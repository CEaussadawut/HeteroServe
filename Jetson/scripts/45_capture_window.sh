#!/usr/bin/env bash
# Phase-marked power capture for a run this node does NOT drive.
#
#   ./45_capture_window.sh start           # begin logging, marker = idle
#   ./45_capture_window.sh mark prefill    # label everything from now on
#   ./45_capture_window.sh idle            # back to idle between phases
#   ./45_capture_window.sh stop            # end logging, print the CSV path
#   ./45_capture_window.sh path|status     # where it is writing / is it alive
#
# WHY THIS EXISTS. `bench/run_matrix.py` marks its own windows because it drives
# llama-bench locally: it knows when a prefill starts because it started it. In
# the distributed experiment this node is an RPC worker -- the orchestrator on
# the laptop hands it tensors, never phases -- so nothing here can know which
# phase is running, and the study's central measurement (prefill vs decode
# placement ACROSS nodes) had no way to label this node's power at all.
#
# WHO WRITES THE MARKER: the orchestrator, over ssh, because it is the only
# machine that knows what it asked for. It brackets a command that does ONE
# phase, exactly as docs/10 section 4 splits the local matrix:
#
#   ssh jetson '.../45_capture_window.sh start'
#   ssh jetson '.../45_capture_window.sh mark prefill'
#   <run the prefill leg against the cluster>
#   ssh jetson '.../45_capture_window.sh idle'
#   ssh jetson '.../45_capture_window.sh mark decode'
#   <run the decode leg>
#   ssh jetson '.../45_capture_window.sh stop'
#
# THE LIMIT, stated because it bounds what the resulting number means: an ssh
# round trip puts each marker edge within roughly one sample period of the real
# one. For a leg of seconds that is fine. It is hopeless for the prefill/decode
# boundary INSIDE a single request -- that boundary is sub-second and the two
# phases are not separable this way. Bracket whole legs, not whole requests.
#
# The CSV lands in $RESULTS_DIR under the house naming, and the marker file is
# the SAME path run_matrix.py uses ($RESULTS_DIR/.marker_$NODE_ID), so:
#   - telemetry/energy_window.py reads a cluster CSV with exactly the same
#     commands as a sweep CSV (`--marker prefill`, `--idle-mw auto`);
#   - 90_collect_results.sh tarballs it without being told it exists, and
#     already excludes .marker_* from the bundle.
. "$(dirname "$0")/lib/common.sh"
load_env

MARKER="${RESULTS_DIR}/.marker_${NODE_ID}"
PIDFILE="${RESULTS_DIR}/.capture_${NODE_ID}.pid"
CSVFILE="${RESULTS_DIR}/.capture_${NODE_ID}.path"
ACTION="${1:-}"

capture_pid() { [ -r "$PIDFILE" ] && cat "$PIDFILE" || true; }
capture_alive() {
  local p; p="$(capture_pid)"
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null
}

case "$ACTION" in
  start)
    mkdir -p "$RESULTS_DIR"
    if capture_alive; then
      die "a capture is already running (pid $(capture_pid)); stop it first"
    fi
    # Same reason run_matrix.py refuses: two tegrastats instances cannot both
    # read the INA3221 -- the second silences the first, and the capture then
    # produces a complete-looking CSV with no power in it.
    if tegrastats_running; then
      die "a tegrastats process is already running; stop it before capturing
  (sudo systemctl stop heteroserve-telemetry, or pkill tegrastats)"
    fi
    OUT="$(result_path cluster_power csv)"
    printf 'idle' > "$MARKER"
    nohup python3 "${HS_JETSON_DIR}/telemetry/tegrastats_logger.py" \
      --out "$OUT" --hz "${TELEMETRY_HZ:-5}" \
      --node-id "$NODE_ID" --marker-file "$MARKER" \
      >/dev/null 2>&1 &
    echo $! > "$PIDFILE"
    printf '%s' "$OUT" > "$CSVFILE"
    # Prove it is actually logging before the orchestrator commits a phase to it.
    # A capture that died at startup is otherwise discovered after the cluster
    # run, when the run cannot be repeated cheaply.
    sleep 2
    if ! capture_alive || [ ! -s "$OUT" ]; then
      rm -f "$PIDFILE"
      die "the capture did not start (no samples in $(basename "$OUT"))"
    fi
    ok "capturing to $OUT"
    ;;

  mark)
    [ -n "${2:-}" ] || die "usage: $0 mark <label>   (e.g. prefill, decode)"
    capture_alive || die "no capture is running; $0 start first"
    # The label is written verbatim and the logger stamps it on every sample
    # from the next one onward. energy_window.py matches it exactly or on the
    # `phase:` prefix, so `prefill` and `prefill:cluster:007` both work.
    printf '%s' "$2" > "$MARKER"
    ok "marker = $2"
    ;;

  idle)
    capture_alive || die "no capture is running"
    printf 'idle' > "$MARKER"
    ok "marker = idle"
    ;;

  stop)
    capture_alive || { warn "no capture is running"; exit 0; }
    printf 'idle' > "$MARKER"
    # A trailing idle tail gives the last phase a cooldown to be measured
    # against, the same way the sweep's per-case floor works.
    sleep 3
    kill "$(capture_pid)" 2>/dev/null || true
    sleep 1
    kill -9 "$(capture_pid)" 2>/dev/null || true
    rm -f "$PIDFILE"
    OUT="$(cat "$CSVFILE" 2>/dev/null || true)"
    ok "stopped"
    [ -n "$OUT" ] && echo "$OUT"
    ;;

  path)
    cat "$CSVFILE" 2>/dev/null || true
    ;;

  status)
    if capture_alive; then
      OUT="$(cat "$CSVFILE" 2>/dev/null || true)"
      N="$( set +o pipefail; wc -l < "$OUT" 2>/dev/null || echo 0 )"
      echo "running pid=$(capture_pid) marker=$(cat "$MARKER" 2>/dev/null || echo '?') rows=$((N > 0 ? N - 1 : 0)) out=$OUT"
    else
      echo "stopped"
    fi
    ;;

  *)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
