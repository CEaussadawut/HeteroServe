#!/usr/bin/env bash
# Shared helpers for every HeteroServe Pi 4 script. Source this, do not run it.

set -euo pipefail

HS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HS_ROOT

# shellcheck disable=SC1091
set -a; source "$HS_ROOT/config/node.env"; set +a

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

need_root() { [ "$(id -u)" -eq 0 ] || die "run as root:  sudo $0 $*"; }
have()      { command -v "$1" >/dev/null 2>&1; }

# --- llama.cpp binary + flag resolution ------------------------------------
# Upstream has changed the rpc server's interface twice, and the pinned SHA is
# chosen on the Jetson (its CUDA 10.2 constrains the whole cluster), so neither
# the binary name nor the flag set is knowable when this script is written:
#   * renamed  rpc-server -> ggml-rpc-server  in c299a92 (2026-06-27)
#   * -m/--mem removed                        in 41386cf (2025-10-17)
#   * -c/--cache has ALWAYS been boolean -- it takes no directory. LLAMA_CACHE
#     picks the location and the server appends "rpc/" to it.
#
# Getting this wrong is not a soft failure. On an unknown argument the server
# prints usage and calls exit(0) -- a SUCCESS code -- so systemd's
# Restart=on-failure will not restart it, a `| tee` pipeline still returns 0,
# and `heteroctl rpc up` reports the worker as started. Detect, never assume.
find_bin() {
  local d="$1"; shift
  local n
  for n in "$@"; do
    [ -x "${d}/${n}" ] && { echo "${d}/${n}"; return 0; }
  done
  return 1
}

bin_rpc()   { find_bin "$LLAMA_BIN" ggml-rpc-server rpc-server llama-rpc-server; }
bin_bench() { find_bin "$LLAMA_BIN" llama-bench bench; }
bin_cli()   { find_bin "$LLAMA_BIN" llama-cli main; }

# Echoes the extra flags this build actually accepts, and warns when the Pi 4's
# OOM protection is among the ones it does not. The Pi 4 is the node this matters
# most on: no swap by design, so losing --mem means the placement engine
# sees real free memory and can hand this board a partition with no slack.
rpc_extra_args() {
  local rpc="$1" help extra=""
  help="$("$rpc" --help 2>&1 || true)"
  case "$help" in
    *--mem*) extra="--mem ${RPC_MEM_MB}" ;;
    *) warn "this build has NO --mem (removed upstream 2025-10-17 in 41386cf)."
       warn "The node will advertise its ACTUAL free memory, not the deliberate"
       warn "${RPC_MEM_MB} MiB headroom figure in node.env. On this swapless board"
       warn "that turns a graceful refusal into an OOM kill under MemoryMax=3G." ;;
  esac
  case "$help" in
    *--cache*) extra="${extra:+$extra }--cache" ;;
    *) warn "this build has no --cache; the local tensor cache stays disabled" ;;
  esac
  # ALWAYS pass -t. Upstream defaults n_threads to hardware_concurrency()/2, which is 2 of
  # the 4 cores on this board -- so a cluster run would serve at half the cores while the
  # Phase 2 characterisation sweep measured at 1, 2 AND 4, and nothing would record which.
  # The placement rule is fit on per-thread rows; serving at an unrecorded thread count
  # makes the rule's input and the run's behaviour different experiments. QR8 is precisely
  # about thread-limited operation, so this cannot be left implicit.
  case "$help" in
    *--threads*|*"-t,"*) extra="${extra:+$extra }-t ${RPC_THREADS}" ;;
    *) warn "this build has no -t/--threads: the server will pick its own thread count"
       warn "(upstream default is hardware_concurrency()/2 = 2 of 4 cores here) and the"
       warn "dataset will not record what a cluster run actually served with." ;;
  esac
  printf '%s' "$extra"
}

# Matches every name the rpc server has shipped under. Without -f, pgrep/pkill
# read /proc/<pid>/comm, which is capped at 15 chars -- ggml-rpc-server is
# exactly 15 and survives, but the 16-char llama-rpc-server truncates to
# llama-rpc-serve, hence the optional trailing r.
RPC_COMM_RE='(ggml-|llama-)?rpc-serve[r]?'

# Every long-lived artefact lives under DATA_DIR so one rsync pulls a node's whole
# contribution back to the control node.
ensure_dirs() {
  mkdir -p "$DATA_DIR"/{bench,logs,power,inventory}
  mkdir -p "$RPC_CACHE_DIR"
}

# Raspberry Pi throttling word. Any non-zero value invalidates a measurement run:
# an under-volt or a thermal cap silently changes the clock underneath the benchmark.
# Returns the raw vcgencmd word, or "unavailable" -- NEVER "0x0" -- when vcgencmd is
# missing or fails. Reporting "not throttled" because the instrument is absent is how a
# measurement silently becomes unverifiable: every consumer here treats "0x0" as "this row
# is sound". "unavailable" is the same sentinel node/device_info.py:159 already uses.
throttled_word() {
  have vcgencmd || { echo "unavailable"; return; }
  local w; w="$(vcgencmd get_throttled 2>/dev/null)" || { echo "unavailable"; return; }
  case "$w" in
    throttled=0x*) echo "${w#throttled=}" ;;
    *)             echo "unavailable" ;;
  esac
}
soc_temp_c()     { awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "nan"; }
cpu_mhz()        { have vcgencmd && vcgencmd measure_clock arm | sed 's/.*=//' | awk '{printf "%d", $1/1000000}' || echo 0; }
governor()       { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown; }

assert_clean_state() {
  local t; t="$(throttled_word)"
  [ "$t" != "unavailable" ] || die "vcgencmd is unavailable, so throttling cannot be observed -- a sweep recorded blind cannot be shown to be valid. Install raspi-utils / libraspberrypi-bin, or run this on the Pi."
  [ "$t" = "0x0" ] || die "throttled=$t before the run started -- cool the board / fix the PSU, then clear with a reboot (see docs/05_troubleshooting.md)"
  [ "$(governor)" = "performance" ] || die "governor is '$(governor)', expected 'performance' -- run scripts/01_system_tune.sh"
  local swap; swap=$(awk '/SwapTotal/{print $2}' /proc/meminfo)
  [ "$swap" -eq 0 ] || die "swap is enabled (${swap} kB) -- it ruins latency tails; run scripts/01_system_tune.sh"
}

# Models named in config/models.txt that are NOT on disk. A partial fetch is the quiet
# failure here: scripts/03_fetch_models.sh warns and continues, the sweep warns and skips,
# and the result is a dataset with fewer rows than the matrix and nothing recording that
# it is incomplete. "Some .gguf exists" was never the question -- the question is whether
# every model this node owes the dataset is present.
missing_models() {
  local name rest out
  while read -r name rest; do
    case "$name" in ''|\#*) continue;; esac
    out="$DATA_DIR/models/$name.gguf"
    [ -s "$out" ] || printf '%s ' "$name"
  done < "$HS_ROOT/config/models.txt"
}

# The commit actually built, from the build manifest -- NOT the hand-edited
# LLAMA_PIN_SHA. Nothing compared the two, so a node could report "pinned to abc123"
# while running a binary built from something else: edit the pin without rebuilding, or
# rebuild without updating the pin, and the dataset records a revision that never ran.
built_commit() {
  python3 - "$DATA_DIR/inventory/build_manifest.json" <<'PYEOF' 2>/dev/null || true
import json,sys
try:
    print(json.load(open(sys.argv[1]))["llama_cpp"]["commit"])
except Exception:
    pass
PYEOF
}

# --- link tier: declared vs live ------------------------------------------
# The interconnect sweep (scope item 6) DELIBERATELY shapes the link, so "any netem
# present" cannot mean "not ready" -- that would make the per-run gate unpassable for
# every shaped point, and the protocol asks for both a shaped tier and a passing gate.
# What is actually dangerous is shaping nobody declared: a forgotten rule silently
# degrades this run and every later one. So record what was asked for, and compare.
NET_TIER_STATE="${DATA_DIR}/inventory/net_tier.json"

declare_net_tier() {
  # $1 = tier name, $2 = "shaped" or "unshaped", $3 = free-text detail
  mkdir -p "$(dirname "$NET_TIER_STATE")"
  printf '{"tier":"%s","shaped":%s,"detail":"%s","declared_utc":%s,"iface":"%s"}\n' \
    "$1" "$([ "$2" = shaped ] && echo true || echo false)" "${3:-}" \
    "$(date -u +%s.%N)" "$ETH_IFACE" > "$NET_TIER_STATE"
}

live_shaped() { tc qdisc show dev "$ETH_IFACE" 2>/dev/null | grep -qE 'netem|htb'; }

declared_shaped() {
  [ -f "$NET_TIER_STATE" ] || return 1
  grep -q '"shaped":true' "$NET_TIER_STATE"
}

# Ready iff the live qdisc matches what was declared. Undeclared shaping is the failure
# this guards; declared shaping that has silently disappeared is equally a failure,
# because the row would be labelled with a tier the link is not actually running.
net_tier_ok() {
  if live_shaped; then declared_shaped; else ! declared_shaped; fi
}

net_tier_detail() {
  local live="unshaped" decl="unshaped"
  # `if`, not `x && y=z`: that shape is a recorded set -e trap in this repo.
  if live_shaped;     then live="shaped"; fi
  if declared_shaped; then decl="shaped"; fi
  printf 'live=%s declared=%s (%s)' "$live" "$decl" \
    "$([ -f "$NET_TIER_STATE" ] && cat "$NET_TIER_STATE" || echo 'no declaration on file')"
}

# Phase markers. The energy collector on the control node aligns the INA226 stream to
# these by wall-clock, so they MUST be CLOCK_REALTIME and the node MUST be chrony-synced.
mark() {
  local event="$1"; shift || true
  local ts; ts=$(date -u +%s.%N)
  printf '{"ts":%s,"node":"%s","event":"%s","note":"%s","temp_c":%s,"throttled":"%s"}\n' \
    "$ts" "$NODE_ID" "$event" "${*:-}" "$(soc_temp_c)" "$(throttled_word)" \
    | tee -a "$DATA_DIR/logs/events.jsonl" >/dev/null
}
