#!/usr/bin/env bash
# Single-node characterisation sweep: throughput + energy for this Jetson alone.
# Thin wrapper over bench/run_matrix.py that asserts the measurement conditions
# first, because a sweep run on a thermally throttled or unlocked board is
# wasted hours.
#
#   ./50_bench_device.sh                     # full matrix
#   ./50_bench_device.sh --only gpu_decode_sweep
#   ./50_bench_device.sh --tag 5W            # label a power-mode condition
#   ./50_bench_device.sh --dry-run
. "$(dirname "$0")/lib/common.sh"
load_env

step "Pre-flight"
if is_jetson; then
  # sudo may not be able to prompt (cron, ssh without a tty), and grep returns
  # 1 when the output shape differs between L4T releases. Neither is a reason to
  # abandon a two-hour sweep at the pre-flight step.
  MODE="$( set +o pipefail
    sudo nvpmodel -q 2>/dev/null | grep -A1 'NV Power Mode' | tail -1 || true )"
  echo "  power mode : ${MODE:-unknown} (configured NVP_MODEL=${NVP_MODEL})"
  if [ "${LOCK_CLOCKS}" = "1" ]; then
    sudo jetson_clocks 2>/dev/null || warn "jetson_clocks needs sudo"
    ok "clocks re-locked"
  fi
    # By zone TYPE, not by max: the hottest zone on a Nano is the PMIC-Die dummy
    # at a constant 100 C, so the old `sort -n | tail -1` made this gate report
    # 100 C on a stone-cold board every time -- a warning that always fires is
    # one nobody reads. thermal_zone_c() also prefers the same quantity
    # tegrastats reports as `thermal@`, so this reading and every sweep row's
    # temperature are the same number.
    TZ="$(thermal_zone_c)"
    if [ -n "$TZ" ]; then
      TC="${TZ##* }"
      echo "  temp start : ${TC} C (${TZ%% *})"
      # Starting hot means the first cases run at a different clock than the last.
      [ "$TC" -gt 45 ] && warn "board is warm; let it settle below 45 C for comparable runs"
    else
      warn "no usable thermal zone found; pre-flight temperature not checked"
    fi
  # rpc_running matches every name RPC_BIN_NAMES knows -- cut to the 15 chars
  # /proc/<pid>/comm keeps -- and falls back to RPC_PORT for a binary renamed
  # again upstream. A plain `pgrep -x rpc-server` missed both ggml-rpc-server and
  # the truncated llama-rpc-serve, so the guard went quiet exactly when the sweep
  # was contended. One list in lib/common.sh means the next rename is one edit.
  if rpc_running; then
    warn "an rpc server is running and will steal CPU and RAM from this sweep"
    confirm "continue anyway?" || exit 1
  fi
else
  warn "not a Jetson — running without power logging"
  set -- "$@" --no-power
fi

free -m | sed 's/^/  /'
echo "  models     : $(ls "${MODEL_DIR}"/*.gguf 2>/dev/null | wc -l) gguf in ${MODEL_DIR}"

step "Sweep"
# The full matrix is ~32 runs, roughly 2-3.5 h on a Nano including cooldowns. Each
# run is one phase in its own power window -- see docs/10_measurement_protocol.md
# section 4. Run it under
# tmux/screen so an ssh drop does not kill it.
if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ] && [ -t 1 ]; then
  warn "not inside tmux/screen — an ssh disconnect will abort the sweep"
fi
exec python3 "${HS_JETSON_DIR}/bench/run_matrix.py" "$@"
