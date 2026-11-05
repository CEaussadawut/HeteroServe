#!/usr/bin/env bash
# Run the ggml RPC server so the orchestrator can place layers on this node.
#
#   ./40_run_rpc_server.sh                 # foreground, reads config/node.env
#   ./40_run_rpc_server.sh --mem 1536      # advertise less memory than default
#   ./40_run_rpc_server.sh --cpu           # expose the CPU backend, not CUDA
#
# SECURITY: ggml's RPC server executes whatever the client sends it and has no
# authentication. Bind it to the testbed's isolated LAN only — never to an
# interface that is reachable from the internet.
. "$(dirname "$0")/lib/common.sh"
load_env

FORCE_CPU=0
while [ $# -gt 0 ]; do
  case "$1" in
    --mem)  RPC_MEM_MB="$2"; shift 2 ;;
    --port) RPC_PORT="$2"; shift 2 ;;
    --bind) RPC_BIND="$2"; shift 2 ;;
    --cpu)  FORCE_CPU=1; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done

RPC="$(bin_rpc)" || die "rpc-server not built — run scripts/20_build_llama_cpp.sh"

if [ "${LOCK_CLOCKS}" = "1" ] && is_jetson; then
  # A reboot resets these; re-assert rather than silently benchmarking a
  # throttled board.
  sudo jetson_clocks 2>/dev/null || warn "could not lock clocks (need sudo)"
  echo "${FAN_PWM}" | sudo tee /sys/devices/pwm-fan/target_pwm >/dev/null 2>&1 || true
fi

if [ "${FORCE_CPU}" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=""
  info "CUDA hidden — this node will serve its CPU backend"
elif [ "${CUDA_UNIFIED_MEMORY}" = "1" ]; then
  # Lets an oversized partition spill into swap instead of failing to allocate.
  # Survivability at the cost of throughput; label any run that uses it.
  export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
  warn "unified memory ON — results from this run are not comparable to the default condition"
fi

step "RPC server"
echo "  node    : ${NODE_ID} (${NODE_ROLE})"
echo "  bind    : ${RPC_BIND}:${RPC_PORT}"
echo "  advertise: ${RPC_MEM_MB} MB"
echo "  binary  : ${RPC}"
free -m | sed 's/^/  /'

# -H/-p are accepted by every ref that has the rpc example; -m (free-mem
# override) only by newer ones, so probe rather than assume.
ARGS=( -H "${RPC_BIND}" -p "${RPC_PORT}" )
if "$RPC" --help 2>&1 | grep -q -- '-m,'; then
  ARGS+=( -m "${RPC_MEM_MB}" )
else
  warn "this ref's rpc-server has no -m flag; it will report the real free memory"
fi
# A local cache of uploaded tensors saves re-sending the partition on every
# run — a large win over 1 GbE, and essential over Wi-Fi.
if "$RPC" --help 2>&1 | grep -q -- '-c,'; then
  ARGS+=( -c )
fi

info "exec ${RPC} ${ARGS[*]}"
exec "$RPC" "${ARGS[@]}"
