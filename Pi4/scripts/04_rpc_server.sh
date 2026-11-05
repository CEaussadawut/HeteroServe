#!/usr/bin/env bash
# Start llama.cpp rpc-server -- this node's contribution to the distributed cluster.
# Usage: scripts/04_rpc_server.sh [foreground|status|stop]
source "$(dirname "$0")/lib.sh"
ensure_dirs
mode="${1:-foreground}"

case "$mode" in
  status)
    systemctl is-active --quiet heteroserve-rpc && echo "systemd unit: active" || echo "systemd unit: inactive"
    ss -ltnp 2>/dev/null | grep -q ":$RPC_PORT " && echo "listening on :$RPC_PORT" || echo "not listening on :$RPC_PORT"
    exit 0;;
  stop)
    # BOTH, always -- not `systemctl || pkill`. `systemctl stop` returns 0 even when the
    # unit was already inactive, so the `||` short-circuited and a server started outside
    # systemd (`make rpc`, or a stray foreground run) survived a "stop" that reported
    # success. A stale process holding the port during a sweep contaminates every row on
    # the same cores and the same 5 V rail, which is precisely what it looks like it
    # prevented. Verify the port afterwards rather than trusting either command.
    systemctl stop heteroserve-rpc 2>/dev/null || true
    pkill -x "$RPC_COMM_RE" 2>/dev/null || true
    sleep 1
    if ss -ltn 2>/dev/null | grep -q ":$RPC_PORT "; then
      mark rpc_stop "FAILED still-listening"
      die "something is STILL listening on :$RPC_PORT after stop -- find it with 'ss -ltnp | grep :$RPC_PORT' before recording anything"
    fi
    mark rpc_stop; log "stopped (port :$RPC_PORT confirmed closed)"; exit 0;;
esac

RPC="$(bin_rpc)" || die "no rpc server binary in $LLAMA_BIN (looked for ggml-rpc-server, then rpc-server) -- run scripts/02_build_llama_cpp.sh"
assert_clean_state

# Resolve the flags this build accepts before exec'ing, not after: an unknown
# argument makes the server print usage and exit(0), which is indistinguishable
# from a clean shutdown. See rpc_extra_args in lib.sh.
read -r -a RPC_EXTRA <<<"$(rpc_extra_args "$RPC")"

# -c/--cache takes no directory; the server appends "rpc/" to LLAMA_CACHE.
export LLAMA_CACHE="$RPC_CACHE_DIR"
mkdir -p "$LLAMA_CACHE"

if printf '%s\n' "${RPC_EXTRA[@]:-}" | grep -q -- '--mem'; then
  log "rpc server on $RPC_HOST:$RPC_PORT, advertising ${RPC_MEM_MB} MiB"
  log "  (physical ${NODE_RAM_MB} MiB, usable ~${NODE_USABLE_MB} MiB; the gap is deliberate headroom --"
  log "   an OOM mid-run destroys a measurement, and this node has no swap to fall back on)"
  mark rpc_start "port=$RPC_PORT mem=${RPC_MEM_MB}MiB threads=${RPC_THREADS}"
else
  log "rpc server on $RPC_HOST:$RPC_PORT, advertising ACTUAL free memory (no --mem in this build)"
  mark rpc_start "port=$RPC_PORT mem=actual-free threads=${RPC_THREADS}"
fi
log "  binary: $RPC"
log "  flags : --host $RPC_HOST --port $RPC_PORT ${RPC_EXTRA[*]:-<none>}"
log "  threads: ${RPC_THREADS} of ${NODE_CORES} cores (explicit -t; upstream would default to $((NODE_CORES/2)))"

# Log to file AND terminal without putting the server behind a pipe: a `| tee`
# pipeline reports tee's exit status, which hides the server's. This keeps the
# server as the process systemd supervises, so its exit code survives.
exec > >(tee -a "$DATA_DIR/logs/rpc-server.log") 2>&1
exec "$RPC" --host "$RPC_HOST" --port "$RPC_PORT" ${RPC_EXTRA[@]+"${RPC_EXTRA[@]}"}
