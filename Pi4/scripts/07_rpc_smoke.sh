#!/usr/bin/env bash
# Verify this node actually serves layers to the control node over RPC.
# Run the control-node half from the T580:
#   llama-cli -m <model.gguf> --rpc pi4:50052 -ngl 99 -p "hello" -n 32
# This script covers the node side: it starts rpc-server, waits for a client, and
# reports what the client asked for.
source "$(dirname "$0")/lib.sh"
ensure_dirs

# The lines this script looks for below -- connected / buffer / graph -- are emitted
# through LOG_DBG in ggml/src/ggml-rpc/ggml-rpc.cpp, which is gated on the GGML_RPC_DEBUG
# environment variable (see :26-29 upstream). Without it the server logs nothing per
# request, so the grep found nothing and this script reported failure on a working
# cluster path. This is a smoke test, not a measurement run, so the extra logging is free.
export GGML_RPC_DEBUG=1
log "starting rpc-server in the background for a smoke test (GGML_RPC_DEBUG=1)"
"$HS_ROOT/scripts/04_rpc_server.sh" foreground &
RPC_PID=$!
trap 'kill $RPC_PID 2>/dev/null || true' EXIT
sleep 3

ss -ltn | grep -q ":$RPC_PORT " || die "rpc-server is not listening on :$RPC_PORT"
log "listening. From the control node, run:"
log "  $LLAMA_BIN/llama-cli -m <model.gguf> --rpc $ETH_ADDR:$RPC_PORT -ngl 99 -p 'hello' -n 32"
log "watching the log for 60 s -- Ctrl-C to stop early"
timeout 60 tail -f "$DATA_DIR/logs/rpc-server.log" || true

if grep -qiE 'connected|buffer|graph_compute|alloc' "$DATA_DIR/logs/rpc-server.log"; then
  log "PASS: the control node connected and allocated buffers on this node"
else
  warn "no client activity seen -- check the firewall, the switch port and the --rpc address."
  warn "If the client DID connect, check that this build emits LOG_DBG at all: the strings"
  warn "grepped for here come from ggml-rpc.cpp and need GGML_RPC_DEBUG set on the SERVER."
fi
