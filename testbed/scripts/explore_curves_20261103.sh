#!/usr/bin/env bash
# Per-node batch curves for item 10's checking cost (2026-11-03; the user: "... then ... Per-node batch curves for the
# speculation cost model"). sp10/sp10c carried the laptop's whole-plan curve (explspec B) onto the 9B's laptop + board plans and
# missed the batch of 4 by 12-18 %: the laptop's batch of 4 is cheaper than its batch of 3, a board's is not. Here each node's
# own curve is measured with the model alone on it: one batch of m tokens, m = 1, 2, 3, 4, 6, 8. Not a registered test.
#
#     bash testbed/scripts/explore_curves_20261103.sh
#
# Qwen3.5-9B Q4_K_M (its bytes checked against models.md first), llama-bench -p 1,2,3,4,6,8 -n 0 -r 3 -t 4 (serving's thread
# count), on the Pi 5 and the Pi 4 at once (each alone on its board), then on the laptop alone. Each bench sits between two
# get_throttled / temperature readings (the Pis). No board serves meanwhile (checked).
# Raw output: testbed/results/explcurves_<stamp>/<node>.log (llama-bench -o md, unfiltered).
set -u
cd "$(dirname "$0")/../.."
OUT=testbed/results/explcurves_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
M='~/models/qwen3.5-9b-q4_k_m.gguf'; SHA=d784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43
BENCH="-p 1,2,3,4,6,8 -n 0 -r 3 -t 4 -o md"

board() {   # $1: the board
  local n=$1
  {
    echo "##### $n $(date -Is) explcurves (item 10; not registered)"
    busy=$($SSH $n 'for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null && printf "%s " $p; done; true')
    [ -z "$busy" ] || { echo "FATAL: $n busy: $busy"; return 1; }
    got=$($SSH $n "sha256sum $M" | cut -d' ' -f1)
    echo "sha256 $M: $got $([ "$got" = "$SHA" ] && echo OK || echo MISMATCH)"
    [ "$got" = "$SHA" ] || return 1
    echo "before: $($SSH $n 'vcgencmd get_throttled; vcgencmd measure_temp' | tr '\n' ' ')"
    $SSH $n "cd ~ && ~/llama.cpp/build/bin/llama-bench -m $M $BENCH; echo EXIT_BENCH=\$?"
    echo "after: $($SSH $n 'vcgencmd get_throttled; vcgencmd measure_temp' | tr '\n' ' ')"
    echo "##### $n done $(date -Is)"
  } > "$OUT/$n.log" 2>&1
}
board pi5 & board pi4 & wait
{
  echo "##### laptop $(date -Is) explcurves (item 10; not registered)"
  L=$HOME/models/qwen3.5-9b-q4_k_m.gguf
  got=$(sha256sum "$L" | cut -d' ' -f1)
  echo "sha256 $L: $got $([ "$got" = "$SHA" ] && echo OK || echo MISMATCH)"
  if [ "$got" = "$SHA" ]; then
    echo "load: $(cut -d' ' -f1-3 /proc/loadavg)"
    /opt/heteroserve/llama.cpp/build/bin/llama-bench -m "$L" $BENCH; echo EXIT_BENCH=$?
  fi
  echo "##### laptop done $(date -Is)"
} > "$OUT/laptop.log" 2>&1
echo "DONE $OUT"
