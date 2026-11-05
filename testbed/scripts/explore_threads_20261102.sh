#!/usr/bin/env bash
# Exploration for the scheduler review's item 9 (2026-11-02; the user: "... then i tem 9-10"): do the Pis decode faster at 3
# threads than at 4 (September: +9.7 % on the old Pi 5, +6.4 % on the Pi 4, TinyLlama Q4_K_M), on the replacement Pi 5 and on
# a model of the size the plans serve? Pooled serving runs ggml-rpc-server at -t 4 on both (serve_pool.sh WEXTRA); a request
# objective could choose 3 for a decode-heavy request and 4 for a prefill-heavy one. Not a registered test.
#
#     bash testbed/scripts/explore_threads_20261102.sh
#
# On the Pi 5 and the Pi 4 at once (each alone on its board; the laptop only starts them): each model's bytes checked against
# models.md first, then llama-bench -t 3,4 -r 3 (TinyLlama Q4_K_M: pp512 / tg128, September's tests; Qwen3.5-9B Q4_K_M: pp512 /
# tg64), each bench between two `vcgencmd get_throttled` / `measure_temp` readings. No board serves meanwhile (checked).
# Raw output: testbed/results/explthreads_<stamp>/<node>.log (llama-bench -o md, unfiltered).
set -u
cd "$(dirname "$0")/../.."
OUT=testbed/results/explthreads_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
TINY='~/models/tinyllama-1.1b-q4km.gguf'; TINY_SHA=9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0
Q9B='~/models/qwen3.5-9b-q4_k_m.gguf'; Q9B_SHA=d784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43

run() {   # $1: the board
  local n=$1
  {
    echo "##### $n $(date -Is) explthreads (item 9; not registered)"
    busy=$($SSH $n 'for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null && printf "%s " $p; done; true')
    [ -z "$busy" ] || { echo "FATAL: $n busy: $busy"; return 1; }
    for spec in "$TINY $TINY_SHA -p 512 -n 128" "$Q9B $Q9B_SHA -p 512 -n 64"; do
      set -- $spec; m=$1; sha=$2; shift 2
      got=$($SSH $n "sha256sum $m" | cut -d' ' -f1)
      echo "sha256 $m: $got $([ "$got" = "$sha" ] && echo OK || echo MISMATCH)"
      [ "$got" = "$sha" ] || continue
      echo "before: $($SSH $n 'vcgencmd get_throttled; vcgencmd measure_temp' | tr '\n' ' ')"
      $SSH $n "cd ~ && ~/llama.cpp/build/bin/llama-bench -m $m -t 3,4 $* -r 3 -o md; echo EXIT_BENCH=\$?"
      echo "after: $($SSH $n 'vcgencmd get_throttled; vcgencmd measure_temp' | tr '\n' ' ')"
    done
    echo "##### $n done $(date -Is)"
  } > "$OUT/$n.log" 2>&1
}
run pi5 & run pi4 & wait
echo "DONE $OUT"
