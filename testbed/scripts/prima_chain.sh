#!/bin/bash
# The prima.cpp comparison as registered in testbed/dataset/prima_predictions_20260930.json: arm A (the engine's plan on the
# Pi 5 + Pi 4, llama.cpp at the pin) then arm B (prima.cpp with its Halda scheduler), each by qr3_tpot.py, each started only
# once the Pi 5 is below 50 C (up to 10 min). From the repo root:  bash testbed/scripts/prima_chain.sh
# Raw output: testbed/results/prima_chain_<stamp>.log.
set -u
cd "$(dirname "$0")/../.."
STAMP=$(date +%Y%m%d_%H%M%S)
exec > >(tee -a "testbed/results/prima_chain_$STAMP.log") 2>&1
cold() {   # host limit
  local t i
  for i in $(seq 60); do
    t=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$1" "vcgencmd measure_temp" | tr -dc '0-9.')
    awk -v t="$t" -v l="$2" 'BEGIN { exit !(t < l) }' && { echo "cold    $1 at $t C (< $2) after $(( (i - 1) * 10 )) s"; return 0; }
    sleep 10
  done
  echo "!!!!! $1 still at $t C after 10 min; starting anyway, recorded"
}
echo "########## $(date -Is) prima.cpp comparison, registration $(sha256sum testbed/dataset/prima_predictions_20260930.json | cut -c1-16)"
echo; echo "===== $(date -Is) arm A: the engine on the Pi 5 + Pi 4, llama.cpp"
cold pi5 50
python3 testbed/scripts/qr3_tpot.py run --model qwen3_14b --ctx 4096 --mem testbed/dataset/memory_snapshot_20260930_013353.json \
  --nodes "pi5 pi4" --reqs 10 --n 128 --gen 128 --tag primacmpA || echo "!!!!! RUN FAILED: arm A (exit $?)"
echo; echo "===== $(date -Is) arm B: prima.cpp, Halda's split"
cold pi5 50
python3 testbed/scripts/qr3_tpot.py run --model qwen3_14b --ctx 4096 --engine prima --reqs 10 --n 128 --gen 128 --tag primacmpB \
  || echo "!!!!! RUN FAILED: arm B (exit $?)"
echo "CHAIN DONE $(date -Is)"
