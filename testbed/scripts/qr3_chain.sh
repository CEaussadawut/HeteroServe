#!/bin/bash
# The QR3 campaign as registered in testbed/dataset/qr3_predictions_20260930b.json: its five configs in its run_order, each
# started only once its host is cool (Pi 5 < 50 C, Pi 4 < 57 C, up to 10 min), each by qr3_tpot.py. From the repo root:
#     bash testbed/scripts/qr3_chain.sh
# A config that fails is logged and the chain goes on; qr3_tpot.py stops what serve_pool.sh started and restores the Jetson's
# clocks whatever happens. Raw output: testbed/results/qr3_chain_<stamp>.log.
set -u
cd "$(dirname "$0")/../.."
STAMP=$(date +%Y%m%d_%H%M%S)
exec > >(tee -a "testbed/results/qr3_chain_$STAMP.log") 2>&1
MEM=testbed/dataset/memory_snapshot_20260930_013353.json
cold() {   # host limit
  local t i
  for i in $(seq 60); do
    t=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$1" "vcgencmd measure_temp" | tr -dc '0-9.')
    awk -v t="$t" -v l="$2" 'BEGIN { exit !(t < l) }' && { echo "cold    $1 at $t C (< $2) after $(( (i - 1) * 10 )) s"; return 0; }
    sleep 10
  done
  echo "!!!!! $1 still at $t C after 10 min; starting anyway, recorded"
}
echo "########## $(date -Is) QR3 chain, registration $(sha256sum testbed/dataset/qr3_predictions_20260930b.json | cut -c1-16)"
for spec in "qwen35_9b scored locked pi5 50" "nemo12b scored locked pi5 50" "qwen3_14b scored locked pi5 50" \
            "qwen3moe scored locked pi4 57" "qwen3_14b asfound asfound pi5 50"; do
  set -- $spec
  echo; echo "===== $(date -Is) $1 --tag $2 --clocks $3"
  cold "$4" "$5"
  python3 testbed/scripts/qr3_tpot.py run --model "$1" --ctx 4096 --mem "$MEM" --reqs 10 --n 128 --gen 128 --tag "$2" --clocks "$3" \
    || echo "!!!!! RUN FAILED: $1 $2 (exit $?)"
done
echo "CHAIN DONE $(date -Is)"
