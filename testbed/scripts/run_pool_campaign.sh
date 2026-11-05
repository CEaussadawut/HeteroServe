#!/bin/bash
# The pre-registered pool campaign: run_pool.sh for each cell, in order. From the repo root:
#     bash testbed/scripts/run_pool_campaign.sh [CELL ...]
# A cell REFUSED for memory (live MemAvailable below its pre-registered need) is a result and is skipped. A failed
# gate, a Jetson guard trip, a worker left running or a cell that did not finish stops the campaign: the boards are
# then no longer in the state the predictions assumed.
set -u
cd "$(dirname "$0")/../.."
CELLS=${*:-qwen14b.pi5_pi4_best qwen14b.rule qwen14b.rule_pi4_forced qwen14b.even qwen14b.pi4_host mistral7b.rule mistral7b.even}
LOG=testbed/results/pool_campaign_$(date +%Y%m%d_%H%M%S).log
echo "########## $(date -Is) pool campaign: $CELLS   predictions sha256 $(sha256sum testbed/dataset/pool_predictions.json | cut -c1-16)" | tee -a $LOG
for c in $CELLS; do
  CELL=$c timeout 5400 bash testbed/scripts/run_pool.sh > /dev/null 2>&1; rc=$?
  L=$(ls -t testbed/results/pool_${c}_*.log | head -1)
  echo "$(date -Is) $c rc=$rc log=$L $(grep -hE '^(EXIT_BENCH|REFUSED|FATAL)' $L | tr '\n' ' ')" | tee -a $LOG
  grep -h '^MEMPROBE' $L | sed 's/^/    /' | tee -a $LOG
  if grep -q '^REFUSED' $L; then continue; fi
  if [ $rc != 0 ] || ! grep -q '^CELL DONE' $L || grep -q '^!!!!!' $L || \
     [ "$(grep -h '^STATE after jetson' $L | grep -o 'trips=[0-9]*')" != "$(grep -h '^STATE before jetson' $L | grep -o 'trips=[0-9]*')" ]; then
    echo "STOPPED after $c: failed, left a worker, or tripped the guard" | tee -a $LOG; break
  fi
done
echo "POOL CAMPAIGN DONE $(date -Is)" | tee -a $LOG
