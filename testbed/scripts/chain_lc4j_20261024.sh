#!/bin/bash
# lc4j (2026-10-24): the Jetson's attention law between two candidates (register_lc4j_20261024.py), queued behind rpc7 --
# lc4_20261024.py's gates refuse while any other llama.cpp runs, so it waits for heteroserve-rpc7 to end first.
set -u
cd /mnt/data/KMITL/Final/HeteroServe
LOG=testbed/results/chain_lc4j_20261024_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
echo "########## $(date -Is) chain lc4j: waiting for heteroserve-rpc7"
while systemctl --user is-active --quiet heteroserve-rpc7; do sleep 20; done
echo "$(date -Is) rpc7 ended"
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; exit 1; }
sleep 30                                   # the Jetson's guard and the rpc servers settle
python3 testbed/scripts/register_lc4j_20261024.py || { echo "STOP: not registered"; exit 1; }
python3 testbed/scripts/lc4_20261024.py run testbed/dataset/lc4j_predictions_20261024.json; rc=$?
echo "$(date -Is) run exit $rc"
echo "CHAIN LC4J DONE $(date -Is)"
