#!/bin/bash
# lc4k (2026-10-24): lc4charb's Jetson bench repeated alone (register_lc4k_20261024.py), queued behind c30 and rpc8 --
# lc4_20261024.py's gates refuse while any other llama.cpp runs.
set -u
cd /mnt/data/KMITL/Final/HeteroServe
LOG=testbed/results/chain_lc4k_20261024_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
echo "########## $(date -Is) chain lc4k: waiting for heteroserve-c30 and heteroserve-rpc8"
while systemctl --user is-active --quiet heteroserve-c30 || systemctl --user is-active --quiet heteroserve-rpc8; do sleep 20; done
echo "$(date -Is) both ended"
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; exit 1; }
sleep 30
python3 testbed/scripts/register_lc4k_20261024.py || { echo "STOP: not registered"; exit 1; }
python3 testbed/scripts/lc4_20261024.py run testbed/dataset/lc4k_predictions_20261024.json; rc=$?
echo "$(date -Is) run exit $rc"
echo "CHAIN LC4K DONE $(date -Is)"
