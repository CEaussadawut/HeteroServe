#!/bin/bash
# lc4m (2026-10-25): the Jetson's attention law against a third model (register_lc4m_20261025.py, registered by hand at 03:3x),
# run and scored by lc4_20261024.py.
set -u
cd /mnt/data/KMITL/Final/HeteroServe
LOG=testbed/results/chain_lc4m_20261025_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
echo "########## $(date -Is) chain lc4m"
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; exit 1; }
python3 testbed/scripts/lc4_20261024.py run testbed/dataset/lc4m_predictions_20261025.json; rc=$?
echo "$(date -Is) run exit $rc"
echo "CHAIN LC4M DONE $(date -Is)"
