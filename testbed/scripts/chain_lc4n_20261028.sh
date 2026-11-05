#!/bin/bash
# lc4n (2026-10-28): the Jetson's long-context cost against fa_bench (register_lc4n_20261028.py, registered by hand before
# this runs), run by lc4_20261024.py (its gates, the guard, the clocks; its own scorer gets no hypotheses) and scored by
# score_lc4n_20261028.py.
set -u
cd /mnt/data/KMITL/Final/HeteroServe
LOG=testbed/results/chain_lc4n_20261028_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
echo "########## $(date -Is) chain lc4n"
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; exit 1; }
python3 testbed/scripts/lc4_20261024.py run testbed/dataset/lc4n_predictions_20261028.json; rc=$?
echo "$(date -Is) run exit $rc"
D=$(ls -d testbed/results/lc4n_2* 2>/dev/null | tail -1)
[ -n "$D" ] && python3 testbed/scripts/score_lc4n_20261028.py "$D"; echo "$(date -Is) score exit $?"
echo "CHAIN LC4N DONE $(date -Is)"
