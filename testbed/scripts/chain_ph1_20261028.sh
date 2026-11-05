#!/bin/bash
# ph1 (2026-10-28): the scheduler's phase placement against measurement, energy at the wall (register_ph1_20261028.py,
# registered by hand before this runs), run and scored by phase_study_20261028.py.
set -u
cd /mnt/data/KMITL/Final/HeteroServe
LOG=testbed/results/chain_ph1_20261028_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
echo "########## $(date -Is) chain ph1"
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; exit 1; }
python3 testbed/scripts/phase_study_20261028.py run --reg testbed/dataset/ph1_predictions_20261028.json; rc=$?
echo "$(date -Is) run exit $rc"
echo "CHAIN PH1 DONE $(date -Is)"
