#!/bin/bash
# ph2 (2026-10-28): per-model power rows and 60 s pauses on held-out lengths (register_ph2_20261028.py,
# registered by hand before this runs), run and scored by phase_study_20261028.py.
set -u
cd /mnt/data/KMITL/Final/HeteroServe
LOG=testbed/results/chain_ph2_20261028_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
echo "########## $(date -Is) chain ph2"
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; exit 1; }
python3 testbed/scripts/phase_study_20261028.py run --reg testbed/dataset/ph2_predictions_20261028.json; rc=$?
echo "$(date -Is) run exit $rc"
echo "CHAIN PH2 DONE $(date -Is)"
