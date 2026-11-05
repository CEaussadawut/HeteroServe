#!/bin/bash
# so5 (2026-10-25): the switch-off automation's first live trial (so5_trial_20261025.py), run then scored.
set -u
cd /mnt/data/KMITL/Final/HeteroServe
LOG=testbed/results/chain_so5_20261025_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
echo "########## $(date -Is) chain so5"
RUNOUT=$(mktemp)
python3 testbed/scripts/so5_trial_20261025.py run | tee "$RUNOUT"; rc=${PIPESTATUS[0]}
D=$(tail -1 "$RUNOUT"); rm -f "$RUNOUT"
echo "$(date -Is) run exit $rc, $D"
[ -d "$D" ] && python3 testbed/scripts/so5_trial_20261025.py score "$D"
echo "CHAIN SO5 DONE $(date -Is)"
