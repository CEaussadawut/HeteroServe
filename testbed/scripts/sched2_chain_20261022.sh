#!/bin/bash
# Scheduler v2's night (2026-10-22), as registered in testbed/dataset/sched2_predictions_20261022.json: the quality check
# (sched2_quality_20261022.sh, the laptop alone; the boards idle) and then the speed / energy session (sched2_session_20261022.sh,
# A V A V on all four nodes, scored at its end with the quality log). One after the other, never together: the session waits for
# a quiet laptop itself. From the repo root:
#     systemd-run --user --unit heteroserve-sched2 --collect -p WorkingDirectory=$PWD systemd-inhibit --what=sleep:idle \
#         --why="HeteroServe scheduler v2" bash testbed/scripts/sched2_chain_20261022.sh
set -u
cd "$(dirname "$0")/../.."
LOG=testbed/results/sched2_chain_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
REG=testbed/dataset/sched2_predictions_20261022.json
[ -f "$REG" ] && [ ! -w "$REG" ] || { echo "FATAL: $REG missing or writable (register first)"; exit 2; }
echo "########## $(date -Is) scheduler v2 chain (registration $(sha256sum "$REG" | cut -c1-16))"
bash testbed/scripts/sched2_quality_20261022.sh; rc=$?
echo "quality exit $rc $(date -Is)"
[ "$rc" = 0 ] || { echo "STOPPED: the quality check failed"; exit 1; }
bash testbed/scripts/sched2_session_20261022.sh; rc=$?
echo "session exit $rc $(date -Is)"
echo "CHAIN DONE $(date -Is)"
