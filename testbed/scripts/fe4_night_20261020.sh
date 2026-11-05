#!/bin/bash
# The order of the four-node evaluation day (2026-10-20): the Qwen3-14B evaluation (it completes the user's 10-11 request: the
# laptop plan's energy and the laptop alone), then the engine against prima.cpp with energy on Pi 5 + Pi 4 (it needs both
# boards to itself), then the rest of the registered models (fe4_chain_20261020.sh skips the 14B as already run). Every
# registration was written at 06:14 (testbed/results/fe4_register_20261020_061431/, prima_energy_predictions_20261020.json).
#     systemd-run --user --unit heteroserve-fe4 --collect -p WorkingDirectory=$PWD systemd-inhibit --what=sleep:idle \
#         --why="HeteroServe four-node full evaluation" bash testbed/scripts/fe4_night_20261020.sh
set -u
cd "$(dirname "$0")/../.."
LOG=testbed/results/fe4_night_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
echo "########## $(date -Is) the four-node evaluation day"
NAMES="q14b_fe4_20261020" bash testbed/scripts/fe4_chain_20261020.sh; echo "PART 1 (the 14B) exit $? $(date -Is)"
bash testbed/scripts/prima_energy_20261020.sh; echo "PART 2 (prima.cpp energy) exit $? $(date -Is)"
bash testbed/scripts/fe4_chain_20261020.sh; echo "PART 3 (the rest) exit $? $(date -Is)"
echo "NIGHT DONE $(date -Is)"
