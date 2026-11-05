#!/bin/bash
# Study rpc4 (2026-10-24, after the user's "let's go"): the RPC link's bytes per token, measured, then priced at new rates.
#   1. rpc4_char_20261024.py: register (fresh memory snapshot), run (unshaped; the laptop's TCP counters), score.
#   2. register_rpc4plan_20261024.py from step 1's score (it refuses unless every plan is complete), then
#      ab_session_20261024.sh: H10 L10 LB10 H3 L3 LB3 H10, shaped on the overlay flow, energy at the wall; scored there.
# It stops, registering nothing more, if serve_pool.sh state or shaping is left behind at the start, or step 1 fails.
# From the repo root, under systemd-run + systemd-inhibit:
#     systemd-run --user --unit heteroserve-rpc4 --collect -p WorkingDirectory=$PWD systemd-inhibit --what=sleep:idle \
#         --why="HeteroServe rpc4" bash testbed/scripts/chain_rpc4_20261024.sh
# To stop it:  systemctl --user stop heteroserve-rpc4
set -u
cd "$(dirname "$0")/../.."
LOG=testbed/results/chain_rpc4_20261024_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
COND="2026-10-24 afternoon: the laptop on its charger, battery full, Wi-Fi off, no working apps; one Claude desktop app open (this session's); Claude works on the laptop during the session (text files, light load, no LibreOffice renders), the laptop's load logged; aunsync.timer every 16 min (~20 s)."
echo "########## $(date -Is) chain rpc4: step 1 (bytes per token, unshaped), then step 2 (10 and 3 Mbit/s)"
echo "laptop: AC $(cat /sys/class/power_supply/AC*/online 2>/dev/null) battery $(cat /sys/class/power_supply/BAT0/status 2>/dev/null) $(cat /sys/class/power_supply/BAT0/capacity 2>/dev/null)%, wifi $(nmcli radio wifi 2>/dev/null), load $(cut -d' ' -f1-3 /proc/loadavg)"
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; exit 1; }
NETSHAPE_PEER=192.168.10.5 NETSHAPE_PORT=50052 bash testbed/scripts/netshape.sh check >/dev/null 2>&1 || { echo "STOP: shaping is installed on the Pi 4"; exit 1; }
python3 testbed/scripts/memory_snapshot.py --reads 10 | tail -1
M=$(ls -t testbed/dataset/memory_snapshot_*.json | head -1)
python3 testbed/scripts/rpc4_char_20261024.py register --memfile "$M" --conditions "$COND" || { echo "STOP: step 1 not registered"; exit 1; }
RUNOUT=$(mktemp)
python3 testbed/scripts/rpc4_char_20261024.py run | tee "$RUNOUT"; rc=${PIPESTATUS[0]}
D=$(tail -1 "$RUNOUT"); rm -f "$RUNOUT"
echo "$(date -Is) step 1 run exit $rc, $D"
[ $rc -eq 0 ] && [ -d "$D" ] || { echo "STOP: step 1 did not run"; exit 1; }
python3 testbed/scripts/rpc4_char_20261024.py score "$D" || { echo "STOP: step 1 not scored"; exit 1; }
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state left by step 1"; exit 1; }
sleep 60
python3 testbed/scripts/memory_snapshot.py --reads 10 | tail -1
M2=$(ls -t testbed/dataset/memory_snapshot_*.json | head -1)
python3 testbed/scripts/register_rpc4plan_20261024.py --char "$D" --memfile "$M2" --conditions "$COND" \
  && REG=testbed/dataset/rpc4plan_predictions_20261024.json bash testbed/scripts/ab_session_20261024.sh
rc=$?
echo "$(date -Is) step 2 exit $rc"
echo "CHAIN RPC4 DONE $(date -Is)"
