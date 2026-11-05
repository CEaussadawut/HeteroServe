#!/bin/bash
# After the follow-on (heteroserve-after1024, chain2b_20261024.sh): the network study's third registration. ns4b served every
# arm-run, but its shaping filtered the laptop's router address while the RPC flow ran on the overlay, so nothing was shaped.
#   1. smoke_ns4c_20261024.sh: does the shaping now reach the RPC flow (head last, 5 Mbit/s, the netem counters)? On FAIL, stop:
#      nothing is registered.
#   2. register_ns4c_20261024.py from a fresh memory snapshot, then ab_session_20261024.sh, as the chain does.
# It stops, registering nothing, if the follow-on stopped or did not finish, or left serve_pool.sh state or shaping behind.
# From the repo root, under systemd-run + systemd-inhibit:
#     systemd-run --user --unit heteroserve-after1024c --collect -p WorkingDirectory=$PWD systemd-inhibit --what=sleep:idle \
#         --why="HeteroServe ns4c" bash testbed/scripts/chain2c_20261024.sh
# To stop it:  systemctl --user stop heteroserve-after1024c
set -u
cd "$(dirname "$0")/../.."
LOG=testbed/results/chain2c_20261024_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
COND="2026-10-24, after the follow-on: the laptop on its charger, battery full, Wi-Fi off, no working apps; one Claude desktop app open (this session's); Claude works on the laptop during the session (text files, light load, no LibreOffice renders), the laptop's load logged; aunsync.timer every 16 min (~20 s)."
echo "########## $(date -Is) chain2c: smoke, then ns4c, after heteroserve-after1024"
while systemctl --user is-active --quiet heteroserve-after1024; do sleep 60; done
C=$(ls -t testbed/results/chain2b_20261024_*.log | head -1)
echo "$(date -Is) the follow-on ended; its log $C"
grep -a -q "^STOP:" "$C" && { echo "STOP: the follow-on stopped ($(grep -a '^STOP:' "$C" | tail -1))"; exit 1; }
grep -a -q "^CHAIN2B DONE" "$C" || { echo "STOP: the follow-on did not finish"; exit 1; }
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; exit 1; }
bash testbed/scripts/netshape.sh check >/dev/null 2>&1 || { echo "STOP: shaping is installed on the Pi 4"; exit 1; }
sleep 60
bash testbed/scripts/smoke_ns4c_20261024.sh; rc=$?
echo "$(date -Is) smoke exit $rc"
[ $rc -eq 0 ] || { echo "STOP: the smoke test did not pass; ns4c is not registered"; exit 1; }
bash testbed/scripts/netshape.sh check >/dev/null 2>&1 || { echo "STOP: shaping left installed after the smoke test"; exit 1; }
sleep 60
python3 testbed/scripts/memory_snapshot.py --reads 10 | tail -1
M=$(ls -t testbed/dataset/memory_snapshot_*.json | head -1)
python3 testbed/scripts/register_ns4c_20261024.py --memfile "$M" --conditions "$COND" \
  && REG=testbed/dataset/ns4c_predictions_20261024.json bash testbed/scripts/ab_session_20261024.sh
rc=$?
echo "$(date -Is) ns4c exit $rc"
echo "CHAIN2C DONE $(date -Is)"
