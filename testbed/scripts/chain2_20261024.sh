#!/bin/bash
# After the four-node chain (heteroserve-chain1024, chain_20261024.sh): the draft-length study, registered afresh as spec4d
# (register_spec4d_20261024.py) from a fresh memory snapshot, then run by spec4_session_20261023.sh. It registers nothing,
# and stops, if the chain stopped or did not finish, or left serve_pool.sh state or shaping behind.
# From the repo root, under systemd-run + systemd-inhibit:
#     systemd-run --user --unit heteroserve-chain1024b --collect -p WorkingDirectory=$PWD systemd-inhibit --what=sleep:idle \
#         --why="HeteroServe draft length (spec4d)" bash testbed/scripts/chain2_20261024.sh
# To stop it (before or during the study):  systemctl --user stop heteroserve-chain1024b
set -u
cd "$(dirname "$0")/../.."
LOG=testbed/results/chain2_20261024_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
COND="2026-10-24, after the four-node chain: the laptop on its charger, battery full, Wi-Fi off, no working apps; one Claude desktop app open (this session's); Claude works on the laptop during the session (text files, light load, no LibreOffice renders), the laptop's load logged; aunsync.timer every 16 min (~20 s)."
echo "########## $(date -Is) chain2: spec4d after heteroserve-chain1024"
while systemctl --user is-active --quiet heteroserve-chain1024; do sleep 60; done
C=$(ls -t testbed/results/chain_20261024_*.log | head -1)
echo "$(date -Is) the chain ended; its log $C"
grep -a -q "^STOP:" "$C" && { echo "STOP: the chain stopped ($(grep -a '^STOP:' "$C" | tail -1))"; exit 1; }
grep -a -q "^CHAIN DONE" "$C" || { echo "STOP: the chain did not finish"; exit 1; }
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; exit 1; }
bash testbed/scripts/netshape.sh check >/dev/null 2>&1 || { echo "STOP: shaping is installed on the Pi 4"; exit 1; }
sleep 60
python3 testbed/scripts/memory_snapshot.py --reads 10 | tail -1
M=$(ls -t testbed/dataset/memory_snapshot_*.json | head -1)
python3 testbed/scripts/register_spec4d_20261024.py --memfile "$M" --conditions "$COND" || { echo "STOP: spec4d was not registered"; exit 1; }
REG=testbed/dataset/spec4d_predictions_20261024.json bash testbed/scripts/spec4_session_20261023.sh
echo "$(date -Is) spec4d exit $?"
echo "CHAIN2 DONE $(date -Is)"
