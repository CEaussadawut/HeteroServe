#!/bin/bash
# c30 (2026-10-24): do the 35B's per-board multipliers price Qwen3-30B-A3B Q6_K (register_c30_20261024.py)? Queued behind lc4j
# (which waits for rpc7): every board idle, then a fresh memory snapshot, the registration, and ab_session_20261024.sh, which
# scores it (ab_score_20261024.py with c30_fit_20261024.py).
set -u
cd /mnt/data/KMITL/Final/HeteroServe
LOG=testbed/results/chain_c30_20261024_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
COND="2026-10-24 evening, after rpc7 and lc4j: the laptop on its charger, battery full, Wi-Fi off; one Claude desktop app open (this session's); Claude works on the laptop during the session (writing text files, light load, no LibreOffice renders), the laptop's load logged; aunsync.timer every 16 min (~20 s)."
echo "########## $(date -Is) chain c30: waiting for heteroserve-rpc7 and heteroserve-lc4j"
while systemctl --user is-active --quiet heteroserve-rpc7 || systemctl --user is-active --quiet heteroserve-lc4j; do sleep 20; done
echo "$(date -Is) both ended"
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; exit 1; }
bash testbed/scripts/netshape.sh check >/dev/null 2>&1 || { echo "STOP: shaping is installed on the Pi 4"; exit 1; }
sleep 60
python3 testbed/scripts/memory_snapshot.py --reads 10 | tail -1
M=$(ls -t testbed/dataset/memory_snapshot_*.json | head -1)
python3 testbed/scripts/register_c30_20261024.py --memfile "$M" --conditions "$COND" || { echo "STOP: not registered"; exit 1; }
REG=testbed/dataset/c30_predictions_20261024.json bash testbed/scripts/ab_session_20261024.sh
echo "CHAIN C30 DONE $(date -Is)"
