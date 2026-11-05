#!/bin/bash
set -u
cd /mnt/data/KMITL/Final/HeteroServe
LOG=testbed/results/chain_rpc7_20261024_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
COND="2026-10-24 evening, after so4: the laptop on its charger, battery full, Wi-Fi off; one Claude desktop app open (this session's); the links unshaped; Qwen3-32B and Qwen3.6-35B served; bytes are counted, so the laptop's load does not enter a reported figure."
echo "########## $(date -Is) chain rpc7"
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; exit 1; }
python3 testbed/scripts/memory_snapshot.py --reads 10 | tail -1
M=$(ls -t testbed/dataset/memory_snapshot_*.json | head -1)
python3 testbed/scripts/rpc7_char_20261024.py register --memfile "$M" --conditions "$COND" || { echo "STOP: not registered"; exit 1; }
RUNOUT=$(mktemp)
python3 testbed/scripts/rpc7_char_20261024.py run | tee "$RUNOUT"; rc=${PIPESTATUS[0]}
D=$(tail -1 "$RUNOUT"); rm -f "$RUNOUT"
echo "$(date -Is) run exit $rc, $D"
[ $rc -eq 0 ] && [ -d "$D" ] && python3 testbed/scripts/rpc7_char_20261024.py score "$D"
echo "CHAIN RPC7 DONE $(date -Is)"
