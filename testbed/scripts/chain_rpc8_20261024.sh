#!/bin/bash
set -u
cd /mnt/data/KMITL/Final/HeteroServe
LOG=testbed/results/chain_rpc8_20261024_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
COND="2026-10-24 night, after c30: the laptop on its charger, battery full, Wi-Fi off; one Claude desktop app open (this session's); the links unshaped; Qwen3-30B-A3B Q3_K_M and Qwen3.5-9B Q4_K_M served on the laptop + Pi 5 + Pi 4; bytes are counted, so the laptop's load does not enter a reported figure."
echo "########## $(date -Is) chain rpc8: waiting for heteroserve-c30 (and anything it waits for)"
while systemctl --user is-active --quiet heteroserve-c30 || systemctl --user is-active --quiet heteroserve-lc4j; do sleep 20; done
echo "$(date -Is) c30 ended"; sleep 60
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; exit 1; }
python3 testbed/scripts/memory_snapshot.py --reads 10 | tail -1
M=$(ls -t testbed/dataset/memory_snapshot_*.json | head -1)
python3 testbed/scripts/rpc8_char_20261024.py register --memfile "$M" --conditions "$COND" || { echo "STOP: not registered"; exit 1; }
RUNOUT=$(mktemp)
python3 testbed/scripts/rpc8_char_20261024.py run | tee "$RUNOUT"; rc=${PIPESTATUS[0]}
D=$(tail -1 "$RUNOUT"); rm -f "$RUNOUT"
echo "$(date -Is) run exit $rc, $D"
[ $rc -eq 0 ] && [ -d "$D" ] && python3 testbed/scripts/rpc8_char_20261024.py score "$D"
echo "CHAIN RPC8 DONE $(date -Is)"
