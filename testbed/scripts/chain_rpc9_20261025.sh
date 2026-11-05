#!/bin/bash
set -u
cd /mnt/data/KMITL/Final/HeteroServe
LOG=testbed/results/chain_rpc9_20261025_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
COND="2026-10-25 night (the user: complete everything lefts): the laptop on its charger; the links unshaped; TinyLlama Q4_0 on laptop 4 + Pi 5 6 + Jetson 6 + Pi 4 6; bytes are counted, so the laptop's load does not enter a reported figure."
echo "########## $(date -Is) chain rpc9"
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; exit 1; }
python3 testbed/scripts/memory_snapshot.py --reads 10 | tail -1
M=$(ls -t testbed/dataset/memory_snapshot_*.json | head -1)
python3 testbed/scripts/rpc9_char_20261025.py register --memfile "$M" --conditions "$COND" || { echo "STOP: not registered"; exit 1; }
RUNOUT=$(mktemp)
python3 testbed/scripts/rpc9_char_20261025.py run | tee "$RUNOUT"; rc=${PIPESTATUS[0]}
D=$(tail -1 "$RUNOUT"); rm -f "$RUNOUT"
echo "$(date -Is) run exit $rc, $D"
[ $rc -eq 0 ] && [ -d "$D" ] && python3 testbed/scripts/rpc9_char_20261025.py score "$D"
echo "CHAIN RPC9 DONE $(date -Is)"
