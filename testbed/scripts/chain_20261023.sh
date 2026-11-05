#!/bin/bash
# The afternoon chain (2026-10-23; the user, ~10:40: "run everything else now"). After the equal-limits session (unit
# heteroserve-prima4c) ends, for each job in turn -- spec4 (speculative decoding on scheduler v2), then order35 (the engine's
# pick against the predictor's best order on Qwen3.6-35B-A3B) -- a fresh memory snapshot with the boards idle
# (memory_snapshot.py), the job's registration from that snapshot by its unchanged script (read-only; it refuses rather than
# overwrite, or when its plan does not fit, or for order35 when the two plans coincide), then its session
# (spec4_session_20261023.sh with REG=...). A refused registration skips its session; serve_pool.sh state left behind stops
# the chain. From the repo root, under systemd-run + systemd-inhibit:
#     systemd-run --user --unit heteroserve-chain1023 --collect -p WorkingDirectory=$PWD systemd-inhibit --what=sleep:idle \
#         --why="HeteroServe afternoon chain" bash testbed/scripts/chain_20261023.sh
set -u
cd "$(dirname "$0")/../.."
LOG=testbed/results/chain_20261023_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
COND="2026-10-23 afternoon: the laptop on its charger, battery full, Wi-Fi off, no working apps; a second Claude desktop instance open and idle; Claude may work on the laptop during the session (light load), the laptop's load logged per arm-run; aunsync.timer every 16 min (~20 s)."
echo "########## $(date -Is) chain: spec4, then order35, after heteroserve-prima4c"
while systemctl --user is-active --quiet heteroserve-prima4c; do sleep 30; done
echo "$(date -Is) heteroserve-prima4c has ended"
sleep 60
for job in spec4 order35; do
  echo "===== $(date -Is) $job"
  [ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; break; }
  python3 testbed/scripts/memory_snapshot.py --reads 10 | tail -1
  M=$(ls -t testbed/dataset/memory_snapshot_*.json | head -1)
  if [ $job = spec4 ]; then
    python3 testbed/scripts/register_spec4_20261023.py --memfile "$M" --deadline 2026-10-23T18:30:00+07:00 --arm-min 40 --conditions "$COND"
  else
    python3 testbed/scripts/register_order35_20261023.py --memfile "$M" --conditions "$COND"
  fi || { echo "SKIP $job: its registration refused"; continue; }
  REG=testbed/dataset/${job}_predictions_20261023.json bash testbed/scripts/spec4_session_20261023.sh
  echo "$(date -Is) $job session exit $?"
done
echo "CHAIN DONE $(date -Is)"
