#!/bin/bash
# The 2026-10-24 chain (the user, ~00:30: "Do all lefts"): the four-node re-runs, one study at a time, each registered from a
# fresh memory snapshot (memory_snapshot.py, the boards idle) just before it runs, by its own script, which refuses rather
# than overwrite, or when its plan does not fit. A refused registration skips its run and the chain goes on; serve_pool.sh
# state left behind, or link shaping found installed, stops the chain. In order:
#   c35      Qwen3.6-35B-A3B characterised where it runs (register_c35_20261024.py; ab_session_20261024.sh)
#   pm4      the Jetson's power mode on the four-node 35B plan (register_pm4_20261024.py --c35 <the c35 run>; skipped when the
#            c35 run has no score)
#   ns4      the host's link and the head's place (register_ns4_20261024.py; ab_session_20261024.sh)
#   ring4    the ring laps on the laptop + Pi 5 (register_ring4_20261024.py; ring4_laps_20261024.sh)
#   lc4char  the cost of context depth per board (register_lc4_20261024.py char; lc4_20261024.py run)
#   lc4plan  the plans without and with the depth term (register_lc4_20261024.py plan --char <the lc4char run>)
# From the repo root, under systemd-run + systemd-inhibit:
#     systemd-run --user --unit heteroserve-chain1024 --collect -p WorkingDirectory=$PWD systemd-inhibit --what=sleep:idle \
#         --why="HeteroServe four-node re-runs" bash testbed/scripts/chain_20261024.sh [first-job]
set -u
cd "$(dirname "$0")/../.."
LOG=testbed/results/chain_20261024_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
COND="2026-10-24 night: the laptop on its charger, battery full, Wi-Fi off, no working apps; two Claude desktop apps open (the user's two profiles, this session in one); Claude works on the laptop during the sessions (writing documents as text files, light load, no LibreOffice renders), the laptop's load logged; aunsync.timer every 16 min (~20 s)."
JOBS="c35 pm4 ns4 ring4 lc4char lc4plan"
if [ -n "${1:-}" ]; then JOBS=${JOBS#*"$1"}; JOBS="$1$JOBS"; fi
echo "########## $(date -Is) chain: $JOBS"
while systemctl --user is-active --quiet heteroserve-spec4c; do sleep 30; done
newest() { ls -td testbed/results/$1_2* 2>/dev/null | head -1; }
for job in $JOBS; do
  echo "===== $(date -Is) $job"
  [ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; break; }
  bash testbed/scripts/netshape.sh check >/dev/null 2>&1 || { echo "STOP: shaping is installed on the Pi 4"; break; }
  sleep 60
  python3 testbed/scripts/memory_snapshot.py --reads 10 | tail -1
  M=$(ls -t testbed/dataset/memory_snapshot_*.json | head -1)
  case $job in
    c35)  python3 testbed/scripts/register_c35_20261024.py --memfile "$M" --conditions "$COND" \
            && REG=testbed/dataset/c35_predictions_20261024.json bash testbed/scripts/ab_session_20261024.sh ;;
    pm4)  C=$(newest c35)
          if [ -z "$C" ] || [ ! -f "$C/score.json" ]; then echo "SKIP pm4: no scored c35 run"; continue; fi
          python3 testbed/scripts/register_pm4_20261024.py --memfile "$M" --c35 "$C" --conditions "$COND" \
            && REG=testbed/dataset/pm4_predictions_20261024.json bash testbed/scripts/ab_session_20261024.sh ;;
    ns4)  python3 testbed/scripts/register_ns4_20261024.py --memfile "$M" --conditions "$COND" \
            && REG=testbed/dataset/ns4_predictions_20261024.json bash testbed/scripts/ab_session_20261024.sh ;;
    ring4) python3 testbed/scripts/register_ring4_20261024.py && bash testbed/scripts/ring4_laps_20261024.sh ;;
    lc4char) python3 testbed/scripts/register_lc4_20261024.py char \
            && python3 testbed/scripts/lc4_20261024.py run testbed/dataset/lc4char_predictions_20261024.json ;;
    lc4plan) C=$(newest lc4char)
          if [ -z "$C" ] || [ ! -f "$C/score.json" ]; then echo "SKIP lc4plan: no scored lc4char run"; continue; fi
          python3 testbed/scripts/register_lc4_20261024.py plan --char "$C" --memfile "$M" \
            && [ -f testbed/dataset/lc4plan_predictions_20261024.json ] \
            && python3 testbed/scripts/lc4_20261024.py run testbed/dataset/lc4plan_predictions_20261024.json ;;
  esac
  echo "$(date -Is) $job exit $?"
done
echo "CHAIN DONE $(date -Is)"
