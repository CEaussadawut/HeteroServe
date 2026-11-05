#!/bin/bash
# After the four-node chain (heteroserve-chain1024, chain_20261024.sh): the two studies it could not finish, each registered
# from a fresh memory snapshot just before it runs, as the chain does.
#   pm4     the Jetson's power mode on the 35B's split A (register_pm4_20261024.py --c35 <the c35 run>; ab_session_20261024.sh).
#           The chain skipped it at 04:14: its registration script crashed on an unescaped "%" in a claim string ("1 % of",
#           read as a %o conversion) before writing anything. The string is fixed; the design is unchanged.
#   ns4b    the network study, registered again (register_ns4b_20261024.py; ab_session_20261024.sh). ns4 ran at 04:17 and
#           every arm-run failed qr3_tpot.py's served-host check (TinyLlama is not in its per-model host table, so it
#           expected the Pi 5); qr3_tpot.py now takes the host from the fixed plan (PLACE).
#   lc4charb, lc4planb  the long-context study's two steps, registered again (register_lc4b_20261024.py; lc4_20261024.py).
#           lc4char ran at 04:46 and stopped at its first gate: the Jetson's ssh printed the OpenSSH 10.2 post-quantum
#           warning on stderr, which lc4_20261024.py read as a running llama.cpp process; lc4plan then had no step 1 to
#           plan from. lc4_20261024.py now passes -o LogLevel=ERROR (and waits for a quiet laptop, from 04:43).
#   spec4d  the draft-length study, a fourth registration (register_spec4d_20261024.py; spec4_session_20261023.sh).
# Supersedes chain2_20261024.sh (spec4d only), whose unit was stopped while it was still waiting. It registers nothing, and
# stops, if the chain stopped or did not finish, or left serve_pool.sh state or shaping behind.
# From the repo root, under systemd-run + systemd-inhibit:
#     systemd-run --user --unit heteroserve-after1024 --collect -p WorkingDirectory=$PWD systemd-inhibit --what=sleep:idle \
#         --why="HeteroServe pm4, ns4b, lc4b and spec4d" bash testbed/scripts/chain2b_20261024.sh
# To stop it (before or during a study):  systemctl --user stop heteroserve-after1024
set -u
cd "$(dirname "$0")/../.."
LOG=testbed/results/chain2b_20261024_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
COND="2026-10-24, after the four-node chain: the laptop on its charger, battery full, Wi-Fi off, no working apps; one Claude desktop app open (this session's); Claude works on the laptop during the session (text files, light load, no LibreOffice renders), the laptop's load logged; aunsync.timer every 16 min (~20 s)."
JOBS="pm4 ns4b lc4charb lc4planb spec4d"
echo "########## $(date -Is) chain2b: $JOBS, after heteroserve-chain1024"
while systemctl --user is-active --quiet heteroserve-chain1024; do sleep 60; done
C=$(ls -t testbed/results/chain_20261024_*.log | head -1)
echo "$(date -Is) the chain ended; its log $C"
grep -a -q "^STOP:" "$C" && { echo "STOP: the chain stopped ($(grep -a '^STOP:' "$C" | tail -1))"; exit 1; }
grep -a -q "^CHAIN DONE" "$C" || { echo "STOP: the chain did not finish"; exit 1; }
for job in $JOBS; do
  echo "===== $(date -Is) $job"
  [ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; break; }
  bash testbed/scripts/netshape.sh check >/dev/null 2>&1 || { echo "STOP: shaping is installed on the Pi 4"; break; }
  sleep 60
  python3 testbed/scripts/memory_snapshot.py --reads 10 | tail -1
  M=$(ls -t testbed/dataset/memory_snapshot_*.json | head -1)
  case $job in
    pm4)    R=$(ls -td testbed/results/c35_2* 2>/dev/null | head -1)
            if [ -z "$R" ] || [ ! -f "$R/score.json" ]; then echo "SKIP pm4: no scored c35 run"; continue; fi
            python3 testbed/scripts/register_pm4_20261024.py --memfile "$M" --c35 "$R" --conditions "$COND" \
              && REG=testbed/dataset/pm4_predictions_20261024.json bash testbed/scripts/ab_session_20261024.sh ;;
    ns4b)   python3 testbed/scripts/register_ns4b_20261024.py --memfile "$M" --conditions "$COND" \
              && REG=testbed/dataset/ns4b_predictions_20261024.json bash testbed/scripts/ab_session_20261024.sh ;;
    lc4charb) python3 testbed/scripts/register_lc4b_20261024.py char \
              && python3 testbed/scripts/lc4_20261024.py run testbed/dataset/lc4charb_predictions_20261024.json ;;
    lc4planb) R=$(ls -td testbed/results/lc4charb_2* 2>/dev/null | head -1)
              if [ -z "$R" ] || [ ! -f "$R/score.json" ]; then echo "SKIP lc4planb: no scored lc4charb run"; continue; fi
              python3 testbed/scripts/register_lc4b_20261024.py plan --char "$R" --memfile "$M" \
              && [ -f testbed/dataset/lc4planb_predictions_20261024.json ] \
              && python3 testbed/scripts/lc4_20261024.py run testbed/dataset/lc4planb_predictions_20261024.json ;;
    spec4d) python3 testbed/scripts/register_spec4d_20261024.py --memfile "$M" --conditions "$COND" \
              && REG=testbed/dataset/spec4d_predictions_20261024.json bash testbed/scripts/spec4_session_20261023.sh ;;
  esac
  rc=$?   # taken before any $(...) runs: a command substitution resets $? (the chain logs "exit 0" for every job)
  echo "$(date -Is) $job exit $rc"
done
echo "CHAIN2B DONE $(date -Is)"
