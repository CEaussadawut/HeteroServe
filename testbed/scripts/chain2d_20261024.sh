#!/bin/bash
# After ns4c (heteroserve-after1024c, chain2c_20261024.sh): sampling on the head's board (scheduler upgrade 6) on four nodes.
#   1. Only if ns4c's shaping smoke test passed: the study's point is the 5 Mbit/s arm.
#   2. apply_serve_extra_20261024.py gives serve_pool.sh its opt-in SERVE_EXTRA (and updates test_spec_serve.py's check),
#      then the testbed suite must pass; if it does not, both files are restored from the patcher's backups and it stops.
#   3. smoke_bs4_20261024.sh: does --backend-sampling reach the server and keep the logits off the shaped link? On FAIL, stop.
#   4. register_bs4_20261024.py from a fresh memory snapshot, then ab_session_20261024.sh.
# From the repo root, under systemd-run + systemd-inhibit:
#     systemd-run --user --unit heteroserve-after1024d --collect -p WorkingDirectory=$PWD systemd-inhibit --what=sleep:idle \
#         --why="HeteroServe bs4" bash testbed/scripts/chain2d_20261024.sh
# To stop it:  systemctl --user stop heteroserve-after1024d
set -u
cd "$(dirname "$0")/../.."
LOG=testbed/results/chain2d_20261024_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
COND="2026-10-24, after ns4c: the laptop on its charger, battery full, Wi-Fi off, no working apps; one Claude desktop app open (this session's); Claude works on the laptop during the session (text files, light load, no LibreOffice renders), the laptop's load logged; aunsync.timer every 16 min (~20 s)."
BAK=.handoff/assets/reproduce-2026-10-24/previous
echo "########## $(date -Is) chain2d: bs4 after heteroserve-after1024c"
while systemctl --user is-active --quiet heteroserve-after1024c; do sleep 60; done
C=$(ls -t testbed/results/chain2c_20261024_*.log 2>/dev/null | head -1)
echo "$(date -Is) ns4c's follow-on ended; its log $C"
[ -n "$C" ] && grep -a -q "^CHAIN2C DONE" "$C" || { echo "STOP: ns4c's follow-on did not finish"; exit 1; }
grep -a -q "smoke exit 0" "$C" || { echo "STOP: ns4c's shaping smoke test did not pass, so there is no shaped arm to test"; exit 1; }
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; exit 1; }
bash testbed/scripts/netshape.sh check >/dev/null 2>&1 || { echo "STOP: shaping is installed on the Pi 4"; exit 1; }
python3 testbed/scripts/apply_serve_extra_20261024.py || { echo "STOP: the serve_pool.sh patch did not apply"; exit 1; }
if ! (cd testbed && python3 -m unittest discover -s tests -q); then
  cp -a "$BAK/serve_pool.sh.before_serve_extra_applied" testbed/scripts/serve_pool.sh
  cp -a "$BAK/test_spec_serve.py.before_serve_extra_applied" testbed/tests/test_spec_serve.py
  echo "STOP: the testbed suite failed after the patch; both files restored"; exit 1
fi
sleep 60
bash testbed/scripts/smoke_bs4_20261024.sh; rc=$?
echo "$(date -Is) smoke exit $rc"
[ $rc -eq 0 ] || { echo "STOP: the smoke test did not pass; bs4 is not registered"; exit 1; }
bash testbed/scripts/netshape.sh check >/dev/null 2>&1 || { echo "STOP: shaping left installed after the smoke test"; exit 1; }
sleep 60
python3 testbed/scripts/memory_snapshot.py --reads 10 | tail -1
M=$(ls -t testbed/dataset/memory_snapshot_*.json | head -1)
python3 testbed/scripts/register_bs4_20261024.py --memfile "$M" --conditions "$COND" \
  && REG=testbed/dataset/bs4_predictions_20261024.json bash testbed/scripts/ab_session_20261024.sh
rc=$?
echo "$(date -Is) bs4 exit $rc"
echo "CHAIN2D DONE $(date -Is)"
