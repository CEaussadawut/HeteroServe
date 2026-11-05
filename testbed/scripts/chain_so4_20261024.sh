#!/bin/bash
# The switch-off test (study so4, 2026-10-24; the user, 14:44: "do the switch-off after this"), launched after rpc4 and after
# one hand rehearsal of strip_switch_20261024.py's off-node / on-node cycle on the Pi 4.
#   1. Gates: rpc4's chain ended (CHAIN RPC4 DONE), no serve_pool.sh state, the Pi 4 unshaped, on and answering.
#   2. register_so4_20261024.py from a fresh memory snapshot (it refuses if the plan uses the Pi 4), then
#      ab_session_20261024.sh: A B B A; each B arm-run's pre powers the Pi 4 off cleanly and switches its outlet off, its post
#      switches it on and waits until the board is back (the session's trap runs the post again on any way out).
# From the repo root, under systemd-run + systemd-inhibit:
#     systemd-run --user --unit heteroserve-so4 --collect -p WorkingDirectory=$PWD systemd-inhibit --what=sleep:idle \
#         --why="HeteroServe so4" bash testbed/scripts/chain_so4_20261024.sh
# To stop it:  systemctl --user stop heteroserve-so4   (then check: python3 testbed/scripts/strip_switch_20261024.py status pi4)
set -u
cd "$(dirname "$0")/../.."
LOG=testbed/results/chain_so4_20261024_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
COND="2026-10-24 afternoon, after rpc4: the laptop on its charger, battery full, Wi-Fi off, no working apps; one Claude desktop app open (this session's); Claude works on the laptop during the session (text files, light load, no LibreOffice renders), the laptop's load logged; aunsync.timer every 16 min (~20 s). The strip's outlet 0 (the Pi 4) is switched by the study itself, with the user's go-ahead (14:44)."
echo "########## $(date -Is) chain so4: the switch-off test"
C=$(ls -t testbed/results/chain_rpc4_20261024_*.log 2>/dev/null | head -1)
[ -n "$C" ] && grep -a -q "^CHAIN RPC4 DONE" "$C" || { echo "STOP: rpc4's chain has not ended"; exit 1; }
systemctl --user is-active --quiet heteroserve-rpc4 && { echo "STOP: heteroserve-rpc4 is still active"; exit 1; }
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "STOP: serve_pool.sh state exists"; exit 1; }
NETSHAPE_PEER=192.168.10.5 NETSHAPE_PORT=50052 bash testbed/scripts/netshape.sh check >/dev/null 2>&1 || { echo "STOP: shaping is installed on the Pi 4"; exit 1; }
python3 testbed/scripts/strip_switch_20261024.py status pi4 | grep -q '"output": true' || { echo "STOP: the Pi 4's outlet is not on"; exit 1; }
ssh -n -o BatchMode=yes -o ConnectTimeout=8 pi4 true || { echo "STOP: the Pi 4 does not answer"; exit 1; }
python3 testbed/scripts/memory_snapshot.py --reads 10 | tail -1
M=$(ls -t testbed/dataset/memory_snapshot_*.json | head -1)
python3 testbed/scripts/register_so4_20261024.py --memfile "$M" --conditions "$COND" \
  && REG=testbed/dataset/so4_predictions_20261024.json bash testbed/scripts/ab_session_20261024.sh
rc=$?
echo "$(date -Is) so4 exit $rc"
python3 testbed/scripts/strip_switch_20261024.py status pi4
echo "CHAIN SO4 DONE $(date -Is)"
