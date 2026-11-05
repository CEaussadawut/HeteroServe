#!/bin/bash
# Scope 4 on today's boards (study pd4, 2026-10-24; the user: "complete scope 4"): register_pd4_20261024.py's two steps.
#   0. Wait for the release candidate's verify (heteroserve-verifyrel1024c) to end; check that no server or shaping is left;
#      check Qwen3.5-9B's sha256 on both Pis against testbed/models.md.
#   1. pd4base: register, run (pd4_cross_20261024.py), score.
#   2. pd4cross: registered from step 1's score (it refuses when the hand-off does not beat the Pi 5 alone at gigabit), run,
#      score.
# From the repo root, under systemd-run + systemd-inhibit:
#     systemd-run --user --unit heteroserve-pd4 --collect -p WorkingDirectory=$PWD systemd-inhibit --what=sleep:idle \
#         --why="HeteroServe pd4" bash testbed/scripts/chain_pd4_20261024.sh
# To stop it:  systemctl --user stop heteroserve-pd4   (then: NETSHAPE_PEER=192.168.10.11 NETSHAPE_PORT=50091 bash testbed/scripts/netshape.sh clear)
set -u
cd "$(dirname "$0")/../.."
LOG=testbed/results/chain_pd4_20261024_$(date +%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
COND="2026-10-24 evening: the replacement Pi 5 and the Pi 4 serve; the laptop only orchestrates and timestamps (it computes nothing); the overlay links unshaped except the arm's own transfer port; the strip logs all four outlets (observation only)."
SHA=d784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43
echo "########## $(date -Is) chain pd4"
while systemctl --user is-active --quiet heteroserve-verifyrel1024c; do sleep 30; done
echo "$(date -Is) the release verify has ended"
NETSHAPE_PEER=192.168.10.11 NETSHAPE_PORT=50091 bash testbed/scripts/netshape.sh check >/dev/null 2>&1 || { echo "STOP: shaping is installed on the Pi 4"; exit 1; }
for n in pi5 pi4; do
  busy=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o LogLevel=ERROR $n 'for p in llama-server llama-bench ggml-rpc-server; do pgrep -x $p >/dev/null && printf "%s " $p; done; true')
  [ -z "$busy" ] || { echo "STOP: $n is busy: $busy"; exit 1; }
  got=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o LogLevel=ERROR $n 'sha256sum ~/models/qwen3.5-9b-q4_k_m.gguf | cut -d" " -f1')
  [ "$got" = "$SHA" ] || { echo "STOP: $n's model sha256 is $got, not $SHA"; exit 1; }
  echo "model ok on $n: $got; $(ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o LogLevel=ERROR $n 'vcgencmd get_throttled; vcgencmd measure_temp' | tr '\n' ' ')"
done
python3 testbed/scripts/register_pd4_20261024.py base --conditions "$COND" || { echo "STOP: step 1 not registered"; exit 1; }
python3 testbed/scripts/pd4_cross_20261024.py run --reg testbed/dataset/pd4base_predictions_20261024.json; rc=$?
L1=$(ls -t testbed/results/pd_pd4cross_*.log | grep -v _server | head -1)
echo "$(date -Is) step 1 exit $rc, $L1"
python3 testbed/scripts/pd4_cross_20261024.py score "$L1" --reg testbed/dataset/pd4base_predictions_20261024.json || { echo "STOP: step 1 not scored"; exit 1; }
S1=${L1%.log}_score.json
sleep 30
python3 testbed/scripts/register_pd4_20261024.py cross --base-score "$S1" --conditions "$COND" || { echo "STOP: step 2 not registered (see above)"; echo "CHAIN PD4 DONE $(date -Is)"; exit 0; }
python3 testbed/scripts/pd4_cross_20261024.py run --reg testbed/dataset/pd4cross_predictions_20261024.json; rc=$?
L2=$(ls -t testbed/results/pd_pd4cross_*.log | grep -v _server | head -1)
echo "$(date -Is) step 2 exit $rc, $L2"
python3 testbed/scripts/pd4_cross_20261024.py score "$L2" --reg testbed/dataset/pd4cross_predictions_20261024.json
echo "CHAIN PD4 DONE $(date -Is)"
