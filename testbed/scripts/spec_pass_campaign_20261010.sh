#!/bin/bash
# Scheduler upgrade 1, step 1: the pooled Qwen3-14B's pass time when checking k = 1..5 tokens, exactly as registered
# (register_spec_pass_20261010.py). From the repo root, on the testbed LAN, as a systemd user unit:
#     systemd-run --user --unit=heteroserve-specpass-<HHMMSS> --working-directory="$PWD" \
#         systemd-inhibit --what=idle:sleep --mode=block --who=heteroserve-specpass bash testbed/scripts/spec_pass_campaign_20261010.sh
# The link as it is (unshaped, checked); the Jetson's guard started by its NOTE's procedure first and stopped last; the one
# cell through run_pool.sh (every gate, the memory probe, the throttle poll; bench.tool llama-batched-bench); then
# score_spec_pass_20261010.py. Everything is tee'd to testbed/results/spec_pass_<stamp>/campaign.log.
set -uo pipefail
cd "$(dirname "$0")/../.."
PRED=testbed/dataset/spec_pass_predictions_20261010.json
CELL=sp.q3_14b.pass
[ -f "$PRED" ] || { echo "FATAL: $PRED is not registered (python3 testbed/scripts/register_spec_pass_20261010.py)"; exit 1; }
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=testbed/results/spec_pass_$STAMP
mkdir -p "$OUT"
exec > >(tee -i -a "$OUT/campaign.log") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
GUARD=0
finish() {
  trap '' INT TERM HUP
  if [ "$GUARD" = 1 ]; then
    $SSH jetson 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p stopped"; done; read -t 2 <> <(:); echo "guard procs: $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l)"'
  fi
  echo "########## $(date -Is) campaign exit"
}
trap finish EXIT; trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 129' HUP

echo "########## $(date -Is) upgrade 1 step 1, registration $PRED ($(sha256sum "$PRED" | cut -c1-16))"
ip route get 192.168.1.114 | grep -q " via " && { echo "FATAL: the testbed LAN is not directly connected"; exit 1; }
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "FATAL: serve_pool.sh state exists; stop it first"; exit 1; }
bash testbed/scripts/netshape.sh check || { echo "FATAL: the Pi 4 is shaped, or did not answer; see above"; exit 1; }
g=$($SSH jetson 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum ~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1' 2>/dev/null | tr '\n' ' ')
[ "$g" = "synced $(sha256sum testbed/scripts/jetson_power_guard.sh | cut -d' ' -f1) " ] || { echo "FATAL: guard precheck '$g'"; exit 1; }
n=$($SSH jetson 'grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l')
if [ "$n" = 0 ]; then
  $SSH jetson 'sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo "guard started"'
else echo "!!!!! a guard was already running ($n procs): not starting a second; it is stopped at the end, as the NOTE keeps it"; fi
GUARD=1
$SSH jetson 'read -t 4 <> <(:); echo "guard procs: $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l)"'

echo "----- $(date +%T) cell $CELL"
touch "$OUT/.mark_$CELL"
CELL=$CELL PRED=$PRED bash testbed/scripts/run_pool.sh > "$OUT/$CELL.console" 2>&1; echo "run_pool $CELL exit $?"
LOG=$(find testbed/results -maxdepth 1 -name "pool_${CELL}_2*.log" -newer "$OUT/.mark_$CELL" | sort | tail -1)
[ -n "$LOG" ] || { echo "!!!!! $CELL wrote no log (see $OUT/$CELL.console)"; exit 1; }
cp "$LOG" "$OUT/"
grep -aE "^(THROTTLE|MEMPROBE|EXIT_BENCH|CELL DONE|REFUSED|FATAL)|!!!!!|model buffer size" "$LOG" | cut -c1-220
echo "===== $(date +%T) score"
python3 testbed/scripts/score_spec_pass_20261010.py "$OUT/$(basename "$LOG")" --out "$OUT/score.json"
echo "RUN DONE $(date -Is) $OUT"
