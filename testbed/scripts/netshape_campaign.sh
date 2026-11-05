#!/bin/bash
# The network study's campaign (register_netshape_20261008.py; topic-form scope 6). From the repo root, on the testbed LAN:
#     bash testbed/scripts/netshape_campaign.sh
# For each registered setting, in order: shape the Pi 4 <-> Jetson path (netshape.sh; s0 and s0r stay unshaped), measure
# the shaped link (measure_link.sh, PAIR="pi4 jetson", a transfer size scaled to the rate), run the setting's two cells in
# their registered order (run_pool.sh: every gate, the memory probe and the throttle poll), and clear the shaping. The
# shaping is cleared on any exit. The Jetson's power guard is started by its NOTE's procedure first and stopped last.
# Then compare_pool.py scores every cell against the nominal predictions, and score_netshape_20261008.py re-prices each
# cell with its setting's measured link and applies the registered hypotheses.
# Everything is tee'd to testbed/results/netshape_<stamp>/campaign.log; each setting's link log is copied beside it.
set -uo pipefail
cd "$(dirname "$0")/../.."
PRED=testbed/dataset/pool_predictions_20261008netshape_r2.json   # r2: the Jetson's address after the router change (the first registration's -rpc is stale)
[ -f "$PRED" ] || { echo "FATAL: $PRED is not registered (python3 testbed/scripts/register_netshape_20261008.py)"; exit 1; }
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=testbed/results/netshape_$STAMP
mkdir -p "$OUT"
exec > >(tee -i -a "$OUT/campaign.log") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
GUARD=0
finish() {
  trap '' INT TERM HUP
  bash testbed/scripts/netshape.sh clear || echo "!!!!! the Pi 4 may still be shaped: run netshape.sh clear by hand"
  if [ "$GUARD" = 1 ]; then
    $SSH jetson 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p stopped"; done; read -t 2 <> <(:); echo "guard procs: $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l)"'
  fi
  echo "########## $(date -Is) campaign exit"
}
trap finish EXIT; trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 129' HUP

echo "########## $(date -Is) the network study, registration $PRED ($(sha256sum "$PRED" | cut -c1-16))"
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

mapfile -t SETS < <(python3 -c "
import json; r = json.load(open('$PRED'))
for s in r['settings']:
    cells = [c for c in r['run_order'] if r['configs'][c]['setting'] == s['id']]
    print(s['id'], s['rate_mbit'], '%g' % s['delay_ms_one_way'], ' '.join(cells))")
LOGS=""
for line in "${SETS[@]}"; do
  read -r sid rate delay cells <<<"$line"
  echo "===== $(date +%T) setting $sid: rate ${rate} Mbit/s (0 = none), one-way delay ${delay} ms; cells $cells"
  if [ "$rate" != 0 ] || [ "$delay" != 0 ]; then bash testbed/scripts/netshape.sh apply "$rate" "$delay" || exit 1; fi
  if [ "$rate" = 0 ]; then mib=512; else mib=$(( rate < 16 ? 16 : rate )); fi      # about 8 s per transfer when shaped
  touch "$OUT/.mark_$sid"
  PAIR="pi4 jetson" MIB=$mib REPS=3 bash testbed/scripts/measure_link.sh > /dev/null 2>&1
  L=$(find testbed/results -maxdepth 1 -name 'link_pi4-jetson_*.log' -newer "$OUT/.mark_$sid" | sort | tail -1)
  [ -n "$L" ] || { echo "FATAL: measure_link.sh wrote no new log for $sid"; exit 1; }
  cp "$L" "$OUT/link_$sid.log"
  echo "LINK $sid $(python3 -c "
import sys; sys.path.insert(0, 'testbed/scripts'); import link_to_json as lj
a, b, r = lj.parse('$OUT/link_$sid.log'); print('mbps=%s rtt_ms=%s' % (r['mbps'], r['rtt_ms']))" 2>&1 | tail -1) ($L)"
  for cid in $cells; do
    echo "----- $(date +%T) cell $cid"
    touch "$OUT/.mark_$cid"
    CELL=$cid PRED=$PRED bash testbed/scripts/run_pool.sh > "$OUT/$cid.console" 2>&1; echo "run_pool $cid exit $?"
    log=$(find testbed/results -maxdepth 1 -name "pool_${cid}_2*.log" -newer "$OUT/.mark_$cid" | sort | tail -1)
    if [ -n "$log" ]; then
      LOGS+=" $log"; grep -aE "^(THROTTLE|EXIT_BENCH|CELL DONE|REFUSED|FATAL)|!!!!!" "$log" | cut -c1-220
    else echo "!!!!! $cid wrote no log (see $OUT/$cid.console)"; fi
  done
  if [ "$rate" != 0 ] || [ "$delay" != 0 ]; then bash testbed/scripts/netshape.sh clear || exit 1; fi
done
echo "===== $(date +%T) score"
python3 testbed/scripts/compare_pool.py --pred "$PRED" --out testbed/dataset/pool_results_20261008netshape_r2.json $LOGS | cut -c1-260
python3 testbed/scripts/score_netshape_20261008.py "$OUT"
