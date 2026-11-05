#!/bin/bash
# The TCP diagnostic's campaign (register_tcpdiag_20261009.py). From the repo root, on the testbed LAN:
#     bash testbed/scripts/tcpdiag_campaign_20261009.sh
# As netshape_campaign.sh: for each registered setting, shape the Pi 4 <-> Jetson path (netshape.sh; s0 stays unshaped),
# measure the link (measure_link.sh), run the setting's cells in their registered order (run_pool.sh: every gate, the memory
# probe and the throttle poll), and clear the shaping; the shaping is cleared on any exit, and the Jetson's guard is started
# by its NOTE's procedure first and stopped last. Around each cell, ss_sample.sh records the RPC connection's tcp_info on the
# Jetson (sport 50052) and on the Pi 4 (dport 50052), read-only, copied back as <out>/ss_<cell>_<board>.log.
# Then compare_pool.py and score_tcpdiag_20261009.py. Everything is tee'd to testbed/results/tcpdiag_<stamp>/campaign.log.
set -uo pipefail
cd "$(dirname "$0")/../.."
PRED=testbed/dataset/pool_predictions_20261009tcpdiag.json
[ -f "$PRED" ] || { echo "FATAL: $PRED is not registered (python3 testbed/scripts/register_tcpdiag_20261009.py)"; exit 1; }
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=testbed/results/tcpdiag_$STAMP
mkdir -p "$OUT"
exec > >(tee -i -a "$OUT/campaign.log") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
GUARD=0; SAMPLERS=""
stop_samplers() {   # $1 cell id: stop both samplers, copy their logs
  local kv n p
  for kv in $SAMPLERS; do n=${kv%%:*}; p=${kv#*:}; $SSH $n "kill $p 2>/dev/null"; done
  [ -n "${1:-}" ] && for kv in $SAMPLERS; do
    n=${kv%%:*}; scp -q $n:logs/ss_tcpdiag_${STAMP}_$1.log "$OUT/ss_$1_$n.log" || echo "!!!!! could not copy the $n sampler log for $1"
  done
  SAMPLERS=""
}
finish() {
  trap '' INT TERM HUP
  stop_samplers ""
  bash testbed/scripts/netshape.sh clear || echo "!!!!! the Pi 4 may still be shaped: run netshape.sh clear by hand"
  if [ "$GUARD" = 1 ]; then
    $SSH jetson 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p stopped"; done; read -t 2 <> <(:); echo "guard procs: $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l)"'
  fi
  echo "########## $(date -Is) campaign exit"
}
trap finish EXIT; trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 129' HUP

echo "########## $(date -Is) the TCP diagnostic, registration $PRED ($(sha256sum "$PRED" | cut -c1-16))"
ip route get 192.168.1.114 | grep -q " via " && { echo "FATAL: the testbed LAN is not directly connected"; exit 1; }
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "FATAL: serve_pool.sh state exists; stop it first"; exit 1; }
bash testbed/scripts/netshape.sh check || { echo "FATAL: the Pi 4 is shaped, or did not answer; see above"; exit 1; }
for n in pi4 jetson; do scp -q testbed/scripts/ss_sample.sh $n:ss_sample.sh || { echo "FATAL: could not copy ss_sample.sh to $n"; exit 1; }; done
echo "tcp settings: pi4 $($SSH pi4 'echo ssai=$(cat /proc/sys/net/ipv4/tcp_slow_start_after_idle) rmem=$(cat /proc/sys/net/ipv4/tcp_rmem | tr "\t" ,) wmem=$(cat /proc/sys/net/ipv4/tcp_wmem | tr "\t" ,) cc=$(cat /proc/sys/net/ipv4/tcp_congestion_control)')"
echo "tcp settings: jetson $($SSH jetson 'echo ssai=$(cat /proc/sys/net/ipv4/tcp_slow_start_after_idle) rmem=$(cat /proc/sys/net/ipv4/tcp_rmem | tr "\t" ,) wmem=$(cat /proc/sys/net/ipv4/tcp_wmem | tr "\t" ,) cc=$(cat /proc/sys/net/ipv4/tcp_congestion_control) qdisc=$(tc qdisc show dev eth0 | head -1 | cut -d" " -f2)' 2>/dev/null)"
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
  if [ "$rate" = 0 ]; then mib=512; else mib=$(( rate < 16 ? 16 : rate )); fi
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
    p=$($SSH jetson "mkdir -p ~/logs; setsid nohup bash ~/ss_sample.sh '( sport = :50052 )' ~/logs/ss_tcpdiag_${STAMP}_$cid.log 0.05 > /dev/null 2>&1 < /dev/null & echo \$!" 2>/dev/null)
    SAMPLERS+=" jetson:$p"
    p=$($SSH pi4 "mkdir -p ~/logs; setsid nohup bash ~/ss_sample.sh '( dport = :50052 )' ~/logs/ss_tcpdiag_${STAMP}_$cid.log 0.05 > /dev/null 2>&1 < /dev/null & echo \$!")
    SAMPLERS+=" pi4:$p"
    echo "samplers $SAMPLERS"
    touch "$OUT/.mark_$cid"
    CELL=$cid PRED=$PRED bash testbed/scripts/run_pool.sh > "$OUT/$cid.console" 2>&1; echo "run_pool $cid exit $?"
    stop_samplers "$cid"
    log=$(find testbed/results -maxdepth 1 -name "pool_${cid}_2*.log" -newer "$OUT/.mark_$cid" | sort | tail -1)
    if [ -n "$log" ]; then
      LOGS+=" $log"; cp "$log" "$OUT/"; grep -aE "^(THROTTLE|EXIT_BENCH|CELL DONE|REFUSED|FATAL)|!!!!!" "$log" | cut -c1-220
    else echo "!!!!! $cid wrote no log (see $OUT/$cid.console)"; fi
  done
  if [ "$rate" != 0 ] || [ "$delay" != 0 ]; then bash testbed/scripts/netshape.sh clear || exit 1; fi
done
echo "===== $(date +%T) score"
python3 testbed/scripts/compare_pool.py --pred "$PRED" --out testbed/dataset/pool_results_20261009tcpdiag.json $LOGS | cut -c1-260
python3 testbed/scripts/score_tcpdiag_20261009.py "$OUT"
