#!/bin/bash
# The four-node full evaluation (2026-10-20), run in its registration order (register_fe4_20261020.sh): for each model,
# prechecks, the Jetson's guard by its NOTE's procedure, full_eval.py run / summary, score_full_eval.py, the guard stopped.
# As run_14b.sh (2026-10-09), for every model. From the repo root, under systemd-run + systemd-inhibit:
#     systemd-run --user --unit heteroserve-fe4 --collect -p WorkingDirectory=$PWD systemd-inhibit --what=sleep:idle \
#         --why="HeteroServe four-node full evaluation" bash testbed/scripts/fe4_chain_20261020.sh
# NAMES="..." runs a subset, in the given order. A model already run (a run directory with score.json) is skipped.
# Between models it waits for a quiet laptop (no working apps, 1-minute load under 1.5, three polls 30 s apart; at most
# QUIET_H hours, default 24): the laptop is a node, so the user's own use of it would land in its windows.
# Log: testbed/results/fe4_chain_<stamp>.log.
set -u
cd "$(dirname "$0")/../.."
F=testbed/scripts/full_eval.py
LOG=testbed/results/fe4_chain_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
OUTLETS=pi4=0,jetson=1,pi5=2,laptop=3
NAMES=${NAMES:-"q14b_fe4_20261020 moeq6_fe4_20261020 moeq3_fe4_20261020 q27b_fe4_20261020 nemo_fe4_20261020 q9b_fe4_20261020 tinyq4km_fe4_20261020 tinyq8_fe4_20261020 tinyq40_fe4_20261020 tinyq3_fe4_20261020 tinyq6_fe4_20261020"}
QUIET_H=${QUIET_H:-24}
RERUN=${RERUN:-}   # 2026-10-21: names run again even with a scored run (the 27B, whose first run lost two placements to a split check)
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
APPS='^(zen|antigravity-ide|notion-snap-reb)$'
stop_guard() {
  $SSH jetson 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p stopped"; done; read -t 2 <> <(:); echo "guard procs: $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l)"' 2>/dev/null
}
start_guard() {
  local g want
  g=$($SSH jetson 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum ~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1' 2>/dev/null | tr '\n' ' ')
  want=$(sha256sum testbed/scripts/jetson_power_guard.sh | cut -d' ' -f1)
  [ "$g" = "synced $want " ] || { echo "FATAL: guard precheck '$g'"; return 1; }
  $SSH jetson 'sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo started' 2>/dev/null
  $SSH jetson 'read -t 4 <> <(:); echo "guard procs: $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l)"' 2>/dev/null
}
precheck() {
  ip route get 192.168.1.114 | grep -q " via " && { echo "FATAL: the testbed LAN is not directly connected"; return 1; }
  for h in 192.168.1.114 192.168.1.110 192.168.1.122; do ping -c 1 -W 2 $h >/dev/null 2>&1 || { echo "FATAL: $h does not answer"; return 1; }; done
  python3 testbed/scripts/find_strip.py --fix > /dev/null || { echo "FATAL: the strip does not answer"; return 1; }
  [ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "FATAL: serve_pool.sh state exists"; return 1; }
  bash testbed/scripts/netshape.sh check > /dev/null || { echo "FATAL: the Pi 4 is shaped"; return 1; }
  for n in pi5 pi4 jetson; do
    # the Jetson's systemd 237 (Ubuntu 18.04) has no "timedatectl show": its plain output says "synchronized: yes" (run_14b.sh)
    [ "$($SSH $n 'timedatectl show -p NTPSynchronized --value 2>/dev/null || { timedatectl | grep -q "synchronized: yes" && echo yes; }' 2>/dev/null)" = yes ] \
      || { echo "FATAL: $n's clock is not synchronized"; return 1; }
    b=$($SSH $n 'for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p; done | wc -l' 2>/dev/null)
    [ "$b" = 0 ] || { echo "FATAL: $n is busy ($b llama.cpp processes)"; return 1; }
  done
  [ "$($SSH pi5 'nmcli radio wifi')" = disabled ] || { echo "FATAL: the Pi 5's Wi-Fi is on"; return 1; }
  echo "net: laptop wifi $(nmcli radio wifi 2>/dev/null); the Pi 4 resolves .108 to $($SSH pi4 'ping -c1 -W1 192.168.1.108 >/dev/null; ip neigh show 192.168.1.108' 2>/dev/null | awk '{print $5}') (wired 48:2a:e3:2b:44:9d); AC $(cat /sys/class/power_supply/AC/online) BAT $(cat /sys/class/power_supply/BAT0/status)/$(cat /sys/class/power_supply/BAT1/status 2>/dev/null)"
  for n in pi5 pi4 jetson; do
    t0=$(date +%s.%N); r=$($SSH $n 'date +%s.%N' 2>/dev/null); t1=$(date +%s.%N)
    python3 -c "print('clock offset $n: %+.3f s (ssh round trip %.3f s)' % ($r - ($t0 + $t1) / 2, $t1 - $t0))"
  done
}
charging() { grep -l '^Charging$' /sys/class/power_supply/BAT*/status 2>/dev/null | wc -l; }   # 2026-10-20 21:20
wifi() { nmcli radio wifi 2>/dev/null; }   # the laptop's Wi-Fi on the boards' subnet beside its Ethernet: board traffic can use it
quiet() {   # also waits while a battery charges (full_eval.py refuses then: outlet 3 would meter the charge) or the Wi-Fi is on
  local ok=0 t0=$(date +%s) a l c w
  while :; do
    a=$(ps -eo comm= | grep -E "$APPS" | sort -u | tr '\n' ' '); l=$(cut -d' ' -f1 /proc/loadavg); c=$(charging); w=$(wifi)
    if [ -z "$a" ] && [ "$c" = 0 ] && [ "$w" != enabled ] && awk -v l="$l" 'BEGIN {exit !(l < 1.5)}'; then ok=$((ok + 1)); else ok=0; fi
    [ $ok -ge 3 ] && { echo "quiet: load $l, no battery charging, wifi $w"; return 0; }
    [ $(( $(date +%s) - t0 )) -ge $((QUIET_H * 3600)) ] && { echo "NOT QUIET after ${QUIET_H} h (apps [$a] load $l charging $c wifi $w): going on, recorded"; return 0; }
    sleep 30
  done
}
echo "########## $(date -Is) four-node full evaluation: $NAMES"
for name in $NAMES; do
  reg=testbed/dataset/full_eval_$name.json
  [ -f "$reg" ] || { echo "SKIP $name: not registered"; continue; }
  done_dir=$(ls -d testbed/results/full_eval_${name}_* 2>/dev/null | while read d; do [ -f "$d/score.json" ] && echo "$d"; done | tail -1)
  if [ -n "$done_dir" ] && [[ " $RERUN " != *" $name "* ]]; then echo "SKIP $name: already run ($done_dir)"; continue; fi
  [ -n "$done_dir" ] && echo "RERUN $name (a scored run exists: $done_dir)"
  echo "===== $(date -Is) $name (registration $(sha256sum "$reg" | cut -c1-16))"
  quiet
  precheck || { echo "STOPPED before $name: precheck failed"; break; }
  start_guard || { echo "STOPPED before $name: the guard did not start"; break; }
  python3 $F run "$reg" --outlets $OUTLETS; rc=$?
  RD=$(ls -dt testbed/results/full_eval_${name}_* | head -1)
  echo "run exit $rc: $RD"
  if [ "$rc" != 0 ]; then   # 2026-10-20 21:20: a run that could not start (a FATAL) stops the chain, not the next model
    stop_guard; echo "STOPPED after $name: full_eval.py run exit $rc"; break
  fi
  stop_guard
  $SSH jetson 'grep -c TRIP ~/heteroserve/logs/power-guard.log; tail -n 1 ~/heteroserve/logs/power-guard.log' 2>/dev/null
  python3 $F summary "$RD"; echo "summary exit $?"
  python3 testbed/scripts/score_full_eval.py "$RD"; echo "score exit $?"
  echo "MODEL DONE $name rc=$rc $(date -Is)"
done
echo "CHAIN DONE $(date -Is)"
