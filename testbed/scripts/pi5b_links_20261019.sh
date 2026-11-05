#!/bin/bash
# The replacement Pi 5's links (2026-10-19), on each pair's pooled path: Pi 5 <-> Pi 4 (the overlay), Pi 5 <-> Jetson (the
# router LAN), Pi 5 <-> laptop (the overlay: the laptop's Wi-Fi is on 192.168.1.x since 10-19; measure_link.sh says why).
# The Jetson's power guard is started by its NOTE's procedure (.handoff/nodes/Jetson/2026-10-06-1105-...) for the Jetson
# pair and stopped afterwards, as run_14b.sh does. From the repo root on the laptop, with the cluster idle:
#     bash testbed/scripts/pi5b_links_20261019.sh
# Output: each pair's testbed/results/link_<a>-<b>_<stamp>.log (measure_link.sh tees it) and this session's log.
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=testbed/results/pi5b_links_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -i -a "$LOG") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
stop_guard() {
  $SSH jetson 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p stopped"; done; read -t 2 <> <(:); echo "guard procs: $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l)"' 2>/dev/null
}
echo "########## $(date -Is) the replacement Pi 5's links -> $LOG"
PAIR="pi5 pi4" bash testbed/scripts/measure_link.sh; echo "PAIR pi5-pi4 rc=$?"

echo "===== $(date +%T) the Jetson's guard, by its NOTE's procedure"
g=$($SSH jetson 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum ~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1' 2>/dev/null | tr '\n' ' ')
want=$(sha256sum testbed/scripts/jetson_power_guard.sh | cut -d' ' -f1)
if [ "$g" = "synced $want " ]; then
  $SSH jetson 'sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo started' 2>/dev/null
  $SSH jetson 'read -t 4 <> <(:); echo "guard procs: $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l)"' 2>/dev/null
  PAIR="pi5 jetson" bash testbed/scripts/measure_link.sh; echo "PAIR pi5-jetson rc=$?"
  echo "===== $(date +%T) stop the Jetson's guard"; stop_guard
  $SSH jetson 'grep -c TRIP ~/heteroserve/logs/power-guard.log; tail -n 2 ~/heteroserve/logs/power-guard.log' 2>/dev/null
else
  echo "SKIPPED pi5-jetson: guard precheck '$g' (want 'synced $want')"
fi

PAIR="pi5 laptop" bash testbed/scripts/measure_link.sh; echo "PAIR pi5-laptop rc=$?"
echo "LINKS DONE $(date -Is)"
