#!/bin/bash
# energy_mapcheck_idle.sh — HeteroServe energy, the first run at home (testbed/energy.md, "First run at home", 1–2).
#
# Phase A, the outlet-map check by load. The strip is logged at 5 Hz for 330 s while one board at a time runs a
# self-terminating CPU busy loop, 30 s apart: Pi 5 4 cores 60 s, Pi 4 4 cores 60 s, Jetson 2 cores 45 s (the Jetson
# runs MAXN from micro-USB: light load only, guard running). Only a load proves which board is behind which outlet.
# Phase B, the idle floors. 60 s settle, then 360 s at 5 Hz with nothing started on any board.
#
# The strip is only read (shelly_strip.py is read-only by construction). Each busy loop is bounded by `timeout` on the
# board itself, so a dropped ssh cannot leave it running. Every board's state (throttle flags, temperature, load average;
# the Jetson's power mode and guard) is logged around each phase. markers.tsv holds laptop-clock epoch seconds, the same
# clock as the strip log's wall_time. Nothing is filtered: every tool's output goes to a file in the run folder.
#
# Usage: testbed/scripts/energy_mapcheck_idle.sh [OUTDIR]   (default testbed/results/energy_mapcheck_idle_<stamp>)
# Score: python3 testbed/scripts/score_energy_mapcheck_idle.py OUTDIR
set -u
cd "$(dirname "$0")/../.." || exit 2   # the project root
D=${1:-testbed/results/energy_mapcheck_idle_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$D" || exit 2
S="python3 testbed/scripts/shelly_strip.py"
M="$D/markers.tsv"
LOG="$D/run.log"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5"

quiet() { grep -v -E 'post-quantum|store now, decrypt later|may need to be upgraded'; }   # the Jetson's old sshd banner
mark() { printf '%s\t%s\n' "$(date +%s.%3N)" "$*" >> "$M"; echo "$(date +%T) $*" >> "$LOG"; }
pistate() { $SSH -n "$1" 'echo "$(date +%T.%3N) $(hostname) $(vcgencmd get_throttled) $(vcgencmd measure_temp) loadavg=$(cut -d" " -f1-3 /proc/loadavg)"' 2>&1 | quiet; }
jstate() { $SSH -n jetson 'echo "$(date +%T.%3N) $(hostname) $(cat /var/lib/nvpmodel/status) loadavg=$(cut -d" " -f1-3 /proc/loadavg) guard=$(pgrep -f jetson_power_guard.sh >/dev/null && echo up || echo DOWN) thermal_mC=$(cat /sys/class/thermal/thermal_zone*/temp | tr "\n" " ")"; tail -n 1 ~/heteroserve/logs/power-guard.log' 2>&1 | quiet; }
states() { { pistate pi5; pistate pi4; jstate; } > "$D/$1"; }
load() {  # host cores seconds: that many self-terminating busy loops on the board
  $SSH "$1" "bash -s $2 $3" 2>&1 <<'EOF' | quiet
n=$1; secs=$2; s=$(date +%s.%3N)
for i in $(seq "$n"); do timeout "$secs" bash -c 'while :; do :; done' & done
wait
e=$(date +%s.%3N)
echo "board clock: start $s end $e ran $(awk -v s="$s" -v e="$e" 'BEGIN{printf "%.3f", e-s}') s, $n busy loops of $secs s"
echo "busy loops left: $(pgrep -fc 'while :; do :; done' || true)"
EOF
}

mark "run start, out $D"
$S info > "$D/strip_info_before.txt" 2>&1
states state_A_before.txt

# Phase A: the outlet-map check
mark "A: logger start"
$S log --out "$D/mapcheck.csv" --hz 5 --duration 330 2> "$D/mapcheck_logger.stderr" &
LP=$!
sleep 30
for spec in "pi5 4 60" "pi4 4 60" "jetson 2 45"; do
  set -- $spec
  mark "A: load start $1 cores=$2 secs=$3"
  load "$1" "$2" "$3" > "$D/load_$1.out"
  mark "A: load end $1"
  if [ "$1" = jetson ]; then jstate; else pistate "$1"; fi >> "$D/state_A_after_loads.txt"
  [ "$1" != jetson ] && sleep 30
done
wait "$LP"; mark "A: logger exit $?"

# Phase B: the idle floors
mark "B: settle 60 s"
sleep 60
states state_B_before.txt
mark "B: logger start"
$S log --out "$D/idle.csv" --hz 5 --duration 360 2> "$D/idle_logger.stderr"
mark "B: logger exit $?"
states state_B_after.txt
$S info > "$D/strip_info_after.txt" 2>&1

# The Jetson guard's lines since its latest start: its 1 Hz INA3221 input rail is the DC cross-check for outlet 2.
$SSH -n jetson 'f=~/heteroserve/logs/power-guard.log; n=$(grep -a -n "guard start" "$f" | tail -n 1 | cut -d: -f1); tail -n +"$n" "$f"' 2>&1 | quiet > "$D/jetson_power_guard.log"
mark "run end"
