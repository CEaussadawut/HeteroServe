#!/bin/bash
# The laptop's own measurements, re-armed for the night of 2026-10-20 (the 2026-10-11 session, laptop_session_20261011.sh,
# waited 14 h for a quiet laptop and never ran). The user, 2026-10-20 ~00:05: "Overnight from ~02:45" -- every app closed but
# the Claude app, the charger in, the laptop's Wi-Fi off, and no messages until the alarm. From the repo root:
#     systemd-run --user --unit heteroserve-lapsession --collect -p WorkingDirectory=$PWD \
#         systemd-inhibit --what=sleep:idle --why="HeteroServe laptop session" bash testbed/scripts/laptop_session_20261020.sh
# Quiet = none of the user's working apps (the Zen browser, the Antigravity IDE, Notion, a Vite dev server) and the 1-minute
# load under 1.0 at six polls in a row, 30 s apart. Then, each step's exit logged:
#   1. the laptop's MemAvailable into memory_at_rest.json (laptop_node_datasets_20261011.py memory; it keeps the 09-27 copy);
#   2. its worker rows at five quantisations, the Pi 4 as host (laptop_worker_quants_20261011.py);
#   3. its host rows at every quantisation, the characterisation protocol (run_t580_rows.sh): Q6_K is new, and Q4_K_M, Q8_0,
#      Q4_0 and Q3_K_M are re-taken -- the 2026-09-29 rows spread 6 % within a run and up to 61 % between runs (taken while
#      the desktop was in use; testbed/results/repeatability_20261020_001844), and the planner prices the laptop from them;
#   4. the same at the pooled protocol (PP=64 TG=16);
#   5. laptop_worker_c_20261011.py: the worker rows into testbed/dataset/laptop_worker_20261011c.json.
# Before and after: the laptop's Wi-Fi state, its interfaces on 192.168.1.x, and the MAC the Pi 4 resolves for 192.168.1.108
# (its worker address: a board that resolved it to the Wi-Fi card would send the RPC traffic over Wi-Fi), the batteries, the
# user timers. Log: testbed/results/laptop_session_<stamp>.log. Gives up after WAIT_H hours (default 6) without starting.
set -u
cd "$(dirname "$0")/../.."
LOG=testbed/results/laptop_session_$(date +%Y%m%d_%H%M%S).log
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
APPS='^(zen|antigravity-ide|notion-snap-reb)$'
busy_apps() { ps -eo comm= | grep -E "$APPS" | sort -u | tr '\n' ' '; ps -eo args= | grep -E '[n]ode_modules/\.bin/vite' > /dev/null && printf 'vite '; }
net() {
  say "NET $1: wifi radio $(nmcli radio wifi 2>/dev/null); on 192.168.1.x: $(ip -4 -o addr show | grep -F ' 192.168.1.' | awk '{print $2"="$4}' | tr '\n' ' ');" \
      "the Pi 4 resolves .108 to $(ssh -n -o BatchMode=yes -o ConnectTimeout=8 pi4 'ping -c1 -W1 192.168.1.108 >/dev/null; ip neigh show 192.168.1.108' 2>/dev/null | awk '{print $5, $6}')" \
      "(wired 48:2a:e3:2b:44:9d); batteries $(cat /sys/class/power_supply/BAT0/status)/$(cat /sys/class/power_supply/BAT1/status 2>/dev/null) AC $(cat /sys/class/power_supply/AC/online)"
}
WAIT_H=${WAIT_H:-6}
say "waiting for the laptop to be quiet (apps: $(busy_apps)load $(cut -d' ' -f1 /proc/loadavg))"
ok=0; t0=$(date +%s)
while :; do
  a=$(busy_apps); l=$(cut -d' ' -f1 /proc/loadavg)
  if [ -z "$a" ] && awk -v l="$l" 'BEGIN {exit !(l < 1.0)}'; then ok=$((ok + 1)); else ok=0; fi
  [ $ok -ge 6 ] && break
  [ $(( $(date +%s) - t0 )) -ge $((WAIT_H * 3600)) ] && { say "GAVE UP after ${WAIT_H} h: apps [$a] load $l"; exit 3; }
  sleep 30
done
say "quiet: no working apps, load $(cut -d' ' -f1 /proc/loadavg); MemAvailable $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) MiB"
net start
say "user timers: $(systemctl --user list-timers --no-legend 2>/dev/null | awk '{print $NF}' | tr '\n' ' ')"
python3 testbed/scripts/laptop_node_datasets_20261011.py memory >> "$LOG" 2>&1; say "STEP memory exit $?"
python3 testbed/scripts/laptop_worker_quants_20261011.py >> "$LOG" 2>&1; say "STEP worker_quants exit $?"
ALL="tinyllama-1.1b-q6_k.gguf tinyllama-1.1b-q4km.gguf tinyllama-1.1b-q8_0.gguf tinyllama-1.1b-q4_0.gguf tinyllama-1.1b-q3km.gguf"
MODELS="$ALL" bash testbed/scripts/run_t580_rows.sh >> "$LOG" 2>&1; say "STEP t580_char exit $?"
PP=64 TG=16 MODELS="$ALL" bash testbed/scripts/run_t580_rows.sh >> "$LOG" 2>&1; say "STEP t580_pool exit $?"
python3 testbed/scripts/laptop_worker_c_20261011.py >> "$LOG" 2>&1; say "STEP worker_c exit $?"
net end
say "SESSION DONE"
