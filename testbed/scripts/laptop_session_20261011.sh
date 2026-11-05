#!/bin/bash
# The laptop's own measurements for the laptop-as-node work (2026-10-11), started by themselves once the laptop is quiet:
# the user was asked at ~04:00 to close every app but the Claude app and leave the laptop alone; this waits for that.
#     systemd-run --user --unit heteroserve-lapsession --collect -p WorkingDirectory=$PWD \
#         systemd-inhibit --what=sleep:idle --why="HeteroServe laptop session" bash testbed/scripts/laptop_session_20261011.sh
# Quiet = none of the user's working apps running (the Zen browser, the Antigravity IDE, Notion, a Vite dev server) and the
# 1-minute load under 1.0 at six polls in a row (30 s apart). Then, in order, each step's exit logged:
#   1. the laptop's MemAvailable into memory_at_rest.json (laptop_node_datasets_20261011.py memory);
#   2. its worker rows at five quantisations, the Pi 4 as host (laptop_worker_quants_20261011.py);
#   3. its host rows at Q6_K, the characterisation and the pooled protocol (run_t580_rows.sh).
# Log: testbed/results/laptop_session_<stamp>.log. Gives up after WAIT_H hours (default 14) without starting anything.
set -u
cd "$(dirname "$0")/../.."
LOG=testbed/results/laptop_session_$(date +%Y%m%d_%H%M%S).log
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
APPS='^(zen|antigravity-ide|notion-snap-reb)$'
busy_apps() { ps -eo comm= | grep -E "$APPS" | sort -u | tr '\n' ' '; ps -eo args= | grep -E '[n]ode_modules/\.bin/vite' > /dev/null && printf 'vite '; }
WAIT_H=${WAIT_H:-14}
say "waiting for the laptop to be quiet (apps: $(busy_apps)load $(cut -d' ' -f1 /proc/loadavg))"
ok=0; t0=$(date +%s)
while :; do
  a=$(busy_apps); l=$(cut -d' ' -f1 /proc/loadavg)
  if [ -z "$a" ] && awk -v l="$l" 'BEGIN {exit !(l < 1.0)}'; then ok=$((ok + 1)); else ok=0; fi
  [ $ok -ge 6 ] && break
  [ $(( $(date +%s) - t0 )) -ge $((WAIT_H * 3600)) ] && { say "GAVE UP after ${WAIT_H} h: apps [$a] load $l"; exit 3; }
  sleep 30
done
say "quiet: no working apps, load $(cut -d' ' -f1 /proc/loadavg); MemAvailable $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) MiB; batteries $(cat /sys/class/power_supply/BAT0/status)/$(cat /sys/class/power_supply/BAT1/status 2>/dev/null)"
python3 testbed/scripts/laptop_node_datasets_20261011.py memory >> "$LOG" 2>&1; say "STEP memory exit $?"
python3 testbed/scripts/laptop_worker_quants_20261011.py >> "$LOG" 2>&1; say "STEP worker_quants exit $?"
MODELS=tinyllama-1.1b-q6_k.gguf bash testbed/scripts/run_t580_rows.sh >> "$LOG" 2>&1; say "STEP t580_char exit $?"
PP=64 TG=16 MODELS=tinyllama-1.1b-q6_k.gguf bash testbed/scripts/run_t580_rows.sh >> "$LOG" 2>&1; say "STEP t580_pool exit $?"
say "SESSION DONE"
