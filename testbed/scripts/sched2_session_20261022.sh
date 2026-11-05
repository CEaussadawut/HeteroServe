#!/bin/bash
# Scheduler v2 against v1 on all four nodes, with energy at the wall (2026-10-22; the user: "try your very best to develop further
# at huge improvement of the scheduler"), as registered in testbed/dataset/sched2_predictions_20261022.json:
#   A  v1: the engine's pick for Qwen3-32B Q8_0, served by llama.cpp at the pin (serve_pool.sh, the registered PLACE);
#   V  v2: plan_precision's pick (the fastest official variant at or above the user's near-lossless floor), served the same way.
# ctx 2048, in the order A, V, A, V, each by qr3_tpot.py, inside one 5 Hz strip log of all four outlets with an idle window
# first and last. Before each arm-run: a quiet laptop (no working apps, no battery charging, Wi-Fi off, 1-min load < 1.5), the
# Pi 5 below 50 C and the Pi 4 below 55 C (up to 10 min each); the Jetson's power guard runs for the whole session, by its
# NOTE's procedure. From the repo root, under systemd-run + systemd-inhibit:
#     systemd-run --user --unit heteroserve-sched2 --collect -p WorkingDirectory=$PWD systemd-inhibit --what=sleep:idle \
#         --why="HeteroServe scheduler v2 against v1" bash testbed/scripts/sched2_session_20261022.sh
# Output: testbed/results/sched2_<stamp>/ (session.log, arms.tsv, strip.csv); each arm-run's qr3 log in testbed/results.
set -u
cd "$(dirname "$0")/../.."
REG=testbed/dataset/sched2_predictions_20261022.json
OUT=testbed/results/sched2_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
exec > >(tee -i -a "$OUT/session.log") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
[ -f "$REG" ] && [ ! -w "$REG" ] || { echo "FATAL: $REG missing or writable (register first)"; exit 2; }
echo "########## $(date -Is) scheduler v2 against v1 on four nodes, with energy -> $OUT (registration $(sha256sum "$REG" | cut -c1-16))"
eval "$(python3 - "$REG" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
a, v = r["arms"]["A"], r["arms"]["V"]
print('PLA=%s PHA=%s MODA=%s PLV=%s PHV=%s MODV=%s REQS=%d GEN=%d N=%d CTX=%d MEMFILE=%s' % (
    a["place_env"]["PLACE"], a["place_env"]["PLACE_HEAD"], a["model"], v["place_env"]["PLACE"], v["place_env"]["PLACE_HEAD"],
    v["model"], r["reqs"], r["gen"], r["n"], r["ctx"], r["memfile"]))
EOF
)"
echo "arm A: $MODA PLACE=$PLA PLACE_HEAD=$PHA; arm V: $MODV PLACE=$PLV PLACE_HEAD=$PHV (MEMFILE $MEMFILE); $REQS requests of $N + $GEN per arm-run, ctx $CTX"
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "FATAL: serve_pool.sh state exists"; exit 1; }
for n in pi5 pi4 jetson; do
  [ "$($SSH $n 'timedatectl show -p NTPSynchronized --value 2>/dev/null || { timedatectl | grep -q "synchronized: yes" && echo yes; }' 2>/dev/null)" = yes ] \
    || { echo "FATAL: $n's clock is not synchronized"; exit 1; }
done
APPS='^(zen|antigravity-ide|notion-snap-reb)$'
quiet() {   # as fe4_chain_20261020.sh: the laptop is a node in both arms
  local ok=0 t0=$(date +%s) a l c w
  while :; do
    a=$(ps -eo comm= | grep -E "$APPS" | sort -u | tr '\n' ' '); l=$(cut -d' ' -f1 /proc/loadavg)
    c=$(grep -l '^Charging$' /sys/class/power_supply/BAT*/status 2>/dev/null | wc -l); w=$(nmcli radio wifi 2>/dev/null)
    if [ -z "$a" ] && [ "$c" = 0 ] && [ "$w" != enabled ] && awk -v l="$l" 'BEGIN {exit !(l < 1.5)}'; then ok=$((ok + 1)); else ok=0; fi
    [ $ok -ge 3 ] && { echo "quiet   laptop: load $l, no battery charging, wifi $w"; return 0; }
    [ $(( $(date +%s) - t0 )) -ge 7200 ] && { echo "NOT QUIET after 2 h (apps [$a] load $l charging $c wifi $w): going on, recorded"; return 0; }
    sleep 30
  done
}
cold() {   # node, limit in C
  local t i
  for i in $(seq 60); do
    t=$($SSH "$1" "vcgencmd measure_temp" 2>/dev/null | tr -dc '0-9.')
    awk -v t="$t" -v l="$2" 'BEGIN { exit !(t < l) }' && { echo "cold    $1 at $t C (< $2) after $(( (i - 1) * 10 )) s"; return 0; }
    sleep 10
  done
  echo "!!!!! $1 still at $t C after 10 min; starting anyway, recorded"
}
g=$($SSH jetson 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum ~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1' 2>/dev/null | tr '\n' ' ')
[ "$g" = "synced $(sha256sum testbed/scripts/jetson_power_guard.sh | cut -d' ' -f1) " ] || { echo "FATAL: Jetson guard precheck '$g'"; exit 1; }
$SSH jetson 'sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo guard started' 2>/dev/null
quiet
python3 testbed/scripts/shelly_strip.py log --out "$OUT/strip.csv" --hz 5 > "$OUT/strip_logger.stdout" 2> "$OUT/strip_logger.stderr" &
LPID=$!
sleep 20
kill -0 $LPID 2>/dev/null || { echo "FATAL: the strip logger exited"; cat "$OUT/strip_logger.stderr"; exit 1; }
echo "IDLE_BEFORE start $(date +%s.%N)"; sleep 180; echo "IDLE_BEFORE end $(date +%s.%N)"
: > "$OUT/arms.tsv"
i=0
for arm in A V A V; do
  i=$((i + 1))
  echo "===== $(date -Is) arm-run $i: $arm"
  quiet; cold pi5 50; cold pi4 55
  if [ $arm = A ]; then PL=$PLA; PH=$PHA; MOD=$MODA; else PL=$PLV; PH=$PHV; MOD=$MODV; fi
  PLACE="$PL" PLACE_HEAD="$PH" python3 testbed/scripts/qr3_tpot.py run --model "$MOD" --ctx "$CTX" --nodes "pi5 pi4 jetson laptop" \
    --mem "$MEMFILE" --reqs "$REQS" --n "$N" --gen "$GEN" --tag "s2${arm}${i}"
  rc=$?
  lg=$(ls -t testbed/results/qr3_s2${arm}${i}_${MOD}_*.log 2>/dev/null | grep -v '_server\|_memprobe' | head -1)
  printf '%s\t%s\t%s\t%s\n' "$i" "$arm" "$rc" "$lg" >> "$OUT/arms.tsv"
  echo "ARM $i $arm rc=$rc log=$lg"
  sleep 20
done
echo "IDLE_AFTER start $(date +%s.%N)"; sleep 60; echo "IDLE_AFTER end $(date +%s.%N)"
kill -INT $LPID; wait $LPID 2>/dev/null
$SSH jetson 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p stopped"; done; echo "TRIP lines: $(grep -c TRIP ~/heteroserve/logs/power-guard.log)"' 2>/dev/null
python3 testbed/scripts/score_sched2_20261022.py "$OUT" --registration "$REG" --quality "$(ls -t testbed/results/sched2_quality_*.log | head -1)"
echo "SESSION DONE $(date -Is)"
