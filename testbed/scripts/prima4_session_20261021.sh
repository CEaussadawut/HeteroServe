#!/bin/bash
# The engine against prima.cpp + Halda on all four nodes, with energy at the wall (2026-10-21; user: "I want a model that will do
# with prima cpp and halda to our engine and it use full nodes"), as registered in testbed/dataset/prima4_predictions_20261021.json:
#   A  the engine's plan, served by llama.cpp at the pin (serve_pool.sh with the registered PLACE, the laptop as host, the
#      cluster.json 8 GiB desktop reserve);
#   B  prima.cpp + Halda (its own decision at start), ring laptop (rank 0, the head) -> pi5 -> pi4 -> jetson on the home LAN,
#      the boards on their partial copies (prima_partial_20261021.py manifest, checked range by range; the windows checked
#      against the copies before any request).
# Qwen3-32B Q8_0, ctx 2048 (prima.cpp's Halda has no in-memory plan at 4096 and its disk-offload path asserts: 2026-10-21 21:05),
# in the order A, B, A, B, each by qr3_tpot.py, inside one 5 Hz strip log of all four outlets with an idle window first and last.
# Before each arm-run: a quiet laptop (no working apps, no battery charging, Wi-Fi off, 1-min load < 1.5), the Pi 5 below 50 C
# and the Pi 4 below 55 C (up to 10 min each); the Jetson's power guard runs for the whole session, by its NOTE's procedure.
# From the repo root, under systemd-run + systemd-inhibit:
#     systemd-run --user --unit heteroserve-prima4 --collect -p WorkingDirectory=$PWD systemd-inhibit --what=sleep:idle \
#         --why="HeteroServe engine vs prima.cpp, four nodes" bash testbed/scripts/prima4_session_20261021.sh
# Output: testbed/results/prima4_<stamp>/ (session.log, arms.tsv, strip.csv); each arm-run's qr3 log in testbed/results.
# 2026-10-23 (the laptop leaves at 08:00; tonight it rebooted on battery, off the wire): a registration may carry
#   order     the arm-runs' order (default A B A B);
#   deadline  {epoch, arm_min}: an arm-run starts only if it can end, at its allowed minutes, before the deadline; otherwise the
#             session ends there, as at its normal end (idle-after window, the guard stopped, scored);
#   reserve   HETEROSERVE_RESERVE_MB for arm A's plan and serve_pool.sh's gate (the user's 7 GiB for Qwen3-32B runs).
# The laptop must also be on its charger (AC online) at every quiet check; off it after the wait, the session ends there. The
# battery is logged every 10 s (battery.tsv); off the charger below 25 %, the logger stops the session's unit.
set -u
cd "$(dirname "$0")/../.."
REG=${REG:-testbed/dataset/prima4_predictions_20261021.json}   # REG= a superseding registration (2026-10-22: prima4b)
OUT=testbed/results/prima4_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
exec > >(tee -i -a "$OUT/session.log") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
[ -f "$REG" ] && [ ! -w "$REG" ] || { echo "FATAL: $REG missing or writable (register first)"; exit 2; }
echo "########## $(date -Is) the engine against prima.cpp on four nodes, with energy -> $OUT (registration $(sha256sum "$REG" | cut -c1-16))"
eval "$(python3 - "$REG" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
a = r["arms"]["A"]
print('PL=%s PH=%s REQS=%d GEN=%d N=%d CTX=%d MEMFILE=%s COPIES=%s MODEL=%s OUTLETS=%s' % (
    a["place_env"]["PLACE"], a["place_env"]["PLACE_HEAD"], r["reqs"], r["gen"], r["n"], r["ctx"], r["memfile"],
    r["arms"]["B"]["copies"], r["model"], ",".join("%s=%d" % kv for kv in r["outlets"].items())))
d = r.get("deadline") or {}
print('ORDER="%s" DEADLINE=%d ARM_MIN_A=%d ARM_MIN_B=%d RESERVE="%s"' % (
    " ".join(r.get("order", ["A", "B", "A", "B"])), d.get("epoch", 0), d.get("arm_min", {}).get("A", 0),
    d.get("arm_min", {}).get("B", 0), r.get("reserve_mb", "")))
EOF
)"
echo "arm A: PLACE=$PL PLACE_HEAD=$PH (MEMFILE $MEMFILE); arm B: prima.cpp, copies $COPIES; $REQS requests of $N + $GEN per arm-run, ctx $CTX"
echo "order $ORDER; deadline $([ "$DEADLINE" -gt 0 ] && date -d @$DEADLINE -Is || echo none) (A $ARM_MIN_A min, B $ARM_MIN_B min); reserve '${RESERVE}'"
export HETEROSERVE_RESERVE_MB="$RESERVE"
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "FATAL: serve_pool.sh state exists"; exit 1; }
for n in pi5 pi4 jetson; do
  [ "$($SSH $n 'timedatectl show -p NTPSynchronized --value 2>/dev/null || { timedatectl | grep -q "synchronized: yes" && echo yes; }' 2>/dev/null)" = yes ] \
    || { echo "FATAL: $n's clock is not synchronized"; exit 1; }
done
APPS='^(zen|antigravity-ide|notion-snap-reb)$'
quiet() {   # as fe4_chain_20261020.sh: the laptop is a node in both arms; since 2026-10-23 also on its charger
  local ok=0 t0=$(date +%s) a l c w ac
  while :; do
    a=$(ps -eo comm= | grep -E "$APPS" | sort -u | tr '\n' ' '); l=$(cut -d' ' -f1 /proc/loadavg)
    c=$(grep -l '^Charging$' /sys/class/power_supply/BAT*/status 2>/dev/null | wc -l); w=$(nmcli radio wifi 2>/dev/null)
    ac=$(cat /sys/class/power_supply/AC/online 2>/dev/null)
    if [ -z "$a" ] && [ "$c" = 0 ] && [ "$w" != enabled ] && [ "$ac" = 1 ] && awk -v l="$l" 'BEGIN {exit !(l < 1.5)}'; then ok=$((ok + 1)); else ok=0; fi
    [ $ok -ge 3 ] && { echo "quiet   laptop: load $l, on the charger, no battery charging, wifi $w"; return 0; }
    if [ $(( $(date +%s) - t0 )) -ge 7200 ]; then
      [ "$ac" = 1 ] || { echo "NOT ON THE CHARGER after 2 h (apps [$a] load $l charging $c wifi $w)"; return 1; }
      echo "NOT QUIET after 2 h (apps [$a] load $l charging $c wifi $w): going on, recorded"; return 0
    fi
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
guard_stop() {
  $SSH jetson 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p stopped"; done; echo "TRIP lines: $(grep -c TRIP ~/heteroserve/logs/power-guard.log)"' 2>/dev/null
}
quiet || { echo "FATAL: the laptop is not on its charger"; guard_stop; exit 1; }
# the laptop's battery every 10 s (epoch, AC online, BAT0 status, %, power_now uW, energy_now uWh); off the charger below 25 %
# it stops this session's unit (a heteroserve-* service only), so the laptop is never run flat
UNIT=$(sed -n 's|^0::.*/\(heteroserve-[^/]*\)\.service$|\1|p' /proc/self/cgroup)
( b=/sys/class/power_supply/BAT0
  while :; do
    ac=$(cat /sys/class/power_supply/AC/online 2>/dev/null); cap=$(cat $b/capacity 2>/dev/null)
    echo "$(date +%s) $ac $(cat $b/status) $cap $(cat $b/power_now) $(cat $b/energy_now)"
    if [ "$ac" != 1 ] && [ "${cap:-100}" -lt 25 ]; then
      echo "LOW BATTERY off the charger ($cap %): stopping unit '${UNIT}'"
      [ -n "$UNIT" ] && systemctl --user stop --no-block "$UNIT.service"
    fi
    sleep 10
  done ) > "$OUT/battery.tsv" 2>&1 &
BPID=$!
python3 testbed/scripts/shelly_strip.py log --out "$OUT/strip.csv" --hz 5 > "$OUT/strip_logger.stdout" 2> "$OUT/strip_logger.stderr" &
LPID=$!
sleep 20
kill -0 $LPID 2>/dev/null || { echo "FATAL: the strip logger exited"; cat "$OUT/strip_logger.stderr"; kill $BPID; guard_stop; exit 1; }
echo "IDLE_BEFORE start $(date +%s.%N)"; sleep 180; echo "IDLE_BEFORE end $(date +%s.%N)"
: > "$OUT/arms.tsv"
i=0
late() {   # arm, i: true when the arm-run, at its allowed minutes, could not end before the deadline (checked before the waits too)
  local m=$([ $1 = A ] && echo $ARM_MIN_A || echo $ARM_MIN_B)
  [ "$DEADLINE" -gt 0 ] && [ $(( $(date +%s) + m * 60 )) -gt "$DEADLINE" ] || return 1
  echo "DEADLINE arm-run $2 ($1, allowed $m min) could not end by $(date -d @$DEADLINE +%T): not started; the session ends here"
}
for arm in $ORDER; do
  i=$((i + 1))
  echo "===== $(date -Is) arm-run $i: $arm"
  late $arm $i && break
  quiet || { echo "NOT ON THE CHARGER: the session ends before arm-run $i"; break; }
  cold pi5 50; cold pi4 55
  late $arm $i && break
  if [ $arm = A ]; then
    PLACE="$PL" PLACE_HEAD="$PH" python3 testbed/scripts/qr3_tpot.py run --model "$MODEL" --ctx "$CTX" --nodes "pi5 pi4 jetson laptop" \
      --mem "$MEMFILE" --reqs "$REQS" --n "$N" --gen "$GEN" --tag "p4${arm}${i}"
  else
    python3 testbed/scripts/qr3_tpot.py run --model "$MODEL" --ctx "$CTX" --engine prima --prima-ranks "laptop pi5 pi4 jetson" \
      --prima-copies "$COPIES" --reqs "$REQS" --n "$N" --gen "$GEN" --tag "p4${arm}${i}"
  fi
  rc=$?
  lg=$(ls -t testbed/results/qr3_p4${arm}${i}_${MODEL}_*.log 2>/dev/null | grep -v '_prima_rank\|_server\|_memprobe' | head -1)
  printf '%s\t%s\t%s\t%s\n' "$i" "$arm" "$rc" "$lg" >> "$OUT/arms.tsv"
  echo "ARM $i $arm rc=$rc log=$lg"
  sleep 20
done
echo "IDLE_AFTER start $(date +%s.%N)"; sleep 60; echo "IDLE_AFTER end $(date +%s.%N)"
kill -INT $LPID; wait $LPID 2>/dev/null
kill $BPID 2>/dev/null
guard_stop
python3 testbed/scripts/score_prima4_20261021.py "$OUT" --registration "$REG"
echo "SESSION DONE $(date -Is)"
