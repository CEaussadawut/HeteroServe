#!/bin/bash
# A registered A/B/... session on the four nodes, with energy at the wall (2026-10-24; the user, 2026-10-24: "Do all lefts"):
# spec4_session_20261023.sh (unchanged, hashed by its registrations) generalised. As there, each arm-run is one qr3_tpot.py
# run with the arm's PLACE / PLACE_HEAD and serve environment from the registration, inside one 5 Hz strip log of all four
# outlets with an idle window first and last, in the registered order, with the same gates: a quiet laptop on its charger
# (no working apps, no battery charging, Wi-Fi off, 1-min load < 1.5), the Pi 5 below 50 C and the Pi 4 below 55 C (up to 10
# min each), the battery logged every 10 s, a registered deadline, the Jetson's power guard for the whole session.
# New here, all optional per arm in the registration:
#   "pre":  a shell command run on the laptop just before the arm-run (e.g. a Jetson power mode, a link shaped); its output
#           is logged, and a non-zero exit skips the arm-run (recorded, never retried);
#   "post": a shell command run right after the arm-run, and for EVERY arm again on any way out (a trap), so a board is
#           never left in an arm's state;
#   "qr3_args": extra qr3_tpot.py arguments for this arm (e.g. "--clocks asfound");
#   "nodes": the boards serve_pool.sh may plan on (default "pi5 pi4 jetson laptop").
# The registration's "check" (optional) is a shell command run once before the guard starts; non-zero stops the session.
# From the repo root:
#     systemd-run --user --unit heteroserve-<name> --collect -p WorkingDirectory=$PWD --setenv=REG=<registration> \
#         systemd-inhibit --what=sleep:idle --why="HeteroServe <name>" bash testbed/scripts/ab_session_20261024.sh
# Output: testbed/results/<out_prefix>_<stamp>/ (session.log, arms.tsv, strip.csv, battery.tsv); each arm-run's qr3 log in
# testbed/results.
set -u
cd "$(dirname "$0")/../.."
REG=${REG:?set REG to the registration}
OUTP=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["out_prefix"])' "$REG" 2>/dev/null) || { echo "FATAL: no out_prefix in $REG"; exit 2; }
OUT=testbed/results/${OUTP}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
export OUT   # the arms' pre/post hooks may write beside the session (2026-10-24)
exec > >(tee -i -a "$OUT/session.log") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
[ -f "$REG" ] && [ ! -w "$REG" ] || { echo "FATAL: $REG missing or writable (register first)"; exit 2; }
echo "########## $(date -Is) $OUTP session on four nodes, with energy -> $OUT (registration $(sha256sum "$REG" | cut -c1-16))"
eval "$(python3 - "$REG" <<'PYEOF'
import json, shlex, sys
r = json.load(open(sys.argv[1]))
d = r.get("deadline") or {}
print("MODEL=%s REQS=%d GEN=%d N=%d CTX=%d MEMFILE=%s ORDER=%s DEADLINE=%d TAGP=%s SCORER=%s CHECK=%s" % (
    r["model"], r["reqs"], r["gen"], r["n"], r["ctx"], r["memfile"], shlex.quote(" ".join(r["order"])), d.get("epoch", 0),
    r["tag_prefix"], r["scorer"], shlex.quote(r.get("check", ""))))
for k, a in r["arms"].items():
    env = dict(a["place_env"], **a["serve_env"])
    print("ENV_%s=%s" % (k, shlex.quote(" ".join("%s=%s" % (x, shlex.quote(y)) for x, y in env.items()))))
    print("MIN_%s=%d" % (k, d.get("arm_min", {}).get(k, 0)))
    print("PRE_%s=%s" % (k, shlex.quote(a.get("pre", ""))))
    print("POST_%s=%s" % (k, shlex.quote(a.get("post", ""))))
    print("QARGS_%s=%s" % (k, shlex.quote(a.get("qr3_args", ""))))
    print("NODES_%s=%s" % (k, shlex.quote(" ".join(a.get("nodes", ["pi5", "pi4", "jetson", "laptop"])))))
PYEOF
)" || { echo "FATAL: could not read $REG"; exit 2; }
echo "model $MODEL; order $ORDER; deadline $([ "$DEADLINE" -gt 0 ] && date -d @$DEADLINE -Is || echo none); $REQS requests of $N + $GEN per arm-run, ctx $CTX (MEMFILE $MEMFILE)"
for k in $(printf '%s\n' $ORDER | sort -u); do eval "echo \"arm $k: \$ENV_$k\""; done   # 2026-10-23 15:50: was V and S by name
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "FATAL: serve_pool.sh state exists"; exit 1; }
for n in pi5 pi4 jetson; do
  [ "$($SSH $n 'timedatectl show -p NTPSynchronized --value 2>/dev/null || { timedatectl | grep -q "synchronized: yes" && echo yes; }' 2>/dev/null)" = yes ] \
    || { echo "FATAL: $n's clock is not synchronized"; exit 1; }
done
APPS='^(zen|antigravity-ide|notion-snap-reb)$'
quiet() {   # as prima4_session_20261021.sh (2026-10-23): the laptop is a node in both arms, and on its charger
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
late() {   # arm, i
  local m; eval "m=\$MIN_$1"
  [ "$DEADLINE" -gt 0 ] && [ $(( $(date +%s) + m * 60 )) -gt "$DEADLINE" ] || return 1
  echo "DEADLINE arm-run $2 ($1, allowed $m min) could not end by $(date -d @$DEADLINE +%T): not started; the session ends here"
}
guard_stop() {
  $SSH jetson 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p stopped"; done; echo "TRIP lines: $(grep -c TRIP ~/heteroserve/logs/power-guard.log)"' 2>/dev/null
}
g=$($SSH jetson 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum ~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1' 2>/dev/null | tr '\n' ' ')
[ "$g" = "synced $(sha256sum testbed/scripts/jetson_power_guard.sh | cut -d' ' -f1) " ] || { echo "FATAL: Jetson guard precheck '$g'"; exit 1; }
if [ -n "$CHECK" ]; then
  echo "CHECK: $CHECK"; bash -c "$CHECK" || { echo "FATAL: the registration's check failed"; exit 1; }
fi
restore_all() {   # every arm's post, on any way out: no board is left in an arm's state
  local k p
  for k in $(printf '%s\n' $ORDER | sort -u); do
    eval "p=\$POST_$k"; [ -n "$p" ] && { echo "RESTORE $k: $p"; bash -c "$p" 2>&1 | sed 's/^/    /'; }
  done
}
trap 'restore_all; guard_stop' EXIT
$SSH jetson 'sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo guard started' 2>/dev/null
quiet || { echo "FATAL: the laptop is not on its charger"; guard_stop; exit 1; }
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
for arm in $ORDER; do
  i=$((i + 1))
  echo "===== $(date -Is) arm-run $i: $arm"
  late $arm $i && break
  quiet || { echo "NOT ON THE CHARGER: the session ends before arm-run $i"; break; }
  cold pi5 50; cold pi4 55
  late $arm $i && break
  eval "envs=\$ENV_$arm; pre=\$PRE_$arm; post=\$POST_$arm; qargs=\$QARGS_$arm; anodes=\$NODES_$arm"
  if [ -n "$pre" ]; then
    echo "PRE $arm: $pre"
    bash -c "$pre" > "$OUT/pre_$i.log" 2>&1; prc=$?
    sed 's/^/    /' "$OUT/pre_$i.log"
    if [ "$prc" != 0 ]; then
      echo "PRE FAILED (exit $prc) for arm-run $i ($arm): skipped, recorded"
      printf '%s\t%s\t%s\t%s\n' "$i" "$arm" "pre-failed" "" >> "$OUT/arms.tsv"
      [ -n "$post" ] && bash -c "$post" 2>&1 | sed 's/^/    /'
      continue
    fi
  fi
  eval "env $envs python3 testbed/scripts/qr3_tpot.py run --model \"\$MODEL\" --ctx \"\$CTX\" --nodes \"\$anodes\" \
    --mem \"\$MEMFILE\" --reqs \"\$REQS\" --n \"\$N\" --gen \"\$GEN\" --tag ${TAGP}${arm}${i} $qargs"
  rc=$?
  if [ -n "$post" ]; then echo "POST $arm: $post"; bash -c "$post" 2>&1 | sed 's/^/    /'; fi
  lg=$(ls -t testbed/results/qr3_${TAGP}${arm}${i}_${MODEL}_*.log 2>/dev/null | grep -v '_server\|_memprobe\|_prima_rank' | head -1)
  printf '%s\t%s\t%s\t%s\n' "$i" "$arm" "$rc" "$lg" >> "$OUT/arms.tsv"
  echo "ARM $i $arm rc=$rc log=$lg"
  sleep 20
done
echo "IDLE_AFTER start $(date +%s.%N)"; sleep 60; echo "IDLE_AFTER end $(date +%s.%N)"
kill -INT $LPID; wait $LPID 2>/dev/null
kill $BPID 2>/dev/null
trap - EXIT
restore_all
guard_stop
python3 "$SCORER" "$OUT" --registration "$REG"
echo "SESSION DONE $(date -Is)"
