#!/bin/bash
# The ring-lap study, one session, exactly as registered (testbed/scripts/register_ring_laps_20261009.py). From the repo root:
#     bash testbed/scripts/ring_laps.sh [testbed/dataset/ring_laps_predictions_20261009.json]
#     SMOKE=1 bash testbed/scripts/ring_laps.sh <a registration built with register_ring_laps_20261009.build()>
#         the plumbing only: one cell, the four-lap layout at s0 (SMOKE_LAYOUT=<label> for another), -p 16 -n 4 -r 1, into
#         ring_laps_smoke_<stamp>/ (never scored or reported: one repetition)
#     bash testbed/scripts/ring_laps.sh testbed/dataset/visit_predictions_20261009.json   # the VISIT test, the same way
# A layout with "ngl" (the VISIT registration's in-process host layouts) runs with -ngl <it> in place of the protocol's
# -ngl 99, so the Pi 4 computes its first blocks itself; the ring-lap registration has none, and runs exactly as before.
# Everything goes to testbed/results/ring_laps_<stamp>/run.log (raw, tee'd); score it with score_ring_laps_20261009.py.
# - Gates before anything starts: both boards idle and at the pin (clean trees); the Jetson in MAXN on eth0 only, its power
#   guard started by its NOTE's procedure (clock synchronized, the script's sha256 the repo's) unless one is already
#   running, live and un-tripped; no shaping installed (netshape.sh check); enough free memory; both boards cool; the model's
#   sha256 on the Pi 4 (testbed/models.md).
# - The Jetson's clocks locked for the session (stored, restored after). A 1 Hz throttle poll on the Pi 4 for the session
#   (throttle_poll.sh v2), copied back as throttle_pi4.log; each cell's start and end are logged on the laptop's clock.
# - Servers: the registration's Pi 4 loopback ggml-rpc-servers and Jetson ggml-rpc-servers, started here, found by exact argv
#   and stopped here; readiness by `ss` on the board itself (a bare TCP connect-then-close kills ggml-rpc-server).
# - Each cell: every server of this run checked alive, then llama-bench on the Pi 4 with the layout's --rpc and -ts and the
#   registered protocol. The d5 setting is netshape.sh apply 0 2.5 (+5 ms RTT, the Pi 4 <-> Jetson path only); each
#   (round, setting) block starts with 30 pings from the Pi 4 to the Jetson.
# - The EXIT trap stops this run's servers and poll, clears the shaping, restores the clocks and stops a guard it started.
set -u
cd "$(dirname "$0")/../.."
REG=${1:-testbed/dataset/ring_laps_predictions_20261009.json}
STAMP=$(date +%Y%m%d_%H%M%S)
OUTD=testbed/results/ring_laps_$STAMP
[ "${SMOKE:-0}" = 1 ] && OUTD=testbed/results/ring_laps_smoke_$STAMP
mkdir -p "$OUTD"
exec > >(trap '' INT TERM HUP QUIT PIPE; exec tee -i -a "$OUTD/run.log") 2>&1
trap 'exit 131' QUIT
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=6"
PIN=$(python3 -c "import json;print(json.load(open('ThinkPad/config/cluster.json'))['llama']['pin_sha'])")
MSHA=$(python3 -c "import sys; sys.path.insert(0, 'testbed/scripts'); import full_eval as F; print(F.SHA['tinyllama'])")
eval "$(python3 - "$REG" <<'PY'
import json, shlex, sys
r = json.load(open(sys.argv[1]))
print("MODEL_PATH=%s" % shlex.quote(r["model_path"]))
print("PPORTS=%s" % shlex.quote(" ".join(map(str, r["servers"]["pi4"]["ports"]))))
print("JPORTS=%s" % shlex.quote(" ".join(map(str, r["servers"]["jetson"]["ports"]))))
print("JIP=%s" % r["servers"]["jetson"]["ip"])
print("PROTO=%s" % shlex.quote(r["protocol"]))
print("declare -A RPC=(%s)" % " ".join("[%s]=%s" % (k, shlex.quote(v["rpc"])) for k, v in r["layouts"].items()))
print("declare -A TS=(%s)" % " ".join("[%s]=%s" % (k, shlex.quote(v["ts"])) for k, v in r["layouts"].items()))
print("declare -A NGL=(%s)" % " ".join("[%s]=%s" % (k, shlex.quote(str(v.get("ngl") or ""))) for k, v in r["layouts"].items()))
print("declare -A SHAPE=(%s)" % " ".join("[%s]=%s" % (k, shlex.quote(v["netshape"] or "")) for k, v in r["settings"].items()))
print("SEQ=%s" % shlex.quote(" ".join("%d:%s:%s" % (ri, st, lab) for ri, rnd in enumerate(r["order"], 1)
                                      for st, labs in rnd for lab in labs)))
PY
)" || { echo "FATAL: cannot read $REG"; exit 1; }
if [ "${SMOKE:-0}" = 1 ]; then SEQ="1:s0:${SMOKE_LAYOUT:-k4}"; PROTO="-t 4 -ngl 99 -p 16 -n 4 -r 1 -oe jsonl -v"; echo "SMOKE: plumbing only, never scored"; fi
echo "########## $(date -Is) ring laps: registration $REG ($(sha256sum "$REG" | cut -c1-16)), pin $PIN"
echo "protocol $PROTO; model $MODEL_PATH; pi4 ports [$PPORTS], jetson $JIP ports [$JPORTS]"
echo "sequence $SEQ"

spid() {  # pid of THIS run's server on board $1 at ip $2 port $3 (exact argv), empty if none
  $SSH $1 "for p in \$(pgrep -x ggml-rpc-server); do tr '\\0' ' ' < /proc/\$p/cmdline | grep -qF -- '-H $2 -p $3 ' && echo \$p; done"
}
temp_of() { [ "$1" = jetson ] && $SSH jetson 'echo $(( $(cat /sys/devices/virtual/thermal/thermal_zone1/temp) / 1000 ))' \
                             || $SSH $1 "vcgencmd measure_temp | tr -dc '0-9.' | cut -d. -f1"; }
state() {
  echo "STATE $1 jetson $($SSH jetson 'G=/sys/devices/57000000.gpu/devfreq/57000000.gpu; echo "$(date -Is) $(cut -d" " -f1 /var/lib/nvpmodel/status) CPU=$(( $(cat /sys/devices/virtual/thermal/thermal_zone1/temp) / 1000 ))C GPU=$(( $(cat /sys/devices/virtual/thermal/thermal_zone2/temp) / 1000 ))C cpu0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)/$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq) gpu=$(cat $G/min_freq)/$(cat $G/max_freq) memavail_mib=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo) eth0_rx=$(cat /sys/class/net/eth0/statistics/rx_bytes) eth0_tx=$(cat /sys/class/net/eth0/statistics/tx_bytes) trips=$(awk "/guard start/ {t=0} /TRIP/ {t++} END {print t+0}" ~/heteroserve/logs/power-guard.log)"')"
  echo "STATE $1 pi4 $($SSH pi4 'echo "$(date -Is) $(vcgencmd measure_temp) $(vcgencmd get_throttled) memavail_mib=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo) load=$(cut -d" " -f1 /proc/loadavg)"')"
}

# ---- gates ----
for n in pi4 jetson; do
  b=$($SSH $n 'for p in llama-bench llama-cli llama-server ggml-rpc-server cc1plus cmake; do pgrep -x $p >/dev/null && printf "%s " $p; done; true') || b+=" (ssh exit $?)"
  [ -z "$b" ] || { echo "FATAL: $n is busy or unreachable: $b"; exit 1; }
  h=$($SSH $n 'git -C ~/llama.cpp rev-parse HEAD; git -C ~/llama.cpp status --porcelain --untracked-files=no | wc -l' | tr '\n' ' ')
  [ "$h" = "$PIN 0 " ] || { echo "FATAL: $n tree is '$h', not the pin $PIN clean"; exit 2; }
done
echo "gate    both boards idle, both trees at the pin and clean"
bash testbed/scripts/netshape.sh check || { echo "FATAL: shaping is installed on the Pi 4; clear it first"; exit 3; }
m=$($SSH pi4 "sha256sum $MODEL_PATH | cut -d' ' -f1")
[ "$m" = "$MSHA" ] || { echo "FATAL: $MODEL_PATH on the Pi 4 is '$m', not the pinned $MSHA"; exit 4; }
echo "gate    model sha256 on the Pi 4 is the pinned ${MSHA:0:16}"

GUARD_STARTED=0
if [ "$($SSH jetson 'grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l')" = 0 ]; then
  g=$($SSH jetson 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum ~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1' | tr '\n' ' ')
  want=$(sha256sum testbed/scripts/jetson_power_guard.sh | cut -d' ' -f1)
  [ "$g" = "synced $want " ] || { echo "FATAL: guard precheck '$g'"; exit 5; }
  $SSH jetson 'sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo started'
  GUARD_STARTED=1; sleep 5
fi
g=$($SSH jetson 'm=$(cut -d" " -f1 /var/lib/nvpmodel/status); [ "$m" = pmode:0000 ] || { echo "GATE-GUARD pmode $m"; exit 6; }
  [ "$(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l)" -ge 1 ] || { echo "GATE-GUARD no guard process"; exit 6; }
  G=~/heteroserve/logs/power-guard.log; t=$(awk "/guard start/ {t=0} /TRIP/ {t++} END {print t+0}" $G); [ "$t" = 0 ] || { echo "GATE-GUARD tripped $t since start"; exit 6; }
  a=$(( $(date +%s) - $(stat -c %Y $G) )); [ $a -le 5 ] || { echo "GATE-GUARD log $a s stale"; exit 6; }
  [ "$(ip -4 -o addr show | grep -cF " 192.168.1.")" = 1 ] && ip -4 -o addr show eth0 | grep -qF " 192.168.1.114/" || { echo "GATE-NET not eth0-only on 192.168.1.x"; exit 7; }
  echo "GATE ok: $m, guard live and un-tripped, eth0 is the only interface on the router LAN"')
echo "jetson  $g (guard started by this run: $GUARD_STARTED)"
stop_guard() {
  $SSH jetson 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p stopped"; done; read -t 2 <> <(:); echo "guard procs: $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l)"'
}
if [[ "$g" != "GATE ok"* ]]; then echo "FATAL: Jetson gate failed"; [ $GUARD_STARTED = 1 ] && stop_guard; exit 6; fi
declare -A COLD=([pi4]=60 [jetson]=45) MEMNEED=([pi4]=2000 [jetson]=2000)
for n in pi4 jetson; do
  w=0; while t=$(temp_of $n); [ "${t:-99}" -ge "${COLD[$n]}" ] && [ $w -lt 600 ]; do sleep 15; w=$((w + 15)); done
  a=$($SSH $n "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo")
  echo "gate    $n: ${t} C after ${w} s (threshold ${COLD[$n]} C); MemAvailable ${a} MiB (need ${MEMNEED[$n]})"
  [ "${a:-0}" -ge "${MEMNEED[$n]}" ] || { echo "REFUSED: $n has ${a} MiB free"; [ $GUARD_STARTED = 1 ] && stop_guard; exit 8; }
done

STARTED=""; POLL=""; LOCKED=0
cleanup() {
  trap '' PIPE INT TERM HUP QUIT
  local kv n ip p pid
  [ -n "$POLL" ] && $SSH pi4 "kill $POLL 2>/dev/null"
  for kv in $STARTED; do
    IFS=: read -r n ip p <<<"$kv"
    pid=$(spid $n $ip $p); [ -n "$pid" ] && $SSH $n "kill $pid"
  done
  sleep 2
  for kv in $STARTED; do
    IFS=: read -r n ip p <<<"$kv"
    [ -z "$(spid $n $ip $p)" ] || echo "!!!!! cleanup: server $kv STILL RUNNING"
  done
  [ -n "$STARTED" ] && echo "cleanup servers stopped: $STARTED"
  bash testbed/scripts/netshape.sh check >/dev/null 2>&1 || bash testbed/scripts/netshape.sh clear
  if [ "$LOCKED" = 1 ]; then
    $SSH jetson "m=\$(cut -d' ' -f1 /var/lib/nvpmodel/status); if [ \"\$m\" = pmode:0000 ]; then sudo -n jetson_clocks --restore ~/heteroserve/run/jetson_clocks_laps_$STAMP.stored && echo 'cleanup jetson clocks restored'; else echo \"!!!!! cleanup: jetson power mode is \$m -- clocks NOT restored\"; fi"
  fi
  [ "$GUARD_STARTED" = 1 ] && stop_guard
}
trap 'cleanup' EXIT; trap 'exit 130' INT; trap 'exit 129' HUP; trap 'exit 143' TERM

r=$($SSH jetson "c=/sys/devices/system/cpu/cpu0/cpufreq; g=/sys/devices/57000000.gpu/devfreq/57000000.gpu; if [ \$(cat \$c/scaling_min_freq) = \$(cat \$c/scaling_max_freq) ] && [ \$(cat \$g/min_freq) = \$(cat \$g/max_freq) ]; then echo ALREADY_LOCKED; else mkdir -p ~/heteroserve/run && sudo -n jetson_clocks --store ~/heteroserve/run/jetson_clocks_laps_$STAMP.stored && sudo -n jetson_clocks && echo LOCKED; fi")
echo "jetson  clocks: $r"; [ "$r" = LOCKED ] && LOCKED=1
[ "$r" = LOCKED ] || [ "$r" = ALREADY_LOCKED ] || { echo "FATAL: could not lock the Jetson's clocks"; exit 1; }

start_server() {  # board ip port extra
  $SSH $1 "mkdir -p ~/logs; setsid nohup ~/llama.cpp/build/bin/ggml-rpc-server -H $2 -p $3 $4 > ~/logs/rpc-server_laps_${STAMP}_$3.log 2>&1 < /dev/null &"
  STARTED+=" $1:$2:$3"
  for i in $(seq 1 30); do $SSH $1 "ss -ltn | grep -q ' $2:$3 '" && break; sleep 1; done
  $SSH $1 "ss -ltn | grep -q ' $2:$3 '" || { echo "FATAL: server $1 $2:$3 never listened"; $SSH $1 "cat ~/logs/rpc-server_laps_${STAMP}_$3.log"; exit 1; }
  echo "server  $1 $2:$3 listening (pid $(spid $1 $2 $3))"
}
for p in $PPORTS; do start_server pi4 127.0.0.1 $p "-t 4"; done
for p in $JPORTS; do start_server jetson $JIP $p "-d Vulkan0"; done
alive() {  # every server of this run is up
  local kv n ip p
  for kv in $STARTED; do IFS=: read -r n ip p <<<"$kv"; [ -n "$(spid $n $ip $p)" ] || { echo "$kv"; return 1; }; done
}
scp -q testbed/scripts/throttle_poll.sh pi4:throttle_poll.sh && \
  POLL=$($SSH pi4 "mkdir -p ~/logs; setsid nohup bash ~/throttle_poll.sh ~/logs/throttle_laps_$STAMP.log > /dev/null 2>&1 < /dev/null & echo \$!")
echo "poll    pi4 throttle poll pid ${POLL:-NONE}"
state before

CUR=s0; LAST_BLOCK=""
for item in $SEQ; do
  IFS=: read -r rd st lab <<<"$item"
  if [ "$st" != "$CUR" ]; then
    if [ -n "${SHAPE[$st]}" ]; then bash testbed/scripts/netshape.sh ${SHAPE[$st]} || { echo "FATAL: could not shape for $st"; exit 9; }
    else bash testbed/scripts/netshape.sh clear || { echo "FATAL: could not clear the shaping"; exit 9; }; fi
    CUR=$st
  fi
  if [ "$rd:$st" != "$LAST_BLOCK" ]; then
    echo "PING round=$rd setting=$st $($SSH pi4 "ping -c 30 -i 0.2 -q $JIP | tail -1")"
    LAST_BLOCK="$rd:$st"
  fi
  dead=$(alive) || { echo "FATAL: server $dead is gone before round $rd $st $lab"; exit 10; }
  proto=$PROTO; [ -n "${NGL[$lab]}" ] && proto=${PROTO/-ngl 99/-ngl ${NGL[$lab]}}
  echo "##### CELL round=$rd setting=$st layout=$lab start=$(date +%s.%N)"
  $SSH pi4 "cd ~ && /usr/bin/timeout 900 ~/llama.cpp/build/bin/llama-bench -m $MODEL_PATH --rpc ${RPC[$lab]} -ts ${TS[$lab]} $proto; echo EXIT_BENCH=\$?"
  rc=$?    # before any command substitution, which would overwrite $?
  echo "##### CELL_END round=$rd setting=$st layout=$lab end=$(date +%s.%N) ssh_exit=$rc"
done
bash testbed/scripts/netshape.sh clear; bash testbed/scripts/netshape.sh check && echo "shaping cleared and checked"
state after
if [ -n "$POLL" ]; then
  a=$($SSH pi4 "kill $POLL 2>/dev/null && echo 1 || echo 0"); POLL=""
  scp -q pi4:logs/throttle_laps_$STAMP.log "$OUTD/throttle_pi4.log" && \
    echo "THROTTLE pi4 $(python3 testbed/scripts/throttle_summary.py "$OUTD/throttle_pi4.log" --alive "${a:-0}")"
fi
for kv in $STARTED; do IFS=: read -r n ip p <<<"$kv"; echo "--- server log $kv ---"; $SSH $n "tail -5 ~/logs/rpc-server_laps_${STAMP}_$p.log"; done
echo "RUN DONE $(date -Is) $OUTD"
