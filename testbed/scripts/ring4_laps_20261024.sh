#!/bin/bash
# The ring-lap study on the four-node cluster's two fastest boards, exactly as registered (register_ring4_20261024.py,
# 2026-10-24). From the repo root, on the laptop:
#     bash testbed/scripts/ring4_laps_20261024.sh [testbed/dataset/ring4_predictions_20261024.json]
# ring_laps.sh (2026-10-09: the Pi 4 hosting with loopback stages, the Jetson's stages over the wire) moved onto the laptop and
# the Pi 5: llama-bench runs on the laptop itself and holds only the token embedding; the laptop's stages are ggml-rpc-servers
# on its loopback, the Pi 5's on its router address.
# - Gates before anything starts: the laptop quiet on its charger (no working apps, no battery charging, Wi-Fi off, 1-min
#   load < 1.5, as the session scripts), its tree at the pin and clean; the Pi 5 idle, at the pin and clean, cool (< 50 C)
#   and with 2 GB free; no other llama.cpp on the laptop; the model's sha256 on the laptop (testbed/models.md).
# - A 1 Hz throttle poll on the Pi 5 for the session (throttle_poll.sh v2), copied back as throttle_pi5.log.
# - Servers started here, found by exact argv and stopped here (the EXIT trap); readiness by `ss` on the board itself.
# - Each cell: every server alive, then llama-bench with the layout's --rpc and -ts and the registered protocol; each round
#   starts with 30 pings from the laptop to the Pi 5. No shaping and no energy: a time-only question, as on 10-09.
# Everything goes to testbed/results/ring4_<stamp>/run.log (raw, tee'd); score it with score_ring4_20261024.py.
set -u
cd "$(dirname "$0")/../.."
REG=${1:-testbed/dataset/ring4_predictions_20261024.json}
STAMP=$(date +%Y%m%d_%H%M%S)
OUTD=testbed/results/ring4_$STAMP
mkdir -p "$OUTD"
exec > >(trap '' INT TERM HUP QUIT PIPE; exec tee -i -a "$OUTD/run.log") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=6"
LTREE=/opt/heteroserve/llama.cpp
[ -f "$REG" ] && [ ! -w "$REG" ] || { echo "FATAL: $REG missing or writable (register first)"; exit 2; }
PIN=$(python3 -c "import json;print(json.load(open('ThinkPad/config/cluster.json'))['llama']['pin_sha'])")
MSHA=$(python3 -c "import sys; sys.path.insert(0, 'testbed/scripts'); import full_eval as F; print(F.SHA['tinyllama'])")
eval "$(python3 - "$REG" <<'PY'
import json, shlex, sys
r = json.load(open(sys.argv[1]))
print("MODEL_PATH=%s" % shlex.quote(r["model_path"]))
print("LPORTS=%s" % shlex.quote(" ".join(map(str, r["servers"]["laptop"]["ports"]))))
print("FPORTS=%s" % shlex.quote(" ".join(map(str, r["servers"]["pi5"]["ports"]))))
print("FIP=%s" % r["servers"]["pi5"]["ip"])
print("PROTO=%s" % shlex.quote(r["protocol"]))
print("declare -A RPC=(%s)" % " ".join("[%s]=%s" % (k, shlex.quote(v["rpc"])) for k, v in r["layouts"].items()))
print("declare -A TS=(%s)" % " ".join("[%s]=%s" % (k, shlex.quote(v["ts"])) for k, v in r["layouts"].items()))
print("SEQ=%s" % shlex.quote(" ".join("%d:%s" % (ri, lab) for ri, rnd in enumerate(r["order"], 1) for lab in rnd)))
PY
)" || { echo "FATAL: cannot read $REG"; exit 1; }
echo "########## $(date -Is) ring laps on the laptop + Pi 5: registration $REG ($(sha256sum "$REG" | cut -c1-16)), pin $PIN"
echo "protocol $PROTO; model $MODEL_PATH; laptop ports [$LPORTS], pi5 $FIP ports [$FPORTS]; sequence $SEQ"

lpid() { for p in $(pgrep -x ggml-rpc-server); do tr '\0' ' ' < /proc/$p/cmdline | grep -qF -- "-H $1 -p $2 " && echo $p; done; }
fpid() { $SSH pi5 "for p in \$(pgrep -x ggml-rpc-server); do tr '\\0' ' ' < /proc/\$p/cmdline | grep -qF -- '-H $1 -p $2 ' && echo \$p; done"; }

# ---- gates ----
APPS='^(zen|antigravity-ide|notion-snap-reb)$'
a=$(ps -eo comm= | grep -E "$APPS" | sort -u | tr '\n' ' '); l=$(cut -d' ' -f1 /proc/loadavg)
c=$(grep -l '^Charging$' /sys/class/power_supply/BAT*/status 2>/dev/null | wc -l); w=$(nmcli radio wifi 2>/dev/null)
ac=$(cat /sys/class/power_supply/AC/online 2>/dev/null)
echo "gate    laptop: apps [$a] load $l charging $c wifi $w ac $ac"
{ [ -z "$a" ] && [ "$c" = 0 ] && [ "$w" != enabled ] && [ "$ac" = 1 ] && awk -v l="$l" 'BEGIN {exit !(l < 1.5)}'; } \
  || { echo "FATAL: the laptop is not quiet on its charger"; exit 3; }
for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null && { echo "FATAL: $p runs on the laptop"; exit 3; }; done
h=$(git -C $LTREE rev-parse HEAD; git -C $LTREE status --porcelain --untracked-files=no | wc -l); h=$(echo $h)
[ "$h" = "$PIN 0" ] || { echo "FATAL: the laptop's tree is '$h', not the pin $PIN clean"; exit 2; }
b=$($SSH pi5 'for p in llama-bench llama-cli llama-server ggml-rpc-server cc1plus cmake; do pgrep -x $p >/dev/null && printf "%s " $p; done; true') || b+=" (ssh exit $?)"
[ -z "$b" ] || { echo "FATAL: the Pi 5 is busy or unreachable: $b"; exit 1; }
h=$($SSH pi5 'git -C ~/llama.cpp rev-parse HEAD; git -C ~/llama.cpp status --porcelain --untracked-files=no | wc -l' | tr '\n' ' ')
[ "$h" = "$PIN 0 " ] || { echo "FATAL: the Pi 5's tree is '$h', not the pin $PIN clean"; exit 2; }
echo "gate    both idle, both trees at the pin and clean"
m=$(sha256sum $(eval echo $MODEL_PATH) | cut -d' ' -f1)
[ "$m" = "$MSHA" ] || { echo "FATAL: $MODEL_PATH on the laptop is '$m', not the pinned $MSHA"; exit 4; }
echo "gate    model sha256 on the laptop is the pinned ${MSHA:0:16}"
wt=0; while t=$($SSH pi5 "vcgencmd measure_temp | tr -dc '0-9.' | cut -d. -f1"); [ "${t:-99}" -ge 50 ] && [ $wt -lt 600 ]; do sleep 15; wt=$((wt + 15)); done
fa=$($SSH pi5 "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo")
echo "gate    pi5: ${t} C after ${wt} s (threshold 50 C); MemAvailable ${fa} MiB (need 2000); $($SSH pi5 'vcgencmd get_throttled')"
[ "${fa:-0}" -ge 2000 ] || { echo "REFUSED: the Pi 5 has ${fa} MiB free"; exit 8; }

STARTED=""; POLL=""
cleanup() {
  trap '' PIPE INT TERM HUP QUIT
  local kv n ip p pid
  [ -n "$POLL" ] && $SSH pi5 "kill $POLL 2>/dev/null"
  for kv in $STARTED; do
    IFS=: read -r n ip p <<<"$kv"
    if [ "$n" = laptop ]; then pid=$(lpid $ip $p); [ -n "$pid" ] && kill $pid
    else pid=$(fpid $ip $p); [ -n "$pid" ] && $SSH pi5 "kill $pid"; fi
  done
  sleep 2
  for kv in $STARTED; do
    IFS=: read -r n ip p <<<"$kv"
    if [ "$n" = laptop ]; then [ -z "$(lpid $ip $p)" ] || echo "!!!!! cleanup: server $kv STILL RUNNING"
    else [ -z "$(fpid $ip $p)" ] || echo "!!!!! cleanup: server $kv STILL RUNNING"; fi
  done
  [ -n "$STARTED" ] && echo "cleanup servers stopped: $STARTED"
}
trap 'cleanup' EXIT; trap 'exit 130' INT; trap 'exit 129' HUP; trap 'exit 143' TERM

for p in $LPORTS; do
  mkdir -p ~/logs; setsid nohup $LTREE/build/bin/ggml-rpc-server -H 127.0.0.1 -p $p -t 4 > ~/logs/rpc-server_ring4_${STAMP}_$p.log 2>&1 < /dev/null &
  STARTED+=" laptop:127.0.0.1:$p"
  for i in $(seq 1 30); do ss -ltn | grep -q " 127.0.0.1:$p " && break; sleep 1; done
  ss -ltn | grep -q " 127.0.0.1:$p " || { echo "FATAL: laptop server 127.0.0.1:$p never listened"; cat ~/logs/rpc-server_ring4_${STAMP}_$p.log; exit 1; }
  echo "server  laptop 127.0.0.1:$p listening (pid $(lpid 127.0.0.1 $p))"
done
for p in $FPORTS; do
  $SSH pi5 "mkdir -p ~/logs; setsid nohup ~/llama.cpp/build/bin/ggml-rpc-server -H $FIP -p $p -t 4 > ~/logs/rpc-server_ring4_${STAMP}_$p.log 2>&1 < /dev/null &"
  STARTED+=" pi5:$FIP:$p"
  for i in $(seq 1 30); do $SSH pi5 "ss -ltn | grep -q ' $FIP:$p '" && break; sleep 1; done
  $SSH pi5 "ss -ltn | grep -q ' $FIP:$p '" || { echo "FATAL: pi5 server $FIP:$p never listened"; exit 1; }
  echo "server  pi5 $FIP:$p listening (pid $(fpid $FIP $p))"
done
alive() {
  local kv n ip p
  for kv in $STARTED; do IFS=: read -r n ip p <<<"$kv"
    if [ "$n" = laptop ]; then [ -n "$(lpid $ip $p)" ] || { echo "$kv"; return 1; }
    else [ -n "$(fpid $ip $p)" ] || { echo "$kv"; return 1; }; fi
  done
}
scp -q testbed/scripts/throttle_poll.sh pi5:throttle_poll.sh && \
  POLL=$($SSH pi5 "mkdir -p ~/logs; setsid nohup bash ~/throttle_poll.sh ~/logs/throttle_ring4_$STAMP.log > /dev/null 2>&1 < /dev/null & echo \$!")
echo "poll    pi5 throttle poll pid ${POLL:-NONE}"
echo "STATE before pi5 $($SSH pi5 'echo "$(date -Is) $(vcgencmd measure_temp) $(vcgencmd get_throttled) load=$(cut -d" " -f1 /proc/loadavg)"') laptop load $(cut -d' ' -f1-3 /proc/loadavg)"

LAST=""
for item in $SEQ; do
  IFS=: read -r rd lab <<<"$item"
  if [ "$rd" != "$LAST" ]; then echo "PING round=$rd $(ping -c 30 -i 0.2 -q $FIP | tail -1)"; LAST=$rd; fi
  dead=$(alive) || { echo "FATAL: server $dead is gone before round $rd $lab"; exit 10; }
  echo "##### CELL round=$rd setting=s0 layout=$lab start=$(date +%s.%N)"
  (cd ~ && /usr/bin/timeout 900 $LTREE/build/bin/llama-bench -m $(eval echo $MODEL_PATH) --rpc ${RPC[$lab]} -ts ${TS[$lab]} $PROTO; echo EXIT_BENCH=$?)
  rc=$?
  echo "##### CELL_END round=$rd setting=s0 layout=$lab end=$(date +%s.%N) exit=$rc"
done
echo "STATE after pi5 $($SSH pi5 'echo "$(date -Is) $(vcgencmd measure_temp) $(vcgencmd get_throttled) load=$(cut -d" " -f1 /proc/loadavg)"') laptop load $(cut -d' ' -f1-3 /proc/loadavg)"
if [ -n "$POLL" ]; then
  a=$($SSH pi5 "kill $POLL 2>/dev/null && echo 1 || echo 0"); POLL=""
  scp -q pi5:logs/throttle_ring4_$STAMP.log "$OUTD/throttle_pi5.log" && \
    echo "THROTTLE pi5 $(python3 testbed/scripts/throttle_summary.py "$OUTD/throttle_pi5.log" --alive "${a:-0}")"
fi
for kv in $STARTED; do IFS=: read -r n ip p <<<"$kv"; echo "--- server log $kv ---"
  if [ "$n" = laptop ]; then tail -3 ~/logs/rpc-server_ring4_${STAMP}_$p.log; else $SSH pi5 "tail -3 ~/logs/rpc-server_ring4_${STAMP}_$p.log"; fi
done
python3 testbed/scripts/score_ring4_20261024.py "$OUTD" --registration "$REG"
echo "RUN DONE $(date -Is) $OUTD"
