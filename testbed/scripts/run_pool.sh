#!/bin/bash
# One pooled cell, exactly as pre-registered in testbed/dataset/pool_predictions.json. From the repo root:
#     CELL=qwen14b.rule bash testbed/scripts/run_pool.sh
# The host runs llama-bench (-t 4 -p 64 -n 16 -r 3 -oe jsonl -v, or the cell's registered bench.protocol) with the
# cell's own -rpc / -ngl / -ts; each worker
# runs ggml-rpc-server started here and stopped here. Carries the house rules and run_14b_pooled.sh's traps:
#  - nothing loads a busy board, a tree off the pin, or a board with less free memory NOW than the cell needs plus
#    the reserve the board must keep free (registration memory.reserve_mib; 0 in registrations that predate it) --
#    or, where the registration gives one, the node's own memory.gate_mib: a cell registered as forced past a
#    board's limit (memory.forced) is still gated on the memory that can really run out, and says so; nor a board
#    whose registered share passes its weights cap (memory.weights_margin_mib < 0: the Jetson's max_weights_mb,
#    registered rule GPU_CAP, 2026-09-29) unless the cell is registered as forced there;
#  - readiness via `ss` on the worker itself (a bare TCP connect-then-close kills ggml-rpc-server); processes found
#    by exact name or exact argv, never `pkill -f`; the Jetson's RPC address is its eth0, the only interface it has
#    on the router LAN (checked), and its eth0 byte counters bracket the cell to prove the wire;
#  - the Jetson: MAXN and a live, un-tripped guard before the load; clocks locked (stored first, restored after,
#    only if this run locked them and the power mode is still MAXN); guard trips counted after;
#  - get_throttled and temperatures before and after on the Pis; a 1 Hz memory probe (memprobe.sh) on every board,
#    its log copied back beside the cell's (memprobe_pool_<cell>_<stamp>_<node>.log: a name compare_pool.py's
#    pool_<cell>_2*.log glob cannot mistake for the cell's own log);
#  - a 1 Hz throttle poll (throttle_poll.sh v2, one process start a second) on every Pi from before the bench to after
#    it (2026-10-07: the before/after snapshots read ~68 C and a sticky 0x80000 while the Pi 4 peaked at 83 C with the
#    soft limit active), its log copied back as throttle_pool_<cell>_<stamp>_<node>.log, and one line per Pi,
#    THROTTLE <node> <fields> verdict=clean|THROTTLED|UNKNOWN, from throttle_summary.py, which compare_pool.py reads;
#  - raw output tee'd; the EXIT trap stops this run's workers and probes whatever happens.
# A cell may name its host tool (bench.tool, 2026-10-10): llama-bench (the default, every cell before) or
# llama-batched-bench, whose generation steps decode a batch of -npl tokens with logits for all of them -- the work of a
# speculative check (register_spec_pass_20261010.py). Its flags are the common ones: --rpc, not llama-bench's -rpc.
set -u
cd "$(dirname "$0")/../.."
CELL=${CELL:?set CELL}; PRED=${PRED:-testbed/dataset/pool_predictions.json}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=testbed/results/pool_${CELL}_${STAMP}.log
exec > >(trap '' INT TERM HUP QUIT PIPE; exec tee -i -a "$OUT") 2>&1
trap 'exit 131' QUIT
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=6"
PIN=$(python3 -c "import json;print(json.load(open('ThinkPad/config/cluster.json'))['llama']['pin_sha'])")
eval "$(python3 - "$PRED" "$CELL" <<'PY'
import json, shlex, sys
c = json.load(open(sys.argv[1]))["configs"][sys.argv[2]]
nodes = [p["node"] for p in c["pipeline"]]
print("HOST=%s" % c["bench"]["host"]); print("WORKERS=%s" % shlex.quote(" ".join(nodes[1:])))
print("ARGS=%s" % shlex.quote(c["bench"]["args"]))
print("declare -A NEED=(%s)" % " ".join("[%s]=%d" % (n, v["need_mib"]) for n, v in c["memory"].items()))
print("declare -A RES=(%s)" % " ".join("[%s]=%d" % (n, v.get("reserve_mib", 0)) for n, v in c["memory"].items()))
print("declare -A GATE=(%s)" % " ".join("[%s]=%d" % (n, v.get("gate_mib", v["need_mib"] + v.get("reserve_mib", 0)))
                                        for n, v in c["memory"].items()))
print("declare -A FORCED=(%s)" % " ".join("[%s]=1" % n for n, v in c["memory"].items() if v.get("forced")))
print("declare -A WOVER=(%s)" % " ".join("[%s]='%.0f'" % (n, -v["weights_margin_mib"]) for n, v in c["memory"].items()
                                          if v.get("weights_margin_mib", 0) < 0))
print("PROTO=%s" % shlex.quote(c["bench"].get("protocol", "-t 4 -p 64 -n 16 -r 3 -oe jsonl -v")))
print("declare -A BACKEND=(%s)" % " ".join("[%s]=%s" % (n, b) for n, b in (c.get("backend") or {}).items()))
tool = c["bench"].get("tool", "llama-bench")
if tool not in ("llama-bench", "llama-batched-bench"):
    sys.exit("unknown bench.tool %r" % tool)
print("TOOL=%s" % tool)
PY
)" || { echo "FATAL: no cell $CELL in $PRED, or its bench.tool is unknown"; exit 1; }
declare -A WIP=([pi5]=192.168.10.11 [pi4]=192.168.10.12 [jetson]=192.168.1.114)   # the Jetson: 192.168.100.111 until 2026-10-08, .1.109 until 2026-10-09
declare -A WEXTRA=([pi5]="-t 4" [pi4]="-t 4" [jetson]="-d Vulkan0")
declare -A COLD=([pi5]=55 [pi4]=60 [jetson]=45)
# A cell may run the Jetson's worker on its CUDA build ("backend": {"jetson": "cuda"}, 2026-09-29): the pin plus
# testbed/patches/jetpack46-cuda102-b23701f.patch (v2) in ~/llama.cpp-cuda, on CUDA0. Default: Vulkan, the unpatched pin.
declare -A WTREE=([pi5]='~/llama.cpp' [pi4]='~/llama.cpp' [jetson]='~/llama.cpp')
JPATCH='~/heteroserve/patches/jetpack46-cuda102-b23701f.patch'; JPATCH_SHA=1f43a3e4a5762e97a49c6d3f4982c6a899671c785257080bd34dc9de9a2283c6
if [ "${BACKEND[jetson]:-vulkan}" = cuda ]; then WTREE[jetson]='~/llama.cpp-cuda'; WEXTRA[jetson]="-d CUDA0"; fi
# ALL: every board once. A host may also be a worker (2026-09-30, llama.cpp's default split with every board a device: the Pi 5
# runs llama-bench AND a ggml-rpc-server of its own, and --rpc lists it); every earlier cell has a host that is not a worker.
PORT=50052; ALL=$(printf '%s\n' $HOST $WORKERS | awk '!seen[$0]++' | tr '\n' ' ')
SELFW=""; [[ " $WORKERS " == *" $HOST "* ]] && SELFW=$HOST
echo "########## $(date -Is) cell $CELL: host $HOST, workers [${WORKERS}], $TOOL $ARGS   pin $PIN"
echo "protocol $PROTO"
[[ " $WORKERS " == *" jetson "* ]] && echo "backend jetson=${BACKEND[jetson]:-vulkan} (tree ${WTREE[jetson]}, ${WEXTRA[jetson]})"

wpid() {  # pid of THIS run's worker on node $1 (exact argv), empty if none
  $SSH $1 "for p in \$(pgrep -x ggml-rpc-server); do tr '\\0' ' ' < /proc/\$p/cmdline | grep -qF -- '-H ${WIP[$1]} -p $PORT ' && echo \$p; done"
}
temp_of() { [ "$1" = jetson ] && $SSH jetson 'echo $(( $(cat /sys/devices/virtual/thermal/thermal_zone1/temp) / 1000 ))' \
                             || $SSH $1 "vcgencmd measure_temp | tr -dc '0-9.' | cut -d. -f1"; }
state() {  # $1 label
  local n
  for n in $ALL; do
    if [ $n = jetson ]; then
      echo "STATE $1 jetson $($SSH jetson 'G=/sys/devices/57000000.gpu/devfreq/57000000.gpu; echo "$(date -Is) $(cut -d" " -f1 /var/lib/nvpmodel/status) CPU=$(( $(cat /sys/devices/virtual/thermal/thermal_zone1/temp) / 1000 ))C GPU=$(( $(cat /sys/devices/virtual/thermal/thermal_zone2/temp) / 1000 ))C cpu0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)/$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq) gpu=$(cat $G/min_freq)/$(cat $G/max_freq) swap_used_kB=$(awk "NR>1{s+=\$4} END{print s+0}" /proc/swaps) memavail_mib=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo) eth0_rx=$(cat /sys/class/net/eth0/statistics/rx_bytes) eth0_tx=$(cat /sys/class/net/eth0/statistics/tx_bytes) trips=$(awk "/guard start/ {t=0} /TRIP/ {t++} END {print t+0}" ~/heteroserve/logs/power-guard.log)"')"
    else
      echo "STATE $1 $n $($SSH $n 'echo "$(date -Is) $(vcgencmd measure_temp) $(vcgencmd get_throttled) memavail_mib=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo) load=$(cut -d" " -f1 /proc/loadavg)"')"
    fi
  done
}

# ---- gates (before anything is started, so the cleanup never touches what this run did not start) ----
for n in $ALL; do
  # stdout only (2026-10-05): the laptop's OpenSSH 10.2 prints a post-quantum key-exchange warning on stderr for the
  # Jetson's old sshd, which 2>&1 made read as "busy"; an unreachable board still fails by ssh's exit code
  # llama-batched-b: the kernel keeps 15 characters of a process name, and pgrep -x matches only those
  b=$($SSH $n 'for p in llama-bench llama-batched-b llama-cli llama-server ggml-rpc-server cc1plus cmake; do pgrep -x $p >/dev/null && printf "%s " $p; done; true') || b+=" (ssh exit $?)"
  [ -z "$b" ] || { echo "FATAL: $n is busy or unreachable: $b"; exit 1; }
  if [ $n = jetson ] && [ "${BACKEND[jetson]:-vulkan}" = cuda ]; then
    # the pin plus exactly patch v2: HEAD at the pin, the patch by sha256, the patch reverse-applies, and the changed
    # files (untracked included) are exactly the patch's files
    h=$($SSH jetson "T=${WTREE[jetson]}; P=$JPATCH; git -C \$T rev-parse HEAD; sha256sum \$P | cut -d' ' -f1; git -C \$T apply -R --check \$P && echo applied; diff <(git -C \$T status --porcelain --untracked-files=all | cut -c4- | sort) <(grep '^diff --git' \$P | sed 's|^diff --git a/[^ ]* b/||' | sort) >/dev/null && echo exact" | tr '\n' ' ')
    [ "$h" = "$PIN $JPATCH_SHA applied exact " ] || { echo "FATAL: jetson CUDA tree is '$h', not the pin plus exactly patch v2"; exit 2; }
    continue
  fi
  h=$($SSH $n 'git -C ~/llama.cpp rev-parse HEAD; git -C ~/llama.cpp status --porcelain --untracked-files=no | wc -l' | tr '\n' ' ')
  [ "$h" = "$PIN 0 " ] || { echo "FATAL: $n tree is '$h', not the pin $PIN clean"; exit 2; }
done
echo "gate    every board idle, every tree at the pin and clean"
if [[ " $ALL " == *" jetson "* ]]; then     # host or worker (2026-10-08: a Jetson-hosted cell was ungated)
  g=$($SSH jetson 'm=$(cut -d" " -f1 /var/lib/nvpmodel/status); [ "$m" = pmode:0000 ] || { echo "GATE-GUARD pmode $m"; exit 6; }
    [ "$(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l)" -ge 1 ] || { echo "GATE-GUARD no guard process"; exit 6; }
    G=~/heteroserve/logs/power-guard.log; t=$(awk "/guard start/ {t=0} /TRIP/ {t++} END {print t+0}" $G); [ "$t" = 0 ] || { echo "GATE-GUARD tripped $t since start"; exit 6; }
    a=$(( $(date +%s) - $(stat -c %Y $G) )); [ $a -le 5 ] || { echo "GATE-GUARD log $a s stale"; exit 6; }
    [ "$(ip -4 -o addr show | grep -cF " 192.168.1.")" = 1 ] && ip -4 -o addr show eth0 | grep -qF " 192.168.1.114/" || { echo "GATE-NET not eth0-only on 192.168.1.x"; exit 7; }
    echo "GATE ok: $m, guard live and un-tripped, eth0 is the only interface on the router LAN"')   # stdout only, as the busy check
  echo "jetson  $g"; [[ "$g" == "GATE ok"* ]] || { echo "FATAL: Jetson gate failed"; exit 3; }
fi
for n in $ALL; do   # cool down (bounded), then the live memory check against the pre-registered need
  w=0; while t=$(temp_of $n); [ "${t:-99}" -ge "${COLD[$n]}" ] && [ $w -lt 600 ]; do sleep 15; w=$((w + 15)); done
  # A board's MemAvailable takes minutes to recover after a big cell (7,432 MiB right after one 14B cell, 7,587
  # a few minutes later): wait for it, bounded, rather than refusing a cell the board can hold once it has recovered.
  req=${GATE[$n]}
  wm=0; while a=$($SSH $n "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo"); [ "${a:-0}" -lt "$req" ] && [ $wm -lt 600 ]; do sleep 15; wm=$((wm + 15)); done
  echo "gate    $n: ${t} C after ${w} s (threshold ${COLD[$n]} C); MemAvailable ${a} MiB after ${wm} s, the cell needs ${NEED[$n]} MiB + a ${RES[$n]} MiB reserve"
  [ -n "${FORCED[$n]:-}" ] && echo "FORCED  $n: registered as forced past its limit (need ${NEED[$n]} MiB); gated instead on ${req} MiB, the memory that can really run out"
  [ "${a:-0}" -ge "$req" ] || { echo "REFUSED: $n has ${a} MiB free and the cell needs ${req} MiB (need ${NEED[$n]} MiB + a ${RES[$n]} MiB reserve, or its registered gate) -- not loading"; exit 8; }
  if [ -n "${WOVER[$n]:-}" ]; then
    [ -n "${FORCED[$n]:-}" ] || { echo "REFUSED: $n's registered share is ${WOVER[$n]} MiB over its weights cap (max_weights_mb) -- not loading"; exit 8; }
    echo "FORCED  $n: registered ${WOVER[$n]} MiB over its weights cap"
  fi
done

STARTED=""; PROBES=""; TPROBES=""; LOCKED=0
cleanup() {
  trap '' PIPE INT TERM HUP QUIT
  local n p kv
  for kv in $PROBES $TPROBES; do $SSH ${kv%%:*} "kill ${kv#*:} 2>/dev/null"; done
  for n in $STARTED; do
    p=$(wpid $n); [ -n "$p" ] && $SSH $n "kill $p"
    for i in 1 2 3 4 5 6 7 8 9 10; do [ -z "$(wpid $n)" ] && break; sleep 1; done
    [ -z "$(wpid $n)" ] && echo "cleanup worker on $n stopped" || echo "!!!!! cleanup: worker on $n STILL RUNNING (pid $(wpid $n))"
  done
  if [ "$LOCKED" = 1 ]; then
    $SSH jetson "m=\$(cut -d' ' -f1 /var/lib/nvpmodel/status); if [ \"\$m\" = pmode:0000 ]; then sudo -n jetson_clocks --restore ~/heteroserve/run/jetson_clocks_pool_$STAMP.stored && echo 'cleanup jetson clocks restored'; else echo \"!!!!! cleanup: jetson power mode is \$m -- clocks NOT restored\"; fi"
  fi
}
trap 'cleanup' EXIT; trap 'exit 130' INT; trap 'exit 129' HUP; trap 'exit 143' TERM

if [[ " $ALL " == *" jetson "* ]]; then     # host or worker (2026-10-08: a Jetson-hosted cell was ungated)
  r=$($SSH jetson "c=/sys/devices/system/cpu/cpu0/cpufreq; g=/sys/devices/57000000.gpu/devfreq/57000000.gpu; if [ \$(cat \$c/scaling_min_freq) = \$(cat \$c/scaling_max_freq) ] && [ \$(cat \$g/min_freq) = \$(cat \$g/max_freq) ]; then echo ALREADY_LOCKED; else mkdir -p ~/heteroserve/run && sudo -n jetson_clocks --store ~/heteroserve/run/jetson_clocks_pool_$STAMP.stored && sudo -n jetson_clocks && echo LOCKED; fi")
  echo "jetson  clocks: $r"; [ "$r" = LOCKED ] && LOCKED=1
  [ "$r" = LOCKED ] || [ "$r" = ALREADY_LOCKED ] || { echo "FATAL: could not lock the Jetson's clocks"; exit 1; }
fi
for n in $WORKERS; do
  $SSH $n "mkdir -p ~/logs; setsid nohup ${WTREE[$n]}/build/bin/ggml-rpc-server -H ${WIP[$n]} -p $PORT ${WEXTRA[$n]} > ~/logs/rpc-server_pool_$STAMP.log 2>&1 < /dev/null &"
  STARTED+=" $n"
  for i in $(seq 1 30); do $SSH $n "ss -ltn | grep -q ' ${WIP[$n]}:$PORT '" && break; sleep 1; done
  $SSH $n "ss -ltn | grep -q ' ${WIP[$n]}:$PORT '" || { echo "FATAL: worker on $n never listened"; $SSH $n "cat ~/logs/rpc-server_pool_$STAMP.log"; exit 1; }
  echo "worker  $n listening on ${WIP[$n]}:$PORT (pid $(wpid $n)): $($SSH $n "tail -3 ~/logs/rpc-server_pool_$STAMP.log" | tr '\n' ' ')"
done
for n in $ALL; do
  scp -q testbed/scripts/memprobe.sh $n:memprobe.sh
  pn=ggml-rpc-server; [ $n = $HOST ] && pn=${TOOL:0:15}     # memprobe matches /proc/<pid>/comm: 15 characters
  p=$($SSH $n "mkdir -p ~/logs; setsid nohup bash ~/memprobe.sh ~/logs/memprobe_pool_$STAMP.log $pn > /dev/null 2>&1 < /dev/null & echo \$!")
  PROBES+=" $n:$p"
  if [ "$n" = "$SELFW" ]; then        # the host's own worker: a second probe, on the worker process, into its own log
    p=$($SSH $n "setsid nohup bash ~/memprobe.sh ~/logs/memprobe_pool_${STAMP}_w.log ggml-rpc-server > /dev/null 2>&1 < /dev/null & echo \$!")
    PROBES+=" $n:$p"; WPROBE=$p          # stopped with the others; summarised apart, below
  fi
  case $n in pi*)                       # the throttle poll: its own list, so the MEMPROBE loop below never reads it
    if scp -q testbed/scripts/throttle_poll.sh $n:throttle_poll.sh; then
      p=$($SSH $n "setsid nohup bash ~/throttle_poll.sh ~/logs/throttle_pool_$STAMP.log > /dev/null 2>&1 < /dev/null & echo \$!")
      TPROBES+=" $n:$p"
    else echo "!!!!! could not copy throttle_poll.sh to $n: no throttle poll there, its THROTTLE verdict will be UNKNOWN"; fi ;;
  esac
done
echo "probes $PROBES${TPROBES:+   throttle polls$TPROBES}"
state before
echo "##### bench $CELL on $HOST"
$SSH $HOST "cd ~ && /usr/bin/timeout 2700 ~/llama.cpp/build/bin/$TOOL $ARGS $PROTO; echo EXIT_BENCH=\$?"
r=$?; [ "$r" = 0 ] || echo "!!!!! lost the host mid-cell (ssh exit $r)"
state after
for kv in $TPROBES; do        # stop each Pi's throttle poll (alive=0: it had died, so it does not cover the cell), certify
  n=${kv%%:*}; a=$($SSH $n "kill ${kv#*:} 2>/dev/null && echo 1 || echo 0")
  T="testbed/results/throttle_pool_${CELL}_${STAMP}_$n.log"
  if scp -q $n:logs/throttle_pool_$STAMP.log "$T"; then
    echo "THROTTLE $n $(python3 testbed/scripts/throttle_summary.py "$T" --alive "${a:-0}")"
  else echo "THROTTLE $n verdict=UNKNOWN (could not copy its poll log)"; fi
done
for n in $ALL; do case $n in pi*) [[ "$TPROBES" == *" $n:"* ]] || echo "THROTTLE $n verdict=UNKNOWN (no throttle poll ran)" ;; esac; done
TPROBES=""
for kv in $PROBES; do
  n=${kv%%:*}; $SSH $n "kill ${kv#*:} 2>/dev/null"
  [ -n "$SELFW" ] && [ "$kv" = "$SELFW:${WPROBE:-}" ] && continue        # the host's own worker probe: its own line below
  echo "MEMPROBE $n $($SSH $n "awk '{a=substr(\$2,10)+0; if (min==\"\" || a<min) min=a; for (i=3;i<=NF;i++) {split(\$i,kv,\":\"); v=kv[2]+0; if (kv[1] ~ /^(VmRSS|RssAnon|RssFile|RssShmem)\$/ && v>mx[kv[1]]) mx[kv[1]]=v}} END {printf \"samples=%d min_avail_mib=%d\", NR, min/1024; for (k in mx) printf \" max_%s_mib=%d\", k, mx[k]/1024}' ~/logs/memprobe_pool_$STAMP.log")"
  scp -q $n:logs/memprobe_pool_$STAMP.log "testbed/results/memprobe_pool_${CELL}_${STAMP}_$n.log" || echo "!!!!! could not copy the probe log from $n"
done
if [ -n "$SELFW" ]; then
  echo "MEMPROBE $SELFW worker $($SSH $SELFW "awk '{a=substr(\$2,10)+0; if (min==\"\" || a<min) min=a; for (i=3;i<=NF;i++) {split(\$i,kv,\":\"); v=kv[2]+0; if (kv[1] ~ /^(VmRSS|RssAnon|RssFile|RssShmem)\$/ && v>mx[kv[1]]) mx[kv[1]]=v}} END {printf \"samples=%d min_avail_mib=%d\", NR, min/1024; for (k in mx) printf \" max_%s_mib=%d\", k, mx[k]/1024}' ~/logs/memprobe_pool_${STAMP}_w.log")"
  scp -q $SELFW:logs/memprobe_pool_${STAMP}_w.log "testbed/results/memprobe_pool_${CELL}_${STAMP}_${SELFW}_worker.log" || echo "!!!!! could not copy the worker probe log from $SELFW"
fi
PROBES=""
for n in $WORKERS; do echo "--- worker log $n ---"; $SSH $n "tail -20 ~/logs/rpc-server_pool_$STAMP.log"; done
echo "CELL DONE $CELL $(date -Is)"
