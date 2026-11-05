#!/bin/bash
# Serve a model on the pooled cluster for interactive use: llama-server on the host board, ggml-rpc-server on the
# workers, the split planned by the engine for a SERVING context (the benchmark cells plan for 80 tokens). Not a
# measurement: nothing here is a reported figure. From the repo root on the laptop:
#     bash testbed/scripts/serve_pool.sh start     # plan, gate, start the workers and the server, wait until healthy
#     bash testbed/scripts/serve_pool.sh status
#     bash testbed/scripts/serve_pool.sh stop      # stops exactly what start started (exact argv), and checks
#     MODEL=qwen3moe CTX=4096                      # (defaults) a predict_pool.py model key; the context in tokens
#     MEMFILE=testbed/dataset/memory_snapshot_<stamp>.json   # plan against a dated snapshot of the boards' free memory
#                                                  # (memory_snapshot.py) instead of memory_at_rest.json (the default)
# Carries run_pool.sh's rules: nothing loads a busy board or a tree off the pin; each board's free memory NOW must
# cover the engine's charge plus its reserve; the Jetson must be MAXN with a live, un-tripped guard and eth0 its only
# router-LAN interface; worker readiness via `ss` on the worker itself (a bare TCP connect-then-close kills
# ggml-rpc-server); processes found by exact argv, never pkill -f. The Jetson's clocks are left as found (no lock:
# nothing to restore). The server binds the host's router-LAN address only (the router has no WAN).
set -u
cd "$(dirname "$0")/../.."
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=6"
STATE=$HOME/.cache/heteroserve-wf/serve_pool.state
MODEL=${MODEL:-qwen3moe}; CTX=${CTX:-4096}; PORT=50052; HTTP=8080
declare -A WIP=([pi5]=192.168.10.11 [pi4]=192.168.10.12 [jetson]=192.168.100.111)
declare -A LAN=([pi5]=192.168.100.107 [pi4]=192.168.100.106 [jetson]=192.168.100.111)
declare -A WEXTRA=([pi5]="-t 4" [pi4]="-t 4" [jetson]="-d Vulkan0")

wpid() { $SSH $1 "for p in \$(pgrep -x ggml-rpc-server); do tr '\\0' ' ' < /proc/\$p/cmdline | grep -qF -- '-H ${WIP[$1]} -p $PORT ' && echo \$p; done"; }
spid() { $SSH $1 "for p in \$(pgrep -x llama-server); do tr '\\0' ' ' < /proc/\$p/cmdline | grep -qF -- '--host ${LAN[$1]} --port $HTTP' && echo \$p; done"; }

stop() {
  [ -f "$STATE" ] || { echo "nothing started by serve_pool.sh (no $STATE)"; return 0; }
  . "$STATE"
  local n p i
  for n in $HOST; do
    p=$(spid $n); [ -n "$p" ] && $SSH $n "kill $p"
    for i in $(seq 1 20); do [ -z "$(spid $n)" ] && break; sleep 1; done
    [ -z "$(spid $n)" ] && echo "stop    llama-server on $n stopped" || echo "!!!!! stop: llama-server on $n STILL RUNNING (pid $(spid $n))"
  done
  for n in $WORKERS; do
    p=$(wpid $n); [ -n "$p" ] && $SSH $n "kill $p"
    for i in $(seq 1 10); do [ -z "$(wpid $n)" ] && break; sleep 1; done
    [ -z "$(wpid $n)" ] && echo "stop    worker on $n stopped" || echo "!!!!! stop: worker on $n STILL RUNNING (pid $(wpid $n))"
  done
  rm -f "$STATE"
}

status() {
  [ -f "$STATE" ] && { . "$STATE"; echo "started $STARTED: host $HOST, workers [$WORKERS], $ARGS -c $CTX"; } || echo "no serve_pool.sh state"
  local n
  for n in pi5 pi4 jetson; do echo "$n: server [$(spid $n)] worker [$(wpid $n)]"; done
  curl -s -m 5 "http://${LAN[pi4]}:$HTTP/health"; echo
}

start() {
  [ -f "$STATE" ] && { echo "FATAL: already started (state $STATE); stop first"; exit 1; }
  STAMP=$(date +%Y%m%d_%H%M%S); OUT=testbed/results/serve_${MODEL}_$STAMP.log
  exec > >(trap '' INT TERM HUP QUIT PIPE; exec tee -i -a "$OUT") 2>&1
  PIN=$(python3 -c "import json;print(json.load(open('ThinkPad/config/cluster.json'))['llama']['pin_sha'])")
  eval "$(python3 - "$MODEL" "$CTX" <<'PY'
import dataclasses, json, os, shlex, sys
sys.path.insert(0, "testbed/scripts")
import predict_pool as P
from control.placement import plan_optimal
key, ctx = sys.argv[1], int(sys.argv[2])
ds = "testbed/dataset"; shapes = json.load(open(ds + "/model_shapes.json")); pairs = json.load(open(ds + "/link.json"))["pairs"]
mem = json.load(open(os.environ.get("MEMFILE") or ds + "/memory_at_rest.json")); R = P.rows(ds + "/characterisation.csv", "char")
shape, quant, _ = P.MODELS[key]; q = shapes[shape]; host = P.HOST.get(key, "pi5")
C, _ = P.costs(R, shapes, quant); nodes = [n for n in ("pi5", "pi4", "jetson") if n in C]
L, missing = P.link_view(pairs, host, nodes); assert not missing, missing
devs = P.profiles(C, q, mem, L, nodes); mp = dataclasses.replace(P.model_profile(q, shape), context_len=ctx)
plan = plan_optimal(devs, mp, "decode", host_candidates=[host])
assign = [(n, plan.layers[n]) for n in plan.order if plan.layers.get(n, 0) > 0]
m = P.memory(assign, q, mem, devs, mp); dec, _ = P.predict(assign, q, C, L)
a = P.bench_args(assign, key)["args"]
a = a.replace(" -rpc ", " --rpc ")
if " -ts " in a:
    head, ts = a.split(" -ts "); a = head + " -ts " + ts.replace("/", ",")
print("HOST=%s" % assign[0][0]); print("WORKERS=%s" % shlex.quote(" ".join(n for n, _ in assign[1:])))
print("ARGS=%s" % shlex.quote(a))
print("PLAN=%s" % shlex.quote(" + ".join("%s %d" % (n, k) for n, k in assign) + ", head on " + assign[-1][0]
                              + ("; predicted decode %.1f ms/token (%.2f tok/s)" % (dec, 1000 / dec))))
print("declare -A GATE=(%s)" % " ".join("[%s]=%d" % (n, v["need_mib"] + v["reserve_mib"]) for n, v in m.items()))
print("declare -A MARGIN=(%s)" % " ".join("[%s]=%d" % (n, v["margin_mib"]) for n, v in m.items()))
PY
)" || { echo "FATAL: the engine could not plan $MODEL at ctx $CTX"; exit 1; }
  echo "########## $(date -Is) serve $MODEL, ctx $CTX, planned against ${MEMFILE:-testbed/dataset/memory_at_rest.json}: $PLAN"
  echo "llama-server $ARGS -c $CTX -np 1 -t 4 --host ${LAN[$HOST]} --port $HTTP   pin $PIN"
  local n b h g w a
  for n in $HOST $WORKERS; do
    b=$($SSH $n 'for p in llama-bench llama-cli llama-server ggml-rpc-server cc1plus cmake; do pgrep -x $p >/dev/null && printf "%s " $p; done; true' 2>&1) || b+=" (ssh exit $?)"
    [ -z "$b" ] || { echo "FATAL: $n is busy or unreachable: $b"; exit 1; }
    h=$($SSH $n 'git -C ~/llama.cpp rev-parse HEAD; git -C ~/llama.cpp status --porcelain --untracked-files=no | wc -l' | tr '\n' ' ')
    [ "$h" = "$PIN 0 " ] || { echo "FATAL: $n tree is '$h', not the pin $PIN clean"; exit 2; }
    a=$($SSH $n "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo")
    echo "gate    $n: MemAvailable $a MiB, the plan needs ${GATE[$n]} MiB with its reserve (margin at rest ${MARGIN[$n]} MiB)"
    [ "${a:-0}" -ge "${GATE[$n]}" ] || { echo "REFUSED: $n has $a MiB free"; exit 8; }
  done
  if [[ " $WORKERS " == *" jetson "* ]]; then
    g=$($SSH jetson 'm=$(cut -d" " -f1 /var/lib/nvpmodel/status); [ "$m" = pmode:0000 ] || { echo "GATE-GUARD pmode $m"; exit 6; }
      [ "$(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l)" -ge 1 ] || { echo "GATE-GUARD no guard process"; exit 6; }
      G=~/heteroserve/logs/power-guard.log; t=$(awk "/guard start/ {t=0} /TRIP/ {t++} END {print t+0}" $G); [ "$t" = 0 ] || { echo "GATE-GUARD tripped $t since start"; exit 6; }
      a=$(( $(date +%s) - $(stat -c %Y $G) )); [ $a -le 5 ] || { echo "GATE-GUARD log $a s stale"; exit 6; }
      [ "$(ip -4 -o addr show | grep -c " 192.168.100.")" = 1 ] && ip -4 -o addr show eth0 | grep -q " 192.168.100.111/" || { echo "GATE-NET not eth0-only on 192.168.100.x"; exit 7; }
      echo "GATE ok: $m, guard live and un-tripped, eth0 is the only interface on the router LAN"' 2>&1)
    echo "jetson  $g"; [[ "$g" == "GATE ok"* ]] || { echo "FATAL: Jetson gate failed"; exit 3; }
  fi
  printf 'STARTED=%q\nHOST=%q\nWORKERS=%q\nARGS=%q\nCTX=%q\nLOG=%q\n' "$(date -Is)" "$HOST" "$WORKERS" "$ARGS" "$CTX" "$OUT" > "$STATE"
  for n in $WORKERS; do
    $SSH $n "mkdir -p ~/logs; setsid nohup ~/llama.cpp/build/bin/ggml-rpc-server -H ${WIP[$n]} -p $PORT ${WEXTRA[$n]} > ~/logs/rpc-server_serve_$STAMP.log 2>&1 < /dev/null &"
    for i in $(seq 1 30); do $SSH $n "ss -ltn | grep -q ' ${WIP[$n]}:$PORT '" && break; sleep 1; done
    $SSH $n "ss -ltn | grep -q ' ${WIP[$n]}:$PORT '" || { echo "FATAL: worker on $n never listened"; stop; exit 1; }
    echo "worker  $n listening on ${WIP[$n]}:$PORT (pid $(wpid $n))"
  done
  # No "cd ~ &&" in front: "a && b &" backgrounds a subshell that keeps the ssh channel open until the server exits
  # (2026-09-28: the first start hung here, with the server up). Every path in the command is absolute.
  $SSH $HOST "mkdir -p ~/logs; setsid nohup ~/llama.cpp/build/bin/llama-server $ARGS -c $CTX -np 1 -t 4 --host ${LAN[$HOST]} --port $HTTP > ~/logs/llama-server_serve_$STAMP.log 2>&1 < /dev/null &"
  echo "server  starting on $HOST (pid $(spid $HOST)); loading streams the model from the SD card to the workers"
  for i in $(seq 1 180); do   # up to 15 min
    r=$(curl -s -m 5 "http://${LAN[$HOST]}:$HTTP/health"); [[ "$r" == *'"ok"'* ]] && break
    [ -z "$(spid $HOST)" ] && { echo "FATAL: llama-server exited while loading:"; $SSH $HOST "tail -25 ~/logs/llama-server_serve_$STAMP.log"; stop; exit 1; }
    sleep 5
  done
  [[ "$r" == *'"ok"'* ]] || { echo "FATAL: not healthy after 15 min: $r"; stop; exit 1; }
  echo "READY   $(date -Is): http://${LAN[$HOST]}:$HTTP/ (web chat) -- health $r"
  $SSH $HOST "grep -a -E 'model buffer size|KV self size|llama_kv_cache.*size|compute buffer size|flash_attn|n_ctx ' ~/logs/llama-server_serve_$STAMP.log | head -20"
}

case "${1:-}" in
  start) start ;; stop) stop ;; status) status ;;
  *) echo "usage: $0 start|status|stop"; exit 2 ;;
esac
