#!/bin/bash
# Each board as an RPC WORKER, not as the node running llama.cpp: the characterisation rows' other half.
# ggml-rpc allocates the default CPU buffer type, so a worker runs without the repacked weight layouts the host
# path uses (ggml-cpu/repack.cpp; the Pi 5's dotprod kernels), and a board's host-mode row need not describe it.
# Run from the repo root on the laptop:   bash testbed/scripts/bench_worker_role.sh
# On each board, in parallel: ggml-rpc-server bound to LOOPBACK (127.0.0.1:50053 -- never the network, never the
# RPC port 50052), readiness via ss on the board itself (a bare connect-then-close kills the server), then
# llama-bench on the same board with -rpc 127.0.0.1:50053 -ngl 99: every layer and the output head on the worker.
# The same protocol as run_quant_campaign.sh (-t 4 -p 512 -n 128 -r 3 -oe jsonl, cold gate, get_throttled before
# and after), so build_dataset.py reads the logs (device cpu-rpc-worker). The server is stopped by exact argv.
set -uo pipefail
cd "$(dirname "$0")/../.."
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=testbed/results/worker_role_$STAMP
mkdir -p "$OUT"
exec > >(tee -i -a "$OUT/run.log") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
# PP / TG (2026-09-28): the prompt and generation lengths, default 512 / 128. PP=64 TG=16 takes the rows at the
# pooled protocol instead (run_pool.sh), so the predictor can price a cell from rows measured the way it runs.
PP=${PP:-512}; TG=${TG:-128}
declare -A COLD=([pi5]=50 [pi4]=57)
# Environment overrides for later campaigns; unset, they reproduce the 09-27 run (e.g. 2026-09-28's Q3_K_M row:
#     NODES=pi5 MODELS_PI5=tinyllama-1.1b-q3km.gguf bash testbed/scripts/bench_worker_role.sh)
declare -A MODELS=([pi5]="${MODELS_PI5:-tinyllama-1.1b-q4km.gguf tinyllama-1.1b-q8_0.gguf tinyllama-1.1b-q4_0.gguf}"
                   [pi4]="${MODELS_PI4:-tinyllama-1.1b-q4km.gguf}")
NODES=${NODES:-"pi5 pi4"}
PORT=50053
SRV_ARGS="-H 127.0.0.1 -p $PORT -t 4"
# the pid of THIS run's server, by exact argv; empty if none
SRV_PID="for p in \$(pgrep -x ggml-rpc-server); do tr '\\0' ' ' < /proc/\$p/cmdline | grep -qF -- '$SRV_ARGS' && echo \$p; done"
echo "########## $(date -Is) worker-role rows -> $OUT"

role_bench() {
  local node=$1 m waited t pid
  [ "$($SSH $node 'for p in llama-bench llama-cli llama-server ggml-rpc-server cc1plus cmake; do pgrep -x $p; done | wc -l')" = 0 ] \
    || { echo "SKIP $node: board busy"; return; }
  for m in ${MODELS[$node]}; do
    waited=0
    while t=$($SSH $node "vcgencmd measure_temp | tr -dc '0-9.' | cut -d. -f1"); [ "${t:-99}" -ge "${COLD[$node]}" ] && [ $waited -lt 600 ]; do
      sleep 15; waited=$((waited + 15)); done
    echo "##### $node $m  role=rpc-worker  cold gate: waited ${waited}s, now ${t} C (threshold ${COLD[$node]} C)"
    $SSH $node "mkdir -p ~/logs; setsid nohup ~/llama.cpp/build/bin/ggml-rpc-server $SRV_ARGS > ~/logs/rpc-server-loopback_$STAMP.log 2>&1 < /dev/null &"
    for i in $(seq 1 20); do $SSH $node "ss -ltn | grep -q '127.0.0.1:$PORT '" && break; sleep 1; done
    $SSH $node "ss -ltn | grep -q '127.0.0.1:$PORT '" || { echo "FATAL $node: loopback worker never listened"; $SSH $node "cat ~/logs/rpc-server-loopback_$STAMP.log"; return; }
    $SSH $node "echo \"STATE_BEFORE \$(date -Is) \$(vcgencmd measure_temp) \$(vcgencmd get_throttled) gov=\$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) memavail_mib=\$(awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo) load=\$(cut -d' ' -f1 /proc/loadavg)\""
    $SSH $node "cd ~/llama.cpp && timeout 1800 ./build/bin/llama-bench -m ~/models/$m -rpc 127.0.0.1:$PORT -ngl 99 -t 4 -p $PP -n $TG -r 3 -oe jsonl" 2>&1
    echo "EXIT_${node}_${m}=$?"
    $SSH $node "echo \"STATE_AFTER \$(date -Is) \$(vcgencmd measure_temp) \$(vcgencmd get_throttled)\""
    pid=$($SSH $node "$SRV_PID")
    [ -n "$pid" ] && $SSH $node "kill $pid"
    for i in $(seq 1 10); do [ -z "$($SSH $node "$SRV_PID")" ] && break; sleep 1; done
    [ -z "$($SSH $node "$SRV_PID")" ] && echo "worker stopped ($node pid ${pid:-none})" || echo "!!!!! $node: loopback worker still running"
  done
}
declare -A PID=()
for n in $NODES; do role_bench $n > "$OUT/$n.log" 2>&1 & PID[$n]=$!; done
for n in $NODES; do wait ${PID[$n]}; done
for n in $NODES; do echo "$n: $(grep -c '^EXIT_' $OUT/$n.log) runs, $(grep -cE '^EXIT_.*=0$' $OUT/$n.log) exit 0; $(grep -E '^(SKIP|FATAL|!!!!!)' $OUT/$n.log | tr '\n' ' ')"; done
echo "WORKER ROLE DONE $(date -Is)"
