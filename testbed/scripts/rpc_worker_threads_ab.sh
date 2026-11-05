#!/bin/bash
# What does an RPC worker started WITHOUT -t cost against the worker-role rows' -t 4? A-B on one board.
# From the repo root, on the laptop:     NODE=pi5 bash testbed/scripts/rpc_worker_threads_ab.sh      (or NODE=pi4)
# Why (2026-09-29 evening): ggml-rpc-server's CPU thread count defaults to hardware_concurrency()/2 (tools/rpc/
# rpc-server.cpp at the pin), and no RPC command carries the host's -t. So a worker started without -t runs on 2 of a
# Pi's 4 cores. The instruments pass it: bench_worker_role.sh (the rows the predictor prices a worker from), run_pool.sh
# and serve_pool.sh (WEXTRA "-t 4"), heteroctl_run_pis.sh and run_14b_pooled.sh all start Pi workers with -t 4, so no
# registered cell is affected. (A first reading of run_pool.sh on 2026-09-29 missed WEXTRA and said otherwise; it was
# retracted the same evening.) This measures what omitting -t would cost, for any new script that starts a worker.
# Conditions, alternated A B A B (drift shows as A1 vs A2):
#   T4  ggml-rpc-server -H 127.0.0.1 -p 50053 -t 4     (bench_worker_role.sh's worker)
#   TD  ggml-rpc-server -H 127.0.0.1 -p 50054          (no -t: the default thread count, hardware_concurrency()/2)
# each measured by llama-bench on the same board, -rpc loopback -ngl 99 -t 4 -p $PP -n $TG -r 3 -oe jsonl: the worker-role
# protocol at the pooled lengths. The worker's own thread count is sampled once a second while it runs (/proc Threads:).
# Cold gate and get_throttled before and after every run; raw output tee'd.
set -uo pipefail
cd "$(dirname "$0")/../.."
NODE=${NODE:?NODE=pi5 or pi4}
MODEL=${MODEL:-tinyllama-1.1b-q4km.gguf}; PP=${PP:-64}; TG=${TG:-16}; ROUNDS=${ROUNDS:-2}
declare -A COLD=([pi5]=50 [pi4]=57)
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=testbed/results/worker_threads_${NODE}_$STAMP
mkdir -p "$OUT"
exec > >(tee -i -a "$OUT/run.log") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
echo "########## $(date -Is) RPC worker threads A-B on $NODE: $MODEL pp$PP/tg$TG -r 3, $ROUNDS rounds -> $OUT"
[ "$($SSH $NODE 'for p in llama-bench llama-cli llama-server ggml-rpc-server cc1plus cmake; do pgrep -x $p; done | wc -l')" = 0 ] \
  || { echo "FATAL: $NODE is busy"; exit 1; }
echo "build $($SSH $NODE 'git -C ~/llama.cpp rev-parse HEAD'); nproc $($SSH $NODE nproc); model sha256 $($SSH $NODE "sha256sum ~/models/$MODEL | cut -c1-64")"
one() {   # $1 label, $2 port, $3 extra server args
  local label=$1 port=$2 extra=$3 args waited=0 t pid
  args="-H 127.0.0.1 -p $port $extra"
  while t=$($SSH $NODE "vcgencmd measure_temp | tr -dc '0-9.' | cut -d. -f1"); [ "${t:-99}" -ge "${COLD[$NODE]}" ] && [ $waited -lt 600 ]; do
    sleep 15; waited=$((waited + 15)); done
  echo "##### $label  worker: ggml-rpc-server $args  cold gate: waited ${waited}s, now ${t} C"
  $SSH $NODE "mkdir -p ~/logs; setsid nohup ~/llama.cpp/build/bin/ggml-rpc-server $args > ~/logs/rpc-server-threads_${label}_$STAMP.log 2>&1 < /dev/null &"
  for _ in $(seq 20); do $SSH $NODE "ss -ltn | grep -q '127.0.0.1:$port '" && break; sleep 1; done
  pid=$($SSH $NODE "for p in \$(pgrep -x ggml-rpc-server); do tr '\\0' ' ' < /proc/\$p/cmdline | grep -qF -- '-p $port ' && echo \$p; done")
  [ -n "$pid" ] || { echo "FATAL: no worker on port $port"; exit 4; }
  $SSH $NODE "setsid nohup bash -c 'while [ -d /proc/$pid ]; do awk \"/^Threads:/{print \\\$2}\" /proc/$pid/status; sleep 1; done' > ~/logs/threads_${label}_$STAMP.log 2>&1 < /dev/null &"
  $SSH $NODE "echo \"STATE_BEFORE \$(date -Is) \$(vcgencmd measure_temp) \$(vcgencmd get_throttled) memavail_mib=\$(awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo) load=\$(cut -d' ' -f1 /proc/loadavg)\""
  $SSH $NODE "cd ~/llama.cpp && timeout 1200 ./build/bin/llama-bench -m ~/models/$MODEL -rpc 127.0.0.1:$port -ngl 99 -t 4 -p $PP -n $TG -r 3 -oe jsonl" 2>&1
  echo "EXIT_$label=$?"
  $SSH $NODE "echo \"STATE_AFTER \$(date -Is) \$(vcgencmd measure_temp) \$(vcgencmd get_throttled)\""
  echo "WORKER_THREADS_$label idle/peak: $($SSH $NODE "sort -n ~/logs/threads_${label}_$STAMP.log | sed -n '1p;\$p' | tr '\n' ' '")"
  $SSH $NODE "kill $pid"; sleep 2
  [ -z "$($SSH $NODE "[ -d /proc/$pid ] && echo up")" ] && echo "worker stopped" || echo "!!!!! worker $pid still running"
}
for r in $(seq "$ROUNDS"); do
  one T4_r$r 50053 "-t 4"
  one TD_r$r 50054 ""
done
echo "===== summary (mean t/s of the 3 samples; the lines are llama-bench's own jsonl)"
python3 - "$OUT/run.log" <<'EOF'
import json, re, sys, statistics
label = None
for line in open(sys.argv[1]):
    m = re.match(r"##### (\S+)", line)
    if m: label = m.group(1)
    if line.startswith("{") and '"samples_ts"' in line:
        r = json.loads(line); s = r["samples_ts"]
        test = "pp%d" % r["n_prompt"] if r["n_prompt"] else "tg%d" % r["n_gen"]
        print("%-8s %-6s %8.3f t/s  (%s)  spread %.2f %%" % (label, test, statistics.fmean(s), " / ".join("%.3f" % x for x in s),
                                                          100 * (max(s) - min(s)) / statistics.fmean(s)))
EOF
echo "log: $OUT/run.log"
