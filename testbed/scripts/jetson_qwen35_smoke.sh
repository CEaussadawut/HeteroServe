#!/bin/bash
# Does the Jetson run a qwen35 hybrid as an RPC worker, on CUDA and on Vulkan? A SMOKE TEST: no figure from it is reported.
# From the repo root, on the laptop:  bash testbed/scripts/jetson_qwen35_smoke.sh
# Why: on 2026-09-28 the Jetson's Vulkan worker died silently on the first graph of a qwen35 split (Pi 5 host; EXIT 134,
# no error line, no guard trip; testbed/results/pool_smoke.qwen38_27b.rule_20260928_231251.log), cause not established.
# The Jetson now also has a CUDA build at the pin (patch v2), and the laptop can host the 9B (the Pis are busy), so:
#   the laptop hosts Qwen3.5-9B Q4_K_M (-ngl 6: the Jetson takes the last 5 blocks and the output head -- recurrent 28-30,
#   attention block 31, the MTP block 32 -- so both layer kinds run on it); first the CUDA worker, then the Vulkan worker.
#   (-ngl k gives the workers k-1 blocks PLUS the output head: i_gpu_start = n_layer + 1 - k, and the 9B has n_layer 33.
#   Corrected 2026-09-30: this header first read -ngl as blocks, the misread in testbed/benchmarks.md, "The Jetson and the
#   qwen35 hybrids, corrected".)
# Workers are started here, found by exact argv, and stopped here; the guard must be running; raw output tee'd.
set -u
cd "$(dirname "$0")/../.."
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=testbed/results/jetson_qwen35_smoke_$STAMP.log
exec > >(tee -a "$LOG") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
IP=192.168.1.114; PORT=50052   # 192.168.100.111 until 2026-10-08, .1.109 until 2026-10-09
# knobs (2026-09-29): MODEL, NGL, CASES. Defaults = the first smoke (the 9B, -ngl 6, cuda then vulkan). On the 27B, NGL=5 is
# blocks 61-64 + head and NGL=17 is blocks 49-64 + head (3,667 MiB, over the Jetson's free memory: it fails at allocation).
# Neither is the 09-28 crash cell's share (blocks 10-26, no head, 2,922 MiB, a driver SIGSEGV): replay that with
# testbed/scripts/jetson_rpc_memdiag.sh.
M=${MODEL:-/opt/heteroserve/models/qwen3.5-9b-q4_k_m.gguf}
NGL=${NGL:-6}
CASES=${CASES:-cuda vulkan}
PROBE=${PROBE:-0}   # 1: memprobe.sh v2 on the Jetson (MemAvailable + the worker's resident set, 1 Hz) around each bench
BENCH=/opt/heteroserve/llama.cpp/build/bin/llama-bench
wpid() { $SSH jetson "for p in \$(pgrep -x ggml-rpc-server); do tr '\\0' ' ' < /proc/\$p/cmdline | grep -qF -- '-H $IP -p $PORT ' && echo \$p; done"; }
echo "########## $(date -Is) Jetson qwen35 RPC smoke (laptop host, -ngl $NGL, cases $CASES), model $(basename "$M") $(sha256sum "$M" | cut -c1-16)"
$SSH jetson "echo \"jetson memavail \$(awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo) MiB\""
[ "$($SSH jetson "pgrep -f '[j]etson_power_guard' | wc -l")" -ge 1 ] || { echo "FATAL: the power guard is not running"; exit 3; }
[ -z "$(wpid)" ] || { echo "FATAL: a worker is already on $IP:$PORT"; exit 1; }
declare -A CBIN=([cuda]="~/llama.cpp-cuda/build/bin" [vulkan]="~/llama.cpp/build/bin"); declare -A CDEV=([cuda]=CUDA0 [vulkan]=Vulkan0)
for name in $CASES; do
  bin=${CBIN[$name]} dev=${CDEV[$name]}
  echo; echo "===== $name worker ($bin, -d $dev)"
  $SSH jetson "echo \"before: \$(tail -1 ~/heteroserve/logs/power-guard.log | cut -c1-90)\""
  $SSH jetson "mkdir -p ~/logs; setsid nohup $bin/ggml-rpc-server -H $IP -p $PORT -d $dev > ~/logs/rpc-server_qwen35smoke_${name}_$STAMP.log 2>&1 < /dev/null &"
  sleep 1
  for _ in $(seq 20); do $SSH jetson "ss -ltn | grep -q '$IP:$PORT '" && break; sleep 1; done
  echo "worker pid $(wpid)"
  if [ "$PROBE" = 1 ]; then
    scp -q testbed/scripts/memprobe.sh jetson:memprobe.sh
    PP=$($SSH jetson "setsid nohup bash ~/memprobe.sh ~/logs/memprobe_qwen35smoke_${name}_$STAMP.log ggml-rpc-server > /dev/null 2>&1 < /dev/null & echo \$!")
    echo "probe pid $PP"; sleep 2
  fi
  timeout 900 "$BENCH" -m "$M" -rpc "$IP:$PORT" -ngl "$NGL" -t 4 -p 64 -n 16 -r 1 -oe jsonl < /dev/null
  echo "EXIT_BENCH_$name=$?"
  echo "worker alive after: $([ -n "$(wpid)" ] && echo yes || echo NO)"
  if [ "$PROBE" = 1 ]; then
    $SSH jetson "kill $PP"; scp -q jetson:logs/memprobe_qwen35smoke_${name}_$STAMP.log testbed/results/
    awk '{a=substr($2,10)+0; if (min=="" || a<min) min=a; for (i=3;i<=NF;i++) {split($i,kv,":"); v=kv[2]+0; if (kv[1]=="VmRSS" && v>rss) rss=v; if (kv[1]=="RssAnon" && v>anon) anon=v}}
         END {printf "MEMPROBE %s: min MemAvailable %d MiB, peak worker VmRSS %d MiB, RssAnon %d MiB (%d samples)\n", "'$name'", min/1024, rss/1024, anon/1024, NR}' \
         testbed/results/memprobe_qwen35smoke_${name}_$STAMP.log
  fi
  $SSH jetson "echo '--- worker log tail'; tail -8 ~/logs/rpc-server_qwen35smoke_${name}_$STAMP.log; echo \"after: \$(tail -1 ~/heteroserve/logs/power-guard.log | cut -c1-90)\"; dmesg 2>/dev/null | tail -3"
  for p in $(wpid); do $SSH jetson "kill $p"; done; sleep 2
  [ -z "$(wpid)" ] && echo "worker stopped" || echo "!!!!! worker still running"
done
echo "log: $LOG"
