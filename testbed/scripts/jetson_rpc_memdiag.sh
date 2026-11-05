#!/bin/bash
# The Jetson as an RPC worker, with every device's buffers as llama.cpp itself sizes them. A DIAGNOSTIC: one repetition,
# and no throughput figure from it is reported. From the repo root, on the laptop (the host):
#     CASE=<label> MODEL=<gguf on the laptop> NGL=<n> [TS=a/b] [WORKERS="jetson pi5"|"jetson t580"] [BACKEND=vulkan|cuda]
#     [P=16] [N=4] [EXTRA="<more llama-bench flags>"] [WENV="<env for the Jetson worker>"] [CRASHTRACE=1] [GDB=1]
#     bash testbed/scripts/jetson_rpc_memdiag.sh
# WORKERS is the pipeline order after the host: the laptop keeps the first blocks, each worker the next ones, and the last
# worker the output head. At the pin, -ngl COUNTS THE OUTPUT LAYER (llama-model.cpp: i_gpu_start = n_layer_all + 1 - ngl),
# so -ngl k gives the workers k-1 blocks plus the head. The -v buffer lines below say what each device actually got.
# "t580" as a worker is a CPU ggml-rpc-server on the laptop's loopback (127.0.0.1): a tail that can hold any share and the
# head, so a Jetson share in the MIDDLE of a pipeline can be tested with only that share crossing the network.
# Pi and laptop workers get -t 4, as run_pool.sh's WEXTRA gives them (ggml-rpc-server's default is half the cores).
#
# Why (2026-09-29 evening). jetson_qwen35_smoke.sh read -ngl as blocks only. So its "5 blocks" and "9 blocks" of the 27B
# were 4 and 8 blocks plus the 834 MiB output head, and the "fixed 1.3 GB" in the record was mostly that head. The 09-28
# cell's Jetson (17 middle blocks of the 27B, no head, 2,922 MiB) died at the first graph with 515 MiB available and the
# memory charge matching the drop. There is no OOM line in kern.log and none on stderr; apport.log records signal 11.
# This script replayed it (r1, r2): SIGSEGV again, and under gdb the fault is in NVIDIA's Tegra Vulkan driver
# (libnvidia-eglcore.so.32.7.6) inside vkAllocateDescriptorSets, called by ggml-vulkan's ggml_pipeline_allocate_descriptor_sets
# from ggml_vk_mul_mat_q_f16 while the graph is built. Each worker runs under a bash wrapper that records how it ended
# (WORKER_EXIT: 128+signal when killed); CRASHTRACE=1 preloads glibc's libSegFault.so (its aarch64 backtrace stops at its
# own frame: it proved only the signal); GDB=1 runs the Jetson worker under gdb for the backtrace (gdb's own memory is not
# measured, so a crash at the memory edge might not reproduce under it -- this one did).
#
# Probes: memprobe_tegra.sh on the Jetson (MemAvailable, NvMapMemUsed/Free, CmaFree, the worker's resident set), memprobe.sh
# v2 on a Pi or laptop worker; get_throttled before and after on every Pi; the Jetson's guard must be running. Raw output tee'd.
set -u
cd "$(dirname "$0")/../.."
STAMP=$(date +%Y%m%d_%H%M%S)
CASE=${CASE:-diag}
LOG=testbed/results/jetson_memdiag_${CASE}_$STAMP.log
exec > >(tee -a "$LOG") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
PIN=b23701f77d47dad9de834d59ebfcbe25c9e8b46f
M=${MODEL:-/opt/heteroserve/models/qwen3.8-27b-q3_k_m.gguf}
NGL=${NGL:?NGL is required}; TS=${TS:-}; WORKERS=${WORKERS:-jetson}; BACKEND=${BACKEND:-vulkan}
P=${P:-16}; N=${N:-4}; EXTRA=${EXTRA:-}; WENV=${WENV:-}; CRASHTRACE=${CRASHTRACE:-0}; GDB=${GDB:-0}
BENCH=/opt/heteroserve/llama.cpp/build/bin/llama-bench
declare -A SHA=([qwen3.8-27b-q3_k_m.gguf]=14865aaef0b6c26a013538782d366443c7298b11d5df9f0fb829d34d1963f869
                [qwen3.5-9b-q4_k_m.gguf]=d784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43
                [qwen3-14b-q4km.gguf]=500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0)
declare -A IP=([jetson]=192.168.1.114 [pi5]=192.168.10.11 [pi4]=192.168.10.12 [t580]=127.0.0.1)
declare -A TREE=([jetson]='~/llama.cpp' [pi5]='~/llama.cpp' [pi4]='~/llama.cpp' [t580]=/opt/heteroserve/llama.cpp)
declare -A DEV=([jetson]="-d Vulkan0" [pi5]="-t 4" [pi4]="-t 4" [t580]="-t 4")
[ "$BACKEND" = cuda ] && { TREE[jetson]='~/llama.cpp-cuda'; DEV[jetson]="-d CUDA0"; }
PORT=50052
on() { local n=$1; shift; if [ "$n" = t580 ]; then bash -c "$*" < /dev/null; else $SSH "$n" "$*"; fi; }
put() { if [ "$1" = t580 ]; then cp "$2" ~/"$(basename "$2")"; else scp -q "$2" "$1:$(basename "$2")"; fi; }
get() { if [ "$1" = t580 ]; then cp ~/"$2" "$3"; else scp -q "$1:$2" "$3"; fi; }
wpid() { on $1 "for p in \$(pgrep -x ggml-rpc-server); do tr '\\0' ' ' < /proc/\$p/cmdline | grep -qF -- '-H ${IP[$1]} -p $PORT ' && echo \$p; done"; }
state() {
  case $1 in
    jetson) $SSH jetson "echo \"STATE $2 jetson \$(date -Is) \$(cut -d' ' -f1 /var/lib/nvpmodel/status) CPU=\$(( \$(cat /sys/devices/virtual/thermal/thermal_zone1/temp) / 1000 ))C memavail_mib=\$(awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo) nvmap_used_kb=\$(awk '/NvMapMemUsed/{print \$2}' /proc/meminfo) swap_kb=\$(awk 'NR>1{s+=\$4} END{print s+0}' /proc/swaps) trips=\$(awk '/guard start/ {t=0} /TRIP/ {t++} END {print t+0}' ~/heteroserve/logs/power-guard.log)\"" ;;
    t580) echo "STATE $2 t580 $(date -Is) memavail_mib=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) load=$(cut -d' ' -f1 /proc/loadavg)" ;;
    *) $SSH $1 "echo \"STATE $2 $1 \$(date -Is) \$(vcgencmd measure_temp) \$(vcgencmd get_throttled) memavail_mib=\$(awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo)\"" ;;
  esac
}
b=$(basename "$(readlink -f "$M")"); got=$(sha256sum "$M" | cut -d' ' -f1); key=$(basename "$M")
echo "########## $(date -Is) jetson memdiag $CASE: host t580, workers [$WORKERS], jetson backend $BACKEND${WENV:+, jetson env $WENV}"
echo "model $key ($b) sha256 $got"
[ "$got" = "${SHA[$key]:-none}" ] || { echo "FATAL: $key sha256 is not the pinned ${SHA[$key]:-none}"; exit 2; }
[ "$(git -C /opt/heteroserve/llama.cpp rev-parse HEAD)" = $PIN ] || { echo "FATAL: the laptop's tree is not at the pin"; exit 2; }
[ "$($SSH jetson "pgrep -f '[j]etson_power_guard' | wc -l")" -ge 1 ] || { echo "FATAL: the Jetson's power guard is not running"; exit 3; }
RPCS=""
for w in $WORKERS; do
  h=$(on $w "git -C ${TREE[$w]} rev-parse HEAD; for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x \$p >/dev/null && echo busy:\$p; done" | tr '\n' ' ')
  echo "gate $w: $h"
  [[ "$h" == "$PIN "* ]] || { echo "FATAL: $w ${TREE[$w]} is not at the pin"; exit 2; }
  [[ "$h" != *busy:* ]] || { echo "FATAL: $w is busy"; exit 1; }
  RPCS+="${RPCS:+,}${IP[$w]}:$PORT"
done
KL0=$($SSH jetson 'wc -l < /var/log/kern.log')     # kern.log is read from this line on at the end (the board's locale is Thai)
declare -A PP
for w in $WORKERS; do
  state $w before
  WL="~/logs/rpc-server_memdiag_${CASE}_$STAMP.log"
  pre=""
  if [ $w = jetson ]; then
    pre="$WENV${WENV:+ }"
    [ "$CRASHTRACE" = 1 ] && pre+="LD_PRELOAD=/lib/aarch64-linux-gnu/libSegFault.so SEGFAULT_SIGNALS='segv bus ill fpe abrt' "
  fi
  cmd="${pre}${TREE[$w]}/build/bin/ggml-rpc-server -H ${IP[$w]} -p $PORT ${DEV[$w]}"
  # GDB=1 (Jetson only): the worker under gdb, which prints the faulting thread's backtrace and every thread's.
  [ "$GDB" = 1 ] && [ $w = jetson ] && cmd="${WENV:+env $WENV }gdb -q -batch -ex 'set pagination off' -ex 'set print thread-events off' -ex 'handle SIGPIPE nostop noprint pass' -ex 'handle SIGTERM nostop noprint pass' -ex run -ex 'bt 40' -ex 'info registers pc lr sp x0 x1 x2 x3' -ex 'x/6i \\\$pc-12' -ex 'thread apply all bt 16' --args ${TREE[$w]}/build/bin/ggml-rpc-server -H ${IP[$w]} -p $PORT ${DEV[$w]}"
  on $w "mkdir -p ~/logs; setsid nohup bash -c \"$cmd; echo WORKER_EXIT=\\\$? \\\$(date -Is)\" > $WL 2>&1 < /dev/null &"
  for _ in $(seq 20); do on $w "ss -ltn | grep -q '${IP[$w]}:$PORT '" && break; sleep 1; done
  echo "worker $w pid $(wpid $w): $cmd"
  pr=memprobe.sh; [ $w = jetson ] && pr=memprobe_tegra.sh
  put $w testbed/scripts/$pr
  PP[$w]=$(on $w "setsid nohup bash ~/$pr ~/logs/memprobe_memdiag_${CASE}_$STAMP.log ggml-rpc-server > /dev/null 2>&1 < /dev/null & echo \$!")
done
sleep 2
echo "===== llama-bench -m $M -rpc $RPCS -ngl $NGL ${TS:+-ts $TS} -t 4 -p $P -n $N -r 1 -v $EXTRA"
timeout 1800 "$BENCH" -m "$M" -rpc "$RPCS" -ngl "$NGL" ${TS:+-ts "$TS"} -t 4 -p "$P" -n "$N" -r 1 -v -oe jsonl $EXTRA < /dev/null
echo "EXIT_BENCH=$?"
sleep 3
for w in $WORKERS; do
  echo "----- $w: worker alive after: $([ -n "$(wpid $w)" ] && echo yes || echo NO)"
  on $w "kill ${PP[$w]}"
  on $w "echo '--- worker log'; grep -v 'onnection' ~/logs/rpc-server_memdiag_${CASE}_$STAMP.log | grep -v '^[0-9a-f]*-[0-9a-f]* ' | tail -40"
  [ $w = jetson ] && $SSH jetson "echo '--- kern.log since the start'; sed -n '$((KL0 + 1)),\$p' /var/log/kern.log | grep -v railgate | tail -20"
  get $w logs/memprobe_memdiag_${CASE}_$STAMP.log testbed/results/memprobe_memdiag_${CASE}_${STAMP}_$w.log || echo "!!!!! could not copy the probe log from $w"
  awk -v W=$w '{a=substr($2,10)+0; if (NR==1) f=a; if (min=="" || a<min) min=a
        for (i=3;i<=NF;i++) { if (split($i,e,"=")==2 && e[1]=="nvmap_used_kb" && e[2]+0>nu) nu=e[2]+0
                              if (split($i,kv,":")==2) { v=kv[2]+0; if (v>mx[kv[1]]) mx[kv[1]]=v } } }
        END {printf "MEMPROBE %s: first avail %d MiB, min %d (drop %d); peak NvMapMemUsed %d MiB; peak worker", W, f/1024, min/1024, (f-min)/1024, nu/1024
             for (k in mx) printf " %s %d", k, mx[k]/1024; printf " MiB (%d samples)\n", NR}' testbed/results/memprobe_memdiag_${CASE}_${STAMP}_$w.log
  for p in $(wpid $w); do on $w "kill $p"; done
  sleep 2; [ -z "$(wpid $w)" ] && echo "worker $w stopped" || echo "!!!!! worker $w still running"
  get $w logs/rpc-server_memdiag_${CASE}_$STAMP.log testbed/results/rpc-server_${w}_memdiag_${CASE}_$STAMP.log
  grep WORKER_EXIT testbed/results/rpc-server_${w}_memdiag_${CASE}_$STAMP.log || echo "(no WORKER_EXIT line yet)"
  state $w after
done
echo "===== the host's view of every device (llama.cpp -v)"
grep -E 'using device|offloaded|model buffer size|KV buffer size|RS buffer size|compute buffer size|output buffer size|graph nodes|graph splits|n_ctx  |n_ubatch' "$LOG" | grep -v '^=====' | sort -u
echo "log: $LOG"
