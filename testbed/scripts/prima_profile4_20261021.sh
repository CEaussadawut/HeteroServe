#!/bin/bash
# prima.cpp's own profile of all four nodes (2026-10-21; user: "build on Jetson + laptop", then pick the model where prima.cpp +
# Halda and the engine both use every node). One short prima.cpp run over the ring laptop (rank 0, the head, llama-cli with HiGHS)
# -> Pi 5 (1) -> Pi 4 (2) -> Jetson (3, CPU only) -> laptop, on the home LAN, with TinyLlama-1.1B Q4_K_M (on every node,
# sha256-checked here): each rank profiles its device (CPU FLOPS per weight type, memory, disk by fio for ~2 s), rank 0 prints
# every device's profile (device_print_props) and Halda's windows, then 16 tokens are generated. Not a scored run: its output
# is the input Halda needs, read later from the logs. Workers start first (rank 3, 2, 1), then rank 0; workers are stopped by
# exact PID afterwards (never pkill -f). The Jetson's power guard runs throughout, by its NOTE's procedure.
# From the repo root, with nothing else running on the cluster:  bash testbed/scripts/prima_profile4_20261021.sh
# Raw output tee'd to testbed/results/prima_profile4_<stamp>.log; each rank's log copied beside it.
set -uo pipefail
cd "$(dirname "$0")/../.."
STAMP=$(date +%Y%m%d_%H%M%S); OUT=$PWD/testbed/results/prima_profile4_$STAMP; LOG=$OUT.log   # absolute: rank 0 runs from ~/prima.cpp
exec > >(tee -a "$LOG") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
# M/SUM pick the model (any file on every node): prima.cpp profiles only the weight types the model has (Q8_0: run it on
# tinyllama-1.1b-q8_0.gguf, a4c9bb1d...)
M=${M:-models/tinyllama-1.1b-q4km.gguf}; SUM=${SUM:-9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0}
declare -A IP=([laptop]=192.168.1.108 [pi5]=192.168.1.122 [pi4]=192.168.1.110 [jetson]=192.168.1.114)
RING=(laptop pi5 pi4 jetson)
run() { if [ "$1" = laptop ]; then bash -c "$2"; else $SSH "$1" "$2" 2>/dev/null; fi; }
pids() {   # prima.cpp processes of rank $2 on node $1, by exact argv
  run "$1" "for p in \$(pgrep -x llama-cli; pgrep -x llama-server); do tr '\\0' ' ' < /proc/\$p/cmdline 2>/dev/null | grep -qF -- '--world 4 --rank $2 ' && echo \$p; done"
}
echo "########## $(date -Is) prima.cpp four-node profile, ring ${RING[*]}, $M"
for n in "${RING[@]}"; do
  busy=$(run $n 'for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null && printf "%s " $p; done; true')
  [ -z "$busy" ] || { echo "FATAL: $n is busy: $busy"; exit 1; }
  s=$(run $n "sha256sum \$(readlink -f ~/$M) | cut -d' ' -f1")
  [ "$s" = "$SUM" ] || { echo "FATAL: $n's $M is $s"; exit 1; }
  echo "ok      $n: idle, $M sha256 ${s:0:16}, MemAvailable $(run $n "grep MemAvailable /proc/meminfo | awk '{print int(\$2/1024)}'") MiB"
done
g=$($SSH jetson 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum ~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1' 2>/dev/null | tr '\n' ' ')
[ "$g" = "synced $(sha256sum testbed/scripts/jetson_power_guard.sh | cut -d' ' -f1) " ] || { echo "FATAL: Jetson guard precheck '$g'"; exit 1; }
$SSH jetson 'sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo guard started' 2>/dev/null
for r in 3 2 1; do
  n=${RING[$r]}; nxt=${IP[${RING[$(( (r + 1) % 4 ))]}]}
  cmd="./llama-cli -m ~/$M -t 4 --world 4 --rank $r --master ${IP[laptop]} --next $nxt --prefetch"
  echo "start   rank $r $n: $cmd"
  run $n "mkdir -p ~/logs; cd ~/prima.cpp; setsid nohup $cmd > ~/logs/prima_profile4_rank${r}_$STAMP.log 2>&1 < /dev/null &"
  sleep 3
done
cmd="./llama-cli -m ~/$M -t 4 --world 4 --rank 0 --master ${IP[laptop]} --next ${IP[pi5]} --prefetch -c 1024 -n 16 -p 'The capital of France is'"
echo "start   rank 0 laptop: $cmd ($(date -Is))"
# the laptop has no system fio (prima.cpp runs it to profile the disk): fio 3.41 built in user space from axboe/fio @ fio-3.41
# (ed675d34..., the user's OK 2026-10-21 ~20:40), on rank 0's PATH only
FIO_BIN=$HOME/.local/fio-3.41/bin; echo "fio     laptop: $(PATH=$FIO_BIN:$PATH command -v fio) $(PATH=$FIO_BIN:$PATH fio --version)"
(cd ~/prima.cpp && PATH=$FIO_BIN:$PATH timeout 900 bash -c "$cmd" > "$OUT"_rank0.log 2>&1 < /dev/null; echo "rank 0 exit $? ($(date -Is))")
for r in 1 2 3; do
  n=${RING[$r]}
  echo "rank $r $n: $(run $n "grep -a -E 'Using window size|No layer is assigned' ~/logs/prima_profile4_rank${r}_$STAMP.log | tail -2 | tr '\n' ' '")"
  for p in $(pids $n $r); do run $n "kill $p" && echo "stopped rank $r pid $p on $n"; done
  sleep 2; left=$(pids $n $r); [ -z "$left" ] || echo "WARNING: rank $r on $n still running: $left"
  scp -q $n:logs/prima_profile4_rank${r}_$STAMP.log "$OUT"_rank$r.log
done
$SSH jetson 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p stopped"; done; echo "TRIP lines: $(grep -c TRIP ~/heteroserve/logs/power-guard.log)"' 2>/dev/null
grep -a -E "Using window size|No layer is assigned" "$OUT"_rank0.log | tail -2
echo "PROFILE DONE $(date -Is)"
