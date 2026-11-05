#!/bin/bash
# One prima.cpp llama-cli run over the four-node ring laptop (rank 0, the head) -> pi5 (1) -> pi4 (2) -> jetson (3) on the home LAN
# (2026-10-21), for checks that need prima.cpp itself rather than a scored arm (those go through qr3_tpot.py):
#   - the partial-copy check: the same fixed windows (LW) on full copies and on prima_partial_20261021.py's partial copies must
#     give the same greedy text;
#   - the Halda decision run: no LW, header-only copies on the boards; STOP_AFTER_WINDOWS=1 stops every rank once each has logged
#     its window, before a token is computed from the empty ranges.
# Env: TAG (names the logs), BOARD_MODEL (the path on the boards, e.g. ~/models/X.prima-partial.gguf), LAPTOP_MODEL (default
# the same name under ~/models), LW ("w0,w1,w2,w3", default: Halda decides), N (tokens, default 32), PROMPT, CTX (1024),
# STOP_AFTER_WINDOWS. Greedy (--temp 0, --seed 1). Workers start first (rank 3, 2, 1), then rank 0; stopped by exact PID after.
# The Jetson's power guard runs throughout, by its NOTE's procedure. Raw output in testbed/results/prima_run4_<TAG>_<stamp>*.
set -uo pipefail
cd "$(dirname "$0")/../.."
STAMP=$(date +%Y%m%d_%H%M%S); TAG=${TAG:-run}; OUT=$PWD/testbed/results/prima_run4_${TAG}_$STAMP
exec > >(tee -a "$OUT.log") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
BOARD_MODEL=${BOARD_MODEL:?}; LAPTOP_MODEL=${LAPTOP_MODEL:-$BOARD_MODEL}; N=${N:-32}; CTX=${CTX:-1024}
PROMPT=${PROMPT:-The capital of France is}; LW=${LW:-}; STOP_AFTER_WINDOWS=${STOP_AFTER_WINDOWS:-}
declare -A IP=([laptop]=192.168.1.108 [pi5]=192.168.1.122 [pi4]=192.168.1.110 [jetson]=192.168.1.114)
RING=(laptop pi5 pi4 jetson); W=$(( ${#RING[@]} ))
run() { if [ "$1" = laptop ]; then bash -c "$2"; else $SSH "$1" "$2" 2>/dev/null; fi; }
pids() { run "$1" "for p in \$(pgrep -x llama-cli; pgrep -x llama-server); do tr '\\0' ' ' < /proc/\$p/cmdline 2>/dev/null | grep -qF -- '--world $W --rank $2 ' && echo \$p; done"; }
stop_all() {
  for r in 0 1 2 3; do n=${RING[$r]}; for p in $(pids $n $r); do run $n "kill $p" && echo "stopped rank $r pid $p on $n"; done; done
  sleep 2; for r in 0 1 2 3; do n=${RING[$r]}; left=$(pids $n $r); [ -z "$left" ] || echo "WARNING: rank $r on $n still running: $left"; done
}
lw=""; [ -n "$LW" ] && lw="-lw $LW"
echo "########## $(date -Is) prima.cpp four-node run $TAG: boards $BOARD_MODEL, laptop $LAPTOP_MODEL, ${LW:-Halda decides}, $N tokens"
for n in "${RING[@]}"; do
  busy=$(run $n 'for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null && printf "%s " $p; done; true')
  [ -z "$busy" ] || { echo "FATAL: $n is busy: $busy"; exit 1; }
done
g=$($SSH jetson 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum ~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1' 2>/dev/null | tr '\n' ' ')
[ "$g" = "synced $(sha256sum testbed/scripts/jetson_power_guard.sh | cut -d' ' -f1) " ] || { echo "FATAL: Jetson guard precheck '$g'"; exit 1; }
$SSH jetson 'sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo guard started' 2>/dev/null
for r in 3 2 1; do
  n=${RING[$r]}; nxt=${IP[${RING[$(( (r + 1) % W ))]}]}
  cmd="./llama-cli -m $BOARD_MODEL -t 4 --world $W --rank $r --master ${IP[laptop]} --next $nxt --prefetch $lw"
  echo "start   rank $r $n: $cmd"
  run $n "mkdir -p ~/logs; cd ~/prima.cpp; setsid nohup $cmd > ~/logs/prima_run4_${TAG}_rank${r}_$STAMP.log 2>&1 < /dev/null &"
  sleep 3
done
FIO_BIN=$HOME/.local/fio-3.41/bin      # the laptop's user-space fio (prima.cpp's disk profile), as prima_profile4_20261021.sh
cmd="./llama-cli -m $LAPTOP_MODEL -t 4 --world $W --rank 0 --master ${IP[laptop]} --next ${IP[pi5]} --prefetch $lw -c $CTX -n $N --temp 0 --seed 1 -p '$PROMPT'"
echo "start   rank 0 laptop: $cmd ($(date -Is))"
( cd ~/prima.cpp && PATH=$FIO_BIN:$PATH setsid nohup timeout 1800 bash -c "$cmd" > "$OUT"_rank0.out 2> "$OUT"_rank0.log < /dev/null & echo $! > "$OUT".rank0pid )
if [ -n "$STOP_AFTER_WINDOWS" ]; then
  for i in $(seq 360); do
    got=0
    grep -a -q -E "Using window size|No layer is assigned" "$OUT"_rank0.log 2>/dev/null && got=$((got + 1))
    for r in 1 2 3; do run ${RING[$r]} "grep -a -q -E 'Using window size|No layer is assigned' ~/logs/prima_run4_${TAG}_rank${r}_$STAMP.log" && got=$((got + 1)); done
    [ $got = 4 ] && break
    sleep 5
  done
  echo "windows logged by $got of 4 ranks ($(date -Is))"; stop_all
else
  while kill -0 "$(cat "$OUT".rank0pid)" 2>/dev/null; do sleep 2; done
  echo "rank 0 finished ($(date -Is))"; stop_all
fi
for r in 0 1 2 3; do
  n=${RING[$r]}; [ $r = 0 ] || scp -q $n:logs/prima_run4_${TAG}_rank${r}_$STAMP.log "$OUT"_rank$r.log
  echo "rank $r $n: $(grep -a -E 'Using window size|No layer is assigned' "$OUT"_rank$r.log | tail -1)"
done
$SSH jetson 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p stopped"; done; echo "TRIP lines: $(grep -c TRIP ~/heteroserve/logs/power-guard.log)"' 2>/dev/null
[ -s "$OUT"_rank0.out ] && { echo "--- generated (rank 0 stdout, sha256 $(sha256sum "$OUT"_rank0.out | cut -c1-16)):"; cat "$OUT"_rank0.out; echo; }
echo "RUN DONE $(date -Is)"
