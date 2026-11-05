#!/bin/bash
# Where does the pool predictor's optimism come from? (2026-09-28; user: "find the cause, fix, re-test")
# The predictor prices every board from TinyLlama rows taken with the characterisation protocol
# (run_quant_campaign.sh: llama-bench -m M -t 4 -p 512 -n 128 -r 3 -oe jsonl, nothing else running), but every
# pool cell runs the pool protocol (run_pool.sh: llama-bench -m M -ngl 0 -t 4 -p 64 -n 16 -r 3 -oe jsonl -v, with
# memprobe.sh polling at 1 Hz on the board). On the Pi 5 alone -- no RPC, no network -- this runs three conditions
# on each model, in rounds (R P PQ, R P PQ, ...):
#   R   the characterisation protocol, exactly as run_quant_campaign.sh ran it
#   P   the pool protocol, exactly as run_pool.sh runs a Pi 5-alone cell, without the probe
#   PQ  the same with memprobe_v1.sh running: the probe every pool cell ran with through 2026-09-28
#   PQ2 the same with memprobe.sh (v2, builtins only: no process start per sample) running
# Per phase, R vs P is the protocol (prompt and generation length), P vs PQ the probe, and round against round the
# drift; the large model's R row against TinyLlama's R row, scaled per byte, is the model-size effect.
# Run from the repo root on the laptop:
#     bash testbed/scripts/protocol_ab_pi5.sh
#     MODELS="tinyllama-1.1b-q4km.gguf mistral-nemo-12b-q4_k_m.gguf"  ROUNDS=2  CONDS="R P PQ"  SETTLE=0  (defaults)
# SETTLE seconds of idle after the hash checks: the first run of 2026-09-28 17:47 (TinyLlama, R) came straight after
# the 7.5 GB Nemo hash and read 62.7 ms/token decode against 55.2 in the next round.
# House rules: each model's SHA-256 checked on the board first; -r 3 with every sample in jsonl; raw output tee'd;
# temperature and get_throttled before and after every run. One cold gate for every run: the characterisation's
# 50 C (run_pool.sh gates the Pi 5 at 55 C), so that the start temperature is not one of the differences.
set -uo pipefail
cd "$(dirname "$0")/../.."
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=testbed/results/protocol_ab_$STAMP
mkdir -p "$OUT"
exec > >(tee -i -a "$OUT/run.log") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
N=pi5
COLD=50
MODELS=${MODELS:-"tinyllama-1.1b-q4km.gguf mistral-nemo-12b-q4_k_m.gguf"}
ROUNDS=${ROUNDS:-2}
CONDS=${CONDS:-"R P PQ"}
SETTLE=${SETTLE:-0}
declare -A SHA=([tinyllama-1.1b-q4km.gguf]=9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0
                [mistral-nemo-12b-q4_k_m.gguf]=7c1a10d202d8788dbe5628dc962254d10654c853cae6aaeca0618f05490d4a46)
PIN=$(python3 -c "import json;print(json.load(open('ThinkPad/config/cluster.json'))['llama']['pin_sha'])")
PROBE=""

fatal() { echo "FATAL: $*"; exit 1; }
cleanup() {
  [ -n "$PROBE" ] && $SSH $N "kill $PROBE 2>/dev/null"
  PROBE=""
}
trap cleanup EXIT

echo "########## $(date -Is) protocol A/B on $N: models [$MODELS], $ROUNDS rounds of $CONDS, cold gate $COLD C"
b=$($SSH $N 'for p in llama-bench llama-cli llama-server ggml-rpc-server cc1plus cmake; do pgrep -x $p >/dev/null && printf "%s " $p; done; true') \
  || fatal "ssh $N failed"
[ -z "$b" ] || fatal "$N is busy: $b"
head=$($SSH $N 'git -C ~/llama.cpp rev-parse HEAD; git -C ~/llama.cpp status --porcelain --untracked-files=no | wc -l')
echo "llama.cpp on $N: $(echo $head)   (pin $PIN)"
[ "$(echo "$head" | head -1)" = "$PIN" ] || fatal "$N's llama.cpp is not at the pin"
[ "$(echo "$head" | tail -1)" = 0 ] || fatal "$N's llama.cpp tree has tracked changes"
for m in $MODELS; do
  [ -n "${SHA[$m]:-}" ] || fatal "no pinned SHA-256 for $m"
  s=$($SSH $N "sha256sum ~/models/$m | cut -d' ' -f1") || fatal "cannot hash $m on $N"
  echo "sha256 $m on $N: $s"
  [ "$s" = "${SHA[$m]}" ] || fatal "$m on $N does not match testbed/models.md"
done
scp -q testbed/scripts/memprobe.sh $N:memprobe.sh && scp -q testbed/scripts/memprobe_v1.sh $N:memprobe_v1.sh \
  || fatal "cannot copy the probes"
[ "$SETTLE" -gt 0 ] && { echo "settle: ${SETTLE}s idle after the hash checks"; sleep "$SETTLE"; }

run() {   # run MODEL COND ROUND
  local m=$1 c=$2 r=$3 tag="${1%.gguf}_$2_r$3" t w=0
  while t=$($SSH $N "vcgencmd measure_temp | tr -dc '0-9.' | cut -d. -f1"); [ "${t:-99}" -ge $COLD ] && [ $w -lt 600 ]; do
    sleep 15; w=$((w + 15)); done
  echo "##### RUN $tag: model $m, condition $c, round $r; cold gate waited ${w}s, now ${t} C (threshold $COLD C)"
  $SSH $N "echo \"STATE_BEFORE \$(date -Is) \$(vcgencmd measure_temp) \$(vcgencmd get_throttled) gov=\$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) memavail_mib=\$(awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo) load=\$(cut -d' ' -f1-3 /proc/loadavg) up_since=\$(uptime -s | tr ' ' T)\""
  if [ $c = PQ ] || [ $c = PQ2 ]; then
    pf=memprobe.sh; [ $c = PQ ] && pf=memprobe_v1.sh
    PROBE=$($SSH $N "mkdir -p ~/logs; setsid nohup bash ~/$pf ~/logs/memprobe_ab_${STAMP}_$tag.log llama-bench > /dev/null 2>&1 < /dev/null & echo \$!")
    echo "probe pid $PROBE ($pf)"
  fi
  case $c in
    R)   $SSH $N "cd ~/llama.cpp && ./build/bin/llama-bench -m ~/models/$m -t 4 -p 512 -n 128 -r 3 -oe jsonl; echo EXIT_BENCH=\$?" 2>&1 ;;
    P|PQ|PQ2) $SSH $N "cd ~ && /usr/bin/timeout 2700 ~/llama.cpp/build/bin/llama-bench -m ~/models/$m -ngl 0 -t 4 -p 64 -n 16 -r 3 -oe jsonl -v; echo EXIT_BENCH=\$?" 2>&1 ;;
  esac
  if [ $c = PQ ] || [ $c = PQ2 ]; then
    cleanup
    scp -q $N:logs/memprobe_ab_${STAMP}_$tag.log "$OUT/" && echo "probe log: $OUT/memprobe_ab_${STAMP}_$tag.log"
  fi
  $SSH $N "echo \"STATE_AFTER \$(date -Is) \$(vcgencmd measure_temp) \$(vcgencmd get_throttled)\""
}

for r in $(seq 1 $ROUNDS); do
  for m in $MODELS; do
    for c in $CONDS; do run $m $c $r; done
  done
done
echo "########## $(date -Is) done: $OUT"
