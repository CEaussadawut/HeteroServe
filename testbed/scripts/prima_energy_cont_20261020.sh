#!/bin/bash
# The prima.cpp energy session's fourth arm-run, run again (2026-10-20). The session of 06:16 (prima_energy_20261020.sh,
# testbed/results/prima_energy_20261020_061608/) completed arm-runs 1 (A), 2 (B) and 3 (A); the operator stopped its unit at
# 07:44:28, in run 4's (B) cold gate, before prima.cpp started (no request ran). This runs run 4 by the registered protocol
# (testbed/dataset/prima_energy_predictions_20261020.json: prima.cpp + Halda, 6 requests of 128 + 64, the Pi 5 below 50 C)
# inside its own 5 Hz strip log with its own idle windows, in a continuation folder; score_prima_energy_20261020.py pools it
# with --also. From the repo root:
#     bash testbed/scripts/prima_energy_cont_20261020.sh
set -u
cd "$(dirname "$0")/../.."
REG=testbed/dataset/prima_energy_predictions_20261020.json
FIRST=testbed/results/prima_energy_20261020_061608
OUT=${FIRST}_cont_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
exec > >(tee -i -a "$OUT/session.log") 2>&1
[ -f "$REG" ] && [ ! -w "$REG" ] || { echo "FATAL: $REG missing or writable"; exit 2; }
echo "########## $(date -Is) prima.cpp energy, arm-run 4 (B) again -> $OUT (registration $(sha256sum "$REG" | cut -c1-16); first session $FIRST)"
REQS=$(python3 -c "import json; print(json.load(open('$REG'))['reqs'])")
GEN=$(python3 -c "import json; print(json.load(open('$REG'))['gen'])")
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "FATAL: serve_pool.sh state exists"; exit 1; }
for n in pi5 pi4; do
  [ "$(ssh -n -o BatchMode=yes $n 'timedatectl show -p NTPSynchronized --value')" = yes ] || { echo "FATAL: $n's clock is not synchronized"; exit 1; }
  b=$(ssh -n -o BatchMode=yes $n 'for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p; done | wc -l')
  [ "$b" = 0 ] || { echo "FATAL: $n is busy ($b llama.cpp / prima.cpp processes)"; exit 1; }
done
cold() {   # host limit
  local t i
  for i in $(seq 60); do
    t=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$1" "vcgencmd measure_temp" | tr -dc '0-9.')
    awk -v t="$t" -v l="$2" 'BEGIN { exit !(t < l) }' && { echo "cold    $1 at $t C (< $2) after $(( (i - 1) * 10 )) s"; return 0; }
    sleep 10
  done
  echo "!!!!! $1 still at $t C after 10 min; starting anyway, recorded"
}
python3 testbed/scripts/shelly_strip.py log --out "$OUT/strip.csv" --hz 5 > "$OUT/strip_logger.stdout" 2> "$OUT/strip_logger.stderr" &
LPID=$!
sleep 20
kill -0 $LPID 2>/dev/null || { echo "FATAL: the strip logger exited"; cat "$OUT/strip_logger.stderr"; exit 1; }
echo "IDLE_BEFORE start $(date +%s.%N)"; sleep 180; echo "IDLE_BEFORE end $(date +%s.%N)"
: > "$OUT/arms.tsv"
echo "===== $(date -Is) arm-run 4: B"
cold pi5 50
python3 testbed/scripts/qr3_tpot.py run --model qwen3_14b --ctx 4096 --engine prima --reqs "$REQS" --n 128 --gen "$GEN" --tag peB4
rc=$?
lg=$(ls -t testbed/results/qr3_peB4_qwen3_14b_*.log 2>/dev/null | grep -v '_prima_rank\|_server' | head -1)
printf '%s\t%s\t%s\t%s\n' 4 B "$rc" "$lg" >> "$OUT/arms.tsv"
echo "ARM 4 B rc=$rc log=$lg"
sleep 20
echo "IDLE_AFTER start $(date +%s.%N)"; sleep 60; echo "IDLE_AFTER end $(date +%s.%N)"
kill -INT $LPID; wait $LPID 2>/dev/null
python3 testbed/scripts/score_prima_energy_20261020.py "$FIRST" --registration "$REG" --also "$OUT"
echo "SESSION DONE $(date -Is)"
