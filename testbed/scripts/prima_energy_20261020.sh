#!/bin/bash
# The engine against prima.cpp on Pi 5 + Pi 4, with energy at the wall (2026-10-20), as registered in
# testbed/dataset/prima_energy_predictions_20261020.json: arms A (the engine's plan, llama.cpp) and B (prima.cpp + Halda),
# in the order A, B, A, B, each by qr3_tpot.py, inside one 5 Hz strip log with an idle window first and last; the Pi 5 below
# 50 C before each arm (up to 10 min). From the repo root, with the cluster otherwise idle:
#     bash testbed/scripts/prima_energy_20261020.sh
# Output: testbed/results/prima_energy_<stamp>/ (session.log, arms.tsv, strip.csv; each arm's qr3 log is in testbed/results).
set -u
cd "$(dirname "$0")/../.."
REG=testbed/dataset/prima_energy_predictions_20261020.json
OUT=testbed/results/prima_energy_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
exec > >(tee -i -a "$OUT/session.log") 2>&1
[ -f "$REG" ] && [ ! -w "$REG" ] || { echo "FATAL: $REG missing or writable (register first)"; exit 2; }
echo "########## $(date -Is) the engine against prima.cpp, with energy -> $OUT (registration $(sha256sum "$REG" | cut -c1-16))"
read -r PL PH < <(python3 -c "import json; a=json.load(open('$REG'))['arms']['A']['place_env']; print(a['PLACE'], a['PLACE_HEAD'])")
REQS=$(python3 -c "import json; print(json.load(open('$REG'))['reqs'])")
GEN=$(python3 -c "import json; print(json.load(open('$REG'))['gen'])")
echo "arm A placement: PLACE=$PL PLACE_HEAD=$PH; $REQS requests of 128 + $GEN per arm-run"
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "FATAL: serve_pool.sh state exists"; exit 1; }
for n in pi5 pi4; do
  [ "$(ssh -n -o BatchMode=yes $n 'timedatectl show -p NTPSynchronized --value')" = yes ] || { echo "FATAL: $n's clock is not synchronized"; exit 1; }
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
i=0
for arm in A B A B; do
  i=$((i + 1))
  echo "===== $(date -Is) arm-run $i: $arm"
  cold pi5 50
  if [ $arm = A ]; then
    PLACE="$PL" PLACE_HEAD="$PH" python3 testbed/scripts/qr3_tpot.py run --model qwen3_14b --ctx 4096 --nodes "pi5 pi4" \
      --reqs "$REQS" --n 128 --gen "$GEN" --tag "pe${arm}${i}"
  else
    python3 testbed/scripts/qr3_tpot.py run --model qwen3_14b --ctx 4096 --engine prima --reqs "$REQS" --n 128 --gen "$GEN" \
      --tag "pe${arm}${i}"
  fi
  rc=$?
  lg=$(ls -t testbed/results/qr3_pe${arm}${i}_qwen3_14b_*.log 2>/dev/null | grep -v '_prima_rank\|_server' | head -1)
  printf '%s\t%s\t%s\t%s\n' "$i" "$arm" "$rc" "$lg" >> "$OUT/arms.tsv"
  echo "ARM $i $arm rc=$rc log=$lg"
  sleep 20
done
echo "IDLE_AFTER start $(date +%s.%N)"; sleep 60; echo "IDLE_AFTER end $(date +%s.%N)"
kill -INT $LPID; wait $LPID 2>/dev/null
python3 testbed/scripts/score_prima_energy_20261020.py "$OUT" --registration "$REG"
echo "SESSION DONE $(date -Is)"
