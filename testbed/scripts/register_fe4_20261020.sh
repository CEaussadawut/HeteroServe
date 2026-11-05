#!/bin/bash
# Register the full evaluation on four nodes (2026-10-20): the rest of the model list, at least five rounds each, every
# registration written before the first of them runs (fe4_chain_20261020.sh runs them in this order). The user, 2026-10-11:
# "run the full evaluation over the rest of the model list, at least five rounds each, starting with a model that needs the
# pool even with the laptop"; the advisor, 2026-10-16: "more baseline pair". Qwen3-14B comes first because it also answers
# the user's first 10-11 request (the laptop plan's energy, and the laptop alone); then the 30B MoE at Q6_K, the one model that
# needs the pool even with the laptop; then the rest.
# Each model's placements are derived by fe4_placements_20261020.py (engine, boards, single, even, llama.cpp's default, Halda,
# and any extra named here), its derivation kept beside the registration log and quoted by sha256 in the registration's note.
#     bash testbed/scripts/register_fe4_20261020.sh            # after the dataset carries the new Pi 5 and laptop rows
# Output: testbed/results/fe4_register_<stamp>/ (register.log, placements_<model>.json).
set -u
cd "$(dirname "$0")/../.."
F=testbed/scripts/full_eval.py
OUT=testbed/results/fe4_register_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
exec > >(tee -i -a "$OUT/register.log") 2>&1
NODES="pi5 pi4 jetson laptop"
echo "########## $(date -Is) register the four-node full evaluation -> $OUT"
for f in testbed/dataset/memory_at_rest.json testbed/dataset/characterisation.csv testbed/dataset/link.json testbed/dataset/model_shapes.json \
         $(ls testbed/dataset/model_inventory_*.json | sort | tail -1) $(ls testbed/dataset/laptop_worker_*.json | sort | tail -1); do
  echo "input  $f  $(sha256sum "$f" | cut -c1-16)"
done
python3 -c "import json,sys; m=json.load(open('testbed/dataset/memory_at_rest.json')); sys.exit(0 if 'laptop' in m else 1)" \
  || { echo "FATAL: memory_at_rest.json has no laptop entry (the laptop session's step 1)"; exit 2; }
reg() {  # key name reqs n gen warm_n warm_gen [extra placements ...]
  local key=$1 name=$2 reqs=$3 n=$4 gen=$5 wn=$6 wg=$7; shift 7
  local der=$OUT/placements_$key.json pl
  [ -f testbed/dataset/full_eval_$name.json ] && { echo "SKIP $name: already registered"; return 0; }
  echo "===== $(date +%T) $key -> $name"
  local extra=(); [ $# -gt 0 ] && extra=(--extra "$@")
  python3 testbed/scripts/fe4_placements_20261020.py --model "$key" --nodes "$NODES" --ctx 4096 --out "$der" "${extra[@]}" \
    || { echo "FAILED placements $key"; return 1; }
  pl=$(python3 -c "import json; print(json.load(open('$der'))['placements_arg'])")
  python3 $F register --name "$name" --model "$key" --nodes "$NODES" --placements "$pl" --rounds 5 --reqs "$reqs" --n "$n" \
    --gen "$gen" --warm-n "$wn" --warm-gen "$wg" --ctx 4096 --rtt-trips 1 \
    --note "The four-node full evaluation (2026-10-20), registered with the rest of the model list before any of it ran. Placements and why each was kept or dropped: $der (sha256 $(sha256sum "$der" | cut -c1-16)); a dropped baseline is a duplicate of an earlier one or does not fit the validated memory accounting with the engine's 256 MiB headroom. The new Pi 5 is priced by its own rows (after its bootloader update), the laptop by its rows of 2026-10-20. Outlets at run: pi4=0, jetson=1, pi5=2, laptop=3 (the laptop's outlet includes its USB-C hub)." \
    || { echo "FAILED register $name"; return 1; }
}
# large models: two requests of 128 + 64 tokens per placement per round, warm-up 64 + 8 (the weights are read into memory)
reg qwen3_14b    q14b_fe4_20261020   2 128 64 64 8 "served1009=pi4:27+jetson:13/host" "lap39=pi4:1+laptop:39/last"
reg qwen3moe_q6k moeq6_fe4_20261020  2 128 64 64 8 "lap_pi4=laptop:37+pi4:11/host"   # only two baselines fit: a board swap shows the choice
reg qwen3moe     moeq3_fe4_20261020  2 128 64 64 8
reg qwen38_27b   q27b_fe4_20261020   2 128 64 64 8
reg nemo12b      nemo_fe4_20261020   2 128 64 64 8
reg qwen35_9b    q9b_fe4_20261020    2 128 64 64 8 "pi4alone=pi4:33/last"           # the advisor's single-board baseline
# TinyLlama: the pilot's protocol (four requests of 128 + 96, its default warm-up)
# each board alone (the advisor's first baseline, 05/10): the laptop is the engine's pick, the Pi 5 the boards' pick
reg tinyllama      tinyq4km_fe4_20261020 4 128 96 512 32 "pi4alone=pi4:22/last" "jetsonalone=jetson:22/last"
reg tinyllama_q8_0 tinyq8_fe4_20261020   4 128 96 512 32 "pi4alone=pi4:22/last" "jetsonalone=jetson:22/last"
reg tinyllama_q4_0 tinyq40_fe4_20261020  4 128 96 512 32 "pi4alone=pi4:22/last" "jetsonalone=jetson:22/last"
reg tinyllama_q3km tinyq3_fe4_20261020   4 128 96 512 32 "pi4alone=pi4:22/last" "jetsonalone=jetson:22/last"
reg tinyllama_q6k  tinyq6_fe4_20261020   4 128 96 512 32 "pi4alone=pi4:22/last" "jetsonalone=jetson:22/last"
echo "REGISTERED $(date -Is): $(ls testbed/dataset/full_eval_*_fe4_20261020.json 2>/dev/null | wc -l) registrations"
