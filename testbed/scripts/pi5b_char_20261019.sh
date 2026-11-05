#!/bin/bash
# The replacement Pi 5's characterisation (2026-10-19), registered in testbed/dataset/pi5b_predictions_20261019.json.
# The Pi 5 came back on a replacement board (serial b8ed9049ada12a03) running the old SD card; every Pi 5 row in
# characterisation.csv and its one energy run are the old board's (serial b266e1877252f91e). The user (2026-10-19):
# re-characterise it and compare before merging any data. This takes every row the old board has, by the same scripts
# and protocols, plus Q6_K and the pooled-protocol Q4_0 / Q8_0 rows it never had; then its energy per token at five
# quantisations. From the repo root on the laptop, AFTER the registration, with the cluster otherwise idle:
#     systemd-run --user --unit heteroserve-pi5b --collect -p WorkingDirectory=$PWD systemd-inhibit --what=sleep:idle \
#         --why="HeteroServe Pi 5 characterisation" bash testbed/scripts/pi5b_char_20261019.sh
# Steps (each logs "STEP <n> <name> rc=<rc> dir=<run dir>" and appends to steps.tsv; it stops at the first failure):
#   1 memory at rest (memory_snapshot.py --reads 10)
#   2 host rows pp512 / tg128      3 host rows pp64 / tg16          (run_quant_campaign.sh, NODES=pi5)
#   4 worker rows pp512 / tg128    5 worker rows pp64 / tg16        (bench_worker_role.sh, NODES=pi5, loopback worker)
#   6 the worker prefill-size sweep (pp1,4,16,256 on Q3_K_M, as 2026-09-28 22:03)
#   7-11 energy per token (energy_phase_run.py, outlet 2 "pi5"): Q4_K_M at the 2026-10-05 protocol, then Q4_0, Q8_0,
#        Q3_K_M, Q6_K, each scored by score_energy_phase_run.py against its own registration (written with this one)
# Then score_pi5b_20261019.py scores the whole session. Output: testbed/results/pi5b_session_<stamp>/ (session.log tee'd).
set -uo pipefail
cd "$(dirname "$0")/../.."
REG=testbed/dataset/pi5b_predictions_20261019.json
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=testbed/results/pi5b_session_$STAMP
mkdir -p "$OUT"
exec > >(tee -i -a "$OUT/session.log") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
echo "########## $(date -Is) the replacement Pi 5's characterisation -> $OUT"
[ -f "$REG" ] && [ ! -w "$REG" ] || { echo "FATAL: $REG missing or writable (register first)"; exit 2; }
echo "registration $REG sha256 $(sha256sum "$REG" | cut -d' ' -f1)"
serial=$($SSH pi5 "tr -d '\0' < /proc/device-tree/serial-number")
[ "$serial" = b8ed9049ada12a03 ] || { echo "FATAL: the Pi 5 answers with serial '$serial', not the registered board"; exit 2; }
for n in pi5 pi4 jetson; do
  b=$($SSH $n 'for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p; done | wc -l')
  [ "$b" = 0 ] || { echo "FATAL: $n is busy ($b llama.cpp processes)"; exit 2; }
done
[ "$($SSH pi5 'timedatectl show -p NTPSynchronized --value')" = yes ] || { echo "FATAL: the Pi 5's clock is not synchronized"; exit 2; }
: > "$OUT/steps.tsv"

newest() { ls -d $1 2>/dev/null | sort | tail -1; }
step() {  # n name glob-of-the-run-dir command...
  local n=$1 name=$2 glob=$3 before after rc; shift 3
  before=$(newest "$glob")
  echo "########## $(date -Is) STEP $n $name: $*"
  "$@"; rc=$?
  after=$(newest "$glob")
  [ "$after" != "$before" ] || after=""
  echo "STEP $n $name rc=$rc dir=${after:-none} $(date -Is)"
  printf '%s\t%s\t%s\t%s\n' "$n" "$name" "$rc" "${after:-}" >> "$OUT/steps.tsv"
  [ $rc = 0 ] || { echo "SESSION STOPPED at step $n"; exit 1; }
}
HOST_MODELS="tinyllama-1.1b-q4km.gguf tinyllama-1.1b-q8_0.gguf tinyllama-1.1b-q4_0.gguf tinyllama-1.1b-q3km.gguf tinyllama-1.1b-q6_k.gguf qwen3.5-9b-q4_k_m.gguf"
WORKER_512="tinyllama-1.1b-q4km.gguf tinyllama-1.1b-q8_0.gguf tinyllama-1.1b-q4_0.gguf tinyllama-1.1b-q3km.gguf tinyllama-1.1b-q6_k.gguf"
WORKER_64="$WORKER_512 qwen3.5-9b-q4_k_m.gguf"

step 1 memory 'testbed/dataset/memory_snapshot_2*.json' python3 testbed/scripts/memory_snapshot.py --reads 10
step 2 host_pp512_tg128 'testbed/results/quant_campaign_2*' env NODES=pi5 MODELS="$HOST_MODELS" bash testbed/scripts/run_quant_campaign.sh
step 3 host_pp64_tg16 'testbed/results/quant_campaign_2*' env NODES=pi5 MODELS="$HOST_MODELS" PP=64 TG=16 bash testbed/scripts/run_quant_campaign.sh
step 4 worker_pp512_tg128 'testbed/results/worker_role_2*' env NODES=pi5 MODELS_PI5="$WORKER_512" bash testbed/scripts/bench_worker_role.sh
step 5 worker_pp64_tg16 'testbed/results/worker_role_2*' env NODES=pi5 MODELS_PI5="$WORKER_64" PP=64 TG=16 bash testbed/scripts/bench_worker_role.sh
step 6 worker_pp_sweep 'testbed/results/worker_role_2*' env NODES=pi5 MODELS_PI5=tinyllama-1.1b-q3km.gguf PP=1,4,16,256 TG=0 bash testbed/scripts/bench_worker_role.sh
# a campaign script's own failures (a SKIP, a FATAL inside a board's log) do not set its exit code: check every run
for d in $(cut -f4 "$OUT/steps.tsv" | grep -E 'quant_campaign|worker_role'); do
  bad=$(grep -hE '^(SKIP|FATAL|!!!!!)|^EXIT_.*=[1-9]' "$d"/pi5.log 2>/dev/null)
  runs=$(grep -c '^EXIT_' "$d/pi5.log" 2>/dev/null)
  [ -z "$bad" ] && [ "${runs:-0}" -ge 1 ] || { echo "SESSION STOPPED: $d: runs=${runs:-0} $bad"; exit 1; }
  echo "checked $d: $runs runs, all exit 0"
done

# Energy per token. Reps sized to about 120 s of timed repetitions at the old board's rate (Q6_K: estimated), as on 10-05.
energy() {  # n quant file reps_pp reps_tg idle_after
  local n=$1 q=$2 f=$3 rp=$4 rt=$5 ia=$6 sha d
  sha=$(python3 -c "import json,sys; print(json.load(open('$REG'))['energy']['runs']['$q']['sha256'])")
  d=testbed/results/energy_phase_pi5b_${q}_$(date +%Y%m%d_%H%M%S)
  step $n energy_$q "$d" python3 testbed/scripts/energy_phase_run.py --node pi5 --outlet 2 --outlet-name pi5 \
    --model "~/models/$f" --sha256 "$sha" --pp 512:$rp --tg 128:$rt --cycles 3 --threads 4 \
    --idle-before 300 --idle-after $ia --settle 30 --gap 10 --out "$d"
  python3 testbed/scripts/score_energy_phase_run.py "$d" --registration "testbed/dataset/energy_predictions_20261019_pi5b_$q.json" \
    --out "$d/score.json" > "$d/score.txt" 2>&1
  echo "scored $d rc=$?"; sed -n '1,12p' "$d/score.txt"
}
energy 7 q4km tinyllama-1.1b-q4km.gguf 20 17 300
energy 8 q4_0 tinyllama-1.1b-q4_0.gguf 32 18 120
energy 9 q8_0 tinyllama-1.1b-q8_0.gguf 15 10 120
energy 10 q3km tinyllama-1.1b-q3km.gguf 10 21 120
energy 11 q6k tinyllama-1.1b-q6_k.gguf 13 12 120

python3 testbed/scripts/score_pi5b_20261019.py "$OUT" --registration "$REG"
echo "SESSION DONE $(date -Is)"
