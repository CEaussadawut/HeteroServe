#!/bin/bash
# The replacement Pi 5 after its bootloader update (2026-10-20), registered in testbed/dataset/pi5b_fw_predictions_20261020.json.
# The as-found characterisation (pi5b_char_20261019.sh, bootloader 2024-09-23) measured the new board slower than the old one,
# most where the most bytes move; its registration named the bootloader as the first suspect (the old board ran 2026-01-21,
# and later bootloaders changed the Pi 5's SDRAM settings). With the user's yes (2026-10-20 ~00:20: "Yes, update it"), the
# official update on the card was installed (rpi-eeprom-update -a) and the board rebooted; this re-takes the same rows by the
# same scripts, and the Q4_K_M energy run, so the firmware's share of the gap is measured. From the repo root on the laptop:
#     systemd-run --user --unit heteroserve-pi5fw --collect -p WorkingDirectory=$PWD systemd-inhibit --what=sleep:idle \
#         --why="HeteroServe Pi 5 after its bootloader update" bash testbed/scripts/pi5b_fw_20261020.sh
# Steps as pi5b_char_20261019.sh's 1-11 (memory; host rows pp512/tg128 and pp64/tg16; worker rows both protocols; the worker
# prefill-size sweep; energy per token at Q4_K_M (the 2026-10-05 protocol), Q4_0, Q8_0, Q3_K_M and Q6_K). The as-found session
# was stopped after its step 7 (the Q4_K_M energy run, RB-E's): its other four energy runs would have described a firmware
# state this update ends. Refuses unless the registration is read-only and the board reports the registered bootloader.
# Output: testbed/results/pi5b_fw_session_<stamp>/ (session.log tee'd).
set -uo pipefail
cd "$(dirname "$0")/../.."
REG=testbed/dataset/pi5b_fw_predictions_20261020.json
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=testbed/results/pi5b_fw_session_$STAMP
mkdir -p "$OUT"
exec > >(tee -i -a "$OUT/session.log") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
echo "########## $(date -Is) the replacement Pi 5 after its bootloader update -> $OUT"
[ -f "$REG" ] && [ ! -w "$REG" ] || { echo "FATAL: $REG missing or writable (register first)"; exit 2; }
echo "registration $REG sha256 $(sha256sum "$REG" | cut -d' ' -f1)"
serial=$($SSH pi5 "tr -d '\0' < /proc/device-tree/serial-number")
[ "$serial" = b8ed9049ada12a03 ] || { echo "FATAL: the Pi 5 answers with serial '$serial', not the registered board"; exit 2; }
want=$(python3 -c "import json; print(json.load(open('$REG'))['bootloader_after']['timestamp'])")
got=$($SSH pi5 "vcgencmd bootloader_version | awk '/^timestamp/ {print \$2}'")
echo "bootloader: $($SSH pi5 'vcgencmd bootloader_version | head -1') (timestamp $got, registered $want)"
[ "$got" = "$want" ] || { echo "FATAL: the Pi 5's bootloader timestamp '$got' is not the registered '$want'"; exit 2; }
for n in pi5 pi4 jetson; do
  b=$($SSH $n 'for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p; done | wc -l')
  [ "$b" = 0 ] || { echo "FATAL: $n is busy ($b llama.cpp processes)"; exit 2; }
done
[ "$($SSH pi5 'timedatectl show -p NTPSynchronized --value')" = yes ] || { echo "FATAL: the Pi 5's clock is not synchronized"; exit 2; }
[ "$($SSH pi5 'nmcli radio wifi')" = disabled ] || { echo "FATAL: the Pi 5's Wi-Fi is not off"; exit 2; }
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
for d in $(cut -f4 "$OUT/steps.tsv" | grep -E 'quant_campaign|worker_role'); do
  bad=$(grep -hE '^(SKIP|FATAL|!!!!!)|^EXIT_.*=[1-9]' "$d"/pi5.log 2>/dev/null)
  runs=$(grep -c '^EXIT_' "$d/pi5.log" 2>/dev/null)
  [ -z "$bad" ] && [ "${runs:-0}" -ge 1 ] || { echo "SESSION STOPPED: $d: runs=${runs:-0} $bad"; exit 1; }
  echo "checked $d: $runs runs, all exit 0"
done
energy() {  # n quant file reps_pp reps_tg idle_after
  local n=$1 q=$2 f=$3 rp=$4 rt=$5 ia=$6 sha d
  sha=$(python3 -c "import json; print(json.load(open('$REG'))['energy']['runs']['$q']['sha256'])")
  d=testbed/results/energy_phase_pi5fw_${q}_$(date +%Y%m%d_%H%M%S)
  step $n energy_$q "$d" python3 testbed/scripts/energy_phase_run.py --node pi5 --outlet 2 --outlet-name pi5 \
    --model "~/models/$f" --sha256 "$sha" --pp 512:$rp --tg 128:$rt --cycles 3 --threads 4 \
    --idle-before 300 --idle-after $ia --settle 30 --gap 10 --out "$d"
  python3 testbed/scripts/score_energy_phase_run.py "$d" --registration "testbed/dataset/energy_predictions_20261020_pi5fw_$q.json" \
    --out "$d/score.json" > "$d/score.txt" 2>&1
  echo "scored $d rc=$?"; sed -n '1,12p' "$d/score.txt"
}
energy 7 q4km tinyllama-1.1b-q4km.gguf 20 17 300
energy 8 q4_0 tinyllama-1.1b-q4_0.gguf 32 18 120
energy 9 q8_0 tinyllama-1.1b-q8_0.gguf 15 10 120
energy 10 q3km tinyllama-1.1b-q3km.gguf 10 21 120
energy 11 q6k tinyllama-1.1b-q6_k.gguf 13 12 120

python3 testbed/scripts/score_pi5b_20261019.py "$OUT" --registration "$REG"
python3 testbed/scripts/score_pi5b_fw_20261020.py "$OUT" --registration "$REG"
echo "SESSION DONE $(date -Is)"
