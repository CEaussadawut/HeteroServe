#!/bin/bash
# Standalone characterisation of the Jetson Nano on the model and settings the Pi baselines
# used (TinyLlama-1.1B Q4_K_M, byte-identical to the Pis' copy; llama-bench pp512 / tg128,
# 4 threads), at the cluster pin b23701f7 built with Vulkan (testbed/scripts/
# jetson_build_llama_vulkan.sh). Two cases: the GPU through Vulkan, and CPU-only.
#
# House rules applied: >= 3 repetitions with every sample recorded (-oe jsonl); raw output
# tee'd; power mode, clock lock, swap, temperatures and over-current alarms logged before
# and after each case; the board's own INA3221 traced throughout, with an idle-floor sample
# in the cooldown before each case. jetson_clocks is applied for the run (the kit's
# LOCK_CLOCKS=1) and restored afterwards.
#
# Run ON the board, as user jetson, with jetson_power_guard.sh already running:
#   ssh jetson 'setsid nohup bash ~/heteroserve/tools/jetson_bench_tinyllama.sh </dev/null >/dev/null 2>&1 &'
set -uo pipefail

# MODEL / TAG / CASES / COLD_C are knobs added 2026-09-27 for the quantisation and cold re-take runs;
# every default reproduces the original run (Q4_K_M, both cases, output dir named "tinyllama").
M=${MODEL:-$HOME/models/tinyllama-1.1b-q4km.gguf}
TAG=${TAG:-tinyllama}
CASES=${CASES:-vulkan cpu}
# Cold gate: the protocol wants the CPU below 40 C at the START of every case, and a fixed 30 s
# cooldown did not deliver it (the 2026-09-25 cpu case began at 46 C). Wait up to COLD_WAIT_S for
# it; if it never gets there, run anyway and say so -- the summariser flags a warm start.
COLD_C=${COLD_C:-40}
COLD_WAIT_S=${COLD_WAIT_S:-1200}
# MODE=vulkan (default): Vulkan GPU case + CPU-only case, from the Vulkan build.
# MODE=cuda: the CUDA GPU case, from the CUDA 10.2 build -- same protocol, same model.
MODE=${MODE:-vulkan}
BIN=${BIN:-$HOME/llama.cpp/build/bin}
BUILDINFO=${BUILDINFO:-$HOME/heteroserve/build_info.json}
OUT=$HOME/heteroserve/results/jetson_${TAG}_${MODE}_$(date +%Y%m%d_%H%M%S)
REPS=${REPS:-5}
# PP / TG (2026-09-28): the prompt and generation lengths, default 512 / 128. PP=64 TG=16 takes the rows at the
# pooled protocol instead (run_pool.sh), so the predictor can price a cell from rows measured the way it runs.
PP=${PP:-512}; TG=${TG:-128}
mkdir -p "$OUT"
echo "$OUT" > "$HOME/heteroserve/results/LATEST"
rm -f "$HOME/heteroserve/results/bench.done"
trap 'echo $? > "$HOME/heteroserve/results/bench.done"' EXIT
exec > >(tee -a "$OUT/run.log") 2>&1

GPU=/sys/devices/57000000.gpu/devfreq/57000000.gpu
state() {
  local z t=""
  for z in /sys/devices/virtual/thermal/thermal_zone{1,2}; do
    t+="$(cat $z/type)=$(( $(cat $z/temp) / 1000 ))C "
  done
  echo "[$1] $(date -Is) $(cut -d' ' -f1 /var/lib/nvpmodel/status) online=$(cat /sys/devices/system/cpu/online)" \
       "cpu0 min/max=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)/$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)" \
       "gpu min/max=$(cat $GPU/min_freq)/$(cat $GPU/max_freq)" \
       "swap_used_kB=$(awk 'NR>1{s+=$4} END{print s+0}' /proc/swaps)" \
       "alarms=$(dmesg | grep -ciE 'oc alarm|over.?current|under.?volt') ${t}"
}

echo "########## $(date -Is) Jetson Nano standalone, $(basename "$M"), pp${PP}/tg${TG}, -r ${REPS}, cases: ${CASES}"
echo "mode   : ${MODE}   binaries: ${BIN}"
echo "commit : $(git -C "$(dirname "$(dirname "$BIN")")" rev-parse HEAD)   patched files: $(git -C "$(dirname "$(dirname "$BIN")")" diff --name-only | wc -l)"
echo "model  : $(sha256sum "$M")"
echo "build  : $(tr -d '\n' < "$BUILDINFO" | tr -s ' ')"
"$BIN/llama-bench" --list-devices 2>&1 | head -8

# Measurement protocol: swap off (as on both Pis), clocks locked for the run.
sudo -n swapoff -a
sudo -n jetson_clocks --store "$OUT/jetson_clocks.stored" 2>/dev/null || true
sudo -n jetson_clocks
sleep 5
state "clocks locked"

run_case() {                     # name, then llama-bench arguments
  local name=$1; shift
  echo; echo "##### case ${name}: $*"
  sleep 30                                               # cooldown between cases
  local waited=0 c
  while c=$(( $(cat /sys/devices/virtual/thermal/thermal_zone1/temp) / 1000 )); [ "$c" -ge "$COLD_C" ] && [ "$waited" -lt "$COLD_WAIT_S" ]; do
    sleep 10; waited=$((waited + 10))
  done
  echo "cold gate: waited ${waited}s for CPU < ${COLD_C} C, now ${c} C$([ "$c" -ge "$COLD_C" ] && echo ' -- NOT REACHED, starting warm')"
  state "before ${name}"
  sudo -n timeout 15 tegrastats --interval 500 > "$OUT/idle_before_${name}.txt" 2>/dev/null || true
  sudo -n tegrastats --start --interval 200 --logfile "$OUT/power_${name}.txt"
  echo "power_window_start ${name} $(date +%s.%N)"
  "$BIN/llama-bench" -m "$M" -p "$PP" -n "$TG" -t 4 -r "$REPS" -oe jsonl "$@"
  echo "EXIT_${name}=$?"
  echo "power_window_end ${name} $(date +%s.%N)"
  sudo -n tegrastats --stop
  state "after ${name}"
}

if [ "$MODE" = cuda ]; then
  run_case cuda   -ngl 99
else
  for c in $CASES; do
    case "$c" in
      vulkan) run_case vulkan -ngl 99 ;;
      cpu)    run_case cpu    -ngl 0 -dev none ;;
      *)      echo "unknown case '$c' (known: vulkan cpu)"; exit 2 ;;
    esac
  done
fi

sudo -n jetson_clocks --restore "$OUT/jetson_clocks.stored" 2>/dev/null || true
sleep 3
state "clocks restored"
echo "########## done $(date -Is)"
