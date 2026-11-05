#!/bin/bash
# The Jetson's CPU cost under prima.cpp's own code (2026-10-21): prima.cpp has no Vulkan, so in the four-node prima.cpp arm the
# Jetson computes its layers on its 4 Cortex-A57 cores, a cost the characterisation dataset never measured (its Jetson rows are
# Vulkan). prima.cpp (the Jetson's ~/prima.cpp tree: prima_build_20261021.sh + the gcc-8 NEON patch, CPU
# only), on TinyLlama-1.1B Q8_0 (sha256-checked) at -t 4: tg128 and pp512, three llama-bench runs of -r 3 (the house rule:
# at least three repetitions, the spread across runs quoted, never the within-run +-). The board's temperatures and guard line before and after.
# The power guard runs throughout, by its NOTE's procedure. From the repo root:
#     bash testbed/scripts/jetson_prima_cpu_bench_20261021.sh
# Raw output tee'd to testbed/results/jetson_prima_cpu_bench_<stamp>.log (llama-bench's CSV inside it).
set -uo pipefail
cd "$(dirname "$0")/../.."
STAMP=$(date +%Y%m%d_%H%M%S); LOG=testbed/results/jetson_prima_cpu_bench_$STAMP.log
exec > >(tee -a "$LOG") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
M=models/tinyllama-1.1b-q8_0.gguf; SUM=a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59
echo "########## $(date -Is) prima.cpp llama-bench on the Jetson's CPU, $M"
busy=$($SSH jetson 'for p in llama-bench llama-cli llama-server ggml-rpc-server cc1plus; do pgrep -x $p >/dev/null && printf "%s " $p; done; true' 2>/dev/null)
[ -z "$busy" ] || { echo "FATAL: the Jetson is busy: $busy"; exit 1; }
s=$($SSH jetson "sha256sum ~/$M | cut -d' ' -f1" 2>/dev/null); [ "$s" = "$SUM" ] || { echo "FATAL: $M on the Jetson is $s"; exit 1; }
g=$($SSH jetson 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum ~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1' 2>/dev/null | tr '\n' ' ')
[ "$g" = "synced $(sha256sum testbed/scripts/jetson_power_guard.sh | cut -d' ' -f1) " ] || { echo "FATAL: Jetson guard precheck '$g'"; exit 1; }
$SSH jetson 'sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo guard started' 2>/dev/null
# prima.cpp's llama-bench segfaults at start on the Jetson (2026-10-21 22:01, exit 139 on every run, before any output): it does
# not set up prima.cpp's distributed parameters. Its llama-cli on one device does: --world 1 with -lw 22 (every layer, a set
# window, so no profiler and no Halda), a ~512-token prompt (the first 2,048 bytes of GPL-3) and 128 tokens, --ignore-eos,
# greedy; the timings from its own llama_perf report. Three runs.
$SSH jetson 'head -c 2048 /usr/share/common-licenses/GPL-3 > ~/logs/prima_bench_prompt.txt; sha256sum ~/logs/prima_bench_prompt.txt' 2>/dev/null
echo "before: $($SSH jetson 'for z in /sys/devices/virtual/thermal/thermal_zone*; do printf "%s=%s " $(cat $z/type) $(cat $z/temp); done; tail -1 ~/heteroserve/logs/power-guard.log | cut -c1-100; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq' 2>/dev/null)"
for run in 1 2 3; do   # three runs, the spread quoted across them
  echo "run $run ($(date -Is))"
  out=$($SSH jetson "cd ~/prima.cpp && ./llama-cli -m ~/$M -t 4 -c 2048 --world 1 -lw 22 -n 128 --ignore-eos --temp 0 --seed 1 -f ~/logs/prima_bench_prompt.txt < /dev/null 2>&1 > /dev/null | grep -a -E 'llama_perf|eval time|load time|total time'" 2>/dev/null)
  echo "$out"
  pp=$(echo "$out" | grep -a 'prompt eval time' | sed -E 's/.*\/ *([0-9]+) tokens \( *([0-9.]+) ms per token.*/\1 \2/')
  tg=$(echo "$out" | grep -a -v 'prompt' | grep -a ' eval time' | sed -E 's/.*\/ *([0-9]+) runs *\( *([0-9.]+) ms per token.*/\1 \2/')
  echo "RESULT run $run pp_tokens_ms ${pp:-none} tg_runs_ms ${tg:-none}"
done
echo "after:  $($SSH jetson 'for z in /sys/devices/virtual/thermal/thermal_zone*; do printf "%s=%s " $(cat $z/type) $(cat $z/temp); done; tail -1 ~/heteroserve/logs/power-guard.log | cut -c1-100; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq' 2>/dev/null)"
$SSH jetson 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p stopped"; done; echo "TRIP lines: $(grep -c TRIP ~/heteroserve/logs/power-guard.log)"' 2>/dev/null
echo "BENCH DONE $(date -Is)"
