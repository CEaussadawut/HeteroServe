#!/bin/bash
# Does a Pi slow itself down below the 80 C hard limit? A 1 Hz log of the SoC temperature, the ARM clock the firmware
# actually runs (vcgencmd measure_clock arm) and get_throttled, while the board runs a steady llama.cpp load.
# From the repo root, on the laptop:     NODE=pi4 bash testbed/scripts/pi_thermal_clock_probe.sh
# Why (2026-09-29 evening): the same Qwen3.5-9B split (Pi 5 20 + Pi 4 13) decoded at 1,017.2 ms/token on 2026-09-29 00:18
# and at 1,112.8 today, unsteady (reps 0.904 / 0.860 / 0.931 t/s), with prefill identical (307.6 / 307.7). The Pi 4, which does
# most of that split's decode, finished the first at 58.9 C and today's at 61.3 C, and its get_throttled has carried the sticky
# 0x80000 ("soft temperature limit has occurred") since a 31-minute row at 16:17. If the firmware lowers the ARM clock at a soft
# limit near 60 C, every Pi 4 figure taken above it is slower than the rows the predictor prices it from.
# The probe forks two vcgencmd a second, so it runs only here, never beside a measured cell. The load is llama-bench's own
# (TinyLlama Q4_K_M pp512, -r 12: compute-bound, so a clock change shows fully); its t/s are not reported, the clock is.
set -u
cd "$(dirname "$0")/../.."
NODE=${NODE:?NODE=pi4 or pi5}; REPS=${REPS:-12}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=testbed/results/thermal_clock_${NODE}_$STAMP.log
exec > >(tee -a "$LOG") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
echo "########## $(date -Is) thermal/clock probe on $NODE, load: llama-bench TinyLlama Q4_K_M -p 512 -n 0 -r $REPS -t 4"
[ "$($SSH $NODE 'for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p; done | wc -l')" = 0 ] || { echo "FATAL: $NODE is busy"; exit 1; }
$SSH $NODE 'echo "config: $(vcgencmd get_config temp_soft_limit) $(vcgencmd get_config temp_limit) $(vcgencmd get_config arm_freq) $(vcgencmd get_config arm_freq_min)"; echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor), now $(vcgencmd measure_temp) $(vcgencmd get_throttled) arm=$(vcgencmd measure_clock arm)"'
P=$($SSH $NODE "mkdir -p ~/logs; setsid nohup bash -c 'while :; do echo \"\$(date +%s) \$(( \$(cat /sys/class/thermal/thermal_zone0/temp) / 100 )) \$(vcgencmd measure_clock arm | cut -d= -f2) \$(vcgencmd get_throttled | cut -d= -f2)\"; sleep 1; done' > ~/logs/thermal_clock_$STAMP.log 2>&1 < /dev/null & echo \$!")
echo "probe pid $P"; sleep 20
$SSH $NODE "cd ~/llama.cpp && timeout 1800 ./build/bin/llama-bench -m ~/models/tinyllama-1.1b-q4km.gguf -t 4 -p 512 -n 0 -r $REPS -oe jsonl" 2>&1
echo "EXIT_BENCH=$?"
sleep 60
$SSH $NODE "kill $P"
scp -q $NODE:logs/thermal_clock_$STAMP.log testbed/results/thermal_clock_${NODE}_${STAMP}.samples
echo "===== samples: epoch, temp (tenths of a degree C), ARM clock (Hz), get_throttled -- summarised by temperature band"
awk '{t=$2/10; b=int(t/2)*2; n[b]++; s[b]+=$3; if (!(b in mn) || $3<mn[b]) mn[b]=$3; if ($3>mx[b]) mx[b]=$3; th[b]=th[b] " " $4}
     END {for (b in n) printf "%5.0f-%-3.0f C  %4d s  ARM mean %4.0f MHz (min %4.0f, max %4.0f)  throttled seen:%s\n", b, b+2, n[b], s[b]/n[b]/1e6, mn[b]/1e6, mx[b]/1e6, th[b]}' \
    testbed/results/thermal_clock_${NODE}_${STAMP}.samples | sort -n | awk '{u=""; split($0,a,"throttled seen:"); split(a[2],v," "); delete seen; for (i in v) seen[v[i]]=1; for (k in seen) u=u" "k; print a[1] "throttled seen:" u}'
echo "log: $LOG"
