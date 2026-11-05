#!/bin/bash
# freqprobe.sh LOGFILE -- ten times a second until killed: the time in ms and each CPU's cpufreq scaling_cur_freq (kHz):
#     <epoch_ms> <cpu0_khz> <cpu1_khz> <cpu2_khz> <cpu3_khz>
# Bash builtins only, like memprobe.sh v2: a sample starts no process. Why (2026-09-29 evening): the Pis run the ondemand
# governor (up_threshold 50, sampling_rate 100 ms). A board that computes alone stays at full clock, but in a pipeline each
# board idles while the other computes its part of every token. If ondemand drops the clock in that gap and needs a
# sampling period or more to bring it back, every decoded token pays for it -- a cost no single-board row contains,
# which would vary with how the token period falls against the 100 ms sampling.
f=$1
exec {tick}<> <(:)
while :; do
  t=${EPOCHREALTIME/./}; l=${t:0:13}                  # epoch in ms (bash >= 5)
  for c in 0 1 2 3; do read -r v < /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq; l+=" $v"; done
  printf '%s\n' "$l" >> "$f"
  read -r -t 0.1 -u $tick _
done
