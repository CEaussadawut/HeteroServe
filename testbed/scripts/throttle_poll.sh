#!/bin/bash
# throttle_poll.sh LOGFILE -- v2 (2026-10-07 night): once a second until killed, on a Raspberry Pi: get_throttled, the SoC
# temperature and the clock the governor asks for; the clock the firmware actually runs, only in a second with a bit set.
# run_pool.sh runs it beside every Pi cell, so it must not slow the cell:
#  - v1 (throttle_poll_v1.sh, the 2026-10-07 evening cells) forked date and three vcgencmd a second: 101 processes in 20 s on
#    the Pi 4, the rate at which memprobe v1 cost 4.5 % of decode (benchmarks.md, 2026-09-28).
#  - v2 starts ONE process a second, `vcgencmd get_throttled`. A user has no fork-free route to the bits on these boards:
#    the firmware driver's sysfs get_throttled node does not exist on the Pi 4's 6.18 kernel, and the mailbox (/dev/vcio)
#    is root-only (checked 2026-10-07). Everything else is a bash builtin: the time ($EPOCHREALTIME), the temperature
#    (thermal_zone0, the sensor measure_temp reads: the two agreed to 0.1 C), the governor's clock (cpu0 scaling_cur_freq),
#    and the sleep (read -t on a pipe nobody writes, as memprobe.sh v2), timed to the next whole second, so no second is
#    skipped (v1 drifted: 1,686 samples in 1,726 s).
#  - the measured ARM clock (vcgencmd measure_clock arm, a second process) only in a second whose bits 0-3 are set: how deep
#    a cap went, at no cost while there is none.
# Line:  <epoch> throttled=0x80000 temp=53.5'C gov_khz=1800000 avail_mib=6214 [frequency(48)=1775000000]   (NA: that read failed)
# avail_mib (2026-10-20, the advisor's "resource (memory usages)"): MemAvailable in MiB, read with bash builtins (read on
# /proc/meminfo, which lists it third), so the poll still starts one process a second; the header says +avail_mib.
# Bits 0-3 are the state NOW (under-voltage, ARM capped, throttled, soft temperature limit); 16-19 the same four, sticky
# until reboot. testbed/scripts/throttle_summary.py reads the log and gives the cell its verdict.
f=$1
Z=/sys/class/thermal/thermal_zone0/temp; C=/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq; M=/proc/meminfo
us=${EPOCHREALTIME//[!0-9]/}
printf '# throttle_poll v2 pid=%s start=%s +avail_mib\n' $$ $((us / 1000000)) >> "$f"
exec {tick}<> <(:)                    # a pipe nobody writes: 'read -t' on it is a sleep without a fork
while :; do
  us=${EPOCHREALTIME//[!0-9]/}
  th=$(vcgencmd get_throttled); th=${th#throttled=}; [[ $th == 0x* ]] || th=NA   # no redirection inside $(): with one,
  # bash 5.2 forks twice (counted: 400 starts per 200 against 201); run_pool.sh sends this script's stderr to /dev/null
  t=NA; read -r z 2>/dev/null < $Z && t="$((z / 1000)).$((z % 1000 / 100))'C"
  k=NA; read -r k 2>/dev/null < $C || k=NA
  m=NA; while read -r mk mv _; do [ "$mk" = MemAvailable: ] && { m=$((mv / 1024)); break; }; done 2>/dev/null < $M
  fr=""
  if [ $th != NA ] && [ "${th: -1}" != 0 ]; then
    fr=$(vcgencmd measure_clock arm); [[ $fr == frequency* ]] || fr="frequency(48)=NA"
  fi
  printf '%s throttled=%s temp=%s gov_khz=%s avail_mib=%s%s\n' $((us / 1000000)) $th $t $k $m "${fr:+ $fr}" >> "$f"
  w=$((1000000 - ${EPOCHREALTIME//[!0-9]/} % 1000000)); printf -v w '%d.%06d' $((w / 1000000)) $((w % 1000000))
  read -r -t $w -u $tick _
done
