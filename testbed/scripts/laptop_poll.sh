#!/bin/bash
# The laptop's 1 Hz state while it is a node in a run (2026-10-11): what throttle_poll.sh is for a Pi. Run on the laptop by a
# harness, killed by it at the end:
#     setsid nohup bash testbed/scripts/laptop_poll.sh <run>/laptop_poll.log &
# One line a second, every field key=value:
#     t=<epoch.ms> load=<1-min load> mhz=<mean of the cores' current MHz> pkg=<package C> core_thr=<core throttle count>
#     pkg_thr=<package throttle count> avail=<MemAvailable MiB> ac=<0|1> bat0=<status>/<capacity> bat1=<status>/<capacity>
# Why each field: the laptop is also the user's desktop, and the 2026-10-11 long-context run had one disturbed repetition
# (desktop activity) found only in the journal afterwards; outlet 1's power is the laptop's only while no battery charges
# (testbed/energy.md); and it has no vcgencmd, so throttling shows as the Intel thermal-throttle counters rising.
set -u
OUT=${1:?usage: laptop_poll.sh <log>}
T=/sys/devices/system/cpu
pkgz=""
for z in /sys/class/thermal/thermal_zone*; do [ "$(cat "$z/type" 2>/dev/null)" = x86_pkg_temp ] && pkgz=$z/temp; done
bat() { local b=/sys/class/power_supply/$1; [ -d "$b" ] && echo "$(tr ' ' '_' < $b/status)/$(cat $b/capacity)" || echo "-"; }
while :; do
  mhz=$(awk -F: '/^cpu MHz/ {s += $2; n++} END {if (n) printf "%.0f", s / n}' /proc/cpuinfo)
  printf 't=%s load=%s mhz=%s pkg=%s core_thr=%s pkg_thr=%s avail=%s ac=%s bat0=%s bat1=%s\n' \
    "$(date +%s.%3N)" "$(cut -d' ' -f1 /proc/loadavg)" "$mhz" "$( [ -n "$pkgz" ] && echo $(( $(cat $pkgz) / 1000 )) || echo -)" \
    "$(cat $T/cpu0/thermal_throttle/core_throttle_count 2>/dev/null || echo -)" \
    "$(cat $T/cpu0/thermal_throttle/package_throttle_count 2>/dev/null || echo -)" \
    "$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)" "$(cat /sys/class/power_supply/AC*/online 2>/dev/null | head -1)" \
    "$(bat BAT0)" "$(bat BAT1)" >> "$OUT"
  sleep 1
done
