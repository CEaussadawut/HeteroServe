#!/bin/bash
# Power guard for the Jetson Nano while it runs from MICRO-USB (rated 5 V / 2 A) in MAXN.
#
# Every second: read the board's own INA3221 on the 5 V input rail (bus voltage, current,
# power) and count over-current / undervoltage alarms in the kernel log. If the input sags
# below GUARD_MIN_MV for 3 consecutive samples, or any NEW alarm appears, drop the board to
# 5 W mode (nvpmodel -m 1) and stay there. A sag is the warning sign before a brownout, and
# a brownout mid-write can corrupt the SD card; 5 W halves the online cores and caps clocks,
# which cuts the draw well inside what micro-USB can deliver.
#
# Must run as root (the INA3221 nodes are root-only on L4T R32.7). Deployed 2026-09-25:
#   scp testbed/scripts/jetson_power_guard.sh jetson:heteroserve/tools/
#   ssh jetson 'sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 &'
# Stop:  ssh jetson 'sudo -n pkill -f jetson_power_guard.sh'
# Log:   ~jetson/heteroserve/logs/power-guard.log  (one line per second; TRIP lines on action)

GUARD_MIN_MV=${GUARD_MIN_MV:-4750}
D=/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0
LOG=${GUARD_LOG:-/home/jetson/heteroserve/logs/power-guard.log}
mkdir -p "$(dirname "$LOG")"

[ "$(cat "$D/rail_name_0")" = "POM_5V_IN" ] || { echo "channel 0 is not POM_5V_IN" >> "$LOG"; exit 1; }

alarms() { dmesg | grep -ciE 'oc alarm|over.?current|under.?volt'; }
base=$(alarms)
low=0; tripped=0; minv=99999; maxp=0
echo "$(date -Is) guard start: min ${GUARD_MIN_MV} mV, baseline alarms ${base}, $(cat /var/lib/nvpmodel/status)" >> "$LOG"

while true; do
  v=$(cat "$D/in_voltage0_input"); i=$(cat "$D/in_current0_input"); p=$(cat "$D/in_power0_input")
  a=$(alarms)
  [ "$v" -lt "$minv" ] && minv=$v
  [ "$p" -gt "$maxp" ] && maxp=$p
  echo "$(date +%T) V=${v} I=${i} P=${p} alarms=${a} minV=${minv} maxP=${maxp} $(cat /var/lib/nvpmodel/status | cut -d' ' -f1)" >> "$LOG"
  if [ "$v" -lt "$GUARD_MIN_MV" ]; then low=$((low + 1)); else low=0; fi
  if [ "$tripped" -eq 0 ] && { [ "$low" -ge 3 ] || [ "$a" -gt "$base" ]; }; then
    echo "$(date -Is) TRIP: V=${v} mV (low x${low}), alarms ${a} vs baseline ${base} -> nvpmodel -m 1" >> "$LOG"
    nvpmodel -m 1 < /dev/null >> "$LOG" 2>&1
    tripped=1
  fi
  sleep 1
done
