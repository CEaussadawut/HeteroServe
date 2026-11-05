#!/bin/bash
# throttle_poll.sh LOGFILE -- once a second until killed, on a Raspberry Pi: get_throttled, the SoC temperature and the ARM
# clock actually running. get_throttled's bits 16-19 are sticky until reboot, so a snapshot after a cell cannot say whether
# THAT cell was capped; bit 3 (soft temperature limit ACTIVE now) and the measured clock, polled through the cell, can.
f=$1
while :; do
  echo "$(date +%s) $(vcgencmd get_throttled) $(vcgencmd measure_temp) $(vcgencmd measure_clock arm)" >> "$f"
  sleep 1
done
