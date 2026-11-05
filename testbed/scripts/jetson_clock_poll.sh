#!/bin/bash
# jetson_clock_poll.sh LOGFILE -- the Jetson's clocks about 10 times a second until killed (scheduler upgrade 5,
# 2026-10-10). Read-only. Runs as root, because the EMC rate is in debugfs:
#   sudo -n setsid nohup bash ~/heteroserve/tools/jetson_clock_poll.sh ~/heteroserve/logs/clockpoll_<stamp>.log </dev/null >/dev/null 2>&1 &
# Stop it by its exact argv (grep -a -l "[j]etson_clock_poll" /proc/[0-9]*/cmdline, then sudo -n kill), never pkill -f.
#
# Bash builtins only inside the loop: no fork per sample, so the poll adds no load the CPU governor could react to.
# L4T R32's bash is 4.4 (no EPOCHREALTIME), so each line carries /proc/uptime (10 ms resolution). The first line anchors
# it to the wall clock: "anchor <epoch seconds.ns> <uptime>".
# Sample line: <uptime> <gpu Hz> <gpu load, per mille> <cpu0 kHz> <emc Hz> <online cpus> <gpu runtime status>
G=/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq
L=/sys/devices/gpu.0/load
C=/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
E=/sys/kernel/debug/clk/emc/clk_rate
O=/sys/devices/system/cpu/online
R=/sys/devices/gpu.0/power/runtime_status
LOG=${1:?usage: jetson_clock_poll.sh LOGFILE}
mkdir -p "$(dirname "$LOG")"
read -r up _ < /proc/uptime
echo "anchor $(date +%s.%N) $up" >> "$LOG"
exec 3<> <(:)                     # a pipe nobody writes: read -t on it is a builtin sleep
while :; do
  read -r up _ < /proc/uptime
  read -r g < $G; read -r l < $L; read -r c < $C; read -r e < $E; read -r o < $O; read -r r < $R
  echo "$up $g $l $c $e $o $r"
  read -r -t 0.1 -u 3 _
done >> "$LOG"
