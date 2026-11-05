#!/bin/bash
# v1: the probe every pool cell ran with through 2026-09-28 -- about five process starts a second (date, awk, pgrep,
# awk, sleep), which cost ~5 % of decode on the Pi 5 (protocol_ab_pi5.sh, 2026-09-28). Kept to reproduce those cells.
# memprobe.sh LOGFILE PROCNAME -- once a second until killed: MemAvailable, and each PROCNAME process's resident set
# split by kind (VmRSS, RssAnon, RssFile, RssShmem, kB). run_pool.sh starts it on every board of a pooled cell, so
# the resident set of llama-bench (host) and ggml-rpc-server (workers) is measured, not inferred from an OOM record.
f=$1; n=$2
while :; do
  l="$(date +%s) avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)"
  for p in $(pgrep -x "$n"); do
    l="$l pid=$p $(awk '/^(VmRSS|RssAnon|RssFile|RssShmem):/{printf "%s%s ", $1, $2}' /proc/$p/status 2>/dev/null)"
  done
  echo "$l" >> "$f"
  sleep 1
done
