#!/bin/bash
# memprobe.sh LOGFILE PROCNAME -- once a second until killed: MemAvailable, and each PROCNAME process's resident set
# split by kind (VmRSS, RssAnon, RssFile, RssShmem, kB). run_pool.sh starts it on every board of a pooled cell, so
# the resident set of llama-bench (host) and ggml-rpc-server (workers) is measured, not inferred from an OOM record.
# v2 (2026-09-28): bash builtins only, so a sample starts no process. v1 (memprobe_v1.sh) started about five a
# second, and on a 4-core board running llama.cpp on all four cores that cost ~5 % of decode (protocol_ab_pi5.sh).
# Same arguments and the same line format as v1:
#     <epoch> avail_kb=<kB> pid=<pid> VmRSS:<kB> RssAnon:<kB> RssFile:<kB> RssShmem:<kB> [pid=...]
# PROCNAME is matched exactly against /proc/<pid>/comm, as pgrep -x does.
f=$1; n=$2
exec {tick}<> <(:)                    # a pipe nobody writes: 'read -t 1' on it is a one-second sleep without a fork
while :; do
  printf -v now '%(%s)T' -1
  avail=""
  while read -r k v _; do [ "$k" = MemAvailable: ] && { avail=$v; break; }; done < /proc/meminfo
  l="$now avail_kb=$avail"
  for d in /proc/[0-9]*; do
    read -r c 2>/dev/null < "$d/comm" || continue
    [ "$c" = "$n" ] || continue
    s=""
    while read -r k v _; do
      case $k in VmRSS:|RssAnon:|RssFile:|RssShmem:) s+="$k$v " ;; esac
    done 2>/dev/null < "$d/status"
    l+=" pid=${d#/proc/} $s"
  done
  printf '%s\n' "$l" >> "$f"
  read -r -t 1 -u $tick _
done
