#!/bin/bash
# memprobe_tegra.sh LOGFILE PROCNAME -- memprobe.sh v2 for the Jetson, with the Tegra memory pools beside MemAvailable:
#     <epoch> avail_kb=<kB> nvmap_used_kb=<kB> nvmap_free_kb=<kB> cma_free_kb=<kB> pid=<pid> VmRSS:<kB> RssAnon:<kB> ...
# Why (2026-09-29): on Tegra a Vulkan or CUDA buffer is nvmap memory. It lowers MemAvailable when it is allocated, but it
# shows in the worker's resident set only once the process touches it, and nvmap keeps freed pages in its own pool
# (NvMapMemFree), which MemAvailable does not count. The fields added here say where the memory went.
# The line keeps memprobe.sh's format: field 2 is avail_kb, and the added fields are key=value (not key:value), so
# every reader of memprobe.sh logs (run_pool.sh, jetson_qwen35_smoke.sh) reads this one unchanged.
# memprobe.sh itself is left as it is: it is the instrument of the registered cells.
f=$1; n=$2
exec {tick}<> <(:)                    # a pipe nobody writes: 'read -t 1' on it is a one-second sleep without a fork
while :; do
  printf -v now '%(%s)T' -1
  avail="" nu="" nf="" cf=""
  while read -r k v _; do
    case $k in
      MemAvailable:) avail=$v ;;
      NvMapMemUsed:) nu=$v ;;
      NvMapMemFree:) nf=$v ;;
      CmaFree:) cf=$v ;;
    esac
  done < /proc/meminfo
  l="$now avail_kb=$avail nvmap_used_kb=$nu nvmap_free_kb=$nf cma_free_kb=$cf"
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
