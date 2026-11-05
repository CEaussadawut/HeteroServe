#!/bin/bash
# TEST A: Qwen2.5-14B Q4_K_M (8.37 GiB) on the Pi 5 ALONE. Expect: cannot allocate.
M=~/models/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf
OUT=~/results/testA_14b_alone_$(date +%Y%m%d_%H%M%S).log
mkdir -p ~/results
{
echo "=== $(date -Is) TEST A: 14B Q4_K_M on Pi 5 alone"
echo "model bytes  : $(du -cb ~/models/qwen2.5-14b*.gguf | tail -1 | cut -f1)"
echo "MemAvailable : $(free -m | awk "/Mem:/{print \$7}") MiB"
echo "swap         : $(swapon --show | wc -l) entries"
echo "commit       : $(cd ~/llama.cpp && git rev-parse HEAD)"
echo "oom before   : $(sudo dmesg | grep -ciE "out of memory|oom-kill|oom_reaper")"
echo "--- run ---"
/usr/bin/timeout 1200 ~/llama.cpp/build/bin/llama-bench -m $M -ngl 0 -t 4 -p 64 -n 16 -r 1
rc=$?
echo "EXIT=$rc"
echo "oom after    : $(sudo dmesg | grep -ciE "out of memory|oom-kill|oom_reaper")"
sudo dmesg | grep -iE "out of memory|oom-kill|oom_reaper|Killed process" | tail -5
echo "=== $(date -Is) end"
} 2>&1 | tee $OUT
echo "OUTFILE=$OUT"
