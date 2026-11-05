#!/bin/bash
# QR1 proof: a model that fits NO single node, run by pooling two nodes.
# Qwen2.5-14B-Instruct Q4_K_M (8.37 GiB) vs Pi 5 ~7.6 GiB / Pi 4 ~7.2 GiB available.
# The RPC worker on the Pi 4 is started by the CONTROL NODE, not from here.
set -u
M=~/models/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf
W=192.168.10.12:50052          # Pi 4 over the WIRED overlay
B=~/llama.cpp/build/bin
OUT=~/results/pool14b_$(date +%Y%m%d_%H%M%S)
mkdir -p ~/results
exec > >(tee -a $OUT.log) 2>&1

echo "########## $(date -Is)  QR1 pooling experiment"
echo "model bytes : $(du -cb ~/models/qwen2.5-14b*.gguf | tail -1 | cut -f1)"
echo "pi5 avail MiB: $(free -m | awk "/Mem:/{print \$7}")"
echo "commit      : $(cd ~/llama.cpp && git rev-parse HEAD)"
echo "temp/throttle: $(vcgencmd measure_temp) $(vcgencmd get_throttled)"

echo; echo "########## DEVICES visible with the RPC worker registered"
$B/llama-bench -rpc $W --list-devices 2>&1 | head -20

echo; echo "########## TEST A - Pi 5 ALONE (expect: cannot allocate)"
/usr/bin/timeout 900 $B/llama-bench -m $M -ngl 0 -t 4 -p 64 -n 16 -r 1
echo "TEST_A_EXIT=$?"
echo "--- kernel OOM records ---"
sudo dmesg | grep -ciE "out of memory|oom-kill" || true

echo; echo "########## TEST B - POOLED, Pi5 CPU + Pi4 over RPC (wired)"
for NGL in 24 30; do
  echo "----- ngl=$NGL (of 48 layers) -----"
  /usr/bin/timeout 2400 $B/llama-bench -m $M -rpc $W -ngl $NGL -t 4 -p 64 -n 16 -r 1
  echo "TEST_B_ngl${NGL}_EXIT=$?"
  echo "temp/throttle: $(vcgencmd measure_temp) $(vcgencmd get_throttled)"
done
echo "########## done $(date -Is)"
echo "OUTFILE=$OUT.log"
