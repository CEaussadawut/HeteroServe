#!/bin/bash
# The Jetson's prima.cpp build, resumed (2026-10-21 ~20:10). prima_build_20261021.sh unpacked and Qwen3-patched ~/prima.cpp there
# (src/llama.cpp ab8b99ce..., as on every node), but make stopped after 42 s: gcc-8's arm_neon.h has no vld1q_u8_x4/vld1q_s8_x4
# (gcc 9 added them; prima.cpp's README asks for gcc >= 9). testbed/patches/prima-288df6a-gcc8-neon-v1.patch defines the two
# loads as four plain 16-byte loads of the same bytes under gcc < 9 only, as prima.cpp's own 32-bit branch does. This applies it
# on the Jetson (only there: the other nodes build with gcc >= 9 and the laptop is x86) and runs the same make, with the power
# guard by its NOTE's procedure. From the repo root:  bash testbed/scripts/prima_build_jetson_gcc8_20261021.sh
# Raw output tee'd to testbed/results/prima_build_jetson_gcc8_<stamp>.log; the board's make log copied back beside it.
set -uo pipefail
cd "$(dirname "$0")/../.."
P=testbed/patches/prima-288df6a-gcc8-neon-v1.patch
STAMP=$(date +%Y%m%d_%H%M%S); LOG=testbed/results/prima_build_jetson_gcc8_$STAMP.log
exec > >(tee -a "$LOG") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
echo "########## $(date -Is) the Jetson's prima.cpp build with $P ($(sha256sum $P | cut -c1-16))"
got=$($SSH jetson 'cd ~/prima.cpp && sha256sum src/llama.cpp | cut -d" " -f1')
[ "$got" = ab8b99ce4ff8fd243b949ed4c4473635b53675485687c1f476b361db55160d0f ] || { echo "FATAL: the Jetson's src/llama.cpp is $got"; exit 1; }
scp -q $P jetson:src/ || { echo "FATAL: copy failed"; exit 1; }
$SSH jetson "cd ~/prima.cpp && git apply --check ~/src/$(basename $P) && git apply ~/src/$(basename $P) && echo applied && sha256sum ggml/src/ggml-cpu-impl.h" || { echo "FATAL: the patch did not apply"; exit 1; }
g=$($SSH jetson 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum ~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1' 2>/dev/null | tr '\n' ' ')
[ "$g" = "synced $(sha256sum testbed/scripts/jetson_power_guard.sh | cut -d' ' -f1) " ] || { echo "FATAL: Jetson guard precheck '$g'"; exit 1; }
$SSH jetson 'sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo guard started' 2>/dev/null
echo "build   jetson: make -j2 llama-cli llama-server, CPU, g++-8 ($(date -Is)); MemAvailable $($SSH jetson 'grep MemAvailable /proc/meminfo')"
$SSH jetson "cd ~/prima.cpp && t0=\$(date +%s); make -j2 llama-cli llama-server CC=gcc-8 CXX=g++-8 CPPFLAGS=-I\$HOME/prima-deps/include LDFLAGS=-L\$HOME/prima-deps/lib > ~/logs/prima_build_jetson_gcc8_$STAMP.log 2>&1; echo jetson: EXIT_MAKE=\$? elapsed_s=\$(( \$(date +%s) - t0 ))"
$SSH jetson "grep -c -i ' error' ~/logs/prima_build_jetson_gcc8_$STAMP.log; tail -4 ~/logs/prima_build_jetson_gcc8_$STAMP.log; cd ~/prima.cpp && sha256sum llama-cli llama-server 2>/dev/null; ldd llama-cli 2>/dev/null | grep -i zmq"
scp -q jetson:logs/prima_build_jetson_gcc8_$STAMP.log testbed/results/prima_build_jetson_gcc8_${STAMP}_make.log
$SSH jetson 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p stopped"; done; echo "TRIP lines: $(grep -c TRIP ~/heteroserve/logs/power-guard.log)"' 2>/dev/null
echo "BUILD DONE $(date -Is)"
