#!/bin/bash
# Glue run on the Jetson, detached, 2026-09-25: wait for the shaderc build, install glslc,
# build llama.cpp (jetson_build_llama_vulkan.sh), let the board cool, then benchmark
# (jetson_bench_tinyllama.sh). Progress in ~/heteroserve/logs/chain.status.
set -uo pipefail
L=$HOME/heteroserve/logs
st() { echo "$(date +%T) $*" | tee -a "$L/chain.log" > "$L/chain.status"; }

st "waiting for shaderc"
while [ ! -f "$L/shaderc_build.done" ]; do sleep 20; done
[ "$(cat "$L/shaderc_build.done")" = 0 ] || { st "FAILED: shaderc build exit $(cat "$L/shaderc_build.done")"; exit 1; }

mkdir -p "$HOME/opt/shaderc/bin"
cp "$HOME/opt/src/shaderc/build/glslc/glslc" "$HOME/opt/shaderc/bin/glslc"
"$HOME/opt/shaderc/bin/glslc" --version > "$L/glslc_version.txt" 2>&1 || { st "FAILED: glslc does not run"; exit 1; }

st "building llama.cpp"
bash "$HOME/heteroserve/tools/jetson_build_llama_vulkan.sh"
[ "$(cat "$L/llama_build.done" 2>/dev/null)" = 0 ] || { st "FAILED: llama.cpp build exit $(cat "$L/llama_build.done" 2>/dev/null)"; exit 1; }

# Cool down before measuring: an hour of compiling leaves the SoC warm, and the benchmark's
# own 30 s pause would not give a cold start. Wait for CPU < 40 C, at most 10 minutes.
st "cooling down"
for _ in $(seq 1 60); do
  t=$(( $(cat /sys/devices/virtual/thermal/thermal_zone1/temp) / 1000 ))
  [ "$t" -lt 40 ] && break
  sleep 10
done
st "benchmarking (CPU at ${t} C when started)"
bash "$HOME/heteroserve/tools/jetson_bench_tinyllama.sh"
st "DONE: benchmark exit $(cat "$HOME/heteroserve/results/bench.done" 2>/dev/null), results in $(cat "$HOME/heteroserve/results/LATEST" 2>/dev/null)"
