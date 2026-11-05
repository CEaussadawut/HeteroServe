#!/bin/bash
# Build llama.cpp at the CLUSTER PIN b23701f7 on the Jetson Nano with CUDA 10.2 + RPC.
# This is "Path 2" of Jetson/docs/50_pooled_gpu_options.md, queued at the user's request
# (2026-09-25) as a CUDA-vs-Vulkan comparison on the same GPU and the same revision.
#
# CUDA 10.2 (JetPack 4.6.6) cannot build a 2026 llama.cpp as-is. The port is
# testbed/patches/jetpack46-cuda102-b23701f.patch: the Z841973620/llama.cpp-jetpack4.6
# patch set (made for b10621), three-way merged onto the pin, 13 conflicts resolved by hand,
# plus two changes of our own -- a C++14 dispatcher for FlashAttention's per-quant gating,
# and GGML_CUDA_NO_MMA_FA, which leaves out MMA FlashAttention (it needs tensor cores and is
# never selected on sm_53). See testbed/node-jetson.md for the classification of every change.
#
# Run on the board, detached:
#   ssh jetson 'setsid nohup bash ~/heteroserve/tools/jetson_build_llama_cuda.sh </dev/null >/dev/null 2>&1 &'
set -euo pipefail

PIN=b23701f77d47dad9de834d59ebfcbe25c9e8b46f
PATCH=$HOME/heteroserve/patches/jetpack46-cuda102-b23701f.patch
SRC=$HOME/llama.cpp-cuda
BUILD=$SRC/build
LOGD=$HOME/heteroserve/logs
CMAKE=$HOME/opt/cmake/bin/cmake
NVCC=/usr/local/cuda-10.2/bin/nvcc
JOBS=${JOBS:-2}          # nvcc's cicc/ptxas are memory-hungry; 4 GB board
mkdir -p "$LOGD"
rm -f "$LOGD/cuda_build.done"
trap 'echo $? > "$LOGD/cuda_build.done"' EXIT

# CUDA 10.2 refuses host compilers newer than gcc 8; 18.04 ships g++-8 (8.4). The C/C++
# side keeps gcc 11 -- only the .cu host passes go through g++-8.
command -v g++-8 >/dev/null || sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq g++-8 > "$LOGD/cuda_gxx8_install.log" 2>&1
export CC=gcc-11 CXX=g++-11

# zram swap: the benchmark turns it off for measurement, but an nvcc build on a 4 GB board
# wants it. Re-enable the two JetPack zram devices for the duration of the build; the next
# benchmark run turns them off again.
sudo -n swapon /dev/zram0 /dev/zram1 2>/dev/null || true

# A separate tree, so the Vulkan build (~/llama.cpp) stays byte-identical to the Pis' source.
rm -rf "$SRC"
cp -a "$HOME/llama.cpp" "$SRC"
rm -rf "$BUILD"
cd "$SRC"
[ "$(git rev-parse HEAD)" = "$PIN" ] || { echo "source is not at the cluster pin" >&2; exit 2; }
[ -z "$(git status --porcelain)" ]   || { echo "source tree is dirty before patching" >&2; exit 2; }
git apply --check "$PATCH"
git apply "$PATCH"
git diff --stat > "$LOGD/cuda_patch_applied.stat"

CMAKE_ARGS=(
  -S "$SRC" -B "$BUILD"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11
  -DLLAMA_CURL=OFF
  -DGGML_RPC=ON -DGGML_NATIVE=ON
  -DGGML_CUDA=ON
  -DCMAKE_CUDA_COMPILER="$NVCC"
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-8
  -DCMAKE_CUDA_ARCHITECTURES=53        # Tegra X1 / Maxwell GM20B
  -DGGML_CUDA_NO_MMA_FA=ON              # MMA FlashAttention needs tensor cores
)
"$CMAKE" "${CMAKE_ARGS[@]}" > "$LOGD/cuda_configure.log" 2>&1
nice -n 5 "$CMAKE" --build "$BUILD" -j "$JOBS" > "$LOGD/cuda_build.log" 2>&1

OPTS=$(printf '%s ' "${CMAKE_ARGS[@]}" | sed 's/\\/\\\\/g; s/"/\\"/g; s/ $//')
cat > "$HOME/heteroserve/build_info_cuda.json" <<JSON
{
  "node_id": "jetson-nano-01",
  "built_at_utc": "$(date -u +%Y%m%dT%H%M%SZ)",
  "llama_cpp_commit": "$PIN",
  "llama_cpp_patch": "jetpack46-cuda102-b23701f.patch sha256 $(sha256sum "$PATCH" | cut -d' ' -f1)",
  "backend": "cuda",
  "cmake_options": "${OPTS}",
  "cuda_version": "$("$NVCC" --version | grep -o 'release [0-9.]*')",
  "cuda_arch": "53",
  "cuda_host_cxx": "$(g++-8 --version | head -1)",
  "host_cxx": "$(g++-11 --version | head -1)",
  "l4t": "$(sed -n 's/^# R\([0-9]*\) (release), REVISION: \([0-9.]*\).*/\1.\2/p' /etc/nv_tegra_release)",
  "cmake": "$("$CMAKE" --version | head -1)"
}
JSON
