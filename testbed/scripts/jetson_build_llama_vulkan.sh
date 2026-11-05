#!/bin/bash
# Build llama.cpp at the CLUSTER PIN on the Jetson Nano with the Vulkan backend + RPC.
# This is "Path 1" of Jetson/docs/50_pooled_gpu_options.md -- chosen 2026-09-25 after the
# driver reported shaderFloat16 = true (testbed/scripts/vk_fp16_probe.py). Same revision as
# both Pis, so the node can join pooled RPC runs without the mixed-revision hazard.
#
# Toolchain (user space unless noted; all installed 2026-09-25, see testbed/node-jetson.md):
#   gcc/g++ 11.4       ppa:ubuntu-toolchain-r/test (apt, root)   -- 18.04's gcc 7 is too old
#   cmake 3.31.12      Kitware aarch64 binary, SHA-256 checked    -- ggml-vulkan needs >= 3.19
#   Vulkan-Headers     v1.4.363 (2026-09-18), newest tag before the llama.cpp commit
#   glslc              shaderc v2026.4 (2026-09-11) built on the board, deps at its pinned revs
#   SPIRV-Headers      04fd3caa1e82 (shaderc v2026.4's pin), installed as a CMake package --
#                      ggml-vulkan does find_package(SPIRV-Headers CONFIG REQUIRED)
#   Vulkan loader      /usr/lib/aarch64-linux-gnu/libvulkan.so.1 -> NVIDIA L4T 1.2.141
#
# Run on the board (detached, it takes about an hour):
#   ssh jetson 'setsid nohup bash ~/heteroserve/tools/jetson_build_llama_vulkan.sh </dev/null >/dev/null 2>&1 &'
set -euo pipefail

PIN=b23701f77d47dad9de834d59ebfcbe25c9e8b46f
SRC=$HOME/llama.cpp
BUILD=$SRC/build
LOGD=$HOME/heteroserve/logs
CMAKE=$HOME/opt/cmake/bin/cmake
GLSLC=$HOME/opt/shaderc/bin/glslc
VKINC=$HOME/opt/src/Vulkan-Headers/include
VKLIB=/usr/lib/aarch64-linux-gnu/libvulkan.so.1
SPIRVH=$HOME/opt/spirv-headers
JOBS=${JOBS:-3}          # not 4: the ggml-vulkan translation units are memory-hungry on 4 GB
# vulkan-shaders-gen is built as a separate ExternalProject that is handed NO compiler on a
# native build, so its CMake falls back to $CC/$CXX, then to plain c++ -- gcc 7, whose
# libstdc++ has no <filesystem>. Export gcc-11 so that sub-build inherits it.
export CC=gcc-11 CXX=g++-11
mkdir -p "$LOGD"
rm -f "$LOGD/llama_build.done"
trap 'echo $? > "$LOGD/llama_build.done"' EXIT

cd "$SRC"
[ "$(git rev-parse HEAD)" = "$PIN" ] || { echo "source is not at the cluster pin $PIN" >&2; exit 2; }
[ -z "$(git status --porcelain)" ]   || { echo "source tree is dirty" >&2; exit 2; }
[ -x "$GLSLC" ]                      || { echo "glslc missing at $GLSLC" >&2; exit 2; }

# Same core options as the Pis' builds (Release, shared libs by default, RPC, native CPU
# flags, OpenMP by default) and as Jetson/scripts/20_build_llama_cpp.sh (LLAMA_CURL=OFF),
# with Vulkan in place of CUDA.
CMAKE_ARGS=(
  -S "$SRC" -B "$BUILD"          # default "Unix Makefiles": 18.04's ninja 1.8.2 cannot build
                                # ggml-vulkan's shader rules (several outputs sharing one
                                # depfile needs ninja >= 1.10)
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11
  -DLLAMA_CURL=OFF
  -DGGML_RPC=ON -DGGML_NATIVE=ON
  -DGGML_VULKAN=ON
  -DVulkan_INCLUDE_DIR="$VKINC" -DVulkan_LIBRARY="$VKLIB" -DVulkan_GLSLC_EXECUTABLE="$GLSLC"
  -DCMAKE_PREFIX_PATH="$SPIRVH"
  # ggml-vulkan finds SPIRV-Headers but never adds its include dir: upstream relies on the
  # LunarG SDK keeping spirv/ beside vulkan/. Without it, ggml-vulkan-types.h's
  # __has_include probe fails -- and gcc bug 80753 (a failed __has_include silences the
  # later #include of the same header) turns "file not found" into "spv not declared".
  "-DCMAKE_CXX_FLAGS=-isystem $SPIRVH/include"
)
if [ "${RESUME:-0}" != 1 ]; then
  rm -rf "$BUILD"          # configure from clean; RESUME=1 continues an interrupted build
  "$CMAKE" "${CMAKE_ARGS[@]}" > "$LOGD/llama_configure.log" 2>&1
  : > "$LOGD/llama_build.log"
fi
# Two phases. The generated shader translation units are huge -- mul_mm.comp.cpp alone reached
# 2.5 GB resident under gcc 11 -O3 and was OOM-killed at -j3 on this 4 GB board (2026-09-25).
# So the Vulkan backend is compiled one file at a time, then everything else in parallel.
nice -n 5 "$CMAKE" --build "$BUILD" --target ggml-vulkan -j 1 >> "$LOGD/llama_build.log" 2>&1
nice -n 5 "$CMAKE" --build "$BUILD" -j "$JOBS" >> "$LOGD/llama_build.log" 2>&1

# Build provenance, in the format Jetson/scripts/20_build_llama_cpp.sh writes and
# Jetson/scripts/60_collect_node_facts.sh reads (${HS_ROOT}/build_info.json).
OPTS=$(printf '%s ' "${CMAKE_ARGS[@]}" | sed 's/\\/\\\\/g; s/"/\\"/g; s/ $//')
cat > "$HOME/heteroserve/build_info.json" <<JSON
{
  "node_id": "jetson-nano-01",
  "built_at_utc": "$(date -u +%Y%m%dT%H%M%SZ)",
  "llama_cpp_ref": "$(git describe --tags --always 2>/dev/null)",
  "llama_cpp_commit": "$(git rev-parse HEAD)",
  "backend": "vulkan",
  "cmake_options": "${OPTS}",
  "cuda_version": "n/a -- Vulkan build (Path 1)",
  "cuda_arch": "n/a",
  "host_cxx": "$(g++-11 --version | head -1)",
  "glslc": "$("$GLSLC" --version 2>&1 | head -1)",
  "vulkan_headers": "v1.4.363",
  "spirv_headers": "04fd3caa1e82 (shaderc v2026.4 pin)",
  "vulkan_loader": "$(readlink -f "$VKLIB")",
  "l4t": "$(sed -n 's/^# R\([0-9]*\) (release), REVISION: \([0-9.]*\).*/\1.\2/p' /etc/nv_tegra_release)",
  "kernel": "$(uname -r)",
  "cmake": "$("$CMAKE" --version | head -1)"
}
JSON
