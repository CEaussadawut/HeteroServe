#!/usr/bin/env bash
# Build llama.cpp on the Jetson Nano, CPU-only (NEON) with the RPC backend.
# The CUDA path was REMOVED 2026-09-29 (user: "remove the CUDA that can't build on jetson"): at the cluster pin an
# unpatched CUDA 10.2 build cannot compile. CUDA at the pin is testbed/scripts/jetson_build_llama_cuda.sh (the pin +
# testbed/patches/jetpack46-cuda102-b23701f.patch v2); Vulkan is testbed/scripts/jetson_build_llama_vulkan.sh.
#
#   ./20_build_llama_cpp.sh                  # use config/node.env
#   ./20_build_llama_cpp.sh --ref b4000      # try a specific llama.cpp ref
#   ./20_build_llama_cpp.sh --cpu-only       # accepted for old callers; every build is CPU-only now
#   ./20_build_llama_cpp.sh --clean
. "$(dirname "$0")/lib/common.sh"
load_env

CLEAN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --ref)      LLAMA_CPP_REF="$2"; shift 2 ;;
    --cpu-only) BUILD_CUDA=0; shift ;;
    --clean)    CLEAN=1; shift ;;
    -j)         BUILD_JOBS="$2"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

need_cmd git
need_cmd cmake
export PATH="${HOME}/.local/bin:${PATH}"

step "Source"
if [ ! -d "${LLAMA_SRC}/.git" ]; then
  info "cloning ${LLAMA_CPP_URL}"
  git clone "${LLAMA_CPP_URL}" "${LLAMA_SRC}"
fi
cd "${LLAMA_SRC}"
git fetch --all --tags --quiet || warn "fetch failed (offline?) — building whatever is checked out"
if [ -n "${LLAMA_CPP_REF}" ]; then
  git checkout --quiet "${LLAMA_CPP_REF}" || die "no such ref: ${LLAMA_CPP_REF}"
fi
BUILT_REF="$(git rev-parse --short HEAD)"
BUILT_DESC="$(git describe --tags --always 2>/dev/null || echo "$BUILT_REF")"
ok "llama.cpp at ${BUILT_DESC} (${BUILT_REF})"

step "Backend"
# A CUDA request is refused, not silently turned into a CPU build: a CPU binary where a CUDA one was asked for would
# produce plausible numbers under the wrong label.
if [ "${BUILD_CUDA:-0}" = "1" ]; then
  die "BUILD_CUDA=1: this script no longer builds CUDA. At the cluster pin CUDA 10.2 needs patch v2 -- use
     testbed/scripts/jetson_build_llama_cuda.sh (see config/node.env). Set BUILD_CUDA=0 for the CPU-only build here."
fi
CUDA_VER="$(cuda_version)"   # provenance only: what JetPack has installed; this build does not use it
ok "CPU-only (NEON) build"

step "Configure"
[ "${CLEAN}" = "1" ] && rm -rf "${LLAMA_BUILD}"

# The ggml build options were renamed LLAMA_* -> GGML_* on 2024-06-26 (PR #8006,
# "llama : reorganize source code + improve CMake"). We may be sitting on either
# side of that rename, so pick ONE spelling from the checked-out source rather
# than passing both.
#
# Passing both is not harmless. Post-rename CMakeLists.txt carries a deprecation
# shim, and while most entries are warnings:
#     llama_option_depr(WARNING     LLAMA_RPC    GGML_RPC)
#     llama_option_depr(WARNING     LLAMA_CUDA   GGML_CUDA)
# the cuBLAS one is fatal:
#     llama_option_depr(FATAL_ERROR LLAMA_CUBLAS GGML_CUDA)
# `--no-warn-unused-cli` silences unused-variable notices; it cannot suppress an
# explicit message(FATAL_ERROR ...). So -DLLAMA_CUBLAS=ON on any modern ref stops
# configure dead. It is never passed unless the source knows no other spelling.
if grep -rqs 'GGML_RPC' "${LLAMA_SRC}/CMakeLists.txt" "${LLAMA_SRC}/ggml/CMakeLists.txt"; then
  OPT_ERA="ggml"
else
  OPT_ERA="llama"
fi
info "llama.cpp CMake option era: ${OPT_ERA}"

CMAKE_ARGS=(
  -S "${LLAMA_SRC}" -B "${LLAMA_BUILD}"
  --no-warn-unused-cli
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=OFF
  -DLLAMA_CURL=OFF
)
# RPC backend: this is what makes the Nano a HeteroServe worker at all.
if [ "${OPT_ERA}" = "ggml" ]; then
  CMAKE_ARGS+=( -DGGML_RPC=ON -DGGML_NATIVE=ON )
else
  CMAKE_ARGS+=( -DLLAMA_RPC=ON -DLLAMA_NATIVE=ON )
fi
if command -v ccache >/dev/null 2>&1; then
  CMAKE_ARGS+=( -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache )
fi

{
  # CPU-only is a measurement condition in its own right (the same-device
  # CPU-vs-GPU comparison), not a fallback, so the flag has to actually take
  # effect rather than be silently dropped as an unknown option -- hence the
  # era split below rather than one spelling.
  #
  # BLAS is OFF, decided cluster-wide by the user on 2026-09-14. This branch used
  # to link OpenBLAS, on the reasoning that the Jetson's CPU condition is its own
  # axis. That reasoning is sound for the same-device CPU-vs-GPU comparison, but
  # it broke the CROSS-NODE table: the Pi 4 and Pi 5 both build without BLAS, and
  # the shared results schema has no column that distinguishes a BLAS-linked CPU
  # build from a plain one -- its only backend-ish column is n_gpu_layers, which
  # is 0 for a CPU build on every node. A Jetson cpu-neon row and a Pi 5 row were
  # therefore indistinguishable, and profiles.per_layer_times takes one median per
  # node across whatever it is handed, so the difference landed in c_i -- the
  # number Rule A divides layers by.
  #
  # Turning it off costs this node some CPU prefill throughput and keeps the
  # same-device comparison intact (both arms are now plain NEON); leaving it on
  # would have cost the cross-node comparison, which is the study's headline.
  # Explicit OFF rather than omitted, so the build record states it either way.
  if [ "${OPT_ERA}" = "ggml" ]; then
    CMAKE_ARGS+=( -DGGML_CUDA=OFF -DGGML_BLAS=OFF )
  else
    CMAKE_ARGS+=( -DLLAMA_BLAS=OFF )
  fi
  BACKEND="cpu-neon"
}

info "cmake ${CMAKE_ARGS[*]}"
cmake "${CMAKE_ARGS[@]}"

step "Compile (-j${BUILD_JOBS})"
# -j4 with nvcc on 4 GB reliably invokes the OOM killer. -j2 plus the swapfile
# from 00_system_prep.sh is the combination that finishes.
info "this takes roughly 40-70 min on a cold SD-card build"
cmake --build "${LLAMA_BUILD}" --config Release -j "${BUILD_JOBS}"

step "Verify"
for f in bin_bench bin_cli bin_rpc; do
  p="$($f 2>/dev/null || true)"
  if [ -n "$p" ]; then ok "$(basename "$p") -> $p"; else warn "${f#bin_} binary not found"; fi
done
RPC="$(bin_rpc || true)"
[ -n "$RPC" ] || die "rpc-server was not built — GGML_RPC did not take effect on this ref"

# Record exactly what was built; the characterisation dataset needs the build
# provenance next to every throughput number.
# The exact configure line, so the build record states which backend options were in
# force. Without it "backend": "cpu-neon" was true of a BLAS-linked build and of a plain
# one alike, and nothing downstream could tell the two apart.
CMAKE_OPTS_STR="$(printf '%s ' "${CMAKE_ARGS[@]}" | sed 's/\\/\\\\/g; s/"/\\"/g; s/ $//')"

BUILDINFO="${HS_ROOT}/build_info.json"
cat > "$BUILDINFO" <<JSON
{
  "node_id": "${NODE_ID}",
  "built_at_utc": "$(timestamp)",
  "llama_cpp_ref": "${BUILT_DESC}",
  "llama_cpp_commit": "${BUILT_REF}",
  "backend": "${BACKEND}",
  "cmake_options": "${CMAKE_OPTS_STR}",
  "cuda_version": "${CUDA_VER}",
  "cuda_arch": "${CUDA_ARCH}",
  "host_cxx": "$(${HOST_CXX} --version 2>/dev/null | head -1)",
  "l4t": "$(l4t_version)",
  "kernel": "$(uname -r)",
  "cmake": "$(cmake --version | head -1)"
}
JSON
ok "build provenance -> ${BUILDINFO}"
cat "$BUILDINFO"

echo
echo "Next: scripts/30_fetch_models.sh"
