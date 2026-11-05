#!/usr/bin/env bash
# 02 -- build the pinned llama.cpp revision with RPC support for Cortex-A76.
#
# Reproducibility contract: every number in the characterisation dataset must come
# from ONE llama.cpp revision, identical on the Pi 5, the Pi 4 and the Jetson. On the
# first build LLAMA_PIN_SHA is empty, the script resolves whatever master is, and
# writes the SHA into build_manifest.json. Copy that SHA into config/node.env (and
# into the other nodes' node.env) and never change it again.
. "$(dirname "$0")/lib/common.sh"
load_env
assert_this_board

[ -w "$(dirname "$LLAMA_SRC")" ] || need_root "$@"

if [ ! -d "$LLAMA_SRC/.git" ]; then
  log "cloning $LLAMA_REPO -> $LLAMA_SRC"
  git clone "$LLAMA_REPO" "$LLAMA_SRC"
fi
cd "$LLAMA_SRC"

if [ -n "${LLAMA_PIN_SHA:-}" ]; then
  log "checking out pinned revision ${LLAMA_PIN_SHA}"
  git fetch --quiet origin
  git checkout --quiet --detach "$LLAMA_PIN_SHA" \
    || die "pinned SHA ${LLAMA_PIN_SHA} not found in the upstream history"
else
  warn "LLAMA_PIN_SHA is empty -- building current master and recording the SHA."
  warn "Put that SHA into config/node.env before the first measurement run."
  git fetch --quiet origin
  git checkout --quiet --detach origin/HEAD 2>/dev/null || git checkout --quiet master
fi
RESOLVED_SHA="$(git rev-parse HEAD)"
ok "llama.cpp at ${RESOLVED_SHA}"

# ONE list, used for both the configure step and the manifest below. These were two
# hand-maintained copies and had already drifted apart: the manifest omitted
# CMAKE_BUILD_TYPE=Release -- the single most consequential option in the whole build --
# and LLAMA_BUILD_TESTS=OFF. A build record that does not state the optimisation level is
# not a reproducibility record, and this node's entire purpose is reproducible numbers.
# (The arch flags are recorded separately, in "arch_flags".)
#
# LLAMA_CURL is deliberately absent. Upstream CMakeLists.txt:195 is
# `llama_option_depr(WARNING LLAMA_CURL)` with no replacement argument, so on a current
# revision it prints "LLAMA_CURL is deprecated and will be ignored" and sets nothing; on
# an older pin it links libcurl for llama.cpp's model-URL loader, which this node never
# uses -- models arrive through 03_fetch_models.sh. Either way it was a no-op here.
CMAKE_OPTS=(
  -DCMAKE_BUILD_TYPE=Release
  -DGGML_RPC=ON
  -DGGML_NATIVE=OFF
  -DGGML_BLAS=OFF
  -DGGML_OPENMP=ON
  -DBUILD_SHARED_LIBS=OFF
  -DLLAMA_BUILD_TESTS=OFF
)
CMAKE_OPTS_STR="$(printf '%s ' "${CMAKE_OPTS[@]#-D}")"; CMAKE_OPTS_STR="${CMAKE_OPTS_STR% }"

log "configuring (RPC on, BLAS off, OpenMP on, no GPU backend on this node)"
rm -rf build
cmake -S . -B build -G Ninja \
  "${CMAKE_OPTS[@]}" \
  -DCMAKE_C_FLAGS="${LLAMA_ARCH_FLAGS}" \
  -DCMAKE_CXX_FLAGS="${LLAMA_ARCH_FLAGS}"
# GGML_NATIVE=OFF is deliberate: -march=native would silently produce a different
# binary on a different board revision and break the Pi4/Pi5 comparison. The ISA
# level is stated explicitly in LLAMA_ARCH_FLAGS instead.
#
# GGML_BLAS=OFF is deliberate. Two reasons, both about keeping the measurement
# interpretable:
#   1. The headline result is a Pi4-vs-Pi5 comparison. If one node links OpenBLAS and
#      the other does not, the prefill gap conflates an ISA difference (dotprod) with
#      a BLAS difference, and the comparison answers no question at all.
#   2. OpenBLAS runs its own thread pool. With LLAMA_THREADS=4 and CPUAffinity=0-3 the
#      two pools contend for the same four A76 cores, which is exactly the variable
#      the 1/2/4-thread scaling cells are supposed to isolate.
# llama.cpp's own aarch64 kernels are what the dataset is meant to characterise.
#
# STATE OF THE CLUSTER, 2026-09-14 -- BLAS is OFF on every node, decided by the user, and
# stated explicitly everywhere rather than inherited from an upstream default (an inherited
# default leaves no trace in a build manifest, so two identical builds can look different):
#   Pi 4   -- -DGGML_BLAS=OFF in its CMAKE_OPTS array.
#   Pi 5   -- -DGGML_BLAS=OFF, here.
#   T580   -- -DGGML_BLAS=OFF too. The control node is not exempt: docs/05_experiment_
#             protocol.md puts it into the measurement as a prefill node when serves_rpc is
#             set, so its build has to be comparable with the workers'.
#   Jetson -- Jetson/scripts/20_build_llama_cpp.sh, explicit OFF in both option eras. That
#             branch used to link OpenBLAS for the same-device CPU-vs-GPU comparison, which
#             was sound on its own axis but broke the cross-node one: the shared results
#             schema has no column distinguishing a BLAS-linked CPU build from a plain one
#             (its only backend-ish column is n_gpu_layers, 0 for a CPU build on every
#             node), so a Jetson cpu-neon row and a Pi 5 row were pooled into one per-node
#             c_i -- the number Rule A divides layers by.
# libopenblas-dev is installed on none of the four, so a stray -DGGML_BLAS=ON fails to
# configure instead of silently producing an incomparable node. Belt and braces on every
# node: the flag says what was built, the absent headers make the opposite impossible.
# Caveat: editing a bootstrap uninstalls nothing on a board that was already provisioned --
# the structural half only holds after a re-flash or a manual `apt-get purge`.
#
# GGML_OPENMP=ON and BUILD_SHARED_LIBS=OFF are here for the same reason: Pi4/scripts/
# 02_build_llama_cpp.sh sets both, and the Pi4-vs-Pi5 comparison is only about the ISA
# if EVERY other build option matches. Changing one of these on one node silently turns
# the headline result into a comparison of build configurations.

log "building with -j${BUILD_JOBS}"
cmake --build build -j "${BUILD_JOBS}"

for b in llama-cli llama-bench; do
  [ -x "build/bin/$b" ] || die "expected binary build/bin/$b was not produced"
done
# Upstream renamed the RPC server target rpc-server -> ggml-rpc-server in c299a92
# (2026-06-27). Which name this build produces depends on the pinned SHA, and the SHA
# is chosen on the Jetson (CUDA 10.2 constrains the cluster), so accept either and
# record which one it was. 04_install_services.sh detects it again at install time.
if   [ -x build/bin/ggml-rpc-server ]; then RPC_BIN_NAME=ggml-rpc-server
elif [ -x build/bin/rpc-server ];      then RPC_BIN_NAME=rpc-server
else die "neither build/bin/ggml-rpc-server nor build/bin/rpc-server was produced -- was -DGGML_RPC=ON honoured?"
fi
ok "rpc binary: ${RPC_BIN_NAME}"
ok "binaries: $(ls build/bin | tr '\n' ' ')"

log "confirming the A76 feature set actually reached the object code"
DOTPROD_OK=no
grep -q ' asimddp' /proc/cpuinfo && DOTPROD_OK=yes
[ "$DOTPROD_OK" = yes ] || warn "the kernel does not report asimddp -- dotprod kernels will not be used"

MANIFEST="${DATA_DIR}/build_manifest.json"
install -d "$DATA_DIR"
cat > "$MANIFEST" <<JSON
{
  "node_id": "${NODE_ID}",
  "built_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "llama_cpp_sha": "${RESOLVED_SHA}",
  "llama_cpp_repo": "${LLAMA_REPO}",
  "pinned_in_node_env": "${LLAMA_PIN_SHA:-}",
  "arch_flags": "${LLAMA_ARCH_FLAGS}",
  "cmake_options": "${CMAKE_OPTS_STR}",
  "rpc_binary": "${RPC_BIN_NAME}",
  "compiler": "$(gcc --version | head -1)",
  "cmake": "$(cmake --version | head -1)",
  "kernel": "$(uname -srm)",
  "os": "$(. /etc/os-release; echo "$PRETTY_NAME")",
  "cpu_features": "$(awk -F': ' '/^Features/{print $2; exit}' /proc/cpuinfo)",
  "dotprod_reported_by_kernel": "${DOTPROD_OK}"
}
JSON
ok "wrote ${MANIFEST}"

if [ -z "${LLAMA_PIN_SHA:-}" ]; then
  printf '\n%sNEXT:%s set  LLAMA_PIN_SHA=%s  in %s and rebuild the other nodes against it.\n' \
    "$c_ylw" "$c_off" "$RESOLVED_SHA" "$HS_CONFIG"
fi
