#!/usr/bin/env bash
# Build the pinned llama.cpp revision for Cortex-A72 and record exactly what was built.
# Run as the normal user (NOT root) -- /opt/heteroserve is owned by that user.
source "$(dirname "$0")/lib.sh"
[ "$(id -u)" -ne 0 ] || die "do NOT run this as root"
ensure_dirs

if [ ! -d "$LLAMA_SRC/.git" ]; then
  log "cloning $LLAMA_REPO -> $LLAMA_SRC"
  git clone "$LLAMA_REPO" "$LLAMA_SRC"
fi
cd "$LLAMA_SRC"
git fetch --all --tags --quiet

if [ -n "${LLAMA_PIN_SHA:-}" ]; then
  log "checking out pinned revision $LLAMA_PIN_SHA"
  git checkout --quiet --detach "$LLAMA_PIN_SHA"
else
  warn "LLAMA_PIN_SHA is empty -- building tip of default branch."
  warn "This is acceptable ONLY for the very first bring-up. Copy the SHA printed below"
  warn "into config/node.env (and into every other node's node.env) before Phase 2."
  git checkout --quiet "$(git symbolic-ref --short refs/remotes/origin/HEAD | sed 's|origin/||')"
  git pull --quiet --ff-only
fi
SHA="$(git rev-parse HEAD)"

# ONE list, used for the cmake line AND for the build manifest below. They used to be
# written out separately and had already drifted: the manifest omitted CMAKE_BUILD_TYPE,
# the arch flags and LLAMA_CURL, so the reproducibility record did not describe the build
# it was recording. A manifest that disagrees with the build is worse than none.
#
# -DLLAMA_CURL is deliberately NOT passed: upstream deprecated it to a no-op
# (CMakeLists.txt:195, `llama_option_depr(WARNING LLAMA_CURL)`), so it only emits a
# warning. This node fetches its own models with scripts/03_fetch_models.sh and never
# needs llama.cpp to download anything.
CMAKE_OPTS=(
  -DCMAKE_BUILD_TYPE=Release
  -DGGML_RPC=ON
  -DGGML_NATIVE=OFF
  -DGGML_OPENMP=ON
  # Explicit, though upstream already defaults it off outside Apple: the manifest below is
  # generated from this array, so an inherited default leaves no trace in the build record
  # while the Pi 5's manifest does carry GGML_BLAS=OFF. Two builds identical on this axis
  # would then look different to any cross-node parity check.
  -DGGML_BLAS=OFF
  -DBUILD_SHARED_LIBS=OFF
)

log "configuring (arch flags: $LLAMA_ARCH_FLAGS)"
cmake -S . -B build "${CMAKE_OPTS[@]}" \
  -DCMAKE_C_FLAGS="$LLAMA_ARCH_FLAGS" \
  -DCMAKE_CXX_FLAGS="$LLAMA_ARCH_FLAGS"

# -j2 was chosen when this board was believed to be 4 GB: there, with no swap, -j4 OOMs the
# compiler on the heavier translation units. It is 8 GB (revision d03115) and -j4 has not been
# tried on it, so -j2 stays until it is. -j2 is slower but finishes.
log "building with -j2 (chosen for a 4 GB board; -j4 is untested on this 8 GB one)"
cmake --build build -j2

for b in llama-bench llama-cli; do
  [ -x "$LLAMA_BIN/$b" ] || die "expected binary missing: $LLAMA_BIN/$b"
done
# The rpc server target was renamed rpc-server -> ggml-rpc-server in c299a92
# (2026-06-27), so which name appears here depends on the pinned ref. Accept
# either and record the answer -- the launcher and the healthcheck resolve it
# the same way, via bin_rpc in lib.sh.
RPC_BIN="$(bin_rpc)" || die "no rpc server binary in $LLAMA_BIN (looked for ggml-rpc-server, then rpc-server) -- was -DGGML_RPC=ON honoured?"
RPC_BIN_NAME="$(basename "$RPC_BIN")"
log "rpc server binary: $RPC_BIN_NAME"

# --- build manifest: the reproducibility record -----------------------------
MANIFEST="$DATA_DIR/inventory/build_manifest.json"
cat > "$MANIFEST" <<JSON
{
  "node": "$NODE_ID",
  "built_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "llama_cpp": {
    "repo": "$LLAMA_REPO",
    "commit": "$SHA",
    "describe": "$(git describe --tags --always 2>/dev/null || echo unknown)",
    "pinned": $([ -n "${LLAMA_PIN_SHA:-}" ] && echo true || echo false),
    "rpc_binary": "$RPC_BIN_NAME"
  },
  "toolchain": {
    "cc": "$(cc --version | head -1)",
    "cmake": "$(cmake --version | head -1)",
    "arch_flags": "$LLAMA_ARCH_FLAGS",
    "cmake_options": "${CMAKE_OPTS[*]//-D/} CMAKE_C_FLAGS=$LLAMA_ARCH_FLAGS CMAKE_CXX_FLAGS=$LLAMA_ARCH_FLAGS"
  },
  "os": {
    "pretty_name": "$(. /etc/os-release; echo "$PRETTY_NAME")",
    "kernel": "$(uname -r)",
    "arch": "$(uname -m)"
  },
  "cpu_features": "$(awk -F': ' '/^Features/{print $2; exit}' /proc/cpuinfo)"
}
JSON
log "build manifest -> $MANIFEST"
jq . "$MANIFEST" 2>/dev/null || cat "$MANIFEST"

echo
log "BUILT COMMIT: $SHA"
[ -z "${LLAMA_PIN_SHA:-}" ] && warn ">>> put LLAMA_PIN_SHA=$SHA into config/node.env now <<<"
log "next: scripts/03_fetch_models.sh"
