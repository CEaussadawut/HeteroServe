#!/usr/bin/env bash
# Build the pinned llama.cpp revision on the control node (x86-64, AVX2).
source "$(dirname "$0")/lib.sh"

SRC="$(cfg paths llama_src)"
BIN="$(cfg paths llama_bin)"
PIN="$(cfg llama pin_sha)"
REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"

[ -d "$SRC/.git" ] || { log "cloning into $SRC"; git clone "$REPO" "$SRC"; }
cd "$SRC"
git fetch --all --tags --quiet

if [ -n "$PIN" ]; then
  log "checking out the cluster's pinned revision $PIN"
  git checkout --quiet --detach "$PIN"
else
  warn "llama.pin_sha is empty in config/cluster.json - building the default branch."
  warn "Pin it (here AND in every node's node.env) before Phase 2 starts."
  git checkout --quiet "$(git symbolic-ref --short refs/remotes/origin/HEAD | sed 's|origin/||')"
  git pull --quiet --ff-only
fi
SHA="$(git rev-parse HEAD)"

# The T580's i5-8350U has AVX2 and FMA but not AVX-512. Spelled out rather than -march=native
# so the control node's build is as reproducible as the workers' builds.
log "configuring (AVX2 + FMA, RPC on)"
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_RPC=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DGGML_OPENMP=ON \
  -DGGML_BLAS=OFF \
  -DLLAMA_CURL=ON

log "building with -j$(nproc)"
cmake --build build -j"$(nproc)"

for b in llama-cli llama-bench llama-server; do
  [ -x "$BIN/$b" ] || die "missing binary after build: $BIN/$b"
done
# The rpc server target was renamed rpc-server -> ggml-rpc-server in c299a92
# (2026-06-27), so which name lands here depends on the pinned ref. Accept
# either rather than failing a perfectly good build, and record the answer in
# the manifest -- the workers resolve it the same way.
RPC_BIN="$(bin_rpc "$BIN")" || die "no rpc server binary in $BIN (looked for ggml-rpc-server, then rpc-server) -- was -DGGML_RPC=ON honoured?"
RPC_BIN_NAME="$(basename "$RPC_BIN")"
log "rpc server binary: $RPC_BIN_NAME"

MANIFEST="$(cfg paths data)/inventory/control_build_manifest.json"
mkdir -p "$(dirname "$MANIFEST")"
cat > "$MANIFEST" <<JSON
{
  "node": "t580",
  "built_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "llama_cpp": {"repo": "$REPO", "commit": "$SHA", "pinned": $([ -n "$PIN" ] && echo true || echo false),
                "rpc_binary": "$RPC_BIN_NAME"},
  "toolchain": {"cc": "$(cc --version | head -1)", "cmake": "$(cmake --version | head -1)",
                "cmake_options": "GGML_RPC=ON GGML_NATIVE=OFF AVX2 FMA F16C OPENMP GGML_BLAS=OFF LLAMA_CURL=ON"},
  "os": {"pretty_name": "$(. /etc/os-release; echo "$PRETTY_NAME")", "kernel": "$(uname -r)"}
}
JSON
log "manifest -> $MANIFEST"
echo
log "BUILT COMMIT: $SHA"
[ -z "$PIN" ] && warn ">>> set llama.pin_sha to $SHA in config/cluster.json <<<"
