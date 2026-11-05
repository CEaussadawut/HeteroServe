#!/usr/bin/env bash
# CMake and Python tooling. Ubuntu 18.04 (JetPack 4.6) ships CMake 3.10, which
# is below llama.cpp's floor; CUDA 10.2 in turn refuses any host gcc above 8.
. "$(dirname "$0")/lib/common.sh"
load_env

step "Host compiler"
need_cmd gcc-8 "run 00_system_prep.sh first"
gcc-8 --version | head -1
CUDA_MAJ="$(cuda_major)"
if [ -n "$CUDA_MAJ" ] && [ "$CUDA_MAJ" -le 10 ]; then
  ok "CUDA ${CUDA_MAJ}.x -> host compiler must be gcc/g++ 8 (configured: ${HOST_CXX})"
fi

step "CMake"
CM="$(command -v cmake || true)"
CMV="$([ -n "$CM" ] && cmake --version | head -1 | awk '{print $3}' || echo 0)"
info "found cmake ${CMV:-none}"
if [ "$(printf '%s\n3.14.0\n' "${CMV:-0}" | sort -V | head -1)" != "3.14.0" ]; then
  info "installing a modern cmake into ~/.local via pip (aarch64 wheels exist)"
  python3 -m pip install --user --upgrade pip
  python3 -m pip install --user "cmake>=3.22,<4"
  export PATH="${HOME}/.local/bin:${PATH}"
  grep -q '.local/bin' "${HOME}/.bashrc" || echo 'export PATH="${HOME}/.local/bin:${PATH}"' >> "${HOME}/.bashrc"
  ok "cmake $(cmake --version | head -1 | awk '{print $3}') at $(command -v cmake)"
else
  ok "cmake new enough"
fi

step "Python packages for telemetry and analysis"
# Deliberately small: the Nano is a measurement node, not an analysis box.
# Heavy analysis (pandas/matplotlib) happens on the laptop over the CSVs.
python3 -m pip install --user -r "${HS_JETSON_DIR}/telemetry/requirements.txt"

step "ccache"
# Rebuilding llama.cpp from scratch on a Nano is ~40-70 min. ccache turns an
# incremental rebuild after a re-pin into minutes.
if command -v ccache >/dev/null 2>&1; then
  ccache -M 5G >/dev/null
  ok "ccache max size 5G"
fi

echo
echo "Next: scripts/20_build_llama_cpp.sh"
