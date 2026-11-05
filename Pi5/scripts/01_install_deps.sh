#!/usr/bin/env bash
# 01 -- packages the Pi 5 node needs. Raspberry Pi OS (64-bit), Bookworm or later.
. "$(dirname "$0")/lib/common.sh"
load_env
need_root "$@"

export DEBIAN_FRONTEND=noninteractive
log "apt update"
apt-get update -qq

log "installing build, measurement and network-shaping packages"
# sched_netem lives in linux-modules-extra on some images; on Raspberry Pi OS the
# netem qdisc is built in, so only the userspace tooling is listed here.
# No libopenblas-dev: the cluster builds llama.cpp with GGML_BLAS=OFF on every node
# so the Pi4-vs-Pi5 prefill gap measures the ISA difference and not a BLAS
# difference, and so OpenBLAS's own thread pool cannot contend with llama.cpp's
# for the four A76 cores. See the rationale in 02_build_llama_cpp.sh.
apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build git pkg-config ccache \
  python3 python3-venv python3-pip \
  libcurl4-openssl-dev \
  iproute2 ethtool iperf3 jq curl ca-certificates

log "python side (agent + telemetry + model fetch) in an isolated venv"
python3 -m venv /opt/heteroserve/venv
/opt/heteroserve/venv/bin/pip install --quiet --upgrade pip
/opt/heteroserve/venv/bin/pip install --quiet "huggingface_hub[cli]" psutil

ok "gcc     $(gcc -dumpversion)"
ok "cmake   $(cmake --version | head -1 | awk '{print $3}')"
ok "python  $(/opt/heteroserve/venv/bin/python -V | awk '{print $2}') (/opt/heteroserve/venv)"
