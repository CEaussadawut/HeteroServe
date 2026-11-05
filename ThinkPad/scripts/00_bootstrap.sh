#!/usr/bin/env bash
# One-time setup of the control node (Ubuntu LTS on the ThinkPad T580).
source "$(dirname "$0")/lib.sh"

log "installing packages (sudo)"
sudo apt-get update -qq
# No libopenblas-dev: the cluster builds llama.cpp with GGML_BLAS=OFF on every node, and
# this one is not exempt -- docs/05_experiment_protocol.md puts the T580 into the
# measurement as a prefill node when serves_rpc is set, so its build has to be comparable
# with the workers'. The analysis stack needs no BLAS either: control/ is stdlib-only
# (csv, json, math, statistics) and the optional matplotlib pulls a numpy wheel that
# vendors its own. Absent headers mean a stray -DGGML_BLAS=ON fails to configure.
sudo apt-get -y -qq install \
  build-essential cmake git ccache pkg-config \
  python3 python3-venv python3-pip \
  libcurl4-openssl-dev \
  chrony iperf3 ethtool jq rsync tmux \
  minicom picocom \
  python3-serial

log "creating the venv at $VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$HS_ROOT/requirements.txt"
"$VENV/bin/pip" install --quiet -e "$HS_ROOT"

log "serial access to the Pico (no sudo needed to read the DAQ)"
sudo usermod -aG dialout "$USER"
warn "log out and back in for the dialout group to take effect"

log "creating /opt/heteroserve"
sudo install -d -o "$USER" -g "$USER" /opt/heteroserve
mkdir -p /opt/heteroserve/{models,data/{runs,sweeps,power,inventory,logs}}

# The control node is the cluster's clock. Every energy figure depends on the workers
# agreeing with it, so this is not optional infrastructure.
log "configuring chrony as the cluster's NTP server"
SUBNET="$(python3 -c "import json;print(json.load(open('$CLUSTER_JSON'))['cluster']['subnet'])")"
sudo tee /etc/chrony/conf.d/heteroserve.conf >/dev/null <<CHRONY
# HeteroServe: this machine is the time reference for the cluster.
allow $SUBNET
local stratum 10
CHRONY
sudo systemctl restart chrony
chronyc sources || warn "chrony is not yet synced upstream"

log "bootstrap done. Next: scripts/01_build_llama_cpp.sh"
