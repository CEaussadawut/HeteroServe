#!/usr/bin/env bash
# Phase 1 -- one-time OS bootstrap for the Pi 4 worker node.
# Idempotent: safe to re-run after a re-flash.
source "$(dirname "$0")/lib.sh"
need_root "$@"

log "bootstrap: $NODE_ID ($NODE_DESC)"

log "apt update / upgrade"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get -y -qq upgrade

log "installing toolchain + measurement dependencies"
# No libopenblas-dev: the cluster builds llama.cpp with GGML_BLAS=OFF on every node, so the
# Pi4-vs-Pi5 prefill gap measures the ISA difference (dotprod) and not a BLAS difference.
# Leaving the headers off is the second half of that guarantee -- a stray -DGGML_BLAS=ON
# fails to configure rather than quietly producing a node whose CPU numbers are not
# comparable with the others'. See the rationale in 02_build_llama_cpp.sh.
apt-get -y -qq install \
  build-essential cmake git ccache pkg-config \
  python3 python3-venv python3-pip \
  libcurl4-openssl-dev \
  chrony linux-cpupower stress-ng \
  iperf3 ethtool iproute2 tcpdump \
  htop lm-sensors jq rsync tmux

log "creating /opt/heteroserve"
install -d -o "${SUDO_USER:-pi}" -g "${SUDO_USER:-pi}" /opt/heteroserve
ensure_dirs
chown -R "${SUDO_USER:-pi}:${SUDO_USER:-pi}" /opt/heteroserve

# --- shared time base -------------------------------------------------------
# Joules-per-token attribution only works if the power samples and the inference
# events share a clock. The control node is the NTP source; everything else drifts.
log "pointing chrony at the control node ($CONTROL_NTP)"
cat > /etc/chrony/conf.d/heteroserve.conf <<CHRONY
# HeteroServe: the control node (T580) is the single time reference for the cluster.
server $CONTROL_NTP iburst minpoll 4 maxpoll 6 prefer
makestep 0.1 -1
CHRONY
systemctl restart chrony
sleep 2
chronyc sources || warn "chrony has no source yet -- check that the control node serves NTP"

log "setting hostname to $NODE_ID"
hostnamectl set-hostname "$NODE_ID"
grep -q "127.0.1.1.*$NODE_ID" /etc/hosts || sed -i "s/^127.0.1.1.*/127.0.1.1\t$NODE_ID/" /etc/hosts

log "bootstrap done. Next: sudo scripts/01_system_tune.sh"
