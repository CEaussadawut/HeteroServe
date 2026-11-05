#!/usr/bin/env bash
# Put a freshly flashed Jetson Nano into the state the measurements assume:
# headless, swapped, clock-locked, fan pinned, packages present.
# Idempotent — safe to re-run. Needs sudo.
. "$(dirname "$0")/lib/common.sh"
load_env
require_jetson

step "System facts"
echo "  L4T      : $(l4t_version)"
echo "  CUDA     : $(cuda_version)"
echo "  Kernel   : $(uname -r)"
echo "  RAM      : $(awk '/MemTotal/{printf "%.0f MB", $2/1024}' /proc/meminfo)"

step "Packages"
sudo apt-get update
# No libopenblas-dev: the cluster builds llama.cpp with BLAS OFF on every node
# (decided 2026-09-14), so the headers are not needed and their absence is a second
# line of defence -- a stray -DGGML_BLAS=ON would fail to configure rather than
# quietly producing a node whose CPU numbers are not comparable with the others'.
sudo apt-get install -y --no-install-recommends \
  build-essential gcc-8 g++-8 git git-lfs curl ca-certificates \
  python3 python3-pip python3-dev python3-setuptools \
  pkg-config ccache jq bc lm-sensors ethtool iperf3 chrony
ok "apt packages installed"

step "Headless boot (frees ~700-900 MB of the 4 GB pool)"
if [ "$(systemctl get-default)" != "multi-user.target" ]; then
  sudo systemctl set-default multi-user.target
  warn "desktop disabled; takes effect on next reboot"
else
  ok "already multi-user.target"
fi

step "Swap"
# The Nano ships with ~2 GB of zram (nvzramconfig). zram is compressed RAM, so
# it does not help when the pressure IS ram. Add a real file-backed swap for the
# nvcc build and for deliberate over-commit experiments.
SWAPFILE=/var/swapfile
SWAP_GB=${SWAP_GB:-8}
SWAP_BYTES=$((SWAP_GB * 1024 * 1024 * 1024))

# Is $SWAPFILE a COMPLETE, correctly sized swap area? "The file exists" is not
# the same question, and conflating them is what wedged this script: allocating
# 8 GB on an SD card takes minutes, so a Ctrl-C or a power cut leaves a partial
# file that was never mkswap'd. The old guard was `[ ! -f "$SWAPFILE" ]`, which
# is then false, so it skipped BOTH the allocation and the mkswap and went
# straight to `swapon`, which failed -- and under `set -e` the script died.
# Every subsequent run took the identical path, so the box could never be
# prepped again without hand-deleting the file.
swapfile_ready() {
  [ -f "$SWAPFILE" ] || return 1
  local have
  have="$( set +o pipefail; stat -c %s "$SWAPFILE" 2>/dev/null || echo 0 )"
  [ "$have" = "$SWAP_BYTES" ] || return 1
  # `swaplabel` reads the header mkswap writes; no header means the allocation
  # finished but mkswap did not, or neither did.
  sudo swaplabel "$SWAPFILE" >/dev/null 2>&1
}

if ! swapon --show=NAME --noheadings | grep -qx "$SWAPFILE"; then
  if ! swapfile_ready; then
    if [ -f "$SWAPFILE" ]; then
      warn "${SWAPFILE} exists but is not a usable ${SWAP_GB}G swap area"
      warn "(partial allocation, or mkswap never ran) -- rebuilding it"
      sudo rm -f "$SWAPFILE"
    fi
    info "allocating ${SWAP_GB}G at ${SWAPFILE} (slow on SD; be patient)"
    sudo fallocate -l "${SWAP_GB}G" "$SWAPFILE" 2>/dev/null \
      || sudo dd if=/dev/zero of="$SWAPFILE" bs=1M count=$((SWAP_GB*1024)) status=progress
    sudo chmod 600 "$SWAPFILE"
    sudo mkswap "$SWAPFILE"
  fi
  if ! sudo swapon "$SWAPFILE"; then
    # Do not die: everything after this step is still worth running, and the
    # build just needs -j1 instead of -j2 without swap.
    warn "swapon failed for ${SWAPFILE}; continuing WITHOUT extra swap"
    warn "the CUDA build may OOM -- use BUILD_JOBS=1"
  else
    grep -q "$SWAPFILE" /etc/fstab || echo "$SWAPFILE none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
    ok "swap on"
  fi
else
  ok "swap already on"
fi
free -h

step "Power mode and clocks"
# Mode 0 (MAXN, 10 W) needs the 5V/4A barrel jack AND the J48 jumper fitted.
# On micro-USB this will brown out under a GPU+4-core load — stay in mode 1.
if [ "${NVP_MODEL}" = "0" ]; then
  warn "mode 0 (MAXN/10W) assumes barrel jack + J48 jumper"
fi
sudo nvpmodel -m "${NVP_MODEL}"
sudo nvpmodel -q || true
if [ "${LOCK_CLOCKS}" = "1" ]; then
  # Without this the governor wanders and every benchmark repetition differs.
  sudo jetson_clocks
  sudo jetson_clocks --show || true
  ok "clocks locked"
fi
if [ -w /sys/devices/pwm-fan/target_pwm ] || sudo test -w /sys/devices/pwm-fan/target_pwm; then
  echo "${FAN_PWM}" | sudo tee /sys/devices/pwm-fan/target_pwm >/dev/null || true
  ok "fan pwm = ${FAN_PWM}"
else
  warn "no pwm-fan node; if you have no fan, expect thermal throttling above ~45 min of load"
fi

step "Clock discipline (for cross-node time alignment)"
# Energy-per-token needs the four nodes' logs to line up. chrony against the
# orchestrator (or any LAN NTP source) keeps the skew inside a few ms.
sudo systemctl enable --now chrony 2>/dev/null || sudo systemctl enable --now chronyd 2>/dev/null || true
chronyc tracking 2>/dev/null | head -5 || warn "chrony not reporting yet"

step "Network"
for i in $(ls /sys/class/net | grep -v lo); do
  printf '  %-8s %s\n' "$i" "$(ethtool "$i" 2>/dev/null | sed -n 's/\tSpeed: /speed /p' | head -1)"
done
warn "run the cluster benchmarks over wired GbE; Wi-Fi is a separate, labelled condition"

step "Power rails"
# The Nano B01 dev kit carries an INA3221 on the CARRIER board; tegrastats exposes
# POM_5V_IN / _GPU / _CPU in milliwatts. POM_5V_IN is therefore the whole dev
# kit's 5 V input — module plus carrier plus USB/HDMI draw — not module-only
# power (that is a TX2/Xavier property). Only wall-adapter conversion loss is
# outside it. There is no external meter: this is the study's only power signal.
if command -v tegrastats >/dev/null 2>&1; then
  timeout 3 tegrastats --interval 1000 | head -2 || true
  ok "tegrastats available"
else
  warn "tegrastats not found — power logging will be unavailable"
fi

step "Done"
echo "Next: scripts/10_install_toolchain.sh"
[ "$(systemctl get-default)" = "multi-user.target" ] && echo "Reboot now if the desktop was still running."
