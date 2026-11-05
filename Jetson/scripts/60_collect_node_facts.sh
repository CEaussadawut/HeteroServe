#!/usr/bin/env bash
# Emit this node's row of the device-characterisation dataset as JSON.
# Run once per node after 00/10/20 and commit the output; every later
# measurement is interpreted against it.
#   ./60_collect_node_facts.sh > ../results/jetson-nano-01_facts.json
. "$(dirname "$0")/lib/common.sh"
load_env

j() { python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))'; }

# Prefix for every sysfs/procfs lookup below. Empty on a real node; a test points
# it at a fake tree so the clock-cap logic can be exercised without a Jetson --
# which matters, because the paths differ between L4T releases and getting one
# wrong is exactly how gpu.max_freq_hz came to be 0 on every run.
SYSFS="${HS_SYSFS_ROOT:-}"

first_path() {
  # First readable path among the arguments. Unmatched globs are left literal by
  # the shell and simply fail the -r test, so no `ls` and no pipefail guard.
  local c
  for c in "$@"; do
    [ -r "$c" ] && { printf '%s' "$c"; return 0; }
  done
  return 1
}

CPU_MODEL="$( set +o pipefail
  awk -F': ' '/model name|Model|CPU part/{print $2; exit}' /proc/cpuinfo || true )"
CPU_CORES="$(nproc)"

# The clock ceilings are the experimental condition, not trivia. nvpmodel caps
# the CPU through scaling_max_freq and the GPU through its devfreq node; the
# silicon ceilings (cpuinfo_max_freq, the top of available_frequencies) do NOT
# move with the mode. Recording only those makes a 5 W row and a 10 W row look
# identical in the dataset -- and telling those two apart is the whole reason
# this fact is collected. So record the cap in force AND the ceiling it was cut
# from, and let the pair speak.
CPU_MAXF_F="$( first_path \
  "${SYSFS}/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq" \
  "${SYSFS}/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq" || true )"
CPU_MAXF="$(cat "$CPU_MAXF_F" 2>/dev/null || echo 0)"
CPU_HWMAXF_F="$( first_path \
  "${SYSFS}/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq" \
  "${SYSFS}/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_max_freq" || true )"
CPU_HWMAXF="$(cat "$CPU_HWMAXF_F" 2>/dev/null || echo 0)"

# On Tegra X1 under L4T R32 the GPU devfreq node hangs off the platform device,
# not the /sys/devices/gpu.0 path this kit used -- which is why this field read 0
# on every run. Try the spellings in order rather than betting on one.
GPU_MAXF_F="$( first_path \
  "${SYSFS}/sys/devices/57000000.gpu/devfreq/57000000.gpu/max_freq" \
  "${SYSFS}/sys/devices/platform/host1x/57000000.gpu/devfreq/57000000.gpu/max_freq" \
  "${SYSFS}/sys/class/devfreq/57000000.gpu/max_freq" \
  "${SYSFS}/sys/devices/gpu.0/devfreq/57000000.gpu/max_freq" \
  ${SYSFS}/sys/class/devfreq/*.gpu/max_freq \
  ${SYSFS}/sys/class/devfreq/*.gm20b/max_freq \
  ${SYSFS}/sys/class/devfreq/*.gp10b/max_freq || true )"
GPU_MAXF="$(cat "$GPU_MAXF_F" 2>/dev/null || echo 0)"
# available_frequencies sits beside max_freq: one space-separated line. Take the
# maximum rather than trusting the ordering.
GPU_HWMAXF=0
if [ -n "$GPU_MAXF_F" ] && [ -r "${GPU_MAXF_F%/max_freq}/available_frequencies" ]; then
  GPU_HWMAXF="$( set +o pipefail
    tr ' ' '\n' < "${GPU_MAXF_F%/max_freq}/available_frequencies" \
    | awk '$1+0 > m {m=$1+0} END{printf "%d", m+0}' || echo 0 )"
fi

# The power mode actually in force, read from the board rather than echoed back
# from the config knob: a facts file that says mode 0 because node.env says so,
# while the board is in mode 1, mislabels every row collected beside it. nvpmodel
# records the live mode in /var/lib/nvpmodel/status as "pmode:0001 fmode:quiet",
# and that file needs no sudo -- `nvpmodel -q` does, and a facts run that stops
# for a password prompt is the #24/#6 bug class all over again.
NVP_ACTIVE="$(sed -n 's/.*pmode:0*\([0-9][0-9]*\).*/\1/p' \
  "${SYSFS}/var/lib/nvpmodel/status" 2>/dev/null || true)"

MEM_TOTAL_MB="$(awk '/MemTotal/{printf "%d", $2/1024}' /proc/meminfo)"
SWAP_TOTAL_MB="$(awk '/SwapTotal/{printf "%d", $2/1024}' /proc/meminfo)"
# A node with no default route, no link, or no ethtool is a fact worth
# recording -- not a reason to emit nothing at all.
NIC="$( set +o pipefail; ip -o route get 1.1.1.1 2>/dev/null \
  | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' || true )"
NIC_SPEED="$( set +o pipefail; ethtool "${NIC:-none}" 2>/dev/null \
  | sed -n 's/.*Speed: \(.*\)/\1/p' | head -1 || true )"
IP="$( set +o pipefail; ip -o -4 addr show "${NIC:-none}" 2>/dev/null \
  | awk '{print $4}' | cut -d/ -f1 || true )"
# Resolved here rather than inside the heredoc: nesting a subshell that sources
# os-release inside an expanding heredoc is fragile.
DISTRO="$(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}")"
L4T="$(l4t_version)"
KERNEL="$(uname -r)"
ARCH="$(uname -m)"
NOW="$(timestamp)"

# tegrastats one-shot for the COLD-BOARD idle figure. This is a device-table
# number -- what this node burns doing nothing, once, for the node comparison --
# and it is NOT the floor subtracted from a sweep. That floor is per case and is
# measured from the cooldown that preceded each case (bench/run_matrix.py's
# idle_floor_at(); every joined row carries it as idle_mw). The two must not be
# swapped: leakage rises with junction temperature, so this cold number sits
# several hundred milliwatts below the floor a case an hour into a sweep ran
# against, and using it as the subtrahend biases the result along case order.
#
# Sampled for 15 s and averaged over everything after the first 3 s: the first
# tegrastats line lands while the daemon is still starting, and three
# instantaneous readings of a switching rail are not a mean of anything.
IDLE_MW=""
IDLE_N=""
if command -v tegrastats >/dev/null 2>&1; then
  # tegrastats never exits on its own, so `timeout` ALWAYS returns 124 here --
  # that is the design, not a failure. Unguarded under pipefail it killed this
  # script at the 6 s mark, before the heredoc below that is its entire output,
  # and `make facts | tee ...` then reported success over a zero-byte JSON file.
  IDLE_SAMPLES="$( set +o pipefail
    timeout 15 tegrastats --interval 1000 2>/dev/null \
    | sed -n 's/.*POM_5V_IN \([0-9]*\)\/.*/\1/p' || true )"
  # drop the first 3 (one per second) -- daemon start-up and clock settling
  IDLE_MW="$( set +o pipefail; printf '%s\n' "$IDLE_SAMPLES" \
    | awk 'NR>3 {s+=$1; n++} END{if(n) printf "%d", s/n}' || true )"
  IDLE_N="$( set +o pipefail; printf '%s\n' "$IDLE_SAMPLES" \
    | awk 'NR>3 {n++} END{printf "%d", n+0}' || true )"
fi

BUILDINFO="${HS_ROOT}/build_info.json"
[ -f "$BUILDINFO" ] || BUILDINFO=/dev/null

python3 - "$BUILDINFO" <<PY
import json, sys
build = {}
try:
    build = json.load(open(sys.argv[1]))
except Exception:
    pass
facts = {
  "node_id": "${NODE_ID}",
  "role": "${NODE_ROLE}",
  "soc": "${NODE_SOC}",
  "collected_at_utc": "${NOW}",
  "os": {
    "l4t": "${L4T}",
    "kernel": "${KERNEL}",
    "distro": "${DISTRO}",
    "arch": "${ARCH}"
  },
  "cpu": {
    "model": "${CPU_MODEL}",
    "cores": ${CPU_CORES:-0},
    "max_freq_khz": ${CPU_MAXF:-0},
    "hw_max_freq_khz": ${CPU_HWMAXF:-0}
  },
  "gpu": {
    "name": "128-core Maxwell (GM20B)",
    "compute_capability": "5.3",
    "max_freq_hz": ${GPU_MAXF:-0},
    "hw_max_freq_hz": ${GPU_HWMAXF:-0},
    "max_freq_source": "${GPU_MAXF_F}",
    "fp16": True,
    "int8_dp4a": False
  },
  "memory": {
    "total_mb": ${MEM_TOTAL_MB:-0},
    "swap_mb": ${SWAP_TOTAL_MB:-0},
    "unified_cpu_gpu": True
  },
  "power": {
    "nvpmodel": "${NVP_MODEL}",
    "nvpmodel_active": "${NVP_ACTIVE}",
    "clocks_locked": ${LOCK_CLOCKS:-0} == 1,
    "idle_board_mw": ${IDLE_MW:-0},
    "idle_board_mw_basis": "cold board, otherwise unloaded: mean POM_5V_IN over a 15 s tegrastats run, first 3 s dropped, ${IDLE_N:-0} samples. Device-table figure only -- the floor subtracted from a sweep is per case, from the cooldown before it (joined CSV column idle_mw)",
    "rail_source": "carrier-board INA3221 via tegrastats POM_5V_IN",
    "rail_scope": "whole dev kit 5V input: module + carrier + USB/HDMI; excludes wall-adapter loss",
    "external_meter": False
  },
  "network": {
    "iface": "${NIC}",
    "ipv4": "${IP}",
    "speed": "${NIC_SPEED}"
  },
  "build": build
}
json.dump(facts, sys.stdout, indent=2, default=str)
print()
PY
