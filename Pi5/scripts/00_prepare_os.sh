#!/usr/bin/env bash
# 00 -- one-time OS preparation for the Pi 5 worker.
# Pins clocks, disables the sources of variance that would otherwise show up as
# noise in the characterisation dataset, and points the clock at the control node.
. "$(dirname "$0")/lib/common.sh"
load_env
need_root "$@"
assert_this_board

log "disabling swap (a swapped-out run is a corrupted measurement, not a slow one)"
if command -v dphys-swapfile >/dev/null 2>&1; then
  dphys-swapfile swapoff || true
  systemctl disable --now dphys-swapfile.service 2>/dev/null || true
fi
swapoff -a || true
ok "swap: $(free -m | awk '/^Swap:/{print $2" MB"}')"

log "pinning cpu governor to ${CPU_GOVERNOR}"
# Count the outcome per core instead of a bare `[ -w ] && echo >`: a core that silently
# failed to pin is an unpinned measurement, and the cpu0 readout below would otherwise
# imply the whole package was set. 06_verify_node.sh gates on all four.
pinned=0; unpinned=0
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  if [ -w "$g" ] && echo "$CPU_GOVERNOR" > "$g" 2>/dev/null; then
    pinned=$((pinned+1))
  else
    unpinned=$((unpinned+1)); warn "could not pin ${g}"
  fi
done
if [ "$unpinned" -gt 0 ]; then
  warn "${unpinned} core(s) NOT pinned to ${CPU_GOVERNOR} -- clocks are not uniform;"
  warn "do not record a dataset until 06_verify_node.sh reports every core pinned"
fi
ok "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) (${pinned} core(s) pinned)"

if [ "${FAN_FORCE_FULL:-0}" = "1" ]; then
  log "pinning the active cooler to full for the duration of the sweeps"
  # The Pi 5 fan is a kernel cooling device; forcing cur_state to max keeps the
  # thermal history of every cell identical instead of letting the fan curve
  # introduce a variable that is not in the experiment design.
  for cd in /sys/class/thermal/cooling_device*; do
    [ -e "$cd/type" ] || continue
    case "$(cat "$cd/type")" in
      *fan*|*pwm*)
        max="$(cat "$cd/max_state")"
        if echo "$max" > "$cd/cur_state" 2>/dev/null; then
          ok "$(basename "$cd") -> state $max"
        else
          warn "could not force $(basename "$cd") to state ${max} -- the fan curve stays"
          warn "variable across cells, which is exactly what FAN_FORCE_FULL exists to stop"
        fi ;;
    esac
  done
fi

log "synchronising the clock against the control node (${CONTROL_NTP})"
# Every INA226 sample and every harness event is joined on timestamp, so the node
# clock has to track the collector's, not a public pool.
if [ -d /etc/systemd/timesyncd.conf.d ] || mkdir -p /etc/systemd/timesyncd.conf.d; then
  cat > /etc/systemd/timesyncd.conf.d/heteroserve.conf <<TSCONF
[Time]
NTP=${CONTROL_NTP}
FallbackNTP=
TSCONF
  systemctl restart systemd-timesyncd || warn "systemd-timesyncd not running"
fi
timedatectl set-timezone UTC || warn "could not force UTC"
ok "clock: $(date -u '+%Y-%m-%dT%H:%M:%SZ') UTC, source ${CONTROL_NTP}"

log "creating the node tree"
install -d -m 0755 "${LLAMA_SRC%/*}" "$MODEL_DIR" "$DATA_DIR" "$RESULTS_DIR" "$RPC_CACHE_DIR"
ok "tree under /opt/heteroserve ready"

log "recording the boot-time baseline"
{
  echo "captured_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model=$(tr -d '\0' < /proc/device-tree/model)"
  echo "kernel=$(uname -srm)"
  echo "os=$(. /etc/os-release; echo "$PRETTY_NAME")"
  echo "firmware=$(vcgencmd version 2>/dev/null | tr '\n' ' ' || echo n/a)"
  echo "ram_mb=$(free -m | awk '/^Mem:/{print $2}')"
  echo "cpu_features=$(awk -F': ' '/^Features/{print $2; exit}' /proc/cpuinfo)"
} > "${DATA_DIR}/node_baseline.txt"
ok "wrote ${DATA_DIR}/node_baseline.txt"

warn "governor and fan settings are NOT persistent across reboot -- rerun this script after every boot, or enable heteroserve-prepare.service (see systemd/)"
