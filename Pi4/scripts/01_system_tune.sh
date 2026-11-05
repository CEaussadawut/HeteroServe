#!/usr/bin/env bash
# Put the Pi 4 into the fixed, documented state that every measurement assumes.
# Re-run after every reboot (or install the systemd unit -- see systemd/README notes).
source "$(dirname "$0")/lib.sh"
need_root "$@"

log "pinning CPU governor to 'performance'"
# Rationale: ondemand makes prefill (a short compute burst) race the governor's ramp-up,
# which shows up as variance of tens of percent in TPOT. A fixed governor trades a little
# idle power for a measurable-at-all result.
for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do echo performance > "$c"; done

log "disabling swap"
# The straggler study deliberately overloads this node. With swap on, an overload becomes
# microSD thrashing and the latency tail measures the SD card, not the CPU. With swap off
# it becomes an honest OOM, which is the result we actually want to record.
systemctl stop  dphys-swapfile 2>/dev/null || true
systemctl disable dphys-swapfile 2>/dev/null || true
swapoff -a || true

log "disabling background services that perturb power and latency"
for svc in bluetooth triggerhappy avahi-daemon ModemManager cups cups-browsed unattended-upgrades apt-daily.timer apt-daily-upgrade.timer man-db.timer; do
  systemctl disable --now "$svc" 2>/dev/null || true
done

log "forcing $ETH_IFACE to 1000baseT full duplex (the GbE tier must not auto-negotiate down)"
ethtool -s "$ETH_IFACE" autoneg on speed 1000 duplex full 2>/dev/null || warn "could not force link speed on $ETH_IFACE"

log "clearing any tc qdisc left over from a netem sweep"
tc qdisc del dev "$ETH_IFACE" root 2>/dev/null || true

log "state now:"
printf '  governor : %s\n  swap     : %s kB\n  temp     : %s C\n  clock    : %s MHz\n  throttled: %s\n' \
  "$(governor)" "$(awk '/SwapTotal/{print $2}' /proc/meminfo)" "$(soc_temp_c)" "$(cpu_mhz)" "$(throttled_word)"

log "tune done. Next: scripts/02_build_llama_cpp.sh (as the normal user, not root)"
