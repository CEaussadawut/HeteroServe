#!/bin/bash
# Make the control-node laptop the testbed's NTP server. Run ONCE, as root: sudo bash testbed/scripts/laptop_ntp_server.sh
#
# Why: since 2026-09-26 the boards sit behind a True GIGATEX T628L (192.168.100.1) that has NO WAN, so they have no
# time source. Their clocks drifted ~11 h, which breaks cross-node log correlation and the power-guard log dating.
# The laptop has real time (it reaches the internet over Wi-Fi), so it serves it to the testbed subnets.
#
# What it changes, and nothing else:
#   - installs chrony (on Ubuntu 24.04 this replaces systemd-timesyncd; chrony then keeps the laptop itself in sync
#     from Ubuntu's default pools, as timesyncd did);
#   - adds /etc/chrony/conf.d/heteroserve-lan.conf: serve 192.168.100.0/24 and 192.168.10.0/24;
#   - adds two ufw rules allowing udp/123 from those two subnets only. No other ufw rule is touched.
# Undo: sudo rm /etc/chrony/conf.d/heteroserve-lan.conf && sudo ufw delete allow from 192.168.100.0/24 to any port 123 proto udp
#       && sudo ufw delete allow from 192.168.10.0/24 to any port 123 proto udp && sudo apt-get install -y systemd-timesyncd
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run as root: sudo bash $0"; exit 1; }

echo "== installing chrony"
DEBIAN_FRONTEND=noninteractive apt-get install -y chrony

echo "== testbed config"
install -d -m 0755 /etc/chrony/conf.d
cat > /etc/chrony/conf.d/heteroserve-lan.conf <<'EOF'
# HeteroServe testbed (2026-09-26): serve time to the boards. The T628L router has no WAN, so they have no other source.
allow 192.168.100.0/24
allow 192.168.10.0/24
# Keep serving if the laptop itself loses internet. Stratum 10 marks it as a local, non-authoritative clock:
# the boards stay consistent with each other and with the laptop, which is what log correlation needs.
local stratum 10
EOF
grep -q '^confdir /etc/chrony/conf.d' /etc/chrony/chrony.conf || { echo "!! chrony.conf has no 'confdir /etc/chrony/conf.d' line -- config would be ignored"; exit 2; }
systemctl enable --now chrony
systemctl restart chrony

echo "== firewall (udp/123 from the two testbed subnets only)"
ufw allow from 192.168.100.0/24 to any port 123 proto udp comment 'HeteroServe NTP'
ufw allow from 192.168.10.0/24  to any port 123 proto udp comment 'HeteroServe NTP'

echo "== verify"
sleep 3
systemctl is-active chrony
ss -lunp | grep -E ':123\b' || echo "!! chrony is not listening on udp/123"
chronyc -n tracking | sed -n '1,6p'
ufw status | grep -E '123/udp' || true
echo "== done. Tell Claude, and it will point the boards at 192.168.100.105."
