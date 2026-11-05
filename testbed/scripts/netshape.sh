#!/bin/bash
# Shape the Pi 4 <-> Jetson path, from the Pi 4 alone (the network study, topic-form scope 6; 2026-10-08). From the repo root:
#     bash testbed/scripts/netshape.sh apply RATE_MBIT DELAY_MS   # RATE_MBIT 0 = no rate limit; DELAY_MS one way, per direction
#     bash testbed/scripts/netshape.sh clear
#     bash testbed/scripts/netshape.sh check                      # exit 0 only when nothing of ours is installed
#     bash testbed/scripts/netshape.sh show
# Only the traffic between the Pi 4's eth0 and the Jetson (192.168.1.114 since 2026-10-09; .109 on 2026-10-08) is shaped. The Jetson is not touched (its SD
# card is the suspect in three hangs), so both directions are shaped on the Pi 4:
#   - egress TO the Jetson: a prio root, whose third band (1:3) carries only what a u32 filter on the Jetson's address puts
#     there, with netem on that band. Every other packet keeps the default bands, unshaped;
#   - ingress FROM the Jetson: an ingress qdisc whose u32 filter on the Jetson's source address redirects (act_mirred) to
#     ifb0, with the same netem on ifb0's root.
# So an added delay d adds 2d to the pair's round trip, and a rate limit applies to each direction. The figures a cell is
# priced with are MEASURED on the shaped path (measure_link.sh), never taken from these settings.
# netem's queue is 10,000 packets, so a slow link queues rather than drops. ssh, the strip and the laptop are untouched.
set -uo pipefail
NODE=pi4; DEV=eth0; IFB=ifb0
# 2026-10-10 (the draft's placement under shaping): NETSHAPE_PEER picks the other end (default the Jetson, as every campaign
# before), and NETSHAPE_PORT narrows the shaping to the TCP traffic to and from one port on it -- e.g. the laptop's draft
# ggml-rpc-server, 192.168.1.108:50053 -- so that this laptop's ssh and HTTP to the Pi 4 stay unshaped. Shaping every packet
# to a peer other than the Jetson is refused: for the laptop that would shape this control session too.
PEER=${NETSHAPE_PEER:-192.168.1.114}; PORT=${NETSHAPE_PORT:-}
# 2026-10-24: or on the wired overlay (192.168.10.x). serve_pool.sh reaches the Pi 4 at 192.168.10.12, so the laptop's RPC
# traffic with it runs 192.168.10.5 <-> 192.168.10.12 on the same NICs, and a filter on the laptop's router address
# (192.168.1.108) never matched it (ns4b, 2026-10-24: installed, verified, and no RPC packet shaped).
[[ "$PEER" =~ ^192\.168\.(1|10)\.[0-9]+$ ]] || { echo "FATAL: NETSHAPE_PEER must be on the router LAN (192.168.1.x) or the wired overlay (192.168.10.x)"; exit 2; }
[[ -z "$PORT" || "$PORT" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "FATAL: NETSHAPE_PORT: port numbers, comma-separated"; exit 2; }
[ "$PEER" = 192.168.1.114 ] || [ -n "$PORT" ] || { echo "FATAL: shaping every packet to $PEER: give NETSHAPE_PORT"; exit 2; }
# one egress (the Pi 4 -> PEER[:port]) and one ingress (PEER[:port] -> the Pi 4) u32 filter per port, all into the same netem
# queue, so several ports share one shaped link (the draft's RPC port and a port to measure that link on)
F_OUT=""; F_IN=""
if [ -z "$PORT" ]; then             # every packet to and from the peer, as every campaign before 2026-10-10
  F_OUT="\$T filter add dev $DEV parent 1: protocol ip prio 1 u32 match ip dst $PEER/32 flowid 1:3"$'\n'
  F_IN="\$T filter add dev $DEV parent ffff: protocol ip prio 1 u32 match ip src $PEER/32 action mirred egress redirect dev $IFB"$'\n'
fi
for pt in ${PORT//,/ }; do         # the port on EITHER end of a connection: whichever side listens (2026-10-10: a transfer
  for side in dport sport; do         # the laptop sent to a Pi 4 listener went unshaped with only one side matched)
    F_OUT+="\$T filter add dev $DEV parent 1: protocol ip prio 1 u32 match ip dst $PEER/32 match ip $side $pt 0xffff flowid 1:3"$'\n'
    F_IN+="\$T filter add dev $DEV parent ffff: protocol ip prio 1 u32 match ip src $PEER/32 match ip $side $pt 0xffff action mirred egress redirect dev $IFB"$'\n'
  done
done
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8 ${NETSHAPE_SSH_OPTS:-}"   # e.g. a Tailscale route for a test off the LAN
cmd=${1:-show}
run() { $SSH $NODE "bash -s" -- "$@"; }
case "$cmd" in
  apply)
    rate=${2:?usage: apply RATE_MBIT DELAY_MS}; delay=${3:?usage: apply RATE_MBIT DELAY_MS}
    [[ "$rate" =~ ^[0-9]+$ && "$delay" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "FATAL: rate (integer Mbit/s) and delay (ms)"; exit 2; }
    ne="limit 10000"; [ "$delay" != 0 ] && ne+=" delay ${delay}ms"; [ "$rate" != 0 ] && ne+=" rate ${rate}mbit"
    [ "$ne" = "limit 10000" ] && { echo "FATAL: nothing to apply (rate 0 and delay 0); use clear"; exit 2; }
    run <<EOF
set -e
T="sudo -n /usr/sbin/tc"; IP="sudo -n /usr/sbin/ip"
if \$T qdisc show dev $DEV | grep -qE "netem|prio|ingress" || \$T qdisc show dev $IFB 2>/dev/null | grep -q netem; then
  echo "FATAL: shaping already installed; clear first"; exit 3
fi
sudo -n /usr/sbin/modprobe ifb
\$IP link show $IFB >/dev/null 2>&1 || \$IP link add $IFB type ifb
\$IP link set $IFB up
\$T qdisc add dev $DEV root handle 1: prio
\$T qdisc add dev $DEV parent 1:3 handle 30: netem $ne
$F_OUT\$T qdisc add dev $DEV handle ffff: ingress
$F_IN\$T qdisc add dev $IFB root handle 1: netem $ne
echo "APPLIED $DEV<->$PEER${PORT:+ port $PORT}: $ne (each direction)"
EOF
    ;;
  clear)
    run <<EOF
T="sudo -n /usr/sbin/tc"; IP="sudo -n /usr/sbin/ip"
\$T qdisc del dev $DEV root 2>/dev/null
\$T qdisc del dev $DEV ingress 2>/dev/null
\$T qdisc del dev $IFB root 2>/dev/null
\$IP link set $IFB down 2>/dev/null
if \$T qdisc show dev $DEV | grep -qE "netem|prio|ingress" || \$T qdisc show dev $IFB 2>/dev/null | grep -q netem; then
  echo "!!!!! CLEAR FAILED: \$(\$T qdisc show dev $DEV | tr '\n' ' ')"; exit 4
fi
echo "CLEARED $DEV: \$(\$T qdisc show dev $DEV | head -1)"
EOF
    ;;
  check)
    run <<EOF
T="sudo -n /usr/sbin/tc"
if \$T qdisc show dev $DEV | grep -qE "netem|prio|ingress" || \$T qdisc show dev $IFB 2>/dev/null | grep -q netem; then
  echo "SHAPED: \$(\$T qdisc show dev $DEV | tr '\n' ' ')"; exit 1
fi
echo "UNSHAPED"
EOF
    ;;
  show)
    run <<EOF
T="sudo -n /usr/sbin/tc"
echo "== qdisc $DEV"; \$T qdisc show dev $DEV
echo "== filter $DEV"; \$T filter show dev $DEV; \$T filter show dev $DEV parent ffff: 2>/dev/null
echo "== qdisc $IFB"; \$T qdisc show dev $IFB 2>/dev/null
EOF
    ;;
  *) echo "usage: netshape.sh apply RATE_MBIT DELAY_MS | clear | check | show"; exit 2 ;;
esac
