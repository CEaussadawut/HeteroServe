#!/usr/bin/env bash
# Set this node's link tier for the interconnect sweep (proposal scope item 6).
#
#   sudo scripts/05_net_tier.sh gbe              real 1 GbE through the managed switch
#   sudo scripts/05_net_tier.sh wifi             real Wi-Fi tier (eth0 down, wlan0 up)
#   sudo scripts/05_net_tier.sh shape 100 1.0    netem: 100 Mbit, +1.0 ms one-way delay
#   sudo scripts/05_net_tier.sh clear            remove shaping
#   sudo scripts/05_net_tier.sh verify           iperf3 + ping against the control node
source "$(dirname "$0")/lib.sh"
cmd="${1:-verify}"

apply_clear() { tc qdisc del dev "$ETH_IFACE" root 2>/dev/null || true; }

case "$cmd" in
  gbe)
    need_root "$@"; apply_clear
    ip link set "$WLAN_IFACE" down 2>/dev/null || true
    ip link set "$ETH_IFACE" up
    ethtool -s "$ETH_IFACE" speed 1000 duplex full autoneg on 2>/dev/null || true
    # Record what the link ACTUALLY negotiated, not what we asked for. A cable or switch
    # port that quietly came up at 100 Mb/s would otherwise be labelled "gbe" in
    # events.jsonl for the whole sweep, and the interconnect study would attribute the
    # resulting slowdown to the model instead of to the cable.
    sleep 2   # autoneg needs a moment before /sys reports the settled speed
    speed="$(cat "/sys/class/net/$ETH_IFACE/speed" 2>/dev/null || echo unknown)"
    declare_net_tier gbe unshaped "speed_mbps=$speed"
    mark net_tier "gbe speed_mbps=$speed"
    if [ "$speed" != "1000" ]; then
      warn "link negotiated ${speed} Mb/s, not 1000 -- this is NOT the GbE tier."
      warn "Check the cable and the switch port before recording anything as 'gbe'."
    fi
    log "tier = GbE on $ETH_IFACE (negotiated ${speed} Mb/s)";;

  wifi)
    need_root "$@"; apply_clear
    ip link set "$ETH_IFACE" down
    ip link set "$WLAN_IFACE" up
    declare_net_tier wifi unshaped ""
    mark net_tier "wifi"; log "tier = Wi-Fi on $WLAN_IFACE (eth0 down so traffic cannot silently take the fast path)";;

  shape)
    need_root "$@"
    rate="${2:?rate in Mbit}"; delay="${3:-0}"
    apply_clear
    # htb caps bandwidth, netem adds the one-way delay. Both are needed: the threshold the
    # proposal looks for is where activation transfer time starts to dominate per-token
    # compute, and that is a function of BOTH rate and RTT.
    tc qdisc add dev "$ETH_IFACE" root handle 1: htb default 10
    tc class add dev "$ETH_IFACE" parent 1: classid 1:10 htb rate "${rate}mbit" ceil "${rate}mbit" burst 15k
    tc qdisc add dev "$ETH_IFACE" parent 1:10 handle 10: netem delay "${delay}ms"
    declare_net_tier shaped shaped "rate=${rate}mbit delay=${delay}ms"
    mark net_tier "shaped rate=${rate}mbit delay=${delay}ms"
    log "tier = shaped ${rate} Mbit / ${delay} ms"
    warn "netem shapes EGRESS only -- apply the same shaping on the peer node for a symmetric link"
    tc -s qdisc show dev "$ETH_IFACE";;

  clear)
    need_root "$@"; apply_clear; declare_net_tier clear unshaped ""
    mark net_tier "clear"; log "shaping removed";;

  verify)
    log "link: $(ethtool "$ETH_IFACE" 2>/dev/null | awk '/Speed|Duplex/{printf "%s ", $0}')"
    log "ping $CONTROL_HOST:"; ping -c 20 -i 0.2 -q "$CONTROL_HOST" || warn "control node unreachable"
    if have iperf3; then
      log "iperf3 -> $CONTROL_HOST (needs 'iperf3 -s' running there)"
      # ONE file -- the one this run just wrote. The glob used to be "$DATA_DIR"/logs/
      # iperf3_*.json, i.e. every measurement ever taken on this node: jq printed a line
      # per file and `tail -1` took the last. With `set -o pipefail` (lib.sh) a malformed
      # newest file made jq exit non-zero AFTER printing the older ones, so the operator
      # saw a stale throughput number and a "failed" warning at the same time and had no
      # way to tell which was true.
      out="$DATA_DIR/logs/iperf3_$(date -u +%Y%m%dT%H%M%SZ).json"
      if iperf3 -c "$CONTROL_HOST" -t 10 -J > "$out" 2>/dev/null; then
        jq -r '.end.sum_received.bits_per_second/1e6 | "throughput: \(.|floor) Mbit/s"' "$out" \
          || warn "iperf3 wrote $out but jq could not read it -- inspect that file"
      else
        rm -f "$out"   # never leave a truncated file for a later run to read
        warn "iperf3 failed -- start 'iperf3 -s' on the control node"
      fi
    fi;;
  *) die "unknown mode: $cmd";;
esac
