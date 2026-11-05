#!/usr/bin/env bash
# 05 -- bring up this node's link tiers and offer the tc/netem shaping used in the
# Phase 4 interconnect study.
#
#   ./05_network_setup.sh addr              static addresses on the wired tiers
#   ./05_network_setup.sh tier gbe|2.5gbe|wifi   select the active tier
#   ./05_network_setup.sh shape <rate> [delay_ms] [loss_pct]   synthesise a link
#   ./05_network_setup.sh clear             remove all shaping
#   ./05_network_setup.sh show              report link state
. "$(dirname "$0")/lib/common.sh"
load_env

cmd="${1:-show}"

iface_of_tier() {
  case "$1" in
    gbe)    echo "$ETH_IFACE" ;;
    2.5gbe) echo "${ETH25_IFACE:-}" ;;
    wifi)   echo "$WLAN_IFACE" ;;
    *) die "unknown tier '$1' (gbe|2.5gbe|wifi)" ;;
  esac
}

case "$cmd" in
addr)
  need_root "$@"
  log "wired GbE: ${ETH_ADDR}/24 on ${ETH_IFACE}"
  ip addr replace "${ETH_ADDR}/24" dev "$ETH_IFACE"
  ip link set "$ETH_IFACE" up
  if [ -n "${ETH25_IFACE:-}" ] && ip link show "$ETH25_IFACE" >/dev/null 2>&1; then
    log "2.5GbE: ${ETH25_ADDR}/24 on ${ETH25_IFACE}"
    ip addr replace "${ETH25_ADDR}/24" dev "$ETH25_IFACE"
    ip link set "$ETH25_IFACE" up
    # A USB 3.0 2.5GbE adapter only reaches ~275 MB/s with a large MTU and the
    # offloads left on; both are recorded here because they affect the measured
    # activation-transfer cost.
    ethtool "$ETH25_IFACE" 2>/dev/null | awk '/Speed|Duplex/{print "    "$0}'
  else
    warn "2.5GbE adapter not present (ETH25_IFACE='${ETH25_IFACE:-}') -- the Pi 5 stays on its native GbE"
  fi
  ;;
tier)
  need_root "$@"
  tier="${2:?usage: $0 tier gbe|2.5gbe|wifi}"
  want="$(iface_of_tier "$tier")"
  [ -n "$want" ] || die "tier '$tier' has no interface configured in node.env"
  # Only one tier carries cluster traffic at a time, otherwise the route the
  # activations actually took is ambiguous and the measurement is unattributable.
  #
  # Taking the other tiers down is the point -- but it will also drop the shell running
  # this command if you are logged in over one of them, leaving the node unreachable
  # until someone attaches a keyboard to it. Detect that case and refuse, rather than
  # discovering it the hard way in a lab you have to walk to.
  # This branch requires root, and the only way anyone runs it is `sudo`. sudo's default
  # env_reset does NOT keep SSH_CONNECTION, so gating the check on that variable made it
  # dead code in exactly the situation it exists for. Resolve the session address from
  # SSH_CONNECTION when it survived (env_keep, or a root login shell), and otherwise ask
  # the kernel directly -- root can always read the socket table, and iproute2 is already
  # a dependency of this script.
  my_addr=""
  if [ -n "${SSH_CONNECTION:-}" ]; then
    # "<client ip> <client port> <server ip> <server port>"
    my_addr="$(printf '%s\n' "$SSH_CONNECTION" | awk '{print $3}')"
  else
    my_addr="$(ss -tnH state established '( sport = :22 )' 2>/dev/null \
               | awk '{print $3}' | sed 's/:[0-9]*$//' | tr -d '[]' | head -1)"
  fi

  if [ -n "$my_addr" ]; then
    my_iface="$(ip -4 -o addr show | awk -v a="$my_addr" '$4 ~ "^"a"/" {print $2; exit}')"
    if [ -n "$my_iface" ] && [ "$my_iface" != "$want" ]; then
      if [ "${HS_FORCE_TIER:-0}" != 1 ]; then
        # `sudo env VAR=1 ...` -- env is a command, so this survives env_reset. Telling the
        # operator `sudo HS_FORCE_TIER=1 ...` would have been another thing stripped.
        die "you are connected over ${my_iface} (${my_addr}), which switching to tier
  '${tier}' would take down -- this session would hang and the node would be
  unreachable over the network.
  Reconnect over ${want} first, or run from the console. To override:
    sudo env HS_FORCE_TIER=1 $0 tier ${tier}"
      fi
      warn "HS_FORCE_TIER=1: taking down ${my_iface}, which carries this SSH session"
    fi
  elif [ "${HS_FORCE_TIER:-0}" != 1 ]; then
    # Neither source produced an address. Do not fall through silently -- the whole point
    # is that the failure mode is an unreachable node in a lab someone has to walk to.
    die "cannot determine which interface carries this session, so the lockout check
  could not be performed. Run from the console, or override with:
    sudo env HS_FORCE_TIER=1 $0 tier ${tier}"
  fi
  for cand in "$ETH_IFACE" "${ETH25_IFACE:-}" "$WLAN_IFACE"; do
    [ -n "$cand" ] || continue
    ip link show "$cand" >/dev/null 2>&1 || continue
    if [ "$cand" = "$want" ]; then ip link set "$cand" up
    else ip link set "$cand" down; fi
  done
  ok "active tier: ${tier} (${want})"
  # cluster.hosts advertises this node at ETH_ADDR. If the active tier is not the one
  # holding that address, the control node cannot reach it, whatever the link state says.
  if [ "$want" != "$ETH_IFACE" ]; then
    warn "cluster.hosts lists this node at ${ETH_ADDR} on ${ETH_IFACE}, which is now down --
  update the orchestrator's inventory or the control node will not find this worker"
  fi
  ;;
shape)
  need_root "$@"
  rate="${2:?usage: $0 shape <rate e.g. 100mbit> [delay_ms] [loss_pct]}"
  delay="${3:-0}"; loss="${4:-0}"
  dev="$ETH_IFACE"
  tc qdisc del dev "$dev" root 2>/dev/null || true
  q="tc qdisc add dev $dev root netem rate $rate"
  [ "$delay" != 0 ] && q="$q delay ${delay}ms"
  [ "$loss" != 0 ] && q="$q loss ${loss}%"
  eval "$q"
  ok "shaped ${dev}: rate=${rate} delay=${delay}ms loss=${loss}%"
  warn "netem shapes EGRESS only -- apply the mirrored qdisc on the peer for a symmetric link"
  ;;
clear)
  need_root "$@"
  for dev in "$ETH_IFACE" "${ETH25_IFACE:-}" "$WLAN_IFACE"; do
    [ -n "$dev" ] || continue
    tc qdisc del dev "$dev" root 2>/dev/null && ok "cleared shaping on $dev" || true
  done
  ;;
show)
  for dev in "$ETH_IFACE" "${ETH25_IFACE:-}" "$WLAN_IFACE"; do
    [ -n "$dev" ] || continue
    ip link show "$dev" >/dev/null 2>&1 || { warn "$dev absent"; continue; }
    state="$(cat "/sys/class/net/$dev/operstate")"
    speed="$(cat "/sys/class/net/$dev/speed" 2>/dev/null || echo '-')"
    addr="$(ip -4 -o addr show dev "$dev" | awk '{print $4}' | tr '\n' ' ')"
    printf '  %-8s %-6s %6s Mb/s  %s\n' "$dev" "$state" "$speed" "${addr:-no address}"
    # Written as a case, not `... | grep -q netem && printf ...`. Under `set -e` an
    # AND-list in final position of a loop body becomes the exit status of the whole
    # script, so the unshaped case -- the normal one -- made `05 show` exit 1 and took
    # 06_verify_node.sh down with it before it reached the peers and services checks.
    qd="$(tc qdisc show dev "$dev" 2>/dev/null | head -1)"
    case "$qd" in *netem*) printf '           %s\n' "$qd" ;; esac
  done
  ;;
*) die "usage: $0 {addr|tier|shape|clear|show}" ;;
esac
