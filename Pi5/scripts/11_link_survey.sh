#!/usr/bin/env bash
# 11 -- measure this node's links and emit the link_survey.json the placement engine reads.
#
# Why this exists
# ---------------
# ThinkPad/control/placement.py minimises  T + lambda*E  where  T = sum_i n_i*c_i + C,
# and C is the communication time across the pipeline hops. Its DeviceProfile carries
# link_mbps and rtt_ms, and ThinkPad/control/profiles.py loads them from link_survey.json.
# Until this script existed nothing produced that file, so profiles.py fell through to its
# defaults -- link_mbps=1000.0, rtt_ms=0.3 -- and every placement was computed partly from
# a datasheet guess. The module docstring is explicit that this is not allowed: "Nothing is
# taken from a datasheet."
#
# Two directions and two tiers matter here:
#   * The Pi 5 has native GbE on eth0 and a USB 3.0 2.5GbE adapter on eth1. The adapter
#     does not reach 2.5 Gbit/s in practice, which is exactly the kind of fact this
#     project is supposed to measure rather than assume.
#   * Layer-pipelined placement sends activations worker -> worker, not only control ->
#     worker, so --all surveys every peer in cluster.hosts, not just the orchestrator.
#
#   ./11_link_survey.sh                      survey the orchestrator over the active tier
#   ./11_link_survey.sh --all                survey every peer in cluster.hosts
#   ./11_link_survey.sh --peer pi4           survey one peer
#   ./11_link_survey.sh --tier 2.5gbe        bind to the 2.5GbE tier instead
#   ./11_link_survey.sh serve                run iperf3 -s so a peer can measure toward us
. "$(dirname "$0")/lib/common.sh"
load_env

PEER=""; ALL=0; TIER=""; DURATION=10; PINGS=20; OUT=""
MODE="survey"

while [ $# -gt 0 ]; do
  case "$1" in
    serve)       MODE="serve"; shift ;;
    --peer)      PEER="${2:?--peer needs a node id}"; shift 2 ;;
    --all)       ALL=1; shift ;;
    --tier)      TIER="${2:?--tier needs gbe|2.5gbe}"; shift 2 ;;
    --duration)  DURATION="${2:?--duration needs seconds}"; shift 2 ;;
    --pings)     PINGS="${2:?--pings needs a count}"; shift 2 ;;
    --out)       OUT="${2:?--out needs a path}"; shift 2 ;;
    # Print the comment banner and stop at the first line of code -- a fixed line range
    # silently starts including source once the banner is edited.
    -h|--help)   awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) die "usage: $0 [serve] [--peer NODE|--all] [--tier gbe|2.5gbe] [--duration N] [--pings N] [--out FILE]" ;;
  esac
done

need_cmd iperf3
need_cmd ping
need_cmd python3
# 06 and 10 both do this, and for the same reason: the output is filed under NODE_ID, so
# running it from the wrong shell silently attributes another node's link to this one.
assert_this_board

HOSTS="${HS_NODE_DIR}/config/cluster.hosts"
[ -f "$HOSTS" ] || die "no cluster.hosts at $HOSTS"

if [ "$MODE" = "serve" ]; then
  log "iperf3 server on :5201 -- run the survey from the peer, Ctrl-C to stop"
  exec iperf3 -s -p 5201
fi

# ---- which interface are we measuring over? --------------------------------------
case "${TIER:-}" in
  "")      IFACE="$ETH_IFACE" ; TIER="gbe" ;;
  gbe)     IFACE="$ETH_IFACE" ;;
  2.5gbe)  IFACE="${ETH25_IFACE:-}"
           [ -n "$IFACE" ] || die "no 2.5GbE interface configured (ETH25_IFACE empty in node.env)" ;;
  *) die "unknown tier '$TIER' (gbe|2.5gbe)" ;;
esac
ip link show "$IFACE" >/dev/null 2>&1 || die "interface $IFACE is not present"
BIND="$(ip -4 -o addr show dev "$IFACE" | awk '{split($4,a,"/"); print a[1]; exit}')"
[ -n "$BIND" ] || die "$IFACE has no IPv4 address -- run 05_network_setup.sh addr first"

# ---- refuse to measure a shaped link ---------------------------------------------
# 05_network_setup.sh can attach netem to synthesise a slow link for the Phase 4 study.
# Surveying while that is attached measures the shaping, not the hardware, and the result
# is indistinguishable from a real measurement once it is in the JSON. Same class of trap
# as a thermally throttled bench cell, so it gets the same treatment: refuse by default.
QDISC="$(tc qdisc show dev "$IFACE" 2>/dev/null | head -1)"
case "$QDISC" in
  *netem*)
    die "netem is attached to ${IFACE}:
    ${QDISC}
  A survey taken now measures the shaping, not the link. Clear it first:
    sudo ./scripts/05_network_setup.sh clear" ;;
esac

LINK_SPEED="$(cat "/sys/class/net/${IFACE}/speed" 2>/dev/null || echo '')"
MTU="$(cat "/sys/class/net/${IFACE}/mtu" 2>/dev/null || echo '')"
log "tier ${TIER} on ${IFACE} (${BIND}) speed=${LINK_SPEED:-?} Mb/s mtu=${MTU:-?}"

# ---- peer list -------------------------------------------------------------------
# cluster.hosts columns: node_id  host  rpc_port  role  iface  notes
peer_rows() {
  awk -v me="$NODE_ID" '
    /^[[:space:]]*#/ {next} NF<2 {next}
    $1 != me {print $1"\t"$2"\t"$4}' "$HOSTS"
}

SELECT=""
if [ "$ALL" -eq 1 ]; then
  SELECT="$(peer_rows)"
elif [ -n "$PEER" ]; then
  SELECT="$(peer_rows | awk -v p="$PEER" -F'\t' '$1==p')"
  [ -n "$SELECT" ] || die "peer '$PEER' is not in cluster.hosts"
else
  # Default: the orchestrator. Its entry is the one profiles.py reads for this node.
  SELECT="$(peer_rows | awk -F'\t' '$3=="orchestrator"')"
  [ -n "$SELECT" ] || die "no orchestrator row in cluster.hosts -- pass --peer explicitly"
fi

ORCH_ID="$(peer_rows | awk -F'\t' '$3=="orchestrator"{print $1; exit}')"
# Is the orchestrator actually in scope for this run? `--peer pi4` is a documented mode for
# measuring a pipeline hop, and it has no orchestrator row by definition. Treating that as
# a failure made link_report exit 1, which under `set -e` killed this script before it
# could say where it had written the file.
EXPECT_ORCH=0
if [ -n "$ORCH_ID" ] && printf '%s\n' "$SELECT" | awk -F'\t' -v o="$ORCH_ID" '$1==o{f=1} END{exit !f}'; then
  EXPECT_ORCH=1
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-${RESULTS_DIR}/link_survey_${NODE_ID}_${RUN_ID}.json}"
install -d "$(dirname "$OUT")"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

TEMP_START="$(soc_temp_c)"
echo '[]' > "${WORK}/measurements.json"

measure_peer() {
  local pid="$1" host="$2" role="$3"
  local ping_out="" fwd="" rev=""

  log "peer ${pid} (${host}, ${role})"

  # latency first: it is cheap, and if it fails the bandwidth test will too
  # LC_ALL=C so the summary is the English text link_report.py's regexes parse; otherwise
  # packet loss comes back unknown and the link is published as clean.
  if ping_out="$(LC_ALL=C ping -c "$PINGS" -i 0.2 -W 2 -I "$BIND" -q "$host" 2>&1)"; then
    :
  else
    warn "  ${pid}: unreachable over ${IFACE} -- skipped"
    printf '%s\n' "$ping_out" | sed 's/^/    /' >&2
    return 0
  fi

  if fwd="$(iperf3 -c "$host" -p 5201 -B "$BIND" -t "$DURATION" -J 2>"${WORK}/fwd.err")"; then
    :
  else
    warn "  ${pid}: iperf3 refused. Start a server there:  ./scripts/11_link_survey.sh serve"
    sed 's/^/    /' "${WORK}/fwd.err" >&2
    return 0
  fi
  # -R measures the other direction. Both matter: weights are pushed to this node once,
  # activations come back every token, and a USB adapter is not symmetric.
  if rev="$(iperf3 -c "$host" -p 5201 -B "$BIND" -t "$DURATION" -R -J 2>"${WORK}/rev.err")"; then
    :
  else
    warn "  ${pid}: reverse leg failed; recording the forward direction only"
    rev=""
  fi

  printf '%s' "$fwd" > "${WORK}/fwd.json"
  printf '%s' "$rev" > "${WORK}/rev.json"
  printf '%s' "$ping_out" > "${WORK}/ping.txt"

  if PEER_ID="$pid" PEER_HOST="$host" PEER_ROLE="$role" \
     IFACE="$IFACE" TIER="$TIER" LINK_SPEED="$LINK_SPEED" MTU="$MTU" \
     python3 "${HS_NODE_DIR}/bench/link_report.py" --accumulate "${WORK}/measurements.json" \
         --ping "${WORK}/ping.txt" --forward "${WORK}/fwd.json" --reverse "${WORK}/rev.json"; then
    return 0
  fi
  # This is the last command of the function, so a bare failure here would become the
  # function's exit status and kill the script under `set -e` with nothing printed --
  # the same shape as the bug that silently ended 06_verify_node.sh at its network
  # section. Fail with a sentence instead.
  die "could not record the ${pid} measurement (bench/link_report.py failed)"
}

while IFS="$(printf '\t')" read -r pid host role; do
  [ -n "$pid" ] || continue
  measure_peer "$pid" "$host" "$role"
done <<< "$SELECT"

TEMP_END="$(soc_temp_c)"

NODE_ID="$NODE_ID" ORCH_ID="${ORCH_ID:-}" RUN_ID="$RUN_ID" EXPECT_ORCH="$EXPECT_ORCH" \
TEMP_START="$TEMP_START" TEMP_END="$TEMP_END" \
QDISC_OBSERVED="${QDISC:-none}" \
python3 "${HS_NODE_DIR}/bench/link_report.py" --finish "${WORK}/measurements.json" --out "$OUT"

ok "link survey -> ${OUT}"
echo
log "this file carries THIS node's entry only. Merge it into the control node's"
log "inventory/link_survey.json -- passing it to build_devices as-is raises ProfileError"
log "for every other worker, because their entries are absent rather than unmeasured."
