#!/usr/bin/env bash
# Prove this node is usable by the cluster before spending a night on a sweep.
# Run it ON THE ORCHESTRATOR (the laptop) or on the Jetson itself; it checks
# reachability, link speed, RPC handshake, and a two-node inference.
#
#   ./70_cluster_smoke.sh                 # check every host in config/cluster.hosts
#   ./70_cluster_smoke.sh jetson-nano-01
. "$(dirname "$0")/lib/common.sh"
load_env

HOSTS="${HS_JETSON_DIR}/config/cluster.hosts"
[ -f "$HOSTS" ] || die "missing ${HOSTS} — cp config/cluster.hosts.example config/cluster.hosts"

WANT="${1:-}"
FAIL=0

step "Inventory"
grep -v '^#' "$HOSTS" | grep -v '^[[:space:]]*$' | while read -r id host port role iface note; do
  [ -n "$WANT" ] && [ "$id" != "$WANT" ] && continue
  printf '  %-20s %-16s %-8s %s\n' "$id" "$host" "${port:--}" "$role"
done

while read -r id host port role iface note; do
  case "$id" in ''|'#'*) continue ;; esac
  [ -n "$WANT" ] && [ "$id" != "$WANT" ] && continue
  [ "$port" = "-" ] && continue   # orchestrator has no rpc-server

  step "$id ($host:$port)"

  if ping -c 3 -W 2 "$host" >/dev/null 2>&1; then
    RTT="$( set +o pipefail
      ping -c 10 -i 0.2 -W 2 "$host" 2>/dev/null | tail -1 | cut -d'=' -f2 || true )"
    ok "reachable, rtt min/avg/max/mdev = ${RTT}"
  else
    warn "unreachable"; FAIL=1; continue
  fi

  # The KV-cache and tensor traffic in a split run is the thing that decides
  # whether wired beats Wi-Fi, so measure the link rather than assuming it.
  if command -v iperf3 >/dev/null 2>&1; then
    if timeout 20 iperf3 -c "$host" -t 5 -f m 2>/dev/null | tail -4 | grep -q receiver; then
      BW="$( set +o pipefail
        timeout 20 iperf3 -c "$host" -t 5 -f m 2>/dev/null \
        | grep receiver | awk '{print $7, $8}' || true )"
      ok "throughput ${BW}"
      # 1 GbE should give ~940 Mbit/s. Anything near 95 means a 100M link -
      # usually a bad cable or a switch port that negotiated down.
      case "$BW" in *Mbits*) B="${BW%% *}"; [ "${B%.*}" -lt 400 ] && warn "link far below 1 GbE — check cable/switch port" ;; esac
    else
      warn "iperf3 server not running on ${host} (start it there: iperf3 -s)"
    fi
  fi

  if command -v nc >/dev/null 2>&1; then
    if nc -z -w 3 "$host" "$port" 2>/dev/null; then
      ok "rpc port ${port} open"
    else
      warn "rpc port ${port} closed — start scripts/40_run_rpc_server.sh on ${id}"
      FAIL=1; continue
    fi
  fi
done < "$HOSTS"

step "Two-node inference"
CLI="$(bin_cli || true)"
if [ -z "$CLI" ]; then
  warn "no llama-cli here; run this step from the orchestrator"
else
  MODEL="$( set +o pipefail; ls "${MODEL_DIR}"/*.gguf 2>/dev/null | head -1 || true )"
  if [ -z "$MODEL" ]; then
    warn "no model available for the smoke test"
  else
    RPCLIST="$( set +o pipefail; grep -v '^#' "$HOSTS" \
      | awk 'NF>=3 && $3!="-" {printf "%s%s:%s", (n++?",":""), $2, $3}' || true )"
    info "llama-cli --rpc ${RPCLIST}"
    # -ngl 99 so layers are actually handed to the remote backends; without it
    # the RPC servers connect and then sit idle, which looks like success.
    "$CLI" -m "$MODEL" --rpc "$RPCLIST" -ngl 99 -n 32 -p "Explain what prefill and decode are, in one sentence." \
      2>&1 | tail -20 || { warn "distributed run failed"; FAIL=1; }
  fi
fi

step "Result"
[ "$FAIL" = "0" ] && ok "cluster smoke test passed" || warn "one or more checks failed (see above)"
exit "$FAIL"
