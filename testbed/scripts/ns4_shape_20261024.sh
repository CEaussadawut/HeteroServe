#!/bin/bash
# The four-node network study's link (register_ns4_20261024.py, 2026-10-24): the Pi 4's RPC link with the laptop, shaped once
# the laptop's llama-server is healthy (so the model's load is never shaped), and cleared. From the repo root:
#     bash testbed/scripts/ns4_shape_20261024.sh when-served RATE_MBIT DELAY_MS LOG   # wait for /health "ok", apply, log
#     bash testbed/scripts/ns4_shape_20261024.sh clear                              # netshape.sh clear, then its check
# netshape.sh (unchanged) shapes from the Pi 4 the traffic to and from the laptop's address on the Pi 4's RPC port 50052
# only (NETSHAPE_PEER, NETSHAPE_PORT), so ssh and the HTTP stream stay unshaped. RATE_MBIT 0 = no rate limit; DELAY_MS one
# way per direction (the pair's round trip gains 2 x DELAY_MS). when-served appends one JSON line to LOG with the epoch the
# shaping was in place, which the study's scorer compares with the first scored request; it keeps its pid in LOG.pid while
# it waits.
set -u
cd "$(dirname "$0")/../.."
export NETSHAPE_PEER=192.168.1.108 NETSHAPE_PORT=50052
case "${1:-}" in
  when-served)
    rate=$2; delay=$3; log=$4
    echo $$ > "$log.pid"; t0=$(date +%s)
    until curl -s -m 5 http://192.168.1.108:8080/health | grep -q '"ok"'; do
      if [ $(( $(date +%s) - t0 )) -gt 1800 ]; then
        echo "{\"ok\": false, \"why\": \"no healthy server in 1800 s\"}" >> "$log"; rm -f "$log.pid"; exit 1
      fi
      sleep 1
    done
    if bash testbed/scripts/netshape.sh apply "$rate" "$delay" >> "$log.apply.out" 2>&1; then
      echo "{\"ok\": true, \"rate_mbit\": $rate, \"delay_ms\": $delay, \"applied_epoch\": $(date +%s.%N)}" >> "$log"
      rm -f "$log.pid"; exit 0
    fi
    echo "{\"ok\": false, \"why\": \"netshape.sh apply failed\"}" >> "$log"; rm -f "$log.pid"; exit 1 ;;
  clear)
    bash testbed/scripts/netshape.sh clear && bash testbed/scripts/netshape.sh check ;;
  *)
    echo "usage: $0 when-served RATE_MBIT DELAY_MS LOG | clear"; exit 2 ;;
esac
