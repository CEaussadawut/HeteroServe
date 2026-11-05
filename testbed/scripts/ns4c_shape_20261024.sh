#!/bin/bash
# The four-node network study's link, third registration (register_ns4c_20261024.py, 2026-10-24): ns4_shape_20261024.sh with
# the laptop's OVERLAY address as the peer (192.168.10.5: serve_pool.sh reaches the Pi 4 at 192.168.10.12, so that is the
# flow; ns4b's filter on 192.168.1.108 matched nothing), and a counters step that records the netem qdiscs' own packet and
# byte counts before each clear, so the study can show its RPC traffic was shaped. ns4_shape_20261024.sh's notes follow.
#
# The four-node network study's link (register_ns4_20261024.py, 2026-10-24): the Pi 4's RPC link with the laptop, shaped once
# the laptop's llama-server is healthy (so the model's load is never shaped), and cleared. From the repo root:
#     bash testbed/scripts/ns4c_shape_20261024.sh when-served RATE_MBIT DELAY_MS LOG  # wait for /health "ok", apply, log
#     bash testbed/scripts/ns4c_shape_20261024.sh counters LOG                        # the netem counters, one JSON line
#     bash testbed/scripts/ns4c_shape_20261024.sh clear                               # netshape.sh clear, then its check
# netshape.sh (unchanged) shapes from the Pi 4 the traffic to and from the laptop's address on the Pi 4's RPC port 50052
# only (NETSHAPE_PEER, NETSHAPE_PORT), so ssh and the HTTP stream stay unshaped. RATE_MBIT 0 = no rate limit; DELAY_MS one
# way per direction (the pair's round trip gains 2 x DELAY_MS). when-served appends one JSON line to LOG with the epoch the
# shaping was in place, which the study's scorer compares with the first scored request; it keeps its pid in LOG.pid while
# it waits.
set -u
cd "$(dirname "$0")/../.."
export NETSHAPE_PEER=192.168.10.5 NETSHAPE_PORT=50052
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
  counters)
    log=$2
    raw=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o LogLevel=ERROR pi4 \
          'sudo -n /usr/sbin/tc -s qdisc show dev eth0; echo "== ifb0"; sudo -n /usr/sbin/tc -s qdisc show dev ifb0 2>/dev/null' 2>&1)
    python3 - "$log" "$raw" <<'PY'
import json, re, sys, time
log, raw = sys.argv[1], sys.argv[2]
eth, _, ifb = raw.partition("== ifb0")
def netem(txt):
    m = re.search(r"qdisc netem .*?Sent (\d+) bytes (\d+) pkt", txt, re.S)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)
eb, ep = netem(eth); ib, ip = netem(ifb)
open(log, "a").write(json.dumps({"epoch": time.time(), "egress_bytes": eb, "egress_pkts": ep, "ingress_bytes": ib,
                                 "ingress_pkts": ip, "raw": raw[-1500:]}) + "\n")
print("counters: egress %s pkts %s B, ingress %s pkts %s B" % (ep, eb, ip, ib))
PY
    ;;
  clear)
    bash testbed/scripts/netshape.sh clear && bash testbed/scripts/netshape.sh check ;;
  *)
    echo "usage: $0 when-served RATE_MBIT DELAY_MS LOG | counters LOG | clear"; exit 2 ;;
esac
