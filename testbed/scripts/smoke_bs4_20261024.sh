#!/bin/bash
# Smoke test for the head-board sampling study (register_bs4_20261024.py), run BEFORE it is registered: does
# SERVE_EXTRA="--backend-sampling" reach llama-server, and does it stop the logits crossing the shaped link? Not a measurement
# for any reported figure. From the repo root, once apply_serve_extra_20261024.py has run:
#     bash testbed/scripts/smoke_bs4_20261024.sh      # exit 0: it does; 1: it does not; 2: refused
# TinyLlama Q4_0 on laptop 11 + Pi 4 11, the head LAST on the Pi 4, with backend sampling: one greedy /completion unshaped,
# then one with the Pi 4's link shaped to 5 Mbit/s (ns4c_shape_20261024.sh). Without backend sampling 5 Mbit/s costs the
# head-last plan ~205 ms a token (128 KB of logits; predicted 325 ms, ns4c measures it); with it, ~120. PASS when the shaped
# decode is below 200 ms per token, netem passed at least 100 packets each way, and serve_pool.sh's command line carries the flag.
set -u -o pipefail   # serve_pool.sh start is piped through tee below: its failure must not read as tee's success
cd "$(dirname "$0")/../.."
OUT=testbed/results/smoke_bs4_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
exec > >(tee -i -a "$OUT/smoke.log") 2>&1
echo "########## $(date -Is) smoke bs4 -> $OUT"
grep -q "SERVE_EXTRA" testbed/scripts/serve_pool.sh || { echo "REFUSED: serve_pool.sh has no SERVE_EXTRA"; exit 2; }
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "REFUSED: serve_pool.sh state exists"; exit 2; }
bash testbed/scripts/netshape.sh check >/dev/null 2>&1 || { echo "REFUSED: shaping is installed on the Pi 4"; exit 2; }
M=$(ls -t testbed/dataset/memory_snapshot_*.json | head -1)
SH="bash testbed/scripts/ns4c_shape_20261024.sh"
stop_all() {
  [ -f "$OUT/shape.log.pid" ] && kill "$(cat "$OUT/shape.log.pid")" 2>/dev/null
  $SH clear >/dev/null 2>&1
  MODEL=tinyllama_q4_0 CTX=2048 bash testbed/scripts/serve_pool.sh stop >/dev/null 2>&1
}
trap stop_all EXIT
MODEL=tinyllama_q4_0 CTX=2048 CLOCKS=locked MEMFILE="$M" NODES="laptop pi4" PLACE=laptop:11,pi4:11 PLACE_HEAD=last SERVE_DRAFT=none \
  SERVE_EXTRA="--backend-sampling" bash testbed/scripts/serve_pool.sh start | tee "$OUT/serve.out" || { echo "FAIL: serve_pool.sh start"; exit 1; }
grep -q -- "--backend-sampling" "$OUT/serve.out" || { echo "FAIL: the server's command line has no --backend-sampling"; exit 1; }
ask() {   # one greedy completion of exactly 24 tokens; prints the server's decode ms per token, or fails
  # ignore_eos: without it TinyLlama ends this prompt at its first token, and the server reports 0.00 ms per token (ns4c smoke, 09:53)
  curl -s -m 600 http://192.168.1.108:8080/completion -H 'Content-Type: application/json' \
    -d '{"prompt": "The quick brown fox jumps over the lazy dog. Summarise this sentence in five words.", "n_predict": 24,
         "temperature": 0, "top_k": 1, "cache_prompt": false, "ignore_eos": true}' \
    | python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; n=t["predicted_n"]; n == 24 or sys.exit("decoded %d tokens, not 24" % n); print("%.2f" % t["predicted_per_token_ms"])'
}
ask >/dev/null || { echo "FAIL: the warm-up completion did not decode 24 tokens (instrument)"; exit 1; }
A=$(ask) || { echo "FAIL: the unshaped completion did not decode 24 tokens (instrument)"; exit 1; }
echo "unshaped decode, backend sampling: $A ms/token"
$SH when-served 5 0 "$OUT/shape.log" || { echo "FAIL: shaping not applied"; exit 1; }
B=$(ask) || { echo "FAIL: the shaped completion did not decode 24 tokens (instrument)"; exit 1; }
echo "shaped (5 Mbit/s) decode, backend sampling: $B ms/token"
$SH counters "$OUT/counters.log"
$SH clear
read -r EP IP < <(python3 -c 'import json,sys; c=json.loads(open(sys.argv[1]).read().splitlines()[-1]); print(c["egress_pkts"] or 0, c["ingress_pkts"] or 0)' "$OUT/counters.log")
echo "netem packets: egress $EP, ingress $IP"
if python3 -c "import sys; b, e, i = map(float, sys.argv[1:]); sys.exit(0 if b < 200 and e >= 100 and i >= 100 else 1)" "$B" "$EP" "$IP"; then
  echo "PASS: backend sampling keeps the logits off the shaped link ($B ms/token at 5 Mbit/s)"; exit 0
fi
echo "FAIL: at 5 Mbit/s the decode is $B ms/token: backend sampling did not stop the logits crossing (or nothing was shaped)"; exit 1
