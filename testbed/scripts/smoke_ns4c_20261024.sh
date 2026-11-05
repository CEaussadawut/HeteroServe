#!/bin/bash
# Smoke test for the network study's third registration (register_ns4c_20261024.py), run BEFORE it is registered: does the
# shaping now reach the laptop <-> Pi 4 RPC flow? (ns4b's filter matched nothing: the flow is on the overlay addresses.)
# Not a measurement for any reported figure; it decides whether ns4c is worth registering. From the repo root:
#     bash testbed/scripts/smoke_ns4c_20261024.sh            # exit 0: the shaping reaches the flow; 1: it does not; 2: refused
# TinyLlama Q4_0 on laptop 11 + Pi 4 11 with the head LAST on the Pi 4, so every token's 128 KB of logits crosses the link:
# one greedy /completion unshaped, then one with the link shaped to 5 Mbit/s (ns4c_shape_20261024.sh: peer 192.168.10.5,
# port 50052), the netem counters read, the shaping cleared, the server stopped. PASS when the shaped decode is at least
# 100 ms per token slower (128 KB at 5 Mbit/s is ~210 ms) and netem passed at least 100 packets each way.
set -u
cd "$(dirname "$0")/../.."
OUT=testbed/results/smoke_ns4c_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
exec > >(tee -i -a "$OUT/smoke.log") 2>&1
echo "########## $(date -Is) smoke ns4c -> $OUT"
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
  bash testbed/scripts/serve_pool.sh start || { echo "FAIL: serve_pool.sh start"; exit 1; }
ask() {   # one greedy completion of exactly 24 tokens; prints the server's decode ms per token, or fails
  # ignore_eos: without it TinyLlama ends this prompt at its first token, and the server reports 0.00 ms per token (09:53 run)
  curl -s -m 600 http://192.168.1.108:8080/completion -H 'Content-Type: application/json' \
    -d '{"prompt": "The quick brown fox jumps over the lazy dog. Summarise this sentence in five words.", "n_predict": 24,
         "temperature": 0, "cache_prompt": false, "ignore_eos": true}' \
    | python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; n=t["predicted_n"]; n == 24 or sys.exit("decoded %d tokens, not 24" % n); print("%.2f" % t["predicted_per_token_ms"])'
}
ask >/dev/null || { echo "FAIL: the warm-up completion did not decode 24 tokens (instrument)"; exit 1; }
A=$(ask) || { echo "FAIL: the unshaped completion did not decode 24 tokens (instrument)"; exit 1; }
echo "unshaped decode: $A ms/token"
$SH when-served 5 0 "$OUT/shape.log" || { echo "FAIL: shaping not applied"; exit 1; }
B=$(ask) || { echo "FAIL: the shaped completion did not decode 24 tokens (instrument)"; exit 1; }
echo "shaped (5 Mbit/s) decode: $B ms/token"
$SH counters "$OUT/counters.log"
$SH clear
read -r EP IP < <(python3 -c 'import json,sys; c=json.loads(open(sys.argv[1]).read().splitlines()[-1]); print(c["egress_pkts"] or 0, c["ingress_pkts"] or 0)' "$OUT/counters.log")
echo "netem packets: egress $EP, ingress $IP"
if python3 -c "import sys; a, b, e, i = map(float, sys.argv[1:]); sys.exit(0 if b - a >= 100 and e >= 100 and i >= 100 else 1)" "$A" "$B" "$EP" "$IP"; then
  echo "PASS: shaping reaches the RPC flow (decode +$(python3 -c "print('%.1f' % ($B - $A))") ms/token at 5 Mbit/s)"; exit 0
fi
echo "FAIL: shaping does not reach the RPC flow"; exit 1
