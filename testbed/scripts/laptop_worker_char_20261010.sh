#!/bin/bash
# The laptop (cluster.json t580) as an RPC WORKER, characterised for planning (2026-10-10 night; the user: "build the
# measured context-length cost into the planner (registered), then try a long-context plan where the laptop holds some
# layers"). Read by register_longctx_20261011.py; nothing here is scored.
# Why a worker row: at the pin the x86 CPU backend repacks Q4_K for AVX2 (ggml-cpu/repack.cpp, q4_K_8x8_q8_K) when the
# laptop runs llama.cpp itself; an RPC worker allocates the default CPU buffer type and never repacks. So the laptop's
# host rows (characterisation.csv, t580 cpu) do not price it as a worker.
# bench_worker_role.sh's protocol, run locally: ggml-rpc-server bound to LOOPBACK (127.0.0.1:50053, never the network),
# readiness via ss, llama-bench -rpc 127.0.0.1:50053 -ngl 99 -t 4 (every layer and the head on the worker), -oe jsonl, raw
# output tee'd. Three invocations of each:
#   1. TinyLlama Q4_K_M, -p 512 -n 128 -r 3: the worker row the predictor prices Q4_K_M from (costs(): the TinyLlama row
#      at the target quantisation);
#   2. Qwen3-0.6B Q8_0, -p 0 -n 32 -d 0,1024,2048,4096 -r 3: the attention cost at 2 query heads per KV head (as the
#      Pi 4 and Jetson cells of kv_depth_20261010);
#   3. Qwen3-14B Q4_K_M, -p 0 -n 16 -d 1024,0 -r 3: the attention cost at 5 query heads per KV head, both depths decoded
#      hot (kv14b's lesson).
# Every model's sha256 is checked first (testbed/models.md). From the repo root, as a systemd user unit, with the laptop
# otherwise quiet (the 16:59 characterisation found this session's own activity on the laptop moving its timings).
set -uo pipefail
cd "$(dirname "$0")/../.."
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=testbed/results/laptop_worker_$STAMP
mkdir -p "$OUT"
exec > >(tee -i -a "$OUT/run.log") 2>&1
BIN=/opt/heteroserve/llama.cpp/build/bin
PIN=b23701f77d47dad9de834d59ebfcbe25c9e8b46f
PORT=50053
SRV_ARGS="-H 127.0.0.1 -p $PORT -t 4"
TINY=/home/aun/.cache/heteroserve-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
Q06=/home/aun/.cache/heteroserve-models/Qwen3-0.6B-Q8_0.gguf
Q14=/home/aun/.cache/heteroserve-models/Qwen3-14B-Q4_K_M.gguf
declare -A SHA=([$TINY]=9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0
                [$Q06]=9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
                [$Q14]=500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0)
srv_pid() { for p in $(pgrep -x ggml-rpc-server); do tr '\0' ' ' < /proc/$p/cmdline | grep -qF -- "$SRV_ARGS" && echo $p; done; }
cleanup() { p=$(srv_pid); [ -n "$p" ] && kill $p && echo "worker stopped (pid $p)"; echo "########## $(date -Is) exit"; }
trap cleanup EXIT
echo "########## $(date -Is) laptop worker characterisation -> $OUT"
[ -f ~/.cache/heteroserve-wf/serve_pool.state ] && { echo "FATAL: serve_pool.sh state exists"; exit 1; }
[ "$(git -C /opt/heteroserve/llama.cpp rev-parse HEAD) $(git -C /opt/heteroserve/llama.cpp status --porcelain --untracked-files=no | wc -l)" = "$PIN 0" ] \
  || { echo "FATAL: the laptop's llama.cpp is not the pin clean"; exit 1; }
[ -z "$(srv_pid)" ] && ! ss -ltn | grep -q "127.0.0.1:$PORT " || { echo "FATAL: port $PORT is taken"; exit 1; }
for m in "$TINY" "$Q06" "$Q14"; do
  s=$(sha256sum "$m" | cut -d' ' -f1); [ "$s" = "${SHA[$m]}" ] || { echo "FATAL: $m sha256 $s"; exit 1; }
  echo "sha256 OK $(basename $m)"
done
setsid nohup $BIN/ggml-rpc-server $SRV_ARGS > "$OUT/rpc-server-loopback.log" 2>&1 < /dev/null &
for i in $(seq 1 20); do ss -ltn | grep -q "127.0.0.1:$PORT " && break; sleep 1; done
ss -ltn | grep -q "127.0.0.1:$PORT " || { echo "FATAL: the loopback worker never listened"; exit 1; }
echo "worker listening on 127.0.0.1:$PORT (pid $(srv_pid))"
state() { echo "STATE $1 $(date -Is) load=$(cut -d' ' -f1 /proc/loadavg) memavail_mib=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) temp=$(sensors 2>/dev/null | awk '/Package id 0/{print $4; exit}') mhz=$(awk '/MHz/{s+=$4;n++} END{printf "%.0f", s/n}' /proc/cpuinfo)"; }
bench() {  # tag model args
  local tag=$1 m=$2; shift 2
  for inv in 1 2 3; do
    state "before $tag.$inv"
    echo "##### $tag invocation $inv: llama-bench -m $(basename $m) -rpc 127.0.0.1:$PORT -ngl 99 -t 4 $*"
    timeout 3600 $BIN/llama-bench -m "$m" -rpc 127.0.0.1:$PORT -ngl 99 -t 4 "$@" -oe jsonl 2>&1 | tee "$OUT/$tag.$inv.log" | grep -E '^\{' | cut -c1-80 > /dev/null
    echo "EXIT_${tag}_$inv=${PIPESTATUS[0]}"
    grep -h '^{' "$OUT/$tag.$inv.log" | python3 -c "
import sys, json
for l in sys.stdin:
    r = json.loads(l)
    print('ROW %s inv $inv: p%d n%d d%d %.2f ms/token' % ('$tag', r['n_prompt'], r['n_gen'], r['n_depth'], 1000.0 / r['avg_ts']))"
    state "after $tag.$inv"
  done
}
bench tiny_q4km "$TINY" -p 512 -n 128 -r 3
bench q06_depth "$Q06" -p 0 -n 32 -d 0,1024,2048,4096 -r 3
bench q14_depth "$Q14" -p 0 -n 16 -d 1024,0 -r 3
echo "LAPTOP WORKER DONE $(date -Is) $OUT"
