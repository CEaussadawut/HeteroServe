#!/bin/bash
# Device characterisation, 2026-09-27: TinyLlama at three quantisations (Q4_K_M, Q8_0, Q4_0) on the
# Pi 5, Pi 4 and Jetson, plus the Jetson's CPU-only Q4_K_M row re-taken from cold (the 2026-09-25 one
# started warm and is provisional). Run from the repo root on the laptop:
#     bash testbed/scripts/run_quant_campaign.sh
# Later campaigns reuse it through the environment; unset, every override below reproduces the 09-27 run exactly:
#     MODELS="<files on the boards>"  NODES="pi5 pi4 jetson"  JETSON_STEPS='<step lines, one per line>'
# e.g. the Q3_K_M rows (2026-09-28): MODELS=tinyllama-1.1b-q3km.gguf NODES="pi5 pi4", then the Jetson alone with
# NODES=jetson JETSON_STEPS='step tinyllama-q3km MODEL=$HOME/models/tinyllama-1.1b-q3km.gguf "CASES=vulkan cpu"'.
# The boards have no internet (the router has no WAN), so the models come from the laptop's cache,
# fetched from TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF at revision 52e7645b and checked against the
# SHA-256 Hugging Face publishes for each file. Every copy is re-hashed on the board before use.
# House rules: -r 3 (Pis) / -r 5 (Jetson protocol) with every sample in jsonl; raw output tee'd;
# get_throttled and temperatures logged around every Pi run; the Jetson under its power guard.
set -uo pipefail
cd "$(dirname "$0")/../.."
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=testbed/results/quant_campaign_$STAMP
mkdir -p "$OUT/jetson"
exec > >(tee -i -a "$OUT/campaign.log") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
CACHE=$HOME/.cache/heteroserve-models
PIN=$(python3 -c "import json;print(json.load(open('ThinkPad/config/cluster.json'))['llama']['pin_sha'])")
declare -A SRC=([tinyllama-1.1b-q8_0.gguf]=tinyllama-1.1b-chat-v1.0.Q8_0.gguf
                [tinyllama-1.1b-q4_0.gguf]=tinyllama-1.1b-chat-v1.0.Q4_0.gguf
                [tinyllama-1.1b-q3km.gguf]=tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf
                [tinyllama-1.1b-q6_k.gguf]=tinyllama-1.1b-chat-v1.0.Q6_K.gguf)   # 2026-10-11: the Q6_K rows (the 30B MoE at Q6_K)
declare -A SHA=([tinyllama-1.1b-q4km.gguf]=9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0
                [tinyllama-1.1b-q8_0.gguf]=a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59
                [tinyllama-1.1b-q4_0.gguf]=da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556
                [tinyllama-1.1b-q3km.gguf]=ec461c4d2b60896152bca3978aa49cd70edad7298715e4222e12af1eefea2125
                [qwen3.5-9b-q4_k_m.gguf]=d784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43   # 2026-09-29: the hybrid rows
                [tinyllama-1.1b-q6_k.gguf]=9fb2be7c8bbc0ca358ae88cef6d9e6844b81929eaffa5016a543e6f56b22cb6d)   # 2026-10-11, TheBloke @ 52e7645b
MODELS=${MODELS:-"tinyllama-1.1b-q4km.gguf tinyllama-1.1b-q8_0.gguf tinyllama-1.1b-q4_0.gguf"}
NODES=${NODES:-"pi5 pi4 jetson"}
DEFAULT_STEPS='step tinyllama-q4km-cold MODEL=$HOME/models/tinyllama-1.1b-q4km.gguf CASES=cpu
step tinyllama-q8_0      MODEL=$HOME/models/tinyllama-1.1b-q8_0.gguf "CASES=vulkan cpu"
step tinyllama-q4_0      MODEL=$HOME/models/tinyllama-1.1b-q4_0.gguf "CASES=vulkan cpu"'
JETSON_STEPS=${JETSON_STEPS:-$DEFAULT_STEPS}
has() { [[ " $NODES " == *" $1 "* ]]; }
for m in $MODELS; do [ -n "${SHA[$m]:-}" ] || { echo "FATAL: no pinned sha256 for $m"; exit 2; }; done
fatal() { echo "FATAL: $*"; echo "CAMPAIGN FAILED"; exit 2; }

echo "########## $(date -Is) quant campaign -> $OUT   pin $PIN   nodes [$NODES]   models [$MODELS]"

# 1. Models: copy what is missing, then verify every file on every board (never trust a copy).
for node in $NODES; do
  for dst in $MODELS; do
    [ -n "${SRC[$dst]:-}" ] || continue
    if ! $SSH $node "echo '${SHA[$dst]}  models/$dst' | sha256sum -c --quiet" >/dev/null 2>&1; then
      scp -q "$CACHE/${SRC[$dst]}" "$node:models/.$dst.part" || fatal "scp $dst -> $node"
      $SSH $node "mv models/.$dst.part models/$dst" || fatal "mv $dst on $node"
    fi
  done
  for m in $MODELS; do
    got=$($SSH $node "sha256sum models/$m | cut -d' ' -f1")
    [ "$got" = "${SHA[$m]}" ] || fatal "$node $m sha256 $got != ${SHA[$m]}"
    echo "model   $node $m sha256 OK"
  done
done

# 2. Every tree at the pin and clean -- a figure from another revision is not a characterisation row.
for node in $NODES; do
  head=$($SSH $node 'git -C ~/llama.cpp rev-parse HEAD'); dirty=$($SSH $node 'git -C ~/llama.cpp status --porcelain --untracked-files=no | wc -l')
  [ "$head" = "$PIN" ] && [ "$dirty" = 0 ] || fatal "$node tree $head dirty=$dirty (pin $PIN)"
  echo "tree    $node at pin, clean"
done

# 3. Jetson: guard live, board idle; install the bench script atomically; start its chain.
if has jetson; then
g=$($SSH jetson "pgrep -f '[j]etson_power_guard' | wc -l")
[ "$g" -ge 1 ] || fatal "Jetson power guard is not running (start it first; it is not a service)"
$SSH jetson 'pgrep -x llama-bench >/dev/null' && fatal "llama-bench already running on the Jetson"
TRIPS0=$($SSH jetson 'grep -c TRIP ~/heteroserve/logs/power-guard.log 2>/dev/null; true' | head -1)
scp -q testbed/scripts/jetson_bench_tinyllama.sh jetson:heteroserve/tools/.jetson_bench_tinyllama.sh.new || fatal "push bench script"
$SSH jetson 'mv ~/heteroserve/tools/.jetson_bench_tinyllama.sh.new ~/heteroserve/tools/jetson_bench_tinyllama.sh'
cat > "$OUT/jetson_chain.sh" <<'CHAIN'
#!/bin/bash
# Runs ON the Jetson. Stops between steps if the guard tripped or the power mode left MAXN.
T=$HOME/heteroserve/tools/jetson_bench_tinyllama.sh; R=$HOME/heteroserve/results; L=$R/quant_chain.log
t0=$(grep -c TRIP $HOME/heteroserve/logs/power-guard.log 2>/dev/null; true)
ok() { local t; t=$(grep -c TRIP $HOME/heteroserve/logs/power-guard.log 2>/dev/null; true)
       [ "$t" = "$t0" ] && [ "$(cut -d' ' -f1 /var/lib/nvpmodel/status)" = pmode:0000 ]; }
step() { local tag=$1; shift; ok || { echo "STOPPED before $tag: guard trip or pmode change" >> $L; return 1; }
         env "$@" TAG=$tag bash $T; echo "$tag rc=$? $(date -Is)" >> $L; }
rm -f $R/quant_chain.done; : > $L
CHAIN
printf '%s\n' "$JETSON_STEPS" | sed '$!s/$/ \&\&/' >> "$OUT/jetson_chain.sh"
echo 'echo DONE > $R/quant_chain.done' >> "$OUT/jetson_chain.sh"
scp -q "$OUT/jetson_chain.sh" jetson:heteroserve/tools/quant_chain.sh || fatal "push chain"
$SSH jetson 'setsid nohup bash ~/heteroserve/tools/quant_chain.sh </dev/null >/dev/null 2>&1 &'
echo "jetson  chain started $(date -Is) (guard trips before: $TRIPS0)"
fi

# 4. Pis, in parallel with each other and with the Jetson. Cold gate: wait (max 10 min) for the
#    SoC to fall below a per-board threshold near its idle temperature; log what was reached.
# PP / TG (2026-09-28): the prompt and generation lengths, default 512 / 128. PP=64 TG=16 takes the rows at the
# pooled protocol instead (run_pool.sh), so the predictor can price a cell from rows measured the way it runs.
PP=${PP:-512}; TG=${TG:-128}
declare -A COLD=([pi5]=50 [pi4]=57)
bench_pi() {
  local node=$1 m waited t
  for m in $MODELS; do
    [ "$($SSH $node 'for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p; done | wc -l')" = 0 ] \
      || { echo "SKIP $node $m: board busy"; continue; }
    waited=0
    while t=$($SSH $node "vcgencmd measure_temp | tr -dc '0-9.' | cut -d. -f1"); [ "${t:-99}" -ge "${COLD[$node]}" ] && [ $waited -lt 600 ]; do
      sleep 15; waited=$((waited + 15)); done
    echo "##### $node $m  cold gate: waited ${waited}s, now ${t} C (threshold ${COLD[$node]} C)"
    $SSH $node "echo \"STATE_BEFORE \$(date -Is) \$(vcgencmd measure_temp) \$(vcgencmd get_throttled) gov=\$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) memavail_mib=\$(awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo) load=\$(cut -d' ' -f1 /proc/loadavg)\""
    $SSH $node "cd ~/llama.cpp && ./build/bin/llama-bench -m ~/models/$m -t 4 -p $PP -n $TG -r 3 -oe jsonl" 2>&1
    echo "EXIT_${node}_${m}=$?"
    $SSH $node "echo \"STATE_AFTER \$(date -Is) \$(vcgencmd measure_temp) \$(vcgencmd get_throttled)\""
  done
}
declare -A PID=()
for n in pi5 pi4; do has $n || continue; bench_pi $n > "$OUT/$n.log" 2>&1 & PID[$n]=$!; done
for n in pi5 pi4; do has $n || continue; wait ${PID[$n]}
  echo "$n     done $(date -Is) ($(grep -c '^EXIT_' "$OUT/$n.log") runs, $(grep -cE '^EXIT_.*=0$' "$OUT/$n.log") exit 0)"; done

# 5. Wait for the Jetson chain (max 2.5 h), then collect its result dirs and the guard log.
if has jetson; then
TAGS=$(printf '%s\n' "$JETSON_STEPS" | awk '$1 == "step" {print $2}')
for i in $(seq 1 150); do
  $SSH jetson 'test -f ~/heteroserve/results/quant_chain.done' && break; sleep 60
done
$SSH jetson 'cat ~/heteroserve/results/quant_chain.log'
for d in $($SSH jetson "ls -d $(for g in $TAGS; do printf '~/heteroserve/results/jetson_%s_* ' $g; done) 2>/dev/null"); do
  scp -q -r "jetson:$d" "$OUT/jetson/" && echo "collect $(basename $d)"
done
scp -q jetson:heteroserve/logs/power-guard.log "$OUT/jetson/power-guard.log"
echo "guard   trips now: $(grep -c TRIP "$OUT/jetson/power-guard.log"; true) (before: $TRIPS0)"
for d in "$OUT"/jetson/jetson_tinyllama-*; do
  python3 -B testbed/scripts/summarise_jetson_bench.py "$d" "$OUT/jetson/power-guard.log" > "$d/summary.txt" 2>&1 \
    && echo "summary $(basename $d): $(grep -E '^(vulkan|cpu) +(pp|tg)' "$d/summary.txt" | awk '{printf "%s %s %s | ", $1,$2,$3}')"
done
fi
echo "CAMPAIGN DONE $(date -Is)"
