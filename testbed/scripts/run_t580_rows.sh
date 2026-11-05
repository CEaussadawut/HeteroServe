#!/bin/bash
# The ThinkPad T580 characterised the way the boards are (remaining work item 2, "the T580 as the prefill node";
# user: "let's finish The ThinkPad as a prefill node", 2026-09-29). From the repo root, on the laptop:
#     bash testbed/scripts/run_t580_rows.sh                 # the characterisation rows, pp512 / tg128
#     PP=64 TG=16 bash testbed/scripts/run_t580_rows.sh     # the pooled-protocol rows
# TinyLlama at the four quantisations every board has rows for (the same bytes: each file is re-hashed here), llama-bench
# -t 4 -r 3 -oe jsonl -v -- the Pis' protocol -- from the laptop's pinned build (/opt/heteroserve/llama.cpp, b23701f,
# AVX2 FMA F16C, GGML_NATIVE=OFF, GGML_BLAS=OFF).
# The log is run_quant_campaign.sh's Pi format (<run>/t580.log), so build_dataset.py reads it as node t580. The laptop
# has no vcgencmd: STATE lines carry the Intel thermal-throttle event counters instead (cpu0 core/package), and the
# dataset flags a row "throttled" when either rose during it. Also recorded: package temperature, governor/EPP, turbo,
# power profile, AC and load -- none of them changed here (they are the user's system settings).
# House rules: raw output tee'd, nothing filtered; >= 3 repetitions with every sample in the jsonl. The laptop is also the
# control node of this session: nothing else heavy runs while the rows are taken.
set -u
cd "$(dirname "$0")/../.."
PP=${PP:-512}; TG=${TG:-128}; REPS=${REPS:-3}; COLD=${COLD:-65}
BIN=/opt/heteroserve/llama.cpp/build/bin
M=/opt/heteroserve/models
MODELS=${MODELS:-"tinyllama-1.1b-q4km.gguf tinyllama-1.1b-q8_0.gguf tinyllama-1.1b-q4_0.gguf tinyllama-1.1b-q3km.gguf"}
declare -A SHA=([tinyllama-1.1b-q4km.gguf]=9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0
                [tinyllama-1.1b-q8_0.gguf]=a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59
                [tinyllama-1.1b-q4_0.gguf]=da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556
                [tinyllama-1.1b-q3km.gguf]=ec461c4d2b60896152bca3978aa49cd70edad7298715e4222e12af1eefea2125
                [tinyllama-1.1b-q6_k.gguf]=9fb2be7c8bbc0ca358ae88cef6d9e6844b81929eaffa5016a543e6f56b22cb6d)   # 2026-10-11
OUT=${OUT:-testbed/results/quant_campaign_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
LOG=$OUT/t580.log
exec > >(tee -a "$LOG") 2>&1
T=/sys/devices/system/cpu
pkg() { for z in /sys/class/thermal/thermal_zone*; do [ "$(cat "$z/type")" = x86_pkg_temp ] && echo $(( $(cat "$z/temp") / 1000 )); done; }
state() {
  echo "$1 $(date -Is) temp=$(pkg)'C throttle_events=$(cat $T/cpu0/thermal_throttle/core_throttle_count)/$(cat $T/cpu0/thermal_throttle/package_throttle_count)" \
       "gov=$(cat $T/cpu0/cpufreq/scaling_governor)/$(cat $T/cpu0/cpufreq/energy_performance_preference)" \
       "turbo=$([ "$(cat $T/intel_pstate/no_turbo)" = 0 ] && echo on || echo off) profile=$(powerprofilesctl get 2>/dev/null || echo '?')" \
       "ac=$(cat /sys/class/power_supply/AC*/online 2>/dev/null) load=$(cut -d' ' -f1 /proc/loadavg)"
}
echo "########## $(date -Is) t580 rows: pp$PP / tg$TG, -t 4 -r $REPS, build $(git -C /opt/heteroserve/llama.cpp rev-parse HEAD)"
echo "cpu    $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) logical CPUs"
for m in $MODELS; do
  got=$(sha256sum "$M/$m" | cut -d' ' -f1)
  [ "$got" = "${SHA[$m]:-none}" ] || { echo "FATAL: $m sha256 $got, pinned ${SHA[$m]:-none}"; exit 2; }
  w=0; while [ "$(pkg)" -ge "$COLD" ] && [ $w -lt 600 ]; do sleep 10; w=$((w + 10)); done
  echo "##### t580 $m  cold gate: waited ${w}s, now $(pkg) C (threshold $COLD C)"
  state STATE_BEFORE
  "$BIN/llama-bench" -m "$M/$m" -t 4 -p "$PP" -n "$TG" -r "$REPS" -oe jsonl -v
  echo "EXIT_t580_$m=$?"
  state STATE_AFTER
done
# The Intel counters register a throttle a moment after the load that caused it, so the last row's window needs a
# reading after the load has gone: build_dataset.py takes a row's window up to the next STATE_BEFORE, or this.
sleep 30; state STATE_END
echo "log: $LOG"
