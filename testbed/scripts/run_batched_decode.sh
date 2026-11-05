#!/bin/bash
# How much does a second (and fourth) user cost a board? llama-batched-bench on each board ALONE, the first step of
# prefill/decode separation's two-user question (remaining work item 6: "batched decode on each board at 1, 2 and 4
# sequences. Whether a second sequence costs a memory-bound board almost nothing sets how much the Pi 5 alone can already
# serve"). From the repo root, on the laptop:
#     bash testbed/scripts/run_batched_decode.sh                       # Pi 5, Pi 4 and the laptop, in parallel
#     NODES=pi5 NPL=1,2,4,8 bash testbed/scripts/run_batched_decode.sh
# Per node: Qwen3.5-9B Q4_K_M (the separation model; sha256 checked on the node), -npp $NPP prompt tokens and -ntg $NTG
# generated tokens per sequence, -npl sequences at once, -t 4, ctx $CTX, prompts NOT shared (independent users). The whole
# bench runs ROUNDS times (default 3): llama-batched-bench has no repetitions of its own, and the house rule wants three
# with the across-run spread quoted. Each board runs on its own (no network); the three run at once because none of them
# touches another. Cold gate and get_throttled (Pis) / Intel throttle counters (laptop) before and after every round;
# raw output tee'd. The Pi 4 runs NPL_PI4 (default 1,2): four 128-token prompts cost it over five minutes of prefill a
# round (it does not slow down when hot: it held 1,800 MHz to 82 C, testbed/scripts/pi_thermal_clock_probe.sh).
set -uo pipefail
cd "$(dirname "$0")/../.."
NODES=${NODES:-"pi5 pi4 t580"}; NPP=${NPP:-128}; NTG=${NTG:-32}; NPL=${NPL:-1,2,4}; NPL_PI4=${NPL_PI4:-1,2}
CTX=${CTX:-4096}; ROUNDS=${ROUNDS:-3}
# EXTRA: more llama-batched-bench flags, default none (2026-09-29: EXTRA=--no-repack on the Pi 5 tests whether its repacked
# kernels are why 2 sequences cost it almost a whole second decode step)
EXTRA=${EXTRA:-}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=testbed/results/batched_decode_$STAMP
mkdir -p "$OUT"
exec > >(tee -i -a "$OUT/run.log") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
SHA9=d784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43
declare -A COLD=([pi5]=50 [pi4]=57 [t580]=65)   # the Pi 4 holds 1,800 MHz to 82 C (pi_thermal_clock_probe.sh): 57 is the rows' usual gate
on() { local n=$1; shift; if [ "$n" = t580 ]; then bash -c "$*" < /dev/null; else $SSH "$n" "$*"; fi; }
bin() { [ "$1" = t580 ] && echo /opt/heteroserve/llama.cpp/build/bin || echo '~/llama.cpp/build/bin'; }
model() { [ "$1" = t580 ] && echo /opt/heteroserve/models/qwen3.5-9b-q4_k_m.gguf || echo '~/models/qwen3.5-9b-q4_k_m.gguf'; }
temp() { if [ "$1" = t580 ]; then for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type)" = x86_pkg_temp ] && echo $(( $(cat $z/temp) / 1000 )); done
         else on $1 "echo \$(( \$(cat /sys/class/thermal/thermal_zone0/temp) / 1000 ))"; fi; }
state() {
  if [ "$1" = t580 ]; then
    T=/sys/devices/system/cpu
    echo "$2 t580 $(date -Is) temp=$(temp t580)'C throttle_events=$(cat $T/cpu0/thermal_throttle/core_throttle_count)/$(cat $T/cpu0/thermal_throttle/package_throttle_count) load=$(cut -d' ' -f1 /proc/loadavg)"
  else
    on $1 "echo \"$2 $1 \$(date -Is) \$(vcgencmd measure_temp) \$(vcgencmd get_throttled) arm=\$(vcgencmd measure_clock arm | cut -d= -f2) load=\$(cut -d' ' -f1 /proc/loadavg)\""
  fi
}
board() {
  local n=$1 npl=$NPL r w t
  [ $n = pi4 ] && npl=$NPL_PI4
  # llama-batched-b: the kernel keeps 15 characters of a process name, so pgrep -x never matches "llama-batched-bench"
  [ "$(on $n 'for p in llama-bench llama-batched-b llama-cli llama-server ggml-rpc-server; do pgrep -x $p; done | wc -l')" = 0 ] || { echo "SKIP $n: busy"; return; }
  local got; got=$(on $n "sha256sum $(model $n) | cut -c1-64")
  [ "$got" = $SHA9 ] || { echo "SKIP $n: model sha256 $got is not the pinned one"; return; }
  echo "##### $n build $(on $n "git -C $(bin $n)/../.. rev-parse HEAD"), model sha256 ok, -npp $NPP -ntg $NTG -npl $npl -c $CTX -t 4 $EXTRA"
  for r in $(seq "$ROUNDS"); do
    w=0; while t=$(temp $n); [ "${t:-99}" -ge "${COLD[$n]}" ] && [ $w -lt 900 ]; do sleep 15; w=$((w + 15)); done
    echo "===== $n round $r  cold gate: waited ${w}s, now ${t} C (threshold ${COLD[$n]} C)"
    state $n STATE_BEFORE
    on $n "timeout 3000 $(bin $n)/llama-batched-bench -m $(model $n) -c $CTX -b 2048 -ub 512 -npp $NPP -ntg $NTG -npl $npl -t 4 -tb 4 $EXTRA" 2>&1
    echo "EXIT_${n}_r$r=$?"
    state $n STATE_AFTER
  done
  sleep 30; state $n STATE_END
}
echo "########## $(date -Is) batched decode, nodes [$NODES]${EXTRA:+, extra flags $EXTRA} -> $OUT"
declare -A PID=()
for n in $NODES; do board $n > "$OUT/$n.log" 2>&1 & PID[$n]=$!; done
for n in $NODES; do wait ${PID[$n]}; done
echo "===== summary: per round, B sequences -> per-user decode ms/token (T_TG / NTG), aggregate decode t/s, prefill t/s"
python3 - "$OUT" $NODES <<'EOF'
import os, re, statistics, sys
out, nodes = sys.argv[1], sys.argv[2:]
for n in nodes:
    p = os.path.join(out, n + ".log")
    if not os.path.exists(p): continue
    rows, rnd = {}, None
    for line in open(p, errors="replace"):
        m = re.match(r"===== \S+ round (\d+)", line)
        if m: rnd = int(m.group(1)); continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 10 and c[0].isdigit():
            pp, tg, b = int(c[0]), int(c[1]), int(c[2])
            rows.setdefault(b, []).append((rnd, float(c[4]), float(c[5]), float(c[6]), float(c[7]), tg))
    for b in sorted(rows):
        r = rows[b]
        tpot = [1000 * x[3] / x[5] for x in r]                  # T_TG over TG steps: one token per user per step
        agg = [x[4] for x in r]; spp = [x[2] for x in r]
        sp = lambda v: 100 * (max(v) - min(v)) / statistics.fmean(v) if len(v) > 1 else float("nan")
        print("%-5s B=%d  per-user decode %7.1f ms/token (rounds %s; spread %.1f %%)  aggregate %6.3f t/s  prefill %6.2f t/s (spread %.1f %%)  n=%d"
              % (n, b, statistics.fmean(tpot), " / ".join("%.1f" % v for v in tpot), sp(tpot), statistics.fmean(agg),
                 statistics.fmean(spp), sp(spp), len(r)))
EOF
echo "DONE $(date -Is)"
