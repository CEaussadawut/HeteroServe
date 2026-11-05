#!/bin/bash
# What does the 1 Hz throttle poll cost a cell? (2026-10-07 night, before run_pool.sh polls in every Pi cell.)
# memprobe v1, about five process starts a second, cost 4.5 % of decode on the Pi 5 (benchmarks.md, 2026-09-28).
# throttle_poll_v1.sh also starts five a second (101 in 20 s on the Pi 4) and ran beside the three 14B cells of
# 2026-10-07 17:42-18:12; throttle_poll.sh v2 starts one (24 in 20 s, 21 samples). On the Pi 4 alone -- no RPC, no
# network -- three conditions, in ABC-CBA rounds so that a steady drift cancels:
#   R   llama-bench alone
#   V1  with throttle_poll_v1.sh running
#   V2  with throttle_poll.sh (v2) running
# TinyLlama Q4_K_M, -t 4 -p 64 -n 128 -r 5: the pool protocol's prefill, and a decode sample long enough (about 11 s) for
# the poll to land in it many times. Run from the repo root on the laptop:
#     bash testbed/scripts/throttle_poll_ab_pi4.sh                     ROUNDS=3 (default: R V1 V2, V2 V1 R, R V1 V2)
# House rules: the model's SHA-256 on the board and the tree at the pin first; every sample in jsonl; raw output tee'd;
# temperature and get_throttled before and after every run, and each poll's own log kept; one cold gate for every run
# (55 C, at most 10 min), so that the start temperature is not one of the differences.
set -uo pipefail
cd "$(dirname "$0")/../.."
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=testbed/results/throttle_poll_ab_$STAMP
mkdir -p "$OUT"
exec > >(trap '' INT TERM HUP QUIT PIPE; exec tee -i -a "$OUT/run.log") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=6"
N=pi4; COLD=55; ROUNDS=${ROUNDS:-3}
M=tinyllama-1.1b-q4km.gguf; MSHA=9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0
PIN=$(python3 -c "import json;print(json.load(open('ThinkPad/config/cluster.json'))['llama']['pin_sha'])")
PROBE=""
fatal() { echo "FATAL: $*"; exit 1; }
cleanup() { trap '' PIPE INT TERM HUP; [ -n "$PROBE" ] && $SSH $N "kill $PROBE 2>/dev/null" && echo "cleanup: poll $PROBE stopped"; PROBE=""; }
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 129' HUP; trap 'exit 143' TERM

echo "########## $(date -Is) throttle-poll A/B on $N: $M, -t 4 -p 64 -n 128 -r 5, $ROUNDS rounds, cold gate $COLD C"
b=$($SSH $N 'for p in llama-bench llama-cli llama-server ggml-rpc-server cc1plus cmake; do pgrep -x $p >/dev/null && printf "%s " $p; done; true') \
  || fatal "ssh $N failed"
[ -z "$b" ] || fatal "$N is busy: $b"
h=$($SSH $N 'git -C ~/llama.cpp rev-parse HEAD; git -C ~/llama.cpp status --porcelain --untracked-files=no | wc -l' | tr '\n' ' ')
[ "$h" = "$PIN 0 " ] || fatal "$N tree is '$h', not the pin $PIN clean"
s=$($SSH $N "sha256sum ~/models/$M | cut -d' ' -f1"); echo "sha256 $M on $N: $s"
[ "$s" = "$MSHA" ] || fatal "$M on $N does not match testbed/models.md"
scp -q testbed/scripts/throttle_poll_v1.sh $N:throttle_poll_v1.sh && scp -q testbed/scripts/throttle_poll.sh $N:throttle_poll.sh \
  || fatal "cannot copy the polls"
echo "polls on $N: $($SSH $N 'cd ~ && sha256sum throttle_poll_v1.sh throttle_poll.sh' | tr '\n' ' ')"

state() {  # $1 label
  $SSH $N "echo \"$1 \$(date -Is) \$(vcgencmd measure_temp) \$(vcgencmd get_throttled) gov=\$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) fan=\$(cat /sys/class/thermal/cooling_device0/cur_state) memavail_mib=\$(awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo) load=\$(cut -d' ' -f1-3 /proc/loadavg)\""
}
run() {   # run COND ROUND
  local c=$1 r=$2 tag="$1_r$2" t w=0 a pf
  while t=$($SSH $N "cat /sys/class/thermal/thermal_zone0/temp"); [ "${t:-99999}" -ge $((COLD * 1000)) ] && [ $w -lt 600 ]; do
    sleep 15; w=$((w + 15)); done
  echo "##### RUN $tag: condition $c, round $r; cold gate waited ${w}s, now $((t / 1000)).$((t % 1000 / 100)) C (threshold $COLD C)"
  state STATE_BEFORE
  case $c in V1) pf=throttle_poll_v1.sh ;; V2) pf=throttle_poll.sh ;; *) pf="" ;; esac
  if [ -n "$pf" ]; then
    PROBE=$($SSH $N "mkdir -p ~/logs; setsid nohup bash ~/$pf ~/logs/throttle_ab_${STAMP}_$tag.log > /dev/null 2>&1 < /dev/null & echo \$!")
    echo "poll pid $PROBE ($pf)"
  fi
  $SSH $N "cd ~ && /usr/bin/timeout 900 ~/llama.cpp/build/bin/llama-bench -m ~/models/$M -t 4 -p 64 -n 128 -r 5 -oe jsonl; echo EXIT_BENCH=\$?"
  if [ -n "$pf" ]; then
    a=$($SSH $N "kill $PROBE 2>/dev/null && echo 1 || echo 0"); PROBE=""
    if scp -q $N:logs/throttle_ab_${STAMP}_$tag.log "$OUT/throttle_$tag.log"; then
      echo "THROTTLE $N $(python3 testbed/scripts/throttle_summary.py "$OUT/throttle_$tag.log" --alive $a)"
    else echo "!!!!! could not copy the poll log of $tag"; fi
  fi
  state STATE_AFTER
}
for r in $(seq 1 $ROUNDS); do
  if [ $((r % 2)) = 1 ]; then order="R V1 V2"; else order="V2 V1 R"; fi
  for c in $order; do run $c $r; done
done
echo "##### SUMMARY (ms per token: 1000 / mean of the five samples' t/s; change against the same round's R)"
python3 - "$OUT/run.log" <<'PY'
import json, re, statistics, sys
runs, cur = {}, None
for ln in open(sys.argv[1], encoding="utf-8", errors="replace"):
    m = re.match(r"##### RUN (\S+): condition (\S+), round (\d+)", ln)
    if m:
        cur = (m.group(2), int(m.group(3))); runs[cur] = {}
    elif ln.startswith('{"build_commit') and cur:
        d = json.loads(ln); ph = "pp64" if d["n_prompt"] else "tg128"
        runs[cur][ph] = 1000 / statistics.fmean(d["samples_ts"])
for ph in ("tg128", "pp64"):
    print(ph)
    for (c, r), v in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if ph in v:
            ref = runs.get(("R", r), {}).get(ph)
            print("  round %d %-2s %8.2f ms/token%s" % (r, c, v[ph], "   %+.2f %% vs R" % (100 * (v[ph] / ref - 1)) if ref and c != "R" else ""))
    for c in ("R", "V1", "V2"):
        xs = [v[ph] for (cc, _), v in runs.items() if cc == c and ph in v]
        if xs:
            print("  %-2s mean %8.2f ms/token over %d runs, spread %.2f %%" % (c, statistics.fmean(xs), len(xs), 100 * (max(xs) - min(xs)) / statistics.fmean(xs)))
PY
echo "########## $(date -Is) done; log $OUT/run.log"
