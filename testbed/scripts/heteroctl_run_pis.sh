#!/usr/bin/env bash
# heteroctl runs driven from the laptop, with the Pi 5 and Pi 4 as RPC workers and no node agents (user, 2026-09-29:
# "Laptop-driven runs, no agents"). From the repo root:
#
#   testbed/scripts/heteroctl_run_pis.sh REPS MODEL PROFILES [extra heteroctl run args...]
#   e.g. testbed/scripts/heteroctl_run_pis.sh 3 qwen3-14b-q4km.gguf testbed/results/heteroctl_20260929/profiles_q4km.json
#
# The workers are started here and stopped here, the testbed's way (run_pool.sh):
#  - gates: board idle; ~/llama.cpp at the pin and clean;
#  - `ggml-rpc-server -H <overlay> -p 50052 -t 4`;
#  - readiness via `ss` on the board, never a bare connect, which kills the server;
#  - processes found by exact argv.
# heteroctl itself runs unchanged: its agent calls (rpc up, markers, conditions) fail and are recorded as such.
# Board state (get_throttled, temperature, MemAvailable) is logged before and after, and everything is tee'd raw to
# testbed/results/heteroctl_20260929/runs_<stamp>.log.
set -uo pipefail
REPS=$1 MODEL=$2 PROFILES=$3; shift 3
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ROOT/testbed/results/heteroctl_20260929
CFG=$OUT/cluster.pis-only.json
PIN=$(python3 -c "import json;print(json.load(open('$ROOT/ThinkPad/config/cluster.json'))['llama']['pin_sha'])")
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=$OUT/runs_$STAMP.log
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
declare -A WIP=([pi5]=192.168.10.11 [pi4]=192.168.10.12)
PORT=50052
exec > >(tee -a "$LOG") 2>&1

wpid() { $SSH $1 "for p in \$(pgrep -x ggml-rpc-server); do tr '\\0' ' ' < /proc/\$p/cmdline | grep -qF -- '-H ${WIP[$1]} -p $PORT ' && echo \$p; done"; }
state() { for n in pi5 pi4; do echo "STATE $1 $n $($SSH $n 'echo "$(date -Is) $(vcgencmd measure_temp) $(vcgencmd get_throttled) memavail_mib=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo) load=$(cut -d" " -f1 /proc/loadavg)"')"; done; }
stop_workers() { for n in pi5 pi4; do for p in $(wpid $n); do $SSH $n "kill $p"; done; done; sleep 2
                 for n in pi5 pi4; do [ -z "$(wpid $n)" ] && echo "worker $n stopped" || echo "!!!!! worker $n STILL RUNNING"; done; }

echo "########## $(date -Is) heteroctl run x$REPS: $MODEL, profiles $PROFILES, config $CFG, pin $PIN, extra: $*"
for n in pi5 pi4; do
  b=$($SSH $n 'for p in llama-bench llama-cli llama-server ggml-rpc-server cc1plus cmake; do pgrep -x $p >/dev/null && printf "%s " $p; done; true')
  [ -z "$b" ] || { echo "FATAL: $n is busy: $b"; exit 1; }
  h=$($SSH $n 'git -C ~/llama.cpp rev-parse HEAD; git -C ~/llama.cpp status --porcelain --untracked-files=no | wc -l' | tr '\n' ' ')
  [ "$h" = "$PIN 0 " ] || { echo "FATAL: $n tree is '$h', not the pin clean"; exit 2; }
done
echo "gate    both boards idle, both trees at the pin and clean"
trap stop_workers EXIT
for n in pi5 pi4; do
  $SSH $n "mkdir -p ~/logs; setsid nohup ~/llama.cpp/build/bin/ggml-rpc-server -H ${WIP[$n]} -p $PORT -t 4 > ~/logs/rpc-server-heteroctl_$STAMP.log 2>&1 < /dev/null &"
done
for n in pi5 pi4; do
  for _ in $(seq 20); do $SSH $n "ss -ltn | grep -q '${WIP[$n]}:$PORT '" && break; sleep 1; done
  $SSH $n "ss -ltn | grep -q '${WIP[$n]}:$PORT '" || { echo "FATAL: $n worker never listened"; $SSH $n "tail -5 ~/logs/rpc-server-heteroctl_$STAMP.log"; exit 3; }
  echo "worker  $n listening on ${WIP[$n]}:$PORT (pid $(wpid $n))"
done
state before
cd "$ROOT/ThinkPad"
for r in $(seq "$REPS"); do
  for n in pi5 pi4; do [ -n "$(wpid $n)" ] || { echo "FATAL: $n worker is gone before rep $r (see its kernel log for an OOM kill)"; exit 4; }; done
  echo "=== rep $r $(date -Is)"
  python3 -m control.cli --config "$CFG" run "$MODEL" --profiles "$ROOT/$PROFILES" --skip-preflight "$@"
  echo "=== rep $r exit $?"
  state "after_rep$r"
done
echo "log: $LOG"
