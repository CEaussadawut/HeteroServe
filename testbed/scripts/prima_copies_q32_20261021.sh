#!/bin/bash
# The boards' partial copies of Qwen3-32B Q8_0 for prima.cpp (user, 2026-10-21 ~20:45: "Partial copies"), from Halda's own
# decision at ctx 2048 (testbed/results/prima_run4_q32decide2048_20261021_213732*: laptop 39, pi5 11, pi4 10, jetson 4, so
# layers 0-10 on the Pi 5, 11-20 on the Pi 4, 21-24 on the Jetson), each with 2 layers to spare either side. One node at a
# time; prima_partial_20261021.py verifies every range on the board against the laptop's verified file and writes the manifest.
# The Jetson's power guard runs while it is written. From the repo root:  bash testbed/scripts/prima_copies_q32_20261021.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=testbed/results/prima_copies_q32_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
M=~/models/qwen3-32b-q8_0.gguf
echo "########## $(date -Is) partial copies of $M ($(sha256sum $M | cut -c1-16) on the laptop)"
for nl in "pi5 0-12" "pi4 9-22" "jetson 19-26"; do
  set -- $nl
  if [ $1 = jetson ]; then
    g=$($SSH jetson 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum ~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1' 2>/dev/null | tr '\n' ' ')
    [ "$g" = "synced $(sha256sum testbed/scripts/jetson_power_guard.sh | cut -d' ' -f1) " ] || { echo "FATAL: Jetson guard precheck '$g'"; exit 1; }
    $SSH jetson 'sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo guard started' 2>/dev/null
  fi
  echo "df before $1: $($SSH $1 'df -h --output=avail ~ | tail -1' 2>/dev/null)"
  t0=$(date +%s); python3 testbed/scripts/prima_partial_20261021.py copy --model $M --node $1 --layers $2 2>&1 | grep -v -E "post-quantum|store now|upgraded"
  rc=${PIPESTATUS[0]}; echo "copy $1 exit $rc in $(( $(date +%s) - t0 )) s; df after: $($SSH $1 'df -h --output=avail ~ | tail -1' 2>/dev/null)"
  if [ $1 = jetson ]; then
    $SSH jetson 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p stopped"; done; echo "TRIP lines: $(grep -c TRIP ~/heteroserve/logs/power-guard.log)"' 2>/dev/null
  fi
  [ "$rc" = 0 ] || { echo "STOPPED: the copy to $1 failed"; exit 1; }
done
echo "COPIES DONE $(date -Is)"
