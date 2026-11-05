#!/bin/bash
# Copy model files board to board over the wired overlay, verified. From the repo root on the laptop:
#     SRC=pi5 DST=pi4 bash testbed/scripts/copy_model_between_boards.sh FILE=SHA256 [FILE=SHA256 ...]
# The boards have no internet and no keys to each other, so the bytes go straight across the overlay with OpenBSD
# netcat (port 5201; never 50052): the receiver writes ~/models/.<file>.part, the file is SHA-256-checked ON THE
# RECEIVER against the value given (from testbed/models.md), and only a match is renamed into place. A mismatch
# leaves the .part file for inspection and fails the run.
set -uo pipefail
cd "$(dirname "$0")/../.."
SRC=${SRC:-pi5}; DST=${DST:-pi4}
declare -A IP=([pi5]=192.168.10.11 [pi4]=192.168.10.12)
OUT=testbed/results/copy_${SRC}-${DST}_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -i -a "$OUT") 2>&1
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8"
P=5201; rc=0
echo "########## $(date -Is) copy $SRC -> $DST over ${IP[$DST]}:$P"
for arg in "$@"; do
  f=${arg%%=*}; want=${arg#*=}
  if $SSH -n $DST "echo '$want  models/$f' | sha256sum -c --quiet" >/dev/null 2>&1; then echo "present $f (sha256 OK), skipped"; continue; fi
  $SSH -n $DST "timeout 3600 nc -l $P > ~/models/.$f.part" &
  bg=$!
  for i in $(seq 1 100); do $SSH -n $DST "ss -ltn | grep -q ':$P '" && break; done
  t0=$(date +%s)
  $SSH -n $SRC "nc -N ${IP[$DST]} $P < ~/models/$f"
  wait $bg
  got=$($SSH -n $DST "sha256sum ~/models/.$f.part | cut -d' ' -f1")
  if [ "$got" = "$want" ]; then
    $SSH -n $DST "mv ~/models/.$f.part ~/models/$f" && echo "copied $f in $(( $(date +%s) - t0 )) s, sha256 OK on $DST"
  else
    echo "!!!!! $f: sha256 on $DST is $got, expected $want -- left as .$f.part"; rc=1
  fi
done
echo "COPY DONE rc=$rc $(date -Is)"
exit $rc
