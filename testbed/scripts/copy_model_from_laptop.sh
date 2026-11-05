#!/bin/bash
# Copy verified model files from the laptop's cache to one board over the wired LAN, re-verified ON the board.
# From the repo root on the laptop:
#     DST=pi5 bash testbed/scripts/copy_model_from_laptop.sh CACHE_FILE=BOARD_NAME=SHA256 [CACHE_FILE=BOARD_NAME=SHA256 ...]
# CACHE_FILE is in ~/.cache/heteroserve-models/ (fetched by .handoff/assets/models-2026-09-28/fetch_models.py and checked
# there against the manifest's sha256); BOARD_NAME is the file's name in the board's ~/models/. The boards have no
# internet, so this is the only way in. Each file goes to ~/models/.<BOARD_NAME>.part, is SHA-256-checked on the board
# against the value given, and only a match is renamed into place. A mismatch leaves the .part file and fails the run.
# A file already present with the right hash is skipped. Nothing is ever deleted to make room: a board short of space
# refuses before copying. A full-size .part left by an interrupted run is verified rather than copied again.
set -uo pipefail
cd "$(dirname "$0")/../.."
DST=${DST:?set DST (pi5, pi4 or jetson)}
CACHE=$HOME/.cache/heteroserve-models
OUT=testbed/results/copy_laptop-${DST}_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -i -a "$OUT") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
# Right after a large copy a Pi can spend minutes writing it back to its SD card and miss an 8 s ssh timeout
# (Pi 4, 2026-09-28 15:49): the on-board check gets a longer timeout and five tries.
board_sha() {
  local i got
  for i in 1 2 3 4 5; do
    got=$(ssh -n -o BatchMode=yes -o ConnectTimeout=60 $DST "sha256sum ~/models/.$1.part | cut -d' ' -f1") && [ -n "$got" ] && { echo "$got"; return 0; }
    echo "      on-board check of .$1.part: try $i failed" >&2; sleep 30
  done
  return 1
}
rc=0
echo "########## $(date -Is) copy laptop -> $DST"
for arg in "$@"; do
  src=${arg%%=*}; rest=${arg#*=}; name=${rest%%=*}; want=${rest#*=}
  [ -f "$CACHE/$src" ] || { echo "!!!!! $src: not in $CACHE"; rc=1; continue; }
  [ "$(sha256sum "$CACHE/$src" | cut -d' ' -f1)" = "$want" ] || { echo "!!!!! $src: the laptop copy does not match $want"; rc=1; continue; }
  if $SSH $DST "echo '$want  models/$name' | sha256sum -c --quiet" >/dev/null 2>&1; then echo "present $name on $DST (sha256 OK), skipped"; continue; fi
  size=$(stat -c %s "$CACHE/$src")
  t0=$(date +%s)
  if [ "$($SSH $DST "stat -c %s ~/models/.$name.part 2>/dev/null")" = "$size" ]; then
    echo "found .$name.part on $DST at full size ($size bytes): verifying it instead of copying again"
  else
    need=$(( size / 1024 ))
    free=$($SSH $DST "df --output=avail -k ~/models | tail -1 | tr -d ' '")
    [ "${free:-0}" -gt $(( need + 1048576 )) ] || { echo "!!!!! $name: $DST has ${free} KiB free, needs ${need} + 1 GiB headroom -- not copied"; rc=1; continue; }
    scp -q "$CACHE/$src" "$DST:models/.$name.part" || { echo "!!!!! scp $name -> $DST failed"; rc=1; continue; }
  fi
  got=$(board_sha "$name")
  if [ "$got" = "$want" ]; then
    if ssh -n -o BatchMode=yes -o ConnectTimeout=60 $DST "mv ~/models/.$name.part ~/models/$name"; then
      echo "placed $name in $(( $(date +%s) - t0 )) s, sha256 OK on $DST"
    else
      echo "!!!!! $name: sha256 OK on $DST but the rename failed -- left as .$name.part"; rc=1
    fi
  else
    echo "!!!!! $name: sha256 on $DST is $got, expected $want -- left as .$name.part"; rc=1
  fi
done
echo "COPY DONE $(date -Is) rc=$rc"
exit $rc
