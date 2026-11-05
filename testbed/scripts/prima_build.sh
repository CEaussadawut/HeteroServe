#!/bin/bash
# Build prima.cpp on the boards: OpenCPIL/prima.cpp @ 288df6a2 (testbed/prima.md) + testbed/patches/prima-288df6a-qwen3-v1.patch
# (the Qwen3 port). From the repo root on the laptop:
#     NODES="pi5 pi4" bash testbed/scripts/prima_build.sh          # HIGHS=<prefix on the node> adds USE_HIGHS=1 (rank 0)
# The source is the laptop's pinned tree as `git archive` (sha256 logged), unpacked on each board into ~/prima.cpp. The patch
# is applied there with `git apply`, and the patched src/llama.cpp must hash the same as the laptop's before anything
# builds. Then `make -j4 llama-cli llama-server`. It refuses to touch an existing ~/prima.cpp; nothing is deleted.
# Raw output tee'd to testbed/results/prima_build_<stamp>.log; each board's make log to ~/logs/ there, tail copied back.
set -uo pipefail
cd "$(dirname "$0")/../.."
NODES=${NODES:-pi5 pi4}; HIGHS=${HIGHS:-}
TREE=/opt/heteroserve/prima.cpp; PIN=288df6a2cc2120387e12d6f40798f30cc525c9ea
PATCH=testbed/patches/prima-288df6a-qwen3-v1.patch
STAMP=$(date +%Y%m%d_%H%M%S)
exec > >(tee -a "testbed/results/prima_build_$STAMP.log") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
TMP=${TMPDIR:-/tmp}/prima_build_$STAMP; mkdir -p "$TMP"      # the archive, kept for the record
git -C $TREE archive --prefix=prima.cpp/ $PIN | gzip -n > $TMP/prima-288df6a.tar.gz
WANT=$(git -C $TREE diff --quiet $PIN -- src/llama.cpp && echo unpatched || sha256sum $TREE/src/llama.cpp | cut -d' ' -f1)
echo "########## $(date -Is) prima.cpp build on [$NODES]: pin $PIN, archive $(sha256sum $TMP/prima-288df6a.tar.gz | cut -c1-16), patch $(sha256sum $PATCH | cut -c1-16), patched src/llama.cpp must be $WANT"
[ "$WANT" = unpatched ] && { echo "FATAL: the laptop tree is not patched"; exit 1; }
for n in $NODES; do
  $SSH $n 'test -e ~/prima.cpp' && { echo "FATAL: $n already has ~/prima.cpp (not overwritten)"; exit 1; }
  $SSH $n 'mkdir -p ~/src ~/logs'
  scp -q $TMP/prima-288df6a.tar.gz $n:src/ && scp -q $PATCH $n:src/prima-288df6a-qwen3-v1.patch || { echo "FATAL: copy to $n failed"; exit 1; }
  got=$($SSH $n 'cd ~ && tar xzf src/prima-288df6a.tar.gz && cd prima.cpp && git apply ~/src/prima-288df6a-qwen3-v1.patch && sha256sum src/llama.cpp | cut -d" " -f1')
  echo "source  $n: patched src/llama.cpp $got"
  [ "$got" = "$WANT" ] || { echo "FATAL: $n's patched source differs"; exit 1; }
done
pids=""
for n in $NODES; do
  flags=""; [ -n "$HIGHS" ] && [ "$n" = pi5 ] && flags="USE_HIGHS=1 HIGHS_CPPFLAGS=-isystem\ $HIGHS/include/highs HIGHS_LDFLAGS=-L$HIGHS/lib\ -lhighs"
  echo "build   $n: make -j4 llama-cli llama-server $flags ($(date -Is))"
  ( $SSH $n "cd ~/prima.cpp && t0=\$(date +%s); make -j4 llama-cli llama-server $flags > ~/logs/prima_build_$STAMP.log 2>&1; echo EXIT_MAKE=\$? elapsed_s=\$(( \$(date +%s) - t0 ))" | sed "s/^/$n: /" ) &
  pids+=" $!"
done
for p in $pids; do wait $p; done
for n in $NODES; do
  echo "--- $n"; $SSH $n "tail -4 ~/logs/prima_build_$STAMP.log; grep -c -i 'warning' ~/logs/prima_build_$STAMP.log; cd ~/prima.cpp && sha256sum llama-cli llama-server 2>/dev/null"
done
echo "BUILD DONE $(date -Is)"
