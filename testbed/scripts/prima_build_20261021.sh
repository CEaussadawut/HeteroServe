#!/bin/bash
# prima.cpp on the two nodes it was never built on (user, 2026-10-21 ~17:55: "Yes, build on Jetson + laptop"), so that prima.cpp +
# Halda can be compared with the engine on all four nodes. The Pis keep their 2026-09-30 builds (testbed/scripts/prima_build.sh).
# Same source as there: OpenCPIL/prima.cpp @ 288df6a2 as `git archive`, plus testbed/patches/prima-288df6a-qwen3-v1.patch, the
# patched src/llama.cpp hash-checked against the laptop tree's before anything builds. Into ~/prima.cpp; never over an existing one.
#   laptop  rank 0, the head (it runs Halda): `make -j4 llama-cli llama-server USE_HIGHS=1`. Needs <zmq.h> (the user's
#           `sudo apt install libzmq3-dev`) and HiGHS v1.15.1's official x86-64 static build unpacked in $HIGHS
#           (default ~/highs-1.15.1; the download the user approved, its sha256 checked against GitHub's digest).
#   jetson  CPU only: prima.cpp has no Vulkan, and its CUDA path is not patched for CUDA 10.2. g++-8 (the board's CUDA host
#           compiler; prima.cpp builds as C++11) and the board's own libzmq.so.5 (ZeroMQ 4.2.5), with ZeroMQ 4.2.5's zmq.h
#           (zeromq/libzmq @ v4.2.5, the download the user approved) in ~/prima-deps/include and a libzmq.so link in
#           ~/prima-deps/lib. make -j2 (4 GiB of RAM). The power guard is started first by its NOTE's procedure
#           (.handoff/nodes/Jetson/2026-10-06-1105-NOTE-third-hang-backup-guard-down.md) and stopped after.
# From the repo root, with no measurement running:
#     NODES="laptop jetson" bash testbed/scripts/prima_build_20261021.sh       # ZMQH=<path of the v4.2.5 zmq.h on the laptop>
# Raw output tee'd to testbed/results/prima_build_<stamp>.log; each node's make log kept (laptop: in the results directory;
# Jetson: ~/logs/, tail copied back).
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
NODES=${NODES:-laptop jetson}; HIGHS=${HIGHS:-$HOME/highs-1.15.1}; ZMQH=${ZMQH:-}
TREE=/opt/heteroserve/prima.cpp; PIN=288df6a2cc2120387e12d6f40798f30cc525c9ea
PATCH=testbed/patches/prima-288df6a-qwen3-v1.patch
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=testbed/results/prima_build_$STAMP.log
exec > >(tee -a "$LOG") 2>&1
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8"
TMP=${TMPDIR:-/tmp}/prima_build_$STAMP; mkdir -p "$TMP"
git -C $TREE archive --prefix=prima.cpp/ $PIN | gzip -n > $TMP/prima-288df6a.tar.gz
WANT=$(git -C $TREE diff --quiet $PIN -- src/llama.cpp && echo unpatched || sha256sum $TREE/src/llama.cpp | cut -d' ' -f1)
echo "########## $(date -Is) prima.cpp build on [$NODES]: pin $PIN, archive $(sha256sum $TMP/prima-288df6a.tar.gz | cut -c1-16), patch $(sha256sum $PATCH | cut -c1-16), patched src/llama.cpp must be $WANT"
[ "$WANT" = unpatched ] && { echo "FATAL: the laptop tree is not patched"; exit 1; }
cp $TMP/prima-288df6a.tar.gz testbed/results/prima_build_${STAMP}_source.tar.gz   # the exact source, kept with the log

for n in $NODES; do
  case $n in
  laptop)
    [ -e ~/prima.cpp ] && { echo "FATAL: the laptop already has ~/prima.cpp (not overwritten)"; exit 1; }
    [ -f /usr/include/zmq.h ] || { echo "FATAL: no /usr/include/zmq.h on the laptop (the user's: sudo apt install libzmq3-dev)"; exit 1; }
    [ -f $HIGHS/lib/libhighs.a ] || { echo "FATAL: no HiGHS in $HIGHS"; exit 1; }
    echo "zmq     laptop: $(grep -m1 -E '^#define ZMQ_VERSION_MAJOR' /usr/include/zmq.h | awk '{print $3}').$(grep -m1 -E '^#define ZMQ_VERSION_MINOR' /usr/include/zmq.h | awk '{print $3}').$(grep -m1 -E '^#define ZMQ_VERSION_PATCH' /usr/include/zmq.h | awk '{print $3}'), HiGHS libs: $(ls $HIGHS/lib | tr '\n' ' ')"
    got=$(cd ~ && tar xzf $TMP/prima-288df6a.tar.gz && cd prima.cpp && git apply $ROOT/$PATCH 1>&2 && sha256sum src/llama.cpp | cut -d' ' -f1)
    echo "source  laptop: patched src/llama.cpp $got"
    [ "$got" = "$WANT" ] || { echo "FATAL: the laptop's patched source differs"; exit 1; }
    ;;
  jetson)
    [ -n "$ZMQH" ] && [ -f "$ZMQH" ] || { echo "FATAL: ZMQH must name ZeroMQ 4.2.5's zmq.h on the laptop"; exit 1; }
    grep -q '^#define ZMQ_VERSION_MINOR 2$' "$ZMQH" && grep -q '^#define ZMQ_VERSION_PATCH 5$' "$ZMQH" || { echo "FATAL: $ZMQH is not 4.2.5's"; exit 1; }
    $SSH jetson 'test -e ~/prima.cpp' && { echo "FATAL: the Jetson already has ~/prima.cpp (not overwritten)"; exit 1; }
    so=$($SSH jetson 'ls /usr/lib/aarch64-linux-gnu/libzmq.so.5.* 2>/dev/null | head -1')
    [ -n "$so" ] || { echo "FATAL: no libzmq.so.5 on the Jetson"; exit 1; }
    $SSH jetson 'command -v g++-8 >/dev/null && command -v gcc-8 >/dev/null' || { echo "FATAL: no gcc-8/g++-8 on the Jetson"; exit 1; }
    echo "zmq     jetson: $so; df: $($SSH jetson 'df -h --output=avail ~ | tail -1')"
    $SSH jetson "mkdir -p ~/src ~/logs ~/prima-deps/include ~/prima-deps/lib && ln -sfn $so ~/prima-deps/lib/libzmq.so"
    scp -q "$ZMQH" jetson:prima-deps/include/zmq.h && scp -q $TMP/prima-288df6a.tar.gz jetson:src/ \
      && scp -q $PATCH jetson:src/prima-288df6a-qwen3-v1.patch || { echo "FATAL: copy to the Jetson failed"; exit 1; }
    got=$($SSH jetson 'cd ~ && tar xzf src/prima-288df6a.tar.gz && cd prima.cpp && git apply ~/src/prima-288df6a-qwen3-v1.patch && sha256sum src/llama.cpp | cut -d" " -f1')
    echo "source  jetson: patched src/llama.cpp $got"
    [ "$got" = "$WANT" ] || { echo "FATAL: the Jetson's patched source differs"; exit 1; }
    ;;
  *) echo "FATAL: unknown node $n (laptop, jetson)"; exit 1 ;;
  esac
done

for n in $NODES; do
  case $n in
  laptop)
    # the static build: libhighs.a needs libhighs_extras.a and zlib (as on the Pi 5, testbed/prima.md); highs_export.h is in include/
    flags="USE_HIGHS=1 HIGHS_CPPFLAGS=-isystem\\ $HIGHS/include/highs\\ -isystem\\ $HIGHS/include HIGHS_LDFLAGS=-L$HIGHS/lib\\ -lhighs\\ -lhighs_extras\\ -lz"
    echo "build   laptop: make -j4 llama-cli llama-server $flags ($(date -Is))"
    ( cd ~/prima.cpp && t0=$(date +%s); eval make -j4 llama-cli llama-server $flags > "$ROOT/testbed/results/prima_build_${STAMP}_laptop_make.log" 2>&1
      echo "laptop: EXIT_MAKE=$? elapsed_s=$(( $(date +%s) - t0 ))" )
    tail -4 testbed/results/prima_build_${STAMP}_laptop_make.log
    (cd ~/prima.cpp && sha256sum llama-cli llama-server 2>/dev/null)
    ;;
  jetson)
    g=$($SSH jetson 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum ~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1' 2>/dev/null | tr '\n' ' ')
    [ "$g" = "synced $(sha256sum testbed/scripts/jetson_power_guard.sh | cut -d' ' -f1) " ] || { echo "FATAL: Jetson guard precheck '$g'"; exit 1; }
    $SSH jetson 'sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo guard started' 2>/dev/null
    echo "build   jetson: make -j2 llama-cli llama-server, CPU, g++-8 ($(date -Is))"
    $SSH jetson "cd ~/prima.cpp && t0=\$(date +%s); make -j2 llama-cli llama-server CC=gcc-8 CXX=g++-8 CPPFLAGS=-I\$HOME/prima-deps/include LDFLAGS=-L\$HOME/prima-deps/lib > ~/logs/prima_build_$STAMP.log 2>&1; echo jetson: EXIT_MAKE=\$? elapsed_s=\$(( \$(date +%s) - t0 ))"
    $SSH jetson "tail -4 ~/logs/prima_build_$STAMP.log; cd ~/prima.cpp && sha256sum llama-cli llama-server 2>/dev/null"
    scp -q jetson:logs/prima_build_$STAMP.log testbed/results/prima_build_${STAMP}_jetson_make.log
    $SSH jetson 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p stopped"; done; grep -c TRIP ~/heteroserve/logs/power-guard.log' 2>/dev/null
    ;;
  esac
done
echo "BUILD DONE $(date -Is)"
