#!/bin/bash
# TEST B -- the QR1 pooling proof.
# Qwen2.5-14B-Instruct Q4_K_M (8.37 GiB) across Pi 5 (main, CPU) + Pi 4 (RPC worker),
# over the WIRED 192.168.10.0/24 overlay.
#
# Run from the ThinkPad control node. The worker is started and stopped from here,
# which is where cluster orchestration belongs by design.
set -u
W_IP=192.168.10.12; W_PORT=50052
M='~/models/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf'
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=testbed/results/pool14b_${STAMP}.log
mkdir -p testbed/results
# tee -i: Ctrl-C signals the whole process group. A plain tee dies of it, the EXIT trap's
# first write hits the closed pipe, and bash dies of SIGPIPE before the cleanup ssh runs.
# -i covers SIGINT only: a group SIGHUP (terminal closed), SIGTERM or SIGQUIT (Ctrl-\) kills
# the tee just the same and takes the cleanup's report with it, so the tee starts with all
# four ignored. SIGPIPE too: with the output piped on (`2>&1 | tee raw.txt`, `| less`), a
# reader that Ctrl-C killed or that was quit early kills the tee at its next write, and the
# report (or the rest of the run) with it. Ignored, that write fails and the log goes on.
exec > >(trap '' INT TERM HUP QUIT PIPE; exec tee -i -a "$OUT") 2>&1
# bash itself IGNORES SIGQUIT: untrapped, Ctrl-\ ends only the ssh it waits on, and the run
# carries on -- into the next cell. Trapped from here, it stops the run wherever it lands
# (through the cleanup, once the EXIT trap below is set).
trap 'exit 131' QUIT

echo "########## $(date -Is)  TEST B: 14B Q4_K_M pooled over wired RPC"
echo "pi5 avail MiB : $(ssh -o BatchMode=yes pi5 'free -m | awk "/Mem:/{print \$7}"')"
echo "pi4 avail MiB : $(ssh -o BatchMode=yes pi4 'free -m | awk "/Mem:/{print \$7}"')"
echo "commit (pi5)  : $(ssh -o BatchMode=yes pi5 'cd ~/llama.cpp && git rev-parse HEAD')"
echo "commit (pi4)  : $(ssh -o BatchMode=yes pi4 'cd ~/llama.cpp && git rev-parse HEAD')"

# Never load a busy board: rpc_smoke_pi5_jetson.sh uses the Pi 5 as its main node too, and
# the worker start below kills ANY ggml-rpc-server on the Pi 4, whoever started it. Checked
# before the EXIT trap is set: the cleanup kills unconditionally, and must not kill a worker
# this run refused to touch. An unreachable board fails the check as well. So does a
# llama.cpp rebuild (cmake --build, cc1plus): it may be relinking the binaries started
# below, and a -j2 compile beside ~6 GiB of weights on the swapless Pi 4 invites the OOM
# killer.
for h in pi5 pi4; do
  B=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 $h 'for p in llama-bench llama-cli llama-server ggml-rpc-server cc1plus cmake; do pgrep -x $p >/dev/null && printf "%s " $p; done; true' 2>&1) || B+=" (ssh exit $?)"
  [ -z "$B" ] || { echo "FATAL: $h is busy or unreachable -- not loading it: $B"; exit 1; }
done
echo "pi5, pi4      : no llama-bench / llama-cli / llama-server / ggml-rpc-server / rebuild running"

# NOTE: pkill -f would match this launcher's OWN bash -c line (it contains the string
# "ggml-rpc-server"), killing the shell before it can start the worker. Use -x, which
# matches the process NAME exactly.
# The ssh runs before any echo, with PIPE/INT/TERM/HUP/QUIT ignored: neither a dead tee nor
# a 2nd Ctrl-C can cut it short. It fails (exit != 0) only if a worker is still running
# afterwards or pi4 is unreachable -- a worker that had already died is not a failure.
# The worker start has no pty: interrupted in its first second, its remote side still
# spawns the worker ~1 s later, after the 1st pkill found nothing -- hence the 2nd one. A
# start ssh still connecting (it outlives a SIGTERM sent to bash alone) is stopped first.
# ConnectTimeout bounds only the connect; ServerAlive ends a session that stalls after it
# (~10 s), which INT/TERM/HUP/QUIT, ignored here, no longer can. If ssh to `pi4` (its DHCP
# address, via the router) fails outright (255), the same command is retried at the static
# overlay address, which needs neither the lease nor the alias; HostKeyAlias checks it
# against the host key already known for the alias.
# A run that left the worker up does not exit 0 (as in rpc_smoke_pi5_jetson.sh): the EXIT
# trap passes the status in, and a run that would have exited 0 exits 5 instead.
P4_KEYHOST=$(ssh -G pi4 2>/dev/null | awk '$1 == "hostname" {print $2; exit}')
CLEANED=0
cleanup() {
  trap '' PIPE INT TERM HUP QUIT
  [ "$CLEANED" = 1 ] && return 0
  CLEANED=1
  local out o2 rc cmd so=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
  pkill -P $$ -x ssh
  cmd='pkill -x ggml-rpc-server && echo "worker stopped" || echo "no worker was running"; sleep 1.5; pkill -x ggml-rpc-server && echo "2nd pkill hit a worker (still up, or started after the 1st)"; sleep 1; if pgrep -x ggml-rpc-server >/dev/null; then echo "WORKER STILL RUNNING"; exit 1; fi'
  out=$(ssh "${so[@]}" pi4 "$cmd" 2>&1); rc=$?
  if [ "$rc" = 255 ] && [ -n "$P4_KEYHOST" ] && [ "$P4_KEYHOST" != "$W_IP" ]; then
    o2=$(ssh "${so[@]}" -o HostName=$W_IP -o HostKeyAlias=$P4_KEYHOST pi4 "$cmd" 2>&1); rc=$?
    out+=$'\n'"retry over the wired overlay ($W_IP):"$'\n'"$o2"
  fi
  echo "--- stopping worker ---"
  echo "$out"; echo "cleanup ssh exit=$rc"
  [ "$rc" = 0 ] || echo "!!!!!!!!!! CLEANUP FAILED -- ggml-rpc-server may still be running on the Pi 4; stop it by hand: ssh pi4 'pkill -x ggml-rpc-server'"
  # An exit in the EXIT trap sets the final status. A non-zero status or signal death stays.
  if [ "$rc" != 0 ] && [ "${1:-0}" = 0 ]; then exit 5; fi
}
trap 'cleanup $?' EXIT
# ssh CATCHES SIGINT and exits 255, and bash then runs the next command (it stops only for
# a child that died OF the signal): without this, Ctrl-C mid-cell would start the next cell.
trap 'exit 130' INT
# HUP needs one too: untrapped, a group SIGHUP (terminal closed) that lands during a
# $(a | b), as in thermals(), kills bash 5.2 WITHOUT running the EXIT trap -- the worker
# stays up and the log never says so.
trap 'exit 129' HUP

echo; echo "--- starting RPC worker on the Pi 4, bound to the wired interface ---"
ssh -o BatchMode=yes pi4 "mkdir -p ~/logs; pkill -x ggml-rpc-server; sleep 1; \
  setsid nohup ~/llama.cpp/build/bin/ggml-rpc-server -H $W_IP -p $W_PORT -t 4 \
    > ~/logs/rpc-server.log 2>&1 < /dev/null & sleep 3; \
  echo worker-pid=\$(pgrep -x ggml-rpc-server | head -1)"

# DO NOT probe the worker with a bare TCP connect (/dev/tcp, nc -z). ggml-rpc-server
# listens with backlog 1 and a connect-then-close KILLS IT -- observed 2026-09-20, the
# worker printed its banner and vanished. Check the listening socket from the worker's
# own side instead, which touches nothing.
echo "--- waiting for the worker to listen (checked via ss, no client connect) ---"
for i in $(seq 1 20); do
  if ssh -o BatchMode=yes pi4 "ss -ltn 2>/dev/null | grep -q ':$W_PORT '"; then
    echo "worker listening on $W_IP:$W_PORT after ${i}s"; break
  fi
  sleep 1
  if [ "$i" = 20 ]; then
    echo "FATAL: worker never came up"; ssh -o BatchMode=yes pi4 'cat ~/logs/rpc-server.log'; exit 1
  fi
done
ssh -o BatchMode=yes pi4 'tail -5 ~/logs/rpc-server.log'

echo; echo "--- devices as the main node sees them ---"
ssh -o BatchMode=yes pi5 "~/llama.cpp/build/bin/llama-bench -rpc $W_IP:$W_PORT --list-devices 2>&1 | head -15"

# House rule (CLAUDE.md): log vcgencmd get_throttled with every Pi run. Both boards, before
# AND after each cell, so a slow cell can be told apart from a hot one. get_throttled bit 19
# (0x80000) is STICKY until reboot: a 0x0 -> 0x80000 change across one cell pins the event
# to that cell, but once set it says nothing about any later cell. Reboot before a sweep,
# and to certify every cell, poll bit 3 (0x8, limit active NOW) during it -- these
# snapshots bracket a cell and cannot see its peak (2026-09-25: flag raised, no snapshot
# above 71.5 C).
# One snapshot per assignment, not two $(...) in one word: a trapped signal landing in the
# first of two makes bash 5.2 misparse its trap ("unexpected EOF") and carry on -- Ctrl-C
# there would start the next cell. The printed line is unchanged.
thermals() {
  local p5 p4
  p5=$(ssh -o BatchMode=yes pi5 'vcgencmd measure_temp; vcgencmd get_throttled' | tr '\n' ' ')
  p4=$(ssh -o BatchMode=yes pi4 'vcgencmd measure_temp; vcgencmd get_throttled' | tr '\n' ' ')
  echo "$1 pi5 $p5| pi4 $p4"
}

# The gate again at every cell boundary: a job another session started after the check above
# (a rebuild, a benchmark) is never loaded beside, and the cell it may have overlapped is
# named -- the last one too. Same list, but the Pi 4's ggml-rpc-server is this run's own
# worker, so only the Pi 5 is checked for one. $1: when; $2: what a busy board means then.
regate() {
  local h L B
  for h in pi5 pi4; do
    L='llama-bench llama-cli llama-server cc1plus cmake'; [ $h = pi5 ] && L+=' ggml-rpc-server'
    B=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 $h "for p in $L; do pgrep -x \$p >/dev/null && printf '%s ' \$p; done; true" 2>&1) || B+=" (ssh exit $?)"
    [ -z "$B" ] || { echo "FATAL: $h became busy or unreachable $1 -- $2: $B"; exit 1; }
  done
  echo "pi5, pi4      : still free $1"
}

PREV=
for NGL in ${NGL_LIST:-24 30 36}; do
  # b23701f: -ngl counts the output layer, and offloads it first: ngl=N puts the output head and the
  # LAST N-1 blocks on the worker, and keeps the first 49-N blocks and the token embedding on the Pi 5.
  echo; echo "##### ngl=$NGL of 49 offloadable layers (48 blocks + the output layer, which goes first) on the remote worker, remainder on the Pi 5 CPU"
  regate "before ngl=$NGL" "not loading it${PREV:+, and ngl=$PREV may have shared the board}"
  thermals "before:"
  # -r 3, never -r 1: -r 1 prints "± 0.00", which looks like precision but is ONE sample.
  # -oe jsonl puts each repetition's own samples (samples_ts) on stderr, so the reported
  # spread comes from the repetitions themselves rather than llama-bench's summary ±.
  ssh -o BatchMode=yes pi5 "cd ~ && /usr/bin/timeout 2400 ~/llama.cpp/build/bin/llama-bench -m $M -rpc $W_IP:$W_PORT -ngl $NGL -t 4 -p 64 -n 16 -r 3 -oe jsonl; echo EXIT_ngl${NGL}=\$?"
  r=$?
  # The remote command ends in an echo, so non-zero is ssh's own 255: the laptop lost the
  # Pi 5 (mid-cell: after ServerAlive, ~90 s), and with no pty its llama-bench runs on. Stop
  # now (the next gate would only see that orphan as a busy board); the cleanup stops the
  # worker, which ends that llama-bench too. (Ctrl-C, Ctrl-\ and a hangup exit before this.)
  [ "$r" = 0 ] || { echo "FATAL: lost the Pi 5 mid-cell (ssh exit $r) -- its llama-bench may still be running; stopping"; exit 1; }
  echo "pi5 mem: $(ssh -o BatchMode=yes pi5 'free -m | sed -n 2p')"
  echo "pi4 mem: $(ssh -o BatchMode=yes pi4 'free -m | sed -n 2p')"
  thermals "after: "
  PREV=$NGL
done
regate "after ngl=$PREV, the last cell" "it may have shared the board"

echo; echo "########## done $(date -Is)"
echo "OUTFILE=$OUT"
