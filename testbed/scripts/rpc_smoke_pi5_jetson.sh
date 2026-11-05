#!/bin/bash
# FUNCTIONAL SMOKE TEST, not a characterisation: can the Jetson join a pooled run on the
# cluster pin? Pi 5 as main node (CPU), Jetson as ggml-rpc-server worker exposing its GPU
# through Vulkan (Path 1). It also takes a MATCHED LOCAL BASELINE: the RPC cells' exact
# llama-bench settings on the Jetson alone (its own llama-bench on Vulkan0, the device the
# worker exposes; no RPC), in the order local1 -> ngl=99 -> ngl=11 -> local2. That A-B-A
# bracket's two ends give the drift (~2% for one config minutes apart) an RPC-minus-local
# difference has to be judged against, and the Jetson's clocks are LOCKED (jetson_clocks)
# for the whole run, so its GPU cannot clock down while it waits on the network. The script
# does not compute the comparison: that is done afterwards, from the raw records in the log.
# There is no mixed-revision hazard only because both ~/llama.cpp trees sit AT the pin in
# ThinkPad/config/cluster.json, unmodified, and every binary the run starts is tied to it:
# the Jetson's worker and llama-bench by the build_info.json their build wrote, the Pi 5's
# llama-bench by its own --version -- checked below, exit 2 if not -- and every cell's jsonl
# rows by their build_commit (BUILD-MISMATCH if not).
#
# Run from the ThinkPad control node, from the repo root. The Jetson gets a TEMPORARY
# address on the wired overlay (192.168.10.13, `ip addr add`, gone on reboot), the subnet
# the Pis' RPC runs use. Until 2026-09-26 the Pi 5 would otherwise have reached the Jetson
# over Wi-Fi; the boards are wired-only since (the ssh aliases are their router-LAN eth0
# addresses, 192.168.100.x), but the overlay stays the one RPC path. The address alone
# does not guarantee the wire (ARP flux, below), so the path is proven by MAC before the
# first load and by the Jetson's per-NIC byte counters around every cell.
#
# Carries the traps from run_14b_pooled.sh: readiness via `ss` on the worker's own side (a
# bare TCP connect-then-close kills ggml-rpc-server), processes found by exact name (`pgrep
# -x`) or a [b]racketed pattern, never a plain `pkill -f`, and no laptop-side command with
# two $(...): bash 5.2 misparses a trap that fires while it waits on the first of two, and
# then aborts (EXIT trap and all) or carries on past the Ctrl-C. One per assignment, then print.
#
# Nothing loads the Jetson until both trees are clean at the pin, both boards' model bytes
# match the full SHA-256 in testbed/models.md, both boards are idle and the Jetson's power
# guard is live and armed -- and each load (the clock lock, the worker, --list-devices, every
# cell) is gated again on evidence taken just before it, the Jetson's inside the very ssh
# that locks or launches. Each change made on the Jetson (clock lock, worker, address, ARP
# sysctls) is flagged BEFORE it is made and undone by the EXIT trap -- only what this run
# changed. The flags also go into a run record on the Jetson (~/heteroserve/run/rpc_smoke.state)
# ahead of each change, and only a cleanup that succeeded deletes it: after a run that could
# not clean up (SIGKILL, crash, laptop off the network) the next run refuses to start and
# prints the undo, rather than taking the leftovers for the board's own settings. The record
# is the lock as well: created exclusively, it keeps a second run of this script off the
# board, and it names the laptop and pid that hold it, so a live run is not taken for a
# crashed one (and handed an undo that would pull its worker mid-cell).
# Exit: 0 all cells ok | 1 fatal, a cell not ok, or an earlier run's record is still on the
# Jetson | 2 revision mismatch, a tree not clean, or a binary not provably built at the pin
# | 3 power guard not live, or wrong power mode | 4 a cell tripped (or lost) the guard | 5
# otherwise fine, but the cleanup failed and a board may still carry this run's changes.
set -u
J_IP=192.168.10.13; J_PORT=50052; P5_IP=192.168.10.11
J_ETH_MAC=00:04:4b:ec:52:f2; J_WLAN_MAC=40:1c:83:41:38:9a; P5_ETH_MAC=88:a2:9e:f0:e7:8a
M='~/models/tinyllama-1.1b-q4km.gguf'
BENCH='-t 4 -p 64 -n 16 -r 3 -oe jsonl'   # every cell's llama-bench settings, RPC and local alike
LOCAL_ARGS="-dev Vulkan0 -ngl 99 ${BENCH}"   # a local cell: Vulkan0 (the worker's -d), every layer on it
EXPECT_PMODE=${EXPECT_PMODE:-pmode:0000}  # MAXN -- the mode the guard is there to protect
GUARD_MAX_AGE=5                           # s; the guard writes a sample about once a second
WLAN_MAX=1048576                          # B on the Jetson's wlan0 in one cell = not proven wired
# A board is busy when one of these runs, matched in /proc/*/cmdline. Bracketed
# ([l]lama-bench): the ssh's own shell carries the pattern too, and must not match itself.
# The Jetson runs its own jobs (jetson_chain_build_bench.sh: build, then a clock-locked
# Vulkan benchmark; a CUDA build queued after it), and the Pi 5 has its own sessions (a
# llama.cpp run, a rebuild, a second RPC worker). This run's worker matches neither; on the
# Jetson any OTHER ggml-rpc-server is busy too (jstate, JGATE: told apart by its argv).
BUSY_J='[l]lama-(bench|cli|server)|[n]vcc|[c]c1plus|[c]make|[j]etson_(chain|bench|build)|[v]k_fp16_probe'
BUSY_P5='[l]lama-(bench|cli|server)|[g]gml-rpc-server|[n]vcc|[c]c1plus|[c]make'
SSHO=(-n -o BatchMode=yes -o ConnectTimeout=8)
jssh() { ssh "${SSHO[@]}" jetson "$@"; }
pssh() { ssh "${SSHO[@]}" pi5 "$@"; }
# CLKS: the Jetson's clock policy as one line, the same text wherever it is read -- the
# snapshots, the lock's check, the undo's: GPU and CPU0 governor, min, max, then the two
# other settings jetson_clocks changes that read without root, GPU rail gating and fan mode.
CLKS='k1=/sys/devices/57000000.gpu; k2=/sys/devices/system/cpu/cpu0/cpufreq; echo "gpu $(cat $k1/devfreq/57000000.gpu/governor) $(cat $k1/devfreq/57000000.gpu/min_freq) $(cat $k1/devfreq/57000000.gpu/max_freq) cpu0 $(cat $k2/scaling_governor) $(cat $k2/scaling_min_freq) $(cat $k2/scaling_max_freq) railgate $(cat $k1/railgate_enable) fanctl $(cat /sys/devices/pwm-fan/temp_control)"'
RE_CLK='^gpu [A-Za-z0-9_]+ [0-9]+ [0-9]+ cpu0 [A-Za-z0-9_]+ [0-9]+ [0-9]+ railgate [0-9]+ fanctl [0-9]+$'

[ -f ThinkPad/config/cluster.json ] && [ -f testbed/models.md ] || { echo "run from the repo root"; exit 1; }
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=testbed/results/rpc_smoke_pi5_jetson_${STAMP}.log
REC='~/heteroserve/run/rpc_smoke.state'             # the run record on the Jetson (see header)
CLKF="~/heteroserve/run/jetson_clocks_${STAMP}.stored"   # this run's jetson_clocks --store there
WLOG="~/heteroserve/logs/rpc-server_${STAMP}.log"   # this run's worker log there; no run reuses it
P5PID=/tmp/rpc_smoke_${STAMP}.pid                   # on the Pi 5: the running cell's timeout pid
JPID=/tmp/rpc_smoke_${STAMP}.pid                    # on the Jetson: the running local cell's, likewise
mkdir -p testbed/results
# tee -i: Ctrl-C signals the whole process group. A plain tee dies of it, the EXIT trap's
# first write hits the closed pipe, and bash dies of SIGPIPE before the cleanup ssh runs.
# -i covers SIGINT only: a group SIGTERM/SIGHUP does the same (bash's own "Terminated"
# notice is the fatal write), and so does SIGQUIT (Ctrl-\), so the tee starts with all four
# ignored -- and SIGPIPE, as run_14b_pooled.sh's does: with the output piped on (`| less`),
# a reader quit early would kill the tee, and the rest of the log with it. fd 3 keeps the
# terminal for the cleanup's report in case the tee dies anyway.
exec 3>&1
exec > >(trap '' INT TERM HUP QUIT PIPE; exec tee -i -a "$OUT") 2>&1
TEE_PID=$!

# What this run has changed on the Jetson. Each flag is set BEFORE its change is issued, so
# a change interrupted half-way is still undone; nothing unflagged is ever touched. The run
# record carries the same flags; REC_WRITTEN means it may exist, HELD that this run made it.
WORKER_STARTED=0; ADDR_ADDED=0; ADDR_PRESENT=0; SYSCTL_RESTORE=""; REC_WRITTEN=0; HELD=0
CLOCKS_LOCKED=0; CLK_PRE=""; CLK_PMODE=""
CELL_RUNNING=0; LOCAL_RUNNING=0; CLEANED=0; WLOG_OFF=0
J_KEYHOST=$(ssh -G jetson 2>/dev/null | awk '$1 == "hostname" {print $2; exit}')

# rec [claim]: the Jetson command that (re)writes the run record from the current flags. Each
# change is chained behind it with &&, so none is made unless the record that covers it was
# written. The record is the lock too: the first write (claim) creates it exclusively (`set
# -C` opens it O_EXCL) and exits 7 if one is there -- another run of this script got to the
# board since the check below -- and each later write replaces it only while it is still
# this run's own (its stamp): a record removed or taken over stops the next change.
rec() {
  local w="printf \"stamp=%s\\nhost=%s\\npid=%s\\nsysctl_restore=%s\\naddr_added=%s\\nworker_started=%s\\nclocks_locked=%s\\nclocks_pre=%s\\nclocks_pmode=%s\\n\" ${STAMP} ${HOSTNAME} $$ \"${SYSCTL_RESTORE}\" ${ADDR_ADDED} ${WORKER_STARTED} ${CLOCKS_LOCKED} \"${CLK_PRE}\" \"${CLK_PMODE}\""
  if [ "${1:-}" = claim ]; then printf '%s' "mkdir -p ~/heteroserve/run && { (set -C; ${w} > ${REC}) || exit 7; }"
  else printf '%s' "grep -qx stamp=${STAMP} ${REC} && ${w} > ${REC}.${STAMP}.tmp && mv -f ${REC}.${STAMP}.tmp ${REC}"; fi
}

# undo_cmd WORKER ADDR_ADDED "SYSCTL_RESTORE" alias|wired STAMP HELD CLOCKS "CLK_PRE" CLK_PMODE
# [LOCAL]: the Jetson command that undoes run STAMP's changes, built from its flags. First
# that run's local llama-bench (LOCAL=1: a cell cut short), by its pid file, as on the Pi 5.
# The rest only while the run record is STAMP's own, or gone: a record another run holds
# means that run's worker, address and settings are live, and nothing is touched -- a
# failure if STAMP did hold the board (HELD=1). Worker first, then the clocks, the ARP
# sysctls, the address LAST: over the wired fallback the session itself runs on .13, and once
# the address is gone the Jetson can send nothing more on it (the kernel will not route from
# a non-local source), so there the removal is detached and 2 s late, after the session has
# returned. The run record (and the clock snapshot) goes only if every step succeeded -- over
# the fallback, with the address. Every step is safe to repeat, and there is no single quote
# inside: it is printed for pasting.
undo_cmd() {
  local c='f=0; ' k="~/heteroserve/run/jetson_clocks_${5}.stored" kr=""
  if [ "${10:-0}" = 1 ]; then
    c+='for t in $(seq 1 8); do [ -s '"$JPID"' ] && break; sleep 0.25; done; p=$(cat '"$JPID"' 2>/dev/null); '
    c+='if [ -n "$p" ] && tr "\0" " " < /proc/$p/cmdline 2>/dev/null | grep -qF -- "'"$LOCAL_ARGS"'"; then kill $p; '
    c+='for t in $(seq 1 20); do [ -d /proc/$p ] || break; sleep 0.25; done; '
    c+='if [ -d /proc/$p ]; then echo "LOCAL LLAMA-BENCH OF THIS RUN STILL RUNNING, timeout pid $p"; f=1; else echo "local llama-bench of this run stopped, timeout pid $p"; fi; '
    c+='else echo "no local llama-bench of this run running"; fi; rm -f '"$JPID"'; '
  fi
  c+='o=$(sed -n "s/^stamp=//p" '"$REC"' 2>/dev/null | head -1); if [ -n "$o" ] && [ "$o" != '"$5"' ]; then '
  if [ "$6" = 1 ]; then c+='echo "NOTHING UNDONE: the run record there is now run $o, not '"$5"' -- its worker, address and settings may be live"; f=1; '
  else c+='echo "nothing undone: the run record there is run $o -- this run never held the board"; '; fi
  c+='else '
  if [ "$1" = 1 ]; then
    # This run's worker, by its argv (the .13 bind), until none has been seen for 2 s: a
    # launch cut short by Ctrl-C keeps running on the Jetson (no tty) and can spawn the
    # worker after the first look. TERM, KILL after 3 s, 10 s at most.
    c+='wp() { for p in $(pgrep -x ggml-rpc-server); do tr "\0" " " < /proc/$p/cmdline 2>/dev/null | grep -qF -- "-H '"$J_IP"' -p '"$J_PORT"' " && echo $p; done; }; '
    c+='n=0; q=0; for t in $(seq 1 40); do p=$(wp); if [ -n "$p" ]; then n=$((n + 1)); q=0; if [ $n -le 12 ]; then kill $p; else kill -9 $p; fi 2>/dev/null; else q=$((q + 1)); [ $q -ge 8 ] && break; fi; sleep 0.25; done; '
    c+='if [ -n "$(wp)" ]; then echo "WORKER STILL RUNNING"; f=1; elif [ $n = 0 ]; then echo "no worker of this run running"; else echo "worker stopped"; fi; '
  fi
  if [ "$7" = 1 ]; then
    # The clocks, from this run's own --store snapshot, and only in the power mode it was
    # taken in: after a guard trip (nvpmodel -m 1, 5 W) the MAXN snapshot would put the CPU
    # cores 5 W took offline back online. Checked by reading the policy back: it must be as
    # before the lock. No snapshot means no lock was made (it is chained behind a complete
    # one) -- unless the policy still reads LOCKED.
    kr=" ${k}"
    c+='if [ -s '"$k"' ]; then m=$(cut -d" " -f1 /var/lib/nvpmodel/status 2>/dev/null); '
    c+='if [ "$m" != '"$9"' ]; then echo "CLOCKS NOT RESTORED: the power mode is $m, not '"$9"', the mode they were stored in (a guard trip?) -- restoring that snapshot now would undo part of it; restore once the board is back in '"$9"'"; f=1; '
    c+='elif ! sudo -n jetson_clocks --restore '"$k"'; then echo "CLOCKS NOT RESTORED: jetson_clocks --restore failed"; f=1; '
    c+='else n=$('"$CLKS"'); if [ "$n" = "'"$8"'" ]; then echo "clocks restored, as before the run: $n"; else echo "CLOCKS NOT RESTORED AS STORED: now $n, before the run '"$8"'"; f=1; fi; fi; '
    c+='else n=$('"$CLKS"'); read -r _ _ a b _ _ x y _ <<< "$n"; if [ -n "$a" ] && [ "$a" = "$b" ] && [ "$x" = "$y" ]; then echo "CLOCKS STILL LOCKED, and no snapshot '"$k"' to restore them from: $n"; f=1; else echo "clocks not locked by this run (no snapshot '"$k"'): $n"; fi; fi; '
  fi
  [ -n "$3" ] && c+='if sudo -n sysctl -w'"$3"'; then echo "ARP sysctls restored"; else echo "SYSCTLS NOT RESTORED"; f=1; fi; '
  local has='ip -4 -o addr show dev eth0 | grep -qF "inet '"$J_IP"'/"'
  local unrec='if [ $f = 0 ]; then if [ ! -e '"$REC"' ]; then echo "run record already gone"; elif grep -qx stamp='"$5"' '"$REC"'; then if rm -f '"$REC$kr"'; then echo "run record removed"; else echo "RUN RECORD NOT REMOVED"; f=1; fi; else echo "run record left alone: it is now another run"; fi; else echo "run record kept: a step above failed"; fi; '
  if [ "$2" = 1 ] && [ "$4" = wired ]; then
    c+='if '"$has"'; then setsid sh -c "sleep 2; sudo -n ip addr del '"$J_IP"'/24 dev eth0 && [ $f = 0 ] && grep -qx stamp='"$5"' '"$REC"' && rm -f '"$REC$kr"'" < /dev/null > /dev/null 2>&1 & '
    c+='echo "address '"$J_IP"' removal scheduled, detached, in 2 s (this session runs over it), so it is unconfirmed; the run record goes with it if every step above succeeded"; '
    c+='else echo "address '"$J_IP"' not present"; '"$unrec"'fi; '
  else
    [ "$2" = 1 ] && c+='if '"$has"'; then if sudo -n ip addr del '"$J_IP"'/24 dev eth0; then echo "address '"$J_IP"' removed"; else echo "ADDRESS NOT REMOVED"; f=1; fi; else echo "address '"$J_IP"' not present"; fi; '
    c+=$unrec
  fi
  printf '%s' "${c}"'fi; echo "undo exit status: $f"; exit $f'
}

# p5_stop_cmd: the Pi 5 command that stops THIS run's cell, by the pid the cell wrote (the
# timeout wrapper, which passes the TERM on to llama-bench), checked against this run's argv
# first -- never by name: the Pi 5 has its own sessions, and their llama-bench is not ours.
p5_stop_cmd() {
  local c='for t in $(seq 1 8); do [ -s '"$P5PID"' ] && break; sleep 0.25; done; p=$(cat '"$P5PID"' 2>/dev/null); '
  c+='if [ -n "$p" ] && tr "\0" " " < /proc/$p/cmdline 2>/dev/null | grep -qF -- "-rpc '"${J_IP}:${J_PORT}"' -ngl "; then kill $p; '
  c+='for t in $(seq 1 20); do [ -d /proc/$p ] || break; sleep 0.25; done; '
  c+='if [ -d /proc/$p ]; then echo "LLAMA-BENCH OF THIS RUN STILL RUNNING, timeout pid $p"; exit 1; fi; echo "llama-bench of this run stopped, timeout pid $p"; '
  c+='else echo "no llama-bench of this run running"; fi; rm -f '"$P5PID"
  printf '%s' "$c"
}

# status_of OUTPUT: the status the undo reported, if its last line arrived
status_of() { sed -n 's/^undo exit status: \([0-9]*\)$/\1/p' <<<"$1" | tail -1; }

cleanup() {
  trap '' PIPE INT TERM HUP QUIT   # nothing may cut the undo short: not a dead tee, not a 2nd Ctrl-C
  local status=$1 cmd="" wcmd="" out="" o2="" rc=0 s wired=0 how="ssh jetson" pre="" p5cmd="" p5out="" p5rc=0
  [ "$CLEANED" = 1 ] && return 0
  CLEANED=1
  if [ "$REC_WRITTEN" = 1 ] || [ "$WORKER_STARTED" = 1 ] || [ "$ADDR_ADDED" = 1 ] || [ -n "$SYSCTL_RESTORE" ] \
     || [ "$CLOCKS_LOCKED" = 1 ] || [ "$LOCAL_RUNNING" = 1 ]; then
    # First the worker's output the log does not have yet -- the cell a Ctrl-C cut short.
    [ "$WORKER_STARTED" = 1 ] && pre="echo \"--- worker log from byte ${WLOG_OFF} of ${WLOG} ---\"; tail -c +$((WLOG_OFF + 1)) ${WLOG} | awk 1; echo \"--- end of worker log ---\"; "
    cmd=$pre$(undo_cmd "$WORKER_STARTED" "$ADDR_ADDED" "$SYSCTL_RESTORE" alias "$STAMP" "$HELD" "$CLOCKS_LOCKED" "$CLK_PRE" "$CLK_PMODE" "$LOCAL_RUNNING")
    # The ssh runs before any echo: if stdout is already gone, the undo has still happened.
    out=$(ssh "${SSHO[@]}" -o ServerAliveInterval=5 -o ServerAliveCountMax=2 jetson "$cmd" 2>&1); rc=$?
    # A session that dropped after the undo reported its status did not fail the undo.
    s=$(status_of "$out"); if [ "$rc" = 255 ] && [ -n "$s" ]; then rc=$s; how+=" -- the session dropped after the undo reported"; fi
    if [ "$rc" = 255 ] && [ -n "$J_KEYHOST" ] && [ "$ADDR_PRESENT" = 1 ]; then
      # `jetson` is its router-LAN address (192.168.100.111 since 2026-09-26). The laptop is
      # on the wired overlay too (enp0s31f6 192.168.10.5), so retry at .13 while it exists.
      # HostKeyAlias checks the host key already known for the alias rather than trusting a
      # new one.
      wired=1; how="ssh jetson via ${J_IP}, after ssh jetson failed: $(tail -1 <<<"$out")"
      wcmd=$pre$(undo_cmd "$WORKER_STARTED" "$ADDR_ADDED" "$SYSCTL_RESTORE" wired "$STAMP" "$HELD" "$CLOCKS_LOCKED" "$CLK_PRE" "$CLK_PMODE" "$LOCAL_RUNNING")
      o2=$(ssh "${SSHO[@]}" -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -o HostName=${J_IP} -o HostKeyAlias=${J_KEYHOST} jetson "$wcmd" 2>&1); rc=$?
      s=$(status_of "$o2"); if [ "$rc" = 255 ] && [ -n "$s" ]; then rc=$s; how+=" -- the session dropped after the undo reported"; fi
      out+=$'\n'"$o2"
    fi
  fi
  # A cell cut short leaves its llama-bench running on the Pi 5 against a dead worker.
  if [ "$CELL_RUNNING" = 1 ]; then
    p5cmd=$(p5_stop_cmd)
    p5out=$(ssh "${SSHO[@]}" -o ServerAliveInterval=5 -o ServerAliveCountMax=2 pi5 "$p5cmd" 2>&1); p5rc=$?
  fi
  # The report must reach the log whatever ended the run: leave the tee (it may be dead,
  # e.g. killed outright), give it up to 2 s to flush, then write through a fresh one.
  exec >&3 2>&3
  for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$TEE_PID" 2>/dev/null || break; sleep 0.2; done
  {
    echo; echo "--- cleanup (script exit ${status}) ---"
    if [ -z "$cmd" ]; then echo "jetson: nothing to undo -- this run changed nothing there"
    else sed 's/^/jetson: /' <<<"$out"; echo "cleanup ssh (${how}) exit=${rc}"; fi
    [ "$CELL_RUNNING" = 1 ] && { sed 's/^/pi5: /' <<<"$p5out"; echo "cleanup ssh (ssh pi5) exit=${p5rc}"; }
    if [ "$rc" != 0 ]; then
      echo "!!!!!!!!!! CLEANUP FAILED -- the Jetson may still carry this run's worker / clock lock / address / ARP sysctls"
      echo "!!!!!!!!!! (or its local llama-bench); its run record ${REC} stays there, so the next run refuses to start until"
      echo "!!!!!!!!!! this is undone."
      # The hint names the path that answered. After the wired fallback it names both: that
      # path lasts only as long as .13, which its own undo removes 2 s later.
      if [ "$wired" = 1 ]; then
        echo "!!!!!!!!!! undo by hand over the wired overlay, while ${J_IP} is still there: ssh -o HostName=${J_IP} -o HostKeyAlias=${J_KEYHOST} jetson '${wcmd}'"
        echo "!!!!!!!!!! or, once ssh jetson answers again:"
      fi
      echo "!!!!!!!!!! undo by hand: ssh jetson '${cmd}'"
    fi
    if [ "$p5rc" != 0 ]; then
      echo "!!!!!!!!!! CLEANUP FAILED -- this run's llama-bench may still be running on the Pi 5"
      echo "!!!!!!!!!! undo by hand: ssh pi5 '${p5cmd}'"
    fi
  } 2>&1 | tee -a "$OUT"
  # A run that left a board dirty does not exit 0.
  if [ "$rc" != 0 ] || [ "$p5rc" != 0 ]; then [ "$status" = 0 ] && exit 5; exit "$status"; fi
}
trap 'cleanup $?' EXIT
# PIPE too: should the tee die anyway, a write to it then runs the cleanup instead of killing bash.
# QUIT as in run_14b_pooled.sh: bash itself IGNORES SIGQUIT, so untrapped, Ctrl-\ ends only the
# ssh it waits on -- a cell's, whose llama-bench then runs on unwatched -- and the run carries on.
trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 129' HUP; trap 'exit 141' PIPE; trap 'exit 131' QUIT

kv() { sed -n "s/^$2=//p" <<<"$1" | head -1; }     # value of key $2 in a key=value snapshot $1
hms() { [[ "$1" =~ ^([0-9]{2}):([0-9]{2}):([0-9]{2}) ]] && echo $((10#${BASH_REMATCH[1]}*3600 + 10#${BASH_REMATCH[2]}*60 + 10#${BASH_REMATCH[3]})); }
mc() { [[ "$1" =~ ^[0-9]+$ ]] && echo "$(( $1 / 1000 )).$(( $1 % 1000 / 100 )) C" || echo "?(${1:-none})"; }
delta() { [[ "$1" =~ ^[0-9]+$ && "$2" =~ ^[0-9]+$ ]] && echo $(( $2 - $1 )) || echo "?"; }
lladdr() { awk '{for (i = 1; i < NF; i++) if ($i == "lladdr") {print $(i + 1); exit}}' <<<"$1"; }
thr() { local v; v=$(kv "$1" throttled); [[ "$v" =~ ^0x[0-9a-fA-F]+$ ]] && echo $(( v )); }   # get_throttled, empty if unreadable
# clk "GOV MIN MAX CUR" UNIT: a clock policy as read -- LOCKED when min == max (jetson_clocks
# engaged, as the board's own benchmark runs), else DVFS upward from min
clk() { local g mn mx c; read -r g mn mx c <<<"$1"; [[ "$mn" =~ ^[0-9]+$ && "$mx" =~ ^[0-9]+$ ]] || { echo "?(${1:-none})"; return; }
  if [ "$mn" = "$mx" ]; then echo "LOCKED $g $mn-$mx $2"; else echo "DVFS $g $mn-$mx $2, now $c"; fi; }
pol() { cut -d' ' -f1-3 <<<"$1"; }                  # a clock reading without its current frequency
# lk SNAPSHOT: 0 when the GPU's and CPU0's policies both read LOCKED
lk() { local g c; g=$(clk "$(kv "$1" gpu_clk)" Hz); c=$(clk "$(kv "$1" cpu0_clk)" kHz); [[ $g == LOCKED* && $c == LOCKED* ]]; }
treestate() { [ "$2" = 0 ] && [ -z "$1" ] && echo clean || echo "NOT CLEAN (exit $2): $(tr '\n' ' ' <<<"$1")"; }
# rowcheck OUTPUT NGL LOCAL: a cell's own jsonl rows, judged, as "LABELS|what they say". Every
# row from a binary at the pin (build_commit >= 7 hex and a prefix of it) -- else
# BUILD-MISMATCH; one pp64 and one tg16 row, each with the 3 repetitions asked for and
# n_gpu_layers as asked -- else RECORD-INCOMPLETE; a local cell's rows must also prove the GPU,
# the way summarise_jetson_bench.py's ran_on does: devices Vulkan0 (not none or auto), a GPU
# backend registered (backends names Vulkan), a GPU enumerated (gpu_info) -- else
# NOT-PROVEN-VULKAN0.
rowcheck() {
  awk -v pin="$PIN" -v ngl="$2" -v loc="$3" '
    function sv(k) { return match($0, "\"" k "\": \"[^\"]*\"") ? substr($0, RSTART + length(k) + 5, RLENGTH - length(k) - 6) : "?" }
    function nv(k) { return match($0, "\"" k "\": -?[0-9]+") ? substr($0, RSTART + length(k) + 4, RLENGTH - length(k) - 4) : "?" }
    function ns(   s, a, n, i, c) { if (!match($0, /"samples_ts": \[[^]]*\]/)) return 0
      s = substr($0, RSTART + 15, RLENGTH - 16); n = split(s, a, ","); for (i = 1; i <= n; i++) if (a[i] ~ /[0-9]/) c++; return c + 0 }
    /^\{"build_commit": / {
      r++; bc = sv("build_commit"); g = nv("n_gpu_layers"); p = nv("n_prompt"); n = nv("n_gen"); s = ns()
      cs = cs " " bc; gs = gs " " g; ps = ps " " p "/" n; ss = ss " " s
      if (bc !~ /^[0-9a-f]+$/ || length(bc) < 7 || index(pin, bc) != 1) bm = 1
      if (g != ngl || s != 3) inc = 1
      if (p == 64 && n == 0) pp++; else if (p == 0 && n == 16) tg++; else inc = 1
      if (loc) { d = sv("devices"); gi = sv("gpu_info"); be = sv("backends"); ds = ds " " d; gis = gis " [" gi "]"; bes = bes " " be
                 if (d != "Vulkan0" || gi ~ /^ *$/ || gi == "?" || be !~ /Vulkan/) nd = 1 }
    }
    END {
      if (pp != 1 || tg != 1) inc = 1
      if (loc && !r) nd = 1
      l = (bm ? ",BUILD-MISMATCH" : "") (inc ? ",RECORD-INCOMPLETE" : "") (nd ? ",NOT-PROVEN-VULKAN0" : "")
      printf "%s|%d jsonl rows; build_commit%s; n_gpu_layers%s; n_prompt/n_gen%s; samples_ts%s", substr(l, 2), r, cs, gs, ps, ss
      if (loc) printf "; devices%s; gpu_info%s; backends%s", ds, gis, bes
      print "" }' <<<"$1"
}

echo "########## $(date -Is)  RPC smoke: Pi 5 (main, CPU) + Jetson (RPC worker, Vulkan GPU), with a matched local baseline on the Jetson, clocks locked"

# --- same revision: the header's safety claim, enforced ---------------------------------
# cluster.json's llama.exceptions entry for the Jetson is about Jetson/config/node.env (its
# old CUDA ref), not the ~/llama.cpp tree on the board, which the Vulkan build put at the pin.
# The trees are what the RPC handshake runs on, so the trees are what is checked -- and "at
# the pin" means unmodified too, as jetson_build_llama_vulkan.sh demands before it builds.
# --untracked-files=all: a file `git apply` adds is untracked (the 2026-09-25 1206 handoff);
# build/ is gitignored, so a built tree is still clean.
PIN=$(python3 -c 'import json; print(json.load(open("ThinkPad/config/cluster.json"))["llama"]["pin_sha"])' 2>/dev/null)
C_P5=$(pssh 'git -C ~/llama.cpp rev-parse HEAD' 2>&1)
C_J=$(jssh 'git -C ~/llama.cpp rev-parse HEAD' 2>&1)
T_P5=$(pssh 'git -C ~/llama.cpp status --porcelain --untracked-files=all' 2>&1); TR_P5=$?
T_J=$(jssh 'git -C ~/llama.cpp status --porcelain --untracked-files=all' 2>&1); TR_J=$?
# The binaries are not the trees. The Jetson's two (the worker, and llama-bench for the
# local cells) are tied to the pin by the build_info.json the Vulkan build writes after it
# links: the commit it was built from, and binaries there and no newer than that file.
WB=$(jssh 'B=~/heteroserve/build_info.json
sed -n "s/^ *\"\(llama_cpp_commit\|backend\|built_at_utc\)\": *\"\([^\"]*\)\".*/\1=\2/p" $B
if [ -f $B ]; then for x in ggml-rpc-server llama-bench; do b=~/llama.cpp/build/bin/$x
  if [ ! -x $b ]; then r=missing; elif [ $b -nt $B ]; then r=yes; else r=no; fi; echo "${x//-/_}_newer_than_build_info=$r"; done; fi' 2>&1)
# The Pi 5's llama-bench has no build_info.json: its own --version names the commit it was
# built from (llama_print_build_info, on stderr), and that must be the pin's (>= 7 hex, a
# prefix of it). The RPC handshake checks only the protocol version (ggml-rpc.cpp), so a
# binary from a nearby revision would pool without a word.
V_P5=$(pssh '~/llama.cpp/build/bin/llama-bench --version' 2>&1)
VC_P5=$(sed -n 's/^version: .*(build [0-9]*, commit \([0-9a-f]*\))$/\1/p' <<<"$V_P5" | head -1)
TS_P5=$(treestate "$T_P5" "$TR_P5")
TS_J=$(treestate "$T_J" "$TR_J")
echo "pin (cluster.json): ${PIN:-UNREADABLE}"
echo "tree HEAD (pi5)   : $C_P5"
echo "tree HEAD (jetson): $C_J"
echo "tree   (pi5)      : $TS_P5"
echo "tree   (jetson)   : $TS_J"
echo "bench  (pi5)      : ~/llama.cpp/build/bin/llama-bench --version: $(tr '\n' ' ' <<<"$V_P5")"
echo "binaries (jetson) : ~/llama.cpp/build/bin/{ggml-rpc-server,llama-bench}; ~/heteroserve/build_info.json: $(tr '\n' ' ' <<<"$WB")"
if ! [[ "$PIN" =~ ^[0-9a-f]{40}$ ]] || [ "$C_P5" != "$PIN" ] || [ "$C_J" != "$PIN" ]; then
  echo "FATAL: both trees must be at the cluster pin -- refusing a mixed-revision pool"; exit 2
fi
if [ "$TS_P5" != clean ] || [ "$TS_J" != clean ]; then
  echo "FATAL: both trees must be clean at the pin -- an edited tree is not the pin"; exit 2
fi
WC=$(kv "$WB" llama_cpp_commit); WK=$(kv "$WB" backend)
WN1=$(kv "$WB" ggml_rpc_server_newer_than_build_info); WN2=$(kv "$WB" llama_bench_newer_than_build_info)
if [ "$WC" != "$PIN" ] || [ "$WK" != vulkan ] || [ "$WN1" != no ] || [ "$WN2" != no ]; then
  echo "FATAL: the Jetson's ggml-rpc-server or llama-bench is not provably built at the pin (its build_info.json, above)"; exit 2
fi
[[ ${#VC_P5} -ge 7 && "$PIN" == "$VC_P5"* ]] || { echo "FATAL: the Pi 5's llama-bench is not provably built at the pin (its --version, above)"; exit 2; }

# --- are both boards free? ---------------------------------------------------------------
# A GPU worker started under one of the Jetson's own jobs corrupts that job's
# characterisation figures and this test's own, and stacks two loads on the micro-USB
# supply. A run record left by an earlier run means its changes may still be on the board
# -- or that the run is still going: the record names its laptop and pid, and each worker
# is listed with the log it writes.
echo; echo "--- are both boards free? ---"
BUSY=$(jssh 'echo "chain.status: $(cat ~/heteroserve/logs/chain.status 2>/dev/null || echo "(none)")"
echo "bench.done  : $(cat ~/heteroserve/results/bench.done 2>/dev/null || echo MISSING)"
for p in $(grep -a -l -s -E "'"$BUSY_J"'" /proc/[0-9]*/cmdline); do
  echo "busy process: $(tr "\0" " " < $p 2>/dev/null | cut -c1-120)"
done
for p in $(pgrep -x ggml-rpc-server); do echo "busy process: ggml-rpc-server already running, pid $p, stdout $(readlink /proc/$p/fd/1 2>/dev/null)"; done
[ -e '"$REC"' ] && { echo "run record  : present"; sed "s/^/run record  : /" '"$REC"'; }
true' 2>&1); rc=$?
echo "$BUSY"
[ "$rc" = 0 ] || { echo "FATAL: could not inspect the Jetson (ssh exit $rc)"; exit 1; }
if grep -q '^run record  : ' <<<"$BUSY"; then
  # The record's saved values are the board's originals; the values there now may be that
  # run's leftovers. Nothing is changed here -- the undo is printed, built from the record,
  # unless that run is still going. Only a run from this laptop can be checked: by its pid,
  # if that is still this script (and not this run). One from elsewhere is judged by its
  # worker, which a live run has up from launch to cleanup -- but so may a crashed one.
  rv() { sed -n "s/^run record  : $1=//p" <<<"$BUSY" | head -1; }
  re_sys='^( net\.ipv4\.conf\.(all|eth0|wlan0)\.arp_(ignore|announce)=[0-9]+)*$'
  rs=$(rv stamp); rh=$(rv host); rp=$(rv pid); rw=$(rv worker_started); ra=$(rv addr_added); ry=$(rv sysctl_restore)
  rl=$(rv clocks_locked); rk=$(rv clocks_pre); rpm=$(rv clocks_pmode)
  ok=0; live=""; rcmd=""; rwk=""
  if [[ "$rs" =~ ^[0-9]{8}_[0-9]{6}$ && "$rh" =~ ^[A-Za-z0-9._-]+$ && "$rp" =~ ^[0-9]+$ && "$rw" =~ ^[01]$ && "$ra" =~ ^[01]$ \
        && "$ry" =~ $re_sys && "$rl" =~ ^[01]$ ]]; then
    ok=1; [ "$rl" = 1 ] && ! [[ "$rk" =~ $RE_CLK && "$rpm" =~ ^pmode:[0-9]{4}$ ]] && ok=0
  fi
  if [ "$ok" = 1 ] && [ "$rh" = "$HOSTNAME" ] && [ "$rp" != $$ ]; then
    rcmd=$(tr '\0' ' ' 2>/dev/null < "/proc/$rp/cmdline")
    [[ "$rcmd" == *rpc_smoke_pi5_jetson.sh* ]] && live=here
  fi
  [ "$ok" = 1 ] && rwk=$(sed -n "s/^busy process: ggml-rpc-server already running, pid \([0-9]*\), stdout .*rpc-server_${rs}\.log$/\1/p" <<<"$BUSY" | head -1)
  if [ "$live" = here ]; then
    echo "FATAL: run ${rs} is still in progress: its script is running on this laptop, pid ${rp} (${rcmd})."
    echo "       It holds the Jetson (its run record, above) -- not touching the board, and no undo: wait for it to end."
    exit 1
  elif [ "$ok" = 1 ] && [ "$rh" != "$HOSTNAME" ] && [ -n "$rwk" ]; then
    echo "FATAL: run ${rs}, from ${rh} (pid ${rp}), may still be in progress: its worker is up on the Jetson (pid ${rwk}),"
    echo "       and this laptop cannot check that host. Not touching the board. Only once that run has certainly"
    echo "       ended, undo with the command built from its record, then re-run:"
  elif [ "$ok" = 1 ] && [ "$rh" != "$HOSTNAME" ]; then
    echo "FATAL: run ${rs}, from ${rh} (pid ${rp}), did not finish its cleanup -- or is still starting up there: check that"
    echo "       host first. Its run record is on the Jetson, so the clock lock, the ARP sysctls and ${J_IP} there may be its"
    echo "       leftovers, not the board's own settings. Once it has certainly ended, undo with the command built from"
    echo "       that record, then re-run:"
  else
    echo "FATAL: an earlier run (${rs}) did not finish its cleanup: its run record is still on the Jetson, so the clock"
    echo "       lock, the ARP sysctls and ${J_IP} there may be its leftovers, not the board's own settings. Undo with"
    echo "       the command built from that record, then re-run:"
  fi
  if [ "$ok" = 1 ]; then
    u=$(undo_cmd "$rw" "$ra" "$ry" alias "$rs" 1 "$rl" "$rk" "$rpm")
    echo "  ssh jetson '${u}'"
  else echo "       (the record is not in the expected form -- inspect it by hand: ssh jetson 'cat ${REC}')"; fi
  exit 1
fi
CHAIN=$(sed -n 's/^chain.status: //p' <<<"$BUSY")
case "$CHAIN" in
  *DONE:*|*FAILED:*|"(none)") ;;
  *) echo "FATAL: the Jetson's own chain is in progress -- not loading its GPU under it"; exit 1 ;;
esac
grep -q '^bench.done  : MISSING' <<<"$BUSY" && { echo "FATAL: no bench.done -- the board's benchmark has not finished"; exit 1; }
grep -q '^busy process:' <<<"$BUSY" && { echo "FATAL: the Jetson is busy (see above)"; exit 1; }
P5BUSY=$(pssh 'for p in $(grep -a -l -s -E "'"$BUSY_P5"'" /proc/[0-9]*/cmdline); do c=$(tr "\0" " " < $p 2>/dev/null | cut -c1-120); [ -n "$c" ] && echo "busy=$c"; done; true' 2>&1); rc=$?
P5B1=$(tr '\n' ' ' <<<"$P5BUSY")
[ "$rc" = 0 ] && [ -z "$P5BUSY" ] || { echo "FATAL: the Pi 5 is busy or unreachable: $P5B1"; exit 1; }
echo "pi5: no llama.cpp run, RPC worker or build running"

# --- model bytes (house rule: SHA-256 checked on the node before any reported figure) ---
# The full 64-hex value from the TinyLlama section's 'SHA-256 (full)' row. The 16-hex row
# is weaker than the evidence on record, so a missing full row is fatal, not a fallback.
# Both boards': the Pi 5's for the RPC cells, the Jetson's own for the local ones. After the
# busy check: hashing reads the whole file (~10 s of SD and CPU on the Jetson), not a load to
# put under another job.
REF=$(awk '/^## TinyLlama/ {f=1; next} /^## / {f=0} f && /SHA-256 \(full\)/' testbed/models.md | grep -oE '\b[0-9a-f]{64}\b' | head -1)
HO=$(pssh "sha256sum $M" 2>&1); rc=$?
H_P5=$(awk 'NR == 1 {print $1}' <<<"$HO")
HJ=$(jssh "sha256sum $M" 2>&1); rcj=$?
H_J=$(awk 'NR == 1 {print $1}' <<<"$HJ")
echo "model  (pi5)      : $H_P5"
echo "model  (jetson)   : $H_J"
echo "model  (models.md): ${REF:-NOT FOUND}  (full SHA-256 compared)"
[ -n "$REF" ] || { echo "FATAL: testbed/models.md has no 'SHA-256 (full)' row for TinyLlama"; exit 1; }
[ "$rc" = 0 ] && [[ "$H_P5" =~ ^[0-9a-f]{64}$ ]] || { echo "$HO"; echo "FATAL: could not hash the Pi 5's model (exit ${rc})"; exit 1; }
[ "$H_P5" = "$REF" ] || { echo "FATAL: the Pi 5's model is not the TinyLlama recorded in testbed/models.md"; exit 1; }
[ "$rcj" = 0 ] && [[ "$H_J" =~ ^[0-9a-f]{64}$ ]] || { echo "$HJ"; echo "FATAL: could not hash the Jetson's model (exit ${rcj})"; exit 1; }
[ "$H_J" = "$REF" ] || { echo "FATAL: the Jetson's model is not the TinyLlama recorded in testbed/models.md"; exit 1; }

# --- power guard: live and armed, or nothing loads --------------------------------------
# The Jetson runs MAXN from micro-USB and jetson_power_guard.sh stands between a sag and a
# brownout mid-write. A dead guard's last log line still reads healthy, so "live" means:
# the process is there, its last per-second sample is within GUARD_MAX_AGE s of the BOARD's
# clock (the lines carry HH:MM:SS only, so the log's mtime is checked as well), and it has
# not tripped since it started (it latches after one trip and never acts again).
# jstate: one read-only key=value snapshot of the Jetson -- power mode, temperatures, clock
# policy, fan, eth0's negotiated link, both NICs' byte counters, load, busy processes (and
# any ggml-rpc-server but this run's), then the guard, and a last line that says it all
# arrived. The clock policy decides the GPU figures: jetson_clocks pins min to max, as the
# board's own benchmark runs and this run does, below; otherwise the GPU runs DVFS from
# 76.8 MHz. The fan and the link go into the log with every result (Jetson/docs/
# 10_measurement_protocol.md: a result whose conditions are not recorded is not a result).
# The board clock is read LAST: a sample the guard writes while the snapshot is taken is
# then never stamped later than `now`. Taken here and before/after every cell.
jstate() {
  jssh 'G=~/heteroserve/logs/power-guard.log
echo "guard_procs=$(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l)"
echo "pmode=$(cut -d" " -f1 /var/lib/nvpmodel/status 2>/dev/null)"
echo "cpu_mC=$(cat /sys/devices/virtual/thermal/thermal_zone1/temp)"
echo "gpu_mC=$(cat /sys/devices/virtual/thermal/thermal_zone2/temp)"
D=/sys/devices/57000000.gpu/devfreq/57000000.gpu; C=/sys/devices/system/cpu/cpu0/cpufreq
echo "gpu_clk=$(cat $D/governor) $(cat $D/min_freq) $(cat $D/max_freq) $(cat $D/cur_freq)"
echo "cpu0_clk=$(cat $C/scaling_governor) $(cat $C/scaling_min_freq) $(cat $C/scaling_max_freq) $(cat $C/scaling_cur_freq)"
printf "clocks="; '"$CLKS"'
echo "fan_pwm=$(cat /sys/devices/pwm-fan/target_pwm)"
echo "fan_ctl=$(cat /sys/devices/pwm-fan/temp_control)"
echo "eth0_link=$(cat /sys/class/net/eth0/speed) $(cat /sys/class/net/eth0/duplex)"
for i in eth0 wlan0; do for d in rx tx; do echo "${i}_${d}=$(cat /sys/class/net/$i/statistics/${d}_bytes)"; done; done
echo "load=$(cut -d" " -f1-3 /proc/loadavg)"
for p in $(grep -a -l -s -E "'"$BUSY_J"'" /proc/[0-9]*/cmdline); do c=$(tr "\0" " " < $p 2>/dev/null | cut -c1-120); [ -n "$c" ] && echo "busy=$c"; done
for p in $(pgrep -x ggml-rpc-server); do c=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null); case "$c" in ""|*"-H '"$J_IP"' -p '"$J_PORT"' "*) ;; *) echo "busy=ggml-rpc-server not of this run, pid $p: ${c:0:120}" ;; esac; done
L=$(tail -n 50 $G 2>/dev/null | grep -a -E "^[0-9]{2}:[0-9]{2}:[0-9]{2} V=" | tail -2)
echo "last=$(echo "$L" | tail -1)"
echo "prev=$(echo "$L" | sed -n "1{\$!s/ .*//p}")"
echo "trips=$(grep -a -c TRIP $G 2>/dev/null)"
echo "trips_since_start=$(awk "/guard start/ {t=0} /TRIP/ {t++} END {print t+0}" $G 2>/dev/null)"
m=$(stat -c %Y $G 2>/dev/null || echo 0); echo "log_age=$(( $(date +%s) - m ))"
echo "now=$(date +%T)"
echo jstate=ok' 2>&1
}
# guard_live SNAPSHOT: prints the verdict; returns 0 only for a live, armed guard
guard_live() {
  local s=$1 now last age
  now=$(hms "$(kv "$s" now)"); last=$(hms "$(kv "$s" last)")
  [ "$(kv "$s" guard_procs)" -ge 1 ] 2>/dev/null || { echo "no jetson_power_guard process"; return 1; }
  [ -n "$now" ] && [ -n "$last" ] || { echo "no parsable guard sample in its log"; return 1; }
  # `now` is read after `last` on the same clock, so last <= now: mod 86400 is for midnight
  # only. A sample up to GUARD_MAX_AGE s AHEAD of `now` means the board's clock stepped back.
  age=$(( (now - last + 86400) % 86400 ))
  [ "$age" -gt $(( 86400 - GUARD_MAX_AGE )) ] && age=$(( age - 86400 ))
  [ "$age" -le "$GUARD_MAX_AGE" ] || { echo "last guard sample is ${age} s old -- stale"; return 1; }
  [ "$(kv "$s" log_age)" -le "$GUARD_MAX_AGE" ] 2>/dev/null || { echo "guard log last written $(kv "$s" log_age) s ago -- stale"; return 1; }
  [ "$(kv "$s" trips_since_start)" = 0 ] || { echo "guard has tripped since it started and is latched"; return 1; }
  if [ "$age" -ge 0 ]; then echo "live (last sample ${age} s old)"
  else echo "live (last sample $(( -age )) s AHEAD of the board clock -- did it step back?)"; fi
}
# JGATE: the Jetson-side gate, run in the same ssh as the load it guards (the clock lock, the
# worker's launch), so that neither starts on evidence seconds old: no busy process and no
# ggml-rpc-server but this run's (exit 5), then MAXN and a live, armed guard -- judged as
# guard_live judges a snapshot, but on the board, right before the start (exit 6).
JGATE='for p in $(grep -a -l -s -E "'"$BUSY_J"'" /proc/[0-9]*/cmdline); do c=$(tr "\0" " " < $p 2>/dev/null | cut -c1-120); [ -n "$c" ] && { echo "GATE-BUSY $c"; exit 5; }; done
for p in $(pgrep -x ggml-rpc-server); do tr "\0" " " < /proc/$p/cmdline 2>/dev/null | grep -qF -- "-H '"$J_IP"' -p '"$J_PORT"' " || { echo "GATE-BUSY a ggml-rpc-server not of this run, pid $p"; exit 5; }; done
G=~/heteroserve/logs/power-guard.log; m=$(cut -d" " -f1 /var/lib/nvpmodel/status 2>/dev/null)
[ "$m" = '"$EXPECT_PMODE"' ] || { echo "GATE-GUARD power mode is $m, expected '"$EXPECT_PMODE"'"; exit 6; }
[ "$(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l)" -ge 1 ] || { echo "GATE-GUARD no jetson_power_guard process"; exit 6; }
t=$(awk "/guard start/ {t=0} /TRIP/ {t++} END {print t+0}" $G 2>/dev/null); [ "$t" = 0 ] || { echo "GATE-GUARD guard has tripped since it started (${t:-no log}) and is latched"; exit 6; }
L=$(tail -n 50 $G 2>/dev/null | grep -a -E "^[0-9]{2}:[0-9]{2}:[0-9]{2} V=" | tail -1 | cut -c1-8); w=$(stat -c %Y $G 2>/dev/null || echo 0); e=$(date +%s); n=$(date +%T)
[[ $L =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || { echo "GATE-GUARD no parsable guard sample in its log"; exit 6; }
a=$(( (10#${n:0:2} * 3600 + 10#${n:3:2} * 60 + 10#${n:6:2} - 10#${L:0:2} * 3600 - 10#${L:3:2} * 60 - 10#${L:6:2} + 86400) % 86400 ))
[ $a -gt $(( 86400 - '"$GUARD_MAX_AGE"' )) ] && a=$(( a - 86400 ))
[ $a -le '"$GUARD_MAX_AGE"' ] || { echo "GATE-GUARD last guard sample is $a s old -- stale"; exit 6; }
[ $(( e - w )) -le '"$GUARD_MAX_AGE"' ] || { echo "GATE-GUARD guard log last written $(( e - w )) s ago -- stale"; exit 6; }
echo "GATE ok: $m, guard live (last sample $a s old), nothing busy"
'
echo; echo "--- power guard ---"
S0=$(jstate)
echo "jetson: $(tr '\n' ' ' <<<"$S0")"
why=$(guard_live "$S0") || { echo "FATAL: power guard not live: $why"; exit 3; }
p0=$(kv "$S0" pmode)
[ "$p0" = "$EXPECT_PMODE" ] || { echo "FATAL: Jetson power mode is '$p0', expected $EXPECT_PMODE"; exit 3; }
t0=$(kv "$S0" trips)
echo "guard $why; trips=$t0; $p0"
g0=$(clk "$(kv "$S0" gpu_clk)" Hz); c0=$(clk "$(kv "$S0" cpu0_clk)" kHz)
echo "jetson clocks: GPU $g0; CPU0 $c0"

# --- wired path: ARP hygiene, temporary overlay address, proof by MAC -------------------
# ARP flux, measured 2026-09-25: with arp_ignore=0 the Jetson answers ARP for any of its
# addresses on EVERY interface, and its wlan0 hears the Pi 5's broadcasts through the
# router's Wi-Fi bridge -- a broadcast for .111 got replies from eth0 AND wlan0. So wlan0
# may answer for .13 and the RPC traffic silently cross Wi-Fi, while `ip route get` on the
# Pi 5 says "dev eth0" regardless. arp_ignore=1 (answer only for addresses on the receiving
# interface; 2 is stricter and also fine) and arp_announce=2 stop it. The kernel takes
# max(all, <if>), so all/eth0/wlan0 are set. Saved first (into the run record before any
# write); only the keys this run changes are restored, to exactly the saved value. wlan0 has
# had no IPv4 since the boards went wired-only (2026-09-26); the mitigation stays, per run.
# That first record write is the claim on the board (rec, above): exit 7 means another run
# of this script took it since the check above, and this one leaves everything alone.
echo; echo "--- wired path ---"
arp_read='for i in all eth0 wlan0; do for k in arp_ignore arp_announce; do echo "$i.$k=$(cat /proc/sys/net/ipv4/conf/$i/$k)"; done; done'
SYS=$(jssh "$arp_read" 2>&1)
echo "jetson ARP sysctls before: $(tr '\n' ' ' <<<"$SYS")"
SYSOK=$(grep -E '^(all|eth0|wlan0)\.arp_(ignore|announce)=[0-9]+$' <<<"$SYS")   # an ssh warning line is not a key
NSYS=$(grep -c . <<<"$SYSOK")
[ "$NSYS" = 6 ] || { echo "FATAL: could not read the Jetson's ARP sysctls"; exit 1; }
SET=""; RESTORE=""
while IFS='=' read -r key val; do
  case "$key" in
    *.arp_ignore)   want=1; { [ "$val" = 1 ] || [ "$val" = 2 ]; } && continue ;;
    *.arp_announce) want=2; [ "$val" = 2 ] && continue ;;
    *) continue ;;
  esac
  SET+=" net.ipv4.conf.${key}=${want}"; RESTORE+=" net.ipv4.conf.${key}=${val}"
done <<<"$SYSOK"
REC_WRITTEN=1; SYSCTL_RESTORE=$RESTORE
if [ -n "$SET" ]; then jssh "$(rec claim) && sudo -n sysctl -w${SET}"; else jssh "$(rec claim)"; fi
rc=$?
if [ "$rc" = 7 ]; then
  REC_WRITTEN=0; SYSCTL_RESTORE=""
  echo "FATAL: a run record appeared on the Jetson since the check above: another run of this script holds the board -- left alone"; exit 1
fi
[ "$rc" = 255 ] || HELD=1
[ "$rc" = 0 ] || { echo "FATAL: could not write the run record or set the Jetson's ARP sysctls (ssh exit ${rc})"; exit 1; }
SYS=$(jssh "$arp_read" 2>&1)
echo "jetson ARP sysctls now   : $(tr '\n' ' ' <<<"$SYS")"
NSYS=$(grep -cE '^(all|eth0|wlan0)\.(arp_ignore=[12]|arp_announce=2)$' <<<"$SYS")
[ "$NSYS" = 6 ] || { echo "FATAL: the Jetson's ARP sysctls are not at arp_ignore=1|2, arp_announce=2"; exit 1; }

# Is .13 on the Jetson already? A permanent overlay address is what node-jetson.md
# recommends, and .13 is the obvious one: if it is there, this run did not add it and must
# not delete it. (With no run record on the board -- checked above -- it is not a leftover.)
A=$(jssh "ip -4 -o addr show | grep -F ' ${J_IP}/' || echo none" 2>&1) || { echo "FATAL: could not read the Jetson's addresses: $A"; exit 1; }
case "$A" in
  none)       ADDR_ADDED=1; ADDR_PRESENT=1
              jssh "$(rec) && sudo -n ip addr add ${J_IP}/24 dev eth0" || { echo "FATAL: could not add ${J_IP} to the Jetson's eth0"; exit 1; } ;;
  *" eth0 "*) ADDR_PRESENT=1; echo "${J_IP} was already on the Jetson's eth0 -- not added by this run, will not be removed" ;;
  *)          echo "FATAL: ${J_IP} is on another Jetson interface: $A"; exit 1 ;;
esac
jssh "ip -br -4 addr show eth0"
R=$(pssh "ip route get ${J_IP} | head -1" 2>&1)
echo "route pi5 -> jetson   : $R"
[[ "$R" == *" dev eth0 "* ]] || { echo "FATAL: the Pi 5 does not route ${J_IP} out of eth0"; exit 1; }
# Flush both ends' entries first, so the MACs checked below come from THIS run's ARP and
# not from a stale entry (possibly learned from the wlan0 MAC by an earlier run).
N=$(pssh "sudo -n ip neigh flush to ${J_IP}; ip neigh show ${J_IP}" 2>&1)
[[ "$N" == *lladdr* ]] && { echo "FATAL: could not flush the Pi 5's neighbour entry for ${J_IP}: $N"; exit 1; }
N=$(jssh "sudo -n ip neigh flush to ${P5_IP}; ip neigh show ${P5_IP}" 2>&1)
[[ "$N" == *lladdr* ]] && { echo "FATAL: could not flush the Jetson's neighbour entry for ${P5_IP}: $N"; exit 1; }
PING=$(pssh "ping -c 5 -q ${J_IP}" 2>&1); rc=$?
PING2=$(tail -2 <<<"$PING" | tr '\n' ' ')
echo "ping  pi5 -> jetson   : $PING2"
[ "$rc" = 0 ] || { echo "FATAL: the Pi 5 cannot reach ${J_IP}"; exit 1; }
N=$(pssh "ip neigh show ${J_IP}" 2>&1)
echo "neigh pi5 -> jetson   : $N"
NL=$(lladdr "$N")
if [ "$NL" != "$J_ETH_MAC" ] || [[ "$N" != *" dev eth0 "* ]]; then
  [ "$NL" = "$J_WLAN_MAC" ] && echo "that MAC is the Jetson's wlan0: ARP flux, the RPC path would cross Wi-Fi"
  echo "FATAL: ${J_IP} did not resolve to the Jetson's eth0 (${J_ETH_MAC}) on the Pi 5's eth0"; exit 1
fi
N=$(jssh "ip neigh show ${P5_IP}" 2>&1)
echo "neigh jetson -> pi5   : $N"
NL=$(lladdr "$N")
if [ "$NL" != "$P5_ETH_MAC" ] || [[ "$N" != *" dev eth0 "* ]]; then
  echo "FATAL: ${P5_IP} did not resolve to the Pi 5's eth0 (${P5_ETH_MAC}) on the Jetson's eth0"; exit 1
fi

# --- the Jetson's clocks: locked for the whole run --------------------------------------
# jetson_clocks pins every CPU's and the GPU's minimum clock to its maximum (and turns off
# GPU rail gating and CPU idle states), as the board's own benchmark runs: a GPU that waits
# on the network cannot then clock down, and the RPC and local cells run at the same clocks.
# As jetson_bench_tinyllama.sh does it, but no step's status is thrown away: the policy is
# stored first, to this run's own new file (--store asks before it overwrites one, and with
# no terminal the answer is no), the change is in the run record before it is made, and the
# lock is read back: no cell runs unless it reads LOCKED. The cleanup restores and checks
# it. A board found LOCKED is left as found. Locked clocks draw more from the micro-USB
# supply (the locked standalone run's lowest guard sample: 4808 mV; the guard acts at 4750):
# the guard stays the protection -- JGATE right before the lock -- and the run stops at its
# first trip.
echo; echo "--- jetson clocks: locked for the whole run ---"
C0=$(jstate)
echo "jetson: $(tr '\n' ' ' <<<"$C0")"
why=$(guard_live "$C0") || { echo "FATAL: power guard not live before the clock lock: $why"; exit 3; }
p0=$(kv "$C0" pmode)
[ "$p0" = "$EXPECT_PMODE" ] || { echo "FATAL: power mode is '$p0' before the clock lock"; exit 3; }
g0=$(clk "$(kv "$C0" gpu_clk)" Hz); c0=$(clk "$(kv "$C0" cpu0_clk)" kHz); k0=$(kv "$C0" clocks)
if lk "$C0"; then
  echo "jetson clocks: GPU $g0; CPU0 $c0 -- already LOCKED before this run: left as found (not stored, locked or restored by it)"
else
  [[ "$k0" =~ $RE_CLK ]] || { echo "FATAL: could not read the Jetson's clock policy (clocks=${k0}) -- not locking what could not be put back and checked"; exit 1; }
  CLK_PRE=$k0; CLK_PMODE=$p0; CLOCKS_LOCKED=1
  L=$(jssh "${JGATE}$(rec) || exit 4; [ ! -e ${CLKF} ] || { echo \"CLOCK SNAPSHOT ${CLKF} EXISTS\"; exit 8; }; \
    sudo -n jetson_clocks --store ${CLKF} < /dev/null || exit 9; \
    grep -q cpu0/cpufreq/scaling_min_freq: ${CLKF} && grep -q 57000000.gpu/min_freq: ${CLKF} || { echo \"CLOCK SNAPSHOT INCOMPLETE\"; cat ${CLKF}; exit 9; }; \
    sed \"s/^/clocks stored: /\" ${CLKF}; sudo -n jetson_clocks < /dev/null || exit 10; echo \"clocks locked, policy stored in ${CLKF}\"" 2>&1); rc=$?
  echo "$L"
  if grep -q '^GATE-BUSY' <<<"$L"; then echo "FATAL: the Jetson became busy before the clock lock (above) -- not locked"; exit 1; fi
  if grep -q '^GATE-GUARD' <<<"$L"; then echo "FATAL: power guard not live, or wrong power mode, at the clock lock (above) -- not locked"; exit 3; fi
  [ "$rc" = 0 ] || { echo "FATAL: could not store and lock the Jetson's clocks (ssh exit ${rc}) -- no cell runs"; exit 1; }
  C1=$(jstate)
  echo "jetson: $(tr '\n' ' ' <<<"$C1")"
  g1=$(clk "$(kv "$C1" gpu_clk)" Hz); c1=$(clk "$(kv "$C1" cpu0_clk)" kHz)
  echo "jetson clocks: GPU $g0 -> $g1; CPU0 $c0 -> $c1"
  lk "$C1" || { echo "FATAL: the Jetson's clocks did not lock (above) -- no cell runs"; exit 1; }
fi

# wlog_dump LABEL: the worker's output since the last dump, into the run log (house rule:
# tee raw output) -- a Vulkan device-lost, an allocation fallback or a disconnect on the
# worker's side then sits next to the cell it hit. Advances WLOG_OFF.
wlog_dump() {
  local o e
  o=$(jssh "s=\$(stat -c %s ${WLOG}) && [ \$s -ge ${WLOG_OFF} ] || exit 1; echo \"--- worker log, $1: bytes ${WLOG_OFF}..\$s of ${WLOG} on the Jetson ---\"; \
    tail -c +$((WLOG_OFF + 1)) ${WLOG} | head -c \$((s - ${WLOG_OFF})) | awk 1; echo \"--- end of worker log at byte \$s ---\"" 2>&1)
  echo "$o"
  e=$(sed -n 's/^--- end of worker log at byte \([0-9]*\) ---$/\1/p' <<<"$o" | tail -1)
  [ -n "$e" ] && WLOG_OFF=$e
}

echo; echo "--- starting ggml-rpc-server on the Jetson, Vulkan0 only ---"
# -d/--device takes a comma-separated device list at the pin (`ggml-rpc-server --help` on
# the board, 2026-09-25), and the board's Vulkan device is named Vulkan0. No pkill and no
# sleep ahead of it: the busy check refused a running worker, so one found now is not this
# run's and is left alone, and the launcher returns the moment the worker is spawned (the
# cleanup outlasts a launch cut short anyway). The log is this run's own file. JGATE first,
# in the same ssh: the checks above are seconds old by now (the ping alone takes 4 s).
WORKER_STARTED=1
L=$(jssh "${JGATE}$(rec) || exit 4; mkdir -p ~/heteroserve/logs; \
  if pgrep -x ggml-rpc-server >/dev/null; then echo \"FOREIGN-WORKER pid \$(pgrep -x ggml-rpc-server | head -1)\"; exit 3; fi; \
  setsid nohup ~/llama.cpp/build/bin/ggml-rpc-server -H ${J_IP} -p ${J_PORT} -d Vulkan0 \
    > ${WLOG} 2>&1 < /dev/null & echo worker-pid=\$! log=${WLOG}" 2>&1); rc=$?
echo "$L"
if grep -q '^FOREIGN-WORKER' <<<"$L"; then
  WORKER_STARTED=0; echo "FATAL: a ggml-rpc-server that is not this run's is running on the Jetson -- left alone"; exit 1
fi
if grep -q '^GATE-BUSY' <<<"$L"; then WORKER_STARTED=0; echo "FATAL: the Jetson became busy before the worker's launch (above) -- nothing started"; exit 1; fi
if grep -q '^GATE-GUARD' <<<"$L"; then WORKER_STARTED=0; echo "FATAL: power guard not live, or wrong power mode, at the worker's launch (above) -- nothing started"; exit 3; fi
[ "$rc" = 0 ] || { echo "FATAL: could not note the worker in the run record, or start it (ssh exit ${rc})"; exit 1; }
for i in $(seq 1 20); do
  if jssh "ss -ltn 2>/dev/null | grep -q ' ${J_IP}:${J_PORT} '"; then
    echo "worker listening on ${J_IP}:${J_PORT} after ${i}s"; break
  fi
  sleep 1
  if [ "$i" = 20 ]; then echo "FATAL: worker never came up"; wlog_dump "startup"; exit 1; fi
done
wlog_dump "startup"

# p5state: the Pi 5's side of a snapshot -- temperature and get_throttled (house rule: with
# every Pi run), its MAC for .13, eth0's negotiated link, load and busy processes, and a last
# line that says it all arrived.
p5state() {
  pssh 'vcgencmd measure_temp; vcgencmd get_throttled; ip neigh show '"$J_IP"'
echo "eth0_link=$(cat /sys/class/net/eth0/speed) $(cat /sys/class/net/eth0/duplex)"
echo "load=$(cut -d" " -f1-3 /proc/loadavg)"
for p in $(grep -a -l -s -E "'"$BUSY_P5"'" /proc/[0-9]*/cmdline); do c=$(tr "\0" " " < $p 2>/dev/null | cut -c1-120); [ -n "$c" ] && echo "busy=$c"; done
echo p5state=ok' 2>&1
}
# gate WHEN P5: nothing loads the boards but on fresh, complete evidence -- the snapshots B
# (and PB, with its ssh's exit PBRC, when the Pi 5 takes part: P5=1) taken just before: a
# live, armed guard and MAXN (else exit 3), then the clocks LOCKED as this run holds them,
# both snapshots complete (their last line) and neither board busy (else exit 1). A
# snapshot that did not arrive is not an idle board.
gate() {
  local why pm bj bp
  why=$(guard_live "$B") || { echo "FATAL: power guard not live before $1: $why"; exit 3; }
  pm=$(kv "$B" pmode)
  [ "$pm" = "$EXPECT_PMODE" ] || { echo "FATAL: power mode is '$pm' before $1"; exit 3; }
  grep -qx 'jstate=ok' <<<"$B" || { echo "FATAL: could not confirm the Jetson is idle before $1 (snapshot incomplete)"; exit 1; }
  bj=$(sed -n 's/^busy=//p' <<<"$B" | head -1)
  [ -z "$bj" ] || { echo "FATAL: the Jetson is busy before $1: $bj"; exit 1; }
  lk "$B" || { echo "FATAL: the Jetson's clocks no longer read LOCKED before $1 (above) -- the cells must all run locked"; exit 1; }
  [ "$2" = 1 ] || return 0
  [ "$PBRC" = 0 ] && grep -qx 'p5state=ok' <<<"$PB" || { echo "FATAL: could not confirm the Pi 5 is idle before $1 (ssh exit ${PBRC}, snapshot incomplete)"; exit 1; }
  bp=$(sed -n 's/^busy=//p' <<<"$PB" | head -1)
  [ -z "$bp" ] || { echo "FATAL: the Pi 5 is busy before $1: $bp"; exit 1; }
}

echo; echo "--- devices as the Pi 5 sees them ---"
# A load on both boards (the Pi 5's llama-bench, the worker enumerating its device for it):
# gated like a cell, on fresh snapshots of both.
B=$(jstate); PB=$(p5state); PBRC=$?
echo "jetson before: $(tr '\n' ' ' <<<"$B")"
echo "pi5    before: $(tr '\n' ' ' <<<"$PB")"
gate "--list-devices" 1
pssh "~/llama.cpp/build/bin/llama-bench -rpc ${J_IP}:${J_PORT} --list-devices 2>&1 | head -10"

# settle: a cell's "after" snapshot, read only once the guard has logged two samples stamped
# later than the board's clock right after the cell. The guard samples, writes the line, may
# TRIP, then sleeps 1 s (gaps up to 2 s seen), so an alarm in a cell's last second reaches
# the log only that much later; the second sample proves the first one's TRIP check was
# written. Gives up after ~12 s -- a guard that stopped writing is then found stale.
settle() {
  local s t0 p d i
  s=$(jstate); t0=$(hms "$(kv "$s" now)")
  for i in $(seq 1 24); do
    [ -n "$t0" ] && [ "$(kv "$s" guard_procs)" -ge 1 ] 2>/dev/null || break
    p=$(hms "$(kv "$s" prev)")
    if [ -n "$p" ]; then d=$(( (p - t0 + 86400) % 86400 )); [ "$d" -ge 1 ] && [ "$d" -lt 43200 ] && break; fi
    sleep 0.5; s=$(jstate)
  done
  printf '%s\n' "$s"
}

# TinyLlama has 22 layers. ngl=99: every layer on the Jetson's GPU; ngl=11: half on the
# Jetson's GPU, half on the Pi 5's CPU -- a genuinely heterogeneous split. local1/local2: the
# ngl=99 work on the Jetson alone, with no network in between -- before and after the RPC
# cells, with the worker up but idle, so all four run in the same board state.
# Around every cell: the guard's TRIP count and pmode (a trip mid-cell mixes MAXN and 5 W
# repetitions; a stale `tail -1` would still read healthy), temperatures, the Jetson's clock
# policy and fan, busy processes and load (a job started by another session since the check
# above), the Jetson's per-NIC byte counters (a local cell's should carry no tensors; an RPC
# cell's run to hundreds of MB, so which NIC carried them is not ambiguous), the worker's own
# log for the cell, and the cell's own jsonl rows (llama-bench's output is tee'd to the log
# AND kept to judge them). For an RPC cell also the Pi 5's get_throttled, judged (bits 0-3
# now, 16-19 sticky until reboot), its MAC for .13 and both ends' negotiated link. A busy
# board refuses the cell; one that became busy during it marks the cell.
add() { V="${V:+$V,}$1"; }
RESULT=""; FINAL=0; LASTC=""; LASTV=""; RES_BEFORE=""
for CELL in local1 ngl=99 ngl=11 local2; do
  case "$CELL" in local*) LOCAL=1; NGL=99 ;; *) LOCAL=0; NGL=${CELL#ngl=} ;; esac
  echo; echo "##### ${CELL}"
  B=$(jstate)
  echo "jetson before: $(tr '\n' ' ' <<<"$B")"
  PB=""; PBRC=0; PA=""
  if [ "$LOCAL" = 0 ]; then
    PB=$(p5state); PBRC=$?
    echo "pi5    before: $(tr '\n' ' ' <<<"$PB")"
  fi
  gate "$CELL" $(( 1 - LOCAL ))
  # The cell's sh writes its pid, then becomes the timeout wrapper: the cleanup stops this
  # cell by that pid, and nothing else on the board. The output goes to the log as it comes
  # (tee) and into CO, whose status is the ssh's.
  if [ "$LOCAL" = 1 ]; then
    LOCAL_RUNNING=1
    CO=$(jssh "cd ~ && sh -c 'echo \$\$ > ${JPID}; exec /usr/bin/timeout 900 ~/llama.cpp/build/bin/llama-bench -m $M ${LOCAL_ARGS}'; rc=\$?; rm -f ${JPID}; echo EXIT_${CELL}=\$rc; exit \$rc" 2>&1 | tee /dev/stderr; exit "${PIPESTATUS[0]}")
    RC=$?
    [ "$RC" = 255 ] || LOCAL_RUNNING=0   # 255: the ssh failed, and the cell may still run there
  else
    CELL_RUNNING=1
    CO=$(pssh "cd ~ && sh -c 'echo \$\$ > ${P5PID}; exec /usr/bin/timeout 900 ~/llama.cpp/build/bin/llama-bench -m $M -rpc ${J_IP}:${J_PORT} -ngl ${NGL} ${BENCH}'; rc=\$?; rm -f ${P5PID}; echo EXIT_ngl${NGL}=\$rc; exit \$rc" 2>&1 | tee /dev/stderr; exit "${PIPESTATUS[0]}")
    RC=$?
    [ "$RC" = 255 ] || CELL_RUNNING=0     # 255: the ssh failed, and the cell may still run there
  fi
  A=$(settle)
  echo "jetson after : $(tr '\n' ' ' <<<"$A")"
  if [ "$LOCAL" = 0 ]; then
    PA=$(p5state)
    echo "pi5    after : $(tr '\n' ' ' <<<"$PA")"
  fi
  tb=$(kv "$B" trips); ta=$(kv "$A" trips); pb=$(kv "$B" pmode); pa=$(kv "$A" pmode)
  GA=$(guard_live "$A"); GARC=$?
  echo "jetson guard : trips ${tb} -> ${ta}, ${pb} -> ${pa}; after: ${GA}"
  tcb=$(mc "$(kv "$B" cpu_mC)"); tca=$(mc "$(kv "$A" cpu_mC)"); tgb=$(mc "$(kv "$B" gpu_mC)"); tga=$(mc "$(kv "$A" gpu_mC)")
  echo "jetson temp  : CPU ${tcb} -> ${tca}, GPU ${tgb} -> ${tga}"
  gcb=$(kv "$B" gpu_clk); gca=$(kv "$A" gpu_clk); ccb=$(kv "$B" cpu0_clk); cca=$(kv "$A" cpu0_clk)
  kgb=$(clk "$gcb" Hz); kga=$(clk "$gca" Hz); kcb=$(clk "$ccb" kHz); kca=$(clk "$cca" kHz)
  echo "jetson clocks: GPU ${kgb} -> ${kga}; CPU0 ${kcb} -> ${kca}"
  fpb=$(kv "$B" fan_pwm); fpa=$(kv "$A" fan_pwm); fcb=$(kv "$B" fan_ctl); fca=$(kv "$A" fan_ctl)
  echo "jetson fan   : target_pwm ${fpb} -> ${fpa}, temp_control ${fcb} -> ${fca}"
  de_rx=$(delta "$(kv "$B" eth0_rx)" "$(kv "$A" eth0_rx)");   de_tx=$(delta "$(kv "$B" eth0_tx)" "$(kv "$A" eth0_tx)")
  dw_rx=$(delta "$(kv "$B" wlan0_rx)" "$(kv "$A" wlan0_rx)"); dw_tx=$(delta "$(kv "$B" wlan0_tx)" "$(kv "$A" wlan0_tx)")
  echo "jetson NICs  : eth0 rx +${de_rx} B, tx +${de_tx} B | wlan0 rx +${dw_rx} B, tx +${dw_tx} B"
  P5V=""; LV=""
  if [ "$LOCAL" = 0 ]; then
    jlb=$(kv "$B" eth0_link); jla=$(kv "$A" eth0_link); plb=$(kv "$PB" eth0_link); pla=$(kv "$PA" eth0_link)
    for x in "$jlb" "$jla" "$plb" "$pla"; do [ "$x" = "1000 full" ] || LV=LINK-NOT-GBE; done
    echo "eth0 link    : jetson ${jlb} -> ${jla} | pi5 ${plb} -> ${pla}: ${LV:-1000 Mb/s full duplex at both ends, before and after}"
    # get_throttled: bits 0-3 are "now", 16-19 "since boot". A cell is throttled if a now-bit
    # is set at either end or a sticky bit changed across it; sticky bits already set before
    # it hide any event during it, so such a cell cannot be certified until a reboot.
    tpb=$(thr "$PB"); tpa=$(thr "$PA")
    if [ -z "$tpb" ] || [ -z "$tpa" ]; then P5V=PI5-UNCERTIFIED; p5why="get_throttled unreadable -- this cell is not certified"
    elif (( (tpb | tpa) & 0xF )) || (( (tpb ^ tpa) & 0xF0000 )); then P5V=PI5-THROTTLED; p5why="under-voltage or throttling now (bits 0-3), or newly flagged across the cell (bits 16-19)"
    elif (( tpb & 0xF0000 )); then P5V=PI5-UNCERTIFIED; p5why="sticky bits 16-19 already set before this cell -- an event during it would not show; reboot the Pi 5 to clear them"
    else p5why="bits 0-3 clear before and after, no sticky bit 16-19"; fi
    thb=$(kv "$PB" throttled); tha=$(kv "$PA" throttled)
    echo "pi5 throttle : ${thb} -> ${tha}: ${P5V:-ok} -- ${p5why}"
  fi
  BUSYA=$( { sed -n 's/^busy=/jetson: /p' <<<"$A"; sed -n 's/^busy=/pi5: /p' <<<"$PA"; } )
  [ -n "$BUSYA" ] && sed 's/^/busy after   : /' <<<"$BUSYA"
  wlog_dump "${CELL}"
  RO=$(rowcheck "$CO" "$NGL" "$LOCAL")
  RL=${RO%%|*}
  echo "records      : ${RO#*|}: ${RL:-ok}"
  V=""
  if ! [[ "$ta" =~ ^[0-9]+$ ]]; then V=GUARD-LOST           # after-snapshot unreadable
  elif [ "$ta" != "$tb" ] || [ "$pa" != "$pb" ]; then V=TRIPPED
  elif [ "$GARC" != 0 ]; then V=GUARD-LOST
  else
    [ "$RC" = 0 ] || add "FAILED(exit ${RC})"
    [ -n "$RL" ] && add "$RL"
    if [ "$LOCAL" = 0 ]; then
      LLA=$(lladdr "$(grep ' lladdr ' <<<"$PA")")
      if ! [[ "$dw_rx" =~ ^[0-9]+$ && "$dw_tx" =~ ^[0-9]+$ ]] || [ $(( dw_rx + dw_tx )) -gt "$WLAN_MAX" ] \
         || [ "$LLA" != "$J_ETH_MAC" ]; then add NOT-PROVEN-WIRED; fi
      [ -n "$LV" ] && add "$LV"
      [ -n "$P5V" ] && add "$P5V"
    fi
    [ -n "$BUSYA" ] && add BUSY
    pgb=$(pol "$gcb"); pga=$(pol "$gca"); pcb=$(pol "$ccb"); pca=$(pol "$cca")
    { [ "$pgb" != "$pga" ] || [ "$pcb" != "$pca" ]; } && add CLOCKS-CHANGED
  fi
  echo "CELL ${CELL}: ${V:-ok}"
  RES_BEFORE=$RESULT; LASTV=$V; LASTC=$CELL
  RESULT+=" ${CELL}:${V:-ok}"
  case "$V" in
    TRIPPED)    echo "STOP: the guard acted during ${CELL} -- its repetitions mix power modes and are not usable"; exit 4 ;;
    GUARD-LOST) echo "STOP: the guard cannot be confirmed live after ${CELL} -- the cell may have run unguarded"; exit 4 ;;
    "") ;;
    *) FINAL=1 ;;
  esac
done

# The last cell's after-snapshot waited for the guard; look once more before calling the
# run clean, so nothing reads "ok" while the board sits latched at 5 W.
echo; echo "--- guard after the last cell ---"
F=$(jstate)
echo "jetson final : $(tr '\n' ' ' <<<"$F")"
tf=$(kv "$F" trips); pf=$(kv "$F" pmode); V=""
if ! [[ "$tf" =~ ^[0-9]+$ ]]; then V=GUARD-LOST-AFTER
elif [ "$tf" != "$ta" ] || [ "$pf" != "$pa" ]; then V=TRIPPED-AFTER
elif ! why=$(guard_live "$F"); then V=GUARD-LOST-AFTER; fi
GF=$(guard_live "$F")
echo "jetson guard : trips ${ta} -> ${tf}, ${pa} -> ${pf}; ${GF}"
if [ -n "$V" ]; then
  RESULT="${RES_BEFORE} ${LASTC}:${LASTV:+$LASTV,}${V}"
  echo "cells:${RESULT}"
  if [ "$V" = TRIPPED-AFTER ]; then echo "STOP: TRIPPED after ${LASTC}'s after-snapshot -- the board left ${EXPECT_PMODE}; that cell's last repetitions may not be MAXN"
  else echo "STOP: the guard cannot be confirmed live after ${LASTC} -- the cell may have run unguarded"; fi
  exit 4
fi

echo; echo "########## done $(date -Is)"
echo "cells:${RESULT}"
echo "OUTFILE=$OUT"
exit $FINAL
