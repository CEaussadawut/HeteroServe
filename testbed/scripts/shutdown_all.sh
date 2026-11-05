#!/bin/bash
# Power off every testbed node cleanly: Jetson Nano, Pi 5, Pi 4 -- and, with --laptop, this
# control node last. A clean poweroff matters on these boards: pulling power mid-write can
# corrupt the SD card (the Jetson is also on micro-USB, where a brownout already risks it).
#
# Run from the laptop:
#   bash testbed/scripts/shutdown_all.sh            # asks for confirmation
#   bash testbed/scripts/shutdown_all.sh --yes      # no prompt
#   bash testbed/scripts/shutdown_all.sh --force    # even if a node is busy (llama-bench / RPC worker running)
#   bash testbed/scripts/shutdown_all.sh --laptop   # also power off this machine at the end
#
# Refuses a node that is running llama-bench, llama-cli, llama-server or ggml-rpc-server unless
# --force: shutting a node down under a measurement ruins the run without saying so.
# The Jetson is HALTED, not powered off: on 2026-09-26 15:31 a clean `systemctl poweroff` made this
# Nano (B01, micro-USB) boot straight back up 20 s later. `systemctl halt` still stops every service
# and unmounts the filesystems, then leaves the CPU stopped with the power on -- safe to unplug, and it
# cannot come back by itself. The Pis are powered off normally.
# A node that answers again with a NEW boot_id is reported as REBOOTED, not as down.
# Needs passwordless sudo for shutdown on each node (`sudo -n`); a node without it is reported,
# not skipped silently. Uses `pgrep -x`, never `pgrep -f` over ssh (a plain -f pattern matches
# the ssh command line itself).
set -u
NODES=(jetson pi5 pi4)
YES=0; FORCE=0; LAPTOP=0
for a in "$@"; do
  case "$a" in
    --yes|-y) YES=1 ;;
    --force)  FORCE=1 ;;
    --laptop) LAPTOP=1 ;;
    -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done
# -n: never let ssh read the terminal -- without it, keys typed while the checks run are swallowed or
# left half-read, and the confirmation below saw junk instead of "y" (2026-09-26 15:34, printed "aborted").
SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=8)

echo "########## $(date -Is) shutdown: ${NODES[*]}$([ $LAPTOP = 1 ] && echo ' + laptop')"
declare -A STATE BOOT
cmd_for() { [ "$1" = jetson ] && echo halt || echo poweroff; }
for n in "${NODES[@]}"; do
  out=$("${SSH[@]}" "$n" 'echo "host=$(hostname) up=$(cut -d. -f1 /proc/uptime)s boot=$(cat /proc/sys/kernel/random/boot_id)"
    busy=""; for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x "$p" >/dev/null && busy="$busy $p"; done
    echo "busy=${busy:-none}"
    sudo -n true 2>/dev/null && echo sudo=ok || echo sudo=NO' 2>/dev/null)
  if [ -z "$out" ]; then echo "  $n: UNREACHABLE (already off?)"; STATE[$n]=skip; continue; fi
  echo "  $n: $(echo $out)"
  BOOT[$n]=$(grep -o 'boot=[0-9a-f-]*' <<<"$out" | cut -d= -f2)
  if ! grep -q 'busy=none' <<<"$out" && [ $FORCE = 0 ]; then STATE[$n]=busy
  elif grep -q 'sudo=NO' <<<"$out"; then STATE[$n]=nosudo
  else STATE[$n]=go; fi
done

busy=$(for n in "${NODES[@]}"; do [ "${STATE[$n]}" = busy ] && echo -n "$n "; done)
[ -n "$busy" ] && { echo "REFUSING: busy node(s): $busy-- stop the run first, or pass --force"; exit 1; }

if [ $YES = 0 ]; then
  while read -r -t 0.1 -n 256 _ < /dev/tty; do :; done     # drop anything typed during the checks
  read -r -p "Shut down the nodes above (Jetson: halt)$([ $LAPTOP = 1 ] && echo ' and THIS laptop')? [y/N] " ans < /dev/tty
  ans=$(tr -d '[:space:]' <<<"$ans")
  [[ "$ans" =~ ^([yY]|[yY][eE][sS])$ ]] || { echo "aborted (answer was '${ans}')"; exit 1; }
fi

rc=0
for n in "${NODES[@]}"; do
  case "${STATE[$n]}" in
    skip)   continue ;;
    nosudo) echo "  $n: NO passwordless sudo -- run 'sudo $(cmd_for "$n")' on it yourself"; rc=1; continue ;;
  esac
  # this drops the connection, so a non-zero ssh exit here is expected; success is judged below
  c=$(cmd_for "$n")
  "${SSH[@]}" "$n" "sudo -n systemctl $c" >/dev/null 2>&1
  echo "  $n: $c sent"
done

echo "--- waiting for the nodes to go down (max 90 s) ---"
for n in "${NODES[@]}"; do
  [ "${STATE[$n]}" = go ] || continue
  down=0
  for _ in $(seq 1 18); do
    "${SSH[@]}" -o ConnectTimeout=3 "$n" true 2>/dev/null || { down=1; break; }
    sleep 5
  done
  if [ $down = 0 ]; then echo "  $n: STILL UP after 90 s"; rc=1; continue; fi
  # a board that reboots instead of stopping is back on ssh within ~25 s: look again before calling it down
  sleep 45
  b=$("${SSH[@]}" -o ConnectTimeout=3 "$n" 'cat /proc/sys/kernel/random/boot_id' 2>/dev/null)
  if [ -z "$b" ]; then echo "  $n: down ($(cmd_for "$n"))"
  elif [ "$b" != "${BOOT[$n]}" ]; then echo "  $n: REBOOTED instead of shutting down (new boot_id)"; rc=1
  else echo "  $n: answered again with the SAME boot_id -- shutdown did not happen"; rc=1; fi
done
echo "Nodes reported down are safe to unplug (the Jetson stays powered but halted: its LED stays on)."

if [ $LAPTOP = 1 ]; then
  [ $rc = 0 ] || { echo "not powering off the laptop: a node did not shut down cleanly"; exit $rc; }
  echo "powering off this laptop in 30 s (Ctrl-C to cancel)"; sleep 30
  systemctl poweroff
fi
exit $rc
