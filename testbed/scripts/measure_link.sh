#!/bin/bash
# The link between two boards, measured on the path their pooled runs use. From the repo root on
# the laptop, with both boards idle:
#     bash testbed/scripts/measure_link.sh                 # Pi 5 <-> Pi 4, the wired 192.168.10.x overlay
#     PAIR="pi5 jetson" bash testbed/scripts/measure_link.sh   # Pi 5 <-> Jetson, the router LAN (192.168.1.x; the Pi 5 back 2026-10-19)
#     PAIR="pi4 jetson" bash testbed/scripts/measure_link.sh   # Pi 4 <-> Jetson, the router LAN: the path a pool
#                                                              # the Pi 4 hosts uses to its Jetson worker
# The Jetson has no overlay address; its eth0 is its only interface on the router LAN (checked below,
# so the path cannot be Wi-Fi), and each Pi reaches it there on the same eth0 and switch.
# Raw TCP on port 5201 (never 50052, the RPC port): OpenBSD netcat sends MIB MiB from dd (1 GiB by default), REPS
# per direction. Two figures per transfer:
#   XFER  dd's on the SENDER, from its first write. Bytes still in the socket buffers when dd stops count as delivered:
#         under 1 % for 1 GiB at gigabit, but +12.4 % at 20 Mbit/s with the 20 MiB the shaped campaigns send
#         (2026-10-09, testbed/benchmarks.md "do all 5", section 3).
#   RECV  the RECEIVER's, from its first byte to EOF (nc -N closes after the last write), so nothing in flight is
#         counted; the bytes of the first read are left out of the rate, which is (bytes - first) x 8 / seconds.
#         The receiver is a few lines of python3 (3.6 on the Jetson), sent base64-encoded as pd_handoff.py does.
# link_to_json.py takes RECV where the log has it (2026-10-09 on), XFER before. RTT is ping, both directions.
# Output is tee'd to testbed/results/link_<a>-<b>_<stamp>.log; link_to_json.py turns the logs into
# testbed/dataset/link.json.
set -uo pipefail
cd "$(dirname "$0")/../.."
read -r A B <<<"${PAIR:-pi5 pi4}"
case "$A-$B" in
  pi5-pi4) declare -A IP=([pi5]=192.168.10.11 [pi4]=192.168.10.12); NET=192.168.10. ;;
  pi5-jetson) declare -A IP=([pi5]=192.168.1.122 [jetson]=192.168.1.114); NET=192.168.1. ;;   # 2026-10-19: the Pi 5 back (a
              # replacement board) on the home router; 192.168.100.107/.111 on the T628L before 2026-10-05
  pi5-laptop) declare -A IP=([pi5]=192.168.10.11 [laptop]=192.168.10.5); NET=192.168.10. ;;   # 2026-10-19: the overlay, the same
              # cable as 192.168.1.x: since 10-19 the laptop's Wi-Fi is also on 192.168.1.x, and on the overlay only its wired
              # NIC has an address (one interface on the net, as link_to_json.py requires); the laptop's end runs here
  pi4-jetson) declare -A IP=([pi4]=192.168.1.110 [jetson]=192.168.1.114); NET=192.168.1. ;;   # the home router since 2026-10-08
  pi4-laptop) declare -A IP=([pi4]=192.168.1.110 [laptop]=192.168.1.108); NET=192.168.1. ;;   # 2026-10-10: the draft's link; the
              # laptop's end runs here, not over ssh (MEASURE_PORT picks the port, so a port-shaped link can be measured)
  laptop-jetson) declare -A IP=([laptop]=192.168.1.108 [jetson]=192.168.1.114); NET=192.168.1. ;;   # 2026-10-11: the laptop as a
              # node hosts the Jetson as a worker over this path; the laptop's end runs here
  *) echo "unknown pair: $A $B"; exit 2 ;;
esac
OUT=testbed/results/link_${A}-${B}_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -i -a "$OUT") 2>&1
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8"
P=${MEASURE_PORT:-5201}; MIB=${MIB:-1024}; REPS=${REPS:-3}
on() {   # a command on a board over ssh, or here when the node is this laptop
  if [ "$1" = laptop ]; then bash -c "$2"; else $SSH -n "$1" "$2"; fi
}
read -r -d '' RECV <<'PY'
import socket, sys, time
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((sys.argv[1], int(sys.argv[2]))); s.listen(1); s.settimeout(110)
c, _ = s.accept(); c.settimeout(110); n, first, t0 = 0, 0, None
while True:
    b = c.recv(1 << 20)
    if not b:
        break
    if t0 is None:
        t0, first = time.monotonic(), len(b)
    n += len(b)
t1 = time.monotonic()
print("%d bytes, first %d, %.6f s from the first byte to EOF" % (n, first, t1 - (t0 if t0 is not None else t1)))
PY
RECV64=$(printf '%s\n' "$RECV" | base64 -w0)

echo "########## $(date -Is) link $A <-> $B, ${MIB} MiB x ${REPS} per direction, port $P"
for n in $A $B; do
  on $n "echo \"STATE $n \$(date -Is) load=\$(cut -d' ' -f1 /proc/loadavg) busy=\$(for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x \$p; done | wc -l) listening_$P=\$(ss -ltn | grep -c ':$P ') ifaces_on_net=\$(ip -4 -o addr show | grep -cF ' ${NET}')\""
done

xfer() {  # sender receiver
  local tx=$1 rx=$2 bg i got rxf
  rxf=$(mktemp)
  on "$rx" "timeout 120 python3 -c \"import base64;exec(base64.b64decode('$RECV64'))\" ${IP[$rx]} $P" > "$rxf" 2>&1 &
  bg=$!
  for i in $(seq 1 100); do on "$rx" "ss -ltn | grep -q ':$P '" && break; sleep 0.2; done
  got=$(on "$tx" "(dd if=/dev/zero bs=1M count=$MIB | nc -N ${IP[$rx]} $P) 2>&1 | tail -1")
  wait $bg
  echo "XFER $tx->$rx $got"
  echo "RECV $tx->$rx $(grep -v -e 'post-quantum' -e 'store now' -e 'openssh.com' "$rxf" | tail -1)"
  rm -f "$rxf"
}
for r in $(seq 1 "$REPS"); do xfer $A $B; xfer $B $A; done

echo "PING $A->$B"; on $A "ping -c 50 -i 0.2 ${IP[$B]} | tail -2"
echo "PING $B->$A"; on $B "ping -c 50 -i 0.2 ${IP[$A]} | tail -2"
echo "LINK DONE $(date -Is)"
