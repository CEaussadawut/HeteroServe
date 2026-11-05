#!/bin/bash
# ss_sample.sh FILTER OUT [INTERVAL_S]: the kernel's tcp_info for the matching established sockets, every INTERVAL_S seconds
# (default 0.05), each sample headed "T <epoch seconds>" -- read-only (ss), no setting changed. Copied to a board and started
# there by tcpdiag_campaign_20261009.sh, e.g.  bash ~/ss_sample.sh '( sport = :50052 )' ~/logs/ss_x.log 0.05
F=${1:?filter}; O=${2:?out}; I=${3:-0.05}
while :; do echo "T $(date +%s.%N)"; ss -tin state established "$F"; sleep "$I"; done > "$O" 2>&1
