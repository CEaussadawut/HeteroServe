#!/bin/bash
# Scheduler v2's quality check (2026-10-22; the user's rule, 2026-10-21 ~23:40: "near-lossless only", measured and reported):
# how far Qwen3-32B Q6_K and Q4_K_M are from Q8_0, all three Qwen/Qwen3-32B-GGUF @ 938a7432 (sha256-checked first).
# llama-perplexity at the pin (/opt/heteroserve/llama.cpp, b23701f7) on the laptop ALONE: the same x86 kernels for every layer
# of every variant, so the comparison is the quantisation's alone, not the cluster's. The files are mapped (Q8_0 does not fit
# in memory; the page cache re-reads it per chunk); -t 4, -c 512, 8 chunks of the fixed text /usr/share/common-licenses/GPL-3
# (the text qr3_tpot.py's prompts come from). Q8_0 first, saving its logits (--kl-divergence-base); then each variant against
# them (--kl-divergence): perplexity, mean KL divergence and "Same top p" (the share of positions where both rank the same
# token first). From the repo root, with no measurement running:  bash testbed/scripts/sched2_quality_20261022.sh
# Raw output tee'd to testbed/results/sched2_quality_<stamp>.log; the base logits in /mnt/data/heteroserve-models/kld/.
set -uo pipefail
cd "$(dirname "$0")/../.."
STAMP=$(date +%Y%m%d_%H%M%S); LOG=testbed/results/sched2_quality_$STAMP.log
exec > >(tee -a "$LOG") 2>&1
B=/opt/heteroserve/llama.cpp/build/bin/llama-perplexity; TXT=/usr/share/common-licenses/GPL-3
K=/mnt/data/heteroserve-models/kld; mkdir -p $K; BASE=$K/qwen3-32b-q8_0_gpl3_c512x8_$STAMP.kld
COMMON="-f $TXT -c 512 --chunks 8 -t 4"
declare -A SUM=([q8_0]=2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169
                [q6_k]=c4c7c3cb6da260df1fe1d3cfd090a32dc7cc348f1278158be18e301f390d6f6e
                [q4_k_m]=efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689)
echo "########## $(date -Is) Qwen3-32B quality: Q6_K and Q4_K_M against Q8_0 on $TXT ($(sha256sum $TXT | cut -c1-16)), $COMMON"
echo "llama-perplexity $(sha256sum $B | cut -c1-16), llama.cpp $(git -C /opt/heteroserve/llama.cpp log -1 --format=%h)"
for v in q8_0 q6_k q4_k_m; do
  f=$(readlink -f ~/models/qwen3-32b-$v.gguf); s=$(sha256sum "$f" | cut -d' ' -f1)
  echo "model   $v: $f sha256 ${s:0:16}"; [ "$s" = "${SUM[$v]}" ] || { echo "FATAL: $v is not the pinned bytes"; exit 1; }
done
for b in llama-server llama-cli llama-bench llama-perplexity ggml-rpc-server; do pgrep -x $b >/dev/null && { echo "FATAL: $b running here"; exit 1; }; done
echo "===== $(date -Is) base: Q8_0 (saving logits to $BASE)"
t0=$(date +%s); $B -m ~/models/qwen3-32b-q8_0.gguf $COMMON --kl-divergence-base $BASE < /dev/null; echo "base exit $? in $(( $(date +%s) - t0 )) s"
ls -l $BASE && sha256sum $BASE
for v in q6_k q4_k_m; do
  echo "===== $(date -Is) $v against the Q8_0 logits"
  t0=$(date +%s); $B -m ~/models/qwen3-32b-$v.gguf $COMMON --kl-divergence-base $BASE --kl-divergence < /dev/null; echo "$v exit $? in $(( $(date +%s) - t0 )) s"
done
echo "QUALITY DONE $(date -Is)"
