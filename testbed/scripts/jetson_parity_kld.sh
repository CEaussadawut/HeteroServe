#!/bin/bash
# CPU-backend parity check for the Jetson's GPU builds. The pin note (cluster.json llama._pin_note) says a patch that
# touches kernel math or precision "wants a CPU-backend parity check on the same node", and patch v2 does (bf16 -> half,
# the FlashAttention dispatch). TinyLlama 1.1B Q4_K_M, a fixed public text, ctx 512, CHUNKS chunks:
#   1. the reference: the UNPATCHED pin (~/llama.cpp) on the CPU (-ngl 0 -dev none) -> perplexity, and its logits saved;
#   2. CUDA, the pin + patch v2 (~/llama.cpp-cuda), -ngl 99, scored against those logits: KL divergence, top-token agreement;
#   3. Vulkan, the unpatched pin, -ngl 99, the same -- the other GPU backend, for scale.
# Run ON the board, as user jetson, with jetson_power_guard.sh running:
#   ssh jetson 'setsid nohup bash ~/heteroserve/tools/jetson_parity_kld.sh </dev/null >/dev/null 2>&1 &'
# Raw output is tee'd to ~/heteroserve/results/jetson_parity_<stamp>/run.log; the exit code goes to results/parity.done.
set -u
M=${MODEL:-$HOME/models/tinyllama-1.1b-q4km.gguf}
TXT=${TEXT:-/usr/share/common-licenses/GPL-3}
CHUNKS=${CHUNKS:-8}
V=$HOME/llama.cpp/build/bin
C=$HOME/llama.cpp-cuda/build/bin
OUT=$HOME/heteroserve/results/jetson_parity_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
rm -f "$HOME/heteroserve/results/parity.done"
trap 'echo $? > "$HOME/heteroserve/results/parity.done"' EXIT
echo "$OUT" > "$HOME/heteroserve/results/LATEST_PARITY"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "########## $(date -Is) Jetson CPU-backend parity: $(basename "$M"), ctx 512, $CHUNKS chunks of $TXT"
pgrep -f '[j]etson_power_guard' >/dev/null || { echo "FATAL: the power guard is not running"; exit 3; }
echo "model  $(sha256sum "$M")"
echo "text   $(sha256sum "$TXT")"
echo "cpu    $V  commit $(git -C "$HOME/llama.cpp" rev-parse HEAD), $(git -C "$HOME/llama.cpp" status --porcelain --untracked-files=no | wc -l) modified files"
echo "cuda   $C  commit $(git -C "$HOME/llama.cpp-cuda" rev-parse HEAD), patch $(sha256sum "$HOME/heteroserve/patches/jetpack46-cuda102-b23701f.patch" | cut -d' ' -f1)"
echo "power  $(cut -d' ' -f1 /var/lib/nvpmodel/status); CPU $(( $(cat /sys/devices/virtual/thermal/thermal_zone1/temp) / 1000 )) C"

echo; echo "===== 1. CPU reference (the unpatched pin, -ngl 0 -dev none), logits saved"
"$V/llama-perplexity" -m "$M" -f "$TXT" -c 512 --chunks "$CHUNKS" -t 4 -ngl 0 -dev none --kl-divergence-base "$OUT/cpu.kld"
echo "EXIT_cpu=$?"
echo; echo "===== 2. CUDA (pin + patch v2, -ngl 99) against the CPU logits"
"$C/llama-perplexity" -m "$M" -c 512 -t 4 -ngl 99 --kl-divergence-base "$OUT/cpu.kld" --kl-divergence
echo "EXIT_cuda=$?"
echo; echo "===== 3. Vulkan (the unpatched pin, -ngl 99) against the CPU logits"
"$V/llama-perplexity" -m "$M" -c 512 -t 4 -ngl 99 --kl-divergence-base "$OUT/cpu.kld" --kl-divergence
echo "EXIT_vulkan=$?"
echo "########## $(date -Is) done; logits file $(stat -c %s "$OUT/cpu.kld" 2>/dev/null) bytes"
