#!/bin/bash
# Qwen2.5-14B-Instruct GGUF Q4_K_M, pinned to repo revision b466e1f8...
REV=b466e1f8c07172155743e8e1307507d8a4f91fbd
BASE=https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/$REV
cd ~/models || exit 1
for i in 00001 00002 00003; do
  f=qwen2.5-14b-instruct-q4_k_m-${i}-of-00003.gguf
  echo "=== $(date -Is) fetching $f"
  wget -c -q --show-progress --progress=dot:giga -O "$f" "$BASE/$f" || { echo "FAILED $f"; exit 1; }
done
echo "=== $(date -Is) done"
ls -la ~/models/qwen2.5-14b*
sha256sum ~/models/qwen2.5-14b* 
