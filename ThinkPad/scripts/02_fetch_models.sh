#!/usr/bin/env bash
# Download the GGUF models the cluster serves. These live only on the control node:
# workers receive layer shards over RPC, they do not hold the file.
source "$(dirname "$0")/lib.sh"
MODEL_DIR="$(cfg paths models)"
mkdir -p "$MODEL_DIR"

while read -r name repo file; do
  case "$name" in ''|\#*) continue;; esac
  out="$MODEL_DIR/$name.gguf"
  [ -f "$out" ] && { log "have $name"; continue; }
  url="https://huggingface.co/$repo/resolve/main/$file"
  log "fetching $name"
  curl -fL --retry 3 --retry-delay 5 -C - -o "$out.part" "$url" \
    && mv "$out.part" "$out" || { warn "FAILED $name"; rm -f "$out.part"; }
done < "$HS_ROOT/config/models.txt"

log "checksums (record these with the dataset)"
( cd "$MODEL_DIR" && sha256sum ./*.gguf | tee "$(cfg paths data)/inventory/model_sha256.txt" )

log "model shapes as the placement engine reads them:"
for f in "$MODEL_DIR"/*.gguf; do py -m control.gguf "$f" | head -12; echo; done
