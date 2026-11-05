#!/usr/bin/env bash
# Pull the small GGUF models this node runs in isolation (Phase 2 characterisation).
# Large models are NOT stored here; they reach this node as layer shards over RPC.
source "$(dirname "$0")/lib.sh"
ensure_dirs
MODEL_DIR="$DATA_DIR/models"; mkdir -p "$MODEL_DIR"

# 128 GB card, models are ~0.7-2.2 GB each. Warn early rather than half-download.
avail=$(df -m "$MODEL_DIR" | awk 'NR==2{print $4}')
[ "$avail" -gt 8000 ] || warn "only ${avail} MB free on the model volume"

while read -r name repo file; do
  case "$name" in ''|\#*) continue;; esac
  out="$MODEL_DIR/$name.gguf"
  if [ -f "$out" ]; then log "have $name"; continue; fi
  url="https://huggingface.co/$repo/resolve/main/$file"
  log "fetching $name  <- $url"
  curl -fL --retry 3 --retry-delay 5 -C - -o "$out.part" "$url" || { warn "FAILED $name"; rm -f "$out.part"; continue; }
  mv "$out.part" "$out"
done < "$HS_ROOT/config/models.txt"

miss="$(missing_models)"
if [ -n "$miss" ]; then
  warn "INCOMPLETE model set -- missing: $miss"
  warn "The sweep skips a missing model with a warning, so the dataset would simply have"
  warn "fewer rows than the matrix and nothing would record why. Re-run this script."
fi

log "model inventory:"
( cd "$MODEL_DIR" && sha256sum *.gguf 2>/dev/null | tee "$DATA_DIR/inventory/model_sha256.txt" )
ls -lh "$MODEL_DIR"
