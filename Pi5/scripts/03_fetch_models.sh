#!/usr/bin/env bash
# 03 -- pull the GGUF models listed in config/models.txt into MODEL_DIR.
# Only the models the Pi 5 runs in ISOLATION during Phase 2 live here. The oversized
# configurations (7B Q8_0, 14B Q4) are never resident on a worker; they arrive as
# layer shards over RPC from the control node.
. "$(dirname "$0")/lib/common.sh"
load_env

# huggingface_hub renamed its entry point huggingface-cli -> hf; accept either so the
# script does not break on a venv rebuild.
HF=""
for cand in /opt/heteroserve/venv/bin/hf /opt/heteroserve/venv/bin/huggingface-cli; do
  if [ -x "$cand" ]; then HF="$cand"; break; fi
done
[ -n "$HF" ] || die "no hf/huggingface-cli in /opt/heteroserve/venv -- run 01_install_deps.sh first"
install -d "$MODEL_DIR"

total=0; got=0
while read -r name repo file; do
  case "${name:-#}" in ''|\#*) continue ;; esac
  total=$((total+1))
  # A split GGUF (<stem>-00001-of-000NN.gguf) cannot survive the fetch-and-rename
  # below: llama.cpp locates the sibling shards by pattern from the first shard's own
  # filename, so renaming it to <local-name>.gguf yields a model that will not load --
  # and a 4 GB file that looks fine sitting in MODEL_DIR. Refuse it loudly instead.
  case "$file" in
    *-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].gguf)
      warn "skipping ${name}: ${file} is one shard of a split GGUF"
      warn "  point config/models.txt at a single-file repack of the same quant"
      continue ;;
  esac
  dest="${MODEL_DIR}/${name}.gguf"
  if [ -f "$dest" ]; then ok "have ${name}"; got=$((got+1)); continue; fi
  log "fetching ${name}  <-  ${repo}/${file}"
  tmp="$(mktemp -d "${MODEL_DIR}/.dl.XXXXXX")"
  if "$HF" download "$repo" "$file" --local-dir "$tmp" >/dev/null; then
    mv "${tmp}/${file}" "$dest"
    got=$((got+1)); ok "${name}  $(du -h "$dest" | cut -f1)"
  else
    warn "could not fetch ${name} -- check the repo/file names in config/models.txt"
  fi
  rm -rf "$tmp"
done < "${HS_NODE_DIR}/config/models.txt"

log "checksumming for the reproducibility artifact"
( cd "$MODEL_DIR" && sha256sum ./*.gguf > "${DATA_DIR}/models.sha256" ) 2>/dev/null || true
ok "${got}/${total} models present in ${MODEL_DIR}"
df -h "$MODEL_DIR" | tail -1
