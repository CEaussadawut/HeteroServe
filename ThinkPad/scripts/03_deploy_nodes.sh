#!/usr/bin/env bash
# Push each node's folder to the node and verify its agent answers.
# Usage: scripts/03_deploy_nodes.sh [node-id ...]   (default: every enabled worker)
source "$(dirname "$0")/lib.sh"
REPO_ROOT="$(cd "$HS_ROOT/.." && pwd)"

while IFS=$'\t' read -r id host user script_dir; do
  [ -n "${1:-}" ] && ! printf '%s\n' "$@" | grep -qx "$id" && continue
  [ -n "$user" ] || { warn "$id has no ssh_user - skipping"; continue; }

  local_dir="$REPO_ROOT/$(basename "$script_dir")"
  [ -d "$local_dir" ] || { warn "no local folder $local_dir for $id - skipping"; continue; }

  log "deploying $local_dir -> $user@$host:$script_dir"
  ssh -o BatchMode=yes "$user@$host" "sudo install -d -o $user -g $user $script_dir /opt/heteroserve"
  rsync -a --delete \
    --exclude '.venv' --exclude '__pycache__' --exclude 'data/' --exclude '*.gguf' \
    "$local_dir/" "$user@$host:$script_dir/"
  # rsync from a Windows checkout loses the exec bit; restore it rather than debugging
  # "permission denied" on the far side later.
  ssh -o BatchMode=yes "$user@$host" "chmod +x $script_dir/scripts/*.sh $script_dir/node/*.py 2>/dev/null || true"
done < <(nodes_tsv)

log "agent check:"
py -m control.cli status
