#!/usr/bin/env bash
# Shared helpers for the control-node scripts. Source this, do not run it.
set -euo pipefail

HS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HS_ROOT
CLUSTER_JSON="$HS_ROOT/config/cluster.json"
VENV="$HS_ROOT/.venv"

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- llama.cpp binary resolution -------------------------------------------
# Upstream renamed the rpc server target rpc-server -> ggml-rpc-server in
# c299a92 (2026-06-27). The cluster pins ONE SHA across four heterogeneous
# nodes and that SHA is chosen on the Jetson (its CUDA 10.2 will not build
# current master), so the name is not knowable when this script is written.
# Resolve by trying every known name, newest first. Mirrors bin_rpc in
# Pi4/scripts/lib.sh and Jetson/scripts/lib/common.sh.
find_bin() {
  local d="$1"; shift
  local n
  for n in "$@"; do
    [ -x "${d}/${n}" ] && { echo "${d}/${n}"; return 0; }
  done
  return 1
}

bin_rpc() { find_bin "${1:-$BIN}" ggml-rpc-server rpc-server llama-rpc-server; }

py() { "$VENV/bin/python" "$@"; }

# cfg paths models  ->  value of cluster.json["paths"]["models"]
cfg() {
  python3 - "$CLUSTER_JSON" "$@" <<'CFGPY'
import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2:]:
    value = value[key]
print(value)
CFGPY
}

# Node list as "id host ssh_user script_dir" lines, enabled workers only.
nodes_tsv() {
  python3 - "$CLUSTER_JSON" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
for n in cfg["nodes"]:
    if n.get("role") == "control" or not n.get("enabled"):
        continue
    print("\t".join([n["id"], n["host"], n.get("ssh_user") or "", n.get("script_dir", "")]))
PY
}
