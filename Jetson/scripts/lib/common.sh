#!/usr/bin/env bash
# Shared helpers for every HeteroServe Jetson script.
# Source, do not execute:  . "$(dirname "$0")/lib/common.sh"

set -euo pipefail

# Every script here inherits those options, which has one sharp edge worth
# knowing before you add a line: a command substitution containing a pipe,
#
#     X="$(a | b | c)"
#
# aborts the WHOLE script if any stage fails. The stages used in this kit fail
# routinely and legitimately -- `timeout` returns 124 on a command that never
# exits by design, `ip route get` fails with no default route, `ethtool` fails
# on a name it does not know, `grep` returns 1 on no match, `ls` returns 2 on an
# unmatched glob. Those are precisely the degraded conditions these scripts
# exist to REPORT, so dying on them is the opposite of useful -- and it dies
# before printing anything, which under `make ... | tee` looks like success.
#
# So for a fact-gathering pipeline whose absence is survivable, write:
#
#     X="$( set +o pipefail; a | b | c || true )"
#
# The subshell keeps the relaxed setting local, and the `|| true` makes the
# substitution itself succeed. Leave a bare pipeline unguarded only when you
# genuinely want the script to stop.

HS_JETSON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HS_JETSON_DIR

_c() { printf '\033[%sm%s\033[0m\n' "$1" "$2"; }
info()  { _c '0;36' "==> $*"; }
ok()    { _c '0;32' "  ok  $*"; }
warn()  { _c '1;33' "  !!  $*" >&2; }
die()   { _c '0;31' "  XX  $*" >&2; exit 1; }
step()  { printf '\n'; _c '1;37' "── $* ─────────────────────────────"; }

load_env() {
  local f="${HS_JETSON_DIR}/config/node.env"
  [ -f "$f" ] || die "missing ${f} — run: cp config/node.env.example config/node.env"
  # shellcheck disable=SC1090
  set -a; . "$f"; set +a
  : "${NODE_ID:?NODE_ID unset}"
  mkdir -p "$RESULTS_DIR" "$MODEL_DIR" "$HS_ROOT"
}

require_jetson() {
  [ -f /etc/nv_tegra_release ] || die "this is not a Jetson (no /etc/nv_tegra_release)"
}

# Only warn — several scripts (parsing, plotting) are useful off-device too.
is_jetson() { [ -f /etc/nv_tegra_release ]; }

# These report a fact or print nothing. They must never abort the caller: every
# one of them is asked off-device too (parsing and plotting run on a laptop), and
# a fact-gatherer that kills the script when the fact is absent is useless. See
# the pipefail note at the top -- a bare `awk ... | head` here is fatal on any
# machine without /etc/nv_tegra_release.
l4t_version() {
  # e.g. "32.7.3" from: # R32 (release), REVISION: 7.3, GCID: ...
  ( set +o pipefail
    awk -F'[ ,]+' '/^# R/{r=$2; sub("R","",r)} /REVISION:/{for(i=1;i<=NF;i++) if($i=="REVISION:") print r"."$(i+1)}' \
      /etc/nv_tegra_release 2>/dev/null | head -1 || true )
}

thermal_zone_c() {
    # Board temperature as "<zone-type> <degrees C>", or nothing.
    #
    # Selected by the zone's `type` file, never by zone NUMBER and never by
    # taking the max. Both of those were wrong here: the numbers are not stable
    # across L4T releases, and the hottest zone on a Nano is PMIC-Die, a dummy
    # that reports a constant 100 C -- so a `sort -n | tail -1` pre-flight gate
    # reported 100 C on a stone-cold board, every time, making the "let it settle
    # below 45 C" warning fire always and mean nothing.
    #
    # thermal-fan-est first: that is the same quantity tegrastats prints as
    # `thermal@`, which the logger records as temp_thermal_c, so the pre-flight
    # reading and every sweep row are the same number. A board with no fan
    # estimator registered falls back to the hotter of CPU-therm and GPU-therm.
    # PMIC-Die can never be selected.
    local root="${HS_SYSFS_ROOT:-}"
    local want zone t type_f best_t="" best_type=""
    for want in thermal-fan-est CPU-therm GPU-therm; do
      for zone in "${root}"/sys/devices/virtual/thermal/thermal_zone*/ \
                  "${root}"/sys/class/thermal/thermal_zone*/; do
        type_f="${zone}type"
        [ -r "$type_f" ] || continue
        [ "$(cat "$type_f" 2>/dev/null)" = "$want" ] || continue
        t="$(cat "${zone}temp" 2>/dev/null || true)"
        case "$t" in ''|*[!0-9-]*) continue ;; esac
        if [ "$want" = "thermal-fan-est" ]; then
          printf '%s %d' "$want" $((t / 1000))
          return 0
        fi
        # CPU-therm / GPU-therm: keep the hotter of the two
        if [ -z "$best_t" ] || [ "$t" -gt "$best_t" ]; then
          best_t="$t"; best_type="$want"
        fi
      done
    done
    [ -n "$best_t" ] && printf '%s %d' "$best_type" $((best_t / 1000))
    return 0
}

cuda_version() {
  ( set +o pipefail
    if command -v nvcc >/dev/null 2>&1; then
      nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1
    elif [ -f /usr/local/cuda/version.txt ]; then
      sed -n 's/.*Version \([0-9.]*\).*/\1/p' /usr/local/cuda/version.txt | head -1
    else
      echo ""
    fi
    true )
}

cuda_major() { ( set +o pipefail; cuda_version | cut -d. -f1 || true ); }

# llama.cpp renamed its binaries (main -> llama-cli, etc.) and the rpc server
# target became ggml-rpc-server in c299a92 (2026-06-27), from rpc-server.
# Resolve by trying every known name, newest first, so a pinned older build still
# works. The pinned SHA is chosen on this node (CUDA 10.2 constrains the whole
# cluster), so the name is not knowable at authoring time.
#
# CAVEAT for llama-rpc-server: /proc/<pid>/comm holds 15 chars, so that name
# truncates to "llama-rpc-serve" and the telemetry RSS probe -- which matches on
# the substring "rpc-server" -- would silently report -1. ggml-rpc-server is
# exactly 15 and survives. No known ref builds llama-rpc-server; it is kept only
# as a last resort.
find_bin() {
  local d="$1"; shift
  local n
  for n in "$@"; do
    [ -x "${d}/${n}" ] && { echo "${d}/${n}"; return 0; }
  done
  return 1
}

# ONE list of rpc-server names, newest first. Both the binary lookup and the
# "is a worker already running" guard read it, so a future rename cannot fix one
# and silently leave the other blind -- which is exactly what a second,
# hand-maintained list in 50_bench_device.sh used to do.
RPC_BIN_NAMES="ggml-rpc-server rpc-server llama-rpc-server"

rpc_comm_re() {
  # An ERE matching every name in RPC_BIN_NAMES as /proc/<pid>/comm holds it.
  # comm is capped at 15 characters (TASK_COMM_LEN-1), which is why the 16-char
  # llama-rpc-server has to be matched as llama-rpc-serve -- see the caveat above.
  local n out=""
  for n in $RPC_BIN_NAMES; do
    out="${out:+$out|}${n:0:15}"
  done
  printf '%s' "$out"
}

rpc_running() {
  # Anchored to the process NAME (-x, no -f). `pgrep -f rpc-server` would also
  # match the operator's own `vim scripts/40_run_rpc_server.sh`, the `less`
  # reading the audit, and the ssh command that started the sweep -- a false
  # warning during ordinary work trains people to answer y to the confirm
  # prompt, which costs more than the check saves.
  pgrep -x "$(rpc_comm_re)" >/dev/null 2>&1 && return 0
  # Port fallback: the only check that survives the NEXT upstream rename, since
  # a worker has to bind the port to be of any use whatever it is called.
  [ -n "${RPC_PORT:-}" ] || return 1
  ( set +o pipefail
    ss -ltn 2>/dev/null | awk -v p=":${RPC_PORT}$" '$4 ~ p {found=1} END{exit !found}' )
}

tegrastats_running() {
  # A second tegrastats instance makes the first one's output stop, so the
  # sweep's logger dies at startup and every energy column comes back empty.
  # Checked before a sweep, not diagnosed after one.
  pgrep -x tegrastats >/dev/null 2>&1
}

bin_bench()  { find_bin "$LLAMA_BIN" llama-bench bench; }
bin_cli()    { find_bin "$LLAMA_BIN" llama-cli main; }
bin_rpc()    { find_bin "$LLAMA_BIN" $RPC_BIN_NAMES; }
bin_server() { find_bin "$LLAMA_BIN" llama-server server; }

timestamp() { date -u +%Y%m%dT%H%M%SZ; }

# Everything written to results/ carries the node id and a UTC stamp so files
# from four devices can be merged without collisions.
result_path() { echo "${RESULTS_DIR}/${NODE_ID}_$1_$(timestamp).$2"; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1${2:+ ($2)}"; }

confirm() {
  [ "${HS_YES:-0}" = "1" ] && return 0
  read -r -p "$1 [y/N] " a; [ "$a" = "y" ] || [ "$a" = "Y" ]
}
