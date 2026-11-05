#!/usr/bin/env bash
# Readiness check (proposal scope item 1: "verify readiness before every experimental run").
# Exit 0 = this node may take part in a measurement run. Non-zero = do not record data.
source "$(dirname "$0")/lib.sh"
fail=0
chk() { if eval "$2"; then printf '  \033[32mPASS\033[0m %s\n' "$1"; else printf '  \033[31mFAIL\033[0m %s -- %s\n' "$1" "$3"; fail=1; fi; }

echo "HeteroServe readiness -- $NODE_ID @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"

chk "no throttling / under-voltage" '[ "$(throttled_word)" = "0x0" ]' "throttled=$(throttled_word) -- see docs/05_troubleshooting.md"
chk "governor = performance"        '[ "$(governor)" = "performance" ]' "run sudo scripts/01_system_tune.sh"
chk "swap disabled"                 '[ "$(awk "/SwapTotal/{print \$2}" /proc/meminfo)" -eq 0 ]' "run sudo scripts/01_system_tune.sh"
chk "SoC temp < 70 C"               'awk -v t="$(soc_temp_c)" "BEGIN{exit !(t<70)}"' "currently $(soc_temp_c) C -- let it cool"
chk "free memory > 3000 MB"         '[ "$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)" -gt 3000 ]' "something is holding RAM"
chk "rpc server built"              'bin_rpc >/dev/null' "run scripts/02_build_llama_cpp.sh (looked for ggml-rpc-server, then rpc-server)"
chk "llama-bench built"             '[ -x "$LLAMA_BIN/llama-bench" ]' "run scripts/02_build_llama_cpp.sh"
# Same lesson as the rpc-server flag drift: a capability the measurement silently depends
# on must be checked by the gate that says "this node may record data", not discovered
# afterwards in the CSV.
chk "llama-bench supports --progress" '"$LLAMA_BIN/llama-bench" --help 2>&1 | grep -q -- "--progress"' \
    "the pinned llama.cpp predates --progress: energy windows would fall back to whole-process and would not be comparable with the rest of the dataset"
chk "window tooling deployed"       '[ -f "$HS_ROOT/node/stamp_lines.py" ] && [ -f "$HS_ROOT/node/bench_windows.py" ]' \
    "rsync the node/ folder again"
# systemd ACCEPTS MemoryMax= whether or not the kernel exposes the memory controller; on
# Raspberry Pi OS it is off unless cgroup_enable=memory cgroup_memory=1 is in
# /boot/cmdline.txt. Without it the rpc unit's ceiling enforces nothing and an OOM takes
# the whole board down mid-sweep instead of one service. Silent either way, so gate on it.
chk "memory cgroup controller on"   'grep -qw memory /sys/fs/cgroup/cgroup.controllers 2>/dev/null || [ -d /sys/fs/cgroup/memory ]' \
    "add 'cgroup_enable=memory cgroup_memory=1' to /boot/cmdline.txt and reboot -- without it systemd/heteroserve-rpc.service's MemoryMax is silently inert"
chk "build manifest present"        '[ -f "$DATA_DIR/inventory/build_manifest.json" ]' "rebuild to regenerate"
chk "llama.cpp revision pinned"     '[ -n "${LLAMA_PIN_SHA:-}" ]' "set LLAMA_PIN_SHA in config/node.env"
# A pin that does not describe the binary on disk is worse than no pin: the dataset would
# record a revision that never ran. Compares the hand-edited value against the commit the
# build manifest says was actually built.
chk "pin matches the built commit"  '[ -z "${LLAMA_PIN_SHA:-}" ] || [ -z "$(built_commit)" ] || [ "$(built_commit)" = "$LLAMA_PIN_SHA" ]' \
    "LLAMA_PIN_SHA=${LLAMA_PIN_SHA:-<unset>} but the manifest says $(built_commit) was built -- rebuild, or correct the pin"
chk "clock synced to control node"  'chronyc tracking 2>/dev/null | grep -q "Leap status.*Normal"' "chrony not synced -- energy/event alignment will be wrong"
clock_offset() { chronyc tracking 2>/dev/null | awk '/System time/{print $4; found=1} END{if(!found) print 9}'; }
chk "clock offset < 5 ms"           'awk -v o="$(clock_offset)" "BEGIN{exit !(o<0.005)}"' "offset $(clock_offset) s is too large for phase attribution"
chk "control node reachable"        'ping -c2 -W2 -q "$CONTROL_HOST" >/dev/null' "check the switch / cabling"
# NOT "no netem": the interconnect sweep shapes the link on purpose, and a gate that
# refused every shaped point would be bypassed for that whole sweep. What is dangerous is
# shaping nobody declared -- or a declared tier the link is no longer running.
chk "tc shaping matches declared tier" 'net_tier_ok' "$(net_tier_detail) -- run sudo scripts/05_net_tier.sh {shape|clear} to re-declare"
# Not "any .gguf exists": a partial fetch would pass that and silently shorten the sweep.
chk "every model in models.txt present" '[ -z "$(missing_models)" ]' \
    "missing: $(missing_models)-- run scripts/03_fetch_models.sh"

echo
if [ "$fail" -eq 0 ]; then
  printf '\033[32mREADY\033[0m -- clock offset %s s, temp %s C, %s MHz\n' \
    "$(chronyc tracking 2>/dev/null | awk '/System time/{print $4}')" "$(soc_temp_c)" "$(cpu_mhz)"
else
  printf '\033[31mNOT READY\033[0m -- do not record measurements from this node until the failures above are cleared\n'
fi
exit "$fail"
