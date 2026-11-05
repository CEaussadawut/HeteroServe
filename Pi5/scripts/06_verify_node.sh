#!/usr/bin/env bash
# 06 -- acceptance check. Run this before every measurement session and paste the
# output into the lab notebook. Nothing here modifies the node.
. "$(dirname "$0")/lib/common.sh"
load_env

fails=0
check() { # check <label> <command...>
  local label="$1"; shift
  if "$@" >/dev/null 2>&1; then ok "$label"; else warn "FAILED: $label"; fails=$((fails+1)); fi
}

printf '\n== identity ==\n'
assert_this_board
ok "node_id=${NODE_ID} role=${NODE_ROLE}"
ok "kernel $(uname -r)  |  $(. /etc/os-release; echo "$PRETTY_NAME")"

printf '\n== cpu features (the Pi4/Pi5 difference the dataset rests on) ==\n'
feats="$(awk -F': ' '/^Features/{print $2; exit}' /proc/cpuinfo)"
for f in asimd asimddp fphp asimdhp crc32; do
  case " $feats " in *" $f "*) ok "has $f" ;; *) warn "missing $f" ;; esac
done
case " $feats " in
  *" i8mm "*) warn "kernel reports i8mm on a Cortex-A76 -- unexpected, re-check the board" ;;
esac

printf '\n== clocks and thermals ==\n'
# Gate on EVERY core, not just cpu0: 00_prepare_os.sh pins them individually, and a
# single core left on ondemand makes the node's clocks non-uniform for the whole sweep.
govs="$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | tr '\n' ' ')"
if [ "$govs" = "${CPU_GOVERNOR} " ]; then
  ok "governor ${CPU_GOVERNOR} on all cores"
else
  warn "governors are not uniformly ${CPU_GOVERNOR} (saw: ${govs}) -- re-run 00_prepare_os.sh"
  fails=$((fails+1))
fi
ok "cpu0 now $(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) / 1000 )) MHz (max $(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq) / 1000 )) MHz)"
assert_cool_enough
if command -v vcgencmd >/dev/null 2>&1; then
  t="$(vcgencmd get_throttled)"
  if [ "$t" = "throttled=0x0" ]; then ok "$t (clean since boot)"
  else warn "$t -- a non-zero word means under-voltage or capping has occurred; results from this boot are suspect"; fails=$((fails+1)); fi
else
  # Fail closed. Without the firmware word there is no way to tell a clean board from a
  # throttled one, and this gate exists precisely to refuse a run that cannot be validated.
  # Silently skipping it let the acceptance gate print "NODE READY -- 0 blocking issues"
  # on a node whose thermal history was entirely unknown.
  warn "vcgencmd not found -- cannot read the firmware throttle word, so thermal validity is unknown"
  fails=$((fails+1))
fi

printf '\n== memory ==\n'
free -m | awk '/^Mem:/{printf "   total %s MB, available %s MB\n",$2,$7} /^Swap:/{printf "   swap  %s MB\n",$2}'
swap_total="$(free -m | awk '/^Swap:/{print $2}')"
[ "$swap_total" -eq 0 ] && ok "swap disabled" || { warn "swap is ON (${swap_total} MB) -- a swapped run is a corrupted measurement"; fails=$((fails+1)); }
avail="$(free -m | awk '/^Mem:/{print $7}')"
[ "$avail" -ge "$RPC_MEM_MB" ] && ok "can honour the advertised ${RPC_MEM_MB} MB" \
  || warn "only ${avail} MB available but ${RPC_MEM_MB} MB is advertised to the placement engine"

printf '\n== build ==\n'
if [ -f "${DATA_DIR}/build_manifest.json" ]; then
  sha="$(grep -o '"llama_cpp_sha": "[^"]*"' "${DATA_DIR}/build_manifest.json" | cut -d'"' -f4)"
  ok "llama.cpp ${sha}"
  if [ -n "${LLAMA_PIN_SHA:-}" ] && [ "$sha" != "$LLAMA_PIN_SHA" ]; then
    warn "built SHA does not match LLAMA_PIN_SHA -- rebuild before measuring"; fails=$((fails+1))
  fi
  [ -n "${LLAMA_PIN_SHA:-}" ] || { warn "LLAMA_PIN_SHA is still empty in node.env"; fails=$((fails+1)); }
else
  warn "no build manifest -- run 02_build_llama_cpp.sh"; fails=$((fails+1))
fi
for b in llama-cli llama-bench; do
  [ -x "${LLAMA_BIN}/$b" ] && ok "binary $b" || { warn "missing binary $b"; fails=$((fails+1)); }
done
# Either name is correct depending on the pinned SHA: upstream renamed the target
# rpc-server -> ggml-rpc-server in c299a92 (2026-06-27).
if   [ -x "${LLAMA_BIN}/ggml-rpc-server" ]; then ok "binary ggml-rpc-server"
elif [ -x "${LLAMA_BIN}/rpc-server" ];      then ok "binary rpc-server"
else warn "missing the rpc server binary (looked for ggml-rpc-server and rpc-server)"; fails=$((fails+1)); fi
# The unit launches a generated wrapper, not the binary directly, so that the --mem /
# --cache flags match the build. If it is missing, 04_install_services.sh has not been
# re-run since the last rebuild and the service is launching a stale interface.
if [ -x /opt/heteroserve/bin/rpc-server.sh ]; then
  ok "rpc launcher present ($(awk '/^# Detected flags/{ $1=""; $2=""; $3=""; print substr($0,4) }' /opt/heteroserve/bin/rpc-server.sh))"
else
  warn "no /opt/heteroserve/bin/rpc-server.sh -- re-run 04_install_services.sh"; fails=$((fails+1))
fi

printf '\n== models ==\n'
n=0
for m in "${MODEL_DIR}"/*.gguf; do [ -f "$m" ] && { n=$((n+1)); printf '   %-28s %s\n' "$(basename "$m")" "$(du -h "$m" | cut -f1)"; }; done
[ "$n" -gt 0 ] && ok "${n} GGUF files" || { warn "no models -- run 03_fetch_models.sh"; fails=$((fails+1)); }

printf '\n== network ==\n'
# Defence in depth: 05 is a reporting script and must never be able to abort the
# acceptance gate. If it does exit non-zero, that is itself a finding -- record it
# and carry on to the remaining checks rather than dying here with half a report.
"${HS_SCRIPT_DIR}/05_network_setup.sh" show \
  || { warn "05_network_setup.sh show exited $? -- network state above may be incomplete"; fails=$((fails+1)); }
check "control node ${CONTROL_HOST} reachable" ping -c1 -W2 "$CONTROL_HOST"
check "clock synchronised" bash -c 'timedatectl show -p NTPSynchronized --value | grep -q yes'
offset="$(chronyc tracking 2>/dev/null | awk -F': ' '/System time/{print $2}')"
[ -n "$offset" ] && ok "clock offset ${offset}"

printf '\n== services ==\n'
for u in heteroserve-rpc heteroserve-agent heteroserve-telemetry; do
  if systemctl is-active --quiet "$u"; then ok "$u active"
  else warn "$u not active"; fails=$((fails+1)); fi
done
if command -v curl >/dev/null 2>&1; then
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${AGENT_PORT}/ready" || echo 000)"
  [ "$code" = 200 ] && ok "agent /ready -> 200" \
    || { warn "agent /ready -> ${code}"; curl -s "http://127.0.0.1:${AGENT_PORT}/ready" | head -20; fails=$((fails+1)); }
fi

printf '\n== peers ==\n'
while read -r id host port role iface rest; do
  case "${id:-#}" in ''|\#*) continue ;; esac
  [ "$id" = "$NODE_ID" ] && continue
  if ping -c1 -W1 "$host" >/dev/null 2>&1; then ok "${id} (${host}) up"; else warn "${id} (${host}) unreachable"; fi
done < "${HS_NODE_DIR}/config/cluster.hosts"

printf '\n'
if [ "$fails" -eq 0 ]; then ok "NODE READY -- ${fails} blocking issues"; exit 0
else warn "${fails} blocking issue(s); do not record measurements until they are cleared"; exit 1; fi
