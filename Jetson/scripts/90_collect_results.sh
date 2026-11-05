#!/usr/bin/env bash
# Bundle this node's results, facts and build provenance for transfer to the
# analysis machine. Everything a figure in the report is derived from should be
# inside this tarball.
#   ./90_collect_results.sh [outfile.tar.gz]
. "$(dirname "$0")/lib/common.sh"
load_env

OUT="${1:-${HS_ROOT}/${NODE_ID}_results_$(timestamp).tar.gz}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
D="${STAGE}/${NODE_ID}"
mkdir -p "$D"

step "Staging"
"${HS_JETSON_DIR}/scripts/60_collect_node_facts.sh" > "${D}/node_facts.json" 2>/dev/null \
  && ok "node_facts.json" || warn "could not collect node facts"
[ -f "${HS_ROOT}/build_info.json" ] && cp "${HS_ROOT}/build_info.json" "$D/"
cp "${HS_JETSON_DIR}/config/node.env" "${D}/node.env.snapshot" 2>/dev/null || true
cp "${HS_JETSON_DIR}/bench/matrix.json" "${D}/matrix.json" 2>/dev/null || true
[ -f "${MODEL_DIR}/SHA256SUMS" ] && cp "${MODEL_DIR}/SHA256SUMS" "${D}/models.sha256"
mkdir -p "${D}/results"
# .marker_* is scratch state, not data.
find "$RESULTS_DIR" -maxdepth 1 -type f ! -name '.marker_*' -exec cp {} "${D}/results/" \; 2>/dev/null || true
dmesg 2>/dev/null | grep -iE 'oom|throttl|thermal' | tail -50 > "${D}/kernel_warnings.txt" || true

step "Archive"
tar -czf "$OUT" -C "$STAGE" "$NODE_ID"
ok "$(du -h "$OUT" | cut -f1)  ${OUT}"
tar -tzf "$OUT" | sed 's/^/  /' | head -40
echo
echo "Copy to the analysis machine:  scp ${OUT} user@laptop:~/heteroserve/incoming/"
