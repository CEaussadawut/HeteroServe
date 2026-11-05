#!/usr/bin/env bash
# 10 -- Phase 2 isolated device characterisation for the Pi 5.
#
# Runs llama-bench over bench/sweep.conf with the node in its pinned measurement
# state, tags the node telemetry and the DAQ stream with the same cell id, and
# writes one CSV row per cell. This produces this node's slice of the device
# characterisation dataset -- the project's first deliverable -- and in particular
# the per-layer per-token compute time c_i that the placement rule (Appendix B.6)
# consumes.
#
#   ./10_isolated_bench.sh [--reps 3] [--sweep bench/sweep.conf] [--dry-run]
. "$(dirname "$0")/lib/common.sh"
load_env
assert_this_board

REPS=3; SWEEP="${HS_NODE_DIR}/bench/sweep.conf"; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --reps) REPS="$2"; shift 2 ;;
    --sweep) SWEEP="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) die "usage: $0 [--reps N] [--sweep FILE] [--dry-run]" ;;
  esac
done

[ -x "${LLAMA_BIN}/llama-bench" ] || die "llama-bench missing -- run 02_build_llama_cpp.sh"
[ -f "$SWEEP" ] || die "no sweep file at $SWEEP"
[ -n "${LLAMA_PIN_SHA:-}" ] || die "LLAMA_PIN_SHA is empty -- pin the revision before recording a dataset"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${RESULTS_DIR}/${NODE_ID}_isolated_${RUN_ID}"
install -d "$OUT"
log "run ${RUN_ID} -> ${OUT}"

# Provenance travels with the numbers, or the numbers are not reproducible.
cp "$SWEEP" "${OUT}/sweep.conf"
cp "$HS_CONFIG" "${OUT}/node.env.snapshot"
[ -f "${DATA_DIR}/build_manifest.json" ] && cp "${DATA_DIR}/build_manifest.json" "$OUT/"
[ -f "${DATA_DIR}/models.sha256" ] && cp "${DATA_DIR}/models.sha256" "$OUT/"

INDEX="${OUT}/cells.csv"
# throttled_start/throttled_end are the firmware word at each cell boundary. Comparing the
# two is the only way to tell "this cell throttled" from the sticky since-boot bits, which
# latch for the rest of the boot and would otherwise flag every row after the first blip.
# idle_start/idle_end bound the cool-down that PRECEDED this cell, as epoch seconds, so a
# marginal (above-idle) power figure can be computed per row instead of against a
# session-wide mean that pools a cold board with a hot one.
echo "cell_id,started_utc,ended_utc,model,n_layers,n_prompt,n_gen,threads,rep,exit_code,temp_start_c,temp_end_c,throttled_start,throttled_end,idle_start,idle_end,raw_json" > "$INDEX"

# The cool-down before the FIRST cell has not happened yet, so these start empty and are
# filled by the cool-down at the end of each iteration, for the cell that follows it.
IDLE_START=""; IDLE_END=""

tag_run() { # tell the agent (and through it the DAQ) which cell is running
  curl -s -m 2 -X POST -H 'Content-Type: application/json' \
    -d "{\"tag\":\"$1\"}" "http://127.0.0.1:${AGENT_PORT}/run-tag" >/dev/null 2>&1 || true
}

cell_n=0
while read -r model np ng th; do
  case "${model:-#}" in ''|\#*) continue ;; esac
  gguf="${MODEL_DIR}/${model}.gguf"
  [ -f "$gguf" ] || { warn "skipping ${model}: not in ${MODEL_DIR}"; continue; }

  # llama-bench does NOT emit a layer count -- its JSON field list is fixed and has
  # model_size/model_n_params/model_type but no block count (n_gpu_layers is the
  # offload count, 0 on this CPU-only node). So read it from the GGUF header here and
  # carry it in cells.csv. Without it c_per_layer_ms -- the c_i Rule A consumes -- is
  # not computable, which would make every row of this sweep useless to the placement
  # engine, so a failure here is fatal rather than a warning.
  n_layers="$(python3 "${HS_NODE_DIR}/node/gguf_meta.py" "$gguf" --field n_layers)" \
    || die "cannot read the layer count from ${gguf} -- the dataset would carry no c_i"
  log "${model}: ${n_layers} layers"

  for rep in $(seq 1 "$REPS"); do
    cell_n=$((cell_n+1))
    CELL="${NODE_ID}_${model}_p${np}_g${ng}_t${th}_r${rep}"
    printf '\n'
    log "cell ${cell_n}: ${CELL}"

    # Gate before every single cell, not once per sweep: thermal history is the
    # variable most likely to make two identical cells disagree (risk R5).
    assert_cool_enough

    t0="$(soc_temp_c)"; s0="$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
    w0="$(throttled_word)"
    tag_run "$CELL"

    raw="${OUT}/${CELL}.json"
    if [ "$DRY" = 1 ]; then
      log "dry-run: ${LLAMA_BIN}/llama-bench -m ${gguf} -p ${np} -n ${ng} -t ${th}"
      rc=0; echo '[]' > "$raw"
    else
      set +e
      "${LLAMA_BIN}/llama-bench" \
        --model "$gguf" \
        --n-prompt "$np" --n-gen "$ng" \
        --threads "$th" \
        --repetitions 5 \
        --output json \
        > "$raw" 2> "${OUT}/${CELL}.stderr"
      rc=$?
      set -e
    fi
    t1="$(soc_temp_c)"; s1="$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
    w1="$(throttled_word)"
    tag_run ""

    echo "${CELL},${s0},${s1},${model},${n_layers},${np},${ng},${th},${rep},${rc},${t0},${t1},${w0},${w1},${IDLE_START},${IDLE_END},$(basename "$raw")" >> "$INDEX"
    [ "$rc" -eq 0 ] && ok "cell done (${t0} C -> ${t1} C)" || warn "cell exited ${rc}; see ${CELL}.stderr"

    # Cool-down between cells so the next one starts from the same thermal state
    # rather than inheriting this one's heat.
    # Bracketed in epoch seconds -- the same time base the DAQ uses -- so this settle is
    # an idle window the energy join can integrate over, not just a delay. It becomes the
    # NEXT cell's idle_start/idle_end, because Pi 4's schema defines those as the idle
    # window recorded immediately BEFORE a run.
    log "cooling to <= ${THERMAL_START_MAX_C} C"
    idle_from="$(date -u +%s.%N)"
    for _ in $(seq 1 120); do
      ct="$(soc_temp_c)"
      # A bad sensor must not spin here for ten minutes; the next cell's gate will die.
      [ "$ct" = "nan" ] && break
      awk -v t="$ct" -v m="${THERMAL_START_MAX_C}" 'BEGIN{exit !(t+0<=m+0)}' && break
      sleep 5
    done
    IDLE_START="$idle_from"; IDLE_END="$(date -u +%s.%N)"
  done
done < "$SWEEP"

printf '\n'
ok "${cell_n} cells written to ${OUT}"
log "parse with:  node/../bench/parse_bench.py ${OUT}"
