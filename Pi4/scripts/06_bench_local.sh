#!/usr/bin/env bash
# Phase 2 -- isolated device characterisation of the Pi 4.
# Produces the rows this node contributes to the device-characterisation dataset,
# which is the project's FIRST deliverable and the input to the placement rule.
#
#   scripts/06_bench_local.sh              full sweep (all models x all shapes)
#   scripts/06_bench_local.sh quick        one model, one shape (smoke test)
#   scripts/06_bench_local.sh validate     R=1,3,5 cross-check of the energy window
source "$(dirname "$0")/lib.sh"
ensure_dirs
assert_clean_state

# An ISOLATED characterisation row measures this board alone. If the rpc server is up the
# node may be serving cluster traffic on the same four cores and the same 5 V rail the
# INA226 is watching, and nothing downstream could tell such a row from a clean one --
# the contamination just looks like ordinary slowness, which is this node's whole subject.
#
# This lives HERE and not in assert_clean_state: scripts/04_rpc_server.sh also calls that
# helper, and it is heteroserve-rpc's own ExecStart. systemd marks a Type=simple unit
# active as soon as the process forks, so the check would fire against the script's own
# unit, die, and Restart=always would loop it for ever.
if systemctl is-active --quiet heteroserve-rpc 2>/dev/null; then
  die "heteroserve-rpc is active -- an isolated characterisation row cannot be taken while this node serves the cluster (same cores, same rail). Stop it first: scripts/04_rpc_server.sh stop"
fi

# --progress is the only in-process signal that says where the repetitions are. Without
# it the energy window silently reverts to the whole process -- which is the exact bug
# this sweep was rewritten to remove -- and the CSV would still look self-consistent.
# Refuse to record anything rather than record something that cannot be interpreted.
# NOTE the `--` in `grep -q --`: without it grep parses --progress as its own option.
"$LLAMA_BIN/llama-bench" --help 2>&1 | grep -q -- "--progress" \
  || die "this llama-bench has no --progress flag: repetition sub-windows cannot be derived, so joules-per-token would silently include model load and a full warmup prefill. Check LLAMA_PIN_SHA and rebuild (scripts/02_build_llama_cpp.sh) before recording."
[ -f "$HS_ROOT/node/stamp_lines.py" ]   || die "node/stamp_lines.py missing -- deploy is incomplete"
[ -f "$HS_ROOT/node/bench_windows.py" ] || die "node/bench_windows.py missing -- deploy is incomplete"

MODEL_DIR="$DATA_DIR/models"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
MODE="${1:-full}"

# validate mode writes to a SEPARATE directory, deliberately. Its filenames carry a third
# underscore field (_r<REPS>) and node/bench_parse.py recovers the model name with
# rsplit("_", 2) -- point it at this directory and every row joins to the wrong window,
# which is a silently wrong number of exactly the kind this sweep exists to avoid.
if [ "$MODE" = "validate" ]; then
  OUT="$DATA_DIR/bench/${RUN_ID}-validate"
else
  OUT="$DATA_DIR/bench/$RUN_ID"
fi
mkdir -p "$OUT"
cp "$DATA_DIR/inventory/build_manifest.json" "$OUT/" 2>/dev/null || warn "no build manifest -- results will not be reproducible"

VALIDATE=0
REPS_SET=()
if [ "$MODE" = "quick" ]; then
  MODELS=(tinyllama-1.1b-q4km); PP=(128); TG=(64); REPS=2
elif [ "$MODE" = "validate" ]; then
  # Independent cross-check of the repetition-window split, using NO timestamps at all.
  # Energy over a whole llama-bench process is affine in the repetition count:
  #     E_process(R) = E_fixed + R * E_rep
  # so running the same cell at R = 1, 3, 5 and fitting a line recovers E_rep (the slope)
  # and E_fixed (the intercept) by the method of differences. The slope must agree with the
  # mean per-repetition energy derived from the --progress sub-windows, and the intercept
  # with startup + warmup energy.
  #
  # This is the paragraph that answers "how do you know your sub-window is right?": two
  # routes to the same split, and the second one does not depend on trusting any timestamp,
  # any clock, or any assumption about llama.cpp's internals.
  MODELS=(tinyllama-1.1b-q4km llama3.2-3b-q4km); PP=(128); TG=(128)
  REPS_SET=(1 3 5); VALIDATE=1
else
  MODELS=(tinyllama-1.1b-q4km tinyllama-1.1b-q8 qwen2.5-1.5b-q4km llama3.2-3b-q4km)
  PP=(128 512); TG=(128); REPS=5
fi
if [ "$VALIDATE" -eq 1 ]; then
  THREADS_SWEEP=(4)       # the fit is per-cell; sweeping threads too would multiply cost
else
  THREADS_SWEEP=(1 2 4)   # 4 = all cores; 1-2 feed the straggler study (QR8: reduced load)
fi

log "run $RUN_ID  ->  $OUT"

# Prefill and decode are run as SEPARATE llama-bench invocations, never in one mixed run.
# llama-bench would happily interleave pp and tg inside one process, but then the INA226
# stream has no boundary to attribute energy to -- and separating prefill energy from
# decode energy is a stated requirement (scope item 5). One invocation = one phase = one
# clean power window bracketed by markers.
run_one() {
  local model="$1" phase="$2" flag="$3" val="$4" threads="$5"
  local path="$MODEL_DIR/$model.gguf"
  [ -f "$path" ] || { warn "missing model $model -- skipping"; return; }
  local tag="${model}_${phase}${val}_t${threads}"
  # `if`, not `[ ... ] && tag=...`: under `set -e` that shape is a known trap in this
  # repo when it lands as a function's last command, and it is not worth the ambiguity.
  if [ "$VALIDATE" -eq 1 ]; then tag="${tag}_r${REPS}"; fi   # separate dir; see MODE block
  local json="$OUT/${tag}.json"

  # settle: a run that starts on a hot core measures the previous run's heat.
  # The middle 12 s of that settle is also the cheapest honest idle baseline this node can
  # record -- same board temperature, same fan duty, same minute as the run it precedes,
  # which a once-per-session baseline is not. First 5 s skipped: the board is still
  # shedding the previous run's heat and the page cache is still writing back. Last 3 s
  # skipped so the marker is not adjacent to the next run's first instruction.
  sleep 5
  mark "idle_baseline_begin" "settle run=$RUN_ID before=$tag"
  local ti0; ti0=$(date -u +%s.%N)
  sleep 12
  local ti1; ti1=$(date -u +%s.%N)
  mark "idle_baseline_end" "settle run=$RUN_ID before=$tag"
  sleep 3

  local temp0 thr0; temp0=$(soc_temp_c); thr0=$(throttled_word)
  mark "bench_begin" "run=$RUN_ID model=$model phase=$phase n=$val threads=$threads"
  local t0; t0=$(date -u +%s.%N)

  # --progress makes llama-bench announce the warmup and each repetition on stderr;
  # stamp_lines.py turns that into a wall-clock trace. This is what lets the energy
  # integral be taken over the REPETITIONS ONLY. Without it the window also contains
  # process start, ggml_backend_load_all, the GGUF page-in off the microSD and a
  # FULL-LENGTH warmup prefill (upstream warms pp with test_prompt(n_prompt), not 32
  # tokens) -- none of which contribute a token to the denominator.
  #
  # `2>&1 >"$json"` : stderr is dup'd to the PIPE first, THEN stdout is sent to the file.
  # The order is NOT interchangeable -- `>"$json" 2>&1` would put both streams in the
  # JSON file and corrupt it. A plain pipeline is used rather than process substitution
  # because bash waits for both ends of a pipeline, so there is no race between the
  # stamper flushing and t1 being taken.
  set +e
  "$LLAMA_BIN/llama-bench" -m "$path" "$flag" "$val" \
      $( [ "$phase" = "pp" ] && echo "-n 0" || echo "-p 0" ) \
      -t "$threads" -r "$REPS" --progress -o json \
      2>&1 >"$json" | python3 "$HS_ROOT/node/stamp_lines.py" > "$OUT/${tag}.stderr.tsv"
  local rc=${PIPESTATUS[0]}   # MUST be one line: a bare `local rc` resets PIPESTATUS
  set -e

  local t1; t1=$(date -u +%s.%N)
  mark "bench_end" "run=$RUN_ID model=$model phase=$phase n=$val threads=$threads"
  [ "$rc" -eq 0 ] || warn "llama-bench exited $rc for $tag -- the row will be flagged, sweep continues"

  # One window record per run. t_start/t_end keep their existing meaning (the RAW process
  # window) forever; bench_windows.py adds the repetition sub-windows on top, and
  # bench_parse.py decides which of them the energy join actually uses.
  python3 "$HS_ROOT/node/bench_windows.py" \
      --run "$RUN_ID" --node "$NODE_ID" --model "$model" --phase "$phase" \
      --n "$val" --threads "$threads" --reps "$REPS" \
      --t-start "$t0" --t-end "$t1" --rc "$rc" \
      --json "$(basename "$json")" --stderr-tsv "${tag}.stderr.tsv" \
      --bench-json "$json" --stamped "$OUT/${tag}.stderr.tsv" \
      --idle-start "$ti0" --idle-end "$ti1" \
      --temp-start-c "$temp0" --throttled-start "$thr0" \
      --temp-end-c "$(soc_temp_c)" --throttled-end "$(throttled_word)" \
      >> "$OUT/windows.jsonl"

  local th; th=$(throttled_word)
  [ "$th" = "0x0" ] || warn "THROTTLED during $tag (throttled=$th) -- this row is suspect, flag it"
  log "  done $tag  (temp $(soc_temp_c) C)"
}

for m in "${MODELS[@]}"; do
  for t in "${THREADS_SWEEP[@]}"; do
    if [ "$VALIDATE" -eq 1 ]; then
      for r in "${REPS_SET[@]}"; do
        REPS="$r"
        for p in "${PP[@]}"; do run_one "$m" pp "-p" "$p" "$t"; done
        for n in "${TG[@]}"; do run_one "$m" tg "-n" "$n" "$t"; done
      done
    else
      for p in "${PP[@]}"; do run_one "$m" pp "-p" "$p" "$t"; done
      for n in "${TG[@]}"; do run_one "$m" tg "-n" "$n" "$t"; done
    fi
  done
done

if [ "$VALIDATE" -eq 1 ]; then
  # Deliberately NOT parsed into results.csv: the _r<REPS> filename field breaks
  # bench_parse.py's model-name recovery. The fit is done on the control node once the
  # energy join has filled in joules, by node/bench_validate.py.
  log "validate sweep complete: $OUT"
  log "  windows.jsonl holds R = ${REPS_SET[*]} for each cell."
  log "  After the energy join, fit with:"
  log "    python3 node/bench_validate.py $OUT --energy <energy.json>"
else
  log "normalising to CSV"
  python3 "$HS_ROOT/node/bench_parse.py" "$OUT" --out "$OUT/results.csv"
  log "sweep complete: $OUT/results.csv"
  log "pull it to the control node:  rsync -a ${NODE_ID}:$DATA_DIR/bench/ ./data/bench/"
fi
