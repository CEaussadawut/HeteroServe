# Cross-node schema emitter — adversarial verification

Four lenses attacked `bench/emit_results_csv.py` and its hook in
`bench/run_matrix.py`. **All four returned `broken`**, and one of them proved it
by running the real control node (`ThinkPad/control/`) against the output.

## The two CRITICALs, both since FIXED

**1. `throttle-unknown` deleted the node from the table it was written to join.**
`ThinkPad/control/analysis.py:224` defines
`THROUGHPUT_INVALIDATING = ("throttled", "throttle-unknown")`, so flagging every
row with that token meant `profiles.per_layer_times` returned
`{'_dropped_flagged_rows': 2}` with no node entry at all, and
`characterisation_table` gave `tokens_per_s=None, throughput_runs=0` on every
row. The projection looked perfect and delivered zero rows. The emitter's own
docstring had called this "a decision for the cross-node table's owner" — the
decision was already in the tree, in writing, naming that exact token.

*Fixed* by stating the condition in the column instead: `throttled_end="unavailable"`,
which is Pi 4's and Pi 5's own spelling for an unread word, is read by no
control-node module, and is not matched by `THROUGHPUT_INVALIDATING`. Verified by
re-running the consumer: `{'jetson': {'prefill': 0.0349, ...},
'_dropped_flagged_rows': 0}`.

**2. Every failed and every skipped run vanished.** `row_to_shared()` keyed on
`row_phase`, which `run_matrix.main()` never set on those records — they carry
`phase` only. Both returned `None` and the omission was reported with the same
wording as the deliberate `combined` omission, so data loss was indistinguishable
from a design choice. *Fixed* on both sides: the driver now sets `row_phase`, and
the emitter falls back to `phase`.

## Also fixed

- `node_id` was `jetson-nano-01`; every consumer keys on cluster.json's `jetson`
  with no alias resolution, so the node sat in a group of its own. Now carried as
  `CLUSTER_NODE_ID` in `config/node.env` and emitted from there, with a
  `node-id-unmapped` flag if absent.
- The driver wrote only `<base>_results.csv`; `analysis.py:135` opens
  `run_dir / "results.csv"` by exact name. Both are written now.
- `n_threads` / `n_gpu_layers` came from the matrix REQUEST; both other nodes take
  them from the bench row. Now bench-first with a request fallback.
- Docstring reason #2 was inverted: a 5 Hz trace produces `gap:`/`late-start:`
  flags, which are NOT throughput-invalidating. Corrected.
- A projection failure now reaches the archived run log, not just stdout.
- Two unit tests passed only because their fixture injected `row_phase` where the
  driver never wrote it. The fixture now mirrors the driver's real record shape.

## Still open

See the entries below. The notable one is the `model` naming question, which is
genuinely cross-node and not this folder's to settle alone.

---

## Lens: `data-honesty` — broken

Column-by-column, most of the projection is honest and several of the suspicions in the brief are refuted by the code (model_size_mb, rep_busy_s, temp_end_c, n_params/n_batch/model_type and t_proc_* all survive attack). But the file's central claim — "what the cross-node table actually consumes for the placement rule is TIMING ... and those this node can supply honestly" — is false as shipped, and I proved it end to end by running the control node's own code against a results.csv the driver emitted.

(1) RUN: ALWAYS_FLAGGED = ["throttle-unknown"] on every row collides with the consumer's published vocabulary. ThinkPad/control/analysis.py:226 sets THROUGHPUT_INVALIDATING = ("throttled", "throttle-unknown") and states in comments that throttle-unknown "counts because the node-side parser already resolves an unreadable throttle word as throttled". So every Jetson row is dropped from characterisation_table's tokens_per_s AND from profiles.per_layer_times (the placement rule). On a real 4-row emit: per_layer_times returned {'_dropped_flagged_rows': 4} — no node entry at all; characterisation_table returned tokens_per_s=null, throughput_runs=0 for every cell; and report.characterisation printed verbatim "2 row(s) were throttled on every run ... That is a result about the board, not a missing measurement - cool it or lower the clock ceiling and re-run." The board was never observed throttling; it has no throttle sensor. The emitter converts "no sensor" into a printed empirical claim of thermal throttling in the thesis's own characterisation report. That is the peer session's "silently erased the node", except worse: it is erased under a false cause.

(2) RUN: every failed and every skipped row is silently dropped, not flagged. The driver never sets row_phase on those records (it sets `phase`), so PHASE_OF.get("") returns None. A model-missing sweep and an OOM sweep each produced a header-only results.csv with the message "1 omitted: no pp/tg phase" — which asserts a phase reason that is not the real one. The bench-exit:/model-missing branches in row_to_shared are unreachable from the driver.

(3) RUN: n_gpu_layers emits the REQUESTED value while llama-bench reported a different one. With a ref that ignores -ngl, the row says n_gpu_layers=99 when bench_n_gpu_layers=0. Pi 4 emits llama-bench's reported value. When counts_mismatch also fires, the config mismatch disappears from flags entirely.

(4) READ: load_s is not Pi 4's load_s.

Neither the 56-case regression nor the 17 unittests can see any of this; two of the 17 assert the opposite of (2) by injecting a field the driver never writes.

### Findings (7)

**[CRITICAL] `throttle-unknown` on every row is not a neutral "not measured" marker — in this cluster's vocabulary it means "resolve as throttled". It deletes 100% of this node's throughput from the cross-node table and makes the generated report assert the Jetson thermally throttled on every run.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/emit_results_csv.py:134`

*Reproduction:*
```
RAN. Copied the cluster to a tmpdir, ran the driver on bench/fixtures/matrix_split.json with the fake tegrastats/llama-bench (4 pp/tg rows emitted, all flags=='throttle-unknown'), then ran the control node's own code against that file:

  >>> import analysis; analysis.invalidates_throughput('throttle-unknown')
  True
  >>> analysis.characterisation_table('..._results.csv')
  {... "phase": "pp", "tokens_per_s": null, "tokens_per_s_sd": null, "joules_per_token": null, "runs": 2, "energy_runs": 0, "throughput_runs": 0, "flagged": 2}
  {... "phase": "tg", "tokens_per_s": null, ... "throughput_runs": 0, "flagged": 2}
  >>> from control import profiles; profiles.per_layer_times(f, {'qwen2.5-1.5b-instruct-q4_k_m': 28})
  {'_dropped_flagged_rows': 4}          # no node key at all
  >>> from control import report; print(report.characterisation(f))
  | jetson-nano-01 | ... | pp | 4 |  |  |  |  |  | 2 | 0 | 0 | 2 |
  | jetson-nano-01 | ... | tg | 4 |  |  |  |  |  | 2 | 0 | 0 | 2 |
  > 2 row(s) were throttled on every run, so they have no throughput figure:
  > jetson-nano-01/qwen2.5-1.5b-instruct-q4_k_m/pp, .../tg. That is a result about
  > the board, not a missing measurement - cool it or lower the clock ceiling and re-run.

The gate is ThinkPad/control/analysis.py:226-231 (THROUGHPUT_INVALIDATING + startswith match), consumed at analysis.py:391 and profiles.py:114. profiles.build_devices(strict=True) then raises ProfileError "unmeasured inputs, so no placement can be computed" for this node.
```

*Why it matters:* The file exists so the Jetson can finally be read by the cross-node table, and its docstring says the timing "this node can supply honestly" is what the placement rule consumes. Measured against the actual consumer, the file supplies exactly nothing: tok/s blank in every Jetson cell, zero rows reaching Rule A. That would be merely a null result. What makes it a data-honesty failure is the caption: the report prints that the board was throttled on every run and tells the reader to cool it and re-run. That is an empirical claim about the hardware that no measurement supports — this board has no vcgencmd word to have read. The docstring's own hedge ("unless the consumer chooses to accept the token -- which is a decision for the cross-node table's owner") is stale: the owner's decision is already written down in analysis.py and it is 'treat as throttled'. An examiner reading the characterisation report would take away a thermal finding that was manufactured by a flag token.

*Fix:* Stop spending the shared token. `throttled_end` is a value column and is the right place to say the absence: write a literal there (e.g. `no-sensor`) instead of "", which answers the fail-open hazard the docstring correctly identifies without asserting throttling. If a flag is still wanted, use a node-local token that is not a THROUGHPUT_INVALIDATING prefix — the file already has that idiom (`jetson-per-token:<status>`), so `jetson-throttle:no-sensor` reads the same to a human and does not trigger the startswith match. Excluding this node from the placement rule must be an explicit edit on the control node by that table's owner, not a side effect of token choice here. Then add the guard that would have caught this: import ThinkPad/control/analysis.py LIVE (the same pattern tests/test_shared_schema.py already uses for Pi4) and assert `not analysis.invalidates_throughput(row['flags'])` for a normal ok row. Nothing in the current 56+17 can see it.

**[HIGH] Every failed and every skipped row is silently dropped from the shared CSV. row_to_shared reads `row_phase`, which the driver never sets on a failed or skipped record — it sets `phase` — so PHASE_OF.get("") returns None and the row vanishes. The bench-exit:/model-missing flag branches are unreachable from the driver.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/emit_results_csv.py:170`

*Reproduction:*
```
RAN. Two driver runs with --no-power against a one-case matrix:
 (a) empty MODEL_DIR -> "[1/1] SKIP sp|... (model missing)" then
     "wrote ..._results.csv  (0 shared-schema rows, 1 omitted: no pp/tg phase)"
     and `cat` of the file shows the 40-column header and nothing else.
 (b) llama-bench stub that prints "CUDA error: out of memory" and exits 1 ->
     "FAILED rc=1" then "(0 shared-schema rows, 1 omitted: no pp/tg phase)", header only.
The standalone CLI names the cause: `emit_results_csv.py ..._bench_energy.csv` prints
  omitted 1 row(s) with no shared-schema phase: sp/None
and inspecting the joined CSV confirms the column is absent, not 'combined':
  fix2 row_phase=None phase='prefill' status=skipped rc=''
  fix3 row_phase=None phase='prefill' status=failed  rc='1'
run_matrix.py:686-698 (skipped) and 743-750 (failed) build those records without row_phase; only bench_records() (run_matrix.py:536) sets it.
```

*Why it matters:* An OOM on the 3B rung is a headline result for a 4 GB Jetson Nano, and Pi 4 does publish it — bench_parse.py:238 raises `bench-exit:%s` on a nonzero rc and keeps the row. Here it disappears, so in the pooled table 'the Jetson OOM'd on this model' is indistinguishable from 'this model was never in the Jetson's matrix'. run_matrix.py:679 states that exact rule for the node's own CSV ("A run that silently vanishes from it is indistinguishable from a run that was never in the matrix") and the emitter reintroduces it for the artifact the examiner actually reads. The docstring at line 165 claims the opposite ("everything else comes through with `flags` saying what is wrong with it, because a row that vanishes cannot be audited"), and the printed reason — 'no pp/tg phase' — asserts a phase problem when the real cause is a field the driver does not write. A sweep where every model path is wrong produces a header-only file and exit 0.

*Fix:* One line: `phase = PHASE_OF.get((rec.get("row_phase") or rec.get("phase") or "").strip())`. `phase` is set on every record path (run_matrix.py:687, 744, 771) and holds prefill/decode/combined, so combined still omits correctly and failed/skipped rows come through carrying bench-exit:/model-missing plus no-throughput, which is what the docstring already promises. Then change the driver's and CLI's omission message to name the omitted row's actual phase value so 'None' can never again read as 'combined'.

**[HIGH] `n_gpu_layers` emits what the driver ASKED for, not what llama-bench reported it ran. The joined record carries both (`n_gpu_layers` = the matrix request, `bench_n_gpu_layers` = the reported value) and the emitter picks the request. Pi 4 emits the reported value.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/emit_results_csv.py:243`

*Reproduction:*
```
RAN. Driver on a matrix asking n_gpu_layers=99 with HS_FAKE_IGNORE_NGL=1 (the fixture's CPU-only-build simulation):
  requested ngl=99 | emitted n_gpu_layers=99  n_prompt=512  flags=throttle-unknown;jetson-per-token:refused_config_mismatch;temp-unknown
  joined:  bench_n_gpu_layers=0  config_mismatch=1  per_token_status=refused_config_mismatch
And with HS_FAKE_IGNORE_NGL=1 plus HS_FAKE_WRONG_COUNTS=1 the evidence disappears entirely, because per_token_status() returns only its first reason (run_matrix.py:474 counts check precedes the 481 config check):
  requested ngl=99 | emitted n_gpu_layers=99  n_prompt=128  flags=throttle-unknown;jetson-per-token:refused_counts_mismatch;temp-unknown
  joined:  bench_n_gpu_layers=0  config_mismatch=1  counts_mismatch=1  per_token_status=refused_counts_mismatch
Pi 4 for comparison: bench_parse.py:290 `"n_gpu_layers": e.get("n_gpu_layers", 0)` — llama-bench's own field.
```

*Why it matters:* n_gpu_layers is the placement axis this whole node exists to characterise: the GPU-offload result IS the comparison between ngl 0 and ngl 99. A row that says 99 when the work demonstrably ran on the CPU is the single most damaging false assertion this schema can carry, and in the second case there is nothing anywhere in the row to contradict it. The driver already detects the condition and run_matrix.py:479-484 explains precisely why it matters; the emitter then discards the detection and republishes the request as if it were a measurement. It also makes the two nodes assert different things under one column name, which is exactly the divergence the COLUMNS drift test exists to prevent — and the drift test cannot see it, because it compares column names, not what is put in them.

*Fix:* Emit the reported value when the ref gave one, falling back to the request only when it did not: `_int(rec.get("bench_n_gpu_layers"), _int(rec.get("n_gpu_layers")))` — matching Pi 4. Independently, carry config_mismatch into flags on its own rather than only via per_token_status (which reports one reason and can hide it): `if _int(rec.get("config_mismatch"), 0): flags.append("config-mismatch:ngl")`. Add a Projection test for a record with n_gpu_layers=99/bench_n_gpu_layers=0.

**[MEDIUM] `load_s` is a different quantity from Pi 4's `load_s` under the same column name: Pi 4 measures the model-load phase from llama-bench's own --progress stamps; this node computes process_span minus the measured compute slice, which adds teardown and rests on an ESTIMATED warm-up. `warmup_s` is left empty even though the estimate exists.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/emit_results_csv.py:281`

*Reproduction:*
```
READ + RAN. Pi4/node/bench_parse.py:308 `"load_s": _delta(win.get("t_load_start"), win.get("t_load_end"))` with Pi4/node/bench_windows.py:201-204 defining t_load_start = t_proc_start and t_load_end = t_warmup_start — strictly process-start to warm-up-start, teardown excluded, warm-up reported separately as warmup_s. Jetson: run_matrix.py:844 `rec["load_s"] = round(dur - ce["duration_s"], 3)` where dur is the WHOLE-PROCESS span [t_start_unix, t_end_unix] and ce is the trailing compute slice anchored on the marker's last sample. run_matrix.py:1010-1014 states that teardown, the CSV write and interpreter shutdown all sit between the marker anchor and process exit — so they are charged to load_s. The compute slice itself includes an ESTIMATED warm-up (row_compute_seconds, run_matrix.py:327-357: warm_ns = ns for pp, ns/n_g for tg, and an explicit upper bound for a fused row), so load_s inherits that estimate's error with no flag. In my run (fixture HS_FAKE_LOAD_S=3.0, near-zero teardown) load_s came out 3.025 / 3.022 — right here only because the fake exits instantly; on hardware the CUDA context teardown is not instant.
```

*Why it matters:* Pooled in one table, a Pi 4 row's load_s and a Jetson row's load_s are read as the same measurement and differenced across nodes — and load time off storage is a real characterisation result for a microSD-booted board, so somebody will difference them. One is a measured sub-window, the other is an arithmetic residue that silently absorbs teardown and an estimate. Leaving warmup_s blank compounds it: a reader reconstructing the process budget as load_s + warmup_s + rep_busy_s gets an unbalanced sum on Jetson rows and a balanced one on Pi 4 rows, with nothing in the row explaining why.

*Fix:* Either leave load_s empty (it is a derived residue, and the node's own CSV keeps load_s/load_joules with their semantics documented), or keep it and fill warmup_s from the same estimate the slice used so the two columns are at least self-consistent, plus add a flag token naming it as derived-by-subtraction (e.g. `load-s:derived`). Do not leave load_s populated and warmup_s blank — that is the combination that reads as a measurement.

**[MEDIUM] Two of the 17 unittests assert the opposite of the real behaviour by injecting `row_phase` into records where the driver never writes it, so the suite is green precisely on the code path that is broken.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/tests/test_shared_schema.py:207`

*Reproduction:*
```
READ + RAN. Projection.rec() (test_shared_schema.py:124-139) hard-codes "row_phase": "prefill" in the base dict, so test_a_failed_run_is_flagged_and_keeps_its_row (line 207) and the skipped half of test_model_is_the_short_name_on_every_row_shape (line 179) both feed row_to_shared a record shape the driver cannot produce for status=failed/skipped (proved above: `fix3 row_phase=None ... status=failed rc='1'`). Both tests pass; the production behaviour is the row being dropped. The 56-case regression is equally blind: bench/fixtures/run_regression.sh:707 asserts `all(r["phase"] in ("pp","tg") for r in rows)`, which is trivially satisfied once the non-conforming rows have been deleted, and it never runs a failing or model-missing case through the emitter.
```

*Why it matters:* The test named test_a_failed_run_is_flagged_and_keeps_its_row is the only written statement of the intended contract for failure rows, and it certifies a contract the system does not honour. A reviewer reading the suite concludes failures are preserved and audited. This is the same class of problem the file's own header warns about for Pi 4 drift ("stale documentation about a load-bearing column is how the last schema divergence survived so long"), applied to the guard itself.

*Fix:* Delete "row_phase" from Projection.rec()'s base dict and set it explicitly per test, so a record can only carry it when the test means it to. Add tests that build the failed and skipped records the way run_matrix.py:687 and :744 build them (phase set, row_phase absent) and assert the row IS emitted with bench-exit:/model-missing. Add one end-to-end case to run_regression.sh driving a nonzero-exit llama-bench and asserting the row survives into _results.csv.

**[LOW] The auto-emitted file is named `<base>_results.csv`, but ThinkPad/control/analysis.py's join_run() requires a file named exactly `results.csv` in the run directory, and the standalone CLI's default `--out` writes that second, differently-named file into the same directory.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:957`

*Reproduction:*
```
READ. run_matrix.py:957 `shared_csv = os.path.join(out_dir, base + "_results.csv")`. ThinkPad/control/analysis.py:135-137: `results = run_dir / "results.csv"` ... `raise FileNotFoundError("no results.csv in %s" % run_dir)`. The emitter's own CLI default (emit_results_csv.py:313) writes `results.csv` beside the input — so running it by hand in the results dir leaves two files with the same content under different names, and only the hand-made one is joinable.
```

*Why it matters:* Minor next to the rest, but the stated purpose of the hook is that the step happens without anyone remembering to run it, and the file it produces automatically is the one the control node cannot find. Worse, an operator who then runs the CLI by hand creates a second copy whose run_id is derived independently, so a stale results.csv can sit next to a fresh <base>_results.csv with no way to tell which is current.

*Fix:* Pick one name. Either emit `results.csv` in out_dir (matching join_run and the CLI default) or document in the README that the control-node collector must rename/symlink. If both are kept, at minimum have the CLI refuse to write results.csv when a <base>_results.csv with a different run_id already sits in the same directory.

**[LOW] When the ref omits llama-bench's count columns, n_prompt/n_gen are back-filled from the MATRIX REQUEST and emitted as though measured; the only trace is an indirect `jetson-per-token:refused_counts_unverified` flag.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/emit_results_csv.py:179`

*Reproduction:*
```
RAN. Driver with HS_FAKE_NO_COUNTS=1 (fixture ref that drops n_prompt/n_gen):
  NO_COUNTS ref -> n_prompt=512 n_gen=0 phase=pp flags=throttle-unknown;jetson-per-token:refused_counts_unverified;temp-unknown
The joined record's counts_source is "matrix" (run_matrix.py:539), i.e. the numbers are the request echoed back. ThinkPad/control/analysis.py:126-127 then computes tokens as n * samples from exactly those cells.
```

*Why it matters:* n_prompt is a denominator on the control node. A cell that says 512 because the driver asked for 512, with no reported count behind it, is not the same assertion as a cell that says 512 because llama-bench reported 512 — and the row does not distinguish them except through a flag whose name is about per-token ENERGY status, not about counts. Low because the flag does travel and because a ref that drops those columns is unlikely on the pinned build; but the emitted cell reads as a measurement either way.

*Fix:* Raise the condition in its own right rather than only through per_token_status: `if (rec.get("counts_source") or "") != "bench": flags.append("counts-unverified")`. The value can stay (it is the best available) as long as the row says the count was not reported.

### Attacks that FAILED
- model_size_mb 2^20-vs-10^6 or double conversion — REFUTED BY READING. llama-bench's `model_size` is a byte count (Pi5/node/gguf_meta.py:14-17 cites test::get_fields() in tools/llama-bench/llama-bench.cpp). Jetson emit_results_csv.py:235 divides by 1024**2; Pi4/node/bench_parse.py:283 `round(float(e.get("model_size", 0) or 0) / (1024 ** 2), 1)`; Pi5/bench/parse_bench.py:419 `round(_num(e.get("model_size"), 0.0) / (1024 ** 2), 1)`. All three are identical, all three are MiB, and Pi5's own findings doc (.handoff/2026-09-13-1238, section 'model_size_mb') already litigated the unit. Observed 378.6 for a 397,000,000-byte model — correct MiB. No double conversion: the joined CSV carries `bench_model_size` raw off the bench row (run_matrix.py:553), untouched.
- n_params / n_batch / model_type invented by the fixture — REFUTED. These map to llama-bench's real `model_n_params`, `n_batch`, `model_type` (run_matrix.py:552-555), the same fields Pi4/node/bench_parse.py:284/289/282 and Pi5/bench/parse_bench.py:420/425 read from real hardware output, and the same names ThinkPad/.handoff/2026-09-13-0053 lists off llama-bench's field list. The fixture was edited to match upstream, not to invent. The driver will not carry empties on real hardware for these three.
- rep_busy_s is a different quantity from Pi 4's — REFUTED BY READING. Pi4 computes sum(samples_ns)/1e9; llama-bench's avg_ns is the arithmetic mean of samples_ns, so avg_ns * repetitions == sum(samples_ns) to within integer-nanosecond truncation (< reps ns). Neither includes the warm-up: samples_ns holds only the timed repetitions, and the Jetson's warm-up term lives in row_compute_seconds/compute_s, which never touches rep_busy_s. The columns are comparable. (Residual nit, not a finding: Jetson's `samples` is the matrix's -r while Pi 4's is len(samples_ns), the count actually performed — they differ only if llama-bench runs fewer repetitions than asked.)
- temp_end_c comes from the wrong window, or is still the max — REFUTED BY READING AND RUNNING. run_matrix.py:1134 takes temps[-1] of the window passed to energy_between, and the call that lands on the record is run_matrix.py:814 `e = energy_between(power_csv, rec["t_start_unix"], rec["t_end_unix"])` followed by `rec.update(e)` — so it IS the whole-process window, NOT the compute slice (the compute-slice call at :832 has only selected keys pulled out of it, never temp). That is the right window: Pi 4's temp_end_c is `soc_temp_c()` sampled immediately after the bench_end marker (Pi4/scripts/06_bench_local.sh:150), i.e. the same instant. Measured on a real emit, the last sample in the window sat 0.140 s / 0.164 s before process exit (one 5 Hz period), and temp_end_c matched it. The temp_max_c -> temp_end_c change is correct and temp_max_c correctly stays in the node's own CSV only.
- t_proc_start / t_proc_end can be mistaken for an energy window downstream — REFUTED. ThinkPad/control/analysis.py:146 reads only t_start/t_end, and when they are blank it skips the row and stamps its own `no-window` flag (:149-151). Nothing in ThinkPad/control/*.py reads t_proc_start, t_proc_end, rep_busy_s, load_s, warmup_s, energy_duty or temp_end_c at all (grepped). energy_window="" further distinguishes a Jetson row from Pi 4's "process"/"reps-span". The risk here is a human reader, not code — and even there the blank energy_window is the tell.
- The COLUMNS / QUANT_HINTS / flag-vocabulary drift guard is decorative — REFUTED. I traced every `flags.append("...")` literal in Pi4/node/bench_parse.py through the test's two-stage normaliser (test_shared_schema.py:86-101): `energy-span:whole-process:`, `low-duty:%.3f`, `bench-exit:%s`, `throttled-at-start:%s`, `throttled:%s`, `temp-unknown`, `no-throughput`, `no-samples:reps-unknown`, `throttle-unknown`, `no-window:energy-join-impossible` all normalise to tokens that are either raised here or present in PI4_FLAGS_NOT_RAISED_HERE, and the reverse (stale-entry) test closes the other direction. The COLUMNS and QUANT_HINTS extractions match Pi 4 byte-for-byte. This part of the work is sound — it just guards names, not the meanings the values carry, which is where findings 1 and 3 live.
- Omitting the `combined` (-pg) row mislabels or loses the fused case — REFUTED. The shared schema genuinely has no phase value meaning 'both'; the row is omitted, reported on stdout (run_matrix.py:970 and emit_results_csv.py:336), and its joules_per_request survives in the node's own <base>_bench_energy.csv with per_token_status=refused_combined_request. Observed: 5 driver rows -> 4 shared rows, '1 omitted: no pp/tg phase'. This is the honest answer.
- The emitted row could break the control node's writer by missing a column, or the header could drift — REFUTED. row_to_shared returns a fixed dict literal whose keys are exactly COLUMNS (verified by writing and re-reading the header), the energy columns are present-but-empty as the docstring says, and the auto-emit path uses shared.COLUMNS as fieldnames (run_matrix.py:967).
- The baseline claims '56 passed, 0 failed' and '17 tests' are inflated — REFUTED. Reproduced verbatim in a fresh copy of the cluster: `./bench/fixtures/run_regression.sh` -> '56 passed, 0 failed'; `python3 -m unittest discover -s tests` -> 'Ran 17 tests ... OK', with the four live Pi 4 drift tests actually running rather than skipping. Both numbers are real. They are also why every finding above needed to be produced by running the consumer's code rather than by reading the suite.

---

## Lens: `consumer-semantics` — broken

The file's most consequential choice is inverted with respect to the actual consumer, and I proved it by running the consumer. `ThinkPad/control/analysis.py` does NOT treat "a non-empty flags" as an opt-out from the placement rule — the emitter's docstring (lines 38-40) and the test file (line 82) both assert that, and both are false. There are exactly two rules, and they are opposites: (a) ANY flag excludes a row from the ENERGY means only (`characterisation_table`, `unflagged = [r for r in rows if not r.get("flags")]`, analysis.py:422); (b) only the THROTTLE flags exclude a row from throughput AND from Rule A — `THROUGHPUT_INVALIDATING = ("throttled", "throttle-unknown")` (analysis.py:224), used by `characterisation_table` (:391) and by `profiles.per_layer_times` (:114). The design deliberately avoided the harmless flag class (`gap:`/`late-start:` from a 5 Hz trace, which cost nothing — I ran a Pi-4 row carrying `no-window` and it kept its tokens_per_s and its c_i) and then raised the one fatal token unconditionally on every row. Result, executed end to end: the Jetson appears in the characterisation table with tok/s, sd, J/tok, W and Wh/1k all blank and `T-runs=0`; `per_layer_times` reports `_dropped_flagged_rows: 2` and yields nothing for the node; and `build_devices(strict=True)` does not merely skip it — it raises `ProfileError: unmeasured inputs, so no placement can be computed` naming `jetson: ms_per_layer_decode, ms_per_layer_prefill`, blocking placement for the whole cluster. Two further independent blockers survive even if the flag is fixed: the row's `node_id` is `jetson-nano-01` (from NODE_ID in config/node.env) while every consumer keys on cluster.json's `jetson`, and the row's `model` is `qwen2.5-1.5b-instruct-q4_k_m` while Pi 4 and Pi 5 both spell that same GGUF `qwen2.5-1.5b-q4km` — so the `--layers` map misses it and `per_layer_times` prints "no layer count was supplied" and discards the rows. The energy decision, by contrast, survives every attack I could mount: nothing downstream requires or divides by the energy columns, and `samples` really is the -r repetition count. Green 56/0 and 17/17 are both true and both hollow — the regression at run_regression.sh:710 asserts `all("throttle-unknown" in r["flags"])`, i.e. the suite enforces the defect, and no test in either suite imports the consumer.

### Findings (9)

**[CRITICAL] `ALWAYS_FLAGGED = ["throttle-unknown"]` is the exact literal the consumer treats as "this run was throttled", so every Jetson row is excluded from the throughput means and from Rule A. Being honest about the throttle state does not merely risk exclusion — it guarantees it.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/emit_results_csv.py:134`

*Reproduction:*
```
RAN. Copied the cluster to a tmpdir, emitted two rows with row_to_shared(), concatenated a clean pi4 row, ran ThinkPad/control/analysis.join_run then characterisation_table then profiles.per_layer_times.
  analysis.invalidates_throughput('throttle-unknown') -> True
  characterisation_table: {"node":"jetson-nano-01",...,"tokens_per_s":null,"tokens_per_s_sd":null,"sd_basis":null,"joules_per_token":null,"mean_power_w":null,"wh_per_1k_tokens":null,"runs":1,"energy_runs":0,"throughput_runs":0,"flagged":1}   (x2, pp and tg)
  same run, pi4 row carrying flags='no-window': tokens_per_s=4.0, throughput_runs=1  <- the ONLY difference is the throttle token
  per_layer_times(..., threads=4) -> {'pi4': {...'decode': 8.9286...}, '_dropped_flagged_rows': 2}   (both Jetson rows dropped)
Counterfactual in the same script: strip only 'throttle-unknown' from the flags and re-run -> Jetson tokens_per_s=12.5, throughput_runs=1, per_layer_times decode=2.8571. Nothing else changed.
```

*Why it matters:* This file exists to put the Jetson into the cross-node table and the placement rule. As written it puts the node into the table as a row of blanks and out of the rule entirely — the single line that was supposed to be the node's honesty costs it the deliverable. analysis.py:222 states why the consumer reads the token that way: "the node-side parser already resolves an unreadable throttle word as throttled (Pi4/node/bench_parse.py:102)". Pi 4 raises `throttle-unknown` conditionally, for "I tried to read the word and could not"; this node raises it unconditionally for "this board has no such sensor". Those are different conditions and the consumer cannot tell them apart.

*Fix:* Do not reuse Pi 4's token for a different condition. Either (a) raise a distinct token for "no throttle sensor exists on this SoC" — I verified `throttle-unmeasurable`, `no-throttle-sensor` and `throttle:no-sensor` all return invalidates_throughput()==False, and with `throttle-unmeasurable` the row yields tok/s=12.5, T-runs=1, flagged=1 (still out of the energy means, which is correct since it carries no energy) and Rule-A c_i=2.8571; or (b) better, because a near-miss string is fragile against a future edit to THROUGHPUT_INVALIDATING, get the token added explicitly to the control node's taxonomy by its owner and pin it with a test. Note the hazard the current comment cites does not exist: `throttled_end` is read by NOTHING in ThinkPad/ (only named in tests/test_analysis.py:39 as a header string), so the "empty reads as checked and clean" argument has no consumer behind it; `flags` is the only channel that is read.

**[CRITICAL] The Jetson's absence from `per_layer_times` does not degrade gracefully — `build_devices(strict=True)` raises and blocks placement for the ENTIRE cluster, not just for this node.**

`/mnt/data/KMITL/Final/HeteroServe/ThinkPad/control/profiles.py:276`

*Reproduction:*
```
RAN, against the real ThinkPad/config/cluster.json (jetson: serves_rpc=true, enabled=true, so it IS in cluster.workers() -> ['pi5','pi4','jetson']):
  profiles.build_devices(cluster, results_csv=<joined>, power_summary=..., link_json=..., n_layers_by_model={both spellings}, threads=4, strict=True)
  stderr: note: 2 result row(s) excluded from per-layer times because their flags invalidate throughput (throttled / throttle-unknown)
  RAISED ProfileError:
  unmeasured inputs, so no placement can be computed:
    pi5: ms_per_layer_decode, ms_per_layer_prefill
    pi4: ms_per_layer_prefill
    jetson: ms_per_layer_decode, ms_per_layer_prefill
```

*Why it matters:* `missing` accumulates across all workers and the raise is at the end, outside the loop. So a Jetson results.csv that the consumer cannot use is not a quiet gap in one row of the table — it is a hard stop on the whole placement run. The emitter's framing ("it costs this node the placement rule unless the consumer chooses to accept the token") understates the blast radius: it costs the cluster the placement rule.

*Fix:* Fix the token (finding 1). Separately, the docstring at emit_results_csv.py:132-133 should say that an unaccepted token is a cluster-wide stop, not a per-node opt-out, so whoever reads it understands the stake.

**[HIGH] `node_id` is written as `jetson-nano-01` (NODE_ID from config/node.env) but every consumer keys on cluster.json's node id `jetson`. The node never matches even when its rows survive.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/emit_results_csv.py:226`

*Reproduction:*
```
RAN. With 'throttle-unknown' stripped so the rows survive:
  per_layer_times -> {'jetson-nano-01': {'n_threads': [4], 'decode': 2.8571, 'prefill': 0.0349, ...}, 'pi4': {...}}
  times.get('jetson') -> None
build_devices does `t = times.get(nid)` with nid from cluster.workers(), i.e. 'jetson'. config/node.env line 10 is NODE_ID=jetson-nano-01; Pi4/config/node.env:5 is NODE_ID=pi4 and Pi5/config/node.env:5 is NODE_ID=pi5 — both match their cluster.json ids exactly. Only this node diverges. ThinkPad/tests/test_repo_hygiene.py:111 carries an explicit alias map {"jetson":"jetson","jetson-nano-01":"jetson"} — the mismatch is known there and nowhere in analysis.py or profiles.py.
```

*Why it matters:* This is a second, independent reason the Jetson is invisible to the placement rule, and it is not visible in any output — `times` simply has a key nobody looks up, and build_devices reports 'jetson: ms_per_layer_decode, ms_per_layer_prefill' as though no data existed. The characterisation table also prints 'jetson-nano-01' beside 'pi4' and 'pi5', so the cross-node table reads as four nodes with inconsistent naming.

*Fix:* Either map NODE_ID to the cluster id when writing the shared projection (the shared schema's node_id is a cross-node join key, not this node's private name), or get the alias map that already exists in test_repo_hygiene.py applied in analysis/profiles. The emitter is the right place: it is the projection layer. Add a test asserting the emitted node_id is one of ThinkPad/config/cluster.json's ids, read live, exactly as the COLUMNS drift test reads Pi 4 live.

**[HIGH] The `model` short name does not group with Pi 4's or Pi 5's for the same GGUF, and the docstring's justification ("each node spells it its own way, consistently") is contradicted by the two nodes it cites: Pi 4 and Pi 5 use an identical shared spelling.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/emit_results_csv.py:193`

*Reproduction:*
```
RAN. Same file, same GGUF: Pi4/config/models.txt and Pi5/config/models.txt BOTH say `qwen2.5-1.5b-q4km   Qwen/Qwen2.5-1.5B-Instruct-GGUF   qwen2.5-1.5b-instruct-q4_k_m.gguf`. Jetson/config/models.tsv says `qwen2.5-1.5b-instruct-q4_k_m`.
  profiles.per_layer_times(<joined>, {"qwen2.5-1.5b-q4km": 28}, threads=4)   # the --layers map, written for the shared spelling
  stderr: warning: 2 result row(s) ignored because no layer count was supplied for qwen2.5-1.5b-instruct-q4_k_m. Add them to the --layers map, or their cells contribute nothing to the placement rule.
  -> {'pi4': {...}, '_dropped_flagged_rows': 0}   # Jetson gone again
And report.characterisation prints the two as separate, unrelatable rows: `| jetson-nano-01 | qwen2.5-1.5b-instruct-q4_k_m |` beside `| pi4 | qwen2.5-1.5b-q4km |`.
```

*Why it matters:* `model` is in the group key AND is the lookup key for n_layers_by_model. A divergent spelling is a third independent exclusion, and unlike the others it fails with a warning on stderr that a batch run will scroll past. The stated precedent is simply wrong: Pi 5 adopted Pi 4's model short names along with its columns, so the convention the cluster actually has is one shared spelling, not per-node spellings.

*Fix:* Emit the cluster-shared short name in the projection (keep the node's own name in <base>_bench_energy.csv, which is authoritative and unchanged). A live drift test, in the same style as the COLUMNS one, should assert that every model this node emits appears in Pi4/config/models.txt — skipping if Pi 4 is absent.

**[HIGH] Docstring reason #2 for leaving t_start/t_end empty is inverted. A 5 Hz trace would have produced `gap:`/`late-start:` flags, and those do NOT drop a node from the placement rule — the consumer draws that line explicitly, and profiles.py was changed specifically to stop dropping them.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/emit_results_csv.py:38`

*Reproduction:*
```
RAN. The pi4 row in my combined CSV came out of join_run carrying flags='no-window' — a power-trace flag — and still produced tokens_per_s=4.0, throughput_runs=1, and per_layer_times decode=8.9286. analysis.py:205-224 spells out the taxonomy: the power-trace flags (gap:, late-start:, early-end:, too-few-samples:, uncalibrated-channel, no-power-data, no-window:, energy-span:, low-duty:) "say the energy join is untrustworthy. The tokens_per_s from that same run is perfectly good". profiles.py:102-113 records that collapsing the two questions was a bug that made EVERY row of EVERY node drop, and names the fix.
```

*Why it matters:* The decision the file is proudest of rests on a claim about the consumer that the consumer contradicts. The reasoning chain was: "flags are fatal, so avoid producing gap flags" — and then it raised, unconditionally, the one token that is actually fatal. The conclusion (leave energy empty) happens to be right for reasons 1 and 3, which are sound; reason 2 should be struck rather than left as a rule other people will apply again.

*Fix:* Rewrite the docstring's reason #2 and the test docstring at tests/test_shared_schema.py:82-83 to state the real rule: ANY flag opts a row out of the ENERGY means; only a token matching THROUGHPUT_INVALIDATING (analysis.py:224) opts it out of throughput and Rule A. Reasons 1 and 3 are independently sufficient and should be kept as-is.

**[MEDIUM] The report renders this node's honest "unknown" as a positive, false assertion that the board thermally throttled, with remediation advice that cannot apply to it.**

`/mnt/data/KMITL/Final/HeteroServe/ThinkPad/control/report.py:87`

*Reproduction:*
```
RAN report.characterisation() on the joined CSV. Verbatim output:
  "> 2 row(s) were throttled on every run, so they have no throughput figure: jetson-nano-01/qwen2.5-1.5b-instruct-q4_k_m/pp, jetson-nano-01/qwen2.5-1.5b-instruct-q4_k_m/tg. That is a result about the board, not a missing measurement - cool it or lower the clock ceiling and re-run."
The board was never throttled. It has no throttle sensor. The fixture rows were clean, at 46.2 C.
```

*Why it matters:* This is worse than exclusion: the thesis deliverable now contains a fabricated thermal finding about the Jetson, phrased as a positive result, and tells the reader to cool a board that is not hot. An examiner reading the table would conclude the Nano thermally throttles on every run of every model. The emitter chose to 'say unknown out loud'; the consumer converts that into 'throttled'.

*Fix:* Follows from finding 1 — a distinct token keeps the rows in T-runs and out of this prose entirely. If the control node instead chooses to accept `throttle-unknown` as non-invalidating, report.py's wording also needs to separate "throttled" from "never checked".

**[MEDIUM] The group key (node_id, model, quant, phase, n_threads) omits n_gpu_layers, so this node's partial-offload sweep — its distinctive contribution — collapses into one averaged cell. No other node varies n_gpu_layers, so this is Jetson-specific and currently masked by finding 1.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/emit_results_csv.py:243`

*Reproduction:*
```
RAN. bench/matrix.json case `partial_offload_decode`: model qwen2.5-1.5b-instruct-q4_k_m, n_gpu_layers [0, 8, 16, 24, 99], one n_gen, default threads -> five rows sharing one group key (confirmed: 5 rows -> 1 group key). Fed tok/s 3.1 / 4.6 / 6.2 / 8.0 / 12.5 through characterisation_table with the throttle token stripped:
  {"node":"jetson",...,"tokens_per_s":6.88,"tokens_per_s_sd":3.633,"sd_basis":"across-runs","runs":5,"throughput_runs":5}
  per_layer_times -> {'jetson': {'decode': 5.7604, 'samples': {'tg': 5}}}
True range was 3.1-12.5 tok/s.
```

*Why it matters:* A CPU-only run and a fully-offloaded CUDA run are averaged into one number and the spread is labelled `sd_basis: across-runs`, which a reader will take as run-to-run repeatability rather than the offload sweep's range. Rule A then gets c_i = 5.76 ms/layer — the median, i.e. the ngl=16 point — a configuration the cluster would never actually run. analysis.py:392 and profiles.py:136 both key on n_threads but nothing keys on n_gpu_layers, because Pi 4 and Pi 5 are CPU-only and never exercised it. Fixing finding 1 unmasks this, so the two must be fixed together.

*Fix:* Options, in order of preference: (a) get n_gpu_layers into the consumer's group key — it is already a column in the shared schema and is the operating point this node's whole sweep is about; (b) short of that, emit the offload variant into `model` or refuse to project rows whose n_gpu_layers is not the cluster operating point, the same way per_layer_times refuses a CSV spanning thread counts (profiles.py:142-148) rather than silently pooling. Either way, do not ship rows the consumer will average across an operating point it cannot see.

**[MEDIUM] Both green suites stop at the Pi 4 header. Nothing in the 17 unit tests or the 56 regression checks imports the consumer, and the regression asserts the defect as a requirement.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/fixtures/run_regression.sh:710`

*Reproduction:*
```
RAN both. `python3 -m unittest discover -s tests` -> Ran 17 tests, OK (all 17 live in the single file tests/test_shared_schema.py). `./bench/fixtures/run_regression.sh` -> 56 passed, 0 failed, and it genuinely does not touch config/node.env (md5 348ebc26c5bb48abf26e23ddfe56527d unchanged, verified in its own output). Both claims are true.
But: run_regression.sh:710 is `assert all("throttle-unknown" in r["flags"] for r in rows)` and test_shared_schema.py:194 is `self.assertIn("throttle-unknown", ...)` — the suites require the token that excludes the node. Its fixture env sets NODE_ID=fixture, so the node_id mismatch (finding 3) is structurally unreachable. And the one assertion that would have caught all of this does not exist anywhere; I wrote it and it fails today:
  analysis.invalidates_throughput(row_to_shared(rec)['flags']) -> True   # expected False
```

*Why it matters:* The suite's stated purpose is that a drifting copy is worse than no copy, and it guards the copy faithfully against Pi 4's file. But the schema is not the contract that matters — the consumer's interpretation of it is, and the consumer is a file both suites can reach (ThinkPad/control/analysis.py is in the same tree the drift tests already walk UP into for Pi4/node/bench_parse.py). Green here means "the header matches Pi 4", not "a row this node emits is usable".

*Fix:* Add a consumer round-trip test alongside the existing drift tests, importing ThinkPad/control/analysis.py live and skipping if absent, asserting for a representative emitted row: (1) not analysis.invalidates_throughput(row['flags']); (2) analysis.characterisation_table over a one-row CSV returns a non-None tokens_per_s; (3) profiles.per_layer_times returns an entry keyed by a cluster.json node id. Then delete the two assertions that currently pin the defect.

**[LOW] `--n-layers` and the `per_layer_ms` column are dead weight: no control-node code reads per_layer_ms, and the automatic hook never passes a layer count anyway.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/emit_results_csv.py:258`

*Reproduction:*
```
grep -rn 'per_layer_ms' ThinkPad/ --include=*.py returns exactly one hit: tests/test_analysis.py:39, where it appears only as a header string in a fixture. profiles.per_layer_times computes its own ms/layer from ms_per_token / n_layers_by_model (profiles.py:136). run_matrix.py:962 calls `shared.row_to_shared(rec, base)` with n_layers defaulting to 0, so the hook always leaves the column empty.
```

*Why it matters:* Harmless, but the comment at :256-257 ("Left to the control node unless a layer count is known here") implies the control node uses it. It does not. A reader could spend effort filling a column nothing consumes, or trust it as the Rule A input it is not.

*Fix:* Say in the comment that the control node derives ms/layer itself from ms_per_token and its own --layers map, and that this column is carried for schema fidelity and human readers only.

### Attacks that FAILED
- Tried to break the empty-energy decision at the consumer: nothing downstream requires or divides by energy_j / joules_per_token / mean_power_w. characterisation_table guards every one with a truthiness test (analysis.py:423-424), report.py renders None as an empty cell, pareto_front skips points with a None on either axis, and the only division (band['energy_j'] / tokens, analysis.py:177) lives inside the joined branch a windowless row never reaches. RAN it end to end: no exception, no zero, no silent substitution. The three stated reasons 1 and 3 hold; the decision is correct.
- Tried to prove `samples` is the power-sample count, not -r. It is the repetition count: tokens_in_row (analysis.py:117-127) does `reps = int(row.get('samples') or 1) or 1; return n * reps`. The near-miss is real but harmless — analysis.integrate() also returns a key called 'samples' meaning len(power window) (analysis.py:68), and join_run never writes it back into the row (it writes only energy_j, mean_power_w, covered_s, joules_per_token), so the column keeps its meaning. The emitter is right, and its comment at :252-253 is right.
- Tried to make a windowless Jetson row crash or vanish in join_run. RAN it: `_add_flag(row, 'no-window')`, skipped += 1, `continue` — the row still reaches the output CSV (join_run writes ALL rows, not just joined ones, analysis.py:199) and still reaches characterisation_table. Reported cleanly as {'rows': 3, 'joined': 0, 'skipped': 3}. The row survives into the report; it is throttle-unknown, not the empty window, that empties it.
- Tried to show that declaring all 40 COLUMNS including the empty energy ones is pointless padding. It is load-bearing and the comment at :285-287 is correct: join_run reuses the producer's fieldnames (analysis.py:142, 186-193) and appends only keys it adds, so a missing energy column would make DictWriter raise after the header is out.
- Tried to break the phase mapping. prefill->pp / decode->tg is required in both consumers: characterisation_table groups on the raw string and per_layer_times reads phases['tg'] / phases['pp'] literally (profiles.py:151-152). 'prefill'/'decode' would group with nothing and produce decode=None. Correct as written.
- Tried to argue the omitted `combined` (-pg) row should have been projected. It should not: the schema has no phase value meaning both, per_layer_times reads only 'tg' and 'pp', and a third value would be counted in `runs` and `flagged` while contributing to nothing. Omitting and reporting it is right.
- Tried to break quant_of against Pi 4's. Verbatim copy, guarded by a live drift test, and correct on this node's own short names: qwen2.5-1.5b-instruct-q4_k_m -> Q4_K_M, qwen2.5-0.5b-instruct-q8_0 -> Q8_0, with no false Q8 match from the '0.5b' in the name. `quant` groups correctly with Pi 4's.
- Tried to break the COLUMNS drift test's regex extraction. `^COLUMNS = \[(.*?)\n\]` does read Pi4/node/bench_parse.py live and, on failure to match, the test asserts IsNotNone and fails loudly rather than passing vacuously. The skip-if-Pi4-absent path is documented in the module docstring. The guard does what it claims — it just guards the wrong contract (see finding 8).
- Tried to catch the regression script writing the live config/node.env. It does not: fixture env is created in a mktemp dir and passed with --env-file, and the script md5s the real file before and after (output: 'ok config/node.env unchanged (348ebc26c5bb48abf26e23ddfe56527d)'). The SAFETY constraint is respected by the code under test.
- Tried to find a consumer for `throttled_end`, to see whether writing "" there really is a fail-open hazard. There is none — grep across all of ThinkPad/ finds it only as a header string in tests/test_analysis.py:39. The hazard the emitter cites at :266-267 is real in Pi 4's own convention but has no consumer behind it, which is why the unconditional flag buys nothing.

---

## Lens: `schema-fidelity` — broken

The literal schema copy is exact — COLUMNS and QUANT_HINTS are byte-for-byte identical across Pi4/node/bench_parse.py, Pi5/bench/parse_bench.py and the Jetson's copy (verified programmatically, 40 columns, same order), and the drift tests genuinely fail on add/remove/reorder/quant-edit/new-flag mutations of Pi 4 (all five run). Both claimed counts reproduce in a clean copy: 56 passed/0 failed and 17 unittests OK. But matching the column LIST is not matching the SCHEMA, and two semantic mismatches defeat the file's entire stated purpose. (1) The unconditional `throttle-unknown` flag is in ThinkPad/control/analysis.py's THROUGHPUT_INVALIDATING tuple, so profiles.per_layer_times drops 100% of this node's rows — I ran it and got `{'_dropped_flagged_rows': 2}` and a characterisation table with tokens_per_s=None, throughput_runs=0 on every row. The file exists to deliver TIMING to the cross-node table and delivers zero. The docstring calls this "a decision for the cross-node table's owner"; the decision is already in the tree, in writing, naming this exact token, and it is "exclude". (2) The auto-emit hook silently drops every failed and every skipped run, because run_matrix's `failed`/`skipped` records carry `phase` but never `row_phase`, which is the only key row_to_shared reads — reproduced end to end: a three-case sweep with one missing model printed "1 shared-schema rows, 2 omitted: no pp/tg phase", the second omission being the skipped run, misreported as a phase problem. The `bench-exit:` and `model-missing` branches are dead code on the auto path. Three further divergences (samples semantics, unverified n_threads, a flag-drift guard weaker than the Pi 5 one it claims parity with) are documented below.

### Findings (6)

**[CRITICAL] ALWAYS_FLAGGED = ["throttle-unknown"] is a token the control node already reads as "this row's throughput is not a measurement". Every Jetson row therefore carries it, and every Jetson row is dropped from the placement rule and from the throughput column of the report table. The file's only stated deliverable — TIMING — reaches the cross-node table at exactly zero rows. This is not an open question for the table's owner: ThinkPad/control/analysis.py:216 names `throttle-unknown` in THROUGHPUT_INVALIDATING with a written rationale citing Pi4/node/bench_parse.py:102, and profiles.py:113 calls invalidates_throughput() on every row before the placement rule sees it.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/emit_results_csv.py:134`

*Reproduction:*
```
RUN. Built a two-row joined CSV (one prefill, one decode, per_token_status=ok, temp_end_c=46.2, status=ok — i.e. a perfectly clean sweep), ran `python3 emit_results_csv.py j_bench_energy.csv`, then fed the output to the real control node:

  from control import profiles, analysis
  profiles.per_layer_times(p, {"qwen2.5-1.5b-instruct-q4_k_m": 28}, threads=4)
  -> {'_dropped_flagged_rows': 2}          # every row dropped, no node entry at all

  analysis.characterisation_table(p)
  -> {'node': 'jetson-nano-01', 'phase': 'pp', 'tokens_per_s': None, 'tokens_per_s_sd': None,
      'sd_basis': None, 'runs': 1, 'throughput_runs': 0, 'flagged': 1}   (and the same for 'tg')

profiles.build_devices then reads `t = times.get(nid, {})` (profiles.py:212) so needed["ms_per_layer_decode"] and ["ms_per_layer_prefill"] are both None, and strict=True — the default, and what the harness uses — raises ProfileError for this node (read, not run: building a Cluster object was out of scope).
```

*Why it matters:* The Jetson was added to the shared schema precisely because it "was the last node the cross-node table could not read". After this change it still cannot be read: it now produces a complete-looking 40-column CSV that the placement rule discards in full, which is strictly worse than an unreadable CSV because the failure is invisible — characterisation_table emits Jetson rows with `runs: 1, throughput_runs: 0`, so the node appears in the table with every number blank, and the thesis's placement rule silently decides across three nodes instead of four. The README's "what the cross-node table consumes for the placement rule is timing, and that this node can supply honestly" is false as shipped.

*Fix:* Stop expressing a permanent board property as a per-row validity flag. On Pi 4 and Pi 5 `throttle-unknown` means "this run's throttle state could not be established" — a fault, recoverable, and absent from a normal Pi 5 sweep (Pi5/bench/parse_bench.py:381 raises it only when the word is unavailable or blank). On the Jetson it means "this SoC has no vcgencmd", which is true of every row forever, so the token's meaning is not preserved across the copy. The in-schema place to say it is `throttled_end`, using Pi 5's own spelling: set throttled_end="unavailable" instead of "". I grepped ThinkPad/control/*.py — nothing there ever reads throttled_end (it appears only in tests/test_analysis.py's header list), so that carries the statement without costing the node the rule. Keep the claim visible in the README and in the node's own CSV. If the flag is nevertheless wanted, it must be agreed with the control node first — adding `throttle-unknown` to an exception list in analysis.THROUGHPUT_INVALIDATING, or a `throttle-unmeasurable` token that tuple does not match — otherwise do not ship the auto-emit hook, because it manufactures rows that are guaranteed to be discarded.

**[HIGH] row_to_shared() keys the phase off `row_phase` alone, but run_matrix.main() never sets `row_phase` on a failed or a skipped run — those records carry `phase` instead. Both therefore return None and vanish from the shared CSV, and the `bench-exit:%s` and `model-missing` branches at emit_results_csv.py:199-202 are unreachable on the automatic path. The docstring's "Returns None only for a row that has no honest place in the shared table at all" and "a row that vanishes cannot be audited" are both violated, and the omission is misreported as a phase problem.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/emit_results_csv.py:170`

*Reproduction:*
```
RUN, end to end. Copied the kit, built a fixture env (fake llama-bench + fake tegrastats, one real model file) and a matrix with three cases: one good prefill, one naming a model that is not on disk, one fused -pg. `python3 bench/run_matrix.py --env-file <fix>/node.env --matrix <fix>/m.json --tag demo` printed:

  [2/3] SKIP missing_model|qwen2.5-3b-instruct-q4_k_m|... (model missing)
  wrote ..._demo_results.csv  (1 shared-schema rows, 2 omitted: no pp/tg phase)

The joined CSV has 3 rows; the shared CSV has 1. Inspecting them:
  case=ok_prefill     status=ok       row_phase='prefill'
  case=missing_model  status=skipped  row_phase=''        <- dropped
  case=fused          status=ok       row_phase='combined' <- dropped by design

Also reproduced at function level against the exact dicts run_matrix builds (run_matrix.py:690-696 skipped, :744-751 failed): both have row_phase=None, phase='prefill', and E.row_to_shared(rec, "r") returns None for each. The unit test test_a_failed_run_is_flagged_and_keeps_its_row passes only because its fixture supplies row_phase='prefill', which run_matrix never puts on a failed record.
```

*Why it matters:* Pi 5 keeps a failed leg: I ran its real pi4_rows() on an exit_code=1 cell and got a full row with flags='bench-exit:1;temp-unknown;no-throughput;no-samples:reps-unknown'. Pi 4 does the same from the window's rc. The Jetson drops it. A sweep where the 3B model OOMs on every case ships a shared CSV that is short by exactly those rows, with one console line that calls them "no pp/tg phase" — the same wording used for the deliberate `combined` omission — so a reader cannot distinguish a design choice from data loss, and the cross-node table's denominator (how many cells this node attempted) is wrong with nothing to detect it. This is the asymmetric-exclusion bias the guard test's own docstring warns about, arriving through a different door.

*Fix:* Fall back to the window's phase when the row has none: `phase = PHASE_OF.get((rec.get("row_phase") or rec.get("phase") or "").strip())`. run_matrix already labels every failed/skipped record with phase_of(run), which returns exactly 'prefill'/'decode'/'combined'/'mixed', so this costs nothing and the existing `bench-exit:`/`model-missing` flags then fire as written. Separately, split the omission report: a `combined`/`mixed` row is a design omission, a status=failed/skipped row that cannot be placed is a defect, and they should not print as one number. Add a regression assertion — the fixture sweep at run_regression.sh:686-710 checks header order and phase values but never that a skipped run survives the projection, which is why this passes 56/56.

**[MEDIUM] The flag-drift guard extracts Pi 4's vocabulary with a regex that only matches `flags.append("...` with a double quote. Pi 5's guard — the one this file's docstring cites as precedent ("Pi5/tests/test_pi4_schema.py does the same and has already caught two real drifts") — uses ast.parse/ast.walk, handles the `"x:" + str(...)` BinOp shape, and asserts it extracted more than five tokens. The regex version is strictly weaker, and the claim of parity is not true.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/tests/test_shared_schema.py:86`

*Reproduction:*
```
RUN. Mutated my copy of Pi4/node/bench_parse.py one way at a time and ran `python3 -m unittest discover -s tests` after each:
  add a column            -> FAILED (guard works)
  remove a column         -> FAILED (works)
  reorder two columns     -> FAILED (works)
  add a QUANT_HINTS spelling -> FAILED (works)
  flags.append("absurd-duration")        -> FAILED (works)
  flags.append(f"absurd-duration:{n}")   -> OK      <- MISSED
  flags.append('absurd-duration')        -> OK      <- MISSED
  flags.extend(["absurd-duration"])      -> OK      <- MISSED
  flags += ["absurd-duration"]           -> OK      <- MISSED
Two further shapes fail loudly but name the wrong token, which is a usability defect rather than a hole: flags.append("%s-clamp" % kind) reports "Pi 4 raises flag(s) this node neither raises nor explains: ['']", and flags.append("duty-95%-low") reports "['duty-95']". A wholesale quote-style change across all ten of Pi 4's literals IS caught, by test_accounted_flags_are_really_pi4s going stale — so total blindness is covered; single incremental additions are not.
Also RUN: adding an ordinary comment inside Pi 4's COLUMNS block that happens to contain a double-quoted word ('# validity -- see the "flags" contract in ...') makes _list_literal report a phantom 41st column and fails test_columns_match_pi4_exactly_including_order with 'First extra element 40: flags' — a purely cosmetic Pi 4 edit breaking this node's suite with a message that points at the wrong thing.
```

*Why it matters:* The whole argument for copying COLUMNS and QUANT_HINTS rather than importing them is that a live drift test makes the copy safe. The test's own docstring says the flag vocabulary is the part that drifted twice before while COLUMNS stayed identical — i.e. flags are the highest-risk axis — and that is the axis where this extraction is weakest. A new Pi 4 flag written as an f-string (Python 3.6+ is available on all these nodes) passes this suite in silence, so the two nodes' rows then enter the combined table on different exclusion criteria: a bias, not a gap, which is exactly what the docstring says the test exists to prevent.

*Fix:* Replace the two regexes with the AST walk already written and working in Pi5/tests/test_pi4_schema.py:85-99: match ast.Call where func.attr in ('append','extend') and func.value.id == 'flags', read ast.Constant and BinOp-left literals (and ast.JoinedStr's first constant part for f-strings), and keep Pi 5's `assertGreater(len(pi4_tokens), 5)` extraction sanity check — the flag test currently passes vacuously if the regex matches nothing on the Pi 4 side. For _list_literal, strip comments before scanning (tokenize, or drop everything after an unquoted '#' per line) so a Pi 4 comment cannot invent a column.

**[MEDIUM] `samples` does not mean the same thing here as it does on the two nodes that define the schema. Pi 4 and Pi 5 both write len(samples_ns) — the number of repetitions llama-bench actually recorded — and both raise `no-samples:reps-unknown` when that list is empty. This node writes the requested -r from the matrix, and its `no-samples:reps-unknown` check reads `rec["repetitions"]`, which run_matrix.expand() defaults to 3 at run_matrix.py:156 and therefore always populates: the flag is unreachable.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/emit_results_csv.py:254`

*Reproduction:*
```
RUN. Called Pi 5's real pi4_rows() and this node's row_to_shared() on the same failed-run scenario (exit_code=1, no samples, avg_ts=0) and diffed field by field:
  column                PI5 (real fn)   JETSON
  samples               0               3
  flags                 ...;no-samples:reps-unknown   (no such token)
  model_size_mb         0.0             ''
  tokens_per_s          0.0             ''
  tokens_per_s_stddev   0.0             ''
  avg_ms                0.0             ''
  stddev_ms             0.0             ''
  throttled_end         '0x0'           ''
The blank-vs-0.0 rows err SAFE and are not the finding (see attacks_that_failed). `samples` does not: on a row where nothing was measured, Pi 5 states 0 observations and flags it, this node asserts 3. Confirmed by reading that llama-bench's CSV output has no samples_ns column (bench/fixtures/fake_llama_bench.py:92-94 mirrors the real header), so the observed count is genuinely unavailable here — but that makes the divergence a fact to declare, not one to paper over.
```

*Why it matters:* analysis.tokens_in_row() is `n * int(row['samples'])` and is the denominator of joules_per_token, the project's headline metric. Today this is inert because the energy columns are blank, but the shared CSV is explicitly built so the control node CAN join power to it later, and the moment a common instrument exists (the README calls that future work, not never) a row that never ran claims a 3x-larger token count. More immediately, the docstring's justification — "This node's `repetitions` is exactly that -r" — is true and beside the point: Pi 4 and Pi 5 do not put -r in this column, they put the observation count, and the two coincide only when the run succeeded. The unit test test_samples_is_the_repetition_count enshrines the wrong one of the two definitions.

*Fix:* Either derive an observation count (run_matrix already knows bench_rows and whether the invocation produced a working row; `samples` should be 0 when working_rows==0) or keep -r but say so: make the no-samples flag fire on the real condition — `if not rec.get("bench_rows") or (rec.get("status") != "ok")` — so a row with no measurements carries `no-samples:reps-unknown` exactly as Pi 4 and Pi 5's does, and amend the comment at :252-254 to state that this column is the requested count, not the observed one, unlike the other two nodes.

**[MEDIUM] `n_threads` and `n_gpu_layers` are copied from the matrix REQUEST, not from the llama-bench row, unlike both other nodes. Pi 5 documents this exact choice and refuses it ("The PI4_COLUMNS contract requires it from the leg's own entry, so flag rather than silently substituting cells.csv's value", parse_bench.py:398-400) and raises `no-threads` when the entry lacks it; this node's `no-threads` check reads rec['threads'], the config value, which expand() always sets — so it too is unreachable. run_matrix's config_mismatch() compares n_gpu_layers and model_filename but NOT n_threads, and bench_records() never records the bench row's n_threads at all, so nothing in the pipeline can detect a ref that ignored -t.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/emit_results_csv.py:241`

*Reproduction:*
```
REASONED from source, not run (the fixture bench echoes -t faithfully and I did not modify it). run_matrix.py:534-548 builds each bench record; it stores bench_n_prompt, bench_n_gen, bench_n_gpu_layers, bench_n_batch, bench_model_type — but no bench n_threads, even though fake_llama_bench.py:92 and the real llama-bench both emit that column. run_matrix.py:423-434 config_mismatch() checks only n_gpu_layers and model_filename. emit_results_csv.py:241 uses _int(rec.get("threads")) (config) and :243 uses _int(rec.get("n_gpu_layers")) (config), while the observed bench_n_gpu_layers sits unused in the same record.
```

*Why it matters:* n_threads is in the control node's group key (analysis.characterisation_table:386) AND is the `threads=` filter in profiles.per_layer_times — the module that explicitly refuses to pool across thread counts because pooling "inflates it 1.35-1.95x per node" and "goes straight into predicted tokens_per_s and joules_per_token, a named deliverable, with no flag". A CUDA-backend llama-bench that clamps or ignores -t therefore publishes the wrong operating point under the right label, and `heteroctl profiles --threads 4` selects rows that were not measured at 4 threads. The same class of error for -ngl is already caught here (config_mismatch -> PT_CONFIG_MISMATCH -> a jetson-per-token flag), which shows the pattern was understood; threads was just left out.

*Fix:* Record bench_n_threads in bench_records() alongside bench_n_gpu_layers, extend config_mismatch() to compare it against run['threads'], and have row_to_shared prefer it: `_int(rec.get("bench_n_threads"), _int(rec.get("threads")))`. Make the `no-threads` flag fire on the absence of the BENCH value (Pi 5's basis) rather than on the config value, which cannot be absent. Prefer bench_n_gpu_layers for n_gpu_layers for the same reason.

**[LOW] The auto-emit hook writes `<base>_results.csv`, but the control node's energy join requires a file named exactly `results.csv` in the run directory, and the same script's own CLI writes `results.csv` by default. One node, two spellings, and the automatic one is the spelling the collector cannot find.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:957`

*Reproduction:*
```
RUN + read. The end-to-end sweep wrote /…/results/fixture_20260914T023349Z_demo_results.csv, while `python3 emit_results_csv.py <joined> ` with no --out writes `results.csv` beside the input (emit_results_csv.py:313-314). ThinkPad/control/analysis.py:135-137: `results = run_dir / "results.csv"` … `raise FileNotFoundError("no results.csv in %s" % run_dir)`. ThinkPad/README.md:136 names the node→control artefact as `results.csv`.
```

*Why it matters:* characterisation_table reads results_energy.csv, which only join_run produces, so the Jetson's rows can only reach the report table through a step that will raise FileNotFoundError on the file the driver actually emits. The operator has to notice and rename, which is precisely the "step nobody remembers to run" the hook's own comment says it exists to eliminate. profiles.py takes an arbitrary --results path so the placement rule is unaffected by the name (it is affected by finding 1 instead).

*Fix:* Write both, or write `results.csv` and hard-link/copy the timestamped name: the per-sweep prefix is useful in a results directory holding many sweeps, but the name the control node looks for must exist. Whichever way, make emit_results_csv.py's --out default and the hook's filename agree, and say in docs/10_measurement_protocol.md §4 which file gets copied to the control node.

### Attacks that FAILED
- COLUMNS drift. Extracted the list literal from all three files programmatically: Pi4/node/bench_parse.py COLUMNS, Pi5/bench/parse_bench.py PI4_COLUMNS and the Jetson's COLUMNS are 40 entries each and compare EQUAL including order. Pi 4 and Pi 5 do not disagree, so there was no wrong parent to copy.
- QUANT_HINTS drift. Same method on the (canonical, [spellings]) pairs: all three identical, 8 entries, same order, same spelling lists. No canonicalisation split is possible between the three nodes.
- "A guard that cannot fail is not a guard" — for COLUMNS and QUANT_HINTS it can. Mutating my copy of Pi 4 made the suite FAIL for: adding a column, removing a column (idle_end), reordering two columns inside a line, and adding a spelling to QUANT_HINTS. Adding an ordinary double-quoted flags.append to Pi 4 also failed the suite. Five of nine mutations caught; the four that escaped are in the medium finding above.
- The claimed test counts. Reproduced both in a fresh `cp -a` of the cluster: `python3 -m unittest discover -s tests` -> Ran 17 tests, OK; `./bench/fixtures/run_regression.sh` -> 56 passed, 0 failed, exit 0, including its own "config/node.env unchanged" self-check. The live /mnt/data copies are byte-identical after all my work (md5 verified).
- per_layer_ms blank being unsafe for a consumer. Grepped the whole tree: nothing averages or compares the per_layer_ms COLUMN across nodes. profiles.per_layer_times ignores it entirely and recomputes ms_per_token / n_layers from the hand-written --layers map (profiles.py:130-133), report.py never reads it, and ThinkPad/control/gguf.py is the control node's own block-count source. The column is already mixed-population before the Jetson arrives — Pi 4 leaves it blank, Pi 5 fills it from the GGUF header and its own comment says "do not average it across nodes" — so the Jetson's blank is the Pi 4 behaviour, not a new hazard. The docstring's "the control node can recompute it" is literally true of the code that exists.
- The energy-blank rationale. All three grounds check out against the actual control-node source rather than its reputation: load_power() really does require t_utc/bus_v/current_a/power_w and skips any row missing them (analysis.py:33-48); integrate() really defaults sample_hz=28.0 with MAX_GAP_FACTOR=4.0, i.e. a 0.143 s gap tolerance against this logger's 0.20 s period, so every row would carry a gap: flag; tokens_in_row() really is `n * reps` with no warm-up term (analysis.py:124-127). Leaving the six columns empty is the right call and I could not break it.
- The %-split normalisation as a silent hole. I looked for a Pi 4 flag shape that slips through the `re.split(r":?%", ...)` normalisation unnoticed, and could not find one: a literal starting with a placeholder ("%s-clamp" % kind) and a literal containing a real percent sign ("duty-95%-low") both make the suite FAIL — loudly, if with a garbled token name. The percent handling is a message-quality defect, not an escape. The actual escapes are quote style, f-strings and extend/+=, reported above.
- PI4_FLAGS_NOT_RAISED_HERE being stale or fabricated. All five keys correspond to flags Pi 4 genuinely raises today (energy-span:whole-process:, low-duty:%.3f, no-window:energy-join-impossible, throttled-at-start:%s, throttled:%s), and I confirmed the reverse guard bites: rewriting all ten of Pi 4's flag literals to single quotes makes test_accounted_flags_are_really_pi4s FAIL rather than pass vacuously.
- Blank-vs-0.0 typing as a defect. The Jetson writes "" where Pi 4 and Pi 5 write 0.0 for model_size_mb, tokens_per_s, tokens_per_s_stddev, avg_ms and stddev_ms on a dead row. I expected this to be the finding and it is the opposite: analysis.characterisation_table filters with `if r.get("tokens_per_s")`, and the STRING "0.0" is truthy, so Pi 4's and Pi 5's zeros are averaged into the reported throughput while the Jetson's blanks are correctly skipped. The divergence is real but it errs in this node's favour; the bug is upstream.
- Omission of the `combined` (-pg) row. Confirmed correct and correctly reported. The shared schema's phase column has only pp and tg, analysis.tokens_in_row() picks n_prompt or n_gen by phase and has no third branch, and the Pi 5 handoff notes record a real incident where a fused entry was attributed to one phase. The CLI names the omitted rows (`omitted 2 row(s) …: missing_model/, fused/combined`). The design is right; only the failed/skipped rows riding on the same code path are wrong.

---

## Lens: `node-erasure` — broken

The projection is arithmetically correct and structurally faithful to Pi 4's schema — and it still erases the node twice over, once on the happy path and once on the failure path. (1) `ALWAYS_FLAGGED = ["throttle-unknown"]` puts a token on 100% of rows that the cross-node table's own code, `ThinkPad/control/analysis.py:224 THROUGHPUT_INVALIDATING`, treats as "this row's tokens_per_s is not a measurement". I ran a healthy, power-on, zero-failure sweep whose rows carry that single flag and nothing else, fed it to the real control node, and got `per_layer_times -> {"_dropped_flagged_rows": 30}` with NO entry for the node, every characterisation row `tokens_per_s: null`, and `report.characterisation` printing the sentence "9 row(s) were throttled on every run ... That is a result about the board, not a missing measurement". The board was never throttled. The docstring's own thesis — "what the cross-node table actually consumes ... is TIMING ... and those this node can supply honestly" — is nullified two paragraphs later by its own flag, and the node is not added to the table, it is added as a block of blank cells plus a fabricated hardware finding. (2) `row_to_shared()` derives phase ONLY from `row_phase`, but `run_matrix.py` never sets `row_phase` on a failed run (:744) or a skipped run (:686) — only `phase`. Every failed and every skipped run is therefore dropped with no row, no flag, and a message that calls it "no pp/tg phase", indistinguishable from the by-design `-pg` omission. An all-OOM sweep produces a header-only results.csv and exits 0. The emitter's own `bench-exit:%s` and `model-missing` flag branches are dead code on the driver path. And the regression's only end-to-end check of the projection happens to run on exactly that header-only file, so "56 passed, 0 failed" contains zero real assertions about the projection's content. Everything below was demonstrated by running code, not by reading it.

### Findings (6)

**[CRITICAL] `ALWAYS_FLAGGED = ["throttle-unknown"]` on every row is the exact token the cross-node table uses to exclude a row's throughput, so a perfectly healthy sweep contributes zero usable numbers and the report prints a false claim that the board throttled on every run.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/emit_results_csv.py:134`

*Reproduction:*
```
RUN. Full power-on sweep against the fixtures (fake tegrastats + fake llama-bench, matrix_split.json, no --no-power): 5/5 runs ok, 4 shared rows, `Counter({'throttle-unknown': 4})` — that one token is the ONLY flag on the healthy path. Then, from HeteroServe/ThinkPad, against that file: `analysis.characterisation_table` -> every group `tokens_per_s: None, joules_per_token: None, throughput_runs: 0`; `profiles.per_layer_times(f, {...}, threads=4)` -> `{"_dropped_flagged_rows": 4}` with NO key for the node. Repeated on the full 32-run bench/matrix.json sweep (32/32 runs ok, 30 shared rows): all 9 characterisation groups null, `{"_dropped_flagged_rows": 30}`. `report.characterisation` on that file renders the whole node as empty cells and then asserts: "9 row(s) were throttled on every run ... That is a result about the board, not a missing measurement - cool it or lower the clock ceiling and re-run." Direct probe: `analysis.invalidates_throughput('throttle-unknown') == True`; the gate is `ThinkPad/control/analysis.py:224 THROUGHPUT_INVALIDATING = ("throttled", "throttle-unknown")`, consumed at `ThinkPad/control/profiles.py:114`.
```

*Why it matters:* This is the erasure the peer session warned about, and it fires on the ONLY path the Jetson ever runs: everything succeeded, nothing was throttled, and the node still contributes nothing. It is worse than absence — the consumer converts "unknown" into an affirmative, printed finding about the hardware that is false. The claim in docs/10_measurement_protocol.md:301-305 that "whether the cross-node table accepts rows carrying that token is a decision for the table's owner" is not true as of this repo state: the owner already decided, in code, with a comment explaining why, and no note was ever sent to ThinkPad raising it (Jetson/.handoff contains no NOTE addressed to the control node). Pi 5 faced the identical question and wrote the opposite rule at Pi5/bench/parse_bench.py:164-167 — "a flag is an opt-out from the placement rule, not a caveat within it. This node raises only what it actually observed." This file copies Pi 5's COLUMNS, QUANT_HINTS and PI4_FLAGS_NOT_RAISED_HERE and then does the one thing Pi 5 said would delete the node.

*Fix:* The premise that the board cannot answer the question is refutable: telemetry/tegrastats_logger.py already records `cpu_freq_mhz` and `gpu_freq_mhz` on EVERY sample at 5 Hz, for every marked window — a strictly richer signal than Pi 4's two-point sticky vcgencmd word. Derive a real verdict per window (clock below the nvpmodel/jetson_clocks ceiling recorded in node facts while the window was busy) and raise `throttled:<detail>` only when it actually happened; raise `throttle-unknown` only when the window has no frequency samples, which keeps the fail-closed property Pi 4 wants. Until that exists, do not ship a projection whose every row is guaranteed-excluded while the docs claim it supplies timing honestly: either get the table's owner to special-case the token (an actual note to ThinkPad, not a sentence in this node's own docs), or say plainly in README/docs that the shared CSV currently contributes no throughput to the cross-node table.

**[CRITICAL] Every failed run and every skipped run vanishes from the shared CSV with no row and no flag, because `row_to_shared()` reads only `row_phase`, which run_matrix.py never sets on those two record shapes.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/emit_results_csv.py:170`

*Reproduction:*
```
RUN. (a) All-failed sweep (llama-bench stub exiting 1): driver prints `wrote ..._oom_results.csv (0 shared-schema rows, 1 omitted: no pp/tg phase)`; the file is header-only; exit 0. The joined CSV's record has `phase='prefill'`, `status='failed'`, and `row_phase` is not even a column. (b) Mixed sweep (one ok prefill + one model-missing skip): 1 shared row, and the joined CSV shows `no_model status=skipped row_phase=''`. (c) A sweep with one `-pg` case AND one skipped run prints `2 omitted: no pp/tg phase` — one is by design, one is an erasure, and the driver's message cannot tell them apart (the standalone CLI prints `e2e/combined, gone/`, so the empty one is at least visible there). (d) Old-format joined CSV: `python3 bench/emit_results_csv.py bench/fixtures/BEFORE_mixed_bench_energy.csv` -> `wrote results.csv (0 rows of 1)`, exit 0, header-only, even though that row carries n_prompt/n_gen and a valid avg_ts. Cause: run_matrix.py:686 (skipped dict) and :744 (failed dict) set `"phase"` but not `"row_phase"`; only bench_records() at :517 copies `"row_phase": rec.get("phase")`, and failed/skipped records never reach it.
```

*Why it matters:* run_matrix.py's own comment at :683 says a skipped run must still emit a record because "a run that silently vanishes from it is indistinguishable from a run that was never in the matrix" — and the projection reintroduces exactly that, for the cross-node table, which is the artefact anyone outside this folder actually reads. On a 4 GB Nano an OOM is the expected failure (the driver is built around surviving one), so the shared table shows the node succeeding at everything it attempted: survivorship bias in the direction that flatters the node, with nothing downstream able to detect it. It also makes emit_results_csv.py's own `bench-exit:%s` (line 200) and `model-missing` (line 202) branches unreachable on the driver path — the two flags written specifically to keep failures auditable can never fire. tests/test_shared_schema.py:207 and :174 assert this behaviour works, but do so on a hand-built record that injects `row_phase='prefill'` (line 127) — a shape the driver never produces — so the suite asserts precisely what is broken.

*Fix:* Two lines in run_matrix.py: add `"row_phase": phase_of(run),` to the skipped dict at :686 and the failed dict at :744, matching what bench_records already does at :517. Belt-and-braces in emit_results_csv.py:170, fall back to the invocation-level label: `PHASE_OF.get((rec.get("row_phase") or rec.get("phase") or "").strip())` — this also fixes old joined CSVs. Then update tests/test_shared_schema.py:207/:174 to build the record the way run_matrix.py does (phase set, row_phase absent) instead of injecting row_phase in the base fixture. Finally, make the driver's omission message name the rows the way the standalone does, and separate "combined (by design)" from "no phase (unexpected)".

**[HIGH] `56 passed, 0 failed` proves nothing about the projection: the regression's only end-to-end check of the shared CSV picks the last `*_results.csv` in the results dir, which is the header-only file left by the all-runs-failed test, so all four of its row-level assertions are vacuous.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/fixtures/run_regression.sh:696`

*Reproduction:*
```
RUN. Copied the script, added one stderr line printing the chosen file and its row count, ran the full suite from the kit root: `INSTRUMENT: checked file=fixture_20260914T024556Z_oom_results.csv  data_rows=0`, and the run still ends `56 passed, 0 failed`. Trace of the results dir confirms it: the last `rm -f "$FIX"/results/*` is at :461 and the last run_matrix.py invocation before :686 is the OOM test at :468, so the only `*_results.csv` present at the cross-node block is the one produced by the sweep in which every run failed. `all(r["phase"] in ("pp","tg") ...)`, `all(r["node_id"] ...)`, `all(int(r["samples"]) == 3 ...)` and `all("throttle-unknown" in r["flags"] ...)` are all `all()` over an empty list. `ls "$FIX"/results/*_results.csv` likewise passes on a file with only a header, so "the driver emits it automatically" is satisfied by an empty emission.
```

*Why it matters:* The one test that would have caught the previous finding is standing on that finding's own artefact — a self-concealing failure. Only the header-vs-Pi4 comparison in that block does real work; the claim "rows are pp/tg only" and the node_id/samples/flags guards have never executed against a single row. This is the test that a future session will point at when asked whether the projection survives a real sweep.

*Fix:* Run a fresh known-good sweep inside that block (or `rm -f "$FIX"/results/*` and re-run matrix_split.json) rather than reusing whatever is left over, and make the assertions non-vacuous: `assert len(rows) == 4` (or whatever the sweep's pp/tg count is) before the `all(...)` checks. Add the case the suite is missing entirely — a sweep containing a failed run and a skipped run, asserting that both appear in the shared CSV carrying `bench-exit:`/`model-missing`.

**[MEDIUM] When the auto-emit hook fails, the sweep exits 0, the notice goes to stdout only, and nothing about it reaches the archived `_run.log` — so a 3-hour sweep leaves no durable record that the cross-node file was never written.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:976`

*Reproduction:*
```
RUN. Copied the kit, deleted bench/emit_results_csv.py (the realistic case: a partial rsync/copy of a kit where this file is brand new), ran a 5-run sweep. Result: `!! could not write the shared-schema CSV: No module named 'emit_results_csv'` on stdout, exit code 0, the node's own `_bench_energy.csv` correctly intact, no `_results.csv`, and `grep -c "shared-schema" *_noemit_run.log` -> 0. The `!!` line is also printed BEFORE the `wrote ...` / `N/M rows ok` / per_token_status summary block, so it is not even the last thing on screen.
```

*Why it matters:* The design intent — never let this cost the sweep its own results — is right and works. But the failure signal is the weakest one available: not in the archived log the operator keeps, not in the exit status a Makefile or cron wrapper would check, and buried above a summary block that is what anyone actually reads. The hook exists because "a step nobody remembers to run is a step that does not happen"; a failure nobody can see afterwards puts the node back in exactly that state, and the first time anyone notices is when the cross-node table is missing a node.

*Fix:* Write the exception into `logfh`/`log_path` as well (or reopen the log to append it), print it AFTER the summary block so it is the last line, and make the final summary state the shared-CSV outcome explicitly on every run — `shared-schema CSV: wrote <path> (N rows)` or `shared-schema CSV: NOT WRITTEN (<reason>)`. Consider a non-zero exit (or a distinct exit code) so a wrapper can notice, since the node's own results are already safely on disk by that point.

**[MEDIUM] The driver names the shared file `<base>_results.csv`, but the control node's joiner opens exactly `run_dir/"results.csv"`, and the standalone CLI defaults to that second name — one artefact, two names, and the automatic one is the name the collector cannot find.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:957`

*Reproduction:*
```
READ + RUN. Driver writes `os.path.join(out_dir, base + "_results.csv")` (run_matrix.py:957); observed as `fixture_20260914T023104Z_pwr_results.csv`. Standalone default is `os.path.join(dirname(input), "results.csv")` (emit_results_csv.py:313-314); observed as `results.csv`. `ThinkPad/control/analysis.py:135-137`: `results = run_dir / "results.csv"` ... `raise FileNotFoundError("no results.csv in %s" % run_dir)`.
```

*Why it matters:* The auto-emit exists to remove a manual step, and the name it chooses reinstates one — the operator must rename or the join raises. The failure is loud the first time, but the recovery is a hand-copy into a fixed filename, and a results dir accumulates one `<base>_results.csv` per sweep plus possibly one stale `results.csv` from a standalone invocation. `join_run` will silently take whatever `results.csv` happens to be there, which may be an older or partial sweep, with no timestamp in the filename to contradict it.

*Fix:* Pick one name and document the handoff. Either have the driver also write/symlink `results.csv` for the most recent sweep (and say so in docs/10_measurement_protocol.md §'The cross-node projection', which currently documents only `<base>_results.csv`), or have `scripts/90_collect_results.sh` stage the chosen sweep into a run dir under the exact name `join_run` requires.

**[LOW] The documentation presents the `throttle-unknown` consequence as an open question for the cross-node table's owner, when the owner's code has already answered it and no one has told them.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/docs/10_measurement_protocol.md:301`

*Reproduction:*
```
READ. docs/10:301-305 — "Whether the cross-node table accepts rows carrying that token is a decision for the table's owner — the alternative is silence, which would be worse." README.md:126 likewise frames the shared CSV as carrying "this node's timing". Against that: `ThinkPad/control/analysis.py:218-224` already names `throttle-unknown` specifically and explains why it counts as throttled, and `ThinkPad/control/profiles.py:114` applies it. `grep -rl "throttle-unknown"` over the cluster shows the topic discussed in Pi4/, Pi5/ and ThinkPad/ handoffs, and Jetson/.handoff contains no NOTE addressed to the control node raising it.
```

*Why it matters:* A reader checking whether this node's rows land in the table will read these two paragraphs, conclude the question is pending, and not run the two lines that show it is settled and settled against this node. The framing is also a false dichotomy — 'say throttle-unknown' vs 'silence' — that hides the third option the hardware actually supports (the logged frequency trace). It is what keeps the critical finding above invisible.

*Fix:* Correct both passages to state the current consumer behaviour and its effect (every row excluded from throughput; the node renders as blanks and as a false 'throttled on every run' line in the report), and send an actual cross-node note to ThinkPad/.handoff the way Pi 5 sent 2026-09-14-0210-NOTE-blas-parity-and-shared-schema.md to this folder.

### Attacks that FAILED
- run_id disagreement between the driver and the standalone CLI. REFUTED by running both on the same sweep: driver `{'fixture_20260914T023104Z_pwr'}`, standalone `{'fixture_20260914T023104Z_pwr'}` — identical. `joined_csv = out_dir/base+"_bench_energy.csv"`, and the CLI's `re.sub(r"_bench_energy\.csv$","",basename)` inverts that exactly. The same sweep cannot appear twice under two ids.
- Enumerating every value `row_phase` can take from `phase_of_counts` (run_matrix.py:217). 'prefill'->pp and 'decode'->tg both map. 'combined' is the -pg case and is correctly omitted and reported. 'none' only arises when a bench row reports both counts zero, i.e. a row that did no identifiable work, which has no honest pp/tg home. No value produced by `phase_of_counts` should map to pp/tg and fails to. 'mixed' from `phase_of()` is unreachable because `expand()` refuses mixed cases (verified by the regression's own expand-refusal test). The erasure is not in the vocabulary — it is in `row_phase` being ABSENT, which is a different failure and is reported as a finding.
- `node_id` blank on driver-produced rows. REFUTED: all three record shapes set it from the env (run_matrix.py:694 skipped, :748 failed, :776 ok), so every driver-written row carries it; my sweeps all show `node_id=fixture`. It is blank only for a joined CSV predating the column, which is covered by the row_phase finding. (Residual, not a finding: `env.get("NODE_ID", "unknown")` means a node.env without NODE_ID silently groups the whole node under the id `unknown`.)
- Silent loss on the happy path of the real matrix. REFUTED by running the full shipped bench/matrix.json (cooldown patched to 0 in a temp copy) against the fixtures with --no-power: 32/32 runs ok, `30 shared-schema rows, 2 omitted: no pp/tg phase`, and the 2 are exactly the two `end_to_end_request` `-pg` cases. Every one of the 32 runs is either a row or a named, by-design omission.
- The drift guards being decorative. REFUTED by mutation: in a throwaway copy I added a 41st entry to Pi 4's COLUMNS and replaced a Pi 4 flag literal with `brand-new-flag`; `python3 -m unittest discover -s tests` went from 17/OK to FAILED (failures=3), naming both drifts. The live-read guards genuinely work. (Caveat, not a finding: with Pi4/ absent the suite reports `OK (skipped=4)` — it does announce itself, but run_regression.sh:687-689 only checks the exit status and still prints the label "shared-schema tests pass (incl. live drift checks against Pi4)".)
- Forcing the auto-emit hook to throw via a malformed record. FAILED — `_num`/`_int` swallow every TypeError/ValueError and every field access is `.get()`, so I could not construct a record from the driver's own vocabulary that raises inside `row_to_shared`. The hook's realistic failure is at the import, which is what I reproduced instead.
- `quant_of` mislabelling. In my fixture output the two `qwen2.5-0.5b-instruct-q8_0` rows came back `quant=Q4_K_M`, because `quant_of` scans the CONCATENATION of the node short name and llama-bench's `model_type` in QUANT_HINTS order and Q4_K_M is checked first. NOT REPORTED as a finding: `bench/fixtures/fake_llama_bench.py:97` hardcodes `model_type` as the literal "qwen2 0.5B Q4_K_M" for every model, so this is a fixture artefact — on real hardware llama-bench reports the GGUF's true quant and the two sources agree. The function is also copied verbatim from Pi 4 and enforced identical by the drift test, so any change here would be a cluster-wide decision, not a Jetson one. Worth knowing that the fixture cannot exercise the quantisation axis's group key.
- Whether the empty energy columns themselves cause erasure. REFUTED, and the design's reasoning holds: `ThinkPad/control/analysis.py:147-152` handles an empty t_start/t_end by adding a `no-window` flag and skipping the join rather than raising, and :186-194 now APPENDS any column the join adds instead of writing over the producer's fieldnames — so a missing column would not truncate the output either (that part of the emit_results_csv.py:285-287 comment is stale but harmless). The energy blanks cost the node only energy figures, which are deferred cluster-wide anyway. The throughput erasure comes entirely from the throttle flag, not from the blanks.
