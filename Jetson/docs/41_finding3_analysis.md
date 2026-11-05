# Finding #3 — analysis (five independent lenses)

Input to fixing audit finding #3 (`docs/40_audit_findings.md`): *"For mixed
prefill+decode cases the row pairs decode-only throughput with prefill+decode
energy."*

Produced by a workflow run on 2026-09-13. The **Analyze** phase completed (5/5
lenses, below); the **Design** and **Judge** phases did not finish before the
session was cut off, so there is no chosen design yet — only this evidence.

Findings are labelled `certain` / `likely` / `speculative` by the lens that wrote
them. Anything load-bearing should be confirmed against the code before acting on
it; several `certain` items below were reproduced on a fixture and say so in their
evidence.

---

## Lens: `matrix-semantics`

Exactly 7 of the 23 enabled llama-bench invocations are mixed (partial_offload_sweep 5, quantisation_sweep 2; largest_local_model adds 2 more when enabled), and each emits 2 CSV rows — so the enabled sweep is a 23-invocation / 30-row set whose join is not 1:1. Splitting makes it 30 invocations / 30 rows, costs ~2.7-3.3 min (+1.5-3.6% of the documented 1.5-3 h envelope) and zero extra token work, because `-p 512 -n 128` was never a fused measurement in the first place: llama-bench builds two independent test instances with a fresh, cleared KV cache each, so the tg128 test already decodes from an empty context exactly as a standalone `-n 128` run would. Both sweeps answer their stated questions strictly better after the split, and the real risk is not lost information but a newly-introduced confound between phase and run order.

### Findings (16)

**[CERTAIN] Exactly 7 of the 23 enabled invocations are mixed; the enabled sweep today is 23 invocations producing 30 llama-bench CSV rows, so the record-per-invocation join is already not 1:1.**  
`bench/run_matrix.py:120`  

> Driving expand()/phase_of() over matrix.json gives: cpu_prefill_sweep 4 prefill, cpu_decode_sweep 2 decode, gpu_prefill_sweep 4 prefill, gpu_decode_sweep 2 decode, partial_offload_sweep 5 MIXED, quantisation_sweep 2 MIXED, thread_sweep 4 prefill = 23 enabled; largest_local_model (enabled:false) 2 MIXED. 16 single-row + 7 two-row = 30 bench rows. phase_of() (run_matrix.py:120) returns 'mixed' for exactly those 7.

*Action:* none — this is the baseline the fix is measured against. Quote 23/30, not 'a couple of cases'.

**[CERTAIN] Splitting the two enabled mixed cases yields 30 invocations and still 30 bench rows — i.e. the split's real effect is making the invocation:row ratio 1:1, not adding measurements.**  
`bench/matrix.json:86`  

> partial_offload_sweep 5 -> 10 (5 prefill + 5 decode); quantisation_sweep 2 -> 4; 23 + 7 = 30 invocations, 30 rows. With largest_local_model enabled and split: 25 -> 32 invocations, 34 rows -> 34 rows.

*Action:* State this ratio in the commit/doc note: after the split, 'one marker window = one bench row = one phase' is an invariant the driver can assert, which is what makes the audit's proposed bench_rows>1 guard a pure tripwire rather than a routine code path.

**[CERTAIN] The split adds no token work at all — the divisor is bit-identical before and after, at both reps=3 and reps=2.**  
`bench/run_matrix.py:517`  

> tokens_of() (run_matrix.py:517) = n_p*(reps+1) + n_g*reps + (1 if n_g). reps=3: combined 512*4+128*3+1 = 2433; split = prefill 2048 + decode 385 = 2433. reps=2: combined 1793 = 1536 + 257. This holds because llama-bench runs its warm-up per test instance (a full prompt pass for pp, one token for tg), and the split preserves one instance of each.

*Action:* none — but it kills any objection that the split changes what was measured. Also note the audit's quoted divisor '(512+128)*3 = 1920' is STALE: after findings #8/#9 were fixed it is 2433. Do not re-quote 1920.

**[LIKELY] Total sweep wall-clock grows by roughly 2.7-3.3 minutes, ~1.5-3.6% of the documented 1.5-3 h envelope. The cost is 7 extra cooldowns plus 7 extra warm loads; nothing else.**  
`bench/matrix.json:19`  

> cooldown_s=20 in defaults (matrix.json:19-22) x 7 new invocations = 140 s fixed. Loads: precache() (run_matrix.py:218) reads the GGUF before every window, so the second half of each split pair finds the file in page cache -> ~2 s warm load + CUDA context init, call it 3-8 s each = 21-56 s. Total 161-196 s. scripts/50_bench_device.sh:41 documents the sweep as '1.5-3 h'.

*Action:* Do not treat wall clock as an argument against the split; it is under 4 minutes. If wall clock is ever a concern, the lever is cooldown_s or the 4-point n_prompt sweep, not phase separation.

**[CERTAIN] Nothing in the study depends on a single model load, and precache() plus the compute_s trim makes the second load essentially free in the energy domain — it costs wall clock only.**  
`bench/run_matrix.py:218`  

> The join already excludes load from the per-token columns: it re-integrates only the trailing compute_s seconds and records load_s/load_joules separately with energy_basis saying which window was used (run_matrix.py:~395-420). precache()'s docstring states the cold SD read is paid before the marker window opens. So the extra load lands outside the window; the only row-level effect is that you now get TWO load_joules samples per (model, ngl) instead of one, which is more data on load cost, not less.

*Action:* none. Explicitly reject 'we need one load' as a reason to keep mixed cases.

**[LIKELY] The study loses nothing measurable by never running a combined -p 512 -n 128, because that invocation never measured a combined workload. It is purely an artifact of how the matrix was written.**  
`docs/40_audit_findings.md:98`  

> The audit's own fact-checked (verdict-carrying, source-cited) claim at docs/40_audit_findings.md:98 states from llama-bench.cpp that get_cmd_params_instances builds SEPARATE instances for each -p and each -n value and skips zeros, and that '-p 512 -n 128 is TWO separate tests (pp512 and tg128), not one mixed test'. Each instance gets its own context with the KV cache cleared before each repetition, so the tg128 test decodes from an empty context — identical workload to a standalone -p 0 -n 128 run. The only things the two tests share in a combined invocation are the process, the loaded model, and the thermal state left behind by prefill — all three confounds, none of them signal.

*Action:* Say this plainly in the matrix.json _comment and in docs/10: a genuinely fused prefill->decode request needs llama-bench's -pg 512,128 (one instance, one row labelled pp512+tg128), which build_cmd() cannot emit today. Splitting therefore forfeits nothing that exists.

**[CERTAIN] partial_offload_sweep answers its stated question ('the placement question in miniature') ONLY after the split — the combined form structurally cannot produce the prefill-vs-ngl curve, because bench_tps() keeps the last row and discards pp512 entirely.**  
`bench/run_matrix.py:174`  

> bench_tps() (run_matrix.py:174) loops all rows with no break; its own docstring admits 'Mixed pp+tg invocations emit two rows and this keeps the last, which is why matrix cases should be phase-separated.' README.md:29-33 states the hypothesis the case exists to test: 'decode (bandwidth-bound) is expected to gain far less from -ngl than prefill (compute-bound) does — and measuring that gap on this device is one of the study's results.' Testing a gap between two curves requires both curves; today only the decode one survives.

*Action:* Split it. After the split you get t/s and J/token as functions of ngl for each phase separately, which also makes the placement result COMPOSABLE: energy of any hypothetical request = P*(J/tok)_prefill + G*(J/tok)_decode, so the P:G ratio can be varied in analysis without re-measuring. The combined form gave one point at one fixed 512:128 ratio.

**[LIKELY] quantisation_sweep is the case most damaged by mixing, because Q4_K_M vs Q8_0 is expected to flip sign between phases — averaging them into one number is the worst possible encoding for this specific hypothesis.**  
`bench/matrix.json:105`  

> matrix.json:107-110 note: 'Maxwell has no useful INT8 path, so K-quant dequantisation cost is a real term here.' Prefill is compute-bound (dequant arithmetic on the critical path, so Q8_0 can be competitive or faster); decode is bandwidth-bound on a shared 25.6 GB/s pool (so Q4_K_M's halved byte traffic should win). README.md:25-28 frames exactly this: 'the quantisation that is cheapest on the Pi 5 is not automatically cheapest here.' One mixed avg_ts plus one mixed J/token cannot show an effect that reverses between the two phases.

*Action:* Split it, and say in the note that the expected sign flip between phases is the result, not a nuisance.

**[CERTAIN] Separate case ids with the phase in the name beat a new 'phase' key, and the ids should follow the convention the matrix already uses.**  
`bench/matrix.json:26`  

> The four existing phase-pure cases are named <placement>_<phase>_sweep (cpu_prefill_sweep, cpu_decode_sweep, gpu_prefill_sweep, gpu_decode_sweep — matrix.json:26,42,56,72). --only filters on case['id'] (run_matrix.py:~272) and is the documented selector in both run_matrix.py:26 and scripts/50_bench_device.sh:8. case_id is also embedded in the marker label (run_matrix.py:324, 'phase:case_id:NNN') and lands in the output CSV.

*Action:* Use partial_offload_prefill_sweep / partial_offload_decode_sweep, quantisation_prefill_sweep / quantisation_decode_sweep, largest_local_model_prefill / largest_local_model_decode. A 'phase' key would (a) give no --only handle for re-running just one half at -r 5 per docs/10 §6, (b) leave one id meaning two runs — i.e. it is auto-splitting with extra steps — and (c) force both halves to share one note when the notes should differ. Keep 'n_gen':[0] / 'n_prompt':[0] written out explicitly, matching the existing four cases.

**[CERTAIN] The obvious minimal JSON edit is a trap: putting both values in both axes of ONE case does not split it, it produces three runs including the mixed one.**  
`bench/run_matrix.py:87`  

> expand() (run_matrix.py:87-118) takes the cartesian product of prompts x gens and skips only (0,0). So 'n_prompt':[512,0],'n_gen':[0,128] yields (512,0)=prefill, (512,128)=MIXED, (0,0)=skipped, (0,128)=decode — 3 runs per ngl, one still mixed.

*Action:* Call this out wherever the fix is written down. Two separate cases is the only correct JSON encoding.

**[CERTAIN] largest_local_model should be split too, and the split is more valuable there than anywhere else — it is the only way its stated purpose (recording where the 4 GB wall is) can be attributed to a phase.**  
`bench/matrix.json:140`  

> matrix.json:140-155: enabled:false, ngl [0,99], p 512, g 128, repetitions:2; note says 'this is where the Nano starts swapping -- which is itself a finding worth recording, with swap_peak_mb from the power CSV as evidence.' swap_peak_mb is computed per marker window (energy_between in run_matrix.py), so a combined window cannot say whether the swapping happened in prefill or decode. Separately, a pp512 batch needs a much larger compute buffer than a tg128 batch of 1, so an OOM in the prefill test kills the whole invocation (rc!=0 -> status=failed, no CSV) and takes the decode number with it; split, the decode row survives and the failed prefill row IS the finding.

*Action:* Split it, and put 'repetitions': 2 on BOTH halves. Splitting it also means zero mixed cases ship, so phase_of()'s 'mixed' branch and the bench_rows>1 guard become pure tripwires.

**[LIKELY] Splitting largest_local_model is the one place the wall-clock argument has teeth, because the 4 GB board may not keep a 1.93 GB GGUF in page cache between the two halves.**  
`bench/run_matrix.py:218`  

> precache()'s own docstring: 'On a 4 GB board the page cache can still evict it -- this makes the load cost more uniform across the axis being compared, it does not remove it.' Qwen2.5-3B-Instruct-Q4_K_M is ~1.9 GB; at the 25-45 MB/s SD figure the audit uses (docs/40_audit_findings.md:349) a cold re-read is 43-77 s. Two extra invocations then cost 2.1-3.2 min rather than ~50 s.

*Action:* Accept it; the case is disabled and expensive by design. Worth one sentence in the note so the student is not surprised when the disabled case is the slow one to enable.

**[CERTAIN] Auto-splitting silently inside expand() should be REJECTED, because bench/matrix.json is a provenance artifact that is shipped with the results and must describe the runs that actually happened.**  
`bench/run_matrix.py:87`  

> scripts/90_collect_results.sh copies bench/matrix.json into the tarball, and docs/10_measurement_protocol.md:186-188 states the reason: results are bundled 'with the node facts, the build provenance, the model checksums and the matrix that produced them, so a figure can always be traced back to the bytes it came from.' A matrix saying -p 512 -n 128 next to a CSV containing two phase-labelled rows per ngl breaks that chain. Auto-splitting would also silently decide the run ORDER and whether a cooldown sits between the halves — a methodological choice that belongs in the visible file — and would leave a user who deliberately wants a fused run (llama-bench -pg) with no way to ask for it except abandoning the driver for the manual §5 marker procedure, losing the automatic join.

*Action:* Do the split in the JSON, and add a hard REFUSAL in expand(): raise/exit naming the offending case id when n_prompt and n_gen are both non-zero, unless the case carries an explicit opt-in. That gets the fail-safe of auto-splitting with none of the dishonesty, and it fires during --dry-run — before hours of sweep — rather than at join time. Keep the bench_rows>1 guard in the join as the last line of defence for an unexpected ref. If a deliberate fused run is ever wanted, add it as its own axis that emits -pg 512,128 (phase 'combined', one instance, one row), not by re-permitting -p N -n M.

**[CERTAIN] The joined output CSV has no phase column at all — phase exists only inside the marker string — which makes old mixed rows and new split rows indistinguishable when pooled.**  
`bench/run_matrix.py:373`  

> records.append (run_matrix.py:373-380) spreads **run (case_id, note, model, n_gpu_layers, threads, n_prompt, n_gen, repetitions, flash_attn, cooldown_s) plus marker/status/rc/timestamps/avg_ts/bench_csv/bench_rows/compute_s/precached/tag. phase_of() is called at line 323 only to build the marker label at line 324. Nothing writes 'phase' as a field.

*Action:* Add phase_of(run) as its own column while doing this fix. It is one line, it makes the deliverable self-describing, and it is what lets an analyst reject pre-fix rows (phase=mixed) instead of silently averaging them with post-fix rows.

**[LIKELY] The comment above build_cmd's -p/-n handling is wrong, and 'fixing' it the way it reads would silently reintroduce a mixed invocation on EVERY case.**  
`bench/run_matrix.py:141`  

> run_matrix.py:141 says 'llama-bench rejects -p 0 / -n 0 on some refs; only pass the axis in play' — but the code immediately below always passes both, emitting -p 0 or -n 0 for the unused axis. The audit's fact-check (docs/40_audit_findings.md:98, cited to get_cmd_params_instances) confirms -p 0 / -n 0 are accepted and skipped. Passing them explicitly is REQUIRED, because llama-bench's own defaults are -p 512 -n 128; omitting a flag makes llama-bench supply those defaults and generate the very mixed pair the split is removing.

*Action:* Correct the comment to say the opposite of what it says now: both flags are always passed, and the explicit 0 is what suppresses llama-bench's 512/128 default. This is load-bearing for the split — a well-meaning cleanup here would undo the whole fix.

**[CERTAIN] Once every window is phase-pure, pooling windows by phase becomes the natural analysis move — and the marker label format blocks it, including for the procedure docs/10 §5 tells the student to run.**  
`bench/run_matrix.py:324`  

> label = '%s:%s:%03d' % (phase, case_id, i) at run_matrix.py:324, e.g. 'prefill:cpu_prefill_sweep:001'. energy_window.py's windows_by_marker() matches on exact string equality. docs/10_measurement_protocol.md:168 instructs 'energy_window.py run.csv --marker prefill', which on a driver-produced power CSV matches nothing and exits 'no samples carry marker'.

*Action:* Not required for finding #3, but pair with it: either have the logger stamp a separate phase column, or give energy_window.py prefix/substring matching on --marker. The split is what makes this newly worth having.

### Risks a naive fix runs
- Phase becomes confounded with run order and board temperature. expand() emits runs case-by-case, so after the split all 5 partial_offload prefill runs execute as a block and all 5 decode runs execute ~10 minutes later on a hotter fanless board. Prefill-vs-decode IS the study's headline axis, so any thermal drift now maps directly onto it. Finding #10 already says the fixed 20 s cooldown plus a within-window temp_rise_c check cannot detect this drift. Mitigations, cheapest first: record run_index and temp_start_c on every row so drift is regressable after the fact; or run the sweep once forward and once with the case order reversed and compare; or add a run-level interleave so prefill/decode alternate per ngl. Do NOT ship the split without at least the first.
- Interleaving was an accidental virtue of the combined form: prefill and decode for a given ngl were adjacent in time and therefore at nearly the same temperature. The split trades that local comparability for global phase purity. That is the right trade — but it should be stated in docs/10, not discovered later.
- --only partial_offload_sweep and --only quantisation_sweep stop matching once the ids change; run_matrix.py exits 'no cases selected'. Loud rather than silent, but it invalidates saved commands, the scripts/50_bench_device.sh usage examples' spirit, and anything already written into docs/30_experiment_log.md. Update the docs in the same change.
- largest_local_model's 'repetitions': 2 must be duplicated onto both halves. Omit it on one and that half silently runs at the default 3 — not a numerical error (the -r flag and the divisor stay in step) but prefill and decode then are not comparable, and the comparison is the point. This duplication hazard is the single strongest argument the auto-split camp has; the expand()-level refusal plus a keys-match assertion between split halves is the answer to it.
- Any result CSVs already collected carry mixed rows with the identical schema and no phase column. Pooling old and new files in one analysis pairs a decode-only avg_ts with a phase-pure one under the same column name. Add the phase column in the same change so old rows are self-identifying, and note the cutover in docs/30_experiment_log.md.
- The minimal-looking JSON edit ('n_prompt':[512,0],'n_gen':[0,128] on one case) does not split anything — expand()'s cartesian product yields prefill, mixed and decode, three runs per ngl. Anyone applying this fix quickly will reach for it.
- Splitting matrix.json alone does not make run_matrix.py correct. bench_tps() still keeps the last row and the record is still one-per-invocation; the split merely means no window produces more than one row today. Without the bench_rows>1 guard (bench_rows is already on the row, so it is a two-line check) the bug returns the moment anyone adds a mixed case, uses a ref whose defaults differ, or drops the explicit -p 0 / -n 0.
- Do not let a tidy-up delete the explicit '-p 0' / '-n 0' in build_cmd() on the strength of its own misleading comment. llama-bench's defaults are exactly 512 and 128, so an omitted flag reinstates a mixed pair on every single case in the matrix — a far worse regression than the one being fixed.

---

## Lens: `driver-code`

run_matrix.py collapses one llama-bench invocation into one output record, so a `-p 512 -n 128` case publishes the tg128 throughput next to joules covering prefill+decode+load; on a synthetic but physically-shaped window the published figure is 0.3398 J/tok where the honest per-phase numbers are 0.184 (prefill) and 1.196 (decode). The fix is a per-bench-row record set built by a new `bench_records()`, a `per_token_refusal()` guard that blanks the three per-token columns with a stated reason whenever a window holds more than one row, and a `tokens_of()` that reads the row's own n_prompt/n_gen instead of the matrix's. Two things must not be attempted: splitting a two-row window by time (the unmeasured inter-test context churn makes the prefill error grow from -0.3% at ngl 0 to -24% at ngl 99, i.e. it tracks the very axis being swept), and a naive `for row in rows: records.append(...)` loop, which silently deletes failed and unparseable runs from the output.

### Findings (15)

**[CERTAIN] The run loop emits exactly one record per invocation, so a two-test invocation loses one test entirely. Replace lines 363-380 with a `bench_records()` call that returns one record per bench row, or exactly one placeholder record when there are none.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:372`  

> run_matrix.py:372-380 — `tps = bench_tps(bench_rows)` then a single `records.append({**run, "marker": label, ..., "avg_ts": tps, "bench_rows": len(bench_rows), "compute_s": compute_seconds(bench_rows, run["repetitions"]), ...})`. `bench_rows` is already a count on the row, so the driver knows it emitted 2 rows and still publishes one.

*Action:* Replace 363-381 with:

            csv_text = bench_csv_text(out.stdout or "")
            with open(case_csv, "w") as fh:
                fh.write(csv_text)
            bench_rows = list(csv.DictReader(csv_text.splitlines())) if csv_text else []
            if not bench_rows:
                print("    !! no llama-bench CSV header in stdout (see %s)" % log_path)
                logfh.write("\n!! unparseable stdout for %s:\n%s\n"
                            % (run_label(run), out.stdout or ""))

            # One record per bench row. Everything that describes the WINDOW --
            # marker, timestamps, compute_s, precached, tag -- is computed once
            # here and copied onto each row by bench_records(); everything that
            # describes the TEST -- test, avg_ts, avg_ns, tokens -- comes off
            # that row alone.
            base = {**run, "marker": label, "phase": phase, "status": "ok", "rc": 0,
                    "t_start_unix": t_start, "t_end_unix": t_end,
                    "bench_csv": os.path.basename(case_csv),
                    "compute_s": compute_seconds(bench_rows, run["repetitions"]),
                    "precached": int(cached), "tag": args.tag}
            new = bench_records(run, bench_rows, base)
            records.extend(new)
            for rec in new:
                print("    %-8s %8.2f t/s  %6s tok  %.1f s"
                      % (rec.get("test") or "-", rec.get("avg_ts") or 0.0,
                         rec.get("tokens") or "-", t_end - t_start))
            if len(new) > 1:
                print("    !! %d bench rows in ONE power window -- per-token energy "
                      "refused for this case; split it in matrix.json" % len(new))
            time.sleep(run["cooldown_s"])

and add, next to the other helpers:

def bench_records(run, bench_rows, base):
    """One output record per llama-bench row -- or exactly ONE record when the
    invocation produced none, so a parse failure still appears in the joined
    CSV instead of vanishing from it. Sibling rows share `marker` (unique per
    invocation) and carry bench_rows / bench_row_index / window_shared, so they
    read as ONE measurement seen twice, not two measurements."""
    n = len(bench_rows)
    reps = run["repetitions"]
    if not n:
        rec = dict(base)
        rec["bench_rows"] = 0
        rec["per_token_status"] = per_token_refusal(rec)
        return [rec]

    tests = "+".join(test_label(bench_int(r, "n_prompt") or 0,
                                bench_int(r, "n_gen") or 0) for r in bench_rows)
    mismatch = window_counts_mismatch(run, bench_rows)
    out = []
    for i, r in enumerate(bench_rows):
        n_p, n_g = bench_int(r, "n_prompt"), bench_int(r, "n_gen")
        eff_p = n_p if n_p is not None else run["n_prompt"]
        eff_g = n_g if n_g is not None else run["n_gen"]
        rec = dict(base)
        rec.update({
            "bench_rows": n,
            "bench_row_index": i,
            "window_shared": int(n > 1),
            "window_tests": tests,
            "test": r.get("test") or test_label(eff_p, eff_g),
            "row_phase": phase_of_counts(eff_p, eff_g),
            "bench_n_prompt": n_p,
            "bench_n_gen": n_g,
            "bench_n_depth": bench_int(r, "n_depth"),
            "counts_source": "bench" if (n_p is not None or n_g is not None) else "matrix",
            "counts_mismatch": int(mismatch),
            "avg_ts": row_tps(r),
            "avg_ns": r.get("avg_ns") or "",
            "stddev_ts": r.get("stddev_ts") or "",
            "row_compute_s": row_compute_seconds(r, reps),
        })
        rec["tokens"] = tokens_of(rec)
        rec["per_token_status"] = per_token_refusal(rec)
        out.append(rec)
    return out

Keep unchanged: load_env, as_list, find_bin, model_path, expand, phase_of, run_label, build_cmd, bench_csv_text, precache, _rows.

**[CERTAIN] `bench_tps()` must be deleted, not kept alongside the per-row code. Its entire semantic is the bug: it iterates all rows without breaking and returns the last one's avg_ts.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:174`  

> run_matrix.py:174-185 — `for r in rows: for k in ("avg_ts", ...): if r.get(k): tps = float(r[k])`, with the docstring already conceding "Mixed pp+tg invocations emit two rows and this keeps the last". llama-bench orders instances pp-before-tg (get_cmd_params_instances iterates params.n_prompt then params.n_gen), so "last" is always the decode row.

*Action:* Delete bench_tps() and replace it with a single-row accessor:

def row_tps(row):
    """avg_ts off ONE bench row. Column names have shifted between refs."""
    for k in ("avg_ts", "t/s", "tokens_per_second"):
        if row.get(k):
            try:
                return float(row[k])
            except ValueError:
                pass
    return None

Note for whoever runs the regression snippet: the handoff smoke test at .handoff/2026-09-12-1541-apply-audit-findings.md:372-377 imports bench_tps and asserts on it. Update that line to `from run_matrix import bench_csv_text, row_tps, compute_seconds, tokens_of` / `assert row_tps(list(csv.DictReader(bench_csv_text(banner).splitlines()))[0]) == 10.0`.

**[LIKELY] compute_s stays meaningful for a two-row window as a SUM of timed-loop seconds, but its trailing-trim premise ("all non-compute strictly precedes all compute") is false there: llama-bench creates and frees a context per test instance, so there is an unmeasured multi-watt gap BETWEEN the two timed loops, inside the trimmed slice.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:188`  

> run_matrix.py:188-215 loops over all rows accumulating `ns * reps + warm_ns`; the join at 410-418 then takes `energy_between(power_csv, rec["t_end_unix"] - cs, rec["t_end_unix"])` on the strength of the comment "All of it strictly PRECEDES the first timed loop". On a 5 Hz fixture shaped load 18 s / ctx 1 s / pp512 51.2 s / ctx churn 2 s / tg128 96.25 s / teardown 1.5 s, compute_s = 147.45 s and the slice starts 22.45 s after t_start while prefill compute truly starts at 19.00 s — it eats 3.45 s of real prefill at the front and swallows the churn and teardown instead. Total error is small (826.8 J trimmed vs 837.7 J true, -1.3%), so the aggregate is fine; the alignment is not.

*Action:* Split compute_seconds into a per-row function and keep the sum, so the window total is unchanged and each row also carries its own timed seconds:

def row_compute_seconds(row, reps):
    """Seconds ONE bench row spent inside llama-bench's timed loops, from its
    own avg_ns, plus the unmeasured warm-up: a full prompt pass for a pp test,
    one token for a tg test. Twin of bench_compute_s() in energy_window.py."""
    try:
        ns = float(row.get("avg_ns") or "")
    except (TypeError, ValueError):
        return None
    if not ns:
        return None
    n_p = bench_int(row, "n_prompt") or 0
    n_g = bench_int(row, "n_gen") or 0
    if n_p and n_g:
        warm_ns = ns * (1.0 + 1.0 / n_g) / 2.0   # a -pg row warms both ways
    elif n_p:
        warm_ns = ns
    elif n_g:
        warm_ns = ns / n_g
    else:
        warm_ns = 0.0
    return round((ns * reps + warm_ns) / 1e9, 3)


def compute_seconds(rows, reps):
    """Timed-loop seconds over EVERY row of one invocation. For a one-row
    window this is the trailing slice of the marker window that processed
    tokens. For a multi-row window it is the SUM of two such slices with an
    unmeasured context teardown/creation gap between them -- the right total,
    but no longer one contiguous trailing interval, which is why per-token
    energy is refused there rather than split."""
    total, seen = 0.0, False
    for r in rows:
        s = row_compute_seconds(r, reps)
        if s is not None:
            total += s
            seen = True
    return round(total, 3) if seen else None

Keep computing compute_joules / load_s / load_joules at window level for multi-row windows (they are still the honest not-inside-a-timed-loop residual), but say so on the row via energy_basis (see the join finding).

**[LIKELY] A two-row window cannot be split by time from anything the driver records, and a plausible-looking attempt reintroduces finding #3's exact failure mode: the error is monotonic in the swept axis.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:410`  

> The power CSV carries one marker value for the whole invocation (tegrastats_logger.read_marker stamps whatever the driver last wrote; the driver writes `label` once at 335 and "idle" at 346), so there is no boundary event between pp and tg. The driver's only absolute timestamps are t_start/t_end; avg_ns gives durations, never offsets, and the load time, the inter-test context churn and the teardown are all unmeasured. Simulating the naive split (boundary = t_end - row_compute_s(tg)) over a churn that grows with offload: ngl 0, churn 0.4 s -> prefill est -0.3%; ngl 16, churn 2 s -> -2.5%; ngl 99, churn 6 s -> -24.2%, because prefill compute shrinks with ngl while the CUDA alloc/free gap grows. Decode stays within -0.7% throughout.

*Action:* Do not implement a time split in the driver. Emit `row_compute_s` (above) as a column instead: it is llama-bench's own exact per-phase timed seconds, so a thesis author who wants a per-phase estimate can compute `compute_joules * row_compute_s / compute_s` downstream and own the constant-power assumption explicitly, in the analysis, where it can be stated in the text. The driver's contract stays: it publishes measured joules or nothing.

**[CERTAIN] The refusal guard belongs in one pure function called twice: at record-build time (so it survives a --no-power run, where the join never executes) and again in the join (so it can add the no-energy reason). It must check status first, or a failed OOM run gets labelled a CSV parse failure.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:431`  

> The join block is gated on `if os.path.exists(power_csv)` (run_matrix.py:395), so a --no-power sweep never enters it; a guard written only inside the join would leave per_token_status blank on every row of such a run. In my end-to-end run, seeding the guard at build time and refining it in the join produced: failed record -> "run failed (rc=1): nothing was measured"; both mixed rows -> "2 bench rows share this power window (pp512+tg128); joules covers every phase in it and cannot be attributed"; single-phase record -> "ok" with joules_per_token 0.181055, wh_per_mtok 50.29, active_joules_per_token 0.133555.

*Action:* Add next to tokens_of():

def per_token_refusal(rec):
    """Why this row must NOT carry per-token energy, or "" when it may.

    The numerator is a property of the marker WINDOW; the divisor is a property
    of the bench ROW. They may only be divided when the window holds exactly
    one row and that row is one phase. Everything else gets joules without
    joules_per_token -- an empty cell with a stated reason beats a number that
    silently averages a 0.18 J/tok prefill with a 1.20 J/tok decode.
    """
    if rec.get("status") != "ok":
        return "run failed (rc=%s): nothing was measured" % rec.get("rc")
    n = rec.get("bench_rows")
    if not n:
        return "no bench row: llama-bench produced no parseable CSV for this window"
    if n > 1:
        return ("%d bench rows share this power window (%s); joules covers every "
                "phase in it and cannot be attributed -- split the case in "
                "matrix.json into prefill-only and decode-only"
                % (n, rec.get("window_tests") or "?"))
    if rec.get("row_phase") == "mixed":
        return ("this bench row is itself a fused prompt+gen test (%s); its joules "
                "cover both phases" % (rec.get("test") or "?"))
    if not rec.get("tokens"):
        return "no token count: bench row reports neither n_prompt nor n_gen"
    return ""

The `row_phase == "mixed"` arm is not dead code: it catches a `-pg` row, which llama-bench renders as a single fused pp512+tg128 test. build_cmd() never emits -pg today, so it is a tripwire against a future edit.

In the join, replace 431-436 with:

            reason = rec.get("per_token_status") or per_token_refusal(rec)
            if not reason and energy_j is None:
                reason = "no energy: fewer than 2 power samples in this window"
            if reason:
                rec["per_token_status"] = reason
                continue
            rec["joules_per_token"] = round(energy_j / tokens, 6)
            rec["wh_per_mtok"] = round(energy_j / tokens * 1e6 / 3600.0, 2)
            if "active_joules" in rec:
                rec["active_joules_per_token"] = round(rec["active_joules"] / tokens, 6)
            rec["per_token_status"] = "ok"

Blank the columns by NOT SETTING the keys (DictWriter's restval is ""), never by writing None or 0. Keep writing `tokens` on a refused row: with per-row records the divisor is now honest (2048 for the pp row, 385 for the tg row); it is only the numerator that cannot be attributed, and saying so is the whole point of the status column.

**[CERTAIN] tokens_of() must read the ROW's n_prompt/n_gen, and the parse must distinguish a reported 0 from an absent column — that distinction is precisely the bug finding #7 fixed on the analysis side. It should also carry n_depth, which current llama-bench refs emit.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:517`  

> run_matrix.py:517-532 reads `rec.get("n_prompt")`, which comes from `{**run}` — the matrix's request, not what ran; for a mixed case both are non-zero, giving (512*4 + 128*3 + 1) = 2433 on both rows. The repo's own fact-check (docs/40_audit_findings.md:98) states get_fields() is "build_commit … n_prompt, n_gen, n_depth, test_time, avg_ns, stddev_ns, avg_ts, stddev_ts", i.e. every row carries both counts with the unused one zeroed, and n_depth exists. My run: with the row's own counts, the pp row divides by 2048 and the tg row by 385.

*Action:* Add the strict parser and rewrite tokens_of():

def bench_int(row, key):
    """Integer field off a llama-bench row, or None when the column is absent
    or unparseable. A literal 0 is a VALUE, not a miss: llama-bench writes
    n_prompt=0 on every tg row, and reading that 0 as "missing" is the bug
    finding #7 fixed in energy_window.pick(skip_zero=). Never test truthiness
    to decide presence here."""
    if key not in row:
        return None
    v = row.get(key)
    if v is None or str(v).strip() == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def tokens_of(rec):
    """Tokens actually processed, for ONE bench row. [keep the existing
    per-repetition + warm-up paragraph verbatim, then add:]

    Counts come from the row itself -- what llama-bench ran -- and fall back to
    the matrix's request ONLY when the ref emits no count columns at all. A row
    that reports 0 is reporting a fact. n_depth (newer refs) prefills that many
    tokens before every repetition, so it rides with the prompt term.
    Must stay in step with bench_tokens() in telemetry/energy_window.py.
    """
    n_p, n_g = rec.get("bench_n_prompt"), rec.get("bench_n_gen")
    if n_p is None and n_g is None:
        n_p, n_g = rec.get("n_prompt"), rec.get("n_gen")
    n_p = n_p or 0
    n_g = n_g or 0
    depth = rec.get("bench_n_depth") or 0
    reps = rec.get("repetitions") or 1
    return ((n_p + depth) * (reps + 1) + n_g * reps + (1 if n_g else 0)) or None

Verified behaviour: matrix fallback when the ref has no count columns -> 2048; tg row -> 385; n_depth 256 on a pp512 row -> 3072; bench reports 0/0 -> None (no fallback, so per_token_refusal blanks the row rather than fabricating a divisor from the matrix).

**[CERTAIN] When the bench CSV disagrees with the matrix, the row must win the divisor and the disagreement must be visible. The comparison has to be made at WINDOW level over the set of tests, or every legitimate mixed window falsely flags.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:517`  

> My first attempt compared each row's counts against run["n_prompt"]/run["n_gen"] and set counts_mismatch=1 on both rows of a perfectly normal p512+g128 window, because the pp row carries n_gen=0 against a requested 128. Set-level comparison gives: asked p512+g128 got only pp512 -> True (a test was dropped); asked p512 got pp512 -> False; asked p512 got pp256 -> True; ref with no count columns -> False.

*Action:* Add:

def window_counts_mismatch(run, rows):
    """True when what llama-bench ran is not what the matrix asked for -- a
    dropped test, or a count the ref changed under us. Compared as a SET of
    tests over the whole invocation, because one row only ever exercises one
    axis and zeroes the other."""
    want = set()
    if run["n_prompt"]:
        want.add("pp%d" % run["n_prompt"])
    if run["n_gen"]:
        want.add("tg%d" % run["n_gen"])
    got = set()
    for r in rows:
        n_p, n_g = bench_int(r, "n_prompt"), bench_int(r, "n_gen")
        if n_p is None and n_g is None:
            return False        # ref has no count columns; nothing to compare
        if n_p:
            got.add("pp%d" % n_p)
        if n_g:
            got.add("tg%d" % n_g)
    return want != got

Write it as `counts_mismatch` on every row of the window and print a warning in the run loop when it trips. What breaks if they disagree and the matrix wins instead: the divisor describes a test that did not run, and nothing on the row says so. What breaks if the row wins: nothing measurable — n_prompt/n_gen stay on the row as the request, bench_n_prompt/bench_n_gen as the fact, counts_source names which one tokens_of used.

**[CERTAIN] The join must compute energy once per WINDOW and fan it out, not once per record. Per-record integration re-parses the whole power CSV for every row — with per-row emission that is up to 4 full parses per two-row window — and it makes it structurally possible for two rows of one window to disagree about their shared joules.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:398`  

> run_matrix.py:398-418 calls `energy_between(power_csv, ...)` inside `for rec in records`, and `energy_between` starts with `_rows(power_csv)` (456-458), a full DictReader over the file; the compute trim at 413 is a second full parse. At the default 5 Hz over a 1.5-3 h sweep (scripts/50_bench_device.sh) that file is tens of thousands of rows on a Cortex-A57.

*Action:* Change the two helpers to take parsed rows — they are module-private, used nowhere outside run_matrix.py (grep confirms):

def idle_floor(rows):        # was idle_floor(power_csv)
def energy_between(rows, t0, t1):   # was energy_between(power_csv, t0, t1)

bodies otherwise unchanged (keep the `if len(sel) < 2: return {}` early-out and all the temp/swap handling). Then replace 394-437 with:

    idle_mw = 0.0
    power_rows = _rows(power_csv) if os.path.exists(power_csv) else []
    if len(power_rows) >= 2:
        idle_mw = idle_floor(power_rows)
        print("\nidle floor: %.0f mW" % idle_mw)

        # Energy is a property of the marker WINDOW, and with one record per
        # bench row a window now owns several records. Integrate each window
        # ONCE and copy the result onto its rows: re-parsing a 50k-row power
        # CSV per record is minutes of A57 time, and sharing one dict makes it
        # impossible for two rows of one window to disagree about its joules.
        windows = {}
        for rec in records:
            windows.setdefault(rec["marker"], []).append(rec)

        for marker in sorted(windows):
            group = windows[marker]
            head = group[0]
            e = energy_between(power_rows, head["t_start_unix"], head["t_end_unix"])
            shared = dict(e)
            cs, dur = head.get("compute_s"), e.get("duration_s")
            if cs and dur and 0 < cs < dur and e.get("joules") is not None:
                ce = energy_between(power_rows, head["t_end_unix"] - cs, head["t_end_unix"])
                if ce.get("joules") is not None:
                    shared["compute_joules"] = ce["joules"]
                    shared["compute_mean_mw"] = ce.get("mean_mw")
                    shared["load_s"] = round(dur - cs, 3)
                    shared["load_joules"] = round(e["joules"] - ce["joules"], 4)
            n_rows = head.get("bench_rows") or 0
            if "compute_joules" in shared:
                energy_j, window_s = shared["compute_joules"], cs
                shared["energy_basis"] = (
                    "compute" if n_rows <= 1 else
                    "compute (window holds %d bench rows, not one phase)" % n_rows)
            elif e.get("joules") is not None:
                energy_j, window_s = e["joules"], dur
                shared["energy_basis"] = "whole-process (includes model load)"
            else:
                energy_j, window_s = None, None
            # active_joules is a WINDOW quantity and needs no token count --
            # it was previously trapped inside the token-gated branch, so a
            # window with energy but no parsed tokens lost it for no reason.
            if energy_j is not None:
                active = energy_j - idle_mw / 1000.0 * (window_s or 0.0)
                shared["active_joules"] = round(max(active, 0.0), 4)
            shared["idle_mw"] = round(idle_mw, 1)

            for rec in group:
                rec.update(shared)
                <the per_token_refusal block from the guard finding>

`marker` is a safe group key: label = "%s:%s:%03d" % (phase, case_id, i) with i the global run index (run_matrix.py:324), so it is unique per invocation, and the failed-run record carries it too.

**[CERTAIN] Per-row emission does not make rows ragged — DictWriter pads missing keys — but the union-over-records fieldname build has two real defects that per-row emission makes worse: column order depends on which case ran first, and a column absent from EVERY record disappears from the header entirely.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:439`  

> Measured: `DictWriter(fieldnames=[a,b,c])` on `{"a":1}` writes `1,,` and on `{"a":1,"b":None,"c":2}` writes `1,,2`; an extra key raises `ValueError: dict contains fields not in fieldnames: 'zz'`. run_matrix.py:439-443 builds `keys` by first-seen order across records, so if every enabled case is mixed, no record ever gets joules_per_token and the thesis CSV has no such column at all — a KeyError downstream, indistinguishable from a broken driver.

*Action:* Replace 439-443 with an explicit schema plus an append-anything-else fallback (so a future key is never dropped and DictWriter can never raise):

    # Column order must not depend on which case happened to run first: two
    # sweeps of the same matrix have to concatenate. Listing the per-token
    # columns unconditionally also means a sweep where every window was
    # refused still HAS those columns, uniformly blank, with per_token_status
    # saying why -- rather than silently missing, which reads as a broken run.
    PREFERRED_COLUMNS = [
        "case_id", "note", "tag", "status", "rc", "marker", "phase",
        "model", "n_gpu_layers", "threads", "n_prompt", "n_gen", "repetitions",
        "flash_attn", "cooldown_s", "precached",
        "bench_csv", "bench_rows", "bench_row_index", "window_shared",
        "window_tests", "test", "row_phase", "bench_n_prompt", "bench_n_gen",
        "bench_n_depth", "counts_source", "counts_mismatch",
        "avg_ts", "stddev_ts", "avg_ns", "row_compute_s",
        "t_start_unix", "t_end_unix", "duration_s", "power_samples",
        "mean_mw", "peak_mw", "temp_max_c", "temp_rise_c", "swap_peak_mb",
        "joules", "compute_s", "compute_joules", "compute_mean_mw",
        "load_s", "load_joules", "energy_basis", "idle_mw", "active_joules",
        "tokens", "per_token_status",
        "joules_per_token", "wh_per_mtok", "active_joules_per_token",
    ]
    keys = list(PREFERRED_COLUMNS)
    for rec in records:
        for k in rec:
            if k not in keys:
                keys.append(k)

Measured result: 55 columns, header identical for a --no-power run and a full run, one field count per line.

**[CERTAIN] Failed runs and unparseable-stdout runs must keep producing exactly one record. A literal reading of "emit one record per bench row" deletes both from the output — the second one silently, because it exits with rc=0.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:352`  

> run_matrix.py:352-361 appends one record with status=failed and no bench rows; 366-370 keeps going with `bench_rows = []` and status=ok when the CSV header is missing, printing only "!! no llama-bench CSV header in stdout". Both would produce zero records under a naive `for row in bench_rows` loop. The second case is exactly the failure mode finding #1 was about, so losing its evidence would be a regression on a fixed finding.

*Action:* bench_records() already returns a one-element list when `bench_rows` is empty (see the run-loop finding). Also extend the failure append at 356-359 so it has phase, bench_rows and a reason:

                rec = {**run, "marker": label, "phase": phase, "status": "failed",
                       "rc": out.returncode, "t_start_unix": t_start,
                       "t_end_unix": t_end, "tag": args.tag, "bench_rows": 0}
                rec["per_token_status"] = per_token_refusal(rec)
                records.append(rec)

Verified output for that record: per_token_status = "run failed (rc=1): nothing was measured", all per-token cells blank, energy_basis blank rather than falsely claiming "whole-process". Separately, the model-missing branch at 317-319 still emits NO record at all (it only prints SKIP) — out of scope for #3, but it means a skipped case is invisible in the CSV.

**[CERTAIN] The closing summary counts records, which after per-row emission is bench rows, not runs — it would report "30/30 runs ok" for a 23-run sweep. It should also say out loud how many windows were refused, so a sweep that produced no per-token energy cannot look successful.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:452`  

> run_matrix.py:452-453 — `ok = sum(1 for r in records if r.get("status") == "ok"); print("%d/%d runs ok" % (ok, len(records)))`. With the matrix as shipped, partial_offload_sweep (5) and quantisation_sweep (2) each emit 2 rows, so len(records) exceeds the run count by 7.

*Action:* Replace 452-453 with:

    windows = {r["marker"] for r in records}
    ok = {r["marker"] for r in records if r.get("status") == "ok"}
    refused = sorted({r["case_id"] for r in records
                      if (r.get("per_token_status") or "") not in ("", "ok")})
    print("%d/%d windows ok, %d bench rows" % (len(ok), len(windows), len(records)))
    if refused:
        print("!! per-token energy refused for: %s" % ", ".join(refused))
        print("   see the per_token_status column; split those cases in %s"
              % os.path.basename(args.matrix))

**[CERTAIN] `phase` is computed for the marker but never written onto the record; adding it alongside the new per-row `row_phase` turns the pair into the tripwire that catches a matrix/bench disagreement at a glance.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:323`  

> run_matrix.py:323 `phase = phase_of(run)` feeds only `label` at 324. The joined CSV has no phase column — it is recoverable only by string-splitting `marker`. docs/10_measurement_protocol.md's "Reading the output" table lists no phase column either.

*Action:* Put `"phase": phase` in the `base` dict (case-level, from the matrix) and `row_phase` per bench row (from that row's counts), as in the run-loop code above. Add the helpers:

def phase_of_counts(n_p, n_g):
    if n_p and not n_g:
        return "prefill"
    if n_g and not n_p:
        return "decode"
    if n_p and n_g:
        return "mixed"
    return "none"

and rewrite phase_of(run) as `return phase_of_counts(run["n_prompt"], run["n_gen"])` so the marker and the column can never drift apart. A row where phase="mixed" but row_phase="prefill" is exactly the case #3 is about, readable without joining anything.

**[CERTAIN] There is no `test` column in llama-bench's CSV output, so the pp512/tg128 label must be synthesised by the driver — reading it from the row will silently produce blanks.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:372`  

> docs/40_audit_findings.md:98, a fact-checked (not unverified) claim with upstream sources: "There is NO `test` column in the CSV — get_fields() is build_commit … n_prompt, n_gen, n_depth, test_time, avg_ns, stddev_ns, avg_ts, stddev_ts; `test` only exists in the markdown renderer." telemetry/energy_window.py:300 still does `pick(b, "test", "n_prompt", skip_zero=True)`, which therefore always falls through.

*Action:* Add, and prefer the row's own column only if some future ref actually grows one:

def test_label(n_p, n_g):
    """llama-bench's own pp512 / tg128 naming, derived here because the CSV
    printer has NO `test` column -- it exists only in the markdown renderer --
    so the label has to be reconstructed from the row's counts."""
    if n_p and n_g:
        return "pp%d+tg%d" % (n_p, n_g)
    if n_p:
        return "pp%d" % n_p
    if n_g:
        return "tg%d" % n_g
    return ""

used as `r.get("test") or test_label(eff_p, eff_g)` in bench_records(). The analysis half of the kit needs the same label derived the same way (energy_window.py:300) or the two tools will name the same window differently.

**[CERTAIN] Splitting partial_offload_sweep and quantisation_sweep in matrix.json fixes today's data but not the class of bug: largest_local_model is still mixed, and any hand-edit re-introduces it. The driver guard is the invariant that makes the matrix edit safe to get wrong.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/matrix.json:140`  

> bench/matrix.json:140-155 — largest_local_model, currently "enabled": false, carries n_prompt [512] and n_gen [128] with repetitions 2; its note tells the operator to "Enable once you have confirmed it fits". matrix.json:16 invites exactly this: "Edit freely." phase_of() returns "mixed" for it, which today only changes a string in the marker and nothing else.

*Action:* Ship both halves and keep them independent: the matrix split is the data fix, per_token_refusal() is the invariant. If stronger enforcement is wanted, the alternative is to split inside expand() — after the `if p == 0 and g == 0: continue` guard, emit two runs when both p and g are non-zero — which makes phase purity structural for any matrix anyone writes. Cost, so the choice is made with eyes open: +7 windows on the enabled matrix, each paying its own model load and 20 s cooldown (roughly +5-8 min plus load time on a 1.5-3 h sweep per scripts/50_bench_device.sh), and the `%03d` run indices in every marker shift, so markers are not comparable to an earlier sweep's. Do not do both the matrix split and the expand() split, or the split cases become four windows.

**[CERTAIN] Duplicating the window's joules onto sibling rows is the right default (each row stands alone, and a blank cell keeps meaning "the join failed"), but it makes a naive column sum double-count, so the row must advertise the sharing.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:398`  

> In my end-to-end run both rows of the mixed window carry joules=912.34 and compute_joules=826.76. Summing the joules column over that CSV overstates the sweep's energy by one full window per mixed case.

*Action:* Carry `window_shared` (0/1) and `bench_rows` on every row, as in bench_records(), and add one line to the docs/10_measurement_protocol.md column table: energy columns are per WINDOW and repeat across sibling rows — sum them over `drop_duplicates("marker")`, group throughput and tokens by row. The alternative (writing joules only on bench_row_index 0) was rejected: it makes a blank cell ambiguous between "sibling row" and "join failed", which is the exact ambiguity the refusal guard exists to remove.

### Risks a naive fix runs
- Deleting bench_tps() breaks the regression snippet recorded at .handoff/2026-09-12-1541-apply-audit-findings.md:372-382, which imports it and asserts on it. Update that snippet in the same change or the next session's first smoke test fails for the wrong reason.
- Changing energy_between()/idle_floor() to take parsed rows instead of a path is a silent-corruption hazard if any call site is missed: passing a path string where rows are expected makes `for r in rows` iterate characters, `r.get` raises AttributeError on str, and `energy_between` would return {} — every energy column blank with no error. grep confirms both functions are used only inside run_matrix.py (energy_window.py has its own summarise()), so the change is contained, but all four call sites must move together.
- tokens_of() is explicitly required to stay in step with bench_tokens() in telemetry/energy_window.py (its own docstring says so). Adding the n_depth term and the row-counts-first rule to one and not the other makes the driver and the standalone analysis tool disagree about the divisor for the same window — the exact class of inconsistency finding #22 documents for the idle floor.
- A per-row record set changes the row count of the joined CSV, so anything that assumed one row per run now over-counts. In this kit that is only the closing "%d/%d runs ok" line, but a thesis notebook written against an earlier sweep will double-count mixed cases in any per-row aggregate until it groups by marker.
- Attributing compute_joules per row by time is the tempting next step and must be resisted: on a fixture where the inter-test context churn grows with offload (0.4 s at ngl 0 to 6 s at ngl 99) the prefill estimate degrades from -0.3% to -24.2%, monotonically in the swept axis — it would manufacture exactly the false energy-vs-ngl trend that finding #3 describes, while looking like a fix.
- The refusal guard is only correct if bench_rows is trustworthy. If a future ref prints a second header line or a trailing blank line inside the captured stdout, DictReader yields a junk row, bench_rows becomes 2, and a legitimate single-phase case is refused per-token energy for the rest of the sweep. bench_csv_text() strips the preamble but nothing validates the rows after it; consider dropping rows where both bench_int(n_prompt) and bench_int(n_gen) are None before counting.
- PREFERRED_COLUMNS is a second place that has to be updated whenever a new key is added to a record. The append-leftovers loop means a forgotten column is misplaced rather than lost, but a stale list quietly reintroduces order instability for the new field.

---

## Lens: `energy-window-pairing`

The --bench join in energy_window.py is a bare positional zip with no guard: `for i, res in enumerate(results): if i >= len(bench): break; b = bench[i]`. I reproduced the desync on a synthetic fixture (3 mixed `-p 512 -n 128` windows, 6 bench rows): window 1 (ngl 16) was billed with ngl 0's tg128 row and window 2 (ngl 99) with ngl 16's pp512 row, rows 3-5 were never read, and nothing was printed to stdout or stderr. Worse, the trim is anchored at the window END, which for a mixed invocation is the end of DECODE — so even a correctly paired pp512 row measures decode watts: my ngl99 window reported `compute 377.28 J over 78.769 s` / 0.184 J/tok for prefill when the true prefill was 582.9 J / 0.285 J/tok, with 0% of the reported joules coming from prefill. The documented §5 command `energy_window.py run.csv --bench bench.csv --reps 3` (docs/10:169) has no --marker at all, so `wins = [rows]` makes the whole 940 s session one window joined to bench row 0 — it printed a confident `0.717261 J/tok` under a header reading `marker=idle`.

### Findings (12)

**[CERTAIN] The join is a positional zip with no length check, so one multi-row window shifts every later window onto another run's data — silently, with plausible numbers.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/energy_window.py:291`  

> energy_window.py:288-294: `# Pair the i-th window with the i-th bench row. llama-bench emits its\n# rows in run order, and the driver writes one marker window per run,\n# so the ordering is the join key.\nfor i, res in enumerate(results):\n    if i >= len(bench):\n        break\n    b = bench[i]`. Reproduced on a fixture of 3 mixed invocations (ngl 0/16/99, -p 512 -n 128) under one marker, with the 6-row bench CSV llama-bench emits. Output: `window 0 ... 408.40 s ... 512  7.00 tok/s  0.726455 J/tok` (row 0 = ngl0 pp, right run); `window 1 ... 255.80 s ... None  3.60 tok/s  1.324442 J/tok` (row 1 = ngl0 tg — window is the ngl16 run); `window 2 ... 185.60 s ... 512  14.00 tok/s  0.409053 J/tok` (row 2 = ngl16 pp — window is the ngl99 run). Bench rows 3, 4, 5 (ngl16 tg, ngl99 pp, ngl99 tg) were never read. No warning on stdout or stderr.

*Action:* Before the loop, refuse to pair when `len(bench) != len(wins)`. Exit with both counts, the marker, and the three causes: a `-p N -n M` invocation emits TWO rows for ONE window; concatenated per-case CSVs break order; a --marker that matches a different number of windows than the CSV has rows. Do not silently pair.

**[CERTAIN] For the placement question specifically, the desynced row describes an experiment that was never run: ngl 99's energy is published next to ngl 16's throughput.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/matrix.json:86`  

> Fixture window 2 is the ngl99 run (185.60 s: 13 s load, 78.77 s prefill, 93.90 s decode). It printed `512  14.00 tok/s  0.409053 J/tok  113.63 Wh/Mtok` — 14.00 t/s is ngl16's pp512 rate; the joules come from ngl99's window; the true ngl99 prefill energy is 78.77 s x 7400 mW = 582.9 J = 0.2846 J/tok, so the published figure is 44% high AND carries the wrong layer split's throughput. partial_offload_sweep is exactly this shape: `"n_gpu_layers": [0, 8, 16, 24, 99], "n_prompt": [512], "n_gen": [128]` (bench/matrix.json:90-102), 5 windows -> 10 bench rows.

*Action:* Same count guard; additionally carry the row's own n_gpu_layers/model onto the output so a shifted pairing is visible in the CSV even if a guard is bypassed.

**[CERTAIN] bench_compute_s() takes ONE row's avg_ns, and the slice it cuts is anchored at the window end — which on a mixed window is the end of decode. A pp512 row's 'prefill compute' slice can therefore lie entirely inside the decode segment.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/energy_window.py:220`  

> energy_window.py:220-239 `bench_compute_s(row, reps)` reads `ns = _f(row, "avg_ns")` from one row; energy_window.py:312-314 `if cs and 0.0 < cs < res["duration_s"]: cut = res["end_unix"] - cs; sub = [r for r in wins[i] if (_f(r, "ts_unix") or 0.0) >= cut]`. Test with each window paired to its OWN run's pp512 row (best case a positional join can give): ngl99 window printed `compute   377.28 J over 78.769 s   load 708.16 J over 107.000 s` and `512  26.00 tok/s  0.184219 J/tok`. Ground truth for that window: prefill 78.77 s / 582.9 J, decode 93.90 s / 450.7 J, load 13.00 s / 52.0 J. cs = 78.769 s is exactly the prefill duration but the trailing 78.769 s is wholly inside the 93.90 s decode tail, so the reported 'prefill' energy is 100% decode energy, 35% low (0.184 vs 0.285 J/tok). Length right, location wrong by exactly the decode duration; error = tg_s x (P_prefill - P_decode).

*Action:* Do not attempt a trailing-slice trim when the window produced more than one bench row. When exactly one row exists, the trim is sound (decode is last, prefill is the whole timed region). State in the docstring that the trim assumes the row's test is the LAST thing in the window.

**[CERTAIN] On a mixed window the load_joules/load_s columns report prefill energy as model-load energy — off by an order of magnitude.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/energy_window.py:321`  

> Pairing each window with its own tg128 row, the ngl99 window printed `compute   450.24 J over 93.902 s   load 635.2 J over 91.800 s`. The compute half is correct (truth 450.7 J / 93.90 s, decode really is the trailing segment, J/tok 1.169455 vs truth 1.1707). But the 'load' half is 13 s of real load (52.0 J) plus 78.8 s of prefill (582.9 J): 92% of the column labelled load_joules is prefill energy. docs/10_measurement_protocol.md:128 documents that column as 'the part of the window spent loading the model and starting CUDA'.

*Action:* Emit load_s/load_joules only when the window holds exactly one bench row; otherwise leave them empty alongside the per-token columns.

**[CERTAIN] The invocation printed verbatim in docs/10 §5 has no --marker, so the window is the entire CSV — idle baseline, every run, every cooldown — joined to bench row 0, and it prints a confident J/token. A length check cannot catch this because 1 == 1.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/docs/10_measurement_protocol.md:169`  

> energy_window.py:276-277 `else:\n    wins = [rows]`. Running docs/10_measurement_protocol.md:169 verbatim (`python3 telemetry/energy_window.py run.csv --bench bench.csv --reps 3 --out per_token.csv`) on the fixture gave: `window 0  marker=idle  940.20 s  4702 samples` / `module   4876.56 J` / `512  7.00 tok/s  0.717261 J/tok  199.24 Wh/Mtok` / `compute  1468.95 J over 292.571 s   load 3407.61 J over 647.800 s`. The 292.571 s trailing slice spans the ngl99 run plus part of a cooldown; the 'load' 647.8 s is most of the sweep. The only tell is the header word `marker=idle`, which summarise() takes from `win[0]` (line 135).

*Action:* Refuse --bench when neither --marker nor --from/--to was given: 'with no window selection the window is the entire CSV (idle baseline and cooldowns included); the J/token this would print is the whole session's energy over one test's tokens.' Then fix docs/10:169 to pass a real --marker.

**[CERTAIN] Concatenating per-case CSVs fails in an index-dependent way: a crash if the repeated header lands on a consumed index, silently wrong numbers if it does not.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/energy_window.py:165`  

> load_bench() is `return [r for r in csv.DictReader(fh)]` (energy_window.py:161-165) with no header-repeat filter. `cat case1.csv case2.csv` with 3 windows crashed: `File "telemetry/energy_window.py", line 301, in main / res["bench_tps"] = float(tps) if tps else None / ValueError: could not convert string to float: 'avg_ts'`. The same concatenated file with ONE window (no --marker) never reaches the header row and printed `512  7.00 tok/s  0.717261 J/tok` with no complaint. Nothing in the kit produces a multi-row bench CSV aligned to multiple windows — run_matrix.py:321 writes one `%s_%03d.csv` per run and scripts/90_collect_results.sh only tars — so hand-concatenation is the only way a user can get there.

*Action:* In load_bench(), detect rows whose values equal their own keys (repeated headers) and exit naming the count: 'bench CSV contains N repeated header lines — several per-case CSVs were concatenated. Concatenation is not a supported way to build this join; the row order must match window order.' Filtering them silently is not enough, because the order assumption is still unmet.

**[CERTAIN] The `break` at the top of the loop silently drops every window past the end of the bench CSV — no message, just missing columns.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/energy_window.py:292`  

> energy_window.py:292-293 `if i >= len(bench):\n    break`. With 3 windows and a 1-row bench CSV the table printed windows 1 and 2 with power lines and no J/tok line at all, and per_token.csv came out as: row 0 `bench_tps='7.0' jpt='0.726455'`, row 1 `bench_tps='' jpt=''`, row 2 `bench_tps='' jpt=''`. Nothing on stderr. This is the same shape as finding #7's symptom ('it looks like the join just didn\'t apply rather than a bug').

*Action:* Replace the `break` with the count guard; a short bench CSV is a mismatch, not a stopping condition.

**[LIKELY] A guard cannot match on the row's test label, because llama-bench's CSV has no `test` column — pick() falls through to n_prompt and labels prefill rows '512' and decode rows None.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/energy_window.py:300`  

> energy_window.py:300 `res["bench_test"] = pick(b, "test", "n_prompt", skip_zero=True)`. My fixture used llama-bench's real field set (no `test`); output shows `512  7.00 tok/s` for pp rows and `None  3.60 tok/s` for tg rows. The audit's sourced fact-check states get_fields() is `build_commit … n_prompt, n_gen, n_depth, test_time, avg_ns, stddev_ns, avg_ts, stddev_ts` and that `test` exists only in the markdown renderer (docs/40_audit_findings.md:98). The skip_zero fix for finding #7 corrected the token divisor but left the label reading None on every decode row.

*Action:* Derive the phase from the row itself: `pp%d` when n_prompt and not n_gen, `tg%d` when n_gen and not n_prompt, else 'mixed'. Use that both for bench_test and as the second guard: compare it to the window marker's phase prefix (the driver writes `'%s:%s:%03d' % (phase, case_id, i)`, run_matrix.py:324, so `marker.split(':')[0]` is prefill/decode/mixed; manual §5 markers are already bare phase words). Exit on disagreement; treat an unrecognised marker as 'cannot check', not as a mismatch, so --from/--to windows still work.

**[CERTAIN] --marker cannot address a driver-produced window by phase at all, and its error message blames the logger — which pushes the user onto the unguarded whole-file path.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/energy_window.py:98`  

> windows_by_marker matches exactly: `if (r.get("marker") or "").strip() == label` (energy_window.py:98). The driver writes `label = "%s:%s:%03d" % (phase, run["case_id"], i)` (run_matrix.py:324), e.g. `prefill:partial_offload_sweep:005`. Running the documented `--marker prefill` against a power CSV carrying driver labels: `no samples carry marker 'prefill' (is the logger reading the marker file?)` exit=1. The logger was working correctly. docs/10:168 and the module docstring (lines 23-25) both show the bare-phase form.

*Action:* On no match, list the distinct markers present in the file with sample counts, and say the driver writes '<phase>:<case_id>:<NNN>'. Optionally accept a prefix match on the phase segment. Either way this must be fixed alongside the whole-file guard, or users hit the guard and fall back to the unguarded command.

**[CERTAIN] The two tools compute the compute-slice differently on multi-row data, while the protocol states they are the same.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:199`  

> run_matrix.py:199-215 compute_seconds() loops `for r in rows: ... total_ns += ns * reps + warm_ns` — it SUMS every row, which is correct for the total timed seconds of a mixed invocation. energy_window.py:220-239 bench_compute_s() reads a single row's avg_ns. docs/10_measurement_protocol.md:177-178 claims '`energy_window.py` applies the same `compute_s` trim as the driver when the bench CSV carries `avg_ns`'. They agree only when the CSV has one row. On my fixture window 0, the driver's arithmetic gives 292.57 + 106.94 = 399.5 s against a 408.4 s window, while energy_window used 292.571 s.

*Action:* Once the one-row-per-window rule is enforced the doc claim becomes true; until then either note the divergence or make bench_compute_s refuse multi-row input. The summed-vs-single difference is also a cheap detector: if the bench CSV's total compute_s nearly fills the window but the paired row's does not, the window holds more tests than the row describes.

**[CERTAIN] After the matrix split the desync is unreachable from the normal path, but fully reachable from the documented manual path — so a hard exit costs nothing and buys the case that matters.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/docs/10_measurement_protocol.md:153`  

> With the split (`"n_prompt":[512],"n_gen":[0]` and `"n_prompt":[0],"n_gen":[128]`), build_cmd emits `-p 512 -n 0` or `-p 0 -n 128`; llama-bench skips the zero axis, so each invocation is one test and each `<base>_%03d.csv` has exactly one row against exactly one window — 1:1, guard never trips. Nothing in scripts/ or the Makefile invokes energy_window.py (grep over *.sh and Makefile: no hits), so every use of it is manual, per docs/10 §5, where the natural manual command is `llama-bench -p 512 -n 128` — two rows, one window. §5 is also where the user has no driver to cross-check against.

*Action:* Hard exit on count mismatch and on phase disagreement. Carve out ONE exception: a single window with N>1 rows (the manual mixed run) should degrade rather than die — print duration, joules, mean/peak power, list each row's phase and t/s, omit joules_per_token / wh_per_mtok / compute_* / load_* entirely, and print to stderr: 'window <marker> produced N bench rows (pp512, tg128): one window cannot be attributed to two phases. Reporting joules only. Re-run the prompt-only and generation-only passes in separate marker windows for energy per token.' That mirrors the audit's own rule for run_matrix — write joules, leave the per-token columns empty.

**[CERTAIN] run_matrix.py already records the evidence needed for the same guard on its side but does not act on it.**  
`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:431`  

> run_matrix.py:376 writes `"bench_rows": len(bench_rows)` onto every record, and bench_tps()'s own docstring says 'Mixed pp+tg invocations emit two rows and this keeps the last, which is why matrix cases should be phase-separated' (run_matrix.py:174-176). The join at 431 still writes `rec["joules_per_token"]` regardless of bench_rows.

*Action:* Gate the per-token block on `rec.get("bench_rows") == 1` in run_matrix.py's join, so both tools refuse the same input for the same reason; leave joules, duration_s, mean_mw on the row.

### Risks a naive fix runs
- A `len(wins) != len(bench)` check alone gives false assurance: the worst case I reproduced — no --marker, whole file as one window, one bench row — passes it 1 == 1 and still prints 0.717 J/tok from 940 s of session energy. The count guard and the window-selection guard are both required; neither substitutes for the other.
- Hard-exiting on every multi-row bench CSV would refuse data a student has already collected on a manual `-p 512 -n 128` run, with no path forward and a real temptation to comment the guard out. Keep the single-window/multi-row case as a degraded report (joules yes, per-token no) rather than an exit.
- Matching the window marker's phase against the row's phase must treat an unknown marker as 'cannot check'. --from/--to windows and the whole-file window take their marker from `win[0]` (summarise, line 135), which is usually 'idle' — a strict comparison would reject legitimate ad-hoc windows.
- Silently filtering repeated header rows out of a concatenated CSV would convert the current crash into silent wrong numbers, which is worse: the row ORDER is still unverified after filtering. Detect and refuse, do not clean up.
- Changing bench_test from the n_prompt fallback to a derived 'pp512'/'tg128' changes the values in an existing output column (today: '512' and None). Any notebook keyed on that column needs updating at the same time — and the existing None means no decode row in any per_token.csv produced so far is labelled at all.
- --bench does not verify it was handed a llama-bench CSV. The driver's own `<base>_bench_energy.csv` carries n_prompt, n_gen and avg_ts, so it parses cleanly, and because it has no avg_ns the trim silently never applies (energy_basis stays 'whole-window (includes model load)'). It half-works, which is the most dangerous outcome. Require avg_ns (or a known llama-bench column signature) before accepting the file.
- Splitting the matrix cases changes run indices, so marker labels like 'prefill:partial_offload_sweep:005' shift. Any previously collected power CSV analysed with a hand-written label will silently select a different window; note the split in docs/30_experiment_log.md so old and new sweeps are not mixed.
- The fixture numbers here are synthetic (power levels and t/s are invented, structured after the real phase ordering). The ORDERING conclusions are structural and hold regardless; the specific 35% and 44% error magnitudes depend on the prefill/decode power gap and should be restated from real hardware before they go in the thesis.

---

## Lens: `downstream-consumers`

Splitting the two mixed cases moves the enabled run count from 23 to 30 and renames two case ids; emitting one record per bench row breaks the kit's remaining "one run = one row = one _NNN.csv" assumptions in the driver's summary line, the energy join, energy_window.py's positional window/row join, and six doc locations. The biggest gap in the audit's own fix is that largest_local_model is also a mixed case and is not named in it; the second is that a literal "one record per bench row" deletes failed runs (zero bench rows means zero records) and duplicates one window's joules onto every record of that window. Already-stale items sitting in the blast radius: bench_test resolves to None on every decode row, the audit doc's run_matrix.py:272 anchors point at unrelated code, and _rows() re-parses the whole power CSV once per record.

### Findings (49)

**[CERTAIN] largest_local_model is also a mixed case; the audit's fix text does not name it, so the mixed path survives and returns the moment the case is enabled.**  
`bench/matrix.json:140`  

> matrix.json:140-155 'id': 'largest_local_model', 'enabled': false, 'n_prompt': [512], 'n_gen': [128], 'repetitions': 2. Verified by running expand()/phase_of() over the matrix: largest_local_model enabled=False runs=2 phases=['mixed']. Audit line 656 names it; the fix at line 366 drops it.

*Action:* Split largest_local_model into prefill-only and decode-only halves in this same edit, both enabled:false. Otherwise finding #3 is 'fixed' with a live mixed case still in the file.

**[CERTAIN] The _comment header states the one-invocation-per-case model the split is abandoning, and never states the invariant the split exists to create.**  
`bench/matrix.json:4`  

> matrix.json:4-5 'Each case becomes one llama-bench invocation, wrapped in its own power window, so every throughput number has joules attached to it.'

*Action:* Rewrite to: each case expands to one llama-bench invocation per point of its axis product, and every invocation must exercise exactly ONE phase. A case with both n_prompt and n_gen non-zero becomes two llama-bench tests inside one power window, and that window's joules cannot be attributed to either.

**[CERTAIN] 'Edit freely' invites re-introducing a mixed case, and nothing validates the invariant.**  
`bench/matrix.json:16`  

> matrix.json:16 "Edit freely. 'enabled': false skips a case without deleting it."

*Action:* Append: never give one case a non-zero n_prompt AND a non-zero n_gen; put the prefill and the decode in separate cases. State that 'make check' enforces it.

**[CERTAIN] The per-case 'note' string is copied verbatim onto every output record, so the split produces two rows carrying a note that describes a mixed measurement.**  
`bench/matrix.json:88`  

> run_matrix.py:107 '"note": merged.get("note", "")' inside the run dict, and records.append({**run, ...}) at line 373 puts it in the CSV. matrix.json:88 note is 'The placement question in miniature: how many layers should live on the GPU when RAM is shared with the CPU?'

*Action:* Give each half its own note, e.g. '...placement question in miniature -- PREFILL half (512-token prompt, no generation); pair with the decode half'. Same for quantisation_sweep's note at matrix.json:107.

**[CERTAIN] Renaming the case ids breaks any --only invocation using the old ids, including the audit doc's own failure scenario.**  
`bench/matrix.json:86`  

> docs/40_audit_findings.md:659 './scripts/50_bench_device.sh --only partial_offload_sweep produces 5 rows'. run_matrix.py:270-271 'if not runs: sys.exit("no cases selected")'.

*Action:* Either keep the id 'partial_offload_sweep' on both halves (so --only still selects both and case_id still groups them) or pick partial_offload_prefill/partial_offload_decode and accept that old --only lines exit loudly. Decide once and record the choice; do not ship two ids differing by a character the reader may miss.

**[CERTAIN] The final summary line counts records, which stops being the number of runs once one record is emitted per bench row.**  
`bench/run_matrix.py:452`  

> run_matrix.py:452-453 'ok = sum(1 for r in records if r.get("status") == "ok")' then 'print("%d/%d runs ok" % (ok, len(records)))'.

*Action:* Count distinct runs (unique rec['marker']) for this line, not records. Otherwise the sweep reports a runs-ok figure that is really a rows figure.

**[CERTAIN] With N records per window, energy_between() is called once per record over the same interval, so identical joules land on every record of that window; summing the joules column then double-counts sweep energy.**  
`bench/run_matrix.py:399`  

> run_matrix.py:399 'e = energy_between(power_csv, rec["t_start_unix"], rec["t_end_unix"]); rec.update(e)' inside 'for rec in records'. t_start_unix/t_end_unix come from the run (lines 358, 374), not the bench row.

*Action:* Add a window_id column (the marker is already unique per run) and either write the window-scoped columns only on the first record of a window or document on the row that they repeat. Say so in the protocol's column table so nobody sums joules.

**[CERTAIN] A literal 'one record per bench row' deletes failed and unparseable runs from the output, because those produce zero bench rows.**  
`bench/run_matrix.py:366`  

> run_matrix.py:352-361 the returncode!=0 path appends a record with status='failed' and no bench data; lines 366-371 'bench_rows = list(csv.DictReader(...)) if csv_text else []' and 'if not bench_rows: print("    !! no llama-bench CSV header in stdout")' still fall through to records.append at 373.

*Action:* Keep the invariant 'at least one record per run': when bench_rows is empty emit exactly one record with a status and blank bench columns. docs/10_measurement_protocol.md:119-120 already promises a status=failed row and must stay true.

**[CERTAIN] The trailing-compute_s trim is valid only for the LAST bench row of a window; applied per record it would charge the prefill row a slice of the decode test's energy.**  
`bench/run_matrix.py:410`  

> run_matrix.py:404-407 comment 'All of it strictly PRECEDES the first timed loop, so the trailing compute_s seconds of the window are the part that did the tokens', and line 413 'ce = energy_between(power_csv, rec["t_end_unix"] - cs, rec["t_end_unix"])'.

*Action:* Compute compute_joules only when the window produced exactly one bench row. For a multi-row window leave compute_s/compute_joules/load_s/load_joules blank and set energy_basis to e.g. 'unattributable (2 tests in one window)'. The refusal must cover the compute columns, not just the per-token ones.

**[CERTAIN] compute_seconds() sums avg_ns across ALL rows of an invocation, which is the right window-level number but the wrong per-record number.**  
`bench/run_matrix.py:188`  

> run_matrix.py:198-215 'total_ns = 0.0 ... for r in rows: ... total_ns += ns * reps + warm_ns', called at line 377 as compute_seconds(bench_rows, run['repetitions']).

*Action:* Keep compute_seconds(rows, reps) for the window level and add a per-row variant identical to telemetry/energy_window.py bench_compute_s (line 220). Per-record compute_s must come from that row's own avg_ns.

**[CERTAIN] bench_tps()'s last-row-wins behaviour and its docstring are the artefact finding #3 describes; both go stale once each record carries its own row's avg_ts.**  
`bench/run_matrix.py:174`  

> run_matrix.py:174-176 'avg_ts off the parsed rows. Mixed pp+tg invocations emit two rows and this keeps the last, which is why matrix cases should be phase-separated.'

*Action:* Take avg_ts from each bench row directly. If bench_tps() is kept for the console line, rewrite the docstring to say it is meaningful only for a single-row window. It cannot simply be deleted without breaking the documented verification snippet (see the handoff finding).

**[CERTAIN] tokens_of(rec) reads the run's requested n_prompt/n_gen, not the bench row's; per-row records must divide by the row's own counts and the two sources can disagree silently.**  
`bench/run_matrix.py:517`  

> run_matrix.py:530-532 'n_p = rec.get("n_prompt") or 0; n_g = rec.get("n_gen") or 0; reps = rec.get("repetitions") or 1'. Records are built as {**run, ...} at line 373, so these are the matrix's values.

*Action:* Overwrite n_prompt/n_gen on the record from the bench row (llama-bench is authoritative for what it ran) and keep the requested values as req_n_prompt/req_n_gen so a clamp or ref difference is visible. Update the docstring, which must stay in step with energy_window.bench_tokens().

**[CERTAIN] The _NNN suffix is a run index from enumerate(runs,1) and skipped runs consume an index, so it was never a record index; per-row records make it a one-to-many mapping with no way to say which row.**  
`bench/run_matrix.py:321`  

> run_matrix.py:315 'for i, run in enumerate(runs, 1)', line 318 the model-missing branch continues without appending a record, line 321 'case_csv = os.path.join(out_dir, "%s_%03d.csv" % (base, i))', line 375 '"bench_csv": os.path.basename(case_csv)'.

*Action:* Add a bench_row_index column (0-based position inside that file) alongside bench_csv, and state in the docs that NNN is the run index and that several output rows may share one bench_csv. Never describe NNN as a row number.

**[CERTAIN] The marker label is composite, so slicing the sweep's power CSV by phase -- exactly what the split enables -- never matches energy_window.py's exact-equality marker test.**  
`bench/run_matrix.py:324`  

> run_matrix.py:324 'label = "%s:%s:%03d" % (phase, run["case_id"], i)'; telemetry/energy_window.py:98 'if (r.get("marker") or "").strip() == label'. So 'energy_window.py sweep_power.csv --marker prefill' exits with 'no samples carry marker'.

*Action:* Either make --marker accept a prefix/regex so '--marker prefill' selects every prefill window of a sweep, or document the exact label form in docs/10_measurement_protocol.md section 5 beside the --marker examples. Phase-separated windows are only useful if they can be selected by phase.

**[CERTAIN] The per-run console line prints one throughput number, which is wrong for any window that still emits two rows.**  
`bench/run_matrix.py:380`  

> run_matrix.py:380 'print("    %.2f t/s, %.1f s" % (tps or 0.0, t_end - t_start))'.

*Action:* Print one line per bench row (test label plus t/s), and print a visible warning when bench_rows > 1 saying the per-token columns will be blank for that window.

**[CERTAIN] The module docstring describes one row per case, which is what this change ends.**  
`bench/run_matrix.py:4`  

> run_matrix.py:4-5 'write one tidy CSV that has throughput and energy on the same row'; line 7 'What it does per case'; line 11 'runs llama-bench with -o csv for that case'.

*Action:* Rewrite to: one output row per llama-bench test row, i.e. one row per phase; each marker window covers exactly one phase because the matrix never mixes n_prompt and n_gen in one case. Change 'per case' to 'per run' throughout -- a case expands to many runs and the distinction becomes load-bearing after the split.

**[CERTAIN] phase_of() still needs its 'mixed' branch after the split, but nothing says what a mixed window now means for the output.**  
`bench/run_matrix.py:120`  

> run_matrix.py:120-127 returns 'prefill' / 'decode' / 'mixed'. matrix.json invites free editing, so a user-authored mixed case stays reachable.

*Action:* Keep the branch and comment it: a mixed window is a user-authored case the shipped matrix never produces; its row gets joules but no per-token columns. Warn at expand()/dry-run time when any selected run is mixed, so it is caught before two hours are spent rather than after.

**[CERTAIN] Already stale and made worse by this edit: _rows() re-parses the entire power CSV on every energy_between() call, so cost scales with record count.**  
`bench/run_matrix.py:456`  

> run_matrix.py:456-458 '_rows(power_csv): with open(power_csv) as fh: return list(csv.DictReader(fh))', called at 476 in energy_between and 462 in idle_floor. The join calls energy_between twice per record (lines 399 and 413). A 2 h sweep at 5 Hz is ~36k rows; that is 1 + 2*23 = 47 full parses today, 61 after the split, more with per-row records.

*Action:* Parse the power CSV once and pass the rows to idle_floor/energy_between. Cheap sweep-up while the join is already being rewritten.

**[CERTAIN] The window-to-bench-row join is positional and documented as 'one marker window per run'; it silently drops surplus bench rows, which is exactly the mixed-case shape.**  
`telemetry/energy_window.py:288`  

> energy_window.py:288-294 'Pair the i-th window with the i-th bench row. llama-bench emits its rows in run order, and the driver writes one marker window per run, so the ordering is the join key.' then 'for i, res in enumerate(results): if i >= len(bench): break'. Iterating over results means a 2-row bench CSV against 1 window uses row 0 (pp) and discards row 1 (tg) with no message.

*Action:* Add a loud mismatch check before the loop: if len(bench) != len(results), warn naming both counts. Update the comment to say each window must contain exactly one bench row, which the matrix split now guarantees for driver-produced data.

**[CERTAIN] Already stale from last session: bench_test is None on every decode row, because llama-bench's CSV has no 'test' column and skip_zero now rejects the n_prompt fallback.**  
`telemetry/energy_window.py:300`  

> energy_window.py:300 'res["bench_test"] = pick(b, "test", "n_prompt", skip_zero=True)'. pick() (lines 168-186) returns None when no candidate survives; a tg row has no 'test' key and n_prompt == '0', which skip_zero skips. Audit fact-check line 98: 'There is NO test column in the CSV -- get_fields() is build_commit ... n_prompt, n_gen, n_depth, test_time, avg_ns, stddev_ns, avg_ts, stddev_ts; test only exists in the markdown renderer.' The print at line 366 then shows '?' for every decode window.

*Action:* Derive the label from the row: 'pp%g' % n_prompt when n_prompt, else 'tg%g' % n_gen. This is the same 'which phase is this row' question finding #3 is about, so fix it here too; the driver's per-row records need the identical derived label as a phase/test column.

**[CERTAIN] windows_by_marker's docstring claims repetitions produce one window each, which is false for driver-produced CSVs and misleads anyone slicing a sweep by phase.**  
`telemetry/energy_window.py:92`  

> energy_window.py:93-95 'A benchmark that repeats a phase produces one window per repetition, which is what you want: the variance across repetitions is itself a result.' The driver sets the marker once per invocation (run_matrix.py:335 set_marker(label) before subprocess.run, 346 set_marker('idle') after), so all -r repetitions live inside one contiguous window.

*Action:* Correct to: one contiguous window per marker value. For a driver sweep that is one window per llama-bench invocation with all -r repetitions inside it -- which is why --reps exists and why per-repetition variance is not visible in these windows.

**[CERTAIN] The docstring's --marker examples work only for hand-written ad-hoc markers, not for a sweep power CSV.**  
`telemetry/energy_window.py:23`  

> energy_window.py:23-25 './energy_window.py power.csv --marker decode' and '--marker prefill --idle-mw 1900'; the driver writes labels like 'decode:gpu_decode_sweep:006' (run_matrix.py:324).

*Action:* Say which CSV each example applies to and add one driver example using the composite label (or the prefix form, if added).

**[CERTAIN] Minor and pre-existing: the --reps warning quotes one ratio that is only right for prefill.**  
`telemetry/energy_window.py:285`  

> energy_window.py:285-287 'note: --reps not given, assuming 1. bench/matrix.json defaults to repetitions=3 -- pass --reps 3 or J/token is ~4x too low.' At r=3 a prefill row is 2048/512 = 4.0x, a decode row 385/129 = 3.0x.

*Action:* Reword to 'roughly r+1x for a prefill row and r x for a decode row'. One-line touch while the file is open for bench_test.

**[CERTAIN] The 'Reading the output' table's avg_ts row is the exact line the audit says produces the wrong reading, and it still says 'for the case'.**  
`docs/10_measurement_protocol.md:126`  

> docs/10_measurement_protocol.md:126 '| `avg_ts` | llama-bench tokens/s for the case |'. Audit line 656: 'Anyone reading <base>_bench_energy.csv per docs/10_measurement_protocol.md section "Reading the output" ("avg_ts | llama-bench tokens/s for the case") will pair a decode rate with a mixed-phase energy figure.'

*Action:* Change to: '| avg_ts | llama-bench tokens/s for THIS row's test (pp<N> or tg<M>) -- one row per phase, never a fused number |'.

**[CERTAIN] The column table omits every column a reader now needs to tell one row from another, including ones added last session.**  
`docs/10_measurement_protocol.md:124`  

> docs/10_measurement_protocol.md:124-134 lists avg_ts, joules, load_s/load_joules, compute_s/compute_joules, energy_basis, the per-token columns, tokens, temp_rise_c, swap_peak_mb and nothing else. The driver writes n_prompt, n_gen, bench_rows, bench_csv, precached, idle_mw, mean_mw, power_samples, status, rc, tag (run_matrix.py:373-379, 437).

*Action:* Add rows for: phase (prefill/decode, from the marker), n_prompt/n_gen (this row's test, per repetition), bench_rows (tests in this window; >1 means the per-token columns are blank by design), bench_csv + bench_row_index, window_id (rows sharing a window repeat its joules -- do not sum), precached, status. The audit's own fact-check correction (line 101) already asks for n_prompt/n_gen here.

**[CERTAIN] The tokens formula implies both terms are always present, which stops being true once every row is single-phase.**  
`docs/10_measurement_protocol.md:132`  

> docs/10_measurement_protocol.md:132 '| `tokens` | the divisor: `n_prompt*(r+1) + n_gen*r + 1` -- see below |'.

*Action:* State it per phase: a prefill row is n_prompt*(r+1); a decode row is n_gen*r + 1; no row carries both any more, which is the point of the split.

**[CERTAIN] Section 4's step list describes one power window per case and cooldown between cases; both are per-run, and the run count is what the split changes.**  
`docs/10_measurement_protocol.md:114`  

> docs/10_measurement_protocol.md:113-117 '2. runs each case from bench/matrix.json, marking its power window, 3. cools down between cases so heat does not carry over, 4. joins throughput to energy and writes results/<node>_<stamp>_<tag>_bench_energy.csv'. run_matrix.py:381 'time.sleep(run["cooldown_s"])' is inside the per-run loop.

*Action:* Rewrite: 2. runs every point of every enabled case (23 runs today, 30 after the phase split), each in its own power window covering exactly one phase; 3. cools down between runs; 4. joins throughput to energy and writes one row per llama-bench test row.

**[CERTAIN] The failed-case sentence must survive the per-row rewrite, and is what a naive 'one record per bench row' implementation breaks.**  
`docs/10_measurement_protocol.md:119`  

> docs/10_measurement_protocol.md:119-120 'A failed case (an OOM on 4 GB is normal for the larger models) is recorded as a row with status=failed and the sweep continues.'

*Action:* Keep the promise and make it explicit: every run produces at least one row; a run that failed or emitted no parseable CSV produces exactly one row with status set and blank bench columns.

**[CERTAIN] 'Why the window is trimmed' does not cover the new refusal case, only the missing-avg_ns case.**  
`docs/10_measurement_protocol.md:136`  

> docs/10_measurement_protocol.md:136-144 ends 'If a ref does not emit avg_ns the trim cannot be made and energy_basis says whole-process (includes model load) -- treat those rows as an upper bound, not as energy per token.'

*Action:* Add the second case: if a window produced more than one llama-bench test (bench_rows > 1) the trim and the per-token columns are refused outright -- that window's joules cannot be split between a prefill test and a decode test, so the row carries joules only. The shipped matrix never produces such a window; seeing one means a case was edited to mix phases.

**[CERTAIN] Section 4 nowhere states the methodological rule the split establishes, so the reason the matrix looks the way it does would live only in a JSON comment.**  
`docs/10_measurement_protocol.md:103`  

> docs/10_measurement_protocol.md:103-151 covers the driver steps, the column table, the window trim and the token count, but never the one-phase-per-window invariant.

*Action:* Add a paragraph to section 4: why the matrix never mixes n_prompt and n_gen in one case -- llama-bench runs -p 512 -n 128 as two independent tests inside one marker window, so the joules are prefill+decode while the throughput is one test's; the matrix therefore ships a prefill case and a decode case for every placement and quantisation point. This is the paragraph the thesis methodology section gets written from.

**[CERTAIN] Section 5's ad-hoc recipe gives no guidance for the sweep's own CSV, and its --bench example does not mention the one-row-per-window requirement.**  
`docs/10_measurement_protocol.md:168`  

> docs/10_measurement_protocol.md:160-169 'echo prefill > /tmp/hs_marker' ... 'python3 telemetry/energy_window.py run.csv --marker prefill --idle-mw 1900' and '--bench bench.csv --reps 3 --out per_token.csv'.

*Action:* Add: against a sweep's power CSV the marker labels are phase:case_id:NNN, so pass the full label (or the prefix form if added); and the --bench CSV must have exactly one row per window -- each results/<base>_NNN.csv has one row after the phase split, and a two-row file means the window mixed phases and its per-token numbers must not be used.

**[CERTAIN] Section 6's provenance bullet is true, but the bundle carries the current matrix.json, which after the id rename cannot reproduce older results.**  
`docs/10_measurement_protocol.md:186`  

> docs/10_measurement_protocol.md:186-188 'Keep the raw CSVs. 90_collect_results.sh bundles them with the node facts, the build provenance, the model checksums and the matrix that produced them'. scripts/90_collect_results.sh:20 'cp "${HS_JETSON_DIR}/bench/matrix.json" "${D}/matrix.json"' copies whatever is on disk at collect time.

*Action:* Add a line: bundle immediately after a sweep -- the matrix.json in the tarball is the one on disk now, not the one that ran. Note that case ids changed with the phase split, so rows from before and after cannot be concatenated by case_id. results/ currently holds only .gitkeep, so nothing is lost today; this is forward-looking.

**[LIKELY] The README's sweep duration is derived from the run count, which rises from 23 to 30.**  
`README.md:57`  

> README.md:57 'make bench                       # single-node throughput + energy sweep (1.5-3 h)'. Counted with expand(): 4+2+4+2+5+2+4 = 23 enabled runs today; the split makes partial_offload 10 and quantisation 4, so 30. Each new run adds a full model load (1.1 GB for qwen2.5-1.5b, ~530 MB for q8_0) plus a 20 s cooldown.

*Action:* Update the estimate (2-3.5 h is the safe restatement) and say it is about 30 runs. The same number appears in scripts/50_bench_device.sh:48.

**[CERTAIN] The layout blurb describes the driver's output as one row per case.**  
`README.md:89`  

> README.md:89 'run_matrix.py            driver: throughput and watts on the same row'.

*Action:* Change to 'driver: one row per phase, throughput and watts on the same row'.

**[CERTAIN] 'Measurement honesty' lists four commitments and omits the one this edit establishes, which is the study's central methodological claim.**  
`README.md:111`  

> README.md:111-128 covers POM_5V_IN scope, no external meter, reporting energy twice, jetson_clocks, and wired/Wi-Fi tagging. Nothing about phase separation.

*Action:* Add a bullet: no power window ever contains more than one phase. The matrix ships prefill-only and decode-only cases; a window that produced two llama-bench tests gets joules but no joules-per-token, because that energy cannot be attributed to either phase.

**[LIKELY] The sweep duration comment carries the same run-count-derived estimate as the README.**  
`scripts/50_bench_device.sh:48`  

> scripts/50_bench_device.sh:48 '# The full matrix is roughly 1.5-3 h on a Nano including cooldowns. Run it under' (line number after the rpc-server pgrep edit that landed mid-analysis).

*Action:* Update alongside README.md:57 and keep the two in agreement. Consider stating the run count so the next matrix edit makes the drift obvious.

**[CERTAIN] The --only example survives the split (gpu_decode_sweep is not split), but there is no pre-flight validation of the phase invariant before two hours are spent.**  
`scripts/50_bench_device.sh:8`  

> scripts/50_bench_device.sh:8 '#   ./50_bench_device.sh --only gpu_decode_sweep'; the pre-flight block checks power mode, clocks, temperature, the rpc server, free memory and model count -- nothing about the matrix.

*Action:* No change needed to line 8. Consider adding the matrix invariant check to the pre-flight, or rely on 'make check'; today a mixed case is discovered only when the joined CSV is read afterwards.

**[CERTAIN] 'make check' validates that matrix.json is JSON and nothing else, so the invariant the whole fix rests on is unenforced in a file whose own comment says 'Edit freely'.**  
`Makefile:47`  

> Makefile:47 '@python3 -c "import json;json.load(open(0x27bench/matrix.json0x27));print(0x27matrix.json ok0x27)"' (quotes elided).

*Action:* Extend the check to assert, for every case whether enabled or not: no case has both a non-zero n_prompt and a non-zero n_gen, and case ids are unique. This is the cheapest place to keep finding #3 fixed.

**[CERTAIN] The bundle listing truncates at 40 entries; the split pushes the archive past that, so the operator's confirmation of what was collected silently stops being complete.**  
`scripts/90_collect_results.sh:30`  

> scripts/90_collect_results.sh:30 'tar -tzf "$OUT" | sed 0x27s/^/  /0x27 | head -40'. Contents: node_facts.json, build_info.json, node.env.snapshot, matrix.json, models.sha256, kernel_warnings.txt, two directory entries, plus per sweep one power.csv, one bench_energy.csv, one run.log, one facts json and one _NNN.csv per run -- 23 today (~36 entries), 30 after the split (~43).

*Action:* Raise to head -60 or drop the head and print a count line. Line 24's find-based copy needs no change.

**[CERTAIN] Verified NOT stale: the bundling logic copies by glob and is unaffected by run count or row count.**  
`scripts/90_collect_results.sh:24`  

> scripts/90_collect_results.sh:24 'find "$RESULTS_DIR" -maxdepth 1 -type f ! -name 0x27.marker_*0x27 -exec cp {} "${D}/results/" \;'.

*Action:* none

**[CERTAIN] The log template's result block assumes one row per case for both throughput and energy; after the split the two phases come from two different rows.**  
`docs/30_experiment_log.md:33`  

> docs/30_experiment_log.md:33-35 'tokens/s        : prefill ......   decode ......' / 'J/token         : total ......     active ......' / 'Wh per Mtok     :'.

*Action:* Split the energy lines by phase to match the throughput line: 'J/token (prefill): total ... active ...' and 'J/token (decode): total ... active ...', and note that the prefill and decode numbers for one placement point come from two rows of the same sweep (two invocations, two power windows).

**[CERTAIN] The Artefacts block names a single 'Bench CSV', which was already ambiguous and is now one joined file plus ~30 per-run files.**  
`docs/30_experiment_log.md:29`  

> docs/30_experiment_log.md:28-30 'Power CSV       : results/...' / 'Bench CSV       : results/...' / 'Run log         : results/..._run.log'.

*Action:* Name them: 'Joined CSV : results/..._bench_energy.csv' and 'Per-run bench CSVs : results/..._NNN.csv (one per run; NNN is the run index)'.

**[CERTAIN] The anomaly checklist has no entry for the failure signal this change introduces.**  
`docs/30_experiment_log.md:56`  

> docs/30_experiment_log.md:56-62 lists temp_rise_c above ~8 C, any status=failed row, swap_peak_mb > 0, repetitions disagreeing, a link that did not negotiate 1 Gbit, and kernel_warnings.txt.

*Action:* Add: 'any row with bench_rows > 1 -- its per-token columns are blank by design because the window mixed two phases; the matrix was edited' and 'a placement or quantisation point whose prefill half completed but whose decode half did not (or vice versa) -- the pair is the result, not either half'.

**[CERTAIN] Already stale: every file:line anchor in the summary table points at pre-fix line numbers; run_matrix.py:272, cited for both #3 and #20, is now the no-cases-selected exit.**  
`docs/40_audit_findings.md:294`  

> docs/40_audit_findings.md:294 '| 3 | critical | `bench/run_matrix.py:272` | ...' and line 311 '| 20 | medium | `bench/run_matrix.py:272` | ...'. bench/run_matrix.py:270-271 now reads 'if not runs: sys.exit("no cases selected")'; the avg_ts extraction those findings describe no longer exists.

*Action:* When marking #3 and #20 addressed, add a one-line status note under each (the doc currently carries no fixed markers at all, not even for #1/#2/#7/#8/#9/#11/#15) saying which session fixed it and that the cited line numbers are pre-fix. Do not renumber the anchors.

**[CERTAIN] Finding #20 proposes an alternative design (pp_ts / tg_ts columns on one row) that this change rejects; leaving it unannotated invites a later session to implement both.**  
`docs/40_audit_findings.md:662`  

> docs/40_audit_findings.md:662 'Fix. Parse all rows and keep them separately instead of collapsing: build per_test = [...] and emit pp_ts and tg_ts as distinct columns (leaving avg_ts set only when exactly one test ran). Better still for the join, split mixed cases in expand() into two runs.'

*Action:* Record that the chosen design is the matrix split plus one record per bench row, and that side-by-side pp_ts/tg_ts columns were deliberately not taken -- they would put two phases back on one row with one joules figure, which is the defect itself.

**[CERTAIN] Already stale: the audit's own example row and run counts do not match the matrix.**  
`docs/40_audit_findings.md:335`  

> docs/40_audit_findings.md:335 'the avg_ts column ... is empty for all ~26 rows'; lines 670 and 673 'the final line still says 26/26 runs ok'. Measured with expand(): 23 enabled runs today, 30 after the split.

*Action:* Low priority, but while the doc is open for #3/#20, note that the enabled matrix produces 23 runs (30 after the phase split) and that '26' in those scenarios is illustrative.

**[CERTAIN] The fact-checked correction for the llama-bench claim asks for a docs change this edit should complete, not just a code change.**  
`docs/40_audit_findings.md:101`  

> docs/40_audit_findings.md:101 'docs/10_measurement_protocol.md: add n_prompt/n_gen to the column table and state that reported energy includes llama-bench0x27s warm-up pass.' The column table (lines 124-134) still has neither, though the warm-up is explained in prose at 146-151.

*Action:* Add n_prompt/n_gen to the table as part of this edit -- they are now the row's own bench-reported per-repetition values, which is exactly what a per-row record needs to be readable.

**[CERTAIN] The documented verification snippet imports bench_tps and calls compute_seconds with a list of rows; removing or re-signaturing either breaks the only off-hardware check the kit has.**  
`.handoff/2026-09-12-1541-apply-audit-findings.md:372`  

> handoff line 372 'from run_matrix import bench_csv_text, bench_tps, compute_seconds, tokens_of', line 377 'assert bench_tps(list(csv.DictReader(bench_csv_text(banner).splitlines()))) == 10.0', and 'assert compute_seconds([{"avg_ns":"1000000000","n_prompt":"0","n_gen":"128"}], 3) == 3.008'.

*Action:* Do not edit the historical handoff. Either keep bench_tps and compute_seconds(rows, reps) intact (adding a per-row helper alongside) or carry a corrected snippet into the new handoff and mark the old one superseded. Also update the fake-Jetson fixture recipe: it must now emit a two-row CSV for at least one window to exercise the refusal guard.

**[CERTAIN] The synthetic fixture writes plain prefill/decode markers, so it models the ad-hoc flow only and cannot exercise the driver's composite labels or the new multi-row join.**  
`telemetry/make_synthetic_power_csv.py:29`  

> make_synthetic_power_csv.py:31-34 PHASES entries ('idle', ...), ('prefill', 6.0, 7400.0, ...), ('decode', 12.0, 4800.0, ...), and line 10 './energy_window.py /tmp/power.csv --marker decode --idle-mw 1900'.

*Action:* Optional but cheap: add a flag or second PHASES table producing driver-shaped labels like 'prefill:partial_offload_prefill:005', so the marker-selection path the split depends on is testable off-hardware. At minimum note in the docstring that these are ad-hoc-style markers.

### Risks a naive fix runs
- Splitting the mixed cases does NOT change what is physically measured: llama-bench already ran -p 512 -n 128 as two independent tests (pp512 and tg128), so the split only costs one extra model load and one extra cooldown per point. Anyone later 'simplifying' it to a fused -pg 512,128 test would reintroduce exactly the one-window-two-phases defect this fixes.
- If the study actually wants decode measured at a 512-token context depth (which '-p 512 -n 128' superficially suggests), neither the old mixed invocation nor the new decode-only case delivers that -- both measure tg128 from a near-empty KV cache. The right knob is llama-bench's -d/--n-depth (the CSV already carries an n_depth column per the audit's fact-check), and the thesis must not claim depth-512 decode from a depth-0 measurement.
- A literal 'one record per bench row' loop deletes failed and no-CSV runs from the output, because those windows produce zero bench rows. The invariant must be 'at least one record per run'.
- With N records sharing one window, the window-scoped columns (joules, duration_s, mean_mw, peak_mw, power_samples, temp_*, swap_peak_mb, idle_mw) are duplicated on every record; summing joules to get total sweep energy then double-counts. Needs a window_id and an explicit note in the protocol's column table.
- The trailing-compute_s trim is valid only for the LAST bench row of a window. Applying it per record in a multi-row window charges the prefill row a slice of the decode test's energy -- a subtler version of the same defect.
- Renaming the case ids breaks --only lines in notes and makes rows from before and after the split non-concatenable by case_id. results/ currently holds only .gitkeep, so nothing is lost today, but the decision belongs in the experiment log.
- largest_local_model stays mixed unless it is split too. If 'make check' gains the no-mixed-case assertion (recommended), leaving it unsplit makes 'make check' fail on a clean tree.
- Taking n_prompt/n_gen from the bench row instead of the matrix silently discards the requested values; if a ref clamps or reinterprets them the divisor changes with no record of the disagreement. Keep both.
- Removing bench_tps() or changing compute_seconds()'s signature breaks the verification snippet in .handoff/2026-09-12-1541-apply-audit-findings.md, the only documented off-hardware check of the driver.
- Phase-separated windows are only useful if they can be selected by phase, and the driver's composite marker (phase:case_id:NNN) plus energy_window.py's exact-equality match means '--marker prefill' returns nothing. Fixing the matrix without fixing that leaves the benefit unreachable from the analysis tool.
- The run count grows 23 -> 30, adding roughly 7 extra model loads (including five of the 1.1 GB qwen2.5-1.5b) and 7 extra 20 s cooldowns. Two documented duration estimates (README.md:57, scripts/50_bench_device.sh:48) are derived from that count and both need the same update, or they will disagree.

---

## Lens: `methodology-honesty`

Splitting the two mixed cases in matrix.json is necessary but not sufficient: the divisor still comes from the request (matrix.json) while the numerator comes from the power log and the throughput from llama-bench, so failed runs and unparseable-CSV runs already publish a plausible joules_per_token for tokens that were never processed, and nothing in expand()/phase_of()/build_cmd() prevents a mixed window from being reintroduced by one edit. The audit's proposed "refuse when bench_rows > 1" guard is keyed on the wrong predicate and can blank the entire sweep if the ref emits a degenerate row for the zero axis. I recommend keeping exactly one combined case, but re-denominated as joules_per_request rather than left as a blank-column control, and reporting the refusal as an explicit per_token_status enum on every row rather than as a bare blank.

### Findings (20)

**[CERTAIN] After the split, the largest remaining phase/scope mismatch is not mixed windows at all: a failed run still gets joules_per_token computed from the token count it *requested*. The join loop has no status guard, and tokens_of() reads n_prompt/n_gen/repetitions out of the run record (which comes from matrix.json), not out of any bench row.**  
`bench/run_matrix.py:420`  

> run_matrix.py:356 appends `{**run, "marker": label, "status": "failed", "rc": ..., "t_start_unix": ..., "t_end_unix": ...}` — n_prompt, n_gen and repetitions ride in via `**run`. The join at :398 is `for rec in records:` with no status test; :420 `tokens = tokens_of(rec)`; :431 `if energy_j and tokens:` then writes joules_per_token / wh_per_mtok / active_joules_per_token. An OOM at model-load time therefore yields a row with status=failed, real joules from the crash window, tokens=2433, and a J/token that looks like every other row. docs/10_measurement_protocol.md:118 tells the reader an OOM on 4 GB is normal and recorded as a row.

*Action:* Compute per-token columns only when status==ok AND the window produced exactly one usable bench row AND the row's own token counts are present. Never derive a divisor from matrix.json.

**[CERTAIN] Same path, second door: a run that exits 0 but whose stdout carries no parseable CSV is recorded as status="ok" with bench_rows=0, avg_ts empty, compute_s None — and still gets joules_per_token on the whole-process basis from the requested token count. A row with no throughput and a full J/token is worse than a missing row.**  
`bench/run_matrix.py:372`  

> run_matrix.py:366-380: `bench_rows = list(csv.DictReader(...)) if csv_text else []`; `if not bench_rows:` prints a warning and *continues into* `records.append({**run, ..., "status": "ok", "rc": 0, "avg_ts": tps, "bench_rows": len(bench_rows), ...})`. Nothing downstream reads bench_rows.

*Action:* bench_rows==0 must set status (or per_token_status) to a refusal value and suppress every per-token column, not just leave avg_ts blank.

**[LIKELY] The audit's own proposed guard — refuse per-token columns when a window produced more than one bench row — is keyed on the wrong predicate and can silently blank the whole sweep. build_cmd() always passes both axes, using 0 for the idle one; if the installed ref emits a row for the zero-valued test (a pp0 or tg0 row), then after the split EVERY case reports 2 rows and EVERY per-token column in the study goes blank.**  
`bench/run_matrix.py:143`  

> run_matrix.py:143-149: `if run["n_prompt"]: cmd += ["-p", str(...)] else: cmd += ["-p", "0"]` and the same for -n. The comment directly above at :144 says "llama-bench rejects -p 0 / -n 0 on some refs; only pass the axis in play" — the code does the opposite of its own comment and always passes both.

*Action:* Refuse on phase heterogeneity, not on row count: drop rows whose n_prompt and n_gen are both 0, then accept a window only if the surviving rows agree on phase and there is exactly one of them. Log the discarded degenerate rows so a chatty ref is visible rather than silently blanking the sweep.

**[CERTAIN] bench_tps()'s last-row-wins still misattributes even in a single-phase window, because a literal "0" is a truthy string. A trailing degenerate row overwrites a good avg_ts with 0.0, while joules_per_token is still written (its divisor does not come from the bench rows). The result is a row reading 0.00 t/s next to a real energy-per-token figure.**  
`bench/run_matrix.py:174`  

> run_matrix.py:174-185: `for r in rows: for k in ("avg_ts", "t/s", "tokens_per_second"): if r.get(k): try: tps = float(r[k])` — no break, and `r.get("avg_ts") == "0.00"` is a non-empty string, so it passes the truthiness test and sets tps = 0.0.

*Action:* Take avg_ts from the one accepted bench row rather than by iterating; treat a non-positive avg_ts as a refusal condition, not as a measurement.

**[CERTAIN] Editing matrix.json does not make the split durable. expand() still cartesian-products n_prompt x n_gen, phase_of() still has a "mixed" branch, and largest_local_model is already an armed mixed case whose note tells a future student to enable it. One edit re-creates finding #3 with no warning anywhere.**  
`bench/run_matrix.py:102`  

> run_matrix.py:98-104 `for p in prompts: for g in gens: if p == 0 and g == 0: continue` — the only rejected combination is (0,0), so (512,128) is produced by construction; run_matrix.py:127 `return "mixed"`. matrix.json:140-152: largest_local_model, enabled false, `"n_prompt":[512], "n_gen":[128]`, note "Enable once you have confirmed it fits". matrix.json:9 states the file's own rationale: "Separating them is the whole point of the study -- a single combined number hides the fact that the two phases want different hardware."

*Action:* Make the code, not the data, enforce it: in expand(), skip or hard-fail any run with both p and g non-zero unless the case carries an explicit opt-in key, and fix largest_local_model at the same time as the other two. Leaving phase_of()'s mixed branch is fine only if a mixed run can no longer be produced by accident.

**[LIKELY] The explicit `-n 0` / `-p 0` is load-bearing and currently documented backwards. llama-bench's own defaults are -p 512 -n 128; omitting a flag does not mean "no test", it means "the default test". A future cleanup that drops the else-branches to "only pass the axis in play" (as the comment literally instructs) would reintroduce a mixed window for every case in the matrix.**  
`bench/run_matrix.py:144`  

> run_matrix.py:144 comment "only pass the axis in play" vs :145 `cmd += ["-p", "0"]`. No doc anywhere states why the zero is passed.

*Action:* Rewrite the comment to say the opposite of what it says now, and state the reason (llama-bench defaults to -p 512 -n 128) in docs/10_measurement_protocol.md so it survives the next refactor.

**[CERTAIN] Splitting separates the two phases in time and therefore in thermal state. expand() nests models -> ngl -> threads -> prompts -> gens, so two separate case ids run all five prefill points first and all five decode points afterwards — up to ten cases and ~15 minutes apart, on a fanless board, with one pooled idle floor subtracted from both. The mixed invocation at least measured the pair back to back.**  
`bench/run_matrix.py:96`  

> run_matrix.py:96-102 loop nesting; idle_floor() at :461 averages pow_in_mw over every idle-marked row in the file, pre-sweep baseline and all 20 s cooldowns pooled, then :434 subtracts that one scalar from every row.

*Action:* Keep one case id and give it a `phases` list (prefill spec, decode spec) iterated inside the ngl loop, so the pair is emitted adjacently and `--only partial_offload_sweep` cannot run half of it. Two separate case ids also break the existing --only examples in 50_bench_device.sh and the docs, and let a student measure prefill and decode in two different sweeps against two different idle floors.

**[LIKELY] The compute_s trim is right-anchored to process exit, not to the end of the last timed loop, so the slice contains llama-bench's teardown (CUDA context destruction, munmap, process exit) and excludes an equal duration of real compute at the front. The bias is not common-mode across the phases or across the ngl axis — CUDA teardown costs a -ngl 99 run more than a -ngl 0 run — which is exactly the axis the split exists to compare.**  
`bench/run_matrix.py:413`  

> run_matrix.py:413 `ce = energy_between(power_csv, rec["t_end_unix"] - cs, rec["t_end_unix"])`, where t_end is `time.time()` taken after `subprocess.run(...)` returns (:344). compute_seconds() at :188 derives cs purely from avg_ns plus warm-up; it has no knowledge of when the process actually exited.

*Action:* Either accept it and state the bound in the docs (teardown seconds x the power gap, worst case on the GPU rows), or stop inferring the boundary: have the driver mark the window from a llama-bench progress signal, or subtract a measured teardown from a `-p 1 -n 0 -r 1` calibration invocation per (model, ngl).

**[CERTAIN] A compute slice too short to contain two power samples silently reverts the row to whole-process (load-inclusive) energy. energy_between returns {} for <2 samples, the trim is skipped, and the only trace is the energy_basis string — which any plotting script ignores. The probability of reverting rises as the configuration gets faster, so the GPU rows are the ones most likely to be charged model load while the CPU rows are not.**  
`bench/run_matrix.py:485`  

> run_matrix.py:485 `if len(sel) < 2: return {}`; :414 `if ce.get("joules") is not None:` guards the assignment, and :428-430 falls through to `energy_basis = "whole-process (includes model load)"` with the per-token columns still written from the untrimmed joules. At the documented 5 Hz that floor is ~0.4 s of compute.

*Action:* Treat a failed trim as a refusal for the per-token columns, not as a fallback to a different (larger) scope — or at minimum raise --hz for short cases and refuse when power_samples in the compute slice is below a stated minimum. Publishing a load-inclusive number in the same column as a compute-only one, distinguished only by a prose string, is the mismatch this whole task is about.

**[CERTAIN] Window edges snap inward to the nearest power sample while the token count does not, so every J/token is under-stated by roughly one sample interval of energy, and the error scales inversely with window length — i.e. it is largest on the decode rows, the short side of the study's headline prefill-vs-decode comparison.**  
`bench/run_matrix.py:494`  

> energy_between() selects `t0 <= t <= t1` then takes `e1 - e0` between the first and last *selected* samples (run_matrix.py:494), and reports duration_s from those same samples; tokens_of() returns the full requested count regardless. The previous session's handoff records the symptom empirically: "The two tools' decode J/token differ by ~6% on the same window (0.0572 vs 0.0539)... 5 Hz sampling granularity on a 3 s window plus different window edges".

*Action:* Report tokens_per_second_implied = tokens / compute_s alongside avg_ts so the reader can see when the two disagree, and state the sampling-granularity bound per phase in the docs. Do not quietly compare a 3 s decode window against a 20 s prefill window without saying the shorter one carries several times the edge error.

**[CERTAIN] The joined CSV contains no GPU or CPU rail energy at all, although the protocol says the prefill-vs-decode attribution rests on those rails. After the split, the first question a reader will ask of a prefill row ("where did those watts go?") cannot be answered from the deliverable.**  
`bench/run_matrix.py:494`  

> energy_between() reads only `energy_in_j` and `pow_in_mw` (run_matrix.py:494-496), yet docs/10_measurement_protocol.md says "POM_5V_GPU and POM_5V_CPU are genuine SoC-rail measurements, so the prefill-vs-decode attribution built on them stands." tegrastats_logger.py already writes energy_gpu_j and energy_cpu_j into every sample, and energy_window.py already summarises all three rails.

*Action:* Add gpu_joules / cpu_joules (and their compute-trimmed counterparts) to energy_between's output. It is a four-line change against data already in the CSV, and it is what turns the split from a bug fix into evidence.

**[CERTAIN] Nothing cross-checks the bench row's own n_prompt/n_gen against what was requested. The phase label stamped on the power window is an assertion about intent; if a ref ignores -n 0, clamps n_prompt to the context, or reorders tests, the window is still labelled prefill and the divisor is still the requested one.**  
`bench/run_matrix.py:120`  

> phase_of(run) at run_matrix.py:120 takes the run dict (from matrix.json) as its only input; the marker is built from it at :331 `label = "%s:%s:%03d" % (phase, run["case_id"], i)`. bench_rows are parsed at :366 but their n_prompt/n_gen are used only by compute_seconds(), never compared to the request.

*Action:* When emitting one record per bench row (the audit's fix), carry that row's n_prompt/n_gen/avg_ts/avg_ns and set a phase_mismatch flag if they disagree with the request. This is the change that makes every number on the row come from the same source.

**[CERTAIN] energy_window.py — the standalone tool the docs point at for re-deriving these numbers — pairs the i-th marker window with the i-th bench row positionally and never checks that assumption. The split is what makes the assumption true; the tool should say so, and it also cannot select driver windows by phase at all, because the driver's marker is 'prefill:case_id:007', not 'prefill'.**  
`telemetry/energy_window.py:100`  

> energy_window.py: "Pair the i-th window with the i-th bench row... the ordering is the join key" with `for i, res in enumerate(results): if i >= len(bench): break`. windows_by_marker() matches on exact equality: `if (r.get("marker") or "").strip() == label`. docs/10_measurement_protocol.md:168 shows `--marker prefill`, which matches nothing in a sweep-produced power.csv.

*Action:* Make windows_by_marker accept a prefix (so --marker prefill selects every prefill window of a sweep), and refuse the positional join with a clear error when len(windows) != len(bench rows) instead of silently truncating and misaligning every subsequent pair.

**[CERTAIN] On the blank-column question: a blank is the right value for the numeric columns and the wrong signal on its own. In this CSV a blank per-token cell already means at least five different things, and the ones that matter for defensibility are indistinguishable from each other.**  
`bench/run_matrix.py:441`  

> The DictWriter fieldnames are the union of all record keys (run_matrix.py:441-448), so any key a record lacks is written as an empty cell. Today a blank joules_per_token can mean: the run failed (:356 record has no avg_ts/compute_s), no llama-bench CSV was parsed (:367), the logger produced <2 samples so energy_between returned {} (:485), the sweep ran --no-power (:404 `if os.path.exists(power_csv)`), or joules was 0/None (:431). A reader in pandas gets NaN for all five; `df.joules_per_token.mean()` silently drops them and changes the population without saying so, and a reader in a spreadsheet is as likely to read the blank as "zero energy" or "not run".

*Action:* Keep the numeric columns blank (never a sentinel — a 0 or -1 in a numeric column is how a wrong mean gets published), and add a short machine-readable per_token_status enum populated on EVERY row, good ones included: ok | multi_test_window | no_bench_csv | run_failed | no_power_samples | compute_trim_failed | combined_request. Keep energy_basis as it is — basis says what the numerator is, status says whether there is one; they are different axes. Print a closing count ("N/M rows carry per-token energy; K withheld: ...") next to the existing '%d/%d runs ok' line, and state in the docs that no row whose per_token_status is not ok may appear in a figure.

**[LIKELY] On keeping one deliberately mixed case: keep exactly one, but not as a blank-column control demonstrating that attribution is impossible. That framing is a negative result about your own harness that you can prove on paper for free, and a row that looks like the others with nothing to plot is precisely what invites a future reader to compute the ratio themselves. Keep it with a different denominator instead — the defect was never 'mixed case', it was 'mixed case divided by a token count'.**  
`bench/matrix.json:86`  

> Two positive uses exist. (1) Additivity check: if prefill(512) energy + decode(128) energy reconciles with the combined invocation's energy minus one model load, then the marker windowing, the compute_s trim and the idle floor are mutually consistent — one extra invocation buys a validation of the instrument, which a thesis that measures energy should have. (2) Per-request energy: a combined run has a perfectly well-defined denominator, one request, and joules_per_request is the only number in the study comparable to a served workload. Note also that llama-bench's -p 512 -n 128 runs two independent tests, so its tg128 is decode from an empty KV cache; genuine decode-at-depth needs -pg 512,128, which is a single combined test and therefore the un-attributable shape by construction.

*Action:* Add one case id `phase_additivity_check`, enabled:false with a note saying run it once and record it as its own EXP number; suppress per-token columns structurally (per_token_status = combined_request) and write joules_per_request instead. If the discipline of a separate denominator is not going to be implemented, delete the mixed case entirely — shipping it enabled inside the main matrix with only blank columns to distinguish it is a trap.

**[CERTAIN] docs/10_measurement_protocol.md needs a new subsection in §4, before 'Reading the output'. Draft follows in action_required, in the voice of the existing doc.**  
`docs/10_measurement_protocol.md:103`  

> §4 currently describes the driver in four numbered steps and goes straight to the column table; nothing in the document tells a reader that a case is single-phase, why, or what the split costs. docs/10_measurement_protocol.md:103-122.

*Action:* Insert after the numbered list in §4:

### Every case measures one phase

`llama-bench -p 512 -n 128` is not one measurement. llama-bench treats it as two tests — `pp512` and `tg128` — runs them one after the other in the same process, and prints one CSV row for each. The driver opens one power window per *process*. So a case with both axes produced a row whose throughput came from the second test and whose joules came from both tests plus the model load: a decode tokens/s sitting next to an energy figure that is mostly prefill. Raising `-ngl` speeds prefill much more than it speeds decode, so on such a row the energy column moves and the throughput column does not, and the layer split that looks best on the plot is not the one that is best.

Every case in `bench/matrix.json` therefore sets exactly one of `n_prompt` and `n_gen` to a non-zero value. `partial_offload_sweep` and `quantisation_sweep` used to be `-p 512 -n 128`; each offload level and each quantisation now runs twice, once as `-p 512 -n 0` and once as `-p 0 -n 128`. `phase_of()` labels the window `prefill` or `decode`, never `mixed`, and the phase is on every row.

Two things follow that are easy to get wrong.

**The zero is not optional.** `llama-bench`'s own defaults are `-p 512 -n 128`. Dropping `-n` does not mean "no decode test", it means "the default decode test". `build_cmd()` passes `-n 0` explicitly for a prefill case and `-p 0` explicitly for a decode case for exactly that reason. Removing either would put the mixed window back, silently, for every case in the matrix.

**Run the pair in one sweep.** Prefill and decode for the same offload level are two rows now, and they are comparable only if they were measured against the same idle floor and at a similar board temperature. They are emitted adjacently for that reason, and they share one case id so `--only` cannot separate them. Do not separate them by hand either: two sweeps means two idle floors and two thermal histories under one comparison.

**What the split costs.** Seven more `llama-bench` invocations, so seven more model loads and seven more cooldowns — about five minutes on a sweep that already takes 1.5–3 h, and the 1.1 GB model is read twice per offload level instead of once. In exchange, the throughput, the token count and the joules on a row all describe the same piece of work.

**What the split does not fix.** A `tg` test generates its tokens from an empty KV cache whether or not a `pp` test ran before it in the same process. The decode numbers here are decode at ~zero context depth, in the split matrix and in the old mixed one alike. Decode at a realistic context depth needs `llama-bench -pg 512,128`, which is a single combined test — and a single combined test is the one shape whose energy cannot be attributed to a phase. That trade is stated, not worked around; see §7.

**[CERTAIN] Two rows of the 'Reading the output' table change meaning under the split and are now misleading as written, and three columns are missing from it.**  
`docs/10_measurement_protocol.md:126`  

> docs/10_measurement_protocol.md:126 `| \`avg_ts\` | llama-bench tokens/s for the case |` — 'for the case' is precisely the ambiguity that produced finding #3. :132 `| \`tokens\` | the divisor: \`n_prompt*(r+1) + n_gen*r + 1\` — see below |` — after the split one of the two terms is always zero on every row.

*Action:* Replace those rows and add three:

| `phase` | `prefill` or `decode` — what this window measured. There is no `mixed`. |
| `avg_ts` | llama-bench tokens/s for the one test in this window: a `pp` rate on a prefill row, a `tg` rate on a decode row. Never both. |
| `bench_rows` | how many tests llama-bench reported inside this window. Anything but 1 means the per-token columns are withheld. |
| `tokens` | the divisor. A prefill row: `n_prompt*(r+1)`. A decode row: `n_gen*r + 1`. Work it out by hand against the row once; if it does not match, something changed in llama-bench. |
| `per_token_status` | `ok`, or why the per-token columns are blank. Never plot a row whose status is not `ok`. |

**[CERTAIN] README.md's 'Measurement honesty' list needs a fourth bullet; without it the README promises a result the kit could not produce. §'This node's role' item 2 says measuring the decode-vs-prefill `-ngl` gap 'is one of the study's results', but the only multi-level offload data — partial_offload_sweep, five ngl values on the 1.5B model — reported a decode tokens/s against prefill-dominated joules and dropped the prefill throughput entirely.**  
`README.md:111`  

> README.md:31 "This is why decode (bandwidth-bound) is expected to gain far less from `-ngl` than prefill (compute-bound) does — and measuring that gap on this device is one of the study's results." bench/run_matrix.py:174 bench_tps keeps the last row, so the pp512 throughput of every partial_offload run was never written to the joined CSV at all.

*Action:* Add to the 'Measurement honesty' list (README.md:111):

- **Every case measures one phase, never both.** `llama-bench -p 512 -n 128` is two tests inside one process and one power window, so its row paired a decode tokens/s with prefill-dominated joules — and dropped the prefill throughput entirely. Prefill cases run `-p N -n 0`, decode cases run `-p 0 -n N`, and the driver refuses to divide joules by tokens whenever a window did not report exactly one test: the row keeps its joules and says in `per_token_status` why the per-token columns are blank. In a study whose subject is the difference between the two phases, a number that mixes them is worse than a missing number.

**[CERTAIN] docs/30_experiment_log.md's Result block encodes the old shape: it asks for prefill and decode tokens/s but only one J/token pair, which is the same phase/scope mismatch in notebook form.**  
`docs/30_experiment_log.md:33`  

> docs/30_experiment_log.md:33-35: `tokens/s        : prefill ......   decode ......` / `J/token         : total ......     active ......` / `Wh per Mtok     :`.

*Action:* Replace with a two-column-per-phase block so the notebook cannot record a mixed figure:

```
                  prefill        decode
tokens/s        : ......         ......
J/token (total) : ......         ......
J/token (active): ......         ......
Wh per Mtok     : ......         ......
per_token_status: ......         ......
```

Add to 'What counts as an anomaly worth recording': any row whose `per_token_status` is not `ok`, and any row whose `energy_basis` is not `compute`.

**[CERTAIN] On what the split changes about numbers already described: partial_offload_sweep's energy Pareto becomes two curves, and that is an upgrade rather than a cost — the single curve could not answer the question the case exists for. quantisation_sweep changes the same way, and its own stated rationale needs the prefill rows specifically.**  
`bench/matrix.json:86`  

> matrix.json:88 calls partial_offload_sweep 'The placement question in miniature'; its five rows previously carried tg128 throughput against mixed joules, and no pp512 throughput anywhere. matrix.json:107 justifies quantisation_sweep with 'Maxwell has no useful INT8 path, so K-quant dequantisation cost is a real term here' — a compute-bound, prefill-side claim, and the only throughput the case reported was decode.

*Action:* Plan for two panels sharing an x-axis (ngl 0/8/16/24/99): prefill t/s and prefill J/token above, decode t/s and decode J/token below. The claim the thesis can now make is stronger and is the actual research question — 'prefill keeps improving with ngl while decode flattens at N layers, so the energy-optimal split differs by phase.' Note in the results chapter that the endpoint comparison (ngl 0 vs 99) was always available from the four *_sweep cases on the 0.5B model; it is the intermediate curve on the 1.5B model that the split makes usable. Sweep length goes from 23 to 30 invocations (and largest_local_model from 2 to 4 when enabled); the 1.5–3 h estimate in README.md and 50_bench_device.sh still holds.

### Risks a naive fix runs
- Refusing per-token columns on `bench_rows != 1` can blank the entire sweep, not just the mixed cases: build_cmd always passes -p 0 or -n 0, and a ref that emits a row for the zero-valued test gives every window two rows. Gate on phase homogeneity after dropping degenerate (0,0) rows instead, and log what was dropped.
- Emitting one record per bench row without also splitting the invocation makes energy double-counting easy: two records would share one power window, and copying `joules` onto both inflates the sweep's total energy by the width of every mixed window. Write the window's joules once, tag both records with a shared window id, and leave per-token math off both.
- 'Simplifying' build_cmd to only pass the axis in play — which its own comment instructs — reintroduces finding #3 for every case in the matrix, because llama-bench's defaults are -p 512 -n 128. The explicit zero is the mechanism, not noise.
- Splitting partial_offload_sweep into two case ids breaks `--only partial_offload_sweep` (used in 50_bench_device.sh's header and the docs) and lets a student run the prefill half and the decode half in two different sweeps — two idle floors, two thermal histories, one comparison. Keep one case id with a phases list.
- Even inside one sweep, the pair is no longer back to back unless expand()'s loop order is changed; all-prefill-then-all-decode puts ~15 minutes and several degrees C between the two members of every comparison, and idle_floor() subtracts one pooled scalar across that drift (finding #22).
- Splitting doubles the model loads. With --no-precache the cold/warm SD asymmetry now differs between the prefill and decode members of a pair — the second load of the pair is page-cache warm and the first is not, so the load_joules column stops being comparable within a pair.
- If the prefill and decode halves are ever given different `repetitions`, the divisor and compute_s scale differently and the pair stops being a like-for-like comparison, silently. Pin repetitions at the case level, not per phase.
- Adding per_token_status changes the CSV schema while 90_collect_results.sh copies old and new sweeps into the same tarball. Any analysis notebook must key on the column's presence rather than its position, and a pre-split CSV must not be treated as per_token_status=ok by default.
- Blanking per-token columns without also giving the one legitimately combined case a correct denominator (joules_per_request) leaves a row that looks like the others with nothing plottable on it — which is what makes a reader recompute the forbidden ratio by hand.
- Suppressing per-token columns on failed and CSV-less rows will make the joined CSV look emptier than before on a sweep with OOMs. That is the point, but state it in the docs or the next operator will read the blanks as a regression in the driver.
