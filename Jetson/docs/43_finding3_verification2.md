# Finding #3 follow-ups — second adversarial verification

Three lenses attacked the CRITICAL + 6 HIGH follow-up fixes recorded in
`docs/42_finding3_verification.md`. **All three returned `broken`.**

All three independently found the same CRITICAL, which is the strongest signal in
either verification round: the fix that made a missing model emit a record instead
of vanishing crashed the entire join with `KeyError: 't_start_unix'` whenever power
logging was on — so the fix meant to stop one run disappearing made the whole
archived CSV disappear, on hardware, after a 1.5–3 h sweep. The regression suite
was blind to it because its model-missing test used `--no-power`.

## Status — every CRITICAL and HIGH below is now fixed

`bench/fixtures/run_regression.sh`: **37 passed, 0 failed** (was 30).

| Finding | Fix |
|---|---|
| CRITICAL (all three lenses): skipped record had no `t_start_unix`; the join crashed and the sweep lost its joined CSV | skipped records carry a zero-length window and a unique marker, and the join guards any record without timestamps. The regression now runs the missing-model case **with power on** — the blind spot that hid it |
| HIGH: `--from/--to` satisfied the window check without bounding anything | a join window containing idle samples is refused, naming the marker to use instead |
| HIGH: prefix matching merged adjacent same-prefix windows | `windows_by_marker()` breaks on any change of the **full** marker, not merely on the prefix ceasing to match |
| HIGH: nothing compared the ref's counts with what was asked | `counts_mismatch` / `refused_counts_mismatch`, plus `refused_counts_unverified` when the ref reports no counts and the divisor would come from the request |
| HIGH: a single-row window with unreadable counts still published | covered by `refused_counts_unverified` — `row_phase` derived from the request could never mismatch, so the phase check was comparing the request with itself |
| HIGH: `joules_per_request` could swallow load energy | the request window must now lie inside a **verified** compute window; if the compute trim was refused, no per-request number is published either |
| HIGH: the positional join compared counts, not identity | the window's marker phase is cross-checked against the bench row's own counts |
| HIGH: subprocess text pipes decoded with the locale encoding | all three pin `encoding="utf-8", errors="replace"` |
| HIGH: the two tools disagreed by 7.4 % / 13.0 %, and both understated | both now anchor on the marker and interpolate the counter to the exact bounds. They agree **bit-identically**, and accuracy went from −9.8 %/−6.7 % to −1.4 %/+0.3 % against hand-computed truth. Asserted in the regression |
| HIGH: docs/42 claimed "all of it is now fixed" | corrected to point here |

### Medium and low — also all now fixed

`bench/fixtures/run_regression.sh`: **41 passed, 0 failed**.

| # | Finding | Fix |
|---|---|---|
| M1 | subprocess text pipes decoded with the locale encoding | all three pin `encoding="utf-8"`; the regression grep now also catches `text=True` |
| M2 | skipped rows shared `marker=""`, collapsing the invocation count | unique `(skipped):case:NNN` marker; invocations counted over non-skipped rows; the checker emits a NOTE naming the cases that never ran |
| M3 | skipped runs left no trace in the run log | the skip branch writes to it |
| M4 | docs/10 §5 gave two contradictory `--marker` instructions | reconciled: full label with `--bench`, prefix form for per-phase aggregation, and prefix matching never merges invocations |
| M5 | `window_shared` had three states, breaking `int()` | skipped rows carry `0`; asserted over a CSV that contains one |
| M6 | the no-token-columns fallback divided by the WHOLE window while the numerator was trimmed | divides by the same interval, and is marked `estimated_from_throughput` — never `ok` |
| M7 | nothing compared the ref's `n_gpu_layers` / `model_filename` with the invocation | `config_mismatch` + `refused_config_mismatch`. A ref that ignores `-ngl` would otherwise publish CPU numbers under `ngl99`, and the GPU-offload result *is* that comparison |
| M8 | the checker routed to notes on the status string alone, so a row that both said `refused_` and published a number exited 0 | routing now depends on whether a per-unit column is actually populated; publishing while refused is a violation |
| M9 | the corrected `--reps` warning had the right direction but the wrong magnitude for the path the shipped ref takes | says what actually happens on each path: with `avg_ns` the trim rescales both sides so J/token barely moves but `tokens`/`compute_s`/`load_*` are wrong; without it, the 2×/3× inflation applies |
| M10 | `joules_per_request` was checked with a ±7 % band that hid a real artifact | the counter interpolation fixed the artifact; tolerance tightened to 3 % and the row is asserted self-consistent (`active_joules_per_request × requests == active_joules`) |
| M11 | `energy_basis` had three values where docs said two, and a combined row's `active_joules` sat on a different basis than the label claimed | documented all three; `active_joules` is recomputed on the request slice so the label is true |
| M12 | `model_path` was emitted but documented nowhere | documented, and moved into the fixed lead column order |
| M13 | `HS_FAKE_LOAD_S` described in the wrong units | corrected, with the expected `load_s` / `load_joules` spelled out |

---

Reproductions are quoted as each lens ran them; several built purpose-made fake
refs.

---

## Lens: `regressions-and-semantics` — broken

The eight follow-up fixes did break things, and the 30-test suite is green because it exercises each new path in exactly the one configuration where the breakage cannot appear.

The worst is fix #4. The skipped-run record is NOT "the same shape of record the rc!=0 branch does" as docs/42 prescribed: it omits `t_start_unix`/`t_end_unix` (19 columns are blank on it, vs 0 on the failed-run record). The join loop at run_matrix.py:723 indexes those keys unconditionally, so **any sweep with power logging on that skips even one run dies with KeyError and writes no joined CSV at all** — exit 1, after the full sweep has already run. I reproduced it: the driver that was changed so a skipped run "does not vanish from the joined CSV" now makes the joined CSV itself vanish. The regression's skip test passes only because it is the one test in the file that passes `--no-power`, which is also the one flag that skips the crashing loop. On the shipped matrix this is live: 10 of 32 runs need Qwen2.5-1.5B and 2 need the Q8_0, and `50_bench_device.sh` adds `--no-power` only when NOT on a Jetson — i.e. the crash is armed precisely on the real hardware runs.

Fix #5 is bypassable and, worse, its companion prefix rule re-creates the defect it was written to stop. The guard tests only that a window flag is *present*, so `--from 0 --to 99999999999` sails through and prints a confident J/tok under `marker=idle` with `basis whole-window (includes model load)` — measured 0.351299 vs the true 0.143377 J/tok. And `windows_by_marker()`'s new prefix matching glues adjacent same-prefix windows into one: a real driver run at `cooldown_s: 0` put three llama-bench invocations into a single "prefill" window, and on a crafted power CSV the merged window published 0.001953 J/tok against a true 0.007227 (3.7x low) under a header naming only the first run. Before prefix matching, that same command exited "no samples carry marker 'prefill'". The docs contradict themselves about which form to use in the same section.

Fix #7 is incomplete rather than wrong: all 16 `open()` calls do pin utf-8, but the two `universal_newlines=True` subprocess pipes decode with the locale encoding, which is exactly the ASCII-under-LC_ALL=C-on-3.6 hazard the fix is about — and one of them decodes the output of `env`, i.e. the student's entire environment. The regression's grep only looks for `open(`.

What I could not break: the per-token/per-request arithmetic, the phase-mismatch and multi-phase guards, the compute_s trim, the divisor formula, the --reps direction (measured exactly 2.0x prefill / 2.98x decode), the `window_shared` semantic change (the docs remain accurate under the new meaning), and a full 32-run sweep in both --no-power and power modes.

### Findings (8)

**[CRITICAL] Fix #4's skipped-run record omits t_start_unix/t_end_unix, so the energy join at run_matrix.py:723 raises KeyError and the entire joined *_bench_energy.csv is never written, whenever power logging is on and any run is skipped.**

`bench/run_matrix.py:723`

*Reproduction:*
```
RAN, in a copy of the tree, with the shipped fake ref and fake tegrastats on PATH (no --no-power):

  $ python3 bench/run_matrix.py --env-file $FIX/node.env --matrix $FIX/skippower.json --tag skippower
  ...
  [1/2] have|qwen2.5-0.5b-instruct-q4_k_m|ngl0|t4|p512|g0
      pp512       1024.00 t/s   2048 tok   3.0 s
  [2/2] SKIP gone|qwen2.5-3b-instruct-q4_k_m|ngl0|t4|p512|g0 (model missing)
  idle floor: 1900 mW
  Traceback (most recent call last):
    File ".../bench/run_matrix.py", line 972, in <module>
      main()
    File ".../bench/run_matrix.py", line 723, in main
      e = energy_between(power_csv, rec["t_start_unix"], rec["t_end_unix"])
  KeyError: 't_start_unix'
  EXIT CODE = 1

  $ ls $FIX/results/*skippower*
  ..._skippower_001.csv   ..._skippower_power.csv   ..._skippower_run.log
  (no *_bench_energy.csv -- the deliverable was not written)

Also RAN: the same matrix with --no-power completes normally, which is why the regression's skip test (run_regression.sh:281, the only test in the file that passes --no-power) is green. Column audit of a skipped row: 19 of 42 columns blank, including t_start_unix, t_end_unix, row_phase, window_shared, marker; the rc!=0 record at run_matrix.py:664-668 carries t_start_unix/t_end_unix and is therefore safe (confirmed green by the oom test, which runs WITH power).
```

*Why it matters:* This is the fix inverting itself. It was added so a skipped run would not vanish from the archived artefact; it instead deletes the artefact. The loss happens at the very end of a 1.5-3 h sweep, after the logger has been terminated, and the shipped matrix triggers it for any student who has not downloaded all four GGUFs (10/32 runs need Qwen2.5-1.5B, 2 need the Q8_0). scripts/50_bench_device.sh adds --no-power only when the host is NOT a Jetson, so every real measurement run is in the crashing configuration and every off-hardware rehearsal is in the safe one -- the failure is invisible until it costs an afternoon. The per-case CSVs and the power CSV survive, so the data is hand-recoverable, but the joined CSV every downstream figure is built from is not produced and the process exits 1.

*Fix:* Give the skipped record the same window keys the rc!=0 branch has. At run_matrix.py:621, add `t_start_unix`/`t_end_unix` (e.g. both `time.time()`; an empty window makes energy_between return {} harmlessly), or belt-and-braces guard the join loop with `if rec.get("t_start_unix") is None: continue` before line 723. Then add a skip case WITHOUT --no-power to run_regression.sh -- the current test cannot see this.

**[HIGH] Fix #5's --bench window requirement checks only that --from/--to were passed, not that they bound anything, so `--from 0 --to 99999999999` reproduces verbatim the whole-file join the refusal exists to prevent.**

`telemetry/energy_window.py:284`

*Reproduction:*
```
RAN, against the kit's own synthetic power CSV:

  $ python3 telemetry/energy_window.py p.csv --from 0 --to 99999999999 --bench one.csv --reps 3
  window 0  marker=idle  35.50 s  72 samples
    module    135.25 J   mean   3783.3 mW   peak   7400.0 mW
    tg128  10.00 tok/s  0.351299 J/tok  97.58 Wh/Mtok
    tokens  385   (128 gen x 3 reps + 1 warmup tok)
    basis   whole-window (includes model load)

  $ python3 telemetry/energy_window.py p.csv --marker decode --bench one.csv --reps 3
    tg128  10.00 tok/s  0.143377 J/tok  39.83 Wh/Mtok     <- the true number

2.45x too high, printed under a header reading marker=idle -- the exact narrative the refusal message quotes (0.001672 vs 0.00651). The refusal itself still fires with no flags at all (RAN: 'no-window join refused').
```

*Why it matters:* The refusal text tells the user the tool needs a window and offers --from/--to as an equivalent. The most natural way to satisfy a reluctant range check is to pass a range that covers everything, and that is precisely the state being refused. The docs (docs/10:293) present --from/--to as co-equal to --marker with no warning that an unbounded range is the failure mode itself. A student who hits the refusal and works around it gets a number that is wrong by 2.5x and is labelled marker=idle, which is the only visible clue.

*Fix:* Make the guard about the window, not the flags. After computing `wins`, refuse (or at minimum warn loudly on stderr) when a --bench window's dominant marker is 'idle'/empty, or when a single window spans more than some large fraction of the file's span. The `energy_basis` string already says 'whole-window (includes model load)'; promote that from a printed line to a refusal when --bench is in play.

**[HIGH] Fix #5's windows_by_marker() prefix matching merges ADJACENT same-prefix windows from different invocations into one window, turning what used to be a loud 'no samples carry marker' exit into a silently wrong J/tok joined against a single run's bench row.**

`telemetry/energy_window.py:92`

*Reproduction:*
```
RAN, two parts.

(a) Real driver run, matrix with cooldown_s: 0 (a legal, explicitly editable setting; default is 20), three prefill points, fake tegrastats at TELEMETRY_HZ=5:
  marker sequence in the power CSV: ['idle', 'prefill:pf:001', 'prefill:pf:002', 'prefill:pf:003', 'idle']
  NEW prefix matching, --marker prefill: 1 window(s)
      prefill:pf:001 -> prefill:pf:003  samples 34  J 46.784
  OLD exact matching, --marker prefill: 0
Three invocations, one window. No idle sample lands between runs when the cooldown is short relative to the sampling period, and only the prefix rule can then bridge two differently-labelled windows.

(b) Numeric harm, on a hand-built power CSV with two adjacent prefill windows at different power (7400 mW then 2000 mW, no idle between) and a 1-row bench CSV for the first run:
  $ energy_window.py adj.csv --marker prefill --bench hot.csv --reps 3
  window 0  marker=prefill:hot:001  5.80 s  30 samples
    pp512  1024.00 tok/s  0.001953 J/tok  0.54 Wh/Mtok
    compute      4.0 J over 2.000 s   load 22.72 J over 3.800 s
  $ energy_window.py adj.csv --marker prefill:hot:001 --bench hot.csv --reps 3
    pp512  1024.00 tok/s  0.007227 J/tok  2.01 Wh/Mtok    <- the truth
3.7x too low, under a header naming prefill:hot:001 while the trimmed compute slice is entirely inside prefill:cold:002, and with a fabricated 'load 22.72 J over 3.800 s' that is another run's compute. The len(bench) != len(results) guard does not fire, because the merge makes the counts match (1 window, 1 row).
```

*Why it matters:* The prefix rule was added so `--marker prefill` would work against a sweep CSV. It does -- but it also removes the one property exact matching guaranteed: one window per invocation. When it merges, every downstream safety net passes: the count guard sees 1==1, the compute_s trim happily trims into a neighbouring run, and the printed marker names only the first of the merged runs. That is the identical shape as the defect fix #5 closed (a confident J/tok under a misleading marker header, off by several times), arriving through the fix's own new code path. Precondition is a short cooldown relative to the telemetry period; at the shipped cooldown_s: 20 / TELEMETRY_HZ 5 I could NOT make it merge (verified: a 32-run sweep at cooldown_s 1 produced 19 clean prefill windows, 11 decode, 2 combined). So this is a loaded gun rather than a currently-firing one -- but cooldown_s is a per-case matrix key the docs invite editing, and TELEMETRY_HZ is a node.env knob.

*Fix:* Never merge across distinct full labels. In windows_by_marker(), break the current window whenever the sample's full marker differs from the previous sample's full marker, even if both satisfy the prefix test -- i.e. group by (matches-prefix, exact marker) runs rather than by matches-prefix alone. That keeps `--marker prefill` returning every prefill window while making it impossible for two invocations to share one. Then fix docs/10:276-277, which currently says 'pass the full label' twenty lines before docs/10:297-299 says `--marker prefill` selects every prefill window.

**[MEDIUM] Fix #7 pinned all 16 text open() calls but not the two subprocess text pipes, which decode with the locale encoding under universal_newlines=True -- the same ASCII-under-LC_ALL=C-on-3.6 failure the fix is about, and one of them decodes the output of `env`.**

`bench/run_matrix.py:62`

*Reproduction:*
```
PARTIALLY reproduced -- mechanism demonstrated, not run on a real 3.6 (none available here; this box is 3.12, where locale.getpreferredencoding(False) returns 'utf-8' even under LC_ALL=C PYTHONCOERCECLOCALE=0, so the 3.12 run passes and proves nothing).

Forcing the encoding 3.6 would choose under LC_ALL=C:
  $ python3 -c 'subprocess.check_output(["bash","-c","set -a; . \"$1\"; env","bash",envfile], universal_newlines=True, encoding="ascii")'
  UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2 in position 2217: ordinal not in range(128)
  (node.env carried NODE_NOTE="Jetson Nano B01 <U+2014> wired")
  $ same for the llama-bench capture shape (subprocess.run(..., universal_newlines=True)):
  UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2 in position 6

Also RAN: the regression's locale test (run_regression.sh:295) greps only for `open(` and explicitly skips lines containing 'subprocess', so neither call is in scope. Confirmed by byte scan that the shipped Python and the Python-read config are clean: only telemetry/energy_window.py:55 has non-ASCII (a comment, not in __doc__, so --help is safe); config/models.tsv, config/node.env.example and bench/matrix.json are pure ASCII as claimed.
```

*Why it matters:* load_env() does not decode node.env -- it decodes the output of `env`, i.e. the student's entire environment. One non-ASCII byte anywhere in it (a LANG-ish value, a path, a prompt string, a hand-added NODE_NOTE) kills the driver at startup on the target platform, in the same C-locale condition models.tsv's U+2014 used to. The llama-bench capture is lower risk but the same class: a model filename or a CUDA/backend banner with a non-ASCII byte aborts the sweep after the runs have happened. docs/42 states the fix as 'all 16 text open() calls pin encoding=utf-8', which is true and also not the whole surface -- and the regression test is written so it can never notice.

*Fix:* Pass encoding="utf-8" (and errors="replace" on the llama-bench capture, whose bytes come from a foreign binary) to both subprocess calls: run_matrix.py:62-64 check_output and run_matrix.py:651-652 subprocess.run. On 3.6 `encoding=` is accepted alongside/instead of universal_newlines. Widen the regression grep to any `universal_newlines=True` or `text=True` without an `encoding=`.

**[MEDIUM] Fix #4's skipped rows all carry marker="", so N skipped runs collapse to one entry in the invocation count and violate the per-marker uniqueness the docs' own de-duplication recipe depends on.**

`bench/run_matrix.py:621`

*Reproduction:*
```
RAN, 5-run matrix with 4 runs' model deleted, --no-power:
  1/5 rows ok (2 invocations)
    per_token_status model_missing              4
    per_token_status no_power_window            1
  distinct markers: Counter({'': 4, 'prefill:have:001': 1})
One run was actually invoked; the summary says 2. Four unrelated skipped runs share a marker value.

Also RAN: check_finding3_invariants.py on that CSV prints 'checked 1 rows (5 total) ... all finding-#3 invariants hold' with no NOTE and rc=0.
```

*Why it matters:* docs/10:180 documents rows that share a marker as one window seen twice ('do not sum them'), and docs/41:622 gives analysts the concrete recipe 'sum them over drop_duplicates("marker")'. Skipped rows break that key: a groupby or drop_duplicates on marker fuses four unrelated runs into one. The energy consequence is nil today (skipped rows carry no joules), but the run-accounting consequence is not -- the '(N invocations)' line is now wrong, and fix #8's NOTES machinery, which exists precisely to say loudly 'this sweep is not complete', says nothing at all about the one case where a third of the sweep never ran. 'checked 1 rows (5 total)' is the only signal, and it is silent about why.

*Fix:* Give skipped rows a distinguishable, unique marker (e.g. `skipped:%s:%03d % (case_id, i)`, matching the label format) rather than "", and count invocations as `len({r["marker"] for r in records if r.get("status") != "skipped"})`. In check_finding3_invariants.py, add a NOTE line when rows with status=skipped are present -- 'M of N planned runs never ran' is exactly the 'suspect but correctly refused' category the notes channel was built for.

**[MEDIUM] Fix #4 addressed only half of the finding it cites: skipped runs still leave no trace in the run log, which docs/42's own prescribed fix called for explicitly.**

`bench/run_matrix.py:620`

*Reproduction:*
```
RAN, 5-run matrix with 4 skips:
  $ grep -ci skip results/*_multiskip_run.log
  0
The log contains only the one invoked run's '=== ... $ cmd ...' block. The SKIP line at run_matrix.py:620 goes to stdout only; logfh is not written. docs/42:~95 states the fix as: 'emit the same shape of record the rc!=0 branch does ... before continue, and write the skip line into logfh as well as stdout', and the finding text is titled '...produces ZERO rows in the joined CSV and leaves no trace in the run log'.
```

*Why it matters:* The run log is part of the archived tarball (90_collect_results.sh copies it); stdout is not. Once the terminal scrollback is gone, the log is the only place a reader can reconstruct what the driver actually attempted, and it still says nothing about runs that were planned and skipped. Half a fix for a finding whose whole point was 'a run that appeared only on stdout is indistinguishable later from a run that was never in the matrix'.

*Fix:* Add `logfh.write("\n=== SKIP %s (model missing: %s)\n" % (run_label(run), mpath or "<not in models.tsv>")); logfh.flush()` in the skip branch at run_matrix.py:620. Note logfh is opened at line 612, after the skip branch's enclosing loop starts -- it is in scope, so this is a one-line change.

**[LOW] docs/10_measurement_protocol.md section 5 gives two contradictory instructions for --marker against a sweep CSV, 20 lines apart, and the one that is now wrong is also the safe one.**

`docs/10_measurement_protocol.md:276`

*Reproduction:*
```
READ, not executed. Line 276-277: 'Against a **sweep's** power CSV the marker labels are `phase:case_id:NNN`, so pass the full label.' Line 297-299: '`--marker` matches the sweep's `phase:case_id:NNN` labels on their prefix, so `--marker prefill` selects every prefill window and `--marker prefill:thread_sweep:027` selects exactly one.'
```

*Why it matters:* A reader following line 276 is safe from the window-merge in finding 3; a reader following line 299 is not. Having both in one section means the doc cannot be used to settle which form is correct, and the section is the only place the --marker contract is written down.

*Fix:* Reconcile them: state that the prefix form selects every window of that phase and is for per-phase aggregation, that the full label is required whenever --bench is involved, and (once finding 3 is fixed) that prefix matching never merges two invocations.

**[LOW] window_shared now has three states in the joined CSV -- "0", "1" and "" -- because skipped rows never reach bench_records(), so a consumer that int()s it breaks on a CSV that legitimately contains skipped runs.**

`bench/run_matrix.py:487`

*Reproduction:*
```
RAN, column audit of the 5-row multiskip CSV:
  window_shared on skipped: ''   on ok: '0'
  blank on skipped but present on ok: row_phase, test, window_shared, bench_row_index, marker, bench_n_prompt, bench_n_gen, tokens, avg_ts, rc, t_start_unix, t_end_unix, bench_csv, compute_s, precached, counts_source, avg_ns, stddev_ts, row_compute_s  (19 of 42 columns)
The semantic change itself is fine: I checked docs/10:180 ('window_shared=1 means sibling rows repeat the same joules -- do not sum them') and it remains accurate under the row-count definition, and is in fact more accurate, since the nc2 case (2 rows, 0 parseable) now correctly reports 1. RAN the nc2 shape: window_shared == '1' as the regression asserts.
```

*Why it matters:* Minor on its own -- but it is the visible half of finding 1's root cause. The skipped record is assembled by hand at run_matrix.py:621 instead of going through bench_records(), so it is missing every field the rest of the pipeline assumes exists, and t_start_unix is simply the one that crashes first. Fixing only the two timestamps leaves 17 other columns inconsistent.

*Fix:* Route the skip through bench_records(run, [], window) with a window dict carrying status='skipped' and real timestamps, so a skipped row gets the same column set as a no-CSV row (bench_rows=0, working_rows=0, window_shared=0, row_phase=phase) and per_token_status() returns PT_MODEL_MISSING on its own via the existing `status == "skipped"` branch at line 418.

### Attacks that FAILED — evidence those guards hold
- Full 32-run shipped-matrix sweep with --no-power against the fake ref: RAN, coherent. 32 records for 32 runs, run_index contiguous 1..32, all markers unique, window_shared all '0', all statuses 'ok', 30 x no_power_window + 2 x refused_combined_request, and check_finding3_invariants.py reports 'all finding-#3 invariants hold'. No dropped or duplicated run.
- Same 32-run sweep WITH power logging (cooldown_s 1): RAN. 32/32 rows ok, 30 ok + 2 refused_combined_request, and windows_by_marker() on the resulting power CSV returns 19 prefill / 11 decode / 2 combined windows, each starting and ending on the same full label -- i.e. at any realistic cooldown the prefix rule does NOT merge. The merge in finding 3 needs cooldown_s short relative to the telemetry period.
- Fix #6, the --reps direction: measured, not reasoned. Decode tg128 at -r 3 gives 0.143377 J/tok, and omitting --reps gives 0.427907 -- 2.98x too HIGH. Prefill pp512: 0.019873 vs 0.039746 -- exactly 2.0x too HIGH. Code and docs both state it in that direction now. Correct.
- Fix #3's demotion loop demoting a legitimate 0.0-joules row: could not construct one. `joules` comes from a subtraction of the logger's running trapezoidal integral over a window with >= 2 samples; at any non-zero pow_in_mw and any dt the difference is non-zero (a 0.2 s gap at the 1900 mW idle floor is already 0.38 J). A true 0.0 needs pow_in_mw == 0 for the whole window, i.e. the board off. The checker's has(row,'joules') would treat '0.0' as present while the driver treats it as absent, but since the driver demotes first, no rule in check_finding3_invariants.py can fire on the mismatch. Not reproduced; reported as a non-finding.
- window_shared's changed meaning breaking a consumer: grepped every reader. Only docs/10:180, the nc2 regression assertion, and docs/41/42 prose refer to it. docs/10:180's claim ('sibling rows repeat the same joules -- do not sum them') is still true, and strictly more true, under the row-count definition. docs/30_experiment_log.md does not mention it. No regression.
- Fix #1's phase-mismatch guard producing false positives on well-formed refs: traced every path. A ref emitting n_prompt/n_gen normally always yields row_phase == phase for pp, tg and -pg. A ref omitting the count columns entirely falls back to the matrix's own counts, so single-phase runs still match; only a -pg run with no count columns mismatches, and that is a correct refusal (the driver genuinely cannot tell). No false positive found.
- Fix #2's multi-phase guard blanking legitimate single-row windows: RAN the 32-run sweep and every fixture shape. working_rows == 1 and bench_rows == 1 on all 30 single-phase rows; the guard never fired spuriously.
- The invariant checker choking on the new row shapes: RAN it against a CSV containing 4 skipped rows and against a --no-power combined-only CSV. No crash, no false violation, rc=0 in both. (Its silence about the 4 skipped runs is reported as finding 5, not a crash.)
- scripts/90_collect_results.sh breaking on skipped rows: read it -- it copies files out of RESULTS_DIR and tars them, and never parses the joined CSV. Unaffected.
- Marker-label prefix collisions between case ids in the shipped matrix: checked all 12 ids against the phase:case_id:NNN format. No id is a prefix of another followed by ':', and the startswith(label + ':') form cannot half-match a run number (`--marker prefill:cpu_prefill_sweep:00` selects nothing). Safe.
- Non-ASCII surviving in anything Python decodes at runtime: byte-scanned every .py and every config file. Only telemetry/energy_window.py:55 has non-ASCII, in a comment below the module docstring, so __doc__ stays ASCII and argparse --help cannot raise UnicodeEncodeError under a C locale. config/models.tsv, config/node.env.example and bench/matrix.json are pure ASCII as claimed. (config/cluster.hosts.example has 3 non-ASCII bytes but is read only by bash.)
- The claim under attack itself: RAN ./bench/fixtures/run_regression.sh in my copy -- 30 passed, 0 failed, and it correctly reported config/node.env unchanged. The suite is honest about what it tests; it just does not test the skip path with power on, the --from/--to escape hatch, adjacent same-prefix windows, or the subprocess text pipes.

---

## Lens: `break-the-guards` — broken

I reproduced the baseline first: `./bench/fixtures/run_regression.sh` really is 30 passed, 0 failed in my copy. Then I built refs and matrices aimed at the new guards and RAN them (fake tegrastats + purpose-built fake llama-bench forks, all in my own tmpdir, --env-file always, real node.env md5 unchanged before and after).

The guards fail on three fronts.

(1) One of the seven fixes is a live crash. Fix #4 (a missing model emits a record) creates a record with no `t_start_unix`; the join loop at run_matrix.py:723 indexes that key unconditionally. With power logging on — i.e. on the Jetson, the only place it matters — one missing GGUF ends a completed sweep with `KeyError: 't_start_unix'` and the joined CSV is never written. The regression cannot see it because its model-missing test passes --no-power.

(2) The phase cross-check is a label check, and on one path a tautology. It compares `phase` against `row_phase`, where `row_phase` is computed from `eff_p/eff_g`, which fall back to the matrix request when the ref omits the count columns (run_matrix.py:480). So a ref with no n_prompt/n_gen columns has its `row_phase` derived from the request itself and can never mismatch; the multi-phase guard's new second clause needs `bench_rows > 1`, so a single-row window sails through with working_rows=0 and publishes J/token over the requested divisor. Measured: 0.006571 J/tok published for a window that actually ran a decode test — 5.3x below truth. And when counts ARE present, nothing compares their magnitude to what was asked (the `counts_mismatch` column docs/41:437 designs is still absent): a ref reporting n_prompt=128 while running the 512 it was given publishes 0.026055 J/tok against a true 0.006514 — exactly 4.00x, the same magnitude as the original finding #3 — with per_token_status=ok, and check_finding3_invariants.py prints "all finding-#3 invariants hold" and exits 0. Same blind spot for identity: nothing reads the row's `model_filename` or `n_gpu_layers`, so a ref that ignores -ngl publishes CPU numbers under ngl99 and the checker passes that too.

(3) The energy_window.py window guard checks that a flag was passed, not that the window means anything. `--from 0 --to 9999999999` satisfies it and reproduces, verbatim, the exact number the guard's own error text cites as the disaster it prevents: 0.001672 J/tok against a true 0.006514, under a header reading marker=idle. With --marker, the len(bench)==len(results) check passes on right-count/wrong-rows: two tg128 rows joined to the two prefill windows of a real sweep printed "tg128 … 0.051449 J/tok" under marker=prefill:split_prefill:001.

Also broken: joules_per_request is written for a window bounded only by `rws <= dur`, so an inflated avg_ns drags model-load energy into the numerator — 22.62 J/request measured against a true 7.86 — while the driver had ALREADY concluded on that same row that the timing was impossible (compute_s 16.8 s > duration 14.83 s, so the compute trim was skipped and compute_joules/load_s/load_joules are blank) and then overwrote energy_basis with "timed requests" anyway. Checker: exit 0.

What did hold: expand()'s mixed refusal, the two-row/zero-working-row refusal, the -pg-answered-as-decode refusal, the --no-power demotion, marker prefix boundaries, and preamble poisoning all behaved exactly as advertised under attack.

### Findings (10)

**[CRITICAL] Fix #4 (a missing model emits a record) crashes the driver at join time whenever power logging is on: the skipped record built at run_matrix.py:621 has no t_start_unix, and run_matrix.py:723 indexes that key unconditionally. A completed sweep dies with KeyError and the joined CSV -- the archived deliverable -- is never written.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:723`

*Reproduction:*
```
RAN. Matrix with one present model (prefill) and one absent model, fake tegrastats on PATH, --env-file to a tmpdir:
$ PATH=$A/bin:$PATH HS_FAKE_PHASE=$A/phase python3 bench/run_matrix.py --env-file $A/node.env --matrix $A/missing_power.json --tag missingpower
...
[1/2] has_model|...|p512|g0
    pp512       1024.00 t/s   2048 tok   5.0 s
[2/2] SKIP no_model|...|p0|g128 (model missing)
idle floor: 1900 mW
Traceback (most recent call last):
  File "bench/run_matrix.py", line 972, in <module>
    main()
  File "bench/run_matrix.py", line 723, in main
    e = energy_between(power_csv, rec["t_start_unix"], rec["t_end_unix"])
KeyError: 't_start_unix'
exit=1
Results dir afterwards: only *_001.csv, *_power.csv, *_run.log -- NO *_bench_energy.csv. The regression's model-missing test passes only because it uses --no-power (run_regression.sh, 'a run whose model is missing still appears in the CSV').
```

*Why it matters:* The fix meant to stop a run vanishing from the CSV instead makes the whole CSV vanish, on hardware, after the 1.5-3 h sweep has already been paid for. A missing or mistyped GGUF is exactly the scenario the fix was written for, and 30/30 green cannot see it.

*Fix:* Give the skipped record a window: skipped["t_start_unix"] = skipped["t_end_unix"] = time.time() (energy_between returns {} for a zero-length window, which is already handled), or guard the loop: `if rec.get("t_start_unix") is None: continue`. Add a regression case that runs the missing-model matrix WITHOUT --no-power.

**[HIGH] energy_window.py's new --bench window requirement (line 284) checks only that a window flag was passed, not that the window is meaningful. --from/--to spanning the file reproduces verbatim the wrong number the guard's own error text cites as the reason it exists.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/energy_window.py:284`

*Reproduction:*
```
RAN against a real 5-run sweep power CSV produced by the driver with the shipped fake ref:
$ python3 telemetry/energy_window.py <sweep>_power.csv --bench <sweep>_001.csv --reps 3
--bench needs a window: pass --marker LABEL, or --from/--to.  (refused, as designed)

$ python3 telemetry/energy_window.py <sweep>_power.csv --from 0 --to 9999999999 --bench <sweep>_001.csv --reps 3
window 0  marker=idle  68.10 s  341 samples
  module  238.1452 J   mean 3492.4 mW
  pp512  1024.00 tok/s  0.001672 J/tok  0.46 Wh/Mtok
  tokens  2048   (512 prompt x 3 reps+warmup)
  basis   compute
  compute   3.4243 J over 2.000 s   load 234.7209 J over 66.302 s
exit=0
The driver's own answer for that same run 001 is joules_per_token=0.006514. 0.001672 is the exact figure docs/42 and the guard's error text quote as the disaster case; the 2.000 s 'compute' slice is the tail of run 005 (the fused case), not run 001.
```

*Why it matters:* The guard's error message actively steers the analyst to --from/--to, and the obvious --from/--to is the sweep's own start and end from the run log -- which is precisely this invocation. The headline J/tok is 3.9x low, labelled marker=idle, with nothing in the output flagging it.

*Fix:* Validate the window, not the flag: refuse a join when the selected window contains samples marked idle (or when duration_s exceeds, say, 3x bench_compute_s), and cross-check identity as docs/42 already proposes -- the marker encodes phase:case_id:NNN and the per-case CSV filename encodes the same NNN; refuse on disagreement. At minimum refuse --bench when the chosen window's marker is 'idle'.

**[HIGH] Nothing compares a bench row's reported counts with what the invocation asked for. The phase cross-check compares labels only, so a ref that answers -p 512 with n_prompt=128 (same phase, wrong magnitude) publishes a 4x wrong J/token that passes the driver AND check_finding3_invariants.py. The counts_mismatch column docs/41:437 designs is still not implemented (grep -c counts_mismatch bench/run_matrix.py -> 0).**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:480`

*Reproduction:*
```
RAN. Fake ref that does the full 512-token prefill work it was given but reports n_prompt=128 / test=pp128 (a fork printing the batch size), matrix asks n_prompt:[512]:
$ PATH=$A/bin3:$PATH HS_FAKE_PHASE=$A/phase python3 bench/run_matrix.py --env-file $A/node_lie.env --matrix $A/nocounts1.json --tag liecount
Joined row: phase=prefill row_phase=prefill test=pp128 per_token_status=ok bench_rows=1 working_rows=1 counts_source=bench n_prompt=512 bench_n_prompt=128 tokens=512 compute_joules=13.3401 joules_per_token=0.026055 wh_per_mtok=7.24
true divisor for the work actually done = 2048 -> true J/tok 0.006514; published 0.026055 = 4.00x too HIGH
$ python3 bench/check_finding3_invariants.py <that csv>
checked 1 rows (1 total)
all finding-#3 invariants hold
exit=0
```

*Why it matters:* The row carries n_prompt=512 and bench_n_prompt=128 side by side and no code path or checker rule compares them. A 4x divisor error is the same magnitude as the original finding #3, published as ok, signed off by the dedicated finding-#3 checker. Checker rule 4 parses the test label for PHASE only, never for the number in it, so 'pp128' corroborates nothing.

*Fix:* Implement counts_mismatch as designed: set-compare the expected {pp512}/{tg128} labels built from the run against the rows' own counts, write counts_mismatch on every row of the window, print a warning in the run loop, and add a checker rule that counts_mismatch=1 may not carry joules_per_token. Also compare the number in the test label with bench_n_prompt/bench_n_gen in check rule 4.

**[HIGH] The multi-phase guard's new second clause needs bench_rows > 1, so a SINGLE-row window whose counts the driver cannot read still publishes -- and on that path the phase cross-check is a tautology, because eff_p/eff_g fall back to the matrix request, so row_phase is derived from `phase` itself and can never disagree. The divisor is then what was ASKED for, not what ran.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:432`

*Reproduction:*
```
RAN. Ref fork emitting ONE row with no n_prompt/n_gen and no `test` column (the shape docs/42:459 says real llama-bench actually has), which runs a tg128 test regardless of the -p 512 it was given:
$ PATH=$A/bin2:$PATH HS_FAKE_PHASE=$A/phase python3 bench/run_matrix.py --env-file $A/node_liar.env --matrix $A/nocounts1.json --tag liar1
[1/1] prefill_asked|...|p512|g0
    pp512        128.00 t/s   2048 tok   6.0 s
1/1 rows ok (1 invocations)
  per_token_status ok                         1
Row: phase=prefill row_phase=prefill test=pp512 bench_rows=1 working_rows=0 window_shared=0 counts_source=matrix tokens=2048 compute_joules=13.4575 joules_per_token=0.006571
True divisor for the decode test that actually ran = 385 -> 0.034955 J/tok. Published 0.006571 = 5.3x too LOW.
checker: FAIL (1) - joules_per_token=0.006571 but working_rows=0  (exit 1)

Also RAN with the kit's OWN fixture ref and the regression's own env var, one row:
$ HS_FAKE_NO_COUNTS=1 python3 bench/run_matrix.py ... --tag nocounts
-> per_token_status=ok bench_rows=1 working_rows=0 counts_source=matrix tokens=2048 joules_per_token=0.005085 ; checker FAIL.
run_regression.sh performs exactly this run under the heading 'driver: a ref with no count columns cannot publish a number either' and asserts NOTHING about it -- only the two-row nc2 variant is checked.
```

*Why it matters:* per_token_status()'s own comment says zero identifiable working rows means 'the driver cannot tell which test the tokens describe, which is the same problem wearing a different hat' -- and then it publishes anyway whenever the window has one row. The CSV is the archived artefact and the driver prints '1/1 rows ok'. Mitigation is real: the invariant checker does catch it (exit 1), so this is only survivable if the checker is run on every sweep, which nothing enforces.

*Fix:* Drop the bench_rows > 1 precondition: refuse whenever working_rows != 1 for any window with at least one row. Equivalently, refuse when counts_source == 'matrix' -- if the counts could not be read, neither the divisor nor the phase cross-check means anything. Then make run_regression.sh assert on its existing --tag nocounts artefact instead of discarding it.

**[HIGH] joules_per_request is written for any window where request_window_s <= duration_s, with no requirement that the request window fall inside the timed loops. An overstated avg_ns drags model-load energy into the numerator, and the driver publishes it as energy_basis='timed requests' even on a row where it had already rejected the compute trim as physically impossible (compute_s > duration_s).**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:775`

*Reproduction:*
```
RAN two -pg sweeps with the same real work (3 reps x 1.5 s) and a 10 s model load; only the reported avg_ns differs.
Honest (avg_ns=1.5e9): request_window_s=4.5 requests=3 request_joules=23.5733 joules_per_request=7.8578 compute_s=6.0 duration_s=14.823 load_s=8.823 load_joules=46.8644 energy_basis='timed requests'
Liar  (avg_ns=4.2e9): request_window_s=12.6 requests=3 request_joules=67.8566 joules_per_request=22.6189 compute_s=16.8 duration_s=14.828 compute_joules=None load_s=None load_joules=None energy_basis='timed requests'
-> 22.6189 vs 7.8578 = 2.88x too high; the extra 8.1 s of the 'request' window is model load at 5200 mW.
$ python3 bench/check_finding3_invariants.py <liar csv>  -> all finding-#3 invariants hold, exit 0 (rule 8 only checks 22.6189 == 67.8566/3).
Note compute_s=16.8 > duration_s=14.828: the compute trim was skipped for exactly this reason and energy_basis was set to 'whole-process (includes model load)', then line 785 overwrote it with 'timed requests'.
```

*Why it matters:* joules_per_request is the headline number for the only case in the shipped matrix that deliberately spans both phases, and its whole justification is that the trailing avg_ns x reps seconds contain no load and no warm-up. The only bound is rws <= dur, and rws == dur is accepted, so 'timed requests' can legally mean the entire process including CUDA init. The driver detects the self-contradiction on that same row and does not act on it.

*Fix:* Refuse the per-request figure when the row's timing is inconsistent with the window: require rws < compute_s <= duration_s (compute_s = ns*(reps+1) for a fused row is always > rws when avg_ns is honest), and add a status such as refused_timing_inconsistent whenever compute_s >= duration_s. Never overwrite energy_basis after the compute trim has been refused. Add a checker rule: joules_per_request requires request_window_s < compute_s <= duration_s.

**[HIGH] energy_window.py's positional-join guard compares only the COUNT of windows to the count of bench rows. With --marker prefix matching now selecting several sweep windows at once, a bench CSV with the right number of rows from the wrong runs joins silently -- decode rows against prefill windows.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/energy_window.py:360`

*Reproduction:*
```
RAN against a real driver sweep (markers prefill:split_prefill:001, decode:split_decode:002, prefill:split_prefill:003, decode:split_decode:004, combined:fused_request:005). --marker prefill selects 2 windows. Bench CSV = the two DECODE per-case CSVs concatenated (2 rows, tg128, n_gen=128):
$ python3 telemetry/energy_window.py <sweep>_power.csv --marker prefill --bench $A/decode_rows.csv --reps 3 --idle-mw 1900
window 0  marker=prefill:split_prefill:001  4.81 s
  module 29.1841 J  mean 6080.0 mW
  tg128  128.00 tok/s  0.051449 J/tok  14.29 Wh/Mtok
  tokens 385  (128 gen x 3 reps + 1 warmup tok)
  basis  compute
window 1  marker=prefill:split_prefill:003 -> 0.051441 J/tok
exit=0
The driver's own decode figure for those rows is 0.034947-0.034969 J/tok; the energy here is prefill energy (7400 mW) over a decode divisor. Every row carries a `test`, `model_filename` and `n_gpu_layers` that contradict the window label; none is consulted.
```

*Why it matters:* Prefix matching was added so analysts WOULD use --marker, and it makes multi-window selections the normal case -- which is exactly when a positional join needs identity, not arithmetic. The output looks completely ordinary: a plausible J/tok, a real marker, a real window.

*Fix:* Join on identity: parse phase:case_id:NNN out of the window's marker and compare it with the bench row's own phase (from its n_prompt/n_gen) and, where available, the per-case CSV's run index; refuse on disagreement. Minimum: refuse when the row's phase (prefill/decode/combined) does not match the phase segment of the window's marker.

**[MEDIUM] energy_window.py's no-token-columns fallback estimates tokens as avg_ts x the WHOLE window duration while the numerator has already been trimmed to the compute slice, so the published J/token is short by exactly duration_s/compute_s -- and it is reported as per_token_status=ok.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/energy_window.py:393`

*Reproduction:*
```
RAN on the shipped synthetic power CSV with a bench row that has avg_ns and avg_ts but no count columns:
$ python3 telemetry/energy_window.py $A/p.csv --marker decode --bench $A/notok.csv --reps 3 --idle-mw 1900
window 0  marker=decode  11.50 s   module 55.2 J
  None  10.00 tok/s  0.250435 J/tok  69.57 Wh/Mtok
  tokens  115   (avg_ts x window duration (no token columns))
  basis   compute
  compute     28.8 J over 6.000 s   load 26.4 J over 5.500 s
per_token_status=ok. Either self-consistent basis gives 0.48 J/tok (28.8/60 on the compute basis, 55.2/115 on the whole-window basis). Published value is 1.92x low = 11.5/6.0.
```

*Why it matters:* The two halves of the ratio are measured over different intervals: joules over the trailing 6 s that produced tokens, tokens over 11.5 s that includes 5.5 s of model load in which zero tokens were produced. The printed output states both facts on adjacent lines and the tool still calls it ok.

*Fix:* In the avg_ts fallback multiply by the same interval the numerator uses: tokens = bench_tps * (res.get('compute_s') or res['duration_s']). Better, mark the fallback per_token_status='estimated_from_throughput' so it can never be mistaken for a counted divisor.

**[MEDIUM] The driver never cross-checks the row's own model_filename or n_gpu_layers against the invocation, though it now does so for phase. A ref that ignores -ngl (a CPU-only build, or a clamp to the model's layer count) publishes its numbers under the requested ngl, and the checker passes.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:480`

*Reproduction:*
```
RAN. Ref with honest counts that always reports n_gpu_layers=0, matrix asks n_gpu_layers: 99:
$ PATH=$A/bin5:$PATH ... python3 bench/run_matrix.py --env-file $A/node_ngl.env --matrix $A/nocounts1.json --tag nglliar
Per-case CSV row: "n_gpu_layers"="0"
Joined row: n_gpu_layers=99 per_token_status=ok counts_source=bench tokens=2048 joules_per_token=0.006511
$ python3 bench/check_finding3_invariants.py <that csv>  -> all finding-#3 invariants hold, exit 0
Static confirmation: no code anywhere in run_matrix.py reads a row's model_filename / n_gpu_layers / n_threads / build_commit (checked by scanning the module source for r.get/row.get/bench_int on those names -- zero hits).
```

*Why it matters:* GPU offload is the study's main independent variable. The phase cross-check establishes the principle that the driver must compare what it asked for with what came back; the same row carries three more fields it asked for (model, ngl, threads) and none is compared. A ref that silently falls back to CPU produces a full, clean, internally consistent ngl-sweep that measures nothing.

*Fix:* Extend the cross-check: compare bench_int(r,'n_gpu_layers') and the row's model_filename basename with run['n_gpu_layers'] / the model path; on disagreement write a config_mismatch column and refuse the per-token figure, with a matching checker rule.

**[MEDIUM] The checker's VIOLATIONS/NOTES split routes failures to notes on the strength of `refused = status.startswith('refused_')` alone, and then prints 'no bad number was published' without ever checking whether the row publishes one. Since refused_combined_request is a status that legitimately DOES publish joules_per_request, and since the column scans that would catch a published number live only inside rules 5 and 6 (which need the optional `phase` / `pg_n_*` columns), a CSV missing those columns downgrades a real violation to a note and exits 0.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/check_finding3_invariants.py:122`

*Reproduction:*
```
RAN. One-row CSV with no `phase` and no pg_n_* columns, a fused test label, decode counts, a refusal status, and a PUBLISHED, arithmetically consistent joules_per_request:
case_id,row_phase,test,per_token_status,bench_rows,working_rows,bench_n_prompt,bench_n_gen,tokens,requests,request_joules,joules,joules_per_request,energy_basis
sneaky,decode,pp512+tg128,refused_phase_mismatch,1,1,0,128,385,3,60.0,80.0,20.0,timed requests
$ python3 bench/check_finding3_invariants.py $A/notes_abuse.csv
  1 row(s) carry suspect data that the driver correctly refused:
    note: row 0 (sneaky / pp512+tg128): test=pp512+tg128 is a fused label but the row reports n_prompt=0 n_gen=128 ...
    -> no bad number was published, ...
all finding-#3 invariants hold
exit=0
```

*Why it matters:* The checker advertises itself as independent of how the fix is spelled, so it must not depend on optional columns to notice a published number. Its notes text asserts an invariant ('no bad number was published') that the code never evaluates -- the one sentence a reader would trust most.

*Fix:* Compute `published = any(has(r,c) for c in PER_TOKEN + PER_REQUEST)` once per row and make every (notes if refused else fails) site read (notes if refused and not published else fails); treat a populated per-unit column on any row whose status starts with refused_, other than refused_combined_request's per-request columns, as a violation regardless of which other columns exist.

**[LOW] Fix #6's corrected --reps warning has the direction right but the magnitude wrong for the path the shipped ref actually takes: when avg_ns is present the compute trim rescales the numerator by the same factor as the divisor, so a wrong --reps changes J/token by ~0%, not 2x/3x. The documented 2.0x/3.0x holds only when the bench row has no avg_ns.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/energy_window.py:319`

*Reproduction:*
```
RAN on the synthetic power CSV.
With avg_ns (compute basis): decode tg128 --reps 3 -> 0.037403 J/tok; --reps omitted (=1) -> 0.037209 J/tok (ratio 1.005, claimed 3.0x). prefill pp512 --reps 3 -> 0.007227; omitted -> 0.007227 (ratio 1.000, claimed 2.0x).
Without avg_ns (whole-window basis): prefill --reps 3 -> 0.019873; omitted -> 0.039746 (ratio exactly 2.0, as documented).
```

*Why it matters:* Minor, and it errs safe -- but the warning is the operator's only guide to how much a --reps slip costs, and it overstates it 3x on the normal path while understating the real consequence there: with a wrong --reps the headline J/token is right while the reported tokens, compute_s, load_s and load_joules on the same record are all wrong by that factor.

*Fix:* Reword to say what actually happens: on the compute basis a wrong --reps leaves J/token nearly unchanged but makes tokens, compute_s, load_s and load_joules wrong by ~reps; the 2x/3x inflation applies only when the row carries no avg_ns. Or simply read repetitions from the joined CSV instead of asking for it.

### Attacks that FAILED — evidence those guards hold
- Single row claiming BOTH phases for a single-phase request. bench_records({'n_prompt':512,'n_gen':0}, [{'n_prompt':'512','n_gen':'128','test':'pp512+tg128',...}], phase='prefill') -> refused_phase_mismatch. The phase check alone catches it; no per-token and no per-request figure.
- Reaching the per-request path for a request the matrix never asked to fuse. joules_per_request is gated on per_token_status == PT_COMBINED, which needs row_phase == 'combined' AND phase == 'combined', and phase_of() only returns 'combined' when pg_n_prompt and pg_n_gen are set. A fused row returned for a decode or prefill request is refused as a phase mismatch instead (verified at unit level for both).
- Making row_phase empty/None so `want and got` short-circuits inside the driver. phase_of_counts returns 'none' for (0,0) -- a truthy string -- so a row carrying avg_ts with both counts zero scores as 'none' and is refused as a phase mismatch, not skipped. The no-bench-rows path sets row_phase = phase but is caught earlier by PT_NO_BENCH_CSV. (The equivalent short-circuit IS reachable in the checker, via a CSV with no phase column -- reported as the notes/VIOLATIONS finding.)
- --marker prefix over-matching. windows_by_marker(rows,'prefill') on markers ['prefill','prefill_extra:c:1','prefill:c:2','prefillX','prefill:c:3'] selects exactly ['prefill'], ['prefill:c:2'], ['prefill:c:3'] -- the ':' boundary holds, 'prefill_extra:c:1' and 'prefillX' are not matched. A partial case-id prefix ('prefill:split') selects nothing and exits with the no-samples error.
- Poisoning bench_csv_text()'s preamble stripping with a stdout log line containing a header hint ('llama_model_loader: build_commit=a1b2c3d'). The junk line becomes the header, the real header and the data row become two rows with unreadable counts, and the window degrades to bench_rows=2 / working_rows=0 -> refused_multi_phase on both rows. No number published.
- The documented two-row/zero-identifiable-count case (HS_FAKE_NO_COUNTS=1 HS_FAKE_EXTRA_ROW=1): refused_multi_phase on both rows with window_shared=1, exactly as claimed. The new second clause of the guard does work -- it just needs bench_rows > 1.
- The -pg-answered-as-decode CRITICAL (HS_FAKE_HIDE_PROMPT=1): refused_phase_mismatch, no joules_per_token, no joules_per_request, and the checker reports it loudly as a note while exiting 0. Reproduced as claimed.
- expand()'s mixed-case refusal, the --no-power 'ok' demotion, bench_csv_text preamble stripping, the divisor n_prompt*(r+1)+n_gen*r+1, and the per-request physics check (8.5 J hand-computed) all pass: I re-ran ./bench/fixtures/run_regression.sh unmodified in my copy and got 30 passed, 0 failed.
- Clobbering the live config: config/node.env md5 is 348ebc26c5bb48abf26e23ddfe56527d before and after everything above; every driver invocation used --env-file pointing into my scratchpad, and all runs happened in a copy under /tmp. (The live tree did change under me at 19:02 -- a peer session added a 4-line comment at run_matrix.py:798 and appended an Outcome section to the newest handoff -- so I re-verified every line number in this report against the live files.)

---

## Lens: `py36-docs-and-numbers` — broken

I obtained a real CPython 3.6.13 (Anaconda linux-64 tarball, LD_LIBRARY_PATH) and ran everything under it; every finding below marked "ran it" was executed, not reasoned about.

(a) Python 3.6.9 — HOLDS, with one hole. All 12 .py files compile under real 3.6.13, and `./bench/fixtures/run_regression.sh` gives **30 passed, 0 failed** under 3.6.13 with LC_ALL=C. No walrus, no dataclasses, no `a|b` dict merge, no `datetime.fromisoformat`, no `subprocess` `text=`/`capture_output`, no f-string `=`. `pick(row, *names, skip_zero=False)` is keyword-only-after-*args, PEP 3102, Python **3.0** — a non-issue. Dict-order reliance is safe: the 63-column header is byte-identical under 3.6.13 and 3.12. The hole is that fix #7 only covered `open()`. Three text pipes still decode with the locale encoding — `run_matrix.load_env` (:64), `subprocess.run` of llama-bench (:652), and `tegrastats_logger` (:158). Under LC_ALL=C on 3.6 that is ASCII, and I reproduced a `UnicodeDecodeError` that kills the sweep on run 1. The regression's "text I/O does not depend on the locale" test cannot see it: line 304 excludes any line containing "subprocess".

(b) Docs vs code — mostly good, one large false claim. The `per_token_status` table is an exact three-way match with the PT_* constants and the checker's KNOWN_STATUS, both directions, no extras either way. "32 runs, combined 2, decode 11, prefill 19" is exact. 2.0×/2.98× for `--reps` is right and the direction is right. `ok` ⇒ non-empty `joules` is enforced for every record, including skipped and failed, outside the power-CSV guard. What is false is docs/42's own headline: "**Status as of 2026-09-13 12:0x — all of it is now fixed / Every finding below has been addressed**". Five findings the same document marks **STILL LIVE** are still live, and I re-measured four of them. `model_path` is documented nowhere, and `energy_basis` emits three values against a column table that says "which of the two".

(c) The numbers — this is where it breaks. `tokens_of()` is exact at r=1,2,3,5,10 and the split halves sum to the fused value at every r. `request_window_s = avg_ns × reps` is exact and `joules_per_request` really is load-independent (7.8641 at HS_FAKE_LOAD_S=3 vs 7.8632 at 12). But the hand-computed 8.5 J does **not** reproduce: I measured 7.86–8.36 across 10 runs, always low, and the regression's ±0.6 tolerance failed 3 of 6 times on an equivalent single-case matrix. The cause is systematic and much wider than the fused case: `energy_between()` integrates between the first and last *sample* inside [t0,t1], while the driver divides by the *nominal* window and subtracts the idle floor over the *nominal* window. Prefill J/token came out 9.8 % low and decode 6.7 % low against hand-computed truth — the bias is (sampling interval)/(window length), so it differs per phase and distorts the prefill-vs-decode ratio, which is the entire deliverable. `energy_window.py`, the tool the protocol offers for spot-checking, gets the same window right to 0.2 % and disagrees with the published CSV by 7.4 % on J/token and 13.0 % on active J/token. And separately: any skipped model with power logging on crashes the driver before it writes the joined CSV.

### Findings (9)

**[CRITICAL] Follow-up fix #4 (a missing model emits a record) crashes the whole sweep: the skipped record has no `t_start_unix`, and the join loop indexes it unconditionally. With power logging on — i.e. every real run on the Jetson — one missing GGUF means a `KeyError` after all the measurement work is done and NO joined CSV at all. The regression's model-missing test passes only because it uses `--no-power`, which skips the entire block that crashes.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:723`

*Reproduction:*
```
Ran it, on a copy, under real CPython 3.6.13 with the shipped fakes. Matrix = 3 good cases + 1 case whose model is absent from models.tsv:

  $ python3 bench/run_matrix.py --env-file $FIX/node.env --matrix $FIX/full.json --tag full
  collecting 30 s idle baseline...
  [1/4] split_prefill|... pp512 1024.00 t/s 2048 tok 5.0 s
  [2/4] split_decode|...  tg128  128.00 t/s  385 tok 6.0 s
  [3/4] fused|...   pp512+tg128 426.67 t/s 2433 tok 8.0 s
  [4/4] SKIP nomodel|not-in-models-tsv|ngl0|t4|p512|g0 (model missing)
  idle floor: 1900 mW
  Traceback (most recent call last):
    File "bench/run_matrix.py", line 972, in <module>
      main()
    File "bench/run_matrix.py", line 723, in main
      e = energy_between(power_csv, rec["t_start_unix"], rec["t_end_unix"])
  KeyError: 't_start_unix'

  $ ls $FIX/results/
  ..._full_001.csv ..._full_002.csv ..._full_003.csv ..._full_power.csv ..._full_run.log
  (no *_bench_energy.csv — the deliverable was never written)

Same matrix with --no-power: rc=0, CSV written, 'per_token_status model_missing 1'. That is the only configuration the regression exercises (run_regression.sh:281-282 passes --no-power).

Line 621 builds the skipped record from {**run, marker, phase, run_index, status, rc, bench_rows, working_rows, model_path, tag} — no t_start_unix/t_end_unix, unlike the failed record at line 666 which has both.
```

*Why it matters:* The shipped matrix needs three distinct GGUFs (0.5B Q4_K_M, 0.5B Q8_0, 1.5B Q4_K_M) across 32 runs taking 2–3.5 h. A partially populated MODEL_DIR — the normal state while downloads are in progress, and the exact scenario fix #4 was written for — now destroys the entire sweep instead of losing one row. It is strictly worse than the bug it replaced: before, the run vanished from the CSV; now there is no CSV. docs/10:120 'Every run produces at least one row' is false in the direction that costs hours of board time, and the operator only finds out at the very end.

*Fix:* Give the skipped record `t_start_unix`/`t_end_unix` (e.g. both = time.time() at the skip, yielding an empty window), or guard the join: `if rec.get('t_start_unix') is None: continue` before line 723. Then change run_regression.sh's model-missing case to run WITH power so the path is actually covered — the --no-power flag there is what hides it.

**[HIGH] `compute_joules`, `joules_per_token` and `active_joules` are systematically understated, and by a different amount for prefill than for decode. `energy_between()` returns the energy counter difference between the first and last power *sample* inside [t_end-compute_s, t_end] — a span shorter than compute_s by up to one sampling interval — but the driver divides that energy by tokens and subtracts the idle floor over the *nominal* compute_s. The published row contradicts itself: `compute_mean_mw × compute_s / 1000 ≠ compute_joules`.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:754`

*Reproduction:*
```
Ran it, three times, under CPython 3.6.13 with the shipped fakes (prefill 7400 mW, decode 4800 mW, idle 1900 mW, 5 Hz, r=3). Hand-computed truth: prefill compute = 2.0 s @ 7400 mW = 14.80 J over 2048 tok = 0.0072266 J/tok; decode compute = 3.0078 s @ 4800 mW = 14.4375 J over 385 tok = 0.037500 J/tok.

  run 1 pref tokens=2048 compute_s=2.0   compute_joules=13.3481 J/tok=0.006518 active_j=9.5481
  run 2 pref tokens=2048 compute_s=2.0   compute_joules=13.3456 J/tok=0.006516 active_j=9.5456
  run 3 pref tokens=2048 compute_s=2.0   compute_joules=13.3511 J/tok=0.006519 active_j=9.5511
  run 1 dec  tokens=385  compute_s=3.008 compute_joules=13.4655 J/tok=0.034975 active_j=7.7503
  run 2 dec  tokens=385  compute_s=3.008 compute_joules=13.4652 J/tok=0.034975 active_j=7.7500

  prefill J/tok  -9.8 %   decode J/tok  -6.7 %   prefill active_joules  -13.2 % (truth 11.00 J)
  decode/prefill J/tok ratio: truth 5.189, published 5.366  -> +3.4 % distortion

Mechanism, measured by calling energy_between() on the driver's own power CSV with the driver's own compute_s:
  pref compute_s(nominal)=2.000  ACTUAL sampled span=1.804 s  lost=0.196 s (9.8%)
       compute_joules=13.3488  compute_mean_mw=7400.0 -> mean_mw*compute_s/1000 = 14.8000 J  (row inconsistent with itself)
       active_joules = compute_joules - 1900 mW x 2.000 s, but compute_joules only spans 1.804 s;
       floor subtracted over 2.000 s = 3.8000 J while only 3.4276 J of floor is inside the measured energy
  dec  compute_s(nominal)=3.008  ACTUAL sampled span=2.806 s  lost=0.202 s (6.7%)

The lost interval is ~1 sample at 5 Hz in every run, so the relative bias is (1/hz)/compute_s — which is why it is 9.8 % on the 2.0 s prefill window and 6.7 % on the 3.0 s decode window.
```

*Why it matters:* Prefill-vs-decode energy per token IS the thesis. A bias whose size depends on window length does not cancel in that comparison, and the matrix deliberately sweeps window length (n_prompt 128..1024, n_gen 32 and 128), so the bias varies across the very axes being plotted. On a real Nano the windows are longer so the absolute percentage shrinks, but the short arms — tg32 at -ngl 99, pp128 — stay the most affected, and nothing on the row tells a reader this happened. docs/42 catalogues only the idle-floor half of this as [LOW]; the larger half is that compute_joules itself is short.

*Fix:* `energy_between()` already computes the sampled `duration_s` of the sub-window. Use it: set `cs_measured = ce['duration_s']`, divide by tokens using compute_joules as-is (it is honest energy over that span) but report `compute_s = cs_measured` so the row is self-consistent, and subtract the idle floor over `cs_measured`, not over `cs`. Better still, interpolate the counter to the exact t0/t1 so no compute energy is dropped at all. Add a checker rule: |compute_mean_mw x compute_s/1000 - compute_joules| must be within tolerance.

**[HIGH] `run_matrix.py` and `telemetry/energy_window.py` — documented as twins that must be kept in step, and the protocol names energy_window.py as the way to re-derive a published figure by hand — disagree by 7.4 % on J/token and 13.0 % on active J/token for the same power CSV, the same bench row and the same --reps. The published tool is the wrong one; energy_window.py is within 0.2 % of truth. docs/42 lists this as STILL LIVE at 1.8 %/3.0 %; on shorter windows it is four times that.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/energy_window.py:356`

*Reproduction:*
```
Ran it, under CPython 3.6.13. Same driver-produced power CSV, same per-case bench CSV (results/*_hdr_002.csv, the tg128 row), --reps 3 --idle-mw 1900:

  run_matrix.py joined row : compute_joules=13.4663 joules_per_token=0.034977 active_joules_per_token=0.020133
  energy_window.py --json  : compute_joules=14.4680 joules_per_token=0.037579 active_joules_per_token=0.022744
  hand-computed truth      : compute_joules=14.4375 joules_per_token=0.037500 active_joules_per_token=0.022656

  energy_window vs run_matrix: J/tok +7.4 %, active J/tok +13.0 %
  energy_window vs truth     : +0.2 % and +0.4 %   (it anchors on the last sample and subtracts the floor over the SAMPLED duration)
  run_matrix   vs truth      : -6.7 % and -11.1 %

Same invocation also reproduces docs/42's other STILL LIVE item: module_active_joules=18.019 (whole 5.813 s window) is printed next to energy_basis='compute' and active_joules_per_token=0.022744 -> 0.022744 x 385 = 8.756 J. 18.019 vs 8.756 is +106 % (docs/42 measured +74 %).
```

*Why it matters:* docs/10 §5 tells the student to spot-check a published row with energy_window.py. When the two tools differ by 7–13 %, a spot-check can never distinguish a real error from tool skew — and the one the thesis publishes is the biased one. The 106 % gap between the two 'active' figures printed in the same block is worse: the human-readable printer puts a whole-window active energy directly above a compute-basis J/tok line, and anyone reconstructing active-J/token from what is on screen gets a number twice too high.

*Fix:* Pick one anchor and one duration and use them in both tools: the last power sample inside the window, and the sampled duration of the trimmed slice. In energy_window.py, when the trim succeeds also overwrite res['module_active_joules'] (line 378 rebinds the local `active` but leaves the dict entry stale) or emit it as compute_active_joules with its scope named. Add a regression step that runs energy_window.py over a driver-produced power CSV and asserts the two joules_per_token agree to 1e-6.

**[HIGH] Fix #7 covered `open()` but not the three subprocess text pipes, which still decode with the locale encoding and strict errors. On real Python 3.6.9 under LC_ALL=C (ASCII) one non-ASCII byte from llama-bench's stderr aborts the sweep on the first run. The regression test that claims 'text I/O does not depend on the locale' explicitly skips any line containing 'subprocess'.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:652`

*Reproduction:*
```
Ran it, under real CPython 3.6.13 (`locale.getpreferredencoding(False)` == 'ANSI_X3.4-1968' under LC_ALL=C, confirmed):

  fake llama-bench writes u"ggml: device — NVIDIA Tegra X1\n" to stderr, then a valid CSV to stdout.
  $ LC_ALL=C python3.6 bench/run_matrix.py --env-file $FIX/node.env --matrix $FIX/pref.json --tag utf --no-power
  [1/2] pref|qwen2.5-1.5b-instruct-q4_k_m|ngl99|t4|p512|g0
  Traceback (most recent call last):
    File "bench/run_matrix.py", line 652, in main
      universal_newlines=True, check=False)
    ...
  UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2 in position 13: ordinal not in range(128)
  Same command under LC_ALL=C.UTF-8: rc=0, '2/2 rows ok'.

Second site, load_env (:64), same interpreter and locale:
  node.env with the em dash in a COMMENT   -> OK (comments never reach `env` output)
  node.env with LLAMA_CPP_REF="b3500 — pinned for CUDA 10.2" -> UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2 in position 301

And the LIVE /mnt/data/KMITL/Final/HeteroServe/Jetson/config/node.env still carries U+2014 on lines 1 and 52 (`grep -nP '[^\x00-\x7F]'`), even though node.env.example and models.tsv were de-unicoded. Harmless today only because both are comments.

Why the regression cannot see any of this — run_regression.sh:303-304:
  if re.search(r"\bopen\(", line) and "encoding=" not in line \
     and '\"rb\"' not in line and "subprocess" not in line:
The 'subprocess' exclusion (there to skip `Popen(`) exempts exactly the calls that still use the locale.
```

*Why it matters:* This is the same trap fix #7 was written for, one layer down, and 3.7+ coerces C to C.UTF-8 (PEP 538) so no workstation ever sees it — only the Jetson does. Nothing in the kit sets a locale: scripts/, systemd/, node.env.example and the Makefile contain no LANG/LC_ALL at all, and systemd units start with no locale environment. llama.cpp prints GGUF metadata (general.name, tokenizer strings, chat templates) and micro-second signs on stderr, and the README itself tells the operator to record the llama.cpp pin as free text in node.env. The section header 'text I/O does not depend on the locale' is a stronger claim than the test behind it.

*Fix:* On 3.6 `subprocess.run`/`check_output`/`Popen` all accept `encoding=` and `errors=`. Replace `universal_newlines=True` with `encoding='utf-8', errors='replace'` at run_matrix.py:64 and :652 and tegrastats_logger.py:158-162 — 'replace' rather than 'utf-8' alone, since a ref could emit latin-1. Then narrow the regression exclusion to `'Popen(' not in line` so subprocess text pipes are actually audited, and add a fixture whose stderr carries a non-ASCII byte.

**[HIGH] docs/42's headline — 'Status as of 2026-09-13 12:0x — all of it is now fixed. Every finding below has been addressed and locked into bench/fixtures/run_regression.sh (30 passed, 0 failed)' — is false. Five findings the same document explicitly marks STILL LIVE are still live in the tree, plus three of its doc-fix items. The status table lists only 8 of the document's 36 findings.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/docs/42_finding3_verification.md:12`

*Reproduction:*
```
Ran each against the live tree (md5 20da70255299e81de2bc8b0a78e0fb8e) under CPython 3.6.13:

1. docs/42:650 'a fused row carries two mutually inconsistent active figures' — STILL LIVE. My fused run: active_joules=22.1052, active_joules_per_request=5.0141, requests=3 -> 5.0141 x 3 = 15.04, not 22.11 (a 47 % gap). Cause unchanged: run_matrix.py:754 fixes active_joules on the compute basis before :785 overwrites energy_basis to 'timed requests'.
2. docs/42:637 'row_compute_seconds() uses warm_ns = ns for a fused row' — STILL LIVE (run_matrix.py:351-352 still `warm_ns = ns`). Measured with the now-upstream-accurate fake: compute_s=6.0 against a truth of 0.5x4 + 1.0x3 + 1.0/128 = 5.008 s (+19.8 %); load_joules=10.4244 against a truth of ~15.6 J (-33 %).
3. docs/42:676 'energy_window.py emits module_active_joules over the WHOLE window next to energy_basis=compute' — STILL LIVE, measured +106 % (see the twin-disagreement finding).
4. docs/42:689 'active_joules subtracts the idle floor over the NOMINAL compute_s' — STILL LIVE, measured (see the compute_joules finding); far larger than the [LOW] it is filed under.
5. docs/42:715 'idle_floor() pools the 30 s baseline with every cooldown tail' — STILL LIVE by code read (run_matrix.py:848-860 averages every sample with marker=='idle'); contradicts docs/10:82 'taken from the 30 s idle baseline the driver collects before the sweep'. Not reproduced numerically — the fake returns a flat 1900 mW for idle, so there is no thermal tail to measure.
6. docs/42:485 'energy_basis ... the code now writes three distinct values' — docs/10:185 still reads '| `energy_basis` | which of the two the per-unit columns were built from |'; my fused row came back energy_basis='timed requests'.
7. docs/42:563 'bench/fixtures/README.md says HS_FAKE_LOAD_S is what load_joules should come back as' — README.md:66 still says it; it is seconds, and load_joules came back 15.86 J for LOAD_S=3.
8. docs/42:576 'README Layout omits every file this change added' — README.md:65-99 still omits bench/check_finding3_invariants.py, bench/fixtures/*, telemetry/make_synthetic_power_csv.py and docs/40/41/42.

What the header IS right about: I reproduced '30 passed, 0 failed' verbatim under 3.6.13 with LC_ALL=C.
```

*Why it matters:* This document is the audit trail. A blanket 'all of it is now fixed' over a list that contains its own STILL LIVE labels means the next reader cannot tell which findings were closed and which were merely written down — and the two biggest remaining numeric errors (items 1-4) are exactly the ones that touch the published energy columns. The 'locked into run_regression.sh' clause is the load-bearing part and it is not true: none of items 1-8 has a regression assertion.

*Fix:* Replace the blanket sentence with a per-finding status column (fixed / still live / doc-only), and move the six STILL LIVE items into an open list with their line numbers. Then apply the three one-line doc fixes (docs/10:185 naming all three energy_basis values, bench/fixtures/README.md:66 seconds-not-joules, README's Layout tree) since they cost nothing.

**[MEDIUM] joules_per_request does not reproduce the 8.5 J the fixture's own physics implies, and the regression assertion that checks it (`abs(jpr - 8.5) < 0.6`, a ±7 % tolerance) is flaky. Measured 7.86–8.36 across 10 runs, always low, bimodal on whether one 0.2 s power sample falls inside the requested window. The load-independence half of the claim does hold.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/fixtures/run_regression.sh:204`

*Reproduction:*
```
Ran it, under CPython 3.6.13. Truth from the fake's own mW table: 3 timed reps x (0.5 s @ 7400 mW + 1.0 s @ 4800 mW) / 3 = 8.50 J/request.

Six runs of a single-case fused matrix, applying the regression's own assertion:
  run 1 jpr=7.8631 request_joules=23.5893  |8.5-jpr|=0.6369  assert(<0.6)=False
  run 2 jpr=8.3571 request_joules=25.0712  |8.5-jpr|=0.1429  assert=True
  run 3 jpr=8.3711 request_joules=25.1133  |8.5-jpr|=0.1289  assert=True
  run 4 jpr=7.8628 request_joules=23.5883  |8.5-jpr|=0.6372  assert=False
  run 5 jpr=8.3621 request_joules=25.0863  |8.5-jpr|=0.1379  assert=True
  run 6 jpr=7.8608 request_joules=23.5824  |8.5-jpr|=0.6392  assert=False
  -> 3 of 6 would have failed the shipped assertion. 25.09 - 23.59 = 1.50 J = exactly one 0.2 s sample at 7.4 W.

Four runs of the actual matrix_split.json the regression uses were stable at 8.3581 / 8.3595 / 8.3590 / 8.3590 (still -1.7 % vs truth), so with that specific matrix the test happens to sit on the right side of the grid — the instability is real, the shipped arrangement just masks it.

Load-independence VERIFIED: HS_FAKE_LOAD_S=3 -> jpr=7.8641, HS_FAKE_LOAD_S=12 -> jpr=7.8632 (0.01 %), with joules moving 43.93 -> 90.27 and load_s 1.823 -> 10.835. request_window_s = 4.5 s = avg_ns(1.5e9) x reps(3) exactly, in every run.

Root cause is the same edge truncation: the driver asks for [t_end-4.5, t_end] but energy_between returns the counter difference between the first and last sample inside it — an actual span of 4.209 s in the run I traced sample by sample — and then divides by 3 requests as if 4.5 s had been measured.
```

*Why it matters:* joules_per_request is the headline number for the two shipped end_to_end_request runs, and it is the one figure in the study measured over a window the driver chose rather than one the instrument delimited. A -1.7 % to -7.5 % bias that flips with sample-grid alignment is not a property a reader can see on the row, and the test that is supposed to catch it is a coin flip whose tolerance (±7 %) is wider than the error it is meant to detect.

*Fix:* Same root fix as the compute-window finding: divide by the SAMPLED duration of the request slice, or interpolate the energy counter to the exact t0/t1. Then tighten run_regression.sh:204 to something like 2 % and add the second assertion docs/42 already suggests — run the fused case at two HS_FAKE_LOAD_S values and assert the two J/request agree (that part already holds, so it would lock in real behaviour instead of a wide band).

**[MEDIUM] `energy_basis` emits three distinct values but docs/10's column table says 'which of the two'; and on a combined row the label is wrong for the columns next to it — energy_basis reads 'timed requests' while `joules`, `compute_joules` and `active_joules` on that same row are on the whole-process and compute bases.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/docs/10_measurement_protocol.md:185`

*Reproduction:*
```
Ran it. My fused row (LOAD_S=3), every non-empty cell:
  energy_basis=timed requests  joules=43.9296  compute_joules=33.5052  active_joules=22.1052
  compute_s=6.0  request_window_s=4.5  request_joules=23.5924  joules_per_request=7.8641  active_joules_per_request=5.0141
  22.1052 = 33.5052 - 1.9 x 6.0  -> active_joules is on the COMPUTE basis, not the timed-request basis the label names.
  5.0141 x 3 = 15.04 != 22.1052.
grep of the live tree: run_matrix.py:747 'compute', :753 'whole-process (includes model load)', :785 'timed requests'. docs/10:185 documents two. The prose at docs/10:220-222 also names only 'compute' and 'whole-process (includes model load)'.
The invariant checker does not catch it either: check_finding3_invariants.py:170 picks `src = 'compute_joules' if basis.startswith('compute') else 'joules'`, so for basis='timed requests' it falls back to `joules` — and since joules_per_token is (correctly) absent on a combined row, rule 8 never runs.
```

*Why it matters:* energy_basis is the single column a reader is told to consult to know what the numerator is. A reader who sees 'timed requests' has no table entry to look it up under, and if they take active_joules from the same row as the timed-request active energy they are 47 % high. This is a labelling failure on the one case in the matrix that deliberately spans both phases.

*Fix:* docs/10:185 -> 'which slice of the window the per-unit columns were built from: `compute` (the trailing compute_s), `timed requests` (a fused row's avg_ns x r), or `whole-process (includes model load)` when no trim could be made — treat the last as an upper bound.' In run_matrix.py, on a PT_COMBINED row either recompute active_joules from (request_joules, request_window_s) or drop active_joules entirely and keep only active_joules_per_request. Add a checker rule: active_joules_per_request x requests == active_joules.

**[LOW] `model_path` is a real emitted column documented nowhere, and because of the KeyError above it can only ever appear in a `--no-power` CSV — so the one artefact that carries it is the one the protocol says is not a measurement.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:621`

*Reproduction:*
```
Ran it. `--no-power` header, 42 columns, ends: ... row_compute_s, request_window_s, model_path. Power-on header, 63 columns, has no model_path (no run can be skipped without crashing first). `grep -rn model_path docs/ README.md bench/fixtures/README.md` returns only docs/42's quoted source snippet — no entry in the docs/10 §4 column table, which lists request_joules, request_window_s, run_index and window_shared but not model_path. (docs/42:595 claims '61 columns from a fixture sweep'; the current header is 63.)
```

*Why it matters:* Minor on its own, but model_path is the column that says WHICH file was looked for when a run was skipped — the only thing that lets someone six months later tell a typo in models.tsv from an incomplete download. Documenting it costs one table row, and it is the piece of provenance the skip record exists to carry.

*Fix:* Add to the docs/10 §4 column table: '| `model_path` | the GGUF the driver looked for; present on `status=skipped` rows so a missing-model gap names the file it wanted |'. Put model_path in the LEAD list so it does not float to the end of the header. And fix the KeyError so it can appear in a real sweep at all.

**[LOW] bench/fixtures/README.md tells the reader HS_FAKE_LOAD_S 'is what load_joules should come back as'. It is seconds; load_joules is joules; and even load_s comes back short because compute_s is overstated for a fused row.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/fixtures/README.md:66`

*Reproduction:*
```
Ran it. HS_FAKE_LOAD_S=3 -> prefill row: load_s=2.81, load_joules=15.8568 J; fused row: load_s=1.823, load_joules=10.4244 J against a truth of 3.0 s @ 5200 mW = 15.6 J. HS_FAKE_LOAD_S=12 -> fused load_s=10.835, load_joules=57.3206. Neither number equals 3 or 12.
```

*Why it matters:* This is the line a reader uses to sanity-check the fixture, so it teaches a wrong unit at exactly the moment someone is deciding whether the harness is trustworthy. It also hides the fused-row load_s understatement (-40 % here), since a reader who expects joules will not notice that load_s itself is wrong.

*Fix:* 'seconds spent in the `load` phase before the timed loops (default 3.0); it should show up as `load_s` on a split row, and `load_joules` should be roughly that many seconds at the fake load power of 5200 mW. On a `combined` row `load_s` comes back short — see docs/42.'

### Attacks that FAILED — evidence those guards hold
- Python 3.6 syntax: every one of the 12 .py files compiles under real CPython 3.6.13 (py_compile with doraise, whole-tree sweep, zero failures). No walrus, no dataclasses, no `a | b` dict merge, no f-string `=` specifier, no positional-only `/`, no `from __future__` needed.
- Python 3.6 stdlib: grepped and ran — no `datetime.fromisoformat` (energy_window.py:53 documents avoiding it deliberately and hand-rolls strptime with the `+00:00` -> `+0000` normalisation 3.6's %z needs), no `subprocess` `text=`/`capture_output=`, no `math.prod`, no `importlib.metadata`. `{**run, ...}` is PEP 448 (3.5), `isoformat(timespec=)` is 3.6, `print(flush=)` is 3.3.
- `pick(row, *names, skip_zero=False)`: keyword-only arguments after *args are PEP 3102, Python 3.0 — not a 3.7+ feature. Verified it runs: the regression's `pick({'n_prompt':'0','n_gen':'128'}, 'n_prompt','n_gen', skip_zero=True) == '128'` assertion passes under 3.6.13.
- The full regression under a real 3.6: `LC_ALL=C PATH=<3.6 shim> ./bench/fixtures/run_regression.sh` -> **30 passed, 0 failed**, rc=0, and it did not touch config/node.env (its own md5 check confirmed 348ebc26c5bb48abf26e23ddfe56527d unchanged).
- Dict insertion-order reliance in the CSV header builder: the 63-column header is byte-identical produced by CPython 3.6.13 and by CPython 3.12 on the same matrix. Column order holds.
- per_token_status completeness, both directions: the 9 PT_* constants in run_matrix.py, the 9 rows of the docs/10 table, and check_finding3_invariants.py's KNOWN_STATUS are an exact three-way match. `code - docs`, `docs - code`, `code - checker`, `checker - code` are all empty.
- 'ok always means a non-empty joules': enforced. The demotion loop at run_matrix.py:797-799 sits outside the `if os.path.exists(power_csv)` guard and runs over every record including skipped and failed ones — and skipped/failed can never reach PT_OK anyway (per_token_status returns PT_MODEL_MISSING / PT_RUN_FAILED first). Verified end to end: a `--no-power` sweep gives 'per_token_status no_power_window 2' and 'model_missing 1', never 'ok'.
- '32 runs, combined 2, decode 11, prefill 19': exact. Counted the matrix by hand (4+2+4+2+5+5+2+2+4+2 = 32) and reproduced the driver's own line `phases    : combined 2, decode 11, prefill 19` from --dry-run.
- The `--reps` warning direction and magnitudes: computed under 3.6 — prefill 512x(3+1)/(512x2) = 2.0000, decode (128x3+1)/(128x1+1) = 2.9845. The docs' '2.0x prefill, 3.0x decode at -r 3' and the too-HIGH direction are both right, in energy_window.py's stderr note, its --reps help text, and docs/10 §5.
- tokens_of() at every r: prefill = n_p x (r+1), decode = n_g x r + 1, fused = n_p x (r+1) + n_g x r + 1, and prefill + decode == fused at r = 1, 2, 3, 5 and 10 (1024+129=1153, 1536+257=1793, 2048+385=2433, 3072+641=3713, 5632+1281=6913). The split really does change no measured work.
- request_window_s = avg_ns x reps, exactly: 1.5e9 x 3 / 1e9 = 4.500 in every fused run, and requests_of() returns the timed repetitions (3), not reps+1. Both return None on a non-combined row.
- joules_per_request is load-independent: 7.8641 at HS_FAKE_LOAD_S=3 vs 7.8632 at HS_FAKE_LOAD_S=12 (0.01 %), while `joules` moved 43.93 -> 90.27 and load_s 1.823 -> 10.835. The timed-repetition window really does exclude the load and the warm-up. (It is biased low for a different reason — see the findings.)
- active_joules on the whole-process basis IS consistent: when the compute trim is refused, active = joules - idle_mw x duration_s, and duration_s is the sampled span energy_between actually measured. Only the compute and timed-request bases mismatch.
- Every guard the follow-up fixes added fires: HS_FAKE_HIDE_PROMPT -> refused_phase_mismatch with phase=combined/row_phase=decode and no per-token OR per-request figure; HS_FAKE_EXTRA_ROW -> both rows refused_multi_phase with no per-token column written at all; HS_FAKE_NO_COUNTS+EXTRA_ROW -> refused_multi_phase with window_shared=1 (row-count driven, not parse-success driven); expand() refuses all four malformed case shapes by name at --dry-run.
- energy_window.py's --bench guard and prefix matching: `--bench` with no window exits non-zero; a 2-row bench CSV against 1 window is refused; `--marker decode:x:001` against a CSV with no such label is refused; `--marker decode:sd:002` against a real sweep CSV resolves to exactly one window. parse_time resolves all four spellings to one value under 3.6.
- config/models.tsv and config/node.env.example are pure ASCII (byte-checked). telemetry/energy_window.py's source does contain U+2014, but .py source is decoded as UTF-8 regardless of locale (PEP 3120), so that is a non-issue — I confirmed it imports fine under LC_ALL=C on 3.6.13.
- The shipped fake_llama_bench.py's -pg branch now enacts the upstream warm-up (one prompt pass + TG_NS/pg_g for a single token, then reps of prompt+generate), so docs/42's [LOW] 'the fixture cannot detect any fused-path error' is genuinely fixed — that is how I was able to measure the fused-path errors at all.
- POM_5V_IN scope: no live file claims module-only. docs/10 §2, README's honesty bullets and tegrastats_logger.py's docstring all say whole-devkit and all say there is no external meter and no across-device comparison. Consistent everywhere.
- Separate stdout/stderr capture and bench_csv_text() preamble stripping both still work — the fake writes a CUDA banner to stderr and it never reaches a case CSV; bench_csv_text('no csv here') returns ''.
