# Finding #3 fix — adversarial verification

Five independent lenses attacked the finding-#3 fix on 2026-09-13, each told to
REFUTE rather than confirm. **All five returned `fix-holds-with-caveats`:** the
core mechanism survived every attack on the matrix side, but the ref side (what
happens when `llama-bench` returns something unexpected) has real holes.

The Critic phase had not returned when the session was cut off, so nothing below
has been through a second opinion. Findings marked SNAPSHOT were already fixed
during the session and are kept only so the history reads straight.

## Status — first-pass findings fixed, then a SECOND pass found more

The eight findings below were fixed and locked into
`bench/fixtures/run_regression.sh`. A third adversarial pass
(`wf_ea30ba50-e68`, three lenses) was then run against those fixes and returned
**`broken` from all three lenses** — see `docs/43_finding3_verification2.md`.
All three independently found the same CRITICAL: fix #4 below, the one that makes
a missing model emit a record, crashed the join with `KeyError: 't_start_unix'`
whenever power logging was on, so a completed sweep lost its entire joined CSV.
The regression could not see it because its model-missing test passed
`--no-power`.

Treat the table below as "the first pass's findings were addressed", not as
"this area is now correct". The second pass's findings are tracked separately.

| Finding | Fix |
|---|---|
| CRITICAL: a `-pg` row reporting `n_prompt=0` scored as decode and published J/token | `per_token_status()` compares the phase the driver ASKED for against the phase that came BACK, and refuses on mismatch (`refused_phase_mismatch`) |
| HIGH: `phase_rows()` → 0 working rows defeats the `> 1` guard | a multi-row window is now refused unless EXACTLY ONE row did identifiable work; `window_shared` is driven by row count, not parse success |
| HIGH: `per_token_status=ok` on rows with no energy | the `no_power_window` demotion was hoisted out of the `if os.path.exists(power_csv)` guard. `ok` now always implies non-empty `joules`, and the checker asserts it |
| HIGH: `--bench` with no window joins the whole CSV | `--bench` now requires `--marker` or `--from/--to`; `--marker` gained `phase:case_id:NNN` prefix matching so the sweep's own labels are usable |
| HIGH: the `--reps` warning stated its error backwards | corrected in `energy_window.py` and `docs/10_measurement_protocol.md`: too low a `--reps` makes J/token too **HIGH** (2.0× prefill, 3.0× decode at `-r 3`) |
| HIGH: `open()` without `encoding=` dies on 3.6 under `LC_ALL=C` | all 16 text `open()` calls pin `encoding="utf-8"`; `config/models.tsv` and `node.env.example` de-unicoded. A regression test greps for any bare `open()` returning |
| HIGH: a run with a missing model produced no row at all | it now emits one row with `status=skipped` / `per_token_status=model_missing` |
| HIGH (fixed before the lens reported): the fixture clobbered `config/node.env` | `run_matrix.py` gained `--env-file`; the regression fingerprints the real file and fails if it changes |

The two SNAPSHOT findings were already fixed during the session that ran this
verification. **The Critic phase never returned**, so nothing below has had a
second opinion.

---

Reproductions are quoted as the lens ran them. Several used a real CPython 3.6.13
or a purpose-built fake `llama-bench`; those say so.

---

## Lens: `regression` — fix-holds-with-caveats

Every prior fix the brief named (#1, #2, #7/#8/#9, #11/#15, #19) survives this edit, and I confirmed each by running it rather than reading it: the CUDA banner still never reaches a case CSV (I also fed the driver a ref that leaks a preamble onto stdout, which bench_csv_text still strips), the compute_s trailing trim still fires on single-row windows with compute_joules/load_joules/energy_basis all present and arithmetically correct, making compute_seconds() a sum over row_compute_seconds() changes the single-row result by exactly nothing across a 64-point grid, the divisor identities hold at every (n_p, n_g, r) I tried and the split halves sum to the pre-split count, parse_time resolves 12 spellings to one value and every .py still parses at feature_version=(3,6), and no live file claims module-only POM scope or an external meter. The regressions I did find are in the new harness and the prose around it, not in the measurement path: the new run_regression.sh destroys the operator's live config/node.env (reproduced, and happening in the shared tree right now), a model-missing run still vanishes from the joined CSV while the docs newly promise it cannot, and per_token_status reports "ok" on rows that have no energy at all. NOTE: the shared tree changed under me twice during this audit (another agent is revising the -pg warm-up semantics), so cite the quoted code, not the line numbers.

### Findings (6)

**[HIGH] bench/fixtures/run_regression.sh — the new 19/19-passing harness — writes the operator's live config/node.env and then deletes it unconditionally on exit, destroying irreplaceable machine config. This is the fourth-plus repeat of a clobber a peer session explicitly asked to stop, in writing, in this repo's own newest handoff.**

`bench/fixtures/run_regression.sh:23`

*Reproduction:*
```
cp -a <shared tree> $(mktemp -d)/kit3; cd kit3; wrote a realistic real config:
  # REAL machine config -- irreplaceable, gitignored\nNODE_ID=jetson-nano-01\nETH_ADDR=192.168.50.15\nINA_CHANNEL=3\nLLAMA_BIN="$HOME/llama.cpp/build/bin" ...
  md5sum -> 6a1eb7277564e83226d2d3a37c71087e
then ./bench/fixtures/run_regression.sh  ->  '19 passed, 0 failed', rc=0
ls -la config/  ->  cluster.hosts.example, models.tsv, node.env.example ONLY
cat config/node.env -> 'No such file or directory'

And live, in the shared tree, twice while I audited (I only read it):
  12:09:58  config/node.env = NODE_ID=fixture, LLAMA_BIN="/tmp/tmp.EEHGPfvGMy/bin" ...
  12:14     config/node.env = NODE_ID=fixture, LLAMA_BIN="/tmp/tmp.1eeISkXVOn/bin" ...
(run_regression.sh mtime 12:09:58 — same second.) Confirmed still present in the live tree: diff of my copy vs the shared tree shows only the unit-test block changed; line 23 is untouched.
```

*Why it matters:* config/node.env is in .gitignore, so it is machine-local and unrecoverable — there is no copy to restore from. .handoff/2026-09-13-0110-finding3-mixed-phase.md carries two separate notes from the ThinkPad session (Notes 1 and 2) asking for exactly this to stop: 'config/node.env is live machine config, the clobber is silent, and it is the fourth write to that path today', 'please point the fixture at a tempfile.TemporaryDirectory()'. This edit answered that by shipping a script that does it again, plus an rm -f that the earlier ad-hoc fixtures did not have. The same note records the downstream cost: ThinkPad/tests/test_repo_hygiene.py::TestClusterAddressing is red (62/64) purely because of this file, and a genuine new ThinkPad failure can hide behind the two standing ones. On the Nano itself, losing node.env costs MODEL_DIR, LLAMA_BIN, NVP_MODEL, FAN_PWM and the (still-blank) LLAMA_CPP_REF — the exact conditions docs/30_experiment_log.md says must be recorded per experiment. The cleanup() also runs on the EXIT trap, so Ctrl-C deletes it too.

*Fix:* Root cause is that load_env() has no override: envfile = os.path.join(JETSON_DIR, "config", "node.env") with nothing to point it elsewhere. Add one — an --env argument plus HS_NODE_ENV fallback — then have run_regression.sh and the bench/fixtures/README.md recipe (line 31, same pattern) write into $FIX and pass --env "$FIX/node.env". If the fixed path must be used at all, cleanup() has to save any pre-existing file first and restore it, never rm -f. Either way the script should refuse to start when config/node.env exists and does not say NODE_ID=fixture.

**[MEDIUM] A run whose GGUF is missing produces ZERO rows in the joined CSV and leaves no trace in the run log — the same silent-vanishing failure the fix set out to eliminate for rc!=0 and no-CSV runs — while docs/10_measurement_protocol.md now states outright that it cannot happen.**

`bench/run_matrix.py:574`

*Reproduction:*
```
Matrix with 3 prefill cases; only the first model present on disk.
  python3 bench/run_matrix.py --matrix missing.json --tag missingtest --no-power
  runs      : 3
  [1/3] have_it|... pp512 1024.00 t/s 2048 tok
  [2/3] SKIP missing_model|qwen2.5-3b-instruct-q4_k_m|... (model missing)
  [3/3] SKIP unknown_name|not-in-models-tsv|... (model missing)
  1/1 rows ok (1 invocations)
    per_token_status ok   1
joined CSV: row count: 1  (only have_it)
run.log: contains only the have_it block — 'SKIP' appears nowhere in it.
check_finding3_invariants.py on that CSV: 'all finding-#3 invariants hold', rc=0.
```

*Why it matters:* docs/10_measurement_protocol.md:120 now says 'Every run produces at least one row. A run that failed ... or emitted no parseable CSV produces exactly one row'. That is false for the skip path, and the summary compounds it: '1/1 rows ok (1 invocations)' reads as a clean 100% sweep with no mention that a third of the planned runs never ran. The handoff's own 'Decisions already made' says 'failed and unparseable runs vanish from the output entirely. Failed runs must still emit exactly one record' — the rc!=0 and no-CSV paths got that treatment; the skip path did not. With the shipped matrix this is not hypothetical: 10 of the 32 runs need Qwen2.5-1.5B and 2 more need the Q8_0, so a student who downloaded only the 0.5B Q4 gets a joined CSV missing the entire partial_offload placement sweep, with nothing in any artefact saying so once the terminal scrollback is gone.

*Fix:* In the model-missing branch, emit the same shape of record the rc!=0 branch does — status='skipped_model_missing', a new PT_ status, bench_rows/working_rows 0 — before continue, and write the skip line into logfh as well as stdout. Then either make docs/10:120 true or narrow it to 'every run that was attempted'.

**[MEDIUM] per_token_status stays 'ok' on every row of a sweep that collected no power at all, because the PT_NO_POWER reclassification lives inside `if os.path.exists(power_csv)`. scripts/50_bench_device.sh adds --no-power automatically off-Jetson, so this is the default state of every rehearsal run.**

`bench/run_matrix.py:707`

*Reproduction:*
```
python3 bench/run_matrix.py --matrix missing.json --tag missingtest --no-power
  !! no power logging (tegrastats missing or --no-power)
  per_token_status ok   1
joined row: case_id=have_it status=ok per_token_status=ok tokens=2048 joules=None energy_basis=None
python3 bench/check_finding3_invariants.py <that csv>  ->  'all finding-#3 invariants hold', rc=0
```

*Why it matters:* docs/10_measurement_protocol.md defines ok as 'the per-token columns are populated and trustworthy', and the whole point of the enum (per its own comment: 'Blank numeric cells already mean at least five different things ... Every row carries one of these') is that a blank per-token cell must never be ambiguous. A --no-power CSV is exactly the case where every per-token cell is blank, and every row claims ok. PT_NO_POWER exists for it and is simply never reached. The invariant checker does not catch it either: rule 7 is `if status == 'ok' and e is not None and jpt is None`, and e is None. scripts/50_bench_device.sh:40 does `set -- "$@" --no-power` whenever is_jetson is false, so the off-board dry run — the run most likely to be eyeballed as a template for what the real output looks like — is the one that lies.

*Fix:* Move the status demotion out of the power-CSV branch: after the join block, for every record still marked PT_OK that has no joules, set PT_NO_POWER (and when --no-power was passed, set it on all of them up front). Add the mirror assertion to check_finding3_invariants.py: per_token_status=='ok' with no joules_per_token is a violation regardless of whether energy is present.

**[LOW] The 'How to verify' smoke test in .handoff/2026-09-12-1541-apply-audit-findings.md still imports the deleted bench_tps and dies at import. It is the only dangling reference to anything this edit renamed or removed — and nothing anywhere in the kit points at run_regression.sh, the harness that replaced it.**

`.handoff/2026-09-12-1541-apply-audit-findings.md:372`

*Reproduction:*
```
Ran the handoff's snippet verbatim:
  from run_matrix import bench_csv_text, bench_tps, compute_seconds, tokens_of
  ImportError: cannot import name 'bench_tps' from 'run_matrix' (...). Did you mean: 'bench_int'?  rc=1
grep -rn 'bench_tps' over the whole kit: run_matrix.py has none; energy_window.py's four hits are the unrelated OUTPUT COLUMN res['bench_tps']; docs/40 and docs/41 are historical narrative; .handoff/2026-09-12-1541:372,377 is the only live caller.
grep -rn 'run_regression' over the whole kit: one hit, the script's own usage line (bench/fixtures/run_regression.sh:8). Not in README.md, not in Makefile's `check` target, not in bench/fixtures/README.md, not in docs/.
Column check, for completeness: diff of BEFORE_mixed_bench_energy.csv's header against a fresh joined CSV — REMOVED: [] , ADDED: 21 columns. Purely additive, and grep shows no script or Makefile target reads any CSV column name.
```

*Why it matters:* docs/41_finding3_analysis.md:1077 anticipated this precisely — 'Do not edit the historical handoff. Either keep bench_tps ... or carry a corrected snippet into the new handoff and mark the old one superseded' — and neither half happened: bench_tps is gone and no handoff was written for this session. The newest handoff on disk (2026-09-13-0110) still says 'No fix has been applied yet' and 'bench/run_matrix.py unchanged'. So a /pickup lands on a handoff that misdescribes the tree, walks back to the 1541 handoff's verification block, watches it fail at import, and has no way to discover that a 19-check harness exists. `make check` does not run it either — it only py_compiles and json-loads.

*Fix:* Write the session's handoff, state that the 1541 snippet is superseded and why, and give run_regression.sh as the replacement. Reference it from README.md and bench/fixtures/README.md, and consider wiring it into a `make regression` target so it is reachable without prior knowledge.

**[LOW] build_cmd()'s comment states the opposite of what the four lines under it do, and contradicts both phase_rows()'s docstring and the kit's own audit. Acting on the comment would silently put three tests in the new -pg case's power window.**

`bench/run_matrix.py:239`

*Reproduction:*
```
Read, not executed. run_matrix.py:239 reads '# llama-bench rejects -p 0 / -n 0 on some refs; only pass the axis in play.' — and :240-247 then always emit both, using "0" for the idle axis. phase_rows()'s docstring says the opposite of the comment: 'build_cmd() always passes both -p and -n, using 0 for the idle axis'. docs/40_audit_findings.md:98 records the upstream source check: '-p 0/-n 0 are accepted (get_cmd_params_instances skips them with if (n_prompt == 0) { continue; })' — so the comment's premise is false too. Confirmed the flags are actually emitted, from my fixture run log: '-o csv -r 3 -t 4 -ngl 0 -p 512 -n 0'.
```

*Why it matters:* The explicit -p 0 -n 0 is load-bearing for the case this edit introduced. run_matrix.py:248-250 spells out why: 'The -p 0 -n 0 above is what stops llama-bench also running its default standalone pp and tg tests, which would put three rows in this window.' A maintainer who trusts the comment at :239 and drops the else-branches leaves params.n_prompt/n_gen empty, llama-bench substitutes its defaults {512}/{128}, and end_to_end_request becomes pp512 + tg128 + pp512+tg128 in one window. The driver does catch that (working_rows=3 -> refused_multi_phase), so it is a lost measurement rather than a wrong number — but the comment is an active invitation to make it happen.

*Fix:* Replace :239 with what the code actually does and why: '-p 0 / -n 0 are accepted and skipped by get_cmd_params_instances; pass both explicitly so the axis not in play cannot fall back to llama-bench's defaults (512 / 128) and add tests to this window. Load-bearing for -pg -- see below.'

**[LOW] A single bench row whose own phase contradicts the window's request is published as `ok`, with a per-token figure, under a marker naming the other phase. phase_rows() counts rows; nothing compares phases.**

`bench/run_matrix.py:412`

*Reproduction:*
```
python3 -c importing bench_records: driver asked for prefill (-p 512 -n 0), ref returns ONE tg128 row.
  rec = bench_records({'repetitions':3,'n_prompt':512,'n_gen':0},
                      [{'n_prompt':'0','n_gen':'128','test':'tg128','avg_ts':'9.9','avg_ns':'1000000000'}],
                      {'status':'ok','phase':'prefill'})[0]
  -> window phase = prefill   row_phase = decode   tokens = 129   per_token_status = ok   working_rows = 1
So joules_per_token would be written from a window whose marker string is 'prefill:<case>:NNN', over a decode divisor, and check_finding3_invariants.py has no rule comparing phase to row_phase.
```

*Why it matters:* docs/10_measurement_protocol.md:166 tells the reader that phase and row_phase 'differ only if a ref returned something unexpected' — which is exactly the condition that carries no status, no console warning and no invariant. The row lands in the CSV indistinguishable from a clean one, and the marker label is what docs §5 tells the analyst to pass to energy_window.py, so the mislabelling propagates into the ad-hoc analysis path too. The handoff's 'Decisions already made' lists this predicate as the alternative that was weighed ('Consider keying on "the row\'s own counts disagree with the window\'s phase" instead. Decide deliberately'); the row-count predicate was chosen for good reason, but nothing was added to cover what it does not see.

*Fix:* In per_token_status(), return a new PT_PHASE_MISMATCH when rec['row_phase'] not in (rec['phase'], 'none') — cheap, since both are already on the record — print it like the multi-phase warning, and add the matching assertion to check_finding3_invariants.py so the CSV cannot carry a silent phase disagreement.

### Attacks that FAILED — evidence the fix holds
- Finding #1, stdout/stderr separation: ran the driver against a ref that writes the ggml_init_cublas banner to stderr — banner lands in <base>_run.log, all five <base>_NNN.csv files start with the real "build_commit",... header. Then built a harsher ref that leaks 'ggml_init_cublas: found 1 CUDA devices:' AND 'llama_model_loader: loaded meta data with 26 key-value pairs' onto STDOUT before the CSV: bench_csv_text() still stripped the preamble, case CSV line 1 is the header, joined row avg_ts=1024.0 test=pp512 tokens=2048 status=ok. Could not get a log line into a case CSV.
- Finding #2, the compute_s trailing trim on a single-row window: real driver run, five windows. Every row carries compute_s, row_compute_s, joules, compute_joules, load_s, load_joules, active_joules and energy_basis='compute'. Numbers check out against the fixture's ground truth: prefill compute_s=2.000 (fake sleeps (0.5e9*3+0.5e9)/1e9), decode compute_s=3.008 (matches the 3.008 the 1541 handoff recorded), load_s 2.80-3.00 against HS_FAKE_LOAD_S=3.0, joules_per_token = compute_joules/tokens = 13.3343/2048 = 0.006511 exactly, active_joules = 13.3343 - 1.9*2.0 = 9.5343 exactly.
- Finding #2, whether making compute_seconds() a sum over row_compute_seconds() moved the single-row answer: swept avg_ns in {1e9, 5e8, 1.23456e9, 4.1e7} x reps {1,2,3,5} x (n_p,n_g) in {(512,0),(0,128),(0,32),(1024,0)} and compared four quantities per point — the pre-split formula round((ns*reps+warm)/1e9,3), row_compute_seconds(), compute_seconds([row]), and energy_window's bench_compute_s(). 0 mismatches in 64 points. The double rounding I expected to bite only applies to multi-row windows, which are refused anyway.
- Findings #7/#8/#9, the divisor: swept n_p in {0,32,128,512,1024} x n_g in {0,32,128} x r in {1,2,3,5}. prefill == n_p*(r+1) and decode == n_g*r+1 at every point, 0 mismatches, and energy_window.bench_tokens() returns the identical value for every single-phase row. The split's two halves sum to the pre-split combined count at every r (512/128 at r=3: 2048 + 385 = 2433, the exact number in the BEFORE baseline). Could not find a shape where the split changed the measured work.
- Finding #19 / Python 3.6: parse_time() collapses 12 spellings to the single value 1789226343.0 — bare epoch, epoch with a decimal, 'Z', lowercase 'z', '+00:00', '+0000', space-separated, no offset at all, '+07:00', microseconds with 'Z', microseconds with '+00:00', and a leading/trailing-whitespace form. Separately, ast.parse(..., feature_version=(3,6)) accepts all seven .py files, so nothing 3.7+ crept in with the new code; the driver also still uses stdout=PIPE/stderr=PIPE rather than 3.7's capture_output.
- Findings #11/#15: grepped every live .md/.py/.sh. Nothing claims module-only POM scope — README.md:113, docs/10:37-44, telemetry/tegrastats_logger.py:14-16 and scripts/00_system_prep.sh:88-91 all say whole dev kit. The only 'module only' hits outside the audit document itself are in .handoff/2026-09-12-1519 (the historical handoff the 1541 one explicitly corrects). No file asserts an external meter exists; README.md:118, docs/10:26/54/301 and tegrastats_logger.py:20-21 all say there is none and that across-device comparison is deferred. docs/40 carries 'Status: FIXED 2026-09-12 — POM_5V_IN documented as whole-devkit scope in five places' and 'ADDRESSED 2026-09-12 — there is no meter'.
- Tried to break interleave_groups(): ran it over the real shipped matrix and checked for lost or duplicated run objects (len equal, all identities unique). It pairs prefill/decode of the SAME configuration, not merely alternating phases — partial_offload comes out ngl 0,0,8,8,16,16,24,24,99,99 and quantisation q4_k_m,q4_k_m,q8_0,q8_0. Also confirmed both halves of every group carry matching repetitions (largest_local's `repetitions: 2` is duplicated onto both), matching model lists and matching n_gpu_layers lists, and that selecting one half alone with --only degrades to a no-op instead of crashing.
- Re-ran ./bench/fixtures/run_regression.sh end to end on a clean copy: 19 passed, 0 failed, rc=0 — the claim reproduces. Its pieces reproduce individually too: --dry-run on the shipped matrix prints 'phases    : combined 2, decode 11, prefill 19' (32, matching README.md:57, scripts/50_bench_device.sh:48, docs/10:114 and bench/fixtures/README.md:19), matrix_mixed.json is refused by case name at --dry-run with exit 1, and the BEFORE baseline still fails the invariant checker.
- Ran the documented §5 ad-hoc workflow against my own sweep's real power CSV, using the full 'phase:case_id:NNN' marker plus the per-case <base>_NNN.csv, for a prefill, a decode and the fused window. All three join cleanly; the prefill J/tok agrees with the driver to the last digit (0.006511); the fused window prints 'no J/tok: refused_combined_request' with the reason line. The decode window differs from the driver by 7% (0.037548 vs 0.034948) — that is the pre-existing 5 Hz window-edge disagreement the 1541 handoff already recorded as finding #29, not something this edit introduced.
- Looked for downstream breakage from the rename/reshape and found none beyond the one handoff: the joined CSV is purely additive (21 columns added, 0 removed vs the BEFORE baseline), no script or Makefile target reads any CSV column name, --only gpu_decode_sweep in scripts/50_bench_device.sh still resolves, and the three renamed case ids are documented in docs/10:282-284 with an explicit instruction not to pool pre-split rows.
- Could not get a per-token number out of a window holding more than one working row: HS_FAKE_EXTRA_ROW=1 gives 'refused_multi_phase 2' and the joined CSV has joules_per_token / wh_per_mtok / active_joules_per_token empty on both rows. A ref exiting 1 still yields exactly one row with status=failed / run_failed and no per-token figure. The zero-work-axis row is correctly excluded by phase_rows(), so the guard does not blank the sweep.
- One thing I could not test and am flagging rather than claiming: config/node.env ships with LLAMA_CPP_REF="", so llama-bench is whatever HEAD builds, and nothing in the kit verifies that the installed ref supports -pg or that -p 0 -n 0 suppresses its defaults. The new end_to_end_request case depends on both. The kit audited -p 0 / -n 0 against upstream source (docs/40:98) but never -pg. Failure is graceful either way (run_failed, or refused_multi_phase if three rows appear), so this is an unverifiable-claim caveat, not a demonstrated defect.

---

## Lens: `py36` — fix-holds-with-caveats

I obtained a real CPython 3.6.13 (Anaconda linux-64 tarball, run via LD_LIBRARY_PATH) and executed the whole kit against it, not just reasoned about it: all 7 modules py_compile, and `bench/fixtures/run_regression.sh` reports 19 passed / 0 failed under 3.6 on two successive snapshots of the tree. The finding-#3 change itself is clean for 3.6.9 — no f-strings, walrus, dataclasses, typing syntax, `a|b` merge, `fromisoformat`, `text=`, `capture_output`, or post-3.6 math/statistics/typing members anywhere; the newest thing in the kit is `{**a, **b}` and `subprocess.run` (both 3.5), and vermin --target=3.6 agrees the floor is 3.5. What does NOT hold is text decoding: every `open()` and both `universal_newlines=True` subprocess reads use the locale default encoding, which on 3.6 under a C/POSIX locale is ASCII (Python 3.7's PEP 538 C-locale coercion is exactly what hides this on the 3.12 workstation). `config/models.tsv` ships an em dash on line 8, so `run_matrix.py --dry-run` dies with UnicodeDecodeError on 3.6 before a single run — reproduced. Shell is fine: `run_regression.sh` uses nothing newer than bash 3.x constructs and Ubuntu 18.04 ships 4.4.19.

### Findings (2)

**[HIGH] `model_path()` opens `config/models.tsv` with the locale default encoding, and that file ships a U+2014 em dash on line 8. On Python 3.6 with LANG/LC_ALL unset or C, the default encoding is ANSI_X3.4-1968 (ASCII) and the driver aborts with UnicodeDecodeError before any run starts. Python 3.7's PEP 538 coerces C to C.UTF-8, which is why the 3.12 workstation can never see this. `bench/run_matrix.py` line 513 (matrix.json), 574 (run log), 628 (case CSV) and 765 (joined CSV), plus `bench/check_finding3_invariants.py:52` and `telemetry/energy_window.py:46,165,439`, are the same unguarded pattern — today only models.tsv has a non-ASCII trigger that actually ships.**

`bench/run_matrix.py:77`

*Reproduction:*
```
Ran against real CPython 3.6.13 in a private copy of the kit (cp -a to a scratch dir; the shared tree was never written to).

$ LC_ALL=C LANG= python3.6 bench/run_matrix.py --dry-run --only gpu_decode_sweep
  File "bench/run_matrix.py", line 543, in main
    mp = model_path(env, r["model"])
  File "bench/run_matrix.py", line 78, in model_path
    for line in fh:
UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2 in position 427: ordinal not in range(128)

$ LC_ALL=C.UTF-8 python3.6 bench/run_matrix.py --dry-run --only gpu_decode_sweep   # works
$ LC_ALL=C LANG= /usr/bin/python3.12 bench/run_matrix.py --dry-run --only gpu_decode_sweep  # works (PEP 538)

Confirming the byte and the encoding:
$ LC_ALL=C grep -nP '[^\x00-\x7F]' config/models.tsv
8:# the Pi 5 — that asymmetry is one of the things being characterised.
$ LC_ALL=C python3.6 -c "import locale;print(locale.getpreferredencoding(False))"
ANSI_X3.4-1968
```

*Why it matters:* The deliverable is measured data and the driver is the only thing that produces it. This is an unconditional crash, not a degraded number, and it fires on a file that ships in the repo — so whether the thesis sweep runs at all depends on an environment variable nobody thinks about. It fires under systemd, cron, `env -i`, `sudo -i`, and any ssh invocation whose client does not forward LANG; it does not fire on the JetPack desktop image from an interactive shell, which is precisely what makes it the kind of thing that bites once, at 2 a.m., an hour into an unattended run.

*Fix:* Pin the encoding everywhere the kit reads or writes text. Minimum: `bench/run_matrix.py:77` `with open(manifest) as fh:` -> `with open(manifest, encoding="utf-8") as fh:`. Then the same `encoding="utf-8"` on run_matrix.py:513 `open(args.matrix)`, :574 `open(log_path, "w")`, :628 `open(case_csv, "w")`, :765 `open(joined_csv, "w", newline="")`, :789 `open(power_csv, newline="")`; check_finding3_invariants.py:52 `open(path)`; energy_window.py:46, :165, :439; tegrastats_logger.py:136 and :189; make_synthetic_power_csv.py:81. All of `encoding=` has been valid on open() since Python 3.0, so this costs nothing on either interpreter. Belt-and-braces: also de-unicode the models.tsv comment (`—` -> `--`), since every other config file the kit parses is already pure ASCII.

**[MEDIUM] `subprocess.run(cmd, ..., universal_newlines=True)` decodes llama-bench's stdout AND stderr with the locale encoding and strict errors. On 3.6 under a C locale a single non-ASCII byte anywhere in ggml/CUDA's stderr raises UnicodeDecodeError *inside* subprocess.run, so `out` is never bound, no record is appended, and the whole sweep dies with a traceback — defeating the driver's own stated design goal that "a single OOM on a 4 GB board must not abort a two-hour sweep". The paired `logfh.write(out.stderr or "")` at line 610 writes into a file opened at line 574 without an encoding, and fails the same way on the encode side. `load_env()`'s `check_output(..., universal_newlines=True)` at line 52 and tegrastats_logger.py's Popen at line 161 share the defect.**

`bench/run_matrix.py:604`

*Reproduction:*
```
Patched the fixture bench (in my scratch copy) to emit one U+00B5 on stderr, the way a ggml timing line plausibly would, and neutralised the models.tsv em dash so this path could be reached:

$ LC_ALL=C LANG= python3.6 bench/run_matrix.py --matrix $F/one.json --no-power --tag utf4
    stdout, stderr = process.communicate(input, timeout=timeout)
  File ".../python3.6/subprocess.py", line 760, in _translate_newlines
    data = data.decode(encoding, errors)
UnicodeDecodeError: 'ascii' codec can't decode byte 0xc2 in position 13: ordinal not in range(128)

Same matrix under LC_ALL=C.UTF-8 on 3.6: "1/1 rows ok (1 invocations)". Same matrix under LC_ALL=C on 3.12: "1/1 rows ok".

And the proposed fix, verified on 3.6.13 under LC_ALL=C LANG=:
  fixed subprocess.run  : ok, stderr='ggml: 1\xb5s timer' stdout='a,b'
  fixed check_output    : ok 'A=1\n'
  fixed log write       : ok, 17 bytes
  unfixed log write     : DIES -> 'ascii' codec can't encode character '\xb5' in position 7
```

*Why it matters:* Unlike finding 1 this has no shipped trigger, so I am not claiming it will definitely fire — it needs both a C locale and a non-ASCII byte from llama.cpp. But the consequence is worse when it does: the crash happens mid-sweep, after hours of board time, and the driver's failure-tolerance machinery (record the failure as a row, cool down, keep going) never runs because the exception is raised before `out` exists. Note the strict-errors half of this is version-independent: even on a UTF-8 locale, a malformed byte from GGUF metadata echoed to stderr raises the same exception on every Python 3.

*Fix:* `bench/run_matrix.py:604-605`: `out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False)` -> `out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, encoding="utf-8", errors="replace", check=False)`. The `encoding` and `errors` parameters were added to subprocess in Python 3.6 (the same release the Jetson runs), so this is safe there — verified above. Do the same at line 52 (`check_output`) and telemetry/tegrastats_logger.py:158-162 (`Popen(... universal_newlines=True, bufsize=1)` -> add `encoding="utf-8", errors="replace"`, which matters most because the systemd unit runs that logger with no LANG at all). Pair it with `encoding="utf-8"` on the `open(log_path, "w")` at line 574 or the write at 610 still raises on encode.

### Attacks that FAILED — evidence the fix holds
- Executed `bench/fixtures/run_regression.sh` end to end under real CPython 3.6.13, twice, on two snapshots of the tree (the second after another agent added the three-row fused-window assertion block): 19 passed, 0 failed both times, ~2m24s. Every driver run, the fake tegrastats logger, the join, the invariant checker and the energy analysis all ran on the 3.6 interpreter (PATH shim, so the fixtures' `#!/usr/bin/env python3` and the driver's `sys.executable` logger spawn both resolved to 3.6).
- `python3.6 -m py_compile` on all seven modules: run_matrix.py, check_finding3_invariants.py, energy_window.py, tegrastats_logger.py, make_synthetic_power_csv.py, fake_llama_bench.py, fake_tegrastats.py. All compile.
- vermin 1.8.0 --target=3.6 over all seven files: highest requirement is 3.5, from `{**run, "marker": label, ...}` (PEP 448 generalized unpacking, 3.5) and `subprocess.run` (3.5). Nothing in the kit needs 3.6 syntax, let alone 3.7+.
- Grepped for and found ZERO instances of: f-strings, walrus `:=`, `datetime.fromisoformat` (3.7), `subprocess` `text=` (3.7) or `capture_output=` (3.7) — confirmed capture_output raises TypeError on 3.6 and is not used, `dataclasses`, any `typing` import, `a | b` dict merge (3.9), `str.removeprefix/removesuffix/isascii` (3.9), `math.prod/dist/isqrt/comb/perm`, `statistics.fmean/multimode/quantiles/NormalDist`, `functools.cached_property`, `contextlib.nullcontext`, `shlex.join`, `importlib.resources`, `breakpoint()`.
- `pick(row, *names, skip_zero=False)` — keyword-only parameters after *args are PEP 3102, Python 3.0, not 3.7+. Called it on 3.6.13: `pick({"n_prompt":"0","n_gen":"128"}, "n_prompt","n_gen", skip_zero=True)` returns '128'.
- parse_time()'s %z handling is load-bearing, not superstition, and it is correct. On 3.6.13 `strptime('2026-09-12T15:19:03+00:00','...%z')` RAISES ValueError while '+0000' parses — the colon form was only accepted from 3.7. Confirmed `datetime.fromisoformat` is genuinely absent on 3.6 (AttributeError), so the comment at energy_window.py:53 is accurate. Round-tripped the logger's exact output format on 3.6: `datetime.now(timezone.utc).isoformat(timespec='milliseconds')` -> '2026-09-13T05:13:40.668+00:00' -> parse_time -> 1789276420.668. Also verified the naive, space-separated, 'Z', '+00:00', 3-digit-%f and bare-epoch forms all parse on 3.6.
- `datetime.isoformat(timespec=...)` was added in Python 3.6.0 (not 3.7) — used by tegrastats_logger.py:198 and make_synthetic_power_csv.py:54; executed on 3.6.13 and it works.
- Dict insertion-order reliance: run_matrix.main()'s header construction (`for rec in records: for k in rec: if k not in keys: keys.append(k)`, line ~758) and the `{**run, ...}`/`dict(base)`/`.update()` chain in bench_records() DO depend on insertion order. That is an implementation detail in CPython 3.6 rather than a language guarantee (3.7+), but the Jetson runs CPython 3.6.9, so it holds. Proved it empirically: ran the same matrix_split.json sweep with --no-power under 3.6.13 and under 3.12 and the joined CSV header came out byte-identical (`case_id,group,run_index,phase,row_phase,test,...`), as did the first 22 columns of every data row. Also note PEP 468 (3.6) makes **kwargs order a real guarantee. No correctness dependency, only column order.
- csv.DictReader returns OrderedDict on 3.6 (plain dict only from 3.8). Confirmed the type on 3.6.13 and checked every consumer: run_matrix's bench_rows are only touched via .get(), and check_finding3_invariants.py:67's `r.keys() if hasattr(r, "keys")` guard is satisfied by OrderedDict. Nothing branches on the type or the class name.
- Every other stdlib member used predates 3.6: shutil.which (3.3), os.makedirs(exist_ok=) (3.2), Popen.wait(timeout=) (3.3), Popen.terminate/kill (3.0), print(flush=) (3.3), print(file=) (3.0), csv.DictWriter.writeheader (3.2), datetime.timezone (3.2), datetime.timestamp() on an aware datetime (3.3), calendar.timegm (2.x), open(newline=) (3.0), subprocess.check_output (3.1).
- Exercised the paths the regression does not cover, all on 3.6.13: `energy_window.py --json` (valid JSON, nested rail dicts), `--out` (DictWriter flattening of the nested dicts, header built from the union of keys), `--from/--to` with ISO stamps, `tegrastats_logger.py` run for 3 s against the fake tegrastats with --echo (16 samples, trapezoidal integral, marker stamping all correct), `check_finding3_invariants.py` with no args (exit 1, prints __doc__) and against BEFORE_mixed_bench_energy.csv (8 violations, exit 1).
- `--help` for run_matrix.py, energy_window.py and tegrastats_logger.py all succeed on 3.6 under LC_ALL=C LANG= — every module docstring is pure ASCII, so argparse printing __doc__ does not hit the encoding problem. The only non-ASCII character in any .py source is in a `#` comment (energy_window.py:55), and Python 3 source is UTF-8 by default regardless of locale (PEP 3120), so import is unaffected.
- Shell: `bench/fixtures/run_regression.sh` uses only `set -u`, functions, `$(( ))`, `trap ... EXIT`, quoted heredocs, `printf` with ANSI escapes, `mktemp -d`, `chmod`, `head -c /dev/urandom`, `grep -q` and `[ ... -eq ]` — all bash 3.x-era. Grepped scripts/, bench/ and the fixtures for post-4.4 or exotic constructs (`mapfile`, `readarray`, `wait -n`, `declare -A`, `${x@Q}`, `coproc`, `local -n`): none. Ubuntu 18.04 ships bash 4.4.19, so the script is comfortably in range; `bash -n` passes on every .sh (that is the script's own first check). The external tools it leans on off-Jetson (`mktemp`, `head`, `grep`) are coreutils; the ones the other scripts use (`jq`, `bc`, `iperf3`, `timeout`, `nproc`) are installed by scripts/00_system_prep.sh.
- Caveat on the interpreter: I could not get 3.6.9 exactly — nothing in the apt archive and no pyenv/docker here — so I used CPython 3.6.13 (Anaconda linux-64) on x86_64. The gap is bugfix-only and touches none of the APIs above; in particular %z colon support and fromisoformat landed in 3.7, not in any 3.6.x, so 3.6.9 behaves identically for everything I tested. The Jetson is aarch64, but nothing in the kit is arch-sensitive.
- Not a py36 matter and not verifiable from here, so flagging rather than claiming: the new `-pg` code path depends on the pinned llama.cpp ref actually having `-pg` and on llama-bench skipping zero-valued `-p 0 -n 0` axes. config/node.env.example ships LLAMA_CPP_REF="" and docs/40_audit_findings.md argues the only workable CUDA-10.2+RPC pin is b5050 (April 2025). Worth a `llama-bench --help | grep -- -pg` check on the built binary before the first sweep. Separately, the shared tree was being edited while I audited: bench/check_finding3_invariants.py now reads a `request_joules` column and rule 7 at line 137 uses `e` leaked from the rule-6 loop, so it tests against request_joules rather than the row's energy source — a real defect, but mid-edit, outside this lens, and not a 3.6 issue. I wrote nothing into /mnt/data/KMITL/Final/HeteroServe/Jetson; all execution was in copies under my scratchpad.

---

## Lens: `break-it` — fix-holds-with-caveats

The matrix-side refusal is genuinely airtight: 21 malformed case shapes (scalars, strings, string "0", floats, negatives, booleans, nulls, defaults-block carry, pg+n_prompt, pg scalar/3-item/dict) all either got refused or landed as "combined", which never receives a per-token figure. But the ref-side is not: the driver trusts a single bench row's self-reported counts over its own knowledge of what it asked for, and phase_rows() goes completely blind exactly when those counts are unreadable. I produced three phase-mixed or mis-scoped per-token numbers on the current tree, one of which passes check_finding3_invariants.py cleanly while the regression suite reports 20 passed, 0 failed.

### Findings (5)

**[CRITICAL] A -pg (fused) window whose bench row reports n_prompt=0 is scored row_phase=decode and publishes joules_per_token over a decode-only divisor, even though the driver already knows on that same row that phase=combined and pg_n_prompt=512. per_token_status() consults only the row's reported counts and never cross-checks them against the invocation it built. check_finding3_invariants.py passes the result.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:411`

*Reproduction:*
```
Copied the kit, put a llama-bench ref on PATH that honours -pg (ONE fused instance, ONE row, warm-up = full prompt + 1 token, then 3 timed prompt+generate reps) but reports the fused row's prompt count in an n_prompt_tokens column, leaving n_prompt="0". Matrix case: {"id":"end_to_end_request","model":"qwen2.5-0.5b-instruct-q4_k_m","n_gpu_layers":99,"pg":[512,128]}.

$ python3 bench/run_matrix.py --matrix $F/c2.json --tag c2
    pp512+tg128   426.67 t/s   385 tok   8.0 s
1/1 rows ok (1 invocations)
  per_token_status ok                         1

Joined CSV row:
  case_id              'end_to_end_request'
  phase                'combined'
  row_phase            'decode'
  test                 'pp512+tg128'
  per_token_status     'ok'
  pg_n_prompt          '512'
  pg_n_gen             '128'
  bench_n_prompt       '0'
  bench_n_gen          '128'
  tokens               '385'
  compute_joules       '23.5755'
  joules_per_token     '0.061235'
  wh_per_mtok          '17.01'
  joules_per_request   None

$ python3 bench/check_finding3_invariants.py <that csv>
checked 1 rows (1 total)
all finding-#3 invariants hold
exit=0

Re-verified at unit level against the newest run_matrix.py (md5 ec8fb0e7b92c6ede072de71b98355477): bench_records({'n_prompt':0,'n_gen':0,'pg_n_prompt':512,'pg_n_gen':128,'repetitions':3}, [{'n_prompt':'0','n_gen':'128','test':'pp512+tg128','avg_ns':'16e9','avg_ts':'40'}], {'phase':'combined'}) -> phase=combined row_phase=decode tokens=129 status=ok
```

*Why it matters:* This is audit finding #3 reconstituted on the one case in the shipped matrix that deliberately spans both phases. The window's 23.58 J covers four 512-token prefills and four 128-token decodes; the divisor is 385 decode tokens. The published J/token is ~5x the true decode figure, it is labelled 'decode', and the dedicated finding-#3 checker signs off on it. The row itself carries three independent statements that the window was fused (phase=combined, pg_n_prompt=512, test=pp512+tg128) and none of them is consulted.

*Fix:* Cross-check the invocation against the row in per_token_status(): if rec.get("phase") == "combined" (equivalently pg_n_prompt and pg_n_gen are set) and rec.get("row_phase") != "combined", refuse -- the ref did not report the fused instance the driver asked for, so neither a per-token nor a per-request denominator is trustworthy. Add a matching rule to check_finding3_invariants.py: a row with pg_n_prompt/pg_n_gen non-zero must have row_phase=combined and must carry no joules_per_token.

**[HIGH] phase_rows() identifies working rows only via bench_int() on n_prompt/n_gen, so a ref that omits, blanks, or renames those columns yields working_rows=0 for every row in the window. per_token_status()'s multi-phase guard tests `working_rows > 1`, which 0 never satisfies, so a two-test window publishes a per-token figure on BOTH rows -- with window_shared=0, defeating the documented 'do not sum' warning at docs/10_measurement_protocol.md:171.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:398`

*Reproduction:*
```
Ref emitting two rows (pp512 then tg128) with a CSV header carrying no n_prompt/n_gen columns, for a single-phase prefill request (n_prompt:[512], n_gen:[0]).

$ python3 bench/run_matrix.py --matrix $F/attackA.json --tag attackA
[1/1] attackA|qwen2.5-1.5b-instruct-q4_k_m|ngl99|t4|p512|g0
    pp512       1024.00 t/s   2048 tok   8.0 s
    tg128        128.00 t/s   2048 tok   8.0 s
2/2 rows ok (1 invocations)
  per_token_status ok                         2

Joined CSV (both rows, same marker prefill:attackA:001):
 row_phase=prefill test=pp512 per_token_status=ok bench_rows=2 working_rows=0 window_shared=0 bench_n_prompt=512 bench_n_gen=0 tokens=2048 avg_ts=1024.0 joules=43.8564 compute_joules=23.5694 joules_per_token=0.011508 counts_source=matrix
 row_phase=prefill test=tg128 per_token_status=ok bench_rows=2 working_rows=0 window_shared=0 bench_n_prompt=512 bench_n_gen=0 tokens=2048 avg_ts=128.0  joules=43.8564 compute_joules=23.5694 joules_per_token=0.011508 counts_source=matrix

$ python3 bench/check_finding3_invariants.py <that csv>
FAIL (3):
  - row 0 (attackA / pp512): joules_per_token=0.011508 but working_rows=0
  - row 1 (attackA / tg128): joules_per_token=0.011508 but working_rows=0
  - row 1 (attackA / tg128): test=tg128 (decode) but n_prompt=512

Re-verified at unit level on md5 ec8fb0e...: phase_rows(rows) == 0, both records status=ok, tokens=1024, window_shared=0.
```

*Why it matters:* The decode row is published under the prefill divisor and both rows repeat the same 43.86 J, so a naive sum double-counts and the CSV gives no warning (window_shared=0). The guard is disabled precisely on the path where the driver is least sure what the window contained -- counts_source='matrix' means it could not read the counts at all, and it responds by trusting the matrix for every row instead of refusing. Real llama-bench does emit n_prompt/n_gen (the kit's own docs/40_audit_findings.md:98 confirms this), so this is defence-in-depth rather than a live break -- but the fallback path exists, is advertised via the counts_source column, and the invariant checker only catches it after the fact, by which point the driver has already printed '2/2 rows ok'.

*Fix:* Make 'counts unreadable' a refusal, not a pass. In per_token_status(), add: if (rec.get("bench_rows") or 0) > 1 and (rec.get("working_rows") or 0) != 1: return PT_MULTI_PHASE. A multi-row window is only safe when exactly one row did identifiable work; zero identifiable working rows in a multi-row window means the driver cannot tell which test the tokens describe. Also set window_shared = int(len(bench_rows) > 1) rather than int(len(working) > 1), so the 'do not sum' flag is driven by how many rows share the window, not by how many the driver could parse.

**[HIGH] telemetry/energy_window.py with --bench but no --marker and no --from/--to makes the ENTIRE power CSV one window (line 290), and the positional-join count check then passes trivially for any single-row bench CSV. The result is a confident joules_per_token whose energy comes from a different case, a different phase and a different ngl. The tool's own docstring example (line 25) is exactly this invocation.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/energy_window.py:290`

*Reproduction:*
```
Ran an honest 5-run split sweep (bench/fixtures/matrix_split.json) with the shipped fake ref. Driver's own answer for run 001 (split_prefill, ngl0): joules=29.1745 compute_joules=13.3331 tokens=2048 joules_per_token=0.00651.

Then the documented example shape, joining run 001's own single-row per-case CSV:
$ python3 telemetry/energy_window.py <sweep>_g_power.csv --bench <sweep>_g_001.csv --reps 3
window 0  marker=idle  69.10 s  346 samples
  module  242.8905 J   mean   3510.1 mW   peak   7400.0 mW
  pp512  1024.00 tok/s  0.001672 J/tok  0.46 Wh/Mtok
  tokens  2048   (512 prompt x 3 reps+warmup)
  basis   compute
  compute   3.4243 J over 2.000 s   load 239.4662 J over 67.302 s

0.001672 vs the true 0.00651 -- 3.9x too low. The 2.000 s 'compute' slice is the tail of run 5 (the fused case), not run 1 at all; the window is labelled marker=idle and spans the 30 s pre-sweep baseline plus all five cases plus every cooldown. Nothing in the output flags it.
```

*Why it matters:* This is the standalone analysis tool the protocol points the student at, invoked in the shape its own help text prints. The count-equality guard added by the fix protects only against 1-window-vs-N-rows; it cannot see that the single window it chose is the wrong window. Combined with the already-documented fact that --marker prefill matches nothing on a driver-produced power CSV (labels are 'prefill:case_id:001'; docs/41_finding3_analysis.md:132), the path of least resistance for an analyst is to drop --marker -- and that path silently produces a number from another run.

*Fix:* Refuse --bench unless a window was actually selected: require --marker or --from/--to alongside --bench, and reject the implicit wins=[rows] whole-file window for joins. Better still, cross-check identity rather than only count: the driver's marker label already encodes phase and run index, and the per-case CSV filename encodes the same run index -- compare them and refuse on disagreement. At minimum, fix the docstring example at line 25 so it carries a --marker, and give --marker prefix matching so 'prefill' selects 'prefill:case:001'.

**[MEDIUM] Nothing compares a bench row's reported counts with what the matrix asked for. A row reporting n_prompt=128 for a run invoked as -p 512 yields tokens=256 against a 512-token window's energy, per_token_status=ok, counts_source=bench, and no warning column. docs/41_finding3_analysis.md:437 specifies a counts_mismatch column for exactly this; it is absent from run_matrix.py.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:456`

*Reproduction:*
```
PYTHONDONTWRITEBYTECODE=1 python3 against the live run_matrix.py (md5 ec8fb0e7b92c6ede072de71b98355477):

bench_records({'repetitions':3,'n_prompt':512,'n_gen':0},
              [{'n_prompt':'128','n_gen':'0','test':'pp128','avg_ns':'5e8','avg_ts':'256'}],
              {'status':'ok','phase':'prefill'})
-> matrix n_prompt=512, row n_prompt=128 -> tokens=256 status=ok mismatch_col=False

Same for decode: requested n_gen=128, row reports n_gen=32 -> tokens=33, status=ok.

$ grep -c counts_mismatch bench/run_matrix.py
0
```

*Why it matters:* The whole point of the finding-#3 fix is that the ROW's counts drive the divisor. That is right, but it makes the row's counts load-bearing with no corroboration: if a ref clamps, rounds, or renames a count, the divisor silently describes a test that did not run and the row still says ok. A 4x divisor error is the same magnitude as the pre-fix bug, and unlike the pre-fix bug there is no column on the row that would let a reader notice.

*Fix:* Implement counts_mismatch as docs/41_finding3_analysis.md:437 already designs it: build the expected {pp512}/{tg128} set from the run, the actual set from the rows, write counts_mismatch on every row of the window when they differ, and print a warning in the run loop. Keep the row's counts as the divisor (that part is correct) but make the disagreement visible, and add a check_finding3_invariants.py rule that a row with counts_mismatch=1 may not carry joules_per_token.

**[MEDIUM] check_finding3_invariants.py rule 7 is dead code. The rule-6 for-loop at lines 121-133 rebinds `e` to num(r.get(e_col)), whose last value is request_joules; rule 7 at line 137 then reads that leaked `e` instead of the row's own energy, so it is None on every non-fused row and the rule never fires.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/check_finding3_invariants.py:137`

*Reproduction:*
```
Built a one-row CSV that rule 7 is supposed to reject: per_token_status=ok, energy_basis=compute, compute_joules=20.0, tokens=2048, joules_per_token empty.

$ python3 bench/check_finding3_invariants.py /tmp/tmp4ebx4eyd.csv
checked 1 rows (1 total) in /tmp/tmp4ebx4eyd.csv
all finding-#3 invariants hold

Expected: 'per_token_status=ok but no joules_per_token'. Before the rule-6 rewrite, `e = num(r.get(src))` was bound immediately above rule 7; the rewrite moved that binding inside the loop as `e_col`/`e`.
```

*Why it matters:* Rule 7 is the invariant that catches a row claiming 'ok' while silently carrying no number -- the failure mode the per_token_status column exists to eliminate. It now passes everything, and the regression suite still reports 20 passed 0 failed because no fixture exercises it. This is a regression introduced by this session's edits, not a pre-existing hole.

*Fix:* Rename the rule-6 loop variable (e.g. `e_val`) and restore an explicit `e = num(r.get(src))` before rule 7, or inline it: `if status == "ok" and num(r.get(src)) is not None and jpt is None and not (p and g)`. Add a fixture row to run_regression.sh that rule 7 must reject, so the rule cannot go dead silently again.

### Attacks that FAILED — evidence the fix holds
- expand() vs 21 malformed matrix shapes -- every route to a mixed -p N -n M invocation was closed. Refused: n_prompt/n_gen as scalar ints (512/128), as strings ("512"/"128"), as string zero ("0"/128 -- refused, conservatively wrong direction), as floats (512.0/128.0), negative (-512/128), booleans (True/128), a list containing null ([None,512]/[128,None]), the defaults block carrying n_prompt while the case sets n_gen, 'pg' plus n_prompt, 'pg' plus n_prompt:"0", pg as a bare scalar, pg as a 2-char string "ab", pg as a dict, pg with 3 items. Each exited non-zero naming the case id, at --dry-run time.
- pg oddities that DID slip past validation all land harmlessly: pg:[-512,128], pg:[512.5,128.25], pg:[[512,1],[128,1]], pg:["0","128"], pg:[True,True] all produce phase_of()=="combined" runs, which per_token_status() routes to refused_combined_request and which never receive joules_per_token. The only real consequence is a malformed -pg argument that llama-bench will reject, i.e. a run_failed row.
- Three-row window (pp512 + tg128 + a fused pp512+tg128 row) -- all three records get refused_multi_phase, with the combined check correctly ordered after the multi-phase check, so the fused row is not re-denominated per request.
- Ref emitting a row for the zero-valued axis (n_prompt="0", n_gen="0", or "0.0") alongside one real row -- phase_rows() correctly counts 1, the real row keeps its per-token number, and the zero row gets no_token_count. The guard is a tripwire, not a sweep-wide outage, exactly as designed.
- Two working rows WITH proper count columns in one window (HS_FAKE_EXTRA_ROW path) -- both rows refused_multi_phase, window_shared=1, and no joules_per_token / wh_per_mtok / active_joules_per_token column is populated at all.
- A failed run (rc!=0) -- exactly one record, status=failed, per_token_status=run_failed, no per-token columns. The failure does not vanish from the joined CSV.
- energy_window.py joining a two-row bench CSV against one window -- refused with the count-mismatch message, not zipped past. Joining a genuinely fused row (n_prompt=512, n_gen=128) -- refused_combined_request, and the avg_ts x duration fallback correctly does NOT rescue it (I confirmed the fallback only fires when basis is empty, i.e. no deliberate refusal happened).
- Single-phase request where the ref returns ONE row with both counts set -- caught as row_phase=combined / refused_combined_request, so no per-token figure is emitted even though the request was prefill-only.
- The fused-row warm-up arithmetic I expected to be wrong (tokens_of returning (n_p+n_g)*(reps+1)=2560 instead of the upstream shape n_p*(reps+1)+n_g*reps+1=2433, and row_compute_seconds assuming a whole fused warm-up) was already corrected in the tree during this session -- tokens_of now returns 2433 and request_window_s() denominates the fused case over its timed loops alone. I could not break it.
- The BEFORE_mixed_bench_energy.csv baseline still fails check_finding3_invariants.py, so the checker has not been weakened into a no-op overall (only rule 7 specifically went dead).
- Running the full suite on my own copy of the current tree: bench/fixtures/run_regression.sh reports '20 passed, 0 failed' while findings 1-5 above all reproduce against the same files -- the suite's rogue-ref test only covers a ref that DOES emit count columns.

---

## Lens: `docs-code-agreement` — fix-holds-with-caveats

The finding-#3 mechanism itself survives every attack I made: 32 runs / combined 2, decode 11, prefill 19 is exact; the group interleave really alternates prefill/decode and pairs the same ngl/model; prefill+decode tokens equal the old combined count at r=3 (2048+385=2433) and r=2 (1536+257=1793); upstream llama-bench really does skip zero-valued -p/-n and really does build ONE instance and ONE row for -pg; every column named in the docs/10 table is present in the real header. What does not hold is a set of factual claims wrapped around it. The headline sentence "no power window ever contains more than one phase" is false for the 2 shipped -pg runs; "every run produces at least one row" is false for a missing model (I ran it: 2 runs, 1 row); `per_token_status=ok` is written on rows that carry no energy at all whenever there is no power CSV; the `--reps` warning in both the doc and the tool has the error direction inverted (measured 2.0x and 2.98x too HIGH, not "r+1 times too low"); and `test` is documented as llama-bench's own label when llama-bench's CSV printer has no `test` field at all, which makes the docs' cross-check and invariant #4 tautological on real hardware. Separately: I found the fused -pg token/warm-up model (2560 tokens, requests=reps+1) to be wrong against upstream, which warms a fused instance with a full prompt pass plus exactly ONE token — while I was writing that up another agent corrected it in the shared tree (it now emits tokens=2433, requests=3, energy_basis "timed requests"), so I report only its residue below.

### Findings (13)

**[HIGH] docs/10 §5 and energy_window.py's own stderr warning both state the --reps error backwards. Omitting --reps makes J/token come out too HIGH, not "roughly r+1 times too low" / "~4x too low".**

`docs/10_measurement_protocol.md:279`

*Reproduction:*
```
Ran in a copy: `python3 telemetry/energy_window.py p.csv --marker decode --bench dec.csv --reps 1` -> `0.427907 J/tok, tokens 129`; `--reps 3` -> `0.143377 J/tok, tokens 385`. Ratio 2.98x too HIGH. Prefill: `--marker prefill --bench pre.csv --reps 1` -> `0.039746 J/tok, tokens 1024`; `--reps 3` -> `0.019873 J/tok, tokens 2048`. Ratio 2.0x too HIGH. The emitted note reads verbatim: `note: --reps not given, assuming 1. bench/matrix.json defaults to repetitions=3 -- pass --reps 3 or J/token is ~4x too low.`
```

*Why it matters:* J/token = joules/tokens; a too-small divisor inflates the result. A student told the failure mode is "too low" will read an inflated energy-per-token as a conservative floor and publish it. The magnitude is wrong too, and it is not even constant: 2.0x for a 512-token prefill row, 2.98x for a 128-token decode row, because the warm-up term scales differently in the two formulas. This is the one place §5 tells the operator how to interpret an analysis they ran by hand.

*Fix:* docs/10 line 279: "Left out, it assumes 1, so the divisor is short by the repetitions that actually ran and every J/token comes out too HIGH — 2.0x for a 512-token prefill row at -r 3, ~3.0x for a 128-token decode row." energy_window.py line 299: same correction ("J/token is 2-3x too high").

**[HIGH] docs/10 §4 "Every run produces at least one row" is false: a run whose model file is absent is skipped with no record at all, so it vanishes from the joined CSV.**

`bench/run_matrix.py:593`

*Reproduction:*
```
Two-case matrix (one model present, one absent from MODEL_DIR), `--no-power`: driver printed `[2/2] SKIP missing_model|llama-3.2-1b-instruct-q4_k_m|ngl0|t4|p512|g0 (model missing)` and the joined CSV contained exactly 1 row (`has_model 1 ok ok`). Code path: `print("[%d/%d] SKIP %s (model missing)"...); continue` — no `records.append`, unlike the rc!=0 path which does append a `status=failed` record.
```

*Why it matters:* Directly contradicts the paragraph it sits under, which promises that a run that failed "or emitted no parseable CSV produces exactly one row ... and the sweep continues." docs/30's anomaly list then asks the operator to spot "a placement or quantisation point whose prefill half completed but whose decode half did not" — undetectable from the CSV when the half disappeared. The quantisation cases need Qwen2.5-0.5B-Q8_0 and largest_local needs the 3B, so a partially populated MODEL_DIR silently halves a split pair while run_index just gains a gap.

*Fix:* Either append a record in the skip branch (`status: "skipped"`, new per_token_status `model_missing`) so the run is auditable, or change docs/10 line 120 to "Every run that was attempted produces at least one row; a run whose model file is missing is skipped and logged to stdout only." The first is better — the CSV is the archived artefact, stdout is not.

**[HIGH] `per_token_status=ok` is written on rows that have no energy columns at all whenever the power CSV does not exist (--no-power, or tegrastats missing). The documented meanings of `ok` and `no_power_window` are both violated.**

`bench/run_matrix.py:672`

*Reproduction:*
```
`python3 bench/run_matrix.py --matrix missing.json --tag missing --no-power` -> joined row: `has_model 1 ok ok joules= None jpt= None`. Driver summary printed `per_token_status ok 1`. The whole join, including the `if not energy_j and rec[per_token_status] == PT_OK: rec[per_token_status] = PT_NO_POWER` demotion, is inside `if os.path.exists(power_csv):`, which is never entered.
```

*Why it matters:* docs/10's per_token_status table says `ok` = "the per-token columns are populated and trustworthy" and `no_power_window` = "no power samples covered the window". This column was added specifically because "a blank per-token cell otherwise means at least five different things"; the most likely real-world blank — the logger never started, e.g. tegrastats absent from PATH because 00_system_prep did not run — is the one case that still reports `ok`. check_finding3_invariants rule 7 guards on `e is not None` and so passes it too, and docs/30 tells the operator to treat anything other than `ok`/`refused_combined_request` as an anomaly, so this row reads as clean.

*Fix:* Hoist the demotion out of the `os.path.exists(power_csv)` guard: after the loop, for every record with `per_token_status == PT_OK` and no `joules`, set `PT_NO_POWER`. Then add an invariant: a row with per_token_status=ok must carry a non-empty `joules`.

**[MEDIUM] docs/10 documents `test` as "llama-bench's own label". Real llama-bench's CSV printer emits `test::get_fields()`, which contains no `test` field — the pp/tg label is markdown-only — so on hardware the value always comes from the driver's own `test_label()` fallback, derived from the same counts as `row_phase`.**

`docs/10_measurement_protocol.md:174`

*Reproduction:*
```
Fetched tools/llama-bench/llama-bench.cpp at master and examples/llama-bench/llama-bench.cpp at tag b3500. get_fields() at b3500 is verbatim `"build_commit", "build_number", "cuda", ... "n_prompt", "n_gen", "test_time", "avg_ns", "stddev_ns", "avg_ts", "stddev_ts"` — `test_time`, no `test`; the pp/tg name is computed in the markdown printer only. The kit's fake_llama_bench.py DOES emit a `test` column, which is why no fixture catches this. Code path taken on real data: run_matrix.py `"test": r.get("test") or test_label(eff_p, eff_g)` with `r.get("test")` always None.
```

*Why it matters:* Two claims collapse. docs/10 line 173 says `phase`/`row_phase` and the row's test "differ only if a ref returned something unexpected" — they can never differ, because both are computed from bench_n_prompt/bench_n_gen. And check_finding3_invariants.py rule 4 ("the test label must agree with the counts it claims to describe") is tautological on every real row: it compares test_label(p,g) against p,g. A regression suite reporting that check as passing is reporting nothing.

*Fix:* docs/10 line 174: "`test` | the test label. llama-bench's CSV has no `test` column (it is markdown-only), so the driver derives this from the row's own n_prompt/n_gen — it is a restatement of those counts, not independent evidence." Drop or re-scope invariant 4, and make the fake omit the `test` column so the fixture matches the real CSV shape.

**[MEDIUM] The `pg` key emits `-pg`, which does not exist in llama.cpp refs older than roughly mid-2024 — exactly the era README tells the operator to pin for CUDA 10.2. No doc states a minimum ref.**

`bench/matrix.json:204`

*Reproduction:*
```
Fetched examples/llama-bench/llama-bench.cpp at tags b1500, b1700, b2000, b2700: all four have no `-pg` parsing branch and no n_pg loop (b1500 quoted verbatim: only `-p`/`--n-prompt` and `-n`/`--n-gen`). At b3500 the `-pg` branch exists. README.md:100-105 says stock JetPack 4.6 ships CUDA 10.2, current llama.cpp targets CUDA >= 11, "the CUDA build of master will very likely not compile on this board" and offers "pin an older llama.cpp ref" as option one.
```

*Why it matters:* On the ref the README predicts the operator will pin, `-pg 512,128` hits the unknown-argument branch, llama-bench prints usage and exits 1, and both end_to_end_request runs land as `status=failed / run_failed`. That is graceful, but it is 2 of the 32 documented runs and the only case that measures decode at real context depth, and nothing in docs/10 §4 (which introduces `"pg"` as simply available) or bench/matrix.json's _comment warns of it. The same two tags confirm the zero-skip guards (`if (n_prompt == 0) continue;`) predate -pg, so the `-p 0 -n 0` half of the invocation is safe on old refs — only -pg is not.

*Fix:* Add to docs/10 §4 and the matrix.json _comment: "`-pg` requires a llama.cpp ref from roughly b3000 onward (absent at b2700). If LLAMA_CPP_REF is pinned older for CUDA 10.2, disable end_to_end_request — there is no fused-request substitute, and -p N -n M is refused by design." Optionally have build_cmd probe `llama-bench --help` for `-pg` and fail the case at --dry-run rather than mid-sweep.

**[MEDIUM] docs/10's column table says `energy_basis` is "which of the two the per-unit columns were built from". The code now writes three distinct values.**

`docs/10_measurement_protocol.md:183`

*Reproduction:*
```
grep of the current tree: run_matrix.py:714 `rec["energy_basis"] = "compute"`, :720 `"whole-process (includes model load)"`, :752 `"timed requests"`. Ran the shipped fused case against the current code: the row came back `energy_basis timed requests`, `request_window_s 4.5`, `request_joules 23.572`, `joules_per_request 7.8573`.
```

*Why it matters:* `energy_basis` is the column a reader consults to know what the numerator is, and docs/10 §4 elsewhere names only two possibilities ("compute" and the whole-process string). A reader who sees `timed requests` has no entry to look it up under. This is the residue of the fused-warm-up correction that landed in the shared tree mid-audit; the surrounding prose and the `request_joules`/`request_window_s` row were updated, this cell was not.

*Fix:* docs/10 line 183: "`energy_basis` | which slice of the window the per-unit columns were built from: `compute` (the trailing compute_s), `timed requests` (a fused row's avg_ns x r), or `whole-process (includes model load)` when no trim could be made — treat the last as an upper bound."

**[MEDIUM] README's "Measurement honesty" bullet "No power window ever contains more than one phase" and docs/10 §4 step 2 "each in its own power window covering exactly one phase" are both false for the two shipped end_to_end_request runs.**

`README.md:122`

*Reproduction:*
```
`--dry-run` of the shipped matrix lists runs 31 and 32 as `combined end_to_end_request|...|pg512,128`. In a fixture sweep the window's power samples all carry marker `combined:fused_request:005` while the process ran a prefill segment and then a decode segment inside it (the fake switches its phase file from `prefill` to `decode` mid-window; upstream llama-bench likewise runs test_prompt then test_gen inside the one fused instance). So one marker window physically spans both phases.
```

*Why it matters:* These are the two most load-bearing sentences in the kit — the bullet is literally under a heading called "Measurement honesty," and the §4 sentence is the numbered description of what the driver does. Both documents go on, three paragraphs later, to explain the fused exception correctly, so the top-line claim is contradicted by its own page. A thesis reviewer who reads the bullet and then finds a `combined` row has been misled about the instrument's guarantee.

*Fix:* README:122 -> "**No power window mixes two separately-measured phases.** Every per-token figure comes from a window holding exactly one llama-bench test. The one deliberately fused case (`-pg`) spans both phases inside a single test instance and therefore reports `joules_per_request`, never `joules_per_token`." docs/10:115 -> "...each in its own power window holding exactly one llama-bench test."

**[MEDIUM] docs/10 §2 says `idle_mw` is "taken from the 30 s idle baseline the driver collects before the sweep". idle_floor() averages every sample marked idle in the whole log, including every inter-run cooldown and the teardown.**

`bench/run_matrix.py:793`

*Reproduction:*
```
idle_floor() iterates all rows of the power CSV selecting `marker == "idle"`. Measured on a fixture sweep's power CSV: 180 idle-marked samples, of which 139 lie within the first 30 s and 41 (23%) come from the 20 s-equivalent cooldowns between runs and the 3 s tail after the last run.
```

*Why it matters:* A cooldown sample is taken seconds after a multi-watt run, with the board hot and leakage elevated and the rails still settling — it is not the same population as a cold pre-sweep baseline. Pooling them biases `idle_mw` upward, which biases every `active_joules` and `active_joules_per_token` downward, and the bias grows with sweep length. The doc names a specific 30 s population so a reader can reason about it; the code uses a different, sweep-dependent one. (The fixture's fake reports a constant 1900 mW for idle, so this run cannot quantify the hardware magnitude — only that the population differs.)

*Fix:* Either restrict idle_floor() to samples before the first non-idle marker (and keep the pooled figure as a second column, e.g. `idle_mw_cooldowns`, since the gap between them is itself a thermal diagnostic), or change docs/10 line 82 to "the mean of every idle-labelled sample in the log — the pre-sweep baseline plus every cooldown" and say why that is acceptable.

**[MEDIUM] docs/10 §4 promises the two phase curves recombine as `P·(J/tok)_prefill + G·(J/tok)_decode` "so the prompt-to-generation ratio can be varied without re-measuring". The matrix measures decode only from an empty KV cache, so the composition is not valid for a request with a real prefix — and the kit's own analysis doc says so.**

`docs/10_measurement_protocol.md:143`

*Reproduction:*
```
Every decode case in bench/matrix.json sets `n_prompt: [0]`, so llama-bench's tg instance generates from a zero-depth context (each instance gets its own context with the KV cache cleared). docs/41_finding3_analysis.md:1233 states it directly: "A `tg` test generates its tokens from an empty KV cache whether or not a `pp` test ran before it in the same process. The decode numbers here are decode at ~zero context depth, in the split matrix and in the old mixed one alike. Decode at a realistic context depth needs `llama-bench -pg 512,128`".
```

*Why it matters:* Decode cost per token grows with context depth (attention over a longer KV cache, more bandwidth per token), so composing a 512-token prefix request from a zero-depth decode measurement under-states it. docs/10 presents the composition as the payoff that justifies the split, with no caveat; docs/41 raises the caveat but carries no status annotation saying whether it was accepted. The one case that does measure decode at depth (`-pg`) is structurally refused a per-token number, so the study has no way to check the composition against a measurement.

*Fix:* Add one sentence at docs/10:144: "Both curves are measured at ~zero context depth (`n_prompt: [0]` on every decode case), so the composition is exact only for short prefixes; decode at depth costs more. `end_to_end_request` is the one point that measures a real 512-token prefix, and it is reported per request — use it as a sanity check on the composed figure, not as a substitute."

**[LOW] docs/10 §6 tells the reader to identify pre-split rows by `phase=mixed`, but the kit's own pre-fix baseline CSV has no `phase` column at all.**

`docs/10_measurement_protocol.md:298`

*Reproduction:*
```
`python3 -c "import csv; r=list(csv.DictReader(open('bench/fixtures/BEFORE_mixed_bench_energy.csv')))[0]; print('has phase col?', 'phase' in r); print(r['marker'])"` -> `has phase col? False` / `mixed:mixed_case:001`. The 40-column header of BEFORE_mixed_bench_energy.csv runs case_id, note, model, n_gpu_layers, threads, n_prompt, n_gen, ... with no `phase`.
```

*Why it matters:* docs/10 §6 says "pre-split rows carry `phase=mixed`. Reject those rather than pooling them." An operator filtering an archived CSV on a column that does not exist gets an empty rejection set and pools the old rows anyway — the exact outcome the paragraph exists to prevent. The discriminator that does exist in that file is the `marker` prefix (`mixed:`) and `n_prompt`/`n_gen` both non-zero.

*Fix:* docs/10:298 -> "...pre-split rows have both `n_prompt` and `n_gen` non-zero and a `marker` beginning `mixed:` (they predate the `phase` column). Reject those rather than pooling them."

**[LOW] build_cmd's comment says llama-bench "rejects -p 0 / -n 0 on some refs; only pass the axis in play" — the code immediately below always passes both, and upstream has skipped zeros since at least b1500.**

`bench/run_matrix.py:248`

*Reproduction:*
```
The three lines after the comment are `if run["n_prompt"]: cmd += ["-p", str(...)] else: cmd += ["-p", "0"]`, and the same for -n. phase_rows()'s docstring in the same file states the opposite of the comment: "build_cmd() always passes both -p and -n, using 0 for the idle axis". Upstream at b1500 (fetched) already has `for (const auto & n_prompt : params.n_prompt) { if (n_prompt == 0) { continue; } }` and the same for n_gen.
```

*Why it matters:* A maintainer who trusts the comment will conclude the -p 0 -n 0 pair is a bug and "fix" it by dropping the zero axis — which would let llama-bench fall back to its defaults (-p 512 -n 128) and put two tests in one window, reintroducing finding #3 by exactly the route expand() was written to block. The comment is also the only justification given for the `phase_rows()` tripwire that follows, and it asserts the opposite of what that tripwire assumes.

*Fix:* Replace with: "Pass BOTH axes explicitly, 0 for the one not in play. Upstream skips zero-valued -p/-n when building instances (since ~b1500), so this is what stops llama-bench falling back to its defaults of -p 512 -n 128 and putting two tests in one window. phase_rows() ignores any row a ref emits for the zero axis."

**[LOW] bench/fixtures/README.md says HS_FAKE_LOAD_S "is what `load_joules` should come back as" — it is seconds, it maps to `load_s`, and even `load_s` comes back short.**

`bench/fixtures/README.md:66`

*Reproduction:*
```
Ran the split fixture with HS_FAKE_LOAD_S=1.0: emitted rows carry `load_s 0.804 / 0.796 / 0.804 / 0.797` and `load_joules 5.4279 / 5.1659 / 6.9097 / 5.1662`. Neither column equals 1.0; load_s is ~0.8 because it is `duration_s - compute_s` over a 5 Hz sampled window that starts before the process is spawned.
```

*Why it matters:* This is the only guidance the fixture README gives on how to verify the compute_s trim, and it points at the wrong column with the wrong unit. Someone checking the trim against `load_joules` will see ~5 J against an expected 3 and conclude the trim is broken, or worse, adjust the trim to make the number match.

*Fix:* "seconds spent in the `load` phase before the timed loops (default 3.0). `load_s` on the joined row should come back close to this — a few hundred ms short, since the window opens before the process is spawned and the power log is sampled at 5 Hz. `load_joules` is that slice's energy, not this value."

**[LOW] README's Layout tree omits every file this change added, plus three pre-existing ones.**

`README.md:86`

*Reproduction:*
```
README lines 67-96 list bench/ as only `matrix.json` and `run_matrix.py`, telemetry/ as only two files, docs/ as four files. Actually on disk: bench/check_finding3_invariants.py, bench/fixtures/ (7 files including run_regression.sh and the fake binaries), telemetry/make_synthetic_power_csv.py, scripts/lib/common.sh, docs/40_audit_findings.md, docs/41_finding3_analysis.md. `make check` also does not invoke bench/fixtures/run_regression.sh.
```

*Why it matters:* The regression harness is the artefact that makes the finding-#3 fix defensible, and the README — the first file anyone opens, and the one that says "every script exists to make a measurement reproducible" — does not mention that it exists. Nothing tells an operator to run it after editing matrix.json, and `make check` (advertised in the Makefile help as "lint the scripts") does not.

*Fix:* Add to the layout block: `│   ├── check_finding3_invariants.py  assert no row mixes phases` and `│   └── fixtures/                     off-hardware regression (run_regression.sh)`, plus make_synthetic_power_csv.py, scripts/lib/common.sh and docs/40, docs/41. Add a `regression:` target to the Makefile and mention it under "per experiment".

### Attacks that FAILED — evidence the fix holds
- Run counts. `--dry-run` on the shipped matrix printed exactly `runs : 32` and `phases : combined 2, decode 11, prefill 19`, matching docs/10 §4, bench/fixtures/README.md and run_regression.sh. Hand-count by case agrees: prefill 4+4+5+2+4=19, decode 2+2+5+2=11, combined 2.
- Interleaving. The dry-run listing shows runs 13-22 alternating partial_offload_prefill/partial_offload_decode at ngl 0,0,8,8,16,16,24,24,99,99 and runs 23-26 alternating quantisation prefill/decode per model — so the halves both alternate AND pair up on the swept axis, which interleave_groups() does not obviously guarantee. Ungrouped cases kept their original positions, as its docstring claims.
- "-pg produces exactly ONE CSV row." Confirmed against upstream: get_cmd_params_instances pushes one instance per n_pg entry (`for (const auto & n_pg : params.n_pg) { if (n_pg.first == 0 && n_pg.second == 0) continue; ... instances.push_back(instance); }`), present at b3500 and master. This is NOT a new instance of finding #3.
- "-p 0 -n 0 -pg N,M is the right invocation." Confirmed: upstream skips zero-valued -p and -n when building instances (verbatim `if (n_prompt == 0) { continue; }`), and cmd_params_defaults has `/* n_pg */ {}` with the README showing `-pg <pp,tg> (default: )`. So there is no phantom fused row on non-pg runs and no standalone pp/tg pair on the -pg runs. I specifically hunted for a non-empty n_pg default that would have put a pp512+tg128 row in every window in the study; it is not there.
- "The split changes no measured work." Verified arithmetically at both repetition counts: r=3 combined 512*4+128*3+1 = 2433 = prefill 2048 + decode 385; r=2 (largest_local_model) 512*3+128*2+1 = 1793 = 1536 + 257.
- Documented column names. Diffed the real emitted header (61 columns from a fixture sweep) against the docs/10 §4 table: every documented column is present under exactly the documented name — phase, row_phase, test, run_index, group, avg_ts, bench_n_prompt, bench_n_gen, bench_rows, working_rows, bench_row_index, window_shared, per_token_status, joules, load_s, load_joules, compute_s, compute_joules, energy_basis, joules_per_token, wh_per_mtok, active_joules_per_token, joules_per_request, wh_per_krequest, tokens, requests, bench_csv, precached, temp_rise_c, swap_peak_mb. No renamed or missing column.
- expand()'s refusal. All four malformed shapes exit non-zero naming the case id: both axes set, both set via list containing zeros, pg plus n_prompt, and a one-element pg. The legal shapes still expand.
- The refusal guard counting phase_rows() rather than len(rows). With HS_FAKE_EXTRA_ROW the driver marked both rows refused_multi_phase and wrote no per-token column; with a zero-work row present, phase_rows() still returns 1 and the sweep is not blanked.
- energy_window.py's new refusals. The positional join exits on unequal counts; bench_tokens refuses a fused pp+tg row with a reason; and the avg_ts x duration fallback does not rescue that refusal (the row prints `refused_combined_request`).
- The regression suite. `./bench/fixtures/run_regression.sh` really does pass in a clean copy — 19/19 on the tree as I first copied it, 20/20 on the tree as of my last copy. I could not find a test in it that passes for the wrong reason except invariant 4 (see findings).
- A failed run producing exactly one row with status=failed / per_token_status=run_failed and no joules_per_token — reproduced, holds.
- docs/10's per_token_status table: the code defines exactly the seven documented values and no others, and check_finding3_invariants.py's KNOWN_STATUS set matches.

---

## Lens: `numbers` — fix-holds-with-caveats

The per-phase half of the fix is numerically sound: I re-derived tokens_of() and row_compute_seconds() from the upstream llama-bench warm-up source and they match exactly for prefill and decode at r=2 and r=3, interleave_groups() is a faithful permutation in every shape I could build, and the compute_s trim recovers hand-integrated compute energy on split rows to within 2% with J/token stable to 0.03% across a 4x change in model-load time. The NEW combined/-pg machinery was where the numbers were wrong: upstream warms a fused instance with a full prompt pass plus exactly ONE generated token, not a whole fused pass, so the snapshot's tokens divisor was 2560 instead of 2433 and joules_per_request swung 34.19 -> 31.60 J (8.2%) for identical compute purely because model load got shorter. While I was testing, another agent fixed those two in the shared tree (verified: tokens=2433, J/req=33.64 across 2/8/25 s loads vs a hand-computed truth of 34.0 J). What is still live is the same wrong warm-up assumption in row_compute_seconds(), which leaves compute_s 19.8% long and load_joules ~49% short on the fused row, an active_joules that is on a different basis from the rest of its own row, and a ~2-3% disagreement between run_matrix.py and energy_window.py on the same window. The regression fixture cannot see any of this because the fake llama-bench enacts the same wrong warm-up shape.

### Findings (9)

**[HIGH] SNAPSHOT (now fixed upstream of me): tokens_of() gave a fused -pg row (n_p + n_g) * (reps + 1) = 2560 tokens. Upstream llama-bench warms a fused instance with a FULL prompt pass plus test_gen(ctx, 1) -- one token -- so the true count is n_p*(reps+1) + n_g*reps + 1 = 2433. Overstated by 127 tokens (+5.22% at r=3, +7.08% at r=2, +11.0% at r=1).**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:852`

*Reproduction:*
```
WebFetch of https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/llama-bench/llama-bench.cpp returned the warmup block verbatim: `if (t.n_prompt > 0) { ... test_prompt(ctx, t.n_prompt, t.n_batch, t.n_threads); } if (t.n_gen > 0) { ... test_gen(ctx, 1, t.n_threads); }` -- both branches fire for a -pg instance. Then `python3 /tmp/tmp.53otbwKo79/t/t_tokens.py` against the snapshot: 'fused 512+128 r3  code=2560  truth=2433  +5.22%'; 'fused 512+128 r2  code=1920  truth=1793  +7.08%'; prefill and decode rows matched truth exactly at both r. docs/10_measurement_protocol.md:225 documented the wrong formula, and bench/fixtures/run_regression.sh:89 asserted `tokens_of(...512,128,r=3) == 2560` while line 92 noted in the same breath that the split halves sum to 2433. Note: docs/41_finding3_analysis.md:405 had already proposed the CORRECT formula; the shipped code diverged from its own analysis. Re-checked the live tree at md5 ec8fb0e7b92c6ede072de71b98355477: tokens_of now returns 2433 and a live driver run writes tokens=2433 -- already fixed by a concurrent agent.
```

*Why it matters:* `tokens` is published on the combined row as the provenance divisor. Any reader who divides that row's joules by it -- which is exactly what the column invites -- gets a J/token 5% low, and the bias is not constant across the sweep because it depends on -r.

*Fix:* Already applied in the live tree: drop the `if n_p and n_g` special case so the single formula n_p*(reps+1) + n_g*reps + (1 if n_g) covers all three shapes. Also update docs/10_measurement_protocol.md:225 and the run_regression.sh:89 assertion, which still encode 2560 in my snapshot.

**[HIGH] SNAPSHOT (now fixed upstream of me): joules_per_request was energy/(reps+1) over the compute_s-trimmed window, and both halves were wrong -- the warm-up is not a whole request, and compute_s was long enough to drag model-load energy into the numerator. The published number moved 8.2% for identical compute work when only the model-load time changed.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:861`

*Reproduction:*
```
Built an upstream-accurate fake llama-bench (warm-up = prompt pass + 1 token) plus the shipped fake tegrastats (1900/5200/7400/4800 mW), PP=2.0 s and TG=4.0 s per rep, r=3, and ran the SNAPSHOT driver twice in /tmp/tmp.53otbwKo79/kit2 changing only HS_FAKE_LOAD_S. Results: LOAD_S=8 -> compute_s=24.0, compute_joules=136.765, joules_per_request=34.1912; LOAD_S=2 -> compute_s=24.0 > duration_s=21.832 so the trim was refused and it fell back to 'whole-process (includes model load)', joules_per_request=31.5956. An 8.2% swing from load time alone. Hand-integrating the power CSV by phase for the LOAD_S=8 run gave prefill 58.53 J + decode 58.45 J = 116.98 J of true compute against the reported compute_joules of 136.765 (+16.9%), and true load 41.32 J against the reported load_joules of 20.83 (-50%). Control: the split prefill/decode rows in the same harness were load-independent -- pp512 J/token 0.028222 (LOAD_S=8) vs 0.028230 (LOAD_S=2), tg128 0.147353 vs 0.147339. Re-ran the LIVE tree (md5 ec8fb0e7...) with the same fake at LOAD_S 8/2/25: joules_per_request = 33.643 / 33.640 / 33.643 over request_window_s=18.0 s, against a hand-computed truth of 3 x (2 s @ 7.4 W + 4 s @ 4.8 W) / 3 = 34.0 J (-1.05%, the residual being 5 Hz sample-edge snapping). Already fixed.
```

*Why it matters:* joules_per_request is the only number the end_to_end_request case produces, and it was a function of how long the GGUF took to come off the SD card -- the precise confound the compute_s trim exists to remove. Two independent errors partially cancelled at one load duration, which is worse than either alone because the row looks plausible.

*Fix:* Already applied: requests_of() now counts the timed repetitions only and request_window_s() = avg_ns * reps anchors the numerator to the same loops, so no warm-up estimate is needed. Verified load-independent above. Worth keeping: the guard `0 < rws <= dur` is structurally safer than `0 < cs < dur`, because cs was inflated by roughly one whole decode pass and so the trim only ever applied when model load exceeded a full n_gen generation.

**[MEDIUM] STILL LIVE: row_compute_seconds() uses warm_ns = ns for a fused row, i.e. it assumes the warm-up is a whole fused pass. compute_s comes out 19.8% long, and because load_s = duration - compute_s, load_joules is understated by ~49%. At short model-load times cs exceeds the window and compute_joules / load_s / load_joules are silently omitted from that row while every other row carries them.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:351`

*Reproduction:*
```
Live tree (md5 ec8fb0e7b92c6ede072de71b98355477) copied to /tmp/tmp.53otbwKo79/kit3, driven with the upstream-accurate fake at three load durations. compute_s = 24.0 s in all three, against a truth of ns*reps + (prompt pass + 1 token) = 18.0 + 2.031 = 20.031 s (+19.8%). LOAD_S=8: load_s=3.84 (truth ~7.8 after edge snapping), load_joules=20.82 (truth ~40.6, -49%). LOAD_S=25: load_s=20.86 (truth ~24.8), load_joules=109.35 (truth ~130, -16%). LOAD_S=2: compute_s 24.0 > duration_s 21.831, so compute_joules, load_s and load_joules are all blank on that row. The identical assumption is still in telemetry/energy_window.py:246-247 (`if p: warm_ns = ns`, which catches the fused case first), file unchanged at md5 0d5df288eeafe0c563d8382ba2003bb3.
```

*Why it matters:* load_joules is the study's evidence for how much of a window is model load rather than tokens -- the quantity the whole trim exists to expose -- and on the fused row it is off by half. The blank-at-short-load behaviour also means the same column is populated or not depending on SD-card timing, which is not a property a reader can see.

*Fix:* A fused row's avg_ns cannot be split into its pp and tg parts from the row alone, so either (a) estimate warm_ns from the matching split rows of the same (model, ngl) -- the matrix already measures pp512 and tg128 separately -- or (b) stop reporting compute_s / load_s / load_joules on a combined row at all and report only request_window_s and request_joules, which need no warm-up estimate. Do not leave warm_ns = ns as an 'upper bound': it is not reported as a bound, it is reported as a measurement.

**[MEDIUM] STILL LIVE: a fused row carries two mutually inconsistent 'active' figures. active_joules is computed on the compute_s basis while active_joules_per_request is computed on the request-window basis, and energy_basis says 'timed requests' for the whole row. active_joules_per_request * requests does not equal active_joules.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:721`

*Reproduction:*
```
Live tree run at LOAD_S=8: active_joules=91.2083, active_joules_per_request=22.243, requests=3, energy_basis='timed requests'. 22.243 * 3 = 66.73, not 91.21 -- a 37% gap. The cause is that line 721 sets active_joules from (energy_j, window_s) = (compute_joules, compute_s) fixed at lines 713-716, before the per-request block at ~734-737 recomputes act = rj - idle_mw/1000*rws and overwrites energy_basis without touching active_joules. At LOAD_S=2 the row shows active_joules=84.9025 (whole-process basis, since the compute trim was refused) beside the same 22.2404 per-request figure -- so the two numbers are not even on a consistent basis from run to run.
```

*Why it matters:* energy_basis is the single column a reader is told to consult to know what the energy numbers mean. On a combined row it describes some of them and not others, and the totals do not reconcile with the per-unit figures the thesis would quote.

*Fix:* On a PT_COMBINED row either recompute active_joules from (request_joules, request_window_s) so the row is internally consistent, or drop active_joules on that row entirely and keep only active_joules_per_request. Add an invariant to bench/check_finding3_invariants.py asserting active_joules_per_request * requests == active_joules (the checker's rule 6 currently only checks joules_per_request, not the active pair).

**[MEDIUM] STILL LIVE: run_matrix.py and telemetry/energy_window.py -- documented as twins that must be kept in step -- produce different energy numbers from the same power CSV and the same bench row: J/token differs by 1.8% and active J/token by 3.0%.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/energy_window.py:343`

*Reproduction:*
```
Same decode window, same tg128 bench row (/tmp/tmp.53otbwKo79/fix2/results/*split2_002.csv), --reps 3 --idle-mw 1900. energy_window.py: compute_joules 57.7322, load_joules 40.6133, joules_per_token 0.149954, active_joules_per_token 0.090639. run_matrix.py's joined CSV for the same window: compute_joules 56.731, load_joules 41.6145, joules_per_token 0.147353, active_joules_per_token 0.087979. The trailing slice is anchored differently: run_matrix uses rec['t_end_unix'] = time.time() taken after subprocess.run returns, energy_window uses res['end_unix'] = the last power sample's ts_unix inside the marker window, so the two slices are offset by up to one sample interval; and run_matrix subtracts the idle floor over the nominal cs while energy_window subtracts it over the sampled sub-window duration.
```

*Why it matters:* The measurement protocol offers energy_window.py as the way to re-derive a published figure by hand. Two tools that disagree by 2-3% on identical input mean a spot-check can never distinguish a real error from tool skew, and 3% is the same order as several of the effects the study is trying to resolve.

*Fix:* Have both tools derive the slice the same way: pick one anchor (the last sample inside the window is the defensible one, since that is where the energy counter was actually read) and subtract the idle floor over the SAMPLED duration of the slice in both. Then add a regression step that runs energy_window.py over a driver-produced power CSV and asserts the two J/token agree to, say, 1e-6.

**[MEDIUM] STILL LIVE: energy_window.py emits module_active_joules computed over the WHOLE window next to energy_basis='compute' and a compute-scope active_joules_per_token. The two 'active' numbers on one record differ by 74%.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/energy_window.py:356`

*Reproduction:*
```
`python3 telemetry/energy_window.py <split2_power.csv> --marker decode:dec:002 --bench <split2_002.csv> --reps 3 --idle-mw 1900 --json` gives module_active_joules=60.6704 (= 98.3455 - 1.9*19.829, the whole 19.8 s window), energy_basis='compute', compute_joules=57.7322, active_joules_per_token=0.090639. But 0.090639 * 385 tokens = 34.90 J -- the compute-scope active. 60.67 vs 34.90 is +74%. Line 356 rebinds the local `active` to ce's value but never updates res['module_active_joules'], and the human-readable printer at lines 397-399 prints the stale whole-window value under the heading 'active ... (idle floor removed)' directly above the compute-basis J/tok line.
```

*Why it matters:* The printed block is what a student reads when spot-checking a row. It shows a compute-basis J/token and a whole-window active energy side by side with no indication they are different scopes, and anyone who reconstructs active-J/token from the printed active figure gets a number 74% too high.

*Fix:* When the trim succeeds, overwrite res['module_active_joules'] (and its idle_mw label) with ce's compute-scope value, or emit it under a distinct key such as compute_active_joules and print both with their scope named.

**[LOW] STILL LIVE: active_joules subtracts the idle floor over the NOMINAL compute_s, while compute_joules is the counter difference between the first and last power sample actually inside that slice -- a shorter interval. The idle floor is therefore over-subtracted by up to about one sampling interval.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:721`

*Reproduction:*
```
For the realwarm fused run: nominal cs = 24.000 s, sampled slice duration = 23.829 s, shortfall 0.171 s -> 0.325 J over-subtracted at a 1900 mW floor. Reported active_joules 91.165 vs 91.490 using the sampled duration (0.36% here). The error is bounded by (1/hz + the gap between the last sample and process exit) * idle_mw and scales inversely with window length, so it is largest on the short decode rows -- at 5 Hz on a 3 s tg32 window it is ~0.4 J against roughly 8 J of active energy, about 5%.
```

*Why it matters:* Decode windows are the short side of the study's headline prefill-vs-decode comparison, so a bias that grows as the window shrinks lands disproportionately on one arm of the comparison.

*Fix:* Return the sampled duration from energy_between() (it already computes duration_s for the sub-window) and subtract the idle floor over that, not over cs. Same change makes run_matrix agree with energy_window, which already does it this way.

**[LOW] The regression fixture cannot detect any fused-path error, because bench/fixtures/fake_llama_bench.py's -pg branch enacts the same wrong warm-up the driver assumes. '19 passed, 0 failed' is circular for the combined case.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/fixtures/fake_llama_bench.py:37`

*Reproduction:*
```
The -pg branch sleeps (PP_NS*reps + PP_NS)/1e9 then (TG_NS*reps + TG_NS)/1e9 -- i.e. a full extra decode repetition as warm-up. Timed it directly with HS_FAKE_LOAD_S=0: 6.032 s of compute wall time, against the driver's row_compute_seconds() of exactly 6.000 s and an upstream-accurate 0.5*4 + 1.0*3 + 1.0/128 = 5.008 s. So the trim lines up perfectly in the fixture and cannot miss. I reproduced 19 passed / 0 failed on the snapshot in my own copy (/tmp/tmp.53otbwKo79/kit), and the substantive divergence only appeared once I substituted a fake whose -pg warm-up is a prompt pass plus one token.
```

*Why it matters:* The regression is the stated evidence that finding #3 is fixed and stays fixed. For the one case in the matrix that is genuinely fused, it asserts the implementation against itself.

*Fix:* Change the -pg branch to sleep PP_NS once, then TG_NS/pg_g once, then reps iterations of (PP_NS, TG_NS) -- as the harness I used in /tmp/tmp.53otbwKo79/fix2/bin/llama-bench does -- and add a regression assertion that computes the expected joules_per_request from the fake's own mW table (3 timed reps at 2 s @ 7.4 W + 4 s @ 4.8 W = 34.0 J) rather than only checking that the status string is refused_combined_request. Run the fused case at two different HS_FAKE_LOAD_S values and assert the two J/request agree.

**[LOW] idle_floor() averages every sample whose marker is 'idle', pooling the one 30 s pre-sweep baseline with the 20 s cooldown tail after each of the 32 runs, so the floor is dominated by a cooling board rather than a cold one and active_joules is biased low.**

`/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py:807`

*Reproduction:*
```
Not reproduced numerically -- the fake tegrastats returns a flat 1900 mW for the idle phase, so no thermal/governor tail exists to measure. Composition demonstrated instead: on my 2-run fixture sweep, 185 idle-marked samples split 150 baseline / 35 cooldown; on the shipped 32-case matrix it is one 30 s baseline against 32 x 20 s of cooldown, i.e. ~96% of the samples defining idle_mw are post-load tails. Sensitivity measured from real rows: a +200 mW error in the floor moves active_joules by 1.60 J on the pp512 row (3.8% of 42.6 J) and 2.41 J on the tg128 row (7.1% of 33.9 J). Already catalogued in docs/41_finding3_analysis.md as a known item; listed here because it is the largest remaining uncertainty in active_joules and it is not a fixed offset -- it varies with where in the sweep a case sits.
```

*Why it matters:* active_joules is the figure that claims to isolate workload energy from the board's floor, and its denominator-equivalent is estimated from samples that are not actually idle. The bias is larger on decode rows (lower absolute active energy) than on prefill rows.

*Fix:* Restrict idle_floor() to the pre-sweep baseline window (samples before the first run's t_start), or drop the first N seconds of each cooldown before admitting its samples, and report the two floors separately so the difference is visible as a thermal-drift diagnostic rather than averaged into one scalar.

### Attacks that FAILED — evidence the fix holds
- interleave_groups() is a correct permutation. Tested with an odd group (3 prefill runs vs 1 decode), three cases sharing one group, cases with different model lists, a group split by ungrouped cases on both sides, and the shipped 32-run matrix. Multiset of runs preserved in every case, nothing dropped or duplicated, ungrouped runs held their exact positions (indices [0,3] in, [0,3] out), and the shipped matrix's partial_offload pair came out prefill,decode x5 as the regression asserts. The one wrinkle -- with mismatched model lists it pairs by position, so prefill/m1 lands next to decode/m2 -- is not reachable from the shipped matrix, where every grouped pair has identical model and ngl lists.
- tokens_of() for prefill and decode is exactly right. Derived independently from the upstream warmup block (full prompt pass when n_prompt>0, test_gen(ctx,1) when n_gen>0): prefill 512 r3 = 2048, decode 128 r3 = 385, prefill 512 r2 = 1536, decode 128 r2 = 257, decode 32 r3 = 97 -- code matched truth at 0.00% on all five. telemetry/energy_window.py's bench_tokens() returns the identical values, so the twins are in step here.
- row_compute_seconds()'s ns/n_gen warm-up term for a tg row is correct. avg_ns is the mean nanoseconds of one repetition, a repetition generates n_gen tokens, and the warm-up is exactly one token, so ns/n_gen is the right per-token scaling: 1e9 ns, 128 tokens, r=3 -> 3.008 s, which is what the fixture asserts and what the fake actually sleeps. The pp arm (warm_ns = ns, a full prompt pass) is likewise correct.
- The trailing-window trim really does isolate compute on single-phase rows. Hand-integrating the power CSV by phase level for the split run gave true compute energy of 59.06 J (pp512) and 57.73 J (tg128) against reported compute_joules of 57.80 and 56.73 -- within 2.1% and 1.7%, the residual being 5 Hz sample-edge snapping in the expected (conservative) direction. Reported load_joules 41.88 / 41.61 against a hand-integrated 40.62 / 40.61.
- J/token on split rows is load-independent, which is the property the trim exists to provide. Running the identical split workload with model load at 8.0 s and at 2.0 s gave pp512 J/token 0.028222 vs 0.028230 (0.03% apart) and tg128 0.147353 vs 0.147339 (0.01% apart). This is the control that isolates the fused-path failure as fused-specific rather than a general defect in the trim.
- active_joules does NOT pair compute_joules with a whole-window duration, which was the specific mismatch I was asked to look for. Lines 713-721 bind (energy_j, window_s) together as either (compute_joules, cs) or (joules, dur); the scopes match. The only residual is the nominal-cs-vs-sampled-duration gap reported as a separate low finding (0.171 s / 0.325 J measured).
- build_cmd()'s claim that `-p 0 -n 0` silences llama-bench's default standalone pp512/tg128 tests is true. Confirmed from upstream get_cmd_params_instances: `for (const auto & n_prompt : params.n_prompt) { if (n_prompt == 0) { continue; } ... }` and the same guard for n_gen, with -pg handled by a third loop. So a -pg invocation really does emit exactly one row, and phase_rows() is defensive rather than load-bearing here.
- check_finding3_invariants.py rule 6 (each per-unit figure equals the energy it names over its own denominator) held on every CSV I generated, including the fused rows. It is genuine self-consistency checking -- it just cannot catch a divisor that is internally consistent and physically wrong, which is what the fused token count was.
- The refusal machinery works as advertised: the mixed-case refusal fires at --dry-run naming the case, HS_FAKE_EXTRA_ROW produces two refused_multi_phase rows with every per-token column absent from the file (not blank-but-present), a failed run yields exactly one row with status=failed and no J/token, and energy_window.py refuses both the count-mismatched positional join and the fused row without letting the avg_ts x duration fallback rescue either. I reproduced all 19 regression checks passing in my own copy before attacking them.
