# Measurement protocol

How to take a number from this node that you can defend in the report.

## 1. Fix the conditions before you measure

Every one of these is a condition, and every one must be recorded with the
result — `60_collect_node_facts.sh` captures them into the node facts JSON, and
`run_matrix.py` copies the relevant ones onto each row.

| Condition | Set by | Recorded as |
|---|---|---|
| Power mode (5 W / 10 W) | `nvpmodel -m` | `power.nvpmodel` |
| Clocks locked | `jetson_clocks` | `power.clocks_locked` |
| Fan | `target_pwm` | note in the log |
| llama.cpp ref and backend | `20_build_llama_cpp.sh` | `build_info.json` |
| Model bytes | `30_fetch_models.sh` | `models.sha256` |
| Link (wired GbE / Wi-Fi) | cabling | `--tag` on the sweep |
| Thermal starting point | wait | `temp_start_c` |

A result whose conditions are not recorded is not a result.

## 2. The energy chain

**One instrument, not two.** This kit reads the on-board INA3221 and nothing
else. An external inline meter appears below only as future work — the cluster
does not have one, and no script here reads one.

**On-board INA3221, read via `tegrastats`** at 1–5 Hz. Gives `POM_5V_IN`,
`POM_5V_GPU`, `POM_5V_CPU` in milliwatts. High enough rate to attribute energy
to a prefill window versus a decode window.

### What `POM_5V_IN` actually covers

On the Nano **B01 developer kit** the INA3221 sits on the **carrier** board, not
on the module — it enumerates at `/sys/bus/i2c/drivers/ina3221x/6-0040`. So
`POM_5V_IN` is the whole dev kit's 5 V input: module *plus* carrier, *plus*
whatever is drawing off the USB ports and HDMI. It is **not** module-only power.
"Module only" is a TX2/Xavier property; repeating it for the Nano understates the
board. What `POM_5V_IN` does exclude is conversion loss in the wall adapter ahead
of the barrel jack.

`POM_5V_GPU` and `POM_5V_CPU` are genuine SoC-rail measurements, so the
prefill-vs-decode attribution built on them stands. `POM_5V_IN` carries a fixed
carrier-and-peripheral offset on top. Keep the peripheral set **identical across
every run** so that offset stays constant, record it in the node facts, and
always report the idle floor next to the total so a reader can see how much of
the figure it is.

> **Confirm on the board before any of this reaches the thesis:**
> `ls /sys/bus/i2c/drivers/ina3221x/` — a `6-0040` node is the carrier-board
> INA3221, i.e. the whole-devkit scope described above.

### What the sample rate bounds, and a two-minute calibration

`TELEMETRY_HZ` sets the resolution on every window boundary and on any structure
inside a window. **It does not average out with repetitions.** Every repetition
is sampled on the same grid relative to the workload, so a transient shorter than
the interval is missed identically each time and the spread across repetitions
comes back small — reassuringly, and wrongly. A tight repetition spread is not
evidence the rate was sufficient.

The shipped default is now **10 Hz**, not 5. Check what the board actually
achieves before trusting it, once, and put the answer in the experiment log as
the stated sampling uncertainty:

```bash
# 60 s of idle, then 60 s under a decode load, at the rate you intend to use
python3 telemetry/tegrastats_logger.py --out /tmp/cal.csv --hz 10 --duration 120 &
sleep 60; ./scripts/50_bench_device.sh --only gpu_decode_sweep --no-power; wait
python3 - <<'EOF'
import csv
r = list(csv.DictReader(open("/tmp/cal.csv")))
t = [float(x["ts_unix"]) for x in r]
p = [float(x["pow_in_mw"]) for x in r if x["pow_in_mw"]]
gaps = [b - a for a, b in zip(t, t[1:])]
rep = sum(1 for a, b in zip(p, p[1:]) if a == b)
print("achieved %.1f Hz, worst gap %.3f s, rail repeated %.0f%% of samples"
      % ((len(t) - 1) / (t[-1] - t[0]), max(gaps), 100.0 * rep / (len(p) - 1)))
EOF
```

A high repeat fraction means tegrastats is re-printing a value the INA has not
refreshed: the effective bandwidth is below the nominal rate, and raising `--hz`
further buys nothing. Record the rate you chose and that fraction together.

Every joined row carries `sample_gap_max_s` and `pow_in_plateau_frac` so this
bound travels with the number rather than living only in a logbook.

**What is not validated, and cannot be here.** The INA3221's absolute accuracy.
There is no reference instrument on this cluster, so every joule reported is as
accurate as that chip and no better. `joules_vs_mean_pct` on each row compares
the running integral against mean power × duration — two arithmetics over the
*same* samples, so it catches a bookkeeping error and proves nothing about the
samples. tegrastats' own running average is not a second instrument either: same
ADC, averaged. Say this in the limitations section rather than implying the
energy figures carry a validated uncertainty.

### Across-device comparison — deferred, not done

Comparing joules per token between the Jetson and the Raspberry Pis needs one
instrument that reads all four nodes the same way. The Pis have no on-board power
telemetry, so that has to be an external inline meter on each supply lead — and
**the cluster does not have one.**

Until a meter is acquired, this study reports **within-device** energy only:
prefill vs decode, CPU vs GPU placement, and quantisation, all measured on the
Jetson against itself. Cross-node joules-per-token is future work and must not
appear in a results table. Latency and throughput across nodes are unaffected;
those need no power instrument.

### How joules are computed

`tegrastats_logger.py` keeps a running trapezoidal integral of each rail in the
CSV (`energy_in_j`). Window energy is then a subtraction at the window edges,
which is immune to dropped or resampled rows downstream. Windows are delimited
by a marker file that the benchmark driver writes — the logger stamps whatever
label it reads onto each sample.

### Idle subtraction

The board burns roughly 1.5–2 W doing nothing. On a 10 W budget that is most of
a short decode window. So every energy figure is reported **twice**:

- `joules` — total energy over the window
- `active_joules` — net of the idle floor (`idle_mw`) measured for **that case**

Report both. Quoting only total makes the Nano look worse than it is; quoting
only active hides the cost of keeping the node powered at all. For a
"what does this cluster cost to run" claim, total is the honest number.

**The floor is local, not a constant of the board.** SoC leakage rises with
junction temperature, so a fanless Nano an hour into a sweep idles several
hundred milliwatts above where it idled cold. One scalar subtracted from every
case therefore over-subtracts from the early cases and under-subtracts from the
late ones — a bias aligned with case order, which is exactly the axis a reader
cannot separate from the axis under study. So the floor on each row is the mean
power over **the idle interval that ended just before that case opened its
window**, with its first 3 s dropped for clock and fan ramp-down. For the first
case that interval *is* the 30 s pre-sweep baseline, so there is one definition
rather than two.

`idle_mw_source` names the interval used. `idle_mw_baseline` (the cold pre-sweep
floor) and `idle_mw_pooled` (the whole-file idle mean, which is what this used to
subtract) are carried on every row for comparison — neither is subtracted, and
both are there so an older figure stays reproducible.

The node facts `power.idle_board_mw` is a **different number for a different job**:
a cold-board device-table figure, recorded once. It is not the floor any
particular window ran against, and `idle_board_mw_basis` in that JSON says so.

Nothing is clamped at zero. A negative `active_joules` means the floor is wrong
for that row — too high, or measured in a different thermal state — and the old
`max(x, 0.0)` published a plausible 0.0 while throwing the evidence away.

## 3. Cross-node time alignment

Energy per token for a *split* run needs the four nodes' power logs on one
timeline. `00_system_prep.sh` enables `chrony`; point every node at the
orchestrator (or any LAN NTP source) so skew stays in the low milliseconds.

Verify before a cluster experiment:

```bash
chronyc tracking | grep -E 'System time|Last offset'
```

At 5 Hz sampling, anything below ~50 ms of skew is invisible. Above that, the
prefill/decode boundary smears across nodes and the attribution is wrong.

## 4. Single-node sweep

```bash
tmux new -s bench
cd ~/HeteroServe/Jetson
./scripts/50_bench_device.sh --tag wired-gbe
```

The driver:

1. collects a 30 s idle baseline,
2. runs every point of every enabled case — 32 runs with the shipped matrix —
   each in its own power window covering **exactly one phase**,
3. cools down between runs so heat does not carry over,
4. joins throughput to energy and writes one row per `llama-bench` test row into
   `results/<node>_<stamp>_<tag>_bench_energy.csv`.

**Every run produces at least one row.** A run that failed (an OOM on 4 GB is
normal for the larger models), was skipped because its GGUF is missing, or emitted
no parseable CSV, produces exactly one row with `status` and `per_token_status`
set and the bench columns blank; the sweep continues. The joined CSV is the
archived artefact — a run that appeared only on stdout would be indistinguishable
later from a run that was never in the matrix.

### One window, one phase

**No case may set both `n_prompt` and `n_gen`.** `llama-bench -p 512 -n 128` is
not one measurement — it is two independent tests, a `pp512` and a `tg128`, run
back to back inside a single marker window. The window's joules therefore cover
prefill *and* decode, while the CSV's throughput belongs to one test alone. A row
built from those pairs a decode tokens/s with prefill-dominated energy, which is
the exact defect that made the placement sweep unreadable: raising `-ngl` mostly
speeds prefill, so the energy column moved while the throughput column did not,
and the energy Pareto pointed at the wrong layer split.

So the matrix ships a **prefill case and a decode case for every placement and
quantisation point**, and `expand()` refuses a mixed case by name at `--dry-run`
time rather than two hours into a sweep. Splitting costs nothing measured: the
token count is bit-identical either way, because `llama-bench` already ran the two
tests independently with a cleared KV cache. What it buys is that each phase's
energy can be attributed, and that the two curves can be recombined in analysis —
the energy of any hypothetical request is
`P·(J/tok)_prefill + G·(J/tok)_decode`, so the prompt-to-generation ratio can be
varied without re-measuring. The fused form gave one point at one fixed ratio.

The two halves of a split case share a `group`, and runs inside a group are
**interleaved** — prefill, decode, prefill, decode — so the two phases of one
configuration are measured at nearly the same board temperature. On a fanless
Nano, position in the run order is confounded with temperature; putting the halves
hours apart would fold thermal drift into the phase comparison that is the point
of the study.

For a deliberately fused request there is `"pg": [n_prompt, n_gen]`, which emits
`-pg`. That builds a **single** test instance and a single CSV row, so it is one
window, one row, one phase label — `combined`. It reports `joules_per_request`
and never `joules_per_token`, because a fused window's joules genuinely cannot be
attributed to prompt or to generation. That is a different denominator, not a
missing number.

Its energy comes from the **timed repetitions only** — the trailing
`avg_ns × r` seconds — not from `compute_s`. llama-bench warms a `-pg` instance
with a prompt plus a single token, which is neither a whole request nor separable
from `avg_ns`; counting it as a request would divide by more requests than were
served and understate the number. Restricting numerator and denominator to the
timed loops needs no estimate at all.

### Reading the output

One row per `llama-bench` test row, not one row per case.

| Column | Meaning |
|---|---|
| `phase` / `row_phase` | what the window was asked for, and what this row's test actually was — `prefill`, `decode` or `combined`. They differ only if a ref returned something unexpected |
| `test` | llama-bench's own label: `pp512`, `tg128`, `pp512+tg128` |
| `run_index`, `group` | position in the run order, and the split pair this run belongs to. Case order is confounded with board temperature; `run_index` is what makes that auditable |
| `avg_ts` | llama-bench tokens/s **for this row's test**, never a fused number |
| `bench_n_prompt`, `bench_n_gen` | this row's own per-repetition counts, as the CSV reported them. One is always 0 except on a `combined` row |
| `bench_rows`, `working_rows`, `bench_row_index`, `window_shared` | how many tests shared this power window, and which one this row is. `window_shared=1` means sibling rows repeat the same `joules` — **do not sum them** |
| `per_token_status` | why a per-token figure is or is not on this row — see below |
| `joules` | energy over the **whole** window, model load included |
| `load_s`, `load_joules` | the part of the window spent loading the model and starting CUDA |
| `compute_s`, `compute_joules` | the trailing part that actually processed tokens. `compute_s` is the **measured** span the energy covers; `compute_s_nominal` is what `avg_ns` implied |
| `compute_anchor_unix` | where the trim was anchored: the last sample still carrying this run's marker, not the wall-clock process exit |
| `req_n_prompt`, `req_n_gen`, `counts_mismatch`, `counts_source` | what the invocation asked for, whether the ref's own counts agreed, and whether the divisor came from the measurement or the request |
| `idle_mw`, `idle_mw_source` | the idle floor subtracted from this row and the interval it was measured over — see §2 |
| `idle_mw_baseline`, `idle_mw_pooled` | the cold pre-sweep floor and the whole-file idle mean, for comparison only; neither is subtracted |
| `power_samples`, `sample_gap_max_s`, `pow_in_plateau_frac` | how well sampled this window was: the count, the longest hole the integral crossed on a straight line, and how often the rail simply repeated — see §2 |
| `joules_vs_mean_pct` | the running integral vs mean power × duration. A bookkeeping cross-check over the same samples, **not** evidence the samples are right |
| `energy_basis` | which slice the per-unit columns were built from: `compute` (the trailing `compute_s`), `timed requests` (a fused row's `avg_ns × r`), or `whole-process (includes model load)` when no trim could be made — treat the last as an upper bound. `active_joules` is always on the same basis as this says |
| `joules_per_token`, `wh_per_mtok`, `active_joules_per_token` | the energy figures the tables use |
| `joules_per_request`, `wh_per_krequest` | the `combined` case's figures instead of the per-token ones |
| `request_joules`, `request_window_s` | the energy and duration of the timed repetitions alone, which is what those per-request figures are built from |
| `tokens` | the divisor — see below |
| `requests` | the `combined` divisor: the **timed** repetitions, warm-up excluded |
| `bench_csv` | the per-run `results/<base>_NNN.csv` this row was parsed from |
| `model_path` | the GGUF the driver looked for. Present on `status=skipped` rows, so a missing-model gap names the file it wanted |
| `bench_n_gpu_layers`, `bench_model_filename`, `config_mismatch` | what the ref said it ran, and whether that matched the invocation. A ref that ignores `-ngl` would otherwise publish CPU numbers under `ngl99` |
| `precached` | whether the GGUF was pre-read before the window opened |
| `temp_baseline_c`, `temp_start_c`, `temp_drift_c` | the cold board, where this case started, and the difference — **this is the drift figure** |
| `temp_end_c`, `temp_max_c`, `temp_rise_c` | within-window temperatures. `temp_rise_c` is kept for continuity but is **not** a contamination test — see below |
| `cooldown_actual_s`, `cooldown_reached` | how long the gate actually waited, and whether the board came back within `cooldown_target_c`. Blank `cooldown_reached` means no gate ran (`--no-power`) |
| `cpu_freq_start_mhz`, `cpu_freq_min_mhz`, `gpu_freq_start_mhz`, `gpu_freq_min_mhz` | the clocks this case ran at. A case clamped by heat is otherwise indistinguishable from one that was not |
| `drift_control` | 1 on the repeat of case 1 that closes every sweep |
| `swap_peak_mb` | > 0 means the model did not fit; the number measures swap, not the GPU |

**`per_token_status`** — a blank per-token cell otherwise means at least five
different things, and the ones that matter for defensibility are indistinguishable.
Every row says which it is:

| Value | Meaning |
|---|---|
| `ok` | the per-token columns are populated and trustworthy |
| `refused_combined_request` | a fused `-pg` row; see `joules_per_request` |
| `refused_multi_phase` | more than one test shared this power window, so its joules cannot be attributed. Also fires when a window has several rows and the driver cannot tell which one did the work. The shipped matrix never produces this — seeing it means a case was edited to mix phases, or the installed ref emitted unexpected rows |
| `refused_phase_mismatch` | the driver asked for one phase and the ref returned another. Neither denominator is trustworthy then, because the window's joules cover work the row's counts do not describe |
| `refused_counts_mismatch` | the ref reported token counts that differ from what was asked for. Same phase, same label, wrong divisor — a ref answering `-p 512` with `n_prompt=128` would publish a figure four times too high |
| `refused_counts_unverified` | the ref reported no counts at all, so the divisor would come from the request rather than the measurement. An unverifiable divisor is not a divisor |
| `refused_config_mismatch` | the ref reported a different `n_gpu_layers` or model file than the invocation asked for. The GPU-offload result is a comparison between two `-ngl` settings, so a ref that silently ignored the flag would invalidate it |
| `estimated_from_throughput` | `energy_window.py` only: the bench CSV had no token columns, so the divisor is `avg_ts × the interval`. An estimate, not a count — never put it in a table |
| `run_failed` | `llama-bench` exited non-zero |
| `model_missing` | the GGUF was not on disk, so the run never happened. The row exists so the gap is visible in the archived CSV rather than only on stdout |
| `no_bench_csv` | it exited 0 but wrote nothing parseable |
| `no_power_window` | no power samples covered the window (`--no-power`, or no `tegrastats`) |
| `no_token_count` | the CSV carried no usable token counts |

`ok` always means the number is really there: a row carrying `ok` has a non-empty
`joules`. Anything else names its reason.

**Thermal drift, and why `temp_rise_c` cannot see it.** `temp_rise_c` is
`temp_end_c - temp_start_c`: a *within-window* delta. It goes to roughly zero
precisely when the board has reached a hot steady state — so the most
contaminated case in a sweep passes a "`temp_rise_c` > 8 °C means contaminated"
rule, and the rule this document used to state was therefore backwards. The
quantity that matters is *between* runs: `temp_drift_c = temp_start_c −
temp_baseline_c`, how far from cold this case began.

Three things address it, and none of them alone is sufficient:

- **The cooldown is a closed loop.** `cooldown_s` is now only a floor; after it
  the driver waits until the board is within `cooldown_target_c` (default 3 °C)
  of the idle baseline, capped by `max_cooldown_s` (default 300 s). It polls
  `temp_thermal_c` — the same column every row is scored on, so the gate and the
  audit column cannot disagree. `cooldown_reached=0` means the cap was hit and
  the board was still hot; blank means no gate ran at all.
- **Split pairs are interleaved**, so the two phases of one configuration sit
  adjacent in time. That protects a *pair*; it does nothing about the sweep-wide
  ratchet.
- **Case 1 is repeated, unchanged, as the last run** (`drift_control=1`). Same
  case, same configuration, hours apart — so the difference between those two
  rows is drift and nothing else. The driver prints the delta at the end. If it
  is large, every conclusion drawn along the run-order axis is suspect, and that
  is a fact about the sweep you want *before* writing it up, not after.

Randomising the order with a recorded seed was considered and not taken: it
fights the interleaving, which depends on a chosen order, and a sweep that cannot
be re-run in sequence is harder to defend than one whose drift is published.

**How the trim is measured.** Two details matter more than they look:

- It is anchored on the **last power sample still carrying this run's marker**,
  not on the wall-clock time the process exited. Teardown, the CSV write and
  interpreter shutdown all sit between those, and anchoring on the later one
  slides the slice forward and drags load energy in.
- The energy counter is **linearly interpolated to the exact slice bounds**. A
  slice's energy taken as the counter difference between the first and last
  sample *inside* it covers up to one sampling interval less than the slice
  asked for, while the full token count is still divided into it. At the default
  5 Hz that understated prefill J/token by about 10 % and decode by about 6 % —
  not a wash, but a bias that differs between the phases and so distorts the
  prefill-vs-decode ratio the study exists to measure. After interpolation the
  residual is about 1 %.

`bench/run_matrix.py` and `telemetry/energy_window.py` implement both of these
identically and are asserted to produce bit-identical `joules_per_token` for the
same window; §5 names the second as the way to re-derive a published figure by
hand, so "close enough" is not enough.

**Why the window is trimmed.** The marker opens before `llama-bench` starts, so
reading a 400 MB–1.1 GB GGUF off the SD card and building the CUDA context are
inside it — seconds of multi-watt work that produced no tokens, and that cost an
`-ngl 99` run more than an `-ngl 0` run. All of it precedes the first timed loop,
so the driver takes the trailing `compute_s` seconds, derived from llama-bench's
own `avg_ns`, and divides *that* energy by the tokens. Raw `joules` stays on the
row. If a ref does not emit `avg_ns` the trim cannot be made and `energy_basis`
says `whole-process (includes model load)` — treat those rows as an upper bound,
not as energy per token.

If a window produced more than one test, the per-token columns are refused
outright rather than trimmed. The trim assumes all non-compute work *precedes*
all compute, which is true of model load but false between two tests:
`llama-bench` builds and frees a context between instances, so unmeasured
multi-watt work sits inside the trimmed slice. Splitting such a window by time
looks obvious and is wrong — the resulting prefill error tracks `-ngl`, the very
axis being swept. Those rows carry `joules` only.

**Why the token count is not just `n_prompt + n_gen`.** Those are *per
repetition*, and `llama-bench` also runs an unmeasured warm-up before each test's
repetition loop, inside the same window. Stated per phase, since no row carries
both any more:

- a prefill row is `n_prompt × (r + 1)` — the warm-up is a full prompt pass;
- a decode row is `n_gen × r + 1` — the warm-up is one generated token;
- a `combined` row is `n_prompt × (r + 1) + n_gen × r + 1` — the *same* formula,
  because llama-bench warms a `-pg` instance with a full prompt pass plus
  `test_gen(ctx, 1)`, a single token, exactly as a `pp` row and a `tg` row do
  between them. Reported for provenance only; never used as a divisor.

That identity is also why **the split changes nothing measured**: prefill tokens
plus decode tokens equal the old combined count at any `-r`.

Counting only `n_prompt` would inflate prefill J/token by (r+1)/r: +33 % at
`-r 3`, +50 % at `-r 2`, so the bias would not even be constant across the sweep.

### The cross-node projection

Alongside its own CSV the driver writes `<base>_results.csv`, the cluster's shared
40-column schema (`Pi4/node/bench_parse.py` defines it; `Pi5/bench/parse_bench.py`
adopted it, which is what makes it the standard). `bench/emit_results_csv.py`
produces the same file from an existing joined CSV.

It is a **projection, not a replacement**. `<base>_bench_energy.csv` stays the
authoritative record of what this node measured; the shared file is the subset
that can stand beside another node's row without lying. Two consequences worth
knowing before you read one:

- **Its energy columns are empty, deliberately.** The control node fills them by
  integrating a power stream over `[t_start, t_end]` and dividing by
  `n × samples`. It cannot read this node's power stream (it wants
  `t_utc/bus_v/current_a/power_w`; tegrastats gives milliwatts and there is no
  current channel), its `sample_hz=28` gap tolerance would flag every row from a
  5 Hz logger, and its denominator omits the warm-up this node's window contains.
  Handing it a window would produce a confident wrong number instead of a blank.
  Cross-node *energy* comparison is deferred anyway, for want of an instrument
  common to all four nodes; what the cross-node table consumes for the placement
  rule is **timing**, and that this node can supply honestly.
- **Every row is flagged `throttle-unknown`.** Pi 4's convention is that an unread
  throttle word fails closed, so an unflagged blank would read as "checked and
  clean". This board has no `vcgencmd` equivalent, so it says so. Whether the
  cross-node table accepts rows carrying that token is a decision for the table's
  owner — the alternative is silence, which would be worse.

A `combined` (`-pg`) row is omitted from the projection and the omission is
printed: the shared schema's `phase` column has no value meaning "both", and
inventing one would make the row group with nothing while still being counted.

`tests/test_shared_schema.py` reads Pi 4's file **live** and fails if the column
list, the quantisation vocabulary or the flag vocabulary drift apart. Run it with
`python3 -m unittest discover -s tests`.

## 5. Ad-hoc windows

For anything the matrix does not cover — a manual `llama-cli` run, a
cluster experiment driven from the laptop:

```bash
# terminal 1
python3 telemetry/tegrastats_logger.py --out run.csv --hz 5 --marker-file /tmp/hs_marker

# terminal 2
echo prefill > /tmp/hs_marker
<your command>
echo idle > /tmp/hs_marker

# afterwards
python3 telemetry/energy_window.py run.csv --marker prefill
python3 telemetry/energy_window.py run.csv --marker prefill --bench bench.csv \
        --reps 3 --out per_token.csv
```

`--idle-mw` defaults to `auto`, which reads each window's floor off the idle
samples preceding it in the same CSV — the same definition the driver uses. To
reproduce a published row exactly, pass that row's own `idle_mw`
(`--idle-mw 1873`); for total energy only, `--idle-mw none`.

**Always pass `--marker`.** Without it the whole logging session becomes one
window, which is then joined to bench row 0 and prints a confident J/tok under a
header reading `marker=idle` — the idle baseline and every cooldown included in
the numerator.

Against a **sweep's** power CSV the marker labels are `phase:case_id:NNN`, and
`--marker` matches either the full label or its `phase` prefix. Which you want
depends on what you are doing:

- **With `--bench`, pass the full label** (`--marker decode:gpu_decode_sweep:012`)
  and the matching `results/<base>_NNN.csv`, where `NNN` is the same run index.
  One window, one row. The tool refuses an unequal count, and also refuses when
  the window's phase and the bench row's own counts disagree — a positional join
  that silently paired each window with another run's numbers was producing
  prefill labels over decode energy.
- **Without `--bench`, the prefix form is the aggregation tool**
  (`--marker decode` gives you every decode window, one summary each, and the
  spread across them is itself a result). Prefix matching never merges two
  invocations: a window breaks on any change of the full label, not merely on
  the prefix ceasing to match.

`--reps` **must match the `-r` the window was recorded with.** A llama-bench CSV
row is one test aggregated over all repetitions, and its `n_prompt` / `n_gen` are
per-repetition counts, so the tool cannot infer the divisor on its own. Left out,
it assumes 1 and the divisor is short by the repetitions that actually ran. What
that does to the numbers depends on whether the bench row carries `avg_ns`:

- **With `avg_ns`** (every modern ref) the compute trim rescales the numerator by
  the same factor as the divisor, so `joules_per_token` barely moves — but
  `tokens`, `compute_s`, `load_s` and `load_joules` are all wrong by about that
  factor, and a reader checking the arithmetic will not be able to.
- **Without `avg_ns`** there is no trim, and `joules_per_token` itself comes out
  **too high** — measured at 2.0× for a 512-token prefill row at `-r 3` and 3.0×
  for a 128-token decode row.

The joined CSV's `repetitions` column has the right value. The printed
`tokens  N   (…)` line spells out the arithmetic used; check it against the
command you ran.

If the bench CSV carries no token columns at all, the tool falls back to
`avg_ts × the interval the numerator covers` and marks the row
`estimated_from_throughput` — never `ok`. That figure is an estimate of an
estimate; do not put it in a table.

**`--bench` requires a window that means something** — `--marker`, or
`--from`/`--to`, *and* the selected window must contain no idle samples. A flag is
not a window: `--from 0 --to 9999999999` passes the first test and puts the idle
baseline and every cooldown into the numerator. Without one the
whole power CSV becomes a single window, idle baseline and cooldowns included, and
it is then joined to bench row 0: a confident J/tok under a header reading
`marker=idle`, measured at 0.001672 against a true 0.00651. The tool refuses
rather than printing it. `--marker` matches the sweep's `phase:case_id:NNN` labels
on their prefix, so `--marker prefill` selects every prefill window and
`--marker prefill:thread_sweep:027` selects exactly one.

### Marking a window during a CLUSTER run

`bench/run_matrix.py` marks its own windows because it starts the work. In the
distributed experiment it does not: this node is an RPC worker, handed tensors
by the orchestrator, and nothing here can know which phase is running. So the
marker is written **from the orchestrator, over ssh**, by
`scripts/45_capture_window.sh`:

```bash
ssh jetson '~/HeteroServe/Jetson/scripts/45_capture_window.sh start'
ssh jetson '~/HeteroServe/Jetson/scripts/45_capture_window.sh mark prefill'
#   ... run the prefill leg against the cluster ...
ssh jetson '~/HeteroServe/Jetson/scripts/45_capture_window.sh idle'
ssh jetson '~/HeteroServe/Jetson/scripts/45_capture_window.sh mark decode'
#   ... run the decode leg ...
ssh jetson '~/HeteroServe/Jetson/scripts/45_capture_window.sh stop'   # prints the CSV path
```

It writes the **same** marker file the sweep uses and lands the CSV in
`$RESULTS_DIR` under the house naming, so the result is read with exactly the
commands above — `--marker prefill`, `--idle-mw auto` — and
`90_collect_results.sh` bundles it without being told it exists.

**What this can and cannot resolve.** An ssh round trip puts each marker edge
within roughly one sample period of the real one. For a leg lasting seconds that
is immaterial. It is *hopeless* for the prefill→decode boundary inside a single
request, which is sub-second: those two phases are not separable this way, and a
cluster run must therefore bracket whole legs — a prompt-only pass, then a
generation-only pass — exactly as §4 splits the local matrix. A figure that
claims per-phase energy from a single mixed request is not supported by this
method.

## 6. Repetition and reporting

- 3 repetitions minimum per case; 5 or more for any number that goes in a table.
- Report median and spread, not a single run.
- If repetitions of one case differ by more than a few percent, something is
  wrong with the conditions — find it before you average it away.
- Keep the raw CSVs. `90_collect_results.sh` bundles them with the node facts,
  the build provenance, the model checksums and the matrix that produced them,
  so a figure can always be traced back to the bytes it came from. **Bundle
  immediately after a sweep** — the `matrix.json` in the tarball is the one on
  disk now, not necessarily the one that ran.
- Case ids changed when the mixed cases were split (`partial_offload_sweep` →
  `partial_offload_prefill` / `_decode`, same for `quantisation_sweep` and
  `largest_local_model`). Rows from before and after the split cannot be
  concatenated by `case_id`, and pre-split rows carry `phase=mixed`. Reject those
  rather than pooling them.

## 7. What this node can and cannot tell you

It **can** tell you: prefill-vs-decode sensitivity to GPU offload on a shared
memory pool; the energy cost of a layer partition on Maxwell; where the 4 GB
wall is and what crossing it costs; how a FP16-only GPU handles K-quants.

It **cannot** tell you anything about INT8 inference, about GPUs with dedicated
memory, or about absolute throughput competitive with an Orin. Those are limits
of the device, and they belong in the limitations section rather than in a
footnote.

It also **cannot** compare energy against the Raspberry Pis. That needs one
instrument reading all four nodes identically; the Pis have no on-board power
telemetry and the cluster has no external inline meter. See §2 — deferred, and it
belongs in the limitations section too.
