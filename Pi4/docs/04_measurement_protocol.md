# 04 — Measurement protocol for the Pi 4 node

What this node contributes to the dataset, and the rules that keep its rows trustworthy.

## The three things this node is for

1. **Isolated characterisation** (Phase 2). Prefill and decode throughput at each
   quantisation, with the matching energy window, on this device alone. These rows feed
   the placement rule's `c_i` (per-layer compute time) and `E_token` terms.
2. **A cluster member** (Phase 3). Serving layer shards over RPC so the cluster can run a
   model no single node can hold.
3. **The straggler** (research question QR8). The node the placement rule should learn to
   under-load or drop. Its value to the study comes precisely from being slow, so do not
   "fix" it — don't overclock it, don't give it a faster card, don't quietly skip it.

## Before every run

```bash
make health          # exits non-zero if the node is not fit to be measured
```

It blocks on, in the order `scripts/99_healthcheck.sh` checks them: any throttling or
under-voltage, a governor other than `performance`, swap enabled, SoC above 70 °C, free
memory under 3000 MB, a missing rpc server binary, a missing `llama-bench`, **a
`llama-bench` that does not support `--progress`**, **missing window tooling**
(`node/stamp_lines.py`, `node/bench_windows.py`), **a kernel without the memory cgroup
controller** (which makes the rpc unit's `MemoryMax` silently inert), a missing build manifest, an unpinned
llama.cpp revision, chrony not synced, clock offset above 5 ms, an unreachable control
node, leftover `tc` shaping, and missing models.

The two in bold are newer than the rest and are the ones most likely to surprise: without
`--progress` the energy window cannot be cut at repetition boundaries, so
`scripts/06_bench_local.sh` **refuses to run at all** rather than silently recording the
old whole-process window. That refusal is a hard `die`, not a warning.

Every one of those silently changes a number without raising an error at the time. Discard
the run rather than the check.

**Two gates, deliberately not identical.** `make health` is what an operator runs by hand;
the agent's `GET /health` is what the control node polls before a cluster run — so in
practice `/health` is the one that gates real runs. They now cover the same preconditions,
with one deliberate exception: an unpinned `LLAMA_PIN_SHA` is a hard failure for
`make health` and only a warning for `/health`, because the first bring-up build has to
happen before a SHA exists and the agent is polled from bring-up onwards. That asymmetry is
pinned by a test; if it is ever changed, change it knowingly.

## Run discipline

- **Cool start.** The sweep sleeps 20 s between runs. On a long sweep watch the trend in
  `temp_end_c` across `windows.jsonl`; if it climbs monotonically the interval is too short
  and late rows are measuring a hot board.
- **One phase per invocation.** `llama-bench` will happily run pp and tg inside one process,
  but then the power trace has no boundary between them and prefill energy cannot be
  separated from decode energy. The sweep therefore calls `-p N -n 0` and `-p 0 -n N`
  separately, bracketed by markers. This is the main reason the sweep is slow; keep it.
- **Repetitions.** `-r 5` for the real sweep. Report the standard deviation, not just the
  mean — the Pi 4's variance under thermal pressure is itself a result.
- **Thread sweep.** 1, 2 and 4 threads. Four is the headline figure; 1 and 2 are the
  straggler study's "reduced load" conditions and also show how far this node is from
  scaling linearly.
- **Nothing else on the board.** No `apt upgrade`, no `tail -f` over SSH on the file being
  written, no monitoring loop polling `vcgencmd` at 10 Hz. The agent's `/metrics` endpoint
  is cheap but do not poll it during a measured window.

## What gets written

`scripts/06_bench_local.sh` creates `$DATA_DIR/bench/<RUN_ID>/` containing:

| File | Contents |
|---|---|
| `<model>_<phase><n>_t<threads>.json` | raw llama-bench output, untouched |
| `<model>_<phase><n>_t<threads>.stderr.tsv` | every llama-bench stderr line, wall-clock stamped — the raw evidence behind the energy window |
| `windows.jsonl` | one line per run: the raw process window `[t_start, t_end]`, the repetition span `[t_meas_start, t_meas_end]`, per-repetition windows, the load/warmup/teardown split, the contemporaneous idle window, start and end temperature and throttle word |
| `build_manifest.json` | the exact revision and toolchain that produced these rows |
| `results.csv` | this node's normalised rows, written by `node/bench_parse.py`. **Not a cluster-wide schema** — Pi5 and Jetson currently emit incompatible columns, so the three cannot be concatenated as they stand |

`results.csv` leaves `energy_j`, `joules_per_token` and `mean_power_w` empty. The control
node's energy collector fills them by integrating the INA226 channel-2 stream over each
row's window. Keeping the join explicit — rather than having the Pi estimate its own energy
— means one collector owns the power data for all four nodes on one time base.

## What is in the numerator of joules per token

The integral runs over the REPETITIONS ONLY. A llama-bench process does much more than the
repetitions, and none of the rest puts a token in the denominator:

| Sub-window | Recorded as | In the headline numerator? |
|---|---|---|
| process start, `ggml_backend_load_all`, context + KV allocation | `t_load_start` .. `t_load_end` | no |
| warmup — a FULL `n_prompt` prefill for pp, exactly one token for tg, and where the mmap'd GGUF actually faults in off the microSD | `t_warmup_start` .. `t_warmup_end` | no |
| repetition i (i = 1..R) | `reps[i]`, start from the stamped `--progress` line, length from llama.cpp's own `samples_ns` | **yes** |
| inter-repetition bookkeeping | `rep_gap_s` | yes, unavoidably — see below |
| teardown, munmap, process exit | `t_teardown_start` .. `t_teardown_end` | no |

`energy_duty = rep_busy_s / (t_end - t_start)` **as read from `results.csv`**, where
`t_start`/`t_end` are the repetition span. The field of the same name in `windows.jsonl` is
computed against `rep_span_s` instead, because `t_start`/`t_end` there are the process
window — the two are the same quantity only because `results.csv` carries the corrected
pair. It is the fraction of the reported window that
produced tokens. It is > 0.99 on a correct run. It is also computable for every row recorded
BEFORE this fix, from `samples_ns` and the old process window — which is how the size of the
old bias can be stated for the pilot data without re-running anything.

Three figures are *intended*, and the thesis must say which is which. **Only the first is
produced today** — the other two are specified here so that the data needed for them is
recorded now, but no code in any node folder emits them yet. Do not cite them as if they
existed:

1. **`joules_per_token`** — repetitions only, model load and warmup EXCLUDED. This is the
   headline: the marginal energy of producing a token on this device in steady state, which
   is what the placement rule consumes and the only definition comparable across models of
   different size and across pp vs tg. Model load is a per-PROCESS cost; dividing it by an
   arbitrary benchmark token count is not a physical quantity.
2. **`energy_load_j`** — startup + warmup, in JOULES, not per token. Reported separately so a
   reader can amortise it over whatever deployment assumption they hold. It is NOT averaged
   across the sweep: the page cache is warm the second time a model is touched, so it is a
   per-invocation figure with its position in the sweep attached.
   **NOT YET PRODUCED.** Everything it needs is already recorded — `load_s` and `warmup_s`
   are in `results.csv`, and `t_load_start`/`t_warmup_end` are in `windows.jsonl` — so this
   is one integration step on the control node, not a re-run.
3. **`joules_per_token_marginal`** — energy above the contemporaneous idle floor,
   `(E_meas - P_idle * T_meas) / (n * R)`. The INA226 is high-side on the whole 5 V rail, so
   it measures SoC, LPDDR4, microSD, Ethernet PHY and the case fan. On a Pi 4 the idle floor
   is a large fraction of the busy draw, and a slow node pays it for longer — so for the
   straggler question (QR8) this separates "inefficient at computing" from "slow, and
   therefore billed idle power for longer". Both are results; they are different results.
   **NOT YET PRODUCED, but every input is now recorded.** The per-run idle window reaches
   `results.csv` as `idle_start`/`idle_end`, alongside the repetition span — so a marginal
   figure can be computed per row, against the settle that immediately preceded that run.
   What is missing is only the consumer. Note that
   `ThinkPad/control/analysis.py`'s existing `mean_power_w_idle` is **not** a substitute:
   `windows_from_events` appends every `idle_baseline` pair for a node to one list and
   `_mean_over` flattens them into a single mean, so it pools all ~200 per-run settles with
   the 120 s session-level baseline and averages a cold board with a hot one. Subtracting
   that pooled figure would over-correct early rows and under-correct late ones, with
   nothing in the output saying so. Use the per-row window.

The idle floor comes from the middle 12 s of each inter-run settle, marked
`idle_baseline_begin` / `idle_baseline_end`, so it is contemporaneous with the run rather
than measured once per session on a cold board.

## Independently cross-checking the window (`make bench-validate`)

Everything above rests on the `--progress` markers landing where upstream's source says
they do, on chrony holding the node's clock, and on nothing having reordered the stream.
A reader is entitled to doubt all three, so there is a second route to the same answer that
uses **no timestamps at all**.

Energy over a whole `llama-bench` process is affine in the repetition count:

```
E_process(R) = E_fixed + R * E_rep
```

`scripts/06_bench_local.sh validate` runs each cell at R = 1, 3, 5. Fitting a line through
those three points recovers `E_rep` as the slope and `E_fixed` (startup + load + warmup +
teardown) as the intercept, by the method of differences. `node/bench_validate.py` then
compares the slope against what the sub-windows independently claim one repetition costs:

```bash
make bench-validate
# then, on the control node once the energy join has run:
python3 node/bench_validate.py <run>-validate --energy energy.json
```

`energy.json` maps each run's JSON filename to **both** integrals — they are not
interchangeable:

```json
{"tinyllama-1.1b-q4km_pp128_t4_r3.json": {"process": 66.4, "reps": 36.2}}
```

`process` is `[t_start, t_end]` and is the only one that is affine in R, because `E_fixed`
is paid once per process. `reps` is `[t_meas_start, t_meas_end]`. Supplying only a bare
number makes the tool refuse the cross-check rather than perform it on the wrong quantity:
a whole-process energy divided by R still carries `E_fixed/R`, and comparing that against
the slope would manufacture agreement out of nothing.

Agreement inside the DAQ's ±5% is as strong a statement as this instrument supports.
Disagreement is **reported, never reconciled** — the tool exits non-zero and names the cell.

Two cautions. The validate sweep writes to a **separate** `<RUN_ID>-validate/` directory,
because its filenames carry a third underscore field (`_r5`) and `node/bench_parse.py`
recovers the model name with `rsplit("_", 2)`; point the parser at that directory and every
row joins to the wrong window. And three points against two parameters leaves one degree of
freedom, so the residual is a real but weak goodness-of-fit signal — read a large residual
as "investigate", not as a measurement.

## How the energy window is validated

The repetition START comes from the node's CLOCK_REALTIME stamp on llama-bench's `--progress`
line. The repetition LENGTH comes from `samples_ns`, which llama.cpp measured on its own
monotonic clock inside the process. Two independent clocks bound every sub-window, and
`min_gap_s` / `max_gap_s` record their disagreement. A negative gap means the repetitions
overlap, i.e. the alignment is wrong, and the run falls back to the whole-process window with
`energy_window=process` and a flag naming the gate that refused. The pipeline never produces
a number that looks the same but means something different.

Because the stamp is written a few microseconds AFTER llama.cpp's own `t_start`, every
repetition window is shifted LATE by the fprintf, the pipe write and the reader's wake-up —
order 0.1–10 ms against repetitions that are seconds long. The window LENGTH is exact; only
its phase is shifted. That residual is bounded by `min_gap_s`, per run, in the data.

## Phase markers

Both `scripts/lib.sh mark()` and the agent's `POST /mark` append to
`$DATA_DIR/logs/events.jsonl`. Markers are CLOCK_REALTIME and therefore only meaningful
while chrony is synced — which is why the healthcheck treats sync as blocking.

The control node can bracket a whole cluster run by POSTing the same marker to every node's
agent, so a distributed run has one consistent event boundary across four power channels.

## Link tiers

```bash
sudo scripts/05_net_tier.sh gbe            # real GbE
sudo scripts/05_net_tier.sh wifi           # real Wi-Fi
sudo scripts/05_net_tier.sh shape 100 1.0  # netem: 100 Mbit, 1 ms
sudo scripts/05_net_tier.sh clear
```

Two cautions that have bitten this kind of sweep before:

- netem shapes **egress only**. Apply the same rule on the peer node, or you have measured
  an asymmetric link and called it symmetric.
- Always `clear` afterwards. A forgotten qdisc makes the *next* experiment quietly wrong,
  and it is the hardest kind of error to find later. Both gates look for it — `make health`
  and the agent's `/health` — by comparing the live qdisc against the tier last declared
  through `scripts/05_net_tier.sh`, which writes `data/inventory/net_tier.json`. So a
  declared shaped point passes, and an undeclared qdisc blocks. Shape only through that
  script, or the declaration and the link disagree and the gate stops the run.

A suggested sweep, matching Appendix B's argument that the interesting threshold lies well
below GbE: 10, 25, 50, 100, 300, 1000 Mbit, each at 0.2 ms and 1.0 ms added delay.

## Suggested order of work on this node

| Step | Command | Produces |
|---|---|---|
| Inventory | `make info` | `inventory/device_info.json` |
| Smoke test | `make bench-quick` | proves the whole chain incl. the energy join |
| Full sweep | `make bench` | the Phase 2 rows for this node |
| Cluster join | the agent (`POST /rpc/start`) | participation in Phase 3 |

`make rpc` is a bring-up and debugging aid, **not** a measurement path. It runs
`scripts/04_rpc_server.sh` in your shell, outside the systemd unit — so none of
`MemoryMax`, `MemorySwapMax` or `Nice=-5` applies, and on this swapless 8 GB board those
are exactly the controls that turn an over-assignment into one failed service instead of
an OOM that takes the whole node down mid-sweep. Nothing in the recorded artefacts
distinguishes a row served this way from one served by the unit, so never use it for a run
that produces data.
| Straggler study | thread-limited + netem-shaped repeats | the QR8 rows |

Run the quick sweep and complete one full energy join **before** committing to the long
sweep. Discovering that the clocks were skewed after eight hours of benchmarking is the
expensive version of the same lesson.

## Recording the negative results

Two outcomes are likely on this node, and both are results worth keeping rather than
problems to hide:

- Adding the Pi 4 to the cluster may *reduce* end-to-end throughput, because a pipeline
  runs at the pace of its slowest stage.
- Distributing work may *increase* total joules per token, because the interconnect and the
  extra idle nodes cost energy.

The proposal predicts both. Measure them, report them with numbers, and let the placement
rule's response to them be the contribution.
