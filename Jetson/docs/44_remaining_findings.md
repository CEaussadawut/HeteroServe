# Remaining HIGH/MEDIUM findings — assessment and fixes

Seven agents assessed the findings left open in `docs/40_audit_findings.md`
against the CURRENT code (not against the audit's original text, which predates
a great deal of work). Each says whether the finding is `already-fixed`,
`partly-fixed` or `still-live`, and gives exact replacement text for what is left.

**The reconciler phase had not returned when the session was cut off**, so the
edits below have NOT been checked for conflicts with each other. Four of them
touch `bench/run_matrix.py` and two each touch `scripts/60_collect_node_facts.sh`
and `scripts/50_bench_device.sh` — apply with care and re-check every `old`
string against the file before trusting it.

Status summary:
- **already-fixed**: #12, #13, #23
- **partly-fixed**: #10, #21, #25
- **still-live**: #14, #17, #22, #26, #27, #29, #30

---

## Finding #10 — partly-fixed

Six coordinated changes, all in the copy, all verified.

1. `cooldown()` + `tail_sample()` in bench/run_matrix.py replace the two `time.sleep(run["cooldown_s"])` calls. `cooldown_s` becomes the FLOOR; after it the driver polls `temp_thermal_c` from the tail of the power CSV — the same column every row is scored on, so gate and audit column cannot disagree — until the board is within `cooldown_target_c` (3 °C) of the idle baseline, capped by `max_cooldown_s` (300 s). `tail_sample()` reads only the last 16 KB, because the gate polls every 2 s for 3 h and re-parsing a 100k-row CSV would put the driver's own load inside the cooldown it is measuring. With `--no-power` or a logger that wrote no temperature, the wait stays open loop and `cooldown_reached` is written blank rather than `1`, so a row never claims a gate that did not run.

2. `temp_baseline_c` is captured once from the 30 s idle window and copied onto every record (ok, failed and skipped alike, so no column is sometimes-absent).

3. `energy_between()` gains `temp_start_c`, `temp_end_c` (bringing the driver into step with telemetry/energy_window.py, which had them already) and `cpu_freq_start_mhz`/`cpu_freq_min_mhz`/`gpu_freq_start_mhz`/`gpu_freq_min_mhz`. The join then writes `temp_drift_c = temp_start_c - temp_baseline_c` — the BETWEEN-run quantity `temp_rise_c` structurally cannot express.

4. The driver repeats run 1, unchanged, as the final run (`drift_control=1`; `--no-drift-control` opts out). Same case, same configuration, hours apart, so the difference between those two rows is drift and nothing else. The summary prints the t/s delta between them. This is the audit's own "at minimum" suggestion; I deliberately did NOT take its other option, shuffling with a seed — randomising would fight `interleave_groups()`, which depends on a chosen order, and a sweep you cannot re-run in sequence is harder to defend than one whose drift is measured and published.

5. bench/matrix.json gains `cooldown_target_c: 3` / `max_cooldown_s: 300` in defaults, per-case overridable exactly like `cooldown_s`, plus a comment saying interleaving only protects a pair.

6. Docs corrected where they are actively wrong: docs/10's "`temp_rise_c` > 8 °C means contaminated" row is replaced by rows for the baseline, drift, rise (explicitly demoted), cooldown and clock columns; a new "Thermal drift, and why `temp_rise_c` cannot see it" section states the mechanism; docs/20's throttling entry points at `temp_drift_c`, `cpu_freq_min_mhz` and the drift-control pair. energy_window.py's on-screen flag is reworded from "check for throttling" to "this case is still heating", since one window cannot see drift.

One deliberate trade: the cooldown after the LAST run is not skipped, so a sweep can spend up to `max_cooldown_s` cooling at the end. Keeping every row's columns uniform is worth more than those minutes; say so if you disagree and it is a three-line guard.

**Justification (against current code):** Two of the finding's sub-claims are already closed, and the audit note is right to ask. `run_index` IS emitted — bench/run_matrix.py:916 `LEAD = ["case_id", "group", "run_index", "phase", ...]` — and `interleave_groups()` (bench/run_matrix.py:163) really does weave split pairs: "Reorder so the halves of a split case alternate: prefill, decode, prefill, decode ... Cases that share a \"group\" are the two phases of one configuration and get compared directly to each other, so they are the pairs that must not sit hours apart." The regression already proves it (`ok   split pairs interleave`).

But that buys only the PAIR. `interleave_groups()` opens with `if not group or len(idxs) < 2: continue`, and in bench/matrix.json exactly four cases carry a `group` (partial_offload, quantisation, and the two disabled largest_local halves). `cpu_prefill_sweep` and `thread_sweep` — the two the failure scenario names, both 512-token CPU prefill, first and last in the matrix — have NO `group` key at all, so interleaving never touches them. `run_index` makes the confound auditable; it does not measure or remove it.

Everything else in the finding is verbatim live:

1. The cooldown is still open loop. bench/run_matrix.py:740 and :775 are both still the bare `time.sleep(run["cooldown_s"])`, and expand() still ships `"cooldown_s": merged.get("cooldown_s", 20)` as the whole of the wait. Nothing reads a temperature between cases.

2. The row still cannot show drift. run_matrix.py's `energy_between()` writes only
       if temps:
           out["temp_max_c"] = max(temps)
           out["temp_rise_c"] = round(temps[-1] - temps[0], 2)
   — no `temp_start_c`, no baseline, no `cpu_freq_mhz`/`gpu_freq_mhz` at window start, even though the power CSV carries all of them (telemetry/tegrastats_logger.py FIELDS has `cpu_freq_mhz`, `gpu_freq_mhz`, `temp_thermal_c`). telemetry/energy_window.py:197 has reported `temp_start_c`/`temp_end_c` all along — the driver is the one not writing them down, so the two twinned tools disagree about what a window records.

3. The docs still name the broken rule as THE check: docs/10_measurement_protocol.md:197 `| `temp_rise_c` | > 8 °C means the case is thermally contaminated |`, and docs/10:19 lists `temp_start_c` as the recorded "Thermal starting point" — a column that does not exist on any joined row. docs/20_troubleshooting.md:77 sends the operator to `temp_rise_c` and `temp_max_c` and to raising `cooldown_s`.

**Reproduction:** Ran the shipped driver against the fixtures in a copy at /tmp/.../scratchpad/kit (never the live tree).

(a) Columns absent. `python3 bench/run_matrix.py --env-file $FIX/node.env --matrix bench/fixtures/matrix_split.json --tag drift`, then the joined CSV header: 73 columns, containing `temp_max_c`, `temp_rise_c`, `cooldown_s` — and NO `temp_start_c`, `temp_drift_c`, `temp_baseline_c`, `cpu_freq_start_mhz`, `gpu_freq_start_mhz`, `cooldown_actual_s`. A post-hoc throttling audit of the archived artefact is impossible, exactly as the finding says.

(b) The defect itself, measured. I gave bench/fixtures/fake_tegrastats.py thermal mass (HS_FAKE_TAU_S, first-order lag toward a phase-dependent target) and ran the same 4-run split matrix twice — once with the closed-loop gate, once with `cooldown_target_c: 100` so the gate is satisfied on its first poll and the wait degenerates to the old `time.sleep`:

  -- OPEN LOOP (what ships today)      baseline 40.0 C
     run phase     temp_drift_c  temp_rise_c
      1  prefill        0.77        21.63
      2  decode        15.78         5.62
      3  prefill       15.11        10.18
      4  decode        17.72         2.56   <-- most contaminated row in the sweep
      5  prefill       14.36        10.78

  -- CLOSED LOOP (after the fix)       baseline 40.0 C
      1  prefill        0.77        21.63   cooldown 7.0 s (floor 1 s), reached=1
      2  decode         2.80        16.73   cooldown 7.0 s, reached=1
      3  prefill        2.54        20.22   cooldown 7.0 s, reached=1
      4  decode         2.83        16.71   cooldown 7.0 s, reached=1
      5  prefill        2.54        20.22   cooldown 7.0 s, reached=1

Run 4 open-loop is the finding in two numbers: it sits 17.72 °C above where run 1 was measured — completely incomparable with it — and its `temp_rise_c` is 2.56, sailing through the docs' "> 8 °C means contaminated" rule. The rise is SMALLEST where the contamination is WORST. The closed-loop gate holds the same fake board to 2.83 °C, a 6.3x reduction, by waiting 7.0 s against a 1 s configured floor.

**Test:** Off-hardware, in the run_regression.sh idiom, and it runs green. Two new sections were added to a copy of the kit and the whole suite re-run:

  thermal: the cooldown is closed loop, and drift lands on the row
    ok   gate waited past the floor; drift and clocks recorded
    ok   open loop: board ratchets up and temp_rise_c does not say so

The first runs matrix_split under HS_FAKE_TAU_S=3 and asserts, on every ok row: cooldown_actual_s >= cooldown_s (the floor is a floor), at least one run waited more than a second past it (so the gate is genuinely closed loop and not a renamed sleep), temp_baseline_c / temp_start_c / cpu_freq_start_mhz / gpu_freq_start_mhz are populated, and max(temp_drift_c) <= 3.5 (the 3 C target plus one sample). The second re-runs the same fake board with cooldown_target_c: 100, which makes the gate pass on its first poll and reproduces the old open-loop behaviour exactly; it then asserts that the most-drifted row is > 10 C above baseline, that its temp_rise_c is < 8 (i.e. it passes the rule the docs used to rely on), and that open-loop drift exceeds closed-loop drift by more than 3x. Measured 17.72 C vs 2.83 C, a 6.3x gap.

Full suite: 51 passed, 0 failed before; 59 passed, 0 failed after (+8 checks, of which 4 belong to finding #14), and the run still ends with `ok   config/node.env unchanged`. check_finding3_invariants.py passes on every sweep in the suite including the drift-control row, and `ok   both tools agree exactly and match hand-computed truth` still holds, so nothing here perturbs the energy accounting.

What cannot be tested off hardware: the real Nano's time constant, and therefore whether 3 C / 300 s is a reachable target on this board in this room. The fixture proves the mechanism, not the tuning. On first real bring-up, watch for `cooldown_reached=0` and for total sweep time growing past the documented 1.5-3 h — both would say max_cooldown_s or cooldown_target_c needs raising, and both are now visible instead of silent.

**Conflicts:** bench/fixtures/run_regression.sh is being edited concurrently. Its mtime is 01:49, two hours after bench/run_matrix.py's, and it has gained two sections ('bring-up: a missing host compiler fails loudly' and 'bring-up: an interrupted swapfile does not wedge the box') that the findings #12/#13 sessions added after I copied the tree. My four edits to it are three exact-string substitutions plus one insertion anchored on `head_ "regression baseline"`; I verified all four anchors are still present and unique in the live file, but apply them to the current file, not to a stale copy, and expect the total to be higher than the 59 I measured. | The expected dry-run line changes from `phases    : combined 2, decode 11, prefill 19` (32 runs) to `prefill 20` (33). Any other finding's fix that adds or removes a matrix case, or that asserts '32 runs' anywhere, collides here. Note this does NOT reopen finding #3: the extra run is a byte-identical duplicate of an existing single-phase case, so 'zero mixed' still holds and check_finding3_invariants.py passes on every sweep in the suite. | Three existing fixture invocations now pass --no-drift-control because they assert exact row counts ('both tools agree', and the two missing-model sweeps). Any future finding that adds a row-count assertion to a multi-run fixture sweep must do the same or account for the repeat. | scripts/lib/common.sh gains thermal_zone_c() (finding #14's fix). The rpc-binary-name fact-check also edits that file, and both findings' fixes land in it — adjacent hunks, not overlapping ones. thermal_zone_c contains no pipelines, so it complies with the finding-#6 pipefail idiom documented at the top of that file rather than needing the `set +o pipefail` guard. | docs/10_measurement_protocol.md and docs/20_troubleshooting.md are edited by several findings; docs/20 was modified at 01:34 today by another session. My old-strings match the current content of both, but they are the most likely place for a stale-anchor failure if this sits unapplied. | No conflict with finding #11 (no external meter, deferred): the cooldown gate reads temperature only from the tegrastats power CSV. It introduces no second instrument, and no /sys dependency inside the driver. | The cooldown after the FINAL run is not skipped, so a sweep can spend up to max_cooldown_s cooling with nothing left to run. Deliberate — it keeps every row's cooldown columns uniform — but if someone prefers the sweep to end promptly, guard the call with `if i < len(runs)` and accept that the last row's cooldown columns go blank.

### edit — `bench/run_matrix.py`

Makes the gate's two thresholds per-case configurable exactly the way cooldown_s already is, so a case known to run hot can be given a longer ceiling without changing the driver.

**OLD**
```
                            "flash_attn": merged.get("flash_attn", 0),
                            "cooldown_s": merged.get("cooldown_s", 20),
                        })
```

**NEW**
```
                            "flash_attn": merged.get("flash_attn", 0),
                            # cooldown_s is now the FLOOR of a closed-loop wait,
                            # not the whole of it -- see cooldown(). The gate
                            # blocks until the board is back within
                            # cooldown_target_c of the pre-sweep idle baseline,
                            # or max_cooldown_s has passed.
                            "cooldown_s": merged.get("cooldown_s", 20),
                            "cooldown_target_c": merged.get("cooldown_target_c", 3.0),
                            "max_cooldown_s": merged.get("max_cooldown_s", 300),
                        })
```

### edit — `bench/run_matrix.py`

The closed-loop gate itself. Temperature comes from the power CSV rather than /sys so the gate and the row's temp_drift_c are the same instrument, and so the whole thing is exercisable against the fake tegrastats off hardware.

**OLD**
```
def main():
    ap = argparse.ArgumentParser(
```

**NEW**
```
def tail_sample(power_csv, cols=("temp_thermal_c",), window=16384):
    """The most recent power-CSV sample that carries every column in `cols`.

    Reads only the tail of the file. The cooldown gate calls this every couple
    of seconds for the length of a 1.5-3 h sweep, by the end of which the CSV is
    tens of thousands of rows, and re-parsing the whole of it each time would
    put the driver's own CPU load inside the cooldown it is trying to measure.
    """
    try:
        size = os.path.getsize(power_csv)
        with open(power_csv, "rb") as fh:
            header = fh.readline().decode("utf-8", "replace").strip()
            if size > window:
                fh.seek(-window, os.SEEK_END)
                fh.readline()           # discard the partial line we landed in
            blob = fh.read().decode("utf-8", "replace")
    except (OSError, IOError):
        return None
    names = next(csv.reader([header]), None)
    if not names:
        return None
    for row in reversed(list(csv.DictReader(blob.splitlines(), fieldnames=names))):
        got = {}
        for c in cols:
            try:
                got[c] = float(row.get(c) or "")
            except (TypeError, ValueError):
                got = None
                break
        if got:
            return got
    return None


def cooldown(power_csv, baseline_c, run):
    """Wait between cases until the board is back near its pre-sweep baseline.

    The fixed `time.sleep(cooldown_s)` this replaces was open loop: nothing read
    a temperature, so nothing could tell a cold board from a heat-soaked one. A
    fanless Nano devkit shedding ~10 W has a heatsink time constant of minutes,
    so 20 s recovers almost nothing, and across the ~33 runs of the shipped
    matrix the board ratchets monotonically upward until it clamps its own
    clocks. Position in the run order is then confounded with temperature, and
    the within-window `temp_rise_c` cannot see it -- that delta goes to ~0
    precisely when the board has settled at its hottest, so the worst case for
    comparability is the one that passes a "rise < 8 C" rule.

    `cooldown_s` stays as the FLOOR of the wait, because the board needs a
    moment to stop being loaded before its temperature means anything. After it,
    poll the power CSV -- the same `temp_thermal_c` every row is scored on, so
    the gate and the audit column cannot disagree -- until the board is within
    `cooldown_target_c` of the idle baseline captured before the first case.
    `max_cooldown_s` caps it, because a warm room can make the baseline
    permanently unreachable and a sweep that never ends is worse than one whose
    drift is recorded.

    Returns the columns that go on the row just measured. `cooldown_reached=0`
    is not a run to discard: it is the driver saying the NEXT run started hotter
    than the baseline, which is exactly the evidence a reader needs.
    """
    floor_s = run.get("cooldown_s") or 0
    target_c = run.get("cooldown_target_c")
    max_s = max(run.get("max_cooldown_s") or 0, floor_s)
    t0 = time.time()
    time.sleep(floor_s)
    out = {"cooldown_actual_s": round(time.time() - t0, 1), "cooldown_reached": 1}
    if not power_csv or baseline_c is None or not target_c:
        # No thermometer: --no-power, or a logger that never wrote a
        # temperature. The wait was open loop and the row must not imply
        # otherwise by claiming the gate was satisfied.
        out["cooldown_reached"] = ""
        return out
    last = None
    while True:
        got = tail_sample(power_csv, ("temp_thermal_c",))
        last = got["temp_thermal_c"] if got else last
        if last is not None and last <= baseline_c + target_c:
            break
        waited = time.time() - t0
        if waited >= max_s:
            out["cooldown_reached"] = 0
            print("    !! still %s C after %.0f s of cooldown, baseline %.1f C "
                  "+ %.1f C -- the next run starts hot and its temp_drift_c "
                  "will say so" % (last if last is not None else "?", waited,
                                   baseline_c, target_c))
            break
        time.sleep(min(2.0, max_s - waited))
    if last is not None:
        out["cooldown_end_temp_c"] = last
    out["cooldown_actual_s"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser(
```

### edit — `bench/run_matrix.py`

Opt-out for the drift control, needed by the fixture sweeps that assert an exact row count.

**OLD**
```
    ap.add_argument("--no-precache", action="store_true",
```

**NEW**
```
    ap.add_argument("--no-drift-control", dest="drift_control", action="store_false",
                    help="do not repeat the first run as the final run. That "
                         "repeat is the sweep's only direct measure of thermal "
                         "drift: identical case, identical configuration, two "
                         "hours apart, so the difference between the two rows is "
                         "the drift and nothing else.")
    ap.add_argument("--no-precache", action="store_true",
```

### edit — `bench/run_matrix.py`

The explicit drift control. It gets its own run_index and therefore its own marker, so window uniqueness and the finding-#3 invariants are untouched.

**OLD**
```
    runs = interleave_groups(runs)

```

**NEW**
```
    runs = interleave_groups(runs)
    for r in runs:
        r["drift_control"] = 0
    if args.drift_control and len(runs) > 1:
        # Repeat the first run, unchanged, as the last run. Interleaving already
        # keeps the two halves of a split PAIR adjacent, but nothing makes the
        # first case of the matrix comparable to the last: cpu_prefill_sweep runs
        # on a ~40 C board at full clock and thread_sweep -- the same 512-token
        # CPU prefill workload -- runs two hours later at ~85 C with the SoC
        # clamping, and "the 4th core costs energy" is then a statement about
        # elapsed time, not about threads. Two identical runs at the two ends of
        # the sweep put a number on that, and it costs one extra case.
        runs.append(dict(runs[0], drift_control=1))

```

### edit — `bench/run_matrix.py`

The baseline must exist as a name on every path, including --no-power, so the cooldown call and the skipped-record path do not raise.

**OLD**
```
    logger = None
    if not args.no_power and shutil.which("tegrastats"):
```

**NEW**
```
    logger = None
    baseline_c = None
    if not args.no_power and shutil.which("tegrastats"):
```

### edit — `bench/run_matrix.py`

Reuses the idle window the driver already collects for the power floor; the board is never colder than it is here, so it is the only defensible reference for drift.

**OLD**
```
        print("collecting 30 s idle baseline...")
        time.sleep(30)
    else:
```

**NEW**
```
        print("collecting 30 s idle baseline...")
        time.sleep(30)
        # The idle temperature the cooldown gate aims at, and the reference every
        # row's temp_drift_c is measured against. Captured here, once, while the
        # board is as cold as it will be all sweep.
        got = tail_sample(power_csv, ("temp_thermal_c",))
        if got:
            baseline_c = got["temp_thermal_c"]
            print("idle baseline: %.1f C" % baseline_c)
        else:
            print("!! no temp_thermal_c in the power CSV -- cooldown stays open "
                  "loop and no row can report thermal drift")
    else:
```

### edit — `bench/run_matrix.py`

A skipped run must carry the baseline too, or temp_baseline_c becomes a sometimes-absent column and the CSV stops being uniformly machine-readable.

**OLD**
```
                           "model_path": mpath or "", "tag": args.tag}
```

**NEW**
```
                           "model_path": mpath or "", "tag": args.tag,
                           "temp_baseline_c": baseline_c}
```

### edit — `bench/run_matrix.py`

A failed case still heated the board, so the run after it still needs the gate; and the failed row still needs the cooldown columns.

**OLD**
```
                failed["per_token_status"] = per_token_status(failed)
                records.append(failed)
                time.sleep(run["cooldown_s"])
                continue
```

**NEW**
```
                failed["per_token_status"] = per_token_status(failed)
                failed["temp_baseline_c"] = baseline_c
                records.append(failed)
                failed.update(cooldown(power_csv if logger else None, baseline_c, run))
                continue
```

### edit — `bench/run_matrix.py`

The main replacement of the open-loop sleep. Attributing the wait to the run that caused the heat is what makes cooldown_reached=0 readable as 'from here on the board stopped recovering'.

**OLD**
```
                      "energy refused for this case. Split it in matrix.json."
                      % emitted[0].get("working_rows"))
            time.sleep(run["cooldown_s"])
```

**NEW**
```
                      "energy refused for this case. Split it in matrix.json."
                      % emitted[0].get("working_rows"))
            # The cooldown belongs to the run that just finished -- it is what
            # that run handed to the next one -- so it is recorded on its rows.
            cool = cooldown(power_csv if logger else None, baseline_c, run)
            for rec in emitted:
                rec["temp_baseline_c"] = baseline_c
                rec.update(cool)
```

### edit — `bench/run_matrix.py`

The actual contamination check the finding asks for, computed where both operands are finally on the record.

**OLD**
```
            e = energy_between(power_csv, rec["t_start_unix"], rec["t_end_unix"])
            rec.update(e)

```

**NEW**
```
            e = energy_between(power_csv, rec["t_start_unix"], rec["t_end_unix"])
            rec.update(e)

            # Drift is a BETWEEN-run quantity: how much hotter this window
            # started than the idle board did. That is the number that tells a
            # reader whether an early case and a late case measured the same
            # hardware, and it is the one temp_rise_c cannot express.
            if rec.get("temp_start_c") is not None and rec.get("temp_baseline_c") is not None:
                rec["temp_drift_c"] = round(rec["temp_start_c"] - rec["temp_baseline_c"], 2)

```

### edit — `bench/run_matrix.py`

Brings the driver's window stats into step with energy_window.py's, and carries the clocks onto the archived CSV. The minimum, not just the start, is what actually evidences a clamp inside the window.

**OLD**
```
    if temps:
        out["temp_max_c"] = max(temps)
        out["temp_rise_c"] = round(temps[-1] - temps[0], 2)
    if swaps:
```

**NEW**
```
    if temps:
        # temp_start_c is the column that detects DRIFT; temp_rise_c is not, and
        # never was. A rise is a delta WITHIN one window, so it collapses to ~0
        # exactly when the board has settled at its hottest -- the least
        # comparable run in the sweep is the one that passes a "rise < 8 C"
        # rule. What a reader needs is where each window STARTED, against the
        # idle baseline captured before the first case; that difference is
        # temp_drift_c, added at join time. Twinned with window_stats() in
        # telemetry/energy_window.py, which has reported temp_start_c and
        # temp_end_c all along -- the driver was the one not writing them down.
        out["temp_start_c"] = temps[0]
        out["temp_end_c"] = temps[-1]
        out["temp_max_c"] = max(temps)
        out["temp_rise_c"] = round(temps[-1] - temps[0], 2)
    # The clocks at the window's first sample, and the lowest they reached
    # inside it. A row that is slow because the SoC clamped looks exactly like a
    # row that is slow because of the axis being swept, and after the sweep the
    # power CSV is the only place the difference survives -- so carry it onto
    # the joined CSV, which is the archived artefact.
    for src, start_key, min_key in (("cpu_freq_mhz", "cpu_freq_start_mhz", "cpu_freq_min_mhz"),
                                    ("gpu_freq_mhz", "gpu_freq_start_mhz", "gpu_freq_min_mhz")):
        fr = [f for f in (num(r, src) for r in sel) if f is not None]
        if fr:
            out[start_key] = fr[0]
            out[min_key] = min(fr)
    if swaps:
```

### edit — `bench/run_matrix.py`

Pins the thermal-audit block into a fixed position in the header, so it does not drift around by first-seen order when one case fails.

**OLD**
```
            "joules_per_request", "wh_per_krequest", "active_joules_per_request"]
```

**NEW**
```
            "joules_per_request", "wh_per_krequest", "active_joules_per_request",
            "drift_control", "temp_baseline_c", "temp_start_c", "temp_drift_c",
            "temp_rise_c", "temp_max_c",
            "cpu_freq_start_mhz", "cpu_freq_min_mhz",
            "gpu_freq_start_mhz", "gpu_freq_min_mhz",
            "cooldown_s", "cooldown_actual_s", "cooldown_reached",
            "cooldown_end_temp_c"]
```

### edit — `bench/run_matrix.py`

Puts the drift verdict on stdout at the end of the sweep, so the operator learns the table is contaminated before they start writing it up rather than after.

**OLD**
```
    if by_status.get(PT_MULTI_PHASE):
```

**NEW**
```
    drifts = [r["temp_drift_c"] for r in records if r.get("temp_drift_c") is not None]
    if drifts:
        print("thermal drift: %+.1f C max above the idle baseline at window start"
              % max(drifts))
    dc = [r for r in records if str(r.get("drift_control")) == "1" and r.get("avg_ts")]
    if dc:
        # The same case, measured at both ends of the sweep. If these two rows
        # disagree, every comparison the sweep makes between an early case and a
        # late one is suspect by at least that much, whatever temp_rise_c said.
        first = next((r for r in records
                      if r.get("case_id") == dc[0].get("case_id")
                      and r.get("test") == dc[0].get("test")
                      and str(r.get("drift_control")) == "0" and r.get("avg_ts")), None)
        if first:
            print("drift control: %s %.2f -> %.2f t/s (%+.1f%%) from run 1 to run %s"
                  % (dc[0].get("test"), first["avg_ts"], dc[0]["avg_ts"],
                     100.0 * (dc[0]["avg_ts"] - first["avg_ts"]) / first["avg_ts"],
                     dc[0].get("run_index")))
    if by_status.get(PT_MULTI_PHASE):
```

### edit — `bench/matrix.json`

The matrix comment currently implies interleaving solves the thermal confound; it solves it only for grouped pairs, and the file is where a reader looks first.

**OLD**
```
    "'group' pairs the two halves of a split case. Runs inside a group are",
    "INTERLEAVED (prefill, decode, prefill, decode ...) so the two phases of one",
    "configuration are measured at nearly the same board temperature -- on a",
    "fanless Nano, case order is confounded with thermal drift.",

```

**NEW**
```
    "'group' pairs the two halves of a split case. Runs inside a group are",
    "INTERLEAVED (prefill, decode, prefill, decode ...) so the two phases of one",
    "configuration are measured at nearly the same board temperature -- on a",
    "fanless Nano, case order is confounded with thermal drift.",
    "",
    "Interleaving only protects a PAIR. Across the whole matrix the board still",
    "ratchets upward, so the wait between cases is closed loop: cooldown_s is",
    "only the floor, and the driver then blocks until the board is back within",
    "cooldown_target_c of the idle baseline it measured before the first case,",
    "giving up after max_cooldown_s. The actual wait, the baseline, and each",
    "window's starting temperature and clocks are columns on every row, and the",
    "driver repeats the first run as the final run as an explicit drift control.",

```

### edit — `bench/matrix.json`

3 C is tight enough that the sweep's rows stay comparable and loose enough to be reachable on a fanless board; 300 s stops a warm room from stalling the sweep forever.

**OLD**
```
    "flash_attn": 0,
    "cooldown_s": 20
  },
```

**NEW**
```
    "flash_attn": 0,
    "cooldown_s": 20,
    "cooldown_target_c": 3,
    "max_cooldown_s": 300
  },
```

### edit — `bench/fixtures/fake_tegrastats.py`

Without thermal mass in the fake, a closed-loop gate and an open-loop sleep are indistinguishable off hardware and the fix would be untestable. Gated behind HS_FAKE_TAU_S so the 51 existing checks see the exact constant temperature they always did.

**OLD**
```
import os, sys, time
interval = 1000
if "--interval" in sys.argv:
    interval = int(sys.argv[sys.argv.index("--interval") + 1])
phase_file = os.environ["HS_FAKE_PHASE"]
MW = {"idle": (1900, 0, 238), "load": (5200, 200, 3000),
      "prefill": (7400, 3100, 2600), "decode": (4800, 1500, 1900)}
while True:
    try:
        ph = open(phase_file, encoding="utf-8").read().strip() or "idle"
    except OSError:
        ph = "idle"
    p_in, p_gpu, p_cpu = MW.get(ph, MW["idle"])
    sys.stdout.write(
        "RAM 1969/3956MB (lfb 22x4MB) SWAP 120/1978MB (cached 0MB) IRAM 0/252kB(lfb 252kB) "
        "CPU [42%%@1479,38%%@1479,11%%@1479,9%%@1479] EMC_FREQ 21%%@1600 GR3D_FREQ 63%%@921 "
        "APE 25 PLL@46.5C CPU@46.2C PMIC@100C GPU@45.5C AO@52C thermal@46.2C "
        "POM_5V_IN %d/%d POM_5V_GPU %d/%d POM_5V_CPU %d/%d\n"
        % (p_in, p_in, p_gpu, p_gpu, p_cpu, p_cpu))
    sys.stdout.flush()
    time.sleep(interval / 1000.0)
```

**NEW**
```
import math, os, sys, time
interval = 1000
if "--interval" in sys.argv:
    interval = int(sys.argv[sys.argv.index("--interval") + 1])
phase_file = os.environ["HS_FAKE_PHASE"]
MW = {"idle": (1900, 0, 238), "load": (5200, 200, 3000),
      "prefill": (7400, 3100, 2600), "decode": (4800, 1500, 1900)}

# With HS_FAKE_TAU_S set, the board has thermal MASS: the reported temperature
# lags toward a phase-dependent target with that time constant, so a case leaves
# the board hotter than it found it and the next one cannot start until it has
# shed the heat. That is what makes the closed-loop cooldown gate testable off
# hardware -- an open-loop sleep passes a constant-temperature fake trivially.
# Unset (the default) the temperature is the constant 46.2 C it has always been,
# so every test written before this knob existed sees exactly what it saw.
TAU_S = float(os.environ.get("HS_FAKE_TAU_S", "0"))
TEMP_TARGET = {"idle": 40.0, "load": 52.0, "prefill": 78.0, "decode": 66.0}
temp_c = 40.0
last = time.time()
while True:
    try:
        ph = open(phase_file, encoding="utf-8").read().strip() or "idle"
    except OSError:
        ph = "idle"
    p_in, p_gpu, p_cpu = MW.get(ph, MW["idle"])
    now = time.time()
    if TAU_S > 0:
        tgt = TEMP_TARGET.get(ph, TEMP_TARGET["idle"])
        temp_c += (tgt - temp_c) * (1.0 - math.exp(-(now - last) / TAU_S))
    else:
        temp_c = 46.2
    last = now
    sys.stdout.write(
        "RAM 1969/3956MB (lfb 22x4MB) SWAP 120/1978MB (cached 0MB) IRAM 0/252kB(lfb 252kB) "
        "CPU [42%%@1479,38%%@1479,11%%@1479,9%%@1479] EMC_FREQ 21%%@1600 GR3D_FREQ 63%%@921 "
        "APE 25 PLL@%.1fC CPU@%.1fC PMIC@100C GPU@%.1fC AO@52C thermal@%.2fC "
        "POM_5V_IN %d/%d POM_5V_GPU %d/%d POM_5V_CPU %d/%d\n"
        % (temp_c + 0.3, temp_c, temp_c - 0.7, temp_c,
           p_in, p_in, p_gpu, p_gpu, p_cpu, p_cpu))
    sys.stdout.flush()
    time.sleep(interval / 1000.0)
```

### edit — `bench/fixtures/run_regression.sh`

The shipped matrix is now 33 runs: 32 cases plus the drift-control repeat. The added checks pin BOTH counts and prove the repeat is byte-identical to run 1.

**OLD**
```
grep -q "phases    : combined 2, decode 11, prefill 19" "$FIX/dry.txt" \
  && ok "32 runs, zero mixed" || { bad "run/phase counts"; sed -n '1,6p' "$FIX/dry.txt"; }
```

**NEW**
```
grep -q "phases    : combined 2, decode 11, prefill 20" "$FIX/dry.txt" \
  && ok "33 runs (32 + the drift control), zero mixed" \
  || { bad "run/phase counts"; sed -n '1,6p' "$FIX/dry.txt"; }
python3 bench/run_matrix.py --env-file "$FIX/node.env" --no-drift-control --dry-run \
  2>&1 | grep -q "phases    : combined 2, decode 11, prefill 19" \
  && ok "--no-drift-control gives the bare 32" || bad "--no-drift-control"
python3 - "$FIX/dry.txt" <<'PY' && ok "the drift control repeats run 1 exactly" || bad "drift control run"
import re, sys
rows = [l.split() for l in open(sys.argv[1]) if re.match(r"\s+\d+\s+\w", l)]
assert rows[0][1:] == rows[-1][1:], (rows[0], rows[-1])
assert rows[-1][0] == "33", rows[-1][0]
PY
```

### edit — `bench/fixtures/run_regression.sh`

This check asserts n == 4 ok rows; the repeat would make it 5 and the failure would look like an energy regression rather than the extra control run it is.

**OLD**
```
python3 bench/run_matrix.py --env-file "$FIX/node.env" \
  --matrix bench/fixtures/matrix_split.json --tag agree > /dev/null 2>&1
```

**NEW**
```
# --no-drift-control: this test counts rows, and the drift control deliberately
# adds one more by repeating the first run at the end of the sweep.
python3 bench/run_matrix.py --env-file "$FIX/node.env" --no-drift-control \
  --matrix bench/fixtures/matrix_split.json --tag agree > /dev/null 2>&1
```

### edit — `bench/fixtures/run_regression.sh`

Same reason: this block asserts exactly two rows, one per case.

**OLD**
```
python3 bench/run_matrix.py --env-file "$FIX/node.env" --matrix "$FIX/missing.json" \
  --tag nomodel --no-power > "$FIX/nm.txt" 2>&1
```

**NEW**
```
python3 bench/run_matrix.py --env-file "$FIX/node.env" --matrix "$FIX/missing.json" \
  --no-drift-control --tag nomodel --no-power > "$FIX/nm.txt" 2>&1
```

### edit — `bench/fixtures/run_regression.sh`

Same reason; this one also feeds the 'no column is sometimes-blank' check, which counts markers.

**OLD**
```
python3 bench/run_matrix.py --env-file "$FIX/node.env" --matrix "$FIX/missing.json" \
  --tag nomodelpwr > "$FIX/nmp.txt" 2>&1
```

**NEW**
```
python3 bench/run_matrix.py --env-file "$FIX/node.env" --matrix "$FIX/missing.json" \
  --no-drift-control --tag nomodelpwr > "$FIX/nmp.txt" 2>&1
```

### edit — `bench/fixtures/run_regression.sh`

Pure insertion before the existing last section, so it cannot collide with the bring-up sections other sessions added at 01:49. Proves the gate waits, that the audit columns are populated, and — the real point — that the open-loop sweep's worst row passes the old > 8 C rule while sitting 17.7 C above baseline.

**OLD**
```
head_ "regression baseline"
```

**NEW**
```
head_ "thermal: the cooldown is closed loop, and drift lands on the row"
rm -f "$FIX"/results/*
# HS_FAKE_TAU_S gives the fake board thermal MASS, so a case leaves it hotter
# than it found it. Against the constant-temperature fake an open-loop sleep and
# a closed-loop gate are indistinguishable, which is the whole difficulty of
# audit finding #10: the contamination the kit checked for -- temp_rise_c,
# a delta WITHIN one window -- goes to ~0 exactly when the board is at its
# hottest steady state.
HS_FAKE_TAU_S=3 python3 bench/run_matrix.py --env-file "$FIX/node.env" \
  --matrix bench/fixtures/matrix_split.json --tag thermal > "$FIX/th.txt" 2>&1
python3 - "$FIX" <<'EOPY' && ok "gate waited past the floor; drift and clocks recorded" || bad "closed-loop cooldown"
import csv, glob, sys
rows = list(csv.DictReader(open(glob.glob(sys.argv[1] + "/results/*thermal_bench_energy.csv")[0])))
ran = [r for r in rows if r["status"] == "ok"]
assert ran, "no completed runs"
for r in ran:
    # cooldown_s is now only the FLOOR of the wait
    assert float(r["cooldown_actual_s"]) >= float(r["cooldown_s"]), (r["marker"], r["cooldown_actual_s"])
    assert r["temp_baseline_c"], r["marker"]
    assert r["temp_start_c"], r["marker"]
    assert r["cpu_freq_start_mhz"] and r["gpu_freq_start_mhz"], r["marker"]
# at least one case must actually have had to wait -- otherwise the gate is
# still open loop and this test proves nothing
assert any(float(r["cooldown_actual_s"]) > float(r["cooldown_s"]) + 1.0 for r in ran), \
    [(r["marker"], r["cooldown_actual_s"], r["cooldown_s"]) for r in ran]
# and the gate must have brought the board back near the baseline before the
# next window opened: that is the drift it exists to remove
base = float(ran[0]["temp_baseline_c"])
drifts = [float(r["temp_drift_c"]) for r in ran]
assert max(drifts) <= 3.0 + 0.5, (base, drifts)
EOPY
# The contrast that makes the point. cooldown_target_c: 100 can never be
# violated, so the gate is satisfied on its first poll and the wait is exactly
# the old open-loop sleep. Same fake board, same workload.
cat > "$FIX/openloop.json" <<'JSON'
{"defaults": {"repetitions": 3, "threads": 4, "cooldown_s": 1,
              "cooldown_target_c": 100, "max_cooldown_s": 60},
 "cases": [{"id": "split_prefill", "group": "split",
            "model": "qwen2.5-1.5b-instruct-q4_k_m",
            "n_gpu_layers": [0, 99], "n_prompt": [512], "n_gen": [0]},
           {"id": "split_decode", "group": "split",
            "model": "qwen2.5-1.5b-instruct-q4_k_m",
            "n_gpu_layers": [0, 99], "n_prompt": [0], "n_gen": [128]}]}
JSON
HS_FAKE_TAU_S=3 python3 bench/run_matrix.py --env-file "$FIX/node.env" \
  --matrix "$FIX/openloop.json" --tag openloop > /dev/null 2>&1
python3 - "$FIX" <<'EOPY' && ok "open loop: board ratchets up and temp_rise_c does not say so" || bad "open-loop contrast"
import csv, glob, sys
def ran(tag):
    f = glob.glob(sys.argv[1] + "/results/*%s_bench_energy.csv" % tag)[0]
    return [r for r in csv.DictReader(open(f)) if r["status"] == "ok"]
op, cl = ran("openloop"), ran("thermal")
hot = max(op, key=lambda r: float(r["temp_drift_c"]))
# Open loop the board ends the sweep far above its idle baseline...
assert float(hot["temp_drift_c"]) > 10, [r["temp_drift_c"] for r in op]
# ...and the MOST contaminated run in the sweep is the one with the SMALLEST
# within-window rise, because it is the closest to a hot steady state. It sails
# through the docs' old "temp_rise_c > 8 C means contaminated" rule while
# sitting more than 10 C above where the first run was measured. That is audit
# finding #10 in two numbers.
assert abs(float(hot["temp_rise_c"])) < 8, (hot["temp_drift_c"], hot["temp_rise_c"])
# The closed-loop gate holds the same fake board to its target instead.
assert max(float(r["temp_drift_c"]) for r in op) \
     > 3 * max(float(r["temp_drift_c"]) for r in cl), \
    ([r["temp_drift_c"] for r in op], [r["temp_drift_c"] for r in cl])
EOPY

head_ "regression baseline"
```

### edit — `docs/10_measurement_protocol.md`

This row claimed temp_start_c was recorded when no joined row carried it and the only mechanism was 'wait'. Both halves are now true.

**OLD**
```
| Thermal starting point | wait | `temp_start_c` |
```

**NEW**
```
| Thermal starting point | measured idle baseline, then a closed-loop cooldown between cases | `temp_baseline_c`, `temp_start_c`, `temp_drift_c` |
```

### edit — `docs/10_measurement_protocol.md`

'cools down between runs so heat does not carry over' was the claim the fixed 20 s sleep could not support; now it says what the gate actually does.

**OLD**
```
1. collects a 30 s idle baseline,
2. runs every point of every enabled case — 32 runs with the shipped matrix —
   each in its own power window covering **exactly one phase**,
3. cools down between runs so heat does not carry over,
```

**NEW**
```
1. collects a 30 s idle baseline, and takes the board's idle temperature from
   it as `temp_baseline_c`,
2. runs every point of every enabled case — 32 runs with the shipped matrix,
   plus the drift control below — each in its own power window covering
   **exactly one phase**,
3. waits between runs until the board is back within `cooldown_target_c` (3 °C)
   of that baseline, or `max_cooldown_s` (300 s) has passed, and records what it
   actually waited,
```

### edit — `docs/10_measurement_protocol.md`

The protocol currently presents interleaving as the answer to the thermal confound. This states what interleaving does and does not buy, and records why randomisation was rejected so the choice is not silently re-litigated.

**OLD**
```
### Reading the output
```

**NEW**
```
### Thermal drift, and why `temp_rise_c` cannot see it

Interleaving only protects a *pair*. Across the whole matrix the board still
ratchets upward: a Nano dev kit with no fan shedding ~10 W has a heatsink time
constant of minutes, so the first case runs at ~40 °C on full clocks and a case
two hours later runs heat-soaked with the SoC clamping. `cpu_prefill_sweep` and
`thread_sweep` are the same 512-token CPU prefill workload at opposite ends of
the sweep; if nothing corrects for this, "the 4th core costs energy" is a
statement about elapsed time.

`temp_rise_c` does **not** catch this, and a sweep that passes a "rise < 8 °C"
rule on every row can still be entirely drift. A rise is a delta *within* one
window, and it goes to ~0 exactly when the board has reached a hot steady state
— so the worst row in the sweep is the quietest. Three things replace it:

- the cooldown between cases is **closed loop**. `cooldown_s` is only a floor;
  after it the driver polls `temp_thermal_c` from the power CSV — the same
  column the rows are scored on — and blocks until the board is within
  `cooldown_target_c` of `temp_baseline_c`, up to `max_cooldown_s`. What it
  actually waited is on the row as `cooldown_actual_s`;
- every row carries `temp_start_c` and `temp_drift_c` (start minus baseline),
  plus the CPU and GPU clocks at the window's first sample and their minimum
  inside it, so a throttling audit of the archived CSV is possible after the
  fact rather than only while the board is in front of you;
- the driver **repeats the first run as the final run** (`drift_control=1`).
  Identical case, identical configuration, hours apart, so the difference
  between those two rows is drift and nothing else. If they disagree by more
  than the repetition spread, every early-vs-late comparison in the sweep is
  suspect by at least that much. `--no-drift-control` turns it off.

Randomising the run order would also decorrelate drift from case order, but it
is deliberately *not* done here: the interleaving above depends on a chosen
order, and a sweep you cannot re-run in the same sequence is harder to defend
than one whose drift is measured and published.

### Reading the output
```

### edit — `docs/10_measurement_protocol.md`

This single line is the rule the finding says is wrong. It is replaced by the columns that do work, with temp_rise_c explicitly demoted rather than deleted.

**OLD**
```
| `temp_rise_c` | > 8 °C means the case is thermally contaminated |
```

**NEW**
```
| `temp_baseline_c` | the board's idle temperature, measured once before the first case. The reference everything below is relative to |
| `temp_start_c`, `temp_drift_c` | how hot the board was when this window opened, and how far above the idle baseline that is. **`temp_drift_c` is the contamination check**: above ~5 °C the run is not comparable with an early one, whatever `temp_rise_c` says |
| `temp_rise_c`, `temp_max_c` | how much the board heated *within* this one window. Useful for spotting a single runaway case, useless for drift: a rise falls to ~0 precisely when the board has settled at its hottest, so the least comparable run in a sweep has the smallest rise. It is not a contamination check on its own |
| `cooldown_s`, `cooldown_actual_s`, `cooldown_reached`, `cooldown_end_temp_c` | the configured floor, what the gate actually waited, whether it got back to baseline before giving up, and the temperature it gave up at. `cooldown_reached=0` marks the point in the sweep where the board stopped recovering |
| `cpu_freq_start_mhz`, `cpu_freq_min_mhz`, `gpu_freq_start_mhz`, `gpu_freq_min_mhz` | the clocks at the window's first sample and the lowest they reached inside it. A row that is slow because the SoC clamped looks exactly like a row that is slow because of the axis being swept; these are how you tell afterwards |
| `drift_control` | `1` on the repeat of the first run at the end of the sweep — see “Thermal drift” above |
```

### edit — `docs/20_troubleshooting.md`

The one place an operator looks when they see the symptom sent them to the two columns that cannot show it.

**OLD**
```
Thermal throttling. Check `temp_rise_c` and `temp_max_c` in the sweep output.
Fit a fan, raise `cooldown_s` in `bench/matrix.json`, and re-run. A table whose
```

**NEW**
```
Thermal throttling. Check `temp_drift_c` and `cpu_freq_min_mhz` on the late rows,
and compare the two `drift_control` rows — the same case measured at both ends of
the sweep. `temp_rise_c` will look fine and mean nothing: it is a delta within one
window, so it falls to ~0 exactly when the board has settled at its hottest.

If `cooldown_reached` is `0` from some run onward, the board stopped recovering
there. Fit a fan, raise `cooldown_target_c` or `max_cooldown_s` in
`bench/matrix.json`, run the sweep in a cooler room, and re-run. A table whose
```

### edit — `telemetry/energy_window.py`

The tool prints one window at a time and genuinely cannot see drift; saying 'check for throttling' told the reader the absence of the flag meant the run was clean. Wording only — no number changes, so the driver/tool bit-identity check is untouched.

**OLD**
```
            if "temp_rise_c" in r:
                flag = "  <-- check for throttling" if r["temp_rise_c"] > 8 else ""
```

**NEW**
```
            if "temp_rise_c" in r:
                # A rise is a delta WITHIN this window, so it is only evidence
                # about THIS case running away -- it goes to ~0 once the board
                # has settled at its hottest, which is when a sweep's rows stop
                # being comparable with each other. Drift between runs is
                # temp_drift_c on the joined CSV, against the sweep's idle
                # baseline; this tool sees one window and cannot compute it.
                flag = "  <-- this case is still heating" if r["temp_rise_c"] > 8 else ""
```

---

## Finding #12 — already-fixed

No change needed; already fixed and independently verified. Two residuals worth knowing about, neither of which I would change now:

1. `swapfile_ready` depends on `swaplabel`. It is in util-linux and ships on bionic/L4T R32.7, so this is fine on the real box -- but if it were ever absent, `sudo swaplabel` exits 127, the predicate returns 1, and every single `make prep` would silently re-allocate 8 GB on the SD card. Slow, not fatal, and it announces itself with the "rebuilding it" warning, so it is self-diagnosing.

2. The audit proposed allocating to a `.part` sidecar and renaming. The applied fix instead `rm -f`s the stale file and reallocates in place. Functionally equivalent for re-runs, with one narrow difference: if `/etc/fstab` already carries the swap line from an earlier successful run and the operator then raises `SWAP_GB` and Ctrl-Cs the reallocation, the next boot hits a partial `/var/swapfile` in fstab. On Ubuntu 18.04 a failing swap unit does not block boot to `multi-user.target`, and the next `make prep` repairs it, so this is a cosmetic boot error rather than a second wedge.

**Justification (against current code):** Closed in `scripts/00_system_prep.sh` (mtime 2026-09-14 01:47; the swap block is byte-identical to the version I tested at 02:09). The wedge state and its recovery are both now explicit. The old guard was `if [ ! -f "$SWAPFILE" ]` wrapping BOTH the allocation and the `mkswap`, with `sudo swapon "$SWAPFILE"` outside it. The current code replaces the file-existence test with a three-part readiness predicate:

    swapfile_ready() {
      [ -f "$SWAPFILE" ] || return 1
      local have
      have="$( set +o pipefail; stat -c %s "$SWAPFILE" 2>/dev/null || echo 0 )"
      [ "$have" = "$SWAP_BYTES" ] || return 1
      # `swaplabel` reads the header mkswap writes; no header means the allocation
      # finished but mkswap did not, or neither did.
      sudo swaplabel "$SWAPFILE" >/dev/null 2>&1
    }

and drives the rebuild off it:

    if ! swapon --show=NAME --noheadings | grep -qx "$SWAPFILE"; then
      if ! swapfile_ready; then
        if [ -f "$SWAPFILE" ]; then
          warn "${SWAPFILE} exists but is not a usable ${SWAP_GB}G swap area"
          warn "(partial allocation, or mkswap never ran) -- rebuilding it"
          sudo rm -f "$SWAPFILE"
        fi
        info "allocating ${SWAP_GB}G at ${SWAPFILE} (slow on SD; be patient)"
        sudo fallocate -l "${SWAP_GB}G" "$SWAPFILE" 2>/dev/null \
          || sudo dd if=/dev/zero of="$SWAPFILE" bs=1M count=$((SWAP_GB*1024)) status=progress
        sudo chmod 600 "$SWAPFILE"
        sudo mkswap "$SWAPFILE"
      fi
      if ! sudo swapon "$SWAPFILE"; then
        # Do not die: everything after this step is still worth running, and the
        # build just needs -j1 instead of -j2 without swap.
        warn "swapon failed for ${SWAPFILE}; continuing WITHOUT extra swap"
        warn "the CUDA build may OOM -- use BUILD_JOBS=1"
      else

The exact wedging state named by the audit -- file present, no `mkswap` header -- is now caught by the `swaplabel` leg; the *size* leg additionally catches a `dd` interrupted partway, which the audit's own suggested `.part` fix would have caught only on the FIRST interruption. The second change matters as much as the first: `swapon` failing is no longer fatal, so `nvpmodel`, `jetson_clocks`, the fan PWM and chrony still run. Those are the steps the measurement protocol actually depends on, and losing them silently was the real damage.

**Reproduction:** Reproduced the ORIGINAL wedge and the fix side by side, off-hardware. `require_jetson` (hard-coded `/etc/nv_tegra_release`) and `apt-get` make the whole script unrunnable on a laptop, so I lifted the Swap step out verbatim with `sed -n '/^step "Swap"$/,/^free -h$/p'` onto a stub `PATH` (`sudo`=exec, and fakes for `swapon`/`mkswap`/`swaplabel`/`fallocate`; `fallocate` exits 1 to mimic the SD rootfs, forcing the `dd` path).

Pre-fix copy, against 300 MB of a never-mkswap'd 1 G file:
    === OLD code, same wedged state ===
    -- Swap ---
    swapon: /tmp/.../swapfile: read swap header failed: Invalid argument
    exit=255
Nothing after the Swap step ran -- exactly the reported failure.

Current code, same state:
    !!  /tmp/.../swapfile exists but is not a usable 1G swap area
    !!  (partial allocation, or mkswap never ran) -- rebuilding it
    ==> allocating 1G at /tmp/.../swapfile (slow on SD; be patient)
    Setting up swapspace version 1, size = 1073741824
      ok  swap on
    exit=0
Also passed: fresh box (no file), full-size-but-unformatted, and a second run ("swap already on", no-op, fstab line written once). And with `swapon` stubbed to fail outright (kernel refuses the area): `!! swapon failed ... continuing WITHOUT extra swap` / `!! the CUDA build may OOM -- use BUILD_JOBS=1`, exit=0 -- the later steps survive.

**Test:** Already covered: `bench/fixtures/run_regression.sh` gained `bring-up: an interrupted swapfile does not wedge the box`, which extracts `swapfile_ready()` from the real script by regex (so the test tracks the code rather than a copy of it) and asserts five states -- absent, short-and-unformatted, full-size-and-unformatted, short-with-a-trailing-magic, and good -- against a stub `swaplabel`. Full suite in a fresh copy: `53 passed, 0 failed`.

Gap worth closing if someone wants it: that test asserts only the PREDICATE. It never exercises the recovery branch itself (rm -> reallocate -> mkswap -> swapon) nor the new non-fatal `swapon` path, which is the half that keeps `jetson_clocks` and the fan running. My harness above does exercise both and they pass; the idiom that makes it possible off-device is lifting the step out with `sed -n '/^step "Swap"$/,/^free -h$/p'` and stubbing `sudo`/`swapon`/`mkswap`/`swaplabel`/`fallocate` on `PATH`, in the same spirit as the existing `PATH="$FIX/bin:$PATH" bash "$ROOT/scripts/60_collect_node_facts.sh"` case.

**Conflicts:** No conflict with the completed work. The `set +o pipefail` guard inside `swapfile_ready` follows the finding-#6 idiom documented at the top of scripts/lib/common.sh, so the two fixes agree rather than collide. | The tree is being edited live by another agent: scripts/00_system_prep.sh and scripts/20_build_llama_cpp.sh both changed twice while I was reading them (01:47, then 02:08/02:09). I re-diffed the Swap block after the last write and it is unchanged from the version I tested. I deliberately propose no edits so nothing lands on top of a file another session is mid-write on. | A transient `52 passed, 1 failed` I saw came from running the regression against a snapshot taken mid-edit; a fresh copy taken after the writes settled gives 53/0.

---

## Finding #13 — already-fixed

No change needed; already fixed and independently verified. One cosmetic item from the audit was not applied and I would leave it that way for now: the array still expands `"$(command -v ${HOST_CXX})"` with the inner variable unquoted, so a `HOST_CXX` containing whitespace would word-split. It can no longer fail silently -- `for c in "$HOST_CXX"` reaches `command -v "$c"` with the value intact and dies loudly first -- so the only remaining effect is cosmetic. If someone does want it, the exact replacement for the three lines is `-DCMAKE_CUDA_HOST_COMPILER="$(command -v "${HOST_CXX}")"`, `-DCMAKE_C_COMPILER="$(command -v "${HOST_CC}")"`, `-DCMAKE_CXX_COMPILER="$(command -v "${HOST_CXX}")"`.

Separately, the audit's framing blamed the Makefile for letting an operator arrive here: `build:` depends only on `config/node.env`, not on `toolchain`. That is still true and I would not change it -- `make toolchain` pip-installs into ~/.local as a side effect, so forcing it as a prerequisite of every build is worse than a loud error. The loud error was the actual finding.

**Justification (against current code):** Closed in `scripts/20_build_llama_cpp.sh` (mtime 2026-09-14 01:47, guard still intact after the 02:08 edit). The construct that swallowed the message is the command substitution inside the compound array assignment at what are now lines 147-152:

      CMAKE_ARGS+=(
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
        -DCMAKE_CUDA_HOST_COMPILER="$(command -v ${HOST_CXX})"
        -DCMAKE_C_COMPILER="$(command -v ${HOST_CC})"
        -DCMAKE_CXX_COMPILER="$(command -v ${HOST_CXX})"
      )

`command -v` prints nothing and returns 1 when the binary is absent; an assignment takes its exit status from the LAST command substitution in it, so `set -e` kills the script right there with no output. A guard now runs before the splice, inside the `BUILD_CUDA=1` arm:

      HOST_CC="${HOST_CXX/g++/gcc}"
      # Check the host compiler BEFORE it is spliced into CMAKE_ARGS below. There,
      # `command -v ${HOST_CXX}` inside a command substitution returns 1 and prints
      # nothing, and under `set -e` the array assignment takes the whole script down
      # with no output at all -- which is the normal outcome on a fresh JetPack box,
      # because CUDA 10.2 refuses gcc > 8 and g++-8 is not installed by default.
      for c in "$HOST_CXX" "$HOST_CC"; do
        command -v "$c" >/dev/null 2>&1 && continue
        warn "${c} not found, and CUDA $(cuda_version) will not accept the default compiler"
        warn "install it:   sudo apt-get install -y g++-8 gcc-8"
        warn "or build without CUDA:   BUILD_CUDA=0 $0"
        die  "missing host compiler: ${c}"
      done

It matches the era-detection style the task pointed at -- a multi-sentence WHY comment, then the mechanism -- and it uses `warn`/`die` from common.sh, which do exist (an intermediate version I caught mid-write used an `err` helper that common.sh does not define; that would have re-swallowed the message under `set -e`. The shipped version does not have that problem). Checking `$HOST_CC` as well as `$HOST_CXX` is right: `gcc-8` and `g++-8` are separate apt packages and either can be missing alone. The guard is correctly scoped to `BUILD_CUDA=1`, since the CPU-only path never uses either variable for a compiler flag.

**Reproduction:** Reproduced the swallow in isolation first, on bash 5.2:
    $ bash -c 'set -euo pipefail; A=(x); A+=( -DFOO="$(command -v definitely-not-here)" ); echo reached'
    exit=1            # "reached" never printed, nothing on stderr
    $ bash -c 'set -euo pipefail; A=(x); A+=( -DA="$(command -v nope)" -DB="$(echo ok)" ); echo reached'
    reached           # a LATER succeeding substitution masks the earlier failure

Then against a pre-edit copy of the kit (stub `git`/`cmake`/`nvcc` reporting release 10.2, `HOST_CXX=g++-8`, no g++-8 on the box). Output ended at:
    -- Configure ---
    ==> llama.cpp CMake option era: llama
    EXIT=1
Nothing further -- exactly the reported "exit 1 and no message".

Against the current file, same fixture:
    !!  g++-8 not found, and CUDA 10.2 will not accept the default compiler
    !!  install it:   sudo apt-get install -y g++-8 gcc-8
    !!  or build without CUDA:   BUILD_CUDA=0 /.../scripts/20_build_llama_cpp.sh
    XX  missing host compiler: g++-8
    EXIT=1

I also checked the escape hatch that message advertises actually works, since nothing had tested it: `./20_build_llama_cpp.sh --cpu-only` with g++-8 still absent runs to completion, exit=0, and writes build_info.json with `"backend": "cpu-neon"` and a harmless `"host_cxx": ""`. That last field is a command substitution inside a heredoc, and I confirmed separately that a heredoc substitution does NOT trip `set -e` (the exit status belongs to the `cat`), so the CPU-only path has no sibling of this bug.

**Test:** Already covered: `bench/fixtures/run_regression.sh` gained `bring-up: a missing host compiler fails loudly, not silently`, which copies the kit to a tmpdir, writes a fixture node.env with `HOST_CXX=g++-nonexistent-99` and `BUILD_CUDA=1`, puts a fake `nvcc` printing "release 10.2" on `PATH`, `git init`s a one-line CMakeLists.txt as the source tree, runs the real script and greps for both `g++-nonexistent-99 not found` and `apt-get install -y g++-8`. Full suite in a fresh copy: `53 passed, 0 failed`.

Gap worth closing if someone wants it: the test asserts the message but not that the remedy it prints actually works. One more line in the same block covers it, in the harness's own style -- reuse `$BK` and add
    PATH="$BK/fakebin:$PATH" bash "$BK/Jetson/scripts/20_build_llama_cpp.sh" --cpu-only \
      > "$FIX/build_cpu.txt" 2>&1 \
      && ok "and the --cpu-only escape it advertises still completes" \
      || bad "the advertised CPU-only escape does not work"
That needs the fixture `cmake` stub to create a `ggml-rpc-server` under `$LLAMA_BIN` on its `--build` invocation, or the `bin_rpc` check at the Verify step dies; I confirmed the path passes with such a stub.

**Conflicts:** No conflict with the completed work. The guard is a plain `command -v` test outside any pipeline, so it neither needs nor disturbs the finding-#6 `set +o pipefail` idiom. | Overlaps finding #10 in scripts/10_install_toolchain.sh only in spirit: that script already does `need_cmd gcc-8 "run 00_system_prep.sh first"` at line 8. The new guard duplicates that check, deliberately, because `make build` does not depend on `make toolchain`. Do not 'deduplicate' them. | scripts/20_build_llama_cpp.sh was rewritten again at 02:08 by another session while I was testing -- build_info.json now records a `cmake_options` field and the CPU-only arm emits `-DGGML_BLAS=OFF` rather than `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS`. That is someone else's finding, not mine, and I left it alone. I re-grepped after that write: the #13 guard survives it intact. I propose no edits so nothing lands on top of an in-flight rewrite.

---

## Finding #14 — still-live

Select the zone by its `type`, never by max, and put the selection in scripts/lib/common.sh rather than inline.

`thermal_zone_c()` joins l4t_version/cuda_version in the block of fact reporters that 'report a fact or print nothing' and must never abort the caller — which is exactly this function's contract, since the zones do not exist off-device. It prints `"<zone-type> <degrees C>"` or nothing. It prefers thermal-fan-est (= tegrastats `thermal@` = temp_thermal_c, so the pre-flight reading and every sweep row's temperature are the same quantity), falls back to the hotter of CPU-therm and GPU-therm for a board with no fan estimator registered, and can never select PMIC-Die. Zone NUMBERS are not stable across L4T releases, which is the second reason the `type` file is the only safe selector.

Putting it in common.sh is what makes it testable: 50_bench_device.sh's whole pre-flight block sits inside `if is_jetson;`, which is false off-device, so an inline loop could not be exercised at all. `HS_THERMAL_ZONES` overrides the glob for the fixture, in the same spirit as the kit's existing HS_SERVE / HS_FAKE_* knobs, and nothing on a Jetson sets it.

50_bench_device.sh then calls it, prints the zone name alongside the temperature so the reading is self-describing, and warns when no usable zone is found instead of silently recording nothing.

**Justification (against current code):** scripts/50_bench_device.sh:26-28 is verbatim what the audit described:

    T="$( set +o pipefail
      cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null \
      | sort -n | tail -1 || true )"
    if [ -n "$T" ]; then
      TC=$((T/1000))
      echo "  temp start : ${TC} C"
      # Starting hot means the first cases run at a different clock than the last.
      [ "$TC" -gt 45 ] && warn "board is warm; let it settle below 45 C for comparable runs"
    fi

The `set +o pipefail` wrapper is the finding-#6 fix and is correct; the `sort -n | tail -1` MAX over all zones is untouched. The kit's own fixture already contains the counter-evidence — bench/fixtures/fake_tegrastats.py emits `PMIC@100C`, and telemetry/tegrastats_logger.py's sample line at :9 shows the same.

I verified the zone layout rather than taking it on trust. NVIDIA's own forum thread on thermal_zone4 gives the R32.x Nano mapping explicitly: thermal_zone0 AO-therm, 1 CPU-therm, 2 GPU-therm, 3 PLL-therm, 4 PMIC-Die, and a 6th, thermal_zone5 thermal-fan-est, which the audit note omits. PMIC-Die is confirmed a dummy: the value is "fixed to 100C to illustrate the temp of PMIC" because the sensor is not wired to the module, and a second thread documents boards running 8+ hours at a reported 100 C with no thermal shutdown. So the max is 100000 on every board, cold or throttling alike.

The audit's suggested fix (CPU-therm/GPU-therm) is right but incomplete on the consistency point it also raises. telemetry/tegrastats_logger.py maps `thermal@` to `temp_thermal_c` (lines 123-124), and `temp_thermal_c` is the column the sweep's whole thermal story is built on — energy_window.py:194 and run_matrix.py's energy_between() both read it, and nothing else. `thermal@` is the thermal-fan-est zone. So the pre-flight gate should prefer that zone: then the number the operator sees before the sweep and the temp_baseline_c/temp_start_c on every row are the same quantity and can be compared directly.

**Reproduction:** Built the R32.7 zone tree on a cold board — AO 38.5, CPU 31.0, GPU 30.0, PLL 29.0, PMIC-Die 100.0, thermal-fan-est 30.25 — and ran the shipped expression against it:

    T="$( set +o pipefail; cat $Z/thermal_zone*/temp 2>/dev/null | sort -n | tail -1 || true )"
    echo "  temp start : $((T/1000)) C"
    [ "$((T/1000))" -gt 45 ] && echo "  WARN board is warm; let it settle below 45 C"

    temp start : 100 C
    WARN board is warm; let it settle below 45 C

A 31 °C board, reported as 100 °C, warning on every invocation. After the fix, against the same tree:

    thermal-fan-est 30        <- the real temperature, from the zone tegrastats calls thermal@
    CPU-therm 31              <- with thermal_zone5 removed: falls back, never to PMIC-Die
    rc=0 (nothing printed)    <- no zones at all (off-device): silent, does not abort the caller
    CPU-therm 72              <- same tree with CPU-therm set to 72.5 C: the warning now fires for a real reason

**Test:** Off-hardware, in the run_regression.sh idiom, and it runs green:

  thermal: the pre-flight gate does not read the PMIC dummy zone
    ok   the old max-over-zones reads the dummy: 100 C
    ok   type-selected: 30 C from thermal-fan-est, the zone tegrastats calls thermal@
    ok   falls back to CPU-therm 31 C, not PMIC-Die 100 C
    ok   no zones: silent, exit 0, caller survives

A fixture zone tree with the R32.7 layout is built under $FIX, and the FIRST assertion runs the old `cat ... | sort -n | tail -1` against it and requires 100 C — so if the fixture ever stops reproducing the defect, that check fails rather than the other three passing vacuously. The rest exercise thermal_zone_c() through HS_THERMAL_ZONES: preferred zone, fallback with thermal_zone5 removed, and the off-device case (prints nothing, returns 0, does not abort a caller running under `set -euo pipefail`).

What this cannot prove off hardware: that a real JetPack 4.6 Nano actually names its zones AO-therm/CPU-therm/GPU-therm/PLL-therm/PMIC-Die/thermal-fan-est. That rests on NVIDIA's forum documentation, not on a board in front of me. The fallback chain is written so that being wrong is survivable rather than silent: an unexpected set of zone names yields no reading and an explicit warning, never a fabricated 100 C. On first bring-up, `cat /sys/devices/virtual/thermal/thermal_zone*/type` settles it in one command.

**Conflicts:** Shares scripts/lib/common.sh with finding #10's fix (which calls thermal_zone_c) and with the rpc-binary-name fact-check, which edits bin_rpc() further down the same file. Adjacent hunks, no overlap. | Shares the `head_ "regression baseline"` insertion anchor with finding #10's fixture additions. Both are insertions before that line; apply them in either order, but not by pasting one over the other. | bench/fixtures/run_regression.sh is under concurrent edit (mtime 01:49, newer than my copy, with two bring-up sections added by the #12/#13 sessions). The anchor is still present and unique in the live file; apply against the current file. | Depends on nothing in finding #10, but the two are worth landing together: the pre-flight reading is only comparable with the sweep's own temp_baseline_c because both now come from the thermal-fan-est / temp_thermal_c sensor. Landing #14 alone still fixes the 100 C bug; landing it without #10 leaves the sweep with no temperature column to compare it against.

### edit — `scripts/lib/common.sh`

Belongs with the other fact reporters whose absence must be survivable. No pipelines, so it needs no `set +o pipefail` guard and complies with the rule documented at the top of the file. The `-n "$v"` guard keeps `[ -gt ]` from erroring on an unreadable zone, and both comparisons sit inside an `if` condition, where set -e is exempt.

**OLD**
```
cuda_major() { ( set +o pipefail; cuda_version | cut -d. -f1 || true ); }

```

**NEW**
```
cuda_major() { ( set +o pipefail; cuda_version | cut -d. -f1 || true ); }

# Prints "<zone-type> <degrees C>" for the board's real temperature sensor, or
# nothing at all when there is none (off-device, or a kernel without these
# zones). Never aborts the caller -- see the note above this block.
#
# Selects the zone by its `type` file, never by taking a max over all of them.
# On L4T R32.7 a Nano exposes AO-therm, CPU-therm, GPU-therm, PLL-therm,
# PMIC-Die and thermal-fan-est, and PMIC-Die is a DUMMY: its driver has no
# sensor to read and reports a hard-coded 100000 m C. A max over the zones is
# therefore always exactly 100 C on every board, cold or throttling alike. Zone
# NUMBERS are not stable across L4T releases either, so the type file is the
# only safe selector.
#
# thermal-fan-est is preferred because it is the zone tegrastats prints as
# `thermal@`, which telemetry/tegrastats_logger.py records as temp_thermal_c and
# every sweep row is scored on -- so a pre-flight reading and the sweep's own
# temp_baseline_c / temp_start_c columns are the same quantity. CPU-therm and
# GPU-therm are the fallback for a board whose fan estimator is not registered.
#
# HS_THERMAL_ZONES overrides the glob so the selection can be exercised against
# a fixture zone tree off-device; nothing on a Jetson sets it.
thermal_zone_c() {
  local z zt v t="" zone=""
  for z in ${HS_THERMAL_ZONES:-/sys/devices/virtual/thermal/thermal_zone*}; do
    zt="$(cat "$z/type" 2>/dev/null || true)"
    case "$zt" in
      thermal-fan-est)
        t="$(cat "$z/temp" 2>/dev/null || true)"; zone="$zt"; break ;;
      CPU-therm|GPU-therm)
        v="$(cat "$z/temp" 2>/dev/null || true)"
        if [ -n "$v" ] && [ "$v" -gt "${t:-0}" ]; then t="$v"; zone="$zt"; fi ;;
    esac
  done
  [ -n "$t" ] || return 0
  echo "$zone $((t/1000))"
}

```

### edit — `scripts/50_bench_device.sh`

Prints the zone name with the reading so the number is self-describing, and says so when there is no usable sensor instead of silently skipping the condition docs/10 requires to be recorded. The > 45 C warning is unchanged and now fires for a real reason.

**OLD**
```
  T="$( set +o pipefail
    cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null \
    | sort -n | tail -1 || true )"
  if [ -n "$T" ]; then
    TC=$((T/1000))
    echo "  temp start : ${TC} C"
    # Starting hot means the first cases run at a different clock than the last.
    [ "$TC" -gt 45 ] && warn "board is warm; let it settle below 45 C for comparable runs"
  fi

```

**NEW**
```
  # A max over thermal_zone*/temp is always exactly 100 C on this board: the
  # PMIC-Die zone is a dummy that reports a hard-coded value. thermal_zone_c
  # selects the real sensor by its `type` -- see the note in lib/common.sh --
  # and it reports the same quantity the sweep rows carry as temp_start_c.
  TZ="$(thermal_zone_c)"
  if [ -n "$TZ" ]; then
    TZONE="${TZ% *}"; TC="${TZ#* }"
    echo "  temp start : ${TC} C (${TZONE})"
    # Starting hot means the first cases run at a different clock than the last.
    [ "$TC" -gt 45 ] && warn "board is warm; let it settle below 45 C for comparable runs"
  else
    warn "no CPU-therm/GPU-therm/thermal-fan-est zone found; thermal starting point not recorded"
  fi

```

### edit — `bench/fixtures/run_regression.sh`

Pins the bug and the fix side by side against the same fixture tree: the old expression must still read 100 C from it, or the fixture has stopped reproducing the defect and the other three assertions prove nothing.

**OLD**
```
head_ "regression baseline"
```

**NEW**
```
head_ "thermal: the pre-flight gate does not read the PMIC dummy zone"
# A Nano's thermal_zone4 is PMIC-Die, whose driver has no sensor and reports a
# constant 100000 m C. `cat thermal_zone*/temp | sort -n | tail -1` therefore
# printed 100 C on a cold board, every time, and the warning it drives became
# noise. Zone NUMBERS also move between L4T releases, so the selection is by
# `type`. Zone tree below is the R32.7 Nano layout.
ZONES="$FIX/zones"
mkdir -p "$ZONES"
i=0
# AO-therm 38.5, CPU-therm 31.0, GPU-therm 30.0, PLL-therm 29.0,
# PMIC-Die 100.0 (the dummy), thermal-fan-est 30.25 -- a cold, idle board.
for zt in AO-therm:38500 CPU-therm:31000 GPU-therm:30000 \
          PLL-therm:29000 PMIC-Die:100000 thermal-fan-est:30250; do
  mkdir -p "$ZONES/thermal_zone$i"
  printf '%s' "${zt%%:*}" > "$ZONES/thermal_zone$i/type"
  echo "${zt#*:}"         > "$ZONES/thermal_zone$i/temp"
  i=$((i+1))
done
OLD="$( set +o pipefail; cat "$ZONES"/thermal_zone*/temp | sort -n | tail -1 || true )"
[ "$((OLD/1000))" = "100" ] && ok "the old max-over-zones reads the dummy: 100 C" \
  || bad "fixture zone tree does not reproduce the bug"
NEW="$( HS_THERMAL_ZONES="$ZONES/thermal_zone*" \
        bash -c '. scripts/lib/common.sh; thermal_zone_c' )"
[ "$NEW" = "thermal-fan-est 30" ] \
  && ok "type-selected: 30 C from thermal-fan-est, the zone tegrastats calls thermal@" \
  || bad "zone selection returned '$NEW', expected 'thermal-fan-est 30'"
# and with no fan estimator registered it falls back to the hotter of CPU/GPU,
# still never to PMIC-Die
rm -rf "$ZONES/thermal_zone5"
NEW="$( HS_THERMAL_ZONES="$ZONES/thermal_zone*" \
        bash -c '. scripts/lib/common.sh; thermal_zone_c' )"
[ "$NEW" = "CPU-therm 31" ] && ok "falls back to CPU-therm 31 C, not PMIC-Die 100 C" \
  || bad "fallback returned '$NEW', expected 'CPU-therm 31'"
# off-device (no zones at all) it must print nothing and abort nothing
NEW="$( HS_THERMAL_ZONES="$FIX/no-such-zones*" \
        bash -c '. scripts/lib/common.sh; thermal_zone_c; echo rc=$?' )"
[ "$NEW" = "rc=0" ] && ok "no zones: silent, exit 0, caller survives" \
  || bad "off-device zone read returned '$NEW'"

head_ "regression baseline"
```

---

## Finding #17 — still-live

Add `scripts/45_capture_window.sh start|mark|idle|stop|path|status` — the node-side capture the orchestrator drives over ssh. It runs `tegrastats_logger.py` into `$RESULTS_DIR` under the house `result_path` naming, and owns `$RESULTS_DIR/.marker_$NODE_ID`, the SAME marker path `run_matrix.py` uses, so a cluster CSV is read by exactly the same `energy_window.py` commands as a sweep CSV and `90_collect_results.sh` tarballs it without being told.

Who writes the marker, and when: the ORCHESTRATOR does, over ssh, because it is the only machine that knows which phase it asked for — the Jetson is an RPC worker and is handed tensors, never phases. It brackets a command that does ONE phase (`-n 0` for a prefill leg; a long generation for a decode leg), exactly as §4 splits the local matrix. The ssh round trip puts the marker edge within about one sample period of the real one, which is fine for a seconds-long leg and hopeless for the boundary inside a single request — that limit is written down rather than papered over, in §5 and in §7's "cannot" list.

Three things the helper refuses, each of which has a silent-failure history in this kit: it will not start while another `tegrastats` is alive (the board allows one; the always-on unit is the usual culprit and records no markers); it will not report success until the logger has actually produced samples; and it will not accept a label whose phase field is not one `energy_window.py` can select, which would otherwise fail only at analysis time. Markers are written by atomic rename, because a truncating redirect can be read mid-write and a blank marker is counted as an idle sample, which makes `energy_window.py` refuse the join.

`70_cluster_smoke.sh` now rehearses the measurement path, not just the inference: it brackets the two-node run with a capture and asserts a window comes back, so "this node cannot label its own windows" is discovered at smoke time rather than the morning after the experiment.

**Justification (against current code):** Nothing has closed this. Grepping the whole kit for a logger or marker outside the driver returns only the driver itself:

    $ grep -rn "tegrastats_logger\|marker" scripts/ Makefile README.md
    scripts/90_collect_results.sh:23:# .marker_* is scratch state, not data.
    scripts/90_collect_results.sh:24:find "$RESULTS_DIR" -maxdepth 1 -type f ! -name '.marker_*' ...
    README.md:85:│   ├── tegrastats_logger.py     tegrastats -> CSV with running energy integral

So the only writer of the marker file is still `bench/run_matrix.py`, which builds it locally at line 624 (`marker_file = os.path.join(out_dir, ".marker_" + node_id)`) and is the only thing that starts a logger (line 653: `logger = subprocess.Popen([... "--marker-file", marker_file])`). `scripts/70_cluster_smoke.sh` still runs the distributed leg with no power logging whatsoever:

    "$CLI" -m "$MODEL" --rpc "$RPCLIST" -ngl 99 -n 32 -p "Explain what prefill and decode are, in one sentence." \
      2>&1 | tail -20 || { warn "distributed run failed"; FAIL=1; }

The one documented fallback is still the hand recipe in docs/10_measurement_protocol.md §5, and it still names a path that does not match the driver's: `--marker-file /tmp/hs_marker` / `echo prefill > /tmp/hs_marker`, against the driver's `$RESULTS_DIR/.marker_$NODE_ID`.

The always-on unit cannot stand in for it either — systemd/heteroserve-telemetry.service:15 says so in as many words: "This CSV carries NO phase markers, by design: nothing here knows when a prefill or decode window starts." And it would actively block a capture, because the same unit's header (line 11) notes tegrastats permits one reader.

The finding's own severity is right: `telemetry/energy_window.py` refuses an unmarked join outright ("--bench needs a window: pass --marker LABEL"), and `windows_by_marker` exits with "no samples carry marker %r" — so an unmarked cluster CSV yields nothing, after the runs are over.

**Reproduction:** Ran in a copy (`cp -a .../Jetson $(mktemp -d)/kit`), with `bench/fixtures/fake_tegrastats.py` on PATH as `tegrastats`.

Before: there is no entry point to reproduce — the failure is an absence. I reproduced its consequence instead: started the logger by hand with no marker file and joined, which is what a cluster run leaves you holding:

    $ python3 telemetry/energy_window.py <csv> --marker prefill
    no samples carry marker 'prefill' (is the logger reading the marker file?)

After adding scripts/45_capture_window.sh and wiring it into 70_cluster_smoke.sh, in the copy:

    $ ./scripts/45_capture_window.sh start --tag split
      ok  capturing to .../fixture_cluster_split_power_20260913T185009Z.csv at 5 Hz (pid 51322), marker=idle
    $ ./scripts/45_capture_window.sh mark prefill:smoke:001
      ok  marker=prefill:smoke:001 at 2026-09-13T18:50:10.707Z
    ... idle ... mark decode:smoke:002 ... stop
    $ python3 telemetry/energy_window.py <csv> --marker prefill
    window 0  marker=prefill:smoke:001  1.80 s  10 samples
      module    3.4243 J   mean   1900.0 mW ...
    $ python3 telemetry/energy_window.py <csv> --marker decode
    window 0  marker=decode:smoke:002  1.80 s  10 samples

And the whole smoke script, run in a private PID namespace so this shared VM's stray `tegrastats` did not trip the one-instance guard:

    ── Two-node inference ──
    ==> llama-cli --rpc 127.0.0.1:59998
      ok  capturing to .../fixture_cluster_smoke_power_20260913T185818Z.csv at 5 Hz (pid 28), marker=idle
      ok  marker=combined:cluster_smoke:001 at 2026-09-13T18:58:19.626Z
      ok  marker=idle at 2026-09-13T18:58:21.642Z
      ok  phase-marked window captured — .../fixture_cluster_smoke_power_20260913T185818Z.csv

Refusal paths, each exercised: second `start` refused by the pid file; `mark prefil:x:1` refused ("phase must be prefill, decode or combined"); `mark` with no capture refused; a decoy process whose comm is `tegrastats` refused the start; a tegrastats that exits 1 at startup produced "logger produced no samples" and left no stale state.

The existing suite still passes in the copy with every edit applied: `51 passed, 0 failed`, and `63 passed, 0 failed` with the new fixture blocks for #17 and #25 added.

**Test:** Off-hardware, in the style of `bench/fixtures/run_regression.sh` — the fixture block above. It needs no Jetson: `bench/fixtures/fake_tegrastats.py` on PATH as `tegrastats` supplies the samples, and the assertions are on `energy_window.py`'s output, not on any board reading.

Run it as the suite is run:

    cp -a /mnt/data/KMITL/Final/HeteroServe/Jetson $(mktemp -d)/kit
    cd <copy>/kit && ./bench/fixtures/run_regression.sh

Measured in my copy: 51 passed / 0 failed before the new blocks, 63 passed / 0 failed after (12 new assertions across #17 and #25).

One caveat, stated rather than worked around: on a machine where some other process's `comm` is already `tegrastats`, the capture's one-instance guard fires and the happy path skips with a printed note. That guard is system-wide on purpose — the device has exactly one tegrastats — so the fixture reports the skip instead of weakening the guard. I hit exactly this on the shared build VM (a stray `/bin/sh .../tegrastats` from another agent's run) and confirmed the full path by re-running under `unshare -rpf --mount-proc`.

What genuinely cannot be tested off-hardware: the marker-edge accuracy claim. Whether an ssh round trip really lands within one sample period, and how much of a 512-token prefill that costs, needs the real board, a real orchestrator and a real link. The fixture proves the mechanism works and the labels are selectable; the size of the boundary uncertainty has to be measured once on the cluster and written into §5.

**Conflicts:** Depends on #25's `scripts/lib/common.sh` edit: `45_capture_window.sh start` calls `tegrastats_running()`, which that edit introduces. Apply #25's common.sh block first. If #25 is not taken, substitute `if pgrep -x tegrastats >/dev/null; then` for `if tegrastats_running; then` in the helper. | Both findings insert a fixture block immediately before `head_ "regression baseline"` in `bench/fixtures/run_regression.sh`. The anchor stays unique whichever order they are applied in, and both blocks end up before it — but do not apply them as one merged patch expecting a single anchor match. | Deliberately does NOT touch `systemd/heteroserve-telemetry.service`. Finding #18's proposed fix wanted `--marker-file` added to that unit; the unit as it now stands refuses markers on purpose (line 15: "This CSV carries NO phase markers, by design"), and the capture is the correct home for them because it also owns the window's start and end. The helper instead refuses to run alongside the unit and says which to stop. Anything that later adds `--marker-file` to the unit would put two writers on one marker file and reintroduce cross-run contamination. | Adjacent to finding #21 (a logger that dies at startup is never detected), which is still live in `bench/run_matrix.py` — it Popens the logger and never checks `poll()`. The helper closes that hole on the capture path only; #21's own fix in the driver is independent and will not conflict. | Observation, not an edit: `bench/run_matrix.py`'s `set_marker` writes with a truncating `open(marker_file, "w")`, so a sample read in that gap records a blank marker — which `energy_window.py` counts as idle and refuses to join. The helper uses an atomic rename. Making the driver match would be a one-line change but it sits inside the settled finding-#3 code and the sweep marks far less often than a remote does, so I have left it alone and only flagged it.

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/scripts/45_capture_window.sh`

New file, mode 0755. The node-side capture entry point. It is deliberately a thin state machine over the existing logger: it adds no measurement code, only the marker file's lifetime and the three refusals that keep a silent empty capture from being mistaken for a run.

**OLD**
```

```

**NEW**
```
#!/usr/bin/env bash
# Capture a phase-marked power window on THIS node for a run driven from
# somewhere else — the cluster experiment, a manual llama-cli, anything that is
# not bench/run_matrix.py.
#
#   ./45_capture_window.sh start [--tag NAME] [--hz N]
#   ./45_capture_window.sh mark prefill[:case_id[:NNN]]
#   ./45_capture_window.sh idle
#   ./45_capture_window.sh stop
#   ./45_capture_window.sh path      # the CSV this capture is writing
#   ./45_capture_window.sh status
#
# WHY THIS EXISTS. Only run_matrix.py writes the marker file, and it only runs
# when the benchmark is driven locally. In a split run this node is an RPC
# worker: it never learns that a prefill started, so its power CSV carries no
# phase labels and energy_window.py can attribute nothing to it. The study's
# central number is prefill-vs-decode energy ACROSS nodes, so a worker that
# cannot label its own windows makes the deliverable uncomputable.
#
# WHO MARKS, AND WHEN. The orchestrator does, over ssh, because the orchestrator
# is the only machine that knows which phase it asked for:
#
#   ssh jetson-nano-01 HeteroServe/Jetson/scripts/45_capture_window.sh start --tag split
#   ssh jetson-nano-01 HeteroServe/Jetson/scripts/45_capture_window.sh mark prefill:split_a:001
#   llama-cli --rpc jetson-nano-01:50052 -ngl 99 -n 0 -f prompt.txt      # prefill only
#   ssh jetson-nano-01 HeteroServe/Jetson/scripts/45_capture_window.sh idle
#   ...decode leg, marked the same way...
#   ssh jetson-nano-01 HeteroServe/Jetson/scripts/45_capture_window.sh stop
#
# The ssh round trip puts the marker edge within roughly one sample period of
# the real phase edge, so the command being bracketed must do ONE phase — the
# same "one window, one phase" rule the local matrix follows. Marking the two
# phases of a single interleaved request from another machine does not work at
# any sample rate this board offers; see docs/10_measurement_protocol.md §5.
. "$(dirname "$0")/lib/common.sh"
load_env

MARKER="${RESULTS_DIR}/.marker_${NODE_ID}"
STATE="${RESULTS_DIR}/.capture_${NODE_ID}"
LOGGER="${HS_JETSON_DIR}/telemetry/tegrastats_logger.py"

# The marker is read by the logger on every sample, from a machine that may be
# marking at any instant. A plain redirect truncates first, so a sample landing
# in that gap records an EMPTY marker -- which energy_window.py counts as an
# idle sample and refuses to join. Rename is atomic within the directory, so a
# reader sees either the old label or the new one, never nothing.
set_marker() {
  printf '%s' "$1" > "${MARKER}.new"
  mv -f "${MARKER}.new" "$MARKER"
}

read_state() {
  [ -f "$STATE" ] || return 1
  # shellcheck disable=SC1090
  . "$STATE"
  [ -n "${CAP_PID:-}" ] && kill -0 "$CAP_PID" 2>/dev/null
}

CMD="${1:-status}"; shift || true

case "$CMD" in
start)
  TAG=""; HZ="${TELEMETRY_HZ:-5}"
  while [ $# -gt 0 ]; do
    case "$1" in
      --tag) TAG="$2"; shift 2 ;;
      --hz)  HZ="$2";  shift 2 ;;
      *) die "unknown argument: $1" ;;
    esac
  done
  read_state && die "a capture is already running (pid ${CAP_PID}, ${CAP_CSV}) — stop it first"
  # Gate on the instrument, not on /etc/nv_tegra_release: a capture without
  # tegrastats produces an empty CSV whatever the board is, and gating this way
  # lets the fixture drive the whole path with a fake on PATH.
  command -v tegrastats >/dev/null 2>&1 \
    || die "no tegrastats on PATH — a power capture needs the Jetson"
  # tegrastats permits ONE instance per boot. A second one exits immediately,
  # the logger then writes a header and nothing else, and the whole cluster run
  # is measured against an empty CSV that is only noticed at analysis time. The
  # always-on telemetry unit is the usual culprit, and it records no markers, so
  # it cannot stand in for this capture.
  if tegrastats_running; then
    die "tegrastats is already running (the telemetry unit, or a sweep).
       stop it first:  sudo systemctl stop heteroserve-telemetry"
  fi
  CSV="$(result_path "cluster${TAG:+_${TAG}}_power" csv)"
  set_marker idle
  python3 "$LOGGER" --out "$CSV" --hz "$HZ" --node-id "$NODE_ID" \
          --marker-file "$MARKER" >"${CSV}.log" 2>&1 &
  PID=$!
  printf 'CAP_PID=%s\nCAP_CSV=%s\n' "$PID" "$CSV" > "$STATE"
  # Do not report success until the logger has actually produced samples. A
  # logger that dies at startup leaves a plausible-looking CSV with a header and
  # no rows, and an overnight experiment is then measured against nothing.
  N=0
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if [ -f "$CSV" ]; then N="$( set +o pipefail; wc -l < "$CSV" || true )"; fi
    [ "${N:-0}" -ge 3 ] && break
    sleep 1
  done
  if ! kill -0 "$PID" 2>/dev/null || [ "${N:-0}" -lt 3 ]; then
    rm -f "$STATE"
    sed 's/^/       /' "${CSV}.log" >&2 || true
    die "logger produced no samples — see ${CSV}.log"
  fi
  ok "capturing to ${CSV} at ${HZ} Hz (pid ${PID}), marker=idle"
  echo "  mark a phase from the orchestrator:"
  echo "    ssh ${NODE_ID} $(basename "$0") mark prefill:<case>:001"
  ;;

mark|idle)
  read_state || die "no capture is running — start one first"
  if [ "$CMD" = "idle" ]; then
    LABEL="idle"
  else
    LABEL="${1:-}"
    [ -n "$LABEL" ] || die "usage: $(basename "$0") mark <phase>[:<case_id>[:<NNN>]]"
    # energy_window.py selects windows by the label or by its phase prefix, so a
    # label whose first field is not a phase this kit knows selects nothing and
    # fails only at analysis time, with the run already over.
    case "${LABEL%%:*}" in
      prefill|decode|combined) ;;
      *) die "phase must be prefill, decode or combined (got '${LABEL%%:*}')" ;;
    esac
  fi
  set_marker "$LABEL"
  # Printed so the orchestrator's own log can be lined up against this CSV; §3
  # of the protocol is what keeps the two clocks close enough for that to mean
  # something.
  ok "marker=${LABEL} at $(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
  ;;

stop)
  read_state || { rm -f "$STATE"; die "no capture is running"; }
  set_marker idle
  # One sample of idle after the last phase, so the final window has a closing
  # edge rather than ending at whatever the logger happened to write last.
  sleep 1
  kill -TERM "$CAP_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5; do kill -0 "$CAP_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL "$CAP_PID" 2>/dev/null || true
  rm -f "$STATE"
  ok "stopped; ${CAP_CSV}"
  echo "  per-phase energy:"
  echo "    python3 telemetry/energy_window.py ${CAP_CSV} --marker prefill --idle-mw <floor>"
  ;;

path)
  # Just the CSV path, for a caller that has to hand it to energy_window.py.
  read_state || die "no capture is running"
  printf '%s\n' "$CAP_CSV"
  ;;

status)
  if read_state; then
    ok "capturing to ${CAP_CSV} (pid ${CAP_PID}), marker=$(cat "$MARKER" 2>/dev/null)"
  else
    echo "  no capture running"
  fi
  ;;

*) die "usage: $(basename "$0") start|mark <label>|idle|stop|path|status" ;;
esac

```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/scripts/70_cluster_smoke.sh`

This is the line the finding cites. The smoke test proved the cluster could infer but not that the cluster could be measured, which is the only reason the cluster exists here. Gated on tegrastats rather than is_jetson so the same code path runs on the orchestrator (skipped, no instrument) and under the fixture (fake on PATH).

**OLD**
```
    info "llama-cli --rpc ${RPCLIST}"
    # -ngl 99 so layers are actually handed to the remote backends; without it
    # the RPC servers connect and then sit idle, which looks like success.
    "$CLI" -m "$MODEL" --rpc "$RPCLIST" -ngl 99 -n 32 -p "Explain what prefill and decode are, in one sentence." \
      2>&1 | tail -20 || { warn "distributed run failed"; FAIL=1; }
  fi
fi

```

**NEW**
```
    info "llama-cli --rpc ${RPCLIST}"
    # Rehearse the MEASUREMENT path, not just the inference. A split run is only
    # a result if this node can put a phase label on its own power window, and
    # nothing else in the kit does that for a remotely driven run -- run_matrix
    # writes the marker file, and run_matrix is not in this loop. The night of
    # the experiment is the wrong time to find that out.
    CAPTURE="${HS_JETSON_DIR}/scripts/45_capture_window.sh"
    CAP=0
    if command -v tegrastats >/dev/null 2>&1 && "$CAPTURE" start --tag smoke; then
      CAP=1
      "$CAPTURE" mark combined:cluster_smoke:001
    fi
    # -ngl 99 so layers are actually handed to the remote backends; without it
    # the RPC servers connect and then sit idle, which looks like success.
    "$CLI" -m "$MODEL" --rpc "$RPCLIST" -ngl 99 -n 32 -p "Explain what prefill and decode are, in one sentence." \
      2>&1 | tail -20 || { warn "distributed run failed"; FAIL=1; }
    if [ "$CAP" = "1" ]; then
      "$CAPTURE" idle
      CAPCSV="$("$CAPTURE" path)"
      "$CAPTURE" stop >/dev/null
      if python3 "${HS_JETSON_DIR}/telemetry/energy_window.py" "$CAPCSV" \
           --marker combined >/dev/null 2>&1; then
        ok "phase-marked window captured — ${CAPCSV}"
      else
        warn "no window carries the marker; a split run on this node would not be measurable"
        FAIL=1
      fi
    fi
  fi
fi

```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/scripts/90_collect_results.sh`

The capture's pid/path state file lives beside the marker file and is scratch for the same reason. The capture CSV and its .log sidecar need no new rule — they are plain files in RESULTS_DIR and are already collected.

**OLD**
```
# .marker_* is scratch state, not data.
find "$RESULTS_DIR" -maxdepth 1 -type f ! -name '.marker_*' -exec cp {} "${D}/results/" \; 2>/dev/null || true
```

**NEW**
```
# .marker_* and .capture_* are scratch state, not data. (The capture's *.csv.log
# sidecar IS data: it is the only place a logger that died mid-run says so.)
find "$RESULTS_DIR" -maxdepth 1 -type f ! -name '.marker_*' ! -name '.capture_*' \
  -exec cp {} "${D}/results/" \; 2>/dev/null || true
```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/Makefile`

Declare the new target.

**OLD**
```
.PHONY: help all prep toolchain build models facts rpc bench smoke collect clean check install-units
```

**NEW**
```
.PHONY: help all prep toolchain build models facts rpc bench capture smoke collect clean check install-units
```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/Makefile`

The help text is the only index of the kit, so an entry point absent from it does not exist. Leading whitespace on both new lines is a TAB, as everywhere else in this recipe.

**OLD**
```
	@echo "  make smoke       cluster reachability / RPC / two-node check"
```

**NEW**
```
	@echo "  make capture A=\"start --tag split\"   mark this node's power window for a"
	@echo "                   run driven from the orchestrator (start|mark|idle|stop|path)"
	@echo "  make smoke       cluster reachability / RPC / two-node check"
```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/Makefile`

The audit asks for a make target. The verb takes an argument, so it goes through A= rather than being split into four targets that would all have to be kept in step with the script.

**OLD**
```
bench: config/node.env     ; $(S)/50_bench_device.sh
```

**NEW**
```
bench: config/node.env     ; $(S)/50_bench_device.sh
# A= carries the subcommand: make capture A="mark prefill:split_a:001"
capture: config/node.env   ; $(S)/45_capture_window.sh $(A)
```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/README.md`

Keeps the tree listing complete; the numbering slots it between the worker and the local sweep, which is where it sits in the workflow.

**OLD**
```
│   ├── 40_run_rpc_server.sh     the worker
```

**NEW**
```
│   ├── 40_run_rpc_server.sh     the worker
│   ├── 45_capture_window.sh    phase-mark this node's power during a cluster run
```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/docs/10_measurement_protocol.md`

§5 was the only documented path for a cluster window and it was wrong in the way that matters: it named `/tmp/hs_marker`, which no tool in the kit reads. Replacing it with the helper also puts the answer to "who marks, and when" in the protocol, where a reader of the thesis method will look for it, and states the resolution limit instead of leaving it to be discovered.

**OLD**
```
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
python3 telemetry/energy_window.py run.csv --marker prefill --idle-mw 1900
python3 telemetry/energy_window.py run.csv --marker prefill --bench bench.csv \
        --reps 3 --out per_token.csv
```
```

**NEW**
```
For anything the matrix does not cover — a manual `llama-cli` run, or the cluster
experiment driven from the laptop — `scripts/45_capture_window.sh` runs the
logger and owns the marker file. It marks into `$RESULTS_DIR/.marker_$NODE_ID`,
the same path `run_matrix.py` uses, so a capture CSV and a sweep CSV are read by
exactly the same `energy_window.py` commands and `90_collect_results.sh` picks
the capture up without being told about it.

```bash
./scripts/45_capture_window.sh start --tag split
./scripts/45_capture_window.sh mark prefill:split_a:001
<your command>
./scripts/45_capture_window.sh idle
./scripts/45_capture_window.sh stop          # prints the CSV it wrote

python3 telemetry/energy_window.py <csv> --marker prefill --idle-mw 1900
python3 telemetry/energy_window.py <csv> --marker prefill --bench bench.csv \
        --reps 3 --out per_token.csv
```

**In a split run the orchestrator writes the markers, not the Jetson.** This node
is an RPC worker: it is handed tensors, never phases, and it never learns that a
prefill has begun. The orchestrator is the only machine that knows, so it marks
over ssh, around its own command:

```bash
J=jetson-nano-01; C=HeteroServe/Jetson/scripts/45_capture_window.sh
ssh $J $C start --tag split
ssh $J $C mark prefill:split_a:001
llama-cli --rpc $J:50052 -ngl 99 -n 0 -f prompt.txt          # prefill leg only
ssh $J $C idle
ssh $J $C mark decode:split_a:002
llama-cli --rpc $J:50052 -ngl 99 -n 512 -p "..."             # decode leg
ssh $J $C idle
ssh $J $C stop
```

**The bracketed command must do one phase.** The ssh round trip lands the marker
edge within roughly a sample period of the real one, which is fine for a leg that
lasts seconds and useless for the prefill/decode boundary *inside* a single
request: a 512-token prefill on this cluster is well under a second, a handful of
samples at 5 Hz, and no rate this board offers makes that boundary markable from
another machine. So split the legs, exactly as §4 splits the local matrix — `-n 0`
for the prefill leg, a long generation for a decode leg whose prefill is a
rounding error — and recombine in analysis with the same arithmetic §4 relies on,
`P·(J/tok)_prefill + G·(J/tok)_decode`.

Two refusals are built in, because both failures are silent and both have cost a
run before. The capture will not start while another `tegrastats` is alive — the
board permits one reader, and the always-on telemetry unit is the usual culprit;
stop it first (`sudo systemctl stop heteroserve-telemetry`), and note that its
CSV cannot substitute, since it carries no markers by design. And the capture
reports success only once the logger has actually produced samples, so a logger
that died at startup is caught now rather than at analysis time.
```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/docs/10_measurement_protocol.md`

§7 is where the study states what it is not allowed to claim. A per-phase energy figure taken from an interleaved distributed request would be unsupportable, and the honest place to say so is next to the other two deferrals.

**OLD**
```
It also **cannot** compare energy against the Raspberry Pis. That needs one
instrument reading all four nodes identically; the Pis have no on-board power
telemetry and the cluster has no external inline meter. See §2 — deferred, and it
belongs in the limitations section too.
```

**NEW**
```
It also **cannot** compare energy against the Raspberry Pis. That needs one
instrument reading all four nodes identically; the Pis have no on-board power
telemetry and the cluster has no external inline meter. See §2 — deferred, and it
belongs in the limitations section too.

And it **cannot** resolve the prefill/decode boundary inside a single distributed
request. The marker for a remotely driven run is written over ssh by the
orchestrator (§5), which puts the edge within about one sample period, while the
prefill it would be separating is itself shorter than that. Per-phase energy for
this node therefore comes from legs run one phase at a time, the same way the
local matrix gets it — a constraint on the experiment design, not a missing
number, and it belongs in the limitations section as well.
```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/fixtures/run_regression.sh`

Inserted immediately before the existing "regression baseline" section. Drives the whole capture lifecycle against the fake tegrastats and then asserts the thing the finding is actually about: that energy_window.py can select exactly one prefill window and one decode window out of the resulting CSV. The skip branch is there because the one-instance guard is system-wide by design, so a shared build machine with a real tegrastats alive cannot exercise the happy path; on the Jetson and on a clean runner it always runs.

**OLD**
```
head_ "regression baseline"
```

**NEW**
```
head_ "cluster capture: a remotely driven run can be phase-marked"
# Finding #17: only run_matrix.py ever wrote the marker file, so a split run
# driven from the orchestrator left this node's power CSV with no phase labels
# and no per-phase joules -- the deliverable, uncomputable after the fact.
# Runs against a COPY of the kit, because the helper reads config/node.env from
# its own tree and the real one is live machine config.
CAPKIT="$FIX/capkit"
mkdir -p "$CAPKIT" "$FIX/cap/results"
cp -a "$ROOT" "$CAPKIT/Jetson"
cat > "$CAPKIT/Jetson/config/node.env" <<ENV
NODE_ID=captest
HS_ROOT="$FIX/cap"
LLAMA_BIN="$FIX/bin"
MODEL_DIR="$FIX/models"
RESULTS_DIR="$FIX/cap/results"
TELEMETRY_HZ=5
ENV
CAP="$CAPKIT/Jetson/scripts/45_capture_window.sh"

bash "$CAP" mark prefill:x:001 >/dev/null 2>&1 \
  && bad "marked with no capture running" || ok "mark without a capture refused"

if pgrep -x tegrastats >/dev/null; then
  # The guard is deliberately system-wide -- the device has exactly one
  # tegrastats -- so a real one already running makes this path untestable here.
  printf '  \033[33mskip\033[0m the capture path: a tegrastats is already running\n'
else
  bash "$CAP" start --tag split > "$FIX/cap_start.txt" 2>&1 \
    && ok "capture started" || { bad "capture did not start"; cat "$FIX/cap_start.txt"; }
  bash "$CAP" start >/dev/null 2>&1 \
    && bad "a second capture was allowed" || ok "a second capture is refused"
  bash "$CAP" mark prefil:x:001 >/dev/null 2>&1 \
    && bad "a misspelt phase was accepted" || ok "a label energy_window cannot select is refused"
  # The orchestrator marks, over ssh, around a command that does ONE phase.
  bash "$CAP" mark prefill:split_smoke:001 >/dev/null; sleep 2
  bash "$CAP" idle >/dev/null;                          sleep 1
  bash "$CAP" mark decode:split_smoke:002 >/dev/null;   sleep 2
  CAPCSV="$(bash "$CAP" path)"
  bash "$CAP" stop >/dev/null
  for ph in prefill decode; do
    n=$(python3 "$ROOT/telemetry/energy_window.py" "$CAPCSV" --marker "$ph" 2>/dev/null \
        | grep -c "^window ")
    [ "$n" = "1" ] && ok "one $ph window, selectable by --marker $ph" \
                   || bad "--marker $ph selected $n windows, expected 1"
  done
  # No sample may carry an empty marker between two labels: energy_window counts
  # a blank as idle and refuses to join a window that contains one, so a torn
  # read of the marker file would cost the run its per-token number.
  python3 - "$CAPCSV" <<'EOPY' && ok "no torn marker reads" || bad "a sample carries a blank marker"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
assert rows, "no samples"
assert all((r["marker"] or "").strip() for r in rows), \
    [i for i, r in enumerate(rows) if not (r["marker"] or "").strip()]
EOPY
  # A logger that dies at startup must not be reported as a running capture.
  # tegrastats permits one instance per boot and its refusal goes to stderr,
  # which the logger parses into a plausible-looking first row.
  cat > "$FIX/bin/tegrastats" <<'SH'
#!/bin/sh
echo "tegrastats is already running" >&2
exit 1
SH
  chmod +x "$FIX/bin/tegrastats"
  bash "$CAP" start >/dev/null 2>&1 \
    && bad "a dead logger was reported as a running capture" \
    || ok "a logger that produces no samples is reported, not announced as ok"
  bash "$CAP" status 2>/dev/null | grep -q "no capture running" \
    && ok "and no stale capture state is left behind" || bad "stale capture state"
  cp bench/fixtures/fake_tegrastats.py "$FIX/bin/tegrastats"; chmod +x "$FIX/bin/tegrastats"
fi

head_ "regression baseline"
```

---

## Finding #21 — partly-fixed

After the baseline sleep, assert the logger is alive AND that the CSV really contains power, and abort with a sentence naming the likely cause instead of burning two hours. Print the ACHIEVED sample rate, not the requested one, so the operator sees what actually bounds the integral before committing. Take the logger down before exiting so the abort does not itself leave the stray tegrastats behind. Also warn if the logger dies mid-sweep (per_token_status already marks the affected rows, but a half-instrumented sweep reads like a whole one), and stop printing `idle floor: 0 mW` as though it were a measurement.

**Justification (against current code):** The archived artefact no longer lies — that half was closed incidentally by the finding #3 work. `PT_NO_POWER = "no_power_window"` exists, the join sets it inside the power-CSV guard (`if not energy_j and rec.get("per_token_status") == PT_OK: rec["per_token_status"] = PT_NO_POWER`), and a second pass outside that guard catches the --no-power and no-tegrastats cases with the comment "`not joules` deliberately catches 0.0 as well as absent". The end-of-run summary prints the tally per status.

The expensive half is untouched. bench/run_matrix.py:652-663 is verbatim as the audit found it:

    logger = subprocess.Popen(
        [sys.executable, os.path.join(JETSON_DIR, "telemetry", "tegrastats_logger.py"), ...])
    # Let the logger establish an idle baseline before the first case; the
    # idle floor is subtracted from every active-energy figure later.
    print("collecting 30 s idle baseline...")
    time.sleep(30)

No `logger.poll()`, no row count, and the join gate is still `if os.path.exists(power_csv):`. And `idle_floor()` still ends `return sum(vals) / len(vals) if vals else 0.0`, which the driver prints unconditionally as `print("\nidle floor: %.0f mW" % idle_mw)` — so a file with no power still produces the reassuring line `idle floor: 0 mW`.

**Reproduction:** Fixture tegrastats that behaves like a second instance (`echo "tegrastats being started or stopped by another process, please retry"; exit 1`), one prefill case, driver run in a copy of the kit:

  BEFORE
    collecting 30 s idle baseline...
    [1/1] pre|qwen2.5-1.5b-instruct-q4_k_m|ngl0|t4|p512|g0
        pp512       1024.00 t/s   2048 tok   5.0 s
    idle floor: 0 mW
    1/1 rows ok (1 invocations)
      per_token_status no_power_window            1
    RC=0
  The power CSV holds exactly one data row whose only populated field is
  raw="tegrastats being started or stopped by another process, please retry".
  The row says no_power_window — so the audit's "nothing in the run prints a
  warning" is now half wrong — but the sweep still ran to completion and exited 0.

  AFTER
    collecting 30 s idle baseline...
    the telemetry logger exited (rc=0) during the idle baseline.
    tegrastats allows only ONE instance: look for another with `pgrep -a tegrastats`, and stop the service if it is that --
      sudo systemctl stop heteroserve-telemetry
    Refusing to start the sweep: every row would carry empty energy.
    RC=1   (30 s in; no joined CSV written, no case run)

  And the case the poll() check alone would miss — a logger that is alive but whose
  output this parser cannot read:
    the telemetry logger is running but only 0 of the expected ~150 baseline samples carry POM_5V_IN.
    ... ls /sys/bus/i2c/drivers/ina3221x/   (expect a 6-0040 node) ...
    RC=1

  Healthy path, unchanged plus one line:
    telemetry : 150 samples in 29.8 s = 4.99 Hz achieved (5 Hz requested)
    idle floor: 1900 mW
    1/1 rows ok (1 invocations)
      per_token_status ok                         1

One defect in the audit's own proposed fix, found by testing it: a bare `sys.exit()` here orphans the logger and the tegrastats it owns. That is precisely the one-instance blocker the error message tells the operator to go and clear, so the abort would create the problem for the next attempt. Confirmed by pgrep after a bare-exit run (`67098 ... tegrastats_logger.py --out ..._ml_power.csv` still alive); `refuse()` takes it down first, and pgrep is clean after.

**Test:** Fully testable off-hardware — the fixture already fakes tegrastats, so faking a *failing* tegrastats is one more shell script. The block above was run against a patched copy:
  ok   a logger that exited during the baseline aborts, non-zero
  ok   and not one case was run
  ok   a logger producing no power is caught too
  ok   and it took its logger down with it
(the orphan check needs the `telemetry/tegrastats_logger[.]py --out .*mutelogger` pattern, not a bare `mutelogger`: the looser pattern matches the harness's own command line and false-positives.)

Full suite on the patched copy: **51 passed, 0 failed**, including `config/node.env unchanged (348ebc26c5bb48abf26e23ddfe56527d)`.

What is NOT testable off-hardware: that a real second tegrastats on L4T R32.7 exits rather than hangs. The fix does not depend on it — `require_telemetry` catches the exit case via poll() and the hang-with-no-output case via the row count, which between them cover both behaviours.

**Conflicts:** `telemetry_hz()`'s fallback of "10" belongs to finding #29. If #29 is not taken, change it to "5" and drop the second line of its docstring; nothing else in this edit depends on the rate. | `require_telemetry` deliberately does NOT call `_rows()`. It runs while the logger is still writing, and #29 memoises `_rows()` for the join — routing the pre-flight through it would cache a 30 s snapshot and every row's energy would then be computed from 150 samples. The comment in the code says so; do not 'simplify' it later. | Overlaps finding #22 (idle_floor pools cold baseline with hot cooldowns) at the `idle floor:` print. My edit only changes what is printed when the value is 0.0; #22's fix changes how the value is computed. They compose, but #22 should keep the zero-guard. | Does not touch the join or per_token_status, so no conflict with the finding #3 work. no_power_window remains the right label for the rows that a mid-sweep death strands. | Unrelated but worth knowing before anyone blames these edits: the existing `the driver and the analysis tool agree` test has a 3% accuracy tolerance that is load-sensitive off-hardware. On a contended machine it failed at 3.74%; on a quiet one the patched copy scored 1.45% worst against a pristine copy's 1.47%. Consider loosening it to 5%, or reporting sample_gap_max_s on failure (finding #29 adds that column).

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py`

The 30 was a magic number in one place; it is now in two, and the second one is arithmetic that must agree with the first.

**OLD**
```
HERE = os.path.dirname(os.path.abspath(__file__))
JETSON_DIR = os.path.dirname(HERE)

```

**NEW**
```
HERE = os.path.dirname(os.path.abspath(__file__))
JETSON_DIR = os.path.dirname(HERE)

# Seconds of idle collected before the first case. Named because the pre-flight
# telemetry check has to know it too, to work out how many samples should be on
# disk by the time the sleep is over.
BASELINE_S = 30

```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py`

Two checks rather than one, because poll() alone misses the alive-but-mute case (a tegrastats whose line shape this parser does not recognise, or an unreadable INA3221) and that is the same empty-energy sweep. Anchored after idle_floor() rather than before _rows() so it does not collide with finding #29's memoisation of _rows, and it deliberately does not call _rows for the same reason.

**OLD**
```
    return sum(vals) / len(vals) if vals else 0.0


def _num(row, k):
```

**NEW**
```
    return sum(vals) / len(vals) if vals else 0.0


def telemetry_hz(env):
    """Sample rate for the logger, from node.env; 10 Hz when the key is absent,
    matching config/node.env.example.

    Parsed rather than passed through as a string because the driver now has to
    reason about it: the pre-flight check below turns it into "how many samples
    should be on disk by now", and a typo that used to reach the logger verbatim
    and be rejected there would otherwise abort the sweep 30 s in with a
    traceback instead of a sentence.
    """
    raw = str(env.get("TELEMETRY_HZ", "") or "10").strip()
    try:
        hz = float(raw)
    except ValueError:
        sys.exit("TELEMETRY_HZ=%r is not a number" % raw)
    if hz <= 0:
        sys.exit("TELEMETRY_HZ=%r must be positive" % raw)
    return hz


def require_telemetry(logger, power_csv, hz, baseline_s):
    """Refuse to start the sweep unless the logger is alive AND the power CSV is
    really filling up with power.

    tegrastats permits exactly one instance per boot. If heteroserve-telemetry.
    service is enabled, or an earlier sweep left one behind, the second one
    prints its complaint and exits immediately -- and tegrastats_logger.py
    writes the CSV header before it reads its first line, so what is left on
    disk is a file that exists, parses, and contains no power whatsoever. The
    old code only slept 30 s and pressed on. The sweep then ran its full two
    hours, printed a cheerful "idle floor: 0 mW" and "32/32 rows ok", and every
    row came out with empty energy columns. The only cure was to run it again.

    The check is free: the baseline sleep has just finished, so at any sane rate
    there are dozens of samples on disk already. Costing one CSV read to save a
    two-hour re-run is not a trade that needs thinking about.
    """
    def refuse(msg):
        """Take the logger down before leaving.

        A bare sys.exit() here would orphan it, and with it the tegrastats it
        owns -- which is precisely the condition half these messages tell the
        operator to go and clear. Aborting the sweep must not be what creates
        the blocker for the next attempt.
        """
        if logger.poll() is None:
            logger.terminate()
            try:
                logger.wait(timeout=10)
            except Exception:
                logger.kill()
        sys.exit(msg)

    rc = logger.poll()
    if rc is not None:
        refuse(
            "the telemetry logger exited (rc=%s) during the idle baseline.\n"
            "tegrastats allows only ONE instance: look for another with "
            "`pgrep -a tegrastats`, and stop the service if it is that --\n"
            "  sudo systemctl stop heteroserve-telemetry\n"
            "Refusing to start the sweep: every row would carry empty energy."
            % rc)

    # Read the file directly rather than through _rows(): this runs while the
    # logger is still writing, and _rows() memoises for the join, which must see
    # the finished file and not this 30 s snapshot of it.
    try:
        with open(power_csv, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, csv.Error) as exc:
        refuse("cannot read the power CSV the logger should be writing "
               "(%s): %s" % (power_csv, exc))
    live = [r for r in rows if (r.get("pow_in_mw") or "").strip()]

    # A tenth of nominal. The point is to separate "nothing at all" from
    # "sampling", not to police the rate -- a busy board legitimately drops the
    # odd sample, and the achieved rate is reported below for the operator to
    # judge rather than enforced here.
    want = max(2, int(hz * baseline_s / 10.0))
    if len(live) < want:
        refuse(
            "the telemetry logger is running but only %d of the expected ~%d "
            "baseline samples carry POM_5V_IN.\n"
            "Either tegrastats is printing a line this parser does not "
            "recognise, or the INA3221 is not readable -- check\n"
            "  ls /sys/bus/i2c/drivers/ina3221x/   (expect a 6-0040 node)\n"
            "and the raw column of %s.\n"
            "Refusing to start the sweep: energy would be missing or wrong."
            % (len(live), int(hz * baseline_s), power_csv))

    span = 0.0
    try:
        span = float(live[-1]["ts_unix"]) - float(live[0]["ts_unix"])
    except (KeyError, TypeError, ValueError):
        pass
    achieved = (len(live) - 1) / span if span > 0 else 0.0
    # Print the achieved rate, not the requested one. They are not the same
    # number on a 4-core Tegra X1 -- the logger competes with llama-bench for
    # those cores -- and the achieved rate is what bounds the integral, so the
    # operator should see it before committing two hours to it.
    print("telemetry : %d samples in %.1f s = %.2f Hz achieved (%g Hz requested)"
          % (len(live), span, achieved, hz))
    if achieved < hz * 0.8:
        print("!! the logger is sampling well below TELEMETRY_HZ; window energy "
              "will be coarser than the config claims")


def _num(row, k):
```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py`

The sleep already exists and already costs 30 s; the check rides on it for free. Placed after the sleep rather than before it so the row-count half has something to count.

**OLD**
```
    logger = None
    if not args.no_power and shutil.which("tegrastats"):
        logger = subprocess.Popen(
            [sys.executable, os.path.join(JETSON_DIR, "telemetry", "tegrastats_logger.py"),
             "--out", power_csv, "--hz", str(env.get("TELEMETRY_HZ", "5")),
             "--node-id", node_id, "--marker-file", marker_file])
        # Let the logger establish an idle baseline before the first case; the
        # idle floor is subtracted from every active-energy figure later.
        print("collecting 30 s idle baseline...")
        time.sleep(30)
    else:
        print("!! no power logging (tegrastats missing or --no-power)")

```

**NEW**
```
    logger = None
    if not args.no_power and shutil.which("tegrastats"):
        hz = telemetry_hz(env)
        logger = subprocess.Popen(
            [sys.executable, os.path.join(JETSON_DIR, "telemetry", "tegrastats_logger.py"),
             "--out", power_csv, "--hz", "%g" % hz,
             "--node-id", node_id, "--marker-file", marker_file])
        # Let the logger establish an idle baseline before the first case; the
        # idle floor is subtracted from every active-energy figure later.
        print("collecting 30 s idle baseline...")
        time.sleep(BASELINE_S)
        require_telemetry(logger, power_csv, hz, BASELINE_S)
    else:
        print("!! no power logging (tegrastats missing or --no-power)")

```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py`

The pre-flight only proves the logger was alive at t=0. A partial sweep is the harder case to notice, because most rows still say ok and the tally at the bottom looks almost right.

**OLD**
```
    finally:
        set_marker("idle")
        if logger:
            time.sleep(3)
            logger.terminate()
```

**NEW**
```
    finally:
        set_marker("idle")
        if logger:
            # The logger was alive when the sweep started -- require_telemetry()
            # made sure of that -- so if it is gone now it died somewhere in the
            # middle and the rows after that point have no power behind them.
            # Say which, loudly: the per-row per_token_status will say
            # no_power_window, but only for the rows that lost their window, and
            # a half-instrumented sweep is easy to mistake for a whole one.
            if logger.poll() is not None:
                print("\n!! the telemetry logger EXITED DURING THE SWEEP (rc=%s). "
                      "Energy is missing from every case after it died; check "
                      "per_token_status in the joined CSV before using any of it."
                      % logger.poll())
            time.sleep(3)
            logger.terminate()
```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py`

The audit singles this line out ("printing the reassuring-looking `idle floor: 0 mW`"). It stays reachable for --no-power-adjacent cases the pre-flight cannot catch, so it has to stop reading as a result.

**OLD**
```
        print("\nidle floor: %.0f mW" % idle_mw)
```

**NEW**
```
        if idle_mw:
            print("\nidle floor: %.0f mW" % idle_mw)
        else:
            # "idle floor: 0 mW" reads like a measurement. It is not one: it is
            # what idle_floor() returns when it found no POM_5V_IN at all.
            print("\n!! idle floor: NO idle samples carried power -- "
                  "active_joules is meaningless on every row below")
```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/fixtures/run_regression.sh`

Same shape as the existing driver-level blocks. Reuses $FIX/one_prefill.json from the rogue-ref block and reads the achieved-rate line off $FIX/split.txt, which an earlier block already produced, so only two extra driver runs (2 x 30 s) are added. Anchored at the text-I/O block so it does not collide with finding #29's insertion point.

**OLD**
```
head_ "text I/O does not depend on the locale"

```

**NEW**
```
head_ "driver: a telemetry logger that never samples stops the sweep at 30 s"
rm -f "$FIX"/results/*
# tegrastats permits one instance. A second one -- heteroserve-telemetry.service,
# or a stray from an earlier sweep -- prints its complaint and exits, and the
# logger has already written the CSV header, so the file exists and is empty of
# power. That used to run the whole two-hour sweep and report it as ok.
cat > "$FIX/bin/tegrastats" <<'SH'
#!/bin/sh
echo "tegrastats being started or stopped by another process, please retry"
exit 1
SH
chmod +x "$FIX/bin/tegrastats"
python3 bench/run_matrix.py --env-file "$FIX/node.env" --matrix "$FIX/one_prefill.json" \
  --tag deadlogger > "$FIX/dead.txt" 2>&1
if [ $? -ne 0 ] && grep -q "Refusing to start the sweep" "$FIX/dead.txt"; then
  ok "a logger that exited during the baseline aborts, non-zero"
else
  bad "a dead logger did not stop the sweep"; tail -5 "$FIX/dead.txt"
fi
ls "$FIX"/results/*deadlogger_bench_energy.csv > /dev/null 2>&1 \
  && bad "the sweep ran anyway" || ok "and not one case was run"

# alive, but printing a shape this parser cannot read: same refusal, because the
# symptom that matters is an empty energy column, not a dead process
cat > "$FIX/bin/tegrastats" <<'SH'
#!/bin/sh
while :; do echo "RAM 100/3956MB and nothing this parser knows"; sleep 0.2; done
SH
chmod +x "$FIX/bin/tegrastats"
python3 bench/run_matrix.py --env-file "$FIX/node.env" --matrix "$FIX/one_prefill.json" \
  --tag mutelogger > "$FIX/mute.txt" 2>&1
if [ $? -ne 0 ] && grep -q "carry POM_5V_IN" "$FIX/mute.txt"; then
  ok "a logger producing no power is caught too"
else
  bad "a logger producing no power was accepted"; tail -5 "$FIX/mute.txt"
fi
# and the refusal must not leave the tegrastats it started behind -- that would
# BE the one-instance blocker the message tells the operator to go and clear
pgrep -f "telemetry/tegrastats_logger[.]py --out .*mutelogger" > /dev/null 2>&1 \
  && bad "the aborted run orphaned its logger" || ok "and it took its logger down with it"
cp bench/fixtures/fake_tegrastats.py "$FIX/bin/tegrastats"; chmod +x "$FIX/bin/tegrastats"

# ...and a healthy sweep says what rate it ACHIEVED, not what it asked for
grep -qE "telemetry : [0-9]+ samples in .* Hz achieved" "$FIX/split.txt" \
  && ok "achieved sample rate is reported before the sweep commits" \
  || bad "achieved rate not reported"

head_ "text I/O does not depend on the locale"

```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/fixtures/run_regression.sh`

The header states the budget; two more driver runs change it, and saying why they are worth 60 s stops someone deleting them to get the suite back under three minutes.

**OLD**
```
# Needs no Jetson. Budget ~3 min: each driver run sleeps 30 s collecting an idle
# baseline.

```

**NEW**
```
# Needs no Jetson. Budget ~5 min: each driver run sleeps 30 s collecting an idle
# baseline, and two of them exist only to prove that a telemetry logger which is
# not sampling aborts at the end of that sleep instead of two hours later.

```

---

## Finding #22 — still-live

Settle on one definition and make all three places say it: the idle floor is LOCAL and PER CASE — the mean board power over the idle interval that ended just before that case opened its window, with the first 3 s dropped for clock ramp-down. That measures the board in the thermal state the case actually ran in, and for the first case that interval IS the 30 s pre-sweep baseline, so there is one definition rather than two. The pooled mean and the baseline are still written to every row (`idle_mw_pooled`, `idle_mw_baseline`) so the subtraction is auditable from the archived CSV and an older figure stays reproducible; `idle_mw_source` names which interval was used. The node facts number keeps its own, different job — cold-board device-table figure — and the facts script now says so in the JSON itself (`idle_board_mw_basis`) and samples 15 s with the first 3 s dropped instead of three instantaneous readings. telemetry/energy_window.py gets the same definition as its `--idle-mw auto` default (twin functions, kept in step), so the tool the protocol names for re-deriving a published figure agrees with the driver. The `max(..., 0.0)` clamps are dropped in both tools: a negative active energy means the floor is wrong and that must be visible. And precache() is moved under its own `precache:<case>:<NNN>` marker so "idle" means idle — without that, the local floor would be read off a multi-watt SD read.

**Justification (against current code):** All three definitions are still in the tree, unchanged and still mutually inconsistent.

(1) bench/run_matrix.py, the subtrahend actually used: `def idle_floor(power_csv): """Mean module power over the samples labelled idle..."""` whose body is `for r in _rows(power_csv): if (r.get("marker") or "").strip() == "idle" and r.get("pow_in_mw"): vals.append(float(r["pow_in_mw"])) ... return sum(vals) / len(vals) if vals else 0.0` — every idle-marked row in the whole file, baseline and cooldowns pooled. It is called once, outside the per-record loop (`idle_mw = idle_floor(power_csv)` / `print("\nidle floor: %.0f mW" % idle_mw)`), so one scalar is subtracted from all 32 rows: `rec["active_joules"] = (round(max(energy_j - idle_mw / 1000.0 * (window_s or 0.0), 0.0), 4) if energy_j else None)`.

(2) docs/10_measurement_protocol.md still says something else: "`active_joules` — net of the measured idle floor (`idle_mw`, taken from the 30 s idle baseline the driver collects before the sweep)". That is not what the code does.

(3) telemetry/energy_window.py's docstring still says a third thing: "--idle-mw N   subtract an N-milliwatt floor... Take N from the node's facts JSON (power.idle_board_mw)." and scripts/60_collect_node_facts.sh derives that from `timeout 6 tegrastats --interval 1000 ... | tail -3 | awk '{s+=$1; n++} END{...}'` — three instantaneous samples off a cold board, under a comment that asserts the opposite of the code: "tegrastats one-shot for the idle baseline; energy numbers are always reported net of this floor."

The clamp is also exactly as reported, in three places: `round(max(energy_j - idle_mw/1000.0*(window_s or 0.0), 0.0), 4)`, `rec["active_joules"] = round(max(act, 0.0), 4)`, and in energy_window.py `res["module_active_joules"] = round(max(active, 0.0), 4)` plus `ce["module_active_joules"] = round(max(... , 0.0), 4)`.

One thing the audit did not notice, and which decides the shape of the fix: the GGUF pre-read runs *under the idle marker*. `cached = False if args.no_precache else precache(mpath)` sits after the cooldown `time.sleep(run["cooldown_s"])` and before `t_start = time.time(); set_marker(label)`, while the marker still reads "idle" from the previous iteration's `finally: set_marker("idle")`. precache() reads the whole file (`while fh.read(8 << 20)`), which on the board is 400 MB–1.1 GB off the SD card — seconds of multi-watt work stamped `marker=idle`, sitting at the tail of every cooldown. So the pooled floor is inflated by load bursts as well as by heat, and any naive "take the samples just before the window" floor would read the floor off an SD read. That has to be fixed together with the floor, or the local floor is worse than the pooled one.

**Reproduction:** Ran the current `idle_floor()` (from a copy at /tmp/tmp.P8hF0Tv0ZT/kit) against a synthetic power CSV shaped like a real sweep — 30 s cold baseline at 1600 mW, then ten 20 s cases each followed by a 20 s cooldown whose idle power climbs 50 mW per case:

  pooled idle_floor()          = 1839.1 mW
  cold 30 s pre-sweep baseline = 1600.0 mW   (what docs/10 says is used)
  node-facts cold-board figure = 1600.0 mW   (what energy_window.py says to use)
  true local floor, case 1     = 1600.0 mW ; case 10 = 2050.0 mW

Then end to end through the real driver, with the fixture's tegrastats taught to idle hotter once the board has done work (HS_FAKE_IDLE_HOT_MW=2400) — two prefill cases doing identical work at identical power, 8 s cooldown:

  marker       idle_mw  idle_mw_source  compute_J  compute_s  active_J  active_J with the POOLED floor
  prefill:001  1900.0   baseline           14.579      2.000    10.779   10.391
  prefill:002  2400.0   cooldown[1]        14.586      2.000     9.786   10.399

  -> active J/token: case 1 pooled floor gives -3.6% against the local floor, case 2 gives +6.3%.

A 10-point spread, in opposite directions, between two cases that did the same work — manufactured entirely by the subtrahend, and aligned with case order. pooled = 2093.9 mW, baseline = 1900.0 mW on that run.

**Test:** Off-hardware, in the fixture's own idiom. A new block goes into bench/fixtures/run_regression.sh immediately before `head_ "driver: a failed run still produces exactly one row"`. It runs the real driver against a two-case matrix with HS_FAKE_IDLE_HOT_MW=2400 and cooldown_s=8, then asserts on the joined CSV:

  head_ "driver: the idle floor is measured per case, not pooled"
  ...
  HS_FAKE_IDLE_HOT_MW=2400 python3 bench/run_matrix.py --env-file "$FIX/node.env" \
    --matrix "$FIX/two_cases.json" --tag drift > "$FIX/drift.txt" 2>&1
  -> a["idle_mw_source"] == "baseline" and b["idle_mw_source"] == "cooldown[1]"
  -> a["idle_mw"] ~ 1900, b["idle_mw"] ~ 2400 (the two cases did IDENTICAL work)
  -> idle_mw_pooled is identical on both rows and strictly between them
  -> active_joules == compute_joules - idle_mw/1000 * compute_s on every row
  -> the power CSV carries precache:<case>:<NNN> samples and no idle-marked
     sample above the hot idle level (plus a structural check on run_matrix.py
     that precache() runs under that marker and the marker is not reset to idle
     before t_start -- the 8 MB fixture model reads too fast for 5 Hz to catch)
  -> `energy_window.py <power.csv> --marker <marker> --json` with no --idle-mw
     returns the SAME idle_mw and idle_mw_source as the driver wrote, for both
     rows (this is the three-definitions check: the two tools must not disagree)
  -> summarise(win, 2000.0) on a 10 J / 10 s window returns -10.0, not 0.0

The exact block is written out at /tmp/claude-1000/-mnt-data-KMITL-Final-HeteroServe-Jetson/4496d19d-01c0-43d0-b10a-2f74f03bec05/scratchpad/final_block.txt, and a fully patched tree is at /tmp/tmp.P8hF0Tv0ZT/kit. Result of the full suite on that tree: **58 passed, 0 failed** (53 before these five new assertions), and `config/node.env unchanged` still holds. Note the copy was taken at 02:06 today and already contained another agent's additions, so its pre-change baseline is 53, not the 51 in the brief.

What cannot be tested off-hardware: the actual magnitude of the drift. 2400 mW is a stand-in; only a real sweep can say whether the Nano's cooldown floor sits 200 mW or 500 mW above cold. That is a number the sweep will now measure and report (idle_mw against idle_mw_baseline on every row) rather than one anybody has to assume.

**Conflicts:** Touches the same join loop as the finding #3 work and the compute-trim work, but only the idle terms. The compute trim, `compute_anchor_unix`, the interpolation, `energy_basis`, `per_token_status` and the one-record-per-bench-row shape are all untouched, and bench/check_finding3_invariants.py passes unchanged (its rule 8 identity checks joules_per_token, which has no idle term). | The precache marker change makes a new non-idle marker value appear in the power CSV. It is safe against everything currently reading markers -- marker_bounds() is keyed by the run's own label and only skips "idle"; energy_window.py's --marker matching is prefix-based on the run label; its idle-contamination refusal tests for markers in ("", "idle"), and precache samples are correctly not idle. But any FUTURE work that assumes "every non-idle marker in the file corresponds to a joined record" -- e.g. finding #17's phase-marked split-run capture, or finding #21's logger liveness check -- must know about it. | `--idle-mw` changing from a float to a string spec is an interface change to a documented tool. Numeric values behave exactly as before (the regression's existing `--idle-mw 1900` and `--idle-mw "$r[idle_mw]"` calls both still pass, including the bit-identical driver/tool agreement test), but anything outside this kit that passes an empty string would now mean 'none' rather than 0.0 -- which is the same thing. | Removing the max(...,0.0) clamps means a wrong floor now produces negative numbers in a published column. That is the intent, but it needs the docs sentence shipped with it (included above) or somebody will 'fix' the negative by restoring the clamp. | Finding #29 (5 Hz sampling / INA integral validation) shares the settle-window argument: if TELEMETRY_HZ is ever lowered below ~1 Hz, IDLE_SETTLE_S=3.0 leaves too few samples in a 20 s cooldown and rows will start falling back to 'baseline (no preceding idle)'. The fallback is explicit on the row, but the two constants are coupled. | scripts/60_collect_node_facts.sh was edited by another agent at 01:20 today (it already carries the whole-devkit rail_scope and external_meter: False from the #15/#11 work). My edit keeps both and only touches the idle block; re-check the anchor text before applying if that file moves again.

### edit — `bench/run_matrix.py`

Replaces the one pooled definition with the three quantities the kit actually needs, each named for what it is. The pooled figure survives only as a reproducibility column.

**OLD**
```
def idle_floor(power_csv):
    """Mean module power over the samples labelled idle. This is the floor the
    board burns regardless of the workload, and on a 10 W budget it is most of
    a short decode window -- so it is reported alongside, never instead of,
    total energy."""
    vals = []
    for r in _rows(power_csv):
        if (r.get("marker") or "").strip() == "idle" and r.get("pow_in_mw"):
            try:
                vals.append(float(r["pow_in_mw"]))
            except ValueError:
                pass
    return sum(vals) / len(vals) if vals else 0.0
```

**NEW**
```
# Seconds to drop from the HEAD of an idle interval before it counts as idle.
# The board does not become idle the instant the marker says so: the governor
# has to walk the CPU and GPU rails back down, and the page cache is still
# writing out. Three seconds is comfortably longer than the ramp at the 5 Hz
# the protocol specifies, and still leaves 17 of the 20 s cooldown to average.
IDLE_SETTLE_S = 3.0


def idle_intervals(power_csv):
    """Every CONTIGUOUS run of idle-marked samples, in file order, as a list of
    lists of rows.

    The driver writes the same "idle" marker for the 30 s pre-sweep baseline and
    for every inter-case cooldown, so the file cannot be split on the label --
    it has to be split on contiguity. Interval 0 is the cold baseline; interval
    k is the cooldown that ended just before case k+1 started.
    """
    out, cur = [], []
    for r in _rows(power_csv):
        if (r.get("marker") or "").strip() == "idle":
            cur.append(r)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def interval_mean_mw(rows, settle_s=IDLE_SETTLE_S):
    """Mean module power over one idle interval, less its first `settle_s`
    seconds. None when nothing is left to average -- a caller must then say so
    rather than substitute a number from somewhere else in the file."""
    ts = [_num(r, "ts_unix") for r in rows]
    ts = [t for t in ts if t is not None]
    if not ts:
        return None
    cut = ts[0] + settle_s
    vals = []
    for r in rows:
        t, mw = _num(r, "ts_unix"), _num(r, "pow_in_mw")
        if t is None or mw is None or t < cut:
            continue
        vals.append(mw)
    return sum(vals) / len(vals) if vals else None


def idle_floor_at(intervals, t_start, settle_s=IDLE_SETTLE_S):
    """The idle floor for the case that started at `t_start`, and where it came
    from: the last idle interval that ENDED before the case opened its window.

    The floor is a local quantity, not a constant of the board. SoC leakage
    rises with junction temperature, so on a fanless Nano an hour into a sweep
    the board idles several hundred milliwatts above where it idled cold. A
    single scalar subtracted from every case therefore over-subtracts from the
    early cases and under-subtracts from the late ones -- a bias aligned with
    exactly the case-order axis the sweep varies, and the axis a reader cannot
    separate from the axis under study. Taking the floor from the cooldown that
    just ended measures the board in the thermal state the case actually ran in,
    and needs no assumption that the board is stationary over three hours.

    For the first case the answer is the pre-sweep baseline, because that is the
    idle interval immediately preceding it: one definition, not two.

    Returns (mw, source) where source names the interval, or (None, "none").
    """
    best = None
    for k, rows in enumerate(intervals):
        ts = [_num(r, "ts_unix") for r in rows]
        ts = [t for t in ts if t is not None]
        if not ts or ts[-1] > t_start:
            continue
        best = (k, rows)
    if best is None:
        return None, "none"
    k, rows = best
    mw = interval_mean_mw(rows, settle_s)
    if mw is None:
        return None, "none"
    return mw, ("baseline" if k == 0 else "cooldown[%d]" % k)


def pooled_idle_mw(power_csv):
    """Mean module power over EVERY idle-marked sample in the file.

    This is NOT the subtrahend -- see idle_floor_at() for why a pooled figure
    biases along case order. It is written to every row as `idle_mw_pooled` so
    that a figure published before the per-case floor existed can still be
    reproduced from the archived CSV, and so the spread between the two is
    visible rather than a matter of which tool the reader happened to run.
    """
    vals = []
    for r in _rows(power_csv):
        if (r.get("marker") or "").strip() == "idle" and r.get("pow_in_mw"):
            try:
                vals.append(float(r["pow_in_mw"]))
            except ValueError:
                pass
    return sum(vals) / len(vals) if vals else None
```

### edit — `bench/run_matrix.py`

Stops hoisting one scalar out of the loop, and makes the operator see on stdout that the subtracted floor is per row. `idle_mw` is not used outside the loop, so the initialiser goes.

**OLD**
```
    idle_mw = 0.0
    if os.path.exists(power_csv):
        idle_mw = idle_floor(power_csv)
        bounds = marker_bounds(power_csv)
        print("\nidle floor: %.0f mW" % idle_mw)

```

**NEW**
```
    if os.path.exists(power_csv):
        # Three numbers, because they are three different quantities and the
        # kit used to conflate them. `intervals` gives each case its own local
        # floor; `baseline` is the cold pre-sweep one; `pooled` is the old
        # whole-file mean, kept only so an older figure stays reproducible.
        intervals = idle_intervals(power_csv)
        pooled_mw = pooled_idle_mw(power_csv)
        baseline_mw = interval_mean_mw(intervals[0]) if intervals else None
        bounds = marker_bounds(power_csv)
        print("\nidle floor: baseline %s mW, pooled %s mW -- the floor actually "
              "subtracted is per case, on each row as idle_mw/idle_mw_source"
              % ("%.0f" % baseline_mw if baseline_mw is not None else "-",
                 "%.0f" % pooled_mw if pooled_mw is not None else "-"))

```

### edit — `bench/run_matrix.py`

Resolves the floor per record, before anything is subtracted, and never silently substitutes one definition for another.

**OLD**
```
            e = energy_between(power_csv, rec["t_start_unix"], rec["t_end_unix"])
            rec.update(e)

```

**NEW**
```
            # The floor this case ran against, measured from the cooldown that
            # ended just before it -- see idle_floor_at(). Falling back to the
            # baseline and then to the pooled mean keeps a row producible when
            # the sweep was interrupted, but the row says which it got, because
            # the three are not interchangeable.
            idle_mw, idle_src = idle_floor_at(intervals, rec["t_start_unix"])
            if idle_mw is None and baseline_mw is not None:
                idle_mw, idle_src = baseline_mw, "baseline (no preceding idle)"
            if idle_mw is None and pooled_mw is not None:
                idle_mw, idle_src = pooled_mw, "pooled (no preceding idle)"
            if idle_mw is None:
                idle_mw, idle_src = 0.0, "none"

            e = energy_between(power_csv, rec["t_start_unix"], rec["t_end_unix"])
            rec.update(e)

```

### edit — `bench/run_matrix.py`

Drops the clamp the finding calls out. A negative active energy is diagnostic of a wrong floor; a clamped 0.0 is indistinguishable from a correct result.

**OLD**
```
            rec["active_joules"] = (round(max(energy_j - idle_mw / 1000.0 * (window_s or 0.0), 0.0), 4)

```

**NEW**
```
            # No clamp at zero. An active energy that comes out negative means
            # the floor being subtracted is larger than the power the case drew,
            # i.e. the floor is wrong -- a stale baseline, a sweep whose
            # cooldowns never went idle, a board that was doing something else.
            # Clamping published a plausible 0.0 and destroyed the only evidence
            # of that; the negative number is the signal.
            rec["active_joules"] = (round(energy_j - idle_mw / 1000.0 * (window_s or 0.0), 4)

```

### edit — `bench/run_matrix.py`

Same clamp on the fused -pg path. Leaving it would also break the identity active_joules_per_request x requests == active_joules that the fixture already asserts, once one of the two is clamped and the other is not.

**OLD**
```
                        rec["active_joules"] = round(max(act, 0.0), 4)
                        rec["active_joules_per_request"] = round(max(act, 0.0) / reqs, 4)

```

**NEW**
```
                        # Unclamped, for the reason given above.
                        rec["active_joules"] = round(act, 4)
                        rec["active_joules_per_request"] = round(act / reqs, 4)

```

### edit — `bench/run_matrix.py`

Makes the subtrahend auditable from the archived deliverable, and makes the thermal drift (idle_mw against idle_mw_baseline) a measurement rather than an inference.

**OLD**
```
            rec["idle_mw"] = round(idle_mw, 1)

```

**NEW**
```
            # All three floors on every row, so the subtraction is auditable
            # from the archived CSV alone and nobody has to know which tool
            # produced the number. idle_mw is the one that was subtracted.
            rec["idle_mw"] = round(idle_mw, 1)
            rec["idle_mw_source"] = idle_src
            if baseline_mw is not None:
                rec["idle_mw_baseline"] = round(baseline_mw, 1)
            if pooled_mw is not None:
                rec["idle_mw_pooled"] = round(pooled_mw, 1)

```

### edit — `bench/run_matrix.py`

Puts the floor columns next to the energy they were subtracted from, in the fixed LEAD order, rather than leaving them to first-seen order at the far right of the CSV.

**OLD**
```
            "avg_ts", "energy_basis", "joules", "compute_joules", "active_joules",

```

**NEW**
```
            "avg_ts", "energy_basis", "joules", "compute_joules", "active_joules",
            "idle_mw", "idle_mw_source", "idle_mw_baseline", "idle_mw_pooled",

```

### edit — `bench/run_matrix.py`

Without this the per-case floor is read off the tail of a cooldown that contains a 400 MB-1.1 GB SD read stamped marker=idle, which would make the local floor worse than the pooled one. The marker is deliberately NOT set back to idle: the idle interval must end where the pre-read begins.

**OLD**
```
            cached = False if args.no_precache else precache(mpath)

```

**NEW**
```
            #
            # Under its OWN marker, not "idle". The pre-read is seconds of
            # multi-watt SD-card work, and running it under the idle label put a
            # ~5 W burst at the tail of every cooldown -- which is precisely the
            # stretch idle_floor_at() reads the floor from, and which also
            # inflated the pooled mean. Leaving the marker set until t_start
            # means the idle interval ends where the pre-read begins, and the
            # pre-read's own cost becomes a window somebody can measure.
            if args.no_precache:
                cached = False
            else:
                set_marker("precache:%s:%03d" % (run["case_id"], i))
                cached = precache(mpath)

```

### edit — `telemetry/energy_window.py`

This docstring was the third of the three conflicting definitions, and it is the one the analyst reads. It now describes the same definition the driver uses and explicitly forbids the node-facts number as a subtrahend.

**OLD**
```
Idle subtraction
  --idle-mw N   subtract an N-milliwatt floor before reporting "active" energy.
  Take N from the node's facts JSON (power.idle_board_mw). Report BOTH numbers
  in the thesis: total energy is what the wall sees, active energy is what the
  inference actually cost.

```

**NEW**
```
Idle subtraction
  --idle-mw auto  (the default) takes the floor from the idle samples in THIS
  CSV, per window: the idle interval that ended just before the window opened,
  less its first few seconds of clock ramp-down. That is the board in the same
  thermal state the window ran in, which matters because SoC leakage rises with
  junction temperature -- an hour into a sweep a fanless Nano idles several
  hundred milliwatts above where it idled cold, and one scalar subtracted from
  every case biases the result along case order.
  --idle-mw N   subtracts a fixed N instead. Use it to reproduce a published
  row exactly: pass that row's own `idle_mw` from the joined bench_energy CSV.
  Do NOT use the node facts `power.idle_board_mw` here -- that is a cold-board
  figure recorded once for the device table, not the floor any particular
  window ran against.
  Report BOTH numbers in the thesis: total energy is what the wall sees, active
  energy is what the inference actually cost. Active energy is never clamped --
  a negative one means the floor is wrong, and that has to be visible.

```

### edit — `telemetry/energy_window.py`

The twin of the driver's definition, in the tool the protocol names for re-deriving a published figure. Deliberately a duplicate rather than an import: the two files already twin energy_at(), row_compute_seconds()/bench_compute_s() and tokens_of()/bench_tokens() the same way, and the regression asserts they agree.

**OLD**
```
def summarise(win, idle_mw=0.0):
```

**NEW**
```
# Twins of idle_intervals()/interval_mean_mw()/idle_floor_at() in
# bench/run_matrix.py -- keep the three in step, and keep IDLE_SETTLE_S equal.
# docs/10_measurement_protocol.md names this tool as the way to re-derive a
# published figure by hand, so a disagreement here is a disagreement with the
# deliverable.
IDLE_SETTLE_S = 3.0


def idle_intervals(rows):
    """Every CONTIGUOUS run of idle-marked samples. The driver writes the same
    "idle" marker for the pre-sweep baseline and for every cooldown, so the file
    splits on contiguity, not on the label."""
    out, cur = [], []
    for r in rows:
        if (r.get("marker") or "").strip() == "idle":
            cur.append(r)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def interval_mean_mw(win, settle_s=IDLE_SETTLE_S):
    """Mean module power over one idle interval, less its first `settle_s`
    seconds -- the governor needs that long to walk the rails back down, so the
    head of a cooldown is not yet idle."""
    ts = [_f(r, "ts_unix") for r in win]
    ts = [t for t in ts if t is not None]
    if not ts:
        return None
    cut = ts[0] + settle_s
    vals = [_f(r, "pow_in_mw") for r in win
            if (_f(r, "ts_unix") or -1) >= cut and _f(r, "pow_in_mw") is not None]
    return sum(vals) / len(vals) if vals else None


def idle_floor_at(intervals, t_start, settle_s=IDLE_SETTLE_S):
    """The floor for a window that opened at `t_start`: the last idle interval
    that ended before it. Returns (mw, source), or (None, "none") when the CSV
    holds no idle interval before the window -- in which case the caller must
    say so rather than substitute a number from elsewhere in the file."""
    best = None
    for k, win in enumerate(intervals):
        ts = [_f(r, "ts_unix") for r in win]
        ts = [t for t in ts if t is not None]
        if not ts or ts[-1] > t_start:
            continue
        best = (k, win)
    if best is None:
        return None, "none"
    k, win = best
    mw = interval_mean_mw(win, settle_s)
    if mw is None:
        return None, "none"
    return mw, ("baseline" if k == 0 else "cooldown[%d]" % k)


def summarise(win, idle_mw=0.0):
```

### edit — `telemetry/energy_window.py`

Same clamp removal as the driver, so the two tools still produce identical numbers.

**OLD**
```
        res["idle_mw"] = idle_mw
        res["module_active_joules"] = round(max(active, 0.0), 4)
```

**NEW**
```
        res["idle_mw"] = round(idle_mw, 1)
        # Unclamped on purpose. A negative active energy says the floor is
        # larger than the power the window drew, i.e. the floor is wrong;
        # max(x, 0.0) published a plausible 0.0 and threw the evidence away.
        res["module_active_joules"] = round(active, 4)
```

### edit — `telemetry/energy_window.py`

A numeric --idle-mw still works exactly as before (that is how a published row is reproduced); the default stops being 'no floor at all' and becomes the same definition the driver used.

**OLD**
```
    ap.add_argument("--idle-mw", type=float, default=0.0,
                    help="idle floor to subtract for 'active' energy")
```

**NEW**
```
    ap.add_argument("--idle-mw", default="auto",
                    help="idle floor in mW to subtract for 'active' energy, or "
                         "'auto' (default) to take each window's floor from the "
                         "idle interval that preceded it in this CSV, or 'none' "
                         "to report total energy only")
```

### edit — `telemetry/energy_window.py`

One floor per window, resolved before the whole-window figure and the compute slice are built, and the source reported alongside so the reader can see which interval it came from.

**OLD**
```
    results = [summarise(w, args.idle_mw) for w in wins]
```

**NEW**
```
    # Resolve the floor ONCE per window, before anything is summarised, so the
    # whole-window figure, the compute slice and the printed source all use the
    # same number. "auto" is per window; an explicit value applies to all of
    # them, which is what reproducing a published row needs.
    idle_spec = (args.idle_mw or "").strip().lower() \
        if isinstance(args.idle_mw, str) else args.idle_mw
    if idle_spec in ("none", "", "0"):
        floors = [(0.0, "none") for _ in wins]
    elif idle_spec == "auto":
        ivs = idle_intervals(rows)
        floors = []
        for w in wins:
            t0 = _f(w[0], "ts_unix") or 0.0
            mw, src = idle_floor_at(ivs, t0)
            if mw is None:
                print("note: no idle samples precede window at %s -- reporting "
                      "total energy only. Pass --idle-mw N to subtract a floor."
                      % w[0].get("ts_utc"), file=sys.stderr)
                mw, src = 0.0, "none"
            floors.append((mw, src))
    else:
        try:
            fixed = float(idle_spec)
        except (TypeError, ValueError):
            sys.exit("--idle-mw must be a number, 'auto' or 'none' (got %r)"
                     % args.idle_mw)
        floors = [(fixed, "fixed") for _ in wins]

    results = []
    for w, (mw, src) in zip(wins, floors):
        res = summarise(w, mw)
        res["idle_mw_source"] = src
        results.append(res)
```

### edit — `telemetry/energy_window.py`

The compute slice must use the same floor as its parent window; args.idle_mw is now a string spec, not a number.

**OLD**
```
                    ce = summarise(sub, args.idle_mw)
```

**NEW**
```
                    ce = summarise(sub, idle_mw)
```

### edit — `telemetry/energy_window.py`

Brings this window's resolved floor into scope for the compute-slice block below.

**OLD**
```
            cs = bench_compute_s(b, reps)
            if cs and 0.0 < cs < res["duration_s"]:
```

**NEW**
```
            cs = bench_compute_s(b, reps)
            idle_mw = floors[i][0]
            if cs and 0.0 < cs < res["duration_s"]:
```

### edit — `telemetry/energy_window.py`

Same floor, same unclamped subtraction, on the interpolated compute slice.

**OLD**
```
                        if args.idle_mw:
                            ce["module_active_joules"] = round(max(
                                ce["module"]["joules"]
                                - args.idle_mw / 1000.0 * ce["duration_s"], 0.0), 4)
```

**NEW**
```
                        if idle_mw:
                            ce["module_active_joules"] = round(
                                ce["module"]["joules"]
                                - idle_mw / 1000.0 * ce["duration_s"], 4)
```

### edit — `scripts/60_collect_node_facts.sh`

The old comment asserted the opposite of what the code feeds. This says what the number is for, keeps every pipeline under the `set +o pipefail` idiom from scripts/lib/common.sh, and makes the sample a mean of ~12 readings rather than 3.

**OLD**
```
# tegrastats one-shot for the idle baseline; energy numbers are always
# reported net of this floor.
IDLE_MW=""
if command -v tegrastats >/dev/null 2>&1; then
  # tegrastats never exits on its own, so `timeout` ALWAYS returns 124 here --
  # that is the design, not a failure. Unguarded under pipefail it killed this
  # script at the 6 s mark, before the heredoc below that is its entire output,
  # and `make facts | tee ...` then reported success over a zero-byte JSON file.
  IDLE_MW="$( set +o pipefail
    timeout 6 tegrastats --interval 1000 2>/dev/null \
    | sed -n 's/.*POM_5V_IN \([0-9]*\)\/.*/\1/p' | tail -3 \
    | awk '{s+=$1; n++} END{if(n) printf "%d", s/n}' || true )"
fi
```

**NEW**
```
# tegrastats one-shot for the COLD-BOARD idle figure. This is a device-table
# number -- what this node burns doing nothing, once, for the node comparison --
# and it is NOT the floor subtracted from a sweep. That floor is per case and is
# measured from the cooldown that preceded each case (bench/run_matrix.py's
# idle_floor_at(); every joined row carries it as idle_mw). The two must not be
# swapped: leakage rises with junction temperature, so this cold number sits
# several hundred milliwatts below the floor a case an hour into a sweep ran
# against, and using it as the subtrahend biases the result along case order.
#
# Sampled for 15 s and averaged over everything after the first 3 s: the first
# tegrastats line lands while the daemon is still starting, and three
# instantaneous readings of a switching rail are not a mean of anything.
IDLE_MW=""
IDLE_N=""
if command -v tegrastats >/dev/null 2>&1; then
  # tegrastats never exits on its own, so `timeout` ALWAYS returns 124 here --
  # that is the design, not a failure. Unguarded under pipefail it killed this
  # script at the 15 s mark, before the heredoc below that is its entire output,
  # and `make facts | tee ...` then reported success over a zero-byte JSON file.
  IDLE_SAMPLES="$( set +o pipefail
    timeout 15 tegrastats --interval 1000 2>/dev/null \
    | sed -n 's/.*POM_5V_IN \([0-9]*\)\/.*/\1/p' | tail -n +4 || true )"
  IDLE_MW="$( set +o pipefail
    printf '%s\n' "$IDLE_SAMPLES" \
    | awk 'NF{s+=$1; n++} END{if(n) printf "%d", s/n}' || true )"
  IDLE_N="$( set +o pipefail
    printf '%s\n' "$IDLE_SAMPLES" | awk 'NF{n++} END{printf "%d", n+0}' || true )"
fi
```

### edit — `scripts/60_collect_node_facts.sh`

The facts JSON is committed and read months later by someone who will not read this script. It now carries its own provenance and its own warning against being used as the subtrahend, in the same style as the existing rail_scope field.

**OLD**
```
    "idle_board_mw": ${IDLE_MW:-0},
    "rail_source": "carrier-board INA3221 via tegrastats POM_5V_IN",
```

**NEW**
```
    "idle_board_mw": ${IDLE_MW:-0},
    "idle_board_mw_basis": "cold board, otherwise unloaded: mean POM_5V_IN over a 15 s tegrastats run, first 3 s dropped, ${IDLE_N:-0} samples. Device-table figure only -- the floor subtracted from a sweep is per case, from the cooldown before it (joined CSV column idle_mw)",
    "rail_source": "carrier-board INA3221 via tegrastats POM_5V_IN",
```

### edit — `docs/10_measurement_protocol.md`

§2's claim was simply false against the code, and it is the sentence a reader would defend the thesis with. It now states the surviving definition, why it has to be local, and why the node-facts number is not it.

**OLD**
```
- `active_joules` — net of the measured idle floor (`idle_mw`, taken from the
  30 s idle baseline the driver collects before the sweep)

Report both. Quoting only total makes the Nano look worse than it is; quoting
only active hides the cost of keeping the node powered at all. For a
"what does this cluster cost to run" claim, total is the honest number.

```

**NEW**
```
- `active_joules` — net of the idle floor (`idle_mw`) measured for **that case**

Report both. Quoting only total makes the Nano look worse than it is; quoting
only active hides the cost of keeping the node powered at all. For a
"what does this cluster cost to run" claim, total is the honest number.

**The floor is local, not a constant of the board.** SoC leakage rises with
junction temperature, so a fanless Nano an hour into a sweep idles several
hundred milliwatts above where it idled cold. One scalar subtracted from every
case therefore over-subtracts from the early cases and under-subtracts from the
late ones — a bias aligned with case order, which is exactly the axis a reader
cannot separate from the axis under study. So the floor each row carries is the
mean power over **the idle interval that ended just before that case opened its
window** (its first 3 s dropped, because the governor needs that long to walk
the rails back down). For the first case that interval *is* the 30 s pre-sweep
baseline: one definition, not two.

Three columns make the subtraction auditable from the archived CSV alone:

| column | what it is |
| --- | --- |
| `idle_mw` | the floor actually subtracted from this row, and the one to pass to `energy_window.py --idle-mw` to reproduce it |
| `idle_mw_source` | which interval it came from: `baseline`, or `cooldown[k]` |
| `idle_mw_baseline`, `idle_mw_pooled` | the cold pre-sweep floor, and the mean over every idle sample in the file. Neither is subtracted; the spread between them and `idle_mw` is the thermal drift, and is worth reporting |

`active_joules` is **not clamped at zero.** A negative one means the floor being
subtracted is larger than the power the case drew — a stale baseline, a cooldown
that never went idle, a board doing something else — and that is a signal, not
something to round away. Any negative `active_joules` is an anomaly to record
and chase, never a row to publish.

The node facts `power.idle_board_mw` is a **different number with a different
job**: a cold-board reading taken once by `scripts/60_collect_node_facts.sh`, for
the device table and for node-to-node comparison. Never use it as the subtrahend
for a sweep — it is several hundred milliwatts below what a hot case ran
against, and `idle_board_mw_basis` in the facts JSON says so.

```

### edit — `docs/10_measurement_protocol.md`

The column table is the one place the CSV's columns are documented; the fixture already asserts no column is undocumented-and-sometimes-blank.

**OLD**
```
| `energy_basis` | which slice
```

**NEW**
```
| `idle_mw`, `idle_mw_source` | the idle floor subtracted from this row and the interval it was measured over — see §2 |
| `idle_mw_baseline`, `idle_mw_pooled` | the cold pre-sweep floor and the whole-file idle mean, for comparison only; neither is subtracted |
| `energy_basis` | which slice
```

### edit — `docs/10_measurement_protocol.md`

The worked example was handing the tool a hard-coded cold-board number, which is the habit this finding is about.

**OLD**
```
python3 telemetry/energy_window.py run.csv --marker prefill --idle-mw 1900

```

**NEW**
```
python3 telemetry/energy_window.py run.csv --marker prefill

```

### edit — `docs/10_measurement_protocol.md`

States the new default next to the example, and names the one legitimate use of a fixed value.

**OLD**
```
**Always pass `--marker`.**
```

**NEW**
```
`--idle-mw` defaults to `auto`, which reads each window's floor off the idle
samples that precede it in the same CSV — the same definition the driver uses.
To reproduce a published row exactly, pass that row's own `idle_mw` instead
(`--idle-mw 1873`); to report total energy only, `--idle-mw none`.

**Always pass `--marker`.**
```

### edit — `bench/fixtures/fake_tegrastats.py`

Without a fixture whose idle power drifts, a pooled floor and a per-case floor are numerically identical and nothing can be tested. Defaults off, so every existing assertion is untouched.

**OLD**
```
phase_file = os.environ["HS_FAKE_PHASE"]
MW = {"idle": (1900, 0, 238), "load": (5200, 200, 3000),
      "prefill": (7400, 3100, 2600), "decode": (4800, 1500, 1900)}
while True:
    try:
        ph = open(phase_file, encoding="utf-8").read().strip() or "idle"
    except OSError:
        ph = "idle"
    p_in, p_gpu, p_cpu = MW.get(ph, MW["idle"])

```

**NEW**
```
phase_file = os.environ["HS_FAKE_PHASE"]
MW = {"idle": (1900, 0, 238), "load": (5200, 200, 3000),
      "prefill": (7400, 3100, 2600), "decode": (4800, 1500, 1900)}
# HS_FAKE_IDLE_HOT_MW models the one thing a constant idle floor cannot: SoC
# leakage rises with junction temperature, so once the board has done any work
# it idles HIGHER than it did cold, and stays there for the rest of the sweep.
# That is what makes a pooled floor (baseline + every cooldown) differ from the
# floor any individual case actually ran against. Unset, idle is flat and every
# other fixture behaves exactly as before.
HOT_IDLE_MW = int(os.environ.get("HS_FAKE_IDLE_HOT_MW", "0"))
worked = False
while True:
    try:
        ph = open(phase_file, encoding="utf-8").read().strip() or "idle"
    except OSError:
        ph = "idle"
    if ph != "idle":
        worked = True
    p_in, p_gpu, p_cpu = MW.get(ph, MW["idle"])
    if ph == "idle" and worked and HOT_IDLE_MW:
        p_in = HOT_IDLE_MW

```

---

## Finding #23 — already-fixed

Closed — do not re-report it. The two edits below are hardening for the residue only: make the driver count `energy_basis` values on stdout beside the `per_token_status` counts, so a sweep that silently fell back to a load-inclusive numerator is visible while the operator is still at the terminal; and make docs/10 say that the two columns are independent axes (divisor vs numerator) and that a table must filter on both. No new per_token_status value: adding one would force the per-token columns to be withheld (check_finding3_invariants.py fails any `refused_*` row that publishes), which would destroy the upper-bound data docs/10 currently promises.

**Justification (against current code):** Every limb of the finding as written is closed, and I confirmed each against the current code rather than the handoffs.

Numerator. The window is no longer `t_start`/`t_end` around the whole process for the published figure: `cs = rec.get("compute_s") ... anchor = bounds.get(rec.get("marker"), (None, None))[1] or rec["t_end_unix"] ... ce = energy_between(power_csv, anchor - cs, anchor)` and then `rec["compute_joules"] = ce["joules"] ... rec["load_s"] = round(dur - ce["duration_s"], 3); rec["load_joules"] = round(e["joules"] - ce["joules"], 4)`, with `energy_j, window_s = rec["compute_joules"], cs_meas` feeding `joules_per_token`. Model load and CUDA init are out of the numerator and reported separately, exactly as the finding's own Fix proposed ("record a separate load_joules ... and subtracting it").

Divisor. `tokens_of()` is now `return (n_p * (reps + 1) + n_g * reps + (1 if n_g else 0)) or None`, off the row's OWN bench_n_prompt/bench_n_gen, with the docstring made true: "the warm-up before each test's repetition loop -- a full prompt pass for pp, one generated token for tg -- happens inside the same marker window and burns real joules. Both terms belong in the divisor". docs/42_finding3_verification.md records this verified against the upstream warm-up block.

And critically the two halves agree with each other: `row_compute_seconds()` puts the warm-up INSIDE the trimmed slice (`warm_ns = ns` for pp, `ns / n_g` for tg, returning `(ns * reps + warm_ns) / 1e9`), so the numerator covers the warm-up that the divisor counts. Had the trim excluded it, #23 would simply have inverted.

Dead config. `"warmup": true` is gone from bench/matrix.json — `defaults` is now `{"repetitions": 3, "threads": 4, "flash_attn": 0, "cooldown_s": 20}`, and `grep -rn warmup` finds no live reference outside prose.

The failure scenario the finding describes is gone too: the fixture already proves it, asserting `TRUTH = {"prefill": 2.0 * 7.4 / 2048, "decode": 3.008 * 4.8 / 385}` and matching to within 3% with `energy_basis == "compute"`.

The residue is narrower than the finding, and is a labelling hole rather than an accounting one. When the trim cannot be made the driver writes `rec["energy_basis"] = "whole-process (includes model load)"` and then still publishes: `elif rec.get("per_token_status") == PT_OK and energy_j and tokens: rec["joules_per_token"] = round(energy_j / tokens, 6)`. `per_token_status` stays `ok`, and docs/10 defines `ok` as "the per-token columns are populated and trustworthy" while telling the reader elsewhere that no row whose per_token_status is not ok may appear in a figure. So a student filtering on `per_token_status == "ok"` — which is what the protocol tells them to do — can still put a load-inclusive J/token in a table. Publishing it as an upper bound is a deliberate earlier decision (docs/10: "treat those rows as an upper bound") and check_finding3_invariants.py permits it, so the fix is to make the filter rule name both columns, not to withhold the number.

**Reproduction:** Not reproduced as a live defect — it is fixed. Positive confirmation instead, from a driver run through the fixture (fake llama-bench sleeps HS_FAKE_LOAD_S=3.0 under a 5200 mW load phase, then 2.0 s of prefill at 7400 mW):

  marker       compute_J  compute_s  tokens  energy_basis
  prefill:001     14.579      2.000    2048  compute
  prefill:002     14.586      2.000    1024  compute

14.579 J / 2.000 s = 7.29 W, i.e. the prefill rail alone; the ~15.6 J of load energy is on the row as load_joules and is not in the numerator. tokens=2048 is 512*(3+1), the warm-up counted. The whole-process figure would have been roughly (3.0*5.2 + 14.58)/2048 = 0.0148 J/tok against the true 0.00712 — a factor of 2.1, which is the bias the finding describes and which no longer appears.

For the residue: the fallback branch is reachable only when the installed ref emits no `avg_ns` (or reports one longer than the window). The fixture has no knob for that, so I did not force it; the branch is plainly there in the code, quoted above.

**Test:** One grep, added to the finding-#22 fixture block described above, right after the driver run:

  grep -q "energy_basis     compute                    2" "$FIX/drift.txt" \
    && ok "the numerator's basis is reported, not only the divisor's" \
    || { bad "energy_basis not summarised"; tail -6 "$FIX/drift.txt"; }

The already-fixed body of #23 is covered by assertions that are already in the suite and still pass: the tokens_of identities (`tokens_of({bench_n_prompt: 512, bench_n_gen: 0, repetitions: 3}) == 2048`, the split halves summing to the fused count at every -r), `row_compute_seconds({avg_ns: 1e9, n_gen: 128}, 3) == 3.008` (warm-up inside the slice), and the end-to-end 'both tools agree exactly and match hand-computed truth' block that checks `joules_per_token` against `compute_s * watts / tokens` to within 3% on four rows.

What cannot be tested off-hardware without a new fixture knob: the whole-process fallback itself. Triggering it needs a ref that emits no `avg_ns`, and fake_llama_bench.py has no switch for that (HS_FAKE_NO_COUNTS drops the count columns, not avg_ns). If the fallback is ever worth an assertion, add `HS_FAKE_NO_AVG_NS` beside the existing knobs and assert the row comes out `per_token_status=ok` with `energy_basis` starting `whole-` and the warning line on stdout — but on the pinned CUDA-10.2-era refs `avg_ns` is present, so this is a guard against a ref change, not against the shipped configuration.

**Conflicts:** The stdout print lands in the same closing summary block as the finding-#3 per_token_status counts and the PT_MULTI_PHASE warning. It appends rather than reorders, but any fixture that greps that block by line position (none currently do; they all grep by content) would need updating. | Deliberately does NOT add a per_token_status value. Adding one would collide with bench/check_finding3_invariants.py's KNOWN_STATUS set and with its rule that a `refused_*` row publishing any per-unit column is a violation — which would force the whole-process rows to withhold the upper-bound numbers docs/10 explicitly promises. | The docs edit sits in the same §4 status table that the finding-#3 work owns; it appends a paragraph after the existing `ok` sentence and changes nothing above it.

### edit — `bench/run_matrix.py`

A three-hour sweep that fell back to a load-inclusive numerator currently ends with '32/32 rows ok' and nothing else. This puts the numerator's basis next to the divisor's status, in the same summary block, and names the likely cause.

**OLD**
```
    for st, n in sorted(by_status.items()):
        print("  per_token_status %-26s %d" % (st, n))

```

**NEW**
```
    for st, n in sorted(by_status.items()):
        print("  per_token_status %-26s %d" % (st, n))
    # per_token_status is about the DIVISOR; energy_basis is about the
    # numerator, and a row can be "ok" while its joules still include model
    # load (no avg_ns to trim with). That fallback is silent in the CSV unless
    # somebody reads the column, so count it here where the operator sees it.
    by_basis = {}
    for r in records:
        b = r.get("energy_basis")
        if b:
            by_basis[b] = by_basis.get(b, 0) + 1
    for b, n in sorted(by_basis.items()):
        print("  energy_basis     %-26s %d" % (b, n))
    n_whole = sum(n for b, n in by_basis.items() if b.startswith("whole-"))
    if n_whole:
        print("  ^ %d row(s) could not be trimmed to the compute slice, so their "
              "joules include model load. Upper bounds, not energy per token -- "
              "does the installed ref emit avg_ns?" % n_whole)

```

### edit — `docs/10_measurement_protocol.md`

The protocol tells the student to filter figures on per_token_status alone, and defines ok as 'populated and trustworthy'. That is the only remaining route by which finding #23's bias could reach a table.

**OLD**
```
`ok` always means the number is really there: a row carrying `ok` has a non-empty
`joules`. Anything else names its reason.

```

**NEW**
```
`ok` always means the number is really there: a row carrying `ok` has a non-empty
`joules`. Anything else names its reason.

`per_token_status` is about the **divisor**; `energy_basis` is about the
**numerator**, and they are independent. A row can be `ok` — the count is sound,
the ref's own `n_prompt`/`n_gen` verified against the request — while its
`energy_basis` still reads `whole-process (includes model load)`, because the ref
emitted no `avg_ns` to trim with. Its `joules_per_token` then carries the SD-card
read and the CUDA context in the numerator and no tokens for them in the divisor,
which is the old bias this protocol removed. **So filter tables on both**:
`per_token_status == "ok"` *and* `energy_basis == "compute"` for per-token
figures, `energy_basis == "timed requests"` for the fused case. An `ok` row on a
whole-process basis is one to chase — does the installed ref emit `avg_ns`? — not
one to plot. The driver counts both on stdout at the end of a sweep.

```

---

## Finding #25 — partly-fixed

Move the name list into `scripts/lib/common.sh` as `RPC_BIN_NAMES`, and have BOTH consumers read it: `bin_rpc` resolves the binary from it, and a new `rpc_running` builds the `pgrep -x` pattern from it via `rpc_comm_re`, cutting each name to the 15 chars `/proc/<pid>/comm` keeps — which is the reason `llama-rpc-server` has to be matched as `llama-rpc-serve`. That caveat is already documented six lines above `bin_rpc`, so the code now sits where its explanation is instead of restating it in another file.

`rpc_running` also falls back to `RPC_PORT` via `ss -ltn`. That is not belt-and-braces: it is the only check that survives the next upstream rename, since a worker has to bind the port to be of any use whatever it is called.

`50_bench_device.sh:41` becomes `if rpc_running; then` and its comment shrinks to why, not how. `tegrastats_running()` goes in the same place for the same reason — it is needed by #17's capture helper and by anything else that starts a logger.

One thing from the finding's Fix paragraph I did NOT take: `pgrep -f '(^|/)(llama-)?rpc-server'`. `-f` matches the whole command line, so `pgrep -f rpc-server` also matches the operator's own `vim scripts/40_run_rpc_server.sh`, the `less` reading this audit, and the ssh command that started the sweep. A false warning that fires during ordinary work trains people to answer `y` to the confirm prompt, which costs more than the case it catches. Matching `comm` exactly and adding the port test covers the same ground without that.

**Justification (against current code):** The substantive defect is closed; the structural one the finding names is not.

`scripts/50_bench_device.sh:41` no longer reads `pgrep -x rpc-server`. It now reads:

      # -x anchors the ERE to the whole process name, and without -f pgrep reads
      # /proc/<pid>/comm, which is capped at 15 chars. Match every name the rpc
      # server has shipped under (see bin_rpc in lib/common.sh), plus the truncated
      # form of the 16-char llama-rpc-server. A plain `pgrep -x rpc-server` silently
      # misses ggml-rpc-server, so the guard becomes a no-op exactly when the sweep
      # is contended. Verified against real binaries of all three names.
      if pgrep -x '(ggml-|llama-)?rpc-serve[r]?' >/dev/null; then

I verified that regex against real processes rather than trusting it (see reproduction): it matches all three names including the truncated `llama-rpc-serve`, and does not match `llama-server`. So the failure scenario in the finding — a resident worker contaminating a sweep with no warning — no longer happens.

What is still live is precisely what you flagged. `scripts/lib/common.sh:102` holds the authoritative list:

    bin_rpc()    { find_bin "$LLAMA_BIN" ggml-rpc-server rpc-server llama-rpc-server; }

immediately under a comment block (lines 80-90) that already documents the rename history and the 15-char `comm` caveat. The regex in `50_bench_device.sh` is a SECOND, independently maintained spelling of that same list, in a different notation, in a different file. The kit's own history says why that matters: the names have already changed once (`rpc-server` -> `ggml-rpc-server`, c299a92), and when they change again whoever edits `bin_rpc` has no reason to know the pre-flight guard carries a copy. The guard would go quiet again, silently, exactly as before.

Also still open from the finding's Fix paragraph: there is no port check, so a worker built under a name neither list has learned is invisible; and nothing checks the telemetry unit, whose own header (`systemd/heteroserve-telemetry.service:11-13`) says it must be stopped before a sweep.

**Reproduction:** Reproduced the ORIGINAL bug and then confirmed the current code closes it, using real binaries (`cp /bin/sleep` under each name) so `/proc/<pid>/comm` truncation is real and not simulated:

    $ for p in $(pgrep -f 'rpc-server 30|llama-server 30'); do echo "$p $(cat /proc/$p/comm)"; done
    46789 rpc-server
    46790 ggml-rpc-server
    46791 llama-rpc-serve        <- 16 chars, truncated to 15
    46792 llama-server

    $ pgrep -x rpc-server | wc -l                        # the finding's original code
    1                                                    # ...out of 3 workers
    $ pgrep -x '(ggml-|llama-)?rpc-serve[r]?' | wc -l     # the code that is there now
    3

So #25's stated failure is fixed. Then the proposed `rpc_running` from common.sh, same method:

    $ bash -c '. .../scripts/lib/common.sh; echo "re = $(rpc_comm_re)"'
    re = ggml-rpc-server|rpc-server|llama-rpc-serve
    detected rpc-server
    detected ggml-rpc-server
    detected llama-rpc-server
    llama-server correctly ignored
    --- port-only detection (unknown binary name) ---
    detected via port

The last line is a plain Python socket listening on RPC_PORT under no recognised binary name at all — the case neither list can ever cover.

Full suite in my copy with both #17's and #25's edits applied: `63 passed, 0 failed` (51 before the new blocks).

**Test:** Off-hardware, in the fixture harness's own style — the block above. No Jetson and no llama.cpp build needed: four copies of `/bin/sleep` supply real `/proc/<pid>/comm` values, and a Python socket supplies the port case.

    cp -a /mnt/data/KMITL/Final/HeteroServe/Jetson $(mktemp -d)/kit
    cd <copy>/kit && ./bench/fixtures/run_regression.sh

Measured: `63 passed, 0 failed` with this block and #17's (51 before either).

The test is also a regression against the ORIGINAL bug, not only against the current code: swap `rpc_running` for `pgrep -x rpc-server` in `probe` and it reports `contention guard missed: ggml-rpc-server llama-rpc-server`, which is finding #25 verbatim.

Not testable off-hardware: nothing here. The guard's consequence — that a resident worker actually contaminates throughput and J/token on a 4 GB, 4-core board — is a hardware claim, but the guard itself is pure process inspection and is fully covered.

**Conflicts:** #17's `45_capture_window.sh` calls `tegrastats_running()` from this common.sh block. Apply this edit first, or inline `pgrep -x tegrastats` in the helper. | Both findings insert a fixture block immediately before `head_ "regression baseline"`. The anchor stays unique in either order; just do not merge them into a single patch that expects one match. | Touches the `set -euo pipefail` surface that finding #6's work hardened. The `ss` pipeline in `rpc_running` uses the documented `( set +o pipefail; ... )` idiom from the top of common.sh, and is inside a function called from an `if`, so neither `set -e` nor pipefail can abort a caller. The 11 guarded substitutions from that work are untouched. | Does not touch `bin_rpc`'s name ORDER (ggml-rpc-server first, llama-rpc-server last), which is load-bearing per the comment at common.sh:80-90 and per audit finding #1's evidence. `RPC_BIN_NAMES` preserves it exactly. | Leaves the systemd side of the finding's Fix paragraph alone. Checking `systemctl is-active heteroserve-rpc heteroserve-telemetry` in pre-flight would overlap finding #21 (undetected logger death) and #28 (the unit's clock locking), and the units have never been executed on real hardware — a pre-flight that depends on them is untested twice over. `rpc_running`'s port test already catches the RPC unit whether or not systemd is consulted.

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/scripts/lib/common.sh`

Puts the one list, and both things that read it, next to the comment block at lines 80-90 that already explains the rename history and the comm truncation. The `ss` pipeline is wrapped in the `set +o pipefail` subshell idiom the file documents at its top, so a missing default route or an unusual `ss` cannot abort a caller.

**OLD**
```
bin_bench()  { find_bin "$LLAMA_BIN" llama-bench bench; }
bin_cli()    { find_bin "$LLAMA_BIN" llama-cli main; }
bin_rpc()    { find_bin "$LLAMA_BIN" ggml-rpc-server rpc-server llama-rpc-server; }
bin_server() { find_bin "$LLAMA_BIN" llama-server server; }
```

**NEW**
```
# ONE list of rpc-server names, newest first. Both the binary lookup and the
# "is a worker already running" guard read it, so adding a name in a future
# rename cannot fix one and silently leave the other blind -- which is exactly
# what a second, hand-maintained list in 50_bench_device.sh used to do.
RPC_BIN_NAMES="ggml-rpc-server rpc-server llama-rpc-server"

bin_bench()  { find_bin "$LLAMA_BIN" llama-bench bench; }
bin_cli()    { find_bin "$LLAMA_BIN" llama-cli main; }
# shellcheck disable=SC2086  # word-splitting IS the point: one name per argument.
bin_rpc()    { find_bin "$LLAMA_BIN" $RPC_BIN_NAMES; }
bin_server() { find_bin "$LLAMA_BIN" llama-server server; }

# An ERE matching RPC_BIN_NAMES against /proc/<pid>/comm, for `pgrep -x`. Each
# name is cut to the 15 chars the kernel keeps (see the caveat above), which is
# why llama-rpc-server must be matched as llama-rpc-serve -- an -x pattern
# spelled with the full 16-char name matches nothing at all.
rpc_comm_re() {
  local n out=""
  for n in $RPC_BIN_NAMES; do out="${out}${out:+|}${n:0:15}"; done
  printf '%s' "$out"
}

# True when an RPC worker is resident on this node. Two independent tests,
# because each covers the other's blind spot: the name test catches a worker
# started by hand from any known binary, and the port test catches one built
# under a name this list has not learned yet -- a worker has to bind RPC_PORT to
# be of any use, whatever it is called. Either way the sweep must not proceed
# unnoticed: the worker holds a CUDA context and competes for the 4 GB.
rpc_running() {
  pgrep -x "$(rpc_comm_re)" >/dev/null 2>&1 && return 0
  [ -n "${RPC_PORT:-}" ] && command -v ss >/dev/null 2>&1 || return 1
  ( set +o pipefail; ss -ltn 2>/dev/null | grep -q ":${RPC_PORT}[[:space:]]" )
}

# tegrastats permits ONE instance per boot; a second exits immediately and the
# logger that spawned it then writes a header and nothing else. Every caller
# that is about to start a logger has to ask first, or it measures nothing and
# only finds out at analysis time.
tegrastats_running() { pgrep -x tegrastats >/dev/null 2>&1; }
```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/scripts/50_bench_device.sh`

This is the cited line. The comment keeps the history (it is the only record of why the pattern is shaped this way) but stops describing a mechanism that now lives in common.sh, where the mechanism is.

**OLD**
```
  # -x anchors the ERE to the whole process name, and without -f pgrep reads
  # /proc/<pid>/comm, which is capped at 15 chars. Match every name the rpc
  # server has shipped under (see bin_rpc in lib/common.sh), plus the truncated
  # form of the 16-char llama-rpc-server. A plain `pgrep -x rpc-server` silently
  # misses ggml-rpc-server, so the guard becomes a no-op exactly when the sweep
  # is contended. Verified against real binaries of all three names.
  if pgrep -x '(ggml-|llama-)?rpc-serve[r]?' >/dev/null; then
```

**NEW**
```
  # rpc_running matches every name RPC_BIN_NAMES knows -- cut to the 15 chars
  # /proc/<pid>/comm keeps -- and falls back to RPC_PORT for a binary renamed
  # again upstream. A plain `pgrep -x rpc-server` missed both ggml-rpc-server and
  # the truncated llama-rpc-serve, so the guard became a no-op exactly when the
  # sweep was contended. Keeping the names in lib/common.sh means the next
  # rename is one edit, not two.
  if rpc_running; then
```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/fixtures/run_regression.sh`

Inserted immediately before the existing "regression baseline" section. Uses real executables rather than a string fixture, because the whole bug was a kernel-side truncation that only exists on a real process; asserts the false-positive direction too, since a guard that fires on `llama-server` teaches operators to dismiss it.

**OLD**
```
head_ "regression baseline"
```

**NEW**
```
head_ "rpc contention guard sees every name the rpc server ships under"
# Finding #25: `pgrep -x rpc-server` matched one of the three names the kit
# resolves, and /proc/<pid>/comm truncates the longest to 15 chars, so the guard
# went quiet exactly when a sweep was contended. Real processes, real comm
# values -- a regex test against a string would not have caught the truncation.
mkdir -p "$FIX/rpcnames"
for n in rpc-server ggml-rpc-server llama-rpc-server llama-server; do
  cp /bin/sleep "$FIX/rpcnames/$n"
done
probe() { bash -c '. "$1/scripts/lib/common.sh"; RPC_PORT="$2"; rpc_running' _ "$ROOT" "$1"; }
MISSED=""
for n in rpc-server ggml-rpc-server llama-rpc-server; do
  "$FIX/rpcnames/$n" 20 & RP=$!
  sleep 0.5
  probe 59997 || MISSED="$MISSED $n"
  kill "$RP" 2>/dev/null; wait "$RP" 2>/dev/null
done
[ -z "$MISSED" ] && ok "all three rpc-server names detected" \
                 || bad "contention guard missed:$MISSED"
"$FIX/rpcnames/llama-server" 20 & RP=$!
sleep 0.5
probe 59997 && bad "llama-server matched the rpc guard" \
             || ok "llama-server is not mistaken for an rpc worker"
kill "$RP" 2>/dev/null; wait "$RP" 2>/dev/null
# A rename this list has not learned yet still has to bind the port to be of use.
python3 - 59997 <<'EOPY' & RP=$!
import socket, sys, time
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(1); time.sleep(6)
EOPY
sleep 1
probe 59997 && ok "an unknown binary holding RPC_PORT is still detected" \
             || bad "port fallback missed a listener on RPC_PORT"
kill "$RP" 2>/dev/null; wait "$RP" 2>/dev/null

head_ "regression baseline"
```

---

## Finding #26 — still-live

Replace the single hard-coded path with an ordered candidate list — the documented platform-device path, the host1x-nested spelling, the devfreq class symlink, the legacy gpu.0 path the kit had, then `*.gpu` / `*.gm20b` / `*.gp10b` globs for other Tegra generations — taking the first readable one. Record the resolved path as `gpu.max_freq_source` so a 0 is diagnosable, and read the uncapped ceiling from `available_frequencies` beside it as `gpu.hw_max_freq_hz` (the GPU half of #27: nvpmodel caps the GPU through this same devfreq node, 921.6 MHz in mode 0 vs 640 MHz in mode 1). Unmatched globs are left literal by the shell and just fail the `-r` test, so this needs no `ls` and therefore no pipefail guard. The shared `SYSFS` / `first_path()` preamble is introduced by #27's first edit.

**Justification (against current code):** `scripts/60_collect_node_facts.sh` line 15 is unchanged and carries no Status line in docs/40_audit_findings.md:

    GPU_MAXF="$(cat /sys/devices/gpu.0/devfreq/57000000.gpu/max_freq 2>/dev/null || echo 0)"

and line 77 emits it verbatim: `"max_freq_hz": ${GPU_MAXF:-0},`. Nothing else in the kit touches it (`grep -rn max_freq` finds only these two lines plus the audit doc). The `2>/dev/null || echo 0` turns any wrong path into a measured-looking 0, and the JSON records no provenance, so a reader cannot tell "the GPU idles at 0 Hz" from "we looked in the wrong place". Note the audit cites line 14 for this one and line 13 for #27; both are off by one against the current file (14 is CPU, 15 is GPU) because of the pipefail guard added for #6/#16 above them. On the sysfs-path question itself I can only go as far as the evidence allows: `/sys/devices/gpu.0/` is the legacy nvgpu directory (railgate_enable, aelpg_enable, elcg_enable) and NVIDIA's Nano power-management guide for L4T 32.7 documents the devfreq knobs under the platform device, `/sys/devices/57000000.gpu/devfreq/57000000.gpu/`. I cannot confirm on hardware which spelling this particular board uses, which is exactly why the fix tries candidates and records which one answered instead of swapping one hard-coded guess for another.

**Reproduction:** Ran the pristine script from a throwaway copy (`cp -a … $(mktemp -d)/kit`):

    $ bash scripts/60_collect_node_facts.sh | python3 -c "import json,sys; print(json.load(sys.stdin)['gpu'])"
    {'name': '128-core Maxwell (GM20B)', 'compute_capability': '5.3', 'max_freq_hz': 0, 'fp16': True, 'int8_dp4a': False}
    $ ls /sys/devices/gpu.0
    ls: cannot access '/sys/devices/gpu.0': No such file or directory

That is the off-device shape of the bug: an absent path silently becomes `"max_freq_hz": 0` with nothing in the record saying so. The Nano-specific half (that gpu.0 has no devfreq subtree on L4T R32.7) cannot be reproduced here — no Jetson has ever run this kit.

**Test:** Off-hardware, in the run_regression.sh style, using the `HS_SYSFS_ROOT` seam the fix introduces: build fake `/sys` trees under the fixture tmpdir and assert the JSON. Verified — the third edit above was applied to a copy (`cp -a … $(mktemp -d)/kit`) and `./bench/fixtures/run_regression.sh` reported **54 passed, 0 failed** (51 before, plus the three new assertions), with `config/node.env unchanged` still passing. The three new lines are: "5 W and 10 W trees produce different, correctly labelled facts", "GPU max_freq found under the platform, class and legacy nvgpu spellings", "no devfreq node at all: 0 Hz with an empty source, not a silent 0". What CANNOT be tested off-hardware is which candidate the real Nano answers on; the test proves only that all of them are tried and that the winner is recorded, so the first real `make facts` run on the board self-reports the answer in `gpu.max_freq_source`.

**Conflicts:** Shares a hunk boundary with #27: this finding's shell edit uses the `SYSFS` variable and `first_path()` helper introduced by #27's first edit, four lines above it. Apply #27's CPU edit first; applying #26 alone leaves `first_path: command not found` and an unbound `$SYSFS` under `set -u`. | #28's prescribed fix also edits `scripts/60_collect_node_facts.sh` — it wants `power.clocks_locked` computed from measured state (`scaling_min_freq == scaling_max_freq`, plus the GPU devfreq min/max) instead of echoed from `LOCK_CLOCKS`. That lands in the same `"power": {` block #27 edits and needs the same GPU devfreq directory this finding resolves. It composes rather than collides if #28 reuses these: `${CPU_MAXF_F%/*}/scaling_min_freq` and `${GPU_MAXF_F%/max_freq}/min_freq`. Whoever applies #28 should do it after these two and not re-derive a second devfreq path. | Another session is inserting its own block into `bench/fixtures/run_regression.sh` immediately before `head_ "regression baseline"` (findings #12/#13). My block anchors on the end of the existing facts section, so the text does not collide, but the pass count quoted in every Status line will drift as both land — treat "54 passed" as the count at the time of this fix, not a fixed expectation. | `HS_SYSFS_ROOT` is deliberately NOT added to `config/node.env.example`: it is a test seam, and an operator who sets it would silently redirect the facts collection. That is why the resolved path goes into the JSON — a contaminated run is visible in the dataset. Do not 'document' it into node.env.example.

### edit — `scripts/60_collect_node_facts.sh`

Tries every sysfs spelling the GPU devfreq node is known to have instead of the one that does not exist on the Nano, and records the ceiling next to the cap. Depends on the SYSFS/first_path preamble added by #27's first edit, immediately above this.

**OLD**
```
GPU_MAXF="$(cat /sys/devices/gpu.0/devfreq/57000000.gpu/max_freq 2>/dev/null || echo 0)"

```

**NEW**
```

GPU_MAXF_F="$( first_path \
  ${SYSFS}/sys/devices/57000000.gpu/devfreq/57000000.gpu/max_freq \
  ${SYSFS}/sys/devices/platform/host1x/57000000.gpu/devfreq/57000000.gpu/max_freq \
  ${SYSFS}/sys/class/devfreq/57000000.gpu/max_freq \
  ${SYSFS}/sys/devices/gpu.0/devfreq/57000000.gpu/max_freq \
  ${SYSFS}/sys/class/devfreq/*.gpu/max_freq \
  ${SYSFS}/sys/class/devfreq/*.gm20b/max_freq \
  ${SYSFS}/sys/class/devfreq/*.gp10b/max_freq || true )"
GPU_MAXF="$(cat "$GPU_MAXF_F" 2>/dev/null || echo 0)"
# available_frequencies sits beside max_freq and is one space-separated line,
# highest last on every L4T release seen -- but take the maximum rather than
# trusting the order, and fall back to 0 when the node was not found at all.
GPU_HWMAXF="$(awk '{for(i=1;i<=NF;i++) if($i+0>m) m=$i+0} END{printf "%d", m+0}' \
  "${GPU_MAXF_F%/max_freq}/available_frequencies" 2>/dev/null || echo 0)"

```

### edit — `scripts/60_collect_node_facts.sh`

The dataset now carries the cap in force, the silicon ceiling it was cut from, and the path the number came from — so a 0 Hz GPU is provably a lookup failure rather than a measurement.

**OLD**
```
    "max_freq_hz": ${GPU_MAXF:-0},
    "fp16": True,
```

**NEW**
```
    "max_freq_hz": ${GPU_MAXF:-0},
    "hw_max_freq_hz": ${GPU_HWMAXF:-0},
    "max_freq_source": "${GPU_MAXF_F}",
    "fp16": True,
```

### edit — `bench/fixtures/run_regression.sh`

Covers #26 and #27 together: three fake sysfs trees prove a 5 W record cannot be mistaken for a 10 W one, that the GPU lookup finds max_freq under the platform, class-symlink, legacy-nvgpu and gm20b spellings, and that a tree with no devfreq node at all yields 0 Hz with an empty source rather than a silent measured-looking 0.

**OLD**
```
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$FIX/facts_off.json" \
  && ok "off-device output is valid JSON too" || bad "off-device JSON"

```

**NEW**
```
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$FIX/facts_off.json" \
  && ok "off-device output is valid JSON too" || bad "off-device JSON"

head_ "fact collection records the clock cap in force, not the silicon ceiling"
# nvpmodel caps the CPU through scaling_max_freq and the GPU through its devfreq
# node; cpuinfo_max_freq and the top of available_frequencies do not move with
# the mode. A 5 W tree must therefore not read like a 10 W one -- that is the
# only thing that makes the two conditions separable in the dataset. The facts
# script takes its sysfs root from HS_SYSFS_ROOT so this needs no Jetson.
mk_sys() {  # $1 dir, $2 cpu scaling khz, $3 gpu max hz, $4 pmode, $5 gpu subpath
  mkdir -p "$1/sys/devices/system/cpu/cpu0/cpufreq" "$1/$5" "$1/var/lib/nvpmodel"
  echo "$2"      > "$1/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"
  echo 1479000   > "$1/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
  echo "$3"      > "$1/$5/max_freq"
  echo "76800000 153600000 230400000 307200000 384000000 460800000 537600000 614400000 691200000 768000000 844800000 921600000" \
                 > "$1/$5/available_frequencies"
  echo "pmode:000$4 fmode:quiet" > "$1/var/lib/nvpmodel/status"
}
GPUDEV=sys/devices/57000000.gpu/devfreq/57000000.gpu
mk_sys "$FIX/sys10w" 1479000 921600000 0 "$GPUDEV"
mk_sys "$FIX/sys5w"   918000 640000000 1 "$GPUDEV"
for mode in 10w 5w; do
  HS_SYSFS_ROOT="$FIX/sys$mode" bash "$ROOT/scripts/60_collect_node_facts.sh" \
    > "$FIX/facts_$mode.json" 2>/dev/null
done
python3 - "$FIX" <<'EOPY' && ok "5 W and 10 W trees produce different, correctly labelled facts" || bad "nvpmodel cap is not recorded"
import json, sys
ten = json.load(open(sys.argv[1] + "/facts_10w.json"))
five = json.load(open(sys.argv[1] + "/facts_5w.json"))
assert five["cpu"]["max_freq_khz"] == 918000, five["cpu"]
assert ten["cpu"]["max_freq_khz"] == 1479000, ten["cpu"]
assert five["cpu"]["hw_max_freq_khz"] == ten["cpu"]["hw_max_freq_khz"] == 1479000
assert five["gpu"]["max_freq_hz"] == 640000000, five["gpu"]
assert ten["gpu"]["max_freq_hz"] == 921600000, ten["gpu"]
assert five["gpu"]["hw_max_freq_hz"] == ten["gpu"]["hw_max_freq_hz"] == 921600000
assert five["power"]["nvpmodel_active"] == "1", five["power"]
assert ten["power"]["nvpmodel_active"] == "0", ten["power"]
# the two records must not be mistakable for one another
assert five["cpu"]["max_freq_khz"] != ten["cpu"]["max_freq_khz"]
assert five["gpu"]["max_freq_hz"] != ten["gpu"]["max_freq_hz"]
EOPY

# The GPU lookup must survive the other sysfs spellings, and must say which one
# answered -- /sys/devices/gpu.0 is the legacy nvgpu directory the kit used to
# hard-code, and on a tree where only the devfreq class symlink exists a
# single-path lookup silently records 0 Hz for the cluster's only GPU.
mk_sys "$FIX/sysclass" 1479000 921600000 0 sys/class/devfreq/57000000.gpu
mk_sys "$FIX/syslegacy" 1479000 921600000 0 sys/devices/gpu.0/devfreq/57000000.gpu
mk_sys "$FIX/sysgm20b" 1479000 921600000 0 sys/class/devfreq/57000000.gm20b
for t in sysclass syslegacy sysgm20b; do
  HS_SYSFS_ROOT="$FIX/$t" bash "$ROOT/scripts/60_collect_node_facts.sh" \
    > "$FIX/facts_$t.json" 2>/dev/null
  python3 - "$FIX/facts_$t.json" "$t" <<'EOPY' || bad "GPU devfreq candidate not tried: $t"
import json, sys
d = json.load(open(sys.argv[1]))
assert d["gpu"]["max_freq_hz"] == 921600000, (sys.argv[2], d["gpu"])
assert d["gpu"]["max_freq_source"].endswith("/max_freq"), (sys.argv[2], d["gpu"])
EOPY
done
ok "GPU max_freq found under the platform, class and legacy nvgpu spellings"

# An absent node is still reported honestly rather than as a measured 0.
mkdir -p "$FIX/sysnone"
HS_SYSFS_ROOT="$FIX/sysnone" bash "$ROOT/scripts/60_collect_node_facts.sh" \
  > "$FIX/facts_none.json" 2>/dev/null
python3 - "$FIX/facts_none.json" <<'EOPY' && ok "no devfreq node at all: 0 Hz with an empty source, not a silent 0" || bad "missing devfreq node is not distinguishable"
import json, sys
d = json.load(open(sys.argv[1]))
assert d["gpu"]["max_freq_hz"] == 0 and d["gpu"]["max_freq_source"] == "", d["gpu"]
assert d["gpu"]["hw_max_freq_hz"] == 0, d["gpu"]
assert d["power"]["nvpmodel_active"] == "", d["power"]
EOPY

```

### edit — `docs/40_audit_findings.md`

Matches the Status-line convention the other 15 closed findings use, and records honestly that the hardware question behind the finding is answered by probing rather than by verification.

**OLD**
```
### 26. [MEDIUM] GPU devfreq sysfs path does not exist on the Nano; gpu.max_freq_hz is always 0

`scripts/60_collect_node_facts.sh:14` — lens `audit:platform`
```

**NEW**
```
### 26. [MEDIUM] GPU devfreq sysfs path does not exist on the Nano; gpu.max_freq_hz is always 0

> **Status:** FIXED 2026-09-14 — the lookup walks a candidate list (the platform device `57000000.gpu`, the host1x-nested spelling, the `/sys/class/devfreq` symlink, the legacy `gpu.0` path this kit had, then `*.gpu` / `*.gm20b` / `*.gp10b` globs) and takes the first that answers, instead of one hard-coded path. The resolved path is recorded as `gpu.max_freq_source`, so a 0 Hz reading is distinguishable from a lookup failure, and the uncapped ceiling from `available_frequencies` is recorded beside it as `gpu.hw_max_freq_hz`. Regression: `bench/fixtures/run_regression.sh`, section "fact collection records the clock cap in force" (54 passed, 0 failed). Which spelling this board actually uses is still unverified — nothing has run on the hardware — which is why the fix asks all of them and writes down the answer. The cited line number is PRE-FIX; do not renumber it.

`scripts/60_collect_node_facts.sh:14` — lens `audit:platform`
```

---

## Finding #27 — still-live

Read `scaling_max_freq` (the cap actually in force) as `cpu.max_freq_khz` and keep `cpuinfo_max_freq` beside it as `cpu.hw_max_freq_khz`, so the nvpmodel cap is legible only next to the ceiling it was cut from — this is the audit's own prescribed naming. Both are looked up through the same candidate helper as the GPU, since the CPU knobs live under `cpufreq/policy0` on some kernels. Also record the power mode the board is actually in as `power.nvpmodel_active`, read from `/var/lib/nvpmodel/status` (`pmode:0001 fmode:quiet`) rather than the audit's `sudo nvpmodel -q`: that file needs no sudo, and a facts run that stops for a password prompt — or dies under a non-interactive shell, the #24/#6 bug class — is worse than a missing fact. It is a plain `sed` on one file, no pipeline, so no pipefail guard is needed.

**Justification (against current code):** `scripts/60_collect_node_facts.sh` line 14 is unchanged and carries no Status line:

    CPU_MAXF="$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0)"

emitted at line 72 as `"max_freq_khz": ${CPU_MAXF:-0}`. `cpuinfo_max_freq` is the silicon ceiling and does not move with the power mode; nvpmodel enforces its cap through `scaling_max_freq`. Nothing else is recorded — there is no second frequency field. The second half of the finding is live too: the power block reads

    "nvpmodel": "${NVP_MODEL}",

which is the config knob loaded by `load_env` from `config/node.env`, not the board's state. So a facts file collected in mode 1 with a stale `NVP_MODEL=0` claims mode 0 at 1.48 GHz while the CPU is capped at 918 MHz — and since the whole purpose of the facts record is to label the 5 W and 10 W sweeps apart, that is a wrong recorded experimental condition, not a cosmetic gap. Nothing consumes `cpu.max_freq_khz` yet (`grep -rn max_freq` finds only the script and the audit doc; `90_collect_results.sh` only bundles the file), so changing what the key means breaks no reader.

**Reproduction:** Structurally reproduced off-hardware on the dev box, where scaling and cpuinfo happen to agree so only the missing second field shows:

    $ bash scripts/60_collect_node_facts.sh | python3 -c "import json,sys; print(json.load(sys.stdin)['cpu'])"
    {'model': 'Intel(R) Core(TM) i5-8350U CPU @ 1.70GHz', 'cores': 8, 'max_freq_khz': 3600000}

One frequency field, sourced from `cpuinfo_max_freq`, with no `hw_`/`scaling_` distinction and no board-read power mode anywhere in the record. The Nano-specific numbers (1479000 vs 918000) cannot be produced here — no Jetson has run this kit — so the regression fakes the sysfs tree instead.

**Test:** Off-hardware, via the `HS_SYSFS_ROOT` seam, in the regression's own style — the block listed under #26's third edit covers both findings and is the test for this one too. It builds a 5 W tree (`scaling_max_freq=918000`, `cpuinfo_max_freq=1479000`, GPU `max_freq=640000000`, `pmode:0001`) and a 10 W tree (1479000 / 1479000 / 921600000 / `pmode:0000`), runs the real script against each, and asserts the two records differ in `cpu.max_freq_khz` and `gpu.max_freq_hz`, agree on the two `hw_` ceilings, and report `nvpmodel_active` "1" and "0". Verified: applied to a copy, `./bench/fixtures/run_regression.sh` gives **54 passed, 0 failed**. The `pmode:0000` case is worth keeping — a naive `tr -dc '0-9'` on that line yields an empty string, while the `0*\([0-9][0-9]*\)` backtrack correctly yields "0". Not testable off-hardware: that `/var/lib/nvpmodel/status` exists and is world-readable on this board's L4T R32.7 image. If it is absent the field is simply empty, which is honest and harmless — the run still produces valid JSON, asserted by the "no devfreq node at all" case.

**Conflicts:** #26 depends on this finding's first edit — it introduces `SYSFS` and `first_path()`, which the GPU lookup calls. Apply this one first; the two are one coherent change to the same twelve lines and should not be split between sessions. | #28 wants `power.clocks_locked` in this same `"power": {` block replaced with measured state. It will conflict textually with the `nvpmodel_active` line added here unless applied afterwards, and it should reuse `${CPU_MAXF_F%/*}/scaling_min_freq` and `${GPU_MAXF_F%/max_freq}/min_freq` rather than hard-coding a third set of sysfs paths. | Does not touch `config/node.env` or `NVP_MODEL`'s meaning: `power.nvpmodel` still reports the configured knob, so nothing that reads the env file changes behaviour. `50_bench_device.sh`'s own `sudo nvpmodel -q` pre-flight check (#24, already fixed) is left alone — this is a second, independent, sudo-free read and the two do not interact.

### edit — `scripts/60_collect_node_facts.sh`

Records the frequency the run was actually capped at, keeps the silicon ceiling beside it so the cap is legible, and introduces the shared SYSFS/first_path preamble that #26's GPU lookup also uses.

**OLD**
```
CPU_CORES="$(nproc)"
CPU_MAXF="$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0)"

```

**NEW**
```
CPU_CORES="$(nproc)"

# The clock ceilings are the experimental condition, not trivia. nvpmodel caps
# the CPU through scaling_max_freq and the GPU through its devfreq node; the
# silicon ceilings (cpuinfo_max_freq, the top of available_frequencies) do not
# move with the mode, so recording only those makes a 5 W row and a 10 W row look
# identical in the dataset -- and the whole reason this fact is collected is to
# tell them apart. Record the cap in force AND the ceiling it was cut from.
#
# Which file holds it depends on the release: the CPU knobs moved to
# cpufreq/policy0 on some kernels, and on Tegra X1 under L4T R32 the GPU devfreq
# node hangs off the platform device (57000000.gpu) while /sys/devices/gpu.0 is
# the legacy nvgpu directory that carries railgating knobs and no devfreq at all.
# So ask every spelling we know of, take the first that answers, and write down
# which one it was -- a 0 with no path beside it is indistinguishable from a
# genuine 0. An unmatched glob is left literal by the shell and simply fails the
# -r test, which is why this needs no `ls` and therefore no pipeline guard.
#
# HS_SYSFS_ROOT is a test seam: the off-hardware regression points it at a fake
# /sys tree. It must be empty on the node, and because the resolved path is
# recorded in the JSON, a run that forgot is visible in the dataset itself.
SYSFS="${HS_SYSFS_ROOT:-}"

first_path() {
  local f
  for f in "$@"; do
    if [ -r "$f" ]; then echo "$f"; return 0; fi
  done
  return 1
}

CPU_MAXF_F="$( first_path \
  ${SYSFS}/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq \
  ${SYSFS}/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq || true )"
CPU_HWMAXF_F="$( first_path \
  ${SYSFS}/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq \
  ${SYSFS}/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_max_freq || true )"
CPU_MAXF="$(cat "$CPU_MAXF_F" 2>/dev/null || echo 0)"
CPU_HWMAXF="$(cat "$CPU_HWMAXF_F" 2>/dev/null || echo 0)"

```

### edit — `scripts/60_collect_node_facts.sh`

Reads the live power mode without sudo, so intent (NVP_MODEL from node.env) and reality can be compared in the record. Inserted after the GPU block from #26 so the file order stays as tested.

**OLD**
```
MEM_TOTAL_MB="$(awk '/MemTotal/{printf "%d", $2/1024}' /proc/meminfo)"
```

**NEW**
```

# The power mode actually in force, read from the board rather than echoed back
# from the config knob: a facts file that says mode 0 because node.env says so,
# while the board is in mode 1, mislabels every row collected beside it. nvpmodel
# records the live mode in /var/lib/nvpmodel/status as "pmode:0001 fmode:quiet",
# and that file needs no sudo -- `nvpmodel -q` does, and a facts run that stops
# for a password prompt is worse than a missing fact.
NVP_ACTIVE="$(sed -n 's/.*pmode:0*\([0-9][0-9]*\).*/\1/p' \
  "${SYSFS}/var/lib/nvpmodel/status" 2>/dev/null || true)"
MEM_TOTAL_MB="$(awk '/MemTotal/{printf "%d", $2/1024}' /proc/meminfo)"
```

### edit — `scripts/60_collect_node_facts.sh`

`max_freq_khz` now means the cap in force and `hw_max_freq_khz` the silicon ceiling, so a 5 W record and a 10 W record differ in the file rather than reading identically.

**OLD**
```
    "max_freq_khz": ${CPU_MAXF:-0}
  },
```

**NEW**
```
    "max_freq_khz": ${CPU_MAXF:-0},
    "hw_max_freq_khz": ${CPU_HWMAXF:-0}
  },
```

### edit — `scripts/60_collect_node_facts.sh`

Keeps the configured mode and the measured mode side by side; a mismatch between what the operator asked for and what the board did is then visible in the dataset instead of invisible.

**OLD**
```
    "nvpmodel": "${NVP_MODEL}",
```

**NEW**
```
    "nvpmodel": "${NVP_MODEL}",
    "nvpmodel_active": "${NVP_ACTIVE}",
```

### edit — `docs/10_measurement_protocol.md`

The table is the contract for what counts as a recorded condition; it must name the measured field rather than the config knob, and the clock caps are now recorded conditions in their own right.

**OLD**
```
| Power mode (5 W / 10 W) | `nvpmodel -m` | `power.nvpmodel` |
```

**NEW**
```
| Power mode (5 W / 10 W) | `nvpmodel -m` | `power.nvpmodel_active`, read from the board — `power.nvpmodel` is only what `node.env` asked for |
| Clock caps that mode imposes | `nvpmodel -m` | `cpu.max_freq_khz`, `gpu.max_freq_hz`; the uncapped silicon ceilings sit beside them as `cpu.hw_max_freq_khz`, `gpu.hw_max_freq_hz`, so a 5 W record cannot be read as a 10 W one |
```

### edit — `docs/40_audit_findings.md`

Matches the Status-line convention of the other closed findings and records the one deliberate deviation from the audit's prescribed fix (reading the status file instead of shelling out to sudo).

**OLD**
```
### 27. [MEDIUM] cpu.max_freq_khz reads cpuinfo_max_freq, which ignores the nvpmodel cap

`scripts/60_collect_node_facts.sh:13` — lens `audit:platform`
```

**NEW**
```
### 27. [MEDIUM] cpu.max_freq_khz reads cpuinfo_max_freq, which ignores the nvpmodel cap

> **Status:** FIXED 2026-09-14 — `cpu.max_freq_khz` is now `scaling_max_freq`, the cap nvpmodel actually enforces, with the silicon ceiling kept beside it as `cpu.hw_max_freq_khz` (and the same pair for the GPU, see #26). The live power mode is read from `/var/lib/nvpmodel/status` and recorded as `power.nvpmodel_active` next to the configured `power.nvpmodel`, so intent and reality can be compared; it deliberately does NOT call `sudo nvpmodel -q`, which cannot prompt from a detached sweep — the #24/#6 bug class. Regression: `bench/fixtures/run_regression.sh`, section "fact collection records the clock cap in force", which asserts a fake 5 W tree (918000 kHz / 640 MHz / pmode 1) and a fake 10 W tree (1479000 kHz / 921.6 MHz / pmode 0) produce different, correctly labelled records (54 passed, 0 failed). The cited line number is PRE-FIX; do not renumber it.

`scripts/60_collect_node_facts.sh:13` — lens `audit:platform`
```

---

## Finding #29 — still-live

Carry the sampling bound on every joined row instead of leaving it underivable: `sample_gap_max_s` (the longest hole the integral had to cross on one straight line), `pow_in_plateau_frac` (how often the rail repeated, i.e. whether the requested rate is real), and `joules_vs_mean_pct` (labelled as a bookkeeping check, explicitly NOT an aliasing check). Memoise `_rows()` first, because the join's cost is O(records x rows) and the whole point is that the rate becomes raisable. Raise the shipped default from 5 Hz to 10 Hz, say in the logger docstring what the rate bounds and why repetitions cannot expose it, and add a two-minute on-board calibration to docs/10 whose result (rate chosen + plateau fraction) goes in the experiment log as the stated sampling uncertainty. Say plainly, in code and in docs, that the INA's absolute accuracy is unvalidated and unvalidatable here, and that the tegrastats running average is not a second instrument.

**Justification (against current code):** Nothing has changed at telemetry/tegrastats_logger.py. The rate is still `interval_ms = max(50, int(round(1000.0 / args.hz)))` with `ap.add_argument("--hz", type=float, default=5.0, ...)`, config/node.env.example:48 still says `TELEMETRY_HZ=5`, docs/10_measurement_protocol.md:29 still says "at 1–5 Hz. ... High enough rate to attribute energy to a prefill window versus a decode window" with no statement of what that rate costs, and the second POM field is still thrown away at `for name, inst, _avg in RE_POM.findall(line):`. No column anywhere records sampling quality: FIELDS is still `"cpu_util_pct", "cpu_util_per_core", "cpu_freq_mhz", ...` and `energy_between()` still returns only `duration_s / power_samples / mean_mw / peak_mw / joules / temp_* / swap_peak_mb`.

Two things HAVE been closed incidentally and should not be re-reported. The edge-truncation half of the integration error is gone: `energy_at()` now interpolates the counter to the exact slice bounds ("At the default 5 Hz that understated prefill J/token by about 10% and decode by about 6%"). And an integral-vs-mean-power consistency check already exists in the harness — run_regression.sh asserts `abs(mw * cs / 1000.0 - cj) / cj < 0.05` on every ok row. It is simply never computed on real rows and never archived.

On what CAN be done, I have to correct the audit's own proposed fix. It says "Stop discarding the second POM field ... report window energy computed from the averaged field alongside the point-sample trapezoid; the two agreeing is evidence the sampling is adequate." That is wrong and would manufacture false confidence. The INA3221 is a shunt/bus voltage monitor with no charge or energy accumulation register, so there is no hardware integral anywhere on this board. tegrastats' second field is its own running mean over the SAME reads at the SAME instants as the first field, so avg×duration is the rectangle rule over the identical sample set. It cannot see aliasing by construction; it would agree with the trapezoid to within the endpoint residual no matter how badly the sampling aliased. An external meter is settled as out of scope, so there is no second instrument at all, and the INA's absolute accuracy is not checkable here — that half of the finding is documentation, and the docs must say so rather than imply a validation that does not exist.

What is real and cheap is bounding the SAMPLING rather than the instrument, and here the honest finding is that the logger already records every ingredient — `ts_unix` gives the achieved rate and its jitter, `pow_in_mw` gives rail staleness, `energy_in_j` vs `mean_mw` gives the rule residual — and nothing in the toolkit ever looks at any of it. So the fix is three derived columns on the archived row plus a two-minute on-board calibration, not new instrumentation. One of the three is genuinely new information: the fraction of consecutive samples that read byte-identical milliwatts is the only way to discover the rail's true update rate without a second instrument, and it is the empirical test for whether raising TELEMETRY_HZ buys anything at all.

**Reproduction:** Cost measurements on this host (`python3` timing loop over telemetry/tegrastats_logger.py `parse()` + DictWriter + flush, 20000 samples):
  parse+write+flush: 69.7 us/sample  => at 20 Hz that is 0.139% of one core here
  DictReader: 105777 rows/s
So raising the rate is nearly free on the logger side even allowing an order of magnitude for a Tegra X1 A57, but the JOIN re-reads the power CSV ~70 times for a 32-run sweep (idle_floor + marker_bounds + up to 3 energy_between per record), which is why edit 2 below is a prerequisite, not a nicety.

The new columns, verified against a hand-built power CSV with a deliberate 1.0 s hole and a rail that mostly repeats (5 Hz, 16 samples):
  {'duration_s': 3.8, 'power_samples': 16, 'mean_mw': 5000.7, 'joules': 19.0029,
   'sample_gap_max_s': 1.0, 'pow_in_plateau_frac': 0.933, 'joules_vs_mean_pct': -0.0}

And on a real fixture sweep (matrix_split.json, fake tegrastats at 5 Hz):
  prefill:split_prefill:001  err 1.45%  gap_max 0.203  samples 25
  decode:split_decode:002    err 0.28%  gap_max 0.201  samples 30
The plateau fraction reads 0.958 there because the fake emits constant power per phase — which is exactly the signal the column exists to give: at that rate those samples carry almost no new information.

Incidental but load-bearing evidence that the column is worth having: on a contended machine one regression run failed `the driver and the analysis tool agree` with `('prefill:split_prefill:003', 0.006956, 0.0072265625, 0.0374)` — a 3.74% miss against a 3% tolerance. A pristine (unpatched) copy scored 1.47% worst and the patched copy 1.45% worst on a quiet machine, so the patches did not cause it; sampling gaps under load did, and `sample_gap_max_s` is the column that names that cause.

**Test:** Off-hardware, in the fixture idiom. Two layers, both run:

1. The unit block above (`bench/fixtures/run_regression.sh`) builds a 16-sample power CSV with a known 1.0 s hole and a rail that repeats 14 of 15 times, and asserts `sample_gap_max_s == 1.0`, `pow_in_plateau_frac == 0.933`, `abs(joules_vs_mean_pct) < 1.0`. Verified passing.

2. The whole suite on the patched copy: `cp -a .../Jetson $(mktemp -d)/kit && ./bench/fixtures/run_regression.sh` -> **51 passed, 0 failed**, including the existing `both tools agree exactly and match hand-computed truth` (so the added columns do not disturb the energy math) and `every text open() pins utf-8` (the new column code adds no open()).

What CANNOT be tested off-hardware, and I am not going to pretend otherwise: whether 10 Hz is the right number. The fake tegrastats emits constant power per phase, so `pow_in_plateau_frac` reads ~0.96 there by construction. The real rail's update rate is a property of the INA3221 and NVIDIA's ina3221x driver on this board, and the two-minute calibration in docs/10 is the test — it has to run on the Nano.

**Conflicts:** Edit 2 (_rows memoisation) is a PREREQUISITE for edit 7 (10 Hz), not independent of it. Take them together or take neither: raising the rate without the cache multiplies a join cost that is already O(records x rows) and lands entirely after the sweep. | Edit 2 also constrains finding #21's fix: the pre-flight check runs while the logger is still writing, so it must NOT go through _rows() or it caches a 30 s snapshot and the join then computes every row's energy from it. #21's edit below reads the file directly and says why. | The 10 Hz default is what makes `telemetry_hz()`'s fallback 10 in #21's edit. If #29 is declined, change that fallback back to "5". | Interacts with docs/43_finding3_verification2.md's window-merge concern (short cooldown_s relative to the telemetry period can merge two markers into one window) in the SAFE direction: a shorter sample period makes merging strictly less likely. | Must never be written up as validating the INA against ground truth. That would reintroduce the external-meter claim the kit has already settled (finding #24 / README:21,115 / docs/10 section 2), and the three columns explicitly say in their own comments that they bound sampling, not the instrument. | config/node.env is live machine config and is deliberately NOT in the edit list; only node.env.example changes. The operator raises TELEMETRY_HZ in the live file by hand. bench/fixtures/run_regression.sh's own fixture env pins TELEMETRY_HZ=5 and stays that way, so the suite's timings do not move. | No conflict with the trim/interpolation work: energy_at(), the compute_s anchor and the token divisor are untouched. Verified — hand-computed physics still reproduces to 1.45% against the pristine tree's 1.47%. | telemetry/tegrastats_logger.py's own argparse `--hz` default is deliberately left at 5.0: the driver always passes --hz explicitly, and the docs/10 calibration reads more naturally as an explicit "5 then 20" comparison. Flagging it in case a reviewer reads it as an oversight.

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py`

Puts the sampling bound on the archived artefact, where the thesis reads it, instead of leaving it derivable-in-principle from the power CSV and derived by nobody. Each comment says which error the column does and does not cover, so `joules_vs_mean_pct` can never be quoted as an instrument validation.

**OLD**
```
    if e0 is not None and e1 is not None:
        out["joules"] = round(e1 - e0, 4)
    if temps:
```

**NEW**
```
    if e0 is not None and e1 is not None:
        out["joules"] = round(e1 - e0, 4)

    # ---- how far can this row's joules be trusted? -------------------------
    # There is no second instrument on this cluster, so the INA3221's absolute
    # accuracy cannot be checked and is not claimed anywhere. What CAN be
    # measured, for free, from the samples already on disk is whether the
    # SAMPLING was adequate for the burst the window contains -- and that is the
    # error term the audit raised: 5 Hz point sampling against a 67-125 ms token
    # cadence. These three columns are that bound, carried on the row itself so
    # a reader of the archived CSV can apply it without re-deriving it.
    ts = [num(r, "ts_unix") for r in sel]
    ts = [t for t in ts if t is not None]
    if len(ts) > 1:
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        # The longest hole in the window. A 1 s gap in a 10 s decode window is a
        # 10% stretch of the integral carried by a single straight line, and no
        # amount of repetition exposes it as variance.
        out["sample_gap_max_s"] = round(max(gaps), 3)
    if len(ps) > 1:
        # Fraction of consecutive samples that read EXACTLY the same milliwatts.
        # The rail cannot be resolved faster than it updates: if this is near 0
        # the requested rate is real and every sample is new information; if it
        # climbs as TELEMETRY_HZ is raised, the extra samples are duplicates and
        # the trapezoid is being smoothed by the driver, not by the board. This
        # is the only way to find the INA's true update rate without a second
        # instrument, and it costs one pass over a list.
        same = sum(1 for a, b in zip(ps, ps[1:]) if a == b)
        out["pow_in_plateau_frac"] = round(float(same) / (len(ps) - 1), 3)
    if out.get("joules") and out.get("mean_mw") and out.get("duration_s"):
        # The counter difference against mean power x span: the trapezoid
        # against the rectangle over the same samples. This does NOT validate
        # the INA -- both sides come from the same point samples, so it cannot
        # see aliasing at all. It validates the BOOKKEEPING: a sample the
        # logger's `0 < dt < 5` guard dropped, a clock step, or a window edge
        # landing badly all move the two apart while each looks reasonable
        # alone. A few percent is the expected endpoint residual; a large one
        # means the integral on this row lost something.
        rect = out["mean_mw"] * out["duration_s"] / 1000.0
        out["joules_vs_mean_pct"] = round((rect - out["joules"]) / out["joules"] * 100.0, 2)
    if temps:
```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/run_matrix.py`

Prerequisite for raising the rate at all. Without it, doubling TELEMETRY_HZ doubles a join cost that is already O(records x rows) and lands entirely after the measurements are taken, which is the worst possible place to spend twenty minutes on a board this slow.

**OLD**
```
def _rows(power_csv):
    with open(power_csv, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

```

**NEW**
```
_ROWS_CACHE = {}


def _rows(power_csv):
    """Every row of the power CSV, parsed once.

    Memoised because the join re-reads this file about seventy times for a
    32-run sweep -- idle_floor, marker_bounds, and up to three energy_between
    calls per record -- and the file grows linearly with TELEMETRY_HZ. At 20 Hz
    a two-hour sweep is ~144k rows, and re-parsing that seventy times costs the
    Nano roughly twenty minutes of pure CSV parsing AFTER the measurements are
    already in hand. Caching is safe because the logger is terminated in the
    `finally` above before the join reads anything: the file is not growing.
    """
    cached = _ROWS_CACHE.get(power_csv)
    if cached is None:
        with open(power_csv, newline="", encoding="utf-8") as fh:
            cached = list(csv.DictReader(fh))
        _ROWS_CACHE[power_csv] = cached
    return cached

```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/tegrastats_logger.py`

The file already explains the scope of POM_5V_IN in full sentences; this is the missing companion paragraph about the rate. It also nails down, next to the `_avg` that gets discarded, why discarding it is correct — otherwise the next reader adds it back as a validation, which is what the audit itself proposed.

**OLD**
```
There is no second instrument: the cluster has no external inline meter, so this
logger is the only power signal, and across-device comparison is out of scope.
Keep the peripheral set identical across runs so POM_5V_IN's carrier offset is
at least constant.

```

**NEW**
```
There is no second instrument: the cluster has no external inline meter, so this
logger is the only power signal, and across-device comparison is out of scope.
Keep the peripheral set identical across runs so POM_5V_IN's carrier offset is
at least constant.

Sampling rate, and what it bounds
  --hz sets tegrastats' own interval (floored at 50 ms below). Each POM field is
  a POINT sample of an instantaneous register -- the INA3221 has no charge or
  energy accumulator, so there is no hardware integral to read and none to check
  against. The integral in this CSV is therefore built by trapezoid between the
  points, and the rate is what bounds it. Decode on this board is a train of
  short bursts at the per-token cadence: 8-15 tok/s is a 67-125 ms period, the
  same order as the 200 ms interval a 5 Hz setting gives. The resulting error is
  a slow beat that is stable within a run, so REPETITIONS DO NOT EXPOSE IT as
  spread, and the protocol's "repetitions agree, therefore the number is good"
  check cannot see it.

  Raising the rate is nearly free on this side -- parse, format and flush is
  well under a millisecond per sample -- but it only buys anything while the rail
  is still producing new numbers. Find that out on the board instead of assuming
  it: log the same load at two rates and compare how often consecutive
  pow_in_mw values are byte-identical. docs/10_measurement_protocol.md,
  "How accurate is a window's joules", has the procedure and the columns the
  driver writes for it.

  The second POM field (the running average) is tegrastats' own mean over the
  same reads at the same instants, not an independent measurement, so averaging
  it would reproduce the same aliasing with a smoother face on it. It is parsed
  and dropped on purpose; do not present it as a cross-check.

```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/docs/10_measurement_protocol.md`

"High enough rate" was an unqualified claim in the document that defines the protocol. It is true of the window attribution and false of anything finer, and the difference is exactly the finding.

**OLD**
```
**On-board INA3221, read via `tegrastats`** at 1–5 Hz. Gives `POM_5V_IN`,
`POM_5V_GPU`, `POM_5V_CPU` in milliwatts. High enough rate to attribute energy
to a prefill window versus a decode window.

```

**NEW**
```
**On-board INA3221, read via `tegrastats`** at `TELEMETRY_HZ`, 10 Hz by default.
Gives `POM_5V_IN`, `POM_5V_GPU`, `POM_5V_CPU` in milliwatts. That is fast enough
to attribute energy to a prefill window versus a decode window, which is the
claim this study makes. It is **not** fast enough to resolve a single token:
decode runs at 8–15 tok/s here, a 67–125 ms period, the same order as the sample
interval. Per-window energy is a result; a per-token power trace is not, and
must not be drawn from this CSV. "How accurate is a window's joules" below says
what that costs and how to measure it on the board.

```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/docs/10_measurement_protocol.md`

The finding's real content is that the uncertainty on every joule is unmeasured and unreported. This is the section that reports it, states honestly which part cannot be measured on this cluster, and gives the one calibration that can actually be run — whose output belongs in the experiment log, not in a comment.

**OLD**
```
### Idle subtraction

```

**NEW**
```
### How accurate is a window's joules

There is no second instrument, so the INA3221's **absolute** accuracy cannot be
checked here and is nowhere claimed. Every joule in this study is what the
on-board rail read, compared only against other joules read the same way — which
is exactly what a within-device study needs, and all a within-device study may
assert.

What can be checked is whether the **sampling** was adequate for the burst
inside each window, and three columns on every joined row carry that. They cost
nothing: all three come from samples already on disk.

- `sample_gap_max_s` — the longest hole between two samples in the window. The
  integral crosses that hole on one straight line, so a 1 s gap in a 10 s decode
  window makes a tenth of the energy interpolation rather than measurement.
- `pow_in_plateau_frac` — the fraction of consecutive samples reading exactly the
  same milliwatts. This is how the rail's true update rate is found without a
  second instrument: raise `TELEMETRY_HZ` and look at what the fraction does. If
  it climbs, the extra samples are duplicates and the smoother curve belongs to
  the driver, not the board. If it stays near zero, the rate is real.
- `joules_vs_mean_pct` — the running integral against mean power × span. Both
  sides come from the same point samples, so this **cannot** see aliasing; do
  not report it as if it could. It checks the bookkeeping — a sample the
  logger's `0 < dt < 5` guard discarded, a clock step, a window edge landing
  badly. A couple of percent is the expected endpoint residual; ten is a row to
  go and look at.

**Do this once on the board before the campaign.** It is the only calibration
available, and it takes two minutes:

```bash
# a decode load, logged at two rates, one after the other
./telemetry/tegrastats_logger.py --out /tmp/hz5.csv  --hz 5  --duration 60 &
./build/bin/llama-bench -m MODEL.gguf -p 0 -n 256 -r 5; wait
./telemetry/tegrastats_logger.py --out /tmp/hz20.csv --hz 20 --duration 60 &
./build/bin/llama-bench -m MODEL.gguf -p 0 -n 256 -r 5; wait
```

Then compare how often `pow_in_mw` repeats in each file. If 20 Hz shows long
runs of identical values that 5 Hz does not, the rail updates at roughly 5 Hz
and a higher setting buys nothing but file size. Record the rate you chose and
the plateau fraction it produced in `docs/30_experiment_log.md`: that pair is
the stated sampling uncertainty behind every joule in the thesis.

The second field tegrastats prints after each `POM_*` value is its own running
mean of the same reads, not an independent instrument — the INA3221 has no
energy accumulator to read. Averaging it would reproduce the same aliasing with
a smoother face on it, so the logger drops it deliberately. It is not a
cross-check and must never be presented as one.

### Idle subtraction

```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/docs/10_measurement_protocol.md`

The column table is the reference a reader of the archived CSV consults; three new columns that appear there and nowhere else would be read as noise, or worse, as an accuracy claim.

**OLD**
```
| `swap_peak_mb` | > 0 means the model did not fit; the number measures swap, not the GPU |

```

**NEW**
```
| `swap_peak_mb` | > 0 means the model did not fit; the number measures swap, not the GPU |
| `sample_gap_max_s`, `pow_in_plateau_frac`, `joules_vs_mean_pct` | how well this window was sampled — see "How accurate is a window's joules". They bound the **sampling**, not the instrument |

```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/config/node.env.example`

The rate was already configurable; what was missing was a default chosen for a reason and a note saying which reason. config/node.env is live machine config and is NOT edited — the operator changes the live file by hand.

**OLD**
```
TELEMETRY_HZ=5                     # tegrastats sample rate (interval = 1000/HZ ms)

```

**NEW**
```
# tegrastats sample rate; the interval is 1000/HZ ms and the logger floors it at
# 50 ms. 5 Hz beats against the 67-125 ms decode token cadence and the error is a
# slow drift that repetitions cannot expose; 10 Hz halves it for well under 1% of
# one core. Going higher only helps while the rail still produces new numbers --
# measure that first, per docs/10_measurement_protocol.md, "How accurate is a
# window's joules". Whatever you choose, keep it the same for the whole campaign.
TELEMETRY_HZ=10

```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/fixtures/run_regression.sh`

Same shape as the existing helper blocks: a hand-built CSV whose answer is computable on paper, so the column means what the docs say it means rather than merely being present.

**OLD**
```
head_ "analysis tool"

```

**NEW**
```
head_ "telemetry: every window carries how well it was sampled"
python3 - "$FIX" <<'PY' && ok "gap, plateau and integral residual are reported" || bad "sampling-quality columns"
import csv, io, sys
sys.path.insert(0, "bench")
from run_matrix import energy_between
# 5 Hz, with a deliberate 1.0 s hole in the middle and a rail that mostly repeats
p = sys.argv[1] + "/quality.csv"
seq = [(0.2, 5000)] * 5 + [(0.2, 5001)] * 5 + [(1.0, 5001)] + [(0.2, 5001)] * 5
rows, t, e, prev = [], 1000.0, 0.0, None
for dt, mw in seq:
    if prev is not None:
        e += (mw + prev) / 2.0 / 1000.0 * dt
        t += dt
    prev = mw
    rows.append({"ts_unix": round(t, 3), "marker": "x", "pow_in_mw": mw,
                 "energy_in_j": round(e, 6)})
with io.open(p, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["ts_unix", "marker", "pow_in_mw", "energy_in_j"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
out = energy_between(p, rows[0]["ts_unix"], rows[-1]["ts_unix"])
# the hole the integral had to cross on one straight line
assert out["sample_gap_max_s"] == 1.0, out["sample_gap_max_s"]
# 14 of the 15 consecutive pairs read the same milliwatts -- at 5 Hz this rail is
# telling us almost nothing new per sample, which is the whole point of the column
assert out["pow_in_plateau_frac"] == 0.933, out["pow_in_plateau_frac"]
# trapezoid against mean x span: bookkeeping only, and it agrees here
assert abs(out["joules_vs_mean_pct"]) < 1.0, out["joules_vs_mean_pct"]
PY

head_ "analysis tool"

```

---

## Finding #30 — still-live

Average over the cores that exist, not over the cores tegrastats printed, and carry the count so `cpu_util_pct` is interpretable: add `cpu_cores_online` and `cpu_cores_seen` to FIELDS and set them in parse(). Keep the per-core string verbatim with `off` in it, so the power mode stays legible in the row rather than only in `raw`. All-cores-parked yields `cpu_util_pct=None` and `cpu_cores_online=0` rather than a ZeroDivisionError.

**Justification (against current code):** telemetry/tegrastats_logger.py:80-97 is unchanged, comment included:

            mm = RE_CORE.match(core)
            if mm:
                utils.append(float(mm.group(1)))
                freqs.append(int(mm.group(2)))
            else:
                # An offline core reads as "off"; count it as 0% so the mean
                # stays comparable across power modes that park cores.
                utils.append(0.0)
        if utils:
            d["cpu_util_pct"] = round(sum(utils) / len(utils), 2)
            d["cpu_util_per_core"] = "|".join(str(int(u)) for u in utils)

On what tegrastats actually prints: RE_CPU is `CPU \[([^\]]+)\]` and RE_CORE is `(\d+)%@(\d+)`. On L4T R32.7 a core the power mode has parked is printed as the literal token `off` with no percentage and no frequency, so RE_CORE.match() returns None and control falls into the else. nvpmodel's T210 table has mode 0 = MAXN (4x A57 @ 1479 MHz) and mode 1 = 5 W, which sets CORE_2 and CORE_3 offline and caps the A57s at 918 MHz — so a saturated 5 W board prints `CPU [98%@918,97%@918,off,off]` and the parser divides by 4.

The comment has it exactly backwards: counting the parked cores as 0% is what destroys comparability across power modes, because the denominator is the printed core count rather than the online core count. Note `cpu_freq_mhz = max(freqs)` is computed only over matched cores and is therefore already correct — it is the one field in the row that would have contradicted the wrong utilisation, and it does not.

Secondary defect in the same lines, not called out in the audit: `cpu_util_per_core` renders a parked core as the string "0", which is indistinguishable from a genuinely idle online core. The power mode is then invisible everywhere except the `raw` column.

**Reproduction:** Ran parse() on the two lines, identical saturated workload, in a copy of the kit:

  BEFORE
    10W cpu_util_pct= 98.0  per_core= 98|97|98|99  freq= 1479
    5W  cpu_util_pct= 48.75 per_core= 98|97|0|0    freq= 918

  AFTER
    10W 98.0  98|97|98|99   cores_online=4 cores_seen=4 freq=1479
    5W  97.5  98|97|off|off cores_online=2 cores_seen=4 freq=918

The failure scenario in the audit is exact: the 5 W sweep would have reported ~49% CPU against ~98% for the 10 W sweep for the same flat-out workload.

**Test:** Fully testable off-hardware; the parser is a pure function of one line of text. The block above was run against a patched copy and passes; the full suite on that copy reports **51 passed, 0 failed**.

Downstream check done explicitly, because FIELDS grows: telemetry/make_synthetic_power_csv.py imports FIELDS and builds each row with `dict.fromkeys(FIELDS)`, so the two new columns come out blank automatically — verified by regenerating the synthetic CSV and re-running `energy_window.py --marker decode --idle-mw 1900`, which still reproduces the documented `55.2 J`. The new columns appear at positions 16 and 17 of the header.

**Conflicts:** `cpu_util_per_core` now contains the token `off` where it used to contain `0`. Grepped the whole kit: the only writers/readers are this function, telemetry/make_synthetic_power_csv.py (which hard-codes "10|6|4|3") and the logger's own --echo line. Nothing parses it, so this is safe — but it is a value-format change in an archived CSV, so any pre-existing power CSV mixes the two spellings. | FIELDS is additive only; no column is removed or renamed, consistent with docs/42's finding that the joined CSV has stayed purely additive. | No overlap with finding #29: that one touches the module docstring and bench/run_matrix.py, this one touches FIELDS and parse(). The two regression blocks deliberately anchor at different points in run_regression.sh so they do not collide. | Not fixed here, and worth a separate decision: `cpu_cores_online` lives only in the power CSV, not on the joined row. The 5 W-vs-10 W comparison the failure scenario describes is made from the joined CSV, so carrying min(cpu_cores_online) over the window onto the row would make that comparison self-contained. Three lines in energy_between(), but it is scope beyond this finding and would collide with #29's edit to the same function.

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/tegrastats_logger.py`

A mean over a varying denominator is not a number until the denominator is on the row beside it. `cpu_cores_seen` is kept separately so a future L4T that prints a different core count is visible rather than silently folded into the mean.

**OLD**
```
    "cpu_util_pct", "cpu_util_per_core", "cpu_freq_mhz",
```

**NEW**
```
    "cpu_util_pct", "cpu_util_per_core", "cpu_cores_online", "cpu_cores_seen",
    "cpu_freq_mhz",
```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/telemetry/tegrastats_logger.py`

The comment being replaced asserted the opposite of the truth, so it gets replaced with the reason including the concrete tegrastats line and the concrete wrong conclusion the old arithmetic produced. Differs from the audit's own suggested snippet in keeping the per-core string honest (it had dropped the `off` markers) and in not leaving an unused `offline` counter behind.

**OLD**
```
    m = RE_CPU.search(line)
    if m:
        utils, freqs = [], []
        for core in m.group(1).split(","):
            core = core.strip()
            mm = RE_CORE.match(core)
            if mm:
                utils.append(float(mm.group(1)))
                freqs.append(int(mm.group(2)))
            else:
                # An offline core reads as "off"; count it as 0% so the mean
                # stays comparable across power modes that park cores.
                utils.append(0.0)
        if utils:
            d["cpu_util_pct"] = round(sum(utils) / len(utils), 2)
            d["cpu_util_per_core"] = "|".join(str(int(u)) for u in utils)
        if freqs:
            d["cpu_freq_mhz"] = max(freqs)

```

**NEW**
```
    m = RE_CPU.search(line)
    if m:
        utils, freqs, per_core = [], [], []
        for core in m.group(1).split(","):
            core = core.strip()
            mm = RE_CORE.match(core)
            if mm:
                utils.append(float(mm.group(1)))
                freqs.append(int(mm.group(2)))
                per_core.append(mm.group(1))
            else:
                # A core the power mode has parked prints literally "off" (L4T
                # R32.7: "CPU [98%@918,97%@918,off,off]" in nvpmodel mode 1,
                # which leaves 2 of the 4 A57s online). Counting it as 0% and
                # dividing by four would report a board running flat out as
                # ~49% busy -- and since the protocol runs the whole sweep once
                # per power mode, the 5 W condition would then look half idle
                # against the 10 W condition purely from this arithmetic. So
                # the mean is over the cores that EXIST, and the count comes
                # with it: cpu_util_pct is only interpretable next to
                # cpu_cores_online.
                per_core.append("off")
        if utils:
            d["cpu_util_pct"] = round(sum(utils) / len(utils), 2)
        # Kept verbatim, "off" included, so the power mode stays visible in the
        # row itself rather than only in `raw`.
        if per_core:
            d["cpu_util_per_core"] = "|".join(per_core)
            d["cpu_cores_online"] = len(utils)
            d["cpu_cores_seen"] = len(per_core)
        if freqs:
            d["cpu_freq_mhz"] = max(freqs)

```

### edit — `/mnt/data/KMITL/Final/HeteroServe/Jetson/bench/fixtures/run_regression.sh`

Pure parse() unit test in the same idiom as the existing `unit: helpers` block — no driver run, no 30 s baseline, so it costs the suite nothing. Sits with the other unit blocks.

**OLD**
```
head_ "unit: expand() refuses a mixed case"

```

**NEW**
```
head_ "telemetry: a parked core is not a busy core"
python3 - <<'PY' && ok "cpu_util_pct is over the cores that exist" || bad "parked-core arithmetic"
import sys
sys.path.insert(0, "telemetry")
from tegrastats_logger import parse, FIELDS
TEN = ("RAM 1969/3956MB (lfb 22x4MB) SWAP 0/1978MB (cached 0MB) "
       "CPU [98%@1479,97%@1479,98%@1479,99%@1479] EMC_FREQ 6%@1600 GR3D_FREQ 0%@76 "
       "thermal@29.75C POM_5V_IN 1904/1850 POM_5V_GPU 0/0 POM_5V_CPU 238/230")
# nvpmodel mode 1 (5 W) parks 2 of the 4 A57s; L4T R32.7 prints them as "off"
FIVE = TEN.replace("CPU [98%@1479,97%@1479,98%@1479,99%@1479]", "CPU [98%@918,97%@918,off,off]")
a, b = parse(TEN), parse(FIVE)
# a board running flat out reads as running flat out in BOTH power modes. Counting
# the parked cores as 0% made the 5 W condition report ~49% and the write-up would
# have concluded the 5 W mode leaves half the CPU idle.
assert a["cpu_util_pct"] == 98.0, a["cpu_util_pct"]
assert b["cpu_util_pct"] == 97.5, b["cpu_util_pct"]
assert (a["cpu_cores_online"], b["cpu_cores_online"]) == (4, 2)
assert (a["cpu_cores_seen"], b["cpu_cores_seen"]) == (4, 4)
# the power mode stays legible in the row itself, not only in `raw`
assert b["cpu_util_per_core"] == "98|97|off|off", b["cpu_util_per_core"]
assert a["cpu_util_per_core"] == "98|97|98|99", a["cpu_util_per_core"]
assert a["cpu_freq_mhz"] == 1479 and b["cpu_freq_mhz"] == 918
# every core parked is not a divide by zero
z = parse(TEN.replace("CPU [98%@1479,97%@1479,98%@1479,99%@1479]", "CPU [off,off,off,off]"))
assert z["cpu_util_pct"] is None and z["cpu_cores_online"] == 0
for k in ("cpu_cores_online", "cpu_cores_seen"):
    assert k in FIELDS, k
PY

head_ "unit: expand() refuses a mixed case"

```
