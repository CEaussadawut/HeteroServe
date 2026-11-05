"""Join power to benchmark windows and produce the project's headline numbers.

E = integral of P dt over a run's window, joules per token = E / tokens. That is the whole
calculation; everything else here exists to make sure a number is only produced when the
data genuinely supports it.

Refusals, and why each one matters:
  * fewer than MIN_SAMPLES points inside a window  -> the integral is an extrapolation
  * a gap wider than MAX_GAP_FACTOR sample periods -> the DAQ dropped out mid-run
  * the channel has no calibration                 -> cannot carry the +-5% accuracy claim
  * the window is not covered at both ends         -> the run started before recording did

A flagged row keeps its throughput and loses its energy, rather than silently reporting a
joules-per-token figure that rests on three samples.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

MIN_SAMPLES = 8
MAX_GAP_FACTOR = 4.0        # times the nominal sample period
EDGE_TOLERANCE_FACTOR = 2.0


# ------------------------------------------------------------------- loading
def load_power(path, node=None, channel=None, t_from=None, t_to=None) -> list:
    """Read power.csv into (t, bus_v, current_a, power_w) tuples, filtered and sorted."""
    out = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(l for l in fh if not l.startswith("#")):
            try:
                t = float(row["t_utc"])
            except (KeyError, TypeError, ValueError):
                continue
            if t_from is not None and t < t_from:
                continue
            if t_to is not None and t > t_to:
                continue
            if node and row.get("node") != node:
                continue
            if channel and int(row.get("channel", -1)) != int(channel):
                continue
            try:
                out.append((t, float(row["bus_v"]), float(row["current_a"]), float(row["power_w"])))
            except (TypeError, ValueError):
                continue
    out.sort(key=lambda r: r[0])
    return out


def integrate(samples, t_start, t_end, sample_hz=28.0) -> dict:
    """Trapezoidal integration of power over [t_start, t_end].

    Trapezoid rather than rectangle: at ~28 Hz against a decode run of seconds the
    difference is small, but prefill windows can be short enough for the rectangle rule's
    bias to matter, and there is no reason to accept a known bias.
    """
    window = [s for s in samples if t_start <= s[0] <= t_end]
    period = 1.0 / sample_hz if sample_hz else 0.05
    result = {
        "energy_j": None,
        "mean_power_w": None,
        "peak_power_w": None,
        "samples": len(window),
        "duration_s": round(t_end - t_start, 6),
        "flags": [],
    }

    if len(window) < MIN_SAMPLES:
        result["flags"].append("too-few-samples:%d" % len(window))
        return result

    gaps = [b[0] - a[0] for a, b in zip(window, window[1:])]
    worst = max(gaps) if gaps else 0.0
    if worst > MAX_GAP_FACTOR * period:
        result["flags"].append("gap:%.3fs" % worst)

    edge = EDGE_TOLERANCE_FACTOR * period
    if window[0][0] - t_start > edge:
        result["flags"].append("late-start:%.3fs" % (window[0][0] - t_start))
    if t_end - window[-1][0] > edge:
        result["flags"].append("early-end:%.3fs" % (t_end - window[-1][0]))

    energy = 0.0
    for (t0, _, _, p0), (t1, _, _, p1) in zip(window, window[1:]):
        energy += (p0 + p1) / 2.0 * (t1 - t0)

    # Extend to the window EDGES at the boundary sample's power (zero-order hold). Without
    # this the integral spans only [first sample, last sample] and silently drops up to one
    # sample period of energy: -0.18 % on a 20 s decode window, but -16.7 % worst case on a
    # 0.30 s prefill one. Always low, never flagged -- quantisation cannot trip late-start /
    # early-end, whose tolerance is two whole periods -- and it puts pp and tg rows on
    # different scales, so a phase-separated J/tok comparison is not apples-to-apples.
    # The edge checks above bound each extension to EDGE_TOLERANCE_FACTOR periods; anything
    # longer is already flagged, so this is not extrapolation the module elsewhere refuses.
    energy += window[0][3] * (window[0][0] - t_start)
    energy += window[-1][3] * (t_end - window[-1][0])

    duration = t_end - t_start
    covered = window[-1][0] - window[0][0]
    result["energy_j"] = round(energy, 6)
    # Mean over the REQUESTED duration, so the row satisfies its own identity
    # energy_j == mean_power_w * duration_s. Dividing by `covered` made the two describe
    # different spans, which a reader cannot reconcile from the CSV.
    result["mean_power_w"] = round(energy / duration, 4) if duration > 0 else None
    result["peak_power_w"] = round(max(s[3] for s in window), 4)
    result["covered_s"] = round(covered, 6)
    result["effective_hz"] = round(len(window) / covered, 2) if covered > 0 else None
    return result


# --------------------------------------------------------------- token counts
def tokens_in_row(row) -> int:
    """Tokens the run actually processed, which is what joules-per-token divides by.

    llama-bench repeats each measurement, so the window covers reps * n tokens. Dividing
    by n alone would overstate the per-token energy by the repetition count - the kind of
    error that produces a confident, consistent, wrong number.
    """
    phase = row.get("phase")
    n = int(row.get("n_prompt") or 0) if phase == "pp" else int(row.get("n_gen") or 0)
    reps = int(row.get("samples") or 1) or 1
    return n * reps


# ------------------------------------------------------------------- the join
def join_run(run_dir, power_csv, sample_hz=28.0, calibrated_channels=None,
             node_channel=None, out_csv=None) -> dict:
    """Fill energy_j / joules_per_token / mean_power_w in one run's results.csv."""
    run_dir = Path(run_dir)
    results = run_dir / "results.csv"
    if not results.exists():
        raise FileNotFoundError("no results.csv in %s" % run_dir)

    with results.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = reader.fieldnames or []

    cache = {}
    joined = skipped = 0
    for row in rows:
        node = row.get("node_id")
        t_start, t_end = row.get("t_start"), row.get("t_end")
        if not (node and t_start and t_end):
            skipped += 1
            _add_flag(row, "no-window")
            continue

        t_start, t_end = float(t_start), float(t_end)
        if node not in cache:
            cache[node] = load_power(power_csv, node=node)
        samples = cache[node]
        if not samples:
            skipped += 1
            _add_flag(row, "no-power-data")
            continue

        band = integrate(samples, t_start, t_end, sample_hz)
        if band["energy_j"] is None:
            skipped += 1
            for flag in band["flags"]:
                _add_flag(row, flag)
            continue

        tokens = tokens_in_row(row)
        row["energy_j"] = band["energy_j"]
        row["mean_power_w"] = band["mean_power_w"]
        # How much of the window actually carried samples. Without it a reader cannot tell
        # a fully-sampled window from a sparsely-sampled one, and has to take energy_j on
        # trust; energy_j now spans the full window, so coverage is the separate question.
        row["covered_s"] = band.get("covered_s")
        row["joules_per_token"] = round(band["energy_j"] / tokens, 6) if tokens else ""
        for flag in band["flags"]:
            _add_flag(row, flag)
        if calibrated_channels is not None and node_channel:
            ch = node_channel.get(node)
            if ch not in calibrated_channels:
                _add_flag(row, "uncalibrated-channel")
        joined += 1

    # The header came from the producer's CSV, so any column the join ADDS has to be
    # appended or DictWriter raises. This used to work only because the node-side writer
    # happened to pre-declare the energy columns -- a silent coupling to another node's
    # folder. Append in first-seen order so the output stays deterministic.
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    out_csv = Path(out_csv) if out_csv else run_dir / "results_energy.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return {"run": run_dir.name, "rows": len(rows), "joined": joined,
            "skipped": skipped, "out": str(out_csv)}


# Which flags invalidate WHICH column. These are not the same question, and collapsing
# them into one rule costs a real measurement whichever way it collapses:
#
#   * the POWER-TRACE flags -- gap:, late-start:, early-end:, too-few-samples:,
#     uncalibrated-channel, no-power-data, no-window:, energy-span:, low-duty: -- say the
#     energy join is untrustworthy. The tokens_per_s from that same run is perfectly good;
#     discarding it would throw away a valid benchmark because the DAQ hiccuped.
#   * the THROTTLE flags say the clock moved underneath the benchmark, so tokens_per_s is
#     not a measurement of this device at its operating point. The joules actually drawn
#     are still what they were, but the run is not characteristic either way, so a throttle
#     flag excludes from both.
#
# Energy stays deliberately blanket -- ANY flag means the join is suspect -- because every
# flag above either concerns the trace or concerns the run. Throughput is the narrow case.
#
# `throttled-at-start:` counts: the board was already capped when the run began, so the
# whole run is at the wrong clock. `throttle-unknown` counts because the node-side parser
# already resolves an unreadable throttle word as throttled (Pi4/node/bench_parse.py:102)
# and this must not be more lenient than the thing producing the data.
THROUGHPUT_INVALIDATING = ("throttled", "throttle-unknown")


def invalidates_throughput(flags) -> bool:
    """True when a row's flags mean its tokens_per_s is not a usable measurement."""
    return any(part.strip().startswith(THROUGHPUT_INVALIDATING)
               for part in (flags or "").split(";") if part.strip())


def _add_flag(row, flag):
    existing = row.get("flags") or ""
    row["flags"] = (existing + ";" + flag).strip(";") if existing else flag


# ------------------------------------------------------------- power summary
def node_power_summary(power_csv, events_jsonl=None, active_windows=None,
                       idle_windows=None, phase_windows_jsonl=None,
                       active_phase="decode") -> dict:
    """Active and idle mean power per node - the input profiles.py needs for the rule.

    Idle windows come from explicit `idle_baseline` markers, not from "the quiet-looking
    parts of the trace": picking quiet stretches by eye biases the baseline downward and
    therefore overstates every marginal-energy figure derived from it.

    ACTIVE power is the DECODE phase, not the whole run. The only begin/end pair anything
    emits is `run_begin`/`run_end` around the whole llama-cli process, so deriving the
    active window from events averages model load and prefill into P_act: on a 40 s load /
    5 s prefill / 20 s decode run that gives 4.80 W where decode is 7.00 W, collapsing
    delta_power_w from 4.00 to 1.80 - 55 % low, with no flag. P_act is multiplied by decode
    compute time in the energy model (placement.py, README), so only a decode-phase mean is
    the right quantity there.

    `harness._append_window` already records the load/prefill/decode split into
    runs/<id>/windows.jsonl, and its own docstring warns about this exact mistake ("the
    same mistake on the per-node bench path overstated joules-per-token by tens of percent").
    Pass that file as `phase_windows_jsonl` and it is used. Without it, `mean_power_w_active`
    is left None with an `active_note`, mirroring how a missing idle marker is handled, so
    `profiles.build_devices(strict=True)` refuses rather than planning on a contaminated
    P_act. A missing number is recoverable; this one was not.
    """
    samples_by_node = {}
    with Path(power_csv).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(l for l in fh if not l.startswith("#")):
            node = row.get("node")
            if not node:
                continue
            try:
                samples_by_node.setdefault(node, []).append((float(row["t_utc"]), float(row["power_w"])))
            except (TypeError, ValueError, KeyError):
                continue

    if events_jsonl and not (active_windows and idle_windows):
        active_windows, idle_windows = windows_from_events(events_jsonl)

    # A decode split, when available, replaces the whole-process window entirely.
    active_from_phase = False
    if phase_windows_jsonl:
        phase_windows = windows_from_phase_windows(phase_windows_jsonl, phase=active_phase)
        if phase_windows:
            active_windows = phase_windows
            active_from_phase = True

    out = {"nodes": {}}
    for node, samples in samples_by_node.items():
        samples.sort()
        entry = {"samples": len(samples)}
        run_mean = _mean_over(samples, (active_windows or {}).get(node))
        if active_from_phase:
            entry["mean_power_w_active"] = run_mean
        else:
            # Whole-process window: report it, but NOT as the active power the rule uses.
            entry["mean_power_w_active"] = None
            entry["mean_power_w_run"] = run_mean
            if run_mean is not None:
                entry["active_note"] = (
                    "active window was the whole llama-cli process (load + prefill + "
                    "decode), which is not P_act - pass runs/<id>/windows.jsonl as "
                    "--windows for the decode split")
        entry["mean_power_w_idle"] = _mean_over(samples, (idle_windows or {}).get(node))
        if entry["mean_power_w_idle"] is None:
            entry["idle_note"] = ("no idle_baseline marker found - record one with "
                                  "`heteroctl idle-baseline` before computing marginal energy")
        if samples:
            entry["mean_power_w_overall"] = round(statistics.fmean(p for _, p in samples), 4)
            entry["peak_power_w"] = round(max(p for _, p in samples), 4)
        out["nodes"][node] = entry
    return out


def _mean_over(samples, windows):
    if not windows:
        return None
    picked = [p for t, p in samples if any(a <= t <= b for a, b in windows)]
    return round(statistics.fmean(picked), 4) if picked else None


def windows_from_phase_windows(windows_jsonl, phase="decode") -> dict:
    """Per-node windows for ONE phase, from the split `harness._append_window` records.

    Each line of runs/<id>/windows.jsonl carries the full-process span plus a `phases`
    list of load/prefill/decode sub-windows with absolute timestamps. A run that produced
    no timings records no phases, so it contributes nothing here rather than a guess.
    """
    out = {}
    path = Path(windows_jsonl)
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        node = record.get("node")
        if not node:
            continue
        for sub in record.get("phases") or []:
            if sub.get("phase") != phase:
                continue
            start, end = sub.get("t_start"), sub.get("t_end")
            if start is None or end is None or end <= start:
                continue
            out.setdefault(node, []).append((float(start), float(end)))
    return out


def windows_from_events(events_jsonl) -> tuple:
    """Derive active and idle windows from paired begin/end markers in the event log."""
    opens = {}
    active, idle = {}, {}
    for line in Path(events_jsonl).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        node, event, ts = rec.get("node"), rec.get("event", ""), rec.get("ts")
        if not (node and ts):
            continue
        if event.endswith("_begin"):
            opens[(node, event[:-6])] = ts
        elif event.endswith("_end"):
            key = (node, event[:-4])
            start = opens.pop(key, None)
            if start is None:
                continue
            bucket = idle if key[1] == "idle_baseline" else active
            bucket.setdefault(node, []).append((start, ts))
    return active, idle


# -------------------------------------------------------------- presentation
def characterisation_table(results_energy_csv) -> list:
    """Collapse joined rows into the per-node, per-model, per-phase summary table."""
    groups = {}
    with Path(results_energy_csv).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["node_id"], row["model"], row["quant"], row["phase"], row["n_threads"])
            groups.setdefault(key, []).append(row)

    table = []
    for (node, model, quant, phase, threads), rows in sorted(groups.items()):
        # Throughput excludes only the throttle flags; see THROUGHPUT_INVALIDATING above.
        unthrottled = [r for r in rows if not invalidates_throughput(r.get("flags"))]
        tps = [float(r["tokens_per_s"]) for r in unthrottled if r.get("tokens_per_s")]
        # Dispersion. The old code published 0.0 whenever a group held a single row, which
        # is the NORMAL case -- bench_parse.py emits one row per (model, phase, n, threads)
        # with the repetitions already collapsed into avg_ts/stddev_ts -- so every row of
        # this table claimed perfect repeatability while llama-bench's own measured spread
        # sat unread in the tokens_per_s_stddev column of the very CSV being aggregated.
        # A fabricated 0 is the worst possible value here: it reads as a strong result.
        # With several runs, the across-run sample spread is what "repeatable" means; with
        # one, the within-run figure is the only dispersion that exists. Never 0 by default.
        within = [float(r["tokens_per_s_stddev"]) for r in unthrottled
                  if r.get("tokens_per_s_stddev")]
        if len(tps) > 1:
            # Sample, not population: these are repeated draws from the node's behaviour,
            # not the whole population of runs. pstdev understated the spread.
            sd = round(statistics.stdev(tps), 3)
            sd_basis = "across-runs"
        elif within:
            sd = round(within[0], 3)
            sd_basis = "within-run"
        else:
            sd = None
            sd_basis = None
        # ENERGY MEANS EXCLUDE FLAGGED ROWS. A flag on a joined row means the energy is
        # not trustworthy -- `gap:`/`late-start:`/`early-end:` mean the power trace did not
        # cover the window, `too-few-samples:` that it was barely sampled,
        # `uncalibrated-channel` that no calibration was applied at all. Averaging those in
        # produced a complete, plausible J/tok that could be tens of percent out, while
        # report.py printed underneath that they "were excluded from the energy means".
        # They were not. Now they are, and the prose is true.
        #
        unflagged = [r for r in rows if not r.get("flags")]
        jpt = [float(r["joules_per_token"]) for r in unflagged if r.get("joules_per_token")]
        power = [float(r["mean_power_w"]) for r in unflagged if r.get("mean_power_w")]
        table.append({
            "node": node, "model": model, "quant": quant, "phase": phase, "threads": int(threads or 0),
            "tokens_per_s": round(statistics.fmean(tps), 3) if tps else None,
            "tokens_per_s_sd": sd,
            # WHICH dispersion this is. The two are not comparable and the nodes do not
            # produce the same one: the Pi 5 wraps every cell in an outer repeat loop, so
            # its groups hold several rows and sd is the across-run spread; the Pi 4 runs
            # each shape once, so its tg groups hold one row and sd is llama-bench's
            # within-run figure. Printed in adjacent rows of one table, an unlabelled
            # column invites reading a run-to-run spread against a within-run one.
            "sd_basis": sd_basis,
            "joules_per_token": round(statistics.fmean(jpt), 5) if jpt else None,
            "mean_power_w": round(statistics.fmean(power), 3) if power else None,
            "wh_per_1k_tokens": round(statistics.fmean(jpt) * 1000 / 3600, 4) if jpt else None,
            "runs": len(rows),
            # How many runs actually backed the energy means. A blank J/tok with
            # energy_runs=0 says "every run was flagged", which is a result in itself
            # rather than a gap in the table.
            "energy_runs": len(jpt),
            # Likewise for throughput: a blank tok/s with throughput_runs=0 says "every run
            # was throttled", which is a finding about the board, not a gap in the table.
            "throughput_runs": len(tps),
            "flagged": sum(1 for r in rows if r.get("flags")),
        })
    return table


def pareto_front(points, x_key="tokens_per_s", y_key="mean_power_w") -> list:
    """Non-dominated set: more tokens/s is better, fewer watts is better."""
    keep = []
    for p in points:
        x, y = p.get(x_key), p.get(y_key)
        if x is None or y is None:
            continue
        dominated = any(
            (q.get(x_key) or -math.inf) >= x and (q.get(y_key) or math.inf) <= y and q is not p
            and ((q.get(x_key) or -math.inf) > x or (q.get(y_key) or math.inf) < y)
            for q in points)
        if not dominated:
            keep.append(p)
    return sorted(keep, key=lambda p: p[x_key])
