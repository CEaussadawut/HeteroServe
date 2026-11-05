#!/usr/bin/env python3
"""
Turn a tegrastats_logger CSV into per-window energy, and (optionally) into
energy per token by joining it with a llama-bench CSV.

Two ways to define a window:

  --marker LABEL          every contiguous run of rows whose marker == LABEL
  --from T0 --to T1       an explicit UTC or unix-epoch interval

For each window it reports duration, mean/peak module power, and joules on each
rail. Energy is taken from the logger's running integral (a subtraction at the
window edges), not re-integrated here, so dropped or resampled rows cannot
distort it.

Idle subtraction
  --idle-mw N   subtract an N-milliwatt floor before reporting "active" energy.
  Take N from the node's facts JSON (power.idle_board_mw). Report BOTH numbers
  in the thesis: total energy is what the wall sees, active energy is what the
  inference actually cost.

Examples
  ./energy_window.py power.csv --marker decode
  ./energy_window.py power.csv --marker prefill --idle-mw 1900
  ./energy_window.py power.csv --marker decode --bench bench.csv --reps 3
"""
import argparse
import calendar
import csv
import json
import sys
from datetime import datetime


def _f(row, key):
    v = row.get(key, "")
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_samples(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)]
    if not rows:
        sys.exit("no samples in %s" % path)
    return rows


# Kept deliberately free of datetime.fromisoformat: that is Python 3.7+, and the
# Jetson runs JetPack 4.6 / Ubuntu 18.04, which ships Python 3.6.9. strptime's %z
# is also stricter on 3.6 — it rejects the "+00:00" colon form the logger emits —
# so the offset is normalised to "+0000" before parsing.
_TIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
)


def parse_time(s):
    """Accept a unix epoch or an ISO-8601 UTC stamp. A stamp with no offset is
    read as UTC, matching what tegrastats_logger.py writes."""
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass

    t = s.replace(" ", "T")
    if t.endswith(("Z", "z")):
        t = t[:-1] + "+0000"
    # "+00:00" -> "+0000"; %z on 3.6 will not take the colon.
    if len(t) > 6 and t[-6] in "+-" and t[-3] == ":":
        t = t[:-3] + t[-2:]

    for fmt in _TIME_FORMATS:
        try:
            dt = datetime.strptime(t, fmt)
        except ValueError:
            continue
        if dt.tzinfo is None:
            return calendar.timegm(dt.timetuple()) + dt.microsecond / 1e6
        return dt.timestamp()
    raise ValueError("unrecognised time %r (want epoch seconds or ISO-8601)" % s)


def windows_by_marker(rows, label):
    """Contiguous runs of rows carrying `label`. A benchmark that repeats a
    phase produces one window per repetition, which is what you want: the
    variance across repetitions is itself a result.

    Matching is exact, or on the "phase:case_id:NNN" prefix the sweep driver
    writes -- so `--marker prefill` selects every prefill window of a sweep and
    `--marker prefill:thread_sweep:027` selects exactly one. Without the prefix
    form, `--marker prefill` against a sweep CSV would match nothing, which is
    what tempts people to drop --marker and join against the whole file.
    """
    out, cur, cur_marker = [], [], None
    for r in rows:
        m = (r.get("marker") or "").strip()
        hit = (m == label or m.startswith(label + ":"))
        # Break on any change of the FULL marker, not merely on the prefix
        # ceasing to match. Two back-to-back invocations of the same phase
        # (prefill:a:001 then prefill:b:002) both match the prefix, and grouping
        # by prefix alone would fuse two runs into one window -- silently
        # halving the window count and pairing every later window with another
        # run's bench row.
        if hit and m == cur_marker:
            cur.append(r)
            continue
        if cur:
            out.append(cur)
            cur = []
        cur_marker = m if hit else None
        if hit:
            cur.append(r)
    if cur:
        out.append(cur)
    return out


def window_between(rows, t0, t1):
    sel = [r for r in rows if (_f(r, "ts_unix") or 0) >= t0 and (_f(r, "ts_unix") or 0) <= t1]
    return [sel] if sel else []


def energy_at(rows, t, col="energy_in_j"):
    """The running energy integral at exactly time `t`, linearly interpolated
    between the samples that bracket it. Twin of energy_at() in
    bench/run_matrix.py -- keep the two in step.

    Without it, a trimmed slice's energy is the counter difference between the
    first and last sample INSIDE it, which covers up to one sampling interval
    less than the slice asked for, while the full token count is still divided
    into it. At 5 Hz that understates prefill J/token by ~10% and decode by ~6%
    -- a bias that differs between the phases, so it distorts the prefill-vs-
    decode ratio the study exists to measure.
    """
    prev = None
    for r in rows:
        ts, e = _f(r, "ts_unix"), _f(r, col)
        if ts is None or e is None:
            continue
        if ts == t:
            return e
        if ts > t:
            if prev is None:
                return None
            pts, pe = prev
            return e if ts == pts else pe + (e - pe) * (t - pts) / (ts - pts)
        prev = (ts, e)
    return None


# Twins of idle_intervals()/idle_floor_at() in bench/run_matrix.py -- keep the
# two in step, and keep IDLE_SETTLE_S equal. docs/10_measurement_protocol.md
# names this tool as the way to re-derive a published figure by hand, so a
# disagreement here is a disagreement with the deliverable.
IDLE_SETTLE_S = 3.0


def idle_intervals(rows):
    """Contiguous idle runs as (t_start, t_end, mean_mw).

    The driver writes the same "idle" marker for the pre-sweep baseline and for
    every cooldown, so the file splits on CONTIGUITY, not on the label. The first
    IDLE_SETTLE_S of each interval are dropped: clocks and fans are still ramping
    down and are carrying the previous case's heat.
    """
    out, cur = [], []
    for r in rows:
        m = (r.get("marker") or "").strip()
        t, p = _f(r, "ts_unix"), _f(r, "pow_in_mw")
        if m == "idle" and t is not None and p is not None:
            cur.append((t, p))
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)

    spans = []
    for run_ in out:
        t0 = run_[0][0]
        kept = [p for t, p in run_ if t - t0 >= IDLE_SETTLE_S] or [p for _, p in run_]
        spans.append((run_[0][0], run_[-1][0], sum(kept) / len(kept)))
    return spans


def idle_floor_at(spans, t_window_start):
    """The idle interval that ended most recently BEFORE this window opened.

    That is the board in the thermal state the window actually ran in. SoC
    leakage rises with junction temperature, so an hour into a sweep a fanless
    Nano idles several hundred milliwatts above where it idled cold; one scalar
    subtracted from every case biases the result along case order, which is the
    axis a reader cannot separate from the one under study.
    """
    prior = [sp for sp in spans if sp[1] <= t_window_start]
    if not prior:
        return None, "no-idle-before-window"
    t0, t1, mw = prior[-1]
    return mw, "idle[%.3f..%.3f]" % (t0, t1)


def summarise(win, idle_mw=0.0):
    t0, t1 = _f(win[0], "ts_unix"), _f(win[-1], "ts_unix")
    dur = (t1 - t0) if (t0 is not None and t1 is not None) else 0.0

    def rail(ecol, pcol):
        e0, e1 = _f(win[0], ecol), _f(win[-1], ecol)
        joules = (e1 - e0) if (e0 is not None and e1 is not None) else None
        ps = [_f(r, pcol) for r in win]
        ps = [p for p in ps if p is not None]
        return {
            "joules": round(joules, 4) if joules is not None else None,
            "mean_mw": round(sum(ps) / len(ps), 1) if ps else None,
            "peak_mw": max(ps) if ps else None,
        }

    res = {
        "start_utc": win[0].get("ts_utc"),
        "end_utc": win[-1].get("ts_utc"),
        "start_unix": t0,
        "end_unix": t1,
        "duration_s": round(dur, 3),
        "samples": len(win),
        "marker": (win[0].get("marker") or "").strip(),
        "module": rail("energy_in_j", "pow_in_mw"),
        "gpu_rail": rail("energy_gpu_j", "pow_gpu_mw"),
        "cpu_rail": rail("energy_cpu_j", "pow_cpu_mw"),
    }
    # Idle floor: the board burns ~2 W doing nothing, and on a 10 W budget that
    # dominates a short decode window. Never report only one of the two.
    if idle_mw and res["module"]["joules"] is not None:
        active = res["module"]["joules"] - idle_mw / 1000.0 * dur
        res["idle_mw"] = round(idle_mw, 1)
        # Unclamped on purpose. A negative active energy says the floor is larger
        # than the power the window drew, i.e. the floor is wrong for this row --
        # max(x, 0.0) published a plausible 0.0 and threw the evidence away.
        res["module_active_joules"] = round(active, 4)

    temps = [_f(r, "temp_thermal_c") for r in win]
    temps = [t for t in temps if t is not None]
    if temps:
        res["temp_start_c"], res["temp_end_c"], res["temp_max_c"] = temps[0], temps[-1], max(temps)
        # A rising edge across a sweep is the thermal-throttle warning sign on a
        # fanless Nano; flag it rather than silently averaging it away.
        res["temp_rise_c"] = round(temps[-1] - temps[0], 2)
    swaps = [_f(r, "swap_used_mb") for r in win]
    swaps = [s for s in swaps if s is not None]
    if swaps:
        res["swap_peak_mb"] = max(swaps)
    return res


def load_bench(path):
    """llama-bench -o csv. Column names have shifted between refs, so look up
    each quantity by a list of candidates rather than a fixed name."""
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh)]
    except IOError as exc:
        sys.exit("cannot read bench CSV %s: %s" % (path, exc))
    if not rows:
        sys.exit("no rows in %s -- an empty <base>_NNN.csv means llama-bench "
                 "wrote no parseable CSV for that case; check the run log." % path)
    return rows


def pick(row, *names, skip_zero=False):
    """First present value among `names`.

    With skip_zero a literal "0" counts as absent. llama-bench emits BOTH
    n_prompt and n_gen on every row and zeroes the one the test does not
    exercise, so plain "first non-empty" returns the zero on a decode row and
    the token count silently collapses to 0.
    """
    for n in names:
        if n not in row or row[n] in ("", None):
            continue
        if skip_zero:
            try:
                if float(row[n]) == 0.0:
                    continue
            except (TypeError, ValueError):
                pass
        return row[n]
    return None


def bench_tokens(row, reps):
    """Tokens llama-bench actually pushed through the marker window, for one
    `-o csv` row. Returns (tokens, basis) where basis is the arithmetic, so the
    divisor behind every J/token figure is auditable in the output.

    One CSV row is one test aggregated over `-r` repetitions, and its n_prompt /
    n_gen are PER-repetition counts -- the window holds `reps` times that work.
    llama-bench also runs an unmeasured warm-up before each test's repetition
    loop: a full n_prompt-token prefill for a pp test, one generated token for a
    tg test. That warm-up burns real joules inside the window, so its tokens
    belong in the divisor too.
    """
    if "n_prompt" not in row and "n_gen" not in row:
        n = pick(row, "n_tokens", skip_zero=True)
        if n is None:
            return None, ""
        return float(n) * reps, "n_tokens %s x %d reps" % (n, reps)

    p = _f(row, "n_prompt") or 0.0
    g = _f(row, "n_gen") or 0.0
    if not p and not g:
        return None, ""
    if p and g:
        # A fused row (llama-bench -pg) is ONE test spanning both phases. Its
        # joules cannot be attributed to prompt or to generation, so summing the
        # two counts would produce a per-token number that is neither. Refuse it
        # here; the driver reports such a window as joules_per_request instead.
        return None, "fused pp+tg row -- per-token energy is not attributable"
    total = p * (reps + 1) + g * reps + (1.0 if g else 0.0)
    parts = []
    if p:
        parts.append("%g prompt x %d reps+warmup" % (p, reps))
    if g:
        parts.append("%g gen x %d reps + 1 warmup tok" % (g, reps))
    return (total or None), " + ".join(parts)


def bench_compute_s(row, reps):
    """Seconds of one bench row that were actually inside llama-bench's timed
    loops, from its own avg_ns (mean nanoseconds per repetition), plus the
    unmeasured warm-up: a full prompt pass for a pp test, one token for a tg
    test. Model load, CUDA init and the weight upload all precede the first
    timed loop, so this is the length of the TRAILING slice of the marker window
    that processed tokens. None when the ref does not emit avg_ns.
    """
    ns = _f(row, "avg_ns")
    if not ns:
        return None
    p = _f(row, "n_prompt") or 0.0
    g = _f(row, "n_gen") or 0.0
    if p:
        warm_ns = ns
    elif g:
        warm_ns = ns / g
    else:
        warm_ns = 0.0
    return round((ns * reps + warm_ns) / 1e9, 3)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("power_csv")
    ap.add_argument("--marker", help="window label to extract (e.g. prefill, decode)")
    ap.add_argument("--from", dest="t_from", help="window start (unix or ISO UTC)")
    ap.add_argument("--to", dest="t_to", help="window end (unix or ISO UTC)")
    ap.add_argument("--idle-mw", default="auto",
                    help="idle floor in mW to subtract for 'active' energy, or "
                         "'auto' (default) to take each window's floor from the "
                         "idle interval that preceded it in this CSV, or 'none' "
                         "to report total energy only")
    ap.add_argument("--bench", help="llama-bench CSV to join for energy per token")
    ap.add_argument("--reps", type=int, default=None,
                    help="llama-bench -r used for this window (default 1). The CSV's "
                         "n_prompt/n_gen are per-repetition, so this MUST match the "
                         "-r the window was recorded with; too low a value shortens "
                         "the divisor and every J/token comes out too HIGH.")
    ap.add_argument("--out", help="write the joined per-window rows to this CSV")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    rows = load_samples(args.power_csv)

    if args.bench and not (args.marker or (args.t_from and args.t_to)):
        sys.exit(
            "--bench needs a window: pass --marker LABEL, or --from/--to.\n"
            "  With neither, the WHOLE power CSV becomes one window -- the idle\n"
            "  baseline and every cooldown included -- and it is then joined to\n"
            "  bench row 0. That prints a confident J/tok under a header reading\n"
            "  marker=idle, and it is wrong by several times: measured 0.001672\n"
            "  J/tok against a true 0.00651 on a five-run sweep.\n"
            "  A sweep's labels are phase:case_id:NNN, and --marker matches that\n"
            "  prefix, so --marker prefill or --marker prefill:thread_sweep:027.")

    if args.marker:
        wins = windows_by_marker(rows, args.marker)
        if not wins:
            sys.exit("no samples carry marker %r (is the logger reading the marker file?)"
                     % args.marker)
    elif args.t_from and args.t_to:
        try:
            t_from, t_to = parse_time(args.t_from), parse_time(args.t_to)
        except ValueError as exc:
            sys.exit(str(exc))
        wins = window_between(rows, t_from, t_to)
        if not wins:
            sys.exit("no samples inside the requested interval")
    else:
        wins = [rows]

    # Resolved ONCE per window, before anything is summarised, so the whole-window
    # figure, the compute slice and the printed source all use the same number.
    # "auto" is per window; an explicit value applies to all of them, which is
    # what reproducing a published row needs.
    spec = args.idle_mw.strip().lower() if isinstance(args.idle_mw, str) else args.idle_mw
    if spec in ("none", "", "0", 0, 0.0):
        floors = [(0.0, "none") for _ in wins]
    elif spec == "auto":
        ivs = idle_intervals(rows)
        floors = []
        for w in wins:
            mw, src = idle_floor_at(ivs, _f(w[0], "ts_unix") or 0.0)
            if mw is None:
                print("note: no idle samples precede the window at %s -- reporting "
                      "total energy only. Pass --idle-mw N to subtract a floor, or "
                      "use the row's own idle_mw from the joined CSV."
                      % (w[0].get("ts_utc") or "?"), file=sys.stderr)
                mw = 0.0
            floors.append((mw, src))
    else:
        try:
            floors = [(float(spec), "fixed")] * len(wins)
        except ValueError:
            sys.exit("--idle-mw must be a number, 'auto' or 'none' (got %r)" % args.idle_mw)

    results = [summarise(w, floors[i][0]) for i, w in enumerate(wins)]
    for i, res in enumerate(results):
        res["idle_mw_source"] = floors[i][1]

    if args.bench:
        # A window flag is not the same as a window. `--from 0 --to 9999999999`
        # satisfied the check above and reproduced, verbatim, the wrong number
        # the check exists to prevent: 0.001672 J/tok against a true 0.006514,
        # under a header reading marker=idle. So validate what was SELECTED:
        # idle samples inside a join window mean the baseline and the cooldowns
        # are in the numerator.
        for i, w in enumerate(wins):
            idle_n = sum(1 for r in w if (r.get("marker") or "").strip() in ("", "idle"))
            if idle_n:
                sys.exit(
                    "window %d of %d covers %d idle sample(s) out of %d -- refusing "
                    "to join.\n"
                    "  Idle time in the window puts the baseline and the cooldowns "
                    "into the\n  energy that is about to be divided by this run's "
                    "tokens.\n"
                    "  Select the run itself: --marker %s, or a phase prefix like "
                    "--marker prefill."
                    % (i, len(wins), idle_n, len(w),
                       next(((r.get("marker") or "").strip() for r in w
                             if (r.get("marker") or "").strip() not in ("", "idle")),
                            "<phase>:<case_id>:<NNN>")))

        bench = load_bench(args.bench)
        reps = args.reps if args.reps is not None else 1
        if args.reps is None:
            print("note: --reps not given, assuming 1, but bench/matrix.json "
                  "defaults to repetitions=3.\n"
                  "  If the bench row carries avg_ns the compute trim rescales the "
                  "numerator by\n  the same factor as the divisor, so J/token "
                  "barely moves -- but tokens,\n  compute_s, load_s and load_joules "
                  "all come out wrong by about that factor.\n"
                  "  If it does NOT carry avg_ns there is no trim, and J/token "
                  "itself comes out\n  too HIGH: about 2x for a 512-token prefill "
                  "row at -r 3, about 3x for a\n  128-token decode row.\n"
                  "  Either way, pass the -r the window was recorded with; the "
                  "joined CSV's\n  `repetitions` column has it.",
                  file=sys.stderr)
        # Pair the i-th window with the i-th bench row. llama-bench emits its
        # rows in run order and the driver writes one marker window per run, so
        # the ordering IS the join key -- which makes it silent when it is
        # wrong. A mixed -p N -n M invocation puts two rows against one window,
        # so window 1 gets run 0's decode row, window 2 gets run 1's prefill
        # row, and the tail of the CSV is never read at all. Nothing about the
        # output looks unusual: it prints a confident J/tok under a label from a
        # different run. Refuse instead of zipping past the mismatch.
        if len(bench) != len(results):
            sys.exit(
                "cannot join: %d power window(s) but %d bench row(s) in %s.\n"
                "  The join is positional -- one window per bench row -- so an\n"
                "  unequal count would silently pair each window with another\n"
                "  run's numbers.\n"
                "  %s"
                % (len(results), len(bench), args.bench,
                   "Narrow the window with --marker, or pass the per-case CSV "
                   "(<base>_NNN.csv) rather than a concatenated one."
                   if len(bench) < len(results) else
                   "This CSV holds more than one test per window -- a mixed "
                   "-p N -n M invocation. Split the case in bench/matrix.json; "
                   "see docs/10_measurement_protocol.md section 2."))

        # The count check above proves there are as many rows as windows. It
        # cannot prove they are the SAME runs: two tg128 rows joined to the two
        # prefill windows of a real sweep passed it, and printed a decode
        # throughput under marker=prefill:split_prefill:001. So compare identity
        # where the data carries it -- the marker encodes the phase the driver
        # recorded, and the row's own counts say what the test was.
        for i, res in enumerate(results):
            marker = (res.get("marker") or "").strip()
            want = marker.split(":", 1)[0] if marker else ""
            bp = _f(bench[i], "n_prompt") or 0.0
            bg = _f(bench[i], "n_gen") or 0.0
            got = ("combined" if (bp and bg) else
                   "prefill" if bp else "decode" if bg else "")
            if want in ("prefill", "decode", "combined") and got and want != got:
                sys.exit(
                    "window %d is marked %r but bench row %d is a %s test "
                    "(n_prompt=%g n_gen=%g).\n"
                    "  The join is positional, so an equal count of windows and "
                    "rows is not\n  proof they are the same runs. Pass the "
                    "per-case CSV that belongs to this\n  window -- "
                    "results/<base>_NNN.csv, where NNN is the marker's run index."
                    % (i, marker, i, got, bp, bg))

        for i, res in enumerate(results):
            b = bench[i]
            tps = pick(b, "avg_ts", "t/s", "tokens_per_second")
            res["bench_model"] = pick(b, "model_filename", "model", "model_type")
            # "test" is llama-bench's own label (pp512 / tg128). The n_prompt
            # fallback must skip the "0" a decode row carries, or every decode
            # window is labelled "0".
            res["bench_test"] = pick(b, "test", "n_prompt", skip_zero=True)
            res["bench_tps"] = float(tps) if tps else None
            joules = res["module"]["joules"]
            active = res.get("module_active_joules")
            res["energy_basis"] = "whole-window (includes model load)"

            # The marker window opens before llama-bench starts, so model load
            # off the SD card and CUDA context creation sit inside it -- seconds
            # of multi-watt work that produced no tokens. All of it precedes the
            # first timed loop, so trim to the trailing compute slice. Raw
            # whole-window joules stays on the row either way.
            cs = bench_compute_s(b, reps)
            idle_mw = floors[i][0]
            if cs and 0.0 < cs < res["duration_s"]:
                cut = res["end_unix"] - cs
                sub = [r for r in wins[i] if (_f(r, "ts_unix") or 0.0) >= cut]
                if len(sub) >= 2:
                    ce = summarise(sub, idle_mw)
                    # Recover the energy the sampled edges drop, by
                    # interpolating the counter at the exact slice bounds.
                    e_cut = energy_at(wins[i], cut)
                    e_end = energy_at(wins[i], res["end_unix"])
                    if e_cut is not None and e_end is not None:
                        ce["module"]["joules"] = round(e_end - e_cut, 4)
                        ce["duration_s"] = round(res["end_unix"] - cut, 3)
                        if idle_mw:
                            ce["module_active_joules"] = round(
                                ce["module"]["joules"]
                                - idle_mw / 1000.0 * ce["duration_s"], 4)
                    if ce["module"]["joules"] is not None:
                        # Report the MEASURED span of the trimmed slice, and keep
                        # the whole-window active figure under its own name --
                        # leaving module_active_joules (whole window) printed next
                        # to a per-token number derived from the compute slice
                        # made the two differ by over 100%.
                        res["compute_s"] = ce["duration_s"]
                        res["compute_s_nominal"] = cs
                        res["compute_joules"] = ce["module"]["joules"]
                        res["compute_mean_mw"] = ce["module"]["mean_mw"]
                        res["compute_active_joules"] = ce.get("module_active_joules")
                        res["load_s"] = round(res["duration_s"] - ce["duration_s"], 3)
                        res["load_joules"] = round(joules - ce["module"]["joules"], 4)
                        res["energy_basis"] = "compute"
                        joules = ce["module"]["joules"]
                        active = ce.get("module_active_joules")

            tokens, basis = bench_tokens(b, reps)
            if tokens is not None:
                res["bench_reps"] = reps
                res["tokens_basis"] = basis
                res["per_token_status"] = "ok"
            elif basis:
                # bench_tokens refused deliberately (a fused pp+tg row). The
                # avg_ts fallback below must NOT rescue it -- multiplying a
                # fused throughput by the window duration would manufacture
                # exactly the unattributable per-token number being refused.
                res["tokens_basis"] = basis
                res["per_token_status"] = "refused_combined_request"
            elif res["bench_tps"]:
                # Multiply by the SAME interval the numerator covers. Using the
                # whole window here while joules had already been trimmed to the
                # compute slice made J/token short by exactly
                # duration_s / compute_s -- and it was reported as "ok".
                span = res.get("compute_s") or res["duration_s"]
                tokens = res["bench_tps"] * span
                res["tokens_basis"] = ("avg_ts x %.3f s (%s) -- ESTIMATED, the bench "
                                       "CSV carried no token columns"
                                       % (span, "compute slice" if res.get("compute_s")
                                          else "whole window"))
                # Never "ok": a throughput-derived divisor is an estimate, and a
                # reader must be able to tell it from a counted one.
                res["per_token_status"] = "estimated_from_throughput"
            else:
                res["per_token_status"] = "no_token_count"
            if joules is not None and tokens:
                res["tokens"] = tokens
                res["joules_per_token"] = round(joules / tokens, 6)
                res["tokens_per_joule"] = round(tokens / joules, 4) if joules else None
                # Wh per million tokens: the unit the energy tables use, because
                # joules/token at the edge is an unreadable string of zeros.
                res["wh_per_mtok"] = round(joules / tokens * 1e6 / 3600.0, 2)
                if active is not None:
                    res["active_joules_per_token"] = round(active / tokens, 6)

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
    else:
        for i, r in enumerate(results):
            print("window %d  marker=%s  %.2f s  %d samples"
                  % (i, r["marker"] or "-", r["duration_s"], r["samples"]))
            print("  module  %8s J   mean %8s mW   peak %8s mW"
                  % (r["module"]["joules"], r["module"]["mean_mw"], r["module"]["peak_mw"]))
            print("  gpu     %8s J   cpu  %8s J"
                  % (r["gpu_rail"]["joules"], r["cpu_rail"]["joules"]))
            if "module_active_joules" in r:
                print("  active  %8s J   (idle floor %.0f mW removed, whole window)"
                      % (r["module_active_joules"], r["idle_mw"]))
            if r.get("compute_active_joules") is not None:
                print("  active  %8s J   (idle floor removed, compute slice -- this "
                      "is what the per-token figures use)" % r["compute_active_joules"])
            if "temp_rise_c" in r:
                # NOT a contamination test: a within-window rise goes to ~0
                # exactly when the board has reached a hot steady state, so the
                # worst case passes it. The between-run figure is temp_drift_c
                # in the joined CSV; see docs/10 section 4.
                flag = ("  <-- clocks moved inside this window" if r["temp_rise_c"] > 8
                        else "")
                print("  temp    %.1f -> %.1f C (max %.1f)%s"
                      % (r["temp_start_c"], r["temp_end_c"], r["temp_max_c"], flag))
            if r.get("swap_peak_mb"):
                print("  swap    peak %d MB" % r["swap_peak_mb"])
            st = r.get("per_token_status")
            if st not in (None, "ok") and "joules_per_token" not in r:
                print("  %s  %.2f tok/s   no J/tok: %s"
                      % (r.get("bench_test", "?"), r.get("bench_tps") or 0, st))
                if r.get("tokens_basis"):
                    print("          %s" % r["tokens_basis"])
            elif st not in (None, "ok"):
                print("  !! %s -- the figure below rests on an estimated divisor"
                      % st)
            if "joules_per_token" in r:
                print("  %s  %.2f tok/s  %.6f J/tok  %.2f Wh/Mtok"
                      % (r.get("bench_test", "?"), r["bench_tps"] or 0,
                         r["joules_per_token"], r["wh_per_mtok"]))
                print("  tokens  %g   (%s)" % (r["tokens"], r.get("tokens_basis", "?")))
                print("  basis   %s" % r.get("energy_basis", "?"))
                if "compute_joules" in r:
                    print("  compute %8s J over %.3f s   load %s J over %.3f s"
                          % (r["compute_joules"], r["compute_s"],
                             r["load_joules"], r["load_s"]))
            print()

    if args.out:
        keys = []
        flat = []
        for r in results:
            row = {}
            for k, v in r.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        row["%s_%s" % (k, k2)] = v2
                else:
                    row[k] = v
            flat.append(row)
            for k in row:
                if k not in keys:
                    keys.append(k)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for row in flat:
                w.writerow(row)
        print("wrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
