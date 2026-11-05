#!/usr/bin/env python3
"""Cross-check the repetition energy window without trusting any timestamp.

    python3 node/bench_validate.py <validate_run_dir> [--energy energy.json] [--out report.json]

WHY
---
scripts/06_bench_local.sh derives the energy window from llama-bench's `--progress` stderr
markers, stamped with the node's wall clock. That is a good method, but it rests on three
things a reader is entitled to doubt: that the marker really is printed where upstream's
source says it is, that the node's clock agrees with the DAQ's, and that nothing reordered
the stream in between.

This module checks the same split a completely different way, using no timestamps at all.
Energy over a whole llama-bench PROCESS is affine in the repetition count:

    E_process(R) = E_fixed + R * E_rep

`scripts/06_bench_local.sh validate` runs each cell at R = 1, 3, 5. Fitting a straight line
through those three points recovers:

    slope     = E_rep    -- energy of ONE repetition
    intercept = E_fixed  -- startup + model load + warmup + teardown

Both are then compared against what the sub-windows claim. If the two routes agree inside
the DAQ's stated accuracy, the window derivation is corroborated by a method that shares
none of its assumptions. If they disagree, something is wrong and the disagreement is the
finding -- this module never "reconciles" them.

WHAT IT DOES NOT DO
-------------------
It does not compute energy. The Pi has no power data: the INA226 stream lives on the
control node. Supply the per-run integrals via --energy (see ENERGY INPUT below), which is
what `heteroctl join` can produce. With no --energy this reports only the timing-side
consistency of the sweep, which is still worth having but is NOT the cross-check.

Three points and two parameters leaves one degree of freedom, so the residual is a real but
weak goodness-of-fit signal. Treat a large residual as "investigate", not as a measurement.

ENERGY INPUT
------------
A JSON object mapping the window record's `json` field (the per-run filename) to the two
integrals for that run, which are NOT interchangeable:

    {"tinyllama-1.1b-q4km_pp128_t4_r1.json": {"process": 41.2, "reps": 12.1},
     "tinyllama-1.1b-q4km_pp128_t4_r3.json": {"process": 66.4, "reps": 36.2}, ...}

  process -- integrated over [t_start, t_end], the WHOLE llama-bench process. This is
             what the affine fit consumes: only the whole-process energy is affine in R,
             because E_fixed is paid once per process.
  reps    -- integrated over [t_meas_start, t_meas_end], the repetition span. Divided by R
             this is the sub-window's claim about one repetition.

A bare number is accepted and treated as `process`; the fit then still works but the
cross-check cannot run, because dividing a whole-process energy by R does not give the
per-repetition energy -- it still carries E_fixed/R. Comparing those two would be
precisely the kind of incommensurable-quantities error this project keeps finding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Agreement threshold between the two routes.
#
# This is NOT the +-5% figure from the success criteria -- that is the study's acceptance
# target for the final number, not a measure of what this comparison can resolve, and using
# it here would let a 4% disagreement pass unremarked on an instrument good to ~1%.
# docs/02_power_instrumentation.md:63 puts the combined uncertainty of the measurement chain
# at 0.5-1.1% (shunt tolerance, INA226 gain error, multimeter accuracy).
#
# Both routes integrate the SAME power samples, so instrument scale error is common-mode
# and largely cancels in the comparison; what is left is the window derivation itself.
# 1.1% -- the top of the documented chain uncertainty -- is therefore a defensible ceiling
# on "these two methods agree". Loosen it only with a stated reason.
DAQ_TOLERANCE = 0.011


def load_windows(run_dir: Path) -> list:
    path = run_dir / "windows.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def cell_key(rec: dict) -> tuple:
    """A cell is one (model, phase, n, threads) -- the thing swept over R."""
    return (rec.get("model"), rec.get("phase"), rec.get("n"), rec.get("threads"))


def reps_of(rec: dict) -> int:
    """Repetition count for this run, or 0 if the record cannot say.

    Prefers the derived sub-window list, which is authoritative -- it is what actually
    happened. But `reps` is written ONLY when every gate in bench_windows.py passed, so a
    fallback record has no `reps` key at all. Falling back to `reps_declared` is what keeps
    such a run VISIBLE: gating inclusion on `reps` alone silently deleted every fallback
    run from the analysis, and the "these runs fell back" warning then never fired because
    the records it was meant to warn about had already been dropped.
    """
    reps = rec.get("reps")
    if isinstance(reps, list) and reps:
        return len(reps)
    declared = rec.get("reps_declared")
    if isinstance(declared, int) and declared > 0:
        return declared
    return 0


def _energies(value):
    """(process_joules, reps_span_joules) from one --energy entry; None where absent.

    A bare number is the whole-process integral, which is the only one the affine fit can
    use. It deliberately does NOT populate the reps figure: a whole-process energy divided
    by R is not the per-repetition energy, and silently treating it as one would
    manufacture agreement out of nothing.
    """
    if isinstance(value, (int, float)):
        return float(value), None
    if isinstance(value, dict):
        proc = value.get("process")
        reps = value.get("reps")
        return (float(proc) if isinstance(proc, (int, float)) else None,
                float(reps) if isinstance(reps, (int, float)) else None)
    return None, None


def fit_line(points: list) -> dict:
    """Least-squares fit of y = intercept + slope*x over >= 2 (x, y) points."""
    n = len(points)
    if n < 2:
        return {}
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    denom = n * sxx - sx * sx
    if denom == 0:                      # every R identical -- no slope is defined
        return {}
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    resid = [y - (intercept + slope * x) for x, y in points]
    return {
        "slope": slope,
        "intercept": intercept,
        "max_abs_residual": max(abs(r) for r in resid),
        "points": n,
    }


def analyse(run_dir: Path, energy: dict) -> dict:
    cells = {}
    unusable = []
    for rec in load_windows(run_dir):
        r = reps_of(rec)
        if not r:
            # Neither a derived sub-window list nor a declared count: the record cannot be
            # placed on the R axis at all. Counted and reported, never silently discarded.
            unusable.append(rec.get("json") or "<unnamed>")
            continue
        cells.setdefault(cell_key(rec), []).append((r, rec))

    report = {"run_dir": str(run_dir), "tolerance": DAQ_TOLERANCE, "cells": [], "problems": []}
    if unusable:
        report["unusable_records"] = sorted(unusable)
        report["problems"].append(
            "%d window record(s) carry neither derived sub-windows nor reps_declared and "
            "were excluded: %s" % (len(unusable), ", ".join(sorted(unusable))))

    for key, entries in sorted(cells.items(), key=lambda kv: [str(x) for x in kv[0]]):
        model, phase, n, threads = key
        entry = {
            "model": model, "phase": phase, "n": n, "threads": threads,
            "reps_observed": sorted(r for r, _ in entries),
        }

        # timing-side consistency: rep_busy_s must scale with R
        per_rep = []
        for r, rec in entries:
            busy = rec.get("rep_busy_s")
            if isinstance(busy, (int, float)) and r:
                per_rep.append(busy / r)
        if per_rep:
            entry["mean_s_per_rep"] = round(sum(per_rep) / len(per_rep), 6)
            spread = (max(per_rep) - min(per_rep)) / entry["mean_s_per_rep"] if entry["mean_s_per_rep"] else 0
            entry["s_per_rep_spread"] = round(spread, 4)
            if spread > 0.15:
                report["problems"].append(
                    "%s %s%s t%s: seconds-per-repetition varies %.1f%% across R -- the cell is "
                    "not reproducible, so an energy fit over it would be meaningless"
                    % (model, phase, n, threads, 100 * spread))

        fallbacks = [r for r, rec in entries if rec.get("window_kind") != "reps-span"]
        if fallbacks:
            entry["fell_back_at_R"] = sorted(fallbacks)
            report["problems"].append(
                "%s %s%s t%s: window derivation fell back to whole-process at R=%s -- "
                "excluded from the fit" % (model, phase, n, threads, sorted(fallbacks)))

        usable = [(r, rec) for r, rec in entries if rec.get("window_kind") == "reps-span"]

        # Only the WHOLE-PROCESS energy is affine in R -- E_fixed is paid once per process.
        # Feeding reps-span energies to the fit would give a line through the origin and a
        # meaningless intercept.
        points, claims = [], []
        for r, rec in usable:
            proc, reps_j = _energies(energy.get(rec.get("json")))
            if proc is not None:
                points.append((float(r), proc))
            if reps_j is not None and r:
                claims.append(reps_j / r)

        if len(points) >= 2:
            fit = fit_line(points)
            if fit:
                entry["fit"] = {k: round(v, 6) if isinstance(v, float) else v
                                for k, v in fit.items()}
                # Two points and two parameters is zero degrees of freedom: the line passes
                # through both exactly and max_abs_residual is 0.0 by construction, not
                # because the data agreed. Say so, or the report reads as a clean fit.
                if fit["points"] < 3:
                    entry["fit"]["residual_is_meaningless"] = True
                    report["problems"].append(
                        "%s %s%s t%s: only %d usable point(s) -- the fit has no degrees of "
                        "freedom, so its zero residual says nothing about goodness of fit"
                        % (model, phase, n, threads, fit["points"]))
                if claims:
                    # The sub-windows' own claim: energy inside the repetition span, over R.
                    sub = sum(claims) / len(claims)
                    entry["subwindow_j_per_rep"] = round(sub, 6)
                    if fit["slope"]:
                        dev = abs(sub - fit["slope"]) / abs(fit["slope"])
                        entry["agreement_deviation"] = round(dev, 4)
                        entry["agrees"] = dev <= DAQ_TOLERANCE
                        if not entry["agrees"]:
                            report["problems"].append(
                                "%s %s%s t%s: the R=1,3,5 fit says %.3f J per repetition but "
                                "the sub-windows say %.3f J (%.1f%% apart, tolerance %.0f%%) -- "
                                "the window derivation and the method of differences DISAGREE"
                                % (model, phase, n, threads, fit["slope"], sub,
                                   100 * dev, 100 * DAQ_TOLERANCE))
                else:
                    report["problems"].append(
                        "%s %s%s t%s: fit ran but no reps-span energies were supplied, so the "
                        "cross-check did NOT happen -- supply {\"process\":..,\"reps\":..}"
                        % (model, phase, n, threads))
        elif energy:
            report["problems"].append(
                "%s %s%s t%s: only %d usable energy point(s); need at least 2 to fit"
                % (model, phase, n, threads, len(points)))

        report["cells"].append(entry)

    if not energy:
        report["note"] = ("no --energy supplied: timing-side consistency only. The "
                          "cross-check of the energy window has NOT been performed.")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="a *-validate sweep directory")
    ap.add_argument("--energy", type=Path, default=None,
                    help="JSON map of per-run json filename -> joules")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.run_dir.is_dir():
        print("not a directory: %s" % args.run_dir, file=sys.stderr)
        return 2

    energy = {}
    if args.energy:
        try:
            energy = json.loads(args.energy.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print("could not read --energy: %s" % exc, file=sys.stderr)
            return 2
        if not isinstance(energy, dict):
            print("--energy must be a JSON object of filename -> joules", file=sys.stderr)
            return 2

    report = analyse(args.run_dir, energy)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print("wrote %s" % args.out)
    else:
        print(text)

    for p in report["problems"]:
        print("PROBLEM: " + p, file=sys.stderr)
    # A disagreement is the finding, so it is worth a non-zero exit for CI-style use.
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
