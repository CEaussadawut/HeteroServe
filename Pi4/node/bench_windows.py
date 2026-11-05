#!/usr/bin/env python3
"""Turn one llama-bench invocation into a nested energy-window record.

Runs AFTER llama-bench exits, outside the measured window, so its own cost is irrelevant.
Writes exactly ONE json line to stdout; scripts/06_bench_local.sh appends it to
windows.jsonl.

WHY THIS EXISTS
---------------
The energy integral used to run over the whole llama-bench PROCESS. A process does far
more than the repetitions the token count is derived from: process start,
ggml_backend_load_all, the mmap/microSD page-in of a 0.7-2.2 GB GGUF, context creation, a
FULL-LENGTH warmup prefill (upstream warms pp with test_prompt(n_prompt), not 32 tokens)
and teardown. None of that puts a token in the denominator, so joules-per-token was
overstated -- and by a different amount for pp than tg, and for Q8_0 than Q4_K_M, which is
worse than a constant offset because it distorts exactly the comparisons the study makes.

This module cuts the integral at repetition boundaries instead:

  * the repetition START comes from the node's CLOCK_REALTIME stamp on llama-bench's
    `--progress` line (stamped by node/stamp_lines.py as it arrives);
  * the repetition LENGTH comes from `samples_ns` in llama-bench's own JSON, measured on
    llama.cpp's monotonic clock inside the process.

Two independent clocks bound every sub-window and their disagreement is recorded, so the
alignment is a checkable fact rather than an assumption.

Verified against upstream tools/llama-bench/llama-bench.cpp: `uint64_t t_start =
get_time_ns();` is line 2440 and the progress fprintf is 2443-2446, i.e. the stamp is
taken a few microseconds AFTER the interval it marks begins. Every repetition window is
therefore shifted LATE by the fprintf + pipe write + reader wake-up, order 0.1-10 ms
against repetitions that are seconds long. The window LENGTH is exact; only its phase is
shifted. min_gap_s / max_gap_s bound that lag empirically, per run.

FAIL-CLOSED CONTRACT
--------------------
Every gate that does not pass degrades to the whole-process window WITH window_kind
"process" and a window_fallback_reason naming the gate. This module must NEVER raise: a
sweep that dies at hour six because a regex missed is worse than a flagged row, and a row
that silently means something other than its label is worse than either. That failure mode
-- a well-formed artefact containing a number that means something else -- is the one this
whole exercise is about.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

# Verified verbatim against upstream llama-bench.cpp (lines 2375, 2388, 2417, 2434, 2444,
# 2457). "starting" (line 2281) has no "run" and deliberately does not match.
PROGRESS_RE = re.compile(
    r"llama-bench:\s*benchmark\s+\d+/\d+:\s*"
    r"(warmup\s+prompt|warmup\s+generation|depth|prompt|generation)\s+run"
    r"(?:\s+(\d+)\s*/\s*(\d+))?",
    re.I,
)

# The stamp sits just after get_time_ns(), so a stamp may legitimately fall a few ms late
# but must never imply overlapping repetitions.
GAP_MIN_S = -0.050
GAP_MAX_S = 5.0


def _f(value, default=""):
    """float() that never raises, and never returns a non-finite value.

    NaN/Infinity matter here because this module's output is written with json.dumps, and
    Python emits them as the bare tokens `NaN` / `Infinity`. Those are NOT valid JSON
    (RFC 8259). Python reads them back happily, so the corruption is invisible from inside
    this codebase -- but `jq` silently rewrites NaN to `null`, and strict parsers reject
    the line outright. Either way the primary artefact is damaged.

    This is reachable in practice: scripts/lib.sh soc_temp_c() falls back to the literal
    string "nan" when the thermal zone cannot be read, and float("nan") succeeds.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):   # NaN or +/-Inf
        return default
    return out


def parse_stamped(path):
    """Read the stamped stderr TSV into (warmup_stamps, rep_stamps, depth_seen).

    Each line is "%.6f\\t<original stderr text>". Split on the FIRST tab only: llama.cpp
    writes tabs of its own in the table output and they must not shift the fields.
    """
    warmups, reps, depth_seen = [], [], False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                head, sep, text = line.partition("\t")
                if not sep:
                    continue
                ts = _f(head, None)
                if ts is None:
                    continue
                m = PROGRESS_RE.search(text)
                if not m:
                    continue
                kind = re.sub(r"\s+", " ", m.group(1).strip().lower())
                if kind == "depth":
                    depth_seen = True
                elif kind.startswith("warmup"):
                    warmups.append(ts)
                else:                       # "prompt" / "generation" -- a timed repetition
                    idx = int(m.group(2)) if m.group(2) else None
                    reps.append((idx, ts))
    except OSError:
        return [], [], False
    return warmups, reps, depth_seen


def load_samples_ns(path):
    """samples_ns of the single bench entry, or None if the JSON is unusable/multi-entry."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            entries = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list) or len(entries) != 1:
        return None                                   # G1
    samples = entries[0].get("samples_ns")
    if not isinstance(samples, list) or not samples:
        return None
    out = []
    for s in samples:
        v = _f(s, None)
        if v is None or v <= 0:
            return None
        out.append(v)
    return out


def derive(args):
    """Return (extra_fields, fallback_reason). fallback_reason "" means the gates passed."""
    if int(_f(args.rc, 1)) != 0:
        return {}, "G8:bench-exit-%s" % args.rc                       # G8

    samples_ns = load_samples_ns(args.bench_json)
    if samples_ns is None:
        return {}, "G1:bench-json-unusable-or-multi-entry"            # G1

    warmups, reps, depth_seen = parse_stamped(args.stamped)
    if depth_seen:
        return {}, "G7:depth-run-marker-present"                      # G7

    want = int(_f(args.reps, 0))
    if len(reps) != len(samples_ns) or (want and len(reps) != want):
        return {}, "G2:marker-count-%d-vs-samples-%d-vs-reps-%d" % (
            len(reps), len(samples_ns), want)                         # G2
    printed = [i for i, _ in reps]
    if any(i is None for i in printed) or printed != list(range(1, len(reps) + 1)):
        return {}, "G2:marker-indices-not-1..R"                       # G2

    stamps = [ts for _, ts in reps]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        return {}, "G3:stamps-not-increasing"                         # G3

    gaps = []
    for i in range(len(stamps) - 1):
        gap = stamps[i + 1] - (stamps[i] + samples_ns[i] / 1e9)
        if gap < GAP_MIN_S:
            return {}, "G4:repetitions-overlap-%.4fs" % gap           # G4
        if gap > GAP_MAX_S:
            return {}, "G4:gap-too-large-%.3fs" % gap                 # G4
        gaps.append(gap)

    t_meas_start = stamps[0]
    t_meas_end = stamps[-1] + samples_ns[-1] / 1e9
    t_proc_end = _f(args.t_end, None)
    if t_proc_end is None or t_meas_end > t_proc_end:
        return {}, "G5:measured-end-after-process-end"                # G5

    t_proc_start = _f(args.t_start, None)
    t_warmup_start = warmups[0] if warmups else None
    if t_warmup_start is None or not (t_warmup_start < t_meas_start):
        return {}, "G6:no-warmup-marker-before-first-repetition"      # G6
    if t_proc_start is None or t_warmup_start < t_proc_start:
        return {}, "G6:warmup-marker-before-process-start"            # G6

    rep_span_s = t_meas_end - t_meas_start
    rep_busy_s = sum(samples_ns) / 1e9
    rep_windows = [
        {"i": i + 1, "t_start": round(stamps[i], 6),
         "t_end": round(stamps[i] + samples_ns[i] / 1e9, 6), "ns": samples_ns[i]}
        for i in range(len(stamps))
    ]
    return {
        "window_kind": "reps-span",
        "window_fallback_reason": "",
        "t_meas_start": round(t_meas_start, 6),
        "t_meas_end": round(t_meas_end, 6),
        "t_load_start": round(t_proc_start, 6),
        "t_load_end": round(t_warmup_start, 6),
        "t_warmup_start": round(t_warmup_start, 6),
        "t_warmup_end": round(t_meas_start, 6),
        "t_teardown_start": round(t_meas_end, 6),
        "t_teardown_end": round(t_proc_end, 6),
        "reps": rep_windows,
        "rep_span_s": round(rep_span_s, 6),
        "rep_busy_s": round(rep_busy_s, 6),
        "rep_gap_s": round(rep_span_s - rep_busy_s, 6),
        "energy_duty": round(rep_busy_s / rep_span_s, 4) if rep_span_s > 0 else "",
        "min_gap_s": round(min(gaps), 6) if gaps else 0.0,
        "max_gap_s": round(max(gaps), 6) if gaps else 0.0,
    }, ""


def build(args):
    # t_start / t_end keep their ORIGINAL meaning -- the raw process window -- forever.
    # windows.jsonl is the primary artefact; silently redefining a field in it would be
    # the same class of defect this fix exists to remove. Everything else is additive.
    rec = {
        "schema": 2,
        "run": args.run, "node": args.node, "model": args.model, "phase": args.phase,
        "n": int(_f(args.n, 0)), "threads": int(_f(args.threads, 0)),
        # How many repetitions were ASKED for. Recorded on every record, including a
        # fallback one, because the derived `reps` list exists only when all gates pass --
        # so without this a fallback run cannot say what R it was, and any consumer keying
        # off `reps` silently loses it rather than reporting it as excluded.
        "reps_declared": int(_f(args.reps, 0)),
        "t_start": _f(args.t_start, ""), "t_end": _f(args.t_end, ""),
        "window_kind": "process", "window_fallback_reason": "not-derived",
        "idle_start": _f(args.idle_start, ""), "idle_end": _f(args.idle_end, ""),
        "rc": int(_f(args.rc, 1)),
        "json": args.json, "stderr_tsv": args.stderr_tsv,
        "temp_start_c": _f(args.temp_start_c, ""), "temp_end_c": _f(args.temp_end_c, ""),
        "throttled_start": args.throttled_start, "throttled_end": args.throttled_end,
    }
    try:
        extra, reason = derive(args)
    except Exception as exc:                          # pragma: no cover - must never raise
        extra, reason = {}, "exception:%s" % type(exc).__name__
    if extra:
        rec.update(extra)
    else:
        rec["window_fallback_reason"] = reason
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "node", "model", "phase", "n", "threads", "reps",
                 "t-start", "t-end", "rc", "json", "stderr-tsv", "bench-json",
                 "stamped", "idle-start", "idle-end", "temp-start-c", "temp-end-c",
                 "throttled-start", "throttled-end"):
        p.add_argument("--" + name, default="")
    args = p.parse_args(argv)
    rec = build(args)
    # allow_nan=False is the belt to _f()'s braces: writing an invalid-JSON line into the
    # primary artefact is worse than losing one field, so if a non-finite value ever gets
    # this far, drop to a minimal record that still parses and says what happened.
    try:
        line = json.dumps(rec, allow_nan=False)
    except ValueError:
        rec = {"schema": 2, "run": args.run, "node": args.node, "model": args.model,
               "phase": args.phase, "json": args.json,
               "window_kind": "process",
               "window_fallback_reason": "non-finite-value-in-record"}
        line = json.dumps(rec, allow_nan=False)
    sys.stdout.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
