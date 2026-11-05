#!/usr/bin/env python3
"""Normalise llama-bench JSON into this node's device-characterisation rows.

THE SCHEMA IS NOT CLUSTER-WIDE, despite what this docstring used to claim. It said every
node emits the same CSV columns so the control node could concatenate them without
per-node special cases. That is not true and, as far as the tree records, never was:
`Pi5/bench/parse_bench.py` writes a different and non-overlapping `OUT_FIELDS`
(`cell_id`, `started_utc`, `prefill_tok_s`, `ttft_est_ms`, ...), and
`Jetson/bench/run_matrix.py` builds its fieldnames dynamically and carries its own
`energy_basis` column for what this file calls `energy_window`. The three worker nodes
cannot currently be concatenated into one table at all.

That is a real open defect (it is the audit's HIGH finding about incompatible results
CSVs), and it needs a decision about which schema wins before the cross-node
characterisation table can be built. It is recorded here rather than papered over,
because a docstring asserting an invariant that does not hold is exactly the kind of
plausible-but-false artefact this node exists to avoid producing.

    python3 node/bench_parse.py <run_dir> [--out results.csv]

<run_dir> is one directory produced by scripts/06_bench_local.sh: a set of llama-bench
*.json files plus windows.jsonl (the [t_start, t_end] wall-clock window of each run).
The window columns are what the energy collector joins against the INA226 stream to turn
tokens/s into joules per token, so they are carried through even though this script never
touches power data itself.
"""
from __future__ import annotations

import argparse
import csv
import re
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import envfile  # noqa: E402

ENV = envfile.load()

COLUMNS = [
    # identity
    "node_id", "node_role", "run_id", "llama_commit",
    # what was run
    "model", "model_type", "quant", "model_size_mb", "n_params",
    "phase", "n_prompt", "n_gen", "n_threads", "n_batch", "n_gpu_layers",
    # measured
    "tokens_per_s", "tokens_per_s_stddev", "avg_ms", "stddev_ms", "samples",
    # derived, and the field the placement rule actually consumes
    "ms_per_token", "per_layer_ms",
    # the join keys for energy (filled by the control node's collector)
    "t_start", "t_end", "temp_end_c", "throttled_end",
    # WHICH window t_start/t_end denote, and the decomposition behind it. The control
    # node integrates power over [t_start, t_end] and divides by n*reps; these columns
    # are how a reader -- or an examiner -- can see what is actually in that numerator.
    "energy_window", "t_proc_start", "t_proc_end",
    "rep_busy_s", "rep_gap_s", "energy_duty", "load_s", "warmup_s",
    # The idle window recorded immediately BEFORE this run -- same board temperature, same
    # fan duty, same minute. Carried through so a marginal (above-idle) figure can be
    # computed per row. Without these columns the only idle figure available downstream is
    # ThinkPad/control/analysis.py's session-wide mean, which pools every settle in the
    # sweep with the 120 s session baseline and so averages a cold board with a hot one.
    "idle_start", "idle_end",
    "energy_j", "joules_per_token", "mean_power_w",
    # validity
    "flags",
]

# Two spellings have to be recognised: the file naming used by config/models.txt
# ("...-q4km", "...-q8") and llama-bench's own model_type ("llama 1B Q4_K - Medium").
# Both are flattened to bare alphanumerics before matching, so separators cannot matter.
QUANT_HINTS = [
    ("Q4_K_M", ["Q4KM", "Q4KMEDIUM"]),
    ("Q4_K_S", ["Q4KS", "Q4KSMALL"]),
    ("Q5_K_M", ["Q5KM", "Q5KMEDIUM"]),
    ("Q6_K", ["Q6K"]),
    ("Q8_0", ["Q80", "Q8"]),
    ("Q4_0", ["Q40"]),
    ("BF16", ["BF16"]),
    ("F16", ["F16", "FP16"]),
]


def quant_of(name: str, model_type: str) -> str:
    flat = re.sub(r"[^A-Z0-9]", "", (name + " " + model_type).upper())
    for canonical, spellings in QUANT_HINTS:
        if any(s in flat for s in spellings):
            return canonical
    return model_type or "unknown"


# Fraction of the reported window that must have produced tokens. Repetitions are seconds
# long and the inter-repetition work is llama_memory_clear plus bookkeeping, so a correct
# run sits above 0.99. This is the ONLY definition of the duty gate -- an earlier copy in
# node/bench_windows.py was never referenced, so editing it changed nothing.
MIN_DUTY = 0.98

LIVE_BITS = 0x0000000F   # 0 under-voltage, 1 freq-capped, 2 throttled, 3 soft-temp-limit
STICKY_BITS = 0x000F0000  # the same four, latched since boot (see node/device_info.py)


def _throttle_delta(start_word, end_word):
    """(was_live_at_start, throttled_during_this_run) from the two vcgencmd words.

    The word is sticky, so `end != "0x0"` says only that the board throttled at some point
    since boot -- which, mid-sweep, is true of every row after the first blip. A row is
    invalidated by what happened DURING it: bits that were not set at the start and are set
    at the end (live or newly latched). Unparseable input is treated as "throttled", because
    the alternative is silently publishing a row that was never checked.
    """
    try:
        a = int(str(start_word), 16)
    except (TypeError, ValueError):
        a = 0 if not start_word else -1
    try:
        b = int(str(end_word), 16)
    except (TypeError, ValueError):
        b = -1
    if a < 0 or b < 0:
        return (False, True)
    newly = b & ~a                      # anything that appeared while this row was running
    return (bool(a & LIVE_BITS), bool(newly & (LIVE_BITS | STICKY_BITS)) or bool(b & LIVE_BITS))


def _delta(a, b):
    """b - a as a rounded float, or "" if either end is absent/unparseable.

    Used for the load/warmup sub-durations, which are present only on schema-2 window
    records. A legacy row must produce an empty cell, never a spurious 0.0.
    """
    try:
        return round(float(b) - float(a), 6)
    except (TypeError, ValueError):
        return ""


def load_windows(run_dir: Path) -> dict:
    """Index the per-run wall-clock windows by (model, phase, n, threads).

    Every failure here is per-line and non-fatal. This runs at the very END of a sweep that
    took hours, against files written by a process that may have been OOM-killed mid-write,
    so one damaged record must cost one row and not the entire normalisation. `encoding` is
    explicit because the locale default under systemd is not UTF-8 and a stray byte would
    otherwise raise UnicodeDecodeError from read_text() before any of this runs.
    """
    path = run_dir / "windows.jsonl"
    index = {}
    if not path.exists():
        return index
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print("cannot read %s: %s" % (path.name, exc), file=sys.stderr)
        return index
    skipped = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            key = (rec["model"], rec["phase"], int(rec["n"]), int(rec["threads"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            skipped += 1
            continue
        index[key] = rec
    if skipped:
        print("%s: skipped %d unusable window record(s) -- affected rows will carry "
              "no-window flags" % (path.name, skipped), file=sys.stderr)
    return index


def rows_from(json_path: Path, windows: dict, run_id: str) -> list:
    try:
        entries = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        # Reached at the end of a multi-hour sweep, against a file whose writer may have
        # been killed mid-write. One bad file costs its own rows, never the whole run.
        print("skipping unparseable " + json_path.name, file=sys.stderr)
        return []
    if isinstance(entries, dict):
        entries = [entries]

    # scripts/06_bench_local.sh names files <model>_<phase><n>_t<threads>.json
    stem = json_path.stem
    model_name = stem.rsplit("_", 2)[0]

    out = []
    for e in entries:
        n_prompt = int(e.get("n_prompt", 0))
        n_gen = int(e.get("n_gen", 0))
        phase = "pp" if n_prompt > 0 else "tg"
        n = n_prompt if phase == "pp" else n_gen
        threads = int(e.get("n_threads", 0))
        win = windows.get((model_name, phase, n, threads), {})

        avg_ns = float(e.get("avg_ns", 0) or 0)
        std_ns = float(e.get("stddev_ns", 0) or 0)
        tps = float(e.get("avg_ts", 0) or 0)
        samples_ns = e.get("samples_ns") or []
        flags = []

        # ---- which window does the energy integral run over? -------------------
        # windows.jsonl keeps t_start/t_end as the RAW process window forever. That
        # window contains process start, the GGUF page-in off the microSD, a full-length
        # warmup prefill and teardown, none of which produce a token. The energy join
        # must therefore run over the REPETITION SPAN. The CSV's t_start/t_end carry that
        # span -- and `energy_window` says so, in the row, so a legacy row and a fixed row
        # can never be silently pooled in the same mean.
        kind = win.get("window_kind")
        if kind == "reps-span" and win.get("t_meas_start") and win.get("t_meas_end"):
            t_start_out, t_end_out = win["t_meas_start"], win["t_meas_end"]
            energy_window = "reps-span"
        elif win:
            t_start_out, t_end_out = win.get("t_start", ""), win.get("t_end", "")
            energy_window = "process"
            reason = win.get("window_fallback_reason") or ("legacy" if kind is None else kind)
            flags.append("energy-span:whole-process:" + str(reason))
        else:
            t_start_out, t_end_out = "", ""
            energy_window = ""

        # rep_busy_s is computable for EVERY row, legacy ones included, because
        # samples_ns has always been in llama-bench's JSON. energy_duty therefore states
        # exactly how much of the charged window produced tokens -- which retroactively
        # quantifies the size of this bug on data already on disk.
        rep_busy_s = round(sum(float(s) for s in samples_ns) / 1e9, 6) if samples_ns else ""
        try:
            span = float(t_end_out) - float(t_start_out)
            duty = round(rep_busy_s / span, 4) if (rep_busy_s and span > 0) else ""
        except (TypeError, ValueError):
            span, duty = "", ""
        rep_gap_s = round(span - rep_busy_s, 6) if (rep_busy_s != "" and span != "") else ""

        if energy_window == "reps-span" and duty != "" and duty < MIN_DUTY:
            # reps are seconds long and the inter-rep work is llama_memory_clear plus
            # bookkeeping; anything below 0.98 means something else ran inside the window.
            flags.append("low-duty:%.3f" % duty)
        if int(win.get("rc", 0) or 0) != 0:
            flags.append("bench-exit:%s" % win.get("rc"))
        # A throttled run measures the power supply or the heatsink, not the SoC -- but
        # the vcgencmd word is STICKY: bits 16-19 latch for the rest of the boot. Comparing
        # the whole word to "0x0" therefore flags every row after the first blip and no row
        # is ever meaningfully excluded. What invalidates THIS row is either (a) bits that
        # latched during it, or (b) the board actively throttling as it began -- so compare
        # the word at the start of the run against the word at the end, and read the live
        # bits (0-3) separately from the since-boot bits (16-19).
        # Temperature is a validity precondition just as throttle state is, and it has the
        # same fail-open hazard: scripts/lib.sh soc_temp_c() yields "nan" when the thermal
        # zone cannot be read, which bench_windows.py drops to an empty field. A blank
        # temperature must be visible as "not measured", not read as "fine".
        if win and win.get("temp_end_c", "") == "":
            flags.append("temp-unknown")

        thr0, thr1 = win.get("throttled_start", ""), win.get("throttled_end", "")
        if "unavailable" in (thr0, thr1):
            flags.append("throttle-unknown")        # fail closed: never assume "not throttled"
        else:
            live0, latched = _throttle_delta(thr0, thr1)
            if live0:
                flags.append("throttled-at-start:%s" % thr0)
            if latched:
                flags.append("throttled:%s" % (thr1 or thr0))
        if tps <= 0:
            flags.append("no-throughput")
        if not samples_ns:
            # The control node turns this column into the repetition count with
            # `reps = int(row.get("samples") or 1) or 1`, so an empty samples_ns silently
            # becomes reps=1 and joules_per_token is overstated by the true repetition
            # count -- 5x on a full sweep. Nothing downstream can detect it, so flag here.
            flags.append("no-samples:reps-unknown")
        if not win:
            flags.append("no-window:energy-join-impossible")

        out.append({
            "node_id": ENV.get("NODE_ID", ""),
            "node_role": ENV.get("NODE_ROLE", ""),
            "run_id": run_id,
            "llama_commit": e.get("build_commit", ""),
            "model": model_name,
            "model_type": e.get("model_type", ""),
            "quant": quant_of(model_name, e.get("model_type", "")),
            "model_size_mb": round(float(e.get("model_size", 0) or 0) / (1024 ** 2), 1),
            "n_params": e.get("model_n_params", ""),
            "phase": phase,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "n_threads": threads,
            "n_batch": e.get("n_batch", ""),
            "n_gpu_layers": e.get("n_gpu_layers", 0),
            "tokens_per_s": round(tps, 4),
            "tokens_per_s_stddev": round(float(e.get("stddev_ts", 0) or 0), 4),
            "avg_ms": round(avg_ns / 1e6, 3),
            "stddev_ms": round(std_ns / 1e6, 3),
            "samples": len(e.get("samples_ns", []) or []),
            "ms_per_token": round(1000.0 / tps, 3) if tps > 0 else "",
            "per_layer_ms": "",   # filled by the control node once n_layers is known
            "t_start": t_start_out,
            "t_end": t_end_out,
            "temp_end_c": win.get("temp_end_c", ""),
            "throttled_end": win.get("throttled_end", ""),
            "energy_window": energy_window,
            "t_proc_start": win.get("t_start", ""),
            "t_proc_end": win.get("t_end", ""),
            "rep_busy_s": rep_busy_s,
            "rep_gap_s": rep_gap_s,
            "energy_duty": duty,
            "load_s": _delta(win.get("t_load_start"), win.get("t_load_end")),
            "warmup_s": _delta(win.get("t_warmup_start"), win.get("t_warmup_end")),
            "idle_start": win.get("idle_start", ""),
            "idle_end": win.get("idle_end", ""),
            "energy_j": "",            # <- energy collector
            "joules_per_token": "",    # <- energy collector (the project's primary metric)
            "mean_power_w": "",        # <- energy collector
            "flags": ";".join(flags),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_dir():
        ap.error(str(run_dir) + " is not a directory")

    run_id = run_dir.name
    windows = load_windows(run_dir)
    rows = []
    for json_path in sorted(run_dir.glob("*.json")):
        if json_path.name in ("build_manifest.json", "results.json"):
            continue
        rows.extend(rows_from(json_path, windows, run_id))

    out = args.out or (run_dir / "results.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    suspect = sum(1 for r in rows if r["flags"])
    print("wrote " + str(len(rows)) + " rows -> " + str(out))
    if suspect:
        print(str(suspect) + " row(s) carry validity flags - inspect before using them", file=sys.stderr)


if __name__ == "__main__":
    main()
