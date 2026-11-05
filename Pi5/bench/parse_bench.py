#!/usr/bin/env python3
"""Fold one isolated-bench run directory into the device characterisation dataset.

Reads the llama-bench JSON produced per cell by scripts/10_isolated_bench.sh and emits TWO
CSVs, because two different readers need two different shapes:

  dataset.csv   one row per CELL, prefill and decode side by side. This node's own view,
                and the only one carrying c_per_layer_ms -- the c_i in Rule A.
  results.csv   one row per PHASE, in Pi4/node/bench_parse.py's 38-column schema. This is
                the shared cross-node schema the cluster settled on, and what concatenates
                with the Pi 4's results.csv to build the combined characterisation table.
                See the PI4_COLUMNS block below for the mapping rules that matter.

Neither file gates the other: results.csv is written before the c_per_layer_ms check that
can exit this script non-zero.

Derived columns worth naming explicitly:

  prefill_tok_s   compute-bound throughput, the TTFT driver
  decode_tok_s    bandwidth-bound throughput, the TPOT driver
  tpot_ms         1000 / decode_tok_s -- compared against the 500 ms Baseline budget
  c_per_layer_ms  decode time per layer per token. This is the c_i in Rule A of
                  Appendix B.6: L_i = round(L_total * (1/c_i) / sum_j (1/c_j)).
                  It is what makes a node comparable to a node with a different
                  layer count, and it is the whole reason this sweep exists.
                  The layer count behind it comes from the n_layers column of
                  cells.csv, which 10_isolated_bench.sh reads from the GGUF header --
                  llama-bench does not report one. A row without c_per_layer_ms is
                  invisible to Rule A, so this script exits non-zero if any row
                  lacks it rather than emitting a CSV that merely looks complete.
  bw_util_gbs     implied memory bandwidth (model bytes read per decoded token).
                  Sanity check against the Pi 5's 17 GB/s peak: a value above it
                  means the model was not actually resident and the cell is void.

Usage:
    parse_bench.py RESULTS_DIR [-o dataset.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT_FIELDS = [
    "node_id", "run_id", "cell_id", "started_utc", "ended_utc",
    "model", "model_size_bytes", "model_n_params", "n_layers", "quant", "model_type",
    "n_prompt", "n_gen", "threads", "rep",
    "prefill_tok_s", "prefill_tok_s_stddev",
    "decode_tok_s", "decode_tok_s_stddev",
    "ttft_est_ms", "tpot_ms", "c_per_layer_ms", "bw_util_gbs",
    "temp_start_c", "temp_end_c", "throttled",
    "llama_cpp_sha", "backend", "exit_code",
]


def load_manifest(d: Path):
    p = d / "build_manifest.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def pick(entries, kind):
    """llama-bench emits one entry per (n_prompt, n_gen) leg; pp = prefill, tg = decode.

    The fallback is deliberately narrow. It exists for one case only: a schema that reports
    both legs in a SINGLE entry (n_prompt and n_gen both non-zero), where that entry is the
    best available answer for either question.

    It used to be `return out or entries`, which also fired when the requested leg was
    simply ABSENT -- and then returned the other leg. A cell with no decode entry therefore
    reported the PREFILL rate as decode throughput: ~11.5 tok/s instead of ~2.35, so
    c_per_layer_ms came out roughly 5x too small, with no flag and exit 0. Rule A allocates
    layers proportional to 1/c_i, so that silently hands this node far more of the model
    than it can serve. Returning nothing instead leaves the column empty, which main()
    already treats as fatal.
    """
    out, mixed = [], []
    for e in entries:
        n_prompt = int(e.get("n_prompt", 0) or 0)
        n_gen = int(e.get("n_gen", 0) or 0)
        if n_prompt > 0 and n_gen > 0:
            mixed.append(e)
        elif kind == "pp" and n_prompt > 0 and n_gen == 0:
            out.append(e)
        elif kind == "tg" and n_gen > 0 and n_prompt == 0:
            out.append(e)
    return out or mixed


def tok_s(entries):
    """Mean throughput and the spread across llama-bench's repetitions.

    The order of these two branches is the whole point. `samples_ts` holds one value per
    `--repetitions` pass; `avg_ts` is already their mean. Reading `avg_ts` first -- which
    this did -- collapses each leg to a single number, so `statistics` saw one value and
    the stddev came out as exactly 0.0 on every row of every dataset ever folded.

    That is not a cosmetic difference. 0.0 does not read as "spread unknown", it reads as
    "this measurement has no variance" -- a far stronger claim than the sweep can make,
    on a passively-cooled ARM board whose principal risk (R5) is thermal drift during the
    run. The spread was always being measured: 10_isolated_bench.sh passes
    `--repetitions 5` to llama-bench and then repeats the whole cell 3 times on top of
    that. It was only being thrown away before it reached the CSV.

    Sample stddev (n-1), matching llama-bench's own `stdev()` helper, so these columns
    are directly comparable with the `stddev_ts` upstream prints.
    """
    vals = []
    for e in entries:
        samples = e.get("samples_ts")
        if samples:
            vals.extend(float(v) for v in samples)
        elif e.get("avg_ts") is not None:
            vals.append(float(e["avg_ts"]))
    if not vals:
        return None, None
    return (
        statistics.fmean(vals),
        # None, not 0.0. A single sample means the spread was never measured, and 0.0 is
        # the stronger claim that it was measured and found to be zero -- the exact error
        # this function's docstring exists to describe. Callers render None as "".
        statistics.stdev(vals) if len(vals) > 1 else None,
    )


def first(entries, *keys, default=None):
    for e in entries:
        for k in keys:
            if e.get(k) not in (None, ""):
                return e[k]
    return default


# ----------------------------------------------------------------------------------
# The Pi 4 schema.
#
# The cluster settled on Pi4/node/bench_parse.py's COLUMNS as the one cross-node results
# schema, because that is the shape ThinkPad/control/ already reads. This node emits it
# IN ADDITION to dataset.csv -- dataset.csv keeps c_per_layer_ms and the per-cell view the
# thesis needs, results.csv is what concatenates with the Pi 4's for the combined table.
#
# This is a deliberate standalone copy, for the same reason node/gguf_meta.py is one: a
# worker node folder is deployed on its own and cannot import from Pi4/. The column list
# and quant_of() are the contract between them -- if one changes, change both.
# tests/test_pi4_schema.py reads Pi4's COLUMNS live and fails when they drift apart.
#
# Two mapping rules are easy to get wrong and silently destructive, so they are stated here:
#
#   * ONE Pi 5 cell becomes TWO rows. 10_isolated_bench.sh runs one llama-bench process per
#     cell with both --n-prompt and --n-gen, and llama-bench reports each leg as its own
#     entry (the pp entry has n_gen=0, the tg entry has n_prompt=0). n_prompt, n_gen,
#     n_threads, tokens_per_s and the durations must come from THAT LEG'S ENTRY, never from
#     cells.csv, whose values are identical on both legs.
#
#   * `flags` is not a severity field. ThinkPad/control/profiles.py:90-92 is a bare
#     truthiness test -- any non-empty flags cell deletes the row from per_layer_times,
#     which is the function that produces the c_i Rule A consumes, and the drop count it
#     computes at :113 is never printed. A flag is therefore an opt-out from the placement
#     rule, not a caveat within it. This node raises only what it actually observed.
# ----------------------------------------------------------------------------------

PI4_COLUMNS = [
    # identity
    "node_id", "node_role", "run_id", "llama_commit",
    # what was run
    "model", "model_type", "quant", "model_size_mb", "n_params",
    "phase", "n_prompt", "n_gen", "n_threads", "n_batch", "n_gpu_layers",
    # measured
    "tokens_per_s", "tokens_per_s_stddev", "avg_ms", "stddev_ms", "samples",
    # derived
    "ms_per_token", "per_layer_ms",
    # energy join keys, filled by the control node
    "t_start", "t_end", "temp_end_c", "throttled_end",
    "energy_window", "t_proc_start", "t_proc_end",
    "rep_busy_s", "rep_gap_s", "energy_duty", "load_s", "warmup_s",
    # The idle window recorded immediately before the run, so a marginal (above-idle)
    # figure can be computed per row instead of against a session-wide mean that pools a
    # cold board with a hot one. This node cannot fill them: 10_isolated_bench.sh has a
    # cool-down loop between cells but records no timestamps for it, so there is no idle
    # span in cells.csv. Recording t_idle_start/t_idle_end around that loop is the
    # source-side fix; until then these are honestly empty.
    "idle_start", "idle_end",
    "energy_j", "joules_per_token", "mean_power_w",
    # validity
    "flags",
]

# Copied from Pi4/node/bench_parse.py. A divergent re-implementation would split
# analysis.py's group key (node_id, model, quant, phase, n_threads) and silently produce
# two half-populated report rows per cell.
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


# Pi 4 raises flag tokens this node cannot, and `flags` is load-bearing: a non-empty cell
# is an opt-out from the placement rule, so the two nodes must flag on the same basis or
# the combined table excludes them unevenly. Every token Pi 4 can raise is listed here
# either as one this node also raises, or with the reason it cannot.
# tests/test_pi4_schema.py extracts Pi 4's flag literals and fails when a new one appears,
# so the vocabulary cannot drift silently the way the column list once did.
PI4_FLAGS_NOT_RAISED_HERE = {
    "energy-span:whole-process": "this node charges no energy window at all; energy_window"
                                 ' is "" and t_start/t_end are empty, which states the'
                                 " absence in band without claiming the row is invalid",
    "low-duty": "needs a duty ratio, which needs an energy window",
    "no-window:energy-join-impossible": "the empty t_start/t_end and energy_window already"
                                        " say this; a flag would additionally assert the"
                                        " row is not a measurement, and the timing IS valid",
}


def quant_of(name: str, model_type: str) -> str:
    flat = re.sub(r"[^A-Z0-9]", "", (name + " " + model_type).upper())
    for canonical, spellings in QUANT_HINTS:
        if any(sp in flat for sp in spellings):
            return canonical
    return model_type or "unknown"


def iso_to_epoch(stamp):
    """cells.csv holds ISO-8601 ms; the shared schema and the power stream hold epoch float.

    10_isolated_bench.sh stamps `date -u +%Y-%m-%dT%H:%M:%S.%3NZ`, while Pi 4 and the DAQ
    use `date -u +%s.%N`. ThinkPad/control/analysis.py calls float() on these columns
    without a guard, and ValueError is not caught by the CLI -- so handing it an ISO string
    takes down the whole energy join mid-loop, after the output header has been written.
    Anything unparseable becomes "", which the consumers do handle.
    """
    if stamp in (None, ""):
        return ""
    text = str(stamp).strip()
    try:
        return round(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp(), 6)
    except ValueError:
        try:
            return round(float(text), 6)   # already epoch; pass it through
        except (TypeError, ValueError):
            return ""


def load_node_env(d: Path) -> dict:
    """node.env.snapshot is copied unconditionally by 10_isolated_bench.sh.

    build_manifest.json is copied with `[ -f ] && cp`, which does not abort under set -e,
    so the manifest can be absent with no error -- and it was parse_bench's only source of
    node_id. An empty node_id creates a device bucket no cluster.json node matches, and
    nothing downstream reports the orphan. Prefer the snapshot, which is always there.
    """
    env = {}
    p = d / "node.env.snapshot"
    if not p.is_file():
        return env
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, _, val = line.partition("=")
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        env[key.strip()] = val
    return env


# Any temperature that is not a finite, non-zero number means "not measured".
#
# Two sentinels reach this, and guarding only one is how the fail-open came back:
#   * scripts/lib/common.sh's soc_temp_c() emits the literal "nan" when the thermal zone
#     cannot be read. float("nan") parses, and `nan >= 80.0` is False -- so an unguarded
#     nan makes the throttled column report "this cell was fine" about a cell nobody took
#     the temperature of. That is precisely the failure this function exists to prevent,
#     arriving through the value the function did not know about.
#   * 0.0 / -0.0 was the OLD fallback, before common.sh was changed to "nan". Run
#     directories recorded under the old behaviour are still on disk, so it stays guarded.
# inf is covered by the same isfinite test; nothing emits it, but a column that silently
# accepts it is one more way a non-measurement can look like a measurement.
def temp_or_blank(value):
    if value in (None, ""):
        return ""
    f = _num(value, None)
    if f is None or not math.isfinite(f) or abs(f) < 1e-9:
        return ""
    return value


# The Pi 5's peak memory bandwidth. A decode rung implying more than this did not actually
# read the weights from DRAM -- the model was not resident, or the run was short-circuited --
# so the cell is void. The docstring promised this check for a while before it existed.
PEAK_MEM_BW_GBS = 17.0

# The SoC soft-throttles here. node.env carries THERMAL_THROTTLE_C; this is the fallback
# for a run directory recorded before that key existed, and keeps the threshold named
# rather than buried as a literal in the row builder.
THROTTLE_C = 80.0


LIVE_BITS = 0x0000000F   # 0 under-voltage, 1 freq-capped, 2 throttled, 3 soft-temp-limit
STICKY_BITS = 0x000F0000  # the same four, latched since boot


def _throttle_delta(start_word, end_word):
    """(was_live_at_start, throttled_during_this_run) from the two vcgencmd words.

    Copied from Pi4/node/bench_parse.py so both nodes decide this identically. The word is
    sticky: bits 16-19 latch for the rest of the boot, so `end != "0x0"` says only that the
    board throttled at SOME point -- which, mid-sweep, is true of every row after the first
    blip. What invalidates a row is what happened during it. Unparseable input is treated as
    throttled, because the alternative is publishing a row that was never checked.
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
    newly = b & ~a
    return (bool(a & LIVE_BITS),
            bool(newly & (LIVE_BITS | STICKY_BITS)) or bool(b & LIVE_BITS))


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pi4_rows(d: Path, row: dict, entries: list, manifest: dict, node_env: dict) -> list:
    """One cells.csv cell -> one row per llama-bench leg, in Pi 4's column order."""
    node_id = node_env.get("NODE_ID") or manifest.get("node_id", "")
    n_layers = int(row.get("n_layers") or 0)

    # temp_end_c is a cell-level fact stamped after BOTH legs, so it is duplicated onto
    # both rows -- unlike Pi 4, which runs each phase as its own invocation. A dead sensor
    # is emitted as absent rather than as a physically impossible reading beside Pi 4's
    # genuine temperatures; see temp_or_blank for which values count as dead.
    temp_end = temp_or_blank(row.get("temp_end_c", ""))

    thr0 = (row.get("throttled_start") or "").strip()
    thr1 = (row.get("throttled_end") or "").strip()
    t_proc_start = iso_to_epoch(row.get("started_utc"))
    t_proc_end = iso_to_epoch(row.get("ended_utc"))
    cell_rc = str(row.get("exit_code", "0") or "0")

    out = []
    for e in entries:
        n_prompt = int(_num(e.get("n_prompt"), 0))
        n_gen = int(_num(e.get("n_gen"), 0))
        phase = "pp" if n_prompt > 0 else "tg"

        # avg_ts verbatim, not a recomputed mean: ms_per_token is defined as
        # 1000/tokens_per_s and avg_ms comes from avg_ns, so recomputing one of the three
        # would make them mutually inconsistent inside a single row.
        tps = _num(e.get("avg_ts"), 0.0)
        ms_per_token = round(1000.0 / tps, 3) if tps > 0 else ""

        samples_ns = e.get("samples_ns") or []
        rep_busy_s = round(sum(_num(x) for x in samples_ns) / 1e9, 6) if samples_ns else ""

        flags = []
        if cell_rc not in ("0", ""):
            flags.append("bench-exit:%s" % cell_rc)
        # Same basis as Pi 4: a word that was never read means the state was never checked,
        # and fails closed rather than being read as clean. An EMPTY word counts as unknown
        # too -- that is what a run directory recorded before 10_isolated_bench.sh carried
        # these columns looks like, and feeding it to the delta below would produce a bare
        # "throttled:" with no word, which asserts a throttle nobody observed.
        if "unavailable" in (thr0, thr1) or not thr0 or not thr1:
            flags.append("throttle-unknown")
        else:
            live0, latched = _throttle_delta(thr0, thr1)
            if live0:
                flags.append("throttled-at-start:%s" % thr0)
            if latched:
                flags.append("throttled:%s" % (thr1 or thr0))
        # A blank temperature must read as "not measured", never as "fine" -- the same
        # fail-open hazard Pi 4 names for its own nan fallback. Raised on the same basis
        # as Pi 4 so the combined table excludes rows from both nodes alike.
        if temp_end == "":
            flags.append("temp-unknown")
        if tps <= 0:
            flags.append("no-throughput")
        # A missing thread count would otherwise become 0 and group with nothing. The
        # PI4_COLUMNS contract requires it from the leg's own entry, so flag rather than
        # silently substituting cells.csv's value.
        if e.get("n_threads") in (None, ""):
            flags.append("no-threads")
        if not samples_ns:
            # The control node turns this column into the repetition count with
            # `reps = int(row.get("samples") or 1) or 1`, so an empty samples_ns silently
            # becomes reps=1 and joules_per_token is overstated by the true repetition
            # count -- 5x on a full sweep. Nothing downstream can detect it.
            flags.append("no-samples:reps-unknown")

        out.append({
            "node_id": node_id,
            "node_role": node_env.get("NODE_ROLE", ""),
            "run_id": d.name,
            # llama-bench's own short SHA, not the manifest's 40-char rev-parse output --
            # the two nodes must spell the pinned revision identically.
            "llama_commit": e.get("build_commit", ""),
            "model": row.get("model", ""),
            "model_type": e.get("model_type", ""),
            "quant": quant_of(row.get("model", ""), e.get("model_type", "")),
            "model_size_mb": round(_num(e.get("model_size"), 0.0) / (1024 ** 2), 1),
            "n_params": e.get("model_n_params", ""),
            "phase": phase,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "n_threads": int(_num(e.get("n_threads"), 0)),
            "n_batch": e.get("n_batch", ""),
            "n_gpu_layers": e.get("n_gpu_layers", 0),
            "tokens_per_s": round(tps, 4),
            # llama-bench's spread across the 5 INNER --repetitions, not across the 3 outer
            # cell repetitions the sweep drives. The outer spread -- the one that carries
            # thermal drift, risk R5 -- has no column in this schema.
            "tokens_per_s_stddev": (round(_num(e.get("stddev_ts")), 4)
                                    if e.get("stddev_ts") is not None else ""),
            # Whole-leg time per repetition, NOT per token.
            "avg_ms": round(_num(e.get("avg_ns"), 0.0) / 1e6, 3),
            "stddev_ms": (round(_num(e.get("stddev_ns")) / 1e6, 3)
                          if e.get("stddev_ns") is not None else ""),
            # The llama-bench repetition COUNT. analysis.py multiplies the token count by
            # this to get the joules-per-token denominator, so cells.csv's outer `rep`
            # index here would overstate the project's headline metric with no warning.
            "samples": len(samples_ns),
            "ms_per_token": ms_per_token,
            # Pi 4 leaves this empty and lets the control node recompute it. This node has
            # an authoritative block_count from the GGUF header, so it fills it -- the only
            # way that layer count survives into the combined table. The column is
            # therefore mixed-population; do not average it across nodes.
            "per_layer_ms": (round(ms_per_token / n_layers, 4)
                             if (ms_per_token != "" and n_layers) else ""),
            # No per-phase energy window exists. One process runs both legs, so its wall
            # clock cannot be attributed to either without double-counting the cell's
            # joules. energy_window="" says so in band; the raw process span is recorded
            # below, correctly typed, where it cannot feed the energy integral.
            "t_start": "",
            "t_end": "",
            "temp_end_c": temp_end,
            # The firmware word at the end of the cell, recorded by 10_isolated_bench.sh.
            # "unavailable" when vcgencmd could not be read -- never "0x0", which would
            # read as clean.
            "throttled_end": thr1,
            "energy_window": "",
            "t_proc_start": t_proc_start,
            "t_proc_end": t_proc_end,
            "rep_busy_s": rep_busy_s,
            "rep_gap_s": "",
            "energy_duty": "",
            "load_s": "",
            "warmup_s": "",
            # The cool-down that preceded this cell, in epoch seconds, so a marginal
            # (above-idle) figure can be computed per row rather than against a
            # session-wide mean that pools a cold board with a hot one.
            "idle_start": _num(row.get("idle_start"), "") or "",
            "idle_end": _num(row.get("idle_end"), "") or "",
            # Present but empty: the control node's join writes over this file's own
            # fieldnames, so a missing column makes writerows raise after the header is
            # already out, truncating results_energy.csv.
            "energy_j": "",
            "joules_per_token": "",
            "mean_power_w": "",
            "flags": ";".join(flags),
        })
    return out


def load_entries(d: Path, row: dict):
    """The llama-bench document for one cell, or None. Both output schemas read it."""
    raw = d / row["raw_json"]
    if not raw.is_file():
        print("missing %s" % raw, file=sys.stderr)
        return None
    try:
        data = json.loads(raw.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print("unparseable %s: %s" % (raw, exc), file=sys.stderr)
        return None
    entries = data if isinstance(data, list) else data.get("results", [])
    if not entries:
        print("no llama-bench results in %s" % raw, file=sys.stderr)
        return None
    return entries


def parse_cell(d: Path, row: dict, manifest: dict, entries: list,
               node_env: dict | None = None) -> dict | None:
    node_env = node_env or {}
    throttle_c = _num(node_env.get("THERMAL_THROTTLE_C"), THROTTLE_C) or THROTTLE_C
    pf, pf_sd = tok_s(pick(entries, "pp"))
    dc, dc_sd = tok_s(pick(entries, "tg"))

    size_b = first(entries, "model_size", default=0) or 0
    n_params = first(entries, "model_n_params", default=0) or 0
    # llama-bench emits NO layer count: test::get_fields() is a fixed list carrying
    # model_size, model_n_params, model_type and n_gpu_layers -- and n_gpu_layers is
    # the offload count, which is 0 on a CPU-only worker. So the authoritative source
    # is the n_layers column that 10_isolated_bench.sh reads from the GGUF header and
    # writes into cells.csv. The JSON keys below are only a fallback for run
    # directories recorded before that column existed.
    n_layers = int(row.get("n_layers") or 0) \
        or int(first(entries, "n_layer", "model_n_layer", default=0) or 0)
    quant = first(entries, "model_type", default="") or ""
    # Upstream spells it `backends` (plural); `backend_name` never existed and is
    # kept only so a hand-written JSON carrying it still parses.
    backend = first(entries, "backends", "backend_name", default="CPU")

    n_prompt = int(row["n_prompt"])
    tpot_ms = (1000.0 / dc) if dc else None
    # Decode reads the whole weight set once per token, so per-layer per-token time
    # is simply TPOT divided by the layer count -- the form the placement rule wants.
    c_layer = (tpot_ms / n_layers) if (tpot_ms and n_layers) else None
    bw = (float(size_b) * dc / 1e9) if (dc and size_b) else None

    return {
        # Same precedence pi4_rows uses. node.env.snapshot is copied unconditionally by
        # 10_isolated_bench.sh; build_manifest.json is copied with `[ -f ] && cp`, which
        # does not abort under set -e, so the manifest alone could leave this blank.
        "node_id": (node_env.get("NODE_ID") or row.get("node_id")
                    or manifest.get("node_id", "")),
        "run_id": d.name,
        "cell_id": row["cell_id"],
        "started_utc": row["started_utc"],
        "ended_utc": row["ended_utc"],
        "model": row["model"],
        "model_size_bytes": size_b,
        "model_n_params": n_params,
        "n_layers": n_layers,
        # Canonical label, the same normalisation results.csv uses -- the raw llama-bench
        # string used to sit in this column, so `quant` meant two different things in the
        # two files this script writes. The raw string is kept beside it.
        "quant": quant_of(row.get("model", ""), quant),
        "model_type": quant,
        "n_prompt": n_prompt,
        "n_gen": int(row["n_gen"]),
        "threads": int(row["threads"]),
        "rep": int(row["rep"]),
        "prefill_tok_s": round(pf, 3) if pf else "",
        "prefill_tok_s_stddev": round(pf_sd, 3) if pf_sd is not None else "",
        "decode_tok_s": round(dc, 3) if dc else "",
        "decode_tok_s_stddev": round(dc_sd, 3) if dc_sd is not None else "",
        "ttft_est_ms": round(1000.0 * n_prompt / pf, 1) if pf else "",
        "tpot_ms": round(tpot_ms, 1) if tpot_ms else "",
        "c_per_layer_ms": round(c_layer, 4) if c_layer else "",
        "bw_util_gbs": round(bw, 3) if bw else "",
        "temp_start_c": temp_or_blank(row["temp_start_c"]),
        "temp_end_c": temp_or_blank(row["temp_end_c"]),
        # A cell that ended hot is flagged, never silently averaged in (risk R5). With no
        # usable temperature the honest answer is "unknown", not 0 -- a dead sensor used to
        # publish a dataset asserting that no cell was throttled.
        "throttled": ("" if temp_or_blank(row["temp_end_c"]) == ""
                      else int(_num(row["temp_end_c"]) >= throttle_c)),
        "llama_cpp_sha": manifest.get("llama_cpp_sha", ""),
        "backend": backend,
        "exit_code": row["exit_code"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("-o", "--out", default=None,
                    help="this node's dataset.csv (default: RESULTS_DIR/dataset.csv)")
    ap.add_argument("--pi4-out", default=None,
                    help="the shared-schema results.csv that concatenates with the Pi 4's "
                         "(default: RESULTS_DIR/results.csv)")
    args = ap.parse_args()

    d = Path(args.results_dir)
    index = d / "cells.csv"
    if not index.is_file():
        print("no cells.csv in %s" % d, file=sys.stderr)
        return 2
    manifest = load_manifest(d)

    node_env = load_node_env(d)
    rows, shared, lost = [], [], []
    with index.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            entries = load_entries(d, row)
            if not entries:
                # Counted, not just skipped. A cell whose JSON is missing, unparseable or
                # empty used to vanish from both CSVs with a clean exit -- and an empty
                # entry list printed nothing at all, so a run whose every cell failed
                # produced "0 cells", rc=0 and an empty stderr.
                lost.append(row.get("cell_id", "?"))
                continue
            parsed = parse_cell(d, row, manifest, entries, node_env)
            if parsed:
                rows.append(parsed)
            else:
                lost.append(row.get("cell_id", "?"))
            shared.extend(pi4_rows(d, row, entries, manifest, node_env))

    out = Path(args.out) if args.out else d / "dataset.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)

    # Written BEFORE the c_per_layer_ms gate below. That gate is about this node's own
    # dataset; letting it suppress the cross-node file too would mean one degenerate cell
    # removes the Pi 5 from the combined table as well as from Rule A.
    shared_out = Path(args.pi4_out) if args.pi4_out else d / "results.csv"
    with shared_out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PI4_COLUMNS)
        w.writeheader()
        w.writerows(shared)

    print("%d cells -> %s" % (len(rows), out))
    print("%d rows (one per phase) -> %s" % (len(shared), shared_out))
    if shared and not shared[0]["node_id"]:
        print("WARNING: node_id is empty on every shared row -- no NODE_ID in "
              "node.env.snapshot and none in build_manifest.json. The control node buckets "
              "these under \"\", which matches no cluster.json node, and nothing downstream "
              "reports the orphan.", file=sys.stderr)
    hot = [r for r in rows if r["throttled"]]
    if hot:
        print("WARNING: %d cell(s) ended at or above 80 C and are flagged throttled:"
              % len(hot), file=sys.stderr)
        for r in hot:
            print("  %s (%s C)" % (r["cell_id"], r["temp_end_c"]), file=sys.stderr)
    # A blank rc reads as "not recorded, treat as clean" in results.csv (pi4_rows), so the
    # two files must not disagree about the same cell. The recorded value is left untouched.
    bad = [r for r in rows if str(r["exit_code"] or "0") != "0"]
    if bad:
        print("WARNING: %d cell(s) exited non-zero" % len(bad), file=sys.stderr)
    # The docstring has always claimed this check; it did not exist. A rung implying more
    # than the SoC's peak memory bandwidth did not read the weights from DRAM, so the cell
    # is void -- but only that cell, so this warns rather than failing the run.
    peak = _num(node_env.get("PEAK_MEM_BW_GBS"), PEAK_MEM_BW_GBS) or PEAK_MEM_BW_GBS
    void = [r for r in rows if r["bw_util_gbs"] != "" and _num(r["bw_util_gbs"]) > peak]
    if void:
        print("WARNING: %d cell(s) imply more than %.1f GB/s of memory bandwidth and cannot "
              "have been reading the weights from DRAM:" % (len(void), peak), file=sys.stderr)
        for r in void:
            print("  %s (%s GB/s)" % (r["cell_id"], r["bw_util_gbs"]), file=sys.stderr)
    unknown_temp = [r for r in rows if r["temp_end_c"] == ""]
    if unknown_temp:
        print("WARNING: %d cell(s) have no usable end temperature, so their thermal validity "
              "is unknown (risk R5)" % len(unknown_temp), file=sys.stderr)
    # c_per_layer_ms is the c_i the placement rule consumes. A blank one is not a
    # cosmetic gap -- it silently removes the row from Rule A while leaving a CSV that
    # looks complete. Fail loudly rather than hand back a quietly useless dataset.
    if lost:
        print("ERROR: %d of %d cell(s) in cells.csv produced no row and are absent from both "
              "CSVs: %s" % (len(lost), len(lost) + len(rows), ", ".join(lost[:10])),
              file=sys.stderr)
        return 1
    no_ci = [r for r in rows if not r["c_per_layer_ms"]]
    if no_ci:
        print("ERROR: %d of %d cell(s) have no c_per_layer_ms (missing layer count or "
              "decode throughput). Rule A cannot use them; re-run the sweep with a "
              "10_isolated_bench.sh that writes the n_layers column."
              % (len(no_ci), len(rows)), file=sys.stderr)
        for r in no_ci[:5]:
            print("  %s (n_layers=%s decode_tok_s=%s)"
                  % (r["cell_id"], r["n_layers"], r["decode_tok_s"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
