#!/usr/bin/env python3
"""Re-emit this node's sweep as the cluster's shared results.csv schema.

    ./emit_results_csv.py <base>_bench_energy.csv [--out results.csv]

WHY THIS EXISTS. Each worker node writes its own rich CSV, and those three are
not concatenable: Pi 4 writes the 40-column schema in `Pi4/node/bench_parse.py`,
Pi 5 writes its own `dataset.csv` PLUS a Pi-4-shaped `results.csv`, and this node
writes `<base>_bench_energy.csv`, whose columns model cases, groups and marker
windows and carry per-request metrics none of the others have. The control node's
cross-node table -- the one the placement rule reads -- needs one shape. Pi 4's
is it, by precedent rather than by decree: it is the only one two nodes already
emit.

So this is a projection, not a replacement. `<base>_bench_energy.csv` stays the
authoritative record of what this node measured; this file is the subset that can
stand beside another node's row without lying.

WHAT IT DELIBERATELY LEAVES EMPTY, and why that is the honest answer rather than
an omission:

  t_start, t_end, energy_window, energy_j, joules_per_token, mean_power_w

`ThinkPad/control/analysis.py` fills the energy columns itself, by integrating a
power stream over [t_start, t_end] and dividing by `n * samples`. Handing it a
window would be wrong here for three separate reasons, each sufficient on its own:

  1. It cannot read this node's power stream. `load_power()` wants
     t_utc / bus_v / current_a / power_w; the tegrastats logger writes
     ts_unix / pow_in_mw / energy_in_j and has NO bus-voltage or current channel
     at all -- the INA3221 is read through tegrastats, which reports milliwatts
     directly. The overlap between the two column sets is empty. Supplying
     bus_v/current_a would mean inventing two measurements this board does not
     make.
  2. Its sample-rate assumptions do not hold. `integrate()` defaults to
     sample_hz=28 and flags a gap wider than MAX_GAP_FACTOR periods; this logger
     runs at TELEMETRY_HZ=5, i.e. 0.20 s between samples, so every row would come
     back carrying `gap:` and edge flags. Those are NOT in
     THROUGHPUT_INVALIDATING, so they would not delete the node -- they would
     make every row look defective while the numbers were in fact fine, which is
     its own kind of wrong.
  3. Its denominator is `n * samples`, with no warm-up term. This node's own
     divisor deliberately includes the warm-up llama-bench runs before each timed
     set (a full prompt pass for pp, one token for tg), because its energy window
     contains that warm-up's joules. The two conventions are each correct and are
     not interchangeable: handing the control node a warm-up-inclusive window to
     divide by n*samples overstates prefill J/token by (r+1)/r -- about 33% at the
     matrix default of -r 3.

None of that is fixable from inside this folder, and cross-node ENERGY comparison
is in any case already deferred for want of an instrument common to all four
nodes (see README, "Measurement honesty"). What the cross-node table actually
consumes for the placement rule is TIMING -- ms_per_token and per_layer_ms -- and
those this node can supply honestly. Its energy stays in its own CSV, where the
window semantics are documented alongside it.

This mirrors what `Pi5/bench/parse_bench.py` does for a different reason (no
per-phase window exists there), so the combined table already handles rows whose
energy columns are blank.
"""
import argparse
import csv
import os
import re
import sys

# --------------------------------------------------------------- the schema
# Copied verbatim from Pi4/node/bench_parse.py. tests/test_shared_schema.py reads
# that file LIVE and fails when the two drift, because a divergent copy is worse
# than none: the control node groups on (node_id, model, quant, phase, n_threads)
# and a mismatch there produces two half-populated rows per cell instead of one.
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
    "energy_window", "t_proc_start", "t_proc_end",
    "rep_busy_s", "rep_gap_s", "energy_duty", "load_s", "warmup_s",
    "idle_start", "idle_end",
    "energy_j", "joules_per_token", "mean_power_w",
    # validity
    "flags",
]

# Copied verbatim from Pi4/node/bench_parse.py for the same reason as COLUMNS:
# `quant` is part of the control node's group key, so a re-implementation that
# canonicalises one spelling differently splits a model into two report rows.
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

# Flag tokens Pi 4 raises that this node cannot, each with the reason. Listed so
# the drift test can assert every one of Pi 4's literals is either raised here or
# accounted for -- the two nodes must flag on the same basis, or the combined
# table excludes their rows unevenly.
PI4_FLAGS_NOT_RAISED_HERE = {
    "energy-span:whole-process": "this node charges no energy window into the shared"
                                 " schema at all; energy_window is \"\" and t_start/"
                                 "t_end are empty, which states the absence in band",
    "low-duty": "needs a duty ratio, which needs an energy window",
    "no-window:energy-join-impossible": "the empty t_start/t_end and energy_window"
                                        " already say this; a flag would additionally"
                                        " assert the row is not a measurement, and the"
                                        " TIMING is valid and is what the placement"
                                        " rule consumes",
    "throttled-at-start": "a Jetson has no vcgencmd throttle word to compare across a"
                          " run, so no start-vs-end delta exists to raise this from."
                          " The unread state is carried by throttled_end instead",
    "throttled": "same: there is no latched firmware word on this board. Thermal"
                 " behaviour is carried by temp_end_c and, in this node's own CSV,"
                 " by temp_rise_c and the per-sample thermal trace",
    "throttle-unknown": "raising it would delete this node from the table it was"
                        " written to join. ThinkPad/control/analysis.py:224 defines"
                        " THROUGHPUT_INVALIDATING = (\"throttled\","
                        " \"throttle-unknown\"), so the token means \"this row's"
                        " tokens_per_s is not a usable measurement\" -- not \"not"
                        " checked\". On Pi 4 and Pi 5 it marks a RUN that failed to"
                        " read its word, which is recoverable; here it would be true"
                        " of every row forever, so the meaning does not survive the"
                        " copy. The state is stated in band instead, via"
                        " throttled_end=\"unavailable\" -- Pi 4's and Pi 5's own"
                        " spelling for an unread word, and a value no control-node"
                        " module reads as a verdict",
}

# Pi 4's and Pi 5's spelling for "the throttle word could not be read", used here
# unconditionally: vcgencmd is Raspberry Pi firmware tooling with no Tegra
# equivalent, so this board can never read one. Emitted in the column rather than
# as a flag, because the flag is a throughput verdict in this cluster's
# vocabulary and this is a statement about the hardware. An empty string would
# have been the dishonest option -- under Pi 4's fail-closed convention it reads
# as "checked and clean".
THROTTLE_UNAVAILABLE = "unavailable"


def quant_of(name, model_type):
    flat = re.sub(r"[^A-Z0-9]", "", (name + " " + model_type).upper())
    for canonical, spellings in QUANT_HINTS:
        if any(sp in flat for sp in spellings):
            return canonical
    return model_type or "unknown"


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _int(v, default=0):
    n = _num(v)
    return int(n) if n is not None else default


# phase vocabulary. Pi 4 and Pi 5 both spell these "pp" and "tg" -- llama-bench's
# own test-label prefixes -- not "prefill"/"decode". A row whose phase does not
# match is silently dropped from every group-by the control node does.
PHASE_OF = {"prefill": "pp", "decode": "tg"}


def row_to_shared(rec, run_id, n_layers=0):
    """One joined-CSV record -> one shared-schema row, or None to omit it.

    Returns None only for a row that has no honest place in the shared table at
    all; everything else comes through with `flags` saying what is wrong with it,
    because a row that vanishes cannot be audited.
    """
    # `row_phase` is what the ref reported; `phase` is what the driver asked for.
    # Falling back matters: a failed or skipped run has no bench row to report a
    # phase, and keying on row_phase alone once dropped every such run from this
    # file with no row and no flag -- the exact silent-erasure failure this
    # projection exists to avoid.
    phase = PHASE_OF.get((rec.get("row_phase") or rec.get("phase") or "").strip())
    if phase is None:
        # `combined` is this node's deliberately fused -pg case, and `none` is a
        # row whose counts could not be read. Neither has a pp/tg counterpart on
        # the other nodes: the shared schema's phase column has no value that
        # means "both", and inventing one would make the row group with nothing
        # while still being counted. Omitted here and reported by the caller.
        return None

    n_prompt = _int(rec.get("bench_n_prompt"), _int(rec.get("n_prompt")))
    n_gen = _int(rec.get("bench_n_gen"), _int(rec.get("n_gen")))
    tps = _num(rec.get("avg_ts"), 0.0) or 0.0
    # 1000/tokens_per_s, not a recomputed mean: avg_ms comes from avg_ns and
    # ms_per_token from tokens_per_s, so recomputing either would make the three
    # numbers in one row mutually inconsistent.
    ms_per_token = round(1000.0 / tps, 3) if tps > 0 else ""

    # The node's OWN short name from config/models.tsv, which is what Pi 4 puts
    # here too (its models.txt first column, e.g. "tinyllama-1.1b-q4km"). The
    # cluster has no shared file-naming convention, and `model` is part of the
    # control node's group key -- so the rule is "each node spells it its own way,
    # consistently". Using the GGUF basename on some rows and the short name on
    # others would split one model into two groups within this node alone.
    model_name = rec.get("model") or ""
    model_type = rec.get("bench_model_type") or ""

    flags = []
    status = (rec.get("per_token_status") or "").strip()
    run_status = (rec.get("status") or "").strip()
    if run_status == "failed":
        flags.append("bench-exit:%s" % (rec.get("rc") or "?"))
    elif run_status == "skipped":
        flags.append("model-missing")
    if tps <= 0:
        flags.append("no-throughput")
    if not _int(rec.get("repetitions"), 0):
        # The control node turns this column into the repetition count with
        # `reps = int(row.get("samples") or 1) or 1`, so a missing value silently
        # becomes 1 and joules-per-token is overstated by the true repetition
        # count. Nothing downstream can detect it.
        flags.append("no-samples:reps-unknown")
    if not rec.get("threads"):
        flags.append("no-threads")
    # This node's own verdict on whether its numbers are usable. Anything that is
    # not plain "ok" travelled here for a reason, and `flags` is exactly the
    # channel the shared schema provides for it.
    if status and status != "ok":
        flags.append("jetson-per-token:%s" % status)
    # A blank temperature must read as "not measured", never as "fine" -- the
    # same fail-open hazard Pi 4 names for its own nan fallback.
    temp_end = rec.get("temp_end_c")
    if temp_end in (None, "", "0", "0.0"):
        temp_end = ""
        flags.append("temp-unknown")

    # The cluster-canonical id, NOT this node's local NODE_ID spelling.
    # ThinkPad/control/profiles.py groups by row["node_id"] with no alias
    # resolution, so "jetson-nano-01" would sit in a group of its own and the
    # placement rule would never see this machine. Pi 4 and Pi 5 avoid the
    # problem by setting NODE_ID to the canonical id directly; this node keeps a
    # longer local spelling, so the mapping is explicit in config/node.env.
    node_id = (rec.get("cluster_node_id") or "").strip() or (rec.get("node_id") or "")
    if not (rec.get("cluster_node_id") or "").strip():
        flags.append("node-id-unmapped")

    return {
        "node_id": node_id,
        "node_role": rec.get("node_role") or "",
        "run_id": run_id,
        # llama-bench's own short SHA, as the other nodes spell it -- not a
        # 40-char rev-parse, or the pinned revision reads as two revisions.
        "llama_commit": rec.get("build_commit") or "",
        "model": model_name,
        "model_type": model_type,
        "quant": quant_of(model_name, model_type),
        "model_size_mb": (round(_num(rec.get("bench_model_size"), 0.0) / (1024 ** 2), 1)
                          if rec.get("bench_model_size") else ""),
        "n_params": rec.get("bench_n_params") or "",
        "phase": phase,
        "n_prompt": n_prompt,
        "n_gen": n_gen,
        # What llama-bench reported it ran, falling back to what was requested.
        # Both other nodes take these from the bench row, and this node already
        # refuses a row whose reported config contradicts the request
        # (refused_config_mismatch), so the two agree or the row is flagged.
        "n_threads": _int(rec.get("bench_n_threads"), _int(rec.get("threads"))),
        "n_batch": rec.get("bench_n_batch") or "",
        "n_gpu_layers": _int(rec.get("bench_n_gpu_layers"), _int(rec.get("n_gpu_layers"))),
        "tokens_per_s": round(tps, 4) if tps else "",
        "tokens_per_s_stddev": (round(_num(rec.get("stddev_ts")), 4)
                                if _num(rec.get("stddev_ts")) is not None else ""),
        # Whole-repetition time, NOT per token.
        "avg_ms": (round(_num(rec.get("avg_ns"), 0.0) / 1e6, 3)
                   if rec.get("avg_ns") else ""),
        "stddev_ms": (round(_num(rec.get("stddev_ns")) / 1e6, 3)
                      if _num(rec.get("stddev_ns")) is not None else ""),
        # The llama-bench repetition COUNT, which the control node multiplies the
        # token count by. This node's `repetitions` is exactly that -r.
        "samples": _int(rec.get("repetitions"), 1) or 1,
        "ms_per_token": ms_per_token,
        # Left to the control node unless a layer count is known here. This node
        # does not read the GGUF header, so it cannot fill it the way Pi 5 does.
        "per_layer_ms": (round(ms_per_token / n_layers, 4)
                         if (ms_per_token != "" and n_layers) else ""),
        # See the module docstring: the control node cannot read this node's
        # power stream, its rate assumptions do not hold at 5 Hz, and its
        # denominator convention differs. Empty says so in band.
        "t_start": "",
        "t_end": "",
        "temp_end_c": temp_end,
        # See THROTTLE_UNAVAILABLE: "unavailable", never "" (which reads as
        # checked-and-clean) and never a flag (which reads as a throughput verdict).
        "throttled_end": THROTTLE_UNAVAILABLE,
        "energy_window": "",
        # The process span IS known and is recorded here, correctly typed, where
        # it cannot feed an energy integral.
        "t_proc_start": rec.get("t_start_unix") or "",
        "t_proc_end": rec.get("t_end_unix") or "",
        # Timed-loop seconds for this row, from llama-bench's own avg_ns. Real,
        # and useful to a reader comparing nodes, without implying a window.
        "rep_busy_s": (round(_num(rec.get("avg_ns"), 0.0) / 1e9
                             * (_int(rec.get("repetitions"), 1) or 1), 6)
                       if rec.get("avg_ns") else ""),
        "rep_gap_s": "",
        "energy_duty": "",
        "load_s": rec.get("load_s") or "",
        "warmup_s": "",
        "idle_start": "",
        "idle_end": "",
        # Present but empty: the control node writes over this file's own
        # fieldnames, so a missing column makes writerows raise after the header
        # is already out, truncating the joined output.
        "energy_j": "",
        "joules_per_token": "",
        "mean_power_w": "",
        "flags": ";".join(flags),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("joined_csv", help="a <base>_bench_energy.csv written by run_matrix.py")
    ap.add_argument("--out", default="", help="default: results.csv beside the input")
    ap.add_argument("--run-id", default="", help="default: the input's <base>")
    ap.add_argument("--n-layers", type=int, default=0,
                    help="model layer count, to fill per_layer_ms. Unset leaves it empty "
                         "rather than guessing; the control node can recompute it.")
    args = ap.parse_args()

    with open(args.joined_csv, newline="", encoding="utf-8") as fh:
        recs = list(csv.DictReader(fh))
    if not recs:
        sys.exit("no rows in %s" % args.joined_csv)

    base = os.path.basename(args.joined_csv)
    run_id = args.run_id or re.sub(r"_bench_energy\.csv$", "", base)
    out_path = args.out or os.path.join(os.path.dirname(args.joined_csv) or ".",
                                        "results.csv")

    rows, omitted = [], []
    for rec in recs:
        row = row_to_shared(rec, run_id, args.n_layers)
        if row is None:
            omitted.append("%s/%s" % (rec.get("case_id"), rec.get("row_phase")))
        else:
            rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("wrote %s  (%d rows of %d)" % (out_path, len(rows), len(recs)))
    if omitted:
        # Reported, never silent. A `combined` row has no pp/tg counterpart on the
        # other nodes, so it cannot enter the shared table -- but its absence is a
        # fact about the table, not about the measurement, which is still in the
        # node's own CSV.
        print("omitted %d row(s) with no shared-schema phase: %s"
              % (len(omitted), ", ".join(omitted)))
    flagged = sum(1 for r in rows if r["flags"])
    if flagged:
        print("%d row(s) carry flags -- a non-empty `flags` is an opt-out from the "
              "placement rule" % flagged)
    print("energy columns are intentionally empty; see the module docstring")


if __name__ == "__main__":
    main()
