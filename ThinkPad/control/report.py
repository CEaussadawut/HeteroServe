"""Turn recorded data into the tables and figures the report and the paper need.

Markdown and CSV only - no plotting library is required to get the numbers out. Plots are
optional (matplotlib, if installed) because a missing chart should never block the result.
"""
from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from . import analysis


def _md_table(rows, columns, headers=None) -> str:
    if not rows:
        return "_no data_\n"
    headers = headers or columns
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                value = ("%.4f" % value).rstrip("0").rstrip(".")
            cells.append("" if value is None else str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


# ------------------------------------------------- device characterisation
def characterisation(results_energy_csv, out_md=None, out_csv=None) -> str:
    """The project's first deliverable, as a table."""
    table = analysis.characterisation_table(results_energy_csv)

    if out_csv and table:
        with Path(out_csv).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)

    md = ["# Device characterisation",
          "",
          "Generated %s from `%s`." % (datetime.now(timezone.utc).isoformat(), results_energy_csv),
          "",
          "Each row is one node, model, quantisation, phase and thread count. Energy columns are",
          "empty where the energy join was refused - see the `flagged` column and",
          "`docs/03_energy_collector.md` for what causes a refusal.",
          "",
          "`runs` counts every underlying run. `E-runs` counts the unflagged ones backing the",
          "energy means; `T-runs` the unthrottled ones backing the throughput means. The two",
          "differ because the flags mean different things: a power-trace flag (gap, late-start,",
          "uncalibrated-channel) invalidates the energy but not the benchmark, while a throttle",
          "flag means the clock moved under the run and invalidates both.",
          ""]
    md.append(_md_table(
        table,
        ["node", "model", "quant", "phase", "threads", "tokens_per_s", "tokens_per_s_sd",
         "joules_per_token", "mean_power_w", "wh_per_1k_tokens",
         "runs", "energy_runs", "throughput_runs", "flagged"],
        ["node", "model", "quant", "phase", "thr", "tok/s", "sd", "J/tok", "W", "Wh/1k",
         "runs", "E-runs", "T-runs", "flagged"]))

    flagged = sum(r["flagged"] for r in table)
    if flagged:
        md.append("")
        md.append("> %d of %d underlying runs carry validity flags and are excluded from the "
                  "energy means. Throughput means additionally drop any throttled run, and "
                  "only those." % (flagged, sum(r["runs"] for r in table)))
        # A row whose runs were ALL flagged has no energy at all. Saying so is the honest
        # reading: it is a result about the measurement, not a hole in the table.
        def _name(r):
            return "%s/%s/%s" % (r["node"], r["model"], r["phase"])

        empty = [r for r in table if r.get("energy_runs") == 0]
        if empty:
            md.append(">")
            md.append("> %d row(s) have no unflagged run and therefore no energy figure at all: "
                      "%s. Re-run those conditions rather than reading the blank as zero."
                      % (len(empty), ", ".join(_name(r) for r in empty)))
        throttled_out = [r for r in table if r.get("throughput_runs") == 0]
        if throttled_out:
            md.append(">")
            md.append("> %d row(s) were throttled on every run, so they have no throughput "
                      "figure: %s. That is a result about the board, not a missing "
                      "measurement - cool it or lower the clock ceiling and re-run."
                      % (len(throttled_out), ", ".join(_name(r) for r in throttled_out)))

    text = "\n".join(md)
    if out_md:
        Path(out_md).write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------- placement results
def placement_comparison(comparison: dict, out_md=None) -> str:
    """Rule vs even split vs single node vs with/without the straggler."""
    rows = []
    for label, key in (("heteroserve rule", "rule"), ("even split", "even"),
                       ("rule, straggler forced in", "with_straggler"),
                       ("rule, straggler dropped", "without_straggler")):
        p = comparison.get(key)
        if not p:
            continue
        if not p.feasible:
            rows.append({"placement": label, "status": "infeasible: " + p.reason})
            continue
        pred = p.predicted
        rows.append({
            "placement": label,
            "nodes": " -> ".join(p.participants),
            "layers": ", ".join("%s:%d" % (n, p.layers[n]) for n in p.participants),
            "ms_per_token": pred.get("ms_per_token"),
            "tokens_per_s": pred.get("tokens_per_s"),
            "joules_per_token": pred.get("joules_per_token"),
            "mean_power_w": pred.get("mean_power_w"),
            "comm_pct": round(100 * pred.get("comm_fraction", 0), 1),
            "status": "ok",
        })

    for nid, p in (comparison.get("singles") or {}).items():
        rows.append({
            "placement": "single node: " + nid,
            "nodes": nid,
            "layers": "all" if p.feasible else "-",
            "ms_per_token": p.predicted.get("ms_per_token") if p.feasible else None,
            "tokens_per_s": p.predicted.get("tokens_per_s") if p.feasible else None,
            "joules_per_token": p.predicted.get("joules_per_token") if p.feasible else None,
            "mean_power_w": p.predicted.get("mean_power_w") if p.feasible else None,
            "status": "ok" if p.feasible else "infeasible: " + p.reason,
        })

    md = ["# Placement comparison - %s (%s phase)" % (comparison["model"], comparison["phase"]),
          "",
          "Predicted from the measured device profiles, energy weight lambda = %s."
          % comparison["energy_weight"],
          "",
          _md_table(rows,
                    ["placement", "nodes", "layers", "ms_per_token", "tokens_per_s",
                     "joules_per_token", "mean_power_w", "comm_pct", "status"],
                    ["placement", "pipeline", "layers", "ms/tok", "tok/s", "J/tok", "W", "comm%", "status"])]

    if "tpot_improvement_pct" in comparison:
        md += ["", "## Against the even-split baseline", "",
               "- TPOT: **%.1f%%** better than even split (MVP target 10%%, stretch 25%%)"
               % comparison["tpot_improvement_pct"],
               "- Energy: **%.1f%%** better than even split (stretch target 25%%)"
               % comparison["energy_improvement_pct"]]

    if "straggler_cost_pct" in comparison:
        md += ["", "## The straggler question (QR8)", "",
               "Forcing the straggler into the pipeline costs **%.1f%%** in per-token latency "
               "and **%.1f%%** in energy against the best placement that excludes it."
               % (comparison["straggler_cost_pct"], comparison["straggler_energy_pct"]),
               "",
               "A positive number on both means the straggler should be dropped for this model "
               "and link tier; a negative latency number means it is earning its place. The "
               "answer is expected to differ by model size, because a model that does not fit "
               "without it has no alternative."]

    text = "\n".join(md) + "\n"
    if out_md:
        Path(out_md).write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------------ measured sweep
def sweep_report(sweep_json, out_md=None) -> str:
    """Measured results of one sweep, including how well the rule predicted them."""
    from . import experiment  # local import keeps this module importable on its own

    data = json.loads(Path(sweep_json).read_text(encoding="utf-8"))
    rows = experiment.summarise(data)
    errors = experiment.prediction_error(data)

    grouped = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped.setdefault((row["strategy"], row["tier"]), []).append(row)

    summary = []
    for (strategy, tier), group in sorted(grouped.items()):
        tpots = [r["tpot_ms"] for r in group if r.get("tpot_ms")]
        ttfts = [r["ttft_ms"] for r in group if r.get("ttft_ms")]
        summary.append({
            "strategy": strategy, "tier": tier, "runs": len(group),
            "tpot_ms": round(statistics.fmean(tpots), 2) if tpots else None,
            "tpot_sd": round(statistics.pstdev(tpots), 2) if len(tpots) > 1 else 0.0,
            "ttft_ms": round(statistics.fmean(ttfts), 1) if ttfts else None,
            "tok_s": round(1000 / statistics.fmean(tpots), 3) if tpots else None,
            "predicted_ms": group[0].get("predicted_ms_per_token"),
            "pred_error_pct": (errors.get("%s@%s" % (strategy, tier)) or {}).get("mean_pct"),
        })

    md = ["# Sweep: %s" % data.get("sweep_id", Path(sweep_json).stem), ""]
    if data.get("aborted"):
        md += ["**Aborted.** %s" % data["aborted"], ""]

    md += ["## Measured", "",
           _md_table(summary,
                     ["strategy", "tier", "runs", "tpot_ms", "tpot_sd", "ttft_ms", "tok_s",
                      "predicted_ms", "pred_error_pct"],
                     ["strategy", "tier", "n", "TPOT ms", "sd", "TTFT ms", "tok/s",
                      "predicted", "pred err %"])]

    failures = [r for r in rows if r.get("status") not in ("ok", None)]
    if failures:
        md += ["", "## Runs that did not produce a measurement", "",
               _md_table(failures, ["strategy", "tier", "status"], ["strategy", "tier", "what happened"])]

    md += ["", "## Cost-model accuracy", "",
           "`pred err %` is (predicted - measured) / measured. A consistent sign across "
           "strategies means the cost model has a systematic bias worth fixing before the "
           "rule's output is trusted; scatter around zero means the model is unbiased and "
           "merely imprecise.", ""]

    text = "\n".join(md)
    if out_md:
        Path(out_md).write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------------------ figures
def pareto_csv(front: list, out_csv) -> str:
    if not front:
        return ""
    with Path(out_csv).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(front[0].keys()))
        writer.writeheader()
        for row in front:
            row = dict(row)
            row["layers"] = json.dumps(row.get("layers"))
            row["order"] = ",".join(row.get("order", []))
            writer.writerow(row)
    return str(out_csv)


def plot_pareto(front: list, out_png) -> str | None:
    """Tokens/s against watts. Optional: returns None if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not front:
        return None
    xs = [p["tokens_per_s"] for p in front]
    ys = [p["mean_power_w"] for p in front]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, "o-")
    for p in front:
        ax.annotate("lambda=%g" % p["energy_weight"],
                    (p["tokens_per_s"], p["mean_power_w"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("tokens / s")
    ax.set_ylabel("mean cluster power (W)")
    ax.set_title("Energy Pareto front across placements")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return str(out_png)
