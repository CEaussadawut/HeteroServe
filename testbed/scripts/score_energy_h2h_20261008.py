#!/usr/bin/env python3
"""Score the energy head-to-head of 2026-10-08 (engine against Halda's plan, Pi 4 + Jetson, TinyLlama). Read-only.

    python3 testbed/scripts/score_energy_h2h_20261008.py RUN_DIR [--out JSON]

Reads RUN_DIR/score.json (score_energy_pool_run.py's: per run and phase, the pool's gross and net J/token by trapezoid
and by counter, per-board mean W, the measured t/s) and the registration it names, and applies the registered hypotheses
(energy_h2h_predictions_20261008.json, "h2h_hypotheses") as written:
  H-DEC  the engine's mean gross decode J/token below Halda's, by trapezoid AND by counter (means over each plan's runs)
  H-PRE  the two plans' mean gross prefill J/token within +-2 % of each other, by both methods
  ACC    every run's gross J/token (trapezoid) within 10 % of energy_model.energy at that run's measured time per token
Prints the table and the verdicts; writes --out (default RUN_DIR/h2h_score.json).
"""
import json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import energy_model as EM                                   # noqa: E402
import predict_pool as P                                    # noqa: E402


def main(argv):
    if not argv:
        sys.exit(__doc__)
    run = argv[0]
    out = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(run, "h2h_score.json")
    sc = json.load(open(os.path.join(run, "score.json")))
    reg = json.load(open(os.path.join(ROOT, sc["registration"])))
    power = EM.power_table()
    shapes, pairs, C = EM._inputs("Q4_K_M")
    runs, per = [], {}
    for r in sc["runs"]:
        if r.get("exit") != 0:
            runs.append({"label": r["label"], "cell": r["cell"], "failed": True})
            continue
        c = reg["configs"][r["cell"]]
        pipe = [(s["node"], s["blocks"]) for s in c["pipeline"]]
        q = shapes[c["model"]]
        L, _ = P.link_view(pairs, pipe[0][0], [n for n, _ in pipe])
        x = {"label": r["label"], "cell": r["cell"]}
        for ph in ("pp", "tg"):
            m = r["phases"][ph]
            T = 1000.0 / m["bench"]["avg_ts"]
            model = EM.energy(pipe, q, C, L, power, ph, head_at=c.get("head_at"), t_ms=T,
                              wait={"jetson": power["jetson"]["idle"]})
            x[ph] = {"tok_s": m["bench"]["avg_ts"], "gross_trap": m["pool"]["j_per_token_gross_trap"],
                     "gross_counter": m["pool"]["j_per_token_gross_counter"],
                     "net_trap": m["pool"]["j_per_token_net_trap"], "net_counter": m["pool"]["j_per_token_net_counter"],
                     "board_w": {n: m["per_board"][n]["mean_w"] for n in m["per_board"]},
                     "model_gross": model["gross_j"],
                     "model_err_pct": 100 * (model["gross_j"] / m["pool"]["j_per_token_gross_trap"] - 1)}
            per.setdefault((r["cell"], ph), []).append(x[ph])
        runs.append(x)
    mean = lambda cell, ph, k: statistics.fmean(v[k] for v in per[(cell, ph)])
    verdicts = {}
    dec = {k: 100 * (mean("eh.tiny.engine", "tg", k) / mean("eh.tiny.halda", "tg", k) - 1) for k in ("gross_trap", "gross_counter")}
    verdicts["H-DEC"] = {"engine_vs_halda_pct": dec, "pass": all(v < 0 for v in dec.values())}
    pre = {k: 100 * (mean("eh.tiny.engine", "pp", k) / mean("eh.tiny.halda", "pp", k) - 1) for k in ("gross_trap", "gross_counter")}
    verdicts["H-PRE"] = {"engine_vs_halda_pct": pre, "pass": all(abs(v) <= 2.0 for v in pre.values())}
    errs = [x[ph]["model_err_pct"] for x in runs if not x.get("failed") for ph in ("pp", "tg")]
    verdicts["ACC"] = {"worst_pct": max(errs, key=abs) if errs else None, "pass": bool(errs) and all(abs(e) <= 10 for e in errs)}
    print("%-22s %-3s %8s %10s %10s %10s %10s %10s   board W" % ("run", "ph", "tok/s", "gross trap", "gross cnt", "net trap",
                                                                "model", "model err"))
    for x in runs:
        if x.get("failed"):
            print("%-22s FAILED" % x["label"]); continue
        for ph in ("pp", "tg"):
            v = x[ph]
            print("%-22s %-3s %8.2f %10.4f %10.4f %10.4f %10.4f %+9.1f%%   %s" % (
                x["label"], ph, v["tok_s"], v["gross_trap"], v["gross_counter"], v["net_trap"], v["model_gross"],
                v["model_err_pct"], " ".join("%s %.2f" % kv for kv in v["board_w"].items())))
    for cell in ("eh.tiny.engine", "eh.tiny.halda"):
        for ph in ("pp", "tg"):
            if (cell, ph) in per:
                xs = [v["gross_trap"] for v in per[(cell, ph)]]
                print("%-15s %-3s %d runs: gross %.4f J/token (trapezoid), spread %.1f %%; %.2f tok/s" % (
                    cell, ph, len(xs), statistics.fmean(xs), 100 * (max(xs) - min(xs)) / statistics.fmean(xs),
                    mean(cell, ph, "tok_s")))
    for k, v in verdicts.items():
        print("%-6s %s  %s" % (k, "PASS" if v["pass"] else "FAIL", {a: (round(b, 2) if isinstance(b, float) else
                                                                      {c: round(d, 2) for c, d in b.items()} if isinstance(b, dict) else b)
                                                                  for a, b in v.items() if a != "pass"}))
    json.dump({"run": run, "registration": sc["registration"], "runs": runs, "verdicts": verdicts}, open(out, "w"), indent=1)
    print("wrote %s" % os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
