#!/usr/bin/env python3
"""The four-node full evaluation as baseline pairs, for the 2026-10-23 advisor brief (the advisor, 2026-10-16: "more baseline
pair"). Laptop only; reads the scored run directories and their registrations, re-scores nothing.

    python3 testbed/scripts/fe4_tables_20261023.py

Per model, its final run (the superseding one where a model ran twice: the 30B Q6_K's moeq6b, the 27B's q27bnc), per placement:
the plan (from the registration), predicted and measured decode (server TPOT P50 / P95 / mean, its error), TTFT P50 and the
gross J per decoded token with every node powered. Placements taken while the Pi 5 was away are left out, as the user decided
(2026-10-21): the 14B's "served1009" (Pi 4 + Jetson) and "lap39" (Pi 4 + laptop). The engine against each baseline: the speed-up
(baseline P50 / engine P50) and the energy saving (1 - engine J / baseline J). Writes testbed/results/brief_20261023/
{fe4_pairs.json, fe4_pairs.md}.
"""
import json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
R = os.path.join(ROOT, "testbed", "results")
RUNS = [  # (label, run dir, registration)
    ("TinyLlama 1.1B Q3_K_M", "full_eval_tinyq3_fe4_20261020_20261021_111154", "full_eval_tinyq3_fe4_20261020.json"),
    ("TinyLlama 1.1B Q4_0", "full_eval_tinyq40_fe4_20261020_20261021_101903", "full_eval_tinyq40_fe4_20261020.json"),
    ("TinyLlama 1.1B Q4_K_M", "full_eval_tinyq4km_fe4_20261020_20261021_082119", "full_eval_tinyq4km_fe4_20261020.json"),
    ("TinyLlama 1.1B Q6_K", "full_eval_tinyq6_fe4_20261020_20261021_120814", "full_eval_tinyq6_fe4_20261020.json"),
    ("TinyLlama 1.1B Q8_0", "full_eval_tinyq8_fe4_20261020_20261021_091335", "full_eval_tinyq8_fe4_20261020.json"),
    ("Qwen3.5-9B Q4_K_M", "full_eval_q9b_fe4_20261020_20261021_062243", "full_eval_q9b_fe4_20261020.json"),
    ("Mistral-Nemo 12B Q4_K_M", "full_eval_nemo_fe4_20261020_20261021_045420", "full_eval_nemo_fe4_20261020.json"),
    ("Qwen3-14B Q4_K_M", "full_eval_q14b_fe4_20261020_20261020_222013", "full_eval_q14b_fe4_20261020.json"),
    ("Qwen3.8-27B Q3_K_M", "full_eval_q27bnc_fe4_20261021_20261021_170855", "full_eval_q27bnc_fe4_20261021.json"),
    ("Qwen3-30B-A3B Q3_K_M (MoE)", "full_eval_moeq3_fe4_20261020_20261021_022834", "full_eval_moeq3_fe4_20261020.json"),
    ("Qwen3-30B-A3B Q6_K (MoE)", "full_eval_moeq6b_fe4_20261021_20261021_131716", "full_eval_moeq6b_fe4_20261021.json"),
]
EXCLUDE = {"served1009", "lap39"}
WHAT = {"engine": "the engine's pick", "boards": "boards only (no laptop)", "single": "best single node", "even": "equal split",
        "default": "llama.cpp default split", "halda": "Halda's windows", "lap_pi4": "laptop + Pi 4 (overflow board)",
        "pi4alone": "Pi 4 alone", "jetsonalone": "Jetson alone"}


def plan(cfg):
    return " + ".join("%s %d" % (n, k) for n, k in cfg["assign"])


def main():
    out, md = [], []
    for label, run, regf in RUNS:
        s = json.load(open(os.path.join(R, run, "summary.json")))
        sc = json.load(open(os.path.join(R, run, "score.json")))
        reg = json.load(open(os.path.join(ROOT, "testbed", "dataset", regf)))
        rows = []
        for lab, p in s.items():
            if lab in EXCLUDE or not isinstance(p, dict) or "server_tpot" not in p:
                continue
            st = p["server_tpot"]
            rows.append({"label": lab, "what": WHAT.get(lab, lab), "plan": plan(reg["configs"][lab]), "head": p.get("head"),
                         "pred_ms": p["predicted"]["decode_ms"], "p50_ms": st["p50"], "p95_ms": st["p95"], "mean_ms": st["mean"],
                         "err_pct": p.get("mean_tpot_err_pct"), "ttft_ms_p50": p.get("ttft_ms_p50"),
                         "j_tok_all": p.get("j_per_token_all_boards"), "thermal": p.get("thermal"), "n": st["n"]})
        eng = [r for r in rows if r["label"] == "engine"][0]
        for r in rows:
            r["speedup_vs_engine"] = round(r["p50_ms"] / eng["p50_ms"], 3)
            r["energy_saving_by_engine_pct"] = round(100 * (1 - eng["j_tok_all"] / r["j_tok_all"]), 1) if r["j_tok_all"] else None
        verdicts = {k: h.get("verdict") for k, h in sc["hypotheses"].items()}
        if any(x in s for x in EXCLUDE):
            # the user's rule: scored on the four-node placements alone (the registered verdicts stay as they are)
            verdicts = {k: "%s (registered); on the four-node placements: %s" % (h.get("verdict"), h.get("on_data_present", "-"))
                        for k, h in sc["hypotheses"].items()}
        base = [r for r in rows if r["label"] != "engine"]
        fastest_base = min(base, key=lambda r: r["p50_ms"]) if base else None
        out.append({"model": label, "run": run, "registration": regf, "verdicts": verdicts, "placements": rows,
                    "fastest_baseline": fastest_base["label"] if fastest_base else None,
                    "engine_vs_fastest_baseline": round(fastest_base["p50_ms"] / eng["p50_ms"], 3) if fastest_base else None,
                    "engine_vs_slowest_baseline": round(max(r["p50_ms"] for r in base) / eng["p50_ms"], 3) if base else None})
        md.append("### %s  (%s)" % (label, "; ".join("%s %s" % kv for kv in verdicts.items())))
        md.append("| placement | plan | pred ms | P50 ms | P95 ms | err % | J/token | engine faster by | engine saves J |")
        md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(rows, key=lambda r: r["p50_ms"]):
            md.append("| %s | %s | %.1f | %.1f | %.1f | %s | %s | %s | %s |" % (
                r["what"], r["plan"], r["pred_ms"], r["p50_ms"], r["p95_ms"], r["err_pct"], r["j_tok_all"],
                "-" if r["label"] == "engine" else "%.2fx" % r["speedup_vs_engine"],
                "-" if r["label"] == "engine" else "%s %%" % r["energy_saving_by_engine_pct"]))
        md.append("")
    errs = [abs(r["err_pct"]) for m in out for r in m["placements"] if r["err_pct"] is not None]
    summ = {"models": len(out), "placements": sum(len(m["placements"]) for m in out),
            "abs_err_median_pct": round(statistics.median(errs), 2), "abs_err_max_pct": round(max(errs), 2),
            "within_5_pct": sum(1 for e in errs if e <= 5), "within_10_pct": sum(1 for e in errs if e <= 10), "n_err": len(errs)}
    d = os.path.join(R, "brief_20261023"); os.makedirs(d, exist_ok=True)
    json.dump({"summary": summ, "models": out}, open(os.path.join(d, "fe4_pairs.json"), "w"), indent=1)
    open(os.path.join(d, "fe4_pairs.md"), "w").write("\n".join(md) + "\n")
    print(json.dumps(summ))
    for m in out:
        print("%-28s engine %-30s P50 %8.1f  vs fastest baseline (%s) %.2fx, slowest %.2fx" % (
            m["model"], [r["plan"] for r in m["placements"] if r["label"] == "engine"][0],
            [r["p50_ms"] for r in m["placements"] if r["label"] == "engine"][0], m["fastest_baseline"],
            m["engine_vs_fastest_baseline"] or 0, m["engine_vs_slowest_baseline"] or 0))


if __name__ == "__main__":
    main()
