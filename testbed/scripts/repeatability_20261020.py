#!/usr/bin/env python3
"""How repeatable the measurements are, at every level the record allows (the advisor, 2026-10-16: "repeatability").
Laptop only; reads published files, re-scores nothing.

    python3 testbed/scripts/repeatability_20261020.py [--pi5b SESSION_DIR ...]

Levels, smallest first:
  1. within a run: llama-bench's repetitions (characterisation.csv rep_range_pct = (max - min) / mean of the -r 3 samples);
  2. the same configuration in separate runs: rows of characterisation.csv with the same node, device, model and test from
     different run directories (spread = max / min - 1 of their means);
  3. across rounds of a served evaluation: full_eval run directories, each round a fresh load of the model -- the per-round
     server TPOT P50 (qr3_tpot.samples over the round's requests) and the per-round gross J/token (summary.json);
  4. across sessions of the same served plan: spec_serve's two "before" sessions, power_modes' prior sessions;
  5. energy per token across invocations: energy_phase run folders' score.json (spread across the three invocations);
  6. across units of one board model: the replacement Pi 5 against the old one (--pi5b: the scored session folders).
Writes testbed/results/repeatability_<stamp>/{repeatability.json, report.txt}.
"""
import argparse, collections, csv, datetime, glob, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import qr3_tpot as Q                                        # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")
RES = os.path.join(ROOT, "testbed", "results")


def spread(xs):
    xs = [x for x in xs if x is not None]
    return round(100 * (max(xs) / min(xs) - 1), 2) if len(xs) > 1 and min(xs) > 0 else None


def cv(xs):
    xs = [x for x in xs if x is not None]
    return round(100 * statistics.stdev(xs) / statistics.fmean(xs), 2) if len(xs) > 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi5b", nargs="*", default=[])
    a = ap.parse_args()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RES, "repeatability_%s" % stamp)
    os.makedirs(out)
    res = {"made": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "script": os.path.relpath(__file__, ROOT)}
    rows = list(csv.DictReader(open(os.path.join(DS, "characterisation.csv"))))
    # 1. within a run
    by_board = collections.defaultdict(list)
    for r in rows:
        if r["rep_range_pct"] not in ("", None):
            by_board["%s/%s" % (r["node_id"], r["device"])].append(float(r["rep_range_pct"]))
    res["within_run"] = {k: {"rows": len(v), "median_pct": round(statistics.median(v), 2), "max_pct": round(max(v), 2),
                             "share_under_1pct": round(sum(1 for x in v if x < 1) / len(v), 3)} for k, v in sorted(by_board.items())}
    # 2. the same configuration in separate runs
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["node_id"], r["device"], r["model"], r["test"])].append((float(r["mean_ts"]), r["source"], r["date"][:10]))
    res["across_runs"] = [{"node": k[0], "device": k[1], "model": k[2], "test": k[3], "runs": len(v), "spread_pct": spread([x[0] for x in v]),
                           "means": [round(x[0], 3) for x in v], "dates": sorted(set(x[2] for x in v))}
                          for k, v in sorted(groups.items()) if len(set(x[1] for x in v)) > 1]
    # 3. across rounds of served evaluations
    fe = []
    for d in sorted(glob.glob(os.path.join(RES, "full_eval_*_2026*"))):
        if not os.path.isdir(d) or "smoke" in d or not os.path.exists(os.path.join(d, "summary.json")):
            continue
        summ = json.load(open(os.path.join(d, "summary.json")))
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        for ln in open(os.path.join(d, "run.jsonl")):
            x = json.loads(ln)
            if x["kind"] != "req":
                continue
            r = x["data"]
            sg = Q.samples(r)[1]
            per[r["placement"]][r["round"]].extend(sg)
        for lab, rounds in per.items():
            p50s = [Q.pct(v, 50) for _, v in sorted(rounds.items()) if v]
            j = summ.get(lab, {}).get("j_per_token_all_boards_rounds")
            fe.append({"run": os.path.relpath(d, RES), "placement": lab, "rounds": len(p50s),
                       "tpot_p50_by_round_ms": [round(x, 1) for x in p50s], "tpot_p50_spread_pct": spread(p50s), "tpot_p50_cv_pct": cv(p50s),
                       "p50_ci95": summ.get(lab, {}).get("p50_ci95"), "j_by_round": j, "j_spread_pct": spread(j or []), "j_cv_pct": cv(j or [])})
    res["across_rounds"] = fe
    # 4. across sessions of one served plan
    sess = []
    for p in sorted(glob.glob(os.path.join(RES, "spec_serve_*", "score.json"))):
        s = json.load(open(p))
        bs = s.get("arms", {}).get("base", {}).get("by_session")
        if bs:
            sess.append({"run": os.path.relpath(os.path.dirname(p), RES), "what": "spec_serve base (no draft), before and after",
                         "ms_per_token": bs, "drift_pct": s.get("base_drift_pct")})
    for p in sorted(glob.glob(os.path.join(DS, "power_modes_predictions_*.json"))):
        pr = json.load(open(p)).get("prior", {})
        if pr.get("by_session"):
            v = {k: x["T_ms"] for k, x in pr["by_session"].items()}
            sess.append({"run": os.path.relpath(p, ROOT), "what": "power modes' prior: the 14B plan, two sessions", "ms_per_token": v,
                         "drift_pct": spread(list(v.values()))})
    res["across_sessions"] = sess
    # 5. energy across invocations
    en = []
    for p in sorted(glob.glob(os.path.join(RES, "energy_phase_*", "score.json"))):
        s = json.load(open(p))
        ph = s.get("phases", {})
        en.append({"run": os.path.relpath(os.path.dirname(p), RES),
                   **{"%s_%s" % (k, f): (ph.get(k, {}).get(f) or {}).get("spread_pct") for k in ("pp", "tg")
                      for f in ("tps", "j_per_token_net_trap", "j_per_token_net_counter")}})
    res["energy_invocations"] = en
    # 6. across units
    units = []
    for d in a.pi5b:
        p = os.path.join(d, "score.json")
        if os.path.exists(p):
            s = json.load(open(p))
            units.append({"session": os.path.relpath(os.path.abspath(d), ROOT), "registration": s["registration"],
                          "hypotheses": {k: {x: v.get(x) for x in ("verdict", "max_abs_diff_pct", "mean_abs_diff_pct", "diff_pct")}
                                         for k, v in s["hypotheses"].items()}})
    res["across_units"] = units
    json.dump(res, open(os.path.join(out, "repeatability.json"), "w"), indent=1)
    L = ["Repeatability, %s" % res["made"], "", "1. Within a run (llama-bench -r 3: (max - min) / mean)"]
    for k, v in res["within_run"].items():
        L.append("  %-22s %3d rows  median %5.2f %%  max %5.2f %%  under 1 %%: %3.0f %%" % (k, v["rows"], v["median_pct"], v["max_pct"],
                                                                                         100 * v["share_under_1pct"]))
    L += ["", "2. The same configuration in separate runs (max / min - 1)"]
    for x in res["across_runs"]:
        L.append("  %-7s %-15s %-26s %-6s %d runs  %6.2f %%  %s  %s" % (x["node"], x["device"], x["model"], x["test"], x["runs"],
                                                                     x["spread_pct"] or 0, x["means"], ",".join(x["dates"])))
    L += ["", "3. Across rounds of a served evaluation (each round reloads the model)"]
    for x in res["across_rounds"]:
        L.append("  %-44s %-8s %d rounds  TPOT P50 spread %5.2f %% (CV %4.2f %%)  J/token spread %s %% (CV %s %%)" % (
            x["run"], x["placement"], x["rounds"], x["tpot_p50_spread_pct"] or 0, x["tpot_p50_cv_pct"] or 0, x["j_spread_pct"], x["j_cv_pct"]))
    L += ["", "4. Across sessions of one served plan"]
    for x in res["across_sessions"]:
        L.append("  %-44s %s: %s, drift %s %%" % (x["run"], x["what"], x["ms_per_token"], x["drift_pct"]))
    L += ["", "5. Energy per token across the three invocations of a phase run (spread %)"]
    for x in res["energy_invocations"]:
        L.append("  %-46s pp t/s %s  J trap %s  ctr %s | tg t/s %s  J trap %s  ctr %s" % (
            x["run"], *[("%.2f" % x[k]) if x[k] is not None else "?" for k in
                        ("pp_tps", "pp_j_per_token_net_trap", "pp_j_per_token_net_counter", "tg_tps", "tg_j_per_token_net_trap",
                         "tg_j_per_token_net_counter")]))
    L += ["", "6. Across units of one board model"] + ["  %s: %s" % (u["session"], json.dumps(u["hypotheses"])) for u in units]
    open(os.path.join(out, "report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
