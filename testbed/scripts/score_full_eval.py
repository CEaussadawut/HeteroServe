#!/usr/bin/env python3
"""Score a full-evaluation run (full_eval.py) against its registration's three hypotheses, as written.

    python3 testbed/scripts/score_full_eval.py testbed/results/full_eval_<name>_<stamp>

Reads <run>/run.jsonl (the registration it ran, every request, every serve failure), <run>/summary.json (full_eval.py
summary: per placement, server TPOT stats, the 95 % CIs of P50 and P95, gross J/token) and the registration.
  FE-TPOT    the placement with the lowest predicted decode ms has the lowest measured server TPOT P50 and the lowest P95
             (strictly), and both its CIs lie clear of every other placement's: its upper bound below their lower bound,
             for P50 and for P95
  FE-ENERGY  the placement with the lowest predicted gross J/token, every pool board powered (decode_gross_j_all_boards),
             has the strictly lowest measured J/token over every pool board's outlet (j_per_token_all_boards)
  FE-ACC     every placement's measured mean server TPOT within 10 % of its predicted decode ms, the error taken as
             100 * (predicted / measured - 1): the reading of full_eval.py's summary (mean_tpot_err_pct) and of the network
             study's NS-ACC. The other reading, 100 * (measured / predicted - 1), is reported beside it, and a verdict the
             two readings would split is flagged
These readings were written on 2026-10-09 while the 20261009 pilot was in its first round, before its summary existed.
A placement is complete when it has every registered request (rounds x reqs), each of the registered gen tokens. Every
hypothesis compares all placements, so any incomplete placement makes all three INCOMPLETE; what the data present shows is
reported beside the verdict. The thermal verdicts, the throttle polls, the model checks and any code changed since the
registration are reported, not folded into the verdicts.
Writes <run>/score.json.
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ACC_MAX = 10.0


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def completeness(reg, summ, rows):
    """{placement: {"requests", "short", "rounds", "complete"}}: every registered request, each of gen tokens."""
    want, got = reg["rounds"] * reg["reqs"], {}
    for r in rows:
        if r["kind"] != "req":
            continue
        x = r["data"]
        d = got.setdefault(x["placement"], {"requests": 0, "short": 0, "rounds": set()})
        d["requests"] += 1
        d["rounds"].add(x["round"])
        d["short"] += sum(len(e[1]) for e in x["events"] if e[1]) < reg["gen"]
    out = {}
    for lab in reg["configs"]:
        d = got.get(lab, {"requests": 0, "short": 0, "rounds": set()})
        out[lab] = {"requests": d["requests"], "short": d["short"], "rounds": sorted(d["rounds"]),
                    "complete": d["requests"] == want and d["short"] == 0 and lab in summ}
    return out


def score(reg, summ, rows):
    """The three hypotheses' verdicts from the registration, summary.json and run.jsonl's rows (no file access)."""
    cfg, done = reg["configs"], completeness(reg, summ, rows)
    have = [lab for lab in cfg if lab in summ]
    complete = all(v["complete"] for v in done.values())

    def verdict(ok):
        return ("HOLDS" if ok else "FAILS") if complete else "INCOMPLETE"
    hyp = {}
    # FE-TPOT
    best = min(cfg, key=lambda lab: cfg[lab]["predicted"]["decode_ms"])
    t = {"predicted_best": best}
    if best in summ:
        b, others = summ[best], [lab for lab in have if lab != best]
        t["lowest_p50"] = all(b["server_tpot"]["p50"] < summ[lab]["server_tpot"]["p50"] for lab in others)
        t["lowest_p95"] = all(b["server_tpot"]["p95"] < summ[lab]["server_tpot"]["p95"] for lab in others)
        t["ci_gap_p50_ms"] = {lab: round(summ[lab]["p50_ci95"][0] - b["p50_ci95"][1], 2) for lab in others}   # > 0: clear
        t["ci_gap_p95_ms"] = {lab: round(summ[lab]["p95_ci95"][0] - b["p95_ci95"][1], 2) for lab in others}
        ok = (t["lowest_p50"] and t["lowest_p95"] and all(g > 0 for g in t["ci_gap_p50_ms"].values())
              and all(g > 0 for g in t["ci_gap_p95_ms"].values()))
        t["on_data_present"] = "holds" if ok else "fails"
        t["verdict"] = verdict(ok)
    else:
        t["verdict"] = "INCOMPLETE"
    hyp["FE-TPOT"] = t
    # FE-ENERGY
    beste = min(cfg, key=lambda lab: cfg[lab]["predicted"]["decode_gross_j_all_boards"])
    meas = {lab: summ[lab]["j_per_token_all_boards"] for lab in have}
    e = {"predicted_best": beste, "predicted": {lab: cfg[lab]["predicted"]["decode_gross_j_all_boards"] for lab in cfg},
         "measured": meas}
    if beste in meas:
        ok = all(meas[beste] < meas[lab] for lab in have if lab != beste)
        rnd = {lab: summ[lab].get("j_per_token_all_boards_rounds", []) for lab in have}
        n = min(len(v) for v in rnd.values())
        e["rounds_lowest"] = "%d of %d" % (sum(all(rnd[beste][i] < rnd[lab][i] for lab in have if lab != beste)
                                               for i in range(n)), n)      # descriptive, not the claim
        e["on_data_present"] = "holds" if ok else "fails"
        e["verdict"] = verdict(ok)
    else:
        e["verdict"] = "INCOMPLETE"
    hyp["FE-ENERGY"] = e
    # FE-ACC
    acc = {}
    for lab in have:
        m, p = summ[lab]["server_tpot"]["mean"], cfg[lab]["predicted"]["decode_ms"]
        e1, e2 = 100 * (p / m - 1), 100 * (m / p - 1)
        acc[lab] = {"predicted_ms": p, "measured_mean_ms": m, "err_pct": round(e1, 1), "err_vs_predicted_pct": round(e2, 1),
                    "within": abs(e1) <= ACC_MAX, "within_other_reading": abs(e2) <= ACC_MAX}
    ok = bool(acc) and all(v["within"] for v in acc.values())
    ok2 = bool(acc) and all(v["within_other_reading"] for v in acc.values())
    hyp["FE-ACC"] = {"verdict": verdict(ok), "on_data_present": "holds" if ok else "fails",
                     "worst_pct": max((v["err_pct"] for v in acc.values()), key=abs, default=None),
                     "readings_split": ok != ok2, "placements": acc}
    order_pred = sorted(cfg, key=lambda lab: cfg[lab]["predicted"]["decode_ms"])
    order_meas = sorted(have, key=lambda lab: summ[lab]["server_tpot"]["p50"])
    return {"complete": complete, "placements": done, "hypotheses": hyp,
            "order_predicted_decode": order_pred, "order_measured_p50": order_meas,
            "thermal": {lab: {"worst": summ[lab].get("thermal"), "windows": summ[lab].get("thermal_windows")} for lab in have}}


def main(argv):
    if not argv:
        sys.exit(__doc__)
    run = argv[0]
    rows = [json.loads(l) for l in open(os.path.join(run, "run.jsonl"))]
    start = next(r for r in rows if r["kind"] == "start")["data"]
    reg_path = os.path.join(ROOT, start["registration"])
    reg, summ = json.load(open(reg_path)), json.load(open(os.path.join(run, "summary.json")))
    s = score(reg, summ, rows)
    s = {"run": os.path.relpath(run, ROOT), "registration": start["registration"],
         "registration_sha256": sha(reg_path), "registration_unchanged_since_run": sha(reg_path) == start["registration_sha256"],
         "code_changed_since_registration": [f for f, h in reg["inputs_sha256"].items() if sha(os.path.join(HERE, f)) != h],
         "session_finished": any(r["kind"] == "idle-after end" for r in rows),
         "serve_failed": [r["data"] for r in rows if r["kind"] == "serve FAILED"],
         "model_sha256": [r["data"] for r in rows if r["kind"] == "model_sha256"],
         "throttle_polls": [{k: v for k, v in r["data"].items() if k != "session"} for r in rows if r["kind"] == "throttle poll stop"],
         **s}
    json.dump(s, open(os.path.join(run, "score.json"), "w"), indent=1)
    cfg = reg["configs"]
    print("%-8s %-22s %-4s %8s %8s %8s %8s %7s %9s %9s  %s" % ("place", "plan", "head", "pred", "mean", "P50", "P95", "err%",
                                                               "J/tok all", "predicted", "thermal"))
    for lab in s["order_measured_p50"]:
        x, c = summ[lab], cfg[lab]
        print("%-8s %-22s %-4s %8.1f %8.1f %8.1f %8.1f %+6.1f%% %9.4f %9.4f  %s" % (
            lab, " + ".join("%s %d" % tuple(a) for a in c["assign"]), c["head"], c["predicted"]["decode_ms"],
            x["server_tpot"]["mean"], x["server_tpot"]["p50"], x["server_tpot"]["p95"], s["hypotheses"]["FE-ACC"]["placements"][lab]["err_pct"],
            x["j_per_token_all_boards"], c["predicted"]["decode_gross_j_all_boards"], x.get("thermal")))
    for h, v in s["hypotheses"].items():
        print("%-9s %s  %s" % (h, v["verdict"], {k: w for k, w in v.items() if k not in ("verdict", "placements")}))
    print("complete %s; finished %s; serve failures %d; models ok %s; polls %s; registration unchanged %s; code changed %s" % (
        s["complete"], s["session_finished"], len(s["serve_failed"]), all(m.get("ok") for m in s["model_sha256"]),
        [(p.get("node"), p.get("alive"), p.get("copied")) for p in s["throttle_polls"]], s["registration_unchanged_since_run"],
        s["code_changed_since_registration"]))
    print("wrote %s" % os.path.relpath(os.path.join(run, "score.json"), ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
