#!/usr/bin/env python3
"""Scheduler upgrade 2 re-tested on the four-node cluster (2026-10-24): a linear correction of the predictor's decode error,
registered before any fit, then scored. Laptop only; nothing runs on the boards.

    python3 testbed/scripts/correction4_20261024.py register [--dry-run]   # writes testbed/dataset/rule_correction4_20261024.json
    python3 testbed/scripts/correction4_20261024.py score                  # writes testbed/results/correction4_20261024/

The 2026-10-10 test (rule_correction_20261010.json) FAILED (leave-one-family-out family-mean |error| 6.33 -> 5.42 %, ratio
0.856 against the 0.70 needed) on Pi 4 + Jetson-era rows, and its write-up named what a next registration would need: the
visit term entered as milliseconds over the prediction, not as a count. The user's rule of 2026-10-21 puts every upgrade back
on the four nodes, so this registration takes only four-node rows and that feature form.

THE TABLE (embedded at registration with its sha256): every placement of the four-node full evaluation
(testbed/results/brief_20261023/fe4_pairs.json: 11 configurations, 52 placements), y = measured mean / registered
prediction - 1 (the mean server gap; fe4_pairs' mean_ms and pred_ms). Families by model (TinyLlama's five quantisations one
family; the 30B MoE's two one family).
THE FEATURES (fixed here): hybrid (1 for the qwen35-architecture Qwen3.5-9B and Qwen3.8-27B); moe (1 for Qwen3-30B-A3B);
visits_per_ms (the plan's RPC workers over its predicted ms: a fixed cost per visit as a share of the token); boards_share
(the share of the plan's blocks on the Raspberry Pis and the Jetson); throttled (1 when the placement's registered thermal
verdict is THROTTLED). A feature constant over a fit's rows is left out of that fit.
THE MODEL: ordinary least squares of y on an intercept and the features; corrected = predicted x (1 + fitted y).
THE TEST, CORR4-LOFO: each family's rows corrected by the fit on all other families; the statistic is the mean over
families of each family's mean |error|. HOLDS when the corrected statistic is at most 0.70 x the uncorrected one; the
correction is adopted only if it holds. Reported beside it: the row-weighted mean |error| before and after, every fit's
coefficients, and the all-rows fit.
"""
import argparse, datetime, hashlib, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "testbed", "results", "brief_20261023", "fe4_pairs.json")
REG = os.path.join(ROOT, "testbed", "dataset", "rule_correction4_20261024.json")
OUTD = os.path.join(ROOT, "testbed", "results", "correction4_20261024")
FEATURES = ("hybrid", "moe", "visits_per_ms", "boards_share", "throttled")
BOARDS = ("pi5", "pi4", "jetson")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def family(model):
    m = model.lower()
    for k in ("tinyllama", "qwen3.5-9b", "mistral-nemo", "qwen3-14b", "qwen3.8-27b", "qwen3-30b-a3b"):
        if k in m:
            return k
    raise ValueError(model)


def table():
    d = json.load(open(SRC))
    rows = []
    for m in d["models"]:
        fam = family(m["model"])
        for p in m["placements"]:
            parts = [x.split() for x in p["plan"].split(" + ")]
            blocks = {n: int(k) for n, k in parts}
            visits = len(parts) - 1
            rows.append({"id": "%s:%s" % (m["run"], p["label"]), "family": fam, "model": m["model"], "plan": p["plan"],
                         "measured_ms": p["mean_ms"], "predicted_ms": p["pred_ms"], "y": p["mean_ms"] / p["pred_ms"] - 1,
                         "hybrid": 1.0 if fam in ("qwen3.5-9b", "qwen3.8-27b") else 0.0,
                         "moe": 1.0 if fam == "qwen3-30b-a3b" else 0.0,
                         "visits_per_ms": visits / p["pred_ms"],
                         "boards_share": sum(blocks.get(n, 0) for n in BOARDS) / sum(blocks.values()),
                         "throttled": 1.0 if p.get("thermal") == "THROTTLED" else 0.0})
    return rows


def ols(rows, feats):
    """Least squares y ~ 1 + feats by the normal equations (Gaussian elimination); features constant over rows dropped."""
    use = [f for f in feats if len({r[f] for r in rows}) > 1]
    X = [[1.0] + [r[f] for f in use] for r in rows]
    y = [r["y"] for r in rows]
    n = len(X[0])
    A = [[sum(X[k][i] * X[k][j] for k in range(len(X))) for j in range(n)] for i in range(n)]
    b = [sum(X[k][i] * y[k] for k in range(len(X))) for i in range(n)]
    M = [A[i] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            raise ValueError("singular normal equations over %s" % use)
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * z for x, z in zip(M[r], M[c])]
    coef = [M[i][n] / M[i][i] for i in range(n)]
    return {"intercept": coef[0], **dict(zip(use, coef[1:]))}


def corrected_err(r, fit):
    yhat = fit["intercept"] + sum(v * r[f] for f, v in fit.items() if f != "intercept")
    return abs(r["measured_ms"] / (r["predicted_ms"] * (1 + yhat)) - 1)


def register(dry):
    if os.path.exists(REG) and not dry:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(REG, ROOT))
    rows = table()
    t = json.dumps(rows, sort_keys=True)
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/correction4_20261024.py", "features": list(FEATURES),
           "model": "OLS of y = measured / predicted - 1 on an intercept and the features (constant features dropped per fit)",
           "test": {"id": "CORR4-LOFO", "ratio_max": 0.70,
                    "claim": "leave one family out: the mean over families of each family's mean |error| after correction is at "
                             "most 0.70 x before; adopted only if it holds"},
           "replaces": "testbed/dataset/rule_correction_20261010.json (FAILS, 0.856)",
           "table": rows, "table_sha256": hashlib.sha256(t.encode()).hexdigest(),
           "inputs_sha256": {"testbed/results/brief_20261023/fe4_pairs.json": sha(SRC),
                             "testbed/scripts/correction4_20261024.py": sha(os.path.abspath(__file__))}}
    fams = {}
    for r in rows:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    print("%d rows; families %s" % (len(rows), fams))
    if dry:
        return
    json.dump(reg, open(REG, "w"), indent=1)
    os.chmod(REG, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(REG, ROOT), sha(REG)[:16]))


def score():
    reg = json.load(open(REG))
    rows = reg["table"]
    rebuilt = hashlib.sha256(json.dumps(table(), sort_keys=True).encode()).hexdigest()
    fams = sorted({r["family"] for r in rows})
    before, after, fits = {}, {}, {}
    for f in fams:
        test = [r for r in rows if r["family"] == f]
        train = [r for r in rows if r["family"] != f]
        fit = ols(train, reg["features"])
        fits[f] = {k: round(v, 6) for k, v in fit.items()}
        before[f] = statistics.fmean(abs(r["y"]) for r in test)
        after[f] = statistics.fmean(corrected_err(r, fit) for r in test)
    b, a = statistics.fmean(before.values()), statistics.fmean(after.values())
    rb = statistics.fmean(abs(r["y"]) for r in rows)
    ra = statistics.fmean(corrected_err(r, fits[r["family"]]) for r in rows)
    verdict = "HOLDS" if a <= reg["test"]["ratio_max"] * b else "FAILS"
    s = {"scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "registration": os.path.relpath(REG, ROOT), "registration_sha256": sha(REG),
         "table_matches_record": rebuilt == reg["table_sha256"],
         "code_changed_since_registration": sha(os.path.abspath(__file__)) != reg["inputs_sha256"]["testbed/scripts/correction4_20261024.py"],
         "CORR4-LOFO": {"verdict": verdict, "family_mean_before_pct": round(100 * b, 3), "family_mean_after_pct": round(100 * a, 3),
                        "ratio": round(a / b, 4), "ratio_max": reg["test"]["ratio_max"]},
         "row_weighted_before_pct": round(100 * rb, 3), "row_weighted_after_pct": round(100 * ra, 3),
         "by_family": {f: {"rows": sum(1 for r in rows if r["family"] == f), "before_pct": round(100 * before[f], 3),
                           "after_pct": round(100 * after[f], 3), "fit_without": fits[f]} for f in fams},
         "all_rows_fit": {k: round(v, 6) for k, v in ols(rows, reg["features"]).items()},
         "adopted": verdict == "HOLDS"}
    os.makedirs(OUTD, exist_ok=True)
    json.dump(s, open(os.path.join(OUTD, "score.json"), "w"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("register", "score")); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    register(a.dry_run) if a.cmd == "register" else score()
