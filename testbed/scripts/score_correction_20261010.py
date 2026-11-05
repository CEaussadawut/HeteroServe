#!/usr/bin/env python3
"""Score the correction layer's registered test CORR-LOFO (testbed/dataset/rule_correction_20261010.json), as written.

    python3 testbed/scripts/score_correction_20261010.py [--reg PATH] [--out DIR]
        DIR defaults to testbed/results/correction_20261010; writes DIR/score.json and DIR/table_lofo.csv

Fits ONLY the rows the registration embeds. It also rebuilds the table from the record (correction_layer.table()) and
reports whether that still matches the registered sha256: a difference means the record changed after registration, and
is reported, never fitted.
  CORR-LOFO  leave one model family out: each family's rows corrected by the fit on every other family; the statistic is
             the mean over families of the family's mean |error|. HOLDS when the corrected statistic is at most 0.70 x the
             uncorrected one.
Reported beside it, as registered: the row-weighted mean |error|; both statistics without the qwen2.5-14b family (the
figures for Documents/); each family's mean |error|; every fit's coefficients; the all-rows fit; and the CV+ q of the 90 %
interval CORR-COVER will use (10 folds by sha256 of the row id), with the leave-one-family-out q beside it for a family
the table has never seen. CORR-COVER itself is prospective and stays PENDING here.
"""
import csv, hashlib, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import correction_layer as CL                               # noqa: E402

REG = os.path.join(ROOT, "testbed", "dataset", "rule_correction_20261010.json")
OUT = os.path.join(ROOT, "testbed", "results", "correction_20261010")
CUT = 0.70
EXCLUDE_FOR_DOCUMENTS = "qwen2.5-14b"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def stats(rows):
    """{uncorrected, corrected} x {family_mean, row_weighted, by_family} for LOFO-corrected rows."""
    out = {}
    for key, name in (("y", "uncorrected"), ("e_corr", "corrected")):
        fam = CL.family_mae(rows, key)
        out[name] = {"family_mean": statistics.fmean(fam.values()), "row_weighted": CL.mae([r[key] for r in rows]),
                     "by_family": fam}
    return out


def score(reg):
    rows = reg["table"]["rows"]
    if CL.table_sha256(rows) != reg["table"]["sha256"]:
        raise SystemExit("refused: the registration's embedded rows do not hash to its recorded sha256")
    lofo, fits = CL.lofo(rows)
    st = stats(lofo)
    u, c = st["uncorrected"]["family_mean"], st["corrected"]["family_mean"]
    docs = [r for r in lofo if r["family"] != EXCLUDE_FOR_DOCUMENTS]
    st_docs = stats(docs)
    allfit = CL.fit(rows)
    cv = CL.cv_scores(rows)
    lofo_scores = [abs(r["e_corr"]) for r in lofo]
    return lofo, {
        "hypotheses": {"CORR-LOFO": {
            "verdict": "HOLDS" if c <= CUT * u else "FAILS",
            "uncorrected_family_mean_pct": round(100 * u, 3), "corrected_family_mean_pct": round(100 * c, 3),
            "ratio": round(c / u, 4), "threshold_ratio": CUT,
            "row_weighted_pct": {k: round(100 * st[k]["row_weighted"], 3) for k in st},
            "by_family_pct": {f: {k: round(100 * st[k]["by_family"][f], 3) for k in st}
                              for f in st["uncorrected"]["by_family"]},
            "rows_by_family": {f: sum(1 for r in rows if r["family"] == f) for f in st["uncorrected"]["by_family"]}},
            "CORR-COVER": {"verdict": "PENDING", "scored": "prospectively, on the next 200 served requests after adoption"}},
        "without_%s" % EXCLUDE_FOR_DOCUMENTS: {
            "family_mean_pct": {k: round(100 * st_docs[k]["family_mean"], 3) for k in st_docs},
            "row_weighted_pct": {k: round(100 * st_docs[k]["row_weighted"], 3) for k in st_docs},
            "ratio": round(st_docs["corrected"]["family_mean"] / st_docs["uncorrected"]["family_mean"], 4)},
        "adoption": "ADOPT (default-off option)" if c <= CUT * u else "NOT ADOPTED",
        "fits": {"all_rows": allfit, "leave_one_family_out": fits},
        "intervals": {"cv_plus_q90": CL.conformal_q(cv, 0.10), "cv_plus_scores_n": len(cv),
                      "lofo_q90": CL.conformal_q(lofo_scores, 0.10),
                      "note": "CORR-COVER uses cv_plus_q90 with the all-rows fit; lofo_q90 is the width for a family the table "
                              "has never seen"},
    }


def main(argv):
    reg_path = argv[argv.index("--reg") + 1] if "--reg" in argv else REG
    out = argv[argv.index("--out") + 1] if "--out" in argv else OUT
    reg = json.load(open(reg_path))
    lofo, s = score(reg)
    rebuilt, _ = CL.table()
    code = {f: h for f, h in reg["inputs_sha256"].items() if f.startswith("testbed/scripts/")
            and os.path.exists(os.path.join(ROOT, f)) and sha(os.path.join(ROOT, f)) != h}
    s = {"registration": os.path.relpath(reg_path, ROOT), "registration_sha256": sha(reg_path),
         "table_sha256_registered": reg["table"]["sha256"], "table_rebuilt_matches": CL.table_sha256(rebuilt) == reg["table"]["sha256"],
         "code_changed_since_registration": sorted(code), **s}
    os.makedirs(out, exist_ok=True)
    json.dump(s, open(os.path.join(out, "score.json"), "w"), indent=1)
    cols = ["source", "id", "family", "measured_ms", "predicted_ms", "y", "y_hat", "e_corr"] + list(CL.FEATURES)
    with open(os.path.join(out, "table_lofo.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(lofo)
    h = s["hypotheses"]["CORR-LOFO"]
    print("CORR-LOFO %s: family-mean |error| %.3f %% -> %.3f %% (ratio %.4f, threshold %.2f)" % (
        h["verdict"], h["uncorrected_family_mean_pct"], h["corrected_family_mean_pct"], h["ratio"], CUT))
    print("row-weighted: %s" % h["row_weighted_pct"])
    for f, v in h["by_family_pct"].items():
        print("  %-18s %3d rows  %7.3f %% -> %7.3f %%" % (f, h["rows_by_family"][f], v["uncorrected"], v["corrected"]))
    print("without %s: %s" % (EXCLUDE_FOR_DOCUMENTS, s["without_%s" % EXCLUDE_FOR_DOCUMENTS]))
    print("all-rows fit: %s" % s["fits"]["all_rows"])
    print("intervals: %s" % s["intervals"])
    print("adoption: %s; table rebuilt matches: %s; code changed since registration: %s" % (
        s["adoption"], s["table_rebuilt_matches"], s["code_changed_since_registration"]))
    print("wrote %s" % os.path.relpath(os.path.join(out, "score.json"), ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
