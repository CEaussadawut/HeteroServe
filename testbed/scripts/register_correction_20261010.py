#!/usr/bin/env python3
"""Register the correction layer (upgrade 2 of the scheduler survey, 2026-10-10) BEFORE anything is fitted. Writes, read-only,
testbed/dataset/rule_correction_20261010.json; refuses to overwrite it.

    python3 testbed/scripts/register_correction_20261010.py [--dry-run]

The survey (Documents/HeteroServe_Scheduler_Research.html, upgrade 2; the user: "let's upgrade"): "Keep the analytic model
and add a small linear correction on five features named in advance: hybrid, MoE, KV bytes at the context depth, board clock
or temperature, and worker visits. Wrap it in split-conformal intervals (adaptive conformal inference under drift), and let
the planner choose by the 90th percentile of predicted time, not the mean." Its test: "Leave-one-model-family-out error cut by
at least 30 %, and 90 % intervals covering 85-95 % of the next 200 served requests."

What this file fixes, before any fit:
- THE TABLE: correction_layer.table() as of now, embedded row by row with its sha256, and every input file's sha256. The
  scorer fits these rows; it also rebuilds the table and reports any difference.
- THE MODEL: ordinary least squares of y = measured / predicted - 1 on an intercept and the five features
  (correction_layer.FEATURES, defined in its docstring). A feature constant over a fit's rows is left out of that fit.
  The corrected prediction is predicted x (1 + the fitted y).
- TEST 1 (scored now, by score_correction_20261010.py): leave one model family out. Each family's rows are corrected by the
  fit on all other families. The statistic is the mean over families of each family's mean |error|. HOLDS when the
  corrected statistic is at most 0.70 x the uncorrected one.
- TEST 2 (prospective, scored when its data exist): the 90 % intervals must cover 85-95 % of the next 200 served requests.
  - The intervals are split-conformal in their cross-validated form (CV+, 10 folds by sha256 of the row id): 118 rows are
    too few to give up half of them to a plain split.
  - Calibration scores: each row's |measured / corrected - 1| out of fold. q = the ceil((n + 1) x 0.9)-th smallest score.
  - A new prediction p gets [p_c (1 - q), p_c (1 + q)], with p_c = p x (1 + the correction fitted on all rows).
  - Every full evaluation registered with the correction records p_c and the interval per placement. A request is covered
    when its mean server time per output token lies inside its placement's interval.
  - Adaptive conformal inference (Gibbs and Candes, 2021) is reported beside it, not judged: alpha_1 = 0.10,
    alpha_t+1 = alpha_t + 0.005 x (0.10 - miss_t), q_t the matching quantile of the calibration scores.
- ADOPTION: only when Test 1 holds. Then the all-rows correction becomes a DEFAULT-OFF option: new registrations record the
  corrected prediction and its interval beside the analytic one. Choosing plans by the 90th percentile is a later change,
  registered on its own. Until then, nothing in the planner or the predictor changes.

Known limits, stated before the result:
- Each row's prediction is the one registered for its campaign, so predictor versions mix: two trips per worker before
  2026-10-09, RTT1 after. The visits feature absorbs part of that difference.
- TinyLlama is 67 of the 118 rows. Test 1 therefore weights families equally, and the row-weighted figure is reported beside it.
- Qwen2.5-14B rows (10) feed this testbed analysis. Any figure taken to Documents/ uses the statistics without them, which
  are pre-registered below and reported by the scorer (memory documents-no-qwen25).
- Decode only. Prefill errors have their own structure and are left to a later registration.
"""
import datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import correction_layer as CL                               # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "rule_correction_20261010.json")
CODE = ("correction_layer.py", "register_correction_20261010.py", "score_correction_20261010.py", "reproduce.py",
        "score_ring_laps_20261009.py",
        "throttle_summary.py", "predict_pool.py")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def build():
    import collections
    rows, skipped = CL.table()
    fams = collections.Counter(r["family"] for r in rows)
    return {
        "registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "rule": "CORR",
        "name": "correction_20261010",
        "source": "Documents/HeteroServe_Scheduler_Research.html, upgrade 2 (published 2026-10-10); the user: \"let's upgrade\"",
        "target": "y = measured / predicted - 1, decode ms per token; predicted as registered for the row's campaign",
        "features": {"order": list(CL.FEATURES),
                     "definitions": {
                         "hybrid": "1 for a qwen35-architecture hybrid, else 0",
                         "moe": "1 for a mixture of experts (a shape with active bytes per layer), else 0",
                         "kv_frac": "KV bytes read per decoded token at the mean context depth / weight bytes read per decoded "
                                    "token; depth (N - 1) / 2 for llama-bench tg-N, P + (G - 1) / 2 for a served request; KV "
                                    "layers every block, or a qwen35 hybrid's full-attention blocks (one in %d)" % CL.FAI["qwen35"],
                         "pi_temp": "max(0, T - %g) deg C, T the hottest Pi's peak in the row (1 Hz poll), else its end-of-cell "
                                    "STATE temperature; 0 with no Pi" % CL.TEMP_KNEE_C,
                         "visits": "RPC worker stages per token (stages that are not the host's own process)"}},
        "model": "ordinary least squares, intercept + the five features; a feature constant over a fit's rows is left out of "
                 "that fit; corrected = predicted x (1 + fitted y)",
        "table": {"rows": rows, "n": len(rows), "sha256": CL.table_sha256(rows), "skipped": skipped,
                  "by_family": dict(sorted(fams.items())),
                  "by_source": dict(collections.Counter(r["source"] for r in rows))},
        "inputs_sha256": {**{os.path.relpath(f, ROOT): sha(f) for f in CL.input_files()},
                          **{"testbed/scripts/" + f: sha(os.path.join(HERE, f)) for f in CODE}},
        "tests": [
            {"id": "CORR-LOFO", "scored": "now",
             "statistic": "mean over model families of the family's mean |measured / prediction - 1|, each family's rows "
                          "corrected by the fit on every other family (leave one model family out)",
             "holds_when": "corrected <= 0.70 x uncorrected",
             "reported_beside": ["the row-weighted mean |error|, corrected and uncorrected",
                                 "both statistics without the qwen2.5-14b family (the figures for Documents/)",
                                 "each family's mean |error| and the coefficients of every fit"]},
            {"id": "CORR-COVER", "scored": "prospectively, on the next 200 served requests after adoption",
             "interval": "CV+ (10 folds by sha256(row id) mod 10): q = the ceil((n + 1) x 0.9)-th smallest out-of-fold "
                         "|measured / corrected - 1|; interval [p_c (1 - q), p_c (1 + q)], p_c from the all-rows fit",
             "holds_when": "85 % <= the share of those requests whose mean server TPOT lies in its placement's interval <= 95 %",
             "reported_beside": "adaptive conformal inference: alpha_1 = 0.10, alpha_t+1 = alpha_t + 0.005 (0.10 - miss_t)"}],
        "adoption": "only if CORR-LOFO holds: the all-rows correction as a default-off option; new registrations record the "
                    "corrected prediction and its interval beside the analytic one. Plan choice by the 90th percentile is "
                    "registered separately.",
        "limits": ["predictor versions mix: two trips per worker before 2026-10-09, RTT1 after",
                   "tinyllama-1.1b is %d of %d rows, so the test weights families equally" % (fams["tinyllama-1.1b"], len(rows)),
                   "qwen2.5-14b rows feed the analysis; Documents/ use the statistics without them",
                   "decode only"],
        "status": "registered, not fitted",
    }


def main(argv):
    if os.path.exists(OUT) and "--dry-run" not in argv:
        sys.exit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    reg = build()
    t = reg["table"]
    print("table: %d rows (%s), sha256 %s; skipped %d" % (t["n"], t["by_source"], t["sha256"][:16], len(t["skipped"])))
    print("families: %s" % t["by_family"])
    print("inputs: %d files" % len(reg["inputs_sha256"]))
    if "--dry-run" in argv:
        print("dry run: nothing written")
        return reg
    json.dump(reg, open(OUT, "w"), indent=1, ensure_ascii=False)
    os.chmod(OUT, 0o444)
    print("registered %s at %s, sha256 %s (read-only)" % (os.path.relpath(OUT, ROOT), reg["registered"], sha(OUT)[:16]))
    return reg


if __name__ == "__main__":
    main(sys.argv[1:])
