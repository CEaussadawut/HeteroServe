#!/usr/bin/env python3
"""Upgrade 2 (2026-10-10): a linear correction of the analytic predictor's decode error, with calibrated intervals.

    python3 testbed/scripts/correction_layer.py table [--out CSV]     # the error table, from the record only
    python3 testbed/scripts/correction_layer.py counts                # rows per family and source, the features' ranges

The predictor (predict_pool.predict and placement.plan_optimal) stays the backbone. This module learns nothing until
register_correction_20261010.py has fixed the table, the features and the tests, and score_correction_20261010.py fits.

THE TABLE. One row per decode measurement scored against a prediction AS REGISTERED for its campaign, never re-priced:
  pool    every cell of testbed/dataset/pool_results*.json with exit 0 and both a measured and a predicted decode ms/token
          in its results file (cell.decode; the 2026-09-30 campaign keeps them in cell.bench.decode), its model from its
          campaign's registration or an addendum's (pool_predictions*addendum*.json), its tg length from llama-bench's
          own JSONL in its log. A cell without a
          measured decode there is left out: the two forced-thrash cells, the crashed MoE cell, and the TCP diagnostic's
          four, which its own scorer timed differently.
  laps    the ring-lap cells (testbed/results/ring_laps_20261009_201631/score.json) against their RTT1 prediction at their
          setting, the prediction RL-ACC judged.
  visit   the VISIT cells (testbed/results/ring_laps_20261009_235345/score.json) against "rtt1", the predictor without
          the rule under test.
  served  each placement of each SCORED full evaluation (a run dir with summary.json and score.json): the mean server
          time per output token against the registered decode ms.
Target: y = measured / predicted - 1, the relative error the correction learns.

THE FIVE FEATURES (named in the survey before any fit: hybrid, MoE, KV bytes at the context depth, board clock or
temperature, worker visits):
  hybrid   1 for a qwen35-architecture hybrid (Qwen3.5-9B, Qwen3.8-27B), else 0
  moe      1 for a mixture of experts (a shape with active bytes per layer: Qwen3-30B-A3B), else 0
  kv_frac  KV bytes read per decoded token at the run's mean context depth, over the weight bytes read per decoded token
           (each computed block's active bytes, plus the output head). The depth: llama-bench's tg-N starts from an empty
           cache, (N - 1) / 2; a served request of P prompt tokens and G generated, P + (G - 1) / 2. The KV layers: every
           block, or a qwen35 hybrid's full-attention blocks (one in FAI = 4: the RECUR registration's
           full_attention_interval for Qwen3.5-9B, and the shapes' own notes, 8 of 32 and 16 of 64 blocks).
  pi_temp  max(0, T - 60) in deg C, T the hottest Raspberry Pi's temperature during the row: the 1 Hz poll's peak over the
           cell (a served placement: over its requests' windows) where a poll exists, else the later of the cell's STATE
           lines (its end). 0 for a row with no Pi in it. TEMP_SOURCE says which.
  visits   the RPC worker stages one token passes through: every stage that is not the host's own process.
"""
import csv, glob, hashlib, json, math, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(ROOT, "testbed", "dataset")
RESD = os.path.join(ROOT, "testbed", "results")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import reproduce as RP                                      # noqa: E402  registrations(), state_temps()
import score_ring_laps_20261009 as SR                       # noqa: E402  the run.log parser
import throttle_summary as TS                               # noqa: E402  the 1 Hz poll's line parser

FEATURES = ("hybrid", "moe", "kv_frac", "pi_temp", "visits")
TEMP_KNEE_C = 60.0
FAI = {"qwen35": 4}
LAPS_RUN = os.path.join(RESD, "ring_laps_20261009_201631")
VISIT_RUN = os.path.join(RESD, "ring_laps_20261009_235345")
FAMILY = {   # shape -> model family: the unit left out in the leave-one-family-out test
    "tinyllama-1.1b-q4km": "tinyllama-1.1b", "tinyllama-1.1b-q8_0": "tinyllama-1.1b", "tinyllama-1.1b-q4_0": "tinyllama-1.1b",
    "tinyllama-1.1b-q3km": "tinyllama-1.1b", "qwen2.5-14b-instruct-q4_k_m": "qwen2.5-14b", "mistral-7b-q8_0": "mistral-7b",
    "mistral-nemo-12b-q4_k_m": "mistral-nemo-12b", "qwen3-14b-q4_k_m": "qwen3-14b", "qwen3-30b-a3b-q3_k_m": "qwen3-30b-a3b",
    "qwen3.8-27b-q3_k_m": "qwen3.8-27b", "qwen3.5-9b-q4_k_m": "qwen3.5-9b",
    "qwen3-30b-a3b-q6_k": "qwen3-30b-a3b", "tinyllama-1.1b-q6_k": "tinyllama-1.1b"}   # Q6_K: 2026-10-11/21


def shapes():
    return json.load(open(os.path.join(DS, "model_shapes.json")))


def weights_read(q):
    """Bytes a decoded token reads: each computed block's active bytes (a multi-token-prediction block is not computed),
    plus the output head."""
    lb = q.get("layer_active_bytes") or q["layer_bytes"]
    main = len(q["layer_bytes"]) - int(q.get("nextn_layers") or 0)
    return float(sum(lb[:main])) + float(q["head_bytes"])


def kv_layers(q):
    main = len(q["layer_bytes"]) - int(q.get("nextn_layers") or 0)
    fai = FAI.get(q.get("architecture"))
    return sum(1 for i in range(main) if (i + 1) % fai == 0) if fai else main


def kv_frac(q, depth):
    return q["kv_bytes_per_layer_per_token"] * depth * kv_layers(q) / weights_read(q)


def n_gen(protocol):
    m = re.search(r"(?:^|\s)-n (\d+)\b", protocol or "")
    if not m:
        raise ValueError("no -n in the protocol %r" % protocol)
    return int(m.group(1))


def log_n_gen(log):
    """The tg test's n_gen, from llama-bench's own JSONL lines in a cell's log: the record of what ran."""
    gens = {int(g) for g in re.findall(r'"n_gen": (\d+)', open(log, encoding="utf-8", errors="replace").read())} - {0}
    if len(gens) != 1:
        raise ValueError("%s: tg n_gen %s, not exactly one" % (log, sorted(gens)))
    return gens.pop()


def temp_feature(t):
    return max(0.0, t - TEMP_KNEE_C) if t is not None else 0.0


def poll_peak(log, windows):
    """The poll's peak temperature over [t0, t1] windows; None when no sample falls in them."""
    if not os.path.exists(log):
        return None
    best = None
    for ln in open(log, encoding="utf-8", errors="replace"):
        p = TS.parse(ln)
        if not p:
            continue
        t, kv = p
        if any(a <= t <= b for a, b in windows):
            m = re.match(r"([\d.]+)", kv.get("temp", ""))
            if m:
                best = max(best or -1e9, float(m.group(1)))
    return best


def row(source, ident, shape_name, measured, predicted, q, depth, pis, temp, temp_source, visits, extra=None):
    r = {"source": source, "id": ident, "family": FAMILY[shape_name], "shape": shape_name,
         "measured_ms": float(measured), "predicted_ms": float(predicted),
         "y": float(measured) / float(predicted) - 1.0,
         "hybrid": 1.0 if q.get("architecture") in FAI else 0.0,
         "moe": 1.0 if (q.get("layer_active_bytes") or q.get("active_bytes_per_layer")) else 0.0,
         "kv_frac": kv_frac(q, depth), "pi_temp": temp_feature(temp) if pis else 0.0, "visits": float(visits),
         "depth": depth, "pis": ",".join(pis), "pi_temp_c": temp, "temp_source": temp_source if pis else "no pi"}
    r.update(extra or {})
    return r


def pool_rows(sh, skipped):
    regs, out = RP.registrations(), []
    addenda = [json.load(open(a)) for a in sorted(glob.glob(os.path.join(DS, "pool_predictions*addendum*.json")))]
    for p in sorted(glob.glob(os.path.join(DS, "pool_results*.json"))):
        pub = json.load(open(p))
        reg = regs.get(pub.get("predictions_sha256")) or (os.path.join(ROOT, pub["registration"]) if "registration" in pub else None)
        R = json.load(open(reg)) if reg and os.path.exists(reg) else {}
        for cid, c in pub.get("cells", {}).items():
            if not isinstance(c, dict):
                continue
            ident = "%s:%s" % (os.path.basename(p), cid)
            rc = R.get("configs", {}).get(cid) or next((A["configs"][cid] for A in addenda if cid in A.get("configs", {})), {})
            dec = c.get("decode") or (c.get("bench") or {}).get("decode") or {}
            m = (dec.get("measured") or {}).get("ms_per_token") if "measured" in dec else dec.get("ms_per_token")
            pr = dec.get("predicted_ms")
            ok_exit = c.get("exit", 0) == 0 and c.get("exit_ok", True)
            if not (m and pr and rc.get("model") and ok_exit):
                skipped.append({"id": ident, "why": "exit %s" % c.get("exit") if not ok_exit else
                                "no measured and predicted decode in the results file"})
                continue
            q = sh[rc["model"]]
            gen = log_n_gen(os.path.join(ROOT, c["log"]))
            if "pipeline" in c:
                nodes = [(s["node"], s["blocks"]) for s in c["pipeline"] if s["blocks"] > 0]
                visits, names = len(nodes) - 1, [n for n, _ in nodes]       # the host is the first stage, in process
            else:                                                            # the 2026-09-30 format: [[node, blocks], ...]
                nodes = [(n, k) for n, k in c["split"] if k > 0]
                visits, names = sum(1 for n, _ in nodes if n != "host-cpu"), [n for n, _ in nodes if n != "host-cpu"]
            host = (rc.get("bench") or {}).get("host") or (names[0] if names else "")
            pis = sorted({n for n in names + [host] if n.startswith("pi")})
            th = c.get("thermal") if isinstance(c.get("thermal"), dict) else {}
            polled = [float(v["max_temp_c"]) for n, v in (th.get("per_node") or {}).items()
                      if n in pis and re.fullmatch(r"[\d.]+", str(v.get("max_temp_c", "")))]
            if polled:
                temp, src = max(polled), "poll"
            else:
                snap, _ = RP.state_temps(os.path.join(ROOT, c["log"])) if c.get("log") else ("", "")
                after = [float(v.split(">")[1]) for v in (x.split(":", 1)[1] for x in snap.split(";") if x.split(":")[0] in pis)
                         if re.fullmatch(r"[\d.]+", v.split(">")[-1])]
                temp, src = (max(after), "state") if after else (None, "none")
            if pis and temp is None:
                skipped.append({"id": ident, "why": "a Pi in the cell, and neither a poll nor a STATE temperature"})
                continue
            out.append(row("pool", ident, rc["model"], m, pr, q, (gen - 1) / 2.0, pis, temp, src, visits,
                           {"file": RP.rel(p)}))
    return out


def lap_rows(sh, run, source, pred_key, skipped):
    score = json.load(open(os.path.join(run, "score.json")))
    reg = json.load(open(os.path.join(ROOT, score["registration"]))) if "registration" in score else \
        json.load(open(os.path.join(DS, "ring_laps_predictions_20261009.json")))
    import predict_pool as P
    shape = P.MODELS[reg["model"]][0]
    q = sh[shape]
    cells, _ = SR.parse(open(os.path.join(run, "run.log"), encoding="utf-8", errors="replace").read().splitlines())
    poll = os.path.join(run, "throttle_pi4.log")
    out = []
    for key, v in sorted(score["cells"].items()):
        lay = reg["layouts"][v["layout"]]
        c = cells[(v["round"], v["setting"], v["layout"])]
        pr = lay["predicted"][pred_key(v)]["decode_ms"]
        visits = len([s for s in lay["stages"] if s["board"] != "pi4 (host)"])
        temp = poll_peak(poll, [(c["t0"], c["t1"])])
        if temp is None:
            skipped.append({"id": "%s:%s" % (source, key), "why": "no poll sample in the cell's window"})
            continue
        out.append(row(source, "%s:%s" % (os.path.basename(run), key), shape, v["decode_ms"], pr, q,
                       (n_gen(reg["protocol"]) - 1) / 2.0, ["pi4"], temp, "poll", visits, {"file": RP.rel(run)}))
    return out


# The served rows are the full-evaluation runs started before the rule was registered (2026-10-10 02:01:36): the table is the
# registered one, so a later run must not join it (2026-10-21: the four-node runs added 53 rows and broke its sha256).
REGISTERED_STAMP = "20261010_020136"


def full_eval_runs():
    return [rd for rd in sorted(glob.glob(os.path.join(RESD, "full_eval_*")))
            if os.path.isdir(rd) and "_".join(os.path.basename(rd).split("_")[-2:]) < REGISTERED_STAMP
            and os.path.exists(os.path.join(rd, "summary.json")) and os.path.exists(os.path.join(rd, "score.json"))]


def served_rows(sh, skipped):
    import predict_pool as P
    out = []
    for rd in full_eval_runs():
        s = json.load(open(os.path.join(rd, "summary.json")))
        recs = [json.loads(l) for l in open(os.path.join(rd, "run.jsonl"))]
        reg = json.load(open(os.path.join(ROOT, next(r for r in recs if r["kind"] == "start")["data"]["registration"])))
        shape = P.MODELS[reg["model"]][0]
        q = sh[shape]
        for lab, v in s.items():
            pis = sorted({n for n, _ in v["assign"] if n.startswith("pi")})
            wins = [(r["data"]["sent_wall"], r["data"]["sent_wall"] + r["data"]["done_s"]) for r in recs
                    if r["kind"] == "req" and r["data"].get("placement") == lab]
            temp = max([t for t in (poll_peak(os.path.join(rd, "throttle_%s.log" % n), wins) for n in pis) if t is not None],
                       default=None)
            if pis and temp is None:
                skipped.append({"id": "%s:%s" % (os.path.basename(rd), lab), "why": "no poll sample in its request windows"})
                continue
            out.append(row("served", "%s:%s" % (os.path.basename(rd), lab), shape, v["server_tpot"]["mean"],
                           v["predicted"]["decode_ms"], q, reg["n"] + (reg["gen"] - 1) / 2.0, pis, temp, "poll",
                           len(v["assign"]) - 1, {"file": RP.rel(rd)}))
    return out


def table():
    """(rows, skipped): the error table, in a fixed order (pool, laps, visit, served)."""
    sh, skipped = shapes(), []
    rows = pool_rows(sh, skipped)
    rows += lap_rows(sh, LAPS_RUN, "laps", lambda v: "%s_rtt1" % v["setting"], skipped)
    rows += lap_rows(sh, VISIT_RUN, "visit", lambda v: "rtt1", skipped)
    rows += served_rows(sh, skipped)
    return rows, skipped


def table_sha256(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def input_files():
    """Every file the table reads, for the registration's sha256 list."""
    fs = sorted(glob.glob(os.path.join(DS, "pool_results*.json")))
    fs += [os.path.join(DS, f) for f in ("model_shapes.json", "ring_laps_predictions_20261009.json", "visit_predictions_20261009.json")]
    for run in (LAPS_RUN, VISIT_RUN):
        fs += [os.path.join(run, f) for f in ("score.json", "run.log", "throttle_pi4.log")]
    for rd in full_eval_runs():
        fs += sorted(glob.glob(os.path.join(rd, "*.json*"))) + sorted(glob.glob(os.path.join(rd, "throttle_*.log")))
    return [f for f in fs if os.path.exists(f)]


# ------------------------------------------------------------------ the correction
def fit(rows, feats=FEATURES):
    """Ordinary least squares of y on an intercept and the features. A feature constant over these rows is left out of the
    fit (coefficient 0): with one family left out, its own indicator may never be 1. Returns {"intercept", "coef", "dropped"}."""
    import numpy as np
    use = [f for f in feats if len({r[f] for r in rows}) > 1]
    X = np.array([[1.0] + [r[f] for f in use] for r in rows])
    y = np.array([r["y"] for r in rows])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"intercept": float(b[0]), "coef": {f: (float(b[1 + use.index(f)]) if f in use else 0.0) for f in feats},
            "dropped": [f for f in feats if f not in use]}


def correction(model, r):
    """The predicted relative error of a row's prediction; the corrected prediction is predicted x (1 + this)."""
    return model["intercept"] + sum(model["coef"][f] * r[f] for f in model["coef"])


def lofo(rows):
    """Leave one model family out: each family's rows corrected by the fit on every other family. Returns the rows, each with
    "y_hat" (the out-of-family correction) and "e_corr" (measured / corrected - 1), and the per-family fits."""
    fams = sorted({r["family"] for r in rows})
    out, fits = [], {}
    for f in fams:
        m = fit([r for r in rows if r["family"] != f])
        fits[f] = m
        for r in rows:
            if r["family"] == f:
                yh = correction(m, r)
                out.append(dict(r, y_hat=yh, e_corr=r["measured_ms"] / (r["predicted_ms"] * (1.0 + yh)) - 1.0))
    return out, fits


def mae(xs):
    return statistics.fmean(abs(x) for x in xs) if xs else None


def family_mae(rows, key):
    fams = sorted({r["family"] for r in rows})
    return {f: mae([r[key] for r in rows if r["family"] == f]) for f in fams}


def fold_of(r, k=10):
    """A row's cross-validation fold: sha256 of its id, mod k. Deterministic, and blind to every value in the row."""
    return int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % k


def cv_scores(rows, k=10):
    """CV+ calibration scores (the cross-validated form of split-conformal): each row's |measured / corrected - 1|, the
    correction fitted on the other k - 1 folds."""
    out = []
    for f in range(k):
        test = [r for r in rows if fold_of(r, k) == f]
        if not test:
            continue
        m = fit([r for r in rows if fold_of(r, k) != f])
        out += [abs(r["measured_ms"] / (r["predicted_ms"] * (1.0 + correction(m, r))) - 1.0) for r in test]
    return out


def conformal_q(scores, alpha=0.10):
    """Split-conformal quantile: the ceil((n + 1)(1 - alpha))-th smallest score, or inf when n is too small."""
    s = sorted(scores)
    k = math.ceil((len(s) + 1) * (1.0 - alpha))
    return s[k - 1] if k <= len(s) else math.inf


def main(argv):
    if not argv or argv[0] not in ("table", "counts"):
        sys.exit(__doc__)
    rows, skipped = table()
    if argv[0] == "table":
        out = argv[argv.index("--out") + 1] if "--out" in argv else None
        cols = ["source", "id", "family", "shape", "measured_ms", "predicted_ms", "y"] + list(FEATURES) + \
               ["depth", "pis", "pi_temp_c", "temp_source", "file"]
        f = open(out, "w", newline="") if out else sys.stdout
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        print("%d rows, %d skipped, sha256 %s" % (len(rows), len(skipped), table_sha256(rows)[:16]), file=sys.stderr)
        return
    import collections
    print("rows %d (skipped %d)" % (len(rows), len(skipped)))
    print("by source: %s" % dict(collections.Counter(r["source"] for r in rows)))
    print("by family: %s" % dict(sorted(collections.Counter(r["family"] for r in rows).items())))
    print("temperature source: %s" % dict(collections.Counter(r["temp_source"] for r in rows)))
    for f in FEATURES:
        v = [r[f] for r in rows]
        print("%-8s min %.5g  max %.5g  distinct %d" % (f, min(v), max(v), len(set(v))))
    for s in skipped:
        print("skipped %s: %s" % (s["id"], s["why"]))
    print("table sha256 %s" % table_sha256(rows))


if __name__ == "__main__":
    main(sys.argv[1:])
