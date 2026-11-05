#!/usr/bin/env python3
"""The study-specific results of c30 (register_c30_20261024.py: do the 35B's per-board multipliers price Qwen3-30B-A3B Q6_K?),
called by ab_score_20261024.py as its scorer_extra: extra(session, reg, score) -> dict.

  C30-RAW        every fit split's measured server mean over its prediction as it is (terms.decode_ms), minus 1, in [lo, hi];
  the 30B's own multipliers: T_meas(s) - xfer(s) = sum over boards d of k_d busy_d(s), solved exactly on the fit splits
                 (c35_fit_20261024.py's solve, imported), beside the 35B's; with, per split, how far +1 % on its measured
                 value moves each k (the exact fit's sensitivity: what a board's k can be trusted to);
  C30-PLAUSIBLE  every fitted k within [lo, hi].
INCOMPLETE when a fit split is incomplete or has no measurement."""
import importlib.util, os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("c35_fit", os.path.join(HERE, "c35_fit_20261024.py"))
_c35 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_c35)
solve = _c35.solve


def extra(session, reg, score):
    fit, arms, hx = reg["fit"], score["arms"], reg["hypotheses_extra"]
    metric, boards = fit["metric"], reg["nodes"]
    out = {"metric": metric, "boards": boards, "k35": fit["k35"]}
    have = {k: bool(arms.get(k, {}).get("complete") and arms.get(k, {}).get(metric)) for k in fit["on"]}
    raw = {k: round(arms[k][metric] / reg["arms"][k]["terms"]["decode_ms"] - 1, 4) if have[k] else None for k in fit["on"]}
    lo, hi = hx["C30-RAW"]["lo"], hx["C30-RAW"]["hi"]
    hyp = {"C30-RAW": {"verdict": "INCOMPLETE" if not all(have.values()) else
                       ("HOLDS" if all(lo <= v <= hi for v in raw.values()) else "FAILS"), "bias": raw, "lo": lo, "hi": hi}}
    if not all(have.values()):
        hyp["C30-PLAUSIBLE"] = {"verdict": "INCOMPLETE"}
        out["hypotheses"] = hyp
        return out
    A = [[reg["arms"][k]["terms"]["busy_ms"][d] for d in boards] for k in fit["on"]]
    y = [arms[k][metric] - reg["arms"][k]["terms"]["xfer_ms"] for k in fit["on"]]
    k = dict(zip(boards, solve(A, y)))
    sens = {}
    for i, s in enumerate(fit["on"]):
        dy = [0.01 * arms[s][metric] if j == i else 0.0 for j in range(len(fit["on"]))]
        sens[s] = {d: round(v, 4) for d, v in zip(boards, solve(A, dy))}
    nA = {x[0]: x[1] for x in reg["arms"]["A"]["assign"]}
    per_block = {d: {"predicted_ms": round(reg["arms"]["A"]["terms"]["busy_ms"][d] / nA[d], 3),
                     "with_35b_k_ms": round(fit["k35"][d] * reg["arms"]["A"]["terms"]["busy_ms"][d] / nA[d], 3),
                     "fitted_ms": round(k[d] * reg["arms"]["A"]["terms"]["busy_ms"][d] / nA[d], 3)} for d in boards}
    plo, phi = hx["C30-PLAUSIBLE"]["lo"], hx["C30-PLAUSIBLE"]["hi"]
    hyp["C30-PLAUSIBLE"] = {"verdict": "HOLDS" if all(plo <= v <= phi for v in k.values()) else "FAILS",
                            "k": {d: round(v, 4) for d, v in k.items()}}
    out.update({"k": {d: round(v, 4) for d, v in k.items()},
                "k_minus_k35": {d: round(k[d] - fit["k35"][d], 4) for d in boards},
                "dk_per_1pct_on_split": sens, "per_block_ms_at_A": per_block, "hypotheses": hyp})
    return out
