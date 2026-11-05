#!/usr/bin/env python3
"""The fit and the fit-based hypotheses of Qwen3.6-35B-A3B's characterisation (register_c35_20261024.py), called by
ab_score_20261024.py as its scorer_extra: extra(session, reg, score) -> dict.

T_meas(s) - xfer(s) = sum over boards d of k_d busy_d(s), solved exactly on the registered fit splits (four equations,
four unknowns, by Gaussian elimination with partial pivoting); then
  C35-HOLDOUT  the held-out split's predicted server mean within tol of its measured one;
  C35-PAST     order35b's P and F (23 Oct) predicted within tol of their measured server means;
  C35-PLAUSIBLE every k_d within [lo, hi].
INCOMPLETE when a split the fit or a test needs is incomplete or has no measurement."""


def solve(A, b):
    n = len(A)
    M = [list(map(float, row)) + [float(v)] for row, v in zip(A, b)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            raise ValueError("singular system")
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def extra(session, reg, score):
    fit, arms, hx = reg["fit"], score["arms"], reg["hypotheses_extra"]
    metric, boards = fit["metric"], reg["nodes"]
    need = fit["on"] + [fit["held_out"]]
    ok = all(arms.get(k, {}).get("complete") and arms.get(k, {}).get(metric) for k in need)
    out = {"metric": metric, "boards": boards}
    if not ok:
        out["hypotheses"] = {k: {"verdict": "INCOMPLETE"} for k in hx}
        return out
    A = [[reg["arms"][k]["terms"]["busy_ms"][d] for d in boards] for k in fit["on"]]
    y = [arms[k][metric] - reg["arms"][k]["terms"]["xfer_ms"] for k in fit["on"]]
    kk = solve(A, y)
    k = dict(zip(boards, kk))
    pred = lambda busy, xfer: sum(k[d] * busy[d] for d in busy) + xfer
    e = reg["arms"][fit["held_out"]]["terms"]
    e_pred, e_meas = pred(e["busy_ms"], e["xfer_ms"]), arms[fit["held_out"]][metric]
    past = {}
    for name, p in fit["past"].items():
        pp = pred(p["busy_ms"], p["xfer_ms"])
        past[name] = {"predicted": round(pp, 2), "measured": p["measured_server_mean_ms"],
                      "err_pct": round(100 * (pp / p["measured_server_mean_ms"] - 1), 2)}
    per_block = {}
    for d in boards:
        nA = {x[0]: x[1] for x in reg["arms"]["A"]["assign"]}[d]
        per_block[d] = {"predicted_ms": round(reg["arms"]["A"]["terms"]["busy_ms"][d] / nA, 3),
                        "fitted_ms": round(k[d] * reg["arms"]["A"]["terms"]["busy_ms"][d] / nA, 3)}
    err_e = 100 * (e_pred / e_meas - 1)
    hyp = {
        "C35-HOLDOUT": {"verdict": "HOLDS" if abs(err_e) <= 100 * hx["C35-HOLDOUT"]["tol"] else "FAILS",
                        "predicted": round(e_pred, 2), "measured": e_meas, "err_pct": round(err_e, 2)},
        "C35-PAST": {"verdict": "HOLDS" if all(abs(v["err_pct"]) <= 100 * hx["C35-PAST"]["tol"] for v in past.values()) else "FAILS",
                     **past},
        "C35-PLAUSIBLE": {"verdict": "HOLDS" if all(hx["C35-PLAUSIBLE"]["lo"] <= v <= hx["C35-PLAUSIBLE"]["hi"] for v in kk)
                          else "FAILS", "k": {d: round(v, 4) for d, v in k.items()}}}
    out.update({"k": {d: round(v, 4) for d, v in k.items()}, "per_block_ms_at_A": per_block,
                "fit_residuals_ms": {kk_: round(pred(reg["arms"][kk_]["terms"]["busy_ms"], reg["arms"][kk_]["terms"]["xfer_ms"])
                                                - arms[kk_][metric], 3) for kk_ in fit["on"]},
                "hypotheses": hyp})
    return out
