#!/usr/bin/env python3
"""The planner's checking cost (the scheduler review's item 10, 2026-11-03): what speculation with the model's own
multi-token-prediction head (llama.cpp `--spec-type draft-mtp`) does on a plan, from one exploration of the same model on
another plan (explore_spec_20261102.py, read by spec_cost_20261103.py).

A check pass with n drafted tokens costs, as measured on the exploration plans (explspec A and B, 3 Nov),

    pass(n) = dec x R(n + 1) + n x d + i

  dec    the plan's decode step (the planner's predict(), calibrated by the exploration plan's measured / predicted decode);
  R(m)   one batch of m tokens against one decode step: the exploration plan's scan median T(m) / its decode step. A plan
         of the same model carries it over, its links counted at 1 (the links are 2-7 ms of a 270-320 ms step);
  d      one drafted token: two passes of the output head on the head node (the draft's own, and the check's extra row),
         the MTP block on the node holding the last blocks (where llama.cpp loads it), and, when that node is a worker,
         a visit (predict()'s rtt_trips x rtt) and an RPC cost the planner does not price:
             d = KAPPA x (2 H + b_last) + visit + RPC_DRAFT_MS x [the last node is a worker]
         KAPPA from B (the laptop alone: no visit, no RPC), RPC_DRAFT_MS from A (the MTP block on the Pi 4): two
         parameters from two plans, exactly (calibrate()). A plan's d is then priced, not scaled from its exploration;
  i      the exploration plan's intercept (its pass fit), carried over for the same model;
and a pass yields E tokens: the exploration's tokens per pass for the same model, setting and prompt (greedy decoding, the
same prompts, so the same drafts but for device rounding). Speedup = dec x E / pass(n).
"""
import json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import spec_cost_20261103 as SC                             # noqa: E402

SPECS = {"mtp1": 1, "mtp2": 2, "mtp3": 3}


def plan_terms(key, assign, head, mem, ctx=4096):
    """The planner's quantities for a plan: its predicted decode, links, head cost H (on the head node), the cost of one
    block on the node holding the last blocks (the MTP block's node; the mean of its main blocks), and the visit to that
    node (0 when it is the host)."""
    r = P.plan_fixed(key, assign, head, ctx, mem, load_mode="none", **P.PLANNER_V4)
    busy, xfer = EM.per_node(r["assign"], r["q"], r["C"], r["links"], head_at=r["head_at"], rpc_msg=P.RPC_MSG)
    hn = assign[r["head_at"]][0] if r["head_at"] is not None else assign[-1][0]
    role = "host" if hn == assign[0][0] else "worker"
    H = r["C"][hn].get(role + "_head", r["C"][hn][role])[0] * r["q"]["head_bytes"]
    last, kl = assign[-1]
    nx = int(r["q"].get("nextn_layers") or 0)
    b_last = (busy[last]["tg"] - (H if last == hn else 0.0)) / (kl - nx)
    remote = last != assign[0][0]
    visit = P.RTT_TRIPS_DEFAULT * r["links"][last]["rtt_ms"] if remote else 0.0
    return {"decode_ms": r["decode_ms"], "xfer_ms": xfer["tg"], "head_ms": H, "head_node": hn, "last": last,
            "block_last_ms": b_last, "visit_ms": visit, "remote": remote, "nextn_layers": nx}


def calibrate(fit_local, terms_local, fit_remote, terms_remote):
    """(KAPPA, RPC_DRAFT_MS): KAPPA from a plan whose MTP block sits on the host (no visit, no RPC), RPC_DRAFT_MS from one
    whose MTP block sits on a worker."""
    assert not terms_local["remote"] and terms_remote["remote"]
    kappa = fit_local["draft_ms"] / (2 * terms_local["head_ms"] + terms_local["block_last_ms"])
    rpc = fit_remote["draft_ms"] - kappa * (2 * terms_remote["head_ms"] + terms_remote["block_last_ms"]) - terms_remote["visit_ms"]
    return kappa, rpc


def draft_ms(terms, kappa, rpc):
    """One drafted token on a plan: KAPPA x (2 H + b_last) + the visit + RPC_DRAFT_MS when the MTP block is on a worker."""
    return kappa * (2 * terms["head_ms"] + terms["block_last_ms"]) + terms["visit_ms"] + (rpc if terms["remote"] else 0.0)


def exploration_fit(folder, config):
    """From an explore_spec folder: the measured decode step (none's mean ms per token over both prompts), R(m) at every
    scanned m, each MTP setting's pass time and tokens per pass by prompt, and the pass fit pass(n) - dec x R(n + 1) =
    n x d + i by least squares over the MTP settings (both prompts' pass times averaged per setting)."""
    recs = [r for r in json.load(open(os.path.join(folder, "results.json"))) if r["config"] == config]
    none = next(r for r in recs if r["spec"] == "none")
    sp = {s["spec"] + "/" + s["prompt"]: s for s in SC.specs(folder, recs)}
    dec = statistics.fmean([sp["none/free"]["ms_per_token"], sp["none/copy"]["ms_per_token"]])
    R = {s["m"]: s["prompt_ms"] / dec for s in SC.scan(none)}
    pts = []
    for name, n in SPECS.items():
        p = statistics.fmean([sp[name + "/free"]["ms_per_pass"], sp[name + "/copy"]["ms_per_pass"]])
        pts.append((n, p - dec * R[n + 1]))
    mn = statistics.fmean(x for x, _ in pts); mr = statistics.fmean(y for _, y in pts)
    slope = sum((x - mn) * (y - mr) for x, y in pts) / sum((x - mn) ** 2 for x, _ in pts)
    E = {k: v["tokens_per_pass"] for k, v in sp.items() if k.split("/")[0] in SPECS}
    return {"folder": os.path.relpath(os.path.abspath(folder), ROOT), "config": config, "decode_ms": dec,
            "R": {m: round(v, 4) for m, v in R.items()}, "draft_ms": slope, "intercept_ms": mr - slope * mn,
            "residuals": pts, "E": E, "speedups": {k: v["speedup"] for k, v in sp.items()}}


def predict(fit, terms_exp, terms_new, kappa, rpc):
    """A new plan of the exploration's model: dec (calibrated), R carried over, d priced on the new plan, i carried over,
    E carried over. {"decode_ms", "R": {m}, "draft_ms", "intercept_ms", "arms": {spec/prompt: {"pass_ms", "ms_per_token",
    "speedup", "E"}}}."""
    dec = terms_new["decode_ms"] * fit["decode_ms"] / terms_exp["decode_ms"]
    d = draft_ms(terms_new, kappa, rpc)
    out = {"decode_ms": round(dec, 1), "calibration": round(fit["decode_ms"] / terms_exp["decode_ms"], 4),
           "R": {m: v for m, v in fit["R"].items()}, "draft_ms": round(d, 1),
           "intercept_ms": round(fit["intercept_ms"], 1), "arms": {}}
    for name, n in SPECS.items():
        p = dec * fit["R"][n + 1] + n * d + fit["intercept_ms"]
        for pk in ("free", "copy"):
            E = fit["E"][name + "/" + pk]
            out["arms"][name + "/" + pk] = {"pass_ms": round(p, 1), "E": E, "ms_per_token": round(p / E, 1),
                                            "speedup": round(dec * E / p, 3)}
    return out


# ---- per-node batch curves (2026-11-03 evening; the user: "... Per-node batch curves for the speculation cost model") -----
# sp10/sp10c carried the exploration plan's whole-plan R(m) onto a new plan of the same model, and missed the 9B's laptop +
# board plans at m 4 (12.3 % on P1, 18.4 % on P4): the laptop's batch of 4 is cheaper than its batch of 3, a board's is not.
# Here each node has its own curve c_n(m) = T_n(m) / T_n(1), measured in its role: the laptop as host (explspec B's scan),
# a board as an RPC worker (bench_worker_role.sh through a loopback server: a worker gets no repacked weights). A plan's curve
# is the planner's split of its decode step, each node's share scaled by that node's curve and the links by 1:
#     c_plan(m) = (links + sum_n busy_n x c_n(m)) / (links + sum_n busy_n)
# and the new plan's R is the exploration plan's measured R scaled by the two plans' curves (the exploration plan, the laptop
# alone, gives back its own R):  R_new(m) = R_exp(m) x c_new(m) / c_exp(m).


def node_busy(key, assign, head, mem, ctx=4096):
    """{node: predicted decode busy ms} and the links' ms, the planner's split of a plan's decode step (energy_model.per_node)."""
    r = P.plan_fixed(key, assign, head, ctx, mem, load_mode="none", **P.PLANNER_V4)
    busy, xfer = EM.per_node(r["assign"], r["q"], r["C"], r["links"], head_at=r["head_at"], rpc_msg=P.RPC_MSG)
    return {n: v["tg"] for n, v in busy.items()}, xfer["tg"]


def plan_curve(busy, links, curves, m):
    """c_plan(m) from each node's busy share and its own curve (a node without a curve: an error, not a guess)."""
    missing = [n for n in busy if m not in curves.get(n, {})]
    if missing:
        raise KeyError("no batch curve at m %d for %s" % (m, missing))
    return (links + sum(b * curves[n][m] for n, b in busy.items())) / (links + sum(busy.values()))


def worker_curve(log):
    """{m: T(m) / T(1)} from a llama-bench -o(e) jsonl log of one model at pp m (bench_worker_role.sh's <node>.log)."""
    t = {}
    for ln in open(log, errors="replace"):
        if ln.startswith('{"build'):
            x = json.loads(ln)
            if x.get("n_gen", 0) == 0 and x.get("n_prompt", 0) > 0:
                t[x["n_prompt"]] = x["avg_ns"] / 1e6
    return {m: v / t[1] for m, v in t.items()} if 1 in t else {}


def scan_curve(fit):
    """{m: T(m) / T(1)} of an exploration plan's scan (exploration_fit's R, relative to its decode step, made relative to m 1)."""
    return {m: v / fit["R"][1] for m, v in fit["R"].items()}


def predict_nodes(fit, terms_exp, terms_new, kappa, rpc, R_new):
    """predict() with the new plan's R given (from the per-node curves) instead of carried over."""
    p = predict(fit, terms_exp, terms_new, kappa, rpc)
    dec, d = p["decode_ms"], p["draft_ms"]
    p["R"] = dict(R_new)
    for name, n in SPECS.items():
        ps = dec * R_new[n + 1] + n * d + fit["intercept_ms"]
        for pk in ("free", "copy"):
            E = fit["E"][name + "/" + pk]
            p["arms"][name + "/" + pk] = {"pass_ms": round(ps, 1), "E": E, "ms_per_token": round(ps / E, 1),
                                          "speedup": round(dec * E / ps, 3)}
    return p
