#!/usr/bin/env python3
"""Pre-register pooled ENERGY cells on the boards that remain (2026-10-05: the Pi 5 was returned). Reads files only.

Each cell is planned with the placement engine's own functions, imported from predict_pool.py and not changed, so every
earlier registration still reproduces. The plan uses the given nodes and host. The output is a registration in
run_pool.sh's format ("configs": pipeline, bench, memory), so `CELL=<id> PRED=<file> bash testbed/scripts/run_pool.sh`
runs a cell as every pool cell has run. Per cell it adds:
  - an energy protocol (bench.protocol): llama-bench -t 4 -p P -n N -r R --progress -oe jsonl. llama-bench runs the prompt
    test and then the generation test, each R times, so ONE invocation gives a prefill window and a decode window. R is
    sized so that both last at least --phase-s at the engine's predicted rates, and it is never below 3;
  - the throughput the engine predicts (the claim every pool registration makes);
  - the energy hypotheses (QR5): each cell's net energy per token, summed over its boards, is measured as a baseline.
    Where a rule and an even split of the same model are both planned, the rule's net J/token is predicted to be at
    least 25 % below the even split's in each phase (QR5's stretch target), by each method.

Usage:
  register_pool_energy.py --nodes pi4,jetson --host pi4 --cells tinyllama.rule,tinyllama.even,qwen3_14b.rule \\
      --out testbed/dataset/pool_energy_predictions_<stamp>.json [--phase-s 120] [--dry-run]
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import predict_pool as P  # noqa: E402  (the engine's registration tool; only its functions are used)

SHAPE_PN = {"tinyllama": (512, 128)}          # the characterisation shape; every other model: the pool protocol (64/16)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--cells", required=True, help="comma list of <model key>.<rule|even>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase-s", type=float, default=120.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    nodes = a.nodes.split(",")
    if a.host not in nodes:
        raise SystemExit("the host must be one of --nodes")
    if os.path.exists(a.out) and not a.dry_run:
        raise SystemExit("%s exists: a registration is never overwritten" % a.out)
    ds = os.path.join(ROOT, "testbed", "dataset")
    inputs = {f: os.path.join(ds, f) for f in ("characterisation.csv", "model_shapes.json", "link.json", "memory_at_rest.json")}
    shapes = json.load(open(inputs["model_shapes.json"]))
    pairs = json.load(open(inputs["link.json"]))["pairs"]
    mem = json.load(open(inputs["memory_at_rest.json"]))
    R = P.rows(inputs["characterisation.csv"], "char")
    res = {"registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "campaign": "pooled energy on the remaining boards (%s, hosted on the %s), after the Pi 5 was returned" % ("+".join(nodes), a.host),
           "inputs_sha256": {**{f: sha(p) for f, p in inputs.items()}, "cluster.json": sha(P.CLUSTER),
                             "predict_pool.py": sha(os.path.join(HERE, "predict_pool.py")),
                             "register_pool_energy.py": sha(os.path.abspath(__file__)),
                             **{f: sha(os.path.join(HERE, f)) for f in ("energy_pool_run.py", "score_energy_pool_run.py",
                                                                         "run_pool.sh", "shelly_strip.py")},
                             "strip.json": sha(os.path.join(ROOT, "testbed", "energy", "strip.json"))},
           "engine_sha256": {f: sha(os.path.join(ROOT, "ThinkPad", "control", f)) for f in ("placement.py", "gguf.py")},
           "rows": "char (the characterisation rows, as every pool registration before 2026-09-28 night); --size off",
           "configs": {}, "energy_hypotheses": []}
    planned = {}
    for cell in a.cells.split(","):
        key, kind = cell.rsplit(".", 1)
        shape, quant, _ = P.MODELS[key]
        q = shapes[shape]
        C, src = P.costs(R, shapes, quant, False)
        members = [n for n in nodes if n in C]
        L, missing = P.link_view(pairs, a.host, members)
        if missing:
            raise SystemExit("%s: no measured link %s" % (cell, missing))
        devs, mp = P.profiles(C, q, mem, L, members), P.model_profile(q, shape)
        if kind == "rule":
            plan = P.plan_optimal(devs, mp, "decode", host_candidates=[a.host])
        elif kind == "even":
            plan = P.plan_even(devs, mp, node_order=[a.host] + [n for n in members if n != a.host])
        else:
            raise SystemExit("unknown kind %r (rule or even)" % kind)
        pipe = [(n, plan.layers[n]) for n in plan.order if plan.layers.get(n, 0) > 0]
        if not getattr(plan, "feasible", True) or not pipe:
            res["configs"][cell] = {"skipped": "infeasible on %s" % "+".join(members)}
            print("%-18s INFEASIBLE" % cell)
            continue
        Lp, _ = P.link_view(pairs, pipe[0][0], [n for n, _ in pipe])
        dec, pre = P.predict(pipe, q, C, Lp)
        pp_n, tg_n = SHAPE_PN.get(key, (P.PP, P.TG))
        rep_pp, rep_tg = pp_n * pre / 1000.0, tg_n * dec / 1000.0
        reps = max(3, math.ceil(a.phase_s / rep_pp), math.ceil(a.phase_s / rep_tg))
        bench = P.bench_args(pipe, key)
        bench["protocol"] = "-t 4 -p %d -n %d -r %d --progress -oe jsonl" % (pp_n, tg_n, reps)
        memo = P.memory(pipe, q, mem, devs, mp)
        res["configs"][cell] = {
            "model": shape, "quant": quant, "why": "%s on %s, hosted on the %s (energy campaign)" % (kind, "+".join(members), a.host),
            "pipeline": [{"node": n, "blocks": k} for n, k in pipe], "head_on": pipe[-1][0], "bench": bench, "memory": memo,
            "predicted": {"decode_ms_per_token": round(dec, 1), "decode_tok_s": round(1000 / dec, 3),
                          "prefill_ms_per_token": round(pre, 1), "prefill_tok_s": round(1000 / pre, 2),
                          "timed_s": {"pp": round(reps * rep_pp, 1), "tg": round(reps * rep_tg, 1)}},
            "cost_sources": {n: src[n] for n, _ in pipe}, "link_pairs": {n: Lp[n]["pair"] for n, _ in pipe[1:]},
            "engine": {"order": plan.order, "layers": plan.layers}}
        planned[cell] = res["configs"][cell]
        print("%-18s %-22s decode %6.3f t/s  prefill %6.2f t/s  %s  timed pp %.0f s / tg %.0f s  margins %s" % (
            cell, " + ".join("%s %d" % x for x in pipe), 1000 / dec, 1000 / pre, bench["protocol"],
            reps * rep_pp, reps * rep_tg, {n: int(v["margin_mib"]) for n, v in memo.items()}))
    for key in sorted(set(c.rsplit(".", 1)[0] for c in planned)):
        if "%s.rule" % key in planned and "%s.even" % key in planned:
            res["energy_hypotheses"].append({
                "model": key, "rule": "%s.rule" % key, "even": "%s.even" % key, "min_saving_pct": 25.0,
                "claim": "the rule's pool net J/token is at least 25 % below the even split's, in prefill and in decode, by "
                         "each method (QR5 stretch: 'find a placement reducing energy by at least 25%')"})
    res["scoring"] = ("testbed/scripts/score_energy_pool_run.py: throughput against the engine's prediction (within 10 %); per "
                      "phase, the pool's gross and net energy per token summed over its boards' outlets (the router reported "
                      "apart, not charged); the across-run spread; the hypotheses above; PASS / FAIL / N/A, reported as found")
    if a.dry_run:
        print("dry run: nothing written")
        return
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    os.chmod(a.out, 0o444)
    print("wrote %s (registered %s, read-only)" % (os.path.relpath(a.out, ROOT), res["registered"]))


if __name__ == "__main__":
    main()
