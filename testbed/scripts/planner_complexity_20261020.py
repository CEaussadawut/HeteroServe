#!/usr/bin/env python3
"""How the placement engine's work grows with the cluster and the model (the advisor, 2026-10-16: "BigO"). Laptop only;
nothing touches a board.

    python3 testbed/scripts/planner_complexity_20261020.py [--reps 5]

The engine (ThinkPad/control/placement.plan_optimal) enumerates every ordered subset of the N boards -- a pipeline order --
and for each one with two or more boards tries the output head on the last board and on the host; for each (order, head)
the exact fill (_fill_exact) chooses the k-1 block boundaries by a dynamic programme over the L layers, O(k L^2) in the
worst case and cut short where a run of blocks stops fitting a board. So the search is
    sum over k = 1..N of N!/(N-k)! orders  x  (2 heads if k >= 2)  x  O(k L^2),   capped at max_permutations = 5000 orders.
plan_host_choice runs it once per board holding the file, as host; the orders it evaluates are those starting with that
host, so over every candidate it evaluates the same set once.
Measured here, as wall time of the real code (time.perf_counter, the median of --reps calls):
  1. the real cluster: plan_host_choice for each model the engine knows on 2, 3 and 4 boards (the laptop's memory is a
     provisional 27,000 MiB where memory_at_rest.json has none: it changes what fits, not how the search grows);
  2. synthetic clusters of N = 1..8 boards, copies of the four real boards' profiles under new names, for Qwen3-14B (L = 40);
  3. synthetic models of L = 22..352 layers (TinyLlama's blocks repeated) on the real four boards.
Writes testbed/results/planner_complexity_<stamp>/{timings.json, report.txt, complexity.png}.
"""
import argparse, dataclasses, datetime, itertools, json, math, os, statistics, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                    # noqa: E402
from control import placement as PL                        # noqa: E402

PROVISIONAL_LAPTOP_MIB = 27000


def orders(n, cap=5000):
    """(ordered subsets, (order, head) pairs evaluated, orders actually checked under the cap) for n boards."""
    o = sum(math.factorial(n) // math.factorial(n - k) for k in range(1, n + 1))
    return o, n + 2 * (o - n), min(o, cap)


def timed(f, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000, min(ts) * 1000, max(ts) * 1000, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(ROOT, "testbed", "results", "planner_complexity_%s" % stamp)
    os.makedirs(out)
    ds = os.path.join(ROOT, "testbed", "dataset")
    mem = json.load(open(os.path.join(ds, "memory_at_rest.json")))
    if "laptop" not in mem:
        mem["laptop"] = {"mem_total_mib": 31800, "mem_available_mib": PROVISIONAL_LAPTOP_MIB, "note": "provisional, timing only"}
    res = {"taken": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "reps": a.reps,
           "cpu": open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t"),
           "python": sys.version.split()[0], "analytic": {}, "real": [], "synthetic_n": [], "synthetic_l": []}
    for n in range(1, 9):
        o, pairs, chk = orders(n)
        res["analytic"][n] = {"ordered_subsets": o, "order_head_pairs": pairs, "orders_checked_under_cap": chk}

    # 1. the real cluster
    for key in ("tinyllama", "qwen35_9b", "qwen3_14b", "nemo12b", "qwen3moe", "qwen38_27b"):
        for nodes in (["pi4", "jetson"], ["pi5", "pi4", "jetson"], ["pi5", "pi4", "jetson", "laptop"]):
            try:
                med, lo, hi, r = timed(lambda: P.plan_host_choice(key, nodes, 4096, mem, rtt_trips=1, load_mode="none"), a.reps)
                pick = " + ".join("%s %d" % x for x in r["assign"])
                res["real"].append({"model": key, "n_layers": r["q"]["n_layers"], "nodes": nodes, "ms_median": round(med, 2),
                                    "ms_min": round(lo, 2), "ms_max": round(hi, 2), "pick": pick, "host": r["host"],
                                    "candidates": len(r["considered"])})
            except Exception as e:                      # a model the boards cannot hold, or a quant without rows
                res["real"].append({"model": key, "nodes": nodes, "error": str(e)[:160]})

    # the four real boards' profiles at Qwen3-14B, as plan_host_choice builds them with the laptop as host
    shapes = json.load(open(os.path.join(ds, "model_shapes.json")))
    pairs = json.load(open(os.path.join(ds, "link.json")))["pairs"]
    R = P.rows(os.path.join(ds, "characterisation.csv"), "char")
    shape, quant, _ = P.MODELS["qwen3_14b"]
    q = shapes[shape]
    C, _ = P.costs(R, shapes, quant)
    real4 = ["pi5", "pi4", "jetson", "laptop"]
    L4, missing = P.link_view(pairs, "pi5", real4)
    devs4 = P.profiles(C, q, mem, L4, real4)
    mp = dataclasses.replace(P.model_profile(q, shape, "none"), context_len=4096)

    # 2. N boards (copies of the real four, in turn)
    for n in range(1, 9):
        devs = {}
        for i in range(n):
            src = devs4[real4[i % 4]]
            devs["b%d" % i] = dataclasses.replace(src, node_id="b%d" % i)
        med, lo, hi, plan = timed(lambda: PL.plan_optimal(devs, mp, "decode"), a.reps if n < 8 else 1)
        res["synthetic_n"].append({"n_boards": n, "n_layers": mp.n_layers, "ms_median": round(med, 2), "ms_min": round(lo, 2),
                                   "ms_max": round(hi, 2), "feasible": plan.feasible,
                                   "orders_checked": orders(n)[2], "capped": orders(n)[0] > 5000})

    # 3. L layers (TinyLlama's blocks repeated) on the real four
    tq = shapes[P.MODELS["tinyllama"][0]]
    tC, _ = P.costs(R, shapes, P.MODELS["tinyllama"][1])
    tdevs = P.profiles(tC, tq, mem, L4, real4)
    base = P.model_profile(tq, P.MODELS["tinyllama"][0], "none")
    for mult in (1, 2, 4, 8, 16):
        lb = tuple(list(base.layer_bytes) * mult)
        m = dataclasses.replace(base, n_layers=len(lb), layer_bytes=lb, context_len=4096)
        med, lo, hi, plan = timed(lambda: PL.plan_optimal(tdevs, m, "decode"), a.reps if mult < 16 else 2)
        res["synthetic_l"].append({"n_layers": len(lb), "n_boards": 4, "ms_median": round(med, 2), "ms_min": round(lo, 2),
                                   "ms_max": round(hi, 2), "feasible": plan.feasible})

    json.dump(res, open(os.path.join(out, "timings.json"), "w"), indent=1)
    L = ["planner complexity, %s, %s, Python %s, median of %d calls" % (res["taken"], res["cpu"], res["python"], a.reps), "",
         "analytic: N boards -> ordered subsets / (order, head) pairs / orders checked under the 5,000 cap"]
    L += ["  N=%d  %6d  %6d  %5d" % (n, v["ordered_subsets"], v["order_head_pairs"], v["orders_checked_under_cap"])
          for n, v in res["analytic"].items()]
    L += ["", "real cluster: plan_host_choice wall time (ms)"]
    for r in res["real"]:
        L.append("  %-11s L=%-3s %-28s %s" % (r["model"], r.get("n_layers", "?"), "+".join(r["nodes"]),
                                              ("%8.2f ms  pick %s (host %s, %d candidates)" % (r["ms_median"], r["pick"], r["host"],
                                                                                              r["candidates"]))
                                              if "ms_median" in r else "error: " + r["error"]))
    L += ["", "synthetic N (Qwen3-14B, L=40): plan_optimal wall time"]
    L += ["  N=%d  %10.2f ms  orders checked %5d%s" % (r["n_boards"], r["ms_median"], r["orders_checked"],
                                                        "  (capped)" if r["capped"] else "") for r in res["synthetic_n"]]
    L += ["", "synthetic L (TinyLlama blocks repeated, 4 real boards): plan_optimal wall time"]
    L += ["  L=%-4d %10.2f ms" % (r["n_layers"], r["ms_median"]) for r in res["synthetic_l"]]
    open(os.path.join(out, "report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        xs = [r["n_boards"] for r in res["synthetic_n"]]; ys = [r["ms_median"] for r in res["synthetic_n"]]
        ax[0].semilogy(xs, ys, "o-", color="#2a78d6", label="measured")
        k = ys[3] / res["analytic"][4]["order_head_pairs"]
        ax[0].semilogy(xs, [k * min(res["analytic"][n]["order_head_pairs"], 2 * 5000) for n in xs], "--", color="#52514e",
                       label="∝ (order, head) pairs")
        ax[0].set_xlabel("boards N"); ax[0].set_ylabel("plan time (ms)"); ax[0].set_title("Qwen3-14B, L = 40", loc="left")
        ax[0].legend(frameon=False, fontsize=8)
        xs = [r["n_layers"] for r in res["synthetic_l"]]; ys = [r["ms_median"] for r in res["synthetic_l"]]
        ax[1].loglog(xs, ys, "o-", color="#2a78d6", label="measured")
        ax[1].loglog(xs, [ys[0] * (x / xs[0]) ** 2 for x in xs], "--", color="#52514e", label="∝ L²")
        ax[1].set_xlabel("layers L"); ax[1].set_title("4 boards", loc="left"); ax[1].legend(frameon=False, fontsize=8)
        for x in ax:
            x.grid(True, color="#e4e3df"); x.spines["top"].set_visible(False); x.spines["right"].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(out, "complexity.png"), dpi=200)
    except Exception as e:
        print("no figure:", e)
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
