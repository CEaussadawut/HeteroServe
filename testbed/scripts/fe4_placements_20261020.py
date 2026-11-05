#!/usr/bin/env python3
"""The baselines a four-node full evaluation serves beside the engine's pick (the advisor, 2026-10-16: "more baseline pair").
Computes each placement for full_eval.py register, checks it against the engine's validated memory accounting, drops what does
not fit or duplicates an earlier one (saying so), and writes the derivation. Reads files only.

    python3 testbed/scripts/fe4_placements_20261020.py --model KEY [--nodes "pi5 pi4 jetson laptop"] [--ctx 4096] \\
        [--extra "label=node:k+node:k/head" ...] [--out FILE]

Placements, in this order (an earlier label wins a duplicate):
  engine    the engine's pick on every node (full_eval.py computes it itself, from the same inputs)
  boards    the engine's pick on the boards alone, without the laptop: what the cluster does when the laptop is away
  single    the fastest single node that holds the whole model (the engine priced on one node at a time)
  even      equal blocks on every node, the engine's host first, the others in the cluster's order, the head last (llama.cpp's
            -ngl places it on the last device); if that does not fit, the head on the host; else the next order that fits
  default   llama.cpp's own default split (llama-model.cpp "default split, by free memory", the RPC README's recipe with every
            node a device): device i's share is its reported free memory -- a CPU RPC server reports its node's whole RAM
            (MemTotal), the Jetson's Vulkan0 its reported free -- each block il of n_layer + 1 (the output counts as one) to the
            first device whose cumulative share exceeds il / (n_layer + 1); the host first, then the --rpc order
  halda     prima.cpp's Halda (ThinkPad/control/halda.py, validated 2026-10-07) with the engine's host as rank 0: its windows for
            every ring order of the others, the best by our predictor among those that fit, the head on the host as Halda puts it
  --extra   any named placement (e.g. the plan served before, for a cross-session check)
Writes FILE (default testbed/results/fe4_placements_<model>_<stamp>.json) and prints the --placements string.
"""
import argparse, dataclasses, datetime, itertools, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                    # noqa: E402
import compare_halda as CH                                  # noqa: E402
from control import halda                                   # noqa: E402
from control.gguf import read_metadata                      # noqa: E402
from control.placement import EXACT_FILL_HEADROOM_MB        # noqa: E402

ORDER = ["pi5", "pi4", "jetson", "laptop"]
FILES = dict(CH.FILES, **{"tinyllama-1.1b-q6_k": "tinyllama-1.1b-chat-v1.0.Q6_K.gguf",     # the 2026-10-11 files, for n_ff
                          "qwen3-30b-a3b-q6_k": "Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf"})
REPORTED_FREE = {"pi5": 8050, "pi4": 7819, "laptop": 31296,    # MemTotal, as a CPU RPC server reports it (2026-10-19/20)
                 "jetson": 3420}                               # Vulkan0's reported free: 3,250-3,591 seen; the middle


def spec(label, assign, head):
    return "%s=%s/%s" % (label, "+".join("%s:%d" % (n, k) for n, k in assign), head)


def llama_default(nodes, n_layers):
    free = [REPORTED_FREE[n] for n in nodes]
    tot = float(sum(free))
    cum, s = [], 0.0
    for f in free:
        s += f / tot
        cum.append(s)
    count = [0] * len(nodes)
    for il in range(n_layers + 1):
        x = il / float(n_layers + 1)
        i = next(j for j, c in enumerate(cum) if x < c)
        if il < n_layers:
            count[i] += 1
        head_dev = i
    return [(n, k) for n, k in zip(nodes, count) if k > 0], nodes[head_dev]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--nodes", default="pi5 pi4 jetson laptop")
    ap.add_argument("--ctx", type=int, default=4096); ap.add_argument("--rtt-trips", type=int, default=1)
    ap.add_argument("--extra", nargs="*", default=[]); ap.add_argument("--out")
    ap.add_argument("--memfile", help="a memory snapshot (memory_snapshot.py) in place of memory_at_rest.json (2026-10-21)")
    a = ap.parse_args()
    nodes = a.nodes.split()
    ds = os.path.join(ROOT, "testbed", "dataset")
    mem = json.load(open(a.memfile or os.path.join(ds, "memory_at_rest.json")))
    shape, quant, _ = P.MODELS[a.model]
    shapes = json.load(open(os.path.join(ds, "model_shapes.json")))
    q = shapes[shape]
    L = q["n_layers"]
    res = {"made": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "model": a.model, "nodes": nodes,
           "ctx": a.ctx, "rtt_trips": a.rtt_trips, "reported_free_mib": REPORTED_FREE, "placements": [], "dropped": []}

    def fixed(assign, head):
        r = P.plan_fixed(a.model, [(n, int(k)) for n, k in assign], head, a.ctx, mem, rtt_trips=a.rtt_trips, load_mode="none")
        m = P.memory(r["assign"], r["q"], mem, r["devices"], r["model"], head_at=r["head_at"])
        worst = min(v["margin_mib"] for v in m.values())
        return r, worst

    seen = {}

    def add(label, assign, head, how):
        key = (tuple(tuple(x) for x in assign), head if len(assign) > 1 else "last")
        if key in seen:
            res["dropped"].append({"label": label, "why": "the same placement as %s" % seen[key], "assign": assign, "head": head})
            return
        try:
            r, worst = fixed(assign, head)
        except Exception as e:
            res["dropped"].append({"label": label, "why": "cannot be priced: %s" % str(e)[:160], "assign": assign, "head": head})
            return
        if worst < EXACT_FILL_HEADROOM_MB:
            res["dropped"].append({"label": label, "why": "keeps %d MiB on its tightest node (< %d headroom)" % (worst, EXACT_FILL_HEADROOM_MB),
                                   "assign": assign, "head": head})
            return
        seen[key] = label
        res["placements"].append({"label": label, "assign": assign, "head": head, "how": how, "min_margin_mib": worst,
                                  "decode_ms": round(r["decode_ms"], 1), "prefill_ms": round(r["prefill_ms"], 1)})

    e = P.plan_host_choice(a.model, nodes, a.ctx, mem, rtt_trips=a.rtt_trips, load_mode="none")
    host = e["host"]
    add("engine", [list(x) for x in e["assign"]], "host" if e["head_at"] == 0 else "last", "plan_host_choice on %s" % nodes)
    boards = [n for n in nodes if n != "laptop"]
    try:
        b = P.plan_host_choice(a.model, boards, a.ctx, mem, rtt_trips=a.rtt_trips, load_mode="none")
        add("boards", [list(x) for x in b["assign"]], "host" if b["head_at"] == 0 else "last", "plan_host_choice on %s" % boards)
    except ValueError as ex:
        res["dropped"].append({"label": "boards", "why": "the boards alone cannot hold it: %s" % str(ex)[:160]})
    best = None
    for n in nodes:
        try:
            s = P.plan_host_choice(a.model, [n], a.ctx, mem, host=n, rtt_trips=a.rtt_trips, load_mode="none")
        except ValueError:
            continue
        if best is None or s["decode_ms"] < best["decode_ms"]:
            best = s
    if best:
        add("single", [[best["host"], L]], "last", "the fastest single node that holds it")
    else:
        res["dropped"].append({"label": "single", "why": "no single node holds it"})
    rest0 = [n for n in ORDER if n in nodes and n != host]
    base, extra = divmod(L, len(nodes))
    done = False
    for rest in [rest0] + [list(p) for p in itertools.permutations(rest0) if list(p) != rest0]:
        order = [host] + rest
        assign = [[n, base + (1 if i < extra else 0)] for i, n in enumerate(order)]
        for head in ("last", "host"):
            try:
                r, worst = fixed(assign, head)
            except Exception:
                continue
            if worst >= EXACT_FILL_HEADROOM_MB:
                add("even", assign, head, "equal blocks, order %s, head %s" % (order, head)); done = True; break
        if done:
            break
    if not done:
        res["dropped"].append({"label": "even", "why": "no order of equal blocks fits"})
    dnodes = [host] + rest0
    d_assign, d_head_node = llama_default(dnodes, L)
    holders_ = [n for n, _ in d_assign]
    if d_head_node == holders_[-1]:
        d_head, d_note = "last", "the output block falls on %s, the last device with blocks" % d_head_node
    elif d_head_node == holders_[0]:
        d_head, d_note = "host", "the output block falls on the host, %s" % d_head_node
    else:     # llama.cpp would give the output alone to a device with no blocks: not a placement serve_pool.sh can write
        d_head, d_note = "last", ("llama.cpp would put the output alone on %s, which gets no blocks; served with the head on %s, "
                                  "the last device with blocks (the nearest placement serve_pool.sh can write)" % (d_head_node, holders_[-1]))
    add("default", [list(x) for x in d_assign], d_head,
        "llama.cpp's split by reported free memory %s over %s; %s" % ({n: REPORTED_FREE[n] for n in dnodes}, dnodes, d_note))
    if q["architecture"] in CH.PRIMA_ARCH or True:
        kv = read_metadata(os.path.join(CH.CACHE, FILES[shape])) if shape in FILES else None
        if kv:
            arch = kv.get("general.architecture")
            n_ff = int(kv[arch + ".feed_forward_length"]) if (arch + ".feed_forward_length") in kv else None
        else:
            n_ff = None
        R = P.rows(os.path.join(ds, "characterisation.csv"), "char")
        C, _ = P.costs(R, shapes, quant)
        if n_ff and all(n in C for n in nodes):
            hs = []
            for ring, hp, caps in CH.halda_plans([n for n in nodes if n != host], host, q, C, mem, a.ctx, n_ff):
                if hp["feasible"]:
                    assign = [[n, hp["layers"][n]] for n in ring if hp["layers"].get(n, 0) > 0]
                    try:
                        r, worst = fixed(assign, "host" if len(assign) > 1 else "last")
                    except Exception:
                        continue
                    hs.append((r["decode_ms"], worst, assign, ring))
            fit = [h for h in hs if h[1] >= EXACT_FILL_HEADROOM_MB]
            if fit:
                d, w, assign, ring = min(fit)
                add("halda", assign, "host" if len(assign) > 1 else "last",
                    "Halda's windows, ring %s (the best of %d fitting rings by our predictor; prima.cpp %s this architecture)" % (
                        ring, len(fit), "runs" if q["architecture"] in CH.PRIMA_ARCH else "does NOT run"))
            else:
                res["dropped"].append({"label": "halda", "why": "no Halda ring fits (%d priced)" % len(hs)})
        else:
            res["dropped"].append({"label": "halda", "why": "no n_ff in the GGUF header, or a node without rows at %s" % quant})
    for x in a.extra:
        lab, assign, head = (lambda t: (t[0], [[n, int(k)] for n, k in (y.split(":") for y in t[1].split("/")[0].split("+"))],
                                        t[1].split("/")[1]))(x.split("=", 1))
        add(lab, assign, head, "named on the command line")
    labels = [p["label"] for p in res["placements"]]
    res["placements_arg"] = ",".join(["engine"] + [spec(p["label"], p["assign"], p["head"]) for p in res["placements"] if p["label"] != "engine"])
    out = a.out or os.path.join(ROOT, "testbed", "results", "fe4_placements_%s_%s.json" % (a.model, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    json.dump(res, open(out, "w"), indent=1)
    for p in res["placements"]:
        print("%-8s %-40s head %-4s decode %7.1f ms  prefill %7.1f ms  margin %5d MiB  (%s)" % (
            p["label"], " + ".join("%s %d" % tuple(x) for x in p["assign"]), p["head"], p["decode_ms"], p["prefill_ms"],
            p["min_margin_mib"], p["how"][:70]))
    for d in res["dropped"]:
        print("DROPPED %-8s %s" % (d["label"], d["why"]))
    print("--placements '%s'" % res["placements_arg"])
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
