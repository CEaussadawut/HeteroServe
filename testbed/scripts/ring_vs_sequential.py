#!/usr/bin/env python3
"""Sequential pipeline vs prima.cpp's piped ring, priced with our measurements (2026-10-07). Reads files only.

    python3 testbed/scripts/ring_vs_sequential.py [--out testbed/dataset/ring_vs_sequential_20261007.json]

The advisor's question (meeting.txt): "try sequential -- is it faster?". Our runtime is a sequential pipeline: each
token visits each board once (k = 1). prima.cpp's piped ring visits the ring k times per token, each board holding k
windows. Halda prices the network as zero (common.cpp:971 at 288df6a2), so in memory every k ties and it keeps k = 1
(its own 2026-09-30 log shows the ties); the ring exists to overlap DISK prefetch with compute when the model does not
fit (Halda's sets M1-M3).

A. In memory: every engine plan in testbed/dataset/halda_offline_20261007.json (decode, no headroom), priced at k = 1
   (the published prediction) and k = 2..4. Each extra pass costs every worker one more host-worker round trip --
   2 x RTT + the hidden state out and back, predict_pool's own link terms -- and, beside it, the per-visit RPC cost the
   smoke test measured on the Jetson (rpc_smoke_pi5_jetson 2026-09-27: 2.2 ms per token at one worker, ngl=99 vs local).
B. Not in memory: for every model a board set cannot hold (the engine refuses it), the bytes that would stream from
   storage per decoded token -- the weights beyond the set's usable memory -- at the measured SD reads (Pi 5 93.2 MB/s,
   Pi 4 44.7 MB/s, 2026-09-19; the Jetson's card is unmeasured, so a set with it streams only from the Pis' share, a
   lower bound), against the compute time of the same model priced as if it fitted. The ring can at best hide the
   smaller of the two (perfect overlap: max(compute, disk)); sequential pays their sum.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import predict_pool as pp                                   # noqa: E402

SD_MBPS = {"pi5": 93.2, "pi4": 44.7}           # measured sequential reads, testbed/README.md (2026-09-19)
RPC_VISIT_MS = 2.2                             # measured: one RPC worker visit per token (rpc_smoke 2026-09-27)
KS = (1, 2, 3, 4)


def parse(pipeline):
    return [(p.split()[0], int(p.split()[1])) for p in pipeline.split(" + ")]


def main(argv):
    out = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(ROOT, "testbed", "dataset",
                                                                           "ring_vs_sequential_20261007.json")
    ds = os.path.join(ROOT, "testbed", "dataset")
    h = json.load(open(os.path.join(ds, "halda_offline_20261007.json")))
    shapes = json.load(open(os.path.join(ds, "model_shapes.json")))
    pairs = json.load(open(os.path.join(ds, "link.json")))["pairs"]
    mem = json.load(open(os.path.join(ds, "memory_at_rest.json")))
    res = {"n_ctx": h["n_ctx"], "rpc_visit_ms_measured": RPC_VISIT_MS, "sd_read_mbps": SD_MBPS,
           "in_memory": [], "not_in_memory": []}
    print("A. In memory: decode ms/token at k ring passes (k = 1 is our sequential pipeline; extra passes priced with "
          "predict_pool's link terms, and [with the measured %.1f ms per RPC visit])" % RPC_VISIT_MS)
    print("%-11s %-15s %-30s %9s %9s %9s %9s  %s" % ("model", "boards", "plan (k = 1)", "k=1", "k=2", "k=3", "k=4",
                                                   "per extra pass"))
    for c in h["configs"]:
        e = c["engine"]
        shape = pp.MODELS[c["model"]][0]
        q = shapes[shape]
        if "decode_ms" not in e:
            # B: the set cannot hold the model
            usable = sum(mem[n]["mem_available_mib"] for n in c["boards"]) * 2 ** 20
            total = sum(q["layer_bytes"]) + q["head_bytes"]
            over = max(0.0, total - usable)
            pis = [n for n in c["boards"] if n in SD_MBPS]
            share = {n: over * mem[n]["mem_available_mib"] / sum(mem[m]["mem_available_mib"] for m in pis) for n in pis} if pis else {}
            disk_ms = max((share[n] / 1e6) / SD_MBPS[n] * 1000 for n in pis) if pis else None
            res["not_in_memory"].append({"model": c["model"], "boards": c["boards"], "weights_gb": round(total / 1e9, 2),
                                         "usable_gb": round(usable / 1e9, 2), "overflow_gb": round(over / 1e9, 2),
                                         "disk_ms_per_token_lower_bound": round(disk_ms, 0) if disk_ms else None})
            continue
        assign = parse(e["pipeline"])
        nodes = [n for n, _ in assign]
        if len(nodes) < 2:
            continue                                    # one board: there is no ring
        L, _ = pp.link_view(pairs, c["host"], nodes)
        act = q["activation_bytes"]
        per_pass = sum(2 * L[n]["rtt_ms"] + 2 * act / (L[n]["mbps"] * 1e6 / 8 / 1000) for n in nodes[1:])
        per_pass_rpc = RPC_VISIT_MS * (len(nodes) - 1)
        row = {"model": c["model"], "boards": c["boards"], "plan": e["pipeline"], "head_on": e["head_on"],
               "decode_ms": {k: round(e["decode_ms"] + (k - 1) * per_pass, 1) for k in KS},
               "decode_ms_with_measured_rpc": {k: round(e["decode_ms"] + (k - 1) * per_pass_rpc, 1) for k in KS},
               "per_extra_pass_ms": round(per_pass, 2), "per_extra_pass_ms_measured_rpc": round(per_pass_rpc, 2),
               "per_extra_pass_pct": round(100 * per_pass_rpc / e["decode_ms"], 2)}
        res["in_memory"].append(row)
        print("%-11s %-15s %-30s %9.1f %9.1f %9.1f %9.1f  +%.2f ms [+%.1f ms, %.2f %%]" % (
            c["model"], "+".join(c["boards"]), e["pipeline"], *[row["decode_ms"][k] for k in KS], per_pass,
            per_pass_rpc, row["per_extra_pass_pct"]))
    print("\nB. Not in memory: what streaming the overflow from storage would cost per decoded token")
    print("%-11s %-15s %8s %8s %9s %14s" % ("model", "boards", "weights", "usable", "overflow", "disk ms/token"))
    for r in res["not_in_memory"]:
        print("%-11s %-15s %7.2fG %7.2fG %8.2fG %14s" % (r["model"], "+".join(r["boards"]), r["weights_gb"], r["usable_gb"],
                                                        r["overflow_gb"], r["disk_ms_per_token_lower_bound"] or "unmeasured"))
    json.dump(res, open(out, "w"), indent=1)
    print("wrote %s" % os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
