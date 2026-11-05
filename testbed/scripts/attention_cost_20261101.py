#!/usr/bin/env python3
"""The planner's attention-cost table for every node (2026-11-01, the scheduler review's item 4; the user: "do item 1 to 4
then 6-7").

    python3 testbed/scripts/attention_cost_20261101.py

Writes testbed/dataset/attention_cost_20261101.json: attention_cost_20261028.json's boards (the laptop and the Pi 4 by the
query-head law, the Jetson by its own attention op per shape) and two changes:
- the Pi 5 gets its law. It had none, so a plan with depth and the Pi 5 stopped with a TypeError (predict_pool.profiles:
  depth x None). lc4charb measured the replacement board's decode growing 24.605 ms per 1,000 tokens of depth on TinyLlama
  (the least-squares slope over depths 0, 2,048, 4,096 and 8,192; testbed/results/lc4charb_20261024_065032/score.json), so
  its ns per query-head dimension is that slope over TinyLlama's n_head x head_dim x n_layers;
- every board carries the hybrids' attention fraction (one block in full_attention_interval computes attention; the others
  are recurrent, their state fixed, whatever the depth): until now only the Jetson's law applied it, and the CPU laws charged
  every block of a hybrid as if it attended.
The fraction is 1 / full_attention_interval, read from hybrid_shapes_20261101.json (every listed shape of an architecture
must agree). predict_pool.attn_ms applies any board's hybrid_attention_fraction (2026-11-01; the tables before carry it on
the Jetson alone, so every registration that priced depth reproduces). The query-head laws are straight lines through depths
whose cost curves upward on both CPUs (lc4charb: the laptop's TinyLlama 30.0 -> 57.1 -> 100.4 -> 232.6 ms at 0 / 2,048 /
4,096 / 8,192): they over-price a moderate depth and fit the long end.
"""
import datetime, hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(ROOT, "testbed", "dataset")
BASE = os.path.join(DS, "attention_cost_20261028.json")
HYBRID = os.path.join(DS, "hybrid_shapes_20261101.json")
LC4CHARB = os.path.join(ROOT, "testbed", "results", "lc4charb_20261024_065032", "score.json")
TINY = "tinyllama-1.1b-q4km"                                   # lc4charb's model, byte-identical on every node
OUT = os.path.join(DS, "attention_cost_20261101.json")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit("%s exists: never overwritten" % os.path.relpath(OUT, ROOT))
    base = json.load(open(BASE))
    hyb = json.load(open(HYBRID))["shapes"]
    q = json.load(open(os.path.join(DS, "model_shapes.json")))[TINY]
    pi5 = json.load(open(LC4CHARB))["benches"]["pi5"]
    ns = pi5["slope_ms_per_1k"] / 1000.0 / (q["n_head"] * q["head_dim"] * q["n_layers"]) * 1e6   # ms per token -> ns
    frac = {}
    for name, h in hyb.items():
        f = 1.0 / h["full_attention_interval"]
        if frac.setdefault(h["architecture"], f) != f:
            raise SystemExit("%s: its interval disagrees with another %s shape's" % (name, h["architecture"]))
    boards = {n: dict(b) for n, b in base["boards"].items()}
    boards["pi5"] = {"law": "query_heads", "ns_per_qhead_dim": round(ns, 5),
                     "from": "lc4charb: TinyLlama Q4_K_M on the replacement Pi 5 (-t 4), %.3f ms per 1,000 tokens of depth "
                             "(least squares over depths 0, 2,048, 4,096, 8,192; one run of llama-bench -r 3), over its "
                             "n_head x head_dim x n_layers = %d" % (pi5["slope_ms_per_1k"], q["n_head"] * q["head_dim"] * q["n_layers"]),
                     "ms_per_token_by_depth": pi5["ms_per_token_by_depth"]}
    for b in boards.values():
        b["hybrid_attention_fraction"] = dict(frac)
    res = {"made": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "_source": "attention_cost_20261101.py: attention_cost_20261028.json's boards, the Pi 5's law from lc4charb, and "
                      "the hybrids' attention fraction (1 / full_attention_interval, hybrid_shapes_20261101.json) on every board",
           "boards": boards,
           "inputs_sha256": {os.path.relpath(p, ROOT): sha(p) for p in (BASE, HYBRID, LC4CHARB, os.path.join(DS, "model_shapes.json"),
                                                                       os.path.abspath(__file__))}}
    json.dump(res, open(OUT, "w"), indent=1)
    for n, b in sorted(boards.items()):
        print("%-7s %-12s %s; hybrid fraction %s" % (n, b["law"], ("%.5f ns per query-head dim" % b["ns_per_qhead_dim"])
                                                     if b["law"] == "query_heads" else "%d shapes" % len(b.get("table", {})),
                                                     b["hybrid_attention_fraction"]))
    print("wrote %s" % os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
