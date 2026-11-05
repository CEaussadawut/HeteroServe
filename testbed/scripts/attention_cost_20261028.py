#!/usr/bin/env python3
"""The planner's attention-cost table with the Jetson priced by its own attention op (2026-10-28, lc4n).

    python3 testbed/scripts/attention_cost_20261028.py [--fa testbed/results/lc4nfa_20261028_081317/fa.json]

Writes testbed/dataset/attention_cost_20261028.json: attention_cost_20261011b.json's boards (the Pi 4 and the laptop by the
query-head law, as before) with the Jetson's "kv_bytes" law replaced by "fa_table": per attention shape (head size x query
heads x KV heads), the ms per attention layer per token of context depth, read from fa_bench on the Jetson (fa.json): the
least-squares slope of its time per graph over the cache lengths from 1,280 cells (the long range, past the 1,024-cell
point where ggml-vulkan may change the kernel's layout), divided by the graph's layers; and the same below 1,024 cells
(256-768), where it differs. predict_pool.attn_ms() reads it (law "fa_table"). Registered test: lc4n
(testbed/dataset/lc4n_predictions_20261028.json). The 11 October tables stay as they were, so every registration that
used them reproduces.
"""
import argparse, datetime, hashlib, json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = os.path.join(ROOT, "testbed", "dataset", "attention_cost_20261011b.json")
OUT = os.path.join(ROOT, "testbed", "dataset", "attention_cost_20261028.json")
LC4N_SCORE = "testbed/results/lc4n_20261028_081724/score_lc4n.json"


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def slope(points):
    xs, ys = zip(*points)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa", default="testbed/results/lc4nfa_20261028_081317/fa.json")
    a = ap.parse_args()
    fa = json.load(open(os.path.join(ROOT, a.fa)))
    base = json.load(open(BASE))
    table = {}
    for label, s in fa["shapes"].items():
        g = {int(k): v for k, v in s["us_per_graph"].items()}
        hi = [(k, v) for k, v in g.items() if k >= 1280]
        lo = [(k, v) for k, v in g.items() if k <= 768]
        key = "%dx%dx%d" % (s["head_dim"], s["n_head"], s["n_head_kv"])
        table[key] = {"shape": label, "layers_in_graph": s["layers"],
                      "ms_per_layer_per_token_long": round(slope(hi) / s["layers"] / 1000.0, 7),      # us per cell -> ms
                      "ms_per_layer_per_token_short": round(slope(lo) / s["layers"] / 1000.0, 7),
                      "long_from_n_kv": 1280, "short_n_kv": [256, 768]}
    boards = dict(base["boards"])
    boards["jetson"] = {"law": "fa_table", "table": table,
                        "hybrid_attention_fraction": {"qwen35": 0.25, "qwen35moe": 0.25},
                        "from": "fa_bench (testbed/scripts/fa_bench_20261028.cpp) on the Jetson, %s; the whole-model test is lc4n"
                                % a.fa,
                        "why": "lc4nx (the Vulkan backend's per-op timer): of a decoded token's ops only FLASH_ATTN_EXT grows with "
                               "the cache, and ggml-vulkan picks its kernel layout per shape and cache length, so no single law "
                               "in the shape fits (lc4charb, lc4j, lc4k, lc4m)"}
    out = {"_source": "testbed/scripts/attention_cost_20261028.py (the Jetson by its attention op, lc4n); the other boards as "
                      "attention_cost_20261011b.json",
           "made": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "boards": boards,
           "inputs_sha256": {f: sha(f) for f in (a.fa, os.path.relpath(BASE, ROOT), "testbed/scripts/fa_bench_20261028.cpp")
                             + ((LC4N_SCORE,) if os.path.exists(os.path.join(ROOT, LC4N_SCORE)) else ())}}
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote %s" % os.path.relpath(OUT, ROOT))
    for k, v in table.items():
        print("%-10s %-16s long %.5f  short %.5f ms per layer per 1 token of depth (x1000 = per 1,000 tokens)" % (
            k, v["shape"], v["ms_per_layer_per_token_long"], v["ms_per_layer_per_token_short"]))


if __name__ == "__main__":
    main()
