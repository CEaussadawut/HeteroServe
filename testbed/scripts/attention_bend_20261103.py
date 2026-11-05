#!/usr/bin/env python3
"""A depth law that bends (2026-11-03; the user: "... A depth law that bends, so decode can be priced at real context depths").

    python3 testbed/scripts/attention_bend_20261103.py [--out testbed/dataset/attention_cost_20261103.json]

PLANNER_V4's attention table (attention_cost_20261101.json) prices a CPU board's depth cost as a straight line, ns per query-
head dimension per block per token of depth. Measured on the boards, the cost of each extra token of depth grows with depth.
lc4charb (TinyLlama Q4_K_M alone, -t 4, depths 0 / 2,048 / 4,096 / 8,192): the laptop's per-1,000-token increments are 13.3,
21.6 and 33.0 ms; the Pi 5's 11.7, 21.4 and 31.4. A straight line through the far points prices moderate depths high:
- v4d: the 30B's depth term at 4,096 was 52 % high;
- q9b: priced at 544 it was 17-26 % high.

Here each CPU board with a measured curve (the laptop, the Pi 5: lc4charb) gets law(d) = a (1 + d / d1), ns per query-head
dimension per block per token of depth at depth d. The fit is by least squares of law(d) = (T(d) - T(0)) / (blocks x n_head x
head_dim x d) on d. The table keeps each board's law in "ns_per_qhead_dim" (a, the slope at small depths) and adds
"bend_d1"; planner_inputs scales a board's law by (1 + depth / bend_d1) when the table carries it. The Pi 4 (no curve
measured) keeps its straight law; the Jetson keeps its flash-attention table (lc4n: flat with depth on the GPU).
Writes the table and prints the fit and its residuals.
"""
import argparse, datetime, json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(ROOT, "testbed", "dataset")
BASE = os.path.join(DS, "attention_cost_20261101.json")
LC4 = "testbed/results/lc4charb_20261024_065032"
LABELS = {"laptop": "lap", "pi5": "pi5"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DS, "attention_cost_20261103.json"))
    a = ap.parse_args()
    base = json.load(open(BASE))
    score = json.load(open(os.path.join(ROOT, LC4, "score.json")))
    reg = json.load(open(os.path.join(ROOT, score["registration"])))
    shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
    out = json.loads(json.dumps(base))
    out["made"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    out["_source"] = ("attention_bend_20261103.py: attention_cost_20261101.json's boards, with the laptop's and the Pi 5's law "
                      "bent, a (1 + d / d1), fitted on %s's depth curves" % LC4)
    fits = {}
    for node, label in LABELS.items():
        b = reg["benches"][label]
        shape = shapes[os.path.basename(b["model_path"]).replace(".gguf", "").replace("q4km", "q4_k_m")] \
            if os.path.basename(b["model_path"]).replace(".gguf", "") not in shapes else shapes[os.path.basename(b["model_path"]).replace(".gguf", "")]
        blocks = sum(k for _, k in b["assign"]) if b.get("assign") else shape["n_layers"]
        qhd = shape["n_head"] * shape["head_dim"]
        T = {int(d): v for d, v in score["benches"][label]["ms_per_token_by_depth"].items()}
        pts = [(d, (T[d] - T[0]) / (blocks * qhd * d) * 1e6) for d in sorted(T) if d > 0]     # ns per qhd per block per token
        md = statistics.fmean(d for d, _ in pts); ml = statistics.fmean(l for _, l in pts)
        s = sum((d - md) * (l - ml) for d, l in pts) / sum((d - md) ** 2 for d, _ in pts)
        a0 = ml - s * md
        d1 = a0 / s
        res = {d: round(100 * (a0 * (1 + d / d1) / l - 1), 2) for d, l in pts}
        fits[node] = {"a_ns": round(a0, 5), "d1": round(d1, 1), "points": [(d, round(l, 5)) for d, l in pts], "residual_pct": res,
                      "straight_v4_ns": base["boards"][node]["ns_per_qhead_dim"], "blocks": blocks, "qhd": qhd}
        e = out["boards"][node]
        e.update({"ns_per_qhead_dim": round(a0, 5), "bend_d1": round(d1, 1),
                  "from": "attention_bend_20261103.py: %s's %s curve (TinyLlama Q4_K_M, %d blocks x %d query-head dims), law(d) = "
                          "a (1 + d / d1) by least squares; the straight V4 law was %s" % (LC4, label, blocks, qhd,
                                                                                         base["boards"][node]["ns_per_qhead_dim"])})
    out["bend_fits"] = fits
    json.dump(out, open(a.out, "w"), indent=1)
    for n, f in fits.items():
        print("%s: a %.5f ns, d1 %.0f tokens (straight V4 %.5f); law at 544 / 2,048 / 4,096 / 8,192: %s; residuals %s" % (
            n, f["a_ns"], f["d1"], f["straight_v4_ns"],
            [round(f["a_ns"] * (1 + d / f["d1"]), 4) for d in (544, 2048, 4096, 8192)], f["residual_pct"]))
    print("wrote %s" % os.path.relpath(a.out, ROOT))


if __name__ == "__main__":
    main()
