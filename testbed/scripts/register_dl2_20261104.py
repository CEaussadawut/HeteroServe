#!/usr/bin/env python3
"""Register dl2 (2026-11-04; the user: "do all four follow-up studies": a depth law per model architecture): the laptop's depth
cost priced per attention shape, from its attention op timed with no model loaded, on three models never benched for depth.

    python3 testbed/scripts/register_dl2_20261104.py [--dry-run]

Why. The planner prices a CPU board's depth cost per query-head dimension (ATTENTION_V5, a (1 + d / d1) ns per query-head
dimension per block per token of depth), fitted on TinyLlama alone (head size 64). dl1 (3 Nov) found Qwen3-30B-A3B's slope 20-28 %
under it (head size 128, 4 KV heads): one price per query dimension cannot carry the head's layout.
The new law (dl2fa, testbed/dataset/attention_cpu_20261104.json): fa_bench_cpu (fa_bench's graph on the CPU backend, -t 4, the
caches flushed between graphs) timed the attention op of every attention shape the cluster serves at 256-8,192 cache cells, two
passes. A decoded token at depth d attends over n_kv = 256 x ceil((d + 32) / 256) cells in a 32-token llama-bench decode; its
depth cost is, per attention layer, us(n_kv(d)) - us(256), read off the table (linear between its cells). Nothing is fitted:
the table is the measurement. A hybrid's attention layers are its every 4th block (hybrid_shapes_20261101.json).
Checked before registering, on curves the table was not built from (reported in the registration, not tested): lc4charb's
TinyLlama (24 Oct) and dl1's Qwen3-30B-A3B and Qwen3-14B (3 Nov), each slope against the table's and V5's.
The test, at depths none of it has seen, three models with three attention shapes no depth curve was measured on:
  D06  Qwen3-0.6B Q8_0, the laptop alone (128 x 16 query heads, 8 KV heads, 28 layers)
  D4   Qwen3-4B Q4_K_M, the laptop alone (128 x 32, 8 KV heads, 36 layers)
  D9   Qwen3.5-9B Q4_K_M, the laptop alone (a hybrid: 256 x 16, 4 KV heads, 8 attention layers of 32)
llama-bench -ngl 0 -t 4 -p 0 -n 32 -d 0,1024,2048,4096 -r 3, order D06 D4 D9, by lc4_20261024.py (its gates; the laptop quiet).
Checked before registering, the table alone does not settle it: it priced TinyLlama within 3 %, but the 30B 18 % and the 14B
14 % high, and its two passes differ by up to 44 % at some cells. What the curves do say: the 30B (4 KV heads) costs 27 % less per
query dimension than V5 and the 14B (8 KV heads, the same head size) does not, so the KV heads matter, not the head size alone.
So four laws are registered against each other, as lc4m did, each from what was measured before this registration:
  V5   the planner's law: a (1 + d / d1) ns per query-head dimension (TinyLlama's curve)
  FA   the per-shape table above (fa_bench_cpu, dl2fa)
  HD   a per-head cost beside the head size: V5 x (hd + c0) / hd x 64 / (64 + c0), c0 fitted so the 30B's measured slope holds
       (TinyLlama's by construction): the 30B's head size alone explains its discount
  KV3  three terms per layer, in V5's units: n_head (c0 + c1 hd) + c2 n_kv_head hd, fitted exactly on TinyLlama's, the 30B's and
       the 14B's measured slopes (the 14B's three points are noisy: its 512 point sits above its 1,024 point)
Hypotheses (lc4's slope_near, each law and bench): DL2-<law>-<bench>, the bench's slope (ms per 1,000 tokens of depth, least
squares over its depths) within 10 % of the law's. Read together: the law that holds on all three benches, if one does, is the
laptop's depth law per attention shape.
Writes testbed/dataset/dl2_predictions_20261104.json, read-only; refuses to overwrite it.
"""
import argparse, bisect, datetime, hashlib, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "testbed", "dataset", "dl2_predictions_20261104.json")
TABLE = "testbed/dataset/attention_cpu_20261104.json"
V5 = "testbed/dataset/attention_cost_20261103.json"
DEPTHS = (0, 1024, 2048, 4096)
ARGS = "-ngl 0 -t 4 -p 0 -n 32 -d %s -r 3" % ",".join(str(d) for d in DEPTHS)
# label: (model, laptop path, sha256 (testbed/models.md), head_dim, n_head, n_head_kv, attention layers, blocks)
BENCHES = {
    "D06": ("qwen3-0.6b", "~/.cache/heteroserve-models/Qwen3-0.6B-Q8_0.gguf",
            "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031", 128, 16, 8, 28, 28),
    "D4": ("qwen3-4b", "~/.cache/heteroserve-models/Qwen3-4B-Q4_K_M.gguf",
           "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5", 128, 32, 8, 36, 36),
    "D9": ("qwen35_9b", "~/models/qwen3.5-9b-q4_k_m.gguf",
           "d784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43", 256, 16, 4, 8, 32),
}
# the curves measured before (the check, not the test): label, its score file and bench, head_dim, n_head, n_head_kv, layers
CHECKS = {
    "tinyllama (lc4charb lap)": ("testbed/results/lc4charb_20261024_065032/score.json", "lap", 64, 32, 4, 22),
    "qwen3-30b-a3b (dl1 A)": ("testbed/results/dl1_20261103_214735/score.json", "A", 128, 32, 4, 48),
    "qwen3-14b (dl1 C)": ("testbed/results/dl1_20261103_214735/score.json", "C", 128, 40, 8, 40),
}


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def slope(pts):
    md = statistics.fmean(d for d, _ in pts); mv = statistics.fmean(v for _, v in pts)
    return 1000 * sum((d - md) * (v - mv) for d, v in pts) / sum((d - md) ** 2 for d, _ in pts)


def us_at(curve, n):
    """The table's time per op at n cells, linear between its cells (beyond the last, the last two cells' line)."""
    xs = sorted(curve)
    i = min(max(bisect.bisect_left(xs, n), 1), len(xs) - 1)
    x0, x1 = xs[i - 1], xs[i]
    return curve[x0] + (curve[x1] - curve[x0]) * (n - x0) / (x1 - x0)


def n_kv(d):
    return 256 * -(-(d + 32) // 256)


def table_ms(curve, layers, d):
    return layers * (us_at(curve, n_kv(d)) - us_at(curve, 256)) / 1000.0


def v5_ms(law, n_head, hd, layers, d):
    return d * n_head * hd * layers * law["ns_per_qhead_dim"] * (1 + d / law["bend_d1"]) * 1e-6


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    tab = json.load(open(os.path.join(ROOT, TABLE)))
    law = json.load(open(os.path.join(ROOT, V5)))["boards"]["laptop"]
    curve = lambda hd, nh, nkv: {int(k): v for k, v in tab["shapes"]["%dx%dx%d" % (hd, nh, nkv)]["us_per_op"].items()}
    checks = {}
    for name, (f, label, hd, nh, nkv, layers) in CHECKS.items():
        T = {int(k): v for k, v in json.load(open(os.path.join(ROOT, f)))["benches"][label]["ms_per_token_by_depth"].items()}
        meas = slope(sorted(T.items()))
        ds = sorted(T)
        pt = slope([(d, table_ms(curve(hd, nh, nkv), layers, d)) for d in ds])
        pv = slope([(d, v5_ms(law, nh, hd, layers, d)) for d in ds])
        checks[name] = {"depths": ds, "measured_slope": round(meas, 2), "table_slope": round(pt, 2), "v5_slope": round(pv, 2),
                        "table_err_pct": round(100 * (meas / pt - 1), 1), "v5_err_pct": round(100 * (meas / pv - 1), 1)}
    # HD and KV3, fitted on the measured slopes in V5's units: r = measured / V5 per model, c = r x n_head x hd per layer
    shape_of = {name: (nh, hd, nkv) for name, (_, _, hd, nh, nkv, _) in CHECKS.items()}
    c = {name: (checks[name]["measured_slope"] / checks[name]["v5_slope"]) * shape_of[name][0] * shape_of[name][1] for name in CHECKS}
    tl, m30, m14 = "tinyllama (lc4charb lap)", "qwen3-30b-a3b (dl1 A)", "qwen3-14b (dl1 C)"
    r30 = checks[m30]["measured_slope"] / checks[m30]["v5_slope"]           # HD: (128 + c0) / 128 x 64 / (64 + c0) = r30
    hd_c0 = (128 * 64 - r30 * 128 * 64) / (r30 * 128 - 64)
    rows = [[nh, nh * hd, nkv * hd] for nh, hd, nkv in (shape_of[tl], shape_of[m30], shape_of[m14])]
    rhs = [c[tl], c[m30], c[m14]]
    det = lambda M: (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                     + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    D = det(rows)
    kv3 = [det([[rhs[i] if j == k else rows[i][j] for j in range(3)] for i in range(3)]) / D for k in range(3)]   # Cramer
    fits = {"HD": {"c0": round(hd_c0, 3), "from": [tl, m30]},
            "KV3": {"c0": round(kv3[0], 4), "c1": round(kv3[1], 5), "c2": round(kv3[2], 5), "from": [tl, m30, m14],
                    "units": "per layer per token of depth, in V5's ns-per-query-head-dimension units"}}
    factor = {"HD": lambda nh, hd, nkv: (hd + hd_c0) / hd * 64 / (64 + hd_c0),
              "KV3": lambda nh, hd, nkv: (nh * kv3[0] + nh * hd * kv3[1] + nkv * hd * kv3[2]) / (nh * hd)}
    benches, hyps = {}, {}
    for label, (model, path, msha, hd, nh, nkv, layers, blocks) in BENCHES.items():
        inc_t = {str(d): round(table_ms(curve(hd, nh, nkv), layers, d), 3) for d in DEPTHS}
        inc_v = {str(d): round(v5_ms(law, nh, hd, layers, d), 3) for d in DEPTHS}
        st = slope([(int(d), v) for d, v in inc_t.items()]); sv = slope([(int(d), v) for d, v in inc_v.items()])
        laws = {"V5": sv, "FA": st, "HD": sv * factor["HD"](nh, hd, nkv), "KV3": sv * factor["KV3"](nh, hd, nkv)}
        benches[label] = {"host": "laptop", "model": model, "model_path": path, "model_sha256": msha, "args": ARGS,
                          "assign": [["laptop", blocks]], "head_at": None, "servers": [],
                          "shape": {"head_dim": hd, "n_head": nh, "n_head_kv": nkv, "attention_layers": layers, "blocks": blocks},
                          "predicted": {"slope_ms_per_1k": round(st, 3), "law": "FA", "depth_cost_ms_by_depth": inc_t},
                          "predicted_v5": {"depth_cost_ms_by_depth": inc_v, "slope_ms_per_1k": round(sv, 3)},
                          "laws_slope_ms_per_1k": {k: round(v, 3) for k, v in laws.items()}}
        for k, v in laws.items():
            hyps["DL2-%s-%s" % (k, label)] = {"type": "slope_near", "label": label, "value": round(v, 3), "tol": 0.10,
                                              "claim": "%s's slope within 10 %% of law %s's %.2f ms per 1,000 tokens" % (label, k, v)}
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_dl2_20261104.py", "runner": "testbed/scripts/lc4_20261024.py", "out_prefix": "dl2",
           "what": "four depth laws for the laptop (V5, the per-shape table FA, HD, KV3) on three models never benched for depth",
           "follows": "dl1 (testbed/dataset/dl1_predictions_20261103.json: V5 20-28 % high on the 30B MoE's slope)",
           "cold_c": {"pi5": 50, "pi4": 55}, "table": TABLE, "v5": V5, "n_kv_rule": "256 x ceil((d + 32) / 256)",
           "checks_before_registering": checks, "fits": fits, "benches": benches, "order": [["D06"], ["D4"], ["D9"]],
           "hypotheses": hyps,
           "conditions": "Zen may stay open (QUIET_ALLOW_APPS=zen, the owner, 4 Nov); aunsync.timer runs (every 16 min, ~20 s)",
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/lc4_20261024.py", "testbed/scripts/register_dl2_20261104.py", "testbed/scripts/dl2_fa_cpu_20261104.py",
               "testbed/scripts/fa_bench_cpu_20261104.cpp", TABLE, V5, "testbed/dataset/hybrid_shapes_20261101.json",
               "testbed/dataset/model_shapes.json", "ThinkPad/config/cluster.json", "testbed/scripts/throttle_poll.sh",
               "testbed/scripts/jetson_power_guard.sh")}}
    for n, c in checks.items():
        print("check %-26s measured %7.2f, table %7.2f (%+.1f %%), V5 %7.2f (%+.1f %%)" % (
            n, c["measured_slope"], c["table_slope"], c["table_err_pct"], c["v5_slope"], c["v5_err_pct"]))
    print("fits: %s" % fits)
    for l, b in benches.items():
        print("%s %s: slopes by law %s" % (l, b["model"], b["laws_slope_ms_per_1k"]))
    if a.dry_run:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))


if __name__ == "__main__":
    main()
