#!/usr/bin/env python3
"""Register dl3 (2026-11-04; the user: "Time the Pis' CPUs per attention shape, so the per-shape depth table (ATTENTION_V6)
covers more than the laptop"): each Pi's depth cost priced per attention shape, from its attention op timed with no model loaded,
on models whose depth curves that board never measured.

    python3 testbed/scripts/register_dl3_20261104.py [--dry-run]

Why. dl2 found the laptop's depth cost follows its attention op per shape (FA held on three new models, 0.6-4.9 %), which one
price per query-head dimension (V5) could not carry. The Pis are priced by V5's per-dimension laws: the Pi 5's bent law fitted on
its TinyLlama curve (lc4charb, 24 Oct), the Pi 4's straight law from Qwen3-14B on the Pi 4 + Jetson (10 Oct).
The new law (dl3fa, testbed/dataset/attention_cpu_pi_20261104.json): fa_bench_cpu built on each Pi against its pinned tree, dl2fa's
method exactly (every attention shape the cluster serves, 256-8,192 cache cells, -t 4, the median of 15 graphs, two passes). A
decoded token at depth d attends over n_kv = 256 x ceil((d + 32) / 256) cells in a 32-token llama-bench decode; its depth cost
is, per attention layer, us(n_kv(d)) - us(256), read off the board's table (linear between its cells). Nothing is fitted.
Checked before registering (reported, not tested): lc4charb's Pi 5 TinyLlama curve (24 Oct) against the Pi 5's table; V5 was
fitted on that curve. The Pi 4 has no four-node-era depth curve.
The test, at depths none of it has seen, on models whose depth curve the board never measured:
  P5Q06  Qwen3-0.6B Q8_0 on the Pi 5 alone (128 x 16 query heads, 8 KV heads, 28 layers), -d 0,1024,2048,4096
  P5Q9   Qwen3.5-9B Q4_K_M on the Pi 5 alone (a hybrid: 256 x 16, 4 KV heads, 8 attention layers of 32), -d 0,512,1024,2048
  P4Q06  Qwen3-0.6B Q8_0 on the Pi 4 alone, -d 0,1024,2048,4096
  P4TL   TinyLlama Q4_K_M on the Pi 4 alone (64 x 32, 4 KV heads, 22 layers), -d 0,1024,2048,4096
llama-bench -t 4 -p 0 -n 32 -r 3 on the board, by lc4_20261024.py (its gates: every node idle at the pin, the model's sha256 on its
host, the Pis cool, a 1 Hz throttle poll on each Pi), in two pairs, the Pis at once: P5Q06 with P4Q06, then P5Q9 with P4TL.
Two laws per bench:
  FA   the board's per-shape table above
  V5   the planner's law for that board today (attention_cost_20261103.json: a d1-bent price per query-head dimension on the Pi 5,
       a straight one on the Pi 4; a hybrid's attention layers only)
Hypotheses (lc4's slope_near): DL3-<law>-<bench>, the bench's slope (ms per 1,000 tokens of depth, least squares over its depths)
within 10 % of the law's. Read together: whether the per-shape table, which held on the laptop, holds on both Pis.
Writes testbed/dataset/dl3_predictions_20261104.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import register_dl2_20261104 as R2                          # noqa: E402  (slope, us_at, n_kv, table_ms, v5_ms, sha)

OUT = os.path.join(ROOT, "testbed", "dataset", "dl3_predictions_20261104.json")
TABLE = "testbed/dataset/attention_cpu_pi_20261104.json"
V5 = "testbed/dataset/attention_cost_20261103.json"
Q06 = ("qwen3-0.6b", "~/models/qwen3-0.6b-q8_0.gguf", "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031")
Q9 = ("qwen35_9b", "~/models/qwen3.5-9b-q4_k_m.gguf", "d784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43")
TL = ("tinyllama", "~/models/tinyllama-1.1b-q4km.gguf", "9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0")
# label: (board, model, depths, head_dim, n_head, n_head_kv, attention layers, blocks)
BENCHES = {
    "P5Q06": ("pi5", Q06, (0, 1024, 2048, 4096), 128, 16, 8, 28, 28),
    "P5Q9": ("pi5", Q9, (0, 512, 1024, 2048), 256, 16, 4, 8, 32),
    "P4Q06": ("pi4", Q06, (0, 1024, 2048, 4096), 128, 16, 8, 28, 28),
    "P4TL": ("pi4", TL, (0, 1024, 2048, 4096), 64, 32, 4, 22, 22),
}
CHECKS = {"tinyllama on the Pi 5 (lc4charb pi5)": ("testbed/results/lc4charb_20261024_065032/score.json", "pi5", "pi5", 64, 32, 4, 22)}


def v5_board_ms(law, n_head, hd, layers, d):
    """V5's price on a board: a bent law when the board has bend_d1 (the Pi 5), else V4's straight one (the Pi 4)."""
    bend = (1 + d / law["bend_d1"]) if law.get("bend_d1") else 1.0
    return d * n_head * hd * layers * law["ns_per_qhead_dim"] * bend * 1e-6


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    tab = json.load(open(os.path.join(ROOT, TABLE)))["boards"]
    laws = json.load(open(os.path.join(ROOT, V5)))["boards"]
    curve = lambda board, hd, nh, nkv: {int(k): v for k, v in tab[board]["shapes"]["%dx%dx%d" % (hd, nh, nkv)]["us_per_op"].items()}
    checks = {}
    for name, (f, label, board, hd, nh, nkv, layers) in CHECKS.items():
        T = {int(k): v for k, v in json.load(open(os.path.join(ROOT, f)))["benches"][label]["ms_per_token_by_depth"].items()}
        ds = sorted(T)
        meas = R2.slope(sorted(T.items()))
        pt = R2.slope([(d, R2.table_ms(curve(board, hd, nh, nkv), layers, d)) for d in ds])
        pv = R2.slope([(d, v5_board_ms(laws[board], nh, hd, layers, d)) for d in ds])
        checks[name] = {"board": board, "depths": ds, "measured_slope": round(meas, 2), "table_slope": round(pt, 2),
                        "v5_slope": round(pv, 2), "table_err_pct": round(100 * (meas / pt - 1), 1),
                        "v5_err_pct": round(100 * (meas / pv - 1), 1)}
    benches, hyps = {}, {}
    for label, (board, (model, path, msha), depths, hd, nh, nkv, layers, blocks) in BENCHES.items():
        inc_t = {str(d): round(R2.table_ms(curve(board, hd, nh, nkv), layers, d), 3) for d in depths}
        inc_v = {str(d): round(v5_board_ms(laws[board], nh, hd, layers, d), 3) for d in depths}
        st = R2.slope([(int(d), v) for d, v in inc_t.items()]); sv = R2.slope([(int(d), v) for d, v in inc_v.items()])
        args = "-t 4 -p 0 -n 32 -d %s -r 3" % ",".join(str(d) for d in depths)
        benches[label] = {"host": board, "model": model, "model_path": path, "model_sha256": msha, "args": args,
                          "assign": [[board, blocks]], "head_at": None, "servers": [],
                          "shape": {"head_dim": hd, "n_head": nh, "n_head_kv": nkv, "attention_layers": layers, "blocks": blocks},
                          "predicted": {"slope_ms_per_1k": round(st, 3), "law": "FA", "depth_cost_ms_by_depth": inc_t},
                          "predicted_v5": {"depth_cost_ms_by_depth": inc_v, "slope_ms_per_1k": round(sv, 3)},
                          "laws_slope_ms_per_1k": {"FA": round(st, 3), "V5": round(sv, 3)}}
        for k, v in (("FA", st), ("V5", sv)):
            hyps["DL3-%s-%s" % (k, label)] = {"type": "slope_near", "label": label, "value": round(v, 3), "tol": 0.10,
                                              "claim": "%s's slope within 10 %% of law %s's %.2f ms per 1,000 tokens" % (label, k, v)}
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_dl3_20261104.py", "runner": "testbed/scripts/lc4_20261024.py", "out_prefix": "dl3",
           "what": "the Pis' depth cost: each board's per-shape attention table (FA) against the planner's law for it (V5), on "
                   "models whose depth curve the board never measured",
           "follows": "dl2 (testbed/dataset/dl2_predictions_20261104.json: the laptop's per-shape table held on three models)",
           "cold_c": {"pi5": 50, "pi4": 55}, "table": TABLE, "v5": V5, "n_kv_rule": "256 x ceil((d + 32) / 256)",
           "checks_before_registering": checks, "benches": benches, "order": [["P5Q06", "P4Q06"], ["P5Q9", "P4TL"]],
           "hypotheses": hyps,
           "conditions": "Zen may stay open (QUIET_ALLOW_APPS=zen, the owner, 4 Nov); aunsync.timer runs (every 16 min, ~20 s); the "
                         "strip restarted at about 22:05 and every board booted at 22:06 (not by this study); Qwen3-0.6B copied to "
                         "the Pi 5 from the laptop's verified file for P5Q06",
           "inputs_sha256": {f: R2.sha(f) for f in (
               "testbed/scripts/lc4_20261024.py", "testbed/scripts/register_dl3_20261104.py",
               "testbed/scripts/register_dl2_20261104.py", "testbed/scripts/dl3_fa_pi_20261104.py",
               "testbed/scripts/dl2_fa_cpu_20261104.py", "testbed/scripts/fa_bench_cpu_20261104.cpp", TABLE, V5,
               "testbed/dataset/hybrid_shapes_20261101.json", "testbed/dataset/model_shapes.json", "ThinkPad/config/cluster.json",
               "testbed/scripts/throttle_poll.sh")}}
    for n, c in checks.items():
        print("check %-38s measured %7.2f, table %7.2f (%+.1f %%), V5 %7.2f (%+.1f %%)" % (
            n, c["measured_slope"], c["table_slope"], c["table_err_pct"], c["v5_slope"], c["v5_err_pct"]))
    for l, b in benches.items():
        print("%-6s %-4s %-11s depths %s: slopes FA %.2f, V5 %.2f ms per 1,000" % (
            l, b["host"], b["model"], b["args"].split("-d ")[1].split()[0], b["laws_slope_ms_per_1k"]["FA"],
            b["laws_slope_ms_per_1k"]["V5"]))
    if a.dry_run:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), R2.sha(os.path.relpath(OUT, ROOT))[:16]))


if __name__ == "__main__":
    main()
