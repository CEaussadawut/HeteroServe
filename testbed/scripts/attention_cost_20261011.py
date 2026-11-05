#!/usr/bin/env python3
"""Build the planner's measured attention costs and the laptop's worker row (2026-10-11), from scored runs and logs only.
Writes, read-only:
    testbed/dataset/attention_cost_20261011.json   per board: ms per block per token of context depth, by law
    testbed/dataset/laptop_worker_20261011.json    the laptop (cluster.json t580) as an RPC worker: TinyLlama Q4_K_M rows

    python3 testbed/scripts/attention_cost_20261011.py [--dry-run]

The laws (attn_ms() in predict_pool.py turns them into ms per block per token of depth for a model shape):
- pi4, "query_heads": n_head x head_dim x ns. From the 14B itself, both depths hot (kv14b,
  testbed/results/kv14_20261010_213514/score.json: 265.56 ms per 1,000 tokens on Pi 4 27 + Jetson 13). The Jetson's own
  share is taken off first, at its law below. The 0.6B cell alone (kv_depth_20261010_193538) gives 5 % more, which is
  recorded beside it.
- jetson, "kv_bytes": kv_bytes_per_layer_per_token x ns. From the 0.6B (kv_depth_20261010_193538). Its scaling with query
  heads is not measured; at 13 blocks its share of the 14B's slope is 1.4 %.
- laptop, "query_heads", by analogy with the Pi 4's CPU. From the 0.6B as an RPC worker on loopback
  (testbed/results/laptop_worker_20261010_232946, three runs, spread 24 %). The 14B check on the laptop is recorded but
  not used: its d = 0 runs alone spread 502–687 ms, because the laptop is also the user's desktop.
The laptop's worker row: TinyLlama Q4_K_M, pp512 / tg128, the mean of three invocations (bench_worker_role.sh's protocol,
run locally). At the pin the x86 CPU backend repacks Q4_K for AVX2 as a host and never as a worker, so its host rows
do not price it as a worker.
"""
import glob, hashlib, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(ROOT, "testbed", "dataset")
OUT_ATTN = os.path.join(DS, "attention_cost_20261011.json")
OUT_LAP = os.path.join(DS, "laptop_worker_20261011.json")
KVD = os.path.join(ROOT, "testbed", "results", "kv_depth_20261010_193538", "score.json")
KV14B = os.path.join(ROOT, "testbed", "results", "kv14_20261010_213514", "score.json")
LAP = os.path.join(ROOT, "testbed", "results", "laptop_worker_20261010_232946")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rows(tag):
    out = []
    for inv in (1, 2, 3):
        p = os.path.join(LAP, "%s.%d.log" % (tag, inv))
        out.append([json.loads(l) for l in open(p, errors="replace") if l.startswith("{")])
    return out


def slope(pts):
    n = len(pts)
    mx, my = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
    return 1000 * sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)


def build():
    shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
    q14 = shapes["qwen3-14b-q4_k_m"]
    sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
    from control import gguf
    q06 = gguf.shape(os.path.expanduser("~/.cache/heteroserve-models/Qwen3-0.6B-Q8_0.gguf"))
    kvd, kv14b = json.load(open(KVD)), json.load(open(KV14B))
    per06 = 1000.0 * q06["n_layers"]                                  # ms per 1,000 tokens -> per block per token
    jet = kvd["boards"]["jetson"]["slope_ms_per_1k_mean"] / per06 / q06["kv_bytes_per_layer_per_token"] * 1e6   # ns per KV byte
    pi4_06 = kvd["boards"]["pi4"]["slope_ms_per_1k_mean"] / per06 / (q06["n_head"] * q06["head_dim"]) * 1e6
    jet14 = 13 * q14["kv_bytes_per_layer_per_token"] * jet * 1e-6 * 1000              # the Jetson's share, ms per 1,000
    pi4 = (kv14b["slope_ms_per_1k_mean"] - jet14) / (27 * 1000.0) / (q14["n_head"] * q14["head_dim"]) * 1e6
    lap_runs = [slope(sorted((r["n_depth"], 1000.0 / r["avg_ts"]) for r in inv)) for inv in rows("q06_depth")]
    lap = statistics.fmean(lap_runs) / per06 / (q06["n_head"] * q06["head_dim"]) * 1e6
    lap14 = [{str(r["n_depth"]): round(1000.0 / r["avg_ts"], 1) for r in inv} for inv in rows("q14_depth")]
    tiny = rows("tiny_q4km")
    tg = [1000.0 / [r for r in inv if r["n_gen"]][0]["avg_ts"] for inv in tiny]
    pp = [1000.0 / [r for r in inv if r["n_prompt"]][0]["avg_ts"] for inv in tiny]
    src = {os.path.relpath(p, ROOT): sha(p) for p in [KVD, KV14B] + sorted(glob.glob(os.path.join(LAP, "*.log")))}
    attn = {
        "_source": "testbed/scripts/attention_cost_20261011.py; units: ns, turned into ms per block per token of context depth "
                   "by predict_pool.attn_ms()",
        "boards": {
            "pi4": {"law": "query_heads", "ns_per_qhead_dim": round(pi4, 5),
                    "from": "kv14b: Qwen3-14B Pi 4 27 + Jetson 13, both depths hot, 265.56 ms per 1,000 tokens, less the "
                            "Jetson's share (%.2f ms) at its law" % jet14,
                    "check_from_0_6b": round(pi4_06, 5)},
            "jetson": {"law": "kv_bytes", "ns_per_kv_byte": round(jet, 6),
                       "from": "kv_depth: Qwen3-0.6B whole on the Jetson (Vulkan), 2 query heads per KV head; its scaling "
                               "with query heads is not measured"},
            "laptop": {"law": "query_heads", "ns_per_qhead_dim": round(lap, 5),
                       "from": "laptop_worker: Qwen3-0.6B as an RPC worker on loopback, three runs; the law by analogy "
                               "with the Pi 4's CPU",
                       "slope_ms_per_1k_each": [round(x, 2) for x in lap_runs],
                       "spread_pct": round(100 * (max(lap_runs) - min(lap_runs)) / statistics.fmean(lap_runs), 1),
                       "check_14b_not_used": {"ms_per_token_by_depth_each_run": lap14,
                                              "why": "d = 0 alone spread 502-687 ms: the laptop is also the user's desktop"}}},
        "inputs_sha256": src}
    lapw = {"_source": "testbed/scripts/attention_cost_20261011.py, from testbed/results/laptop_worker_20261010_232946",
            "node": "laptop", "cluster_json_id": "t580", "role": "rpc-worker (ggml-rpc-server -t 4, no repack)",
            "rows": {"Q4_K_M": {"model": "tinyllama-1.1b-q4km", "tg_ms": round(statistics.fmean(tg), 3),
                                "pp_ms": round(statistics.fmean(pp), 3), "tg_ms_each": [round(x, 2) for x in tg],
                                "pp_ms_each": [round(x, 2) for x in pp],
                                "tg_spread_pct": round(100 * (max(tg) - min(tg)) / statistics.fmean(tg), 1)}},
            "inputs_sha256": {k: v for k, v in src.items() if "laptop_worker" in k}}
    return attn, lapw


def main(argv):
    attn, lapw = build()
    print(json.dumps({"attention": attn["boards"], "laptop_worker": lapw["rows"]}, indent=1))
    if "--dry-run" in argv:
        print("(dry run: nothing written)")
        return
    for path, obj in ((OUT_ATTN, attn), (OUT_LAP, lapw)):
        if os.path.exists(path):
            sys.exit("%s exists: never rewritten" % path)
        json.dump(obj, open(path, "w"), indent=1)
        os.chmod(path, 0o444)
        print("wrote %s (%s)" % (os.path.relpath(path, ROOT), sha(path)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
