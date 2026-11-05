#!/usr/bin/env python3
"""Register lc4n (2026-10-28; item 2 of the improvements left, "a long-context cost law for the Jetson"), before it runs.

    python3 testbed/scripts/register_lc4n_20261028.py --fa testbed/results/lc4nfa_<stamp>/fa.json [--dry-run]

Three registered tests (lc4charb, lc4j-lc4m) found no single law in a model's shape for the Jetson's decode slope against
context depth. The code says why. The Jetson decodes on Vulkan with flash attention on (auto), and ggml-vulkan at the pin
picks the attention kernel's layout per call from the shape and the cache's length (ggml_vk_flash_attn,
get_fa_tuning_params_scalar): its KV heads, the query heads per KV head, the head size, and whether the cache holds 1,024
cells. lc4nx (the Vulkan backend's own per-op timer, testbed/results/lc4nx_20261028_080414; exploration, not registered)
found every op of a decoded token constant in the cache's length except that attention op, on all three models the Jetson
holds alone. So the law proposed here is not a formula in the shape: it is the attention op itself, timed with no model
loaded by fa_bench (fa_bench_20261028.cpp: one graph of the model's layer count of chained FLASH_ATTN_EXT ops, called as
llama.cpp calls it; lc4n_fa_20261028.py ran it on the Jetson before this registration, --fa). If it holds, the planner can
price the Jetson's depth cost for any attention shape in seconds, including the large models' layers the Jetson holds in a
ring, which no single-board bench can load.
llama-bench sizes the cache to the depth plus the 32 generated tokens, padded to 256 cells, so every timed token at depth d
attends over n_kv = d + 256 cells (d a multiple of 128 up to the pad). The benches (Jetson alone, Vulkan, F16 cache, flash
attention auto as every Jetson figure, -p 0 -n 32 -r 3; lc4_20261024.py run: its gates, the guard, the clocks locked):
  tq  TinyLlama Q4_K_M   -d 0,256,384,512,768,1024,2048,4096
  t8  TinyLlama Q8_0     -d 0,256,384,512,768,1024,2048,4096   (the same attention; weights 1.75x the bytes)
  q6  Qwen3-0.6B Q8_0    -d 0,256,384,512,768,1024,2048,4096
  q4  Qwen3-4B Q4_K_M    -ub 128 -d 0,256,384,512,768          (its memory limit: lc4m's smokes collapsed at 1,536)
Hypotheses (registered here, scored by score_lc4n_20261028.py; lc4_20261024.py's own scorer gets none and reports the
benches only):
  J5-FA      the slope over the long range, depths 1,024-4,096 (tq, t8, q6), and over 256-768 (q4), each within 10 % of
             fa_bench's slope over the same cache lengths
  J5-SWITCH  TinyLlama (tq and t8) decodes FASTER at depth 768 than at 512, as the code's path change predicts (fa_bench:
             the op costs less at 1,024 cells than at 768), and Qwen3-0.6B (q6) slower (no change of path)
  J5-STAIR   depths 256 and 384 (the same 512 padded cells) within 0.5 % of each other on every bench
  J5-QUANT   t8's long-range slope within 5 % of tq's
  J5-REPEAT  every bench and depth: the three repetitions within 2 % (max / min - 1 of the per-repetition ms per token)
Depth 0 is measured too and stated as found (lc4j-lc4m and lc4nx put an offset there that the attention op does not carry).
Writes testbed/dataset/lc4n_predictions_20261028.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, hashlib, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "testbed", "dataset", "lc4n_predictions_20261028.json")
PAD = 256
BENCHES = {   # label: (fa_bench shape, model path on the Jetson, sha256 (testbed/models.md), extra args, depths)
    "tq": ("tinyllama", "~/models/tinyllama-1.1b-q4km.gguf", "9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0", "",
           [0, 256, 384, 512, 768, 1024, 2048, 4096]),
    "t8": ("tinyllama", "~/models/tinyllama-1.1b-q8_0.gguf", "a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59", "",
           [0, 256, 384, 512, 768, 1024, 2048, 4096]),
    "q6": ("qwen3_0.6b", "~/models/qwen3-0.6b-q8_0.gguf", "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031", "",
           [0, 256, 384, 512, 768, 1024, 2048, 4096]),
    "q4": ("qwen3_4b", "~/models/qwen3-4b-q4_k_m.gguf", "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5", "-ub 128",
           [0, 256, 384, 512, 768]),
}
LONG = {"tq": [1024, 2048, 4096], "t8": [1024, 2048, 4096], "q6": [1024, 2048, 4096], "q4": [256, 512, 768]}


def sha(p):
    h = hashlib.sha256()
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def n_kv(depth, gen=32):
    return -(-(depth + gen) // PAD) * PAD


def slope(points):
    xs, ys = zip(*points)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    fa = json.load(open(a.fa))
    if fa["source_sha256"] != sha(os.path.join(HERE, "fa_bench_20261028.cpp")):
        raise SystemExit("refused: %s was not made by today's fa_bench_20261028.cpp" % a.fa)
    benches, preds = {}, {}
    for label, (shape, path, want, extra, depths) in BENCHES.items():
        g = {int(k): v for k, v in fa["shapes"][shape]["us_per_graph"].items()}
        need = sorted({n_kv(d) for d in depths})
        if not all(k in g for k in need):
            raise SystemExit("refused: fa_bench has no %s at %s" % (shape, [k for k in need if k not in g]))
        ms = {d: g[n_kv(d)] / 1000.0 for d in depths}                       # the op's ms per token at each depth
        p = {"n_kv_by_depth": {str(d): n_kv(d) for d in depths},
             "fa_ms_per_token_by_depth": {str(d): round(v, 4) for d, v in ms.items()},
             "long_slope_ms_per_1k": round(1000 * slope([(d, ms[d]) for d in LONG[label]]), 4),
             "long_depths": LONG[label],
             "delta_768_minus_512_ms": round(ms[768] - ms[512], 4)}
        preds[label] = p
        benches[label] = {"host": "jetson", "model_path": path, "model_sha256": want, "shape": shape,
                          "args": ("-ngl 99 %s -p 0 -n 32 -d %s -r 3" % (extra, ",".join(str(d) for d in depths))).replace("  ", " "),
                          "predicted": p}
    sw = {k: preds[k]["delta_768_minus_512_ms"] for k in ("tq", "t8", "q6")}
    if not (sw["tq"] < 0 and sw["t8"] < 0 and sw["q6"] > 0):
        raise SystemExit("refused: fa_bench does not show the switch the code predicts: %s" % sw)
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_lc4n_20261028.py", "runner": "testbed/scripts/lc4_20261024.py",
           "scorer": "testbed/scripts/score_lc4n_20261028.py", "out_prefix": "lc4n",
           "what": "the Jetson's long-context cost is its flash-attention op's: whole-model decode against depth on three models "
                   "(four files), predicted from fa_bench, the op timed alone with no model loaded",
           "cold_c": {"pi5": 50, "pi4": 55}, "pad": PAD, "gen": 32,
           "exploration": "testbed/results/lc4nx_20261028_080414 (GGML_VK_PERF_LOGGER; not registered, not reported as figures)",
           "fa_bench": {"file": os.path.relpath(os.path.abspath(a.fa), ROOT), "source": "testbed/scripts/fa_bench_20261028.cpp",
                        "source_sha256": fa["source_sha256"], "stamp": fa["stamp"],
                        "v1_not_used": "testbed/results/lc4nfa_20261028_080922 (fa_bench v1, sha256 138b5dd4...: the ops unchained, "
                                       "so ggml-vulkan overlapped them; Qwen3-0.6B's slope came 4 % under the in-model timer's. v2 "
                                       "chains them as a model's layers are; only v2's table is used here)"},
           "benches": benches, "order": [["tq"], ["t8"], ["q6"], ["q4"]],
           "hypotheses": {},
           "hypotheses_lc4n": {
               "J5-FA": {"tol": 0.10, "claim": "each bench's slope over its long range (tq, t8, q6: depths 1,024-4,096; q4: 256-768) "
                                               "within 10 % of fa_bench's slope over the same cache lengths"},
               "J5-SWITCH": {"claim": "TinyLlama (tq, t8) decodes faster at depth 768 than at 512; Qwen3-0.6B (q6) slower"},
               "J5-STAIR": {"tol": 0.005, "claim": "depths 256 and 384 (both 512 padded cells) within 0.5 % on every bench"},
               "J5-QUANT": {"tol": 0.05, "claim": "t8's long-range slope within 5 % of tq's"},
               "J5-REPEAT": {"tol": 0.02, "claim": "every bench and depth: the three repetitions within 2 %"}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/lc4_20261024.py", "testbed/scripts/register_lc4n_20261028.py", "testbed/scripts/score_lc4n_20261028.py",
               "testbed/scripts/fa_bench_20261028.cpp", "testbed/scripts/lc4n_fa_20261028.py", "testbed/scripts/jetson_power_guard.sh",
               "testbed/scripts/throttle_poll.sh", os.path.relpath(os.path.abspath(a.fa), ROOT), "ThinkPad/config/cluster.json")}}
    if a.dry_run:
        print(json.dumps(preds, indent=1)); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(OUT)[:16]))
    for k, p in preds.items():
        print("%s: long slope %.3f ms per 1,000 tokens; 768 - 512: %+.3f ms" % (k, p["long_slope_ms_per_1k"], p["delta_768_minus_512_ms"]))


if __name__ == "__main__":
    main()
