#!/usr/bin/env python3
"""Register the first measurement of scheduler upgrade 1 (speculative decoding as a placement decision), before it runs.
Writes, read-only, testbed/dataset/spec_pass_predictions_20261010.json.

    python3 testbed/scripts/register_spec_pass_20261010.py [--dry-run]

The question (Documents/HeteroServe_Scheduler_Research.html, upgrade 1): what does one pass of the pooled Qwen3-14B cost
when it must CHECK k tokens at once (a draft's k - 1 tokens plus the one after them), with logits for every one of them?
The speculative speedup on these boards hinges on it, and nothing measured answers it: the predictor has decode (one
token, memory-bound) and prefill at 64 and 512 tokens (compute-bound), and its prefill curve is flat below 64.

The cell: the engine's own plan as served on 2026-10-09 (Pi 4 host 27 blocks + the output head, Jetson 13 blocks on
Vulkan, the weights read into memory), run by run_pool.sh with bench.tool llama-batched-bench: after a shared 8-token
prompt, 8 generation steps, each a batch of k tokens (k parallel sequences, logits for every token), for k = 1..5,
listed three times so the repetitions interleave and drift spreads over every k. T(k) = t_tg / 8: ms per pass.
(llama-bench's pp tests compute only the last token's logits, so they would leave out the head's work for k - 1 tokens:
on this plan the 638 MB output head is on the Pi 4.)

Predictions, from single-board rows only (predict_pool.costs, RTT1, the measured Pi 4 <-> Jetson link). Per stage i:
dec_i = its decode cost (per byte read, blocks and the head where it is), pre_i = its prefill cost per token (per
parameter, the head's parameters per token: every checked token needs logits). Two models bound the truth:
  ADD(k)  = DEC + (k - 1) x sum(pre_i)            each extra token adds its compute; memory and compute never overlap
  ROOF(k) = sum(max(dec_i, k x pre_i)) + link(k)  a roofline per board: weights are read once, compute hides under it
where DEC = predict_pool.predict's decode ms per token for the plan (RTT1) and link(k) the round trip plus k tokens'
hidden states each way. ADD(1) = ROOF(1) = DEC.
Hypotheses, decided before any data:
  SP-K1    T(1) within 5 % of DEC (a one-token pass is a decode step: the method measures what the predictor prices)
  SP-BAND  every T(k), k = 2..5, within [0.95 x ROOF(k), 1.05 x ADD(k)]
  SP-PAYS  for some k in 2..5, T(k) / T(1) < E(k - 1; 0.7) = (1 - 0.7^k) / 0.3, the tokens a check of k - 1 drafted
           tokens yields at 70 % acceptance: with a free drafter, speculative decoding would pay
Exploratory: the overlap fraction phi fitted as T(k) = DEC + (k - 1) phi sum(pre_i); the ceiling speedup
max_k E(k - 1; a) T(1) / T(k) for a = 0.5 .. 0.9 (no drafter cost: the next step measures the drafter).
"""
import datetime, hashlib, json, os, stat, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                    # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "spec_pass_predictions_20261010.json")
KEY, NODES, CTX, CELL = "qwen3_14b", ["pi4", "jetson"], 4096, "sp.q3_14b.pass"
KS = [1, 2, 3, 4, 5]
ROUNDS = 3
ALPHA = 0.7
PROTOCOL = "-t 4 -c 512 -npp 8 -ntg 8 -npl %s -pps --output-format jsonl" % ",".join(str(k) for k in KS * ROUNDS)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def tokens_per_check(k, a):
    """Expected tokens one check of k - 1 drafted tokens yields at per-token acceptance a: the accepted run plus the
    target's own next token, (1 - a^k) / (1 - a)."""
    return (1 - a ** k) / (1 - a)


def pass_models(assign, q, C, links, head_at, rtt_trips=1):
    """{"stages": [...], "dec_ms", "add": {k: ms}, "roof": {k: ms}} for a k-token check of the pipeline."""
    lb, ppl = q["layer_bytes"], q["params_per_layer"]
    stages, start = [], 0
    h = len(assign) - 1 if head_at is None else head_at
    for i, (n, k) in enumerate(assign):
        pb, pp = C[n]["host" if i == 0 else "worker"]
        dec = pb * sum(lb[start:start + k]) + (pb * q["head_bytes"] if i == h else 0.0)
        pre = pp * ppl * k + (pp * q["head_params"] if i == h else 0.0)     # logits for every checked token
        stages.append({"node": n, "blocks": k, "head": i == h, "dec_ms": dec, "pre_ms_per_token": pre})
        start += k
    dec_total, _ = P.predict(assign, q, C, links, head_at=head_at, rtt_trips=rtt_trips)
    link_of = {}
    for k in KS:
        t = 0.0
        for i, (n, _) in enumerate(assign):
            if i > 0:
                bw = links[n]["mbps"] * 1e6 / 8 / 1000
                t += rtt_trips * links[n]["rtt_ms"] + 2 * k * q["activation_bytes"] / bw
        link_of[k] = t
    pre_sum = sum(s["pre_ms_per_token"] for s in stages)
    add = {k: dec_total + (k - 1) * pre_sum for k in KS}
    roof = {k: sum(max(s["dec_ms"], k * s["pre_ms_per_token"]) for s in stages) + link_of[k] for k in KS}
    return {"stages": stages, "dec_ms": dec_total, "pre_sum_ms": pre_sum, "link_ms": link_of, "add": add, "roof": roof}


def build():
    mem = json.load(open(os.path.join(DS, "memory_at_rest.json")))
    r = P.plan_host_choice(KEY, NODES, CTX, mem, rtt_trips=1, load_mode="none")
    assign, head_at = [tuple(x) for x in r["assign"]], r["head_at"]
    if assign != [("pi4", 27), ("jetson", 13)] or head_at != 0:
        sys.exit("refused: the engine now plans %s head_at %s, not the 27 + 13 head-on-host plan served on 2026-10-09"
                 % (assign, head_at))
    m = pass_models(assign, r["q"], r["C"], r["links"], head_at)
    memo = P.memory(r["assign"], r["q"], mem, r["devices"], r["model"], head_at=head_at)
    for n, v in memo.items():
        if v["margin_mib"] < 0:
            sys.exit("refused: %s does not fit at rest: %s" % (n, v))
    b = P.bench_args(assign, KEY, head_at=head_at, load_mode="none")
    if " -rpc " not in b["args"]:
        sys.exit("refused: no -rpc in %r" % b["args"])
    args = b["args"].replace(" -rpc ", " --rpc ")               # the common argument parser's spelling
    E = {k: tokens_per_check(k, ALPHA) for k in KS}
    per_k = {str(k): {"add_ms": round(m["add"][k], 1), "roof_ms": round(m["roof"][k], 1),
                      "add_ratio": round(m["add"][k] / m["add"][1], 3), "roof_ratio": round(m["roof"][k] / m["roof"][1], 3),
                      "band_ms": [round(0.95 * m["roof"][k], 1), round(1.05 * m["add"][k], 1)],
                      "tokens_per_check_at_alpha": round(E[k], 3),
                      "ceiling_speedup_add": round(E[k] * m["add"][1] / m["add"][k], 3),
                      "ceiling_speedup_roof": round(E[k] * m["roof"][1] / m["roof"][k], 3)} for k in KS}
    inputs = {f: os.path.join(DS, f) for f in ("characterisation.csv", "model_shapes.json", "link.json", "memory_at_rest.json")}
    code = {f: os.path.join(HERE, f) for f in ("predict_pool.py", "run_pool.sh", os.path.basename(__file__))}
    code["placement.py"] = os.path.join(ROOT, "ThinkPad", "control", "placement.py")
    reg = {
        "registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "name": "spec_pass_20261010",
        "campaign": "scheduler upgrade 1, step 1: the pooled Qwen3-14B's pass time when checking k = 1..5 tokens",
        "survey": "Documents/HeteroServe_Scheduler_Research.html, upgrade 1 (2026-10-10)",
        "model": KEY, "nodes": NODES, "rtt_trips": 1, "load_mode": "none", "alpha": ALPHA, "ks": KS, "rounds": ROUNDS,
        "inputs_sha256": {**{f: sha(p) for f, p in inputs.items()}, **{f: sha(p) for f, p in code.items()}},
        "run_order": [CELL],
        "configs": {CELL: {
            "model": P.MODELS[KEY][0], "quant": P.MODELS[KEY][1],
            "pipeline": [{"node": n, "blocks": k} for n, k in assign], "head_on": assign[0][0], "head_at": head_at,
            "bench": {"host": b["host"], "tool": "llama-batched-bench", "args": args, "protocol": PROTOCOL},
            "memory": memo,
            "predicted": {"decode_ms": round(m["dec_ms"], 1), "pre_sum_ms_per_token": round(m["pre_sum_ms"], 1),
                          "stages": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items()} for s in m["stages"]],
                          "link_ms": {str(k): round(v, 3) for k, v in m["link_ms"].items()},
                          "per_k": per_k},
            "why": "T(k) decides speculative decoding's gain on these boards; the predictor has no batch-1..5 point"}},
        "hypotheses": [
            {"id": "SP-K1", "claim": "T(1) within 5 %% of the predicted decode step, %.1f ms" % m["dec_ms"]},
            {"id": "SP-BAND", "claim": "every T(k), k = 2..5, within [0.95 x ROOF(k), 1.05 x ADD(k)] (per_k band_ms)"},
            {"id": "SP-PAYS", "claim": "for some k in 2..5, T(k) / T(1) < E(k - 1; %.1f) = (1 - %.1f^k) / %.1f" % (ALPHA, ALPHA, 1 - ALPHA)},
        ],
        "scoring": "T(k) = t_tg / 8 of each llama-batched-bench JSONL line with pl = k; per k the mean of the 3 lines, its "
                   "spread (max - min) / mean reported; a hypothesis lacking a k is INCOMPLETE. The cell's THROTTLE "
                   "verdict and the Jetson's RPC model buffer (13 blocks, within 0.5 MiB) are reported beside the verdicts.",
    }
    return reg


def main(argv):
    if os.path.exists(OUT) and "--dry-run" not in argv:
        sys.exit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    reg = build()
    c = reg["configs"][CELL]
    print("plan %s, head on %s; bench %s %s %s" % (c["pipeline"], c["head_on"], c["bench"]["tool"], c["bench"]["args"],
                                                c["bench"]["protocol"]))
    print("memory %s" % {n: (v["need_mib"], v["available_mib"], v["margin_mib"]) for n, v in c["memory"].items()})
    print("stages %s" % c["predicted"]["stages"])
    print("%-3s %9s %9s %7s %7s %13s %6s %8s %8s" % ("k", "ADD ms", "ROOF ms", "ADD x", "ROOF x", "band ms", "E(.7)", "S add", "S roof"))
    for k, v in c["predicted"]["per_k"].items():
        print("%-3s %9.1f %9.1f %7.3f %7.3f %13s %6.3f %8.3f %8.3f" % (k, v["add_ms"], v["roof_ms"], v["add_ratio"], v["roof_ratio"],
              "%.0f-%.0f" % tuple(v["band_ms"]), v["tokens_per_check_at_alpha"], v["ceiling_speedup_add"], v["ceiling_speedup_roof"]))
    if "--dry-run" in argv:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print("registered %s at %s, sha256 %s (read-only)" % (os.path.relpath(OUT, ROOT), reg["registered"], sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
