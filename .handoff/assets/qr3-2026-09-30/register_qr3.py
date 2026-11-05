#!/usr/bin/env python3
"""Register QR3 before any scored request runs (user: "yes demote them, then start QR3"):

    python3 .handoff/assets/qr3-2026-09-30/register_qr3.py [--dry-run]

Writes testbed/dataset/qr3_predictions_20260930.json, read-only, never overwritten. It fixes, before any per-token figure:
the definition of a TPOT sample and of its P95, the plans (the engine's, for a 4096-token serving context, as serve_pool.sh
computes them), what the predictor says of each plan's mean, and the rules the scorer applies as written. Drafted while the
unscored instrument pilot (qr3_tpot.py --tag pilot, the 14B, 2 x 32 tokens) was running; registered (01:31:06) after the pilot had
finished but before any of its output was read.
"""
import dataclasses, datetime, hashlib, json, os, stat, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad", "control"))
import predict_pool as P  # noqa: E402
import gguf as G  # noqa: E402
from control.placement import plan_optimal  # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "qr3_predictions_20260930.json")
CTX = 4096
ORDER = ["qwen35_9b", "nemo12b", "qwen3_14b", "qwen3moe"]
NINE_B = os.path.expanduser("~/.cache/heteroserve-models/Qwen_Qwen3.5-9B-Q4_K_M.gguf")      # as register_recur2.py
NINE_B_SHA = "d784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv):
    dry = "--dry-run" in argv
    if os.path.exists(OUT) and not dry:
        sys.exit("%s exists: a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    inputs = {f: os.path.join(DS, f) for f in ("characterisation.csv", "model_shapes.json", "link.json", "memory_at_rest.json")}
    inputs.update({"cluster.json": os.path.join(ROOT, "ThinkPad", "config", "cluster.json"),
                   "predict_pool.py": os.path.join(ROOT, "testbed", "scripts", "predict_pool.py"),
                   "serve_pool.sh": os.path.join(ROOT, "testbed", "scripts", "serve_pool.sh"),
                   "qr3_tpot.py": os.path.join(ROOT, "testbed", "scripts", "qr3_tpot.py"),
                   "placement.py": os.path.join(ROOT, "ThinkPad", "control", "placement.py")})
    shapes = json.load(open(inputs["model_shapes.json"]))
    pairs = json.load(open(inputs["link.json"]))["pairs"]
    mem = json.load(open(inputs["memory_at_rest.json"]))
    R = P.rows(inputs["characterisation.csv"], "char")
    configs = {}
    for key in ORDER:                                           # serve_pool.sh's planning block, verbatim in effect
        shape, quant, _ = P.MODELS[key]; q = shapes[shape]; host = P.HOST.get(key, "pi5")
        C, _ = P.costs(R, shapes, quant); nodes = [n for n in ("pi5", "pi4", "jetson") if n in C]
        L, missing = P.link_view(pairs, host, nodes); assert not missing, missing
        devs = P.profiles(C, q, mem, L, nodes); mp = dataclasses.replace(P.model_profile(q, shape), context_len=CTX)
        plan = plan_optimal(devs, mp, "decode", host_candidates=[host])
        assign = [(n, plan.layers[n]) for n in plan.order if plan.layers.get(n, 0) > 0]
        m = P.memory(assign, q, mem, devs, mp); dec, pre = P.predict(assign, q, C, L)
        a = P.bench_args(assign, key)["args"].replace(" -rpc ", " --rpc ")
        if " -ts " in a:
            h, ts = a.split(" -ts "); a = h + " -ts " + ts.replace("/", ",")
        c = {"model": shape, "quant": quant, "host": host, "plan": [{"node": n, "blocks": k} for n, k in assign],
             "head_on": assign[-1][0], "server_args": a + " -c %d -np 1 -t 4" % CTX,
             "memory": {n: {k: v[k] for k in ("need_mib", "reserve_mib", "available_mib", "margin_mib")} for n, v in m.items()},
             "predicted": {"decode_ms_per_token": round(dec, 1), "prefill_ms_per_token": round(pre, 1)}}
        if key == "qwen35_9b":                                  # the hybrid: the predictor has no adopted decode term for it
            if sha(NINE_B) != NINE_B_SHA:
                sys.exit("the laptop's 9B is not the pinned bytes")
            md = G.read_metadata(NINE_B)
            md = md[0] if isinstance(md, tuple) else md
            qr = dict(q, full_attention_interval=int(md["qwen35.full_attention_interval"]))   # as register_recur2.py
            th, tw = P.recurrent_terms(inputs["characterisation.csv"], C, qr, shape + ".gguf")
            assert th, "no recurrent terms"
            d2, _ = P.predict(assign, qr, C, L, recur=th, recur_worker=tw)
            c["predicted"]["decode_ms_per_token_with_recur_term"] = round(d2, 1)
            c["predicted"]["recur_term_note"] = ("information only: RECUR's host term, fitted on the Pi 5's own tg16 row of this model, "
                                                 "so for this one-board plan it restates that row; ACCURACY uses the plain figure")
            c["note"] = ("a qwen35 hybrid: the registered rule TRANSFER (2026-09-28) found the predictor 5-14 % optimistic in decode for "
                         "it, and RECUR2's decode term was not adopted, so the plain figure is expected to be low")
        if key == "qwen3moe":
            c["note"] = "the MoE: its warm llama-bench decode on the ctx-80 plan (13/10/25) ran 290.0 ms against 263.3 predicted (-9.2 %)"
        configs[key] = c
    reg = {
        "registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "campaign": "2026-09-30: QR3, TPOT at the 95th percentile for one user, on the engine's serving plan (ctx %d)" % CTX,
        "requirement": "HeteroServe_Proposal_EN.docx: 'TPOT P95 <= 500 ms (MVP) / <= 250 ms (Stretch)'; TPOT measured as decode "
                       "streams tokens back to the client",
        "inputs_sha256": {k: sha(v) for k, v in inputs.items()},
        "instrument": "testbed/scripts/qr3_tpot.py run --model <key> --ctx 4096 --reqs 10 --n 128 --gen 128 (tag 'scored'): "
                      "serve_pool.sh starts the plan; the model's sha256 is checked on its host first; one unscored warm-up "
                      "(512-token prompt, 32 tokens), then 10 requests one after another, each its own 128-token prompt (GPL-3 from "
                      "token 500 x i, then a question), 128 tokens each, greedy, ignore_eos, cache_prompt false, streamed with "
                      "timings_per_token; get_throttled / temperature / MemAvailable on every board between requests; no memory "
                      "probe during scored requests. The configs run in the order listed.",
        "definitions": {
            "tpot_sample": "the gap between two consecutive tokens of one request (the first token is TTFT's): 127 samples a request, "
                           "1,270 a config. An SSE event carrying more than one token gives no sample for the gap it merges; merged "
                           "events are counted and reported.",
            "client": "the laptop's time.perf_counter() at each SSE event's arrival: what a user on the router LAN sees. PRIMARY.",
            "server": "llama-server's timings.predicted_ms per token (stamped after sampling, us resolution); a sample only where "
                      "predicted_n rises by exactly 1. Reported beside the primary, to separate the cluster from the delivery path.",
            "p95": "the 95th percentile of a config's pooled samples, by linear interpolation between order statistics (numpy's "
                   "default, Hyndman-Fan type 7)",
            "ci": "95 % CI of the pooled P95 by bootstrap over whole requests (10,000 resamples, seed 20260930), because tokens of "
                  "one request are not independent",
            "mean_tpot": "the mean of the pooled samples",
        },
        "rules": [
            "QR3-MVP: a config meets the MVP if its pooled client P95 is <= 500 ms. It is reported as 'met' when the whole CI is <= "
            "500, 'met on the point estimate' when only the P95 is, and 'not met' otherwise.",
            "QR3-STRETCH: the same at 250 ms.",
            "PRIMARY: the verdicts are taken on the client samples. If the server samples would give a different verdict, both are "
            "reported.",
            "TAIL: QR3 is decided by the mean decode speed, not by jitter: every config's pooled client P95 is within 10 % above its "
            "mean TPOT (P95 / mean <= 1.10).",
            "DELIVERY: the pooled client and server P95 of every config differ by less than 2 %.",
            "ACCURACY: every config's server mean TPOT is within +-10 % of the predictor's decode figure for its plan (as the engine "
            "and serve_pool.sh use it, with no recurrent term). Expected to hold for the dense models; the 9B hybrid and the MoE carry "
            "the known biases in their notes, and are scored against the rule as written.",
            "PLAN: the plan serve_pool.sh serves must be the registered one (its server args); a config served on another plan is "
            "reported and not scored.",
            "COMPLETE: every scored request must return exactly 128 tokens; a config with a short request is reported, its full "
            "requests still scored.",
        ],
        "predicted_verdicts": {
            "QR3-MVP": "met only by qwen3moe (predicted mean 273.7 ms); the dense models and the 9B are predicted not to meet it "
                       "(their predicted means already exceed 500 ms, except the 9B's plain 466.3, which its known bias puts above)",
            "QR3-STRETCH": "met by none",
        },
        "configs": configs,
    }
    txt = json.dumps(reg, indent=1)
    if dry:
        print(txt); return
    with open(OUT, "w") as f:
        f.write(txt + "\n")
    os.chmod(OUT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print("registered", os.path.relpath(OUT, ROOT), reg["registered"], sha(OUT))
    for k, c in configs.items():
        print("  %-10s %-36s pred decode %s  margins %s" % (k, " + ".join("%s %d" % (p["node"], p["blocks"]) for p in c["plan"]),
                                                           c["predicted"], {n: v["margin_mib"] for n, v in c["memory"].items()}))


if __name__ == "__main__":
    main(sys.argv[1:])
