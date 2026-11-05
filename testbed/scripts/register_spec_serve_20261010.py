#!/usr/bin/env python3
"""Register scheduler upgrade 1, step 3 -- speculative decoding served on the boards, the drafter placed by prediction --
before it runs. Writes, read-only, testbed/dataset/spec_serve_predictions_20261010.json.

    python3 testbed/scripts/register_spec_serve_20261010.py --accept testbed/results/spec_accept_<stamp> [--dry-run]

The plan is the engine's own for Qwen3-14B on Pi 4 + Jetson (27 + 13, the head on the host), served by serve_pool.sh at a
1,024-token context: at 4,096 the Pi 4 has no room for a draft model beside the target. The predictor, from measured parts
only:
  T(k)    the pass time of a k-token check: step 1 (spec_pass_20261010, llama-batched-bench on this plan)
  tau(d)  tokens per check with d drafted tokens: step 2 (spec_accept_20261010, Qwen3-0.6B on 50 prompts)
  t(b)    one drafter step on board b: its measured per-byte decode cost at Q8_0 (TinyLlama Q8_0 rows) x the bytes a
          Qwen3-0.6B step reads (28 blocks and the tied embedding as the output head), plus, for an RPC board, a round trip
          (pinged from the Pi 4 at registration) and the logits back (n_vocab x 4 bytes at the measured gigabit rate)
  i(b)    the draft reading each request's prompt once before it can draft (llama-server does it inside the generation
          timing): the prompt's tokens (step 2's prompt_n for the same prompts) x its prefill cost per token on b
          (per-parameter Q8_0 rate x its parameters), plus a round trip and the logits for an RPC board
  S(d, b) = T(1) / ((T(d + 1) + d t(b)) / tau(d) + i(b) / (n - 1)), n the tokens per request (the first comes from the
          prompt pass in every arm, so n - 1 passes or checks carry the decode time)
Boards: the Pi 4 (its CPU, beside the target: its memory is charged), the laptop (a ggml-rpc-server on its LAN address);
the Jetson keeps about 440 MiB at this context and a 0.6B draft needs about 800, so it is not a candidate. The engine's choice is the
(d, b) with the highest S. The arms: base (no draft), the choice, and the other board at the same d; run base, the Pi 4
arm, the laptop arm, base again. 8 of step 2's prompts, 64 tokens each, greedy, one warm-up request per session.
Hypotheses, decided before any data (the survey's test): SS-BAR, SS-ACC, SS-ORDER, SS-MOE (spec_serve_20261010.py).
"""
import datetime, hashlib, json, os, re, stat, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                    # noqa: E402
from control import gguf                                    # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "spec_serve_predictions_20261010.json")
STEP1 = os.path.join(ROOT, "testbed", "results", "spec_pass_20261010_011447", "score.json")
ACCEPT_REG = os.path.join(DS, "spec_accept_predictions_20261010.json")
DRAFT_LOCAL = os.path.expanduser("~/.cache/heteroserve-models/Qwen3-0.6B-Q8_0.gguf")
LAPTOP = "192.168.1.108"
LAPTOP_RPC = LAPTOP + ":50053"
CTX, N_PREDICT, D_MAX = 1024, 64, 4
PROMPT_IDS = ["e01", "h02", "c01", "m01", "w01", "t02", "th01", "th03"]
DRAFT_HOST_MIB = {"pi4": 800, "laptop": 160}   # the draft beside the target (-lm none: weights + KV at 1,024 + compute);
                                               # on the laptop, only the host's staging copy of its largest tensor
HEADROOM = 256


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def ping_ms():
    p = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "pi4", "ping -c 20 -i 0.2 -q %s" % LAPTOP],
                       capture_output=True, text=True, timeout=60)
    m = re.search(r"= [\d.]+/([\d.]+)/", p.stdout)
    if not m:
        sys.exit("refused: could not ping the laptop from the Pi 4: %s" % p.stdout[-200:])
    return float(m.group(1))


def build(accept_dir):
    s1 = json.load(open(STEP1))
    T = {int(k): v["mean_ms"] for k, v in s1["per_k"].items()}
    s2p = os.path.join(accept_dir, "score.json")
    s2 = json.load(open(s2p))
    if s2["hypotheses"]["SA-PAY1"]["verdict"] == "INCOMPLETE":
        sys.exit("refused: step 2 is incomplete")
    tau = {int(k): v for k, v in s2["all"]["tau"].items()}
    acc = json.load(open(ACCEPT_REG))
    shp = gguf.shape(DRAFT_LOCAL)
    step_bytes = shp["n_layers"] * shp["bytes_per_layer"] + shp["head_bytes"]
    n_vocab = shp["head_params"] / shp["n_embd"]
    R = P.rows(os.path.join(DS, "characterisation.csv"), "char")
    shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
    tiny = shapes[P.TINY["Q8_0"]]
    units = {b: P.unit(R[(n, "cpu", "Q8_0", "tg")], R[(n, "cpu", "Q8_0", "pp")], tiny) for b, n in (("pi4", "pi4"), ("laptop", "t580"))}
    pb = {b: u[0] for b, u in units.items()}
    pp = {b: u[1] for b, u in units.items()}
    links = json.load(open(os.path.join(DS, "link.json")))["pairs"]
    bw = links["pi4-jetson"]["mbps"] * 1e6 / 8 / 1000            # bytes per ms: the measured gigabit on this router
    rtt = ping_ms()
    t = {"pi4": pb["pi4"] * step_bytes,
         "laptop": pb["laptop"] * step_bytes + rtt + (4 * n_vocab + shp["activation_bytes"]) / bw}
    rows2 = [json.loads(l) for l in open(os.path.join(accept_dir, "run.jsonl"))]
    n_prompt = statistics.fmean(r["timings"]["prompt_n"] for r in rows2 if r["id"] in PROMPT_IDS)
    ing = {"pi4": n_prompt * pp["pi4"] * shp["n_params"],
           "laptop": n_prompt * pp["laptop"] * shp["n_params"] + rtt + (4 * n_vocab) / bw}
    S = {(dd, b): T[1] / ((T[dd + 1] + dd * t[b]) / tau[dd] + ing[b] / (N_PREDICT - 1))
         for dd in range(1, D_MAX + 1) for b in t if dd + 1 in T}
    S_core = {(dd, b): tau[dd] * T[1] / (T[dd + 1] + dd * t[b]) for dd in range(1, D_MAX + 1) for b in t if dd + 1 in T}
    best_d, best_b = max(S, key=S.get)
    other = "laptop" if best_b == "pi4" else "pi4"
    mem = json.load(open(os.path.join(DS, "memory_at_rest.json")))
    r = P.plan_fixed("qwen3_14b", [("pi4", 27), ("jetson", 13)], "host", CTX, mem, rtt_trips=1, load_mode="none")
    m = P.memory(r["assign"], r["q"], mem, r["devices"], r["model"], head_at=r["head_at"])
    for n, v in m.items():
        if v["margin_mib"] < 0:
            sys.exit("refused: %s does not fit at ctx %d: %s" % (n, CTX, v))
    pi4_margin = m["pi4"]["margin_mib"] - DRAFT_HOST_MIB["pi4"]
    if pi4_margin < HEADROOM:
        sys.exit("refused: with the draft beside it, the Pi 4 keeps %d MiB, inside the %d MiB headroom" % (pi4_margin, HEADROOM))
    draft = "-md ~/models/qwen3-0.6b-q8_0.gguf --spec-type draft-simple --spec-draft-n-max %d --spec-draft-n-min 0 --spec-draft-p-min 0"

    def arm(b, dd):
        if b == "pi4":
            return {"board": "pi4", "d": dd, "laptop_rpc": None,
                    "env": {"SERVE_DRAFT": (draft % dd) + " -ngld 0", "DRAFT_HOST_MIB": str(DRAFT_HOST_MIB["pi4"])}}
        return {"board": "laptop", "d": dd, "laptop_rpc": LAPTOP_RPC,
                "env": {"SERVE_DRAFT": (draft % dd) + " -devd RPC1 -ngld 99",     # RPC1: after the Jetson's RPC0 (serve_pool.sh)
                        "DRAFT_HOST_MIB": str(DRAFT_HOST_MIB["laptop"]), "SERVE_RPC_EXTRA": LAPTOP_RPC}}
    arms = {"base": {"board": None, "d": 0, "laptop_rpc": None, "env": {}},
            "%sd%d" % (best_b[:3], best_d): arm(best_b, best_d),
            "%sd%d" % (other[:3], best_d): arm(other, best_d)}
    pi4_arm, lap_arm = "pi4d%d" % best_d, "lapd%d" % best_d
    prompts = [p for p in acc["prompts"] if p["id"] in PROMPT_IDS]
    pred = {a: {"board": arms[a]["board"], "d": best_d, "tau": round(tau[best_d], 4), "draft_step_ms": round(t[arms[a]["board"]], 1),
                "speedup": round(S[(best_d, arms[a]["board"])], 3),
                "draft_ingest_ms_per_request": round(ing[arms[a]["board"]], 1),
                "speedup_without_ingest": round(S_core[(best_d, arms[a]["board"])], 3),
                "ms_per_token_vs_T1": round((T[best_d + 1] + best_d * t[arms[a]["board"]]) / tau[best_d]
                                            + ing[arms[a]["board"]] / (N_PREDICT - 1), 1)} for a in (pi4_arm, lap_arm)}
    return {
        "registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "name": "spec_serve_20261010",
        "campaign": "scheduler upgrade 1, step 3: speculative decoding served on Pi 4 + Jetson, the drafter placed by prediction",
        "survey": "Documents/HeteroServe_Scheduler_Research.html, upgrade 1 (2026-10-10)",
        "model": "qwen3_14b", "ctx": CTX, "place": "pi4:27,jetson:13", "place_head": "host", "n_predict": N_PREDICT,
        "host_url": "http://192.168.1.110:8080", "outlets": {"pi4": 0, "laptop": 1, "jetson": 2},
        "chat_format": acc["chat_format"], "sampling": acc["sampling"], "warmup_prompt": "Say hello in one sentence.",
        "prompts": prompts, "idle_before_s": 120, "idle_after_s": 60,
        "laptop_rpc_server": "/opt/heteroserve/llama.cpp/build/bin/ggml-rpc-server",
        "models_on_host": {"~/models/qwen3-14b-q4_k_m.gguf": "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0",
                           "~/models/qwen3-0.6b-q8_0.gguf": acc["draft"]["sha256"]},
        "arms": arms, "order": ["base", pi4_arm, lap_arm, "base"], "base_arm": "base",
        "engine_choice": "%sd%d" % (best_b[:3], best_d),
        "predicted": pred,
        "model_parts": {"T_ms": T, "tau": {str(k): v for k, v in tau.items()}, "draft_step_bytes": step_bytes,
                        "per_byte_ms_q8": pb, "per_param_ms_q8": pp, "draft_step_ms": {b: round(v, 2) for b, v in t.items()},
                        "mean_prompt_tokens": round(n_prompt, 2), "draft_ingest_ms": {b: round(v, 1) for b, v in ing.items()},
                        "laptop_rtt_ms_from_pi4": rtt, "bw_bytes_per_ms": round(bw, 1), "n_vocab": round(n_vocab),
                        "S_all": {"d%d_%s" % k: round(v, 3) for k, v in sorted(S.items())},
                        "memory_ctx1024": {n: {k: v[k] for k in ("need_mib", "available_mib", "margin_mib")} for n, v in m.items()},
                        "pi4_margin_with_draft_mib": pi4_margin, "jetson_draft_candidate": "no: %d MiB to spare, a 0.6B draft needs ~800" % m["jetson"]["margin_mib"]},
        "sources": {"step1": os.path.relpath(STEP1, ROOT), "step1_sha256": sha(STEP1), "step2": os.path.relpath(s2p, ROOT),
                    "step2_sha256": sha(s2p)},
        "inputs_sha256": {f: sha(os.path.join(HERE, f)) for f in ("spec_serve_20261010.py", "serve_pool.sh", "predict_pool.py",
                                                                    os.path.basename(__file__))},
        "moe_note": "Qwen3-30B-A3B (14.7 GB) does not fit Pi 4 + Jetson (about 10.9 GB free): the survey's MoE leg needs the Pi 5",
        "hypotheses": [
            {"id": "SS-BAR", "claim": "the engine's choice is at least 1.30x the base arm (decode ms per token)"},
            {"id": "SS-ACC", "claim": "every draft arm's measured speedup within 10 % of its prediction"},
            {"id": "SS-ORDER", "claim": "the draft arm predicted faster is measured faster"},
            {"id": "SS-MOE", "claim": "not testable on these boards (moe_note)"}],
    }


def main(argv):
    if "--accept" not in argv:
        sys.exit(__doc__)
    if os.path.exists(OUT) and "--dry-run" not in argv:
        sys.exit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    reg = build(argv[argv.index("--accept") + 1])
    mp = reg["model_parts"]
    print("T %s; tau %s; draft step ms %s (laptop rtt %.3f ms)" % (mp["T_ms"], mp["tau"], mp["draft_step_ms"], mp["laptop_rtt_ms_from_pi4"]))
    print("S(d, board): %s" % mp["S_all"])
    print("choice %s; predicted %s" % (reg["engine_choice"], reg["predicted"]))
    print("memory at ctx %d: %s; the Pi 4 with the draft keeps %d MiB" % (reg["ctx"], mp["memory_ctx1024"], mp["pi4_margin_with_draft_mib"]))
    print("order %s; %d prompts x %d tokens" % (reg["order"], len(reg["prompts"]), reg["n_predict"]))
    if "--dry-run" in argv:
        print("dry run: nothing written"); return reg
    json.dump(reg, open(OUT, "w"), indent=1, ensure_ascii=False)
    os.chmod(OUT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print("registered %s at %s, sha256 %s (read-only)" % (os.path.relpath(OUT, ROOT), reg["registered"], sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
