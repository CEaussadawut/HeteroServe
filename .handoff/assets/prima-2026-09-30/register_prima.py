#!/usr/bin/env python3
"""Register the prima.cpp comparison before prima.cpp has run at all (user: "after downloaded do the comparison in the full
evaluation"; "i insist to go with qwen3-14B"):

    python3 .handoff/assets/prima-2026-09-30/register_prima.py [--dry-run]

Writes testbed/dataset/prima_predictions_20260930.json, read-only, never overwritten. Identical conditions, as far as prima.cpp
allows (testbed/prima.md): Qwen3-14B Q4_K_M, the Pi 5 and the Pi 4 (prima.cpp cannot build on the Jetson), the wired overlay,
the QR3 protocol and client (qr3_tpot.py), a 4096-token context, one user.
"""
import dataclasses, datetime, glob, hashlib, json, os, stat, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad", "control"))
import predict_pool as P  # noqa: E402
from control.placement import plan_optimal  # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "prima_predictions_20260930.json")
MEMFILE = os.path.join(DS, "memory_snapshot_20260930_013353.json")      # the warm snapshot QR3 planned against
QR3 = os.path.join(DS, "qr3_results_20260930.json")
KEY, CTX = "qwen3_14b", 4096


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
    inputs = {f: os.path.join(DS, f) for f in ("characterisation.csv", "model_shapes.json", "link.json")}
    inputs.update({"memory": MEMFILE, "qr3_results": QR3,
                   "cluster.json": os.path.join(ROOT, "ThinkPad", "config", "cluster.json"),
                   "predict_pool.py": os.path.join(ROOT, "testbed", "scripts", "predict_pool.py"),
                   "serve_pool.sh": os.path.join(ROOT, "testbed", "scripts", "serve_pool.sh"),
                   "qr3_tpot.py": os.path.join(ROOT, "testbed", "scripts", "qr3_tpot.py"),
                   "placement.py": os.path.join(ROOT, "ThinkPad", "control", "placement.py"),
                   "prima_patch": os.path.join(ROOT, "testbed", "patches", "prima-288df6a-qwen3-v1.patch"),
                   "prima_build.sh": os.path.join(ROOT, "testbed", "scripts", "prima_build.sh"),
                   "highs_package": "/opt/heteroserve/third_party/highs-1.15.1-aarch64-linux-gnu-static-mit.tar.gz"})
    shapes = json.load(open(inputs["model_shapes.json"]))
    pairs = json.load(open(inputs["link.json"]))["pairs"]
    mem = json.load(open(MEMFILE))
    R = P.rows(inputs["characterisation.csv"], "char")
    shape, quant, _ = P.MODELS[KEY]; q = shapes[shape]
    C, _ = P.costs(R, shapes, quant)
    nodes = ["pi5", "pi4"]                                        # serve_pool.sh NODES="pi5 pi4"
    L, missing = P.link_view(pairs, "pi5", nodes); assert not missing, missing
    devs = P.profiles(C, q, mem, L, nodes); mp = dataclasses.replace(P.model_profile(q, shape), context_len=CTX)
    plan = plan_optimal(devs, mp, "decode", host_candidates=["pi5"])
    assign = [(n, plan.layers[n]) for n in plan.order if plan.layers.get(n, 0) > 0]
    m = P.memory(assign, q, mem, devs, mp); dec, pre = P.predict(assign, q, C, L)
    a = P.bench_args(assign, KEY)["args"].replace(" -rpc ", " --rpc ")
    qr3 = json.load(open(QR3))["configs"][KEY]
    reg = {
        "registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "campaign": "2026-09-30: the full evaluation's prima.cpp comparison (proposal Phase 5), Qwen3-14B Q4_K_M, one user",
        "inputs_sha256": {k: sha(v) for k, v in inputs.items()},
        "prima": {"repo": "https://github.com/OpenCPIL/prima.cpp", "commit": "288df6a2cc2120387e12d6f40798f30cc525c9ea",
                  "patch": "testbed/patches/prima-288df6a-qwen3-v1.patch (the qwen3 port: prima.cpp's own qwen2 code with upstream's "
                           "per-head Q/K norms and no Q/K/V biases; its distributed path and profiler taught the two new tensors)",
                  "highs": "HiGHS v1.15.1, highs-1.15.1-aarch64-linux-gnu-static-mit.tar.gz (sha256 = GitHub's published digest), "
                           "linked statically on the Pi 5 (rank 0) only",
                  "not_on_the_jetson": "gcc 7.5 < 9, ZeroMQ 4.2.5 < 4.3.2, make 4.1, fio 3.1, no internet (testbed/prima.md)"},
        "conditions": "identical where prima.cpp allows: the same model file (sha256-checked on every board that loads it), the Pi 5 "
                      "and the Pi 4, the wired overlay (192.168.10.x), -t 4, a 4096-token context, the QR3 protocol and client "
                      "(qr3_tpot.py: one unscored warm-up of 512/32, then 10 requests of 128-token GPL-3 prompts x 128 tokens, greedy, "
                      "ignore_eos, cache_prompt false, streamed and timed per token on the laptop), each arm started only once the Pi 5 "
                      "is below 50 C. Order: A, then B.",
        "arms": {
            "A_ours_pis": {"what": "the engine's plan on the two boards prima.cpp can use, served by llama.cpp at the cluster pin "
                                   "(serve_pool.sh NODES='pi5 pi4', MEMFILE the 01:33 warm snapshot)",
                           "plan": [{"node": n, "blocks": k} for n, k in assign], "head_on": assign[-1][0],
                           "server_args": a + " -c %d -np 1 -t 4" % CTX,
                           "memory": {n: {k: v[k] for k in ("need_mib", "reserve_mib", "available_mib", "margin_mib")} for n, v in m.items()},
                           "predicted": {"decode_ms_per_token": round(dec, 1), "prefill_ms_per_token": round(pre, 1)},
                           "run": "qr3_tpot.py run --model qwen3_14b --ctx 4096 --mem testbed/dataset/memory_snapshot_20260930_013353.json "
                                  "--nodes 'pi5 pi4' --tag primacmpA", "log_glob": "testbed/results/qr3_primacmpA_qwen3_14b_2*.log"},
            "B_prima_halda": {"what": "prima.cpp as its README runs it: the Pi 5 rank 0 (llama-server, the head, which holds the input and "
                                      "output layers), the Pi 4 rank 1 (llama-cli), --prefetch, -t 4, -c 4096, and NO -lw: its Halda "
                                      "scheduler profiles both boards and picks the split, which each rank logs ('Using window size')",
                              "run": "qr3_tpot.py run --model qwen3_14b --ctx 4096 --engine prima --tag primacmpB",
                              "log_glob": "testbed/results/qr3_primacmpB_qwen3_14b_2*.log"},
            "D_ours_full_reference": {"what": "NOT RUN AGAIN: QR3's measured qwen3_14b config -- the engine's plan on the whole cluster "
                                              "(Pi 5 32 + Jetson 8, the Jetson's clocks locked), same protocol, same client, same day",
                                      "measured": {"client_mean_ms": qr3["client"]["mean"], "client_p95_ms": qr3["client"]["p95"],
                                                   "ttft_client_mean_ms": qr3["ttft_client_ms"]["mean"], "log": qr3["log"]}},
        },
        "smoke": "UNSCORED, after this registration and before any scored arm: prima.cpp started as arm B, one 128-token request "
                 "(QR3's request-0 prompt) for 16 tokens, then stopped. It checks the qwen3 port (CORRECTNESS), the server's API "
                 "with the client, and records Halda's split. Its timings are not used.",
        "rules": [
            "CORRECTNESS: the port is taken as correct if prima.cpp's first 8 greedy tokens on QR3's request-0 prompt match llama.cpp's "
            "(arm A's request 0), compared as text. If they do not, both texts are reported and arm B is not scored until it is "
            "explained.",
            "DECODE: the arm with the lower pooled client mean TPOT is the faster; the difference is reported in %. PREDICTED: A is "
            "faster. On two CPU boards with the model in RAM, prima.cpp's ring is a two-stage pipeline like ours; its advantages (GPUs, "
            "disk offload with prefetch, many devices) do not apply. Against that, its late-2024 base does not repack Q4_K on the "
            "Pi 5, which decodes faster (a single sequence 13 % faster, on the 9B, benchmarks.md step 1).",
            "TTFT: the arm with the lower mean client TTFT (128-token prompt) is the faster in prefill. PREDICTED: A, whose Pi 5 "
            "prefills with the repacked kernels (1.55x faster on the 9B).",
            "QR3: each arm's pooled client P95 against 500 / 250 ms, as QR3's rules. PREDICTED: neither arm meets 500.",
            "HALDA-SPLIT: PREDICTED: Halda gives the Pi 5 at most 30 of the 40 blocks, because the Pi 5 must also hold the 609 MiB "
            "output layer (and prima.cpp keeps the input embedding there). Reported with the split it picks, at the smoke run and at B.",
            "ACCURACY (A only): A's server mean TPOT within +-10 % of the predictor (as QR3).",
            "REFERENCE: D, the engine on the whole cluster, is reported beside A and B. prima.cpp cannot use the Jetson; that is a "
            "capability difference, reported as such, and no rule is scored on it.",
            "COMPLETE: every scored request returns 128 tokens (for B: 128 content events and tokens_predicted 128); a short one is "
            "reported and the full ones still scored.",
        ],
    }
    txt = json.dumps(reg, indent=1)
    if dry:
        print(txt); return
    with open(OUT, "w") as f:
        f.write(txt + "\n")
    os.chmod(OUT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print("registered", os.path.relpath(OUT, ROOT), reg["registered"], sha(OUT))
    print("  A:", reg["arms"]["A_ours_pis"]["plan"], reg["arms"]["A_ours_pis"]["predicted"],
          {n: v["margin_mib"] for n, v in reg["arms"]["A_ours_pis"]["memory"].items()})


if __name__ == "__main__":
    main(sys.argv[1:])
