#!/usr/bin/env python3
"""Re-register QR3 before any scored request runs, superseding qr3_predictions_20260930.json:

    python3 .handoff/assets/qr3-2026-09-30/register_qr3b.py [--dry-run]

Why: the first registration planned against memory_at_rest.json (read 2026-09-27), and serve_pool.sh's gate refused its 14B plan
at 01:28:56 -- the Pi 5 had 7,435 MiB available against the 7,619 on file, and the plan's margin was 89 MiB. No server started
and no per-token figure existed. This one plans against a dated snapshot (memory_snapshot_20260930_013353.json, the lowest of
10 reads per board) and copies the first registration's requirement, definitions, rules and predicted verdicts VERBATIM from its
file. Only the plans (and the predictions that follow from them), the memory input and the instrument's hashes change, plus two
conditions the second, unscored pilot (01:35, the 14B as found) showed were missing:
  - the Jetson's clocks: locked, as run_pool.sh locks them for every benchmark cell (the condition of the rows the predictor is
    built on). The pilot, with the governors left to themselves, decoded at ~1,050 ms/token against 878.5 predicted, and the GPU
    idled at 76.8 of 921.6 MHz. A fifth, secondary config repeats the 14B with the clocks as found, to measure what that costs;
  - a cool host before each config (Pi 5 < 50 C, Pi 4 < 57 C, up to 10 min), as run_batched_decode.sh gates.

Writes testbed/dataset/qr3_predictions_20260930b.json, read-only, never overwritten. It fixes, before any per-token figure:
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
OUT = os.path.join(DS, "qr3_predictions_20260930b.json")
FIRST = os.path.join(DS, "qr3_predictions_20260930.json")
MEMFILE = os.path.join(DS, "memory_snapshot_20260930_013353.json")
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
    inputs = {f: os.path.join(DS, f) for f in ("characterisation.csv", "model_shapes.json", "link.json")}
    inputs["memory"] = MEMFILE
    inputs.update({"cluster.json": os.path.join(ROOT, "ThinkPad", "config", "cluster.json"),
                   "predict_pool.py": os.path.join(ROOT, "testbed", "scripts", "predict_pool.py"),
                   "serve_pool.sh": os.path.join(ROOT, "testbed", "scripts", "serve_pool.sh"),
                   "qr3_tpot.py": os.path.join(ROOT, "testbed", "scripts", "qr3_tpot.py"),
                   "placement.py": os.path.join(ROOT, "ThinkPad", "control", "placement.py")})
    shapes = json.load(open(inputs["model_shapes.json"]))
    pairs = json.load(open(inputs["link.json"]))["pairs"]
    mem = json.load(open(MEMFILE))
    first = json.load(open(FIRST))
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
        c["clocks"] = "locked" if any(n == "jetson" for n, _ in assign) else "no jetson"
        c["log_glob"] = "testbed/results/qr3_scored_%s_2*.log" % key
        configs[key] = c
    sec = json.loads(json.dumps(configs["qwen3_14b"]))                # the secondary: the same plan, the clocks as found
    sec.update({"clocks": "asfound", "log_glob": "testbed/results/qr3_asfound_qwen3_14b_2*.log",
                "note": "SECONDARY: the qwen3_14b plan with the Jetson's clocks left to its governors (nvhost_podgov, schedutil), as "
                        "serve_pool.sh serves interactively. The predictor has no term for a governor; its figure is the locked one, and "
                        "ACCURACY is applied as written."})
    configs["qwen3_14b_asfound"] = sec
    reg = {
        "registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "campaign": first["campaign"] + ", planned against a dated memory snapshot",
        "supersedes": {"file": os.path.relpath(FIRST, ROOT), "sha256": sha(FIRST),
                       "why": "its plans used memory_at_rest.json (2026-09-27); serve_pool.sh's gate refused the 14B plan at 01:28:56 "
                              "(Pi 5 7,435 MiB available against 7,619 on file, margin 89 MiB). No server started; no per-token figure "
                              "existed. Requirement, definitions, rules and predicted verdicts are copied from it verbatim."},
        "requirement": first["requirement"],
        "inputs_sha256": {k: sha(v) for k, v in inputs.items()},
        "memory_file": os.path.relpath(MEMFILE, ROOT),
        "instrument": first["instrument"].replace("--ctx 4096 --reqs 10", "--ctx 4096 --mem %s --clocks locked --reqs 10"
                                                  % os.path.relpath(MEMFILE, ROOT))
                      + " Each config starts only once its host is below 50 C (Pi 5) or 57 C (Pi 4), waiting up to 10 min. The Jetson's "
                        "clocks are locked for every config it serves in (run_pool.sh's commands: stored, locked, restored after), except "
                        "the secondary config qwen3_14b_asfound, which leaves them to the governors (--clocks asfound --tag asfound).",
        "definitions": first["definitions"],
        "rules": first["rules"],
        "predicted_verdicts": first["predicted_verdicts"],
        "configs": configs,
        "run_order": ["qwen35_9b", "nemo12b", "qwen3_14b", "qwen3moe", "qwen3_14b_asfound"],
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
