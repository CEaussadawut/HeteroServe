#!/usr/bin/env python3
"""Our engine against prima.cpp's Halda on the same runtime (llama.cpp at the pin), offline. Reads files only.

    python3 testbed/scripts/compare_halda.py [--n-ctx 4096] [--out testbed/dataset/halda_offline_20261007.json]

For every model the predictor knows, and every board set whose links are measured (a link is never borrowed), it plans
with two schedulers.
  ENGINE  the rule (ThinkPad/control/placement.plan_optimal), with head_placement="best": the output head on the host
          or on the last board, whichever is cheaper. It plans decode-optimal, and request-optimal for three
          workloads. The head-last plan, llama.cpp's -ngl default and every plan registered before this, is
          reported beside it.
  HALDA   prima.cpp's scheduler, re-implemented in ThinkPad/control/halda.py and validated against its 2026-09-30
          decision. It is given OUR measured per-layer decode costs and each board's MemAvailable at rest, the most
          favourable inputs it could have, with the host as its head (rank 0) and the other boards in every ring order.
It prices every plan with the registrations' own predictor (predict_pool.predict). Halda's plans carry the output layer
on the host, where Halda puts it. Each plan's memory is checked with the engine's validated accounting
(predict_pool.memory). Both schedulers plan for the same context, n_ctx: Halda's KV term and the engine's context_len.

Also reported:
  - the 2026-09-30 decision recomputed from Halda's own logged profile (the port's check);
  - the engine's decode time under a keep-free headroom (placement.with_headroom) of 0, 1 and 2 GiB per board, with
    Halda's at the same headroom.
Where prima.cpp itself cannot run a model, Halda's rule is applied hypothetically: its distributed path at 288df6a2
takes llama and qwen2, plus qwen3 with our port.
"""
import csv, itertools, json, os, sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as pp                                   # noqa: E402
from control import halda, placement                       # noqa: E402
from control.gguf import read_metadata                     # noqa: E402

WORKLOADS = {"chat": {"prefill": 128, "decode": 128}, "long_prompt": {"prefill": 1024, "decode": 128},
             "long_answer": {"prefill": 128, "decode": 512}}
BOARD_SETS = [("pi5", "pi4", "jetson"), ("pi5", "pi4"), ("pi5", "jetson"), ("pi4", "jetson")]
PRIMA_ARCH = {"llama", "qwen2", "qwen3"}
HEADROOM_MIB = (0, 1024, 2048)
# The laptop's verified copies (testbed/models.md), read for the GGUF header only: the architecture and n_ff.
CACHE = os.path.expanduser("~/.cache/heteroserve-models")
FILES = {"tinyllama-1.1b-q4km": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "tinyllama-1.1b-q8_0": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
         "tinyllama-1.1b-q4_0": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf", "tinyllama-1.1b-q3km": "tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf",
         "qwen2.5-14b-instruct-q4_k_m": "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf", "mistral-7b-q8_0": "mistral-7b-q8_0.gguf",
         "mistral-nemo-12b-q4_k_m": "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf", "qwen3-14b-q4_k_m": "Qwen3-14B-Q4_K_M.gguf",
         "qwen3-30b-a3b-q3_k_m": "Qwen3-30B-A3B-Instruct-2507-Q3_K_M.gguf", "qwen3.8-27b-q3_k_m": "Qwen3.8-27B-Q3_K_M.gguf",
         "qwen3.5-9b-q4_k_m": "Qwen_Qwen3.5-9B-Q4_K_M.gguf"}


def pipeline(plan):
    return [(n, plan.layers[n]) for n in plan.order if plan.layers.get(n, 0) > 0]


def priced(assign, q, C, L, recur, head_at, mem, devs, mp):
    """The registrations' predictor, plus the request times for each workload, and the memory under the engine's
    accounting."""
    dec, pre = pp.predict(assign, q, C, L, recur[0], recur[1], head_at=head_at)
    m = pp.memory(assign, q, mem, devs, mp, head_at=head_at)
    worst = min(min(v["margin_mib"], v.get("weights_margin_mib", v["margin_mib"])) for v in m.values())
    return {"pipeline": " + ".join("%s %d" % a for a in assign), "head_on": assign[head_at if head_at is not None else -1][0],
            "decode_ms": round(dec, 1), "prefill_ms": round(pre, 1),
            "request_ms": {w: round(x["prefill"] * pre + x["decode"] * dec, 0) for w, x in WORKLOADS.items()},
            "min_margin_mib": worst, "fits_engine_accounting": worst >= 0}


def halda_plans(ring_rest, host, q, C, mem, n_ctx, n_ff, keep_free_mib=0):
    """Halda's decision for every ring order (host first): its per-layer cost is our measured host-role decode cost; its
    memory budget is MemAvailable at rest minus the headroom."""
    n_vocab = (q["head_params"] - q["n_embd"]) / q["n_embd"]
    buf = halda.compute_buffer_bytes(q["n_embd"], n_ff, q["n_head"], n_ctx)
    kv = halda.kv_bytes_per_layer(q["n_head_kv"], q["head_dim"], n_ctx)
    per_layer = q.get("active_bytes_per_layer") or q["bytes_per_layer"]
    out = []
    for rest in itertools.permutations(ring_rest):
        ring = [host] + list(rest)
        costs = {n: C[n]["host"][0] * per_layer for n in ring}
        budgets = [(mem[n]["mem_available_mib"] - keep_free_mib) / 1024 for n in ring]
        caps = halda.memory_caps(budgets, q["bytes_per_layer"], kv, [buf] * len(ring), q["head_bytes"],
                                 q["embd_bytes"] / n_vocab)
        out.append((ring, halda.plan(ring, costs, dict(zip(ring, caps)), q["n_layers"]), dict(zip(ring, caps))))
    return out


def check_0930():
    """Halda's 2026-09-30 decision from its own logged profile (testbed/results/qr3_primacmpB_qwen3_14b_20260930_061646_
    prima_rank0.log), with the port: it must give [24, 16]."""
    fl = {"f32": 273202, "f16": 5242880, "q4k": 566231040, "q6k": 94371840}
    costs = {"pi5": halda.layer_ms_from_profile(fl, {"f32": 38.9, "f16": 76.2, "q4k": 89.8, "q6k": 46.6}, 0.02),
             "pi4": halda.layer_ms_from_profile(fl, {"f32": 10.0, "f16": 9.0, "q4k": 23.5, "q6k": 17.5}, 0.01)}
    buf = halda.compute_buffer_bytes(5120, 17408, 40, 4096)
    caps = halda.memory_caps([5.85, 5.72], 198001664, halda.kv_bytes_per_layer(8, 128, 4096), [buf, buf], 638151680, 2880)
    return halda.plan(["pi5", "pi4"], costs, dict(zip(["pi5", "pi4"], caps)), 40)


def main(argv):
    n_ctx = int(argv[argv.index("--n-ctx") + 1]) if "--n-ctx" in argv else 4096
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(ROOT, "testbed", "dataset",
                                                                                 "halda_offline_20261007.json")
    ds = os.path.join(ROOT, "testbed", "dataset")
    char = os.path.join(ds, "characterisation.csv")
    shapes = json.load(open(os.path.join(ds, "model_shapes.json")))
    pairs = json.load(open(os.path.join(ds, "link.json")))["pairs"]
    mem = json.load(open(os.path.join(ds, "memory_at_rest.json")))
    R = pp.rows(char, "char")
    res = {"n_ctx": n_ctx, "workloads": WORKLOADS, "check_20260930": check_0930(), "configs": [], "skipped": [],
           "default_headroom_mib": placement.EXACT_FILL_HEADROOM_MB}
    for key, (shape, quant, path) in pp.MODELS.items():
        q = shapes.get(shape)
        if not q or q.get("bytes_per_layer_source") != "tensor table":
            res["skipped"].append("%s: no tensor-table shape" % key); continue
        kv = read_metadata(os.path.join(CACHE, FILES[shape]))
        arch = kv.get("general.architecture")
        n_ff = int(kv[arch + ".feed_forward_length"])
        C, _ = pp.costs(R, shapes, quant)
        recur = (None, None)
        if arch == "qwen35":                    # the adopted form (ROLE-PRE): prefill terms only, by role
            h, w = pp.recurrent_terms(char, C, q, os.path.basename(path))
            recur = ({b: {"pre": t["pre"]} for b, t in h.items()}, {b: {"pre": t["pre"]} for b, t in w.items()})
        for bset in BOARD_SETS:
            nodes = [n for n in bset if n in C]
            if len(nodes) < len(bset):
                res["skipped"].append("%s on %s: no %s row for %s" % (key, "+".join(bset), quant,
                                                                    ",".join(sorted(set(bset) - set(nodes))))); continue
            host = pp.HOST.get(key, "pi5") if pp.HOST.get(key, "pi5") in bset else bset[0]
            L, missing = pp.link_view(pairs, host, nodes)
            if missing:
                res["skipped"].append("%s on %s: no measured link %s" % (key, "+".join(bset), ", ".join(missing))); continue
            devs = pp.profiles(C, q, mem, L, nodes)
            mp = replace(pp.model_profile(q, shape), context_len=n_ctx)
            mpx = replace(mp, layer_bytes=tuple(q["layer_bytes"]))     # the exact fill's real block sizes
            cfg = {"model": key, "arch": arch, "prima_can_run": arch in PRIMA_ARCH, "boards": list(bset), "host": host}
            def engine(phase, d=devs, placement_="best", fill="average", headroom=0):
                # the published columns name their fill and headroom: from 2026-10-07 the default is the exact fill
                # with EXACT_FILL_HEADROOM_MB, reported apart as "engine_default"
                r = placement.plan_optimal(d, mpx if fill == "exact" else mp, phase, host_candidates=[host],
                                           head_placement=placement_, fill=fill, headroom_mb=headroom)
                if not r.feasible:
                    return r, {"infeasible": r.reason[:200]}
                return r, priced(pipeline(r), q, C, L, recur, 0 if r.head_node else None, mem, devs, mp)
            cfg["engine"] = engine("decode")[1]
            cfg["engine_head_last"] = engine("decode", placement_="last")[1]
            cfg["engine_exact"] = engine("decode", fill="exact")[1]
            cfg["engine_default"] = engine("decode", fill="exact", headroom=placement.EXACT_FILL_HEADROOM_MB)[1]
            cfg["engine_request"] = {wname: engine(wl)[1] for wname, wl in WORKLOADS.items()}
            rest = [n for n in nodes if n != host]
            cfg["halda"] = []
            for ring, hp, caps in halda_plans(rest, host, q, C, mem, n_ctx, n_ff):
                row = {"ring": ring, "halda_caps": caps, "removed": hp.get("removed", [])}
                if hp["feasible"]:
                    assign = [(n, hp["layers"][n]) for n in ring if n in hp["layers"]]
                    row.update(priced(assign, q, C, L, recur, 0, mem, devs, mp))
                else:
                    row["infeasible"] = hp["reason"]
                cfg["halda"].append(row)
            cfg["headroom"] = {}
            for kf in HEADROOM_MIB:
                e = placement.plan_optimal(placement.with_headroom(devs, kf), mp, "decode", host_candidates=[host],
                                           head_placement="best", fill="average", headroom_mb=0)
                hb = [h for _, h, _ in halda_plans(rest, host, q, C, mem, n_ctx, n_ff, kf) if h["feasible"]]
                dkf = placement.with_headroom(devs, kf)  # (also the exact fill's memory check, below)
                hdec = []
                for h in hb:              # only rings that fit the engine's accounting at the same headroom
                    a = [(n, h["layers"][n]) for n in h["ring"] if n in h["layers"]]
                    mm = pp.memory(a, q, mem, dkf, mp, head_at=0)
                    if min(min(v["margin_mib"], v.get("weights_margin_mib", v["margin_mib"])) for v in mm.values()) >= 0:
                        hdec.append(pp.predict(a, q, C, L, recur[0], recur[1], head_at=0)[0])
                ex = placement.plan_optimal(placement.with_headroom(devs, kf), mpx, "decode", host_candidates=[host],
                                            head_placement="best", fill="exact", headroom_mb=0)
                exm = pp.memory(pipeline(ex), q, mem, dkf, mp, head_at=0 if ex.head_node else None) if ex.feasible else {}
                cfg["headroom"][kf] = {"engine_decode_ms": round(pp.predict(pipeline(e), q, C, L, recur[0], recur[1],
                                                                       head_at=0 if e.head_node else None)[0], 1)
                                       if e.feasible else None,
                                       "engine_exact_decode_ms": round(pp.predict(pipeline(ex), q, C, L, recur[0], recur[1],
                                                                                  head_at=0 if ex.head_node else None)[0], 1)
                                       if ex.feasible else None,
                                       "engine_exact_pipeline": " + ".join("%s %d" % a for a in pipeline(ex)) if ex.feasible else None,
                                       "engine_exact_min_margin_mib": min(min(v["margin_mib"], v.get("weights_margin_mib", v["margin_mib"]))
                                                                          for v in exm.values()) if exm else None,
                                       "halda_best_decode_ms": round(min(hdec), 1) if hdec else None}
            res["configs"].append(cfg)
    json.dump(res, open(out_path, "w"), indent=1)
    show(res)
    print("wrote %s" % os.path.relpath(out_path, ROOT))


def best_halda(c, fits_only=True):
    hs = [h for h in c["halda"] if "decode_ms" in h and (h["fits_engine_accounting"] or not fits_only)]
    return min(hs, key=lambda h: h["decode_ms"]) if hs else None


def show(res):
    c0 = res["check_20260930"]
    print("port check, 2026-09-30 profile: Halda windows %s (prima.cpp logged [24, 16])" % c0.get("layers"))
    print("decode, n_ctx %d. ENGINE = head where cheaper (* = on the host); HALDA = its best ring that fits the engine's "
          "accounting ('!' = only rings that would not fit)." % res["n_ctx"])
    print("%-11s %-15s %-4s | %-28s %7s %7s | %8s | %-28s %7s %7s | %6s" % (
        "model", "boards", "host", "ENGINE", "dec ms", "pre ms", "head-last", "HALDA", "dec ms", "pre ms", "Δdec%"))
    for c in res["configs"]:
        e, el = c["engine"], c["engine_head_last"]
        if "decode_ms" not in e:
            print("%-11s %-15s %-4s | engine infeasible" % (c["model"], "+".join(c["boards"]), c["host"])); continue
        hb = best_halda(c) or best_halda(c, fits_only=False)
        tag = "" if not hb or hb["fits_engine_accounting"] else " !"
        d = ("%+6.1f" % (100.0 * (hb["decode_ms"] - e["decode_ms"]) / e["decode_ms"])) if hb else "   n/a"
        print("%-11s %-15s %-4s | %-28s %7.1f %7.1f | %8s | %-28s %7s %7s | %6s%s" % (
            c["model"], "+".join(c["boards"]), c["host"][:4], e["pipeline"] + ("*" if e["head_on"] == c["host"] and
            "+" in e["pipeline"] else ""), e["decode_ms"], e["prefill_ms"],
            ("%.1f" % el["decode_ms"]) if "decode_ms" in el else "-",
            (hb["pipeline"] + tag) if hb else "disk offload", ("%.1f" % hb["decode_ms"]) if hb else "-",
            ("%.1f" % hb["prefill_ms"]) if hb else "-", d,
            "" if c["prima_can_run"] else "  [prima.cpp cannot run %s]" % c["arch"]))
    print("\nrequest time (s): the engine's request-optimal plan against Halda's best fitting ring")
    print("%-11s %-15s | %s" % ("model", "boards", "   ".join("%-24s" % w for w in res["workloads"])))
    for c in res["configs"]:
        hb = best_halda(c)
        cells = []
        for w in res["workloads"]:
            e = c["engine_request"].get(w, {})
            if "request_ms" not in e:
                cells.append("%-24s" % "engine infeasible"); continue
            ev = e["request_ms"][w] / 1000.0
            hv = hb["request_ms"][w] / 1000.0 if hb else None
            cells.append("%-24s" % ("%.1f vs %s%s" % (ev, ("%.1f" % hv) if hv else "-",
                                                      (" (%+.0f%%)" % (100 * (hv - ev) / ev)) if hv else "")))
        print("%-11s %-15s | %s" % (c["model"], "+".join(c["boards"]), "   ".join(cells)))
    print("\ndecode ms/token with each board keeping 0 / 1 / 2 GiB free: engine | Halda's best fitting ring ('-' = none fits)")
    for c in res["configs"]:
        if "decode_ms" not in c["engine"]:
            continue
        cells = ["%s | %s" % (h["engine_decode_ms"] if h["engine_decode_ms"] is not None else "-",
                              h["halda_best_decode_ms"] if h["halda_best_decode_ms"] is not None else "-")
                 for h in (c["headroom"][k] for k in HEADROOM_MIB)]
        print("%-11s %-15s | %s" % (c["model"], "+".join(c["boards"]), "   ".join("%-20s" % x for x in cells)))
    print("\nthe default since 2026-10-07 (exact fill + %.0f MiB headroom per board) against the average fill and Halda"
          % res.get("default_headroom_mib", 0))
    for c in res["configs"]:
        e, x = c["engine"], c.get("engine_default", {})
        hb = best_halda(c)
        print("%-11s %-15s | average %-26s %7s | default %-26s %7s (margin %4s) | Halda %7s" % (
            c["model"], "+".join(c["boards"]), e.get("pipeline", "-"), e.get("decode_ms", "-"),
            x.get("pipeline", "infeasible"), x.get("decode_ms", "-"), x.get("min_margin_mib", "-"),
            hb["decode_ms"] if hb else "-"))
    print("\nexact fill (fill=\"exact\", real block sizes): where its decode plan differs from the average fill's")
    for c in res["configs"]:
        e, x = c["engine"], c.get("engine_exact", {})
        if (e.get("pipeline"), e.get("decode_ms")) != (x.get("pipeline"), x.get("decode_ms")):
            print("%-11s %-15s | average %-26s %7s | exact %-26s %7s  (min margin %s MiB)" % (
                c["model"], "+".join(c["boards"]), e.get("pipeline", "-"), e.get("decode_ms", "-"),
                x.get("pipeline", "infeasible"), x.get("decode_ms", "-"), x.get("min_margin_mib", "-")))


if __name__ == "__main__":
    main(sys.argv[1:])
