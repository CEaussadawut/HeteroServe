#!/usr/bin/env python3
"""Register DEFAULT before any of its cells runs: the engine's placement against llama.cpp's own default split. From the repo root:

    python3 .handoff/assets/placement-2026-09-30/register_default.py [--dry-run]

Why (user, 2026-09-30: "run the llama.cpp default split comparison"). The rule's QR8 margin (-42 to -45 % decode) was measured against
an EVEN split, which the study chose. llama.cpp's own default is different, and it is what a user who follows its RPC README gets
(tools/rpc/README.md at the pin): "llama-cli ... -ngl 99 --rpc host1,host2 ... By default, llama.cpp distributes model weights and the
KV cache across all available devices -- both local and remote -- in proportion to each device's available memory"
(llama-model.cpp: "default split, by free memory", used when no --tensor-split is given). Every CPU RPC server reports the board's
whole RAM as free (Pi 5 8,062 MiB, Pi 4 7,819); the Jetson's Vulkan0 reports 3,250-3,591 MiB, varying run to run. A CPU-only main
host is not one of those devices, so the README's recipe leaves it computing nothing. Measured here, Qwen3-14B Q4_K_M (the dense model
that needs the pool), pp64 / tg16 x 3 (run_pool.sh), in this order:
  d.q3_14b.rule      the engine's plan (v3.qwen3_14b.rule, 2026-09-28): Pi 5 35 (host) + Jetson 5 + head; re-run in this session
  d.q3_14b.default   every board a device: the Pi 5 hosts AND runs a ggml-rpc-server of its own; --rpc pi5,pi4,jetson -ngl 99
  d.q3_14b.readme    the README verbatim, the Jetson listed first: --rpc jetson,pi4 -ngl 99; the Pi 5 hosts and computes nothing
Registered and NOT run: the README verbatim with the Pi 4 listed first (--rpc pi4,jetson). It gives the Jetson 11 blocks plus the
608.6 MiB head, and the engine's accounting says the Jetson cannot hold that (below). The 09-29 OOM on the Jetson took
systemd-journal with the worker, and the power guard lives on that board, so it is not run to watch it fail.
The split llama.cpp makes for d.q3_14b.default depends on the Jetson's free figure at start, so all three possible splits are priced;
the scorer uses the one the -v log shows.

Writes testbed/dataset/pool_predictions_20260930default.json, refuses if it exists, leaves it read-only; --dry-run writes nothing.
"""
import datetime, hashlib, json, os, stat, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad", "control"))
import predict_pool as P  # noqa: E402
from control.placement import memory_need_mb, MB  # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "pool_predictions_20260930default.json")
KEY, SHAPE = "qwen3_14b", "qwen3-14b-q4_k_m"
MODEL = "~/models/qwen3-14b-q4_k_m.gguf"
EP = P.ENDPOINT
FREE = {"pi5": 8062, "pi4": 7819}                         # what their CPU RPC servers report (every pool log since 2026-09-27)
LOOPBACK = {"mbps": 20000.0, "rtt_ms": 0.05, "pair": "loopback (not measured: the worker-role rows were taken over loopback)"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def llama_split(free, n_layer=40):
    """llama.cpp's default split (llama-model.cpp at the pin): with -ngl 99 every layer and the output (n_layer + 1) goes to the
    devices in --rpc order; layer il goes to the first device whose cumulative share of the free memory exceeds il / (n_layer + 1).
    Returns blocks per device (the last device also holds the output head)."""
    tot, cum, acc = sum(free), [], 0.0
    for f in free:
        acc += f; cum.append(acc / tot)
    counts = [0] * len(free)
    for il in range(n_layer):                              # the blocks; the output (il = n_layer) always lands on the last device
        x = il / (n_layer + 1)
        counts[next(i for i, c in enumerate(cum) if c > x)] += 1
    return counts


def main(argv):
    dry = "--dry-run" in argv
    if os.path.exists(OUT) and not dry:
        sys.exit("%s exists: a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    inputs = {f: os.path.join(DS, f) for f in ("characterisation.csv", "model_shapes.json", "link.json", "memory_at_rest.json")}
    shapes = json.load(open(inputs["model_shapes.json"]))
    pairs = json.load(open(inputs["link.json"]))["pairs"]
    mem = json.load(open(inputs["memory_at_rest.json"]))
    Rc = P.rows(inputs["characterisation.csv"], "char")
    C, src = P.costs(Rc, shapes, "Q4_K_M", size=True)
    q = shapes[SHAPE]
    mp = P.model_profile(q, SHAPE)
    Lall, missing = P.link_view(pairs, "pi5", ["pi5", "jetson", "pi4"])
    assert not missing, missing
    devs = P.profiles(C, q, mem, Lall, ["pi5", "jetson", "pi4"])
    lb = q["layer_bytes"]

    def price(assign, loopback_pi5=False):
        L = dict(Lall)
        if loopback_pi5:
            L["pi5"] = LOOPBACK
        d, p = P.predict(assign, q, C, L)
        return {"decode_ms_per_token": round(d, 1), "decode_tok_s": round(1000 / d, 3),
                "prefill_ms_per_token": round(p, 1), "prefill_tok_s": round(1000 / p, 2)}

    def memo(assign):
        """Per board: the need of every role it plays, summed (the Pi 5 is both host and worker in d.q3_14b.default)."""
        out, start = {}, 0
        for i, (n, k) in enumerate(assign):
            need = memory_need_mb(devs[n], mp, sum(lb[start:start + k]), k, holds_head=(i == len(assign) - 1), holds_embd=(i == 0),
                                  role=("host" if i == 0 else "worker"), uploads=(i == 0 and len(assign) > 1))
            e = out.setdefault(n, {"blocks": 0, "need_mib": 0.0, "roles": []})
            e["blocks"] += k; e["need_mib"] += need; e["roles"].append("host" if i == 0 else "worker")
            if devs[n].max_weights_mb is not None and i > 0:
                w = (sum(lb[start:start + k]) + (q["head_bytes"] if i == len(assign) - 1 else 0.0)) / MB
                e.update({"weights_mib": round(w, 1), "weights_cap_mib": round(devs[n].max_weights_mb),
                          "weights_margin_mib": round(devs[n].max_weights_mb - w)})
            start += k
        for n, e in out.items():
            avail, res = mem[n]["mem_available_mib"], devs[n].memory_reserve_mb
            e.update({"need_mib": round(e["need_mib"]), "reserve_mib": round(res), "available_mib": avail,
                      "margin_mib": round(avail - e["need_mib"] - res)})
        return out

    cells, variants = {}, {}
    rule = [("pi5", 35), ("jetson", 5)]
    cells["d.q3_14b.rule"] = {"why": "the engine's plan for Qwen3-14B (v3.qwen3_14b.rule, 2026-09-28), re-run in this session",
                              "assign": rule, "bench": P.bench_args(rule, KEY), "pred": price(rule), "memory": memo(rule)}
    # every board a device: the split llama.cpp would make for each possible Jetson free figure (3,250-3,591 MiB seen)
    for fj in range(3200, 3651, 10):
        k5, k4, kj = llama_split([FREE["pi5"], FREE["pi4"], fj])
        variants.setdefault((k5, k4, kj), []).append(fj)
    dflt = {}
    for (k5, k4, kj), fjs in sorted(variants.items()):
        a = [("pi5", 0), ("pi5", k5), ("pi4", k4), ("jetson", kj)]
        dflt["%d/%d/%d" % (k5, k4, kj)] = {"jetson_free_mib": [min(fjs), max(fjs)], "pipeline": [{"node": n, "blocks": b} for n, b in a],
                                            "predicted": price(a, loopback_pi5=True), "memory": memo(a)}
    typical = next(k for k, v in dflt.items() if v["jetson_free_mib"][0] <= 3400 <= v["jetson_free_mib"][1])
    a2 = [(x["node"], x["blocks"]) for x in dflt[typical]["pipeline"]]
    cells["d.q3_14b.default"] = {
        "why": "llama.cpp's default split, every board a device (the Pi 5 hosts and runs its own ggml-rpc-server): --rpc pi5,pi4,jetson -ngl 99, "
               "no --tensor-split",
        "assign": a2, "bench": {"host": "pi5", "args": "-m %s -rpc %s,%s,%s -ngl 99" % (MODEL, EP["pi5"], EP["pi4"], EP["jetson"])},
        "pred": dflt[typical]["predicted"], "variants": dflt,
        "memory": {n: dict(v, need_mib=max(dd["memory"][n]["need_mib"] for dd in dflt.values()),
                           margin_mib=min(dd["memory"][n]["margin_mib"] for dd in dflt.values()))
                   for n, v in dflt[typical]["memory"].items()}}   # the gate: the largest need among the possible splits
    kj1, k41 = llama_split([3400, FREE["pi4"]])
    a3 = [("pi5", 0), ("jetson", kj1), ("pi4", k41)]
    splits_b = sorted({tuple(llama_split([fj, FREE["pi4"]])) for fj in range(3200, 3651, 10)})
    cells["d.q3_14b.readme"] = {
        "why": "llama.cpp's RPC README verbatim, the Jetson listed first: --rpc jetson,pi4 -ngl 99; the Pi 5 hosts and computes nothing",
        "assign": a3, "bench": {"host": "pi5", "args": "-m %s -rpc %s,%s -ngl 99" % (MODEL, EP["jetson"], EP["pi4"])},
        "pred": price(a3), "memory": memo(a3), "possible_splits": [list(s) for s in splits_b]}
    k42, kj2 = llama_split([FREE["pi4"], 3400])
    a4 = [("pi5", 0), ("pi4", k42), ("jetson", kj2)]
    m4 = memo(a4)
    not_run = {"d.q3_14b.readme_pi4_first": {
        "why": "the README verbatim, the Pi 4 listed first: --rpc pi4,jetson -ngl 99",
        "pipeline": [{"node": n, "blocks": b} for n, b in a4], "memory": m4, "predicted_if_it_ran": price(a4),
        "verdict": "NOT RUN: the engine's accounting says the Jetson cannot hold its share (margin %d MiB after its %d MiB reserve)"
                   % (m4["jetson"]["margin_mib"], m4["jetson"]["reserve_mib"])}}
    res = {"registered": None,
           "campaign": "2026-09-30: the engine's placement against llama.cpp's own default split (DEFAULT), Qwen3-14B Q4_K_M",
           "inputs_sha256": {**{f: sha(p) for f, p in inputs.items()}, "cluster.json": sha(P.CLUSTER),
                             "predict_pool.py": sha(os.path.join(ROOT, "testbed", "scripts", "predict_pool.py")),
                             "run_pool.sh": sha(os.path.join(ROOT, "testbed", "scripts", "run_pool.sh"))},
           "engine_sha256": {f: sha(os.path.join(ROOT, "ThinkPad", "control", f)) for f in ("placement.py", "gguf.py")},
           "llama_default": {"source": "llama-model.cpp at the pin, 'default split, by free memory'; tools/rpc/README.md",
                             "reported_free_mib": {"pi5": FREE["pi5"], "pi4": FREE["pi4"], "jetson": "3,250-3,591 (varies)"}},
           "protocol": "llama-bench -t 4 -p 64 -n 16 -r 3 -oe jsonl -v; memprobe.sh v2 (run_pool.sh); order rule, default, readme",
           "rules": [
               "DEFAULT: the engine's placement beats llama.cpp's default split if d.q3_14b.rule's measured decode ms/token is below both "
               "d.q3_14b.default's and d.q3_14b.readme's. Prefill is compared and reported the same way.",
               "ACCURACY: each cell's decode and prefill within +-10 % of the prediction for the split llama.cpp actually made (read from "
               "the -v layer assignment); the predictor is untested on a host that is also a worker, hence 10 % in both phases.",
               "SPLIT: d.q3_14b.default's actual split must be one of the registered variants, and d.q3_14b.readme's one of its "
               "possible_splits; a split outside them is reported, and that cell's prediction is not scored.",
               "STEADY: a cell's phase is steady if its three repetitions agree within 5 %; one that is not is reported and still scored."],
           "not_run": not_run, "configs": {}}
    for cid, c in cells.items():
        res["configs"][cid] = {"model": SHAPE, "quant": "Q4_K_M", "why": c["why"],
                               "pipeline": [{"node": n, "blocks": b} for n, b in c["assign"]], "head_on": c["assign"][-1][0],
                               "bench": c["bench"], "memory": c["memory"], "predicted": c["pred"]}
        if "variants" in c:
            res["configs"][cid]["variants"] = c["variants"]
        if "possible_splits" in c:
            res["configs"][cid]["possible_splits"] = c["possible_splits"]
        print("%-18s %-42s decode %7.1f ms  prefill %6.1f ms  memory %s" % (
            cid, " + ".join("%s %d" % x for x in c["assign"]), c["pred"]["decode_ms_per_token"], c["pred"]["prefill_ms_per_token"],
            {n: (v["need_mib"], v["margin_mib"], v.get("weights_margin_mib")) for n, v in c["memory"].items()}))
    for k, v in dflt.items():
        print("   default split %-8s (Jetson free %d-%d MiB): decode %7.1f  prefill %6.1f" % (k, v["jetson_free_mib"][0], v["jetson_free_mib"][1],
              v["predicted"]["decode_ms_per_token"], v["predicted"]["prefill_ms_per_token"]))
    print("NOT RUN %s: %s" % (list(not_run)[0], not_run["d.q3_14b.readme_pi4_first"]["verdict"]))
    if dry:
        print("dry run: nothing written (would have written %s)" % os.path.relpath(OUT, ROOT)); return
    res["registered"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    with open(OUT, "x") as f:
        json.dump(res, f, indent=1)
    os.chmod(OUT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print("registered %s at %s, sha256 %s" % (os.path.relpath(OUT, ROOT), res["registered"], sha(OUT)))


if __name__ == "__main__":
    main(sys.argv[1:])
