#!/usr/bin/env python3
"""Pre-register the network study (topic-form scope 6, 2026-10-08): a shaped Pi 4 <-> Jetson link, and whether the planner's
link terms -- and the head placement they decide -- hold when the link is slow. Reads files only; writes one new file.

    python3 testbed/scripts/register_netshape_20261008.py [--dry-run] [--out FILE --supersedes "WHY"]

Why. At gigabit the link terms are 1-2 % of a token, so no campaign has tested them. They decide one choice outright: with
the output head on the last board (the Jetson), every decoded token brings the logits back to the host (4 x 32,000 bytes for
TinyLlama); with the head on the host (-ot "^output\\.weight=CPU"), only the hidden state (4 x 2,048 bytes). At gigabit the
head belongs on the faster Jetson (2026-10-07: 95.6 / 95.9 against 101.7 ms). Below some bandwidth the logits' transfer
outweighs the Pi 4's slower head, and the host wins. The predictor puts that crossover near 130 Mbit/s.

Cells: TinyLlama-1.1B Q4_K_M, Pi 4 host (1 block) + Jetson (21 blocks, Vulkan), two layouts:
  last   head on the Jetson (llama.cpp's -ngl default; the engine's pick at gigabit)
  host   head on the Pi 4 (-ot "^output\\.weight=CPU"; Halda's rule)
at eight link settings, applied by testbed/scripts/netshape.sh on the Pi 4 (both directions, the Jetson's address only):
  s0 unshaped | b200 b100 b50 b20: a rate limit, Mbit/s | d1 d5: added one-way delay, ms, each direction (RTT +2d) |
  s0r unshaped again, last (the session's drift)
the two layouts alternating first and last from setting to setting. Every setting's link is MEASURED on the shaped path
(measure_link.sh, PAIR="pi4 jetson") before its cells, and the cells are scored against predictions re-priced with that
measured link (score_netshape_20261008.py); the nominal predictions below are registered as well.
Protocol: run_pool.sh, -t 4 -p 64 -n 16 -r 3 -oe jsonl -v (the pool protocol), the throttle poll and memory probe in every cell.
Hypotheses (scored as written, reported as found):
  NS-FLIP   at every setting where the re-priced predictions of the two layouts differ by >= 3 %, the layout
            predicted faster measures faster in decode
  NS-CROSS  the crossover bandwidth interpolated from the measured decode gaps (linear in 1/bandwidth between the two
            settings that bracket the sign change) is within a factor 1.5 of the re-priced prediction's crossover
  NS-ACC    every cell's measured decode is within 10 % of its re-priced prediction
  NS-RTT    per layout, decode's increase over s0 at d1 and d5, fitted through the origin against the measured RTT increase,
            gives a slope (round trips per token per hop) in [1.5, 2.5]; the predictor charges 2
Output: testbed/dataset/pool_predictions_20261008netshape.json, read-only; never overwritten.
"""
import copy, datetime, hashlib, json, os, stat, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as pp                                   # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "pool_predictions_20261008netshape.json")
KEY, HOST, NODES = "tinyllama", "pi4", ["pi4", "jetson"]
ASSIGN = [("pi4", 1), ("jetson", 21)]
LAYOUTS = {"last": None, "host": 0}                          # head_at
SETTINGS = [("s0", 0, 0.0), ("b200", 200, 0.0), ("b100", 100, 0.0), ("b50", 50, 0.0), ("b20", 20, 0.0),
            ("d1", 0, 1.0), ("d5", 0, 5.0), ("s0r", 0, 0.0)]          # (id, rate Mbit/s or 0, one-way delay ms)
CODE = ("predict_pool.py", "run_pool.sh", "compare_pool.py", "netshape.sh", "measure_link.sh", "link_to_json.py",
        "netshape_campaign.sh", "score_netshape_20261008.py", "register_netshape_20261008.py")     # pinned before any data


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def nominal_link(base, rate, delay):
    """The link a setting is meant to give: the measured base, its rate replaced by the limit, its RTT plus 2 x delay."""
    return {"mbps": float(rate) if rate else base["mbps"], "rtt_ms": base["rtt_ms"] + 2 * delay}


def crossover(base_links, q, C, lo=5.0, hi=1000.0):
    """The bandwidth (Mbit/s) where the two layouts' predicted decode are equal, at the base RTT; None if no sign change."""
    def gap(mbps):
        L = copy.deepcopy(base_links)
        L["jetson"]["mbps"] = mbps
        return pp.predict(ASSIGN, q, C, L, head_at=0)[0] - pp.predict(ASSIGN, q, C, L, head_at=None)[0]
    a, b = gap(lo), gap(hi)
    if (a > 0) == (b > 0):
        return None
    for _ in range(60):                                        # bisection: gap(host - last) rises with bandwidth
        m = (lo * hi) ** 0.5
        if (gap(m) > 0) == (b > 0):
            hi = m
        else:
            lo = m
    return round((lo * hi) ** 0.5, 1)


def main(argv):
    global OUT
    if "--out" in argv:
        OUT = os.path.abspath(argv[argv.index("--out") + 1])
    if os.path.exists(OUT) and "--dry-run" not in argv:
        sys.exit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    ds = os.path.join(ROOT, "testbed", "dataset")
    inputs = {f: os.path.join(ds, f) for f in ("characterisation.csv", "model_shapes.json", "link.json", "memory_at_rest.json")}
    shapes = json.load(open(inputs["model_shapes.json"]))
    pairs = json.load(open(inputs["link.json"]))["pairs"]
    mem = json.load(open(inputs["memory_at_rest.json"]))
    R = pp.rows(inputs["characterisation.csv"], "char")
    shape, quant, _ = pp.MODELS[KEY]
    q = shapes[shape]
    C, src = pp.costs(R, shapes, quant)
    L0, missing = pp.link_view(pairs, HOST, NODES)
    if missing:
        sys.exit("refused: no measured link %s" % missing)
    base = dict(L0["jetson"])
    devs = pp.profiles(C, q, mem, L0, NODES)
    from dataclasses import replace
    mp = replace(pp.model_profile(q, shape), context_len=4096)
    reg = {"registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "campaign": "the network study: a shaped Pi 4 <-> Jetson link (netshape.sh), TinyLlama Pi 4 1 + Jetson 21, the head "
                       "last or on the host; does the predictor's link term, and the head placement it decides, hold?",
           "inputs_sha256": {**{f: sha(p) for f, p in inputs.items()},
                             **{f: sha(os.path.join(HERE, f)) for f in CODE}},
           "engine_sha256": {f: sha(os.path.join(ROOT, "ThinkPad", "control", f)) for f in ("placement.py", "gguf.py")},
           "protocol": "-t 4 -p 64 -n 16 -r 3 -oe jsonl -v", "base_link": {"pair": base.get("pair"), "mbps": base["mbps"],
                                                                         "rtt_ms": base["rtt_ms"]},
           "settings": [{"id": s, "rate_mbit": r, "delay_ms_one_way": d} for s, r, d in SETTINGS],
           "run_order": [], "configs": {},
           "crossover_mbps_predicted_nominal": crossover({"jetson": dict(base)}, q, C),
           "supersedes": ({"file": "testbed/dataset/pool_predictions_20261008netshape.json",
                           "sha256": sha(os.path.join(ROOT, "testbed", "dataset", "pool_predictions_20261008netshape.json")),
                           "why": argv[argv.index("--supersedes") + 1]} if "--supersedes" in argv else None),
           "hypotheses": [
               {"id": "NS-FLIP", "claim": "at every setting where the re-priced predictions of the two layouts differ "
                                          "by >= 3 %, the layout predicted faster measures faster in decode"},
               {"id": "NS-CROSS", "claim": "the crossover interpolated from the measured decode gaps (linear in 1/bandwidth "
                                           "between the bracketing settings) is within a factor 1.5 of the re-priced crossover"},
               {"id": "NS-ACC", "claim": "every cell's measured decode is within 10 % of its re-priced prediction"},
               {"id": "NS-RTT", "claim": "per layout, decode's increase over s0 at d1 and d5 against the measured RTT increase, "
                                         "fitted through the origin, has a slope in [1.5, 2.5] (the predictor charges 2)"}]}
    for i, (sid, rate, delay) in enumerate(SETTINGS):
        order = list(LAYOUTS) if i % 2 == 0 else list(reversed(list(LAYOUTS)))
        L = {"pi4": dict(L0["pi4"]), "jetson": dict(base, **nominal_link(base, rate, delay))}
        for lay in order:
            head_at = LAYOUTS[lay]
            dec, pre = pp.predict(ASSIGN, q, C, L, head_at=head_at)
            cid = "ns.tiny.%s.%s" % (lay, sid)
            reg["run_order"].append(cid)
            reg["configs"][cid] = {
                "model": shape, "quant": quant, "setting": sid, "layout": lay, "rate_mbit": rate, "delay_ms_one_way": delay,
                "why": "network study 2026-10-08: setting %s, head %s" % (sid, "on the Jetson" if lay == "last" else "on the host"),
                "pipeline": [{"node": n, "blocks": k} for n, k in ASSIGN],
                "head_on": ASSIGN[0][0] if head_at == 0 else ASSIGN[-1][0], "head_at": head_at,
                "bench": pp.bench_args(ASSIGN, KEY, head_at=head_at),
                "memory": pp.memory(ASSIGN, q, mem, devs, mp, head_at=head_at),
                "link_nominal": L["jetson"],
                "predicted": {"decode_ms_per_token": round(dec, 1), "decode_tok_s": round(1000 / dec, 3),
                              "prefill_ms_per_token": round(pre, 1), "prefill_tok_s": round(1000 / pre, 2)},
                "cost_sources": {n: src[n] for n, _ in ASSIGN}}
    for cid in reg["run_order"]:
        x = reg["configs"][cid]
        print("%-20s %-5s rate %4s delay %3.1f  link %7.1f Mbit/s %6.3f ms  decode %6.1f  prefill %5.1f  %s" % (
            cid, x["layout"], x["rate_mbit"] or "-", x["delay_ms_one_way"], x["link_nominal"]["mbps"], x["link_nominal"]["rtt_ms"],
            x["predicted"]["decode_ms_per_token"], x["predicted"]["prefill_ms_per_token"], x["bench"]["args"]))
    print("predicted crossover (nominal, base RTT): %s Mbit/s" % reg["crossover_mbps_predicted_nominal"])
    if "--dry-run" in argv:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print("registered %s at %s, sha256 %s (read-only)" % (os.path.relpath(OUT, ROOT), reg["registered"], sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
