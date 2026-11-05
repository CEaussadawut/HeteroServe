#!/usr/bin/env python3
"""Register the test of the predictor rule VISIT (2026-10-09) before it runs. Writes, read-only,
testbed/dataset/visit_predictions_20261009.json.

    python3 testbed/scripts/register_visit_20261009.py [--dry-run] [--out PATH]
        --out writes the same registration elsewhere (a smoke run's plumbing; never the dataset's file)

The rule, VISIT (predict_pool.VISIT_MS): each worker visit costs a fixed 1.587 ms of decode per token beyond RTT1's link
terms. Fitted on the ring-lap study (ring_laps_predictions_20261009.json, scored 2026-10-09): a lap of two visits, a Jetson
stage and a Pi 4 loopback stage, cost 4.450 ms against RTT1's 1.277 ms of link terms, so 3.173 ms per lap, split evenly here.
That fit is in sample. This is its test on fresh cells, which also separate the two kinds of visit the laps always added
together, and include the layout real pools use: the host computing its own blocks.

The cells: TinyLlama Q4_K_M with the Pi 4 holding blocks 0-3 and the Jetson 4-21 and the output layer in EVERY layout, so
each board holds the same bytes in every layout and only the stages change:
  h1  H4 | J18                  the Pi 4 computes its blocks inside llama-bench (-ngl 19): 1 Jetson visit
  h2  H4 | J9 | J9              2 Jetson visits
  h4  H4 | J5 | J5 | J4 | J4    4 Jetson visits
  a1  P4 | J18                  every stage an RPC device (-ngl 99), the ring-lap layout k1: 1 loopback + 1 Jetson visit
  p4  P1 | P1 | P1 | P1 | J18   4 loopback visits + 1 Jetson visit
  j3  P4 | J6 | J6 | J6         1 loopback + 3 Jetson visits
P is a Pi 4 loopback ggml-rpc-server (127.0.0.1:50060-50063, -t 4) and J a Jetson ggml-rpc-server (192.168.1.114:50052-50055,
Vulkan0). Between two J stages the hidden state passes through the host: at the pin a copy between two RPC servers returns
false and the scheduler reads from one and writes to the other (ggml-rpc.cpp:752-761). Run by ring_laps.sh at s0 only (the
link as it is): 3 rounds, the layouts rotated by round, the ring-lap protocol (-t 4 -p 64 -n 32 -r 3).

Predictions: predict_pool.predict with RTT1 (rtt_trips 1), without VISIT (visit_ms 0) and with it (VISIT_MS); the Pi 4 in
the host role for H and the worker role for P stages; the links of the ring-lap registration (link.json's Pi 4 <-> Jetson
pair; the loopback 0.041 ms at a nominal 10 Gbit/s). Hypotheses, decided here before any data:
  VISIT-J    the remote visit, as real pools have it: the least-squares slope of decode ms/token over Jetson visits in the
             in-process host series (h1, h2, h4; every round), minus RTT1's predicted slope over the same series, lies in
             VISIT_MS +- 0.5 ms, [1.087, 2.087].
  VISIT-P    the loopback visit: the slope over loopback visits in a1 and p4 (every round), minus RTT1's predicted slope,
             lies in [1.087, 2.087].
  VISIT-MAE  over the six layouts, the mean absolute error of VISIT's decode predictions, 100 x (predicted / measured - 1)
             against each layout's mean over rounds, is below RTT1's.
Adoption: VISIT is adopted for new registrations (predict(..., visit_ms=VISIT_MS)) when VISIT-J and VISIT-MAE hold.
VISIT-P decides only whether a loopback visit, which exists only in such experiments, costs the same.
Exploratory: each kind of visit's cost fitted on these cells together; prefill's per-visit cost.
"""
import bisect, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                    # noqa: E402
import register_ring_laps_20261009 as RL                    # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "visit_predictions_20261009.json")
MODEL = "tinyllama"
LAYOUTS = {   # stages in order: "H" the Pi 4 inside llama-bench, "P" a Pi 4 loopback server, "J" a Jetson server
    "h1": [("H", 4), ("J", 18)],
    "h2": [("H", 4), ("J", 9), ("J", 9)],
    "h4": [("H", 4), ("J", 5), ("J", 5), ("J", 4), ("J", 4)],
    "a1": [("P", 4), ("J", 18)],
    "p4": [("P", 1), ("P", 1), ("P", 1), ("P", 1), ("J", 18)],
    "j3": [("P", 4), ("J", 6), ("J", 6), ("J", 6)],
}
SERIES = {"VISIT-J": ("h1", "h2", "h4"), "VISIT-P": ("a1", "p4")}
BAND_MS = 0.5
ROUNDS = 3


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def devices(stages):
    """The RPC stages (P and J) in order, with their endpoints; an H stage is the host's CPU, not a device."""
    pi = ji = 0
    out = []
    for t, k in stages:
        if t == "P":
            out.append(("127.0.0.1:%d" % RL.P_PORTS[pi], k)); pi += 1
        elif t == "J":
            out.append(("%s:%d" % (RL.J_IP, RL.J_PORTS[ji]), k)); ji += 1
    return out


def args_of(stages):
    """(--rpc, -ts, -ngl) that realise the stages at the pin: -ngl counts the output layer and offloads the LAST layers
    (llama-model.cpp:1521-1529: i_gpu_start = n_layer + 1 - ngl); -ts over the devices, each boundary half a layer inside
    its share (predict_pool.bench_args's rule), so float rounding cannot move a block."""
    dev = devices(stages)
    n = [k for _, k in dev]
    ts = [n[0] - 0.5] + n[1:-1] + [n[-1] + 1.5] if len(n) > 1 else [1]
    ngl = sum(n) + 1
    return ",".join(ep for ep, _ in dev), "/".join("%g" % x for x in ts), ngl


def layer_map(stages, n_layer):
    """Block -> device index (-1 for the host's CPU), as llama.cpp at the pin assigns it from args_of's -ngl and -ts: an
    independent check of the formula. The last entry is the output layer."""
    _, ts, ngl = args_of(stages)
    w = [float(x) for x in ts.split("/")]
    cum, s = [], 0.0
    for x in w:
        s += x; cum.append(s)
    splits = [c / s for c in cum]
    start, act = max(n_layer + 1 - ngl, 0), min(ngl, n_layer + 1)
    return [-1 if (il < start or il - start >= act) else bisect.bisect_right(splits, (il - start) / act)
            for il in range(n_layer + 1)]


def build():
    """The registration, as a dict (main() writes it; the tests build it without writing)."""
    shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
    pairs = json.load(open(os.path.join(DS, "link.json")))["pairs"]
    R = P.rows(os.path.join(DS, "characterisation.csv"), "char")
    shape, quant, path = P.MODELS[MODEL]
    q = shapes[shape]
    C, src = P.costs(R, shapes, quant)
    CC = {"pi4": C["pi4"], "pi4L": C["pi4"], "jetson": C["jetson"]}
    jl = {"mbps": pairs["pi4-jetson"]["mbps"], "rtt_ms": pairs["pi4-jetson"]["rtt_ms"]}
    links = {"pi4L": RL.LOOPBACK, "jetson": jl}
    layouts = {}
    for lab, stages in LAYOUTS.items():
        assert sum(k for _, k in stages) == q["n_layers"]
        assert [k for t, k in stages if t in "HP"] and sum(k for t, k in stages if t in "HP") == 4    # blocks 0-3, every layout
        assert all(t in "HP" for t, _ in stages[:len([1 for t, _ in stages if t in "HP"])])             # the Pi 4's stages first
        dev = devices(stages)
        want = [-1] * stages[0][1] if stages[0][0] == "H" else []
        want += [i for i, (_, k) in enumerate(dev) for _ in range(k)] + [len(dev) - 1]
        assert layer_map(stages, q["n_layers"]) == want, (lab, layer_map(stages, q["n_layers"]), want)
        rpc, ts, ngl = args_of(stages)
        host = stages[0][1] if stages[0][0] == "H" else 0
        assign = [("pi4", host)] + [("pi4L" if t == "P" else "jetson", k) for t, k in stages if t != "H"]
        pred = {}
        for name, v in (("rtt1", 0.0), ("visit", P.VISIT_MS)):
            d, p = P.predict(assign, q, CC, links, rtt_trips=1, visit_ms=v)
            pred[name] = {"decode_ms": round(d, 2), "prefill_ms": round(p, 3)}
        layouts[lab] = {"stages": [{"board": {"H": "pi4 (host)", "P": "pi4", "J": "jetson"}[t], "blocks": k} for t, k in stages],
                        "jetson_visits": sum(1 for t, _ in stages if t == "J"),
                        "loopback_visits": sum(1 for t, _ in stages if t == "P"),
                        "host_blocks": host, "rpc": rpc, "ts": ts, "ngl": ngl if ngl != 99 and host else None,
                        "args": "-m %s --rpc %s -ts %s %s" % (path, rpc, ts, RL.PROTO.replace("-ngl 99", "-ngl %d" % ngl)
                                                                if host else RL.PROTO),
                        "predicted": pred}

    def pslope(series, key, kind):
        xs = [layouts[l][kind] for l in series]
        ys = [layouts[l]["predicted"][key]["decode_ms"] for l in series]
        return round(RL_slope(xs, ys), 3)
    predicted_slopes = {h: {"rtt1": pslope(s, "rtt1", "jetson_visits" if h == "VISIT-J" else "loopback_visits"),
                            "visit": pslope(s, "visit", "jetson_visits" if h == "VISIT-J" else "loopback_visits")}
                        for h, s in SERIES.items()}
    labs = list(LAYOUTS)
    order = [labs[r % len(labs):] + labs[:r % len(labs)] for r in range(ROUNDS)]
    return {"registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "name": "visit_20261009", "model": MODEL, "model_path": path, "host": "pi4",
            "rule_under_test": {"id": "VISIT", "visit_ms": P.VISIT_MS, "with": "RTT1 (rtt_trips 1)",
                                "fitted_on": "ring_laps_predictions_20261009.json (3.173 ms per lap of two visits beyond "
                                             "RTT1's link terms, split evenly)"},
            "servers": {"pi4": {"ip": "127.0.0.1", "ports": list(RL.P_PORTS), "extra": "-t 4"},
                        "jetson": {"ip": RL.J_IP, "ports": list(RL.J_PORTS), "extra": "-d Vulkan0"}},
            "protocol": RL.PROTO, "settings": {"s0": {"netshape": None}}, "rounds": ROUNDS,
            "order": [[["s0", o]] for o in order], "layouts": layouts, "series": {h: list(s) for h, s in SERIES.items()},
            "predicted_slopes_ms_per_visit": predicted_slopes,
            "links": {"jetson": dict(jl, pair="pi4-jetson"), "loopback": dict(RL.LOOPBACK, note="ping 2026-10-09; bandwidth nominal")},
            "cost_sources": {n: src[n] for n in ("pi4", "jetson")},
            "inputs_sha256": {**{f: sha(os.path.join(DS, f)) for f in ("characterisation.csv", "model_shapes.json", "link.json")},
                              **{f: sha(os.path.join(HERE, f)) for f in ("predict_pool.py", "ring_laps.sh", "netshape.sh",
                                                                          "register_ring_laps_20261009.py",
                                                                          "register_visit_20261009.py")}},
            "hypotheses": [
                {"id": "VISIT-J", "claim": "least-squares slope of decode over Jetson visits in h1, h2, h4 (every round), minus "
                                           "RTT1's predicted slope over them, in [%.3f, %.3f] ms" % (P.VISIT_MS - BAND_MS,
                                                                                                   P.VISIT_MS + BAND_MS)},
                {"id": "VISIT-P", "claim": "the same over loopback visits in a1 and p4, in [%.3f, %.3f] ms" % (
                    P.VISIT_MS - BAND_MS, P.VISIT_MS + BAND_MS)},
                {"id": "VISIT-MAE", "claim": "over the six layouts, VISIT's mean absolute decode error (against each layout's "
                                             "mean over rounds) is below RTT1's"}],
            "adoption": "VISIT is adopted for new registrations when VISIT-J and VISIT-MAE hold; VISIT-P decides only "
                        "whether a loopback visit costs the same"}


def RL_slope(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def main(argv):
    out = OUT
    if "--out" in argv:
        out = os.path.abspath(argv[argv.index("--out") + 1])
        if os.path.dirname(out) == DS:
            sys.exit("refused: --out is for a smoke run's plumbing, never the dataset")
    if os.path.exists(out) and "--dry-run" not in argv:
        sys.exit("refused: %s exists; a registration is never overwritten" % os.path.relpath(out, ROOT))
    reg = build()
    for lab, v in reg["layouts"].items():
        print("%s  visits J %d P %d  host %d  --rpc %s  -ts %s  -ngl %s" % (lab, v["jetson_visits"], v["loopback_visits"],
                                                                         v["host_blocks"], v["rpc"], v["ts"], v["ngl"] or 99))
        print("      decode %.2f / %.2f ms (RTT1 / VISIT)   prefill %.3f" % (
            v["predicted"]["rtt1"]["decode_ms"], v["predicted"]["visit"]["decode_ms"], v["predicted"]["rtt1"]["prefill_ms"]))
    print("predicted slopes, ms per visit: %s" % reg["predicted_slopes_ms_per_visit"])
    print("order: %s" % reg["order"])
    if "--dry-run" in argv:
        print("dry run: nothing written")
        return reg
    json.dump(reg, open(out, "w"), indent=1)
    os.chmod(out, 0o444)
    print("registered %s at %s, sha256 %s (read-only)" % (os.path.relpath(out, ROOT), reg["registered"], sha(out)[:16]))
    return reg


if __name__ == "__main__":
    main(sys.argv[1:])
