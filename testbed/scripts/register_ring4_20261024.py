#!/usr/bin/env python3
"""Register the ring-lap study on the four-node cluster's boards (2026-10-24) before it runs. Writes, read-only,
testbed/dataset/ring4_predictions_20261024.json.

    python3 testbed/scripts/register_ring4_20261024.py [--dry-run]

The question is the advisor's "try sequential: is it faster?" (meeting of 2026-10-05), answered on 2026-10-09 with the Pi 4
and the Jetson (ring_laps_predictions_20261009.json: one lap fastest in every round, +4.45 ms per extra lap); the user's
rule of 2026-10-21 puts it back on the four-node cluster. The design is that study's (register_ring_laps_20261009.py),
moved onto the laptop and the Pi 5: TinyLlama Q4_K_M, the laptop holding 4 blocks (two of each of TinyLlama's two block
sizes in every layout) and the Pi 5 18, cut into k = 1, 2, 3, 4 alternating stages, so each token visits the Pi 5 k times
and each board holds the same bytes throughout. Every stage is an RPC device: the laptop's are ggml-rpc-servers on its
loopback (127.0.0.1:50060-50063, -t 4), the Pi 5's on its router address (192.168.1.122:50052-50055, -t 4); llama-bench on
the laptop holds only the token embedding. Three rounds, the layouts rotated by round, unshaped (the round-trip rule is
re-tested by the four-node network study, register_ns4_20261024.py). Time only, as on 10-09.

Predictions: predict_pool.predict with the laptop's stages in the worker role on its loopback (0.034 ms, ping 2026-10-24,
nominal 10 Gbit/s) and the Pi 5's on the measured Pi 5 - laptop pair (link.json), RTT1 (the adopted rule) with two trips
beside it. Hypotheses (decided here, before any data):
  R4-SEQ  k = 1 has the lowest decode ms/token in every round, and the least-squares slope of decode over k is positive.
  R4-ACC  every layout's mean decode within 10 % of its RTT1 prediction.
Exploratory, not a hypothesis: the per-lap cost against the predicted one; the difference is the per-visit overhead the
link terms leave out (about 3.2 ms per lap on 10-09).
"""
import datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import register_ring_laps_20261009 as RL                    # noqa: E402  layer_map, LAYOUTS

DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "ring4_predictions_20261024.json")
MODEL = "tinyllama"
LAYOUTS = {lab: [("L" if t == "P" else "F", k) for t, k in st] for lab, st in RL.LAYOUTS.items()}   # 10-09's cuts
L_PORTS, F_PORTS = (50060, 50061, 50062, 50063), (50052, 50053, 50054, 50055)
F_IP = "192.168.1.122"
LOOPBACK = {"mbps": 10000.0, "rtt_ms": 0.034}
ROUNDS = 3
PROTO = "-t 4 -ngl 99 -p 64 -n 32 -r 3 -oe jsonl -v"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rpc_and_ts(stages):
    li = fi = 0
    eps = []
    for t, _ in stages:
        if t == "L":
            eps.append("127.0.0.1:%d" % L_PORTS[li]); li += 1
        else:
            eps.append("%s:%d" % (F_IP, F_PORTS[fi])); fi += 1
    n = [k for _, k in stages]
    ts = [n[0] - 0.5] + n[1:-1] + [n[-1] + 1.5]
    return ",".join(eps), "/".join("%g" % x for x in ts)


def build():
    shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
    pairs = json.load(open(os.path.join(DS, "link.json")))["pairs"]
    R = P.rows(os.path.join(DS, "characterisation.csv"), "char")
    shape, quant, path = P.MODELS[MODEL]
    q = shapes[shape]
    C, src = P.costs(R, shapes, quant)
    CC = {"laptop": C["laptop"], "laptopL": C["laptop"], "pi5": C["pi5"]}
    fl = {"mbps": pairs["pi5-laptop"]["mbps"], "rtt_ms": pairs["pi5-laptop"]["rtt_ms"]}
    links = {"laptopL": LOOPBACK, "pi5": fl}
    layouts, l_bytes = {}, set()
    for lab, stages in LAYOUTS.items():
        assert sum(k for _, k in stages) == q["n_layers"] and sum(k for t, k in stages if t == "L") == 4
        start, b, stage_mib = 0, 0, []
        for t, k in stages:
            sb = sum(q["layer_bytes"][start:start + k])
            b += sb if t == "L" else 0
            stage_mib.append(round(sb / 2 ** 20, 2))
            start += k
        l_bytes.add(round(b))
        lm = RL.layer_map(stages, q["n_layers"])
        want = [i for i, (_, k) in enumerate(stages) for _ in range(k)] + [len(stages) - 1]
        assert lm == want, (lab, lm, want)
        rpc, ts = rpc_and_ts(stages)
        assign = [("laptop", 0)] + [("laptopL" if t == "L" else "pi5", k) for t, k in stages]
        pred = {}
        for r in (1, 2):
            d, p = P.predict(assign, q, CC, links, rtt_trips=r)
            pred["rtt%d" % r] = {"decode_ms": round(d, 2), "prefill_ms": round(p, 3)}
        layouts[lab] = {"stages": [{"board": "laptop" if t == "L" else "pi5", "blocks": k, "block_mib": m}
                                   for (t, k), m in zip(stages, stage_mib)],
                        "pi5_visits": sum(1 for t, _ in stages if t == "F"), "rpc": rpc, "ts": ts,
                        "args": "-m %s --rpc %s -ts %s %s" % (path, rpc, ts, PROTO), "predicted": pred}
    assert len(l_bytes) == 1, ("the laptop must hold the same bytes in every layout", l_bytes)
    per_lap = {"rtt%d" % r: round((layouts["k4"]["predicted"]["rtt%d" % r]["decode_ms"]
                                    - layouts["k1"]["predicted"]["rtt%d" % r]["decode_ms"]) / 3, 3) for r in (1, 2)}
    order = []
    for r in range(ROUNDS):
        labs = list(LAYOUTS)
        order.append(labs[r % len(labs):] + labs[:r % len(labs)])
    return {"registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "name": "ring4_20261024", "model": MODEL, "model_path": path, "model_sha256_key": "full_eval.SHA['tinyllama']",
            "host": "laptop",
            "servers": {"laptop": {"ip": "127.0.0.1", "ports": list(L_PORTS), "extra": "-t 4"},
                        "pi5": {"ip": F_IP, "ports": list(F_PORTS), "extra": "-t 4"}},
            "protocol": PROTO, "rounds": ROUNDS, "order": order, "layouts": layouts,
            "laptop_block_bytes_every_layout": l_bytes.pop(),
            "links": {"pi5": dict(fl, pair="pi5-laptop"), "loopback": dict(LOOPBACK, note="ping 2026-10-24; bandwidth nominal")},
            "predicted_per_lap_ms": per_lap, "cost_sources": {n: src[n] for n in ("laptop", "pi5")},
            "replicates": "testbed/dataset/ring_laps_predictions_20261009.json (Pi 4 + Jetson)",
            "inputs_sha256": {**{f: sha(os.path.join(DS, f)) for f in ("characterisation.csv", "model_shapes.json", "link.json")},
                              **{f: sha(os.path.join(HERE, f)) for f in ("predict_pool.py", "ring4_laps_20261024.sh",
                                                                          "register_ring4_20261024.py", "score_ring4_20261024.py",
                                                                          "register_ring_laps_20261009.py")},
                              "ThinkPad/config/cluster.json": sha(os.path.join(ROOT, "ThinkPad", "config", "cluster.json"))},
            "hypotheses": [
                {"id": "R4-SEQ", "claim": "k = 1 has the lowest decode ms/token in every round, and the least-squares slope of "
                                          "decode ms/token over k (all rounds) is positive"},
                {"id": "R4-ACC", "claim": "every layout's mean decode within 10 % of its RTT1 prediction"}]}


def main(argv):
    if os.path.exists(OUT) and "--dry-run" not in argv:
        sys.exit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    reg = build()
    for lab, v in reg["layouts"].items():
        print("%s  visits %d  --rpc %s  -ts %s  decode %.2f / %.2f ms (RTT1 / RTT2)" % (
            lab, v["pi5_visits"], v["rpc"], v["ts"], v["predicted"]["rtt1"]["decode_ms"], v["predicted"]["rtt2"]["decode_ms"]))
    print("per lap: %s; order: %s" % (reg["predicted_per_lap_ms"], reg["order"]))
    if "--dry-run" in argv:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s at %s, sha256 %s (read-only)" % (os.path.relpath(OUT, ROOT), reg["registered"], sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
