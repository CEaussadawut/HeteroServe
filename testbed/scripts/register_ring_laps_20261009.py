#!/usr/bin/env python3
"""Register the ring-lap study (2026-10-09) before it runs. Writes, read-only, testbed/dataset/ring_laps_predictions_20261009.json.

    python3 testbed/scripts/register_ring_laps_20261009.py [--dry-run]

The question, from meeting.txt ("try sequential: is it faster?"), measured instead of modeled: our pooled runs are a
sequential ring, each token visiting each board once (k = 1); prima.cpp's piped ring laps it k times. Here llama.cpp
itself laps it: TinyLlama Q4_K_M with the Pi 4 holding 4 blocks and the Jetson 18 (the head on the last Jetson stage),
cut into k = 1, 2, 3, 4 alternating stages, so each token visits the Jetson k times; the Pi 4's four blocks are
two of each of TinyLlama's two block sizes in every layout, so each board holds the same bytes throughout. Every stage is an RPC device: the
Pi 4's stages are ggml-rpc-servers on its own loopback (127.0.0.1:50060-50063, -t 4), the Jetson's are ggml-rpc-servers on
its eth0 (192.168.1.114:50052-50055, Vulkan0); llama-bench on the Pi 4 holds only the token embedding. So every layout
has the same blocks on the same boards in the same roles, and only the number of visits changes. llama.cpp's
pipeline-parallel mode is on in all four (every layer offloaded); every pass is one micro-batch, so it overlaps nothing.

Settings: s0, the link as it is; d5, +5 ms RTT on the Pi 4 <-> Jetson path only (testbed/scripts/netshape.sh apply 0 2.5:
2.5 ms each way; loopback is untouched), with the RTT measured by ping in each round. Three rounds; each round runs both
settings (s0 first in rounds 1 and 3, d5 first in round 2), the layouts rotated by round within each.

Predictions: predict_pool.predict, the host a stage with no blocks, both Pi 4 stages and Jetson stages in the worker role,
links: the Jetson the measured Pi 4 <-> Jetson pair (link.json), the loopback 0.041 ms (ping, 2026-10-09) at a nominal
10 Gbit/s (its transfer term is under 0.002 ms a visit). Both rtt_trips 1 (RTT1, testbed/dataset/rule_rtt1_20261009.json)
and 2. Hypotheses (decided here, before any data):
  RL-SEQ  sequential is fastest: at s0, k = 1 has the lowest decode ms/token in every round, and the slope of decode ms/token
          over k (least squares, all s0 rounds) is positive.
  RL-RTT  the rule test: c = the slope through the origin of [decode at d5 - decode at s0] over k, divided by the round's
          ping RTT increase, averaged over rounds. c in [0.75, 1.25]: RTT1 holds (adopted). c in [1.75, 2.25]: the 2-trip
          model stands. Otherwise neither.
  RL-ACC  every s0 layout's mean decode within 10 % of its RTT1 prediction (100 x (predicted / measured - 1)).
Exploratory, not a hypothesis: the per-lap cost at s0 against the 2026-10-07 model (0.6-2.4 ms by link terms, 2.2-4.4 ms
with the measured RPC visit), and the per-visit overhead the link terms leave out (the s0 slope minus RTT1's).
"""
import datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                    # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "ring_laps_predictions_20261009.json")
MODEL = "tinyllama"
LAYOUTS = {   # stages in order; "P" a Pi 4 loopback server, "J" a Jetson server
    # Q4_K_M gives TinyLlama two block sizes (26.6 MiB: blocks 0 1 4 7 8 9 12 15 18 20; 23.6 MiB: the rest), so the Pi 4's
    # blocks are chosen to be two of each in every layout (0 1 2 3 / 0 1 10 11 / 0 7 13 14 / 0 6 12 17): the same bytes on
    # each board in every layout, asserted below, so only the number of visits differs.
    "k1": [("P", 4), ("J", 18)],
    "k2": [("P", 2), ("J", 8), ("P", 2), ("J", 10)],
    "k3": [("P", 1), ("J", 6), ("P", 1), ("J", 5), ("P", 2), ("J", 7)],
    "k4": [("P", 1), ("J", 5), ("P", 1), ("J", 5), ("P", 1), ("J", 4), ("P", 1), ("J", 4)],
}
P_PORTS, J_PORTS = (50060, 50061, 50062, 50063), (50052, 50053, 50054, 50055)
J_IP = "192.168.1.114"
LOOPBACK = {"mbps": 10000.0, "rtt_ms": 0.041}
D5_ADDED_RTT_MS = 5.0
ROUNDS = 3
PROTO = "-t 4 -ngl 99 -p 64 -n 32 -r 3 -oe jsonl -v"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rpc_and_ts(stages):
    """(--rpc list, -ts string) for the stages: each P stage the next free loopback server, each J the next Jetson server.
    With -ngl 99 every block and the output layer go to the devices (llama-model.cpp:1521-1546 at the pin: layer il goes to
    the first device whose cumulative split exceeds il / (n_layer + 1)), so -ts n0-0.5, n1, ..., n_last+1.5 gives each
    device exactly its blocks and the last one the output layer too (the same rule as predict_pool.bench_args)."""
    pi = ji = 0
    eps = []
    for t, _ in stages:
        if t == "P":
            eps.append("127.0.0.1:%d" % P_PORTS[pi]); pi += 1
        else:
            eps.append("%s:%d" % (J_IP, J_PORTS[ji])); ji += 1
    n = [k for _, k in stages]
    ts = [n[0] - 0.5] + n[1:-1] + [n[-1] + 1.5]
    return ",".join(eps), "/".join("%g" % x for x in ts)


def layer_map(stages, n_layer):
    """Block -> stage index, as llama.cpp assigns it from rpc_and_ts's -ts (an independent check of the formula)."""
    import bisect
    n = [k for _, k in stages]
    ts = [n[0] - 0.5] + n[1:-1] + [n[-1] + 1.5]
    cum, s = [], 0.0
    for x in ts:
        s += x; cum.append(s)
    splits = [c / s for c in cum]
    return [bisect.bisect_right(splits, il / (n_layer + 1)) for il in range(n_layer + 1)]   # the last entry: the output


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
    links = {"s0": {"pi4L": LOOPBACK, "jetson": jl},
             "d5": {"pi4L": LOOPBACK, "jetson": {"mbps": jl["mbps"], "rtt_ms": jl["rtt_ms"] + D5_ADDED_RTT_MS}}}
    layouts, pi4_bytes = {}, set()
    for lab, stages in LAYOUTS.items():
        assert sum(k for _, k in stages) == q["n_layers"] and sum(k for t, k in stages if t == "P") == 4
        start, b = 0, 0
        for t, k in stages:
            b += sum(q["layer_bytes"][start:start + k]) if t == "P" else 0
            start += k
        pi4_bytes.add(round(b))
        lm = layer_map(stages, q["n_layers"])
        want = [i for i, (_, k) in enumerate(stages) for _ in range(k)] + [len(stages) - 1]
        assert lm == want, (lab, lm, want)
        rpc, ts = rpc_and_ts(stages)
        assign = [("pi4", 0)] + [("pi4L" if t == "P" else "jetson", k) for t, k in stages]
        pred = {}
        for st, L in links.items():
            for r in (1, 2):
                d, p = P.predict(assign, q, CC, L, rtt_trips=r)
                pred["%s_rtt%d" % (st, r)] = {"decode_ms": round(d, 2), "prefill_ms": round(p, 3)}
        layouts[lab] = {"stages": [{"board": "pi4" if t == "P" else "jetson", "blocks": k} for t, k in stages],
                        "jetson_visits": sum(1 for t, _ in stages if t == "J"), "rpc": rpc, "ts": ts,
                        "args": "-m %s --rpc %s -ts %s %s" % (path, rpc, ts, PROTO), "predicted": pred}
    assert len(pi4_bytes) == 1, ("the Pi 4 must hold the same bytes in every layout", pi4_bytes)
    per_lap = {"s0_rtt%d" % r: round((layouts["k4"]["predicted"]["s0_rtt%d" % r]["decode_ms"]
                                       - layouts["k1"]["predicted"]["s0_rtt%d" % r]["decode_ms"]) / 3, 3) for r in (1, 2)}
    order = []
    for r in range(ROUNDS):
        labs = list(LAYOUTS)
        rot = labs[r % len(labs):] + labs[:r % len(labs)]
        sets = ["s0", "d5"] if r % 2 == 0 else ["d5", "s0"]
        order.append([[s, rot] for s in sets])
    reg = {"registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "name": "ring_laps_20261009", "model": MODEL, "model_path": path, "host": "pi4",
           "servers": {"pi4": {"ip": "127.0.0.1", "ports": list(P_PORTS), "extra": "-t 4"},
                       "jetson": {"ip": J_IP, "ports": list(J_PORTS), "extra": "-d Vulkan0"}},
           "protocol": PROTO, "settings": {"s0": {"netshape": None},
                                           "d5": {"netshape": "apply 0 2.5", "added_rtt_ms_nominal": D5_ADDED_RTT_MS}},
           "rounds": ROUNDS, "order": order, "layouts": layouts, "pi4_block_bytes_every_layout": pi4_bytes.pop(),
           "links": {"jetson": dict(jl, pair="pi4-jetson"), "loopback": dict(LOOPBACK, note="ping 2026-10-09; bandwidth nominal")},
           "predicted_per_lap_s0_ms": per_lap,
           "cost_sources": {n: src[n] for n in ("pi4", "jetson")},
           "inputs_sha256": {**{f: sha(os.path.join(DS, f)) for f in ("characterisation.csv", "model_shapes.json", "link.json")},
                             **{f: sha(os.path.join(HERE, f)) for f in ("predict_pool.py", "netshape.sh", "ring_laps.sh",
                                                                         "register_ring_laps_20261009.py")}},
           "rule_under_test": "RTT1 (testbed/dataset/rule_rtt1_20261009.json)",
           "hypotheses": [
               {"id": "RL-SEQ", "claim": "at s0, k = 1 has the lowest decode ms/token in every round, and the least-squares "
                                         "slope of decode ms/token over k (all s0 rounds) is positive"},
               {"id": "RL-RTT", "claim": "c = slope through the origin of [decode at d5 - decode at s0] over k, divided by the "
                                         "round's ping RTT increase, averaged over rounds; RTT1 holds for c in [0.75, 1.25], "
                                         "the 2-trip model for c in [1.75, 2.25], otherwise neither"},
               {"id": "RL-ACC", "claim": "every s0 layout's mean decode within 10 % of its RTT1 prediction"}]}
    return reg


def main(argv):
    if os.path.exists(OUT) and "--dry-run" not in argv:
        sys.exit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    reg = build()
    layouts, per_lap, order = reg["layouts"], reg["predicted_per_lap_s0_ms"], reg["order"]
    for lab, v in layouts.items():
        print("%s  visits %d  --rpc %s  -ts %s" % (lab, v["jetson_visits"], v["rpc"], v["ts"]))
        print("      decode s0 %.2f / %.2f ms (RTT1 / RTT2)   d5 %.2f / %.2f   prefill s0 %.3f / %.3f" % (
            v["predicted"]["s0_rtt1"]["decode_ms"], v["predicted"]["s0_rtt2"]["decode_ms"], v["predicted"]["d5_rtt1"]["decode_ms"],
            v["predicted"]["d5_rtt2"]["decode_ms"], v["predicted"]["s0_rtt1"]["prefill_ms"], v["predicted"]["s0_rtt2"]["prefill_ms"]))
    print("per lap at s0: %s" % per_lap)
    print("order: %s" % order)
    if "--dry-run" in argv:
        print("dry run: nothing written")
        return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s at %s, sha256 %s (read-only)" % (os.path.relpath(OUT, ROOT), reg["registered"], sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
