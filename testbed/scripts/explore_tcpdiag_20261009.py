#!/usr/bin/env python3
"""EXPLORATORY, post hoc (2026-10-09): what the TCP diagnostic's data say about the network study's two misses. Not a
registered test; reads files only and prints. The registered verdict is score_tcpdiag_20261009.py's (TD-SSAI).

    python3 testbed/scripts/explore_tcpdiag_20261009.py

1. The link figure at low rates. measure_link.sh times dd on the SENDER, so bytes still in socket buffers count as delivered:
   under 1 % for 1 GiB at gigabit, but the network campaigns send only `rate` MiB at each shaped rate. netem's rate limits
   whole frames, so the TCP payload rate is about rate x 1448 / 1514. Compared: measure_link's figure, that payload rate, and
   (at 20 Mbit/s) the Jetson's own tcp_info delivery rate during the logits bursts.
2. The 20 Mbit/s cells re-priced at the payload rate (predict_pool.predict, both rtt_trips).
3. The crossover. With the transfer as modeled, the head-last layout's extra bytes per token (the logits minus the hidden
   state) cost extra_bits / b ms at b Mbit/s, so the gap (head on host - head last) falls to zero where
   extra_bits / b = g0 + extra_bits / b0, g0 being the gap measured at gigabit (b0). The crossover therefore follows g0,
   which is listed for every session that measured both layouts at gigabit.
"""
import glob, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as pp                                   # noqa: E402

PAYLOAD = 1448 / 1514
NS = os.path.join(ROOT, "testbed", "results", "netshape_20261008_182208", "score.json")
TD = sorted(glob.glob(os.path.join(ROOT, "testbed", "results", "tcpdiag_2*", "score.json")))[-1]
PILOT = os.path.join(ROOT, "testbed", "results", "full_eval_tiny_pilot_20261009_20261009_190242", "summary.json")
NOMINAL = {"b200": 200, "b100": 100, "b50": 50, "b20": 20}


def main():
    ns, td = json.load(open(NS)), json.load(open(TD))
    shapes = json.load(open(os.path.join(ROOT, "testbed", "dataset", "model_shapes.json")))
    R = pp.rows(os.path.join(ROOT, "testbed", "dataset", "characterisation.csv"), "char")
    shape, quant, _ = pp.MODELS["tinyllama"]
    q = shapes[shape]
    C, _ = pp.costs(R, shapes, quant)
    assign = [("pi4", 1), ("jetson", 21)]
    print("1. the link figure (Mbit/s): measure_link, the payload rate netem allows, and the over-statement")
    for s, r in NOMINAL.items():
        m = ns["links"][s]["mbps"]
        print("   10-08 %-5s %7.1f %7.1f %+6.1f %%" % (s, m, r * PAYLOAD, 100 * (m / (r * PAYLOAD) - 1)))
    lb = td["cells"]["tcp.tiny.last.b20"]
    print("   10-09 b20   %7.1f %7.1f %+6.1f %%; the Jetson's delivery rate in the logits bursts: %.1f" % (
        lb["link_measured"]["mbps"], 20 * PAYLOAD, 100 * (lb["link_measured"]["mbps"] / (20 * PAYLOAD) - 1),
        lb["jetson_delivery_mbps_median"]))
    print("2. the 20 Mbit/s cells re-priced at the payload rate (error = predicted / measured - 1)")
    rows = [("10-08", "last", ns["cells"]["ns.tiny.last.b20"]["decode_ms"], ns["links"]["b20"]["rtt_ms"], None),
            ("10-08", "host", ns["cells"]["ns.tiny.host.b20"]["decode_ms"], ns["links"]["b20"]["rtt_ms"], 0),
            ("10-09", "last", td["cells"]["tcp.tiny.last.b20"]["decode_ms"], lb["link_measured"]["rtt_ms"], None),
            ("10-09", "host", td["cells"]["tcp.tiny.host.b20"]["decode_ms"], lb["link_measured"]["rtt_ms"], 0)]
    for day, lay, meas, rtt, ha in rows:
        out = []
        for t in (2, 1):
            d = pp.predict(assign, q, C, {"jetson": {"mbps": 20 * PAYLOAD, "rtt_ms": rtt}}, head_at=ha, rtt_trips=t)[0]
            out.append("%.1f (%+.1f %%, rtt_trips %d)" % (d, 100 * (d / meas - 1), t))
        print("   %s %-4s measured %.1f; at the payload rate %s" % (day, lay, meas, "; ".join(out)))
    extra_bits = (4 * (q["head_params"] - q["n_embd"]) / q["n_embd"] - q["activation_bytes"]) * 8
    b0 = ns["links"]["s0"]["mbps"]
    print("3. the crossover from the gigabit gap g0 (head on host - head last); extra %.0f bits a token" % extra_bits)
    gaps = [("10-08 network study, s0 (llama-bench tg16)", ns["cells"]["ns.tiny.host.s0"]["decode_ms"] - ns["cells"]["ns.tiny.last.s0"]["decode_ms"]),
            ("10-08 network study, s0r", ns["cells"]["ns.tiny.host.s0r"]["decode_ms"] - ns["cells"]["ns.tiny.last.s0r"]["decode_ms"]),
            ("10-09 TCP diagnostic, s0 (tg64)", td["cells"]["tcp.tiny.host.s0"]["decode_ms"] - td["cells"]["tcp.tiny.last.s0"]["decode_ms"])]
    if os.path.exists(PILOT):
        p = json.load(open(PILOT))
        gaps.append(("10-09 pilot, served (P50)", p["halda"]["server_tpot"]["p50"] - p["pooled"]["server_tpot"]["p50"]))
    L0 = {"jetson": {"mbps": b0, "rtt_ms": ns["links"]["s0"]["rtt_ms"]}}
    gm = pp.predict(assign, q, C, L0, head_at=0)[0] - pp.predict(assign, q, C, L0, head_at=None)[0]
    gaps.append(("the model (predict_pool, rtt_trips 2)", gm))
    for name, g in gaps:
        b = (extra_bits / 1e3) / (g + (extra_bits / 1e3) / b0)
        print("   %-44s g0 %5.2f ms -> crossover %6.1f Mbit/s" % (name, g, b))
    print("   (the network study's measured crossover: %s Mbit/s; its re-priced prediction: %s)" % (
        ns["hypotheses"]["NS-CROSS"]["measured_mbps"], ns["hypotheses"]["NS-CROSS"]["repriced_mbps"]))


if __name__ == "__main__":
    main()
