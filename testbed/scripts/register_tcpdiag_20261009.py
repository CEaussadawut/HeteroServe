#!/usr/bin/env python3
"""Pre-register the TCP diagnostic (2026-10-09): why the logits cost more than size / rate on a slow link. Reads files only;
writes testbed/dataset/pool_predictions_20261009tcpdiag.json, read-only, never overwritten.

    python3 testbed/scripts/register_tcpdiag_20261009.py [--dry-run]

Why. The network study (2026-10-08) found the head-last layout's 128 KB of logits per token slower under shaping than their
size over the measured rate: 158.1 ms per token at 20 Mbit/s against 146.2 re-priced, while the head-on-host layout (8 KB
back) was only +4.3 ms over. Its write-up named a candidate, untested: TCP restarting its congestion window after each idle
gap between tokens (slow start after idle; tcp_slow_start_after_idle = 1 on both boards). That restart needs an idle gap
longer than the sender's RTO, at least 200 ms in Linux, and a token at 20 Mbit/s takes about 158 ms, so it is expected NOT
to be the cause; this diagnostic checks that, and records what the sender and the receiver do instead.

Cells: the network study's two layouts, TinyLlama-1.1B Q4_K_M, Pi 4 host (1 block) + Jetson (21, Vulkan): "last" (head on
the Jetson: 128 KB of logits back per token) and "host" (head on the Pi 4, -ot "^output\\.weight=CPU": 8 KB back), at b20
(netshape.sh apply 20 0) and s0 (unshaped), b20 first. Protocol -t 4 -p 64 -n 64 -r 3 -oe jsonl -v (64 decoded tokens a
run, for more bursts than the study's 16). Each setting's link is measured first (measure_link.sh). During every cell a
read-only sampler (ss_sample.sh, ss -tin every 0.05 s) records the RPC connection's tcp_info on the Jetson (the logits'
sender: sport 50052) and on the Pi 4 (the receiver: dport 50052). Nothing on either board is reconfigured beyond the
study's own shaping.
Hypothesis (scored as written, reported as found):
  TD-SSAI  slow start after idle is not what slows the logits: in tcp.tiny.last.b20's decode runs, the measured time per
           token is below the Jetson's median RTO on the connection (so no idle gap between tokens can reach it), and no
           Jetson sample taken with data in flight shows cwnd <= 10 (the restart window)
Exploratory, per cell, over the decode runs: the Jetson's retransmissions (the counter's increase), its cwnd and smoothed
RTT with data in flight and when idle, its delivery rate; the Pi 4's receive space, receive threshold and advertised window;
and the decode time against the prediction re-priced at the measured link (rtt_trips 2 and 1).
"""
import datetime, hashlib, json, os, stat, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as pp                                   # noqa: E402
from register_netshape_20261008 import nominal_link          # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "pool_predictions_20261009tcpdiag.json")
KEY, HOST, NODES = "tinyllama", "pi4", ["pi4", "jetson"]
ASSIGN = [("pi4", 1), ("jetson", 21)]
LAYOUTS = {"last": None, "host": 0}                          # head_at
SETTINGS = [("b20", 20, 0.0), ("s0", 0, 0.0)]                  # (id, rate Mbit/s or 0, one-way delay ms)
PROTOCOL = "-t 4 -p 64 -n 64 -r 3 -oe jsonl -v"
CODE = ("predict_pool.py", "run_pool.sh", "compare_pool.py", "netshape.sh", "measure_link.sh", "link_to_json.py",
        "ss_sample.sh", "tcpdiag_campaign_20261009.sh", "score_tcpdiag_20261009.py", "register_tcpdiag_20261009.py")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv):
    if os.path.exists(OUT) and "--dry-run" not in argv:
        sys.exit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    ds = os.path.join(ROOT, "testbed", "dataset")
    inputs = {f: os.path.join(ds, f) for f in ("characterisation.csv", "model_shapes.json", "link.json", "memory_at_rest.json")}
    shapes, pairs = json.load(open(inputs["model_shapes.json"])), json.load(open(inputs["link.json"]))["pairs"]
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
           "campaign": "the TCP diagnostic: why the logits cost more than size / rate on a shaped Pi 4 <-> Jetson link",
           "inputs_sha256": {**{f: sha(p) for f, p in inputs.items()}, **{f: sha(os.path.join(HERE, f)) for f in CODE}},
           "protocol": PROTOCOL, "base_link": {"pair": base.get("pair"), "mbps": base["mbps"], "rtt_ms": base["rtt_ms"]},
           "settings": [{"id": s, "rate_mbit": r, "delay_ms_one_way": d} for s, r, d in SETTINGS],
           "sampler": {"script": "ss_sample.sh", "interval_s": 0.05, "jetson": "( sport = :50052 )", "pi4": "( dport = :50052 )"},
           "run_order": [], "configs": {},
           "hypotheses": [
               {"id": "TD-SSAI", "claim": "in tcp.tiny.last.b20's decode runs, the measured time per token is below the "
                                          "Jetson's median RTO on the connection, and no Jetson sample with data in flight "
                                          "shows cwnd <= 10"}],
           "exploratory": ["the Jetson's retransmissions, cwnd and smoothed RTT (in flight and idle) and delivery rate",
                           "the Pi 4's rcv_space, rcv_ssthresh and rcv_wnd", "the decode time against the re-priced prediction"]}
    for i, (sid, rate, delay) in enumerate(SETTINGS):
        order = list(LAYOUTS) if i % 2 == 0 else list(reversed(list(LAYOUTS)))
        L = {"pi4": dict(L0["pi4"]), "jetson": dict(base, **nominal_link(base, rate, delay))}
        for lay in order:
            head_at = LAYOUTS[lay]
            dec, pre = pp.predict(ASSIGN, q, C, L, head_at=head_at)
            cid = "tcp.tiny.%s.%s" % (lay, sid)
            reg["run_order"].append(cid)
            bench = pp.bench_args(ASSIGN, KEY, head_at=head_at)
            bench["protocol"] = PROTOCOL
            reg["configs"][cid] = {
                "model": shape, "quant": quant, "setting": sid, "layout": lay, "rate_mbit": rate, "delay_ms_one_way": delay,
                "why": "TCP diagnostic 2026-10-09: setting %s, head %s" % (sid, "on the Jetson" if lay == "last" else "on the host"),
                "pipeline": [{"node": n, "blocks": k} for n, k in ASSIGN],
                "head_on": ASSIGN[0][0] if head_at == 0 else ASSIGN[-1][0], "head_at": head_at,
                "bench": bench, "memory": pp.memory(ASSIGN, q, mem, devs, mp, head_at=head_at),
                "link_nominal": L["jetson"],
                "predicted": {"decode_ms_per_token": round(dec, 1), "decode_tok_s": round(1000 / dec, 3),
                              "prefill_ms_per_token": round(pre, 1), "prefill_tok_s": round(1000 / pre, 2)},
                "cost_sources": {n: src[n] for n, _ in ASSIGN}}
    for cid in reg["run_order"]:
        x = reg["configs"][cid]
        print("%-18s link %7.1f Mbit/s %6.3f ms  decode %6.1f  prefill %5.1f  %s %s" % (
            cid, x["link_nominal"]["mbps"], x["link_nominal"]["rtt_ms"], x["predicted"]["decode_ms_per_token"],
            x["predicted"]["prefill_ms_per_token"], x["bench"]["args"], x["bench"]["protocol"]))
    if "--dry-run" in argv:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print("registered %s at %s, sha256 %s (read-only)" % (os.path.relpath(OUT, ROOT), reg["registered"], sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
