#!/usr/bin/env python3
"""Register the network study on the four-node cluster again, as a NEW registration (ns4b; 2026-10-24), before it is measured.

ns4 (ns4_predictions_20261024.json, 04:17:50) is INCOMPLETE: every arm-run served, then failed qr3_tpot.py's served-host
check, because qr3_tpot.py took the expected host from a per-model table that lists only the large models, so TinyLlama
defaulted to the Pi 5 while the plan hosts it on the laptop. qr3_tpot.py now takes the host from the fixed plan (PLACE)
when one is given. Nothing else changes: the design, the arms, the order, the hypotheses and the predictions' method are
ns4's, recomputed here from a fresh memory snapshot. ns4's docstring follows.

Register the network study on the four-node cluster (topic-form scope 6; 2026-10-24), before it is measured.

    python3 testbed/scripts/register_ns4b_20261024.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

The 2026-10-08 study (pool_predictions_20261008netshape_r2.json) ran on the Pi 4 and the Jetson; the user's rule of
2026-10-21 puts it back on the four nodes. On the four nodes the laptop hosts every plan, so the question is how the
host's link moves a plan and where the output head belongs as the link slows. The split is fixed: TinyLlama Q4_0, the
laptop hosting blocks 0-10 and the Pi 4 holding 11-21 as an RPC worker, with the output head
  H  on the host (the laptop computes it; the Pi 4 returns its hidden state, 8 KB a token), or
  L  last, on the Pi 4 (the Pi 4 computes it and returns the logits, 32,000 x 4 = 128 KB a token).
The Pi 4's RPC link with the laptop (its port 50052 only) is shaped from the Pi 4 by netshape.sh, through
ns4_shape_20261024.sh once the server is healthy (the model's load is never shaped): U unshaped; R20 20 Mbit/s; R5 5
Mbit/s; D +5 ms each way (+10 ms round trip). Eight arms, HU LU H20 L20 H5 L5 HD LD, then LU HU again (drift), through
ab_session_20261024.sh with the Jetson and the Pi 5 idle and metered.
Predictions: predict_pool.predict on the split with the Pi 4's link set to the NOMINAL setting (its measured 934.0 Mbit/s
and 0.801 ms, link.json, replaced by the rate, plus twice the delay): ASSUMED that the shaped link delivers its nominal rate
(the 10-08 study priced measured shaped links). RTT1 (one round trip per worker per token, the rule adopted 2026-10-09);
the two-trip figure is recorded beside each prediction. Writes testbed/dataset/ns4b_predictions_20261024.json, read-only.
"""
import argparse, copy, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import qr3_tpot as Q                                        # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "ns4b_predictions_20261024.json")
KEY, CTX = "tinyllama_q4_0", 2048           # qr3_tpot.py serves the TinyLlama files its MODEL_FILE lists (Q4_0, not Q4_K_M)
ASSIGN = [("laptop", 11), ("pi4", 11)]
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}
SETTINGS = {"U": (0, 0), "20": (20, 0), "5": (5, 0), "D": (0, 5)}      # (rate Mbit/s, one-way delay ms)
SHAPE = "bash testbed/scripts/ns4_shape_20261024.sh"


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--reqs", type=int, default=6)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--order", default="HU LU H20 L20 H5 L5 HD LD LU HU"); ap.add_argument("--conditions", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    mem = json.load(open(a.memfile))
    e = P.plan_host_choice(KEY, [n for n, _ in ASSIGN], CTX, mem, host="laptop", load_mode="none")
    base = e["links"]["pi4"]
    arms, power = {}, EM.power_table()
    for s, (rate, delay) in SETTINGS.items():
        links = copy.deepcopy(e["links"])
        links["pi4"]["mbps"] = float(rate) if rate else base["mbps"]
        links["pi4"]["rtt_ms"] = base["rtt_ms"] + 2 * delay
        for head, head_at, hname in (("H", 0, "host"), ("L", None, "last")):
            m = P.memory(ASSIGN, e["q"], mem, e["devices"], e["model"], head_at=head_at)
            if min(v["margin_mib"] for v in m.values()) < 256:
                raise SystemExit("the split does not fit: %s" % m)
            d1, p1 = P.predict(ASSIGN, e["q"], e["C"], links, head_at=head_at, rtt_trips=1)
            d2, p2 = P.predict(ASSIGN, e["q"], e["C"], links, head_at=head_at, rtt_trips=2)
            en = EM.energy(ASSIGN, e["q"], e["C"], links, power, "tg", head_at=head_at, rtt_trips=1)
            idle = sum(power[n]["idle"] for n in OUTLETS if n not in dict(ASSIGN)) * en["T_ms"] / 1000
            k = head + ("U" if s == "U" else s)
            arm = {"what": "head %s, the Pi 4's link %s" % (hname, "unshaped" if s == "U" else ("%d Mbit/s" % rate if rate else
                                                                                               "+%d ms each way" % delay)),
                   "assign": ASSIGN, "head": hname, "setting": {"rate_mbit": rate, "delay_ms": delay},
                   "link_priced": {"mbps": links["pi4"]["mbps"], "rtt_ms": links["pi4"]["rtt_ms"]},
                   "place_env": {"PLACE": ",".join("%s:%d" % x for x in ASSIGN), "PLACE_HEAD": hname},
                   "serve_env": {"SERVE_DRAFT": "none"}, "nodes": [n for n, _ in ASSIGN],
                   "predicted": {"mean_ms_per_token": round(d1, 2), "server_mean_ms": round(d1, 2),
                                 "j_per_token_all": round(en["gross_j"] + idle, 4), "rtt_trips_2_ms": round(d2, 2),
                                 "prefill_ms": round(p1, 2)}}
            if s != "U":
                arm["pre"] = "nohup %s when-served %d %s \"$OUT/ns4_shape.log\" >> \"$OUT/ns4_shape.out\" 2>&1 &" % (SHAPE, rate, delay)
                arm["post"] = "[ -f \"$OUT/ns4_shape.log.pid\" ] && kill $(cat \"$OUT/ns4_shape.log.pid\") 2>/dev/null; %s clear" % SHAPE
            arms[k] = arm
    pr = lambda k: arms[k]["predicted"]["mean_ms_per_token"]
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_ns4b_20261024.py", "session": "testbed/scripts/ab_session_20261024.sh",
           "supersedes": "testbed/dataset/ns4_predictions_20261024.json (INCOMPLETE: every arm-run failed qr3_tpot.py's served-host check; qr3_tpot.py fixed to take the host from PLACE)",
           "scorer": "testbed/scripts/ab_score_20261024.py", "scorer_extra": "testbed/scripts/ns4_extra_20261024.py",
           "tag_prefix": "ns4b", "out_prefix": "ns4b",
           "check": "NETSHAPE_PEER=192.168.1.108 NETSHAPE_PORT=50052 bash testbed/scripts/netshape.sh check",
           "what": "the host's link and the output head's place on four nodes: TinyLlama Q4_0, laptop 11 + Pi 4 11, the Pi 4's RPC link "
                   "unshaped, 20 and 5 Mbit/s, and +10 ms round trip",
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1], "ctx": CTX,
           "nodes": [n for n, _ in ASSIGN], "outlets": OUTLETS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT),
           "base_link": base, "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN],
           "order": a.order.split(), **({"conditions": a.conditions} if a.conditions else {}), "cold_c": {"pi5": 50, "pi4": 55},
           "arms": arms,
           "hypotheses": dict(
               {"NS4-HOST-%s" % s: {"type": "lower", "metric": "mean_ms_per_token", "a": "H" + s, "b": "L" + s,
                                    "claim": "the head on the host is faster than the head last (predicted %.1f against %.1f ms)"
                                             % (pr("H" + s), pr("L" + s))} for s in SETTINGS},
               **{"NS4-RTT": {"type": "diff", "metric": "mean_ms_per_token", "a": "HD", "b": "HU", "lo": 7.0, "hi": 13.0,
                              "claim": "+10 ms of round trip adds one round trip per token (RTT1): HD - HU within 7-13 ms "
                                       "(two trips would add 20)"},
                  "NS4-ACC": {"type": "acc", "metric": "mean_ms_per_token", "tol": 0.15,
                              "claim": "each arm within 15 % of its prediction at the nominal setting"},
                  "NS4-DRIFT": {"type": "spread", "metric": "mean_ms_per_token", "arm": "HU", "tol": 0.03,
                                "claim": "HU's first and last arm-runs within 3 %"}}),
           "hypotheses_extra": {
               "NS4-GAP": {"claim": "the head-last penalty L - H grows as the rate falls: U < 20 Mbit/s < 5 Mbit/s"},
               "NS4-SHAPED": {"claim": "in every shaped arm-run the shaping was in place before its first scored request"}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
               "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py",
               "testbed/scripts/full_eval.py", "testbed/scripts/ab_session_20261024.sh", "testbed/scripts/ab_score_20261024.py",
               "testbed/scripts/ns4_extra_20261024.py", "testbed/scripts/ns4_shape_20261024.sh", "testbed/scripts/netshape.sh",
               "testbed/scripts/register_ns4b_20261024.py", "testbed/scripts/score_prima4_20261021.py",
               "testbed/dataset/link.json", "ThinkPad/config/cluster.json", os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    if a.dry_run:
        for k, v in arms.items():
            print(k, v["link_priced"], v["predicted"])
        return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))
    for k, v in arms.items():
        print(k, v["predicted"]["mean_ms_per_token"], "ms")


if __name__ == "__main__":
    main()
