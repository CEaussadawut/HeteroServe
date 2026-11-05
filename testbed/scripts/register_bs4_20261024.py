#!/usr/bin/env python3
"""Register sampling on the head's board (scheduler upgrade 6) on the four-node cluster (2026-10-24), before it is measured.

    python3 testbed/scripts/register_bs4_20261024.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

Upgrade 6 was measured on 2026-10-10 with the Pi 4 and the Jetson only (head_bs_predictions_20261010b.json); the user's rule of
2026-10-21 puts every upgrade back on the four nodes. With llama-server's --backend-sampling the node that holds the output
head samples the next token itself, so only the token comes back to the host instead of the logits (32,000 x 4 = 128 KB a
token for TinyLlama). The plan is ns4c's head-last arm: TinyLlama Q4_0, the laptop hosting blocks 0-10 and the Pi 4 holding
11-21 and the head, served through serve_pool.sh with SERVE_EXTRA="--backend-sampling" for the B arms (the flag added after
the night's sessions by apply_serve_extra_20261024.py; this script refuses if it is not there). Four arms:
  LU  head last, unshaped;  LBU the same with backend sampling;
  L5  head last, the Pi 4's RPC link shaped to 5 Mbit/s (ns4c_shape_20261024.sh: the overlay flow, the netem counters kept);
  LB5 the same with backend sampling.
Order LU LBU L5 LB5 LB5 L5 LBU LU, six requests of 128 + 64 per arm-run, through ab_session_20261024.sh; scored by
ab_score_20261024.py with scorer_extra bs4_extra_20261024.py (BS4-SHAPED).
Predictions: L arms by predict_pool.predict at the link's nominal rate (RTT1), as ns4c; B arms the same less the logits' transfer
the predictor charges the head-last worker, 4 x n_vocab bytes at the link's rate (the sampled token's 4 bytes left in): ASSUMED,
backend sampling changes nothing else (the Pi 4 still computes the head). Energy per token: the L arm's prediction scaled by the
B arm's time, ASSUMED (the saved time is all link wait). Writes testbed/dataset/bs4_predictions_20261024.json, read-only.
"""
import argparse, copy, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import qr3_tpot as Q                                        # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "bs4_predictions_20261024.json")
KEY, CTX = "tinyllama_q4_0", 2048
ASSIGN = [("laptop", 11), ("pi4", 11)]
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}
SETTINGS = {"U": 0, "5": 5}                                 # the Pi 4's RPC link: unshaped, or a rate in Mbit/s
SHAPE = "bash testbed/scripts/ns4c_shape_20261024.sh"
BS = "--backend-sampling"


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
    ap.add_argument("--order", default="LU LBU L5 LB5 LB5 L5 LBU LU"); ap.add_argument("--conditions", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    if "SERVE_EXTRA" not in open(os.path.join(HERE, "serve_pool.sh"), encoding="utf-8").read():
        raise SystemExit("refused: serve_pool.sh has no SERVE_EXTRA yet (run apply_serve_extra_20261024.py when no session uses it)")
    mem = json.load(open(a.memfile))
    e = P.plan_host_choice(KEY, [n for n, _ in ASSIGN], CTX, mem, host="laptop", load_mode="none")
    base = e["links"]["pi4"]
    q = e["q"]
    n_vocab = (q["head_params"] - q["n_embd"]) / q["n_embd"]
    m = P.memory(ASSIGN, q, mem, e["devices"], e["model"], head_at=None)
    if min(v["margin_mib"] for v in m.values()) < 256:
        raise SystemExit("the split does not fit: %s" % m)
    arms, power = {}, EM.power_table()
    for s, rate in SETTINGS.items():
        links = copy.deepcopy(e["links"])
        links["pi4"]["mbps"] = float(rate) if rate else base["mbps"]
        d1, p1 = P.predict(ASSIGN, q, e["C"], links, head_at=None, rtt_trips=1)
        en = EM.energy(ASSIGN, q, e["C"], links, power, "tg", head_at=None, rtt_trips=1)
        idle = sum(power[n]["idle"] for n in OUTLETS if n not in dict(ASSIGN)) * en["T_ms"] / 1000
        j1 = en["gross_j"] + idle
        bw = links["pi4"]["mbps"] * 1e6 / 8 / 1000          # bytes per ms
        saving = (4 * n_vocab - 4) / bw                     # the logits, less the sampled token's 4 bytes
        for b, name in (("", "L"), ("B", "LB")):
            k = name + s
            dd = d1 - (saving if b else 0.0)
            arm = {"what": "head last on the Pi 4%s, its link %s" % (", backend sampling" if b else "",
                                                                    "unshaped" if not rate else "%d Mbit/s" % rate),
                   "assign": ASSIGN, "head": "last", "setting": {"rate_mbit": rate, "delay_ms": 0},
                   "link_priced": {"mbps": links["pi4"]["mbps"], "rtt_ms": links["pi4"]["rtt_ms"]},
                   "place_env": {"PLACE": ",".join("%s:%d" % x for x in ASSIGN), "PLACE_HEAD": "last"},
                   "serve_env": dict({"SERVE_DRAFT": "none"}, **({"SERVE_EXTRA": BS} if b else {})),
                   "nodes": [n for n, _ in ASSIGN],
                   "predicted": {"mean_ms_per_token": round(dd, 2), "server_mean_ms": round(dd, 2),
                                 "j_per_token_all": round(j1 * dd / d1, 4), "logits_saving_ms": round(saving, 3) if b else 0.0,
                                 "prefill_ms": round(p1, 2)}}
            if rate:
                arm["pre"] = "nohup %s when-served %d 0 \"$OUT/ns4_shape.log\" >> \"$OUT/ns4_shape.out\" 2>&1 &" % (SHAPE, rate)
                arm["post"] = ("[ -f \"$OUT/ns4_shape.log.pid\" ] && kill $(cat \"$OUT/ns4_shape.log.pid\") 2>/dev/null; "
                               "%s counters \"$OUT/ns4c_counters.log\"; %s clear" % (SHAPE, SHAPE))
            arms[k] = arm
    pr = lambda k: arms[k]["predicted"]["mean_ms_per_token"]
    sv = lambda s: pr("L" + s) - pr("LB" + s)
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_bs4_20261024.py", "session": "testbed/scripts/ab_session_20261024.sh",
           "scorer": "testbed/scripts/ab_score_20261024.py", "scorer_extra": "testbed/scripts/bs4_extra_20261024.py",
           "tag_prefix": "bs4", "out_prefix": "bs4",
           "check": "NETSHAPE_PEER=192.168.10.5 NETSHAPE_PORT=50052 bash testbed/scripts/netshape.sh check",
           "what": "sampling on the head's board on four nodes: TinyLlama Q4_0, laptop 11 + Pi 4 11 with the head on the Pi 4, "
                   "with and without --backend-sampling, the Pi 4's RPC link unshaped and at 5 Mbit/s",
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1], "ctx": CTX,
           "nodes": [n for n, _ in ASSIGN], "outlets": OUTLETS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT),
           "base_link": base, "n_vocab": n_vocab, "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN],
           "order": a.order.split(), **({"conditions": a.conditions} if a.conditions else {}), "cold_c": {"pi5": 50, "pi4": 55},
           "arms": arms,
           "hypotheses": {
               "BS4-FASTER-5": {"type": "lower", "metric": "mean_ms_per_token", "a": "LB5", "b": "L5",
                                "claim": "at 5 Mbit/s backend sampling is faster (predicted %.1f against %.1f ms)" % (pr("LB5"), pr("L5"))},
               "BS4-SAVE-5": {"type": "diff", "metric": "mean_ms_per_token", "a": "L5", "b": "LB5",
                              "lo": round(0.8 * sv("5"), 1), "hi": round(1.2 * sv("5"), 1),
                              "claim": "at 5 Mbit/s the saving is the logits' transfer within 20 %% (predicted %.1f ms)" % sv("5")},
               "BS4-SAVE-U": {"type": "diff", "metric": "mean_ms_per_token", "a": "LU", "b": "LBU", "lo": -1.0, "hi": 3.0,
                              "claim": "unshaped the saving is small: LU - LBU within -1 to +3 ms (predicted %.2f)" % sv("U")},
               "BS4-ENERGY-5": {"type": "lower", "metric": "j_per_token_all", "a": "LB5", "b": "L5",
                                "claim": "at 5 Mbit/s backend sampling uses less energy per decoded token"},
               "BS4-TEXT": {"type": "text_across", "arms": ["LU", "LBU", "L5", "LB5"],
                            "claim": "request 0's first 16 tokens (greedy) are the same in every arm-run, with and without"},
               "BS4-ACC": {"type": "acc", "metric": "mean_ms_per_token", "tol": 0.15,
                           "claim": "each arm within 15 % of its prediction"},
               "BS4-DRIFT": {"type": "spread", "metric": "mean_ms_per_token", "arm": "LU", "tol": 0.03,
                             "claim": "LU's first and last arm-runs within 3 %"}},
           "hypotheses_extra": {
               "BS4-SHAPED": {"claim": "in every shaped arm-run the shaping was in place before its first scored request, and at "
                                       "least 1,000 packets went through netem in each direction"}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
               "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py",
               "testbed/scripts/full_eval.py", "testbed/scripts/ab_session_20261024.sh", "testbed/scripts/ab_score_20261024.py",
               "testbed/scripts/bs4_extra_20261024.py", "testbed/scripts/ns4c_shape_20261024.sh", "testbed/scripts/netshape.sh",
               "testbed/scripts/register_bs4_20261024.py", "testbed/scripts/score_prima4_20261021.py",
               "testbed/dataset/link.json", "ThinkPad/config/cluster.json", os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    if a.dry_run:
        for k, v in arms.items():
            print(k, v["serve_env"], v["link_priced"], v["predicted"])
        print({k: (h.get("lo"), h.get("hi")) for k, h in reg["hypotheses"].items() if h["type"] == "diff"})
        return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))
    for k, v in arms.items():
        print(k, v["predicted"]["mean_ms_per_token"], "ms")


if __name__ == "__main__":
    main()
