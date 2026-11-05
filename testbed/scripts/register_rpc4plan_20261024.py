#!/usr/bin/env python3
"""Register the RPC link term on the four-node cluster (study rpc4, step 2; 2026-10-24), from step 1's measured bytes, before
the shaped runs it predicts.

    python3 testbed/scripts/register_rpc4plan_20261024.py --char testbed/results/rpc4char_<stamp> \
        --memfile testbed/dataset/memory_snapshot_<stamp>.json [--conditions TEXT] [--dry-run]

Step 1 (rpc4_char_20261024.py, its score.json) measured, per decoded token over the Pi 4's RPC link, the payload and the
data segments each way for three plans of TinyLlama Q4_0 on laptop 11 + Pi 4 11: H (the head on the host), L (the head
last, on the Pi 4) and LB (the head last with --backend-sampling). This step prices the same plans on links shaped to rates
never measured before, 10 and 3 Mbit/s (ns4c and bs4 used 20 and 5), two ways:
  new  predict_pool's unshaped decode for the plan (RTT1, the link at its measured gigabit rate), less the transfer it
       already charges there (the tensors it counts, at that rate), plus step 1's bytes on the wire at the shaped rate:
       (payload + 66 B of Ethernet, IP and TCP headers per data segment, both ways) / rate. LB is priced from L's compute (the
       Pi 4 still computes the head) with LB's own bytes. ASSUMED: the two directions do not overlap (request, then
       response), the ACKs cost nothing, and netem charges every frame's length at its rate.
  old  predict_pool as registered before (ns4c, bs4): the tensors it counts, at the nominal rate; LB = L less the logits.
Arms H10 L10 LB10 H3 L3 LB3, then H10 again; six requests of 128 + 64 per arm-run after qr3's warm-up, through
ab_session_20261024.sh (the shaping: ns4c_shape_20261024.sh, the overlay flow, its netem counters kept); scored by
ab_score_20261024.py with rpc4_extra_20261024.py (RP-SHAPED, RP-BETTER). Writes testbed/dataset/rpc4plan_predictions_20261024.json,
read-only.
"""
import argparse, copy, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import qr3_tpot as Q                                        # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "rpc4plan_predictions_20261024.json")
KEY, CTX = "tinyllama_q4_0", 2048
ASSIGN = [("laptop", 11), ("pi4", 11)]
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}
RATES = (10, 3)
SHAPE = "bash testbed/scripts/ns4c_shape_20261024.sh"
BS = "--backend-sampling"
HDR = 66                                                    # Ethernet 14 + IPv4 20 + TCP 20 + timestamps 12, per data segment


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", required=True); ap.add_argument("--memfile", required=True)
    ap.add_argument("--reqs", type=int, default=6); ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--gen", type=int, default=64); ap.add_argument("--order", default="H10 L10 LB10 H3 L3 LB3 H10")
    ap.add_argument("--conditions", default=""); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    char = os.path.join(os.path.abspath(a.char), "score.json")
    sc = json.load(open(char))
    if not all(sc["plans"].get(p, {}).get("complete") for p in ("H", "L", "LB")):
        raise SystemExit("refused: step 1 is not complete for every plan: %s" % {p: x.get("complete") for p, x in sc["plans"].items()})
    if sc["code_changed_since_registration"]:
        raise SystemExit("refused: step 1's code changed since its registration: %s" % sc["code_changed_since_registration"])
    mem = json.load(open(a.memfile))
    e = P.plan_host_choice(KEY, [n for n, _ in ASSIGN], CTX, mem, host="laptop", load_mode="none")
    base, q = e["links"]["pi4"], e["q"]
    n_vocab = (q["head_params"] - q["n_embd"]) / q["n_embd"]
    bwg = base["mbps"] * 1e6 / 8 / 1000                     # bytes per ms at the measured gigabit rate
    counted = {"H": q["activation_bytes"] * 2, "L": q["activation_bytes"] + 4 * n_vocab, "LB": q["activation_bytes"] + 4 * n_vocab}
    wire = {p: sc["plans"][p]["per_token"]["bytes_acked"] + sc["plans"][p]["per_token"]["bytes_received"]
            + HDR * (sc["plans"][p]["per_token"]["data_segs_out"] + sc["plans"][p]["per_token"]["data_segs_in"])
            for p in ("H", "L", "LB")}
    power = EM.power_table()
    arms = {}
    for plan, head_at, hname in (("H", 0, "host"), ("L", None, "last"), ("LB", None, "last")):
        m = P.memory(ASSIGN, q, mem, e["devices"], e["model"], head_at=head_at)
        if min(v["margin_mib"] for v in m.values()) < 256:
            raise SystemExit("the split does not fit: %s" % m)
        dg, pg = P.predict(ASSIGN, q, e["C"], e["links"], head_at=head_at, rtt_trips=1)     # unshaped, as the predictor prices it
        for rate in RATES:
            links = copy.deepcopy(e["links"]); links["pi4"]["mbps"] = float(rate)
            bw = rate * 1e6 / 8 / 1000
            d_old, p_old = P.predict(ASSIGN, q, e["C"], links, head_at=head_at, rtt_trips=1)
            if plan == "LB":
                d_old -= (4 * n_vocab - 4) / bw                                            # bs4's model of the option
            d_new = dg - counted[plan] / bwg + wire[plan] / bw
            en = EM.energy(ASSIGN, q, e["C"], links, power, "tg", head_at=head_at, rtt_trips=1)
            idle = sum(power[n]["idle"] for n in OUTLETS if n not in dict(ASSIGN)) * en["T_ms"] / 1000
            j_old = (en["gross_j"] + idle) * (d_old / en["T_ms"]) if plan == "LB" else en["gross_j"] + idle
            k = "%s%d" % (plan, rate)
            arm = {"what": "head %s on %s%s, the Pi 4's link %d Mbit/s" % (hname, "the host" if head_at == 0 else "the Pi 4",
                                                                              ", backend sampling" if plan == "LB" else "", rate),
                   "assign": ASSIGN, "head": hname, "setting": {"rate_mbit": rate, "delay_ms": 0},
                   "place_env": {"PLACE": ",".join("%s:%d" % x for x in ASSIGN), "PLACE_HEAD": hname},
                   "serve_env": dict({"SERVE_DRAFT": "none"}, **({"SERVE_EXTRA": BS} if plan == "LB" else {})),
                   "nodes": [n for n, _ in ASSIGN],
                   "terms": {"unshaped_predicted_ms": round(dg, 3), "counted_bytes": counted[plan],
                             "counted_at_gigabit_ms": round(counted[plan] / bwg, 4), "wire_bytes_per_token": round(wire[plan], 1),
                             "wire_at_rate_ms": round(wire[plan] / bw, 3)},
                   "predicted": {"mean_ms_per_token": round(d_new, 2), "server_mean_ms": round(d_new, 2),
                                 "j_per_token_all": round(j_old * d_new / d_old, 4), "prefill_ms": round(p_old, 2)},
                   "predicted_old": {"mean_ms_per_token": round(d_old, 2)},
                   "pre": "nohup %s when-served %d 0 \"$OUT/ns4_shape.log\" >> \"$OUT/ns4_shape.out\" 2>&1 &" % (SHAPE, rate),
                   "post": ("[ -f \"$OUT/ns4_shape.log.pid\" ] && kill $(cat \"$OUT/ns4_shape.log.pid\") 2>/dev/null; "
                            "%s counters \"$OUT/ns4c_counters.log\"; %s clear" % (SHAPE, SHAPE))}
            arms[k] = arm
    pr = lambda k: arms[k]["predicted"]["mean_ms_per_token"]
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_rpc4plan_20261024.py", "session": "testbed/scripts/ab_session_20261024.sh",
           "scorer": "testbed/scripts/ab_score_20261024.py", "scorer_extra": "testbed/scripts/rpc4_extra_20261024.py",
           "tag_prefix": "rpc4", "out_prefix": "rpc4plan",
           "check": "NETSHAPE_PEER=192.168.10.5 NETSHAPE_PORT=50052 bash testbed/scripts/netshape.sh check",
           "what": "the RPC link term on four nodes: TinyLlama Q4_0, laptop 11 + Pi 4 11, the head on the host (H), last (L) or "
                   "last with backend sampling (LB), the Pi 4's RPC link at 10 and 3 Mbit/s, priced from step 1's measured "
                   "bytes per token",
           "char": os.path.relpath(char, ROOT), "char_sha256": sha(os.path.relpath(char, ROOT)),
           "wire_bytes_per_token": {p: round(w, 1) for p, w in wire.items()}, "counted_bytes_per_token": counted,
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1], "ctx": CTX,
           "nodes": [n for n, _ in ASSIGN], "outlets": OUTLETS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT),
           "base_link": base, "n_vocab": n_vocab, "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN],
           "order": a.order.split(), **({"conditions": a.conditions} if a.conditions else {}), "cold_c": {"pi5": 50, "pi4": 55},
           "arms": arms,
           "hypotheses": {
               "RP-ACC": {"type": "acc", "metric": "mean_ms_per_token", "tol": 0.10,
                          "claim": "each arm within 10 % of its prediction from the measured bytes"},
               "RP-HOST-3": {"type": "lower", "metric": "mean_ms_per_token", "a": "H3", "b": "L3",
                             "claim": "at 3 Mbit/s the head on the host is faster than the head last (predicted %.1f against %.1f ms)"
                                      % (pr("H3"), pr("L3"))},
               "RP-LB-3": {"type": "lower", "metric": "mean_ms_per_token", "a": "LB3", "b": "L3",
                           "claim": "at 3 Mbit/s backend sampling is faster than the head last without it (predicted %.1f against "
                                    "%.1f ms)" % (pr("LB3"), pr("L3"))},
               "RP-DRIFT": {"type": "spread", "metric": "mean_ms_per_token", "arm": "H10", "tol": 0.03,
                            "claim": "H10's first and last arm-runs within 3 %"}},
           "hypotheses_extra": {
               "RP-SHAPED": {"claim": "in every arm-run the shaping was in place before its first scored request, and at least "
                                      "1,000 packets went through netem in each direction"},
               "RP-BETTER": {"claim": "in every arm the new term's error is smaller in magnitude than the old term's"}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
               "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py",
               "testbed/scripts/full_eval.py", "testbed/scripts/ab_session_20261024.sh", "testbed/scripts/ab_score_20261024.py",
               "testbed/scripts/rpc4_extra_20261024.py", "testbed/scripts/ns4c_shape_20261024.sh", "testbed/scripts/netshape.sh",
               "testbed/scripts/register_rpc4plan_20261024.py", "testbed/scripts/rpc4_char_20261024.py",
               "testbed/scripts/score_prima4_20261021.py", "testbed/dataset/link.json", "ThinkPad/config/cluster.json",
               os.path.relpath(char, ROOT), os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    if a.dry_run:
        for k, v in arms.items():
            print(k, "new %.1f  old %.1f  terms %s" % (v["predicted"]["mean_ms_per_token"], v["predicted_old"]["mean_ms_per_token"], v["terms"]))
        return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))
    for k, v in arms.items():
        print(k, "new", v["predicted"]["mean_ms_per_token"], "old", v["predicted_old"]["mean_ms_per_token"])


if __name__ == "__main__":
    main()
