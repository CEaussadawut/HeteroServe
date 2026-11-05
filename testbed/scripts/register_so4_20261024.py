#!/usr/bin/env python3
"""Register the switch-off test on the four-node cluster (study so4, 2026-10-24; the user, 14:44: "do the switch-off after
this"), before it is measured.

    python3 testbed/scripts/register_so4_20261024.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--conditions TEXT] [--dry-run]

The question: does switching off a board that the plan leaves idle save what the energy model says? Scheduler v2 serves
Qwen3-32B Q6_K on the laptop, the Pi 5 and the Jetson and leaves the Pi 4 out, yet the Pi 4 stays powered at its outlet
(2.7-3.6 J per token in the Q6_K sessions, 5-6 % of the cluster). Two arms on the same plan, planned here as spec4d planned
(plan_host_choice, order_check, the host the laptop, its serve-time reserve plus 1,024 MiB of headroom), no draft:
  A  the Pi 4 on and idle, as in every run so far;
  B  the Pi 4 powered off cleanly and its outlet switched off before the arm-run (strip_switch_20261024.py off-node), and
     switched back on after it, the board up again with its clock synchronized and its overlay address (on-node).
Order A B B A, three requests of 128 + 64 per arm-run, through ab_session_20261024.sh with energy at all four outlets; scored
by ab_score_20261024.py with so4_extra_20261024.py (SO-ZERO, SO-BACK). It refuses if the plan uses the Pi 4.
Predictions: the time per token is predict_pool's for both arms. Energy per token: energy_model.energy over the plan's nodes,
plus each idle outlet's idle power for the token's time: in A the Pi 4's included, in B left out (its outlet off: 0 W).
The saving B - A is therefore the Pi 4's idle wall power times the time per token. Writes
testbed/dataset/so4_predictions_20261024.json, read-only.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import qr3_tpot as Q                                        # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "so4_predictions_20261024.json")
KEY, CTX, NODES = "qwen3_32b_q6k", 2048, ["pi5", "pi4", "jetson", "laptop"]
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}
RESERVE, HEADROOM_MIB = 7168, 1024
SW = "python3 testbed/scripts/strip_switch_20261024.py"


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--reqs", type=int, default=3)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--order", default="A B B A"); ap.add_argument("--conditions", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    os.environ["HETEROSERVE_RESERVE_MB"] = "laptop=%d" % (RESERVE + HEADROOM_MIB)
    mem = json.load(open(a.memfile))
    e = P.plan_host_choice(KEY, NODES, CTX, mem, host="laptop", load_mode="none", order_check=True)
    used = [n for n, _ in e["assign"]]
    if "pi4" in used:
        raise SystemExit("refused: today's plan uses the Pi 4 (%s); this test needs a plan that leaves it idle" % e["assign"])
    m = P.memory(e["assign"], e["q"], mem, e["devices"], e["model"], head_at=e["head_at"])
    if min(v["margin_mib"] for v in m.values()) < 0:
        raise SystemExit("refused: the plan does not fit at rest: %s" % m)
    head = "host" if e["head_at"] == 0 else "last"
    place = {"PLACE": ",".join("%s:%d" % tuple(x) for x in e["assign"]), "PLACE_HEAD": head}
    power = EM.power_table()
    en = EM.energy(e["assign"], e["q"], e["C"], e["links"], power, "tg", head_at=e["head_at"], rtt_trips=e["rtt_trips"])
    t = e["decode_ms"]
    idle_out = {n: power[n]["idle"] for n in OUTLETS if n not in used}         # watts at the wall, each idle outlet
    j_a = en["gross_j"] + sum(idle_out.values()) * en["T_ms"] / 1000
    j_b = en["gross_j"] + sum(w for n, w in idle_out.items() if n != "pi4") * en["T_ms"] / 1000
    save = j_a - j_b
    common = {"assign": e["assign"], "head": head, "place_env": place, "nodes": used,
              "serve_env": {"SERVE_DRAFT": "none", "HETEROSERVE_RESERVE_MB": "laptop=%d" % RESERVE}}
    arms = {"A": dict(common, what="the Pi 4 on and idle (the plan leaves it out)",
                      predicted={"mean_ms_per_token": round(t, 1), "j_per_token_all": round(j_a, 4)}),
            "B": dict(common, what="the Pi 4 powered off cleanly and its outlet switched off for the arm-run",
                      predicted={"mean_ms_per_token": round(t, 1), "j_per_token_all": round(j_b, 4)},
                      pre="%s off-node pi4 --log \"$OUT/so4_power.log\"" % SW,
                      post="%s on-node pi4 --log \"$OUT/so4_power.log\"" % SW)}
    memfile = os.path.relpath(os.path.abspath(a.memfile), ROOT)
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_so4_20261024.py", "session": "testbed/scripts/ab_session_20261024.sh",
           "scorer": "testbed/scripts/ab_score_20261024.py", "scorer_extra": "testbed/scripts/so4_extra_20261024.py",
           "tag_prefix": "so4", "out_prefix": "so4",
           "what": "the switch-off test on four nodes: scheduler v2's Qwen3-32B Q6_K plan, which leaves the Pi 4 out, with the Pi 4 "
                   "on and idle (A) or powered off at its outlet (B); energy at all four outlets",
           "plan_options": {"order_check": True, "load_mode": "none", "host": "laptop",
                            "planning_reserve_mb": "laptop=%d" % (RESERVE + HEADROOM_MIB), "headroom_mib": HEADROOM_MIB},
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1], "ctx": CTX, "nodes": NODES,
           "outlets": OUTLETS, "memfile": memfile, "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN],
           "order": a.order.split(), **({"conditions": a.conditions} if a.conditions else {}), "cold_c": {"pi5": 50, "pi4": 55},
           "energy_terms": {"plan_gross_j": round(en["gross_j"], 4), "T_ms": round(en["T_ms"], 2),
                            "idle_outlets_w": idle_out, "pi4_idle_w": power["pi4"]["idle"], "saving_j_per_token": round(save, 4)},
           "memory_at_rest_mib": {n: v["margin_mib"] for n, v in m.items()},
           "arms": arms,
           "hypotheses": {
               "SO-SAVE": {"type": "diff", "metric": "j_per_token_all", "a": "A", "b": "B",
                           "lo": round(0.7 * save, 3), "hi": round(1.3 * save, 3),
                           "claim": "switching the idle Pi 4 off saves its idle wall power times the time per token, within 30 %% "
                                    "(predicted %.2f J per token)" % save},
               "SO-TIME": {"type": "ratio", "metric": "mean_ms_per_token", "a": "B", "b": "A", "lo": 0.98, "hi": 1.02,
                           "claim": "the time per token is unchanged (B / A within 0.98-1.02): the Pi 4 is not in the plan"},
               "SO-ACC": {"type": "acc", "metric": "mean_ms_per_token", "tol": 0.15,
                          "claim": "each arm within 15 %% of the predicted %.1f ms per token" % t},
               "SO-DRIFT": {"type": "spread", "metric": "mean_ms_per_token", "arm": "A", "tol": 0.03,
                            "claim": "A's first and last arm-runs within 3 %"},
               "SO-TEXT": {"type": "text_across", "arms": ["A", "B"],
                           "claim": "request 0's first 16 tokens (greedy) are the same in every arm-run"}},
           "hypotheses_extra": {
               "SO-ZERO": {"claim": "in every B arm-run the Pi 4's outlet reads off and 0 W in every strip sample of its requests"},
               "SO-BACK": {"claim": "after every B arm-run the Pi 4 is back: a new boot id, its overlay address up and its clock "
                                    "synchronized, within 10 minutes of its outlet switching on"}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
               "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py",
               "testbed/scripts/full_eval.py", "testbed/scripts/ab_session_20261024.sh", "testbed/scripts/ab_score_20261024.py",
               "testbed/scripts/so4_extra_20261024.py", "testbed/scripts/strip_switch_20261024.py",
               "testbed/scripts/register_so4_20261024.py", "testbed/scripts/score_prima4_20261021.py",
               "testbed/energy/strip.json", "testbed/dataset/link.json", "ThinkPad/config/cluster.json", memfile)}}
    if a.dry_run:
        print("plan", e["assign"], "head", head, "decode %.1f ms" % t, "memory at rest", reg["memory_at_rest_mib"])
        print("energy: plan %.3f J, idle outlets %s, A %.3f, B %.3f, saving %.3f J/token" % (en["gross_j"], idle_out, j_a, j_b, save))
        print({k: (h.get("lo"), h.get("hi")) for k, h in reg["hypotheses"].items() if h["type"] in ("diff", "ratio")})
        return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))
    for k, v in arms.items():
        print(k, v["predicted"])


if __name__ == "__main__":
    main()
