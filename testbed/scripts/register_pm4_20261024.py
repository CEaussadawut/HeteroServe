#!/usr/bin/env python3
"""Register the Jetson's power mode as a plan variable on the FOUR-node cluster (2026-10-24), before it is measured.

    python3 testbed/scripts/register_pm4_20261024.py --memfile testbed/dataset/memory_snapshot_<stamp>.json \\
        --c35 testbed/results/c35_<stamp> [--dry-run]

Scheduler upgrade 5 was measured on 2026-10-10 with the Pi 4 and the Jetson only (power_modes_predictions_20261010.json:
all six tests held, the saving small); the user's rule of 2026-10-21 puts every upgrade back on the four nodes. The plan is
the engine's four-node pick for Qwen3.6-35B-A3B Q6_K on order35b's ring order, split A of the characterisation
(register_c35_20261024.py), the head on the host. Three arms, by the 10-10 study's modes (power_modes_char_20261010.py):
  L  locked: MAXN with jetson_clocks, serve_pool.sh's CLOCKS=locked, as every run since 2026-09-30;
  U  unlocked: MAXN with the governors free (as found), CLOCKS=asfound;
  W  5 W: nvpmodel -m 1 and jetson_clocks at its maxima, CLOCKS=asfound, switched by pm4_mode_20261024.py once the
     laptop's llama-server is healthy (serve_pool.sh starts only in MAXN), before the warm-up ends; the switch is logged
     and checked against the first scored request by pm4_extra_20261024.py.
Order L U W W U L, through ab_session_20261024.sh (the as-found clocks stored first by the session's check, every arm's
post restoring them, also on any way out).
Predictions, INFORMED BY MEASUREMENTS and stated as such:
  T_L, E_L  split A as measured by the characterisation session (--c35: its two A arm-runs, the server mean gap and the
            gross J per decoded token at all four outlets), the Jetson locked there too;
  s         the Jetson's share of a token: k_J x its predicted busy time on A (the characterisation's fit), over the 10-10
            study's locked forward pass of its block (619.74 ms, pm_char_20261010_165919);
  T_m = T_L + dT_m x s x 619.74 ms, dT the 10-10 study's relative slowdown of that block (unlocked +1.464 %, 5 W +25.20 %:
            9.072 and 156.19 ms over 619.74);
  E_m = E_other x T_m / T_L + Pw_m x T_m + A_m x s, with the Jetson's waiting power Pw and per-block energy A fitted by the
        10-10 study per mode (L 3.2076 W / 2.9849 J, U 3.3114 / 2.5899, W 2.9679 / 2.0287) and E_other = E_L - (Pw_L T_L
        + A_L s), the other boards' energy taken to scale with the token's time (they wait for most of it): ASSUMED.
Writes testbed/dataset/pm4_predictions_20261024.json, read-only.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import qr3_tpot as Q                                        # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "pm4_predictions_20261024.json")
KEY, CTX = "qwen36moe_q6k", 4096
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}
FWD_L = 619.74                                              # ms, pm_char_20261010_165919: the locked forward pass, median
MODES = {"L": {"Pw": 3.2076, "A": 2.9849, "dT": 0.0},
         "U": {"Pw": 3.3114, "A": 2.5899, "dT": 9.072 / FWD_L},
         "W": {"Pw": 2.9679, "A": 2.0287, "dT": 156.1883 / FWD_L}}       # power_modes_predictions_20261010.json "predictions"
PMODE = "python3 testbed/scripts/pm4_mode_20261024.py"


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--c35", required=True)
    ap.add_argument("--reqs", type=int, default=6); ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--gen", type=int, default=64); ap.add_argument("--order", default="L U W W U L")
    ap.add_argument("--conditions", default=""); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    c35 = json.load(open(os.path.join(a.c35, "score.json")))
    creg = json.load(open(os.path.join(ROOT, c35["registration"])))
    if c35["code_changed_since_registration"]:
        raise SystemExit("the characterisation's code changed since its registration: %s" % c35["code_changed_since_registration"])
    A = c35["arms"].get("A") or {}
    fit = (c35.get("extra") or {}).get("k")
    if not (A.get("complete") and A.get("server_mean_ms") and A.get("j_per_token_all") and fit):
        raise SystemExit("the characterisation has no complete split A or no fit: %s" % {k: A.get(k) for k in ("complete", "server_mean_ms")})
    assign = [tuple(x) for x in creg["arms"]["A"]["assign"]]
    mem = json.load(open(a.memfile))
    e = P.plan_host_choice(KEY, [n for n, _ in assign], CTX, mem, host="laptop", load_mode="none", order_check=True)
    m = P.memory(assign, e["q"], mem, e["devices"], e["model"], head_at=0)
    low = {n: v["margin_mib"] for n, v in m.items() if v["margin_mib"] < 256}
    if low:
        raise SystemExit("split A does not fit today with 256 MiB to spare: %s" % low)
    T_L, E_L = A["server_mean_ms"], A["j_per_token_all"]
    s = fit["jetson"] * creg["arms"]["A"]["terms"]["busy_ms"]["jetson"] / FWD_L
    E_other = E_L - (MODES["L"]["Pw"] * T_L / 1000 + MODES["L"]["A"] * s)
    pred = {}
    for k, md in MODES.items():
        T = T_L + md["dT"] * s * FWD_L
        E = E_other * T / T_L + md["Pw"] * T / 1000 + md["A"] * s
        pred[k] = {"server_mean_ms": round(T, 2), "mean_ms_per_token": round(T, 2), "j_per_token_all": round(E, 4)}
    place = {"PLACE": ",".join("%s:%d" % x for x in assign), "PLACE_HEAD": "host"}
    arms = {
        "L": {"what": "the Jetson locked (MAXN + jetson_clocks), serve_pool.sh's own lock", "assign": assign, "head": "host",
              "place_env": place, "serve_env": {"SERVE_DRAFT": "none"}, "qr3_args": "--clocks locked",
              "pre": "%s set unlocked" % PMODE, "post": "%s set unlocked" % PMODE, "predicted": pred["L"]},
        "U": {"what": "the Jetson in MAXN with its governors free (as found)", "assign": assign, "head": "host",
              "place_env": place, "serve_env": {"SERVE_DRAFT": "none"}, "qr3_args": "--clocks asfound",
              "pre": "%s set unlocked" % PMODE, "post": "%s set unlocked" % PMODE, "predicted": pred["U"]},
        "W": {"what": "the Jetson in its 5 W mode (nvpmodel -m 1, jetson_clocks at its maxima), switched once the server is healthy",
              "assign": assign, "head": "host", "place_env": place, "serve_env": {"SERVE_DRAFT": "none"},
              "qr3_args": "--clocks asfound",
              "pre": "nohup %s set-when-served 5w --log \"$OUT/pm4_switch.log\" >> \"$OUT/pm4_switch.out\" 2>&1 &" % PMODE,
              "post": "[ -f \"$OUT/pm4_switch.log.pid\" ] && kill $(cat \"$OUT/pm4_switch.log.pid\") 2>/dev/null; %s restore" % PMODE,
              "predicted": pred["W"]}}
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_pm4_20261024.py", "session": "testbed/scripts/ab_session_20261024.sh",
           "scorer": "testbed/scripts/ab_score_20261024.py", "scorer_extra": "testbed/scripts/pm4_extra_20261024.py",
           "tag_prefix": "pm4", "out_prefix": "pm4", "check": "%s store" % PMODE,
           "what": "the Jetson's power mode (locked, governors free, 5 W) on the engine's four-node 35B plan, energy at the wall",
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1], "ctx": CTX,
           "nodes": [n for n, _ in assign], "outlets": OUTLETS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT),
           "memory": m, "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN], "order": a.order.split(),
           **({"conditions": a.conditions} if a.conditions else {}), "cold_c": {"pi5": 50, "pi4": 55},
           "informed_by": {"characterisation": c35["session"], "A_server_mean_ms": T_L, "A_j_per_token_all": E_L,
                           "k_jetson": fit["jetson"], "jetson_share_s": round(s, 5), "E_other_j": round(E_other, 4),
                           "modes_10_10": MODES, "fwd_locked_ms": FWD_L},
           "arms": arms,
           "hypotheses": {
               "PM4-T5": {"type": "lower", "metric": "server_mean_ms", "a": "L", "b": "W",
                          "claim": "the 5 W mode slows the token: L's server mean below W's (predicted %.1f against %.1f ms)"
                                   % (pred["L"]["server_mean_ms"], pred["W"]["server_mean_ms"])},
               "PM4-TU": {"type": "ratio", "metric": "server_mean_ms", "a": "U", "b": "L", "lo": 0.98, "hi": 1.02,
                          "claim": "freeing the governors moves the token's time by less than 2 %"},
               "PM4-E5": {"type": "ratio", "metric": "j_per_token_all", "a": "W", "b": "L", "lo": 0.99, "hi": 10.0,
                          "claim": "the 5 W mode does not save more than 1 %% of the cluster's energy per token (predicted W/L %.4f)"
                                   % (pred["W"]["j_per_token_all"] / pred["L"]["j_per_token_all"])},
               "PM4-ACC": {"type": "acc", "metric": "server_mean_ms", "tol": 0.05,
                           "claim": "each arm's server mean within 5 % of its prediction"},
               "PM4-EACC": {"type": "acc", "metric": "j_per_token_all", "tol": 0.05,
                            "claim": "each arm's gross J per decoded token within 5 % of its prediction"}},
           "hypotheses_extra": {"PM4-SWITCH": {"claim": "in every W arm-run the 5 W mode was verified before its first scored request"}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
               "testbed/scripts/predict_pool.py", "testbed/scripts/shelly_strip.py", "testbed/scripts/full_eval.py",
               "testbed/scripts/ab_session_20261024.sh", "testbed/scripts/ab_score_20261024.py",
               "testbed/scripts/pm4_extra_20261024.py", "testbed/scripts/pm4_mode_20261024.py",
               "testbed/scripts/power_modes_char_20261010.py", "testbed/scripts/spec_serve_20261010.py",
               "testbed/scripts/register_pm4_20261024.py", "testbed/scripts/score_prima4_20261021.py",
               "testbed/dataset/power_modes_predictions_20261010.json", os.path.join(c35["session"], "score.json"),
               "ThinkPad/config/cluster.json", os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    if a.dry_run:
        print(json.dumps({"informed_by": reg["informed_by"], "predicted": pred, "memory_margins": {n: v["margin_mib"] for n, v in m.items()}}, indent=1))
        return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s): %s" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16], pred))


if __name__ == "__main__":
    main()
