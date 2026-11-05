#!/usr/bin/env python3
"""Register ph2 (2026-10-28; after ph1): per-model power rows for the phase scheduler, and ph1's turbo explanation, tested
on held-out prompt lengths. Registered before it runs.

    python3 testbed/scripts/register_ph2_20261028.py [--dry-run]

ph1 found every pick right but its energies 3-14 % above prediction, and the hand-off's first token late by 6-8 % at two
lengths. Read after the data: the laptop's prefill draws ~41 W for its first ~20 s of load and ~31 W after (its CPU's
short-term turbo budget), it idles at ~17 W while serving, and it prefills up to 18 % slower right after a request that kept
it busy. phase_power_20261028.py fitted power rows per phase on ph1's strip data (testbed/dataset/phase_power_20261028.json);
in-sample they price ph1's twelve arms within -5.8 to +2.3 % (the table: +3.3 to +14.3 %). This tests both on new data:
  prompts of 1,536 and 6,144 tokens (held out: ph1 used 1,024-8,192 in powers of two, the calibration 512-8,192), 256
  generated, Qwen3-0.6B Q8_0, plans L (laptop), J (Jetson), H (hand-off), two repetitions, interleaved; and 60 s between
  requests (ph1: 10 s), so the laptop's turbo budget refills before each one.
Times are phase_plan_20261028's (the calibration, unchanged); energies are phase_power_20261028.energy() with the new rows,
and, for comparison, phase_plan's with the TinyLlama table (energy_model.power_table()).
Hypotheses (registered; phase_study_20261028.py score):
  PH-PICK-E  at each n, the least predicted energy (rows) is the least measured (mean of two)
  PH-PICK-T  the same for total time
  PH-TIME    every arm's mean total time within 10 % of its prediction
  PH-ENERGY  every arm's mean energy within 7 % of the rows' prediction (above the in-sample maximum, 5.8 %, which came
             with a 4.9 % time error)
  PH-BETTER  the rows' |error| below the table's on every arm
  PH-TTFT    at each n, H's time to first token within 3 % of L's (the budget refilled before each request)
  PH-REPEAT  every arm's two total times within 3 % of each other
Writes testbed/dataset/ph2_predictions_20261028.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import phase_plan_20261028 as PP                           # noqa: E402
import phase_power_20261028 as W                           # noqa: E402
import energy_model as EM                                   # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "ph2_predictions_20261028.json")
NS, GEN, PAUSE = [1536, 6144], 256, 60
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}
QUESTION_TOKENS = 12


def sha(p):
    h = hashlib.sha256()
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    costs, rows, table = PP.load(), W.load()["rows"], EM.power_table()
    arms, picks = {}, {"energy": {}, "time": {}}
    for n in NS:
        ne = n + QUESTION_TOKENS
        pr = {pl: W.energy(costs, pl, ne, GEN, rows) for pl in PP.PLANS}
        pt = {pl: PP.plan(costs, pl, ne, GEN, table) for pl in PP.PLANS}
        picks["energy"][str(n)] = min(pr, key=lambda k: pr[k]["energy_j"])
        picks["time"][str(n)] = min(pr, key=lambda k: pr[k]["total_s"])
        for pl in PP.PLANS:
            p = pr[pl]
            arms["n%d_%s" % (n, pl)] = {"n": n, "gen": GEN, "plan": pl, "predicted": {
                "ttft_s": round(p["ttft_s"], 4), "decode_s": round(p["decode_s"], 4), "total_s": round(p["total_s"], 4),
                "handoff_s": round(p.get("handoff_s", 0.0), 4), "energy_j": round(p["energy_j"], 2),
                "energy_j_table": round(pt[pl]["energy_j"], 2), "prefill_node": p["prefill_node"], "decode_node": p["decode_node"]}}
    rot = [("L", "J", "H"), ("H", "L", "J")]
    order = []
    for i, n in enumerate(NS):
        order += ["n%d_%s" % (n, pl) for pl in rot[i % 2]]
    for i, n in reversed(list(enumerate(NS))):
        order += ["n%d_%s" % (n, pl) for pl in reversed(rot[i % 2])]
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_ph2_20261028.py", "runner": "testbed/scripts/phase_study_20261028.py",
           "out_prefix": "ph2", "follows": "ph1 (testbed/results/ph1_20261028_090849)", "model": "Qwen3-0.6B Q8_0",
           "model_sha256": "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
           "what": "per-model power rows (with the laptop's turbo burst) and 60 s between requests, on held-out prompt lengths: "
                   "the scheduler's energy and its hand-off's first token",
           "costs": {"file": "testbed/dataset/phase_costs_20261028.json", "sha256": sha("testbed/dataset/phase_costs_20261028.json")},
           "power_rows": {"file": "testbed/dataset/phase_power_20261028.json", "sha256": sha("testbed/dataset/phase_power_20261028.json"),
                          "rows": rows},
           "power_table": table, "outlets": OUTLETS, "pause_s": PAUSE, "question_tokens_assumed": QUESTION_TOKENS,
           "arms": arms, "order": order, "picks": picks,
           "hypotheses": {
               "PH-PICK-E": {"claim": "at each n, the least predicted energy (rows) is the least measured (mean of two)"},
               "PH-PICK-T": {"claim": "at each n, the least predicted total time is the least measured"},
               "PH-TIME": {"tol": 0.10, "claim": "every arm's mean total time within 10 % of its prediction"},
               "PH-ENERGY": {"tol": 0.07, "claim": "every arm's mean energy within 7 % of the rows' prediction"},
               "PH-BETTER": {"claim": "the rows' |error| below the TinyLlama table's on every arm"},
               "PH-TTFT": {"tol": 0.03, "claim": "at each n, H's time to first token within 3 % of L's, 60 s between requests"},
               "PH-REPEAT": {"tol": 0.03, "claim": "every arm's two total times within 3 % of each other"}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/register_ph2_20261028.py", "testbed/scripts/phase_study_20261028.py",
               "testbed/scripts/phase_serve_20261028.py", "testbed/scripts/phase_plan_20261028.py",
               "testbed/scripts/phase_power_20261028.py", "testbed/scripts/jetson_proc_20261028.py",
               "testbed/scripts/pd_handoff.py", "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py",
               "testbed/scripts/jetson_power_guard.sh", "testbed/dataset/phase_costs_20261028.json",
               "testbed/dataset/phase_power_20261028.json", "ThinkPad/config/cluster.json")}}
    for n in NS:
        print("n %5d: pick by energy %s, by time %s;  %s" % (n, picks["energy"][str(n)], picks["time"][str(n)], "  ".join(
            "%s %.1f s %.0f J (table %.0f)" % (pl, arms["n%d_%s" % (n, pl)]["predicted"]["total_s"], arms["n%d_%s" % (n, pl)]["predicted"]["energy_j"],
                                             arms["n%d_%s" % (n, pl)]["predicted"]["energy_j_table"]) for pl in ("L", "J", "H"))))
    if a.dry_run:
        print("order", order); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(OUT)[:16]))


if __name__ == "__main__":
    main()
