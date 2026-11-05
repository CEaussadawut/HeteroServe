#!/usr/bin/env python3
"""Register ph1 (2026-10-28; item 3, "the prefill-decode hand-off inside the scheduler"): the scheduler's phase placement
for one request, against measurement, energy at the wall. Registered before it runs.

    python3 testbed/scripts/register_ph1_20261028.py [--dry-run]

The scheduler (phase_plan_20261028.py) prices three plans for a request of n prompt tokens and gen generated tokens on
Qwen3-0.6B Q8_0: L, the laptop alone; J, the Jetson alone (its GPU, Vulkan); H, the laptop prefills and hands its slot to
the Jetson, which decodes. It picks the plan with the least predicted energy (all four outlets) or the least time. Its
costs are phase_calib_20261028.py's fits (testbed/dataset/phase_costs_20261028.json), measured through the same serving path
before this registration, and the power table energy_model.power_table() (single-board strip runs of 5-20 October). None
of the calibration's requests is reused here.
Why it might pay for ONE user (the home-use scope): the laptop prefills faster, while the Jetson decodes a long context
faster (its attention grows ~8 ms per 1,000 tokens of depth on this model, lc4n; the laptop's CPU far more) and the cluster
draws ~10 W less with the Jetson busy than with the laptop busy (the laptop's 14 W idle counts in every plan). For a short
prompt the laptop should win both phases; for a long one, the hand-off or the Jetson alone. The calibration found the
Jetson's GPU prefilling about as fast as the laptop by 4,096 tokens (and faster at 8,192), so the hand-off may never pay
for one user; the test points span the predicted changes: n = 1,024, 2,048, 4,096 and 8,192 prompt tokens (the GPL-3
text, then a question), 256 generated, greedy; every plan at every n, two repetitions,
interleaved so each plan takes each position (phase_study_20261028.py run; the strip at 5 Hz on all four outlets; the
laptop quiet on its charger; the Jetson's procedure).
Hypotheses (registered):
  PH-PICK-E  at each n, the plan with the least predicted energy is the one with the least measured energy (mean of two)
  PH-PICK-T  the same for total time
  PH-TIME    every arm's mean total time within 10 % of its prediction
  PH-ENERGY  every arm's mean energy within 15 % of its prediction (the power table is from TinyLlama runs)
  PH-TTFT    at each n, H's time to first token within 5 % of L's (the first token comes from the laptop in both)
  PH-REPEAT  every arm's two total times within 3 % of each other
Writes testbed/dataset/ph1_predictions_20261028.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import phase_plan_20261028 as PP                           # noqa: E402
import energy_model as EM                                   # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "ph1_predictions_20261028.json")
NS, GEN = [1024, 2048, 4096, 8192], 256
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}
QUESTION_TOKENS = 12          # the question appended to the cut text (pd_handoff.QUESTION; tokenized on the servers at run time)


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
    costs = PP.load()
    if costs.get("failed"):
        print("calibration notes: %s" % costs["failed"])
    power = EM.power_table()
    arms, picks = {}, {"energy": {}, "time": {}}
    for n in NS:
        ne = n + QUESTION_TOKENS
        r = PP.choose(ne, GEN, "energy", costs, power)
        picks["energy"][str(n)] = r["pick"]
        picks["time"][str(n)] = PP.choose(ne, GEN, "time", costs, power)["pick"]
        for plan, p in r["plans"].items():
            arms["n%d_%s" % (n, plan)] = {"n": n, "gen": GEN, "plan": plan,
                                          "predicted": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()
                                                        if k != "handoff_parts_s"} | (
                                              {"handoff_parts_s": {k: round(v, 4) for k, v in p["handoff_parts_s"].items()}}
                                              if "handoff_parts_s" in p else {})}
    rot = [("L", "J", "H"), ("J", "H", "L"), ("H", "L", "J")]
    order = []
    for i, n in enumerate(NS):
        order += ["n%d_%s" % (n, pl) for pl in rot[i % 3]]
    for i, n in reversed(list(enumerate(NS))):
        order += ["n%d_%s" % (n, pl) for pl in reversed(rot[i % 3])]
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_ph1_20261028.py", "runner": "testbed/scripts/phase_study_20261028.py",
           "out_prefix": "ph1", "model": "Qwen3-0.6B Q8_0", "model_sha256": "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
           "what": "the scheduler's phase placement for one request (the laptop, the Jetson, or prefill on the laptop handed to "
                   "the Jetson for decode), predicted from its calibration and the power table, against measured time and "
                   "energy at the wall",
           "costs": {"file": "testbed/dataset/phase_costs_20261028.json", "sha256": sha("testbed/dataset/phase_costs_20261028.json"),
                     "from": costs.get("from")},
           "power_table": power, "outlets": OUTLETS, "question_tokens_assumed": QUESTION_TOKENS,
           "arms": arms, "order": order, "picks": picks,
           "hypotheses": {
               "PH-PICK-E": {"claim": "at each n, the least predicted energy is the least measured energy (mean of two)"},
               "PH-PICK-T": {"claim": "at each n, the least predicted total time is the least measured"},
               "PH-TIME": {"tol": 0.10, "claim": "every arm's mean total time within 10 % of its prediction"},
               "PH-ENERGY": {"tol": 0.15, "claim": "every arm's mean energy (all four outlets) within 15 % of its prediction"},
               "PH-TTFT": {"tol": 0.05, "claim": "at each n, H's time to first token within 5 % of L's"},
               "PH-REPEAT": {"tol": 0.03, "claim": "every arm's two total times within 3 % of each other"}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/register_ph1_20261028.py", "testbed/scripts/phase_study_20261028.py",
               "testbed/scripts/phase_serve_20261028.py", "testbed/scripts/phase_plan_20261028.py",
               "testbed/scripts/phase_calib_20261028.py", "testbed/scripts/jetson_proc_20261028.py",
               "testbed/scripts/pd_handoff.py", "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py",
               "testbed/scripts/jetson_power_guard.sh", "testbed/dataset/phase_costs_20261028.json", "ThinkPad/config/cluster.json")}}
    for n in NS:
        print("n %5d: pick by energy %s, by time %s;  %s" % (n, picks["energy"][str(n)], picks["time"][str(n)], "  ".join(
            "%s %.1f s %.0f J" % (pl, arms["n%d_%s" % (n, pl)]["predicted"]["total_s"], arms["n%d_%s" % (n, pl)]["predicted"]["energy_j"])
            for pl in ("L", "J", "H"))))
    if a.dry_run:
        print("order", order); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(OUT)[:16]))


if __name__ == "__main__":
    main()
