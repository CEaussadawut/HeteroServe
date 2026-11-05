#!/usr/bin/env python3
"""Register ph3 (2026-11-03; the user: "... Power rows for the other models, for the phase scheduler"): the phase scheduler on a
second model, Qwen3-4B Q4_K_M, with the power rows measured on Qwen3-0.6B. Do per-model rows carry to another model?

    PHASE_MODEL=qwen3_4b python3 testbed/scripts/register_ph3_20261103.py [--dry-run]

The scheduler (phase_plan_20261028.py) prices three plans for one request: L, the laptop alone; J, the Jetson alone (its GPU);
H, the laptop prefills and hands its slot to the Jetson, which decodes. ph2 found its energy within 2.1 % with power rows per
phase fitted on Qwen3-0.6B's own strip data (phase_power_20261028.json). Here the times come from Qwen3-4B Q4_K_M's own
calibration (phase_calib_20261028.py with PHASE_MODEL=qwen3_4b: testbed/dataset/phase_costs_qwen3_4b_20261103.json), and the
energies from the 0.6B's rows. For comparison they are also priced with the TinyLlama table (energy_model.power_table()).
Prompts of 768 and 1,536 tokens (held out: the calibration used 256-2,048 in powers of two; the 4B's context is 2,560 cells on
the Jetson), 256 generated, plans L and J, two repetitions, interleaved, 60 s between requests (phase_study_20261028.py run,
PHASE_MODEL=qwen3_4b; the strip at 5 Hz on all four outlets). The study's strip data then fits the 4B's own rows
(phase_power_20261028.py fit <run> ...), the deliverable.
No H (2026-11-04): the 4B's calibration (phase_calib_20261104_004942.log) found its hand-off thrashing the 4 GB Jetson's memory
from n = 1,024 (the slot file in /dev/shm plus the restore's copy, on top of the model and its 2,560 cells): restore 214 s and
53 min for 17 decoded tokens at n = 1,024, where n = 512 took 0.3 s; then three hand-offs failed on ssh and the n = 2,048 restore
never returned. So H is infeasible at these lengths here, and its costs are not fitted.
Hypotheses (registered; phase_study_20261028.py score):
  PH-PICK-E  at each n, the least predicted energy (the 0.6B's rows) is the least measured (mean of two)
  PH-PICK-T  the same for total time
  PH-TIME    every arm's mean total time within 10 % of its prediction
  PH-ENERGY  every arm's mean energy within 10 % of the 0.6B's rows' prediction (ph2's tolerance was 7 %, on the model fitted)
  PH-BETTER  the rows' |error| below the TinyLlama table's on every arm
  PH-REPEAT  every arm's two total times within 3 % of each other
Writes testbed/dataset/ph3_predictions_20261103.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import phase_plan_20261028 as PP                           # noqa: E402
import phase_power_20261028 as W                           # noqa: E402
import energy_model as EM                                   # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "ph3_predictions_20261103.json")
NS, GEN, PAUSE = [768, 1536], 256, 60
COSTS4B = "testbed/dataset/phase_costs_qwen3_4b_20261103.json"
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}
QUESTION_TOKENS = 12
PLANS = ("L", "J")                                          # no H: see above
CALIB = "testbed/results/phase_calib_20261104_004942.log"


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
    import phase_serve_20261028 as S
    if S.PHASE_MODEL != "qwen3_4b":
        raise SystemExit("refused: run with PHASE_MODEL=qwen3_4b (phase_serve picks the model from it)")
    costs, rows, table = PP.load(os.path.join(ROOT, COSTS4B)), W.load()["rows"], EM.power_table()
    arms, picks = {}, {"energy": {}, "time": {}}
    for n in NS:
        ne = n + QUESTION_TOKENS
        pr = {pl: W.energy(costs, pl, ne, GEN, rows) for pl in PLANS}
        pt = {pl: PP.plan(costs, pl, ne, GEN, table) for pl in PLANS}
        picks["energy"][str(n)] = min(pr, key=lambda k: pr[k]["energy_j"])
        picks["time"][str(n)] = min(pr, key=lambda k: pr[k]["total_s"])
        for pl in PLANS:
            p = pr[pl]
            arms["n%d_%s" % (n, pl)] = {"n": n, "gen": GEN, "plan": pl, "predicted": {
                "ttft_s": round(p["ttft_s"], 4), "decode_s": round(p["decode_s"], 4), "total_s": round(p["total_s"], 4),
                "handoff_s": round(p.get("handoff_s", 0.0), 4), "energy_j": round(p["energy_j"], 2),
                "energy_j_table": round(pt[pl]["energy_j"], 2), "prefill_node": p["prefill_node"], "decode_node": p["decode_node"]}}
    rot = [PLANS, PLANS[::-1]]
    order = []
    for i, n in enumerate(NS):
        order += ["n%d_%s" % (n, pl) for pl in rot[i % 2]]
    for i, n in reversed(list(enumerate(NS))):
        order += ["n%d_%s" % (n, pl) for pl in reversed(rot[i % 2])]
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_ph3_20261103.py", "runner": "testbed/scripts/phase_study_20261028.py",
           "out_prefix": "ph3", "follows": "ph2 (testbed/results/ph2_20261028_122445)", "model": "Qwen3-4B Q4_K_M",
           "phase_model": "qwen3_4b", "model_sha256": S.MODEL_SHA,
           "what": "the phase scheduler on a second model (Qwen3-4B Q4_K_M), with its own calibrated times and the power rows "
                   "measured on Qwen3-0.6B: do per-model rows carry to another model?",
           "costs": {"file": COSTS4B, "sha256": sha(COSTS4B)},
           "plans": list(PLANS),
           "handoff_excluded": {"calibration": CALIB, "sha256": sha(CALIB),
                                "why": "the 4B's hand-off thrashed the 4 GB Jetson's memory from n = 1,024 (the slot file in /dev/shm "
                                       "plus the restore's copy, on top of the model and its 2,560 cells): restore 214 s and 53 min "
                                       "for 17 decoded tokens at n = 1,024 (0.3 s at n = 512), three hand-offs then failed on ssh, "
                                       "and the n = 2,048 restore never returned; H is infeasible at these lengths here"},
           "power_rows": {"file": "testbed/dataset/phase_power_20261028.json", "sha256": sha("testbed/dataset/phase_power_20261028.json"),
                          "rows": rows},
           "power_table": table, "outlets": OUTLETS, "pause_s": PAUSE, "question_tokens_assumed": QUESTION_TOKENS,
           "conditions": "aunsync.timer (Google Drive sync, every 16 min, ~20 s) runs; the quiet gate is checked once before the "
                         "run (phase_calib_20261028.laptop_quiet), not during it; the Zen browser stays open, allowed by the "
                         "owner (2026-11-04: \"it's ok with zen open, RAM 8GB is reserved\"; the gate runs with "
                         "QUIET_ALLOW_APPS=zen), so the laptop's outlet includes it; the 4B's calibration ran the same way",
           "arms": arms, "order": order, "picks": picks,
           "hypotheses": {
               "PH-PICK-E": {"claim": "at each n, the least predicted energy (rows) is the least measured (mean of two)"},
               "PH-PICK-T": {"claim": "at each n, the least predicted total time is the least measured"},
               "PH-TIME": {"tol": 0.10, "claim": "every arm's mean total time within 10 % of its prediction"},
               "PH-ENERGY": {"tol": 0.10, "claim": "every arm's mean energy within 10 % of the 0.6B's rows' prediction"},
               "PH-BETTER": {"claim": "the rows' |error| below the TinyLlama table's on every arm"},
               "PH-REPEAT": {"tol": 0.03, "claim": "every arm's two total times within 3 % of each other"}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/register_ph3_20261103.py", "testbed/scripts/phase_study_20261028.py", "testbed/scripts/phase_calib_20261028.py",
               "testbed/scripts/phase_serve_20261028.py", "testbed/scripts/phase_plan_20261028.py",
               "testbed/scripts/phase_power_20261028.py", "testbed/scripts/jetson_proc_20261028.py",
               "testbed/scripts/pd_handoff.py", "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py",
               "testbed/scripts/jetson_power_guard.sh", COSTS4B,
               "testbed/dataset/phase_power_20261028.json", "ThinkPad/config/cluster.json")}}
    for n in NS:
        print("n %5d: pick by energy %s, by time %s;  %s" % (n, picks["energy"][str(n)], picks["time"][str(n)], "  ".join(
            "%s %.1f s %.0f J (table %.0f)" % (pl, arms["n%d_%s" % (n, pl)]["predicted"]["total_s"], arms["n%d_%s" % (n, pl)]["predicted"]["energy_j"],
                                             arms["n%d_%s" % (n, pl)]["predicted"]["energy_j_table"]) for pl in PLANS)))
    if a.dry_run:
        print("order", order); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(OUT)[:16]))


if __name__ == "__main__":
    main()
