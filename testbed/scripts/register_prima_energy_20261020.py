#!/usr/bin/env python3
"""Register, before it runs, the energy head-to-head of the engine against prima.cpp itself on real boards (2026-10-20).

    python3 testbed/scripts/register_prima_energy_20261020.py [--dry-run]

Why: prima.cpp (OpenCPIL @ 288df6a2 + the qwen3 port, testbed/prima.md) was compared with the engine on 2026-09-30 for time
only (Pi 5 + Pi 4, Qwen3-14B: the engine -11.2 % per token, -44.6 % time to first token); the strip arrived on 10-04, and
on 10-05 the Pi 5 went back, so prima.cpp's energy was only ever modelled (the 10-08 head-to-head served Halda's PLAN on
llama.cpp, on Pi 4 + Jetson). The Pi 5 is back (2026-10-19, a replacement board, its bootloader updated 10-20), and prima.cpp
is still built on both Pis' cards. The advisor's baselines (05/10) include "prima.cpp กับ Halda scheduler".
Arms, both Qwen3-14B Q4_K_M at a 4,096-token context, one user, greedy, by qr3_tpot.py (its prompts and timing):
  A  the engine's plan on Pi 5 + Pi 4, the Pi 5 hosting, llama.cpp at the pin (serve_pool.sh with PLACE / PLACE_HEAD)
  B  prima.cpp, the Pi 5 rank 0 (head), the Pi 4 rank 1, its Halda scheduler choosing the split at start
Order A, B, A, B inside one strip log (5 Hz), the Pi 5 below 50 C before each arm; 6 requests of 128 + 64 tokens per arm-run.
Hypotheses (score_prima_energy_20261020.py):
  PE-TPOT    A's client TPOT P50 is below B's (client: prima.cpp's late-2024 server gives no per-token timings)
  PE-ENERGY  A's gross J per decoded token at the Pi 5 + Pi 4 outlets is below B's
  PE-FOLLOW  B / A in J per token is within 0.05 of B / A in TPOT P50 (both arms keep the same two boards busy, so energy
             should follow time)
  PE-REPL    B / A in client TPOT P50 within 0.05 of 2026-09-30's 1.126 (1,339.0 / 1,188.7 ms client means), on the new board
Writes testbed/dataset/prima_energy_predictions_20261020.json, read-only; refuses to overwrite.
"""
import datetime, hashlib, json, os, stat, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "prima_energy_predictions_20261020.json")
CODE = ["testbed/scripts/register_prima_energy_20261020.py", "testbed/scripts/prima_energy_20261020.sh",
        "testbed/scripts/score_prima_energy_20261020.py", "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh",
        "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py"]
DATA = ["testbed/dataset/memory_at_rest.json", "testbed/dataset/characterisation.csv", "testbed/dataset/link.json",
        "testbed/results/qr3_primacmpA_qwen3_14b_20260930_053423.log", "testbed/results/qr3_primacmpB_qwen3_14b_20260930_061646.log"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    dry = "--dry-run" in sys.argv
    if os.path.exists(OUT) and not dry:
        sys.exit("refusing to overwrite %s" % os.path.relpath(OUT, ROOT))
    mem = json.load(open(os.path.join(ROOT, "testbed", "dataset", "memory_at_rest.json")))
    r = P.plan_host_choice("qwen3_14b", ["pi5", "pi4"], 4096, mem, host="pi5", rtt_trips=1, load_mode="none")
    power = EM.power_table()
    e = {ph: EM.energy(r["assign"], r["q"], r["C"], r["links"], power, ph, head_at=r["head_at"], rtt_trips=1) for ph in ("tg", "pp")}
    head = "host" if r["head_at"] == 0 else "last"
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_prima_energy_20261020.py",
           "model": "qwen3_14b", "ctx": 4096, "boards": ["pi5", "pi4"], "outlets": {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3},
           "arms": {"A": {"engine": "llama.cpp b23701f7 (serve_pool.sh)", "assign": r["assign"], "head": head,
                          "place_env": {"PLACE": ",".join("%s:%d" % x for x in r["assign"]), "PLACE_HEAD": head},
                          "predicted": {"decode_ms": round(r["decode_ms"], 1), "prefill_ms": round(r["prefill_ms"], 1),
                                        "decode_gross_j_per_token_plan_boards": round(e["tg"]["gross_j"], 4)}},
                    "B": {"engine": "prima.cpp 288df6a2 + qwen3 port (qr3_tpot.py --engine prima)", "split": "Halda's, at start (logged)"}},
           "order": ["A", "B", "A", "B"], "reqs": 6, "n": 128, "gen": 64, "cold_c": 50,
           "power_table": power,
           "prior_20260930": {"A_client_mean_ms": 1188.62, "B_client_mean_ms": 1339.0, "ratio": round(1339.0 / 1188.62, 3),
                              "A_log": DATA[3], "B_log": DATA[4]},
           "hypotheses": {"PE-TPOT": {"claim": "A's client TPOT P50 below B's"},
                          "PE-ENERGY": {"claim": "A's gross J per decoded token at the Pi 5 + Pi 4 outlets below B's"},
                          "PE-FOLLOW": {"tol": 0.05, "claim": "B/A in J per token within 0.05 of B/A in client TPOT P50"},
                          "PE-REPL": {"tol": 0.05, "expected": round(1339.0 / 1188.62, 3),
                                      "claim": "B/A in client TPOT P50 within 0.05 of 2026-09-30's ratio"}},
           "energy_definition": "gross J per decoded token: the strip's trapezoid energy over each request's decode window (its "
                                "first token's event to its last), summed over the Pi 5 and Pi 4 outlets, over the request's "
                                "gen - 1 token gaps (full_eval.py's definition); the all-outlets figure reported beside it",
           "inputs_sha256": {p: sha256(os.path.join(ROOT, p)) for p in CODE + DATA if os.path.exists(os.path.join(ROOT, p))}}
    print(json.dumps({k: reg[k] for k in ("arms", "prior_20260930")}, indent=1))
    if dry:
        print("dry run: nothing written"); return
    with open(OUT, "w") as f:
        json.dump(reg, f, indent=1); f.write("\n")
    os.chmod(OUT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print("wrote", os.path.relpath(OUT, ROOT), sha256(OUT))


if __name__ == "__main__":
    main()
