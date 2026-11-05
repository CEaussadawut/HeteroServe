#!/usr/bin/env python3
"""Register scope 4's two steps on today's boards (study pd4, 2026-10-24), each before it runs.

    python3 testbed/scripts/register_pd4_20261024.py base [--conditions TEXT]
    python3 testbed/scripts/register_pd4_20261024.py cross --base-score testbed/results/pd_pd4cross_<stamp>_score.json [--conditions TEXT]

Step 1, pd4base: pd_twouser.py's two-user test (2026-09-29: TWOUSER, the ORIGINAL Pi 5) again on the replacement Pi 5 and the
Pi 4: Qwen3.5-9B Q4_K_M, two users of 128 tokens each wanting 32 back, together; arms shared_nr (the Pi 5 alone, --no-repack),
handoff (the Pi 5 prefills user 2, its state goes straight into the Pi 4's RAM, the Pi 4 decodes it), shared (the Pi 5 alone,
repacked); 3 repetitions, interleaved, all on gigabit (the overlay). Hypotheses, from 2026-09-29's outcome on the original board:
  PD4-WIN    the hand-off finishes the two users sooner on average than shared_nr (29 Sep: 55.06 against 65.13 s)
  PD4-WIN-R  and sooner than shared (29 Sep: 55.06 against 61.45 s)
  PD4-STATE  the state is 56,889,668 bytes (52,691,776 + 32,796 x 128 + 4 for the 2-slot header) and the Pi 4 processes exactly
             1 prompt token after the restore
  PD4-XFER   the state's transfer at gigabit within 25 % of 29 Sep's 0.51 s
  PD4-STEADY each arm's three arm-runs within 3 %
Step 2, pd4cross, registered from step 1's scored figures: the hand-off with only its state transfer shaped (netshape.sh on the Pi
4: peer the Pi 5's overlay address, port 50091), at R_hi = 2 B* and R_lo = B* / 2, where B* is where 2026-10-08's model
(pd_network_crossover.py) puts the crossover against shared_nr on today's boards:
  only user 2 waits for the transfer, so the mean completion rises by (x(R) - t0) / 2, x(R) the transfer at R and t0 step 1's;
  x(R) = S x 8 x 1514 / 1448 / R (netem charges whole frames: 66 B of headers per 1,448 B of payload; rpc4 measured it);
  B* = S x 8 x 1514 / 1448 / (t0 + 2 (mean_nr - mean_ho)).
Arms shared_nr, handoff, handoff@R_hi, handoff@R_lo, 3 repetitions. Hypotheses:
  PD4X-ABOVE  at R_hi the hand-off finishes sooner than shared_nr
  PD4X-BELOW  at R_lo it finishes later
  PD4X-ACC    each shaped hand-off's mean completion within 5 % of the model
  PD4X-XFER   each shaped transfer within 10 % of x(R)
  PD4X-SHAPED every shaped arm-run's netem counted at least 1,000 packets each way
  PD4X-STEADY each arm's three arm-runs within 3 %
It refuses to register step 2 if step 1 is incomplete or the hand-off does not beat shared_nr at gigabit (then there is no
crossover to place: the answer is that it never pays, and that is reported).
Writes testbed/dataset/pd4base_predictions_20261024.json or pd4cross_predictions_20261024.json, read-only.
"""
import argparse, datetime, hashlib, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(ROOT, "testbed", "dataset")
N, GEN, REPS = 128, 32, 3
STATE = 52691776 + 32796 * N + 4
FRAME = 1514 / 1448
INPUTS = ("testbed/scripts/pd4_cross_20261024.py", "testbed/scripts/pd_twouser.py", "testbed/scripts/pd_handoff.py",
          "testbed/scripts/netshape.sh", "testbed/scripts/shelly_strip.py", "testbed/scripts/register_pd4_20261024.py")


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path, reg):
    if os.path.exists(path):
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(path, ROOT))
    json.dump(reg, open(path, "w"), indent=1)
    os.chmod(path, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(path, ROOT), sha(os.path.relpath(path, ROOT))[:16]))


def base(a):
    reg = {"name": "pd4base", "registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_pd4_20261024.py", "runner": "testbed/scripts/pd4_cross_20261024.py",
           "what": "two users at once on today's boards (the replacement Pi 5 and the Pi 4), Qwen3.5-9B Q4_K_M, 128 tokens in and 32 "
                   "out each, together: the Pi 5 alone (no-repack, repacked) against the prefill/decode hand-off, on gigabit",
           "model": "qwen3.5-9b-q4_k_m (bartowski @ 182be2fd, testbed/models.md)", "n": N, "gen": GEN, "reps": REPS,
           "arms_order": ["shared_nr", "handoff", "shared"], **({"conditions": a.conditions} if a.conditions else {}),
           "reference_20260929": {"handoff": 55.06, "shared_nr": 65.13, "shared": 61.45, "transfer_s": 0.51,
                                  "source": "testbed/dataset/twouser_results_20260929.json (the original Pi 5)"},
           "hypotheses": {
               "PD4-WIN": {"type": "lower", "a": "handoff", "b": "shared_nr",
                           "claim": "the hand-off finishes the two users sooner on average than the Pi 5 alone without repacking"},
               "PD4-WIN-R": {"type": "lower", "a": "handoff", "b": "shared", "claim": "and sooner than the Pi 5 alone repacked"},
               "PD4-STATE": {"type": "state", "bytes": STATE,
                             "claim": "the state is %d bytes and the Pi 4 processes exactly 1 prompt token after the restore" % STATE},
               "PD4-XFER": {"type": "xfer", "tol": 0.25, "predicted_s": {"handoff": 0.51},
                            "claim": "the transfer at gigabit within 25 % of 29 Sep's 0.51 s"},
               "PD4-STEADY": {"type": "steady", "tol": 0.03, "claim": "each arm's three arm-runs within 3 %"}},
           "inputs_sha256": {f: sha(f) for f in INPUTS}}
    write(os.path.join(DS, "pd4base_predictions_20261024.json"), reg)


def cross(a):
    sc = json.load(open(a.base_score))
    if not sc.get("complete") or sc.get("code_changed_since_registration"):
        raise SystemExit("refused: step 1 is incomplete or its code changed: %s" % sc.get("code_changed_since_registration"))
    arms = sc["arms"]
    ho, nr, rp = arms["handoff"]["mean_done_s"], arms["shared_nr"]["mean_done_s"], arms["shared"]["mean_done_s"]
    t0, S = arms["handoff"]["transfer_s"], arms["handoff"]["state_bytes"][0]
    if not ho < nr:
        raise SystemExit("refused: at gigabit the hand-off (%.2f s) does not beat shared_nr (%.2f s): no crossover to place" % (ho, nr))
    bits = S * 8 * FRAME
    b_star = bits / (t0 + 2 * (nr - ho)) / 1e6                       # Mbit/s at netem's rate
    b_star_r = bits / (t0 + 2 * (rp - ho)) / 1e6 if ho < rp else None
    r_hi, r_lo = max(1, round(2 * b_star)), max(1, round(b_star / 2))
    x = lambda r: bits / (r * 1e6)
    pred = {"handoff@%d" % r: round(ho + (x(r) - t0) / 2, 3) for r in (r_hi, r_lo)}
    reg = {"name": "pd4cross", "registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_pd4_20261024.py", "runner": "testbed/scripts/pd4_cross_20261024.py",
           "what": "where the prefill/decode hand-off stops paying as its state transfer slows: the hand-off at %d and %d Mbit/s "
                   "(the Pi 5 -> Pi 4 transfer shaped on the Pi 4) against the Pi 5 alone without repacking, two users, Qwen3.5-9B"
                   % (r_hi, r_lo),
           "base": {"score": os.path.relpath(os.path.abspath(a.base_score), ROOT), "score_sha256": sha(os.path.relpath(os.path.abspath(a.base_score), ROOT)),
                    "handoff_mean_done_s": ho, "shared_nr_mean_done_s": nr, "shared_mean_done_s": rp, "t0_s": t0, "state_bytes": S},
           "model": {"b_star_mbit": round(b_star, 2), "b_star_vs_repacked_mbit": round(b_star_r, 2) if b_star_r else None,
                     "frame_factor": round(FRAME, 5), "r_hi": r_hi, "r_lo": r_lo,
                     "transfer_s": {"handoff@%d" % r: round(x(r), 3) for r in (r_hi, r_lo)}},
           "n": N, "gen": GEN, "reps": REPS, "arms_order": ["shared_nr", "handoff", "handoff@%d" % r_hi, "handoff@%d" % r_lo],
           **({"conditions": a.conditions} if a.conditions else {}),
           "hypotheses": {
               "PD4X-ABOVE": {"type": "lower", "a": "handoff@%d" % r_hi, "b": "shared_nr",
                              "claim": "at %d Mbit/s the hand-off finishes sooner than shared_nr (predicted %.1f against %.1f s)"
                                       % (r_hi, pred["handoff@%d" % r_hi], nr)},
               "PD4X-BELOW": {"type": "higher", "a": "handoff@%d" % r_lo, "b": "shared_nr",
                              "claim": "at %d Mbit/s it finishes later (predicted %.1f against %.1f s)" % (r_lo, pred["handoff@%d" % r_lo], nr)},
               "PD4X-ACC": {"type": "acc", "tol": 0.05, "predicted": pred,
                            "claim": "each shaped hand-off's mean completion within 5 % of the model"},
               "PD4X-XFER": {"type": "xfer", "tol": 0.10, "predicted_s": {k: round(x(int(k.split("@")[1])), 3) for k in pred},
                             "claim": "each shaped transfer within 10 % of S x 8 x 1514/1448 / R"},
               "PD4X-SHAPED": {"type": "shaped", "min_pkts": 1000, "claim": "every shaped arm-run's netem counted >= 1,000 packets each way"},
               "PD4X-STEADY": {"type": "steady", "tol": 0.03, "claim": "each arm's three arm-runs within 3 %"}},
           "inputs_sha256": {f: sha(f) for f in INPUTS + (os.path.relpath(os.path.abspath(a.base_score), ROOT),)}}
    write(os.path.join(DS, "pd4cross_predictions_20261024.json"), reg)
    print("B* %.2f Mbit/s (vs repacked %s); R_hi %d, R_lo %d; predicted %s" % (b_star, "%.2f" % b_star_r if b_star_r else "none", r_hi, r_lo, pred))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("base"); b.add_argument("--conditions", default="")
    c = sp.add_parser("cross"); c.add_argument("--base-score", required=True); c.add_argument("--conditions", default="")
    a = ap.parse_args()
    {"base": base, "cross": cross}[a.cmd](a)


if __name__ == "__main__":
    main()
