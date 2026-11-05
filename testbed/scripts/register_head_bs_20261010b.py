#!/usr/bin/env python3
"""Registration b for scheduler upgrade 6 (2026-10-10, 19:25), superseding head_bs_predictions_20261010.json BEFORE the
full run. Writes, read-only, testbed/dataset/head_bs_predictions_20261010b.json.

    python3 testbed/scripts/register_head_bs_20261010b.py [--dry-run]

One correction, found by the smoke (testbed/results/head_bs_smoke_20261010_191552, plumbing only, never scored):
- the server's ms per token is predicted_ms / predicted_n, and predicted_ms times n - 1 decode passes. The first token
  comes from the prompt's pass, timed in prompt_ms. The server log of the smoke's e01 request shows it: "eval time =
  982.18 ms / 8 tokens" with n_tokens = 28 = 21 + 7 passes;
- so each token carries (n - 1) / n of a pass, and the saving per token is (n - 1) / n x V / g: 63/64 at 64 tokens.
  Registration a priced one whole transfer per token, 1.6 % too high: 1.6 ms at b50, more than the 1 ms tolerance.
Nothing else changes: the plan, settings, prompts, arms, thresholds and hypotheses are a's. The scorer reads
per_token_factor (absent, so 1, in a) and scores b. a's verdicts are reported beside it.
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import register_head_bs_20261010 as A                       # noqa: E402

OUT = os.path.join(A.DS, "head_bs_predictions_20261010b.json")


def main(argv):
    if os.path.exists(OUT) and "--dry-run" not in argv:
        sys.exit("%s exists: a registration is never rewritten" % OUT)
    a = json.load(open(A.OUT))
    n = a["n_predict"]
    f = (n - 1) / float(n)
    b = dict(a)
    b.update({
        "registered_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": "testbed/scripts/register_head_bs_20261010b.py",
        "supersedes": {"file": "testbed/dataset/head_bs_predictions_20261010.json", "sha256": A.sha(A.OUT),
                       "why": "the server times n - 1 decode passes over n tokens; a priced one transfer per token"},
        "per_token_factor": f,
        "predicted_saving_ms_nominal": {s: round(f * v, 3) for s, v in a["predicted_saving_ms_nominal"].items()},
        "inputs_sha256": {p: A.sha(os.path.join(ROOT, p)) for p in A.INPUTS + ["testbed/scripts/register_head_bs_20261010b.py"]
                          if os.path.exists(os.path.join(ROOT, p))},
    })
    b["hypotheses"] = dict(a["hypotheses"], **{
        "HB-SAVE": "at each setting, |(ms per token off - on) - (n-1)/n x V / g| <= 1 ms, g = the measured Jetson -> Pi 4 goodput",
        "HB-SLOPE": "the least-squares slope of the saving against 1 / g over the three settings within 10 % of (n-1)/n x V"})
    print(json.dumps({k: b[k] for k in ("per_token_factor", "predicted_saving_ms_nominal", "supersedes")}, indent=1))
    if "--dry-run" in argv:
        print("(dry run: nothing written)")
        return
    json.dump(b, open(OUT, "w"), indent=1, ensure_ascii=False)
    os.chmod(OUT, 0o444)
    print("registered %s (%s)" % (os.path.relpath(OUT, ROOT), A.sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
