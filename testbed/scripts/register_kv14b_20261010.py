#!/usr/bin/env python3
"""Register the KV term's 14B test again, with both depths decoded hot (2026-10-10, 21:50). Writes, read-only,
testbed/dataset/kv14b_predictions_20261010.json.

    python3 testbed/scripts/register_kv14b_20261010.py [--dry-run]

Why: in kv14_20261010_201701 (kv14_predictions_20261010.json, HK14-A fails at +182 %, HK14-B at +15.1 %), the d = 0
decode ran first at 1,800 MHz. The d = 1,024 decode ran after ~15 min of prefill with the Pi 4 at its soft limit
(84–85 °C, 1,531–1,775 MHz). So the slope mixes attention with heat.
The change, and the only one: the protocol is `-d 1024,0`. llama-bench runs the depths in the order given, so d = 0
decodes seconds after d = 1,024, at the same heat. The cell, the two models (A: KV bytes, 114.1 ms per 1,000 tokens;
B: query heads on the Pi 4, 279.6), the tolerance (15 %), three runs and the scorer (kv14_20261010.py) are
kv14_predictions_20261010.json's, recomputed by its own build().
Reported beside the verdicts: d = 0 here (hot) against d = 0 in the first test (cool), which is the heat's own cost.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import register_kv14_20261010 as K                          # noqa: E402

OUT = os.path.join(K.DS, "kv14b_predictions_20261010.json")
CELL = "kv14b.d1024hot"


def main(argv):
    if os.path.exists(OUT) and "--dry-run" not in argv:
        sys.exit("%s exists: a registration is never rewritten" % OUT)
    reg = K.build()
    cell = reg["configs"].pop(K.CELL)
    cell["bench"]["protocol"] = cell["bench"]["protocol"].replace("-d 0,1024", "-d 1024,0")
    assert "-d 1024,0" in cell["bench"]["protocol"]
    reg["configs"][CELL] = cell
    reg.update({"cell": CELL, "script": "testbed/scripts/register_kv14b_20261010.py",
                "follows": {"registration": "testbed/dataset/kv14_predictions_20261010.json",
                            "run": "testbed/results/kv14_20261010_201701",
                            "why": "both depths decoded hot: d = 1,024 first, d = 0 seconds after"},
                "question": "the KV term on the served 14B with heat taken out: KV bytes (A) or query heads (B)?"})
    reg["inputs_sha256"]["testbed/scripts/register_kv14b_20261010.py"] = K.sha(os.path.abspath(__file__))
    print(json.dumps({k: reg[k] for k in ("cell", "predictions", "follows")}, indent=1), cell["bench"]["protocol"])
    if "--dry-run" in argv:
        print("(dry run: nothing written)")
        return
    json.dump(reg, open(OUT, "w"), indent=1, ensure_ascii=False)
    os.chmod(OUT, 0o444)
    print("registered %s (%s)" % (os.path.relpath(OUT, ROOT), K.sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
