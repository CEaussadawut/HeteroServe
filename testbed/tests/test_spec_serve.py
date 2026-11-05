"""Scheduler upgrade 1, step 3 (2026-10-10): spec_serve_20261010.py's scorer on synthetic sessions, and serve_pool.sh's
default-off draft options; stdlib unittest only.

    python3 -m unittest discover -s testbed/tests -q
"""
import json, os, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import spec_serve_20261010 as SV                              # noqa: E402

PROMPTS = [{"id": "e01", "lang": "en", "text": "x"}, {"id": "th01", "lang": "th", "text": "y"}]


def reg():
    return {"arms": {"base": {}, "pi4d1": {}, "lapd1": {}}, "order": ["base", "pi4d1", "lapd1", "base"], "base_arm": "base",
            "prompts": PROMPTS, "engine_choice": "lapd1", "outlets": {"pi4": 0, "laptop": 1, "jetson": 2},
            "predicted": {"pi4d1": {"speedup": 1.33}, "lapd1": {"speedup": 1.40}}, "moe_note": "n/a", "inputs_sha256": {}}


def session(d, i, arm, ms, n=64, checks=None):
    rows = []
    for p in PROMPTS:
        rows.append({"session": i, "arm": arm, "id": p["id"], "lang": p["lang"], "t0": 1000.0 * i, "t1": 1000.0 * i + 100,
                     "timings": {"predicted_ms": ms * n, "predicted_n": n, "prompt_ms": 500.0,
                                 "draft_n": 0 if arm == "base" else 40, "draft_n_accepted": 0 if arm == "base" else 28}})
    if checks is not None:
        open(os.path.join(d, "server_%d_%s.log" % (i, arm)), "w").write(
            "".join("I slot operator(): id 0 | task 0 | accepted %2d/ 1 draft tokens\n" % a for a in checks))
    return rows


class TestSpecServeScore(unittest.TestCase):
    def score(self, sessions):
        d = tempfile.mkdtemp()
        rows = [r for s in sessions for r in session(d, *s)]
        open(os.path.join(d, "run.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in rows))
        path = os.path.join(d, "reg.json"); json.dump(reg(), open(path, "w"))
        return SV.score(d, path)

    def test_predictions_that_come_true_hold_all_three(self):
        s = self.score([(0, "base", 2000.0), (1, "pi4d1", 2000.0 / 1.33, 64, [1, 1, 0, 1]), (2, "lapd1", 2000.0 / 1.40, 64, [1, 0, 1, 1]),
                        (3, "base", 2000.0)])
        self.assertEqual({h: v["verdict"] for h, v in s["hypotheses"].items()},
                         {"SS-BAR": "HOLDS", "SS-ACC": "HOLDS", "SS-ORDER": "HOLDS", "SS-MOE": "NOT TESTABLE"})
        self.assertAlmostEqual(s["speedups"]["lapd1"]["speedup"], 1.40, places=2)
        self.assertEqual(s["base_drift_pct"], 0.0)
        self.assertEqual(s["arms"]["lapd1"]["alpha1_boards"], 0.75)

    def test_a_choice_below_the_bar_fails_bar_and_accuracy(self):
        s = self.score([(0, "base", 2000.0), (1, "pi4d1", 2000.0 / 1.33), (2, "lapd1", 2000.0 / 1.20), (3, "base", 2000.0)])
        self.assertEqual(s["hypotheses"]["SS-BAR"]["verdict"], "FAILS")
        self.assertEqual(s["hypotheses"]["SS-ACC"]["verdict"], "FAILS")       # 1.40 predicted against 1.20: +16.7 %
        self.assertEqual(s["hypotheses"]["SS-ORDER"]["verdict"], "FAILS")     # the Pi 4 arm measured faster

    def test_a_missing_session_is_incomplete(self):
        s = self.score([(0, "base", 2000.0), (1, "pi4d1", 1500.0), (2, "lapd1", 1450.0)])   # the closing base never ran
        self.assertEqual(s["hypotheses"]["SS-BAR"]["verdict"], "INCOMPLETE")
        self.assertFalse(s["complete"]["base"])

    def test_serve_pool_draft_options_are_off_by_default(self):
        # Off for every harness, as when this campaign ran: since 2026-10-10 (the user: "Laptop draft") the default draft is
        # only for a person at a terminal (serve_defaults.sh; test_serve_defaults.py). This campaign's base arm, which sets no
        # SERVE_DRAFT and calls serve_pool.sh with its output captured, still serves without a draft.
        import subprocess
        src = open(os.path.join(ROOT, "testbed", "scripts", "serve_pool.sh")).read()
        self.assertIn('DRAFT_GIVEN=${SERVE_DRAFT-__unset__}', src)
        defs = os.path.join(ROOT, "testbed", "scripts", "serve_defaults.sh")
        mode = subprocess.run(["bash", "-c", ". '%s'; draft_mode __unset__ qwen3_14b 0" % defs], capture_output=True, text=True,
                              check=True).stdout.strip()
        self.assertEqual(mode, "none")
        # the node's own tree since 2026-10-11 (ldir: ~/llama.cpp on a board, /opt/heteroserve/llama.cpp on the laptop)
        self.assertIn("${DRAFT:+LLAMA_TRACE=1 }setsid nohup $(ldir $HOST)/build/bin/llama-server $ARGS $DRAFT${EXTRA:+ $EXTRA} -c $CTX", src)
        self.assertIn("ldir() { if [ -n \"${LDIR[$1]:-}\" ]; then echo \"${LDIR[$1]}\"; else echo '~/llama.cpp'; fi; }", src)
        self.assertIn("-dev '$devs'", src)                              # quoted: RPC0[...] is a glob to the host's shell


if __name__ == "__main__":
    unittest.main()
