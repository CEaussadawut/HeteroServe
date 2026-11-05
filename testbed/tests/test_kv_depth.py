"""Scheduler upgrade 4's KV term (2026-10-10): the depth-slope scorer, on synthetic runs; stdlib unittest.

    python3 -m unittest discover -s testbed/tests -q
"""
import os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import kv_depth_20261010 as KV                                # noqa: E402

D = [0, 1024, 2048, 4096]
REG = {"bench": {"depths": D, "invocations": 3}, "tolerance_pct": 15.0,
       "predictions": {"pi4": {"slope_ms_per_1k_tokens": 31.3}, "jetson": {"slope_ms_per_1k_tokens": 15.4}}}


def inv(base, per1k, jitter=0.0):
    return [(d, base + per1k * d / 1000.0 + (jitter if d == 2048 else 0.0)) for d in D]


class TestKV(unittest.TestCase):
    def test_slope_is_recovered(self):
        s, b = KV.slope(inv(170.0, 31.3))
        self.assertAlmostEqual(s, 31.3, places=9)
        self.assertAlmostEqual(b, 170.0, places=9)

    def test_verdicts(self):
        runs = {"pi4": [inv(170, 31.3), inv(171, 32.0), inv(169, 30.8)], "jetson": [inv(60, 25.0)] * 3}
        s = KV.score_rows(REG, runs)
        self.assertEqual(s["hypotheses"]["HK-PI4"]["verdict"], "HOLDS")
        self.assertEqual(s["hypotheses"]["HK-JET"]["verdict"], "FAILS")       # +62 %
        self.assertEqual(s["kv_term"], "FAILS")

    def test_a_missing_invocation_is_incomplete(self):
        s = KV.score_rows(REG, {"pi4": [inv(170, 31.3)] * 2, "jetson": [inv(60, 15.4)] * 3})
        self.assertEqual(s["hypotheses"]["HK-PI4"]["verdict"], "INCOMPLETE")
        self.assertEqual(s["kv_term"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()


import kv14_20261010 as K14                                    # noqa: E402

REG14 = {"runs": 3, "tolerance_pct": 15.0,
         "predictions": {"A_ms_per_1k": 114.07, "B_ms_per_1k": 279.55, "registered_weight_cost_term_ms_per_1k": 38.31}}


class TestKV14(unittest.TestCase):
    def test_a_bytes_like_slope_holds_a_and_fails_b(self):
        runs = [{"d0": 1900.0, "d1024": 1900.0 + 1.024 * x} for x in (110.0, 118.0, 114.0)]
        s = K14.score_rows(REG14, runs)
        self.assertEqual(s["hypotheses"]["HK14-A"]["verdict"], "HOLDS")
        self.assertEqual(s["hypotheses"]["HK14-B"]["verdict"], "FAILS")
        self.assertAlmostEqual(s["slope_ms_per_1k_mean"], 114.0, places=6)

    def test_a_missing_run_is_incomplete(self):
        s = K14.score_rows(REG14, [{"d0": 1900.0, "d1024": 2017.0}] * 2)
        self.assertEqual(s["hypotheses"]["HK14-A"]["verdict"], "INCOMPLETE")
