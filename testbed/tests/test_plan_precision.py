"""Tests for scheduler v2's precision search (predict_pool.plan_precision; 2026-10-21, the user's rule: near-lossless only,
Q6_K or better), stdlib unittest only. plan_host_choice is replaced by a stub, so the selection logic is tested apart from
the planner.

    python3 -m unittest discover -s testbed/tests -q
"""
import os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                      # noqa: E402


class PlanPrecision(unittest.TestCase):
    def setUp(self):
        self.real = P.plan_host_choice

    def tearDown(self):
        P.plan_host_choice = self.real

    def stub(self, times):
        def f(key, nodes, ctx, mem, **kw):
            if times[key] is None:
                raise ValueError("no subset of the cluster can hold %s" % key)
            return {"decode_ms": times[key], "assign": [("laptop", 1)], "key": key}
        P.plan_host_choice = f

    def test_the_floor_is_the_users_rule(self):
        self.assertEqual(P.MIN_BITS, 6.5)
        bits = dict(P.VARIANTS["qwen3_32b"])
        self.assertGreaterEqual(bits["qwen3_32b_q8"], P.MIN_BITS)
        self.assertGreaterEqual(bits["qwen3_32b_q6k"], P.MIN_BITS)
        self.assertLess(bits["qwen3_32b_q4km"], P.MIN_BITS)

    def test_fastest_allowed_variant_wins_and_below_floor_is_never_chosen(self):
        self.stub({"qwen3_32b_q8": 3666.8, "qwen3_32b_q6k": 1658.7, "qwen3_32b_q4km": 1000.0})
        best, out = P.plan_precision("qwen3_32b", ["laptop"], 2048, {})
        self.assertEqual(best["variant"], "qwen3_32b_q6k")
        q4 = [e for e in out if e["variant"] == "qwen3_32b_q4km"][0]
        self.assertTrue(q4["below_floor"])
        self.assertIn("plan", q4)                      # priced and reported, not chosen

    def test_lower_floor_lets_the_lower_variant_win(self):
        self.stub({"qwen3_32b_q8": 3666.8, "qwen3_32b_q6k": 1658.7, "qwen3_32b_q4km": 1000.0})
        best, _ = P.plan_precision("qwen3_32b", ["laptop"], 2048, {}, min_bits=4.0)
        self.assertEqual(best["variant"], "qwen3_32b_q4km")

    def test_infeasible_variant_is_reported(self):
        self.stub({"qwen3_32b_q8": None, "qwen3_32b_q6k": 1658.7, "qwen3_32b_q4km": 1000.0})
        best, out = P.plan_precision("qwen3_32b", ["laptop"], 2048, {})
        self.assertEqual(best["variant"], "qwen3_32b_q6k")
        self.assertIn("infeasible", [e for e in out if e["variant"] == "qwen3_32b_q8"][0])

    def test_tie_goes_to_the_more_precise(self):
        self.stub({"qwen3_32b_q8": 2000.0, "qwen3_32b_q6k": 2000.0, "qwen3_32b_q4km": 2000.0})
        best, _ = P.plan_precision("qwen3_32b", ["laptop"], 2048, {})
        self.assertEqual(best["variant"], "qwen3_32b_q8")

    def test_nothing_allowed_feasible_raises(self):
        self.stub({"qwen3_32b_q8": None, "qwen3_32b_q6k": None, "qwen3_32b_q4km": 1000.0})
        with self.assertRaises(ValueError):
            P.plan_precision("qwen3_32b", ["laptop"], 2048, {})

    def test_family_lookup(self):
        self.assertEqual(P.variant_family("qwen3_32b_q6k"), "qwen3_32b")
        self.assertIsNone(P.variant_family("tinyllama"))


if __name__ == "__main__":
    unittest.main()
