"""The opt-in per-board multipliers (predict_pool.NODE_MULT, 2026-10-24): off by default, and with them the 35B's measured
split A (c35, testbed/results/c35_20261024_025725) is predicted at its measured server mean, 246.2 ms per token."""
import json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "testbed", "scripts"), os.path.join(ROOT, "ThinkPad")]
import predict_pool as P  # noqa: E402

REG = os.path.join(ROOT, "testbed", "dataset", "c35_predictions_20261024.json")
SCORE = os.path.join(ROOT, "testbed", "results", "c35_20261024_025725", "score.json")


class NodeMult(unittest.TestCase):
    def test_off_by_default(self):
        C = {"laptop": {"host": (1.0, 2.0), "worker": (3.0, 4.0)}}
        self.assertIs(P.apply_node_mult(C, "qwen36moe_q6k", None), C)
        self.assertIs(P.apply_node_mult(C, "tinyllama_q4_0", "measured"), C)     # no factors for this model

    def test_decode_only(self):
        C = {"laptop": {"host": (1.0, 2.0), "worker": (3.0, 4.0)}, "pi4": {"host": (5.0, 6.0), "worker": (7.0, 8.0)}}
        out = P.apply_node_mult(C, "x", {"laptop": 2.0})
        self.assertEqual(out["laptop"], {"host": (2.0, 2.0), "worker": (6.0, 4.0)})
        self.assertEqual(out["pi4"], C["pi4"])
        self.assertEqual(C["laptop"]["host"], (1.0, 2.0))                          # the input is not changed

    def test_factors_are_c35s(self):
        k = json.load(open(SCORE))["extra"]["k"]
        self.assertEqual(P.NODE_MULT["qwen36moe_q6k"], k)

    def test_split_a_predicted_at_its_measurement(self):
        reg, score = json.load(open(REG)), json.load(open(SCORE))
        mem = json.load(open(os.path.join(ROOT, reg["memfile"])))
        a = reg["arms"]["A"]
        assign = [tuple(x) for x in a["assign"]]
        raw = P.plan_fixed(reg["model"], assign, "host", reg["ctx"], mem)
        fit = P.plan_fixed(reg["model"], assign, "host", reg["ctx"], mem, node_mult="measured")
        measured = score["arms"]["A"]["server_mean_ms"]
        self.assertAlmostEqual(raw["decode_ms"], a["terms"]["decode_ms"], delta=0.02 * a["terms"]["decode_ms"])
        self.assertAlmostEqual(fit["decode_ms"], measured, delta=0.01 * measured)                 # 246.2 within 1 %
        self.assertAlmostEqual(fit["prefill_ms"], raw["prefill_ms"], places=6)                   # prefill not scaled


if __name__ == "__main__":
    unittest.main()
