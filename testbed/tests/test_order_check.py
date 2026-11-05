"""Tests for the planner's order check (predict_pool.plan_host_choice(order_check=True), best_by_predictor; 2026-10-23),
stdlib unittest only, on the dataset files and the memory snapshot order35b registered against (2026-10-23 15:47), where
the engine's pick and the predictor's best order differ for Qwen3.6-35B-A3B (measured: the predictor's order 3.8 % faster).

    python3 -m unittest discover -s testbed/tests -q
"""
import json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                      # noqa: E402

SNAP = os.path.join(ROOT, "testbed", "dataset", "memory_snapshot_20261023_154748.json")
NODES = ["pi5", "pi4", "jetson", "laptop"]


class OrderCheck(unittest.TestCase):
    def setUp(self):
        self.env = os.environ.get("HETEROSERVE_RESERVE_MB")
        os.environ["HETEROSERVE_RESERVE_MB"] = ""                     # cluster.json's reserves, as order35b
        self.mem = json.load(open(SNAP))

    def tearDown(self):
        if self.env is None:
            os.environ.pop("HETEROSERVE_RESERVE_MB", None)
        else:
            os.environ["HETEROSERVE_RESERVE_MB"] = self.env

    def plan(self, key, ctx=4096, **kw):
        return P.plan_host_choice(key, NODES, ctx, self.mem, host="laptop", load_mode="none", **kw)

    def test_the_default_is_unchanged(self):
        a = self.plan("qwen36moe_q6k")
        self.assertNotIn("order_check", a)
        self.assertEqual([tuple(x) for x in a["assign"]], [("laptop", 26), ("pi4", 2), ("pi5", 10), ("jetson", 3)])

    def test_the_35b_gets_the_predictors_order(self):
        a, b = self.plan("qwen36moe_q6k"), self.plan("qwen36moe_q6k", order_check=True)
        self.assertEqual([tuple(x) for x in b["assign"]], [("laptop", 26), ("jetson", 3), ("pi5", 10), ("pi4", 2)])
        self.assertEqual(b["head_at"], 0)
        self.assertLess(b["decode_ms"], a["decode_ms"])
        self.assertAlmostEqual(b["decode_ms"], 188.1, delta=0.05)    # order35b's registered F
        self.assertEqual([tuple(x) for x in b["order_check"]["replaced"]], [tuple(x) for x in a["assign"]])

    def test_never_slower_and_kept_when_already_best(self):
        for key in ("qwen3_14b", "qwen3moe_q6k", "nemo12b"):
            a, b = self.plan(key), self.plan(key, order_check=True)
            self.assertLessEqual(b["decode_ms"], a["decode_ms"] + 1e-9, key)
            if b["order_check"] == "kept":
                self.assertEqual(b["assign"], a["assign"], key)

    def test_its_plan_fits(self):
        b = self.plan("qwen36moe_q6k", order_check=True)
        m = P.memory(b["assign"], b["q"], self.mem, b["devices"], b["model"], head_at=b["head_at"])
        self.assertGreaterEqual(min(v["margin_mib"] for v in m.values()), 0)


if __name__ == "__main__":
    unittest.main()
