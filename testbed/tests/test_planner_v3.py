"""Tests for predict_pool.PLANNER_V3, the planner's proposed defaults (2026-10-28, def1), stdlib unittest only.

    python3 -m unittest discover -s testbed/tests -q
"""
import json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                      # noqa: E402

SNAP = os.path.join(ROOT, "testbed", "dataset", "memory_snapshot_20261023_170219.json")
NODES = ["pi5", "pi4", "jetson", "laptop"]


class PlannerV3(unittest.TestCase):
    def setUp(self):
        self.env = os.environ.get("HETEROSERVE_RESERVE_MB")
        os.environ["HETEROSERVE_RESERVE_MB"] = ""
        self.mem = json.load(open(SNAP))

    def tearDown(self):
        if self.env is None:
            os.environ.pop("HETEROSERVE_RESERVE_MB", None)
        else:
            os.environ["HETEROSERVE_RESERVE_MB"] = self.env

    def test_the_options(self):
        self.assertEqual(set(P.PLANNER_V3), {"order_check", "rpc_msg", "node_mult"})
        self.assertIs(P.PLANNER_V3["rpc_msg"], P.RPC_MSG)
        self.assertEqual(P.PLANNER_V3["node_mult"], "measured_blocks")

    def test_the_functions_defaults_are_unchanged(self):
        a = P.plan_host_choice("qwen3_14b", NODES, 4096, self.mem, host="laptop", load_mode="none")
        self.assertNotIn("order_check", a)
        self.assertAlmostEqual(a["decode_ms"], 439.5, delta=0.05)

    def test_the_35b_is_priced_with_its_multipliers(self):
        a = P.plan_host_choice("qwen36moe_q6k", NODES, 4096, self.mem, host="laptop", load_mode="none", order_check=True)
        b = P.plan_host_choice("qwen36moe_q6k", NODES, 4096, self.mem, host="laptop", load_mode="none", **P.PLANNER_V3)
        self.assertGreater(b["decode_ms"] / a["decode_ms"], 1.15)

    def test_the_head_keeps_its_boards_unscaled_rate(self):
        # def2's L3: the head on the Pi 4; blocks-only = blocks-and-head less the head's scaled excess
        assign, key = [("laptop", 25), ("pi5", 10), ("jetson", 3), ("pi4", 3)], "qwen36moe_q6k"
        a = P.plan_fixed(key, assign, "last", 4096, self.mem, load_mode="none", node_mult="measured")
        b = P.plan_fixed(key, assign, "last", 4096, self.mem, load_mode="none", node_mult="measured_blocks")
        base = P.plan_fixed(key, assign, "last", 4096, self.mem, load_mode="none")
        excess = base["q"]["head_bytes"] * base["C"]["pi4"]["worker"][0] * (P.NODE_MULT[key]["pi4"] - 1)
        self.assertAlmostEqual(a["decode_ms"] - b["decode_ms"], excess, places=6)

    def test_a_model_without_multipliers_moves_only_by_the_rpc_term(self):
        for key in ("qwen3_14b", "qwen3_32b_q6k", "qwen3moe_q6k"):
            a = P.plan_host_choice(key, NODES, 4096 if key != "qwen3_32b_q6k" else 2048, self.mem, host="laptop",
                                   load_mode="none", order_check=True)
            b = P.plan_host_choice(key, NODES, 4096 if key != "qwen3_32b_q6k" else 2048, self.mem, host="laptop",
                                   load_mode="none", **P.PLANNER_V3)
            self.assertLess(abs(b["decode_ms"] / a["decode_ms"] - 1), 0.01, key)


if __name__ == "__main__":
    unittest.main()
