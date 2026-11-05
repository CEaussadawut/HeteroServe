"""The planner's context-length term and the laptop as a worker (2026-10-11): predict_pool's opt-in additions and the
long-context scorer, on the real dataset and synthetic runs; stdlib unittest.

    python3 -m unittest discover -s testbed/tests -q
"""
import json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                      # noqa: E402
import longctx_20261011 as LC                                 # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")


class TestTerm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
        cls.q = shapes["qwen3-14b-q4_k_m"]
        R = P.rows(os.path.join(DS, "characterisation.csv"), "char")
        cls.C, _ = P.costs(R, shapes, "Q4_K_M")
        cls.L, _ = P.link_view(json.load(open(os.path.join(DS, "link.json")))["pairs"], "pi4", ["pi4", "jetson"])
        cls.T = P.attention_table()
        cls.attn = {n: P.attn_ms(cls.T, n, cls.q) for n in ("pi4", "jetson")}

    def test_depth_zero_is_every_earlier_prediction(self):
        a = [("pi4", 27), ("jetson", 13)]
        self.assertEqual(P.predict(a, self.q, self.C, self.L, head_at=0, rtt_trips=1),
                         P.predict(a, self.q, self.C, self.L, head_at=0, rtt_trips=1, depth=0, attn=self.attn))

    def test_the_term_adds_blocks_times_depth_times_cost(self):
        a = [("pi4", 27), ("jetson", 13)]
        d0, _ = P.predict(a, self.q, self.C, self.L, head_at=0, rtt_trips=1)
        d1, _ = P.predict(a, self.q, self.C, self.L, head_at=0, rtt_trips=1, depth=1024, attn=self.attn)
        self.assertAlmostEqual(d1 - d0, 1024 * (27 * self.attn["pi4"] + 13 * self.attn["jetson"]), places=9)

    def test_the_laws(self):
        b = self.T["boards"]
        self.assertAlmostEqual(self.attn["pi4"], 40 * 128 * b["pi4"]["ns_per_qhead_dim"] * 1e-6, places=12)
        self.assertAlmostEqual(self.attn["jetson"], 4096 * b["jetson"]["ns_per_kv_byte"] * 1e-6, places=12)
        self.assertIsNone(P.attn_ms(self.T, "pi5", self.q))

    def test_profiles_carry_the_term_in_both_roles(self):
        mem = json.load(open(os.path.join(DS, "memory_at_rest.json")))
        p0 = P.profiles(self.C, self.q, mem, self.L, ["pi4", "jetson"])
        p1 = P.profiles(self.C, self.q, mem, self.L, ["pi4", "jetson"], depth=2048, attn=self.attn)
        for n in ("pi4", "jetson"):
            self.assertAlmostEqual(p1[n].ms_per_layer_decode - p0[n].ms_per_layer_decode, 2048 * self.attn[n], places=9)
            self.assertAlmostEqual(p1[n].ms_per_layer_decode_worker - p0[n].ms_per_layer_decode_worker, 2048 * self.attn[n], places=9)
            self.assertEqual(p1[n].ms_per_layer_prefill, p0[n].ms_per_layer_prefill)

    def test_the_laptop_worker_is_priced_from_its_own_row(self):
        shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
        R = P.rows(os.path.join(DS, "characterisation.csv"), "char")
        lap = P.laptop_costs(R, shapes, "Q4_K_M", os.path.join(DS, "laptop_worker_20261011.json"))
        w = json.load(open(os.path.join(DS, "laptop_worker_20261011.json")))["rows"]["Q4_K_M"]
        self.assertEqual(lap["worker"], P.unit(w["tg_ms"], w["pp_ms"], shapes[P.TINY["Q4_K_M"]]))
        self.assertLess(lap["host"][0], lap["worker"][0])          # the host repacks Q4_K for AVX2; the worker does not


REG = {"plan_depth": 4096, "runs": 3, "tolerance_pct": 15.0, "tolerance_slope_pct": 25.0,
       "predictions_ms": {"0": 891.77, "4096": 1336.13}, "predictions_without_the_term_ms": {"0": 891.77, "4096": 891.77},
       "served_plan_measured_d0_ms_kv14b": 1991.08, "predicted_gain_d0": 2.2327,
       "served_plan_predicted_ms": {"0": 2025.73, "4096": 3113.47}}


class TestScore(unittest.TestCase):
    def test_on_prediction_everything_holds(self):
        s = LC.score_rows(REG, [{"0": 891.77, "4096": 1336.13}] * 3)
        self.assertTrue(all(h["verdict"] == "HOLDS" for h in s["hypotheses"].values()))

    def test_a_flat_slope_fails_the_slope_and_the_depth(self):
        s = LC.score_rows(REG, [{"0": 891.77, "4096": 900.0}] * 3)
        self.assertEqual(s["hypotheses"]["LC-SLOPE"]["verdict"], "FAILS")
        self.assertEqual(s["hypotheses"]["LC-4096"]["verdict"], "FAILS")
        self.assertEqual(s["hypotheses"]["LC-0"]["verdict"], "HOLDS")

    def test_a_missing_run_is_incomplete(self):
        s = LC.score_rows(REG, [{"0": 891.77, "4096": 1336.13}] * 2)
        self.assertEqual(s["hypotheses"]["LC-0"]["verdict"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
