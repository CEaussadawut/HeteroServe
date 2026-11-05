"""The draft's placement under shaping (2026-10-10): the registered model and the scorer, on synthetic runs; stdlib unittest.

    python3 -m unittest discover -s testbed/tests -q
"""
import json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import register_spec_shape_20261010 as R                      # noqa: E402
import spec_shape_20261010 as SH                              # noqa: E402


class TestShapeModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = R.build()
        cls.m = cls.reg["model_parts"]

    def test_at_gigabit_it_is_step_3s_registered_model(self):
        s3 = json.load(open(R.STEP3_REG))["model_parts"]["S_all"]
        m = dict(self.m, P=json.load(open(R.STEP3_REG))["model_parts"]["mean_prompt_tokens"])   # step 3's eight prompts
        s = R.speedups(m, self.m["bw_gigabit"], self.m["rtt_registered"])
        self.assertAlmostEqual(s["lapd1"], s3["d1_laptop"], delta=0.002)   # the hidden-state terms are tiny at gigabit
        self.assertAlmostEqual(s["pi4d1"], s3["d1_pi4"], delta=0.002)

    def test_the_crossover_equalises_the_arms_and_the_laptop_wins_above_it(self):
        x = R.crossover(self.m, self.m["rtt_registered"])
        s = R.speedups(self.m, x, self.m["rtt_registered"])
        self.assertAlmostEqual(s["lapd1"], s["pi4d1"], places=6)
        self.assertAlmostEqual(R.mbit(x), 29.76, delta=0.01)
        hi, lo = R.speedups(self.m, 2 * x, 0.85), R.speedups(self.m, x / 2, 0.85)
        self.assertGreater(hi["lapd1"], hi["pi4d1"]); self.assertLess(lo["lapd1"], lo["pi4d1"])

    def test_the_settings_and_the_flip_rule(self):
        n = self.reg["predicted_nominal"]
        self.assertEqual([s["id"] for s in self.reg["settings"]], ["s0", "b100", "b50", "b35", "b25", "b15", "s0r"])
        self.assertEqual({k for k, v in n.items() if abs(v["gap_pct"]) >= 3}, {"s0", "b100", "b15", "s0r"})
        self.assertEqual(n["b15"]["faster"], "pi4d1"); self.assertEqual(n["s0"]["faster"], "lapd1")
        self.assertEqual(self.reg["arms"]["base"]["env"], {"SERVE_DRAFT": "none"})
        self.assertIn("-devd RPC1", self.reg["arms"]["lapd1"]["env"]["SERVE_DRAFT"])


class TestShapeScorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = R.build()
        cls.m = cls.reg["model_parts"]
        cls.links = {s["id"]: ((R.speedups and cls.m["bw_gigabit"]) if not s["rate_mbit"] else s["rate_mbit"] * 1e6 * R.PAYLOAD / 8000.0,
                               0.85) for s in cls.reg["settings"]}

    def _rows(self, lap_scale=1.0, pi4_scale=1.0):
        """Requests whose ms per token are the model's own predictions (scaled), every arm and setting complete."""
        rows, T1, n = [], self.m["T1"], 64
        def req(session, arm, setting, ms):
            for p in self.reg["prompts"]:
                rows.append({"session": session, "arm": arm, "setting": setting, "id": p["id"],
                             "timings": {"predicted_ms": ms * n, "predicted_n": n}})
        p0 = R.speedups(self.m, self.links["s0"][0], 0.85)
        for i, arm in enumerate(self.reg["order"]):
            if arm == "base":
                req(i, arm, "s0", T1)
            elif arm == "pi4d1":
                req(i, arm, "s0", T1 / p0["pi4d1"] * pi4_scale)
            else:
                for s in self.reg["settings"]:
                    req(i, arm, s["id"], T1 / R.speedups(self.m, self.links[s["id"]][0], 0.85)["lapd1"] * lap_scale)
        return rows

    def test_a_run_that_matches_the_model_holds_everything(self):
        s = SH.score_rows(self.reg, self._rows(), self.links)
        self.assertEqual({k: v["verdict"] for k, v in s["hypotheses"].items()},
                         {"SH-FLIP": "HOLDS", "SH-CROSS": "HOLDS", "SH-ACC": "HOLDS", "SH-SLOPE": "HOLDS"})
        self.assertTrue(s["confirmed"])
        self.assertAlmostEqual(s["hypotheses"]["SH-CROSS"]["measured_mbit"], s["hypotheses"]["SH-CROSS"]["predicted_mbit"], delta=1.0)
        self.assertAlmostEqual(s["hypotheses"]["SH-SLOPE"]["c"], 1.0, places=6)

    def test_a_laptop_far_slower_than_priced_fails_the_flip(self):
        s = SH.score_rows(self.reg, self._rows(lap_scale=1.2), self.links)    # 20 % slow at every setting
        self.assertEqual(s["hypotheses"]["SH-FLIP"]["verdict"], "FAILS")
        self.assertFalse(s["confirmed"])

    def test_a_missing_request_is_incomplete(self):
        rows = self._rows()
        rows = [r for r in rows if not (r["arm"] == "pi4d1" and r["session"] == 3 and r["id"] == rows[0]["id"])]
        s = SH.score_rows(self.reg, rows, self.links)
        self.assertEqual(s["hypotheses"]["SH-FLIP"]["verdict"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
