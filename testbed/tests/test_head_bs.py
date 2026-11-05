"""Scheduler upgrade 6 (2026-10-10), backend sampling on the head board: the scorer, on synthetic runs; stdlib unittest.

    python3 -m unittest discover -s testbed/tests -q
"""
import os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import head_bs_20261010 as HB                                 # noqa: E402

V = 607744
REG = {"V_bytes": V, "abba": ["off", "on", "on", "off"], "prompts": [{"id": "a"}, {"id": "b"}],
       "settings": [{"id": "s0"}, {"id": "b100"}, {"id": "b50"}],
       "thresholds": {"bytes_fraction": 0.9, "save_ms": 1.0, "slope_pct": 10.0}}
GOOD = {"s0": 940.0, "b100": 95.0, "b50": 47.5}


def rows(extra_ms=0.0, logits_cross_when_on=False, text_on="x"):
    """Two prompts x ABBA per setting; the off arm pays V / g (+ extra_ms), 64 tokens per request."""
    out = []
    for s, g in GOOD.items():
        cost = V / (g * 1e6 / 8 / 1000.0)
        for rep, arm in enumerate(REG["abba"]):
            for p in ("a", "b"):
                ms = 100.0 + (cost + extra_ms if arm == "off" else 0.0)
                sent = 64 * (V + 300 if (arm == "off" or logits_cross_when_on) else 300)
                out.append({"setting": s, "arm": arm, "rep": rep, "id": p, "jetson_sent_bytes": sent,
                            "timings": {"predicted_ms": 64 * ms, "predicted_n": 64},
                            "content": "x" if arm == "off" else text_on})
    return out


class TestScore(unittest.TestCase):
    def test_exact_transfer_holds_everything(self):
        s = HB.score_rows(REG, rows(), GOOD)
        self.assertEqual(s["upgrade6"], "HOLDS")
        self.assertTrue(s["works_over_rpc"])
        self.assertAlmostEqual(s["hypotheses"]["HB-SLOPE"]["intercept_ms"], 0.0, places=6)

    def test_a_constant_left_over_fails_save_but_not_slope(self):
        s = HB.score_rows(REG, rows(extra_ms=2.0), GOOD)
        h = s["hypotheses"]
        self.assertEqual(h["HB-SAVE-s0"]["verdict"], "FAILS")
        self.assertEqual(h["HB-SLOPE"]["verdict"], "HOLDS")
        self.assertAlmostEqual(h["HB-SLOPE"]["intercept_ms"], 2.0, places=6)

    def test_logits_still_crossing_fails_bytes(self):
        s = HB.score_rows(REG, rows(logits_cross_when_on=True), GOOD)
        self.assertEqual(s["hypotheses"]["HB-BYTES"]["verdict"], "FAILS")
        self.assertFalse(s["works_over_rpc"])

    def test_different_text_fails_same(self):
        self.assertEqual(HB.score_rows(REG, rows(text_on="y"), GOOD)["hypotheses"]["HB-SAME"]["verdict"], "FAILS")

    def test_registration_b_prices_n_minus_1_passes_over_n_tokens(self):
        f = 63 / 64.0
        r = [dict(x, timings={"predicted_ms": x["timings"]["predicted_ms"] if x["arm"] == "on" else
                              64 * 100.0 + f * (x["timings"]["predicted_ms"] - 64 * 100.0), "predicted_n": 64}) for x in rows()]
        self.assertEqual(HB.score_rows(dict(REG, per_token_factor=f), r, GOOD)["upgrade6"], "HOLDS")
        self.assertEqual(HB.score_rows(REG, r, GOOD)["hypotheses"]["HB-SAVE-b50"]["verdict"], "FAILS")   # a: 1.6 ms off

    def test_a_missing_request_makes_its_setting_incomplete(self):
        r = rows()[1:]
        s = HB.score_rows(REG, r, GOOD)
        self.assertEqual(s["hypotheses"]["HB-SAVE-s0"]["verdict"], "INCOMPLETE")
        self.assertEqual(s["upgrade6"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
