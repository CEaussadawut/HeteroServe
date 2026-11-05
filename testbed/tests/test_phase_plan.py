"""The phase scheduler's choice (2026-10-28): phase_plan_20261028.choose() with the calibration, with and without the
per-model power rows (phase_power_20261028), reproduces the picks ph1 and ph2 registered and measured."""
import json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "testbed", "scripts"), os.path.join(ROOT, "ThinkPad")]
import phase_plan_20261028 as PP  # noqa: E402
import phase_power_20261028 as W  # noqa: E402


class PhasePlan(unittest.TestCase):
    def setUp(self):
        self.c, self.rows = PP.load(), W.load()["rows"]
        self.ph1 = json.load(open(os.path.join(ROOT, "testbed", "dataset", "ph1_predictions_20261028.json")))
        self.ph2 = json.load(open(os.path.join(ROOT, "testbed", "dataset", "ph2_predictions_20261028.json")))

    def test_ph1s_picks_with_its_table(self):
        power = self.ph1["power_table"]
        for n, pick in self.ph1["picks"]["energy"].items():
            self.assertEqual(PP.choose(int(n) + 12, 256, "energy", self.c, power)["pick"], pick)
        for n, pick in self.ph1["picks"]["time"].items():
            self.assertEqual(PP.choose(int(n) + 12, 256, "time", self.c, power)["pick"], pick)

    def test_ph2s_picks_with_the_rows(self):
        for n, pick in self.ph2["picks"]["energy"].items():
            self.assertEqual(PP.choose(int(n) + 12, 256, "energy", self.c, self.ph2["power_table"], rows=self.rows)["pick"], pick)

    def test_the_rows_give_ph2s_registered_energies(self):
        for aid, a in self.ph2["arms"].items():
            e = PP.choose(a["n"] + 12, a["gen"], "energy", self.c, self.ph2["power_table"], rows=self.rows)["plans"][a["plan"]]["energy_j"]
            self.assertAlmostEqual(e, a["predicted"]["energy_j"], delta=0.01)

    def test_without_rows_nothing_changes(self):
        a = PP.choose(4108, 256, "energy", self.c, self.ph1["power_table"])
        b = PP.choose(4108, 256, "energy", self.c, self.ph1["power_table"], rows=None)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
