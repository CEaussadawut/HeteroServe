"""Tests for testbed/scripts/full_eval.py's pure parts (placement specs, the two-level bootstrap), stdlib unittest only.

    python3 -m unittest discover -s testbed/tests -q
"""
import os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import full_eval as F                                         # noqa: E402
import qr3_tpot as Q                                          # noqa: E402


class TestFullEval(unittest.TestCase):
    def test_placement_specs(self):
        self.assertEqual(F.parse_placement("engine"), ("engine", None, None))
        self.assertEqual(F.parse_placement("halda=pi4:1+jetson:21/host"), ("halda", [("pi4", 1), ("jetson", 21)], "host"))
        self.assertEqual(F.parse_placement("even=pi4:11+jetson:11/last"), ("even", [("pi4", 11), ("jetson", 11)], "last"))

    def test_thermal_window(self):
        def poll(t_from, t_to, hot=()):
            return ["# throttle_poll v2 test"] + ["%d throttled=0x%x temp=%s'C gov_khz=1800000" % (
                t, 0x80008 if t in hot else 0x0, "81.0" if t in hot else "60.0") for t in range(t_from, t_to + 1)]
        lines = poll(1000, 1100)
        self.assertEqual(F.thermal(lines, 1010.4, 1030.7)["verdict"], "clean")
        self.assertEqual(F.thermal(poll(1000, 1100, hot={1020}), 1010.4, 1030.7)["verdict"], "THROTTLED")
        self.assertEqual(F.thermal(poll(1000, 1100, hot={1040}), 1010.4, 1030.7)["verdict"], "clean")   # outside the window
        self.assertEqual(F.thermal(lines, 1200.0, 1210.0)["verdict"], "UNKNOWN")       # after the poll's last line
        self.assertEqual(F.thermal(poll(1000, 1020), 1010.0, 1030.0)["verdict"], "UNKNOWN")   # it died inside the window
        self.assertEqual(F.thermal(poll(1020, 1100), 1010.0, 1030.0)["verdict"], "UNKNOWN")   # it started late
        self.assertEqual(F.thermal(poll(1000, 1028), 1010.0, 1030.0)["verdict"], "clean")     # within 3 s of the edge
        gap = [l for l in lines if not l.startswith("1015") and not l.startswith("1016")
               and not l.startswith("1017") and not l.startswith("1018")]
        self.assertEqual(F.thermal(gap, 1010.0, 1030.0)["verdict"], "UNKNOWN")          # a 5 s hole inside it

    def test_bootstrap_is_deterministic_and_brackets_the_statistic(self):
        rounds = {r: [[90.0 + r + 0.1 * i for i in range(50)] for _ in range(3)] for r in range(1, 6)}
        pooled = [g for reqs in rounds.values() for req in reqs for g in req]
        a = F.boot2(rounds, lambda s: Q.pct(s, 50), B=300)
        self.assertEqual(a, F.boot2(rounds, lambda s: Q.pct(s, 50), B=300))       # seeded
        self.assertLessEqual(a[0], Q.pct(pooled, 50))
        self.assertGreaterEqual(a[1], Q.pct(pooled, 50))


if __name__ == "__main__":
    unittest.main()
