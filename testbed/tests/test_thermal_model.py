"""thermal_model.py (upgrade 3, offline, 2026-10-10): the one-time-constant fit recovers known parameters from synthetic
1 Hz polls in throttle_poll.sh v2's format; stdlib unittest (and NumPy, as the fit uses it).

    python3 -m unittest discover -s testbed/tests -q
"""
import math, os, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
import thermal_model as TM                                    # noqa: E402


def synthetic(path, t_a, k, tau, pattern, t0=1791500000, T0=None):
    """A v2 poll whose temperature follows the model exactly, busy where pattern(t) is true (rounded to 0.01 C)."""
    T = t_a if T0 is None else T0
    with open(path, "w") as f:
        f.write("# throttle_poll v2 pid=1 start=%d\n" % t0)
        for i in range(1500):
            u = 1.0 if pattern(i) else 0.0
            f.write("%d throttled=0x0 temp=%.2f'C gov_khz=%d\n" % (t0 + i, T, 1800000 if u else 600000))
            a = math.exp(-1.0 / tau)
            T = a * T + (1 - a) * (t_a + k * u)


class TestThermalModel(unittest.TestCase):
    def test_the_fit_recovers_tau_and_k(self):
        with tempfile.TemporaryDirectory() as d:
            p1, p2 = os.path.join(d, "a.log"), os.path.join(d, "b.log")
            synthetic(p1, 50.0, 25.0, 60.0, lambda i: (i // 200) % 2 == 1)
            synthetic(p2, 46.0, 25.0, 60.0, lambda i: (i // 300) % 3 != 0, T0=55.0)
            r = TM.fit([p1, p2])
        self.assertAlmostEqual(r["tau_s"], 60.0, delta=1.5)
        self.assertAlmostEqual(r["k_c"], 25.0, delta=0.3)
        self.assertEqual([l["t_a_c"] for l in r["logs"]], [50.0, 46.0])
        self.assertLess(r["rmse_c"], 0.05)

    def test_steady_state_and_segments(self):
        self.assertEqual(TM.steady(50.0, 30.0, 0.9), 77.0)
        segs = TM.segments([(0, 50, 0), (1, 50, 0), (10, 51, 1), (11, 51, 1)])   # a 9 s gap splits it
        self.assertEqual([len(s) for s in segs], [2, 2])


if __name__ == "__main__":
    unittest.main()
