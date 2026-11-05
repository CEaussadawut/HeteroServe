"""Scheduler upgrade 5 (2026-10-10), the Jetson's power modes: the mode checks, the registered model and the scorer, on
synthetic data; stdlib unittest.

    python3 -m unittest discover -s testbed/tests -q
"""
import os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import power_modes_char_20261010 as PC                        # noqa: E402
import register_power_modes_20261010 as R                     # noqa: E402
import power_modes_20261010 as PM                             # noqa: E402

ASFOUND = "pmode=pmode:0000 online=0-3 cpu=102000-1479000 gpu=76800000-921600000 railgate=1 emcov=0 cpuidle=00"
LOCKED = "pmode=pmode:0000 online=0-3 cpu=1479000-1479000 gpu=921600000-921600000 railgate=0 emcov=1 cpuidle=11"
FIVE = "pmode=pmode:0001 online=0-1 cpu=921600-921600 gpu=614400000-614400000 railgate=0 emcov=1 cpuidle=11"


class TestModeChecks(unittest.TestCase):
    """The states read by hand on the Jetson, 2026-10-10 16:45."""
    def test_each_mode_accepts_its_own_state_only(self):
        self.assertTrue(PC.mode_ok("locked", LOCKED, ASFOUND))
        self.assertTrue(PC.mode_ok("5w", FIVE, ASFOUND))
        self.assertTrue(PC.mode_ok("unlocked", ASFOUND, ASFOUND))
        for m, s in (("locked", ASFOUND), ("locked", FIVE), ("5w", LOCKED), ("5w", ASFOUND), ("unlocked", LOCKED)):
            self.assertFalse(PC.mode_ok(m, s, ASFOUND), (m, s))

    def test_nvpmodel_m0_alone_does_not_count_as_unlocked(self):
        # after 5 W locked, nvpmodel -m 0 leaves railgating off, the EMC override on and cpuidle off: only the restore
        # of the stored state gets back to as found
        m0 = "pmode=pmode:0000 online=0-3 cpu=102000-1479000 gpu=76800000-921600000 railgate=0 emcov=1 cpuidle=11"
        self.assertFalse(PC.mode_ok("unlocked", m0, ASFOUND))
        self.assertFalse(PC.mode_ok("locked", m0, ASFOUND))

    def test_every_switch_passes_through_the_as_found_state(self):
        for m in PC.MODES:
            c = PC.mode_cmd(m, "~/x.stored")
            self.assertIn("jetson_clocks --restore ~/x.stored", c)
            self.assertLess(c.index("--restore"), c.index("nvpmodel -m 1") if "nvpmodel -m 1" in c else len(c))


def analysis(pw, a, dT, gaps=(0.0, 1.1, 1.3, 1.5), lap=600.0):
    """A synthetic analysis.json whose Jetson follows P = Pw + A / T exactly: per mode its waiting power, its block's
    energy above waiting and its block-time change; two blocks per cell, at slightly different cycles."""
    out = {}
    for m in R.MODES:
        for g in gaps:
            blocks = []
            for b, jitter in ((0, 0.0), (1, 40.0)):
                T = lap + dT[m] + 1000 * g + 5 + jitter
                blocks.append({"block": b, "period_power_ms": T, "strip_w": pw[m] + a[m] / (T / 1000.0)})
            out["%s@%.1f" % (m, g)] = {"gap_s": g, "forward_ms_p10": lap + dT[m], "forward_ms_median": lap + dT[m] + 30,
                                        "blocks": blocks}
    return {"cells": out}


class TestModel(unittest.TestCase):
    def setUp(self):
        self.pw = {"locked": 4.0, "unlocked": 3.7, "5w": 3.5}
        self.a = {"locked": 2.0, "unlocked": 2.05, "5w": 2.3}
        self.dT = {"locked": 0.0, "unlocked": 12.0, "5w": 165.0}

    def test_the_fit_recovers_the_waiting_power_and_the_block_energy(self):
        pw, a, r2, res = R.fit_power([(T, 4.0 + 2.0 / (T / 1000.0)) for T in (650.0, 1700.0, 1900.0, 2100.0)])
        self.assertAlmostEqual(pw, 4.0, places=9)
        self.assertAlmostEqual(a, 2.0, places=9)
        self.assertAlmostEqual(r2, 1.0, places=9)

    def test_the_formula_holds(self):
        T0, E0, pwp = 1903.1, 20.195, 2.391
        p = R.predict(analysis(self.pw, self.a, self.dT), T0, E0, pwp)
        L, U, F = (p["modes"][m] for m in R.MODES)
        self.assertAlmostEqual(L["T_ms"], T0)
        self.assertAlmostEqual(L["E_j"], E0)
        self.assertAlmostEqual(U["dT_ms"], 12.0)
        self.assertAlmostEqual(F["T_ms"], T0 + 165.0)
        ej = lambda m, T: self.pw[m] * T / 1000 + self.a[m]
        want = E0 + ej("5w", T0 + 165) - ej("locked", T0) + pwp * 0.165
        self.assertAlmostEqual(F["E_j"], want, places=9)
        self.assertAlmostEqual(U["Pw_w"], 3.7, places=9)

    def test_pick_needs_its_margin(self):
        p = R.predict(analysis(self.pw, self.a, self.dT), 1903.1, 20.195, 2.391)
        self.assertEqual(p["pick"]["judged"], p["pick"]["margin_over_next"] >= R.PICK_MARGIN)

    def test_a_mode_without_enough_cells_is_refused(self):
        an = analysis(self.pw, self.a, self.dT, gaps=(1.3,))
        for k in an["cells"]:
            an["cells"][k]["blocks"] = an["cells"][k]["blocks"][:1]
        with self.assertRaises(ValueError):
            R.predict(an, 1903.1, 20.195, 2.391)


class TestScore(unittest.TestCase):
    def setUp(self):
        self.reg = {"order": ["locked", "unlocked", "5w", "5w", "unlocked", "locked"], "prompts": [{}] * 4,
                    "tolerance": 0.05, "pick_margin": 0.02,
                    "pick": {"least_energy": "unlocked", "margin_over_next": 0.03, "judged": True},
                    "predictions": {"locked": {"T_ms": 1900.0, "E_j": 20.0, "dT_ms": 0.0, "dE_j": 0.0, "Ej_j": 9.0},
                                    "unlocked": {"T_ms": 1915.0, "E_j": 19.4, "dT_ms": 15.0, "dE_j": -0.6, "Ej_j": 8.4},
                                    "5w": {"T_ms": 2065.0, "E_j": 20.0, "dT_ms": 165.0, "dE_j": 0.0, "Ej_j": 8.6}}}
        self.meas = {"locked": {"requests": 8, "T_ms": 1910.0, "E_j": 20.3, "Ej_jetson_j": 9.1},
                     "unlocked": {"requests": 8, "T_ms": 1925.0, "E_j": 19.5, "Ej_jetson_j": 8.3},
                     "5w": {"requests": 8, "T_ms": 2300.0, "E_j": 20.6, "Ej_jetson_j": 8.7}}

    def test_verdicts(self):
        s = PM.score_rows(self.reg, self.meas)
        h = s["hypotheses"]
        self.assertEqual(h["PM-T-locked"]["verdict"], "HOLDS")
        self.assertEqual(h["PM-T-5w"]["verdict"], "FAILS")                      # +11.4 %
        self.assertEqual(h["PM-E-5w"]["verdict"], "HOLDS")
        self.assertEqual(h["PM-PICK"]["verdict"], "HOLDS")
        self.assertEqual(s["upgrade5"], "FAILS")
        self.assertAlmostEqual(s["reported"]["unlocked"]["dE_j_measured"], -0.8)

    def test_a_missing_request_makes_that_mode_incomplete(self):
        m = dict(self.meas, unlocked=dict(self.meas["unlocked"], requests=7))
        s = PM.score_rows(self.reg, m)
        self.assertEqual(s["hypotheses"]["PM-T-unlocked"]["verdict"], "INCOMPLETE")
        self.assertEqual(s["upgrade5"], "INCOMPLETE")

    def test_pick_too_close_is_not_judged(self):
        r = dict(self.reg, pick={"least_energy": "unlocked", "margin_over_next": 0.01, "judged": False})
        self.assertEqual(PM.score_rows(r, self.meas)["hypotheses"]["PM-PICK"]["verdict"], "NOT JUDGED")


if __name__ == "__main__":
    unittest.main()
