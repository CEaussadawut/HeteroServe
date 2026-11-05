"""Tests for testbed/scripts/score_full_eval.py's verdicts, on synthetic runs, stdlib unittest only.

    python3 -m unittest discover -s testbed/tests -q
"""
import copy, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
import score_full_eval as S                                   # noqa: E402

REG = {"rounds": 2, "reqs": 2, "gen": 4,
       "configs": {"a": {"assign": [["x", 22]], "head": "last", "predicted": {"decode_ms": 80.0, "decode_gross_j_all_boards": 1.0}},
                   "b": {"assign": [["x", 1], ["y", 21]], "head": "last",
                         "predicted": {"decode_ms": 90.0, "decode_gross_j_all_boards": 1.2}}}}


def req(lab, rnd, i, ntok=4):
    return {"kind": "req", "data": {"placement": lab, "round": rnd, "i": i,
                                    "events": [[0.1 * k, [k], k + 1, 80.0 * k, 0.0] for k in range(ntok)]}}


def rows(skip=(), short=()):
    return [req(lab, r, i, 3 if (lab, r, i) in short else 4) for r in (1, 2) for lab in ("a", "b") for i in (0, 1)
            if (lab, r, i) not in skip]


def summ(a_mean=81.0, b_mean=91.0, a_ci=(79.0, 82.0), b_ci=(88.0, 93.0), a_j=1.05, b_j=1.25):
    def one(mean, ci, j):
        return {"server_tpot": {"mean": mean, "p50": mean, "p95": mean + 5}, "p50_ci95": list(ci),
                "p95_ci95": [ci[0] + 5, ci[1] + 5], "j_per_token_all_boards": j, "j_per_token_all_boards_rounds": [j, j],
                "thermal": "clean", "thermal_windows": {"clean": 4}}
    return {"a": one(a_mean, a_ci, a_j), "b": one(b_mean, b_ci, b_j)}


class TestScoreFullEval(unittest.TestCase):
    def test_all_hold_when_the_predicted_best_wins_clearly(self):
        s = S.score(REG, summ(), rows())
        self.assertTrue(s["complete"])
        self.assertEqual({h: v["verdict"] for h, v in s["hypotheses"].items()},
                         {"FE-TPOT": "HOLDS", "FE-ENERGY": "HOLDS", "FE-ACC": "HOLDS"})
        self.assertEqual(s["hypotheses"]["FE-TPOT"]["ci_gap_p50_ms"], {"b": 6.0})
        self.assertEqual(s["hypotheses"]["FE-ENERGY"]["rounds_lowest"], "2 of 2")

    def test_overlapping_cis_fail_tpot_even_when_the_point_estimate_is_lowest(self):
        s = S.score(REG, summ(a_ci=(79.0, 89.0)), rows())
        t = s["hypotheses"]["FE-TPOT"]
        self.assertTrue(t["lowest_p50"] and t["lowest_p95"])
        self.assertEqual(t["verdict"], "FAILS")
        self.assertEqual(t["ci_gap_p50_ms"], {"b": -1.0})

    def test_touching_cis_are_not_clear(self):
        self.assertEqual(S.score(REG, summ(a_ci=(79.0, 88.0)), rows())["hypotheses"]["FE-TPOT"]["verdict"], "FAILS")

    def test_energy_fails_when_another_placement_measures_lower(self):
        self.assertEqual(S.score(REG, summ(a_j=1.3), rows())["hypotheses"]["FE-ENERGY"]["verdict"], "FAILS")

    def test_a_missing_or_short_request_makes_every_verdict_incomplete(self):
        for kw in ({"skip": {("b", 2, 1)}}, {"short": {("a", 1, 0)}}):
            s = S.score(REG, summ(), rows(**kw))
            self.assertFalse(s["complete"], kw)
            self.assertEqual({v["verdict"] for v in s["hypotheses"].values()}, {"INCOMPLETE"}, kw)
            self.assertEqual(s["hypotheses"]["FE-TPOT"]["on_data_present"], "holds", kw)

    def test_a_placement_missing_from_the_summary_is_incomplete(self):
        sm = summ()
        del sm["b"]
        s = S.score(REG, sm, rows())
        self.assertFalse(s["placements"]["b"]["complete"])
        self.assertEqual(s["hypotheses"]["FE-ACC"]["verdict"], "INCOMPLETE")

    def test_acc_takes_the_error_against_the_measured_mean_and_flags_a_split(self):
        # predicted 80 and 90; a measures 72.3: 100 * (80 / 72.3 - 1) = +10.65 % (outside), against the prediction -9.6 %
        s = S.score(REG, summ(a_mean=72.3, a_ci=(70.0, 74.0)), rows())
        a = s["hypotheses"]["FE-ACC"]
        self.assertEqual(a["verdict"], "FAILS")
        self.assertEqual(a["placements"]["a"]["err_pct"], 10.7)
        self.assertEqual(a["placements"]["a"]["err_vs_predicted_pct"], -9.6)
        self.assertTrue(a["readings_split"])
        # a slow placement: b measures 99.5, -9.5 % on the registered reading (inside), +10.6 % on the other
        b = S.score(REG, summ(b_mean=99.5, b_ci=(97.0, 101.0)), rows())["hypotheses"]["FE-ACC"]
        self.assertEqual(b["verdict"], "HOLDS")
        self.assertTrue(b["readings_split"])

    def test_inputs_are_not_modified(self):
        reg, sm, rw = copy.deepcopy(REG), summ(), rows()
        before = (copy.deepcopy(reg), copy.deepcopy(sm), copy.deepcopy(rw))
        S.score(reg, sm, rw)
        self.assertEqual((reg, sm, rw), before)


if __name__ == "__main__":
    unittest.main()
