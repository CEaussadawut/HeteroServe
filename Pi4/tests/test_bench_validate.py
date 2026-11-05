"""Tests for node/bench_validate.py -- the independent cross-check of the energy window.

The energy window is derived from llama-bench's `--progress` markers and the node's wall
clock. `bench_validate.py` checks that derivation a second way, by fitting
`E_process(R) = E_fixed + R * E_rep` over R = 1, 3, 5 -- a route that uses no timestamps at
all. These tests care about one thing above all: that it REPORTS DISAGREEMENT rather than
reconciling it, and that it never claims a cross-check happened when it did not.

    python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "node"))

import bench_validate  # noqa: E402

E_FIXED = 30.0
E_REP = 12.0
S_REP = 12.4


def write_sweep(run_dir: Path, *, reps_set=(1, 3, 5), kind="reps-span",
                sub_scale=1.0, s_rep=S_REP, fallback_at=()):
    """A synthetic *-validate sweep: one cell measured at several repetition counts.

    FAITHFULNESS MATTERS HERE. This fixture must emit what node/bench_windows.py really
    emits, key for key. An earlier version wrote the derived `reps` list onto fallback
    records too -- which the real producer never does, because `reps` is added only when
    every gate passes. That single unfaithful key hid a real defect: bench_validate.py
    keyed inclusion off `reps`, so fallback runs were silently deleted before the
    "these runs fell back" warning could fire, and the tool reported agreement over a
    quietly smaller set of points. The test passed because the fixture was wrong.

    `fallback_at` is a set of R values that should fall back; those records get
    `window_kind="process"`, a reason, and NO `reps`/`t_meas_*`/`rep_*` keys at all.
    """
    windows, energy = [], {}
    for r in reps_set:
        tag = "tinyllama-1.1b-q4km_pp128_t4_r%d" % r
        fell_back = (kind != "reps-span") or (r in fallback_at)
        rec = {
            "schema": 2, "run": "V", "node": "pi4", "model": "tinyllama-1.1b-q4km",
            "phase": "pp", "n": 128, "threads": 4,
            "reps_declared": r,          # written on EVERY record, fallback included
            "t_start": 990.0, "t_end": 1000.0 + r * s_rep + 3.0,
            "json": tag + ".json",
        }
        if fell_back:
            rec["window_kind"] = "process"
            rec["window_fallback_reason"] = "G2:marker-count-0-vs-samples-%d" % r
        else:
            ns = int(s_rep * 1e9)
            rec.update({
                "window_kind": "reps-span", "window_fallback_reason": "",
                "t_meas_start": 1000.0, "t_meas_end": 1000.0 + r * s_rep,
                "reps": [{"i": i + 1, "t_start": 1000.0 + i * s_rep,
                          "t_end": 1000.0 + (i + 1) * s_rep, "ns": ns} for i in range(r)],
                "rep_span_s": r * s_rep, "rep_busy_s": r * s_rep,
            })
        windows.append(rec)
        energy[tag + ".json"] = {"process": E_FIXED + r * E_REP,
                                 "reps": r * E_REP * sub_scale}
    (run_dir / "windows.jsonl").write_text(
        "".join(json.dumps(w) + "\n" for w in windows), encoding="utf-8")
    return energy


class TestFit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_fit_recovers_the_per_repetition_and_fixed_energy(self):
        energy = write_sweep(self.dir)
        report = bench_validate.analyse(self.dir, energy)
        cell = report["cells"][0]
        # slope is energy per repetition, intercept is everything paid once per process
        self.assertAlmostEqual(cell["fit"]["slope"], E_REP, places=6)
        self.assertAlmostEqual(cell["fit"]["intercept"], E_FIXED, places=6)
        self.assertTrue(cell["agrees"])
        self.assertEqual(report["problems"], [])

    def test_a_wrong_sub_window_is_reported_not_reconciled(self):
        # The sub-window over-reads by 20%, as it would if warmup still leaked into it.
        energy = write_sweep(self.dir, sub_scale=1.20)
        report = bench_validate.analyse(self.dir, energy)
        cell = report["cells"][0]
        self.assertFalse(cell["agrees"])
        self.assertAlmostEqual(cell["agreement_deviation"], 0.20, places=4)
        self.assertTrue(any("DISAGREE" in p for p in report["problems"]))
        # the fit itself is still correct -- only the sub-window claim moved
        self.assertAlmostEqual(cell["fit"]["slope"], E_REP, places=6)

    def test_whole_process_energy_alone_cannot_pass_the_cross_check(self):
        """A bare number is a process integral. Dividing it by R is NOT per-repetition
        energy -- it still carries E_fixed/R. Accepting it would manufacture agreement."""
        energy = {k: v["process"] for k, v in write_sweep(self.dir).items()}
        report = bench_validate.analyse(self.dir, energy)
        cell = report["cells"][0]
        self.assertNotIn("agrees", cell)
        self.assertNotIn("subwindow_j_per_rep", cell)
        self.assertTrue(any("cross-check did NOT happen" in p for p in report["problems"]))

    def test_no_energy_says_plainly_that_the_check_did_not_run(self):
        write_sweep(self.dir)
        report = bench_validate.analyse(self.dir, {})
        self.assertIn("NOT been performed", report["note"])
        self.assertNotIn("fit", report["cells"][0])
        # the timing side is still checked
        self.assertAlmostEqual(report["cells"][0]["mean_s_per_rep"], S_REP, places=6)

    def test_runs_that_fell_back_to_the_process_window_are_excluded(self):
        energy = write_sweep(self.dir, kind="process")
        report = bench_validate.analyse(self.dir, energy)
        cell = report["cells"][0]
        self.assertEqual(cell["fell_back_at_R"], [1, 3, 5])
        self.assertNotIn("fit", cell)
        self.assertTrue(any("fell back" in p for p in report["problems"]))

    def test_a_fallback_in_the_middle_is_reported_not_silently_dropped(self):
        """The regression that matters: a MIX of good and fallback runs.

        The all-fallback case above passes either way, because with nothing left to fit
        there is no misleading result to produce. The dangerous shape is one bad R among
        good ones: the tool then still fits a line, still reports agreement, and the
        missing point is invisible unless the fallback is counted BEFORE records are
        filtered. That is the defect this test exists for.
        """
        energy = write_sweep(self.dir, fallback_at={3})
        report = bench_validate.analyse(self.dir, energy)
        cell = report["cells"][0]

        # R=3 must still appear on the R axis, not vanish from the analysis
        self.assertEqual(cell["reps_observed"], [1, 3, 5])
        self.assertEqual(cell["fell_back_at_R"], [3])
        self.assertTrue(any("fell back" in p for p in report["problems"]))

        # it is excluded from the FIT (its window is the wrong one) but the exclusion is
        # stated, and the now-degenerate 2-point fit says so rather than looking clean
        self.assertEqual(cell["fit"]["points"], 2)
        self.assertTrue(cell["fit"]["residual_is_meaningless"])
        self.assertTrue(any("no degrees of freedom" in p for p in report["problems"]))

    def test_a_record_that_cannot_state_its_rep_count_is_reported(self):
        """No derived sub-windows and no reps_declared: the run cannot be placed on the R
        axis. It must be named, not quietly skipped."""
        (self.dir / "windows.jsonl").write_text(json.dumps({
            "model": "m", "phase": "pp", "n": 128, "threads": 4,
            "window_kind": "process", "json": "m_pp128_t4_r3.json",
        }) + "\n", encoding="utf-8")
        report = bench_validate.analyse(self.dir, {})
        self.assertEqual(report["unusable_records"], ["m_pp128_t4_r3.json"])
        self.assertTrue(any("neither derived sub-windows nor reps_declared" in p
                            for p in report["problems"]))

    def test_a_single_repetition_count_yields_no_fit(self):
        # One point cannot define a line; the module must not invent a slope.
        energy = write_sweep(self.dir, reps_set=(5,))
        report = bench_validate.analyse(self.dir, energy)
        self.assertNotIn("fit", report["cells"][0])
        self.assertTrue(any("need at least 2" in p for p in report["problems"]))

    def test_an_unreproducible_cell_is_flagged_before_any_energy_claim(self):
        """If seconds-per-repetition drifts across R, the cell is not the same experiment
        at each R and an energy fit over it would be meaningless."""
        (self.dir / "windows.jsonl").write_text("", encoding="utf-8")
        energy = {}
        windows = []
        for r, s in ((1, 12.4), (3, 12.4), (5, 20.0)):   # R=5 ran 60% slower
            tag = "m_pp128_t4_r%d" % r
            reps = [{"i": i + 1, "t_start": 0, "t_end": s, "ns": int(s * 1e9)}
                    for i in range(r)]
            windows.append({
                "model": "m", "phase": "pp", "n": 128, "threads": 4,
                "window_kind": "reps-span", "reps": reps,
                "rep_span_s": r * s, "rep_busy_s": r * s, "json": tag + ".json",
            })
            energy[tag + ".json"] = {"process": E_FIXED + r * E_REP, "reps": r * E_REP}
        (self.dir / "windows.jsonl").write_text(
            "".join(json.dumps(w) + "\n" for w in windows), encoding="utf-8")
        report = bench_validate.analyse(self.dir, energy)
        self.assertTrue(any("not reproducible" in p for p in report["problems"]))

    def test_a_missing_or_empty_sweep_does_not_raise(self):
        self.assertEqual(bench_validate.analyse(self.dir, {})["cells"], [])
        (self.dir / "windows.jsonl").write_text("not json\n\n", encoding="utf-8")
        self.assertEqual(bench_validate.analyse(self.dir, {})["cells"], [])


if __name__ == "__main__":
    unittest.main()
