"""Tests for node/bench_windows.py -- the per-repetition energy window builder.

This module writes `windows.jsonl`, which is the PRIMARY artefact of a sweep: every energy
figure in the thesis is joined through it. Two properties matter more than anything else
here, and both are tested by provoking the failure rather than by inspection:

  1. It must never emit a line that is not valid JSON. Python's json accepts NaN/Infinity
     as bare tokens; the JSON spec does not, and `jq` silently rewrites NaN to `null`.
  2. It must never raise. A sweep that dies at hour six because a sensor was unplugged is
     worse than a flagged row, and a row that silently means something else is worse still.

    python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "node"))

import bench_windows  # noqa: E402

ARG_NAMES = ("run", "node", "model", "phase", "n", "threads", "reps", "t_start", "t_end",
             "rc", "json", "stderr_tsv", "bench_json", "stamped", "idle_start", "idle_end",
             "temp_start_c", "temp_end_c", "throttled_start", "throttled_end")


class Args:
    """Stand-in for the argparse namespace; every field defaults to the empty string."""

    def __init__(self, **kw):
        for name in ARG_NAMES:
            setattr(self, name, "")
        for k, v in kw.items():
            setattr(self, k.replace("-", "_"), v)


def strict_loads(text):
    """json.loads that REJECTS NaN/Infinity, the way a conforming parser would."""
    return json.loads(
        text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError("bare %s" % c)))


class TestFloatGuard(unittest.TestCase):
    def test_non_finite_values_never_enter_the_record(self):
        # scripts/lib.sh soc_temp_c() really does emit the literal "nan" when the thermal
        # zone cannot be read, and float("nan") succeeds -- so this path is reachable.
        self.assertEqual(bench_windows._f("nan", ""), "")
        self.assertEqual(bench_windows._f("inf", ""), "")
        self.assertEqual(bench_windows._f("-inf", ""), "")
        self.assertEqual(bench_windows._f("NaN", ""), "")
        self.assertEqual(bench_windows._f("12.5", ""), 12.5)
        self.assertEqual(bench_windows._f("", "x"), "x")
        self.assertEqual(bench_windows._f(None, "x"), "x")

    def test_a_dead_thermal_sensor_still_yields_valid_json(self):
        rec = bench_windows.build(Args(rc="0", reps="5", t_start="1000", t_end="1030",
                                       temp_start_c="nan", temp_end_c="nan"))
        text = json.dumps(rec, allow_nan=False)      # would raise if NaN survived
        parsed = strict_loads(text)
        self.assertEqual(parsed["temp_start_c"], "")
        self.assertEqual(parsed["temp_end_c"], "")

    def test_main_writes_one_strictly_valid_json_line(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = bench_windows.main([
                "--run", "R", "--node", "pi4", "--model", "m", "--phase", "pp",
                "--n", "128", "--threads", "4", "--reps", "5",
                "--t-start", "1000", "--t-end", "1030", "--rc", "0",
                "--temp-start-c", "nan", "--temp-end-c", "nan",
            ])
        self.assertEqual(rc, 0)
        lines = [x for x in buf.getvalue().splitlines() if x.strip()]
        self.assertEqual(len(lines), 1)
        strict_loads(lines[0])                        # must not raise


class TestFallbackContract(unittest.TestCase):
    def test_every_record_states_its_requested_repetition_count(self):
        """`reps` exists only when derivation succeeded. Without `reps_declared` a fallback
        run cannot say what R it was, and a consumer keying off `reps` deletes it silently
        instead of reporting it as excluded -- which is exactly what happened once."""
        rec = bench_windows.build(Args(rc="3", reps="5", t_start="1000", t_end="1030"))
        self.assertEqual(rec["window_kind"], "process")
        self.assertNotIn("reps", rec)
        self.assertEqual(rec["reps_declared"], 5)

    def test_a_failed_run_falls_back_with_a_named_reason_and_never_raises(self):
        rec = bench_windows.build(Args(rc="3", reps="5", t_start="1000", t_end="1030"))
        self.assertEqual(rec["window_kind"], "process")
        self.assertTrue(rec["window_fallback_reason"].startswith("G8:"))

    def test_missing_input_files_degrade_rather_than_raise(self):
        rec = bench_windows.build(Args(rc="0", reps="5", t_start="1000", t_end="1030",
                                       bench_json="/nonexistent/x.json",
                                       stamped="/nonexistent/x.tsv"))
        self.assertEqual(rec["window_kind"], "process")
        self.assertIn("G1", rec["window_fallback_reason"])

    def test_the_raw_process_window_keeps_its_meaning_on_every_record(self):
        """t_start/t_end are the whole-process bracket, permanently. Redefining a field in
        the primary artefact is the defect class this whole module exists to remove."""
        for rc in ("0", "3"):
            rec = bench_windows.build(Args(rc=rc, reps="5", t_start="1000", t_end="1030"))
            with self.subTest(rc=rc):
                self.assertEqual((rec["t_start"], rec["t_end"]), (1000.0, 1030.0))


if __name__ == "__main__":
    unittest.main()
