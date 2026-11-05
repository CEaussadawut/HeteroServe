"""Tests for bench/parse_bench.py -- the code every number in the dataset passes through.

Every measurement this project publishes is folded by this one script, so a defect here
is not a bug in a tool, it is a wrong result in a thesis. It has already happened once:
`c_per_layer_ms` -- the c_i in Rule A of proposal Appendix B.6 -- was computed from a
layer count read out of llama-bench's JSON, which never contained one. The column came
out blank on every row while the CSV still looked complete, so the dataset silently
carried nothing the placement rule could use.

These tests exist mainly to make that class of failure loud. The fixtures are built from
llama-bench's *real* field list (see llama_bench_fixture.py), because a fixture that
invents field names would have passed that whole time.
"""
from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARSE_BENCH = ROOT / "bench" / "parse_bench.py"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama_bench_fixture as fx  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("parse_bench", PARSE_BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pb = _load()

MANIFEST = {"node_id": "pi5", "llama_cpp_sha": "deadbeef"}


def run_cli(run_dir):
    """Invoke the script the way 10_isolated_bench.sh does, and keep the exit code."""
    proc = subprocess.run(
        [sys.executable, str(PARSE_BENCH), str(run_dir)],
        capture_output=True, text=True)
    return proc


def read_dataset(run_dir):
    with (Path(run_dir) / "dataset.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class RunDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def build(self, cells, manifest=MANIFEST, name="run-20260913-000000"):
        return fx.write_run(self.tmp / name, cells, manifest)


class TestWorkedExample(RunDirCase):
    """The numbers the handoff carried forward as the reference result. If these move,
    either the arithmetic changed or the fixture drifted -- both are worth a stop."""

    def test_derived_columns(self):
        run = self.build([fx.cell()])
        proc = run_cli(run)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rows = read_dataset(run)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["decode_tok_s"], "2.35")
        self.assertEqual(r["prefill_tok_s"], "11.5")
        # 1000 / 2.35
        self.assertEqual(r["tpot_ms"], "425.5")
        # 425.5319... / 32 layers -- the c_i Rule A consumes
        self.assertEqual(r["c_per_layer_ms"], "13.2979")
        # 4.7 GB re-read once per decoded token, at 2.35 tok/s
        self.assertEqual(r["bw_util_gbs"], "11.045")
        # 1000 * 256 prompt tokens / 11.5 tok/s
        self.assertEqual(r["ttft_est_ms"], "22260.9")
        self.assertEqual(r["n_layers"], "32")
        self.assertEqual(r["node_id"], "pi5")
        self.assertEqual(r["llama_cpp_sha"], "deadbeef")
        self.assertEqual(r["backend"], "CPU")
        # Canonical, matching results.csv -- `quant` used to hold the raw llama-bench
        # string here and the canonical label there, so one name meant two things.
        self.assertEqual(r["quant"], "Q4_K_M")
        self.assertEqual(r["model_type"], "Q4_K - Medium")

    def test_bw_util_above_peak_is_visible(self):
        """Above the Pi 5's ~17 GB/s the model was not actually resident and the cell is
        void. The column is the only thing that shows it, so it must be computed even
        when the value is absurd rather than quietly clamped."""
        run = self.build([fx.cell(decode_ts=10.0)])
        run_cli(run)
        self.assertEqual(float(read_dataset(run)[0]["bw_util_gbs"]), 47.0)


class TestLayerCountContract(RunDirCase):
    """Defect 1, the one that silently emptied the dataset. Regression-locked here."""

    def test_missing_layer_count_fails_loudly(self):
        """A run directory recorded before the n_layers column existed must NOT yield a
        clean exit and a CSV with a blank c_per_layer_ms. That is the exact shape of the
        original failure: complete-looking output, nothing Rule A can use."""
        cell = fx.cell()
        del cell["n_layers"]
        run = self.build([cell])
        proc = run_cli(run)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("c_per_layer_ms", proc.stderr)
        self.assertEqual(read_dataset(run)[0]["c_per_layer_ms"], "")

    def test_zero_layer_count_also_fails(self):
        run = self.build([fx.cell(n_layers=0)])
        self.assertEqual(run_cli(run).returncode, 1)

    def test_layer_count_never_comes_from_llama_bench_json(self):
        """llama-bench emits no total layer count -- only n_gpu_layers, the offload count,
        which is 0 on this CPU-only node. If the parser ever starts trusting the JSON
        again, this catches it.

        It did not used to. The fixture's entries carry NO `n_layer` key at all, so
        `first(entries, "n_layer", "model_n_layer", default=0)` returned 0 and
        `cells_value or 0` and `0 or cells_value` are the same answer -- the test passed
        with the two operands swapped, which is exactly the defect it names. A guard that
        cannot fail is worse than no guard, because it is cited as coverage. The JSON must
        carry a CONFLICTING value for the precedence to be observable."""
        run = self.build([fx.cell(n_layers=32, entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5, n_layer=99),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35, n_layer=99),
        ])])
        run_cli(run)
        self.assertEqual(read_dataset(run)[0]["n_layers"], "32",
                         "cells.csv (32) must win over the llama-bench JSON (99)")

    def test_one_bad_row_fails_the_whole_run(self):
        """Partial credit is the wrong behaviour: a dataset that is 17/18 complete gets
        used as if it were whole."""
        run = self.build([fx.cell("c001"), fx.cell("c002", n_layers=0)])
        proc = run_cli(run)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("1 of 2", proc.stderr)


class TestLegSeparation(RunDirCase):
    """prefill and decode are different physical regimes -- one compute-bound, one
    bandwidth-bound. Mixing them makes both columns meaningless."""

    def test_pp_and_tg_are_not_averaged(self):
        run = self.build([fx.cell(prefill_ts=40.0, decode_ts=2.0)])
        run_cli(run)
        r = read_dataset(run)[0]
        self.assertEqual(r["prefill_tok_s"], "40.0")
        self.assertEqual(r["decode_tok_s"], "2.0")

    def test_pick_selects_by_leg(self):
        entries = [fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5),
                   fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35)]
        self.assertEqual(len(pb.pick(entries, "pp")), 1)
        self.assertEqual(pb.pick(entries, "pp")[0]["n_prompt"], 256)
        self.assertEqual(pb.pick(entries, "tg")[0]["n_gen"], 128)

    def test_pick_falls_back_only_for_a_genuinely_mixed_entry(self):
        """A combined pp+tg entry (both counts non-zero) matches neither filter, and is the
        best available answer for either question, so it is returned for both."""
        entries = [fx.entry(n_prompt=256, n_gen=128, avg_ts=5.0)]
        self.assertEqual(pb.pick(entries, "pp"), entries)
        self.assertEqual(pb.pick(entries, "tg"), entries)

    def test_pick_does_not_substitute_the_other_leg(self):
        """The fallback used to be `out or entries`, so asking for the decode leg of a
        prefill-only cell returned the PREFILL entry. decode_tok_s then came out at the
        prefill rate -- ~11.5 tok/s instead of ~2.35 -- making c_per_layer_ms about 5x too
        small. Rule A allocates layers proportional to 1/c_i, so that silently hands this
        node far more of the model than it can serve."""
        prefill_only = [fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5)]
        self.assertEqual(pb.pick(prefill_only, "pp"), prefill_only)
        self.assertEqual(pb.pick(prefill_only, "tg"), [])

        decode_only = [fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35)]
        self.assertEqual(pb.pick(decode_only, "tg"), decode_only)
        self.assertEqual(pb.pick(decode_only, "pp"), [])

    def test_a_missing_decode_leg_is_fatal_not_silent(self):
        """End-to-end: the column comes out empty rather than wrong, and main()'s existing
        fail-closed check turns that into a non-zero exit."""
        run = self.build([fx.cell(entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5),
        ])])
        proc = run_cli(run)
        row = read_dataset(run)[0]
        self.assertEqual(row["decode_tok_s"], "")
        self.assertEqual(row["c_per_layer_ms"], "")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("c_per_layer_ms", proc.stderr)

    def test_a_missing_prefill_leg_leaves_prefill_empty(self):
        run = self.build([fx.cell(entries=[
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35),
        ])])
        run_cli(run)
        row = read_dataset(run)[0]
        self.assertEqual(row["prefill_tok_s"], "")
        self.assertEqual(row["ttft_est_ms"], "")
        # the decode side is still valid, so the row is still usable by Rule A
        self.assertNotEqual(row["c_per_layer_ms"], "")


class TestThermalFlag(RunDirCase):
    """Risk R5: a throttled cell is a different machine, not a slower one. It must be
    flagged, never silently averaged in."""

    def test_at_threshold_is_throttled(self):
        run = self.build([fx.cell(temp_end_c="80.0")])
        run_cli(run)
        self.assertEqual(read_dataset(run)[0]["throttled"], "1")

    def test_just_below_threshold_is_not(self):
        run = self.build([fx.cell(temp_end_c="79.9")])
        run_cli(run)
        self.assertEqual(read_dataset(run)[0]["throttled"], "0")

    def test_throttled_cells_are_reported_on_stderr(self):
        run = self.build([fx.cell("hot", temp_end_c="81.2")])
        proc = run_cli(run)
        self.assertIn("throttled", proc.stderr)
        self.assertIn("hot", proc.stderr)

    def test_blank_temperature_is_unknown_not_cool(self):
        """A cell with no usable temperature must not report throttled=0. A dead sensor
        (common.sh falls back to `echo 0`) otherwise publishes a dataset asserting that no
        cell was throttled, while the R5 thermal gate was disarmed for the whole sweep."""
        run = self.build([fx.cell(temp_end_c="")])
        proc = run_cli(run)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        row = read_dataset(run)[0]
        self.assertEqual(row["throttled"], "")
        self.assertEqual(row["temp_end_c"], "")
        self.assertIn("thermal validity is unknown", proc.stderr)

    def test_dead_sensor_zero_is_unknown_too(self):
        """0.0 C is not a cold board -- it is the `echo 0` fallback."""
        run = self.build([fx.cell(temp_end_c="0")])
        run_cli(run)
        row = read_dataset(run)[0]
        self.assertEqual(row["temp_end_c"], "")
        self.assertEqual(row["throttled"], "")


class TestDamagedInput(RunDirCase):
    """The sweep runs unattended for hours. Individual cells can die; the fold must
    survive them and say so, not abort or invent values."""

    def test_missing_json_skips_the_row(self):
        cell = fx.cell("gone")
        cell["entries"] = None
        run = self.build([cell, fx.cell("ok")])
        proc = run_cli(run)
        self.assertIn("missing", proc.stderr)
        self.assertEqual([r["cell_id"] for r in read_dataset(run)], ["ok"])

    def test_unparseable_json_skips_the_row(self):
        run = self.build([fx.cell("bad"), fx.cell("ok")])
        (run / "bad.json").write_text("{ not json", encoding="utf-8")
        proc = run_cli(run)
        self.assertIn("unparseable", proc.stderr)
        self.assertEqual([r["cell_id"] for r in read_dataset(run)], ["ok"])

    def test_empty_entry_list_skips_the_row(self):
        run = self.build([fx.cell("empty", entries=[]), fx.cell("ok")])
        run_cli(run)
        self.assertEqual([r["cell_id"] for r in read_dataset(run)], ["ok"])

    def test_no_cells_csv_is_exit_2(self):
        (self.tmp / "bare").mkdir()
        self.assertEqual(run_cli(self.tmp / "bare").returncode, 2)

    def test_nonzero_exit_code_is_reported(self):
        run = self.build([fx.cell(exit_code=1)])
        self.assertIn("non-zero", run_cli(run).stderr)

    def test_missing_manifest_leaves_sha_blank(self):
        """No manifest is a provenance gap, not a crash -- but the SHA must come out
        empty rather than being guessed."""
        run = self.build([fx.cell()], manifest=None)
        proc = run_cli(run)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(read_dataset(run)[0]["llama_cpp_sha"], "")

    def test_corrupt_manifest_leaves_sha_blank(self):
        run = self.build([fx.cell()])
        (run / "build_manifest.json").write_text("{oops", encoding="utf-8")
        self.assertEqual(read_dataset(run)[0]["llama_cpp_sha"] if
                         run_cli(run).returncode == 0 else None, "")


class TestNoSilentCellLoss(RunDirCase):
    """A cell that produces no row must not vanish with a clean exit. The sweep runs
    unattended for hours; "0 cells, rc=0, empty stderr" was a reachable outcome when every
    raw JSON was an empty list, which is what a llama-bench that dies before finishing a
    test emits (and what --dry-run writes)."""

    def test_missing_json_is_counted_and_fatal(self):
        cell = fx.cell("gone")
        cell["entries"] = None
        run = self.build([cell, fx.cell("ok")])
        proc = run_cli(run)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("gone", proc.stderr)
        self.assertIn("absent from both CSVs", proc.stderr)

    def test_empty_entry_list_says_so(self):
        run = self.build([fx.cell("empty", entries=[])])
        proc = run_cli(run)
        self.assertIn("no llama-bench results", proc.stderr)
        self.assertEqual(proc.returncode, 1)

    def test_a_wholly_failed_run_is_not_silent(self):
        run = self.build([fx.cell("c1", entries=[]), fx.cell("c2", entries=[])])
        proc = run_cli(run)
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue(proc.stderr.strip(), "a run that produced nothing said nothing")

    def test_a_clean_run_is_still_exit_0(self):
        run = self.build([fx.cell("c1"), fx.cell("c2")])
        self.assertEqual(run_cli(run).returncode, 0)


class TestSanityChecks(RunDirCase):
    def test_bandwidth_above_the_soc_peak_is_reported(self):
        """The docstring promised this check long before it existed. Above ~17 GB/s the
        weights were not being read from DRAM, so the cell is void."""
        run = self.build([fx.cell(decode_ts=10.0)])   # 4.7 GB x 10/s = 47 GB/s
        proc = run_cli(run)
        self.assertIn("memory bandwidth", proc.stderr)
        self.assertIn("47.0", proc.stderr)

    def test_a_plausible_rung_is_not_reported(self):
        run = self.build([fx.cell(decode_ts=2.35)])   # 11.045 GB/s
        self.assertNotIn("memory bandwidth", run_cli(run).stderr)

    def test_blank_exit_code_reads_as_clean_in_both_files(self):
        """results.csv treats a blank rc as 'not recorded, treat as clean'. dataset.csv used
        to call the same cell a non-zero exit, so the two files disagreed about one cell."""
        run = self.build([fx.cell(exit_code="")])
        proc = run_cli(run)
        self.assertNotIn("non-zero", proc.stderr)

    def test_node_id_comes_from_the_snapshot_in_dataset_too(self):
        """build_manifest.json is copied with `[ -f ] && cp`, which does not abort under
        set -e; node.env.snapshot is copied unconditionally."""
        run = self.build([fx.cell()], manifest=None)
        run_cli(run)
        self.assertEqual(read_dataset(run)[0]["node_id"], "pi5")


class TestDerivedColumnsVaryWithInputs(RunDirCase):
    """`test_derived_columns` asserts every derived value at exactly ONE operating point --
    n_layers=32, decode 2.35, prefill 11.5, n_prompt 256, model_size 4.7e9. A single point
    cannot distinguish a formula from a constant, so `c_layer = tpot_ms / 32` and
    `bw = 4.7e9 * dc / 1e9` both shipped green.

    That matters because the sweep's real models are NOT 32 layers: the consumer round-trip
    maps qwen2.5-7b-q4km to 28 and tinyllama-1.1b-q8 to 22, so every published row would
    carry a c_i off by 1.14x or 1.45x -- and c_i is what Rule A divides layers by.

    `tests/test_pi4_schema.py::TestRuleAsInputIsAsserted` already does this correctly for
    results.csv; dataset.csv never got the same treatment. Each case below varies every
    input together so no single constant can satisfy them all."""

    CASES = [
        # n_layers, decode_ts, prefill_ts, n_prompt, model_size
        (28, 2.35, 11.5, 256, 4_700_000_000),
        (22, 6.10, 34.2, 512, 1_100_000_000),
        (32, 2.35, 11.5, 256, 4_700_000_000),   # the worked example, kept as one case
        (24, 3.70, 19.8, 128, 2_300_000_000),
    ]

    def _row(self, n_layers, decode, prefill, n_prompt, size):
        run = self.build([fx.cell(n_layers=n_layers, n_prompt=n_prompt, entries=[
            fx.entry(n_prompt=n_prompt, n_gen=0, avg_ts=prefill, model_size=size),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=decode, model_size=size),
        ])])
        run_cli(run)
        return read_dataset(run)[0]

    def test_c_per_layer_ms_is_a_quotient_not_a_constant(self):
        """The mutation that survived: `tpot_ms / 32`. Also kills `tpot_ms / n_layers`
        replaced by the worked example's literal 13.2979."""
        for n_layers, decode, prefill, n_prompt, size in self.CASES:
            with self.subTest(n_layers=n_layers, decode=decode):
                r = self._row(n_layers, decode, prefill, n_prompt, size)
                tpot = 1000.0 / decode
                self.assertAlmostEqual(float(r["tpot_ms"]), round(tpot, 1), places=1)
                self.assertAlmostEqual(float(r["c_per_layer_ms"]),
                                       round(tpot / n_layers, 4), places=3)

    def test_bw_util_uses_this_cell_s_model_size(self):
        """The mutation that survived: `4.7e9 * dc / 1e9`. bw_util_gbs is the DRAM-residency
        validity gate -- judged against the wrong model size, a void cell passes."""
        for n_layers, decode, prefill, n_prompt, size in self.CASES:
            with self.subTest(size=size, decode=decode):
                r = self._row(n_layers, decode, prefill, n_prompt, size)
                self.assertAlmostEqual(float(r["bw_util_gbs"]),
                                       round(size * decode / 1e9, 3), places=2)
                self.assertEqual(r["model_size_bytes"], str(size))

    def test_ttft_tracks_the_prompt_length(self):
        """The TTFT-vs-prompt-length relationship is what the prefill sweep exists to
        measure; a constant here flattens the curve to a line."""
        for n_layers, decode, prefill, n_prompt, size in self.CASES:
            with self.subTest(n_prompt=n_prompt, prefill=prefill):
                r = self._row(n_layers, decode, prefill, n_prompt, size)
                self.assertAlmostEqual(float(r["ttft_est_ms"]),
                                       round(1000.0 * n_prompt / prefill, 1), places=1)
                self.assertAlmostEqual(float(r["prefill_tok_s"]), prefill, places=3)

    def test_the_cases_would_catch_a_constant(self):
        """Guard on the guard: if every case shared a value, the loops above would pass
        against a hardcoded implementation. Assert the inputs really do differ."""
        for col in range(5):
            values = {c[col] for c in self.CASES}
            self.assertGreater(len(values), 1, "column %d is constant across all cases" % col)


class TestDatasetIdentityColumns(RunDirCase):
    """dataset.csv's whole identity/factor half was unasserted: no test read the VALUE of
    threads, rep, n_gen, model, exit_code, started_utc, ended_utc, model_size_bytes,
    model_n_params or temp_start_c. Feeding `threads` from `n_gen` survived -- and threads
    is one of the sweep's two independent variables, so that silently re-labels every point
    of the thread-scaling curve. Distinct values on every axis is the point: today's
    256/128/4/1 lets several of these be swapped undetected."""

    def test_the_factor_columns_are_not_interchangeable(self):
        run = self.build([fx.cell(n_prompt=512, n_gen=64, threads=2, rep=3)])
        run_cli(run)
        r = read_dataset(run)[0]
        self.assertEqual(r["n_prompt"], "512")
        self.assertEqual(r["n_gen"], "64")
        self.assertEqual(r["threads"], "2")
        self.assertEqual(r["rep"], "3")
        self.assertEqual(r["model"], "qwen2.5-7b-q4km")
        self.assertEqual(r["exit_code"], "0")

    def test_the_model_size_and_parameter_count_are_not_swapped(self):
        """Both are ~10^9 and plausible in each other's place, and bytes-per-parameter is
        how a reader checks the quantisation claim."""
        run = self.build([fx.cell()])
        run_cli(run)
        r = read_dataset(run)[0]
        self.assertEqual(r["model_size_bytes"], "4700000000")
        self.assertEqual(r["model_n_params"], "7615616512")

    def test_the_cell_window_runs_forwards(self):
        run = self.build([fx.cell()])
        run_cli(run)
        r = read_dataset(run)[0]
        self.assertLess(pb.iso_to_epoch(r["started_utc"]), pb.iso_to_epoch(r["ended_utc"]),
                        "the cell's timestamps run backwards")

    def test_both_output_files_agree_about_the_same_cell(self):
        """decode_tok_s (fmean of samples_ts) and the tg row's tokens_per_s (avg_ts) are
        computed by different code paths and both are published as this node's decode
        throughput. Nothing compared them."""
        run = self.build([fx.cell()])
        run_cli(run)
        ds = read_dataset(run)[0]
        with (run / "results.csv").open(encoding="utf-8", newline="") as fh:
            tg = {r["phase"]: r for r in csv.DictReader(fh)}["tg"]
        self.assertAlmostEqual(float(ds["decode_tok_s"]), float(tg["tokens_per_s"]), places=3)
        self.assertAlmostEqual(float(ds["c_per_layer_ms"]), float(tg["per_layer_ms"]), places=3)
        self.assertEqual(ds["node_id"], tg["node_id"])
        self.assertEqual(ds["model"], tg["model"])


class TestOutputShape(RunDirCase):
    def test_header_matches_declared_fields(self):
        run = self.build([fx.cell()])
        run_cli(run)
        with (run / "dataset.csv").open(encoding="utf-8", newline="") as fh:
            header = next(csv.reader(fh))
        self.assertEqual(header, pb.OUT_FIELDS)

    def test_run_id_is_the_directory_name(self):
        """Provenance travels with every number: the run directory names the sweep."""
        run = self.build([fx.cell()], name="run-20260913-120000")
        run_cli(run)
        self.assertEqual(read_dataset(run)[0]["run_id"], "run-20260913-120000")

    def test_out_flag_redirects(self):
        run = self.build([fx.cell()])
        dest = self.tmp / "elsewhere.csv"
        subprocess.run([sys.executable, str(PARSE_BENCH), str(run), "-o", str(dest)],
                       capture_output=True, text=True)
        self.assertTrue(dest.is_file())
        # Existence is not redirection: an -o that creates an empty or header-only file
        # passes `is_file()` while the rows go somewhere else, or nowhere.
        with dest.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 1, "-o produced a file with no data rows")
        self.assertEqual(rows[0]["n_layers"], "32")
        self.assertEqual(rows[0]["decode_tok_s"], "2.35")


class TestRepetitionSpread(RunDirCase):
    """llama-bench is run with -r > 1 so the sweep has run-to-run variance to report.
    That spread is the error bar on every published figure, so it has to survive the
    fold rather than being flattened to zero."""

    def test_stddev_reflects_the_repetitions(self):
        samples = [2.30, 2.35, 2.40]
        run = self.build([fx.cell(entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35, samples_ts=samples),
        ])])
        run_cli(run)
        r = read_dataset(run)[0]
        self.assertEqual(float(r["decode_tok_s"]), 2.35)
        # The VALUE, not merely non-zero. "greater than 0" is satisfied by a population
        # stddev, a scaled one, or a variance -- and the spread is the error bar on every
        # published figure, so the wrong statistic misstates the confidence, not the number.
        self.assertAlmostEqual(float(r["decode_tok_s_stddev"]), 0.05, places=4,
                               msg="decode spread is not the sample stddev (n-1) of the "
                                   "repetitions -- llama-bench's own `stdev()` helper")

    def test_the_prefill_spread_is_its_own(self):
        """prefill_tok_s_stddev was read by no test in either suite -- only the decode side
        was checked -- so it could be blanked, or fed from the decode leg, undetected."""
        import statistics
        pp = [11.2, 11.5, 11.8, 11.35, 11.65]
        tg = [2.30, 2.35, 2.40, 2.33, 2.37]
        run = self.build([fx.cell(entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5, samples_ts=pp),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35, samples_ts=tg),
        ])])
        run_cli(run)
        r = read_dataset(run)[0]
        self.assertAlmostEqual(float(r["prefill_tok_s_stddev"]),
                               round(statistics.stdev(pp), 3), places=3)
        self.assertAlmostEqual(float(r["decode_tok_s_stddev"]),
                               round(statistics.stdev(tg), 3), places=3)
        self.assertNotAlmostEqual(float(r["prefill_tok_s_stddev"]),
                                  float(r["decode_tok_s_stddev"]), places=3)

    def test_single_repetition_has_unknown_spread(self):
        """Empty, not 0.0. One sample means the spread was never measured; 0.0 is the
        stronger claim that it was measured and found to be zero -- which on a passively
        cooled board is not credible and is exactly what this column exists to expose."""
        run = self.build([fx.cell()])
        run_cli(run)
        self.assertEqual(read_dataset(run)[0]["decode_tok_s_stddev"], "")


class TestConstantsAndFallbacksAreLoadBearing(RunDirCase):
    """Four values that a mutation could change while all 292 tests stayed green. Each is
    a number or a fallback that nothing downstream can sanity-check, so the only place the
    error can be caught is here."""

    def test_the_layer_count_falls_back_to_the_llama_bench_json(self):
        """`n_layers` normally comes from the cells.csv column 10_isolated_bench.sh fills
        from the GGUF header, but run directories recorded before that column existed have
        it only in the llama-bench JSON. Deleting the fallback leaves n_layers = 0, and
        `c_layer = tpot_ms / n_layers if (tpot_ms and n_layers)` then yields None -- so the
        row loses the one column Rule A consumes, silently, for a whole class of runs."""
        run = self.build([fx.cell(n_layers="", entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5, n_layer=28),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35, n_layer=28),
        ])])
        run_cli(run)
        r = read_dataset(run)[0]
        self.assertEqual(r["n_layers"], "28",
                         "the JSON fallback for the layer count was not used")
        # 1000/2.35 = 425.5319 ms TPOT, over 28 layers
        self.assertEqual(r["c_per_layer_ms"], "15.1976",
                         "c_per_layer_ms is empty or wrong when n_layers came from JSON")

    def test_the_bandwidth_ceiling_is_the_pi5s_own_peak_not_a_rounder_number(self):
        """`PEAK_MEM_BW_GBS` is the Pi 5's LPDDR4X-4267 peak, and it is the threshold that
        marks a cell void -- a rung implying more than it did not read the weights from
        DRAM. The existing coverage uses 47 GB/s, which is above any plausible ceiling and
        so cannot tell 17 from 40. 23.5 GB/s sits between them: void on the real constant,
        invisible on an inflated one."""
        run = self.build([fx.cell(decode_ts=5.0)])      # 4.7 GB x 5 tok/s = 23.5 GB/s
        proc = run_cli(run)
        self.assertEqual(read_dataset(run)[0]["bw_util_gbs"], "23.5")
        self.assertIn("cannot", proc.stderr,
                      "23.5 GB/s is above the Pi 5's 17 GB/s peak and was not reported void")
        self.assertIn("17.0 GB/s", proc.stderr,
                      "the void warning did not name the Pi 5's actual peak bandwidth")

    def test_node_env_may_override_the_throttle_temperature(self):
        """`throttled` is a published column and a placement opt-out. The threshold is
        `THERMAL_THROTTLE_C` from the run's own node.env.snapshot, falling back to 80 C --
        a node whose snapshot says 70 must be judged at 70. Ignoring the override reads a
        different node's thermal policy into this node's rows."""
        run = fx.write_run(self.tmp / "override", [fx.cell(temp_end_c="75.0")],
                           MANIFEST, {"NODE_ID": "pi5", "THERMAL_THROTTLE_C": "70"})
        run_cli(run)
        self.assertEqual(read_dataset(run)[0]["throttled"], "1",
                         "75.0 C against a node.env threshold of 70 C was not flagged; "
                         "the snapshot's THERMAL_THROTTLE_C was ignored")

    def test_the_start_temperature_is_the_start_not_the_end(self):
        """The pair exists to show thermal history across the cell. Sourcing both ends from
        `temp_end_c` collapses the delta to zero and makes every cell look like it ran at a
        flat temperature -- which is precisely the claim the column is there to support."""
        run = self.build([fx.cell(temp_start_c="58.1", temp_end_c="81.2")])
        run_cli(run)
        r = read_dataset(run)[0]
        self.assertEqual(r["temp_start_c"], "58.1")
        self.assertEqual(r["temp_end_c"], "81.2")
        self.assertNotEqual(r["temp_start_c"], r["temp_end_c"],
                            "both temperature columns came from the same end of the cell")


class TestThroughputIsTheMeanOfTheSamples(RunDirCase):
    """`tok_s` returns (centre, spread) over every sample in the cell. The centre must be
    the arithmetic MEAN: it is paired with a sample stddev taken about the mean, and it is
    what llama-bench's own `avg_ts` reports, so the columns are only comparable across
    nodes if both are means. A median is a defensible statistic and an indefensible one to
    publish beside a stddev -- and on a throttling node the two diverge exactly when it
    matters, because thermal slowdown skews the tail."""

    def test_the_median_is_not_substituted_for_the_mean(self):
        run = self.build([fx.cell(entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35, samples_ts=[2.0, 2.1, 5.0]),
        ])])
        run_cli(run)
        r = read_dataset(run)[0]
        # mean = 3.0333..., median = 2.1
        self.assertEqual(r["decode_tok_s"], "3.033",
                         "decode_tok_s is not the mean of the samples (median is 2.1)")


class TestNodeEnvSnapshotIsParsedAsAShellFile(RunDirCase):
    """`node.env.snapshot` is a verbatim copy of `config/node.env`, which is SOURCED by the
    shell scripts. Two of its spellings had zero coverage because the fixture writes only
    bare `k=v`: quoted values (node.env really does use them -- NODE_DESC, LLAMA_ARCH_FLAGS,
    PSU_MODEL) and the `export ` prefix. Both branches feed NODE_ID, which is the join key
    for every row this node publishes."""

    def _run_with_snapshot(self, text):
        run = self.build([fx.cell()])
        (run / "node.env.snapshot").write_text(text, encoding="utf-8")
        run_cli(run)
        return read_dataset(run)[0]

    # NODE_ID falls back to build_manifest.json, which also says "pi5" -- so a snapshot
    # that FAILS to parse still yields node_id == "pi5" and the assertion proves nothing.
    # Every value below is deliberately one the manifest cannot supply.
    def test_quoted_values_are_unquoted(self):
        r = self._run_with_snapshot('NODE_ID="pi5-alt"\n')
        self.assertEqual(r["node_id"], "pi5-alt",
                         "a quoted value kept its quotes, or failed to parse and fell "
                         "back to the manifest")

    def test_single_quoted_values_are_unquoted(self):
        r = self._run_with_snapshot("NODE_ID='pi5-alt'\n")
        self.assertEqual(r["node_id"], "pi5-alt")

    def test_the_export_prefix_is_stripped(self):
        r = self._run_with_snapshot("export NODE_ID=pi5-alt\n")
        self.assertEqual(r["node_id"], "pi5-alt",
                         "`export NODE_ID=` produced the key 'export NODE_ID', so NODE_ID "
                         "was missing and the manifest's value was used instead")

    def test_both_spellings_together(self):
        r = self._run_with_snapshot('export NODE_ID="pi5-alt"\n')
        self.assertEqual(r["node_id"], "pi5-alt")


class TestPublishedPrecisionAndProvenance(RunDirCase):
    def test_decode_throughput_keeps_three_decimals(self):
        """Every throughput the other tests use is expressible in two decimals, so the
        declared 3-dp precision was pinned by nothing. Real llama-bench output is not so
        tidy, and this column is the input to tpot_ms -> c_per_layer_ms."""
        run = self.build([fx.cell(decode_ts=2.3456, prefill_ts=11.5)])
        run_cli(run)
        self.assertEqual(read_dataset(run)[0]["decode_tok_s"], "2.346")

    def test_the_backend_is_read_from_the_run_not_assumed_to_be_cpu(self):
        """HeteroServe is a HETEROGENEOUS cluster -- the Jetson row is the whole point of
        recording this column. The existing assertion uses "CPU", which is also what a
        hardcoded constant would produce, so the column could be a literal and the suite
        would agree."""
        run = self.build([fx.cell(entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5, backends="CUDA"),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35, backends="CUDA"),
        ])])
        run_cli(run)
        self.assertEqual(read_dataset(run)[0]["backend"], "CUDA",
                         "the backend column ignored the run and reported CPU")

    def test_backends_plural_wins_over_the_spelling_that_never_existed(self):
        """The code keeps `backend_name` only so a hand-written JSON carrying it still
        parses -- upstream has always spelled it `backends`. That documented precedence is
        observable only when an entry carries BOTH, which no real llama-bench output does,
        so the key order was pinned by nothing."""
        run = self.build([fx.cell(entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5,
                     backends="CUDA", backend_name="CPU"),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35,
                     backends="CUDA", backend_name="CPU"),
        ])])
        run_cli(run)
        self.assertEqual(read_dataset(run)[0]["backend"], "CUDA",
                         "the legacy `backend_name` spelling took precedence over the "
                         "upstream `backends`")

    def test_model_metadata_comes_from_the_first_entry(self):
        """`first()` is a documented precedence, not an accident: when the legs disagree
        the earlier one wins. Both legs of a real run agree, so only a disagreeing pair can
        show which end is being read."""
        run = self.build([fx.cell(entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5, model_type="Q4_K - Medium"),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35, model_type="Q8_0"),
        ])])
        run_cli(run)
        self.assertEqual(read_dataset(run)[0]["model_type"], "Q4_K - Medium")


class TestTheBandwidthGateBoundaryAndOverride(RunDirCase):
    def test_node_env_may_override_the_peak_bandwidth(self):
        """Partner to the throttle override. PEAK_MEM_BW_GBS is NOT in config/node.env, so
        this branch is only ever exercised with the key absent -- the override path itself
        was never executed. A node with different memory needs its own ceiling or the
        validity gate is judging it against the Pi 5's DRAM."""
        run = fx.write_run(self.tmp / "peak", [fx.cell(decode_ts=5.0)], MANIFEST,
                           {"NODE_ID": "pi5", "PEAK_MEM_BW_GBS": "50"})
        proc = run_cli(run)
        self.assertEqual(read_dataset(run)[0]["bw_util_gbs"], "23.5")
        self.assertNotIn("cannot", proc.stderr,
                         "23.5 GB/s was called void against a node.env ceiling of 50 GB/s")

    def test_exactly_the_peak_is_not_void(self):
        """The gate's own comment says "more than the SoC's peak". A cell implying exactly
        the ceiling is at the limit, not above it, and the boundary was untested in either
        direction."""
        # 3.4 tok/s x 5.0 GB = 17.0 GB/s exactly
        run = self.build([fx.cell(decode_ts=3.4, entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5, model_size=5_000_000_000),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=3.4, model_size=5_000_000_000),
        ])])
        proc = run_cli(run)
        self.assertEqual(read_dataset(run)[0]["bw_util_gbs"], "17.0")
        self.assertNotIn("cannot", proc.stderr,
                         "a cell at exactly the peak was reported void")


if __name__ == "__main__":
    unittest.main(verbosity=2)
