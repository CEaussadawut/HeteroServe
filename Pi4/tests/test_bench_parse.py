"""Tests for node/bench_parse.py -- the code every dataset row passes through.

This node's contribution to the device-characterisation dataset is exactly what this
module writes. A defect here does not crash anything: it produces a well-formed CSV with
a wrong number in it, which is the failure mode the whole project is least able to
detect. Hence the emphasis below on the *silent* paths -- the energy-join key, the quant
label, and the validity flags.

    python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "node"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_parse  # noqa: E402
import fixtures  # noqa: E402


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class TestQuantDetection(unittest.TestCase):
    """The quant label is a primary axis of the study: Q4_K_M vs Q8_0 is how the
    compute/bandwidth trade-off is measured. Mislabelling one silently merges two
    conditions."""

    def test_every_model_in_models_txt_is_labelled_correctly(self):
        for name, model_type, expected in fixtures.MODELS:
            with self.subTest(model=name):
                self.assertEqual(bench_parse.quant_of(name, model_type), expected)

    def test_label_survives_with_no_model_type(self):
        # llama-bench could omit or rename model_type; the filename alone must still work,
        # because the filename is ours and the JSON field is upstream's.
        for name, _model_type, expected in fixtures.MODELS:
            with self.subTest(model=name):
                self.assertEqual(bench_parse.quant_of(name, ""), expected)

    def test_label_survives_with_no_filename_hint(self):
        self.assertEqual(bench_parse.quant_of("", "llama 1B Q4_K - Medium"), "Q4_K_M")
        self.assertEqual(bench_parse.quant_of("", "llama 1B Q8_0"), "Q8_0")

    def test_q8_hint_does_not_swallow_q4_k_m(self):
        # QUANT_HINTS is ordered and matched by substring; the loose "Q8" spelling must
        # not win against a Q4_K_M model whose name happens to contain an 8.
        self.assertEqual(
            bench_parse.quant_of("qwen2.5-8b-q4km", "qwen2 8B Q4_K - Medium"), "Q4_K_M"
        )

    def test_unknown_quant_is_reported_not_guessed(self):
        self.assertEqual(bench_parse.quant_of("mystery-model", "llama 1B TQ2_0"),
                         "llama 1B TQ2_0")


class TestEnergyJoin(unittest.TestCase):
    """`windows.jsonl` is written by scripts/06_bench_local.sh and read here. The two
    sides derive the join key independently -- the sweep from shell variables, the parser
    from the filename -- so they can drift apart without any error being raised. A miss
    means the control node can never attach energy to that row, and joules per token is
    the project's primary metric."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = fixtures.write_run(Path(self.tmp.name) / "20260912T090000Z")

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_row_joins_to_a_window(self):
        windows = bench_parse.load_windows(self.run_dir)
        rows = []
        for p in sorted(self.run_dir.glob("*.json")):
            if p.name == "build_manifest.json":
                continue
            rows.extend(bench_parse.rows_from(p, windows, self.run_dir.name))

        self.assertTrue(rows, "fixture produced no rows at all")
        unjoined = [r for r in rows if "no-window" in r["flags"]]
        self.assertEqual(
            unjoined, [], "%d of %d rows failed the energy join" % (len(unjoined), len(rows))
        )

    def test_join_key_holds_for_model_names_containing_dots_and_hyphens(self):
        # "llama3.2-3b-q4km" and "qwen2.5-1.5b-q4km" are the awkward ones: the parser
        # recovers the model name with rsplit("_", 2), so a dot or hyphen must not matter
        # -- but an underscore in a model name would.
        windows = bench_parse.load_windows(self.run_dir)
        for name, _mt, _q in fixtures.MODELS:
            path = self.run_dir / ("%s_pp128_t4.json" % name)
            self.assertTrue(path.exists(), "fixture missing %s" % path.name)
            rows = bench_parse.rows_from(path, windows, self.run_dir.name)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model"], name)
            self.assertNotIn("no-window", rows[0]["flags"])

    def test_underscores_in_a_model_name_still_join(self):
        # rsplit("_", 2) peels from the right, so it strips exactly "_<phase><n>" and
        # "_t<threads>" however many underscores the model name itself carries. Asserted
        # explicitly because the obvious reading of that line is that it splits from the
        # left, and "re-deriving the model name from the filename" is the kind of thing a
        # later edit quietly gets wrong.
        tmp = Path(self.tmp.name) / "underscore"
        fixtures.write_run(
            tmp,
            models=[("odd_name_q4_k_m", "llama 1B Q4_K - Medium", "Q4_K_M")],
            threads=(4,),
        )
        windows = bench_parse.load_windows(tmp)
        rows = bench_parse.rows_from(tmp / "odd_name_q4_k_m_pp128_t4.json", windows, tmp.name)
        self.assertEqual(rows[0]["model"], "odd_name_q4_k_m")
        self.assertNotIn("no-window", rows[0]["flags"])

    def test_missing_window_is_flagged_never_silently_blank(self):
        tmp = Path(self.tmp.name) / "nowin"
        fixtures.write_run(
            tmp, threads=(4,), models=fixtures.MODELS[:1],
            skip_windows_for=("tinyllama-1.1b-q4km_pp128_t4",),
        )
        windows = bench_parse.load_windows(tmp)
        rows = bench_parse.rows_from(
            tmp / "tinyllama-1.1b-q4km_pp128_t4.json", windows, tmp.name
        )
        self.assertIn("no-window:energy-join-impossible", rows[0]["flags"])
        self.assertEqual(rows[0]["t_start"], "")
        self.assertEqual(rows[0]["t_end"], "")

    def test_window_timestamps_are_carried_through_unrounded(self):
        # The collector slices a ~28 Hz power stream on these; losing sub-second precision
        # would smear a prefill window across a phase boundary.
        windows = bench_parse.load_windows(self.run_dir)
        rows = bench_parse.rows_from(
            self.run_dir / "tinyllama-1.1b-q4km_tg128_t4.json", windows, self.run_dir.name
        )
        t_start = float(rows[0]["t_start"])
        t_end = float(rows[0]["t_end"])
        self.assertGreater(t_end, t_start)
        self.assertNotEqual(t_start, int(t_start), "sub-second precision was lost")

    def test_windows_for_a_missing_file_do_not_crash_the_load(self):
        with (self.run_dir / "windows.jsonl").open("a", encoding="utf-8") as fh:
            fh.write("not json at all\n\n")
        windows = bench_parse.load_windows(self.run_dir)
        self.assertTrue(windows, "one bad line discarded every good window")


class TestValidityFlags(unittest.TestCase):
    """A throttled run measures the heatsink and the PSU, not the SoC. The row must
    survive -- discarding it hides a real thermal result -- but it must never reach the
    dataset unmarked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "20260912T090000Z"
        fixtures.write_run(
            self.run_dir, models=fixtures.MODELS[:1], threads=(4,),
            throttled_on=("tinyllama-1.1b-q4km_tg128_t4",),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_throttled_row_is_flagged_with_the_word(self):
        windows = bench_parse.load_windows(self.run_dir)
        rows = bench_parse.rows_from(
            self.run_dir / "tinyllama-1.1b-q4km_tg128_t4.json", windows, self.run_dir.name
        )
        self.assertIn("throttled:0x50005", rows[0]["flags"])

    def test_clean_row_carries_no_flags(self):
        windows = bench_parse.load_windows(self.run_dir)
        rows = bench_parse.rows_from(
            self.run_dir / "tinyllama-1.1b-q4km_pp128_t4.json", windows, self.run_dir.name
        )
        self.assertEqual(rows[0]["flags"], "")

    def test_zero_throughput_is_flagged_and_leaves_ms_per_token_blank(self):
        entry = fixtures.bench_entry("llama 1B Q4_K - Medium", n_prompt=0, n_gen=128,
                                     n_threads=4, avg_ts=0.0)
        path = self.run_dir / "tinyllama-1.1b-q4km_tg999_t4.json"
        path.write_text(json.dumps([entry]), encoding="utf-8")
        rows = bench_parse.rows_from(path, bench_parse.load_windows(self.run_dir),
                                     self.run_dir.name)
        self.assertIn("no-throughput", rows[0]["flags"])
        self.assertEqual(rows[0]["ms_per_token"], "",
                         "a divide-by-zero must not become a fabricated latency")

    def test_a_result_without_samples_is_flagged_because_reps_becomes_one(self):
        """ThinkPad/control/analysis.py turns the `samples` column into the repetition
        count with `reps = int(row.get("samples") or 1) or 1`. An empty samples_ns
        therefore becomes reps=1 and joules_per_token is overstated by the true repetition
        count -- 5x on a full sweep. Nothing downstream can notice, so it is flagged here."""
        entry = fixtures.bench_entry("llama 1B Q4_K - Medium", n_prompt=128, n_gen=0,
                                     n_threads=4, avg_ts=9.5)
        entry["samples_ns"] = []
        path = self.run_dir / "tinyllama-1.1b-q4km_pp777_t4.json"
        path.write_text(json.dumps([entry]), encoding="utf-8")
        rows = bench_parse.rows_from(path, bench_parse.load_windows(self.run_dir),
                                     self.run_dir.name)
        self.assertIn("no-samples:reps-unknown", rows[0]["flags"])
        self.assertEqual(rows[0]["samples"], 0)

    def test_an_unreadable_temperature_is_flagged_rather_than_read_as_fine(self):
        """scripts/lib.sh soc_temp_c() yields "nan" when the thermal zone cannot be read,
        which bench_windows.py drops to an empty field. A blank temperature must be visible
        as "not measured" -- a dead sensor is not a cold board."""
        wpath = self.run_dir / "windows.jsonl"
        recs = [json.loads(l) for l in wpath.read_text().splitlines() if l.strip()]
        for r in recs:
            if r["phase"] == "pp" and r["n"] == 128:
                r["temp_end_c"] = ""
        wpath.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        rows = bench_parse.rows_from(
            self.run_dir / "tinyllama-1.1b-q4km_pp128_t4.json",
            bench_parse.load_windows(self.run_dir), self.run_dir.name)
        self.assertIn("temp-unknown", rows[0]["flags"])

    def test_one_damaged_window_record_does_not_abort_the_whole_normalisation(self):
        """This runs at the END of a multi-hour sweep, against files whose writer may have
        been OOM-killed mid-line. One bad record must cost its own row, not the run."""
        wpath = self.run_dir / "windows.jsonl"
        good = wpath.read_text()
        wpath.write_text(
            '{"model": "x"}\n'                      # valid JSON, missing required keys
            'not json at all\n'
            '{"model": "y", "phase": "pp", "n": "NaN", "threads": 4}\n'
            + good, encoding="utf-8")
        windows = bench_parse.load_windows(self.run_dir)   # must not raise
        rows = bench_parse.rows_from(
            self.run_dir / "tinyllama-1.1b-q4km_pp128_t4.json", windows, self.run_dir.name)
        self.assertEqual(rows[0]["flags"], "",
                         "the undamaged rows must still join their windows normally")


class TestSchema(unittest.TestCase):
    """Every node in the cluster emits the same columns so the control node can
    concatenate them without per-node special cases."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = fixtures.write_run(
            Path(self.tmp.name) / "20260912T090000Z", models=fixtures.MODELS[:2],
            threads=(1, 4),
        )
        self.out = self.run_dir / "results.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def _run_cli(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "node" / "bench_parse.py"), str(self.run_dir),
             "--out", str(self.out)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return read_csv(self.out)

    def test_cli_writes_the_declared_columns_in_order(self):
        self._run_cli()
        with self.out.open(newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        self.assertEqual(header, bench_parse.COLUMNS)

    def test_energy_columns_are_left_empty_for_the_control_node(self):
        # Energy is never computed on a node: one collector owns all four channels on one
        # clock. A node that filled these in would be estimating on a second time base.
        for row in self._run_cli():
            self.assertEqual(row["energy_j"], "")
            self.assertEqual(row["joules_per_token"], "")
            self.assertEqual(row["mean_power_w"], "")

    def test_identity_columns_come_from_node_env(self):
        env = bench_parse.ENV
        for row in self._run_cli():
            self.assertEqual(row["node_id"], env["NODE_ID"])
            self.assertEqual(row["node_role"], env["NODE_ROLE"])
            self.assertEqual(row["run_id"], self.run_dir.name)

    def test_build_manifest_is_not_parsed_as_a_result(self):
        rows = self._run_cli()
        self.assertTrue(all(r["model"] != "build_manifest" for r in rows))
        # 2 models x 2 thread settings x (1 pp + 1 tg)
        self.assertEqual(len(rows), 8)

    def test_phase_is_inferred_from_the_token_counts_not_the_filename(self):
        rows = self._run_cli()
        for row in rows:
            expected = "pp" if int(row["n_prompt"]) > 0 else "tg"
            self.assertEqual(row["phase"], expected)
        self.assertEqual(sum(1 for r in rows if r["phase"] == "pp"), 4)
        self.assertEqual(sum(1 for r in rows if r["phase"] == "tg"), 4)

    def test_ms_per_token_is_the_reciprocal_of_throughput(self):
        for row in self._run_cli():
            tps = float(row["tokens_per_s"])
            self.assertAlmostEqual(float(row["ms_per_token"]), 1000.0 / tps, places=2)

    def test_per_layer_ms_is_left_for_the_control_node(self):
        # The layer count is not in llama-bench's output; inventing one here would put a
        # fabricated c_i into the placement rule.
        for row in self._run_cli():
            self.assertEqual(row["per_layer_ms"], "")


class TestRobustness(unittest.TestCase):
    """The sweep runs for hours unattended. A half-written file at the end of an
    interrupted run must not take the whole parse down, and must not quietly disappear
    either."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = fixtures.write_run(
            Path(self.tmp.name) / "20260912T090000Z", models=fixtures.MODELS[:1],
            threads=(4,),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_truncated_json_is_skipped_with_a_message(self):
        (self.run_dir / "tinyllama-1.1b-q4km_pp999_t4.json").write_text(
            '[{"n_prompt": 128, "avg_', encoding="utf-8"
        )
        out = self.run_dir / "results.csv"
        proc = subprocess.run(
            [sys.executable, str(ROOT / "node" / "bench_parse.py"), str(self.run_dir),
             "--out", str(out)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("skipping", proc.stderr.lower() + proc.stdout.lower())
        self.assertEqual(len(read_csv(out)), 2)

    def test_a_bare_object_is_accepted_as_well_as_an_array(self):
        entry = fixtures.bench_entry("llama 1B Q4_K - Medium", n_prompt=512, n_gen=0,
                                     n_threads=4, avg_ts=8.1)
        path = self.run_dir / "tinyllama-1.1b-q4km_pp512_t4.json"
        path.write_text(json.dumps(entry), encoding="utf-8")
        rows = bench_parse.rows_from(path, bench_parse.load_windows(self.run_dir),
                                     self.run_dir.name)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n_prompt"], 512)

    def test_missing_numeric_fields_do_not_fabricate_values(self):
        path = self.run_dir / "tinyllama-1.1b-q4km_pp128_t4.json"
        path.write_text(json.dumps([{"n_prompt": 128, "n_gen": 0, "n_threads": 4}]),
                        encoding="utf-8")
        rows = bench_parse.rows_from(path, bench_parse.load_windows(self.run_dir),
                                     self.run_dir.name)
        self.assertEqual(rows[0]["tokens_per_s"], 0)
        self.assertIn("no-throughput", rows[0]["flags"])

    def test_an_empty_run_directory_produces_a_header_only_csv(self):
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        out = empty / "results.csv"
        proc = subprocess.run(
            [sys.executable, str(ROOT / "node" / "bench_parse.py"), str(empty),
             "--out", str(out)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(read_csv(out), [])

    def test_a_nonexistent_run_directory_fails_loudly(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "node" / "bench_parse.py"),
             str(Path(self.tmp.name) / "nope")],
            capture_output=True, text=True,
        )
        self.assertNotEqual(proc.returncode, 0)


class TestAgreesWithConfig(unittest.TestCase):
    """The fixture hard-codes the model list; if config/models.txt moves, the tests must
    notice rather than quietly testing models the node no longer holds."""

    def test_fixture_models_match_config_models_txt(self):
        names = []
        for line in (ROOT / "config" / "models.txt").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line.split()[0])
        self.assertEqual(sorted(names), sorted(m[0] for m in fixtures.MODELS))

    def test_no_model_name_collides_with_the_filename_grammar(self):
        # The sweep writes "<model>_<phase><n>_t<threads>.json" and the parser recovers the
        # model with rsplit("_", 2). That is safe for any model name -- see
        # test_underscores_in_a_model_name_still_join -- but a name ending in something
        # that looks like the trailing two fields would be recovered short.
        for name, _mt, _q in fixtures.MODELS:
            tail = name.rsplit("_", 2)
            self.assertEqual(
                len(tail), 1,
                "model %r ends in a field the filename grammar would eat" % name,
            )


if __name__ == "__main__":
    unittest.main()

    def test_each_row_carries_the_idle_window_that_preceded_it(self):
        """A marginal (above-idle) figure has to be computed against THIS run's settle.

        The alternative on offer downstream is ThinkPad/control/analysis.py's
        `mean_power_w_idle`, which pools every idle_baseline window of the session with the
        120 s session-level one -- averaging a cold board with a hot one. So the per-run
        window has to survive the CSV boundary, or the only available idle figure is the
        wrong one.
        """
        rows = bench_parse.rows_from(
            self.run_dir / "tinyllama-1.1b-q4km_pp128_t4.json",
            bench_parse.load_windows(self.run_dir), self.run_dir.name)
        row = rows[0]
        self.assertTrue(row["idle_start"] and row["idle_end"],
                        "idle window must reach results.csv")
        self.assertLess(float(row["idle_end"]), float(row["t_start"]),
                        "the idle window must PRECEDE the run it is paired with")
        self.assertGreater(float(row["idle_end"]) - float(row["idle_start"]), 0)
