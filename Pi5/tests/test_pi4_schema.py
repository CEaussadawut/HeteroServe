"""Tests for the shared cross-node results schema emitted by bench/parse_bench.py.

The Pi 4 and the Pi 5 used to emit incompatible results CSVs, so the cross-node
characterisation table -- the project's first deliverable -- could not be assembled at all:
ThinkPad/control/profiles.py raised KeyError('phase') on the first Pi 5 row. The cluster
settled on Pi 4's schema, and this node now emits it alongside its own dataset.csv.

What makes this worth testing rather than eyeballing is that almost every way of getting it
wrong is SILENT. profiles.py:90-92 drops any row whose `flags` cell is non-empty, without a
message and without surfacing the count it computes. analysis.py multiplies the token count
by `samples` to get the joules-per-token denominator. analysis.py calls float() on t_start
with no guard. None of those announce a mistake; they just produce a table with the Pi 5
quietly missing, or a headline metric quietly wrong by a constant factor.

So these tests assert the contract from the consumer's side wherever they can, and the
first one guards the copy itself.
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import math
import re
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARSE_BENCH = ROOT / "bench" / "parse_bench.py"
PI4_BENCH_PARSE = ROOT.parent / "Pi4" / "node" / "bench_parse.py"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama_bench_fixture as fx  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("parse_bench", PARSE_BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pb = _load()
MANIFEST = {"node_id": "pi5", "llama_cpp_sha": "deadbeefcafe1234567890abcdef1234567890ab"}


class TestSchemaDrift(unittest.TestCase):
    """PI4_COLUMNS and quant_of are a deliberate standalone copy -- a worker node folder is
    deployed on its own and cannot import from Pi4/. That is the same reasoning
    node/gguf_meta.py records for the GGUF key names. A copy without a drift check is how
    the two schemas diverged in the first place, so read Pi 4's list live and compare."""

    def _pi4_literal(self, name):
        if not PI4_BENCH_PARSE.is_file():
            self.skipTest("Pi4/node/bench_parse.py not present (node folder deployed alone)")
        src = PI4_BENCH_PARSE.read_text(encoding="utf-8")
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", None) == name for t in node.targets):
                return ast.literal_eval(node.value)
        self.fail("%s not found in Pi4/node/bench_parse.py" % name)

    def test_columns_match_pi4_exactly(self):
        self.assertEqual(pb.PI4_COLUMNS, self._pi4_literal("COLUMNS"))

    def test_quant_hints_match_pi4_exactly(self):
        """A divergent quant table splits analysis.py's group key
        (node_id, model, quant, phase, n_threads) and produces two half-populated report
        rows per cell instead of one."""
        self.assertEqual([(c, list(s)) for c, s in pb.QUANT_HINTS],
                         [(c, list(s)) for c, s in self._pi4_literal("QUANT_HINTS")])


    def test_flag_vocabulary_has_not_drifted(self):
        """`flags` is load-bearing -- a non-empty cell is an opt-out from the placement
        rule -- so the two nodes must flag on the same basis. Pi 4 added `temp-unknown` and
        `no-samples:reps-unknown` after this emitter was first written, while COLUMNS stayed
        identical; a column-only guard would have missed it. Extract Pi 4's flag literals
        and require each to be either raised here or listed with a reason it cannot be."""
        if not PI4_BENCH_PARSE.is_file():
            self.skipTest("Pi4/node/bench_parse.py not present")
        tree = ast.parse(PI4_BENCH_PARSE.read_text(encoding="utf-8"))
        pi4_tokens = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "append"
                    and getattr(node.func.value, "id", "") == "flags"):
                continue
            arg = node.args[0] if node.args else None
            literal = None
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                literal = arg.value                              # "no-throughput"
            elif isinstance(arg, ast.BinOp) and isinstance(arg.left, ast.Constant):
                literal = arg.left.value                         # "x:" + str(...)
            if literal:
                pi4_tokens.add(literal.split(":")[0].rstrip("%s -"))
        self.assertGreater(len(pi4_tokens), 5, "failed to extract Pi 4's flag vocabulary")

        src = (ROOT / "bench" / "parse_bench.py").read_text(encoding="utf-8")
        raised = {t for t in pi4_tokens if ('flags.append("%s' % t) in src}
        documented = {t.split(":")[0] for t in pb.PI4_FLAGS_NOT_RAISED_HERE}
        unaccounted = pi4_tokens - raised - documented
        self.assertEqual(
            unaccounted, set(),
            "Pi 4 raises flag token(s) this node neither raises nor documents: %s. "
            "Decide for each: raise it on the same basis, or add it to "
            "PI4_FLAGS_NOT_RAISED_HERE with the reason." % sorted(unaccounted))


class TestFixtureMatchesTheSweep(unittest.TestCase):
    """The fixture is only evidence if it models what the sweep actually writes.

    Everything in these suites is built on `llama_bench_fixture`, so a fixture that has
    drifted from `10_isolated_bench.sh` makes every test green against a shape the node
    never produces. That already happened once on a different axis: the fixture encoded
    `avg_ns` as nanoseconds per token instead of per test, and nothing caught it because
    nothing compared the fixture to reality.
    """

    def test_cells_header_is_the_one_the_sweep_writes(self):
        sweep = (ROOT / "scripts" / "10_isolated_bench.sh").read_text(encoding="utf-8")
        m = re.search(r'^echo "(cell_id,[^"]*)" > "\$INDEX"', sweep, re.M)
        self.assertIsNotNone(m, "could not find the cells.csv header in 10_isolated_bench.sh")
        self.assertEqual(m.group(1).split(","), fx.CELLS_HEADER)

    def test_the_sweep_writes_one_field_per_header_column(self):
        """The header and the row are two separate `echo`s in the script, so they can drift
        apart silently -- and a row with the wrong field count shifts every later column
        without any parse error, because csv.DictReader will happily read a temperature as
        a filename."""
        sweep = (ROOT / "scripts" / "10_isolated_bench.sh").read_text(encoding="utf-8")
        # Greedy to the end of the line: the field list contains $(basename "$raw"), whose
        # inner quotes stop any [^"]* match well short of the real end.
        m = re.search(r'^\s*echo "(\$\{CELL\}.*)" >> "\$INDEX"', sweep, re.M)
        self.assertIsNotNone(m, "could not find the cells.csv row writer")
        self.assertEqual(len(m.group(1).split(",")), len(fx.CELLS_HEADER),
                         "row writer emits a different number of fields than the header")


class RunCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def build(self, cells, manifest=MANIFEST, name="pi5_isolated_20260913T051450Z", **kw):
        return fx.write_run(self.tmp / name, cells, manifest, **kw)

    def emit(self, run):
        proc = subprocess.run([sys.executable, str(PARSE_BENCH), str(run)],
                              capture_output=True, text=True)
        with (Path(run) / "results.csv").open(encoding="utf-8", newline="") as fh:
            return proc, list(csv.DictReader(fh))

    def one_cell(self, **kw):
        run = self.build([fx.cell(**kw)])
        _, rows = self.emit(run)
        return {r["phase"]: r for r in rows}


class TestRowShape(RunCase):
    def test_header_is_pi4s_columns_in_order(self):
        run = self.build([fx.cell()])
        self.emit(run)
        with (run / "results.csv").open(encoding="utf-8", newline="") as fh:
            self.assertEqual(next(csv.reader(fh)), pb.PI4_COLUMNS)

    def test_one_cell_becomes_one_row_per_phase(self):
        by = self.one_cell()
        self.assertEqual(sorted(by), ["pp", "tg"])

    def test_phase_spelling_is_lowercase(self):
        """profiles.py buckets on exactly "pp"/"tg"; analysis.py treats every non-"pp"
        value as decode."""
        by = self.one_cell()
        self.assertIn("pp", by)
        self.assertIn("tg", by)

    def test_two_cells_make_four_rows(self):
        run = self.build([fx.cell("c001"), fx.cell("c002")])
        _, rows = self.emit(run)
        self.assertEqual(len(rows), 4)


class TestPerLegFields(RunCase):
    """n_prompt, n_gen, n_threads and the throughputs must come from the leg's own entry.
    cells.csv carries the sweep-requested values, which are identical on both legs, so
    sourcing them from there would make the decode row claim it processed a prompt."""

    def test_prompt_and_gen_are_per_leg(self):
        by = self.one_cell(n_prompt=256, n_gen=128)
        self.assertEqual((by["pp"]["n_prompt"], by["pp"]["n_gen"]), ("256", "0"))
        self.assertEqual((by["tg"]["n_prompt"], by["tg"]["n_gen"]), ("0", "128"))

    def test_throughput_is_per_leg(self):
        by = self.one_cell(prefill_ts=11.5, decode_ts=2.35)
        self.assertEqual(float(by["pp"]["tokens_per_s"]), 11.5)
        self.assertEqual(float(by["tg"]["tokens_per_s"]), 2.35)

    def test_ms_per_token_is_the_inverse_of_that_leg(self):
        """profiles.py:101 reads ms_per_token and nothing else for timing."""
        by = self.one_cell(decode_ts=2.35)
        self.assertEqual(float(by["tg"]["ms_per_token"]), round(1000 / 2.35, 3))

    def test_row_is_internally_consistent(self):
        """ms_per_token = 1000/tokens_per_s and avg_ms comes from avg_ns. If any of the
        three were recomputed independently they would disagree inside one row."""
        for r in self.one_cell(n_gen=128, decode_ts=2.35).values():
            n = int(r["n_prompt"]) or int(r["n_gen"])
            self.assertAlmostEqual(
                float(r["avg_ms"]) / 1000 * float(r["tokens_per_s"]), n, places=1)
            # Relative, not absolute: ms_per_token is rounded to 3 dp and n reaches 512 in
            # the sweep, so the rounding alone moves the product by ~0.1 ms.
            lhs, rhs = float(r["ms_per_token"]) * n, float(r["avg_ms"])
            self.assertLess(abs(lhs - rhs) / rhs, 1e-4,
                            "row is internally inconsistent: %r" % r)


class TestDurationUnits(RunCase):
    """avg_ms, stddev_ms and rep_busy_s are whole-leg-per-repetition, not per token.
    Upstream get_ts() is `1e9 * n_tokens / t`, so the inverse carries the n_tokens factor.
    Getting it wrong is low by exactly n (64/128/256/512) and parses cleanly as a float."""

    def test_avg_ms_is_the_whole_leg(self):
        by = self.one_cell(n_gen=128, decode_ts=2.35)
        self.assertAlmostEqual(float(by["tg"]["avg_ms"]), 128 / 2.35 * 1000, places=1)

    def test_rep_busy_s_is_all_repetitions(self):
        by = self.one_cell(n_gen=128, decode_ts=2.35)
        self.assertAlmostEqual(float(by["tg"]["rep_busy_s"]),
                               int(by["tg"]["samples"]) * 128 / 2.35, places=1)

    def test_samples_is_the_repetition_count_not_the_outer_rep(self):
        """analysis.py multiplies the token count by `samples` for the joules-per-token
        denominator. cells.csv `rep` is the outer loop index 1..3, a different quantity;
        putting it here would misstate the project's headline metric with no warning."""
        run = self.build([fx.cell(rep=3, entries=[
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35,
                     samples_ts=[2.3, 2.35, 2.4, 2.33, 2.37]),
        ])])
        _, rows = self.emit(run)
        self.assertEqual(rows[0]["samples"], "5")


class TestEnergyWindow(RunCase):
    """One llama-bench process runs both legs, so its wall clock cannot be attributed to
    either phase without double-counting the cell's joules."""

    def test_t_start_and_t_end_are_empty(self):
        for r in self.one_cell().values():
            self.assertEqual(r["t_start"], "")
            self.assertEqual(r["t_end"], "")

    def test_energy_window_states_the_absence_in_band(self):
        for r in self.one_cell().values():
            self.assertEqual(r["energy_window"], "")

    def test_process_span_is_recorded_as_epoch_seconds(self):
        """analysis.py:139 calls float(t_start) with no guard and the CLI does not catch
        ValueError, so an ISO-8601 string would take the whole join down mid-loop. The
        honest process window lives in t_proc_*, correctly typed, where it cannot feed the
        energy integral."""
        r = self.one_cell()["tg"]
        self.assertGreater(float(r["t_proc_start"]), 1_600_000_000)
        self.assertGreater(float(r["t_proc_end"]), float(r["t_proc_start"]))

    def test_derived_window_columns_stay_empty(self):
        """These need a per-phase energy span, which this node does not have."""
        for r in self.one_cell().values():
            for col in ("rep_gap_s", "energy_duty", "load_s", "warmup_s"):
                self.assertEqual(r[col], "", col)

    def test_the_idle_window_is_carried_as_epoch_seconds(self):
        """Superseded an earlier decision: idle_start/idle_end used to be blank because the
        sweep's cool-down loop was untimed. It is now bracketed in epoch seconds -- the same
        base the DAQ uses -- so the settle before each cell is an idle window the energy join
        can integrate over, giving a per-row marginal (above-idle) figure instead of a
        session-wide mean that pools a cold board with a hot one."""
        r = self.one_cell()["tg"]
        self.assertGreater(float(r["idle_start"]), 1_600_000_000)
        self.assertGreater(float(r["idle_end"]), float(r["idle_start"]))

    def test_a_run_recorded_before_the_idle_columns_existed_stays_empty(self):
        by = self.one_cell(idle_start="", idle_end="")
        self.assertEqual(by["tg"]["idle_start"], "")
        self.assertEqual(by["tg"]["idle_end"], "")

    def test_energy_columns_are_present_but_empty(self):
        """join_run writes with DictWriter over this file's own fieldnames; a missing
        column makes writerows raise after the header is out, truncating the output."""
        for r in self.one_cell().values():
            for col in ("energy_j", "joules_per_token", "mean_power_w"):
                self.assertEqual(r[col], "", col)


class TestFlags(RunCase):
    """profiles.py:90-92 is a bare truthiness test, so any non-empty flags cell deletes the
    row from the function that produces the c_i Rule A consumes -- and the drop count is
    never printed. A flag is an opt-out from the placement rule, not a caveat within it."""

    def test_clean_row_has_no_flags(self):
        for r in self.one_cell().values():
            self.assertEqual(r["flags"], "")

    def test_a_hot_cell_is_not_flagged(self):
        """The 7B distribution baseline is the hottest cell in the sweep. Flagging on
        temperature would delete exactly the largest models from Rule A, and the exclusion
        would correlate with the quantity being measured. Pi 4 has no temperature flag
        either; temp_end_c carries the observation and any consumer can filter on it."""
        by = self.one_cell(temp_end_c="81.2")
        for r in by.values():
            self.assertEqual(r["flags"], "")
            self.assertEqual(r["temp_end_c"], "81.2")

    def test_throttled_end_carries_the_firmware_word(self):
        """Superseded an earlier decision: this column used to be hard-coded empty because
        the sweep never read vcgencmd, which made "unavailable" a self-inflicted wound (it
        implies throttle-unknown, which would have dropped every row). The sweep now records
        the word at both cell boundaries, so the column is a real measurement."""
        for r in self.one_cell().values():
            self.assertEqual(r["throttled_end"], "0x0")
            self.assertEqual(r["flags"], "")

    def test_an_unreadable_word_fails_closed(self):
        """common.sh returns "unavailable", never "0x0", when vcgencmd cannot be read --
        Pi 4's shell helper returns 0x0 there, which reads as clean and is the fail-open its
        own audit flagged."""
        by = self.one_cell(throttled_start="unavailable", throttled_end="unavailable")
        self.assertIn("throttle-unknown", by["tg"]["flags"])

    def test_a_run_recorded_before_the_throttle_columns_existed_is_unknown(self):
        """An empty word is "never checked", not "clean" -- and must not fall through to the
        delta, which would emit a bare `throttled:` asserting a throttle nobody observed."""
        by = self.one_cell(throttled_start="", throttled_end="")
        self.assertIn("throttle-unknown", by["tg"]["flags"])
        self.assertNotIn("throttled:", by["tg"]["flags"])

    def test_the_sticky_bits_do_not_flag_every_later_row(self):
        """The word latches bits 16-19 for the rest of the boot, so `end != 0x0` says only
        that the board throttled at SOME point since boot -- true of every row after the
        first blip. What invalidates a row is what changed DURING it."""
        by = self.one_cell(throttled_start="0x50000", throttled_end="0x50000")
        self.assertEqual(by["tg"]["flags"], "",
                         "a since-boot latch flagged a cell that did not throttle")

    def test_a_throttle_during_the_cell_is_flagged(self):
        by = self.one_cell(throttled_start="0x0", throttled_end="0x50005")
        self.assertIn("throttled:0x50005", by["tg"]["flags"])

    def test_a_board_already_capped_at_the_start_is_flagged(self):
        """The run began at the wrong clock, so its throughput is not characteristic."""
        by = self.one_cell(throttled_start="0x5", throttled_end="0x5")
        self.assertIn("throttled-at-start:0x5", by["tg"]["flags"])

    def test_nonzero_exit_is_flagged(self):
        by = self.one_cell(exit_code=1)
        self.assertIn("bench-exit:1", by["tg"]["flags"])

    def test_no_throughput_is_flagged(self):
        run = self.build([fx.cell(entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=0.0),
        ])])
        _, rows = self.emit(run)
        by = {r["phase"]: r for r in rows}
        self.assertIn("no-throughput", by["tg"]["flags"])
        self.assertEqual(by["pp"]["flags"], "")

    def test_every_dead_sensor_spelling_is_absent(self):
        """A LIVE BUG got in here once and is worth naming. `common.sh` was changed to emit
        the literal "nan" for an unreadable thermal zone, but this guard still only rejected
        `0`/`0.0` -- the OLD fallback. `float("nan")` parses, and `nan >= 80.0` is False, so
        a nan cell was published verbatim with `throttled=0`: "this cell was fine" about a
        cell nobody took the temperature of. Exactly the fail-open the guard exists to
        prevent, arriving through the one value the guard did not know about.

        Parameterised over every spelling, so narrowing the guard to any single one dies."""
        for bad in ("0", "0.0", "0.00", "-0.0", "-0.00", "nan", "NaN", "inf", "-inf"):
            with self.subTest(temp=bad):
                by = self.one_cell(temp_end_c=bad, temp_start_c=bad)
                for phase, r in by.items():
                    self.assertEqual(r["temp_end_c"], "", "%s leaked on the %s row" % (bad, phase))
                    self.assertIn("temp-unknown", r["flags"],
                                  "%s did not raise temp-unknown on the %s row" % (bad, phase))

    def test_a_real_temperature_is_not_discarded(self):
        """The mirror: the guard must not be so wide that it eats measurements."""
        for good in ("44.3", "81.2", "0.1", "-5.0"):
            with self.subTest(temp=good):
                r = self.one_cell(temp_end_c=good)["tg"]
                self.assertEqual(r["temp_end_c"], good)
                self.assertNotIn("temp-unknown", r["flags"])

    def test_the_parser_guards_the_sentinel_the_shell_actually_emits(self):
        """Pins the two files together, and is honest about what that buys.

        The guard is value-class-based -- anything not a finite non-zero number is rejected
        -- so swapping the shell's sentinel for another WORD cannot re-open the hole, and
        this test cannot fail that way. What it does catch is the dangerous direction: a
        sentinel that parses as a number. `-1` would sail straight through `isfinite` and be
        published as -1.0 C with no flag, which is how the original `0` fallback failed. So
        assert both that the sentinel is rejected AND that it is non-numeric by construction.
        """
        common = (ROOT / "scripts" / "lib" / "common.sh").read_text(encoding="utf-8")
        m = re.search(r"echo\s+\"([^\"]+)\"\s*;\s*return 0", common)
        self.assertIsNotNone(m, "could not find soc_temp_c's dead-sensor sentinel in common.sh")
        sentinel = m.group(1)
        self.assertEqual(pb.temp_or_blank(sentinel), "",
                         "common.sh emits %r for a dead sensor and the parser lets it "
                         "through" % sentinel)
        # "nan" DOES parse as a float -- it is simply not finite -- so the check is
        # finiteness, not parseability. A sentinel like "-1" would pass both and is the
        # case this exists to forbid.
        try:
            f = float(sentinel)
        except ValueError:
            return          # non-numeric: safe by construction
        self.assertFalse(
            math.isfinite(f) and abs(f) > 1e-9,
            "the shell's dead-sensor sentinel %r is a finite non-zero number, so a future "
            "widening of the parser's guard would publish it as a real temperature"
            % sentinel)

    def test_dead_thermal_sensor_is_absent_not_zero(self):
        """common.sh falls back to `echo 0` when the thermal zone is unreadable, so a
        literal 0.0 is a dead sensor, not a cold board. Emitting it would place a
        physically impossible reading beside Pi 4's genuine temperatures."""
        by = self.one_cell(temp_end_c="0")
        self.assertEqual(by["tg"]["temp_end_c"], "")

    def test_dead_thermal_sensor_is_flagged_like_pi4(self):
        """A blank temperature must read as "not measured", never as "fine". Pi 4 raises
        temp-unknown on exactly this condition, so raising it here keeps the combined
        table's exclusions even across the two nodes."""
        by = self.one_cell(temp_end_c="0")
        self.assertIn("temp-unknown", by["tg"]["flags"])

    def test_missing_repetitions_are_flagged(self):
        """analysis.py reads `samples` as the repetition count with `or 1`, so an empty
        samples_ns silently becomes reps=1 and overstates joules-per-token by the true
        repetition count -- 5x on a full sweep. Nothing downstream can detect it."""
        run = self.build([fx.cell(entries=[
            dict(fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35), samples_ns=[]),
        ])])
        _, rows = self.emit(run)
        self.assertEqual(rows[0]["samples"], "0")
        self.assertIn("no-samples:reps-unknown", rows[0]["flags"])


class TestIdentity(RunCase):
    def test_node_id_and_role_come_from_the_snapshot(self):
        """node.env.snapshot is copied unconditionally; build_manifest.json is copied with
        `[ -f ] && cp`, which does not abort under set -e."""
        r = self.one_cell()
        self.assertEqual(r["tg"]["node_id"], "pi5")
        self.assertEqual(r["tg"]["node_role"], "worker-strong")

    def test_node_id_survives_a_missing_manifest(self):
        run = self.build([fx.cell()], manifest=None)
        _, rows = self.emit(run)
        self.assertEqual(rows[0]["node_id"], "pi5")

    def test_missing_identity_warns_loudly(self):
        run = self.build([fx.cell()], manifest=None, node_env=None)
        proc, rows = self.emit(run)
        self.assertEqual(rows[0]["node_id"], "")
        self.assertIn("node_id is empty", proc.stderr)

    def test_llama_commit_is_the_short_build_commit(self):
        """Not the manifest's 40-char rev-parse output -- the two nodes must spell the
        pinned revision identically or any cross-node equality check fails."""
        r = self.one_cell()["tg"]
        self.assertEqual(r["llama_commit"], "deadbeef")
        self.assertNotEqual(r["llama_commit"], MANIFEST["llama_cpp_sha"])

    def test_run_id_is_the_directory_name(self):
        self.assertEqual(self.one_cell()["tg"]["run_id"], "pi5_isolated_20260913T051450Z")

    def test_model_is_the_bare_local_name(self):
        """The only join key to the control node's --layers JSON, which skips a miss
        silently. Never the cell_id, which carries _p/_g/_t/_r suffixes."""
        r = self.one_cell(model="qwen2.5-7b-q4km")["tg"]
        self.assertEqual(r["model"], "qwen2.5-7b-q4km")


class TestQuantAndSize(RunCase):
    def test_quant_is_canonical(self):
        r = self.one_cell(model="qwen2.5-0.5b-q4km")["tg"]
        self.assertEqual(r["quant"], "Q4_K_M")

    def test_q8_spelling_from_the_sweep_normalises(self):
        r = self.one_cell(model="tinyllama-1.1b-q8", entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5, model_type="Q8_0"),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35, model_type="Q8_0"),
        ])["tg"]
        self.assertEqual(r["quant"], "Q8_0")

    def test_quant_matches_pi4s_function_exactly(self):
        """Stronger than comparing the hint tables: run Pi 4's own quant_of over the real
        sweep's model names and require identical output. Guards the function body, not
        just the data. Note it flattens name AND model_type together and resolves a
        disagreement by QUANT_HINTS order -- that behaviour is copied, not corrected."""
        if not PI4_BENCH_PARSE.is_file():
            self.skipTest("Pi4/node/bench_parse.py not present")
        import re as _re
        src = PI4_BENCH_PARSE.read_text(encoding="utf-8")
        ns = {"re": _re}
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", None) == "QUANT_HINTS" for t in node.targets):
                ns["QUANT_HINTS"] = ast.literal_eval(node.value)
            if isinstance(node, ast.FunctionDef) and node.name == "quant_of":
                exec(compile(ast.Module([node], []), "<pi4>", "exec"), ns)
        pi4_quant_of = ns["quant_of"]
        names = [ln.split()[0] for ln in
                 (ROOT / "bench" / "sweep.conf").read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.startswith("#")]
        types = ["Q4_K - Medium", "Q8_0", "", "llama 1B Q4_K - Medium", "F16"]
        checked = 0
        for n in set(names):
            for t in types:
                self.assertEqual(pb.quant_of(n, t), pi4_quant_of(n, t),
                                 "diverged on (%r, %r)" % (n, t))
                checked += 1
        self.assertGreater(checked, 20)

    def test_model_type_is_kept_separately(self):
        """Pi5's own dataset.csv conflates these two; the shared schema keeps them apart."""
        r = self.one_cell()["tg"]
        self.assertEqual(r["model_type"], "Q4_K - Medium")
        self.assertNotEqual(r["model_type"], r["quant"])

    def test_model_size_is_mebibytes(self):
        r = self.one_cell()["tg"]
        self.assertAlmostEqual(float(r["model_size_mb"]), 4_700_000_000 / 1024 ** 2, places=1)

    def test_n_gpu_layers_is_the_offload_count(self):
        self.assertEqual(self.one_cell()["tg"]["n_gpu_layers"], "0")

    def test_per_layer_ms_uses_the_gguf_block_count(self):
        """Pi 4 leaves this empty. This node has an authoritative block_count from the GGUF
        header, so filling it is the only way that layer count reaches the shared table."""
        by = self.one_cell(n_layers=32, decode_ts=2.35)
        self.assertAlmostEqual(float(by["tg"]["per_layer_ms"]),
                               round(1000 / 2.35, 3) / 32, places=3)


class TestSourcesCannotSubstitute(RunCase):
    """Every precedence rule in this file is only enforced if the two candidate sources hold
    DIFFERENT values. The default fixture gives them the same value everywhere -- cells.csv
    `threads` equals the leg's `n_threads`, node.env's NODE_ID equals the manifest's, the
    two n_layers sources agree -- so a mutation that reads the wrong source produces
    identical output and the suite stays green.

    These tests deliberately make the sources disagree. The worked-example defaults are left
    alone: they reproduce the reference numbers carried in the handoffs, and that is worth
    keeping as a separate thing from precedence enforcement.
    """

    def test_n_threads_comes_from_the_leg_not_the_cell(self):
        """llama-bench clamps the requested thread count to hardware concurrency, so the
        entry's value is what actually ran and cells.csv's is only what was asked for. The
        1/2/4-thread rungs are an entire axis of the sweep, so publishing the request as the
        result would fabricate the thread-scaling curve."""
        run = self.build([fx.cell(threads=4, entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5, n_threads=3),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35, n_threads=3),
        ])])
        _, rows = self.emit(run)
        for r in rows:
            self.assertEqual(r["n_threads"], "3",
                             "n_threads came from cells.csv (4) not the leg entry (3)")

    def test_prompt_and_gen_come_from_the_leg_not_the_cell(self):
        """cells.csv carries the sweep request on BOTH legs; only the entry knows which leg
        it is. Distinct values on every axis so no two can be confused."""
        run = self.build([fx.cell(n_prompt=512, n_gen=64, entries=[
            fx.entry(n_prompt=512, n_gen=0, avg_ts=11.5),
            fx.entry(n_prompt=0, n_gen=64, avg_ts=2.35),
        ])])
        _, rows = self.emit(run)
        by = {r["phase"]: r for r in rows}
        self.assertEqual((by["pp"]["n_prompt"], by["pp"]["n_gen"]), ("512", "0"))
        self.assertEqual((by["tg"]["n_prompt"], by["tg"]["n_gen"]), ("0", "64"))

    def test_node_identity_prefers_the_snapshot_over_the_manifest(self):
        """node.env.snapshot is copied unconditionally; build_manifest.json with
        `[ -f ] && cp`, which fails silently. A stale manifest relabels a whole run as the
        wrong device -- the worst corruption possible in a device-characterisation dataset,
        precisely because the rows stay internally consistent."""
        run = self.build([fx.cell()],
                         manifest={"node_id": "pi4", "llama_cpp_sha": "deadbeef"},
                         node_env={"NODE_ID": "pi5", "NODE_ROLE": "worker-strong"})
        _, rows = self.emit(run)
        for r in rows:
            self.assertEqual(r["node_id"], "pi5", "the stale manifest won over the snapshot")
        with (Path(run) / "dataset.csv").open(encoding="utf-8", newline="") as fh:
            self.assertEqual(next(csv.DictReader(fh))["node_id"], "pi5",
                             "dataset.csv took node_id from the manifest")

    def test_the_layer_count_prefers_cells_csv_when_both_disagree(self):
        """The existing test for this cannot fail: the JSON reports n_gpu_layers=0, and 0 is
        falsy, so the assertion holds under either precedence. Make both non-zero and
        different."""
        run = self.build([fx.cell(n_layers=28, entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5, n_gpu_layers=99, n_layer=99),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35, n_gpu_layers=99, n_layer=99),
        ])])
        _, rows = self.emit(run)
        by = {r["phase"]: r for r in rows}
        self.assertAlmostEqual(float(by["tg"]["per_layer_ms"]),
                               round(round(1000 / 2.35, 3) / 28, 4), places=3)


class TestRuleAsInputIsAsserted(RunCase):
    """c_per_layer_ms / per_layer_ms is the c_i Rule A divides layers by, and until now no
    test anywhere asserted its VALUE -- only that it was non-empty. A mutation dividing by
    `(n_layers or 1)` published the whole-model time as a per-layer time, 32x too large,
    parsing cleanly with no flag, and every one of the 182 tests stayed green."""

    def test_per_layer_ms_is_the_quotient_not_a_constant(self):
        for n_layers, decode in ((28, 2.35), (22, 6.0), (32, 2.35)):
            with self.subTest(n_layers=n_layers):
                by = self.one_cell(n_layers=n_layers, decode_ts=decode)
                ms_per_token = round(1000.0 / decode, 3)
                self.assertAlmostEqual(float(by["tg"]["per_layer_ms"]),
                                       round(ms_per_token / n_layers, 4), places=4)

    def test_a_missing_layer_count_never_divides_by_one(self):
        """`(n_layers or 1)` is the mutation that survived. An absent layer count must leave
        the column empty, not publish the whole-model time as a per-layer cost."""
        cell = fx.cell()
        del cell["n_layers"]
        run = self.build([cell])
        _, rows = self.emit(run)
        for r in rows:
            self.assertEqual(r["per_layer_ms"], "",
                             "a missing layer count produced a per-layer value")


class TestThrottleDecoder(unittest.TestCase):
    """The R5 thermal-validity decision, exercised with exactly three firmware words before
    this. Each live bit has a distinct meaning and the sticky half latches for the rest of
    the boot, so narrowing LIVE_BITS from 0xF to 0x7 -- dropping soft-temp-limit -- survived
    the whole suite. Table-driven over every bit individually.

    Word layout (Raspberry Pi firmware):
      bit 0 under-voltage now      bit 16 under-voltage since boot
      bit 1 arm frequency capped   bit 17 capped since boot
      bit 2 currently throttled    bit 18 throttled since boot
      bit 3 soft temperature limit bit 19 soft temp limit since boot
    """

    CASES = [
        # (start, end, expect_live_at_start, expect_throttled_during, note)
        ("0x0", "0x0", False, False, "clean board, nothing happened"),
        ("0x0", "0x1", False, True, "under-voltage appeared during the cell"),
        ("0x0", "0x2", False, True, "frequency capped during the cell"),
        ("0x0", "0x4", False, True, "throttled during the cell"),
        ("0x0", "0x8", False, True, "soft temp limit during the cell"),
        ("0x0", "0x10000", False, True, "under-voltage latched during the cell"),
        ("0x0", "0x40000", False, True, "throttle latched during the cell"),
        ("0x0", "0x80000", False, True, "soft-temp latched during the cell"),
        ("0x1", "0x1", True, True, "already under-voltage at the start"),
        ("0x4", "0x4", True, True, "already throttled at the start"),
        ("0x8", "0x8", True, True, "already at soft temp limit at the start"),
        ("0x50000", "0x50000", False, False,
         "sticky-only word unchanged: the board throttled EARLIER in the boot, not here"),
        ("0x50000", "0x50005", False, True, "live bits appeared on top of an old latch"),
        ("0x0", "0x50005", False, True, "the worked example from the Pi 4 handoff"),
    ]

    def test_each_bit_is_decoded(self):
        for start, end, live, during, note in self.CASES:
            with self.subTest(start=start, end=end, note=note):
                self.assertEqual(pb._throttle_delta(start, end), (live, during), note)

    def test_an_unparseable_word_fails_closed(self):
        """Treated as throttled, because the alternative is publishing a row nobody checked."""
        for start, end in (("garbage", "0x0"), ("0x0", "garbage"), ("0x0", "")):
            with self.subTest(start=start, end=end):
                self.assertEqual(pb._throttle_delta(start, end)[1], True)

    def test_the_sticky_half_alone_never_invalidates_a_row(self):
        """The whole point of comparing two words. Every sticky bit, set before the cell and
        unchanged through it, must leave the row valid -- otherwise one blip early in a boot
        flags every remaining row of a multi-hour sweep."""
        for bit in (0x10000, 0x20000, 0x40000, 0x80000):
            with self.subTest(bit=hex(bit)):
                self.assertEqual(pb._throttle_delta(hex(bit), hex(bit)), (False, False))


class TestEpochConversion(unittest.TestCase):
    """Every epoch timestamp in this project was asserted by MAGNITUDE (> 1_600_000_000) and
    ORDERING (end > start), never by value -- so scaling the conversion by 1000, or dropping
    the timezone so a +07 host shifts everything seven hours, both survived. These feed the
    energy join, which matches them against DAQ sample timestamps."""

    def test_a_known_stamp_converts_exactly(self):
        # 2026-09-13T05:14:50.538Z -- computed independently of the implementation.
        from datetime import datetime, timezone
        expected = datetime(2026, 9, 13, 5, 14, 50, 538000, tzinfo=timezone.utc).timestamp()
        self.assertAlmostEqual(pb.iso_to_epoch("2026-09-13T05:14:50.538Z"), expected, places=3)

    def test_the_trailing_z_is_utc_not_local(self):
        """A naive parse on this +07 host shifts every window by 25200 s. The energy join
        would then integrate the wrong seven hours of power trace and no test noticed."""
        utc = pb.iso_to_epoch("2026-09-13T00:00:00.000Z")
        explicit = pb.iso_to_epoch("2026-09-13T00:00:00.000+00:00")
        self.assertAlmostEqual(utc, explicit, places=6)
        self.assertAlmostEqual(utc, 1789257600.0, places=3)

    def test_the_scale_is_seconds_not_milliseconds(self):
        a = pb.iso_to_epoch("2026-09-13T00:00:00.000Z")
        b = pb.iso_to_epoch("2026-09-13T00:00:01.000Z")
        self.assertAlmostEqual(b - a, 1.0, places=6,
                               msg="one second of wall clock did not move epoch by 1.0")

    def test_millisecond_resolution_survives(self):
        a = pb.iso_to_epoch("2026-09-13T00:00:00.000Z")
        b = pb.iso_to_epoch("2026-09-13T00:00:00.250Z")
        self.assertAlmostEqual(b - a, 0.25, places=6)

    def test_an_epoch_string_passes_through_unchanged(self):
        self.assertAlmostEqual(pb.iso_to_epoch("1789257600.5"), 1789257600.5, places=6)

    def test_unparseable_input_is_empty_not_zero(self):
        for bad in ("", "   ", None, "not a time", "2026-13-45T99:99:99Z"):
            with self.subTest(value=bad):
                self.assertEqual(pb.iso_to_epoch(bad), "")


class TestFlagsAreAContract(RunCase):
    """`flags` is parsed by the control node, not just read by humans:
    analysis.invalidates_throughput does `(flags or "").split(";")` then
    `part.startswith(("throttled", "throttle-unknown"))`. Every test here asserted with
    assertIn of a substring, so the SEPARATOR was never pinned -- changing ";" to ","
    survived, and with commas a row flagged "temp-unknown,throttled:0x50005" is ONE part
    beginning "temp-unknown", so a throttled row silently enters the throughput summary."""

    def _invalidates_throughput(self, flags):
        """The consumer's own two lines, vendored the way quant_of already is."""
        return any(part.strip().startswith(("throttled", "throttle-unknown"))
                   for part in (flags or "").split(";") if part.strip())

    def test_multiple_flags_are_semicolon_separated(self):
        by = self.one_cell(exit_code=1, temp_end_c="0")
        flags = by["tg"]["flags"]
        self.assertEqual(sorted(flags.split(";")), ["bench-exit:1", "temp-unknown"])

    def test_a_throttled_row_survives_the_consumers_split(self):
        """The end-to-end version: a row flagged throttled ALONGSIDE another flag must still
        be recognised as throttled after the consumer splits it."""
        by = self.one_cell(temp_end_c="0", throttled_start="0x0", throttled_end="0x50005")
        flags = by["tg"]["flags"]
        self.assertIn("temp-unknown", flags)
        self.assertIn("throttled:0x50005", flags)
        self.assertTrue(self._invalidates_throughput(flags),
                        "the consumer could not find the throttle in %r" % flags)

    def test_a_clean_row_does_not_invalidate_throughput(self):
        self.assertFalse(self._invalidates_throughput(self.one_cell()["tg"]["flags"]))

    def test_a_blank_exit_code_is_not_a_failure(self):
        self.assertEqual(self.one_cell(exit_code="")["tg"]["flags"], "")


class TestRepetitionDurations(RunCase):
    """rep_busy_s is the sum over ALL repetitions. The default cell has one sample, where
    sum, mean, max and first are numerically identical -- so `sum` -> `mean` survived. A 5x
    low numerator makes every Pi 5 row look ~20% duty and trips Pi 4's 0.98 duty gate,
    quietly excluding this node from the energy table."""

    SAMPLES = [2.30, 2.35, 2.40, 2.33, 2.37]

    def _multi(self):
        return self.build([fx.cell(n_gen=128, entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35, samples_ts=self.SAMPLES),
        ])])

    def test_rep_busy_s_sums_every_repetition(self):
        _, rows = self.emit(self._multi())
        tg = {r["phase"]: r for r in rows}["tg"]
        expected = sum(128.0 / s for s in self.SAMPLES)
        self.assertAlmostEqual(float(tg["rep_busy_s"]), expected, places=2)
        self.assertEqual(tg["samples"], "5")
        # and it must be ~5x a single repetition, not ~1x
        self.assertGreater(float(tg["rep_busy_s"]), 4 * (128.0 / 2.40))

    PP_SAMPLES = [11.2, 11.5, 11.8, 11.35, 11.65]

    def _both_legs(self):
        """Different sample sets per leg, so prefill and decode spreads cannot be swapped
        or sourced from each other."""
        return self.build([fx.cell(n_prompt=256, n_gen=128, entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5, samples_ts=self.PP_SAMPLES),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35, samples_ts=self.SAMPLES),
        ])])

    def test_stddev_ms_is_the_spread_of_the_durations(self):
        """This column's arithmetic was UNREACHABLE: the fixture hardcoded stddev_ns=0, and
        0/1e6 is 0 whatever the conversion does, so scaling it by 1000 or blanking it
        outright both passed. The fixture now derives stddev_ns from samples_ns the way
        upstream does."""
        import statistics
        _, rows = self.emit(self._both_legs())
        by = {r["phase"]: r for r in rows}
        for phase, samples, n in (("tg", self.SAMPLES, 128), ("pp", self.PP_SAMPLES, 256)):
            with self.subTest(phase=phase):
                ns = [int(n * 1e9 / x) for x in samples]
                self.assertAlmostEqual(float(by[phase]["stddev_ms"]),
                                       round(int(statistics.stdev(ns)) / 1e6, 3), places=2)
                self.assertGreater(float(by[phase]["stddev_ms"]), 0.0)

    def test_the_two_legs_do_not_share_a_spread(self):
        _, rows = self.emit(self._both_legs())
        by = {r["phase"]: r for r in rows}
        self.assertNotAlmostEqual(float(by["pp"]["stddev_ms"]),
                                  float(by["tg"]["stddev_ms"]), places=1)
        self.assertNotAlmostEqual(float(by["pp"]["tokens_per_s_stddev"]),
                                  float(by["tg"]["tokens_per_s_stddev"]), places=3)

    def test_the_spread_columns_are_values_not_floors(self):
        """stddev was only ever asserted as > 0.0, so rounding it to 0 dp, or swapping
        sample stddev for population stddev, both survived."""
        import statistics
        _, rows = self.emit(self._multi())
        tg = {r["phase"]: r for r in rows}["tg"]
        expected = statistics.stdev(self.SAMPLES)
        self.assertAlmostEqual(float(tg["tokens_per_s_stddev"]), round(expected, 4), places=4)
        self.assertNotAlmostEqual(float(tg["tokens_per_s_stddev"]),
                                  round(statistics.pstdev(self.SAMPLES), 4), places=4,
                                  msg="population stddev would understate every error bar")


class TestQuantTableIsFullyExercised(unittest.TestCase):
    """test_quant_matches_pi4s_function_exactly drew its inputs from sweep.conf, which only
    ever spells q4km and q8 -- so five of the eight QUANT_HINTS rows had no witness and
    truncating the loop to [:6] survived. Drive the test from the table itself."""

    def test_every_hint_row_and_spelling_has_a_witness(self):
        for canonical, spellings in pb.QUANT_HINTS:
            for sp in spellings:
                with self.subTest(canonical=canonical, spelling=sp):
                    self.assertEqual(pb.quant_of("model-" + sp.lower(), ""), canonical)

    def test_an_unknown_quant_falls_back_to_the_raw_label(self):
        self.assertEqual(pb.quant_of("mystery-model", "Q3_K_L"), "Q3_K_L")

    def test_an_unknown_quant_with_no_label_is_unknown(self):
        self.assertEqual(pb.quant_of("mystery-model", ""), "unknown")

    def test_both_argument_sources_are_consulted(self):
        """The fixture's name and model_type always agree, so deleting either argument at
        the call site survived."""
        self.assertEqual(pb.quant_of("tinyllama-1.1b-q8", ""), "Q8_0")
        self.assertEqual(pb.quant_of("mystery", "Q6_K"), "Q6_K")


class TestConsumerRoundTrip(RunCase):
    """Drive the rows through the same expressions the control node uses. These are the
    tests that would have caught the original incompatibility."""

    def test_profiles_per_layer_times_accepts_every_row(self):
        """Mirrors ThinkPad/control/profiles.py:89-104 -- the function that produces c_i."""
        run = self.build([fx.cell("c001"), fx.cell("c002", model="tinyllama-1.1b-q8")])
        _, rows = self.emit(run)
        n_layers_by_model = {"qwen2.5-7b-q4km": 28, "tinyllama-1.1b-q8": 22}
        acc, dropped = {}, 0
        for row in rows:
            if row.get("flags"):
                dropped += 1
                continue
            node, phase, model = row["node_id"], row["phase"], row["model"]
            n_layers = n_layers_by_model.get(model)
            if not n_layers:
                continue
            ms = float(row.get("ms_per_token") or 0)
            if not ms:
                continue
            acc.setdefault(node, {}).setdefault(phase, []).append(ms / n_layers)
        self.assertEqual(dropped, 0, "rows were silently dropped by the flags filter")
        self.assertEqual(sorted(acc["pi5"]), ["pp", "tg"])
        self.assertEqual(len(acc["pi5"]["tg"]), 2)

    def test_analysis_join_skips_cleanly_rather_than_crashing(self):
        """Mirrors analysis.py:132-139. With t_start empty the row is skipped and flagged
        no-window -- the correct outcome for a node with no per-phase window. What must NOT
        happen is float() raising on an ISO-8601 string, which is uncaught by the CLI."""
        _, rows = self.emit(self.build([fx.cell()]))
        for row in rows:
            t_start, t_end = row.get("t_start"), row.get("t_end")
            if not (row.get("node_id") and t_start and t_end):
                continue
            float(t_start), float(t_end)   # must never raise

    def test_every_numeric_column_parses(self):
        _, rows = self.emit(self.build([fx.cell()]))
        numeric = ["model_size_mb", "tokens_per_s", "tokens_per_s_stddev", "avg_ms",
                   "stddev_ms", "samples", "ms_per_token", "per_layer_ms",
                   "t_proc_start", "t_proc_end", "rep_busy_s", "n_prompt", "n_gen",
                   "n_threads", "n_gpu_layers"]
        for row in rows:
            for col in numeric:
                if row[col] != "":
                    float(row[col])

    def test_concatenates_with_a_pi4_row(self):
        """The whole point: one DictReader over both files must see one consistent schema."""
        _, rows = self.emit(self.build([fx.cell()]))
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=pb.PI4_COLUMNS)
        w.writeheader()
        w.writerows(rows)
        pi4_row = {c: "" for c in pb.PI4_COLUMNS}
        pi4_row.update({"node_id": "pi4", "phase": "tg", "model": "tinyllama-1.1b-q4km",
                        "ms_per_token": "980.0", "n_threads": "4", "quant": "Q4_K_M"})
        w.writerow(pi4_row)
        buf.seek(0)
        back = list(csv.DictReader(buf))
        self.assertEqual(len(back), 3)
        self.assertEqual({r["node_id"] for r in back}, {"pi5", "pi4"})
        for r in back:
            self.assertEqual(set(r), set(pb.PI4_COLUMNS))


class TestOutputIndependence(RunCase):
    def test_shared_file_survives_the_c_per_layer_gate(self):
        """parse_bench exits 1 when any row lacks c_per_layer_ms. That gate is about this
        node's own dataset; it must not also remove the Pi 5 from the combined table."""
        cell = fx.cell()
        del cell["n_layers"]
        run = self.build([cell])
        proc, rows = self.emit(run)
        self.assertEqual(proc.returncode, 1)
        self.assertTrue((run / "results.csv").is_file())
        self.assertEqual(len(rows), 2)

    def test_dataset_csv_is_still_written(self):
        run = self.build([fx.cell()])
        self.emit(run)
        self.assertTrue((run / "dataset.csv").is_file())

    def test_pi4_out_flag_redirects(self):
        run = self.build([fx.cell()])
        dest = self.tmp / "shared.csv"
        subprocess.run([sys.executable, str(PARSE_BENCH), str(run),
                        "--pi4-out", str(dest)], capture_output=True, text=True)
        self.assertTrue(dest.is_file())


class TestTheProcessingWindowAndQuantComeFromTheRightSource(RunCase):
    """Two fields that a mutation could re-source from a neighbouring value while every
    column still looked populated and plausible."""

    def test_t_proc_comes_from_the_cell_timestamps_not_the_idle_window(self):
        """`t_proc_start`/`t_proc_end` bound the cell's COMPUTE; `idle_start`/`idle_end`
        bound the cool-down BEFORE it. They sit next to each other in the row and both are
        epoch seconds, so substituting one for the other keeps the row well-formed. It
        would attribute the idle period's energy to the run, which is the joules-per-token
        denominator -- the number the energy half of this study reports."""
        by = self.one_cell(started_utc="2026-09-13T05:14:50Z",
                           ended_utc="2026-09-13T05:19:10Z",
                           idle_start="1700000000.0", idle_end="1700000100.0")
        r = by["tg"]
        self.assertEqual(r["t_proc_start"], str(pb.iso_to_epoch("2026-09-13T05:14:50Z")))
        self.assertEqual(r["t_proc_end"], str(pb.iso_to_epoch("2026-09-13T05:19:10Z")))
        self.assertNotEqual(r["t_proc_start"], r["idle_start"],
                            "the processing window was taken from the idle window")
        self.assertNotEqual(r["t_proc_end"], r["t_proc_start"],
                            "t_proc_end was derived from t_proc_start rather than measured")

    def test_quant_uses_llama_benchs_model_type_when_the_filename_is_silent(self):
        """`quant_of` reads the filename AND llama-bench's `model_type`. Models whose file
        name carries no quant tag are resolved only by the second source, and `quant` is
        part of analysis.py's group key -- an unresolved one splits a model into two groups
        that are then averaged separately."""
        by = self.one_cell(model="mystery-model", entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5, model_type="Q4_K - Medium"),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35, model_type="Q4_K - Medium"),
        ])
        self.assertEqual(by["tg"]["quant"], "Q4_K_M",
                         "quant did not fall back to llama-bench's model_type")


class TestEachLegsFieldsComeFromThatLegsEntry(RunCase):
    """The PI4_COLUMNS contract says the per-leg fields come from the leg's OWN llama-bench
    entry. Every one of these mutations swaps a field for a neighbouring one of the same
    type, or for a constant, and the 40-column row stays well-formed and plausible."""

    def test_a_mixed_entry_is_classified_by_its_prompt_not_its_generation(self):
        """`phase = "pp" if n_prompt > 0 else "tg"`. A llama-bench entry with BOTH n_prompt
        and n_gen is the shape `pick()`'s docstring exists for, and it is the only input
        that can tell this rule from `"tg" if n_gen > 0`. `phase` is the group key the
        analysis splits prefill from decode on -- getting it backwards files the prefill
        measurement as decode, which is the ~5x error class this project has already hit
        once in `pick()`."""
        by = self.one_cell(entries=[
            fx.entry(n_prompt=256, n_gen=128, avg_ts=11.5),
        ])
        self.assertIn("pp", by, "a mixed entry was not classified as prefill")
        self.assertEqual(by["pp"]["n_prompt"], "256")
        self.assertEqual(by["pp"]["n_gen"], "128")

    def test_tokens_per_s_is_llama_benchs_avg_ts_verbatim(self):
        """`ms_per_token` is defined as 1000/tokens_per_s and `avg_ms` comes from avg_ns,
        so recomputing any one of the three from the samples makes the three mutually
        inconsistent INSIDE a single row -- a reader checking 1000/tps against ms_per_token
        would find they disagree. The samples here mean 3.033 while avg_ts says 2.35."""
        by = self.one_cell(entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35, samples_ts=[2.0, 2.1, 5.0]),
        ])
        r = by["tg"]
        self.assertEqual(r["tokens_per_s"], "2.35",
                         "tokens_per_s was recomputed from the samples instead of avg_ts")
        self.assertEqual(r["ms_per_token"], "425.532",
                         "ms_per_token no longer equals 1000/tokens_per_s")

    def test_n_params_is_the_parameter_count_not_the_file_size(self):
        """Both are large integers from the same entry, so a swap is invisible in the row.
        7.6e9 parameters and 4.7e9 bytes describe different things about the same model."""
        by = self.one_cell()
        self.assertEqual(by["tg"]["n_params"], "7615616512")

    def test_n_batch_is_the_batch_not_the_microbatch(self):
        """llama-bench emits both; they differ on a real run (2048 vs 512)."""
        by = self.one_cell(entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5, n_batch=2048, n_ubatch=512),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35, n_batch=2048, n_ubatch=512),
        ])
        self.assertEqual(by["tg"]["n_batch"], "2048")

    def test_n_gpu_layers_is_read_not_assumed_zero(self):
        """0 on this CPU-only node, which is exactly why a hardcoded 0 is invisible here.
        The column exists for the Jetson rows in the shared file."""
        by = self.one_cell(entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5, n_gpu_layers=28),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35, n_gpu_layers=28),
        ])
        self.assertEqual(by["tg"]["n_gpu_layers"], "28")

    def test_a_leg_without_a_thread_count_is_flagged_not_filled_in(self):
        """The contract requires n_threads from the leg's own entry. cells.csv also has a
        threads column, so silently substituting it would look right and be unverifiable;
        the flag is the opt-out that keeps the row out of the placement fit instead."""
        by = self.one_cell(entries=[
            fx.entry(n_prompt=256, n_gen=0, avg_ts=11.5, n_threads=None),
            fx.entry(n_prompt=0, n_gen=128, avg_ts=2.35, n_threads=None),
        ])
        self.assertIn("no-threads", by["tg"]["flags"],
                      "a leg with no thread count was published unflagged")


class TestTheThrottleWordAndIdleWindowAreNotInvented(RunCase):
    def test_an_unreadable_throttle_word_is_never_reported_as_clean(self):
        """`10_isolated_bench.sh` writes "unavailable" when vcgencmd could not be read.
        "0x0" is the firmware word for "nothing ever tripped" -- publishing it for a word
        nobody read turns a missing measurement into a clean bill of health, and
        `throttled_end` is the column the energy analysis uses to discard a cell."""
        by = self.one_cell(throttled_start="unavailable", throttled_end="unavailable")
        self.assertEqual(by["tg"]["throttled_end"], "unavailable",
                         "an unreadable throttle word was published as 0x0 (clean)")

    def test_a_real_throttle_word_is_passed_through(self):
        by = self.one_cell(throttled_start="0x0", throttled_end="0x50005")
        self.assertEqual(by["tg"]["throttled_end"], "0x50005")

    def test_a_non_numeric_idle_marker_does_not_leak_into_a_numeric_column(self):
        """`idle_start`/`idle_end` are epoch seconds that the control node's energy join
        reads as numbers. Passing the raw cell through puts a word in a numeric column;
        `_num(..., "")` turns anything unparseable into an explicit blank."""
        by = self.one_cell(idle_start="n/a", idle_end="n/a")
        self.assertEqual(by["tg"]["idle_start"], "",
                         "a non-numeric idle marker reached the numeric column")
        self.assertEqual(by["tg"]["idle_end"], "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
