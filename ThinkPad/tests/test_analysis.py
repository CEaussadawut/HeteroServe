"""Tests for the energy join.

The point of these is the refusals. Producing a joules-per-token number is easy; producing
one only when the power trace actually supports it is the part that has to be right.
"""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from control import analysis, energy


def write_power(path, node="pi4", start=1000.0, seconds=10.0, hz=28.0, watts=5.0, gap=None):
    """Synthetic power trace: constant `watts`, optionally with a gap in the middle."""
    rows = [analysis.POWER_HEADER] if hasattr(analysis, "POWER_HEADER") else \
           [["t_utc", "channel", "node", "bus_v", "current_a", "power_w", "pico_us"]]
    period = 1.0 / hz
    t = start
    while t <= start + seconds:
        if gap and gap[0] <= t <= gap[1]:
            t += period
            continue
        rows.append(["%.6f" % t, 2, node, 5.1, watts / 5.1, watts, int(t * 1e6)])
        t += period
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    return path


def write_results(run_dir, rows):
    from control.analysis import tokens_in_row  # noqa: F401  (schema sanity)
    fields = ["node_id", "node_role", "run_id", "llama_commit", "model", "model_type", "quant",
              "model_size_mb", "n_params", "phase", "n_prompt", "n_gen", "n_threads", "n_batch",
              "n_gpu_layers", "tokens_per_s", "tokens_per_s_stddev", "avg_ms", "stddev_ms",
              "samples", "ms_per_token", "per_layer_ms", "t_start", "t_end", "temp_end_c",
              "throttled_end", "energy_j", "joules_per_token", "mean_power_w", "flags"]
    path = Path(run_dir) / "results.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            base = {f: "" for f in fields}
            base.update(row)
            writer.writerow(base)
    return path


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_constant_power_integrates_to_power_times_time(self):
        path = write_power(self.dir / "power.csv", watts=5.0, start=1000.0, seconds=10.0)
        samples = analysis.load_power(path, node="pi4")
        result = analysis.integrate(samples, 1001.0, 1006.0)
        # 5 W for ~5 s = ~25 J, allowing for sample-boundary truncation
        self.assertAlmostEqual(result["energy_j"], 25.0, delta=0.5)
        self.assertAlmostEqual(result["mean_power_w"], 5.0, places=2)
        self.assertEqual(result["flags"], [])

    def test_too_few_samples_refuses_rather_than_extrapolates(self):
        path = write_power(self.dir / "power.csv", hz=2.0, seconds=10.0)
        samples = analysis.load_power(path, node="pi4")
        result = analysis.integrate(samples, 1000.0, 1001.0, sample_hz=2.0)
        self.assertIsNone(result["energy_j"])
        self.assertTrue(any(f.startswith("too-few-samples") for f in result["flags"]))

    def test_a_dropout_is_flagged(self):
        path = write_power(self.dir / "power.csv", seconds=20.0, gap=(1005.0, 1008.0))
        samples = analysis.load_power(path, node="pi4")
        result = analysis.integrate(samples, 1000.0, 1015.0)
        self.assertIsNotNone(result["energy_j"])
        self.assertTrue(any(f.startswith("gap:") for f in result["flags"]),
                        "a three-second DAQ dropout must be visible in the flags")

    def test_window_starting_before_the_recording_is_flagged(self):
        path = write_power(self.dir / "power.csv", start=1000.0, seconds=10.0)
        samples = analysis.load_power(path, node="pi4")
        result = analysis.integrate(samples, 990.0, 1008.0)
        self.assertTrue(any(f.startswith("late-start") for f in result["flags"]))


class TestTokenCounting(unittest.TestCase):
    def test_repetitions_are_counted(self):
        # llama-bench repeats each measurement; the window covers every repetition.
        row = {"phase": "tg", "n_gen": "128", "n_prompt": "0", "samples": "5"}
        self.assertEqual(analysis.tokens_in_row(row), 640)

    def test_prefill_uses_prompt_tokens(self):
        row = {"phase": "pp", "n_prompt": "512", "n_gen": "0", "samples": "3"}
        self.assertEqual(analysis.tokens_in_row(row), 1536)

    def test_missing_repetition_count_defaults_to_one(self):
        row = {"phase": "tg", "n_gen": "64", "n_prompt": "0", "samples": ""}
        self.assertEqual(analysis.tokens_in_row(row), 64)


class TestJoin(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.power = write_power(self.dir / "power.csv", start=1000.0, seconds=60.0, watts=6.0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_join_fills_energy_columns(self):
        run = self.dir / "run1"
        run.mkdir()
        write_results(run, [{
            "node_id": "pi4", "model": "tiny", "phase": "tg", "n_gen": "128", "n_prompt": "0",
            "n_threads": "4", "samples": "2", "tokens_per_s": "5.8", "ms_per_token": "172",
            "t_start": "1010.0", "t_end": "1020.0", "quant": "Q4_K_M",
        }])
        result = analysis.join_run(run, self.power, sample_hz=28)
        self.assertEqual(result["joined"], 1)

        with open(result["out"], newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh))
        self.assertAlmostEqual(float(row["energy_j"]), 60.0, delta=1.0)   # 6 W x 10 s
        self.assertAlmostEqual(float(row["joules_per_token"]), 60.0 / 256, delta=0.01)

    def test_row_without_a_window_keeps_throughput_but_gets_no_energy(self):
        run = self.dir / "run2"
        run.mkdir()
        write_results(run, [{"node_id": "pi4", "model": "tiny", "phase": "tg", "n_gen": "128",
                             "n_threads": "4", "samples": "1", "tokens_per_s": "5.8"}])
        result = analysis.join_run(run, self.power)
        self.assertEqual(result["joined"], 0)
        with open(result["out"], newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh))
        self.assertEqual(row["energy_j"], "")
        self.assertEqual(row["tokens_per_s"], "5.8")
        self.assertIn("no-window", row["flags"])


class TestPowerSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_idle_windows_come_from_markers_not_from_guesswork(self):
        power = write_power(self.dir / "power.csv", start=1000.0, seconds=30.0, watts=3.0)
        events = self.dir / "events.jsonl"
        events.write_text("\n".join([
            json.dumps({"ts": 1005.0, "node": "pi4", "event": "idle_baseline_begin"}),
            json.dumps({"ts": 1015.0, "node": "pi4", "event": "idle_baseline_end"}),
        ]), encoding="utf-8")

        summary = analysis.node_power_summary(power, events_jsonl=events)
        self.assertAlmostEqual(summary["nodes"]["pi4"]["mean_power_w_idle"], 3.0, places=2)

    def test_missing_idle_marker_reports_none_and_says_why(self):
        power = write_power(self.dir / "power.csv", start=1000.0, seconds=10.0)
        summary = analysis.node_power_summary(power)
        entry = summary["nodes"]["pi4"]
        self.assertIsNone(entry["mean_power_w_idle"])
        self.assertIn("idle_baseline", entry["idle_note"])


class TestPareto(unittest.TestCase):
    def test_dominated_points_are_removed(self):
        points = [
            {"tokens_per_s": 2.0, "mean_power_w": 10.0},   # dominated by the third
            {"tokens_per_s": 1.0, "mean_power_w": 5.0},
            {"tokens_per_s": 3.0, "mean_power_w": 9.0},
        ]
        front = analysis.pareto_front(points)
        self.assertEqual(len(front), 2)
        self.assertTrue(all(not math.isnan(p["tokens_per_s"]) for p in front))


class TestCalibrationLoading(unittest.TestCase):
    """`Calibration.load_all` decides two things at once: what correction is applied to a
    channel's current, and whether that channel counts as calibrated at all. A file that
    should not have been loaded therefore both rescales the joules and disarms the flag
    that would have reported it. The shipped `ch2.example.json` did exactly this.
    """

    def _write(self, directory, name, **fields):
        payload = {"channel": 2, "gain": 0.994, "offset_a": 0.0012,
                   "calibrated_at": "2026-09-13T00:00:00Z"}
        payload.update(fields)
        (Path(directory) / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_an_example_file_is_not_loaded_as_a_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "ch2.example.json")
            self._write(d, "ch2.json.example")
            self.assertEqual(energy.Calibration.load_all(d), {},
                             "an example file must never count as a calibration")

    def test_a_real_channel_file_is_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "ch2.json")
            loaded = energy.Calibration.load_all(d)
            self.assertEqual(set(loaded), {2})
            self.assertAlmostEqual(loaded[2].gain, 0.994)

    def test_an_undated_calibration_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "ch2.json", calibrated_at="")
            with self.assertRaises(energy.DaqError):
                energy.Calibration.load_all(d)

    def test_the_repo_ships_no_file_that_load_all_would_pick_up(self):
        """Guards the actual `calibration/` directory, not a fixture. Nothing in it may be
        loadable until real points have been measured on real hardware."""
        shipped = Path(__file__).resolve().parent.parent / "calibration"
        self.assertEqual(
            energy.Calibration.load_all(shipped), {},
            "calibration/ must hold no usable calibration until the channels are measured")


class TestProfilesRefuseInventedLinks(unittest.TestCase):
    """`control/profiles.py` states its own contract in its docstring: "Anything that cannot
    be derived from a measurement is left as None and the loader says so, rather than
    substituting a plausible default. A placement computed from invented numbers would look
    exactly like one computed from measured numbers, which is why this refuses."

    Two of its fields broke that contract. `link_mbps=link.get("mbps") or 1000.0` and
    `rtt_ms=... or 0.3` were absent from the `needed` dict, so `strict=True` never fired for
    an unmeasured link and a node with no iperf3 result was profiled as 1 Gbit. On a shaped
    tier that understates the communication term -- the subject of the study's scope item 6 --
    by more than tenfold, with no flag. `or` also swallowed a genuine measured 0, which means
    "unreachable" rather than "fast".
    """

    def _cluster(self):
        from control import config
        return config.load(Path(__file__).resolve().parent.parent / "config" / "cluster.json")

    def _build(self, link_nodes, strict=True):
        from control import profiles
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "link_survey.json").write_text(json.dumps({"nodes": link_nodes}))
            return profiles.build_devices(self._cluster(), strict=strict)

    def test_an_unmeasured_link_is_refused_not_defaulted(self):
        from control import profiles
        with self.assertRaises(profiles.ProfileError) as cm:
            self._build({})
        message = str(cm.exception)
        self.assertIn("link_mbps", message)
        self.assertIn("rtt_ms", message)

    def test_a_zero_mbps_link_counts_as_unmeasured(self):
        # 0 Mbps means unreachable. `or` treated it as falsy and substituted 1 Gbit, which
        # turns "this node is off the network" into "this node has the fastest link".
        from control import profiles
        every = {n.id: {"mbps": 0.0, "rtt_ms": 0.3} for n in self._cluster().nodes}
        with self.assertRaises(profiles.ProfileError) as cm:
            self._build(every)
        self.assertIn("link_mbps", str(cm.exception))

    def test_the_module_does_not_default_the_link_inline(self):
        # Guards the shape of the fix, not just its effect: the fallback must live behind
        # the strict check, never in the DeviceProfile construction itself.
        import inspect
        from control import profiles
        source = inspect.getsource(profiles.build_devices)
        code = "\n".join(l for l in source.splitlines() if not l.lstrip().startswith("#"))
        self.assertNotIn('link.get("mbps") or', code)
        self.assertNotIn('link.get("rtt_ms") or', code)



class TestCharacterisationDispersion(unittest.TestCase):
    """The deliverable table's `sd` column used to be a fabricated 0.0 on every row.

    `characterisation_table` computed `pstdev(tps) if len(tps) > 1 else 0.0`, and
    `len(tps) == 1` is the NORMAL case: `Pi4/node/bench_parse.py` emits one row per
    (model, phase, n, threads) with the repetitions already collapsed into avg_ts/stddev_ts.
    So the published table claimed perfect repeatability for every measurement while
    llama-bench's own dispersion sat unread in the `tokens_per_s_stddev` column of the same
    CSV. A fabricated 0 is the worst possible wrong value here, because it reads as a
    strong result rather than a missing one.
    """

    COLS = ["node_id", "model", "quant", "phase", "n_threads", "tokens_per_s",
            "tokens_per_s_stddev", "joules_per_token", "mean_power_w", "flags"]

    def _table(self, rows):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "e.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=self.COLS)
                w.writeheader()
                w.writerows(rows)
            return analysis.characterisation_table(path)

    def _row(self, **kw):
        base = dict(node_id="pi4", model="m", quant="q4_k_m", phase="tg", n_threads="4",
                    tokens_per_s="6.4", tokens_per_s_stddev="0.42",
                    joules_per_token="0.9375", mean_power_w="6.0", flags="")
        base.update(kw)
        return base

    def test_a_single_run_reports_its_measured_spread_not_zero(self):
        table = self._table([self._row()])
        self.assertEqual(len(table), 1)
        self.assertEqual(table[0]["runs"], 1)
        self.assertEqual(table[0]["tokens_per_s_sd"], 0.42)

    def test_sd_is_never_a_fabricated_zero(self):
        # No measured stddev available either -> unknown, which must read as blank, not 0.
        table = self._table([self._row(tokens_per_s_stddev="")])
        self.assertIsNone(table[0]["tokens_per_s_sd"])

    def test_several_runs_use_the_across_run_sample_spread(self):
        rows = [self._row(tokens_per_s="6.0"), self._row(tokens_per_s="7.0")]
        table = self._table(rows)
        self.assertEqual(table[0]["runs"], 2)
        # sample stdev of (6.0, 7.0) is 0.707; the population figure would be 0.5.
        self.assertAlmostEqual(table[0]["tokens_per_s_sd"], 0.707, places=3)



class TestShortWindowEnergy(unittest.TestCase):
    """`integrate` used to sum only between the FIRST and LAST in-window sample.

    That silently drops up to one sample period of energy. At 28 Hz the deficit is
    period/duration: -0.18 % over a 20 s decode window, but -16.7 % worst case over a
    0.30 s prefill one. It is always one-sided, and sampling quantisation can never trip
    `late-start`/`early-end`, whose tolerance is two whole periods -- so the row ships
    unflagged and goes straight into the characterisation table. It also put pp and tg on
    different scales, which makes a phase-separated J/tok comparison meaningless.
    """

    HZ = 28.0

    def _samples(self, watts=7.0, seconds=60.0, start=1000.0):
        n = int(seconds * self.HZ)
        return [(start + i / self.HZ, 0.0, 0.0, watts) for i in range(n)]

    def _worst(self, samples, win):
        """Worst sub-period alignment, which is where the bias was largest."""
        best = None
        for k in range(20):
            t0 = 1005.0 + k * (1.0 / self.HZ) / 20.0
            r = analysis.integrate(samples, t0, t0 + win, sample_hz=self.HZ)
            if r["energy_j"] is None:
                continue
            if best is None or r["energy_j"] < best["energy_j"]:
                best = r
        return best

    def test_a_short_prefill_window_keeps_the_energy_at_its_edges(self):
        worst = self._worst(self._samples(), 0.30)
        self.assertEqual(worst["flags"], [],
                         "quantisation is not a flag condition, so the number must stand alone")
        self.assertAlmostEqual(worst["energy_j"], 7.0 * 0.30, delta=0.02)

    def test_energy_and_mean_power_describe_the_same_window(self):
        # The emitted row has to satisfy its own identity, or a reader cannot reconcile it.
        for win in (0.30, 1.0, 20.0):
            with self.subTest(window=win):
                r = self._worst(self._samples(), win)
                self.assertAlmostEqual(r["energy_j"], r["mean_power_w"] * r["duration_s"],
                                       delta=0.02)

    def test_prefill_and_decode_windows_share_a_scale(self):
        # Implied mean power must not depend on how long the window is.
        samples = self._samples()
        short = self._worst(samples, 0.30)
        long_ = self._worst(samples, 20.0)
        self.assertAlmostEqual(short["energy_j"] / 0.30, long_["energy_j"] / 20.0, delta=0.05)



class TestPerLayerCostThreadPoint(unittest.TestCase):
    """c_i has to be taken at the thread count the cluster run actually uses.

    `docs/05_experiment_protocol.md` sweeps 1, 2 and 4 threads and feeds that same CSV to
    `heteroctl profiles`, while the cluster run uses llama.cpp's default (all cores). A
    median across the sweep therefore returns roughly the 2-thread cost -- 1.35-1.95x the
    operating cost depending on the node -- and it lands unflagged in predicted
    tokens_per_s and joules_per_token, which are named deliverables.
    `docs/06_troubleshooting.md` already names this as the first thing to check when
    predictions miss by a consistent margin; the code used to contradict that advice, and
    `per_layer_times` had a working `threads=` filter that no caller ever passed.
    """

    COLS = ["node_id", "model", "phase", "n_threads", "ms_per_token", "flags"]

    def _csv(self, d, scaling=((1, 3.6), (2, 1.9), (4, 1.0))):
        path = Path(d) / "r.csv"
        rows = [dict(node_id="pi4", model="m", phase="tg", n_threads=str(t),
                     ms_per_token="%.4f" % (16.7857 * s * 22), flags="")
                for t, s in scaling]
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.COLS)
            w.writeheader()
            w.writerows(rows)
        return path

    def _table_csv(self, d, rows):
        path = Path(d) / "results_energy.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return path

    def test_the_sd_column_says_which_dispersion_it_is(self):
        """The two are not comparable and the nodes do not produce the same one: the Pi 5
        wraps every cell in an outer repeat loop so its groups hold several rows (across-run
        spread), while the Pi 4 runs each shape once so its tg groups hold one row and the
        figure is llama-bench's within-run stddev. Printed unlabelled in adjacent rows of one
        table, a reader compares a run-to-run spread against a within-run one."""
        from control import analysis
        base = dict(model="m", quant="Q4_K_M", phase="tg", n_threads="4", flags="",
                    joules_per_token="", mean_power_w="")
        rows = [dict(base, node_id="pi5", tokens_per_s="2.30", tokens_per_s_stddev="0.01"),
                dict(base, node_id="pi5", tokens_per_s="2.40", tokens_per_s_stddev="0.01"),
                dict(base, node_id="pi4", tokens_per_s="1.10", tokens_per_s_stddev="0.05")]
        with tempfile.TemporaryDirectory() as d:
            table = {r["node"]: r for r in
                     analysis.characterisation_table(self._table_csv(d, rows))}
        self.assertEqual(table["pi5"]["sd_basis"], "across-runs")
        self.assertEqual(table["pi4"]["sd_basis"], "within-run")
        self.assertNotEqual(table["pi5"]["tokens_per_s_sd"], table["pi4"]["tokens_per_s_sd"])

    def test_no_dispersion_at_all_is_labelled_too(self):
        from control import analysis
        rows = [dict(node_id="pi4", model="m", quant="Q4_K_M", phase="tg", n_threads="4",
                     flags="", tokens_per_s="1.10", tokens_per_s_stddev="",
                     joules_per_token="", mean_power_w="")]
        with tempfile.TemporaryDirectory() as d:
            row, = analysis.characterisation_table(self._table_csv(d, rows))
        self.assertIsNone(row["tokens_per_s_sd"])
        self.assertIsNone(row["sd_basis"])

    def test_a_model_missing_from_the_layers_map_is_named(self):
        """The --layers map is hand-written and nothing derives it from the nodes'
        models.txt or GGUF headers, so a model the sweep measured but the map does not name
        is entirely plausible. It used to be skipped with no counter and no message: a map
        written for one node's models silently discarded half of another node's cells, the
        profile still built, and nothing looked wrong."""
        import io as _io
        from contextlib import redirect_stderr
        from control import profiles
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.csv"
            rows = [dict(node_id="pi5", model="qwen2.5-7b-q4km", phase="tg", n_threads="4",
                         ms_per_token="425.5", flags=""),
                    dict(node_id="pi5", model="tinyllama-1.1b-q4km", phase="tg",
                         n_threads="4", ms_per_token="90.0", flags="")]
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=self.COLS)
                w.writeheader()
                w.writerows(rows)
            err = _io.StringIO()
            with redirect_stderr(err):
                t = profiles.per_layer_times(path, {"tinyllama-1.1b-q4km": 22}, threads=4)
            self.assertIn("qwen2.5-7b-q4km", err.getvalue())
            self.assertIn("pi5", t)          # the mapped model still contributes

    def test_link_survey_keeps_the_conditions(self):
        """A figure taken on the 2.5GbE tier, or across a retransmitting link, used to enter
        the placement rule indistinguishable from a clean one -- the survey records both and
        this projected them away one line after reading them."""
        from control import profiles
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "link_survey.json"
            path.write_text(json.dumps({"nodes": {"pi5": {
                "mbps": 2190.0, "rtt_ms": 0.21, "tier": "2.5gbe", "iface": "eth1",
                "measured_utc": "2026-09-14T01:00:00+00:00",
                "warnings": ["900 TCP retransmits"]}}}), encoding="utf-8")
            got = profiles.link_survey(path)["pi5"]
            self.assertEqual(got["mbps"], 2190.0)
            self.assertEqual(got["tier"], "2.5gbe")
            self.assertEqual(got["warnings"], ["900 TCP retransmits"])

    def _flagged_csv(self, d, flags):
        path = Path(d) / "f.csv"
        rows = [dict(node_id="pi5", model="m", phase="tg", n_threads="4",
                     ms_per_token="369.28", flags=f) for f in flags]
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.COLS)
            w.writeheader()
            w.writerows(rows)
        return path

    def test_power_trace_flags_do_not_remove_a_row_from_the_placement_rule(self):
        """Which flags invalidate WHICH column is not one question. `uncalibrated-channel`,
        `gap:` and friends say the ENERGY join is untrustworthy; the tokens_per_s from that
        same run is a perfectly good benchmark. characterisation_table has always drawn that
        line, and per_layer_times did not -- so the report table and Rule A disagreed about
        which runs counted.

        The practical effect: with no calibration file on disk, join_run stamps
        `uncalibrated-channel` on every joined row of every node, so every row vanished from
        the placement rule and build_devices reported "unmeasured inputs" for nodes whose
        throughput data was complete."""
        from control import profiles
        with tempfile.TemporaryDirectory() as d:
            t = profiles.per_layer_times(
                self._flagged_csv(d, ["uncalibrated-channel",
                                      "gap:1.0s",
                                      "no-window:energy-join-impossible"]),
                {"m": 22})
            self.assertIn("pi5", t, "power-trace flags removed every row from Rule A")
            self.assertEqual(t["_dropped_flagged_rows"], 0)
            self.assertEqual(t["pi5"]["samples"]["tg"], 3)

    def test_throttle_flags_still_remove_the_row(self):
        """A throttle flag means the clock moved underneath the benchmark, so tokens_per_s
        is not a measurement of this device at its operating point. That one must still
        exclude -- and `throttle-unknown` counts, because the node-side parser already
        resolves an unreadable word as throttled and this must not be more lenient."""
        from control import profiles
        with tempfile.TemporaryDirectory() as d:
            t = profiles.per_layer_times(
                self._flagged_csv(d, ["throttled:0x5", "throttle-unknown",
                                      "gap:1.0s;throttled-at-start:0x5"]),
                {"m": 22})
            self.assertNotIn("pi5", t)
            self.assertEqual(t["_dropped_flagged_rows"], 3)

    def test_a_mixed_thread_csv_is_refused_not_pooled(self):
        from control import profiles
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(profiles.ProfileError) as cm:
                profiles.per_layer_times(self._csv(d), {"m": 22})
            self.assertIn("thread", str(cm.exception).lower())

    def test_the_operating_thread_count_selects_that_row(self):
        from control import profiles
        with tempfile.TemporaryDirectory() as d:
            t = profiles.per_layer_times(self._csv(d), {"m": 22}, threads=4)
            self.assertAlmostEqual(t["pi4"]["decode"], 16.7857, places=3)
            # emphatically not the pooled median, which is the 2-thread row
            self.assertNotAlmostEqual(t["pi4"]["decode"], 31.8928, places=1)

    def test_the_operating_point_is_recorded(self):
        # A per-layer cost is meaningless without the thread count it was measured at.
        from control import profiles
        with tempfile.TemporaryDirectory() as d:
            t = profiles.per_layer_times(self._csv(d), {"m": 22}, threads=4)
            self.assertEqual(t["pi4"]["n_threads"], [4])

    def test_a_single_thread_csv_needs_no_choice(self):
        # Only one operating point present -> nothing to disambiguate, so it must not raise.
        from control import profiles
        with tempfile.TemporaryDirectory() as d:
            t = profiles.per_layer_times(self._csv(d, scaling=((4, 1.0),)), {"m": 22})
            self.assertAlmostEqual(t["pi4"]["decode"], 16.7857, places=3)

    def test_build_devices_forwards_the_thread_count(self):
        # The regression that shipped: the filter existed and no caller passed it.
        import inspect
        from control import profiles
        self.assertIn("threads", inspect.signature(profiles.build_devices).parameters)
        self.assertIn("threads=threads", inspect.getsource(profiles.build_devices))



class TestActivePowerIsDecodeNotTheWholeRun(unittest.TestCase):
    """P_act is multiplied by DECODE compute time in the energy model, so it has to be a
    decode-phase mean.

    The only begin/end pair anything emits is `run_begin`/`run_end` around the whole
    llama-cli process, so deriving the active window from events averaged model load and
    prefill into P_act: 4.80 W where decode is 7.00 W, collapsing delta_power_w from 4.00
    to 1.80 -- 55 % low, unflagged, straight into a saved deliverable.
    `harness._append_window` already records the split and its docstring warns about this
    exact mistake; `analysis` simply never read it.
    """

    HZ = 28.0
    PHASES = ((40.0, 3.4), (5.0, 7.2), (20.0, 7.0))   # load, prefill, decode

    def _trace(self, d):
        path = Path(d) / "power.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t_utc", "node", "power_w"])
            t = 1000.0
            for secs, watts in self.PHASES:
                for i in range(int(secs * self.HZ)):
                    w.writerow(["%.6f" % (t + i / self.HZ), "pi4", watts])
                t += secs
        events = Path(d) / "events.jsonl"
        events.write_text(
            json.dumps({"ts": 1000.0, "node": "pi4", "event": "run_begin"}) + "\n" +
            json.dumps({"ts": 1065.0, "node": "pi4", "event": "run_end"}) + "\n",
            encoding="utf-8")
        windows = Path(d) / "windows.jsonl"
        windows.write_text(json.dumps({
            "run": "r1", "node": "pi4", "t_start": 1000.0, "t_end": 1065.0, "phase": "full",
            "phases": [{"phase": "load", "t_start": 1000.0, "t_end": 1040.0},
                       {"phase": "prefill", "t_start": 1040.0, "t_end": 1045.0},
                       {"phase": "decode", "t_start": 1045.0, "t_end": 1065.0}],
        }) + "\n", encoding="utf-8")
        return path, events, windows

    def test_the_decode_split_gives_the_decode_power(self):
        with tempfile.TemporaryDirectory() as d:
            power, events, windows = self._trace(d)
            entry = analysis.node_power_summary(
                power, events_jsonl=events, phase_windows_jsonl=windows)["nodes"]["pi4"]
            self.assertAlmostEqual(entry["mean_power_w_active"], 7.0, places=2)

    def test_a_whole_run_window_never_becomes_active_power(self):
        with tempfile.TemporaryDirectory() as d:
            power, events, _ = self._trace(d)
            entry = analysis.node_power_summary(power, events_jsonl=events)["nodes"]["pi4"]
            self.assertIsNone(entry["mean_power_w_active"],
                              "a load-contaminated mean must never be published as P_act")
            # the figure is still reported, just not as the thing the rule consumes
            self.assertAlmostEqual(entry["mean_power_w_run"], 4.8, places=1)
            self.assertIn("decode", entry["active_note"])

    def test_a_run_with_no_recorded_phases_contributes_nothing(self):
        # A failed run records no phases; it must not become a guessed window.
        with tempfile.TemporaryDirectory() as d:
            power, events, _ = self._trace(d)
            empty = Path(d) / "empty.jsonl"
            empty.write_text(json.dumps(
                {"run": "r2", "node": "pi4", "phases": []}) + "\n", encoding="utf-8")
            self.assertEqual(analysis.windows_from_phase_windows(empty), {})



class TestFlaggedRowsExcludedFromEnergyMeans(unittest.TestCase):
    """A flagged row's energy is not a measurement, so it must not reach the headline means.

    `characterisation_table` used to average every row regardless of `flags`, while
    `report.characterisation` printed underneath the same table that flagged runs "were
    excluded from the energy means". They were not: two runs truly at 0.9375 J/tok, one
    with the DAQ down for its first 12 s, published 0.6552 J/tok -- 30 % low -- next to a
    `mean_power_w` that could not be reconciled with it. The table and the prose now agree.

    Throughput is deliberately still unfiltered: `gap:`, `late-start:`, `early-end:`,
    `too-few-samples:` and `uncalibrated-channel` all concern the power trace, not the
    benchmark. `throttled:1` is the exception, and filtering on it needs a flag taxonomy
    rather than a blanket rule.
    """

    COLS = ["node_id", "model", "quant", "phase", "n_threads", "tokens_per_s",
            "tokens_per_s_stddev", "joules_per_token", "mean_power_w", "flags"]

    def _row(self, **kw):
        base = dict(node_id="pi4", model="m", quant="q4_k_m", phase="tg", n_threads="4",
                    tokens_per_s="6.4", tokens_per_s_stddev="0.42",
                    joules_per_token="0.9375", mean_power_w="6.0", flags="")
        base.update(kw)
        return base

    def _table(self, rows):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "e.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=self.COLS)
                w.writeheader()
                w.writerows(rows)
            return analysis.characterisation_table(path)

    def test_a_flagged_run_does_not_move_the_energy_mean(self):
        rows = [self._row(),
                self._row(joules_per_token="0.3733", flags="late-start:12.036s")]
        row = self._table(rows)[0]
        self.assertAlmostEqual(row["joules_per_token"], 0.9375, places=4)
        self.assertEqual(row["runs"], 2)
        self.assertEqual(row["energy_runs"], 1)
        self.assertEqual(row["flagged"], 1)

    def test_every_flag_kind_excludes(self):
        for flag in ("gap:7.0s", "late-start:1.0s", "early-end:1.0s",
                     "too-few-samples:3", "uncalibrated-channel"):
            with self.subTest(flag=flag):
                rows = [self._row(), self._row(joules_per_token="99.0", flags=flag)]
                self.assertAlmostEqual(self._table(rows)[0]["joules_per_token"], 0.9375,
                                       places=4)

    def test_all_runs_flagged_yields_no_energy_rather_than_a_number(self):
        rows = [self._row(flags="gap:7.0s"), self._row(flags="uncalibrated-channel")]
        row = self._table(rows)[0]
        self.assertIsNone(row["joules_per_token"])
        self.assertIsNone(row["mean_power_w"])
        self.assertIsNone(row["wh_per_1k_tokens"])
        self.assertEqual(row["energy_runs"], 0)

    def test_power_trace_flags_do_not_filter_throughput(self):
        # A power-trace flag says nothing about tokens_per_s; dropping it would discard a
        # perfectly good benchmark because the DAQ hiccuped. (Throttle flags DO filter it --
        # see TestThrottledExcludedFromThroughput.)
        rows = [self._row(tokens_per_s="6.0"),
                self._row(tokens_per_s="8.0", flags="uncalibrated-channel")]
        self.assertAlmostEqual(self._table(rows)[0]["tokens_per_s"], 7.0, places=3)

    def test_the_report_prose_matches_the_table(self):
        from control import report
        rows = [self._row(), self._row(joules_per_token="0.3733", flags="late-start:12.0s")]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "e.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=self.COLS)
                w.writeheader()
                w.writerows(rows)
            text = report.characterisation(path)
        self.assertIn("excluded from the energy means", text)
        self.assertIn("E-runs", text)
        # the published figure is the unflagged run's, not the contaminated blend
        self.assertIn("0.9375", text)
        self.assertNotIn("0.6552", text)



class TestThrottledExcludedFromThroughput(unittest.TestCase):
    """A throttled run is not a measurement of the device at its operating point.

    The Raspberry Pi throttle word is the single most important validity signal on these
    boards: any non-zero value means the clock moved underneath the benchmark, so its
    tokens_per_s describes a machine that was not running at the speed being characterised.
    Energy flags and throughput flags are therefore different questions, and this is the
    asymmetry: a `gap:` in the power trace invalidates the joules but not the benchmark,
    while a throttle invalidates the benchmark (and, being blanket, the joules too).
    """

    COLS = ["node_id", "model", "quant", "phase", "n_threads", "tokens_per_s",
            "tokens_per_s_stddev", "joules_per_token", "mean_power_w", "flags"]

    def _row(self, **kw):
        base = dict(node_id="pi4", model="m", quant="q4_k_m", phase="tg", n_threads="4",
                    tokens_per_s="6.0", tokens_per_s_stddev="0.42",
                    joules_per_token="0.9375", mean_power_w="6.0", flags="")
        base.update(kw)
        return base

    def _table(self, rows):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "e.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=self.COLS)
                w.writeheader()
                w.writerows(rows)
            return analysis.characterisation_table(path)

    def test_every_throttle_spelling_excludes_throughput(self):
        # All three are emitted by the node-side parser; `throttle-unknown` counts because
        # bench_parse.py already resolves an unreadable throttle word as throttled, and
        # this must not be more lenient than the thing producing the data.
        for flag in ("throttled:0x5", "throttled-at-start:0x50000", "throttle-unknown"):
            with self.subTest(flag=flag):
                rows = [self._row(tokens_per_s="6.0"),
                        self._row(tokens_per_s="8.0", flags=flag)]
                row = self._table(rows)[0]
                self.assertAlmostEqual(row["tokens_per_s"], 6.0, places=3)
                self.assertEqual(row["throughput_runs"], 1)

    def test_a_power_trace_flag_leaves_throughput_alone(self):
        for flag in ("gap:7.0s", "late-start:1.0s", "early-end:1.0s",
                     "too-few-samples:3", "uncalibrated-channel", "no-power-data"):
            with self.subTest(flag=flag):
                rows = [self._row(tokens_per_s="6.0"),
                        self._row(tokens_per_s="8.0", flags=flag)]
                row = self._table(rows)[0]
                self.assertAlmostEqual(row["tokens_per_s"], 7.0, places=3)
                self.assertEqual(row["throughput_runs"], 2)

    def test_a_throttled_run_does_not_contribute_its_dispersion_either(self):
        rows = [self._row(tokens_per_s="6.0", tokens_per_s_stddev="0.42"),
                self._row(tokens_per_s="8.0", tokens_per_s_stddev="9.99",
                          flags="throttled:0x5")]
        self.assertEqual(self._table(rows)[0]["tokens_per_s_sd"], 0.42)

    def test_all_runs_throttled_yields_no_throughput(self):
        rows = [self._row(flags="throttled:0x5"), self._row(flags="throttled:0x5")]
        row = self._table(rows)[0]
        self.assertIsNone(row["tokens_per_s"])
        self.assertEqual(row["throughput_runs"], 0)

    def test_the_report_names_a_fully_throttled_row(self):
        from control import report
        rows = [self._row(flags="throttled:0x5"), self._row(flags="throttled:0x5")]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "e.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=self.COLS)
                w.writeheader()
                w.writerows(rows)
            text = report.characterisation(path)
        self.assertIn("throttled on every run", text)
        self.assertIn("T-runs", text)

    def test_the_predicate_is_exported_and_precise(self):
        self.assertTrue(analysis.invalidates_throughput("gap:1.0s;throttled:0x5"))
        self.assertFalse(analysis.invalidates_throughput("gap:1.0s;uncalibrated-channel"))
        self.assertFalse(analysis.invalidates_throughput(""))
        self.assertFalse(analysis.invalidates_throughput(None))

    def test_the_predicate_matches_whole_flags_not_substrings(self):
        """A substring test would exclude on any flag merely CONTAINING "throttled".

        These two are the cases that separate a prefix match from a substring match, and
        without them the previous test passes under either implementation -- it asserted
        precision it did not actually check.
        """
        self.assertFalse(analysis.invalidates_throughput("unthrottled"),
                         "a flag that merely contains 'throttled' must not exclude")
        self.assertFalse(analysis.invalidates_throughput("gap:1.0s;not-throttled"))
        # ...while a real throttle flag anywhere in the list still does
        self.assertTrue(analysis.invalidates_throughput("unthrottled;throttled:0x5"))



if __name__ == "__main__":
    unittest.main()


from control import profiles  # noqa: E402 -- the per-byte tests below use it directly


class TestPerBytePricing(unittest.TestCase):
    """heteroctl priced every model's layers as the profiled TinyLlama's, so its prediction for Qwen3-14B was 85 % too
    fast (2026-09-29). A profile now carries unit costs -- decode per byte read, prefill per block parameter, the
    testbed predictor's convention -- and profiles.for_model prices the model being planned."""

    COLS = ["node_id", "phase", "model", "n_threads", "ms_per_token", "flags"]

    def _csv(self, d, rows):
        path = Path(d) / "r.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.COLS)
            w.writeheader()
            w.writerows([dict(node_id=n, model="m", phase=p, n_threads="4", ms_per_token=str(ms), flags="")
                         for n, p, ms in rows])
        return path

    def _device(self, **kw):
        from control.placement import DeviceProfile
        base = dict(node_id="pi5", ms_per_layer_decode=2.27, ms_per_layer_prefill=0.45, usable_mem_mb=7000.0)
        base.update(kw)
        return DeviceProfile(**base)

    def _model(self, **kw):
        from control.placement import ModelProfile
        base = dict(name="big", n_layers=40, bytes_per_layer=2e8, kv_bytes_per_layer_per_token=0.0,
                    activation_bytes=0.0, params_per_layer=3e8)
        base.update(kw)
        return ModelProfile(**base)

    def test_unit_costs_come_from_the_profiled_model_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            t = profiles.per_layer_times(self._csv(d, [("pi5", "tg", 50.0), ("pi5", "pp", 10.0)]), {"m": 22},
                                         threads=4, geometry={"m": (1e9, 2e9)})
        # relative: the unit costs are ~1e-8, far below assertAlmostEqual's absolute 1e-7
        self.assertAlmostEqual(t["pi5"]["decode_per_byte"] / (50.0 / 1e9), 1.0)
        self.assertAlmostEqual(t["pi5"]["prefill_per_param"] / (10.0 / 2e9), 1.0)
        self.assertAlmostEqual(t["pi5"]["decode"], 50.0 / 22, places=4)   # the per-layer figure (rounded) is still there

    def test_for_model_scales_with_the_model_being_planned(self):
        d = {"pi5": self._device(ms_per_byte_decode=5e-8, ms_per_param_prefill=5e-9)}
        got = profiles.for_model(d, self._model())["pi5"]
        self.assertAlmostEqual(got.ms_per_layer_decode, 10.0)         # 5e-8 ms/B x 2e8 B
        self.assertAlmostEqual(got.ms_per_layer_prefill, 1.5)         # 5e-9 ms/param x 3e8
        twice = profiles.for_model(d, self._model(bytes_per_layer=4e8))["pi5"]
        self.assertAlmostEqual(twice.ms_per_layer_decode, 2 * got.ms_per_layer_decode)
        moe = profiles.for_model(d, self._model(active_bytes_per_layer=5e7, active_params_per_layer=6e7))["pi5"]
        self.assertAlmostEqual(moe.ms_per_layer_decode, 2.5)           # an MoE reads its active experts
        self.assertAlmostEqual(moe.ms_per_layer_prefill, 0.3)

    def test_an_older_profile_without_unit_costs_comes_back_unchanged(self):
        got = profiles.for_model({"pi5": self._device()}, self._model())["pi5"]
        self.assertEqual((got.ms_per_layer_decode, got.ms_per_layer_prefill), (2.27, 0.45))

    def test_the_profiled_model_reproduces_its_own_ms_per_token_through_the_engine(self):
        """Per byte, the head is priced by the engine (placement._head_layers) rather than smeared over the layers,
        and the profiled model must come back at exactly the ms/token it was measured at."""
        from control.placement import evaluate
        bpl, head, n, ms_tok = 26e6, 54e6, 22, 45.0
        per_byte = ms_tok / (n * bpl + head)
        tiny = self._model(name="tiny", n_layers=n, bytes_per_layer=bpl, head_bytes=head, params_per_layer=44e6)
        devs = profiles.for_model({"pi5": self._device(ms_per_byte_decode=per_byte, ms_per_param_prefill=1e-9)}, tiny)
        self.assertAlmostEqual(evaluate(["pi5"], {"pi5": n}, devs, tiny)["ms_per_token"], ms_tok, places=2)

    def test_save_and_load_keep_the_unit_costs(self):
        d = {"pi5": self._device(ms_per_byte_decode=5e-8, ms_per_param_prefill=5e-9, profiled_on="m")}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            profiles.save_devices(d, path)
            back = profiles.load_devices(path)["pi5"]
        self.assertEqual((back.ms_per_byte_decode, back.ms_per_param_prefill, back.profiled_on), (5e-8, 5e-9, "m"))
