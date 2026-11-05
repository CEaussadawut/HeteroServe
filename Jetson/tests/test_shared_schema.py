#!/usr/bin/env python3
"""Guard bench/emit_results_csv.py against drift in the cluster's shared schema.

    python3 -m unittest discover -s tests -v

The schema lives in `Pi4/node/bench_parse.py` and this node copies three things
out of it: the 40-column list, QUANT_HINTS, and the flag vocabulary. A COPY that
drifts is worse than no copy at all, because the control node groups rows on
(node_id, model, quant, phase, n_threads) and a mismatch there does not error --
it silently produces two half-populated report rows per cell, or drops this
node's rows from a group they belong to.

So these tests read Pi 4's file LIVE rather than restating what it said when this
was written. `Pi5/tests/test_pi4_schema.py` does the same and has already caught
two real drifts (the idle_start/idle_end addition, and two new flag tokens).

If Pi 4 is not present -- someone checked out this folder alone -- the drift
tests skip rather than fail. A green run then means less than it looks like, and
says so.
"""
import csv
import io
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
JETSON = os.path.dirname(HERE)
CLUSTER = os.path.dirname(JETSON)
PI4 = os.path.join(CLUSTER, "Pi4", "node", "bench_parse.py")

sys.path.insert(0, os.path.join(JETSON, "bench"))
import emit_results_csv as E  # noqa: E402


def pi4_source():
    if not os.path.exists(PI4):
        return None
    with io.open(PI4, encoding="utf-8") as fh:
        return fh.read()


def _list_literal(src, name):
    """The string literals of a top-level `NAME = [ ... ]` block."""
    m = re.search(r"^%s = \[(.*?)\n\]" % name, src, re.S | re.M)
    if not m:
        return None
    return re.findall(r'"([^"]+)"', m.group(1))


class SharedSchemaDrift(unittest.TestCase):
    """Every assertion here reads Pi 4's file as it is on disk right now."""

    def setUp(self):
        self.src = pi4_source()
        if self.src is None:
            self.skipTest("Pi4/node/bench_parse.py not present; drift cannot be checked")

    def test_columns_match_pi4_exactly_including_order(self):
        # Order matters: the control node's join writes over this file's own
        # fieldnames, so a reordered header silently mislabels every value.
        pi4 = _list_literal(self.src, "COLUMNS")
        self.assertIsNotNone(pi4, "Pi4 COLUMNS block not found -- the extraction broke")
        self.assertEqual(pi4, E.COLUMNS)

    def test_quant_hints_match_pi4_exactly(self):
        # `quant` is part of the group key. One spelling canonicalised differently
        # splits a model into two report rows with no error anywhere.
        pi4 = re.search(r"^QUANT_HINTS = \[(.*?)\n\]", self.src, re.S | re.M)
        self.assertIsNotNone(pi4)
        pairs = re.findall(r'\("([^"]+)",\s*\[([^\]]*)\]\)', pi4.group(1))
        expected = [(c, re.findall(r'"([^"]+)"', sp)) for c, sp in pairs]
        self.assertEqual(expected, E.QUANT_HINTS)

    def test_every_pi4_flag_is_raised_here_or_accounted_for(self):
        """Pi 4's flag vocabulary cannot grow without this node deciding about it.

        `flags` is load-bearing: a non-empty cell is an opt-out from the placement
        rule. If Pi 4 starts raising a token on a basis this node also has and this
        node stays silent, the combined table excludes the two nodes' rows on
        different criteria -- which is a bias, not a gap.
        """
        pi4_flags = set()
        for m in re.finditer(r'flags\.append\(\s*"([^"]+)"', self.src):
            pi4_flags.add(m.group(1).split(":%")[0].split(":")[0]
                          if "%" in m.group(1) else m.group(1))
        # normalise: a token is its text up to the first format placeholder
        norm = {re.split(r":?%", f)[0].rstrip(":") for f in pi4_flags}

        raised_here = set()
        with io.open(os.path.join(JETSON, "bench", "emit_results_csv.py"),
                     encoding="utf-8") as fh:
            mine = fh.read()
        for m in re.finditer(r'flags\.append\(\s*"([^"]+)"', mine):
            raised_here.add(re.split(r":?%", m.group(1))[0].rstrip(":"))
        accounted = {re.split(r":?%", f)[0].rstrip(":")
                     for f in E.PI4_FLAGS_NOT_RAISED_HERE}

        unhandled = norm - raised_here - accounted
        self.assertEqual(
            unhandled, set(),
            "Pi 4 raises flag(s) this node neither raises nor explains: %s. Add each "
            "to PI4_FLAGS_NOT_RAISED_HERE with the reason, or raise it." % sorted(unhandled))

    def test_accounted_flags_are_really_pi4s(self):
        """The other direction: an entry explaining a flag Pi 4 no longer raises is
        stale documentation, and stale documentation about a load-bearing column is
        how the last schema divergence survived so long."""
        norm = set()
        for m in re.finditer(r'flags\.append\(\s*"([^"]+)"', self.src):
            norm.add(re.split(r":?%", m.group(1))[0].rstrip(":"))
        stale = {f for f in E.PI4_FLAGS_NOT_RAISED_HERE} - norm
        self.assertEqual(stale, set(),
                         "PI4_FLAGS_NOT_RAISED_HERE explains flag(s) Pi 4 does not "
                         "raise any more: %s" % sorted(stale))


class Projection(unittest.TestCase):
    """What one joined-CSV record becomes. No Pi 4 needed."""

    def rec(self, **over):
        base = {
            "node_id": "jetson-nano-01", "node_role": "cuda-worker",
            # the cluster-canonical id, which is what the driver carries and what
            # every consumer groups on -- NODE_ID above is the local spelling
            "cluster_node_id": "jetson",
            "model": "qwen2.5-1.5b-instruct-q4_k_m",
            # the driver writes BOTH: `phase` is what it asked for, `row_phase`
            # what the ref reported. A failed or skipped run has only the former.
            "phase": "prefill", "row_phase": "prefill",
            "bench_n_prompt": "512", "bench_n_gen": "0",
            "n_prompt": "512", "n_gen": "0",
            "threads": "4", "n_gpu_layers": "99", "repetitions": "3",
            "avg_ts": "1024.0", "avg_ns": "500000000", "stddev_ts": "0.5",
            "stddev_ns": "41000000", "build_commit": "a1b2c3d",
            "bench_model_type": "qwen2 1.5B Q4_K_M", "bench_model_size": "397000000",
            "bench_n_params": "494032768", "bench_n_batch": "2048",
            "status": "ok", "per_token_status": "ok",
            "temp_end_c": "46.2", "t_start_unix": "100.0", "t_end_unix": "105.0",
        }
        base.update(over)
        return base

    def test_prefill_becomes_pp_and_decode_becomes_tg(self):
        # Not "prefill"/"decode": every group-by on the control node keys on the
        # llama-bench spelling, and a row with the wrong one groups with nothing.
        self.assertEqual(E.row_to_shared(self.rec(), "r")["phase"], "pp")
        self.assertEqual(
            E.row_to_shared(self.rec(row_phase="decode", bench_n_prompt="0",
                                     bench_n_gen="128"), "r")["phase"], "tg")

    def test_a_combined_row_is_omitted_not_mislabelled(self):
        # The shared schema has no phase value meaning "both". Inventing one would
        # make the row group with nothing while still being counted.
        self.assertIsNone(E.row_to_shared(self.rec(row_phase="combined",
                                                   bench_n_gen="128"), "r"))

    def test_samples_is_the_repetition_count(self):
        # analysis.py: reps = int(row["samples"]); tokens = n * reps. Anything else
        # here scales the project's headline metric by that factor, silently.
        self.assertEqual(E.row_to_shared(self.rec(), "r")["samples"], 3)

    def test_ms_per_token_is_the_inverse_of_this_rows_throughput(self):
        r = E.row_to_shared(self.rec(), "r")
        self.assertAlmostEqual(r["ms_per_token"], 1000.0 / float(r["tokens_per_s"]), places=3)

    def test_avg_ms_is_the_whole_repetition_not_per_token(self):
        self.assertEqual(E.row_to_shared(self.rec(), "r")["avg_ms"], 500.0)

    def test_quant_is_canonical_from_the_nodes_own_short_name(self):
        self.assertEqual(E.row_to_shared(self.rec(), "r")["quant"], "Q4_K_M")
        self.assertEqual(
            E.row_to_shared(self.rec(model="qwen2.5-0.5b-instruct-q8_0",
                                     bench_model_type="qwen2 0.5B Q8_0"), "r")["quant"],
            "Q8_0")

    def test_model_is_the_short_name_on_every_row_shape(self):
        # Including a skipped row, which is the only one carrying model_path. Two
        # spellings within one node split the group key exactly as a cross-node
        # divergence would.
        ok = E.row_to_shared(self.rec(), "r")["model"]
        skipped = E.row_to_shared(
            self.rec(status="skipped", per_token_status="model_missing",
                     model_path="/m/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
                     avg_ts=""), "r")["model"]
        self.assertEqual(ok, skipped)

    def test_energy_columns_are_empty_by_design(self):
        r = E.row_to_shared(self.rec(), "r")
        for col in ("t_start", "t_end", "energy_window", "energy_j",
                    "joules_per_token", "mean_power_w"):
            self.assertEqual(r[col], "", "%s must be empty -- see the module docstring" % col)

    def test_the_throttle_state_is_stated_in_the_column_not_as_a_flag(self):
        """Both halves matter, and getting either wrong deletes or misleads.

        An empty `throttled_end` reads as "checked and clean" under Pi 4's
        fail-closed convention -- so it must say "unavailable". But raising
        `throttle-unknown` as a flag is worse: ThinkPad/control/analysis.py lists
        that exact token in THROUGHPUT_INVALIDATING, so it would delete every row
        of this node from the placement rule. The state is a property of the
        board, not a verdict on the run.
        """
        r = E.row_to_shared(self.rec(), "r")
        self.assertEqual(r["throttled_end"], "unavailable")
        self.assertNotIn("throttle", r["flags"])

    def test_a_row_missing_the_cluster_id_says_so(self):
        # Falling back to the local NODE_ID silently would put this node in a
        # group of its own at the control node, which looks like a node that ran
        # nothing rather than a misconfiguration.
        r = E.row_to_shared(self.rec(cluster_node_id=""), "r")
        self.assertIn("node-id-unmapped", r["flags"])

    def test_a_failed_or_skipped_run_still_projects(self):
        """The driver builds these records without a bench row, so they carry
        `phase` and no `row_phase`. Keying on row_phase alone dropped every one of
        them from the shared CSV with no row and no flag."""
        for over in ({"status": "failed", "rc": "1", "avg_ts": "0"},
                     {"status": "skipped", "per_token_status": "model_missing",
                      "avg_ts": ""}):
            rec = self.rec(**over)
            rec.pop("row_phase")           # exactly the shape run_matrix once built
            row = E.row_to_shared(rec, "r")
            self.assertIsNotNone(row, "a %s run vanished from the shared CSV" % over["status"])
            self.assertEqual(row["phase"], "pp")

    def test_a_missing_temperature_is_flagged_not_read_as_fine(self):
        r = E.row_to_shared(self.rec(temp_end_c=""), "r")
        self.assertEqual(r["temp_end_c"], "")
        self.assertIn("temp-unknown", r["flags"])
        # 0.0 is common.sh's unreadable-sensor fallback, not a cold board
        self.assertIn("temp-unknown", E.row_to_shared(self.rec(temp_end_c="0.0"), "r")["flags"])

    def test_this_nodes_own_refusal_reaches_the_shared_flags(self):
        r = E.row_to_shared(self.rec(per_token_status="refused_counts_mismatch"), "r")
        self.assertIn("jetson-per-token:refused_counts_mismatch", r["flags"])

    def test_a_failed_run_is_flagged_and_keeps_its_row(self):
        r = E.row_to_shared(self.rec(status="failed", rc="1", avg_ts="0"), "r")
        self.assertIsNotNone(r)
        self.assertIn("bench-exit:1", r["flags"])
        self.assertIn("no-throughput", r["flags"])

    def test_written_header_is_pi4s_column_order(self):
        rows = [E.row_to_shared(self.rec(), "r")]
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                         newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=E.COLUMNS)
            w.writeheader()
            w.writerows(rows)
            path = fh.name
        try:
            with io.open(path, encoding="utf-8") as fh:
                self.assertEqual(next(csv.reader(fh)), E.COLUMNS)
        finally:
            os.unlink(path)


class ControlNodeAccepts(unittest.TestCase):
    """The test that was missing, and that let a node-deleting bug ship green.

    Every other test here stops at Pi 4's header: they prove the COLUMNS match and
    the values are well-formed, and none of them ever asks the consumer what it
    does with them. A row can satisfy all of that and still be discarded in full --
    which is what happened when this file flagged `throttle-unknown` on every row,
    a token ThinkPad/control/analysis.py lists in THROUGHPUT_INVALIDATING. The
    projection looked perfect and delivered zero rows to the placement rule.
    """

    def setUp(self):
        self.control = os.path.join(CLUSTER, "ThinkPad")
        if not os.path.exists(os.path.join(self.control, "control", "analysis.py")):
            self.skipTest("ThinkPad/control not present; the consumer cannot be asked")
        if self.control not in sys.path:
            sys.path.insert(0, self.control)

    def _row(self, **over):
        base = Projection().rec(**over)
        return E.row_to_shared(base, "r")

    def test_a_clean_row_is_not_discarded_by_the_consumer(self):
        from control import analysis
        row = self._row()
        self.assertFalse(
            analysis.invalidates_throughput(row["flags"]),
            "a clean sweep row must reach the placement rule; flags=%r" % row["flags"])

    def test_no_emitted_flag_is_a_throughput_verdict(self):
        """Any token this file can raise, checked against the consumer's own list.

        A flag that means "look at this" to us must not mean "this number is not a
        measurement" to the control node -- that is how an honest annotation
        deletes a node.
        """
        from control import analysis
        raised = set()
        with io.open(os.path.join(JETSON, "bench", "emit_results_csv.py"),
                     encoding="utf-8") as fh:
            for m in re.finditer(r'flags\.append\(\s*"([^"%]+)', fh.read()):
                raised.add(m.group(1).rstrip(":"))
        fatal = sorted(f for f in raised if analysis.invalidates_throughput(f))
        self.assertEqual(
            fatal, [],
            "these flags delete the row's throughput at the control node: %s. Either"
            " stop raising them, or state the condition in a column instead." % fatal)

    def test_the_node_reaches_the_placement_rule(self):
        from control import profiles
        import tempfile
        rows = [self._row(),
                self._row(row_phase="decode", bench_n_prompt="0", bench_n_gen="128",
                          avg_ts="128.0", avg_ns="1000000000")]
        d = tempfile.mkdtemp()
        path = os.path.join(d, "results.csv")
        with io.open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=E.COLUMNS)
            w.writeheader()
            w.writerows(rows)
        t = profiles.per_layer_times(path, {"qwen2.5-1.5b-instruct-q4_k_m": 28}, threads=4)
        self.assertEqual(t.get("_dropped_flagged_rows"), 0, t)
        nid = rows[0]["node_id"]
        self.assertIn(nid, t, "the node is absent from per_layer_times: %s" % t)
        self.assertIsNotNone(t[nid].get("prefill"))
        self.assertIsNotNone(t[nid].get("decode"))

    def test_node_id_is_one_the_cluster_knows(self):
        """profiles.py groups by row["node_id"] with no alias resolution, so a local
        spelling sits in a group of its own and never joins anything."""
        import json
        cj = os.path.join(CLUSTER, "ThinkPad", "config", "cluster.json")
        if not os.path.exists(cj):
            self.skipTest("cluster.json not present")
        with io.open(cj, encoding="utf-8") as fh:
            ids = {n["id"] for n in json.load(fh)["nodes"]}
        self.assertIn(self._row()["node_id"], ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
