"""Tests for the preflight gate in control/nodes.py.

This module had NO test coverage at all, which is how two fail-open paths survived in the
one function whose entire job is to refuse. `heteroctl preflight` is the gate: `ready` False
means do not start the run, and the harness honours it. A gate that cannot say no is worse
than no gate, because the run proceeds and the numbers look ordinary.

The agent is stubbed rather than mocked at the HTTP layer: what is under test is how
`preflight` INTERPRETS a body, so the bodies are supplied directly.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control import config, nodes  # noqa: E402
from control import cli  # noqa: E402


class PreflightHarness(unittest.TestCase):
    """Drives the real `preflight` with a stubbed fanout/info."""

    def setUp(self):
        self.cluster = config.load(ROOT / "config" / "cluster.json")
        self._fanout, self._info = nodes.fanout, nodes.info

    def tearDown(self):
        nodes.fanout, nodes.info = self._fanout, self._info

    def preflight(self, body, require_pin=False, pin_sha="abc123"):
        nodes.fanout = lambda ns, fn, **kw: {n.id: {"ok": True, "data": body} for n in ns}
        nodes.info = lambda n: {"software": {"llama_pin_sha": pin_sha}}
        return nodes.preflight(self.cluster, require_pin=require_pin)


class TestPreflightFailsClosed(PreflightHarness):
    """`h.get("blocking", [])` meant ANY parseable body without that key contributed zero
    blocking reasons, so a proxy page, an error body, a drifted key name or a half-deployed
    agent all read as "no problems reported" and the node passed."""

    def test_a_healthy_node_still_passes(self):
        # `clock_offset_s` is part of the contract: preflight gates on it, so a body that
        # omits it is not a healthy node -- it is a node whose clock is unverified.
        report = self.preflight({"ready": True, "blocking": [], "warnings": [],
                                 "clock_offset_s": 0.001})
        self.assertTrue(report["ready"])
        self.assertEqual(report["blocking"], [])

    def test_a_node_that_states_its_problems_blocks(self):
        report = self.preflight({"ready": False, "blocking": ["swap enabled"], "warnings": []})
        self.assertFalse(report["ready"])
        self.assertTrue(any("swap enabled" in b for b in report["blocking"]))

    def test_ready_false_with_no_reasons_is_still_not_ready(self):
        # The gate's verdict came only from `blocking`; `ready` was recorded and never read,
        # so a node answering ready=false with an empty list was treated as ready.
        report = self.preflight({"ready": False, "blocking": [], "warnings": []})
        self.assertFalse(report["ready"])
        self.assertTrue(any("gave no reason" in b for b in report["blocking"]))

    def test_an_off_contract_body_blocks(self):
        for body in ({"status": "ok", "message": "service running"},
                     {},
                     {"ready": True},                      # no `blocking` list
                     {"blocking": []},                     # no `ready`
                     {"ready": True, "blocking": "none"}):  # `blocking` not a list
            with self.subTest(body=body):
                report = self.preflight(body)
                self.assertFalse(report["ready"],
                                 "an off-contract body must not read as ready")
                self.assertTrue(any("readiness contract" in b for b in report["blocking"]))

    def test_a_non_dict_body_blocks(self):
        report = self.preflight(["ok"])
        self.assertFalse(report["ready"])

    def test_an_unreachable_node_blocks(self):
        nodes.fanout = lambda ns, fn, **kw: {n.id: {"ok": False, "error": "timeout"} for n in ns}
        nodes.info = lambda n: {}
        report = nodes.preflight(self.cluster, require_pin=False)
        self.assertFalse(report["ready"])
        self.assertTrue(any("unreachable" in b for b in report["blocking"]))


class TestPreflightClockGate(PreflightHarness):
    """`clock_offset_s` was recorded at nodes.py:160 and read by nothing but the display
    column, so the gate printed its evidence and never acted on it -- while this very
    function's docstring says a run started on a skewed clock "produces numbers that look
    fine and are wrong".

    It is not hypothetical. Pi 4's agent gates on chrony and supplies the offset
    (node_agent.py:286-299); the Pi 5's readiness() has nine gates and not one concerns the
    clock (Pi5/node/agent.py:126-165), so the Pi 5 can answer ready=true with an arbitrarily
    skewed clock. Node markers drive analysis.windows_from_events -> the idle windows ->
    mean_power_w_idle -> delta_power_w, the marginal-energy figure.
    """

    def test_a_node_that_will_not_report_its_offset_blocks(self):
        """Unverified must not read as synced -- the same rule the pin gate applies."""
        report = self.preflight({"ready": True, "blocking": [], "warnings": []})
        self.assertFalse(report["ready"])
        self.assertTrue(any("will not report its clock offset" in b
                            for b in report["blocking"]))

    def test_an_offset_beyond_five_ms_blocks(self):
        report = self.preflight({"ready": True, "blocking": [], "warnings": [],
                                 "clock_offset_s": 0.02})
        self.assertFalse(report["ready"])
        self.assertTrue(any("exceeds" in b for b in report["blocking"]))

    def test_a_negative_offset_is_just_as_bad(self):
        """Skew has a sign; error does not. A node 20 ms BEHIND is equally misaligned."""
        report = self.preflight({"ready": True, "blocking": [], "warnings": [],
                                 "clock_offset_s": -0.02})
        self.assertFalse(report["ready"])
        self.assertTrue(any("exceeds" in b for b in report["blocking"]))

    def test_an_offset_inside_the_threshold_passes(self):
        report = self.preflight({"ready": True, "blocking": [], "warnings": [],
                                 "clock_offset_s": 0.004})
        self.assertTrue(report["ready"], report["blocking"])

    def test_the_threshold_matches_the_one_pi4_enforces_on_itself(self):
        """Two thresholds for one quantity is how a cluster disagrees with itself."""
        self.assertEqual(nodes.MAX_CLOCK_OFFSET_S, 0.005)

    def test_the_escape_hatch_disables_it(self):
        nodes.fanout = lambda ns, fn, **kw: {
            n.id: {"ok": True, "data": {"ready": True, "blocking": [], "warnings": []}}
            for n in ns}
        nodes.info = lambda n: {"software": {"llama_pin_sha": "abc123"}}
        report = nodes.preflight(self.cluster, require_pin=False, require_clock=False)
        self.assertTrue(report["ready"], report["blocking"])

    def test_an_unreachable_node_is_not_also_blamed_for_its_clock(self):
        """One failure, one reason. An unreachable node already blocks; piling a second,
        misleading reason on it sends the operator to chrony instead of the cable."""
        nodes.fanout = lambda ns, fn, **kw: {
            n.id: {"ok": False, "error": "connection refused"} for n in ns}
        nodes.info = lambda n: {"software": {"llama_pin_sha": "abc123"}}
        report = nodes.preflight(self.cluster, require_pin=False)
        self.assertFalse(report["ready"])
        self.assertFalse(any("clock offset" in b for b in report["blocking"]),
                         report["blocking"])


class TestPreflightPinVerification(PreflightHarness):
    """One pinned llama.cpp revision across the cluster. Rows built from two revisions
    cannot be compared, which the module states is "blocking, not advisory" -- but a node
    that failed to report its revision only raised a warning, so unverified read as agreed.
    The Jetson ships no node agent at all, so it is exactly the node this affects."""

    HEALTHY = {"ready": True, "blocking": [], "warnings": [], "clock_offset_s": 0.001}

    CLUSTER_PIN = "abc123"

    def _preflight(self, cluster_pin, node_pin):
        """Run the real preflight against an EXPLICIT cluster pin.

        `test_an_empty_cluster_pin_blocks` used to call the shared harness, which loads the
        LIVE config/cluster.json -- so it asserted a property of a transient global state.
        It passed only while Q1 was unanswered, and answering Q1 (pinning b5050,
        23106f94ea2bc3da929afb7330655fd5515d08dc) turned it red without a single line of
        preflight logic changing. Worse, the behaviour it covers -- an empty cluster pin must
        block -- would then have had NO coverage at all, because it was the only test of it.

        A test of a guard must supply the state it is testing. The live file is data that
        changes; the guard is the thing under test.
        """
        raw = config.load(ROOT / "config" / "cluster.json")
        raw.llama = dict(raw.llama or {})
        raw.llama["pin_sha"] = cluster_pin
        nodes.fanout = lambda ns, fn, **kw: {n.id: {"ok": True, "data": self.HEALTHY} for n in ns}
        nodes.info = lambda n: {"software": {"llama_pin_sha": node_pin}}
        return nodes.preflight(raw, require_pin=True)

    def _with_pin(self, pin_sha):
        """Cluster pinned to CLUSTER_PIN; `pin_sha` is what the NODES report."""
        return self._preflight(self.CLUSTER_PIN, pin_sha)

    def test_an_empty_cluster_pin_blocks(self):
        report = self._preflight("", self.CLUSTER_PIN)
        self.assertFalse(report["ready"])
        self.assertTrue(any("pin_sha is empty" in b for b in report["blocking"]))

    def test_the_empty_pin_verdict_does_not_depend_on_the_live_config(self):
        """The same nodes, the same code, opposite verdicts -- decided only by the fixture.

        This is the guard against the defect returning: if either assertion ever starts
        reading config/cluster.json again, one of them fails whatever today's repo state is.
        """
        blocked = self._preflight("", self.CLUSTER_PIN)
        agreed = self._preflight(self.CLUSTER_PIN, self.CLUSTER_PIN)
        self.assertFalse(blocked["ready"])
        self.assertTrue(agreed["ready"], agreed["blocking"])
        self.assertTrue(any("pin_sha is empty" in b for b in blocked["blocking"]))
        self.assertFalse(any("pin_sha is empty" in b for b in agreed["blocking"]))

    def test_the_live_cluster_pin_is_not_what_these_tests_assert_on(self):
        """Documents the trap directly: the live pin is now SET, and that must not matter."""
        live = (config.load(ROOT / "config" / "cluster.json").llama or {}).get("pin_sha", "")
        report = self._preflight("", self.CLUSTER_PIN)
        self.assertTrue(any("pin_sha is empty" in b for b in report["blocking"]),
                        "preflight must block on the FIXTURE's empty pin regardless of the "
                        "live config, which currently reads %r" % live)

    def test_a_matching_pin_passes(self):
        self.assertTrue(self._with_pin("abc123")["ready"])

    def test_a_divergent_pin_blocks(self):
        report = self._with_pin("def456")
        self.assertFalse(report["ready"])
        self.assertTrue(any("cluster pins" in b for b in report["blocking"]))

    def test_a_node_that_will_not_report_its_revision_blocks(self):
        report = self._with_pin(None)
        self.assertFalse(report["ready"],
                         "unverified revision agreement must not read as agreement")
        self.assertTrue(any("cannot be verified" in b for b in report["blocking"]))


if __name__ == "__main__":
    unittest.main()


class TestStatusNeverAssertsWhatItDoesNotKnow(unittest.TestCase):
    """`"up" if m.get("rpc_active") else "down"` turned an ABSENT key into an affirmative
    "down". The Pi 5's /metrics wraps its body in {"sample": ...} and so carries no
    top-level `rpc_active` at all, meaning `heteroctl status` reported a running RPC server
    as stopped. An unknown state must read as unknown."""

    class Args:
        config = ROOT / "config" / "cluster.json"

    def _status(self, bodies):
        import contextlib, io
        saved_fanout, saved_metrics = nodes.fanout, nodes.metrics
        cli.nodeclient.fanout = lambda ns, fn, **kw: {
            n.id: {"ok": True, "data": bodies.get(n.id, {})} for n in ns}
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                cli.cmd_status(self.Args())
        finally:
            cli.nodeclient.fanout, cli.nodeclient.metrics = saved_fanout, saved_metrics
        return [l for l in buf.getvalue().splitlines() if l.strip()]

    def test_an_absent_rpc_key_reads_as_unknown_not_as_down(self):
        line = [l for l in self._status({"pi5": {"soc_temp_c": 78.4}}) if "pi5" in l][-1]
        self.assertIn("rpc=?", line)
        self.assertNotIn("rpc=down", line)

    def test_a_genuinely_stopped_server_still_reads_as_down(self):
        """The fix must not cost us the true negative."""
        line = [l for l in self._status({"pi4": {"rpc_active": False}}) if "pi4" in l][-1]
        self.assertIn("rpc=down", line)

    def test_a_running_server_reads_as_up(self):
        line = [l for l in self._status({"pi4": {"rpc_active": True}}) if "pi4" in l][-1]
        self.assertIn("rpc=up", line)

    def test_a_missing_reading_prints_a_question_mark_not_the_word_None(self):
        line = [l for l in self._status({"pi4": {}}) if "pi4" in l][-1]
        self.assertNotIn("None", line)


class TestIdleBaselineProvenance(unittest.TestCase):
    """`nodes` listed the nodes we ATTEMPTED to mark. The Jetson has agent_port 9410 in
    cluster.json and ships no agent, so the primary artefact asserted a marker that cannot
    exist. `mark_all` returns per-node ok/error and both return values were discarded."""

    def _baseline(self, results):
        from control import harness as harness_mod

        class FakeNode:
            def __init__(s, nid): s.id = nid; s.agent_port = 9410

        class FakeCluster:
            @staticmethod
            def workers(): return [FakeNode(n) for n in ("pi4", "pi5", "jetson")]

        class FakeClient:
            @staticmethod
            def mark_all(nodes_, event, note=""): return results

        h = object.__new__(harness_mod.Harness)
        h.cluster = FakeCluster()
        saved_client, saved_sleep = harness_mod.nodeclient, harness_mod.time.sleep
        harness_mod.nodeclient = FakeClient
        harness_mod.time.sleep = lambda s: None
        try:
            return harness_mod.Harness.idle_baseline(h, seconds=0, note="t")
        finally:
            harness_mod.nodeclient, harness_mod.time.sleep = saved_client, saved_sleep

    def test_a_node_with_no_agent_is_not_claimed_as_marked(self):
        rec = self._baseline({
            "pi4": {"ok": True, "data": {}}, "pi5": {"ok": True, "data": {}},
            "jetson": {"ok": False, "error": "connection refused"}})
        self.assertEqual(rec["nodes"], ["pi4", "pi5"])
        self.assertEqual(rec["unmarked"], ["jetson"])

    def test_a_fully_marked_cluster_reports_everyone_and_nothing_unmarked(self):
        rec = self._baseline({n: {"ok": True, "data": {}}
                              for n in ("pi4", "pi5", "jetson")})
        self.assertEqual(rec["nodes"], ["jetson", "pi4", "pi5"])
        self.assertEqual(rec["unmarked"], [])
