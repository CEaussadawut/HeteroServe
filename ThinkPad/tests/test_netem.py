"""Tests for link-tier control in control/netem.py.

Network shaping IS scope item 6 of the study, and this module had no test coverage, which is
how two defects survived in it:

  1. `[n for n in cluster.workers() if n.ssh_user]` silently dropped the control node.
     `config.workers()` includes the control node exactly when `serves_rpc` is set -- the
     prefill-on-the-T580 disaggregation condition -- and `cluster.json` gives t580
     `"ssh_user": null`, so the one node the module's own docstring singles out ("and to the
     control node too when it serves") was the one node never shaped. Its egress stayed at
     1000 Mbit while the tier claimed 10, and `survey()` measures worker->control only, so a
     10/1000 link was recorded as 10/10 and applied to both legs by the placement rule.

  2. A failed `tc` was stored, inspected, and then ignored: `apply_tier` returned normally
     and the CLI exited 0. Since `netem.py:43` hard-codes `05_net_tier.sh`, which exists
     only in `Pi4/`, `heteroctl tier shaped-10` shaped one of three workers and reported
     success. The runs are still labelled `tier=shaped-10`, so the condition is MISLABELLED
     rather than missing -- a gbe-vs-shaped-10 comparison becomes two unshaped conditions
     under different names.

`_run` is stubbed: what is under test is which nodes are reached and what happens when one
of them fails, not ssh or tc themselves.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control import config, netem  # noqa: E402


class NetemHarness(unittest.TestCase):
    def setUp(self):
        self.cluster = config.load(ROOT / "config" / "cluster.json")
        # The disaggregation condition: the control node serves, so it participates.
        self.cluster.node("t580").serves_rpc = True
        self._run = netem._run
        # `apply_tier` calls nodeclient_safe_mark on every success, which makes a REAL HTTP
        # request to the node's agent. Against a cluster that is not powered on, that blocks
        # on TCP connect for each node -- the marking is best-effort and wrapped in a bare
        # except, so it never fails, it just hangs. Stub it out.
        self._mark = netem.nodeclient_safe_mark
        netem.nodeclient_safe_mark = lambda node, event, note: None
        self.calls = []

    def tearDown(self):
        netem._run = self._run
        netem.nodeclient_safe_mark = self._mark

    def stub(self, results=None):
        results = results or {}

        def _run(cluster, node, cmd, timeout):
            self.calls.append((node.id, cmd))
            return results.get(node.id, (0, "ok", ""))

        netem._run = _run

    @property
    def reached(self):
        return sorted({node for node, _ in self.calls})


class TestControlNodeIsShaped(NetemHarness):
    def test_apply_tier_reaches_the_control_node_when_it_serves(self):
        self.stub()
        netem.apply_tier(self.cluster, "shaped-10")
        self.assertIn("t580", self.reached,
                      "the control node serves here, so its egress must be shaped too")

    def test_clear_reaches_the_control_node(self):
        # Shaping outlives the experiment that set it; a tier left on the control node
        # would quietly corrupt the next run.
        self.stub()
        netem.clear(self.cluster)
        self.assertIn("t580", self.reached)

    def test_verify_reports_the_control_node(self):
        self.stub()
        out = netem.verify(self.cluster)
        self.assertIn("t580", out)

    def test_survey_excludes_the_control_node_on_purpose(self):
        # The exception, and for a real reason: it is the iperf3 server, so measuring it
        # against itself is meaningless. This must not regress into shaping-style coverage.
        targets = [n for n in self.cluster.workers() if n.id != self.cluster.control_id]
        self.assertNotIn("t580", [n.id for n in targets])
        self.assertIn("pi4", [n.id for n in targets])


class TestPartialApplicationIsRefused(NetemHarness):
    MISSING = (127, "", "bash: 05_net_tier.sh: No such file or directory")

    def test_a_failed_node_refuses_the_whole_tier(self):
        self.stub({"pi5": self.MISSING, "jetson": self.MISSING})
        with self.assertRaises(netem.NetemError) as cm:
            netem.apply_tier(self.cluster, "shaped-10")
        message = str(cm.exception)
        self.assertIn("2 of 4", message)
        self.assertIn("pi5", message)

    def test_the_error_names_the_actual_cause(self):
        # rc=127 means the script is not there, which is the real situation on Pi5/Jetson.
        # Saying "rc=127" alone would send the reader to the wrong place.
        self.stub({"pi5": self.MISSING})
        with self.assertRaises(netem.NetemError) as cm:
            netem.apply_tier(self.cluster, "shaped-10")
        self.assertIn("05_net_tier.sh is missing", str(cm.exception))

    def test_all_succeeding_returns_ok(self):
        self.stub()
        report = netem.apply_tier(self.cluster, "shaped-10")
        self.assertTrue(report["ok"])
        self.assertEqual(report["failed"], [])
        self.assertEqual(sorted(report["applied"]), sorted(self.reached))

    def test_strict_false_proceeds_but_still_reports_failure(self):
        # The escape hatch must not launder the failure into a success.
        self.stub({"pi5": self.MISSING})
        report = netem.apply_tier(self.cluster, "shaped-10", strict=False)
        self.assertFalse(report["ok"])
        self.assertEqual(len(report["failed"]), 1)
        self.assertNotIn("pi5", report["applied"])

    def test_a_node_with_no_route_at_all_is_a_failure_not_a_skip(self):
        """Previously such a node was filtered out of the target list entirely, so it
        looked like it had been shaped. Now it is a target that fails.

        Exercises the real `_run` directly rather than through `apply_tier`: reaching the
        other nodes would attempt genuine ssh with a 60 s timeout each.
        """
        node = self.cluster.node("pi4")
        node.ssh_user = None
        rc, _out, err = netem._run(self.cluster, node, "true", timeout=5)
        self.assertNotEqual(rc, 0, "a node with no route must not report success")
        self.assertIn("no way to reach it", err)

    def test_the_control_node_runs_locally_rather_than_over_ssh(self):
        # t580 has ssh_user null; it is shaped via subprocess, not ssh.
        rc, out, _err = netem._run(self.cluster, self.cluster.node("t580"),
                                   "echo shaped-locally", timeout=5)
        self.assertEqual(rc, 0)
        self.assertIn("shaped-locally", out)


if __name__ == "__main__":
    unittest.main()
