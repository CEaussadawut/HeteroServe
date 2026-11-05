"""Tests for bench/link_report.py -- the interconnect half of the placement inputs.

placement.py minimises T + lambda*E where T = sum_i n_i*c_i + C. The compute term comes
from the Phase 2 sweep; C comes from here. Before 11_link_survey.sh existed nothing wrote
link_survey.json at all, so profiles.py fell through to link_mbps=1000.0 / rtt_ms=0.3 --
datasheet numbers, in a module whose docstring says "Nothing is taken from a datasheet".
These tests pin the contract so that stays true.

The iperf3 fixtures follow the structure emitted by `iperf_json_printf` in upstream
src/iperf_api.c: `end.sum_sent` and `end.sum_received`, each carrying start/end/seconds/
bytes/bits_per_second, with `retransmits` added to `sum_sent` for TCP. The ping fixtures
are real `ping` output, not paraphrased.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK_REPORT = ROOT / "bench" / "link_report.py"


def _load():
    spec = importlib.util.spec_from_file_location("link_report", LINK_REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lr = _load()

# Captured verbatim from `ping -c 3 -i 0.2 -q 127.0.0.1` on a Linux host.
PING_OK = """PING 127.0.0.1 (127.0.0.1) 56(84) bytes of data.

--- 127.0.0.1 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 411ms
rtt min/avg/max/mdev = 0.029/0.043/0.052/0.010 ms
"""

PING_LOSSY = """PING 192.168.50.10 (192.168.50.10) 56(84) bytes of data.

--- 192.168.50.10 ping statistics ---
20 packets transmitted, 17 received, 15% packet loss, time 3806ms
rtt min/avg/max/mdev = 0.201/0.845/4.301/1.025 ms
"""

PING_UNREACHABLE = """PING 192.168.50.99 (192.168.50.99) 56(84) bytes of data.

--- 192.168.50.99 ping statistics ---
5 packets transmitted, 0 received, 100% packet loss, time 4080ms
"""


def iperf(bps_sent, bps_recv, *, retransmits=0, seconds=10.0, version="iperf 3.16"):
    return {
        "start": {"version": version, "test_start": {"protocol": "TCP", "num_streams": 1,
                                                     "duration": 10, "reverse": 0}},
        "end": {
            "sum_sent": {"start": 0, "end": seconds, "seconds": seconds,
                         "bytes": int(bps_sent * seconds / 8),
                         "bits_per_second": bps_sent, "retransmits": retransmits,
                         "sender": True},
            "sum_received": {"start": 0, "end": seconds, "seconds": seconds,
                             "bytes": int(bps_recv * seconds / 8),
                             "bits_per_second": bps_recv, "sender": False},
        },
    }


GBE_FWD = iperf(948_000_000, 941_000_000)
GBE_REV = iperf(942_000_000, 937_000_000)


class Case(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.store = self.tmp / "measurements.json"
        self.store.write_text("[]", encoding="utf-8")

    def w(self, name, obj):
        p = self.tmp / name
        p.write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")
        return p

    def accumulate(self, *, ping=PING_OK, fwd=GBE_FWD, rev=GBE_REV, env=None):
        argv = ["--accumulate", str(self.store), "--ping", str(self.w("p.txt", ping))]
        if fwd is not None:
            argv += ["--forward", str(self.w("f.json", fwd))]
        if rev is not None:
            argv += ["--reverse", str(self.w("r.json", rev))]
        base = {"PEER_ID": "laptop-t580", "PEER_HOST": "192.168.50.10",
                "PEER_ROLE": "orchestrator", "IFACE": "eth0", "TIER": "gbe",
                "LINK_SPEED": "1000", "MTU": "1500"}
        base.update(env or {})
        old = dict(os.environ)
        os.environ.update(base)
        try:
            lr.main(argv)
        finally:
            os.environ.clear()
            os.environ.update(old)
        return json.loads(self.store.read_text(encoding="utf-8"))

    def finish(self, env=None):
        out = self.tmp / "link_survey.json"
        base = {"NODE_ID": "pi5", "ORCH_ID": "laptop-t580", "RUN_ID": "20260913T000000Z",
                "TEMP_START": "44.1", "TEMP_END": "47.2"}
        base.update(env or {})
        old = dict(os.environ)
        os.environ.update(base)
        try:
            rc = lr.main(["--finish", str(self.store), "--out", str(out)])
        finally:
            os.environ.clear()
            os.environ.update(old)
        return rc, json.loads(out.read_text(encoding="utf-8"))


class TestPingParsing(Case):
    def test_real_ping_output(self):
        p = lr.parse_ping(PING_OK)
        self.assertEqual(p["rtt_ms"], 0.043)
        self.assertEqual(p["rtt_min_ms"], 0.029)
        self.assertEqual(p["rtt_max_ms"], 0.052)
        self.assertEqual(p["rtt_mdev_ms"], 0.010)
        self.assertEqual(p["loss_pct"], 0.0)
        self.assertEqual(p["packets"], 3)

    def test_avg_not_min_is_used(self):
        """rtt_ms feeds the comm term. The minimum flatters the link; the average is what
        a token actually waits."""
        self.assertEqual(lr.parse_ping(PING_LOSSY)["rtt_ms"], 0.845)

    def test_loss_is_captured(self):
        self.assertEqual(lr.parse_ping(PING_LOSSY)["loss_pct"], 15.0)

    def test_unreachable_has_no_rtt_line(self):
        p = lr.parse_ping(PING_UNREACHABLE)
        self.assertIsNone(p["rtt_ms"])
        self.assertEqual(p["loss_pct"], 100.0)

    def test_unparsed_loss_is_not_silence(self):
        """`if ping.get("loss_pct")` conflated 0.0 (measured, clean) with None (summary did
        not parse -- a non-English locale renders "packet loss" differently, and the ping
        inherits the node's locale). A link whose loss is unknown was published as clean
        with an empty warnings list."""
        localised = ("--- 192.168.50.10 ping statistics ---\n"
                     "5 paquets transmis, 5 recus, 0% perte de paquets\n"
                     "rtt min/avg/max/mdev = 0.201/0.284/0.401/0.05 ms\n")
        self.assertIsNone(lr.parse_ping(localised)["loss_pct"])
        rows = self.accumulate(ping=localised)
        self.assertTrue(any("packet loss unknown" in w for w in rows[0]["warnings"]),
                        "unknown loss published as clean: %r" % rows[0]["warnings"])
        # the rtt still parsed, so the row stays usable
        self.assertEqual(rows[0]["rtt_ms"], 0.284)

    def test_zero_loss_is_clean_not_warned(self):
        self.assertEqual(self.accumulate(ping=PING_OK)[0]["warnings"], [])

    def test_garbage_does_not_crash(self):
        self.assertIsNone(lr.parse_ping("nothing useful here")["rtt_ms"])


class TestIperfParsing(Case):
    def test_reads_delivered_goodput(self):
        """sum_received is what arrived. Reporting sum_sent would credit the link with
        bytes that were retransmitted or dropped."""
        got = lr.parse_iperf(self.w("f.json", GBE_FWD))
        self.assertEqual(got["mbps"], 941.0)

    def test_retransmits_are_kept(self):
        got = lr.parse_iperf(self.w("f.json", iperf(9e8, 8.9e8, retransmits=412)))
        self.assertEqual(got["retransmits"], 412)

    def test_error_document_is_surfaced(self):
        got = lr.parse_iperf(self.w("f.json", {"error": "unable to connect to server"}))
        self.assertIn("error", got)

    def test_missing_file_is_empty_not_fatal(self):
        self.assertEqual(lr.parse_iperf(self.tmp / "nope.json"), {})

    def test_empty_file_is_empty_not_fatal(self):
        self.assertEqual(lr.parse_iperf(self.w("e.json", "")), {})

    def test_truncated_json_is_empty_not_fatal(self):
        self.assertEqual(lr.parse_iperf(self.w("t.json", '{"end": {')), {})


class TestBottleneckRule(Case):
    """A hop costs what its bottleneck costs. The Pi 5's 2.5GbE is a USB 3.0 adapter and
    is not symmetric, so averaging the directions would report a link nothing can achieve
    in the direction that matters."""

    def test_slower_direction_wins(self):
        rows = self.accumulate(fwd=iperf(9.4e8, 9.4e8), rev=iperf(6.1e8, 6.0e8))
        self.assertEqual(rows[0]["mbps"], 600.0)
        self.assertEqual(rows[0]["mbps_forward"], 940.0)
        self.assertEqual(rows[0]["mbps_reverse"], 600.0)

    def test_asymmetry_stays_visible(self):
        rows = self.accumulate(fwd=iperf(9.4e8, 9.4e8), rev=iperf(6.1e8, 6.0e8))
        self.assertNotEqual(rows[0]["mbps_forward"], rows[0]["mbps_reverse"])

    def test_one_direction_only_still_reports(self):
        rows = self.accumulate(rev=None)
        self.assertEqual(rows[0]["mbps"], 941.0)
        self.assertIsNone(rows[0]["mbps_reverse"])

    def test_one_direction_only_says_so(self):
        """`mbps` is documented as the slower of the two directions. With one leg missing
        it is not a bottleneck at all, and the only other clue -- mbps_reverse being null
        -- is something no consumer reads. A failed reverse leg leaves 11_link_survey.sh
        writing an empty rev.json, so this is the ordinary failure, not an exotic one."""
        rows = self.accumulate(rev=None)
        self.assertTrue(any("only one direction" in w for w in rows[0]["warnings"]),
                        "a single-direction measurement was published as a bottleneck: %r"
                        % rows[0]["warnings"])

    def test_both_directions_present_is_not_warned(self):
        self.assertEqual(self.accumulate()[0]["warnings"], [])

    def test_qdisc_is_recorded_not_asserted(self):
        """The provenance block used to state 'refused to run with netem attached'
        unconditionally -- a claim about the caller's behaviour that the report cannot
        verify. It now records what the caller actually observed."""
        self.accumulate()
        _, doc = self.finish()
        self.assertIn("qdisc_observed", doc["provenance"])
        self.assertEqual(doc["provenance"]["qdisc_observed"], "not recorded by the caller")

    def test_qdisc_passes_through_when_the_caller_supplies_it(self):
        self.accumulate()
        rc, doc = self.finish(env={"QDISC_OBSERVED": "qdisc fq_codel 0: root refcnt 2"})
        self.assertIn("fq_codel", doc["provenance"]["qdisc_observed"])


class TestWarnings(Case):
    """A number with a fault behind it must not look like a clean measurement."""

    # These assert the warning's CONTENT, not just that some warning was raised. A warning
    # is an operator-facing message; "a warning fired" is satisfied by one quoting the
    # wrong figure or naming the wrong leg, which is worse than silence because it sends
    # the reader after the wrong fault.
    def test_packet_loss_is_flagged_with_the_measured_figure(self):
        rows = self.accumulate(ping=PING_LOSSY)
        loss = [w for w in rows[0]["warnings"] if "packet loss" in w]
        self.assertTrue(loss, "15% packet loss raised no warning")
        self.assertIn("15.0", loss[0],
                      "the loss warning did not quote the measured percentage: %r" % loss[0])

    def test_retransmits_are_flagged_with_the_count(self):
        rows = self.accumulate(fwd=iperf(9e8, 8.9e8, retransmits=900))
        rt = [w for w in rows[0]["warnings"] if "retransmit" in w]
        self.assertTrue(rt, "900 retransmits raised no warning")
        self.assertIn("900", rt[0],
                      "the retransmit warning did not quote the count: %r" % rt[0])
        self.assertIn("forward", rt[0], "the warning did not name the leg")

    def test_far_below_negotiated_speed_is_flagged_with_both_figures(self):
        """A GbE link delivering 300 Mb/s is a fault -- a bad cable, a USB 2.0 fallback,
        a duplex mismatch -- not a property of the topology worth modelling. The warning
        has to carry the delivered figure AND the negotiated one, or the reader cannot see
        which half is wrong."""
        rows = self.accumulate(fwd=iperf(3.1e8, 3.0e8), rev=iperf(3.1e8, 3.0e8))
        neg = [w for w in rows[0]["warnings"] if "negotiated" in w]
        self.assertTrue(neg, "300 Mb/s on a 1000 Mb/s link raised no warning")
        self.assertIn("300", neg[0], "the warning did not quote the delivered speed")
        self.assertIn("1000", neg[0], "the warning did not quote the negotiated speed")

    def test_healthy_gbe_is_not_flagged(self):
        self.assertEqual(self.accumulate()[0]["warnings"], [])

    def test_an_iperf_error_is_flagged_against_the_leg_that_failed(self):
        """A forward/reverse label swap would send the operator to the wrong end of the
        link -- and on an asymmetric USB adapter that is a materially different fault."""
        rows = self.accumulate(fwd={"error": "unable to connect to server: Connection refused"})
        err = [w for w in rows[0]["warnings"] if "Connection refused" in w]
        self.assertTrue(err, "a failed iperf leg raised no warning")
        self.assertTrue(err[0].startswith("forward:"),
                        "the error was not attributed to the failing leg: %r" % err[0])


class TestSurveyDocument(Case):
    def test_shape_matches_what_profiles_py_reads(self):
        """ThinkPad/control/profiles.py: link_survey() does
        data["nodes"][node] -> {"mbps", "rtt_ms"}. Anything else is ignored, but those
        two must be present and numeric."""
        self.accumulate()
        rc, doc = self.finish()
        self.assertEqual(rc, 0)
        self.assertIn("pi5", doc["nodes"])
        entry = doc["nodes"]["pi5"]
        self.assertIsInstance(entry["mbps"], float)
        self.assertIsInstance(entry["rtt_ms"], float)
        self.assertEqual(entry["mbps"], 937.0)
        self.assertEqual(entry["rtt_ms"], 0.043)

    def test_round_trips_through_the_real_loader(self):
        """Parse it with the same expression profiles.link_survey() uses, so a schema
        drift on either side shows up here rather than as a silent 1000 Mb/s default."""
        self.accumulate()
        _, doc = self.finish()
        links = {node: {"mbps": e.get("mbps"), "rtt_ms": e.get("rtt_ms")}
                 for node, e in doc.get("nodes", {}).items()}
        self.assertEqual(links["pi5"]["mbps"], 937.0)
        self.assertEqual(links["pi5"]["rtt_ms"], 0.043)

    def test_peer_links_do_not_become_this_nodes_link(self):
        """Worker-to-worker hops belong in "links" for the pipeline study. If a pi4 row
        landed in "nodes" it would overwrite this node's control-node link."""
        self.accumulate()
        self.accumulate(env={"PEER_ID": "pi4", "PEER_HOST": "192.168.50.13",
                             "PEER_ROLE": "worker-straggler"})
        _, doc = self.finish()
        self.assertEqual(list(doc["nodes"]), ["pi5"])
        self.assertEqual(len(doc["links"]), 2)
        self.assertEqual(doc["nodes"]["pi5"]["peer"], "laptop-t580")

    def test_faster_tier_is_chosen_for_the_contract(self):
        """Surveying both tiers should report the one actually in use, not the last one
        measured."""
        self.accumulate(env={"TIER": "gbe", "IFACE": "eth0", "LINK_SPEED": "1000"})
        self.accumulate(fwd=iperf(2.3e9, 2.28e9), rev=iperf(2.2e9, 2.19e9),
                        env={"TIER": "2.5gbe", "IFACE": "eth1", "LINK_SPEED": "2500"})
        _, doc = self.finish()
        self.assertEqual(doc["nodes"]["pi5"]["tier"], "2.5gbe")
        self.assertEqual(doc["nodes"]["pi5"]["mbps"], 2190.0)

    def test_no_orchestrator_link_is_exit_1(self):
        """Falling back to the 1000 Mb/s default must be loud. A survey that measured
        only peers cannot be used to compute a placement."""
        self.accumulate(env={"PEER_ID": "pi4", "PEER_ROLE": "worker-straggler"})
        rc, doc = self.finish()
        self.assertEqual(rc, 1)
        self.assertEqual(doc["nodes"], {})

    def test_peer_only_survey_is_not_a_failure(self):
        """`--peer pi4` surveys a pipeline hop and has no orchestrator row by definition.
        Returning 1 there killed 11_link_survey.sh under `set -e` before it could print
        where it had written the file, so the documented mode always appeared to fail."""
        self.accumulate(env={"PEER_ID": "pi4", "PEER_ROLE": "worker-straggler"})
        rc, doc = self.finish(env={"EXPECT_ORCH": "0"})
        self.assertEqual(rc, 0)
        self.assertEqual(doc["nodes"], {})
        self.assertEqual(len(doc["links"]), 1)

    def test_missing_orchestrator_is_still_fatal_when_expected(self):
        """The default must stay loud: silently falling back to profiles.py's 1000 Mb/s
        is exactly what this file exists to prevent."""
        self.accumulate(env={"PEER_ID": "pi4", "PEER_ROLE": "worker-straggler"})
        rc, _ = self.finish(env={"EXPECT_ORCH": "1"})
        self.assertEqual(rc, 1)

    def test_provenance_travels_with_the_numbers(self):
        self.accumulate()
        _, doc = self.finish()
        self.assertEqual(doc["source_node"], "pi5")
        self.assertEqual(doc["run_id"], "20260913T000000Z")
        self.assertEqual(doc["provenance"]["soc_temp_start_c"], "44.1")
        self.assertTrue(doc["generated_utc"].endswith("+00:00"))
        self.assertEqual(doc["links"][0]["negotiated_mbps"], 1000)
        self.assertEqual(doc["links"][0]["mtu"], 1500)


class TestBothLegsFailed(Case):
    """Every test that reached finish() had a working orchestrator row; the only no-nodes
    tests use a pi4 peer, so they exercise the PEER filter and never the HAS-A-MEASUREMENT
    filter. Deleting that filter promotes a row whose iperf failed into `nodes` as
    {"mbps": null} and returns 0 -- the operator is told the survey succeeded."""

    def test_no_bandwidth_means_no_contract_entry(self):
        err = {"error": "unable to connect to server: Connection refused"}
        rows = self.accumulate(fwd=err, rev=err)
        self.assertIsNone(rows[0]["mbps"])
        rc, doc = self.finish()
        self.assertEqual(rc, 1, "a survey with no measurement reported success")
        self.assertEqual(doc["nodes"], {})

    def test_the_failed_row_is_still_recorded(self):
        """Absence of a measurement is itself data -- the row belongs in links even though
        it cannot be a contract entry."""
        err = {"error": "unable to connect to server: Connection refused"}
        self.accumulate(fwd=err, rev=err)
        _, doc = self.finish()
        self.assertEqual(len(doc["links"]), 1)
        self.assertTrue(doc["links"][0]["warnings"])


class TestContractEntryIsComplete(Case):
    """profiles.py reads this dict. Four of its ten fields were asserted, so blanking
    warnings, measured_utc or iface all survived -- and reporting mbps_forward twice in
    place of mbps_reverse erases the USB adapter's asymmetry from the one place a
    downstream reader looks."""

    def test_every_field_of_the_node_entry(self):
        self.accumulate(fwd=iperf(9.4e8, 9.41e8), rev=iperf(6.1e8, 6.0e8),
                        env={"TIER": "2.5gbe", "IFACE": "eth1"})
        _, doc = self.finish()
        e = doc["nodes"]["pi5"]
        self.assertEqual(e["mbps"], 600.0)            # the bottleneck, not the mean
        self.assertEqual(e["mbps_forward"], 941.0)
        self.assertEqual(e["mbps_reverse"], 600.0)
        self.assertNotEqual(e["mbps_forward"], e["mbps_reverse"])
        self.assertEqual(e["rtt_ms"], 0.043)
        self.assertEqual(e["tier"], "2.5gbe")
        self.assertEqual(e["iface"], "eth1")
        self.assertEqual(e["peer"], "laptop-t580")
        self.assertEqual(e["warnings"], doc["links"][0]["warnings"])
        self.assertTrue(e["measured_utc"].endswith("+00:00"))

    def test_the_health_check_judges_by_the_bottleneck(self):
        """2300 forward / 400 reverse is the asymmetric-collapse case this module was
        written for. Judging by max(speeds) would leave it unflagged."""
        rows = self.accumulate(fwd=iperf(2.3e9, 2.3e9), rev=iperf(4.1e8, 4.0e8),
                               env={"LINK_SPEED": "2500"})
        self.assertEqual(rows[0]["mbps"], 400.0)
        self.assertTrue(any("negotiated" in w for w in rows[0]["warnings"]),
                        "a link that collapsed in one direction was not flagged")

    def test_the_negotiated_threshold_is_bracketed(self):
        """Previously pinned from neither side (0.30 vs 0.94)."""
        below = self.accumulate(fwd=iperf(4.7e8, 4.6e8), rev=iperf(4.7e8, 4.6e8),
                                env={"LINK_SPEED": "1000"})
        self.assertTrue(any("negotiated" in w for w in below[0]["warnings"]))
        self.setUp()
        above = self.accumulate(fwd=iperf(5.5e8, 5.4e8), rev=iperf(5.5e8, 5.4e8),
                                env={"LINK_SPEED": "1000"})
        self.assertFalse(any("negotiated" in w for w in above[0]["warnings"]))

    def test_a_fractional_loss_percentage_is_read_exactly(self):
        """LOSS_RE losing its decimal point turns a real 0.4% into 4.0."""
        p = ("--- x ping statistics ---\n"
             "1000 packets transmitted, 996 received, 0.4% packet loss, time 1ms\n"
             "rtt min/avg/max/mdev = 0.201/0.284/0.401/0.05 ms\n")
        self.assertEqual(lr.parse_ping(p)["loss_pct"], 0.4)
        rows = self.accumulate(ping=p)
        self.assertIn("0.4% ICMP packet loss", rows[0]["warnings"])


class TestDirtyLinkReachesTheContract(Case):
    """`test_every_field_of_the_node_entry` ties the entry's warnings to the link row on a
    HEALTHY link, so it reduces to [] == [] and blanking the field survived. The entry is
    the only dict profiles.py opens; if its warnings are empty, a link measured through 15%
    packet loss and heavy retransmits enters the placement rule labelled clean."""

    def test_the_entry_carries_the_faults_not_just_the_row(self):
        self.accumulate(ping=PING_LOSSY,
                        fwd=iperf(9.4e8, 9.41e8, retransmits=412),
                        rev=iperf(6.1e8, 6.0e8, retransmits=7))
        _, doc = self.finish()
        entry, row = doc["nodes"]["pi5"], doc["links"][0]
        self.assertTrue(entry["warnings"], "a filthy link reached the contract as clean")
        self.assertEqual(entry["warnings"], row["warnings"],
                         "the entry and the row disagree about the same measurement")
        joined = " ".join(entry["warnings"])
        self.assertIn("packet loss", joined)
        self.assertIn("retransmit", joined)

    def test_both_directions_are_reported_and_not_swapped(self):
        """Asymmetric retransmit counts, so a direction swap is detectable."""
        rows = self.accumulate(ping=PING_LOSSY,
                               fwd=iperf(9.4e8, 9.41e8, retransmits=412),
                               rev=iperf(6.1e8, 6.0e8, retransmits=7))
        self.assertEqual(rows[0]["retransmits_forward"], 412)
        self.assertEqual(rows[0]["retransmits_reverse"], 7)
        joined = " ".join(rows[0]["warnings"])
        self.assertIn("forward: 412 TCP retransmits", joined)
        self.assertIn("reverse: 7 TCP retransmits", joined)

    def test_the_loss_figure_is_the_measured_one(self):
        """Substring checks passed a mutant reporting 85.0% for a 15% link."""
        rows = self.accumulate(ping=PING_LOSSY)
        self.assertEqual(rows[0]["loss_pct"], 15.0)
        self.assertIn("15.0% ICMP packet loss", rows[0]["warnings"])

    def test_an_unreachable_peer_has_no_rtt_not_a_default(self):
        """0.3 ms is literally profiles.py's datasheet fallback that this module exists to
        displace; it must never be manufactured here."""
        rows = self.accumulate(ping=PING_UNREACHABLE)
        self.assertIsNone(rows[0]["rtt_ms"])


class TestTierSelectionIsOrderIndependent(Case):
    """`test_faster_tier_is_chosen_for_the_contract` surveys gbe then 2.5gbe, so 'fastest'
    and 'last surveyed' are the same row and `max(...)` -> `to_orch[-1]` survived. 2.5GbE is
    the primary NIC, so surveying it first is the natural order -- and in that order the
    mutant publishes the 1 GbE fallback, understating the link 2.3x and moving layers off
    this node for no physical reason."""

    def _gbe(self):
        self.accumulate(fwd=iperf(9.4e8, 9.41e8), rev=iperf(9.4e8, 9.39e8),
                        env={"TIER": "gbe", "IFACE": "eth0", "LINK_SPEED": "1000"})

    def _fast(self):
        self.accumulate(fwd=iperf(2.3e9, 2.29e9), rev=iperf(2.2e9, 2.19e9),
                        env={"TIER": "2.5gbe", "IFACE": "eth1", "LINK_SPEED": "2500"})

    def _survey(self, fast_first):
        order = (self._fast, self._gbe) if fast_first else (self._gbe, self._fast)
        for step in order:
            step()
        return self.finish()[1]["nodes"]["pi5"]

    def test_slow_tier_surveyed_first(self):
        e = self._survey(fast_first=False)
        self.assertEqual(e["tier"], "2.5gbe")
        self.assertEqual(e["mbps"], 2190.0)

    def test_fast_tier_surveyed_first(self):
        """The ordering the mutant needs. Same expected answer -- that is the claim."""
        e = self._survey(fast_first=True)
        self.assertEqual(e["tier"], "2.5gbe")
        self.assertEqual(e["mbps"], 2190.0)


class TestCLI(Case):
    def test_finish_without_out_is_an_error(self):
        proc = subprocess.run([sys.executable, str(LINK_REPORT), "--finish",
                               str(self.store)], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)

    def test_no_mode_is_an_error(self):
        proc = subprocess.run([sys.executable, str(LINK_REPORT)],
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)


class TestTheNodeEntryIsOneMeasuredRow(Case):
    """`nodes[node_id]` is what `ThinkPad/control/profiles.py` reads into
    `DeviceProfile.link_mbps` / `rtt_ms`, and it is assembled field-by-field from `best` --
    the fastest tier to the orchestrator. Field-by-field copying is exactly the shape that
    lets ONE field drift to a different row, or to a constant, while the entry still looks
    well-formed. Mutation testing found both: `rtt_ms` could be taken from `to_orch[0]`
    instead of `best`, or defaulted to 0.3 when unmeasured, and all 292 tests still passed.

    0.3 ms is not an arbitrary number -- it is profiles.py's own datasheet fallback for a
    node with no survey. A survey that emits it has produced a value indistinguishable from
    "never measured", which is the one failure this project treats as worse than a crash."""

    def _store(self, rows):
        self.store.write_text(json.dumps(rows), encoding="utf-8")

    def _row(self, **kw):
        row = {"peer": "laptop-t580", "peer_host": "192.168.50.10",
               "peer_role": "orchestrator", "iface": "eth0", "tier": "gbe",
               "negotiated_mbps": 1000, "mtu": 1500, "mbps": 937.0,
               "mbps_forward": 937.0, "mbps_reverse": 940.0, "rtt_ms": 0.043,
               "rtt_min_ms": 0.03, "rtt_max_ms": 0.06, "rtt_mdev_ms": 0.01,
               "loss_pct": 0.0, "icmp_packets": 20, "retransmits_forward": 0,
               "retransmits_reverse": 0, "iperf_version": "iperf 3.12", "warnings": []}
        row.update(kw)
        return row

    def test_every_field_comes_from_the_same_row_as_the_bandwidth(self):
        """Two tiers, the SLOWER one first and carrying a distinguishable rtt. The entry
        must describe the 2.5GbE row throughout -- taking rtt from `to_orch[0]` yields the
        GbE row's 0.900 ms, which is a real measurement of a link that is not the one in
        use, so nothing downstream can tell it is wrong."""
        self._store([
            self._row(tier="gbe", iface="eth0", mbps=937.0, rtt_ms=0.900),
            self._row(tier="2.5gbe", iface="enx0", mbps=2350.0, rtt_ms=0.210),
        ])
        _, doc = self.finish()
        entry = doc["nodes"]["pi5"]
        self.assertEqual(entry["mbps"], 2350.0)
        self.assertEqual(entry["rtt_ms"], 0.210,
                         "rtt_ms came from a different row than mbps: %r" % (entry,))
        self.assertEqual(entry["tier"], "2.5gbe")
        self.assertEqual(entry["iface"], "enx0")

    def test_an_unmeasured_rtt_stays_unmeasured(self):
        """`--ping` is optional, so `rtt_ms` is legitimately None on a bandwidth-only run.
        None must survive into the entry. Substituting any float here -- 0.3 above all --
        publishes a datasheet guess in the column the comm term reads, and `profiles.py`
        cannot distinguish it from a measurement."""
        self._store([self._row(rtt_ms=None, rtt_min_ms=None, rtt_max_ms=None,
                               rtt_mdev_ms=None, loss_pct=None, icmp_packets=None)])
        _, doc = self.finish()
        self.assertIsNone(doc["nodes"]["pi5"]["rtt_ms"],
                          "an unmeasured RTT was published as a number")

    def test_the_datasheet_fallback_value_is_never_emitted_as_a_measurement(self):
        """Names the specific constant, so that if anyone later adds an `or 0.3` anywhere on
        this path the suite says why it is wrong rather than just that a number changed."""
        self._store([self._row(rtt_ms=None)])
        _, doc = self.finish()
        self.assertNotEqual(doc["nodes"]["pi5"]["rtt_ms"], 0.3,
                            "0.3 ms is profiles.py's datasheet fallback for an unsurveyed "
                            "node; emitting it from a survey makes a guess look measured")


class TestAbsentFactsStayAbsent(Case):
    """`accumulate` reads the link's physical facts from the environment that
    11_link_survey.sh exports. Every one of them has an `else None` for "the caller could
    not read it" -- and every one of those branches was unreachable from the test suite,
    because the helper always supplies a full environment.

    Defaulting any of them substitutes a plausible number for a missing measurement. That
    is the same defect as the 0.3 ms RTT fallback, one layer down: the row still validates,
    the column is still populated, and nothing downstream can tell."""

    def test_an_unread_link_speed_is_none_not_a_gigabit(self):
        """`ethtool` prints `Speed: Unknown!` for a down or USB-attached port, so the
        non-numeric case is the normal one on the 2.5GbE adapter, not an edge case.
        `negotiated_mbps` is the reference the '<50% of negotiated' check is written
        against -- defaulting it to 1000 invents the denominator of a fault check."""
        for value in ("", "Unknown!"):
            with self.subTest(link_speed=value):
                self.store.write_text("[]", encoding="utf-8")
                rows = self.accumulate(env={"LINK_SPEED": value})
                self.assertIsNone(rows[0]["negotiated_mbps"],
                                  "an unread link speed was published as a number")
                self.assertFalse([w for w in rows[0]["warnings"] if "negotiated" in w],
                                 "a negotiated-speed fault was raised with nothing to "
                                 "compare against")

    def test_an_unread_mtu_is_none_not_jumbo(self):
        """A fabricated 9000 makes an ordinary 1500-MTU result look like an unexplained
        collapse -- MTU is the first thing anyone checks when 2.5GbE under-delivers."""
        for value in ("", "auto"):
            with self.subTest(mtu=value):
                self.store.write_text("[]", encoding="utf-8")
                rows = self.accumulate(env={"MTU": value})
                self.assertIsNone(rows[0]["mtu"])

    def test_an_unsampled_soc_temperature_is_empty_not_plausible(self):
        """Thermal state is the main confound for the Pi 5's decode cost, so a survey
        stamped with a temperature nobody sampled invites exactly the wrong reading of a
        slow result. The empty string is the explicit-absence convention used throughout
        the provenance block."""
        out = self.tmp / "survey.json"
        old = dict(os.environ)
        os.environ.update({"NODE_ID": "pi5", "ORCH_ID": "laptop-t580",
                           "RUN_ID": "20260913T000000Z"})
        # UNSET, not set-to-empty: `os.environ.get(k, default)` falls back to the default
        # only when the key is ABSENT, so a test that writes "" cannot see a default at all.
        os.environ.pop("TEMP_START", None)
        os.environ.pop("TEMP_END", None)
        try:
            lr.main(["--finish", str(self.store), "--out", str(out)])
        finally:
            os.environ.clear()
            os.environ.update(old)
        prov = json.loads(out.read_text(encoding="utf-8"))["provenance"]
        self.assertEqual(prov["soc_temp_start_c"], "",
                         "a temperature nobody sampled was published as a measurement")
        self.assertEqual(prov["soc_temp_end_c"], "")

    def test_an_absent_run_id_is_empty_not_a_word(self):
        """Same convention: "" means nobody set it. A literal "unknown" is a value, and it
        would join against nothing while looking like an identifier."""
        _, doc = self.finish(env={"RUN_ID": ""})
        self.assertEqual(doc["run_id"], "")


class TestPingCountsAndWarningsReachTheOperator(Case):
    def test_icmp_packets_is_the_transmitted_count(self):
        """`icmp_packets` is the sample size behind `rtt_ms` -- what tells a reader whether
        0.043 ms came from 3 pings or 1000. PING_OK sent 3 and received 3, so the existing
        assertion cannot distinguish the transmitted count from the received one. These two
        fixtures can: publishing `received` would make a lossy link look like a smaller,
        cleaner sample instead of a degraded one."""
        self.assertEqual(lr.parse_ping(PING_LOSSY)["packets"], 20)
        self.assertEqual(lr.parse_ping(PING_UNREACHABLE)["packets"], 5)

    def test_the_rtt_line_is_matched_only_as_pings_rtt_summary(self):
        """Guards the anchor on the rtt summary line rather than a bare min/avg/max/mdev."""
        self.assertEqual(lr.parse_ping(PING_LOSSY)["rtt_ms"], 0.845)
        self.assertIsNone(lr.parse_ping("no summary here at all").get("rtt_ms"))

    def test_the_operator_is_told_when_a_link_was_measured_with_warnings(self):
        """The warnings sit in the JSON either way; this is the line the human running
        11_link_survey.sh actually sees. Losing it means a survey taken through 15% packet
        loss scrolls past looking clean -- a silent failure in the operator's only feedback
        channel, which nothing else in this file covers."""
        self.accumulate(ping=PING_LOSSY)
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            self.finish()
        err = buf.getvalue()
        self.assertIn("link(s) measured with warnings", err,
                      "finish() reported nothing on stderr for a flagged link")
        self.assertIn("packet loss", err,
                      "the summary did not say what the warning was")

    def test_a_bandwidth_only_run_is_not_warned_about_unknown_loss(self):
        """`--ping` is optional. The 'packet loss unknown' warning exists for a ping that
        ran and could not be parsed -- raising it when no ping was requested makes every
        bandwidth-only survey look faulty, and a warning that fires always carries no
        information."""
        argv = ["--accumulate", str(self.store),
                "--forward", str(self.w("f.json", GBE_FWD))]     # no --ping at all
        env = {"PEER_ID": "laptop-t580", "PEER_HOST": "192.168.50.10",
               "PEER_ROLE": "orchestrator", "IFACE": "eth0", "TIER": "gbe",
               "LINK_SPEED": "1000", "MTU": "1500"}
        old = dict(os.environ)
        os.environ.update(env)
        try:
            lr.main(argv)
        finally:
            os.environ.clear()
            os.environ.update(old)
        rows = json.loads(self.store.read_text(encoding="utf-8"))
        self.assertIsNone(rows[0]["loss_pct"], "loss was invented for a run with no ping")
        self.assertFalse([w for w in rows[0]["warnings"] if "packet loss unknown" in w],
                         "a run with no --ping was warned about unparsed packet loss")


class TestTheSchemaTagIsPartOfTheContract(Case):
    def test_the_schema_version_is_pinned(self):
        """The one field a consumer would gate on to decide whether it understands the
        document. Written and never asserted, so the writer side was free to drift."""
        _, doc = self.finish()
        self.assertEqual(doc["schema"], "heteroserve.link_survey/1")


class TestRowProvenanceAndTheOperatorsConsoleLine(Case):
    """Fields that are written on every row and were asserted nowhere, plus the one line
    the operator actually reads while 11_link_survey.sh runs."""

    def test_the_measuring_tools_version_travels_with_the_number(self):
        """iperf3 changed its JSON accounting between 3.9 and 3.16 -- which is why the
        fixture docstring cites iperf_api.c. Without the version a goodput figure is not
        reproducible, only plausible."""
        self.assertEqual(lr.parse_iperf(self.w("v.json", GBE_FWD))["iperf_version"],
                         "iperf 3.16")
        rows = self.accumulate()
        self.assertEqual(rows[0]["iperf_version"], "iperf 3.16")

    def test_the_version_falls_back_to_whichever_leg_ran(self):
        """A failed forward leg must not take the version with it."""
        rows = self.accumulate(fwd=None)
        self.assertEqual(rows[0]["iperf_version"], "iperf 3.16")

    def test_the_row_records_which_peer_and_address_were_measured(self):
        """`peer_role` separates the orchestrator hop from the pi4 pipeline hop when the
        links are read by hand; `peer_host` is the only record of WHICH address was
        measured, and this node has both a 1GbE and a 2.5GbE one."""
        rows = self.accumulate()
        self.assertEqual(rows[0]["peer_host"], "192.168.50.10")
        self.assertEqual(rows[0]["peer_role"], "orchestrator")

    def test_the_icmp_sample_size_reaches_the_row(self):
        """parse_ping being right is not enough -- row assembly can still drop it."""
        rows = self.accumulate(ping=PING_LOSSY)
        self.assertEqual(rows[0]["icmp_packets"], 20)

    def test_the_console_line_shows_the_bottleneck_not_the_fast_direction(self):
        """This line is where an asymmetric collapse is first noticed. The USB 3.0 2.5GbE
        adapter is not symmetric, which is why both legs are run at all -- printing the
        forward figure would show 2300 Mb/s for a link whose bottleneck is 400."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.accumulate(fwd=iperf(2_300_000_000, 2_300_000_000),
                            rev=iperf(400_000_000, 400_000_000))
        line = buf.getvalue()
        self.assertIn("400 Mb/s", line, "the console line did not show the bottleneck")
        self.assertNotIn("2300 Mb/s", line,
                         "the console line showed the fast direction as the link speed")

    def test_the_console_line_says_so_when_there_is_nothing_to_report(self):
        """`no bandwidth` / `no rtt` are the honest renderings of an absent measurement;
        a 0 would read as a measured zero."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.accumulate(ping=PING_UNREACHABLE, fwd=None, rev=None)
        line = buf.getvalue()
        self.assertIn("no bandwidth", line)
        self.assertIn("no rtt", line)

    def test_a_near_miss_summary_line_is_not_read_as_pings_rtt(self):
        """parse_ping takes free text, so the four numbers must be anchored to ping's own
        `rtt` summary rather than to any line that happens to carry min/avg/max/mdev --
        an mtr report or a paraphrased log line would otherwise be parsed as a measurement
        of this link."""
        self.assertIsNone(
            lr.parse_ping("min/avg/max/mdev = 1.000/2.000/3.000/4.000 ms").get("rtt_ms"),
            "a line without ping's `rtt` prefix was read as an RTT measurement")


if __name__ == "__main__":
    unittest.main(verbosity=2)
