"""The switch-off automation (serve_switch_20261024.py, 2026-10-24), an opt-in that is NOT run live: its decisions, its plan
(the same call serve_pool.sh makes), and its dry run, which contacts nothing when --assume-pi4 gives the outlet's state."""
import io, json, os, subprocess, sys, tempfile, unittest
from contextlib import redirect_stdout
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "testbed", "scripts"), os.path.join(ROOT, "ThinkPad")]
import serve_switch_20261024 as S  # noqa: E402

MEM = "testbed/dataset/memory_snapshot_20261024_142953.json"      # so4's snapshot
SO4 = {"MODEL": "qwen3_32b_q6k", "CTX": "4096", "MEMFILE": MEM, "PLACE": "laptop:45,pi5:18,jetson:1", "PLACE_HEAD": "host"}


class Decisions(unittest.TestCase):
    def test_power_up_only_when_needed(self):
        self.assertEqual(S.before_start(["laptop", "pi4"], False), [("on-node", "pi4")])
        self.assertEqual(S.before_start(["laptop", "pi4"], True), [])
        self.assertEqual(S.before_start(["laptop", "pi5"], False), [])

    def test_switch_off_only_an_idle_board_out_of_the_plan(self):
        self.assertEqual(S.after_start(["laptop", "pi5", "jetson"], True, ""), [("off-node", "pi4")])
        self.assertEqual(S.after_start(["laptop", "pi4"], True, ""), [])
        self.assertEqual(S.after_start(["laptop", "pi5"], False, ""), [])
        self.assertEqual(S.after_start(None, True, ""), [])
        self.assertEqual(S.after_start(["laptop", "pi5"], True, "ggml-rpc-server")[0][0], "leave-on")

    def test_only_the_pi4(self):
        self.assertEqual(S.BOARD, "pi4")
        import strip_switch_20261024 as W
        self.assertEqual(sorted(W.ALLOWED), ["pi4"])


class Outlet(unittest.TestCase):
    """The outlet's state as strip_switch_20261024.py status prints it (a top-level "output"), parsed without the strip."""
    def run_status(self, stdout, rc=0):
        done = subprocess.CompletedProcess(["x"], rc, stdout=stdout, stderr="")
        with mock.patch.object(subprocess, "run", return_value=done) as m:
            v = S.outlet_on()
        self.assertEqual(m.call_args[0][0][-2:], ["status", "pi4"])
        return v

    def test_on_and_off(self):
        self.assertIs(self.run_status('{"output": true, "apower": 2.3, "source": "init", "outlet": 0, "board": "pi4"}\n'), True)
        self.assertIs(self.run_status('{"output": false, "apower": 0.0, "source": "HTTP_in", "outlet": 0, "board": "pi4"}\n'), False)

    def test_unreadable_refuses(self):
        for out in ("", "refused: the device at x has MAC y\n", '{"output": null, "board": "pi4"}\n',
                    '{"output": true, "board": "pi5"}\n'):
            with self.assertRaises(SystemExit):
                self.run_status(out, 1 if out.startswith("refused") else 0)

    def test_assumed_state_contacts_nothing(self):
        with mock.patch.object(subprocess, "run", side_effect=AssertionError("contacted")):
            self.assertIs(S.outlet_on("on"), True)
            self.assertIs(S.outlet_on("off"), False)


class Plan(unittest.TestCase):
    def test_so4_plan_leaves_the_pi4_out(self):
        members, r = S.planned_members(SO4)
        self.assertEqual(members, ["laptop", "pi5", "jetson"])
        self.assertEqual(r["head_at"], 0)

    def test_a_plan_with_the_pi4(self):
        members, _ = S.planned_members(dict(SO4, MODEL="tinyllama_q4_0", CTX="2048", PLACE="laptop:11,pi4:11"))
        self.assertIn("pi4", members)

    def test_state_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".state", delete=False) as f:
            f.write("STARTED=2026-10-24T20:00:00+07:00\nHOST=laptop\nWORKERS=pi5\\ jetson\nARGS=x\nCTX=4096\nLOG=y\n")
        try:
            self.assertEqual(S.served_members(f.name), ["laptop", "pi5", "jetson"])
        finally:
            os.unlink(f.name)
        self.assertIsNone(S.served_members("/nonexistent/serve_pool.state"))


class DryRun(unittest.TestCase):
    def run_main(self, argv, env):
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["serve_switch_20261024.py"] + argv), mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(S, "served_members", return_value=None), \
                mock.patch.object(subprocess, "run", side_effect=AssertionError("a dry run with --assume-pi4 contacts nothing")), \
                redirect_stdout(out):
            rc = S.main()
        return rc, out.getvalue()

    def test_dry_run_switches_off_an_unplanned_pi4(self):
        rc, out = self.run_main(["start", "--dry-run", "--assume-pi4", "on"], SO4)
        self.assertEqual(rc, 0)
        self.assertIn("DRY-RUN would run: serve start", out)
        self.assertIn("DRY-RUN would run: off-node pi4", out)
        self.assertNotIn("on-node", out)

    def test_dry_run_powers_up_a_planned_pi4(self):
        rc, out = self.run_main(["start", "--dry-run", "--assume-pi4", "off"],
                                dict(SO4, MODEL="tinyllama_q4_0", CTX="2048", PLACE="laptop:11,pi4:11"))
        self.assertEqual(rc, 0)
        lines = [x for x in out.splitlines() if x.startswith("DRY-RUN")]
        self.assertEqual(lines, ["DRY-RUN would run: on-node pi4", "DRY-RUN would run: serve start"])

    def test_refuses_while_something_is_served(self):
        with mock.patch.object(sys, "argv", ["x", "start", "--dry-run", "--assume-pi4", "on"]), \
                mock.patch.object(S, "served_members", return_value=["laptop", "pi5"]):
            with self.assertRaises(SystemExit):
                S.main()


if __name__ == "__main__":
    unittest.main()
