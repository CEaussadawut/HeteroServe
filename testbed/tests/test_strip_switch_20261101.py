"""strip_switch_20261101.py (the scheduler review's item 7, 2026-11-01): the boards it may switch, how each goes down, the
parallel runner, and the Jetson's guard check -- with the strip and every board faked; nothing is contacted.

    python3 -m unittest discover -s testbed/tests -q
"""
import json, os, sys, threading, time, unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
import strip_switch_20261101 as W                              # noqa: E402


class Boards(unittest.TestCase):
    def test_the_three_boards_never_the_laptop(self):
        self.assertEqual(sorted(W.ALLOWED), ["jetson", "pi4", "pi5"])
        cfg = json.load(open(W.CFG))
        laptop = [k for k, v in cfg["outlets"].items() if v == "laptop"]
        self.assertTrue(laptop and laptop[0] in cfg["do_not_switch_off"])        # the strip config protects it too
        with self.assertRaises(SystemExit):
            W.outlet("laptop")

    def test_the_jetson_is_halted_and_has_no_overlay(self):
        self.assertEqual(W.ALLOWED["jetson"]["cmd"], "halt")                      # a poweroff made this Nano boot back up
        self.assertEqual({b: W.ALLOWED[b]["cmd"] for b in ("pi4", "pi5")}, {"pi4": "poweroff", "pi5": "poweroff"})
        self.assertEqual(W.address(W.ALLOWED["jetson"]), "192.168.1.114")
        self.assertEqual(W.address(W.ALLOWED["pi5"]), "192.168.10.11")
        self.assertTrue(W.ALLOWED["jetson"]["guard"])
        self.assertFalse(W.ALLOWED["pi4"]["guard"] or W.ALLOWED["pi5"]["guard"])

    def test_addresses_match_serve_pool(self):
        s = open(os.path.join(ROOT, "testbed", "scripts", "serve_pool.sh")).read()
        for b, a in W.ALLOWED.items():
            self.assertIn("[%s]=%s" % (b, a["lan"]), s, b)                         # LAN=( ... ) in serve_pool.sh
        self.assertIn("[pi5]=192.168.10.11", s)
        self.assertIn("[pi4]=192.168.10.12", s)

    def test_the_command_line_refuses_another_board(self):
        with mock.patch.object(sys, "argv", ["x", "off-node", "laptop"]):
            with self.assertRaises(SystemExit):
                W.main()


class Parallel(unittest.TestCase):
    def test_each_runs_at_once_and_needs_every_board(self):
        started, lock = [], threading.Lock()

        def slow(b, path):
            with lock:
                started.append((b, time.time()))
            time.sleep(0.3)
            return 0 if b != "jetson" else 1
        t0 = time.time()
        self.assertEqual(W.each(slow, ["pi4", "pi5"], None), 0)
        self.assertLess(time.time() - t0, 0.55)                                   # two boards, one board's time
        self.assertEqual(W.each(slow, ["pi4", "jetson"], None), 1)                # one board failing fails the call

    def test_a_refusal_in_one_thread_is_logged_not_raised(self):
        def refuse(b, path):
            raise SystemExit("refused: test")
        with mock.patch.object(W, "log_line") as ll:
            self.assertEqual(W.each(refuse, ["pi5"], None), 1)
        self.assertIn("refused", ll.call_args[0][1]["result"])


class Guard(unittest.TestCase):
    def fake_ssh(self, answers):
        calls = []

        def ssh(board, cmd, timeout=15):
            calls.append(cmd)
            for key, out in answers:
                if key in cmd:
                    return 0, out
            return 0, ""
        return ssh, calls

    def test_a_live_guard_is_checked_not_restarted(self):
        ssh, calls = self.fake_ssh([("nvpmodel", "pmode:0000 1 0 2"), ("wc -l", "1")])   # the status line also counts: first
        with mock.patch.object(W, "ssh", ssh), mock.patch.object(W, "log_line"):
            self.assertEqual(W.guard("jetson", None, repo_sha="x"), 0)
        self.assertFalse(any("setsid" in c for c in calls))

    def test_a_missing_guard_is_started_after_its_prechecks(self):
        ssh, calls = self.fake_ssh([("nvpmodel", "pmode:0000 1 0 1"), ("sha256sum", "synced abc"), ("wc -l", "0")])
        with mock.patch.object(W, "ssh", ssh), mock.patch.object(W, "log_line"), mock.patch.object(W.time, "sleep"):
            self.assertEqual(W.guard("jetson", None, repo_sha="abc"), 0)
        self.assertTrue(any("setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh" in c for c in calls))

    def test_refuses_a_stale_script_a_trip_or_another_mode(self):
        for answers in ([("sha256sum", "synced OTHER"), ("wc -l", "0")],                     # the board's script differs
                        [("nvpmodel", "pmode:0000 1 2 1"), ("wc -l", "1")],                  # a TRIP since the guard started
                        [("nvpmodel", "pmode:0001 1 0 1"), ("wc -l", "1")],                  # not MAXN
                        [("nvpmodel", "pmode:0000 1 0 40"), ("wc -l", "1")]):                # the guard's log is stale
            ssh, _ = self.fake_ssh(answers)
            with mock.patch.object(W, "ssh", ssh), mock.patch.object(W, "log_line"), mock.patch.object(W.time, "sleep"):
                self.assertEqual(W.guard("jetson", None, repo_sha="abc"), 1, answers)
        with self.assertRaises(SystemExit):
            W.guard("pi5", None)


if __name__ == "__main__":
    unittest.main()
