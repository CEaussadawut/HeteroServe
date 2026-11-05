"""serve_pool.sh's serving default (the user, 2026-10-10: "Laptop draft"): a person at a terminal serving Qwen3-14B gets the
laptop draft; a harness keeps what its registration names. Tests serve_defaults.sh's functions in bash; stdlib unittest only.

    python3 -m unittest discover -s testbed/tests -q
"""
import json, os, re, subprocess, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEF = os.path.join(ROOT, "testbed", "scripts", "serve_defaults.sh")


def bash(cmd):
    return subprocess.run(["bash", "-c", ". '%s'; %s" % (DEF, cmd)], capture_output=True, text=True, check=True).stdout.strip()


class TestServeDefaults(unittest.TestCase):
    def test_who_gets_the_draft(self):
        cases = {("__unset__", "qwen3_14b", "1"): "laptop",      # a person serving the measured target
                 ("__unset__", "qwen3_14b", "0"): "none",        # a harness: as registered (spec_serve's base arm has no env)
                 ("__unset__", "tinyllama", "1"): "none",        # a target the draft was never measured with
                 ("default", "qwen3_14b", "0"): "laptop",        # asked for, from anywhere
                 ("default", "qwen3moe", "1"): "incompatible",
                 ("none", "qwen3_14b", "1"): "none", ("", "qwen3_14b", "1"): "none",
                 ("-md x.gguf -ngld 0", "qwen3_14b", "1"): "custom"}
        for (given, model, tty), want in cases.items():
            self.assertEqual(bash("draft_mode '%s' %s %s" % (given, model, tty)), want, (given, model, tty))

    def test_the_laptop_draft_is_the_measured_arm(self):
        a = bash("laptop_draft_args 1")
        self.assertEqual(a, "-md ~/models/qwen3-0.6b-q8_0.gguf --spec-type draft-simple --spec-draft-n-max 1 "
                            "--spec-draft-n-min 0 --spec-draft-p-min 0 -devd RPC1 -ngld 99")   # spec_serve's lapd1 arm
        self.assertIn("-devd RPC2", bash("laptop_draft_args 2"))   # after two planned workers
        self.assertEqual(bash("echo $LAPTOP_LAN:$LAPTOP_RPC_PORT $LAPTOP_DRAFT_HOST_MIB"), "192.168.1.108:50053 160")

    def test_the_boards_a_person_gets(self):
        # 2026-11-01 (the scheduler review's item 1): the laptop too from a terminal; a harness keeps the old default
        self.assertEqual(bash("default_nodes 1"), "laptop pi5 pi4 jetson")
        self.assertEqual(bash("default_nodes 0"), "pi5 pi4 jetson")
        with open(os.path.join(ROOT, "testbed", "scripts", "serve_pool.sh")) as f:
            s = f.read()
        line = 'NODES=${NODES:-$(default_nodes "$INTERACTIVE")}; export NODES'
        self.assertIn(line, s)
        self.assertLess(s.index("INTERACTIVE=0; [ -t 1 ] && INTERACTIVE=1"), s.index(line))   # decided at the terminal
        self.assertLess(s.index(line), s.index("exec > >("))                                   # before start's tee

    def test_serve_pool_reads_the_terminal_before_its_tee_and_full_eval_says_none(self):
        with open(os.path.join(ROOT, "testbed", "scripts", "serve_pool.sh")) as f:
            s = f.read()
        self.assertIn(". testbed/scripts/serve_defaults.sh", s)
        self.assertLess(s.index("INTERACTIVE=0; [ -t 1 ] && INTERACTIVE=1"), s.index("exec > >("))
        self.assertIn('DRAFT_GIVEN=${SERVE_DRAFT-__unset__}', s)          # unset and empty differ: "" means none
        with open(os.path.join(ROOT, "testbed", "scripts", "full_eval.py")) as f:
            self.assertIn('SERVE_DRAFT="none"', f.read())


class TestMtpDefault(unittest.TestCase):
    """MTP 1 on pooled plans (the user, 2026-11-04: "Turn MTP 1 on"): a person at a terminal, for the models MTP was measured
    with; a harness never; speculation asked for another way is served as given. serve_pool.sh decides "pooled"
    (test_planner_v4.py)."""

    def test_who_gets_it(self):
        u = "__unset__"
        cases = {(u, "qwen35_9b", "1", u, ""): "on",                   # a person, the 9B (sp10c, sp10e)
                 (u, "qwen36moe_q6k", "1", u, ""): "on",               # a person, the 35B (sp10, sp10c)
                 (u, "qwen35_9b", "0", u, ""): "off",                  # a harness: as registered
                 (u, "qwen38_27b", "1", u, ""): "off",                 # has the head, never measured with it
                 (u, "qwen3moe", "1", u, ""): "off",                   # no head
                 ("on", "qwen35_9b", "0", u, ""): "on",                # asked for, from anywhere
                 ("on", "qwen3moe", "0", u, ""): "off",
                 ("off", "qwen35_9b", "1", u, ""): "off", ("", "qwen35_9b", "1", u, ""): "off",
                 (u, "qwen35_9b", "1", "none", ""): "off",             # SERVE_DRAFT given: no speculation by default
                 (u, "qwen35_9b", "1", u, "--spec-type draft-mtp --spec-draft-n-max 3"): "off",   # served as given
                 (u, "qwen35_9b", "1", u, "--backend-sampling"): "on"}
        for (given, model, tty, draft, extra), want in cases.items():
            self.assertEqual(bash("mtp_mode '%s' %s %s '%s' '%s'" % (given, model, tty, draft, extra)), want,
                             (given, model, tty, draft, extra))

    def test_the_default_is_sp10es_mtp1_arm(self):
        with open(os.path.join(ROOT, "testbed", "dataset", "sp10e_predictions_20261103.json")) as f:
            reg = json.load(f)
        self.assertEqual(bash("echo $MTP_DEFAULT_ARGS"), reg["spec_args"]["mtp1"])
        self.assertEqual(bash("echo $MTP_MODELS"), "qwen35_9b qwen36moe_q6k")


class TestSwitchDefault(unittest.TestCase):
    """The Pi 4's switch-off default (the user, 2026-10-25: "Interactive default"): a person at a terminal gets it, a harness
    never; it powers the board up only for a plan that needs it while its outlet is off, and switches it off only when the
    served plan leaves it out, its outlet is on and it runs nothing."""

    def test_who_gets_it(self):
        cases = {("__unset__", "1"): "on", ("__unset__", "0"): "off", ("on", "0"): "on", ("on", "1"): "on",
                 ("off", "1"): "off", ("", "1"): "off", ("yes", "1"): "off"}
        for (given, tty), want in cases.items():
            self.assertEqual(bash("switch_mode '%s' %s" % (given, tty)), want, (given, tty))

    def test_every_board_the_plan_can_leave_out(self):
        # 2026-11-01 (the scheduler review's item 7): the Pi 4, the Pi 5 and the Jetson, through the tool that may switch them
        self.assertEqual(bash("echo $SWITCH_BOARDS"), "pi4 pi5 jetson")
        self.assertEqual(bash("echo $SWITCH_TOOL"), "testbed/scripts/strip_switch_20261101.py")

    def test_before_and_after(self):
        before = {("laptop pi4", "pi4", "off"): "power-up", ("laptop pi4", "pi4", "on"): "", ("laptop pi5", "pi4", "off"): "",
                  ("laptop pi4", "pi4", "?"): "", ("laptop pi5 pi4 jetson", "jetson", "off"): "power-up",
                  ("laptop pi5", "pi5", "off"): "power-up", ("laptop", "jetson", "off"): ""}
        for (members, board, outlet), want in before.items():
            self.assertEqual(bash("switch_before '%s' %s '%s'" % (members, board, outlet)), want, (members, board, outlet))
        after = {("laptop pi5", "pi4", "on", ""): "switch-off", ("laptop", "jetson", "on", ""): "switch-off",
                 ("laptop pi5", "pi4", "on", "ggml-rpc-server "): "leave-on", ("laptop pi4", "pi4", "on", ""): "",
                 ("laptop pi5", "pi4", "off", ""): "", ("laptop pi5", "jetson", "?", ""): "",
                 ("laptop pi5 jetson", "jetson", "on", ""): ""}
        for (members, board, outlet, busy), want in after.items():
            self.assertEqual(bash("switch_after '%s' %s '%s' '%s'" % (members, board, outlet, busy)), want,
                             (members, board, outlet, busy))

    def test_serve_pool_wiring(self):
        with open(os.path.join(ROOT, "testbed", "scripts", "serve_pool.sh")) as f:
            s = f.read()
        self.assertIn("SWITCH_GIVEN=${SERVE_SWITCH-__unset__}", s)
        self.assertLess(s.index("SWITCH_GIVEN="), s.index("exec > >("))        # read with the terminal, before the tee
        up, gates = s.index("switch_before"), s.index("for n in $HOST $WORKERS; do")
        self.assertLess(up, gates)                                              # powered up before the boards are gated
        self.assertLess(s.index('"$SWITCH_TOOL" guard jetson'), gates)         # the Jetson's guard before its gate
        self.assertGreater(s.index("switch_after"), s.index('echo "READY'))    # switched off only once the model is served
        # a harness never reaches the strip: both steps are behind SWITCH = on
        self.assertIn('if [ "$SWITCH" = on ]; then   # every board the plan needs', s)
        self.assertIn('if [ "$SWITCH" = on ]; then   # served without them', s)
        self.assertIn('"$SWITCH_TOOL" on-node $up', s)                         # all at once, one call
        self.assertIn('"$SWITCH_TOOL" off-node $down', s)


if __name__ == "__main__":
    unittest.main()
