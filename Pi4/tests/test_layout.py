"""Structural tests for the shell scripts, systemd units and the Makefile.

None of these can run the scripts -- that needs a Raspberry Pi. What they can do is catch
the drift that only shows up at install time on the hardware, when the board is on the
bench and the sweep was meant to start an hour ago: a unit pointing at a renamed script, a
Makefile target calling a file that no longer exists, a doc referencing a path that was
never written, a sudoers rule that quietly grants more than the agent needs.

    python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Every unit hardcodes this, and docs/01 rsyncs the folder there.
DEPLOY_ROOT = "/opt/heteroserve/Pi4"
DATA_ROOT = "/opt/heteroserve/data"


def find_bash():
    candidates = [shutil.which("bash")]
    if os.name == "nt":
        candidates += [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
        ]
    for cand in candidates:
        if not cand or "system32" in cand.lower():
            continue
        try:
            proc = subprocess.run([cand, "-c", "echo ok"], capture_output=True,
                                  text=True, timeout=20)
        except OSError:
            continue
        if proc.returncode == 0 and proc.stdout.strip() == "ok":
            return cand
    return None


BASH = find_bash()
SCRIPTS = sorted(ROOT.glob("scripts/*.sh"))
UNITS = sorted(ROOT.glob("systemd/*.service"))


class TestScripts(unittest.TestCase):
    def test_the_expected_scripts_are_all_present(self):
        names = {p.name for p in SCRIPTS}
        expected = {
            "lib.sh", "00_bootstrap.sh", "01_system_tune.sh", "02_build_llama_cpp.sh",
            "03_fetch_models.sh", "04_rpc_server.sh", "05_net_tier.sh",
            "06_bench_local.sh", "07_rpc_smoke.sh", "99_healthcheck.sh",
        }
        self.assertEqual(names, expected)

    @unittest.skipUnless(BASH, "no usable bash found")
    def test_every_script_parses(self):
        for path in SCRIPTS:
            with self.subTest(script=path.name):
                proc = subprocess.run([BASH, "-n", path.as_posix()],
                                      capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_every_runnable_script_sources_lib(self):
        # lib.sh is what loads node.env and defines mark()/die()/assert_clean_state. A
        # script that forgets it runs with an empty configuration and no phase markers.
        for path in SCRIPTS:
            if path.name == "lib.sh":
                continue
            with self.subTest(script=path.name):
                self.assertIn("lib.sh", path.read_text(encoding="utf-8"))

    def test_lib_sets_strict_mode(self):
        self.assertIn("set -euo pipefail", (ROOT / "scripts" / "lib.sh").read_text(encoding="utf-8"))

    def test_no_script_re_enables_swap(self):
        # Swap off is a measurement decision, not a tuning preference: with swap on, an
        # overload measures the microSD card's latency tail instead of the CPU's.
        for path in SCRIPTS:
            text = path.read_text(encoding="utf-8")
            with self.subTest(script=path.name):
                self.assertNotIn("swapon", text)

    def test_no_script_overclocks_or_otherwise_speeds_the_node_up(self):
        # The Pi 4 is the designated straggler for research question QR8. A quietly
        # optimised Pi 4 stops being the node the study is about.
        forbidden = ("arm_freq=", "over_voltage", "force_turbo", "gpu_freq=")
        for path in SCRIPTS:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                with self.subTest(script=path.name, token=token):
                    self.assertNotIn(token, text)

    def test_the_readiness_gate_requires_the_WHOLE_model_set(self):
        """"Any .gguf exists" is not the question.

        scripts/03_fetch_models.sh warns and continues on a failed download, and the sweep
        warns and skips a missing model -- so a partial fetch yields a dataset with fewer
        rows than the matrix and nothing recording why. The gate has to compare against
        config/models.txt, not just look for one file.
        """
        text = (ROOT / "scripts" / "99_healthcheck.sh").read_text(encoding="utf-8")
        self.assertIn("missing_models", text,
                      "the readiness gate must check every model in models.txt")
        self.assertNotIn('ls "$DATA_DIR/models"/*.gguf', text,
                         "the weak 'any .gguf exists' check must not come back")

    def test_stopping_the_rpc_server_does_not_short_circuit_the_process_kill(self):
        """`systemctl stop` returns 0 even when the unit was already inactive, so
        `systemctl stop ... || pkill ...` left a hand-started server running while
        reporting success. A stale process on the port contaminates every row of a sweep
        on the same cores and the same rail."""
        text = (ROOT / "scripts" / "04_rpc_server.sh").read_text(encoding="utf-8")
        self.assertNotIn('systemctl stop heteroserve-rpc 2>/dev/null || pkill', text,
                         "stop must not short-circuit past the process kill")
        self.assertIn("still listening", text.lower(),
                      "stop must verify the port actually closed")

    def test_the_healthcheck_exits_non_zero_when_a_check_fails(self):
        # A readiness gate that exits 0 on failure is worse than no gate at all: the
        # harness would record measurements from a node nobody certified.
        text = (ROOT / "scripts" / "99_healthcheck.sh").read_text(encoding="utf-8")
        self.assertRegex(text, r"exit\s+\"?\$\{?fail")

    def test_net_tier_always_offers_a_way_to_clear_shaping(self):
        # A forgotten netem qdisc survives until removed and silently degrades the *next*
        # experiment -- the hardest kind of error to find afterwards.
        text = (ROOT / "scripts" / "05_net_tier.sh").read_text(encoding="utf-8")
        self.assertIn("clear)", text)
        self.assertIn("tc qdisc del", text)


class TestSystemdUnits(unittest.TestCase):
    def test_the_expected_units_are_present(self):
        self.assertEqual(
            {p.name for p in UNITS},
            {"heteroserve-agent.service", "heteroserve-rpc.service",
             "heteroserve-tune.service"},
        )

    def test_each_unit_has_the_required_sections(self):
        for path in UNITS:
            text = path.read_text(encoding="utf-8")
            with self.subTest(unit=path.name):
                self.assertIn("[Unit]", text)
                self.assertIn("[Service]", text)
                self.assertIn("[Install]", text)
                self.assertIn("ExecStart=", text)

    def test_every_exec_and_documentation_path_maps_to_a_file_in_this_repo(self):
        # The units hardcode /opt/heteroserve/Pi4/...; this repo is what gets rsynced
        # there. A renamed script makes the unit fail at boot with a bare 203/EXEC.
        pattern = re.compile(re.escape(DEPLOY_ROOT) + r"[/\w.-]+")
        for path in UNITS:
            for ref in pattern.findall(path.read_text(encoding="utf-8")):
                rel = ref[len(DEPLOY_ROOT) + 1:]
                with self.subTest(unit=path.name, ref=ref):
                    self.assertTrue(
                        (ROOT / rel).exists(),
                        "%s references %s, which does not exist in this folder"
                        % (path.name, ref),
                    )

    def test_log_destinations_all_live_under_the_collected_data_dir(self):
        # One rsync of $DATA_DIR must pull this node's whole contribution back to the
        # control node. A log written elsewhere is a log nobody ever collects.
        for path in UNITS:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith(("StandardOutput=append:", "StandardError=append:")):
                    with self.subTest(unit=path.name, line=line):
                        self.assertIn(DATA_ROOT, line)

    def test_the_rpc_unit_caps_memory_below_the_physical_total(self):
        # No swap by design, so an unbounded rpc-server takes the whole board down with
        # the OOM killer mid-measurement. Failing one service keeps the node reachable.
        text = (ROOT / "systemd" / "heteroserve-rpc.service").read_text(encoding="utf-8")
        self.assertRegex(text, r"MemoryMax=\S+")

    def test_the_rpc_unit_is_not_enabled_at_boot_by_the_install_target(self):
        # The control node starts the worker per run. A worker that came up at boot would
        # be holding memory during the isolated characterisation sweep.
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        enable_lines = [l for l in makefile.splitlines() if "systemctl enable" in l]
        self.assertTrue(enable_lines)
        for line in enable_lines:
            self.assertNotIn("heteroserve-rpc", line)


class TestSudoers(unittest.TestCase):
    """The agent runs unprivileged and needs exactly one privilege: start and stop the
    rpc-server unit for the control node."""

    def setUp(self):
        self.text = (ROOT / "systemd" / "heteroserve-agent.sudoers").read_text(encoding="utf-8")
        # Comments mention paths in prose ("systemctl is /bin/systemctl on Raspberry Pi
        # OS"); only the directives grant anything.
        self.rules = "\n".join(
            line for line in self.text.splitlines() if not line.lstrip().startswith("#")
        )

    def test_grants_nothing_but_the_rpc_unit(self):
        commands = re.findall(r"/\S*bin/systemctl\s+\w+\s+(\S+)", self.rules)
        self.assertTrue(commands, "no systemctl commands found in the sudoers file")
        for unit in commands:
            self.assertEqual(unit.rstrip(","), "heteroserve-rpc")

    def test_covers_both_systemctl_paths(self):
        # systemctl is /bin/systemctl on Raspberry Pi OS and /usr/bin/systemctl on some
        # images; sudo matches on the literal path, so both have to be listed.
        self.assertIn("/bin/systemctl", self.text)
        self.assertIn("/usr/bin/systemctl", self.text)

    def test_does_not_grant_a_blanket_all(self):
        self.assertNotRegex(self.text, r"NOPASSWD:\s*ALL\b")

    def test_the_agent_unit_user_matches_the_sudoers_principal(self):
        agent = (ROOT / "systemd" / "heteroserve-agent.service").read_text(encoding="utf-8")
        user = re.search(r"^User=(\S+)", agent, re.M)
        self.assertIsNotNone(user, "the agent unit does not set User=")
        principal = re.search(r"^(\S+)\s+ALL=\(root\)", self.text, re.M)
        self.assertIsNotNone(principal, "the sudoers file names no principal")
        self.assertEqual(
            user.group(1), principal.group(1),
            "the agent runs as %r but sudo grants %r -- POST /rpc/start will be denied"
            % (user.group(1), principal.group(1)),
        )


class TestMakefile(unittest.TestCase):
    def setUp(self):
        self.text = (ROOT / "Makefile").read_text(encoding="utf-8")

    def test_every_script_a_recipe_calls_exists(self):
        for ref in re.findall(r"scripts/[\w.]+\.sh", self.text):
            with self.subTest(ref=ref):
                self.assertTrue((ROOT / ref).exists(), "Makefile calls missing %s" % ref)

    def test_every_python_module_a_recipe_calls_exists(self):
        for ref in re.findall(r"node/[\w.]+\.py", self.text):
            with self.subTest(ref=ref):
                self.assertTrue((ROOT / ref).exists(), "Makefile calls missing %s" % ref)

    def test_the_bring_up_targets_are_all_declared(self):
        declared = set(re.findall(r"^([a-z0-9_-]+):", self.text, re.M))
        for target in ("bootstrap", "tune", "build", "models", "install", "health",
                       "info", "rpc", "bench", "bench-quick"):
            with self.subTest(target=target):
                self.assertIn(target, declared)

    def test_every_declared_target_is_phony(self):
        declared = set(re.findall(r"^([a-z0-9_-]+):", self.text, re.M))
        phony = set(re.search(r"^\.PHONY:(.*)$", self.text, re.M).group(1).split())
        missing = sorted(declared - phony - {".PHONY"})
        self.assertEqual(missing, [], "targets missing from .PHONY: %s" % missing)

    def test_every_target_carries_a_help_string(self):
        # `make help` greps for the ## marker; a target without one is invisible.
        for line in self.text.splitlines():
            match = re.match(r"^([a-z0-9_-]+):(?!=)", line)
            if match and match.group(1) != "help":
                with self.subTest(target=match.group(1)):
                    self.assertIn("##", line, "target %r has no ## help text"
                                  % match.group(1))


class TestDocs(unittest.TestCase):
    def test_referenced_repo_paths_exist(self):
        # Catches a doc that points at a file which was renamed or never written.
        pattern = re.compile(r"`((?:scripts|node|config|systemd|docs)/[\w./-]+)`")
        missing = []
        for doc in sorted(ROOT.glob("docs/*.md")) + [ROOT / "README.md"]:
            for ref in pattern.findall(doc.read_text(encoding="utf-8")):
                if ref.endswith("/"):
                    continue
                if not (ROOT / ref).exists():
                    missing.append("%s -> %s" % (doc.name, ref))
        self.assertEqual(missing, [], "documentation points at files that do not exist: %s"
                         % missing)

    def test_the_documented_bring_up_order_matches_the_makefile(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        order = [m for m in re.findall(r"^make (\S+)", readme, re.M)]
        for target in ("bootstrap", "tune", "build", "models", "install", "health"):
            self.assertIn(target, order, "README never tells the operator to run make %s"
                          % target)
        self.assertLess(order.index("tune"), order.index("build"))
        self.assertLess(order.index("build"), order.index("models"))
        self.assertLess(order.index("install"), order.index("health"))


if __name__ == "__main__":
    unittest.main()
