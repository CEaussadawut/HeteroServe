"""Tests for node/envfile.py and the real config/node.env.

`config/node.env` is the single source of truth for this node and it is read twice, by
two different languages: bash sources it, and this module parses it. Those two readings
must agree. If they ever disagree, the shell scripts and the Python agent are configuring
different machines while both appear to work -- for example the sweep enforcing one layer
budget while `device_info.json` reports another to the placement engine.

    python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "node"))

import envfile  # noqa: E402

NODE_ENV = ROOT / "config" / "node.env"


def find_bash():
    """An absolute path to a bash that can actually source a file, or None.

    On the Pi this is just /usr/bin/bash. On a Windows dev box a bare "bash" in
    subprocess resolves through CreateProcess's search order to the WSL stub in
    System32, which answers "no installed distributions" on stderr and exits 1 -- a
    failure that looks like the config is broken when nothing is wrong with it. So the
    interpreter is resolved explicitly and proved to work before any test relies on it.
    """
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


class TestParser(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "node.env"

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, text):
        self.path.write_text(text, encoding="utf-8")
        return envfile.load(self.path)

    def test_comments_and_blank_lines_are_ignored(self):
        env = self.write("# a comment\n\n  # indented comment\nA=1\n")
        self.assertEqual(env, {"A": "1"})

    def test_double_and_single_quotes_are_stripped(self):
        env = self.write('A="hello world"\nB=\'single\'\n')
        self.assertEqual(env["A"], "hello world")
        self.assertEqual(env["B"], "single")

    def test_a_value_containing_equals_survives_intact(self):
        # LLAMA_ARCH_FLAGS is exactly this shape: -march=... -mtune=...
        env = self.write('F="-march=armv8-a+crc+simd -mtune=cortex-a72"\n')
        self.assertEqual(env["F"], "-march=armv8-a+crc+simd -mtune=cortex-a72")

    def test_mismatched_quotes_are_left_alone(self):
        env = self.write("A=\"unterminated\n")
        self.assertEqual(env["A"], '"unterminated')

    def test_a_line_without_equals_is_skipped_not_fatal(self):
        env = self.write("this is not a setting\nA=1\n")
        self.assertEqual(env, {"A": "1"})

    def test_shell_environment_overrides_a_declared_key(self):
        # Documented behaviour: RPC_MEM_MB=1800 scripts/04_rpc_server.sh must work for a
        # one-off without editing the file.
        self.path.write_text("RPC_MEM_MB=2600\n", encoding="utf-8")
        os.environ["RPC_MEM_MB"] = "1800"
        try:
            self.assertEqual(envfile.load(self.path)["RPC_MEM_MB"], "1800")
        finally:
            del os.environ["RPC_MEM_MB"]

    def test_shell_environment_cannot_introduce_an_undeclared_key(self):
        # The override is deliberately restricted to keys the file already declares, so a
        # stray variable in the operator's shell cannot invent configuration.
        self.path.write_text("A=1\n", encoding="utf-8")
        os.environ["HETEROSERVE_NOT_A_REAL_KEY"] = "x"
        try:
            self.assertNotIn("HETEROSERVE_NOT_A_REAL_KEY", envfile.load(self.path))
        finally:
            del os.environ["HETEROSERVE_NOT_A_REAL_KEY"]

    def test_default_path_is_the_real_node_env(self):
        self.assertEqual(envfile.load()["NODE_ID"], "pi4")


class TestRealConfig(unittest.TestCase):
    """Assertions about config/node.env itself. These encode the proposal's facts about
    this node, so a careless edit to the config fails a test instead of quietly changing
    what the dataset means."""

    def setUp(self):
        self.env = envfile.load(NODE_ENV)

    def test_identity_is_the_straggler_worker(self):
        self.assertEqual(self.env["NODE_ID"], "pi4")
        self.assertEqual(self.env["NODE_ROLE"], "worker-straggler")
        self.assertEqual(self.env["NODE_ARCH"], "aarch64")
        self.assertEqual(self.env["NODE_CPU"], "Cortex-A72")
        self.assertEqual(self.env["NODE_SOC"], "BCM2711")

    def test_every_key_the_code_reads_is_present(self):
        # Collected by grepping ENV.get(...) across node/ and $VAR across scripts/.
        required = [
            "NODE_ID", "NODE_ROLE", "NODE_DESC", "NODE_CPU", "NODE_SOC",
            "NODE_CORES", "NODE_RAM_MB", "NODE_USABLE_MB",
            "ETH_IFACE", "ETH_ADDR", "WLAN_IFACE", "CONTROL_HOST", "CONTROL_NTP",
            "LLAMA_REPO", "LLAMA_SRC", "LLAMA_BIN", "LLAMA_ARCH_FLAGS", "LLAMA_THREADS",
            "RPC_HOST", "RPC_PORT", "RPC_MEM_MB", "RPC_CACHE_DIR", "RPC_THREADS",
            "AGENT_PORT",
            "INA_CHANNEL", "INA_I2C_ADDR", "INA_SHUNT_OHMS", "INA_MAX_A", "PSU_MODEL",
            "DATA_DIR",
        ]
        missing = [k for k in required if k not in self.env]
        self.assertEqual(missing, [], "config/node.env is missing %r" % missing)

    def test_the_rpc_thread_count_is_explicit_and_fits_the_board(self):
        """Upstream defaults rpc-server to hardware_concurrency()/2 -- 2 of this board's 4
        cores. A cluster run would then serve at a thread count the Phase 2 sweep never
        measured at, and nothing would record it. scripts/lib.sh passes -t explicitly, so
        this key must exist and must be sane."""
        threads = int(self.env["RPC_THREADS"])
        cores = int(self.env["NODE_CORES"])
        self.assertGreaterEqual(threads, 1)
        self.assertLessEqual(threads, cores,
                             "RPC_THREADS=%d exceeds NODE_CORES=%d" % (threads, cores))

    def test_llama_pin_sha_key_exists_even_while_empty(self):
        # It is empty on purpose until the first build prints a SHA, but the KEY must
        # exist: scripts run under `set -u` and `${LLAMA_PIN_SHA}` unset is a hard abort.
        self.assertIn("LLAMA_PIN_SHA", self.env)

    def test_the_layer_budget_leaves_headroom_below_physical_ram(self):
        # The gap is deliberate headroom. There is no swap, and an OOM mid-run does not
        # merely lose a measurement -- it can take the session down and end the sweep.
        #
        # This used to assert RPC_MEM_MB < NODE_USABLE_MB, on the belief that rpc-server
        # always advertises RPC_MEM_MB to the placement engine. That is conditional, not
        # certain: scripts/lib.sh rpc_extra_args() passes --mem only when the build still
        # accepts it, and upstream removed the flag in 41386cf. NODE_USABLE_MB is the
        # figure the control node plans against unconditionally (profiles.py, via the
        # agent's usable_budget_mb), so that is the one that has to carry the headroom.
        usable = int(self.env["NODE_USABLE_MB"])
        total = int(self.env["NODE_RAM_MB"])
        self.assertLess(usable, total, "NODE_USABLE_MB must leave headroom under physical RAM")
        self.assertLessEqual(
            usable, total - 1024,
            "a swapless board needs a real margin, not a token one -- "
            "NODE_USABLE_MB=%d against NODE_RAM_MB=%d" % (usable, total))

    def test_the_cgroup_ceiling_sits_above_the_layer_budget(self):
        """The systemd ceiling is a backstop, not the planner.

        If MemoryMax were below NODE_USABLE_MB the cgroup would kill the server before the
        placement engine ever thought the node was full -- the failure would look like a
        random mid-sweep crash rather than an over-assignment, which is exactly the kind of
        misattribution this dataset cannot afford.
        """
        unit = (ROOT / "systemd" / "heteroserve-rpc.service").read_text(encoding="utf-8")
        usable = int(self.env["NODE_USABLE_MB"])
        found = {}
        for key in ("MemoryHigh", "MemoryMax"):
            m = re.search(r"^%s=(\d+)([MG])" % key, unit, re.M)
            if m:
                found[key] = int(m.group(1)) * (1024 if m.group(2) == "G" else 1)
        self.assertTrue(found, "no MemoryHigh/MemoryMax found in heteroserve-rpc.service")
        for key, mb in found.items():
            with self.subTest(key=key):
                self.assertGreater(
                    mb, usable,
                    "%s=%d MB is not above NODE_USABLE_MB=%d -- the cgroup would bind "
                    "before the planner does" % (key, mb, usable))

    def test_numeric_fields_parse_as_numbers(self):
        for key in ("NODE_CORES", "NODE_RAM_MB", "NODE_USABLE_MB", "RPC_PORT",
                    "RPC_MEM_MB", "RPC_THREADS", "AGENT_PORT", "INA_CHANNEL", "LLAMA_THREADS"):
            with self.subTest(key=key):
                int(self.env[key])
        for key in ("INA_SHUNT_OHMS", "INA_MAX_A"):
            with self.subTest(key=key):
                float(self.env[key])

    def test_rpc_binds_all_interfaces(self):
        # Bound to 127.0.0.1 it works locally and is invisible to the cluster -- a failure
        # that looks like a network problem for an hour.
        self.assertEqual(self.env["RPC_HOST"], "0.0.0.0")

    def test_agent_and_rpc_ports_do_not_collide(self):
        self.assertNotEqual(self.env["AGENT_PORT"], self.env["RPC_PORT"])

    def test_arch_flags_do_not_claim_instructions_the_a72_lacks(self):
        # The Cortex-A72 has neither dotprod nor i8mm. Building with either would compile
        # and then trap at runtime -- and it would also erase the ISA gap that makes this
        # node the straggler, which is the point of the whole comparison.
        flags = self.env["LLAMA_ARCH_FLAGS"]
        self.assertNotIn("dotprod", flags)
        self.assertNotIn("i8mm", flags)
        self.assertIn("cortex-a72", flags)

    def test_thread_count_matches_the_core_count(self):
        self.assertEqual(int(self.env["LLAMA_THREADS"]), int(self.env["NODE_CORES"]))


@unittest.skipUnless(BASH, "no usable bash found")
class TestBashAgreesWithPython(unittest.TestCase):
    """The file is read by two languages. This is the only test that can catch them
    drifting apart.

    Paths are handed to bash as POSIX: on the Pi that is a no-op, but on a Windows dev box
    Git Bash cannot source a `D:\\...` backslash path and the test would fail for a reason
    that has nothing to do with the config.
    """

    ENV_PATH = NODE_ENV.as_posix()

    def _bash_env(self):
        """Everything bash exports after sourcing node.env, as a dict.

        `env` is used rather than indirect expansion (`${!k}`) because the latter has to
        survive two layers of quoting to get here and mangles silently when it does not.
        """
        proc = subprocess.run(
            [BASH, "-c", 'set -a; . "$1"; set +a; env', "bash", self.ENV_PATH],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        out = {}
        for line in proc.stdout.splitlines():
            key, sep, value = line.partition("=")
            if sep:
                out[key] = value
        return out

    def test_bash_sourcing_yields_the_same_values(self):
        shell = self._bash_env()
        py = envfile.load(NODE_ENV)
        # Only keys the file itself declares; bash's `env` also carries the whole ambient
        # environment, which is not this file's business.
        declared = [
            k for k in py
            # A key whose value the surrounding shell has overridden is expected to differ
            # from the file, and that override is a documented feature.
            if k not in os.environ
        ]
        self.assertGreater(len(declared), 20, "node.env looks unexpectedly small")
        for key in declared:
            with self.subTest(key=key):
                self.assertIn(key, shell, "bash never set %s" % key)
                self.assertEqual(
                    shell[key], py[key],
                    "bash and envfile.py disagree about %s: bash=%r python=%r"
                    % (key, shell[key], py[key]),
                )

    def test_quoted_values_survive_both_readers_identically(self):
        # The three values carrying spaces, brackets or an equals sign -- the ones most
        # likely to be read differently by a hand-rolled parser and by the shell.
        shell = self._bash_env()
        py = envfile.load(NODE_ENV)
        for key in ("NODE_DESC", "LLAMA_ARCH_FLAGS", "PSU_MODEL"):
            with self.subTest(key=key):
                self.assertNotIn('"', py[key], "envfile.py left the quotes on %s" % key)
                self.assertEqual(shell[key], py[key])

    def test_the_file_sources_without_error_under_set_eu(self):
        proc = subprocess.run(
            [BASH, "-c", 'set -euo pipefail; set -a; . "$1"; set +a; echo ok', "bash",
             self.ENV_PATH],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ok", proc.stdout)


class TestLineEndings(unittest.TestCase):
    """A file saved with CRLF on Windows and executed on Raspberry Pi OS fails in a way
    that reads as a missing interpreter: `env: 'bash\\r': No such file or directory`. For
    a shell script it is fatal; in `node.env` a trailing CR ends up *inside* a value, so
    `$RPC_PORT` becomes "50052\\r" and every comparison against it silently fails."""

    def _text_files(self):
        for pattern in ("scripts/*.sh", "node/*.py", "tests/*.py", "config/*",
                        "systemd/*", "Makefile"):
            for path in sorted(ROOT.glob(pattern)):
                if path.is_file():
                    yield path

    def test_no_file_shipped_to_the_pi_has_crlf_line_endings(self):
        offenders = []
        for path in self._text_files():
            if b"\r\n" in path.read_bytes():
                offenders.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        self.assertEqual(
            offenders, [],
            "CRLF line endings would break these on Raspberry Pi OS: %s" % offenders,
        )

    def test_every_shell_script_starts_with_a_clean_shebang(self):
        for path in sorted(ROOT.glob("scripts/*.sh")):
            with self.subTest(script=path.name):
                first = path.read_bytes().split(b"\n", 1)[0]
                self.assertEqual(first, b"#!/usr/bin/env bash")


if __name__ == "__main__":
    unittest.main()
