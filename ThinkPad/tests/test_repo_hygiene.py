"""Tests for properties of the repository itself, not of any one module.

These catch the class of defect that is invisible in a diff and fatal at runtime: a file
that is correct in content but wrong in its bytes, or a config that is internally valid
but disagrees with the node it describes. Both have already happened once in this project.
"""
import json
import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent


def parse_env_file(path):
    """Parse a `node.env` well enough to compare values.

    Inline comments matter: the Jetson's house style is
    `NODE_ROLE=cuda-worker              # cuda-worker | cpu-worker | orchestrator`, so a
    naive parser turns `INA_CHANNEL=3  # jetson rail` into `"3  # jetson rail"` and the
    comparison fails for the wrong reason. A test that fails loudly has to fail
    accurately, or the next person learns to ignore it.
    """
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value[:1] in ('"', "'"):
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end != -1 else value[1:]
        else:
            value = value.split("#", 1)[0].strip()
        env[key.strip()] = value
    return env


class TestLineEndings(unittest.TestCase):
    """The control node runs Ubuntu. A file saved with CRLF and executed there fails in a
    way that reads as a missing interpreter: `env: 'bash\\r': No such file or directory`.
    `scripts/lib.sh` was shipped this way and did not parse at all — every script sources
    it, so nothing in `scripts/` could run. The Pi 4 folder had the same defect."""

    def _text_files(self):
        for pattern in ("scripts/*.sh", "control/*.py", "tests/*.py", "firmware/pico2/*.py",
                        "config/*.json", "config/*.txt", "config/experiments/*.json",
                        "systemd/*", "Makefile", "requirements.txt"):
            for path in sorted(ROOT.glob(pattern)):
                if path.is_file():
                    yield path

    def test_no_file_has_crlf_line_endings(self):
        offenders = [
            str(p.relative_to(ROOT)).replace("\\", "/")
            for p in self._text_files()
            if b"\r\n" in p.read_bytes()
        ]
        self.assertEqual(
            offenders, [],
            "CRLF line endings would break these on the control node: %s" % offenders,
        )

    def test_every_shell_script_starts_with_a_clean_shebang(self):
        for path in sorted(ROOT.glob("scripts/*.sh")):
            with self.subTest(script=path.name):
                first = path.read_bytes().split(b"\n", 1)[0]
                self.assertEqual(first, b"#!/usr/bin/env bash")


class TestClusterAddressing(unittest.TestCase):
    """Two nodes sharing an address does not raise anywhere — it produces a complete,
    self-consistent and entirely wrong energy table, because the harness records the work
    against one node id and the collector integrates the other node's rail. The Pi 5 was
    recorded at .13 here and at .12 in its own `node.env` for exactly this reason."""

    def setUp(self):
        self.cluster = json.loads((ROOT / "config" / "cluster.json").read_text(encoding="utf-8"))

    def test_no_two_nodes_share_a_host(self):
        seen = {}
        for node in self.cluster["nodes"]:
            host = node.get("host")
            self.assertNotIn(
                host, seen,
                "%s and %s both claim %s" % (seen.get(host), node["id"], host),
            )
            seen[host] = node["id"]

    def test_no_two_nodes_share_an_ina_channel(self):
        seen = {}
        for node in self.cluster["nodes"]:
            ch = node.get("ina_channel")
            if ch is None:
                continue
            self.assertNotIn(
                ch, seen, "%s and %s both claim channel %s" % (seen.get(ch), node["id"], ch),
            )
            seen[ch] = node["id"]

    # Every spelling a node id appears under, mapped to its `cluster.json` id. Shared by
    # the node.env and cluster.hosts cross-checks.
    ALIASES = {
        "t580": "t580", "laptop-t580": "t580", "laptop-i5-8350u": "t580",
        "pi5": "pi5", "rpi5-01": "pi5",
        "pi4": "pi4", "rpi4-01": "pi4",
        "jetson": "jetson", "jetson-nano-01": "jetson",
        "orangepi-rv2": "orangepi-rv2",
    }

    # cluster.json key for each node.env key that must agree with it.
    CROSS_CHECKED = {"ETH_ADDR": "host", "INA_CHANNEL": "ina_channel"}

    _parse_env = staticmethod(parse_env_file)

    def _provisioned_nodes(self):
        """(node, folder, env) for every cluster.json node that has a real `node.env`.

        A node with no `node.env` is skipped, not failed: `ThinkPad/` and a future
        `OrangePi/` legitimately have none, and the Jetson's is machine-local
        (`Jetson/.gitignore` lists it). Absence means "not provisioned yet"; a file that
        exists but says nothing is the case this class refuses to tolerate.
        """
        for node in self.cluster["nodes"]:
            folder = Path(node.get("script_dir", "")).name
            env_path = REPO / folder / "config" / "node.env"
            if not folder or not env_path.is_file():
                continue
            yield node, folder, self._parse_env(env_path)

    def test_cluster_json_agrees_with_each_node_env(self):
        """`cluster.json` is what the control node acts on; a worker's `node.env` is what
        the node itself believes. Different folders, different sessions, nothing forcing
        them to agree — so assert it here. The node folder is taken from `script_dir`
        rather than guessed from the id, so adding a node needs no change to this test.

        A `node.env` that exists must DECLARE the cross-checked keys, not merely avoid
        contradicting them. This used to `continue` past a missing key, which meant a
        node.env could be silently outside the check while the suite stayed green — and
        it happened: a concurrent session replaced `Jetson/config/node.env` with a test
        fixture carrying neither key, and the Jetson dropped out of this cross-check with
        no failure anywhere. Silent loss of coverage is the thing this test exists to
        prevent, so absence is now as loud as disagreement.
        """
        undeclared, disagreed, checked = [], [], 0
        for node, folder, env in self._provisioned_nodes():
            for env_key, json_key in self.CROSS_CHECKED.items():
                expected = node.get(json_key)
                if env_key not in env:
                    # Only required where cluster.json actually asserts a value; a node
                    # with no INA channel (ina_channel: null) need not declare one.
                    if expected is not None:
                        undeclared.append(
                            "%s: %s/config/node.env declares no %s, but cluster.json says "
                            "%s=%s" % (node["id"], folder, env_key, json_key, expected))
                    continue
                checked += 1
                if str(env[env_key]) != str(expected):
                    disagreed.append(
                        "%s: %s/config/node.env says %s=%s, cluster.json says %s=%s"
                        % (node["id"], folder, env_key, env[env_key], json_key, expected))

        self.assertEqual(
            undeclared, [],
            "a node.env that exists must declare the cross-checked keys, or the node "
            "silently leaves this check: %s" % undeclared)
        self.assertEqual(disagreed, [], "cluster.json and node.env disagree: %s" % disagreed)
        self.assertGreater(checked, 0, "no node.env was found to cross-check against")

    def test_every_node_env_identifies_a_node_this_cluster_knows(self):
        """`NODE_ID` names the node in every results file and in the characterisation
        dataset, so a wrong one mislabels data rather than raising. It does not have to
        equal the `cluster.json` id — `jetson-nano-01` and `jetson` are the same machine —
        but it must be a spelling this cluster recognises, and it must resolve to the node
        whose folder it sits in. `NODE_ID=fixture` (a real clobber, 2026-09-13) is exactly
        what this catches."""
        offenders = []
        for node, folder, env in self._provisioned_nodes():
            node_id = env.get("NODE_ID")
            if node_id is None:
                offenders.append("%s/config/node.env declares no NODE_ID" % folder)
                continue
            canonical = self.ALIASES.get(node_id)
            if canonical is None:
                offenders.append(
                    "%s/config/node.env says NODE_ID=%r, which this cluster does not "
                    "recognise — add it to ALIASES if it is a real new spelling"
                    % (folder, node_id))
            elif canonical != node["id"]:
                offenders.append(
                    "%s/config/node.env says NODE_ID=%r (that is %s), but cluster.json "
                    "puts %s in this folder" % (folder, node_id, canonical, node["id"]))
        self.assertEqual(offenders, [], "; ".join(offenders))

    def test_every_cluster_hosts_inventory_agrees_with_cluster_json(self):
        """Three files list the same five machines under three different naming schemes:
        this `cluster.json`, `Pi5/config/cluster.hosts` and
        `Jetson/config/cluster.hosts.example`. They drifted once — the Orange Pi was given
        the address `cluster.json` had already assigned to the Jetson — so normalise the
        ids and assert the maps match."""
        alias = self.ALIASES
        expected = {n["id"]: n["host"] for n in self.cluster["nodes"]}
        inventories = sorted(REPO.glob("*/config/cluster.hosts")) + \
            sorted(REPO.glob("*/config/cluster.hosts.example"))
        self.assertTrue(inventories, "no cluster.hosts inventory found to cross-check")
        for path in inventories:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 2:
                    continue
                node_id, host = fields[0], fields[1]
                canonical = alias.get(node_id)
                with self.subTest(file=path.name, node=node_id):
                    self.assertIsNotNone(
                        canonical, "%s: unknown node id %r — add it to the alias map"
                        % (path, node_id))
                    self.assertEqual(
                        host, expected[canonical],
                        "%s lists %s at %s; cluster.json says %s"
                        % (path.relative_to(REPO), node_id, host, expected[canonical]))

class TestRpcServerInterface(unittest.TestCase):
    """llama.cpp's rpc server has changed its interface twice, and this cluster pins ONE
    SHA across four heterogeneous nodes — chosen on the Jetson, whose CUDA 10.2 will not
    build current master. So neither the binary name nor the flag set is knowable when a
    script is written, and every node must resolve both at runtime:

      * renamed  rpc-server -> ggml-rpc-server  in c299a92 (2026-06-27)
      * -m/--mem removed                        in 41386cf (2025-10-17)
      * -c/--cache has ALWAYS been boolean — it takes no directory. LLAMA_CACHE picks the
        location and the server appends "rpc/" to it.

    Getting any of these wrong fails in the worst possible way. On an unknown argument the
    server prints usage and calls exit(0) — a SUCCESS code — so systemd's
    Restart=on-failure sees a clean shutdown, a `| tee` pipeline returns 0, and
    `heteroctl rpc up` reports the worker as started. The port is simply closed, and the
    first thing that notices is a run that silently never used that node.
    """

    NODES = ("ThinkPad", "Pi4", "Pi5", "Jetson")

    def _files(self):
        for node in self.NODES:
            for path in sorted((REPO / node).rglob("*")):
                posix = path.as_posix()
                if not path.is_file() or "/.handoff/" in posix or "__pycache__" in posix:
                    continue
                if path.suffix in (".sh", ".py", ".service") or path.name.endswith("node.env"):
                    yield path

    @staticmethod
    def _code_lines(path):
        """Comment lines are excluded: several of these files explain the traps below in
        prose, and that prose must not read as the defect it warns about."""
        return [
            (n, line)
            for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1)
            if not line.lstrip().startswith("#")
        ]

    # The binary path forms that pin the name: $LLAMA_BIN/rpc-server, build/bin/rpc-server,
    # Path(...).joinpath("rpc-server"). Deliberately does NOT match rpc-server.sh (the
    # generated launcher) or rpc-server.log.
    _BIN_PATH = re.compile(
        r"""(?:\$\{?LLAMA_BIN\}?|build/bin|bin)/rpc-server(?![\w.-])"""
        r"""|joinpath\(\s*["']rpc-server["']"""
    )
    # The exemption, for the genuine two-arm probes (Pi5 checks ggml-rpc-server and then
    # falls back to rpc-server in the same construct). Merely mentioning the new name in
    # a message or comment does not qualify, and neither does calling bin_rpc elsewhere
    # in the file: a file that really delegates has no hard-coded path to match at all,
    # so exempting on that would let a delegating file smuggle one back in.
    _PROBES_NEW_NAME = re.compile(
        r"""(?:\$\{?LLAMA_BIN\}?|build/bin|bin)/ggml-rpc-server"""
        r"""|["']ggml-rpc-server["']"""
    )

    def test_no_file_pins_the_old_binary_name(self):
        offenders = []
        for path in self._files():
            text = path.read_text(errors="replace")
            if self._PROBES_NEW_NAME.search(text):
                continue  # the file really resolves both names, or delegates to bin_rpc
            for n, line in self._code_lines(path):
                if self._BIN_PATH.search(line):
                    offenders.append("%s:%d: %s" % (path.as_posix(), n, line.strip()))
        self.assertEqual(
            offenders, [],
            "these resolve the rpc server by the pre-c299a92 name only, so they silently "
            "find nothing on any ref newer than 2026-06-27: %s" % offenders,
        )

    def test_no_file_passes_a_directory_to_cache(self):
        # -c/--cache is boolean in every upstream revision. Passing a path makes the
        # server report an unknown argument and exit(0).
        offenders = [
            "%s:%d: %s" % (path.as_posix(), n, line.strip())
            for path in self._files()
            for n, line in self._code_lines(path)
            if re.search(r"""--cache[ =]+["'$/]""", line)
        ]
        self.assertEqual(
            offenders, [],
            "--cache takes no argument; set LLAMA_CACHE instead: %s" % offenders,
        )

    # A resolver may name the candidates inline on the `bin_rpc()` line, or hold them in a
    # variable that the line expands. The Jetson does the latter -- `RPC_BIN_NAMES` is also
    # what builds its pgrep pattern, so one list feeds both -- and a test that only reads
    # the `bin_rpc()` line cannot see through that. Follow the indirection rather than
    # forcing every node to spell the names in the same place: the invariant under test is
    # the ORDER the names are tried in, not where they are written.
    _INDIRECTION = re.compile(r"\$\{?([A-Z_][A-Z0-9_]*)\}?")

    def _resolver_body(self, path, pattern):
        """The resolver line, with any single variable it expands resolved in-place."""
        text = path.read_text()
        match = re.search(pattern, text, re.MULTILINE)
        if match is None:
            return None
        body = match.group(0)
        for var in self._INDIRECTION.findall(body):
            if var in ("LLAMA_BIN",):      # the search directory, not a candidate list
                continue
            assign = re.search(r"^%s=[\"']?(.*?)[\"']?$" % re.escape(var), text, re.MULTILINE)
            if assign:
                body = body.replace("$" + var, assign.group(1))
                body = body.replace("${%s}" % var, assign.group(1))
        return body

    def test_every_resolver_tries_the_current_name_first(self):
        resolvers = {
            "ThinkPad/scripts/lib.sh": r"^bin_rpc\(\).*$",
            "Pi4/scripts/lib.sh": r"^bin_rpc\(\).*$",
            "Jetson/scripts/lib/common.sh": r"^bin_rpc\(\).*$",
            "Pi4/node/node_agent.py": r"^RPC_BIN_CANDIDATES\s*=.*$",
        }
        for rel, pattern in resolvers.items():
            with self.subTest(resolver=rel):
                path = REPO / rel
                self.assertTrue(path.is_file(), "%s is missing" % rel)
                body = self._resolver_body(path, pattern)
                self.assertIsNotNone(body, "no rpc resolver found in %s" % rel)
                new = body.find("ggml-rpc-server")
                self.assertNotEqual(new, -1, "%s does not offer ggml-rpc-server" % rel)
                # Blank out the new name so the old one is found on its own merits.
                old = body.replace("ggml-rpc-server", "x" * len("ggml-rpc-server")).find("rpc-server")
                self.assertTrue(
                    old == -1 or new < old,
                    "%s must try ggml-rpc-server before rpc-server, got: %s" % (rel, body.strip()),
                )

    def test_unit_files_are_not_executable(self):
        # systemd refuses to parse an executable unit cleanly ("marked executable").
        offenders = [
            p.as_posix() for p in self._files()
            if p.suffix == ".service" and os.access(p, os.X_OK)
        ]
        self.assertEqual(offenders, [], "unit files must be 0644: %s" % offenders)

    def test_no_rpc_unit_restarts_only_on_failure(self):
        # exit(0) on an unknown argument is a success code, so Restart=on-failure leaves
        # the unit sitting "dead" with the port closed and no failure state.
        offenders = [
            "%s:%d" % (p.as_posix(), n)
            for p in self._files()
            if p.suffix == ".service" and "rpc" in p.name
            for n, line in self._code_lines(p)
            if line.strip() == "Restart=on-failure"
        ]
        self.assertEqual(
            offenders, [],
            "an rpc unit needs Restart=always plus StartLimit* in [Unit]: %s" % offenders,
        )


class TestSystemdUnitSpecifiers(unittest.TestCase):
    """`%i` and `%I` are the INSTANCE specifiers. They expand only in a template unit — one
    whose filename ends in `@.service`. In a plain unit they expand to nothing, so
    `User=%i` silently becomes `User=` and systemd rejects the unit at load with "Invalid
    user/group name or numeric ID". `systemd/heteroserve-collector.service` shipped that
    way: the install snippets renamed it on the way to /etc/systemd/system so the DEPLOYED
    unit was fine, but the repo file could not be verified or installed under its own name.

    Deliberately a plain-Python check rather than a `systemd-analyze verify` sweep. Seven
    of the ten units in this repo exit 1 under `verify` purely because ExecStart points
    into /opt/heteroserve, which does not exist on a dev machine — a verify test would be
    red on most of the tree from day one. This one has no external dependency, runs on any
    OS, and targets the actual defect class.
    """

    NODES = ("ThinkPad", "Pi4", "Pi5", "Jetson")

    def _units(self):
        for node in self.NODES:
            for path in sorted((REPO / node).rglob("*.service")):
                if "/.handoff/" not in path.as_posix():
                    yield path

    def test_instance_specifier_implies_a_template_filename(self):
        offenders = []
        for path in self._units():
            if "@" in path.name:
                continue  # a template — the specifier expands as intended
            for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#") or stripped.startswith(";"):
                    continue
                if re.search(r"%[iI](?![a-zA-Z])", line):
                    offenders.append("%s:%d: %s" % (path.as_posix(), n, line.strip()))
        self.assertEqual(
            offenders, [],
            "%%i/%%I expand only in a unit whose filename ends in '@.service'; in a plain "
            "unit they vanish and the directive silently loses its value: %s" % offenders,
        )

    def test_no_exec_line_has_an_unescaped_percent(self):
        """systemd expands `%`-specifiers in `Exec*=` lines BEFORE handing the string to a
        shell, and it does not respect shell quoting. A literal percent must be written
        `%%`. `Jetson/systemd/heteroserve-telemetry.service` shipped
        `$(date -u +%Y%m%dT%H%M%SZ)` undoubled, and every letter in that format is a valid
        specifier (%Y unit-file dir, %m machine ID, %d credentials dir, %H host name, %S
        state dir root) — so `date` never saw its format string. It expanded instead to a
        host-CONSTANT path containing slashes, which (a) reinstated the fixed-filename
        truncation that same unit's comment claims to have fixed, and (b) buried the CSV
        several directories deep, where `90_collect_results.sh` (`find -maxdepth 1`) would
        never collect it. `systemd-analyze verify` exits 0 either way, so only this catches
        it.

        The rule: within an `Exec*=` value, runs of `%` must be even-length. `%%Y` is a
        literal percent; `%Y` is a specifier. If a unit ever genuinely needs a specifier in
        an Exec line, this test is the right place to record that exception explicitly —
        do not silence it by weakening the check.
        """
        offenders = []
        for path in self._units():
            for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#") or stripped.startswith(";"):
                    continue
                if not re.match(r"Exec[A-Za-z]*\s*=", stripped):
                    continue
                for run in re.findall(r"%+", stripped):
                    if len(run) % 2:
                        offenders.append("%s:%d: %s" % (path.as_posix(), n, stripped[:110]))
                        break
        self.assertEqual(
            offenders, [],
            "a literal percent in an Exec* line must be doubled, or systemd eats it as a "
            "specifier before the shell runs: %s" % offenders,
        )

    def test_the_collector_unit_is_a_template(self):
        # Guards the specific rename, so nothing quietly renames it back.
        units = {p.name for p in (REPO / "ThinkPad" / "systemd").glob("*.service")}
        self.assertIn("heteroserve-collector@.service", units)
        self.assertNotIn("heteroserve-collector.service", units)



class TestLlamaRevisionPin(unittest.TestCase):
    """One llama.cpp revision is pinned cluster-wide, and it is spelled THREE different
    ways across four node folders:

        ThinkPad/config/cluster.json   llama.pin_sha
        Pi4/config/node.env            LLAMA_PIN_SHA
        Pi5/config/node.env            LLAMA_PIN_SHA
        Jetson/config/node.env         LLAMA_CPP_REF      <-- different name

    `control/nodes.py` does check the pin, but over the NETWORK — it asks each agent's
    /info for `llama_pin_sha`, so it needs the cluster powered up and every agent
    reachable, and the Jetson ships no node agent at all. Nothing else checks that the
    repo's own files agree. This does, offline.

    The realistic failure is a PARTIAL rollout: someone answers "which SHA?" and updates
    two of the four. Three nodes then build one revision and the fourth builds another,
    and the characterisation dataset silently mixes them — which is the one thing this
    whole study cannot tolerate.

    Until 2026-09-20 this only checked set-vs-unset, so four locations holding four
    DIFFERENT SHAs passed cleanly — the exact failure the docstring describes was not
    actually caught. It now checks VALUES.

    The cluster is deliberately two-revision as of 2026-09-20: the Jetson keeps CUDA/FP16
    and cannot build the cluster pin (JetPack 4.6.x is CUDA 10.2). That is expressed as a
    DECLARED exception in cluster.json under `llama.exceptions` — a mapping of location to
    the exact value that location is allowed to hold. Declared divergence passes; anything
    else fails. The point is that a deliberate split and an accidental one no longer look
    the same.

    THE TEMPLATE IS A REAL LOCATION, not a nicety. `Jetson/config/node.env` is gitignored
    (`Jetson/.gitignore:1`), so on a fresh clone it does not exist, `_values()` skipped it,
    and the node that CHOOSES the pin was the one node this check could not see.
    Demonstrated by removing that file and setting the other three to one SHA: the test
    PASSED on a genuine 3-of-4 rollout. `Jetson/config/node.env.example` IS tracked and is
    what seeds the real file, so a pin recorded everywhere EXCEPT the template still leaves
    a fresh clone building master. Both files are therefore checked.
    """

    LOCATIONS = (
        ("ThinkPad/config/cluster.json", "llama.pin_sha"),
        ("Pi4/config/node.env", "LLAMA_PIN_SHA"),
        ("Pi5/config/node.env", "LLAMA_PIN_SHA"),
        ("Jetson/config/node.env", "LLAMA_CPP_REF"),
        # Tracked, and the source a fresh clone is seeded from. See the docstring.
        ("Jetson/config/node.env.example", "LLAMA_CPP_REF"),
    )

    def _values(self):
        """(label, value) for every location whose file exists. Missing key -> ''."""
        out = []
        for rel, key in self.LOCATIONS:
            path = REPO / rel
            if not path.is_file():
                continue          # node not provisioned yet; not this test's business
            if path.suffix == ".json":
                blob = json.loads(path.read_text(encoding="utf-8"))
                value = (blob.get("llama") or {}).get("pin_sha", "")
            else:
                value = parse_env_file(path).get(key, "")
            out.append(("%s (%s)" % (rel, key), (value or "").strip()))
        return out

    def _declared_exceptions(self):
        """location -> the exact value that location is allowed to hold instead of the pin."""
        blob = json.loads((REPO / "ThinkPad/config/cluster.json").read_text(encoding="utf-8"))
        return dict(((blob.get("llama") or {}).get("exceptions") or {}))

    def test_declared_exceptions_name_real_locations(self):
        """An exception for a path this class does not check is a hole that silently
        exempts nothing, and hides a typo in the path."""
        checked = {rel for rel, _ in self.LOCATIONS}
        for rel in self._declared_exceptions():
            self.assertIn(rel, checked,
                          "llama.exceptions names %r, which is not a checked pin location" % rel)

    def test_the_pinned_revision_is_all_set_or_all_unset(self):
        values = self._values()
        self.assertTrue(values, "no pin location found at all")
        unset = [label for label, value in values if not value]
        if len(unset) == len(values):
            # All-unset is still tolerated, but it is NO LONGER "the correct state today":
            # Q1 was answered on 2026-09-19 and cluster.json now pins b5050
            # (23106f94ea2bc3da929afb7330655fd5515d08dc). Tolerated because deliberately
            # clearing every location to rebuild against master is a legitimate operation,
            # and this test cannot tell that apart from a regression. If the whole cluster
            # ever reads empty again, that is a finding to investigate, not a green light.
            return
        self.assertEqual(
            unset, [],
            "the llama.cpp revision is pinned in some places but not others, so those "
            "nodes would build whatever master is that day: %s. Set it in every location "
            "or none: %s" % (unset, values),
        )

    def test_the_tracked_template_is_one_of_the_checked_locations(self):
        """`Jetson/config/node.env` is gitignored, so the template is the only pin location
        a fresh clone actually has. Dropping it from LOCATIONS restores a hole that a
        3-of-4 rollout walks straight through, and nothing else in the suite would notice."""
        checked = [rel for rel, _ in self.LOCATIONS]
        self.assertIn("Jetson/config/node.env.example", checked)
        self.assertTrue((REPO / "Jetson/config/node.env.example").is_file(),
                        "the tracked Jetson template is missing; a fresh clone has no "
                        "pin location for the node that chooses the pin")

    def test_a_partial_rollout_is_caught_even_when_the_gitignored_file_is_absent(self):
        """The exact fresh-clone scenario, driven through the real _values()."""
        real = [rel for rel, _ in self.LOCATIONS
                if rel != "Jetson/config/node.env" and (REPO / rel).is_file()]
        self.assertIn("Jetson/config/node.env.example", real,
                      "with the gitignored file absent the template must still be read, "
                      "or a 3-of-4 rollout passes unnoticed")

    def test_every_pin_location_agrees_unless_the_split_is_declared(self):
        """Undeclared divergence is a partial rollout and must fail. DECLARED divergence
        is a deliberate engineering decision and must pass -- but only at the exact value
        that was declared, so "the Jetson is special" cannot quietly become "the Jetson is
        on whatever it happens to be on".

        As of 2026-09-20 the cluster IS two-revision on purpose: the Jetson keeps CUDA/FP16
        and JetPack 4.6.x cannot build the cluster pin. See cluster.json llama.exceptions
        and Pi5/docs/PIN_CONFLICT.md."""
        allowed = self._declared_exceptions()
        undeclared = {}
        for rel, key in self.LOCATIONS:
            path = REPO / rel
            if not path.is_file():
                continue
            if path.suffix == ".json":
                value = ((json.loads(path.read_text(encoding="utf-8")).get("llama") or {})
                         .get("pin_sha", ""))
            else:
                value = parse_env_file(path).get(key, "")
            value = (value or "").strip()
            if not value:
                continue
            if rel in allowed:
                self.assertEqual(
                    value, allowed[rel],
                    "%s is a declared exception but holds %r, not the declared %r -- a "
                    "declared exception pins a SPECIFIC revision, it is not a licence to "
                    "drift" % (rel, value[:12], allowed[rel][:12]),
                )
                continue
            undeclared[rel] = value
        distinct = set(undeclared.values())
        self.assertLessEqual(
            len(distinct), 1,
            "the nodes disagree about which llama.cpp revision is pinned, so the "
            "characterisation dataset would mix revisions: %s. If a split is deliberate, "
            "declare it in cluster.json under llama.exceptions with its exact value and a "
            "reason." % (undeclared,),
        )



class TestDocsDoNotHardCodeTheSuiteSize(unittest.TestCase):
    """The docs said "43 unit tests" in four places while the suite actually ran 64.

    Nobody lied; the number was written once and the suite kept growing. A count in prose
    is a fact with no owner -- nothing fails when it drifts, so it drifts, and then a
    reader who runs `make test` and sees a different number has to work out which source
    is stale. Say "unit tests" and let the runner report the number.

    Scoped to lines that are actually talking about this suite (`make test`, `unittest`,
    `tests/`), so experiment prose like "3 test cases per condition" is unaffected.
    """

    COUNT = re.compile(r"\b\d+\s+(?:unit\s+)?tests?\b", re.IGNORECASE)
    ABOUT_THE_SUITE = ("make test", "unittest", "tests/")

    def _doc_files(self):
        yield ROOT / "README.md"
        yield ROOT / "Makefile"
        for path in sorted((ROOT / "docs").glob("*.md")):
            yield path

    def test_no_doc_states_a_test_count(self):
        offenders = []
        for path in self._doc_files():
            if not path.is_file():
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                low = line.lower()
                if not any(marker in low for marker in self.ABOUT_THE_SUITE):
                    continue
                if self.COUNT.search(line):
                    offenders.append("%s:%d: %s"
                                     % (path.relative_to(ROOT).as_posix(), n, line.strip()))
        self.assertEqual(
            offenders, [],
            "these hard-code how many tests the suite has, which silently goes stale; "
            "drop the number: %s" % offenders,
        )



if __name__ == "__main__":
    unittest.main()
