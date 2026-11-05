"""Deployment-consistency checks for the node folder itself.

Everything else in `tests/` checks what the code computes. Nothing checked whether this
folder can actually be INSTALLED: whether the systemd units point at paths something
creates, whether the scripts the README tells an operator to run exist, whether a unit
references a sibling unit that was renamed. Those failures cannot be found by reading one
file, they surface on the board at boot, and this folder has never been on a board.

Both sibling nodes have a suite like this (`Pi4/tests/test_layout.py`,
`ThinkPad/tests/test_repo_hygiene.py`). This one was missing.

The checks are deliberately precise rather than heuristic: each one names a concrete pairing
between two files that must agree, so a failure points at the drift rather than at a guess.
"""
from __future__ import annotations

import re
import stat
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted((ROOT / "scripts").rglob("*.sh"))
UNITS = sorted((ROOT / "systemd").glob("*.service"))
INSTALLER = ROOT / "scripts" / "04_install_services.sh"
# Where docs/SETUP.md tells the operator to symlink this repo folder on the node. Units may
# reference paths beneath it without the installer creating them.
SYMLINK_ROOT = "/opt/heteroserve/Pi5"


class TestScripts(unittest.TestCase):
    def test_there_are_scripts_to_check(self):
        """A guard on the guard: a bad glob would make every test below vacuous."""
        self.assertGreaterEqual(len(SCRIPTS), 9)
        self.assertGreaterEqual(len(UNITS), 4)

    def test_every_entry_point_has_a_bash_shebang(self):
        """lib/ holds sourced libraries, which must NOT have one -- a shebang there invites
        someone to execute a file whose `set -euo pipefail` then applies to nothing."""
        for p in SCRIPTS:
            if p.parent.name == "lib":
                continue
            with self.subTest(script=p.name):
                first = p.read_text(encoding="utf-8").splitlines()[0]
                self.assertTrue(first.startswith("#!"), "no shebang: %s" % p.name)
                self.assertIn("bash", first, "not a bash shebang: %s" % p.name)

    def test_sourced_libraries_say_so_and_carry_no_shebang(self):
        for p in SCRIPTS:
            if p.parent.name != "lib":
                continue
            with self.subTest(script=p.name):
                text = p.read_text(encoding="utf-8")
                self.assertFalse(text.startswith("#!"),
                                 "%s is sourced; a shebang invites executing it" % p.name)
                self.assertIn("ourced", text.splitlines()[1],
                              "%s does not say it is sourced" % p.name)

    def test_entry_point_scripts_are_executable(self):
        """lib/common.sh is sourced, never executed, so it is exempt."""
        for p in SCRIPTS:
            if p.parent.name == "lib":
                continue
            with self.subTest(script=p.name):
                self.assertTrue(p.stat().st_mode & stat.S_IXUSR,
                                "%s is not executable" % p.name)

    def test_no_crlf_anywhere(self):
        """A file saved with CRLF fails on the Pi as `env: 'bash\\r': No such file or
        directory`, which reads as a missing interpreter. It has already happened twice in
        this project, on other nodes."""
        for p in SCRIPTS + UNITS + [ROOT / "config" / "node.env"]:
            with self.subTest(path=p.name):
                self.assertNotIn(b"\r\n", p.read_bytes(), "%s has CRLF" % p.name)

    def test_units_are_not_executable(self):
        """Units are data read by systemd. A stray exec bit has been removed from these
        once already."""
        for p in UNITS:
            with self.subTest(unit=p.name):
                self.assertFalse(p.stat().st_mode & stat.S_IXUSR,
                                 "%s should not be executable" % p.name)


class TestUnitsAgreeWithTheInstaller(unittest.TestCase):
    """The units name absolute paths on the deployed node. Something has to put a file at
    each one: `04_install_services.sh`, an earlier numbered script, or -- for the repo
    checkout itself -- the clone/symlink step documented in `docs/SETUP.md`. A path that no
    step creates fails at boot, not at install."""

    def _directives(self, name):
        out = []
        for u in UNITS:
            for line in u.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(name + "="):
                    out.append((u.name, line.split("=", 1)[1].strip().rstrip("\\").strip()))
        return out

    def test_every_environmentfile_is_installed(self):
        installer = INSTALLER.read_text(encoding="utf-8")
        for unit, value in self._directives("EnvironmentFile"):
            with self.subTest(unit=unit, path=value):
                self.assertIn(value, installer,
                              "%s reads %s, which 04_install_services.sh never creates"
                              % (unit, value))

    # Deployed path -> (where its directory comes from, the repo file behind it or None).
    # Checked by DIRECTORY, because the installer creates dirs and copies globs: the literal
    # string "/opt/heteroserve/node/telemetry.py" appears nowhere, yet the file is installed.
    def test_every_execstart_target_is_accounted_for(self):
        installer = INSTALLER.read_text(encoding="utf-8")
        setup = (ROOT / "docs" / "SETUP.md").read_text(encoding="utf-8")
        for unit, value in self._directives("ExecStart"):
            target = value.split()[0]
            # `/opt/.../python foo.py` -- the interpreter comes from 01_install_deps.sh; the
            # script after it is the thing that must be installed.
            if Path(target).name.startswith("python"):
                target = value.split()[1]
            # Walk up: docs/SETUP.md documents the symlink at /opt/heteroserve/Pi5, and
            # scripts/ lives INSIDE the repo, so the ancestor is what a step has to create.
            ancestors = [str(a) for a in Path(target).parents
                         if str(a) not in ("/", "/opt")]
            # Only DIRECTORY-CREATING lines count. Accepting any mention let an installer
            # that created /opt/heteroserve/nodes while copying into /opt/heteroserve/node
            # pass, which fails at install time with no such file or directory.
            # Token-exact, not substring: "/opt/heteroserve" occurs inside
            # "/opt/heteroserve/bin", so a substring test let an installer that created
            # /opt/heteroserve/nodes while copying into /opt/heteroserve/node pass.
            created = set()
            for ln in installer.splitlines():
                if re.search(r'\b(install\s+-d|mkdir)\b', ln):
                    created.update(re.findall(r'(/[A-Za-z0-9_./-]+)', ln))
            # Two ways a deployed directory legitimately exists, and only two:
            #   1. 04_install_services.sh creates it explicitly. Required to be the
            #      IMMEDIATE parent -- "/opt/heteroserve" existing does not mean
            #      "/opt/heteroserve/node" does.
            #   2. It is inside the repo checkout, which docs/SETUP.md puts in place with a
            #      clone plus a symlink at /opt/heteroserve/Pi5. That step brings the whole
            #      subtree at once, so an ancestor is enough -- but ONLY for that subtree.
            # Letting any SETUP.md mention rescue any path is what made this test vacuous:
            # "/opt/heteroserve" appears in the docs, so every path under it passed.
            parent = str(Path(target).parent)
            # Boundary-anchored: "/opt/heteroserve/Pi5" is a prefix of
            # "/opt/heteroserve/Pi5x", so a plain `in` accepted a docs change that moved
            # the symlink somewhere else entirely.
            documented = re.search(re.escape(SYMLINK_ROOT) + r'(?![A-Za-z0-9_-])', setup)
            via_repo_symlink = SYMLINK_ROOT in ancestors and documented is not None
            with self.subTest(unit=unit, target=target):
                self.assertTrue(
                    parent in created or via_repo_symlink,
                    "%s runs %s, but 04_install_services.sh has no `install -d`/`mkdir` "
                    "for %s, and %s is not inside the repo checkout that docs/SETUP.md "
                    "symlinks to %s" % (unit, target, parent, target, SYMLINK_ROOT))

    def test_every_execstart_target_maps_to_a_file_in_this_repo(self):
        """The deployed tree is this folder plus what the installer generates. A unit naming
        a script that exists in neither is a boot failure nobody sees until the board is on."""
        installer = INSTALLER.read_text(encoding="utf-8")
        for unit, value in self._directives("ExecStart"):
            target = value.split()[0]
            if Path(target).name.startswith("python"):
                target = value.split()[1]
            name = Path(target).name
            with self.subTest(unit=unit, target=target):
                in_repo = ((ROOT / "node" / name).is_file()
                           or (ROOT / "scripts" / name).is_file())
                generated = name in installer      # e.g. the rpc-server.sh launcher
                self.assertTrue(in_repo or generated,
                                "%s runs %s, which is neither a file in this repo nor "
                                "generated by 04_install_services.sh" % (unit, name))

    def test_the_python_services_the_units_run_exist_in_the_repo(self):
        for unit, value in self._directives("ExecStart"):
            for token in value.split():
                if token.endswith(".py"):
                    with self.subTest(unit=unit, module=token):
                        self.assertTrue((ROOT / "node" / Path(token).name).is_file(),
                                        "%s runs %s but node/%s does not exist"
                                        % (unit, token, Path(token).name))

    def test_the_installer_copies_every_python_service_it_runs(self):
        installer = INSTALLER.read_text(encoding="utf-8")
        self.assertIn("node/", installer)
        self.assertRegex(installer, r'install .*node/.*\*\.py|install .*"\$\{HS_NODE_DIR\}/node/"\*\.py',
                         "04_install_services.sh does not appear to copy node/*.py")

    def test_sibling_unit_references_exist(self):
        """A `Before=`/`After=`/`Wants=` naming a heteroserve unit that was renamed is a
        silent ordering loss: systemd ignores unknown units."""
        names = {u.name for u in UNITS}
        for directive in ("Before", "After", "Wants", "Requires", "PartOf"):
            for unit, value in self._directives(directive):
                for ref in value.split():
                    if not ref.startswith("heteroserve-"):
                        continue
                    with self.subTest(unit=unit, references=ref):
                        self.assertIn(ref, names,
                                      "%s references %s, which does not exist in systemd/"
                                      % (unit, ref))


class TestDocsMatchTheTree(unittest.TestCase):
    def test_every_script_the_readme_tells_you_to_run_exists(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        named = set(re.findall(r'\./(scripts/[0-9]{2}_[a-z0-9_]+\.sh)', readme))
        self.assertGreaterEqual(len(named), 6, "README names suspiciously few scripts")
        for rel in sorted(named):
            with self.subTest(script=rel):
                self.assertTrue((ROOT / rel).is_file(),
                                "README tells the operator to run %s, which does not exist"
                                % rel)

    def test_every_numbered_script_is_mentioned_somewhere(self):
        """A script nobody is told to run is either dead or an undocumented step."""
        prose = "\n".join((ROOT / f).read_text(encoding="utf-8")
                          for f in ("README.md", "docs/SETUP.md"))
        for p in SCRIPTS:
            if p.parent.name == "lib":
                continue
            with self.subTest(script=p.name):
                self.assertIn(p.name, prose,
                              "%s is in scripts/ but named in neither README.md nor "
                              "docs/SETUP.md" % p.name)

    def test_the_agent_endpoints_the_docs_list_are_served(self):
        """docs/SETUP.md publishes the agent's routes; the handler is the authority."""
        setup = (ROOT / "docs" / "SETUP.md").read_text(encoding="utf-8")
        handler = (ROOT / "node" / "agent.py").read_text(encoding="utf-8")
        for route in re.findall(r'`(/[a-z-]+)`', setup):
            if route in ("/", "/opt"):
                continue
            with self.subTest(route=route):
                self.assertIn('"%s"' % route, handler,
                              "docs/SETUP.md advertises %s but agent.py does not serve it"
                              % route)


if __name__ == "__main__":
    unittest.main(verbosity=2)
