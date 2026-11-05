#!/usr/bin/env python3
"""Mutation runner for the test-quality measurement (see .handoff/nodes/Pi5/ at the repo root for the method).

Specs are a JSON list of {label, before, after}.
Refuses to report a result unless the edit actually changed the file."""
import json, pathlib, subprocess, sys

sandbox, target, specfile, pattern = sys.argv[1:5]
root = pathlib.Path(sandbox)
f = root / target
orig = f.read_text()
specs = json.loads(pathlib.Path(specfile).read_text())
for s in specs:
    lbl, bef, aft = s["label"], s["before"], s["after"]
    n = orig.count(bef)
    if n != 1:
        print("  ANCHOR(%d)  %s" % (n, lbl)); continue
    mutated = orig.replace(bef, aft)
    if mutated == orig:
        print("  NO-OP      %s  (before == after)" % lbl); continue
    f.write_text(mutated)
    p = subprocess.run([sys.executable, "-m", "unittest", "discover",
                        "-s", "tests", "-t", ".", "-p", pattern],
                       cwd=root, capture_output=True, text=True)
    tail = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ""
    print("  %-9s %s   [%s]" % ("KILLED" if p.returncode else "SURVIVES", lbl, tail))
    f.write_text(orig)
f.write_text(orig)
