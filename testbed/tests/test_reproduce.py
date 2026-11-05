"""Tests for testbed/scripts/reproduce.py's pure parts (the leaf comparison, the STATE-line temperatures), stdlib unittest only.

    python3 -m unittest discover -s testbed/tests -q
"""
import os, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
import reproduce as R                                         # noqa: E402


class TestReproduce(unittest.TestCase):
    def test_leaves_and_compare(self):
        pub = {"cells": {"a": {"decode": {"ms": 95.6}, "split": [1, 2]}, "b": {"x": "s"}}, "scored": "2026-10-08T00:00"}
        self.assertEqual(R.leaves(pub), {"/cells/a/decode/ms": 95.6, "/cells/a/split[0]": 1, "/cells/a/split[1]": 2,
                                         "/cells/b/x": "s"})                     # "scored" is ignored
        same = {"cells": {"a": {"decode": {"ms": 95.6 + 1e-12}, "split": [1, 2], "new": 3}, "b": {"x": "s"}}}
        self.assertEqual(R.compare(pub, same, None), ([], [], 1))                # equal within 1e-9; one new field
        moved = {"cells": {"a": {"decode": {"ms": 95.7}, "split": [1, 2]}}}
        d, po, do = R.compare(pub, moved, None)
        self.assertEqual(d, [("/cells/a/decode/ms", 95.6, 95.7)])
        self.assertEqual(po, ["/cells/b/x"])
        scratch = {"cells": {"a": {"decode": {"ms": 95.6}, "split": [1, 2]}, "b": {"x": "/tmp/scratch-1/out.json"}}}
        self.assertEqual(R.compare(pub, scratch, "/tmp/scratch-1")[0], [])        # a scratch path is not a difference
        tmp = os.path.join(os.path.dirname(R.ROOT), "scratch-2")                  # a clone beside its scratch (/tmp):
        rel = {"cells": {"a": {"decode": {"ms": 95.6}, "split": [1, 2]}, "b": {"x": "../scratch-2/out.json"}}}
        self.assertEqual(R.compare(pub, rel, tmp)[0], [])                         # the relative form is scratch too
        self.assertEqual(len(R.compare(pub, rel, "/tmp/scratch-3")[0]), 1)        # another folder's path still differs

    def test_state_temps(self):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write("STATE before pi4 2026-10-07T16:43:35+07:00 temp=54.5'C throttled=0x0 memavail_mib=7401 load=1.14\n"
                    "STATE before jetson 2026-10-07T16:43:35+07:00 pmode:0000 CPU=39C GPU=38C cpu0=1479000/1479000\n"
                    "noise\n"
                    "STATE after pi4 2026-10-07T16:44:06+07:00 temp=57.4'C throttled=0x80000 memavail_mib=7376 load=1.11\n"
                    "STATE after jetson 2026-10-07T16:44:07+07:00 pmode:0000 CPU=40C GPU=40C cpu0=1479000/1479000\n")
        try:
            self.assertEqual(R.state_temps(f.name), ("pi4:54.5>57.4;jetson:39/38>40/40", "pi4:0x80000"))
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
