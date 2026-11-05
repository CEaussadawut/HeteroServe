#!/usr/bin/env python3
"""Give serve_pool.sh an opt-in SERVE_EXTRA (further llama-server arguments, e.g. "--backend-sampling" for the four-node
head-board sampling study, register_bs4_20261024.py), and update the one test that checks serve_pool.sh's launch line.
Run once, when no session is using serve_pool.sh (after ns4c; chain2d_20261024.sh does it), then the testbed suite:

    python3 testbed/scripts/apply_serve_extra_20261024.py && (cd testbed && python3 -m unittest discover -s tests -q)

Unset (every registration before it), the command line is byte-identical: "$DRAFT${EXTRA:+ $EXTRA}" expands to "$DRAFT".
Tried at 07:05 and reverted before spec4d registered (its literal check in test_spec_serve.py failed), so that spec4d ran on
spec4c's serve_pool.sh. Idempotent: a second run changes nothing. Backups go to .handoff/assets/reproduce-2026-10-24/previous/."""
import os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SP = os.path.join(HERE, "serve_pool.sh")
TEST = os.path.join(ROOT, "testbed", "tests", "test_spec_serve.py")
BAK = os.path.join(ROOT, ".handoff", "assets", "reproduce-2026-10-24", "previous")

EDITS_SP = [
    ('CACHE_RAM=${SERVE_CACHE_RAM:-0}\n',
     'CACHE_RAM=${SERVE_CACHE_RAM:-0}\n'
     '# SERVE_EXTRA (2026-10-24): further llama-server arguments, appended after the draft\'s, e.g. "--backend-sampling" for the\n'
     '# head-board sampling study. Unset (every registration before it), the command line is exactly as before.\n'
     'EXTRA=${SERVE_EXTRA:-}\n'),
    ('  echo "llama-server $ARGS $DRAFT -c $CTX', '  echo "llama-server $ARGS $DRAFT${EXTRA:+ $EXTRA} -c $CTX'),
    ('build/bin/llama-server $ARGS $DRAFT -c $CTX -np 1', 'build/bin/llama-server $ARGS $DRAFT${EXTRA:+ $EXTRA} -c $CTX -np 1'),
]
EDITS_TEST = [
    ('"${DRAFT:+LLAMA_TRACE=1 }setsid nohup $(ldir $HOST)/build/bin/llama-server $ARGS $DRAFT -c $CTX"',
     '"${DRAFT:+LLAMA_TRACE=1 }setsid nohup $(ldir $HOST)/build/bin/llama-server $ARGS $DRAFT${EXTRA:+ $EXTRA} -c $CTX"'),
]


def apply(path, edits, tag):
    s = open(path, encoding="utf-8").read()
    if all(b in s for _, b in edits):
        print("%s: already applied" % os.path.relpath(path, ROOT)); return
    for a, b in edits:
        if s.count(a) != 1:
            sys.exit("refused: %s: %r found %d times" % (os.path.relpath(path, ROOT), a[:60], s.count(a)))
    os.makedirs(BAK, exist_ok=True)
    shutil.copy2(path, os.path.join(BAK, os.path.basename(path) + ".before_serve_extra_" + tag))
    for a, b in edits:
        s = s.replace(a, b)
    open(path, "w", encoding="utf-8").write(s)
    print("%s: applied" % os.path.relpath(path, ROOT))


if __name__ == "__main__":
    apply(SP, EDITS_SP, "applied")
    apply(TEST, EDITS_TEST, "applied")
