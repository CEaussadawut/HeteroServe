#!/usr/bin/env python3
"""Register sp10e (2026-11-03, 22:20): sp10d again, after the Pi 5's fix. sp10d (register_sp10d_20261103.py, registered 20:32)
lost three requests to the Pi 5's network-driver stall and scored INCOMPLETE (testbed/results/sp10d_20261103_203213). The owner
turned the Pi 5's hardware TSO off at about 22:00, and the serve that always hung then held three rounds out of three with the
stall counter unmoved (testbed/results/pi5_tso_check_*). sp10e is sp10d's registration copied: the same plans, predictions,
prompts, sizes, hypotheses and scorer (score_sp10_20261103.py), with its own out_prefix and this note. Nothing is re-derived,
so nothing can drift.

    python3 testbed/scripts/register_sp10e_20261103.py [--dry-run]

Writes testbed/dataset/sp10e_predictions_20261103.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "testbed", "dataset", "sp10d_predictions_20261103.json")
OUT = os.path.join(ROOT, "testbed", "dataset", "sp10e_predictions_20261103.json")


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    reg = json.load(open(SRC))
    reg.update({"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "script": "testbed/scripts/register_sp10e_20261103.py", "out_prefix": "sp10e",
                "what": "sp10d again, after the Pi 5's hardware TSO was turned off: " + reg["what"],
                "copied_from": {"registration": os.path.relpath(SRC, ROOT), "sha256": sha(os.path.relpath(SRC, ROOT)),
                                "why": "sp10d lost three requests to the Pi 5's stall (INCOMPLETE); the stall's fix held its check"}})
    reg["inputs_sha256"] = dict(reg["inputs_sha256"], **{f: sha(f) for f in (
        "testbed/scripts/register_sp10e_20261103.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/sp10_session_20261103.py",
        "testbed/dataset/sp10d_predictions_20261103.json")})
    print("sp10e: plans %s, predictions copied from sp10d (sha256 %s)" % (list(reg["configs"]), reg["copied_from"]["sha256"][:16]))
    if a.dry_run:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))


if __name__ == "__main__":
    main()
