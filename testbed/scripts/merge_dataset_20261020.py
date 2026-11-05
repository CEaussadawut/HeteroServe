#!/usr/bin/env python3
"""Merge the rows of 2026-10-11 and 2026-10-20 into the published dataset, by the merge rules registered for them.

    python3 testbed/scripts/merge_dataset_20261020.py --pi5b SESSION --pi5fw SESSION --laptop LAPTOP_SESSION_LOG [--dry-run]

Appends to testbed/dataset/README.md's Rebuild line, in run order (the csv keeps run order among ties, and the planner keeps
the last row per key, so each later run prices its node):
  1. the TinyLlama Q6_K rows of 2026-10-11 (quant_campaign_20261011_035324: Pi 4 and Jetson; _035930: the Pi 4 pooled);
  2. the replacement Pi 5 as found (pi5b_session: steps 2-6), then after its bootloader update (pi5b_fw_session: steps 2-6)
     -- pi5b_predictions_20261019.json and pi5b_fw_predictions_20261020.json, "merge_rule";
  3. the laptop's rows of the night (its laptop_session log: run_t580_rows.sh's two run dirs), after its 2026-09-29 rows.
Keeps the published characterisation.csv and results.csv first as read-only dated copies (*.20261020_before_merge.csv:
every registration up to now quotes those bytes), rebuilds both with build_dataset.py from the new line, checks that every
row of the old file is still there unchanged, and rewrites the Rebuild line. --dry-run builds into a scratch folder only.
"""
import argparse, csv, glob, os, re, shutil, stat, subprocess, sys, tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DS = os.path.join(ROOT, "testbed", "dataset")
README = os.path.join(DS, "README.md")
LINE = re.compile(r"^Rebuild: `python3 testbed/scripts/build_dataset\.py ([^`]+)`", re.M)
Q6K = ["testbed/results/quant_campaign_20261011_035324", "testbed/results/quant_campaign_20261011_035930"]


def steps_dirs(session):
    out = []
    for ln in open(os.path.join(session, "steps.tsv")):
        n, name, rc, d = (ln.rstrip("\n").split("\t") + [""])[:4]
        if 2 <= int(n) <= 6:
            if rc != "0" or not d:
                sys.exit("refused: %s step %s (%s) rc=%s dir=%r" % (session, n, name, rc, d))
            out.append(d)
    if len(out) != 5:
        sys.exit("refused: %s has %d of the 5 row steps" % (session, len(out)))
    return out


def laptop_dirs(log):
    text = open(log, encoding="utf-8", errors="replace").read()
    for s in ("STEP t580_char exit 0", "STEP t580_pool exit 0"):
        if s not in text:
            sys.exit("refused: %s lacks %r" % (log, s))
    ds = re.findall(r"^log: (testbed/results/quant_campaign_\d+_\d+)/t580\.log$", text, re.M)
    if len(ds) != 2:
        sys.exit("refused: %s names %d t580 run dirs, not 2" % (log, len(ds)))
    return ds


def rows(path):
    return list(csv.reader(open(path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi5b", required=True); ap.add_argument("--pi5fw", required=True); ap.add_argument("--laptop", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    readme = open(README).read()
    m = LINE.search(readme)
    old_dirs = m.group(1).split()
    new_dirs = Q6K + steps_dirs(a.pi5b) + steps_dirs(a.pi5fw) + laptop_dirs(a.laptop)
    for d in new_dirs:
        if not os.path.isdir(os.path.join(ROOT, d)):
            sys.exit("refused: %s is not a directory" % d)
        if d in old_dirs:
            sys.exit("refused: %s is already on the Rebuild line" % d)
    dirs = old_dirs + new_dirs
    out = tempfile.mkdtemp(prefix="hs-merge-") if a.dry_run else DS
    if not a.dry_run:
        for f in ("characterisation.csv", "results.csv"):
            keep = os.path.join(DS, f.replace(".csv", ".20261020_before_merge.csv"))
            if not os.path.exists(keep):
                shutil.copy2(os.path.join(DS, f), keep)
                os.chmod(keep, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                print("kept %s as %s (read-only)" % (f, os.path.basename(keep)))
    before = rows(os.path.join(DS, "characterisation.20261020_before_merge.csv") if not a.dry_run else os.path.join(DS, "characterisation.csv"))
    p = subprocess.run([sys.executable, os.path.join(ROOT, "testbed", "scripts", "build_dataset.py"), "--out", out] + dirs, cwd=ROOT,
                       capture_output=True, text=True)
    print(p.stdout[-2000:], p.stderr[-2000:])
    if p.returncode:
        sys.exit("build_dataset.py failed (%d)" % p.returncode)
    after = rows(os.path.join(out, "characterisation.csv"))
    missing = [r for r in before[1:] if r not in after[1:]]
    if before[0] != after[0] or missing:
        sys.exit("refused: %d old rows changed or vanished (header same: %s)" % (len(missing), before[0] == after[0]))
    print("characterisation.csv: %d rows before, %d after (+%d); every old row unchanged" % (len(before) - 1, len(after) - 1,
                                                                                         len(after) - len(before)))
    by = {}
    for r in after[1:]:
        if r not in before[1:]:
            by[r[0]] = by.get(r[0], 0) + 1
    print("new rows by node:", by)
    if a.dry_run:
        print("dry run: built in %s; README untouched" % out)
        return
    readme = readme[:m.start(1)] + " ".join(dirs) + readme[m.end(1):]
    open(README, "w").write(readme)
    print("Rebuild line: %d run dirs (+%d)" % (len(dirs), len(new_dirs)))


if __name__ == "__main__":
    main()
