#!/usr/bin/env python3
"""Build the release folder for the public release of code and data (the user's decision, 23 Oct 2026: prepare a clean
release folder on the laptop now; publish only on the user's word). Nothing here publishes anything.

    python3 .handoff/assets/release-2026-10-24/make_release.py [--out DIR]

It copies an allowlist of the project into DIR (default: ../HeteroServe_release_20261024, beside the project) and never
writes into the project except PRIVACY_SCAN.md beside this script. The release is laid out like the project, so
testbed/scripts/reproduce.py runs from it unchanged:
- testbed/: the record's public documents (README, REPRODUCE, models, benchmarks, energy, prima, node-*.md), scripts/,
  tests/, patches/, dataset/, results/ (every raw log and score), energy/;
- ThinkPad/, Pi5/, Pi4/, Jetson/: the per-node scripts, config, docs and tests;
- .handoff/assets/{pd-2026-09-29, placement-2026-09-30, prima-2026-09-30, qr3-2026-09-30}/: the four scorer folders that
  reproduce.py verify runs (families twouser and 0930); nothing else of .handoff/;
- laptop_home/logs/: every log in the laptop's ~/logs (11 MB). The A/B scorers take each arm-run's server log from ~/logs
  (the first one started after its client log), so verify needs all of them, not only those a score names: with 13 of
  them, four sessions picked up a later draft log (the first release check, 24 Oct 12:29, 233 checks, 9 failing).
  README_RELEASE.md runs verify with HOME=laptop_home; reproduce.py compares paths under a home folder relative to it.
- laptop_home/.cache/heteroserve-models/README.txt: the two model files verify reads (their GGUF tensor tables), named
  with their pinned revisions and SHA-256, to be downloaded there. They are not shipped (9.6 GB).
Left out: testbed/JOURNAL.md (the working journal: it names the user's accounts), testbed/backups/, Documents/, the
retired proposals, every other .handoff/ file, caches. PRIVACY_SCAN.md lists what personal or network detail the release
still carries (home-network addresses, the laptop's user name in paths), for the user to decide on before publishing."""
import argparse, glob, json, os, re, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_OUT = os.path.abspath(os.path.join(ROOT, "..", "HeteroServe_release_20261024"))
FILES = ["testbed/" + f for f in ("README.md", "REPRODUCE.md", "models.md", "benchmarks.md", "energy.md", "prima.md",
                                  "node-jetson.md", "node-pi4.md", "node-pi5.md")]
DIRS = ["testbed/" + d for d in ("scripts", "tests", "patches", "dataset", "results", "energy")] + \
       ["ThinkPad", "Pi5", "Pi4", "Jetson"] + \
       [".handoff/assets/" + d for d in ("pd-2026-09-29", "placement-2026-09-30", "prima-2026-09-30", "qr3-2026-09-30")]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.swp", "*~", ".handoff")
SCAN = [   # (what, regex): reported, never edited
    ("home-network IPv4 (192.168.x.x)", r"\b192\.168\.\d{1,3}\.\d{1,3}\b"),
    ("Tailscale CGNAT IPv4 (100.64/10)", r"\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b"),
    ("MAC address", r"\b[0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){5}\b"),
    ("the laptop's home path (/home/<user>)", r"/home/[a-z_][a-z0-9_-]*"),
    ("e-mail address", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("secret-like assignment", r"(?i)\b(password|passwd|psk|api_?key|secret)\s*[:=]\s*['\"]?[A-Za-z0-9]{4,}"),
    ("private key block", r"BEGIN [A-Z ]*PRIVATE KEY"),
]
TEXT_EXT = {".py", ".sh", ".md", ".json", ".csv", ".txt", ".log", ".tsv", ".yaml", ".yml", ".service", ".conf", ".cfg",
            ".ini", ".patch", ".diff", ".sudoers", ".toml", ".html", ""}


MODELS_README = """Two model files that reproduce.py verify reads (their GGUF tensor tables); not shipped (9.6 GB). Download each
at its pinned revision into this folder and check its SHA-256 (testbed/models.md):

  Qwen3-0.6B-Q8_0.gguf    Qwen/Qwen3-0.6B-GGUF @ 23749fefcc72300e3a2ad315e1317431b06b590a, 639,446,688 bytes
                          sha256 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031   (the costs family)
  Qwen3-14B-Q4_K_M.gguf   Qwen/Qwen3-14B-GGUF @ 530227a7d994db8eca5ab5ced2fb692b614357fd, 9,001,752,960 bytes
                          sha256 500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0   (the 0930 family's prima.cpp score)

Without them those two checks report an error and every other check stands.
"""

README = """# HeteroServe — code and data (release candidate, 24 October 2026)

KMITL senior project CE69-43. An experimental characterisation study: where should the prefill, decode and model layers
of a large language model go across a cluster of heterogeneous low-cost devices to optimise latency, throughput and
energy per token together? The cluster here is a ThinkPad T580 laptop, a Raspberry Pi 5 (8 GB), a Raspberry Pi 4
(8 GB) and a Jetson Nano (4 GB), pooled with llama.cpp's RPC backend at one pinned revision (`b23701f7`). Energy is
measured at the wall on all four outlets.

**This folder is a release candidate, not yet published.** It was prepared on the project laptop. Its licence is still
to be chosen, and the project owner decides whether and where to publish it.

## What is here

| Path | What it is |
|---|---|
| `testbed/README.md` | Start here: the cluster, how it runs, where everything is |
| `testbed/REPRODUCE.md` | How to reproduce: the three levels (re-derive, re-run on boards, reuse the data) |
| `testbed/benchmarks.md` | The record: every study, registered before it ran, with its verdicts as found |
| `testbed/models.md` | Every model file, pinned by upstream revision and SHA-256 |
| `testbed/dataset/` | The registrations (`*predictions*.json`, read-only, each pinning its inputs by SHA-256), the characterisation dataset, link and memory measurements |
| `testbed/results/` | Every run's raw output (logs, power at 5 Hz, throttle polls) and its score |
| `testbed/scripts/`, `testbed/tests/`, `testbed/patches/` | The harness, the planner (`predict_pool.py`), the scorers, their tests, and the patches applied at the pin |
| `ThinkPad/`, `Pi5/`, `Pi4/`, `Jetson/` | Per-node scripts, configuration, docs and tests (`ThinkPad/config/cluster.json` defines the cluster) |
| `.handoff/assets/{pd-2026-09-29,placement-2026-09-30,prima-2026-09-30,qr3-2026-09-30}/` | Four scorer folders that `reproduce.py verify` runs |
| `laptop_home/logs/` | The laptop's `~/logs`: its llama-server and rpc-server logs, from which the A/B scorers read each arm-run's server log |
| `laptop_home/.cache/heteroserve-models/` | Where the two model files `verify` reads go (its `README.txt` names them) |

Left out:
- the working journal (it names the owner's accounts); `testbed/benchmarks.md` is the public record;
- backups, the project's documents (thesis, reports, paper), the planning notes.

## Re-derive every published result (no boards needed)

Tested with Python 3.14.4 and numpy 2.3.5. Some node scripts also use `lxml`, `matplotlib` and `pyserial`.

First download the two model files into `laptop_home/.cache/heteroserve-models/`, as its `README.txt` says. Then, from
this folder:

```bash
HOME=$PWD/laptop_home PYTHONUSERBASE=$(python3 -m site --user-base) python3 testbed/scripts/reproduce.py verify
```

It re-runs every scorer on scratch copies of the raw data and compares each published value. On the project laptop it
took about 54 minutes: 282 checks, 0 failing, the record untouched (28 October 2026, 13:04–13:58, run from this folder with HOME set as below).

Notes on the command:
- `HOME` points the scorers at the laptop's logs and model cache, as they were when the results were scored.
- `PYTHONUSERBASE` keeps your own installed packages visible.
- Paths under a home folder are compared relative to it, so the different `HOME` is not a difference.

`reproduce.py export` writes the CSVs: software version, settings, temperatures and time per run. `manifest` pins every
file by SHA-256.

## Re-run on hardware

See `testbed/REPRODUCE.md`. Every study has three steps, each a script: register (a read-only prediction file written
before anything runs), run (a session under `systemd-run`), and score (HOLDS / FAILS / INCOMPLETE). Boards of the same
model can differ: the replacement Pi 5 decoded 5–15 % slower than the original. Characterise your own boards first.

## Privacy note for the owner, before publishing

The raw logs carry the home network's private addresses (192.168.x.x), two Tailscale addresses, device MAC addresses and
the devices' user names in paths. They are the evidence `verify` re-derives from, so they were not edited. The full list,
with counts, is in the project's `.handoff/assets/release-2026-10-24/PRIVACY_SCAN.md`, outside this folder.

Built by `make_release.py` (a copy is in this folder).
"""


def copy(out):
    for f in FILES:
        src = os.path.join(ROOT, f)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(os.path.join(out, f)), exist_ok=True)
            shutil.copy2(src, os.path.join(out, f))
    for d in DIRS:
        src = os.path.join(ROOT, d)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(out, d), ignore=IGNORE, symlinks=True)
    logs = set()
    for p in glob.glob(os.path.join(ROOT, "testbed", "results", "*", "score.json")):
        logs.update(re.findall(r"/home/[^/\s\"]+/logs/[A-Za-z0-9_.-]+", open(p, encoding="utf-8", errors="replace").read()))
    shutil.copytree(os.path.expanduser("~/logs"), os.path.join(out, "laptop_home", "logs"), ignore=IGNORE)
    found = [l for l in sorted(logs) if os.path.exists(os.path.join(out, "laptop_home", "logs", os.path.basename(l)))]
    cache = os.path.join(out, "laptop_home", ".cache", "heteroserve-models")
    os.makedirs(cache, exist_ok=True)
    with open(os.path.join(cache, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write(MODELS_README)
    with open(os.path.join(out, "README_RELEASE.md"), "w", encoding="utf-8") as fh:
        fh.write(README)
    return sorted(logs), found


def scan(out):
    hits = {w: {} for w, _ in SCAN}
    rx = [(w, re.compile(r)) for w, r in SCAN]
    for dp, _, fs in os.walk(out):
        for f in fs:
            p = os.path.join(dp, f)
            if os.path.splitext(f)[1].lower() not in TEXT_EXT or os.path.getsize(p) > 50 * 1024 * 1024:
                continue
            try:
                t = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for w, r in rx:
                for m in r.finditer(t):
                    v = m.group(0)
                    e = hits[w].setdefault(v, [0, set()])
                    e[0] += 1
                    e[1].add(os.path.relpath(p, out).split("/")[0] + "/" + (os.path.relpath(p, out).split("/")[1]
                                                                          if "/" in os.path.relpath(p, out) else ""))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if out == ROOT or out.startswith(ROOT + os.sep):
        sys.exit("refused: the release goes beside the project, not inside it")
    if os.path.exists(out):
        sys.exit("refused: %s exists; move it aside first" % out)
    named, found = copy(out)
    shutil.copy2(__file__, os.path.join(out, "make_release.py"))
    hits = scan(out)
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs)
    nfiles = sum(len(fs) for _, _, fs in os.walk(out))
    with open(os.path.join(HERE, "PRIVACY_SCAN.md"), "w", encoding="utf-8") as fh:
        fh.write("# What the release folder still carries (scan of %s)\n\n" % out)
        fh.write("Reported, never edited: the raw logs are the evidence that `reproduce.py verify` re-derives every result "
                 "from, so editing them would break it. Decide on each before publishing.\n\n")
        for w, _ in SCAN:
            vals = hits[w]
            fh.write("## %s: %d distinct, %d matches\n\n" % (w, len(vals), sum(v[0] for v in vals.values())))
            for v, (n, where) in sorted(vals.items(), key=lambda x: -x[1][0])[:25]:
                fh.write("- `%s` × %d (in %s)\n" % (v, n, ", ".join(sorted(where))[:200]))
            if len(vals) > 25:
                fh.write("- … %d more\n" % (len(vals) - 25))
            fh.write("\n")
    print("release: %s (%d files, %.1f MB)" % (out, nfiles, size / 1e6))
    print("laptop logs named by the scores: %d, in laptop_home/logs: %d" % (len(named), len(found)))
    for w, _ in SCAN:
        print("scan: %-40s %4d distinct, %6d matches" % (w, len(hits[w]), sum(v[0] for v in hits[w].values())))
    print("privacy scan:", os.path.relpath(os.path.join(HERE, "PRIVACY_SCAN.md"), ROOT))


if __name__ == "__main__":
    main()
