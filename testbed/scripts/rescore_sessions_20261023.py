#!/usr/bin/env python3
"""Re-score this week's session results from their raw logs on scratch copies, and compare with the published score.json
leaf by leaf (numbers within 1e-9). Ignored: scoring time, the session path, and code_changed_since_registration.
2026-10-23: restored from the session record (its scratchpad copy was lost); the run of 19:47-21:07 printed 181 / 174 / 178 /
214 / 85 / 186 values compared, 0 different ("1,018 values compared, none different", brief v17).
    python3 testbed/scripts/rescore_sessions_20261023.py <an empty scratch directory>
"""
import json, os, shutil, subprocess, sys
ROOT = "/mnt/data/KMITL/Final/HeteroServe"
TMP = sys.argv[1]
JOBS = [  # (session dir, scorer, registration, extra args)
    ("testbed/results/prima4_20261023_030920", "testbed/scripts/score_prima4_20261021.py", "testbed/dataset/prima4b_predictions_20261023.json", []),
    ("testbed/results/prima4_20261023_111055", "testbed/scripts/score_prima4_20261021.py", "testbed/dataset/prima4c_predictions_20261023.json", []),
    ("testbed/results/spec4_20261023_134113", "testbed/scripts/score_spec4_20261023.py", "testbed/dataset/spec4_predictions_20261023.json", []),
    ("testbed/results/order35_20261023_154835", "testbed/scripts/score_order35_20261023.py", "testbed/dataset/order35b_predictions_20261023.json", []),
    ("testbed/results/spec4b_20261023_170315", "testbed/scripts/score_spec4b_20261023.py", "testbed/dataset/spec4b_predictions_20261023.json", []),
    ("testbed/results/sched2_20261022_032239", "testbed/scripts/score_sched2_20261022.py", "testbed/dataset/sched2_predictions_20261022.json",
     ["--quality", "testbed/results/sched2_quality_20261022_012629.log"]),
]
SKIP = {"scored", "session", "code_changed_since_registration", "registration", "quality_log"}

def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if p == "" and k in SKIP:
                continue
            yield from leaves(v, p + "/" + str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, p + "[%d]" % i)
    else:
        yield p, x

total_diff = 0
for d, scorer, reg, extra in JOBS:
    src = os.path.join(ROOT, d); dst = os.path.join(TMP, os.path.basename(d))
    shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
    for f in ("score.json", "score.txt"):
        if os.path.exists(os.path.join(dst, f)):
            os.remove(os.path.join(dst, f))
    r = subprocess.run([sys.executable, os.path.join(ROOT, scorer), dst, "--registration", os.path.join(ROOT, reg)] +
                       [os.path.join(ROOT, e) if e.startswith("testbed/") else e for e in extra],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(os.path.join(dst, "score.json")):
        print("ERROR %s: rc %d %s" % (d, r.returncode, r.stderr[-300:])); total_diff += 1; continue
    a = dict(leaves(json.load(open(os.path.join(src, "score.json")))))
    b = dict(leaves(json.load(open(os.path.join(dst, "score.json")))))
    diff = [k for k in a if k in b and not (a[k] == b[k] or (isinstance(a[k], (int, float)) and isinstance(b[k], (int, float))
                                                         and abs(a[k] - b[k]) <= 1e-9))]
    only = [k for k in a if k not in b]
    total_diff += len(diff)
    print("%-45s %5d values compared, %d differ, %d published-only%s" % (d, len([k for k in a if k in b]), len(diff), len(only),
                                                                          (": " + ", ".join(diff[:5])) if diff else ""))
print("TOTAL DIFFERENCES", total_diff)
