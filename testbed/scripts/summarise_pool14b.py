#!/usr/bin/env python3
"""Summarise a run_14b_pooled.sh log that was taken with `-oe jsonl`.

Reads the JSON lines llama-bench writes to stderr (one per test, each carrying every
repetition in `samples_ts`) and prints, per layer split, the mean and the min-max
across repetitions. The min-max is what gets quoted: llama-bench's own `±` is a
standard deviation printed to two decimals, and it reads `± 0.00` whenever the spread
is under 0.005 tok/s -- indistinguishable, on the page, from a one-sample `-r 1` run.

Also checks the claims Result 7 of the proposal makes -- monotonic rise as layers move
off the Pi 4, and the gain of the best split over the even 24/24 split -- and whether
adjacent splits' repetition ranges overlap (if they do, "monotonic" is not supported
at that step).

    python3 testbed/scripts/summarise_pool14b.py testbed/results/pool14b_<stamp>.log
"""
import json
import re
import sys
from collections import defaultdict

LAYERS = 48

# The 2026-09-20 figures, all llama-bench -r 1 (one sample each), as tabled in
# benchmarks.md. (The 2026-09-20 handoff's carry-over list omitted ngl=16; the table has it.)
R1 = {36: (1.19, 0.51), 30: (1.38, 0.57), 24: (1.64, 0.64), 20: (1.87, 0.69),
      16: (2.19, 0.76), 12: (2.59, 0.82), 8: (3.24, 0.90), 6: (3.73, 0.97)}


def load(path):
    runs = defaultdict(dict)        # ngl -> {"pp": record, "tg": record}
    order = []                      # execution order of splits, from the "#####" lines
    thermal = defaultdict(list)     # ngl -> ["before: ...", "after: ..."]
    exits = {}
    cur = None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"##### ngl=(\d+) ", line)
        if m:
            cur = int(m.group(1))
            order.append(cur)
            continue
        if line.startswith(("before:", "after:")) and cur is not None:
            thermal[cur].append(line.strip())
            continue
        m = re.match(r"EXIT_ngl(\d+)=(\d+)", line)
        if m:
            exits[int(m.group(1))] = int(m.group(2))
            continue
        if line.startswith('{"build_commit"'):
            rec = json.loads(line)
            kind = "pp" if rec["n_prompt"] > 0 else "tg"
            runs[rec["n_gpu_layers"]][kind] = rec
    return runs, order, thermal, exits


def stats(rec):
    s = rec["samples_ts"]
    return sum(s) / len(s), min(s), max(s), len(s)


def main(path):
    runs, order, thermal, exits = load(path)
    commits = {r[k]["build_commit"] for r in runs.values() for k in r}
    print(f"log     : {path}")
    print(f"order   : {' '.join(map(str, order))}   commit(s): {', '.join(sorted(commits))}")
    print()
    print(" remote/local | pp64 mean  [min-max]      n | tg16 mean  [min-max]      n |"
          " -r1 pp / tg   | exit")
    print("--------------+-------------------------------+-------------------------------+"
          "---------------+-----")
    done = sorted(runs, reverse=True)
    for ngl in done:
        r = runs[ngl]
        cells = []
        for k in ("pp", "tg"):
            if k in r:
                mean, lo, hi, n = stats(r[k])
                cells.append(f"{mean:6.3f}  [{lo:.3f}-{hi:.3f}]  {n}")
            else:
                cells.append(f"{'(missing)':>27}")
        old = R1.get(ngl)
        old_s = f"{old[0]:.2f} / {old[1]:.2f}" if old else "not run"
        print(f"   {ngl:2d} / {LAYERS - ngl:2d}     | {cells[0]:>29} | {cells[1]:>29} |"
              f" {old_s:>13} | {exits.get(ngl, '?')}")
    print()

    # Result 7's claims, re-derived
    complete = [n for n in done if "pp" in runs[n] and "tg" in runs[n]]
    if 24 in complete and complete:
        best = min(complete)        # smallest offload that ran
        for k, name in (("pp", "prefill"), ("tg", "decode")):
            b, e = stats(runs[best][k])[0], stats(runs[24][k])[0]
            print(f"{name:8}: best split {best}/{LAYERS - best} vs even 24/24 = "
                  f"{b:.3f} / {e:.3f} = {100 * (b / e - 1):+.1f}%")
    if len(complete) >= 2:
        hi_ngl, lo_ngl = max(complete), min(complete)
        for k, name in (("pp", "prefill"), ("tg", "decode")):
            a, z = stats(runs[hi_ngl][k])[0], stats(runs[lo_ngl][k])[0]
            print(f"{name:8}: {hi_ngl}/{LAYERS - hi_ngl} -> {lo_ngl}/{LAYERS - lo_ngl} = "
                  f"{a:.3f} -> {z:.3f} = {z / a:.2f}x")
        print()
        print("monotonic as layers move off the Pi 4 (adjacent splits, mean and range):")
        for k, name in (("pp", "prefill"), ("tg", "decode")):
            steps = []
            for a, b in zip(complete, complete[1:]):           # e.g. 36 -> 30
                ma, la, ha, _ = stats(runs[a][k])
                mb, lb, hb, _ = stats(runs[b][k])
                if mb <= ma:
                    verdict = "NOT RISING"
                elif lb > ha:
                    verdict = "rises, ranges separate"
                else:
                    verdict = "rises, RANGES OVERLAP"
                steps.append(f"    {a:2d}->{b:2d}: {ma:.3f} -> {mb:.3f}  {verdict}")
            print(f"  {name}:")
            print("\n".join(steps))
    print()
    print("thermals (before / after each cell):")
    for ngl in order:
        for t in thermal.get(ngl, []):
            print(f"  ngl={ngl:2d}  {t}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
