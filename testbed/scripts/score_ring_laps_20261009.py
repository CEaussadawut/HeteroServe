#!/usr/bin/env python3
"""Score the ring-lap study (register_ring_laps_20261009.py) against its registered hypotheses, as written.

    python3 testbed/scripts/score_ring_laps_20261009.py testbed/results/ring_laps_<stamp>

Reads <dir>/run.log (ring_laps.sh: each cell between "##### CELL ..." and "##### CELL_END ...", llama-bench's JSONL lines
and verbose load log inside; one "PING round= setting=" line per (round, setting) block) and <dir>/throttle_pi4.log.
Per cell: decode and prefill ms/token (the mean of llama-bench's samples), each RPC device's model buffer checked against
its stage's blocks (and the output layer on the last stage) within 0.5 MiB, the Pi 4's thermal verdict over the cell
(full_eval.thermal), and the bench's exit.
  RL-SEQ  at s0, k = 1 has the lowest decode in every round, and the least-squares slope of decode over k (every s0 cell)
          is positive
  RL-RTT  c = mean over rounds of [slope through the origin of (decode d5 - decode s0) over k] / (the round's ping RTT at d5
          - at s0); RTT1 holds for c in [0.75, 1.25], the 2-trip model for c in [1.75, 2.25], otherwise neither
  RL-ACC  every s0 layout's mean decode within 10 % of its RTT1 prediction, 100 x (predicted / measured - 1)
A cell that did not exit 0, or whose layout check fails, is reported and left out; a hypothesis that then lacks its cells is
INCOMPLETE. Exploratory: the per-lap cost at s0 against the predictions, and the per-visit overhead the link terms leave
out (the s0 slope minus RTT1's per-lap link cost). Writes <dir>/score.json.
"""
import hashlib, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
REG = os.path.join(ROOT, "testbed", "dataset", "ring_laps_predictions_20261009.json")
MIB = 2 ** 20
CELL = re.compile(r"^##### CELL round=(\d+) setting=(\w+) layout=(\w+) start=([\d.]+)")
END = re.compile(r"^##### CELL_END round=(\d+) setting=(\w+) layout=(\w+) end=([\d.]+)")
PING = re.compile(r"^PING round=(\d+) setting=(\w+) rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms")
BUF = re.compile(r"load_tensors: +RPC\d*\[([^\]]+)\] model buffer size = +([\d.]+) MiB")
BANDS = {"RTT1": (0.75, 1.25), "RTT2": (1.75, 2.25)}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def parse(lines):
    """({(round, setting, layout): {"t0", "t1", "tg", "pp", "bufs", "exit"}}, {(round, setting): ping avg ms})."""
    cells, pings, cur = {}, {}, None
    for ln in lines:
        m = PING.match(ln)
        if m:
            pings[(int(m.group(1)), m.group(2))] = float(m.group(4))
            continue
        m = CELL.match(ln)
        if m:
            cur = (int(m.group(1)), m.group(2), m.group(3))
            cells[cur] = {"t0": float(m.group(4)), "t1": None, "tg": None, "pp": None, "bufs": {}, "exit": None}
            continue
        if cur is None:
            continue
        c = cells[cur]
        m = END.match(ln)
        if m:
            c["t1"] = float(m.group(4)); cur = None
            continue
        m = BUF.search(ln)
        if m and m.group(1) not in c["bufs"]:
            c["bufs"][m.group(1)] = float(m.group(2))
            continue
        if ln.startswith("EXIT_BENCH="):
            c["exit"] = int(ln.strip().split("=")[1])
            continue
        if ln.startswith('{"build_commit"'):
            d = json.loads(ln)
            s = [x / 1e6 for x in d["samples_ns"]]
            if d["n_gen"] > 0 and d["n_prompt"] == 0:
                c["tg"] = statistics.fmean(s) / d["n_gen"]
            elif d["n_prompt"] > 0 and d["n_gen"] == 0:
                c["pp"] = statistics.fmean(s) / d["n_prompt"]
    return cells, pings


def expected_buffers(layout, q):
    """{endpoint: MiB} for a registered layout: each stage's blocks, and the output layer on the last."""
    eps = layout["rpc"].split(",")
    out, start = {}, 0
    for i, (ep, st) in enumerate(zip(eps, layout["stages"])):
        b = sum(q["layer_bytes"][start:start + st["blocks"]])
        if i == len(eps) - 1:
            b += q["head_bytes"]
        out[ep] = b / MIB
        start += st["blocks"]
    return out


def slope(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def score(reg, q, cells, pings, poll_lines=None, thermal=None):
    labs = list(reg["layouts"])
    k_of = {lab: v["jetson_visits"] for lab, v in reg["layouts"].items()}
    good, bad = {}, {}
    for key, c in cells.items():
        r, st, lab = key
        exp = expected_buffers(reg["layouts"][lab], q)
        diff = {ep: round(c["bufs"].get(ep, -1.0) - mib, 2) for ep, mib in exp.items()}
        ok_layout = all(abs(d) <= 0.5 for d in diff.values()) and set(c["bufs"]) == set(exp)
        th = thermal(poll_lines, c["t0"], c["t1"])["verdict"] if (poll_lines and thermal and c["t1"]) else "UNKNOWN"
        row = {"round": r, "setting": st, "layout": lab, "k": k_of[lab], "decode_ms": c["tg"], "prefill_ms": c["pp"],
               "exit": c["exit"], "layout_ok": ok_layout, "buffer_diff_mib": diff, "thermal": th}
        if c["exit"] != 0 or not ok_layout or c["tg"] is None or c["pp"] is None:
            bad["%d.%s.%s" % key] = row
        else:
            good[key] = row
    want = {(r, st, lab) for r in range(1, reg["rounds"] + 1) for st in reg["settings"] for lab in labs}
    missing = sorted("%d.%s.%s" % k for k in want - set(good))
    hyp = {}
    # RL-SEQ
    s0 = [v for v in good.values() if v["setting"] == "s0"]
    per_round = {}
    for r in range(1, reg["rounds"] + 1):
        row = {v["layout"]: v["decode_ms"] for v in s0 if v["round"] == r}
        if len(row) == len(labs):
            per_round[r] = min(row, key=row.get) == labs[0]
    sl = slope([v["k"] for v in s0], [v["decode_ms"] for v in s0]) if len({v["k"] for v in s0}) > 1 else None
    complete_s0 = len(per_round) == reg["rounds"]
    ok = complete_s0 and all(per_round.values()) and sl is not None and sl > 0
    hyp["RL-SEQ"] = {"verdict": ("HOLDS" if ok else "FAILS") if complete_s0 else
                     ("FAILS" if (per_round and not all(per_round.values())) else "INCOMPLETE"),
                     "k1_fastest_by_round": per_round, "slope_ms_per_lap": round(sl, 3) if sl is not None else None}
    # RL-RTT
    cs = {}
    for r in range(1, reg["rounds"] + 1):
        ks, ds = [], []
        for lab in labs:
            a, b = good.get((r, "s0", lab)), good.get((r, "d5", lab))
            if a and b:
                ks.append(a["k"]); ds.append(b["decode_ms"] - a["decode_ms"])
        drtt = pings.get((r, "d5"), 0.0) - pings.get((r, "s0"), 0.0) if (r, "d5") in pings and (r, "s0") in pings else None
        if len(ks) == len(labs) and drtt:
            b = sum(k * d for k, d in zip(ks, ds)) / sum(k * k for k in ks)
            cs[r] = {"slope_ms_per_visit": round(b, 3), "rtt_increase_ms": round(drtt, 3), "c": round(b / drtt, 3),
                     "increase_by_k": dict(zip(ks, [round(d, 2) for d in ds]))}
    c = statistics.fmean(v["c"] for v in cs.values()) if cs else None
    which = next((name for name, (lo, hi) in BANDS.items() if c is not None and lo <= c <= hi), None)
    hyp["RL-RTT"] = {"verdict": "INCOMPLETE" if len(cs) < reg["rounds"] else
                     ("RTT1 HOLDS" if which == "RTT1" else "2-TRIP MODEL STANDS" if which == "RTT2" else "NEITHER"),
                     "c": round(c, 3) if c is not None else None, "rounds": cs}
    # RL-ACC
    acc = {}
    for lab in labs:
        xs = [v["decode_ms"] for v in s0 if v["layout"] == lab]
        if xs:
            m = statistics.fmean(xs)
            p1, p2 = (reg["layouts"][lab]["predicted"]["s0_rtt%d" % t]["decode_ms"] for t in (1, 2))
            acc[lab] = {"measured_ms": round(m, 2), "rtt1_ms": p1, "rtt2_ms": p2, "err_rtt1_pct": round(100 * (p1 / m - 1), 1),
                        "err_rtt2_pct": round(100 * (p2 / m - 1), 1), "n": len(xs)}
    ok = len(acc) == len(labs) and all(abs(v["err_rtt1_pct"]) <= 10 for v in acc.values())
    hyp["RL-ACC"] = {"verdict": ("HOLDS" if ok else "FAILS") if (len(acc) == len(labs) and complete_s0) else
                     ("FAILS" if any(abs(v["err_rtt1_pct"]) > 10 for v in acc.values()) else "INCOMPLETE"),
                     "layouts": acc}
    # exploratory
    pre_s0 = [v for v in s0]
    sl_pre = slope([v["k"] for v in pre_s0], [v["prefill_ms"] for v in pre_s0]) if len({v["k"] for v in pre_s0}) > 1 else None
    lap = reg["predicted_per_lap_s0_ms"]
    expl = {"per_lap_s0_ms": round(sl, 3) if sl is not None else None, "predicted_per_lap_s0_ms": lap,
            "per_visit_overhead_beyond_rtt1_ms": round(sl - lap["s0_rtt1"], 3) if sl is not None else None,
            "prefill_per_lap_s0_ms_per_token": round(sl_pre, 4) if sl_pre is not None else None,
            "model_20261007_ms_per_extra_pass": "0.6-2.4 by link terms, 2.2-4.4 with the measured RPC visit"}
    return {"cells": {"%d.%s.%s" % k: v for k, v in sorted(good.items())}, "left_out": bad, "missing": missing,
            "pings_ms": {"%d.%s" % k: v for k, v in sorted(pings.items())}, "hypotheses": hyp, "exploratory": expl}


def main(argv):
    if not argv:
        sys.exit(__doc__)
    d = argv[0]
    import full_eval as F
    reg = json.load(open(REG))
    shapes = json.load(open(os.path.join(ROOT, "testbed", "dataset", "model_shapes.json")))
    import predict_pool as P
    q = shapes[P.MODELS[reg["model"]][0]]
    lines = open(os.path.join(d, "run.log"), encoding="utf-8", errors="replace").read().splitlines()
    cells, pings = parse(lines)
    tp = os.path.join(d, "throttle_pi4.log")
    poll = open(tp, encoding="utf-8", errors="replace").readlines() if os.path.exists(tp) else None
    s = score(reg, q, cells, pings, poll, F.thermal)
    code = [f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh")) and os.path.exists(os.path.join(HERE, f))
            and sha(os.path.join(HERE, f)) != h]
    s = {"registration_sha256": sha(REG), "run": os.path.relpath(d, ROOT), "code_changed_since_registration": code, **s}
    json.dump(s, open(os.path.join(d, "score.json"), "w"), indent=1)
    print("%-6s %-3s %8s %8s %8s %8s %8s  %s" % ("set", "lay", "r1", "r2", "r3", "mean", "pred1", "thermal"))
    for st in reg["settings"]:
        for lab in reg["layouts"]:
            xs = [s["cells"].get("%d.%s.%s" % (r, st, lab), {}).get("decode_ms") for r in range(1, reg["rounds"] + 1)]
            th = sorted({s["cells"].get("%d.%s.%s" % (r, st, lab), {}).get("thermal") for r in range(1, reg["rounds"] + 1)} - {None})
            ok = [x for x in xs if x is not None]
            print("%-6s %-3s %s %8s %8.2f  %s" % (st, lab, " ".join("%8.2f" % x if x is not None else "       -" for x in xs),
                                                 "%.2f" % statistics.fmean(ok) if ok else "-",
                                                 reg["layouts"][lab]["predicted"]["%s_rtt1" % st]["decode_ms"], th))
    print("pings: %s" % s["pings_ms"])
    for h, v in s["hypotheses"].items():
        print("%-7s %s  %s" % (h, v["verdict"], {k: w for k, w in v.items() if k not in ("verdict", "layouts", "rounds")}))
    print("RL-RTT rounds: %s" % s["hypotheses"]["RL-RTT"]["rounds"])
    print("RL-ACC: %s" % s["hypotheses"]["RL-ACC"]["layouts"])
    print("exploratory: %s" % s["exploratory"])
    print("left out: %s; missing: %s; code changed since registration: %s" % (list(s["left_out"]), s["missing"], code))
    print("wrote %s" % os.path.relpath(os.path.join(d, "score.json"), ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
