#!/usr/bin/env python3
"""Score the four-node ring-lap study (register_ring4_20261024.py, run by ring4_laps_20261024.sh), as registered.

    python3 testbed/scripts/score_ring4_20261024.py testbed/results/ring4_<stamp> --registration <registration>

From run.log: each cell between its CELL and CELL_END lines, llama-bench's JSONL records: decode ms/token = the mean of the
tg record's samples over n_gen, prefill ms/token = the pp record's over n_prompt; the cell's EXIT_BENCH. Per layout, the
mean over rounds.
  R4-SEQ  k = 1 lowest in every round, and the least-squares slope of decode over k (all rounds) positive.
  R4-ACC  every layout's mean decode within 10 % of its RTT1 prediction.
Exploratory: the measured per-lap cost (the slope) against the predicted one (RTT1, RTT2); their difference is the per-visit
overhead the link terms leave out. A cell with no tg record or a non-zero exit makes its layout, and the verdicts, INCOMPLETE.
Writes score.json and score.txt into the run folder.
"""
import argparse, datetime, hashlib, json, os, re, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CELL = re.compile(r"^##### CELL round=(\d+) setting=(\S+) layout=(\S+) start=([\d.]+)")
END = re.compile(r"^##### CELL_END round=(\d+) setting=(\S+) layout=(\S+) end=([\d.]+)")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--registration", required=True)
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    cells, cur = {}, None
    for ln in open(os.path.join(a.run, "run.log"), errors="replace"):
        m = CELL.match(ln)
        if m:
            cur = (int(m.group(1)), m.group(3)); cells[cur] = {"tg": None, "pp": None, "exit": None}
            continue
        if cur is None:
            continue
        if END.match(ln):
            cur = None; continue
        if ln.startswith("EXIT_BENCH="):
            cells[cur]["exit"] = int(ln.strip().split("=")[1]); continue
        if ln.startswith('{"build_commit"'):
            d = json.loads(ln)
            s = [x / 1e6 for x in d["samples_ns"]]
            if d["n_gen"] > 0 and d["n_prompt"] == 0:
                cells[cur]["tg"] = statistics.fmean(s) / d["n_gen"]
            elif d["n_prompt"] > 0 and d["n_gen"] == 0:
                cells[cur]["pp"] = statistics.fmean(s) / d["n_prompt"]
    labs = list(reg["layouts"])
    ks = {lab: int(lab[1:]) for lab in labs}
    rounds = sorted({r for r, _ in cells})
    ok = all(cells.get((r, lab), {}).get("tg") is not None and cells[(r, lab)]["exit"] == 0
             for r in range(1, reg["rounds"] + 1) for lab in labs)
    per = {lab: [cells[(r, lab)]["tg"] for r in rounds if cells.get((r, lab), {}).get("tg") is not None] for lab in labs}
    mean = {lab: round(statistics.fmean(v), 3) if v else None for lab, v in per.items()}
    pp = {lab: round(statistics.fmean([cells[(r, lab)]["pp"] for r in rounds if cells.get((r, lab), {}).get("pp")]), 3)
          if any(cells.get((r, lab), {}).get("pp") for r in rounds) else None for lab in labs}
    xs = [ks[lab] for r in rounds for lab in labs if cells.get((r, lab), {}).get("tg") is not None]
    ys = [cells[(r, lab)]["tg"] for r in rounds for lab in labs if cells.get((r, lab), {}).get("tg") is not None]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    k1_wins = {r: min(labs, key=lambda lab: cells[(r, lab)]["tg"]) == "k1" for r in rounds
               if all(cells.get((r, lab), {}).get("tg") is not None for lab in labs)}
    err = {lab: round(100 * (mean[lab] / reg["layouts"][lab]["predicted"]["rtt1"]["decode_ms"] - 1), 2) if mean[lab] else None
           for lab in labs}
    v = lambda c: ("HOLDS" if c else "FAILS") if ok else "INCOMPLETE"
    hyp = {"R4-SEQ": {"verdict": v(all(k1_wins.values()) and len(k1_wins) == reg["rounds"] and slope > 0),
                      "k1_lowest_by_round": k1_wins, "slope_ms_per_lap": round(slope, 3)},
           "R4-ACC": {"verdict": v(all(e is not None and abs(e) <= 10 for e in err.values())), "err_pct_rtt1": err}}
    s = {"run": os.path.relpath(os.path.abspath(a.run), ROOT), "registration": os.path.relpath(os.path.abspath(a.registration), ROOT),
         "registration_sha256": sha(a.registration), "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and os.path.exists(os.path.join(HERE, f)) and sha(os.path.join(HERE, f)) != h),
         "complete": ok, "decode_ms_by_round": {lab: per[lab] for lab in labs}, "decode_ms_mean": mean, "prefill_ms_mean": pp,
         "exploratory": {"per_lap_measured_ms": round(slope, 3), "per_lap_predicted_ms": reg["predicted_per_lap_ms"],
                         "per_visit_overhead_ms_vs_rtt1": round(slope - reg["predicted_per_lap_ms"]["rtt1"], 3)},
         "hypotheses": hyp}
    json.dump(s, open(os.path.join(a.run, "score.json"), "w"), indent=1)
    L = ["ring laps on the laptop + Pi 5 (%s); code changed since registration: %s" % (reg["name"], s["code_changed_since_registration"] or "none")]
    for k, h in hyp.items():
        L.append("%-8s %-10s %s" % (k, h["verdict"], {x: y for x, y in h.items() if x != "verdict"}))
    for lab in labs:
        L.append("%s: decode mean %s ms (rounds %s), predicted RTT1 %s / RTT2 %s, prefill %s ms/token" % (
            lab, mean[lab], [round(x, 2) for x in per[lab]], reg["layouts"][lab]["predicted"]["rtt1"]["decode_ms"],
            reg["layouts"][lab]["predicted"]["rtt2"]["decode_ms"], pp[lab]))
    L.append("per lap: measured %.3f ms, predicted %s; overhead beyond RTT1's link terms %.3f ms" % (
        slope, reg["predicted_per_lap_ms"], slope - reg["predicted_per_lap_ms"]["rtt1"]))
    open(os.path.join(a.run, "score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
