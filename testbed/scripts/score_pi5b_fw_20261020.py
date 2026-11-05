#!/usr/bin/env python3
"""Score the bootloader follow-up (pi5b_fw_20261020.sh): the replacement Pi 5 after its EEPROM update, against the old board
AND against itself as found (pi5b_char_20261019.sh), by the hypotheses of testbed/dataset/pi5b_fw_predictions_20261020.json.

    python3 testbed/scripts/score_pi5b_fw_20261020.py testbed/results/pi5b_fw_session_<stamp> \\
        --registration testbed/dataset/pi5b_fw_predictions_20261020.json

Run after score_pi5b_20261019.py has scored the same session against the same registration (it rebuilds the rows into
<session>/rows/ and applies RB-PP / RB-TG / RB-MEM / RB-E against the old board). Per matched row (device|model|test):
  gap_as_found = as found / old - 1;   gap_after = after / old - 1;   firmware = after / as found - 1.
Hypotheses (registered):
  FW-ALL      after the update every matched row is within RB's tolerance of the old board's: RB-PP and RB-TG both hold
  FW-HALF-TG  the decode rows' mean |gap_after| is at most half their mean |gap_as_found|
  FW-HALF-PP8 the host Q8_0 pp512 row's |gap_after| is at most half its |gap_as_found|
  FW-NOWORSE  no matched row is slower after the update than as found by more than the registered tolerance
Writes fw_score.json and fw_score.txt into the session folder. Nothing is filled in: a missing row makes a test INCOMPLETE.
"""
import argparse, csv, datetime, hashlib, json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
VOIDING = ("throttled", "exit-", "n<3")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows_of(path):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["node_id"] == "pi5":
            out["%s|%s|%s" % (r["device"], r["model"], r["test"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--registration", required=True)
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    H = reg["fw_hypotheses"]
    rb = json.load(open(os.path.join(a.session, "score.json")))           # score_pi5b_20261019.py's, same registration
    after = rows_of(os.path.join(a.session, "rows", "characterisation.csv"))
    rows = []
    for key, o in sorted(reg["old_rows"].items()):
        af = reg["as_found_rows"].get(key)
        n = after.get(key)
        nt = float(n["mean_ts"]) if n else None
        voided = bool(n) and any(f in (n.get("flags") or "") for f in VOIDING)
        g0 = (af["mean_ts"] / o["mean_ts"] - 1) * 100 if af else None
        g1 = (nt / o["mean_ts"] - 1) * 100 if nt is not None else None
        fw = (nt / af["mean_ts"] - 1) * 100 if (af and nt is not None) else None
        rows.append({"key": key, "phase": o["phase"], "old": o["mean_ts"], "as_found": af["mean_ts"] if af else None,
                     "after": nt, "gap_as_found_pct": g0, "gap_after_pct": g1, "firmware_pct": fw, "voided": voided,
                     "after_flags": (n or {}).get("flags")})
    for key, af in sorted(reg["as_found_rows"].items()):                   # the rows the old board never had
        if key in reg["old_rows"]:
            continue
        n = after.get(key)
        nt = float(n["mean_ts"]) if n else None
        rows.append({"key": key, "phase": af["phase"], "old": None, "as_found": af["mean_ts"], "after": nt,
                     "gap_as_found_pct": None, "gap_after_pct": None,
                     "firmware_pct": (nt / af["mean_ts"] - 1) * 100 if nt is not None else None,
                     "voided": bool(n) and any(f in (n.get("flags") or "") for f in VOIDING), "after_flags": (n or {}).get("flags")})

    def v_all():
        vs = [rb["hypotheses"][h]["verdict"] for h in ("RB-PP", "RB-TG")]
        return "FAILS" if "FAILS" in vs else ("HOLDS" if vs == ["HOLDS", "HOLDS"] else "INCOMPLETE")

    tg = [r for r in rows if r["phase"] == "tg" and r["old"] is not None]
    ok_tg = [r for r in tg if r["gap_after_pct"] is not None and r["gap_as_found_pct"] is not None and not r["voided"]]
    half_tg = None
    if tg and len(ok_tg) == len(tg):
        m0 = sum(abs(r["gap_as_found_pct"]) for r in ok_tg) / len(ok_tg)
        m1 = sum(abs(r["gap_after_pct"]) for r in ok_tg) / len(ok_tg)
        half_tg = {"verdict": "HOLDS" if m1 <= 0.5 * m0 else "FAILS", "mean_abs_gap_as_found_pct": m0, "mean_abs_gap_after_pct": m1}
    else:
        half_tg = {"verdict": "INCOMPLETE", "rows": len(tg), "complete": len(ok_tg)}
    k8 = "cpu|tinyllama-1.1b-q8_0.gguf|pp512"
    r8 = next((r for r in rows if r["key"] == k8), None)
    if r8 and r8["gap_after_pct"] is not None and r8["gap_as_found_pct"] is not None and not r8["voided"]:
        half_pp8 = {"verdict": "HOLDS" if abs(r8["gap_after_pct"]) <= 0.5 * abs(r8["gap_as_found_pct"]) else "FAILS",
                    "gap_as_found_pct": r8["gap_as_found_pct"], "gap_after_pct": r8["gap_after_pct"]}
    else:
        half_pp8 = {"verdict": "INCOMPLETE"}
    tol = H["FW-NOWORSE"]["tol_pct"]
    comp = [r for r in rows if r["firmware_pct"] is not None and not r["voided"]]
    worse = [r["key"] for r in comp if r["firmware_pct"] < -tol]
    noworse = {"verdict": "FAILS" if worse else ("HOLDS" if comp and len(comp) == sum(1 for r in rows if r["as_found"] is not None)
                                                   else "INCOMPLETE"), "tol_pct": tol, "slower": worse}
    s = {"session": os.path.relpath(os.path.abspath(a.session), ROOT),
         "registration": os.path.relpath(os.path.abspath(a.registration), ROOT), "registration_sha256": sha256(a.registration),
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and (not os.path.exists(os.path.join(ROOT, f)) or sha256(os.path.join(ROOT, f)) != h)),
         "hypotheses": {"FW-ALL": {"verdict": v_all(), "RB-PP": rb["hypotheses"]["RB-PP"]["verdict"],
                                   "RB-TG": rb["hypotheses"]["RB-TG"]["verdict"]},
                        "FW-HALF-TG": half_tg, "FW-HALF-PP8": half_pp8, "FW-NOWORSE": noworse},
         "rb_after": {k: {x: v.get(x) for x in ("verdict", "max_abs_diff_pct", "mean_abs_diff_pct", "diff_pct")}
                      for k, v in rb["hypotheses"].items()},
         "rows": rows}
    json.dump(s, open(os.path.join(a.session, "fw_score.json"), "w"), indent=1)
    f = lambda v: "%+.2f" % v if v is not None else "  -  "
    L = ["the replacement Pi 5 after its bootloader update, against the old board and against itself as found",
         "code changed since registration: %s" % (s["code_changed_since_registration"] or "none"), ""]
    for k, h in s["hypotheses"].items():
        L.append("%-12s %-10s %s" % (k, h["verdict"], json.dumps({x: (round(y, 2) if isinstance(y, float) else y)
                                                                   for x, y in h.items() if x != "verdict"})))
    L += ["", "%-50s %9s %9s %9s %8s %8s %9s" % ("row", "old", "as found", "after", "gap0 %", "gap1 %", "firmw. %")]
    for r in rows:
        L.append("%-50s %9s %9s %9s %8s %8s %9s" % (r["key"][:50], "%.3f" % r["old"] if r["old"] else "-",
                                                    "%.3f" % r["as_found"] if r["as_found"] else "-",
                                                    "%.3f" % r["after"] if r["after"] is not None else "MISSING",
                                                    f(r["gap_as_found_pct"]), f(r["gap_after_pct"]), f(r["firmware_pct"])))
    open(os.path.join(a.session, "fw_score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
