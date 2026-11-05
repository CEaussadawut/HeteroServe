#!/usr/bin/env python3
"""SO-ZERO and SO-BACK for the switch-off test (register_so4_20261024.py), called by ab_score_20261024.py as its scorer_extra:
extra(session, reg, score) -> dict.
  SO-ZERO  every strip sample (<session>/strip.csv) inside a B arm-run's request windows reads the Pi 4's outlet off and 0 W
           (the outlet is reg["outlets"]["pi4"]).
  SO-BACK  <session>/so4_power.log (strip_switch_20261024.py's lines): each B arm-run's "outlet off" is followed by an
           on-node line that is "back", with a boot id other than the one before the poweroff, synchronized within 600 s."""
import csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import score_prima4_20261021 as S4                          # noqa: E402


def extra(session, reg, score):
    o = str(reg["outlets"]["pi4"])
    b_runs = [r for r in score["runs"] if r["arm"] == "B" and r.get("log") and r.get("requests")]
    windows = []
    for r in b_runs:
        windows += [(x["sent_wall"], x["sent_wall"] + x["done_s"]) for x in S4.reqs(os.path.join(ROOT, r["log"]))]
    n = bad = 0
    p = os.path.join(session, "strip.csv")
    if os.path.exists(p) and windows:
        for row in csv.DictReader(open(p)):
            try:
                t = float(row["wall_time"])
            except (TypeError, ValueError):
                continue
            if any(a <= t <= b for a, b in windows):
                n += 1
                out, w = row.get("o%s_output" % o), row.get("o%s_apower" % o)
                if not (out in ("0", "False", "false") and w not in (None, "") and float(w) == 0.0):
                    bad += 1
    zero = {"verdict": "INCOMPLETE" if not b_runs or not n else ("HOLDS" if bad == 0 else "FAILS"),
            "b_arm_runs": [r["i"] for r in b_runs], "samples_in_windows": n, "samples_not_off": bad}
    lines = []
    lp = os.path.join(session, "so4_power.log")
    if os.path.exists(lp):
        lines = [json.loads(x) for x in open(lp) if x.strip()]
    pairs = []
    for k, x in enumerate(lines):
        if x.get("action") == "off-node" and x.get("result") == "outlet off":
            nxt = next((y for y in lines[k + 1:] if y.get("action") == "on-node"), None)
            back = bool(nxt and nxt.get("result") == "back" and nxt.get("boot_id") and nxt["boot_id"] != x.get("boot_id")
                        and (nxt.get("synced_after_s") or 1e9) <= 600)
            pairs.append({"off_at": x.get("at"), "boot_id_before": x.get("boot_id"), "down_after_s": x.get("down_after_s"),
                          "apower_before_switch_w": x.get("apower_before_switch_w"),
                          "on": {k2: nxt.get(k2) for k2 in ("at", "boot_id", "ssh_after_s", "synced_after_s", "overlay_up",
                                                            "ntp_synchronized", "result")} if nxt else None, "back": back})
    sch = len([r for r in score["runs"] if r["arm"] == "B"])
    back = {"verdict": "INCOMPLETE" if not pairs or len(pairs) < sch else ("HOLDS" if all(x["back"] for x in pairs) else "FAILS"),
            "cycles": pairs, "b_arm_runs_scheduled": sch}
    return {"hypotheses": {"SO-ZERO": zero, "SO-BACK": back}}
