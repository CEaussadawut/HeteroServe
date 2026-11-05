#!/usr/bin/env python3
"""Score hm1 (register_hm1_20261104.py, run by handoff_mem_20261104.py run): the hand-off's memory check.

    python3 testbed/scripts/score_hm1_20261104.py testbed/results/hm1_<stamp> --registration testbed/dataset/hm1_predictions_20261104.json

From the run's results.json and run.log: the Jetson's MemAvailable after the warm-ups at the new context (the run's "ready" line),
each case's live verdict (handoff_fits at the MemAvailable read just before it), what its hand-off did (restored, or refused by
the guard and decoded on the laptop), the restore's wall time, the Jetson's SwapFree before and after, and, where registered, plan
L's tokens on the same prompt. Hypotheses HM-READY, HM-AGREE, HM-ALLOW, HM-GUARD as registered; INCOMPLETE unless every case ran.
Writes score.json and score.txt.
"""
import argparse, datetime, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import score_prima4_20261021 as S4                          # noqa: E402
MIB = 1024 * 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--registration", required=True)
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    H = reg["hypotheses"]
    res = {r["case"]: r for r in json.load(open(os.path.join(a.run, "results.json")))}
    ready = None
    for ln in open(os.path.join(a.run, "run.log"), encoding="utf-8", errors="replace"):
        m = re.match(r"\S+ ready (\{.*\})$", ln.strip())
        if m:
            x = json.loads(m.group(1))
            if x.get("ctx") == reg["prediction"]["ctx"]:
                ready = x["jetson"]
    ok = ready is not None and all(c["id"] in res and "error" not in res[c["id"]] for c in reg["cases"])   # a skipped case counts as run
    vs = lambda cond: ("HOLDS" if cond else "FAILS") if ok else "INCOMPLETE"
    rows = []
    for c in reg["cases"]:
        r = res.get(c["id"], {})
        h = r.get("H") or {}
        drop = (r["before"]["SwapFree"] - r["after"]["SwapFree"]) / MIB if r.get("before") and r.get("after") else None
        rows.append({"case": c["id"], "prompt_n": r.get("prompt_n"), "registered": c["registered_verdict"],
                     "live": ("allow" if r["verdict"]["fits"] else "refuse") if r.get("verdict") else None,
                     "live_avail_mib": round(r["verdict"]["avail"] / MIB, 1) if r.get("verdict") else None,
                     "live_margin_mib": round(r["verdict"]["margin_bytes"] / MIB, 1) if r.get("verdict") else None,
                     "forced": bool(c.get("forced_refusal")), "plan": h.get("plan"), "restore_s": h.get("restore_wall_s"),
                     "guard_avail_after_landing_mib": round(h["guard"]["after_landing"]["MemAvailable"] / MIB, 1) if h.get("guard") else None,
                     "swap_drop_mib": round(drop, 1) if drop is not None else None,
                     "tokens_match_L": (r.get("tokens_H") == r.get("tokens_L")) if "tokens_L" in r else None,
                     "error": r.get("error"), "skipped": r.get("skipped")})
    avail = ready["MemAvailable"] / MIB if ready else None
    pred = reg["prediction"]["avail_mib"]
    hyp = {"HM-READY": {"verdict": vs(avail is not None and abs(avail - pred) <= H["HM-READY"]["tol_mib"]),
                        "measured_mib": round(avail, 1) if avail else None, "predicted_mib": pred,
                        "err_mib": round(avail - pred, 1) if avail else None},
           "HM-AGREE": {"verdict": vs(all(x["live"] == x["registered"] for x in rows)),
                        "rows": [{k: x[k] for k in ("case", "registered", "live", "live_avail_mib", "live_margin_mib")} for x in rows]},
           "HM-ALLOW": {"verdict": vs(all(x["plan"] == "H" and (x["restore_s"] or 1e9) <= H["HM-ALLOW"]["restore_s"]
                                          and (x["swap_drop_mib"] is not None and x["swap_drop_mib"] < H["HM-ALLOW"]["swap_drop_mib"])
                                          for x in rows if x["live"] == "allow" and not x["forced"])),
                        "rows": [x for x in rows if x["live"] == "allow" and not x["forced"]]},
           "HM-GUARD": {"verdict": vs(bool([x for x in rows if x["plan"] == "H->L"])
                                      and all(x["tokens_match_L"] is True and x["swap_drop_mib"] is not None
                                              and x["swap_drop_mib"] < H["HM-GUARD"]["swap_drop_mib"]
                                              for x in rows if x["plan"] == "H->L")),
                        "rows": [x for x in rows if x["plan"] == "H->L"]}}
    s = {"run": os.path.relpath(os.path.abspath(a.run), ROOT), "registration": os.path.relpath(os.path.abspath(a.registration), ROOT),
         "registration_sha256": S4.sha256(a.registration), "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, hh in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and S4.sha256(os.path.join(ROOT, f)) != hh),
         "ready": ready, "cases": rows, "hypotheses": hyp, "past_handoffs": reg["past_handoffs"]}
    json.dump(s, open(os.path.join(a.run, "score.json"), "w"), indent=1)
    L = ["hm1: the hand-off's memory check; code changed since registration: %s" % (s["code_changed_since_registration"] or "none"), ""]
    for k, h in hyp.items():
        L.append("%-9s %s  %s" % (k, h["verdict"], {kk: v for kk, v in h.items() if kk not in ("verdict", "rows")}))
        for r in h.get("rows", []):
            L.append("    %s" % r)
    L += ["", "cases:"] + ["    %s" % r for r in rows]
    open(os.path.join(a.run, "score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
