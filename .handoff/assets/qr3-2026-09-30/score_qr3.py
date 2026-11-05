#!/usr/bin/env python3
"""Score QR3 against testbed/dataset/qr3_predictions_20260930b.json, applying its registered rules as written:

    python3 .handoff/assets/qr3-2026-09-30/score_qr3.py

For each registered config, the newest testbed/results/qr3_scored_<key>_<stamp>.log (qr3_tpot.py's own log), summarised by
qr3_tpot.summary() -- the same code that printed the run's summary. Writes testbed/dataset/qr3_results_20260930.json.
"""
import glob, json, os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
import qr3_tpot as Q  # noqa: E402

REG = os.path.join(ROOT, "testbed", "dataset", "qr3_predictions_20260930b.json")
OUT = os.path.join(ROOT, "testbed", "dataset", "qr3_results_20260930.json")
MVP, STRETCH, TAIL, DELIVERY, ACCURACY = 500.0, 250.0, 1.10, 2.0, 10.0


def verdict(p95, ci, bound):
    if p95 is None:
        return None
    if ci and ci[1] <= bound:
        return "met"
    if p95 <= bound:
        return "met on the point estimate"
    return "not met"


def health_notes(path):
    """Throttle flags other than the Pi 4's sticky 0x80000, and guard alarms, from the log's health lines."""
    flags, alarms = set(), set()
    for line in open(path):
        parts = line.split(" ", 2)
        if len(parts) < 3 or parts[1] != "health":
            continue
        for m in re.finditer(r"throttled=(0x[0-9a-f]+)", parts[2]):
            flags.add(m.group(1))
        for m in re.finditer(r"alarms=(\d+)", parts[2]):
            alarms.add(int(m.group(1)))
    return sorted(flags), sorted(alarms)


def main():
    reg = json.load(open(REG))
    out = {"registration": os.path.relpath(REG, ROOT), "configs": {}, "rules": {}}
    for key, c in reg["configs"].items():
        logs = sorted(glob.glob(os.path.join(ROOT, c["log_glob"])))
        logs = [l for l in logs if not re.search(r"_(summary|server|memprobe_\w+)\.(log|json)$", l)]
        if not logs:
            out["configs"][key] = {"missing": True}
            continue
        s = Q.summary(logs[-1])
        served = s["meta"].get("served", {})
        args = "%s -c %s -np 1 -t 4" % (served.get("args"), served.get("ctx"))
        plan_ok = args == c["server_args"]
        full = [r for r in s["requests"] if r["tokens"] == 128]
        pc, ps = s["pooled"]["client"], s["pooled"]["server"]
        flags, alarms = health_notes(logs[-1])
        r = {"log": s["log"], "served_args": args, "plan_as_registered": plan_ok, "requests": len(s["requests"]),
             "complete_requests": len(full), "merged_events": s["merged_events"], "client": pc, "server": ps,
             "per_request_mean_tpot": s["per_request_mean_tpot"], "ttft_client_ms": s["ttft_client_ms"],
             "throttle_flags_seen": flags, "guard_alarms_seen": alarms, "predicted_decode_ms": c["predicted"]["decode_ms_per_token"]}
        if plan_ok and pc:
            r["QR3-MVP"] = verdict(pc.get("p95"), pc.get("p95_ci95"), MVP)
            r["QR3-STRETCH"] = verdict(pc.get("p95"), pc.get("p95_ci95"), STRETCH)
            r["server_QR3-MVP"] = verdict(ps.get("p95"), ps.get("p95_ci95"), MVP)
            r["server_QR3-STRETCH"] = verdict(ps.get("p95"), ps.get("p95_ci95"), STRETCH)
            r["PRIMARY_agrees"] = (r["QR3-MVP"] == r["server_QR3-MVP"]) and (r["QR3-STRETCH"] == r["server_QR3-STRETCH"])
            r["tail_ratio"] = round(pc["p95"] / pc["mean"], 4)
            r["TAIL"] = r["tail_ratio"] <= TAIL
            r["delivery_pct"] = round(100 * abs(pc["p95"] - ps["p95"]) / ps["p95"], 3) if ps else None
            r["DELIVERY"] = r["delivery_pct"] is not None and r["delivery_pct"] < DELIVERY
            pred = c["predicted"]["decode_ms_per_token"]
            r["accuracy_pct"] = round(100 * (pred - ps["mean"]) / ps["mean"], 2) if ps else None
            r["ACCURACY"] = r["accuracy_pct"] is not None and abs(r["accuracy_pct"]) <= ACCURACY
        r["COMPLETE"] = len(full) == len(s["requests"]) == 10
        out["configs"][key] = r
    C = out["configs"]
    for rule in ("QR3-MVP", "QR3-STRETCH", "TAIL", "DELIVERY", "ACCURACY", "COMPLETE", "PRIMARY_agrees", "plan_as_registered"):
        out["rules"][rule] = {k: v.get(rule) for k, v in C.items()}
    json.dump(out, open(OUT, "w"), indent=1)
    for k, v in C.items():
        if v.get("missing"):
            print(k, "no scored log"); continue
        pc, ps = v["client"], v["server"]
        print("%-10s plan %s  client mean %.1f p95 %.1f %s  server mean %.1f p95 %.1f  MVP %s  STRETCH %s  tail %.3f  deliv %s%%  acc %s%%"
              % (k, "ok" if v["plan_as_registered"] else "NOT AS REGISTERED", pc["mean"], pc["p95"], pc.get("p95_ci95"), ps["mean"],
                 ps["p95"], v.get("QR3-MVP"), v.get("QR3-STRETCH"), v.get("tail_ratio", 0), v.get("delivery_pct"), v.get("accuracy_pct")))
    print(json.dumps(out["rules"], indent=1))
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
