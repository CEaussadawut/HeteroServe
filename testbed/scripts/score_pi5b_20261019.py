#!/usr/bin/env python3
"""Score the replacement Pi 5's characterisation session (pi5b_char_20261019.sh) against its registration.

    python3 testbed/scripts/score_pi5b_20261019.py testbed/results/pi5b_session_<stamp> \\
        --registration testbed/dataset/pi5b_predictions_20261019.json

Rebuilds the session's rows with build_dataset.py into the session folder (rows/), matches each new Pi 5 row to the old
board's row by (device, model, test), and applies the registered tests:
  RB-PP   every matched prefill row within the registered tolerance of the old board's tokens/s
  RB-TG   every matched decode row within it
  RB-MEM  MemAvailable at rest (step 1: the lowest of 10 reads) within its tolerance of the old board's
  RB-E    TinyLlama Q4_K_M at the 2026-10-05 energy protocol: net J/token (trapezoid) per phase within its tolerance of the
          old board's run, and tokens/s per phase within the row tolerance
Verdicts: HOLDS / FAILS / INCOMPLETE. A test FAILS if any valid compared figure is outside its tolerance; otherwise it is
INCOMPLETE if a registered figure is missing or its row is flagged throttled, exit-N or n<3 (warm-start is reported, not
voiding: the old rows carry the same flag rule); otherwise it HOLDS. Nothing is filled in.
Writes score.json and score.txt into the session folder, with any registered code file changed since registration.
"""
import argparse, csv, datetime, hashlib, json, os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
VOIDING = ("throttled", "exit-", "n<3")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def steps(session):
    out = {}
    for ln in open(os.path.join(session, "steps.tsv")):
        n, name, rc, d = (ln.rstrip("\n").split("\t") + [""])[:4]
        out[int(n)] = {"name": name, "rc": int(rc), "dir": d or None}
    return out


def new_rows(session, dirs):
    rows_dir = os.path.join(session, "rows")
    os.makedirs(rows_dir, exist_ok=True)
    p = subprocess.run([sys.executable, os.path.join(ROOT, "testbed", "scripts", "build_dataset.py"), "--out", rows_dir] + dirs,
                       cwd=ROOT, capture_output=True, text=True)
    open(os.path.join(rows_dir, "build_dataset.log"), "w").write(p.stdout + p.stderr)
    if p.returncode:
        raise SystemExit("build_dataset.py failed (%d): %s" % (p.returncode, p.stderr.strip()[-500:]))
    out = {}
    for r in csv.DictReader(open(os.path.join(rows_dir, "characterisation.csv"))):
        if r["node_id"] == "pi5":
            out["%s|%s|%s" % (r["device"], r["model"], r["test"])] = r   # the last row per key, as predict_pool keeps it
    return out


def verdict(items, tol):
    """items: [(key, diff_pct or None, voided bool)]"""
    if any(d is not None and not v and abs(d) > tol for _, d, v in items):
        return "FAILS"
    if any(d is None or v for _, d, v in items):
        return "INCOMPLETE"
    return "HOLDS" if items else "INCOMPLETE"


def pct(new, old):
    return None if new is None or old in (None, 0) else 100.0 * (new / old - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--registration", required=True)
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    H = reg["hypotheses"]
    st = steps(a.session)
    s = {"session": os.path.relpath(os.path.abspath(a.session), ROOT),
         "registration": os.path.relpath(os.path.abspath(a.registration), ROOT),
         "registration_sha256": sha256(a.registration), "registered_at": reg["registered_at"],
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "steps": st}
    s["code_changed_since_registration"] = sorted(
        f for f, h in reg["inputs_sha256"].items()
        if f.endswith((".py", ".sh")) and (not os.path.exists(os.path.join(ROOT, f)) or sha256(os.path.join(ROOT, f)) != h))

    char_dirs = [v["dir"] for k, v in sorted(st.items()) if 2 <= k <= 6 and v["dir"] and v["rc"] == 0]
    new = new_rows(a.session, char_dirs) if char_dirs else {}
    rows, by_phase = [], {"pp": [], "tg": []}
    for key, o in sorted(reg["old_rows"].items()):
        n = new.get(key)
        nt = float(n["mean_ts"]) if n else None
        flags = (n or {}).get("flags", "") or ""
        voided = bool(n) and any(f in flags for f in VOIDING)
        d = pct(nt, o["mean_ts"])
        rows.append({"key": key, "phase": o["phase"], "old_ts": o["mean_ts"], "new_ts": nt, "diff_pct": d,
                     "new_flags": flags, "new_start_temp_c": (n or {}).get("start_temp_c"),
                     "new_rep_range_pct": (n or {}).get("rep_range_pct"), "new_source": (n or {}).get("source"), "voided": voided})
        by_phase[o["phase"]].append((key, d, voided))
    s["rows"] = rows
    s["extra_rows"] = [{"key": k, "mean_ts": float(r["mean_ts"]), "flags": r["flags"], "rep_range_pct": r["rep_range_pct"]}
                       for k, r in sorted(new.items()) if k not in reg["old_rows"]]
    hyp = {}
    for name, ph in (("RB-PP", "pp"), ("RB-TG", "tg")):
        it = by_phase[ph]
        valid = [abs(d) for _, d, v in it if d is not None and not v]
        hyp[name] = {"verdict": verdict(it, H[name]["tol_pct"]), "tol_pct": H[name]["tol_pct"], "rows": len(it),
                     "compared": len(valid), "max_abs_diff_pct": max(valid) if valid else None,
                     "mean_abs_diff_pct": sum(valid) / len(valid) if valid else None,
                     "outside": [k for k, d, v in it if d is not None and not v and abs(d) > H[name]["tol_pct"]]}

    m = st.get(1)
    mem = None
    if m and m["dir"] and m["rc"] == 0:
        mem = json.load(open(os.path.join(ROOT, m["dir"])))["pi5"]
    old_mem = reg["old_memory"]["mem_available_mib"]
    dm = pct(mem["mem_available_mib"] if mem else None, old_mem)
    hyp["RB-MEM"] = {"verdict": verdict([("mem", dm, False)], H["RB-MEM"]["tol_pct"]), "tol_pct": H["RB-MEM"]["tol_pct"],
                     "old_mib": old_mem, "new": mem, "diff_pct": dm}

    e = st.get(7)
    es = None
    if e and e["dir"] and os.path.exists(os.path.join(ROOT, e["dir"], "score.json")):
        es = json.load(open(os.path.join(ROOT, e["dir"], "score.json")))
    old_e = reg["old_energy"]["phases"]
    items, detail = [], {}
    for ph in ("pp", "tg"):
        ne = (es or {}).get("phases", {}).get(ph, {})
        j = (ne.get("j_per_token_net_trap") or {}).get("mean")
        t = (ne.get("tps") or {}).get("mean")
        dj, dt = pct(j, old_e[ph]["j_per_token_net_trap"]), pct(t, old_e[ph]["tps"])
        jc = (ne.get("j_per_token_net_counter") or {}).get("mean")
        detail[ph] = {"j_per_token_net_trap": j, "old": old_e[ph]["j_per_token_net_trap"], "diff_pct": dj,
                      "tps": t, "old_tps": old_e[ph]["tps"], "tps_diff_pct": dt,
                      "j_per_token_net_counter": jc, "old_counter": old_e[ph]["j_per_token_net_counter"],
                      "counter_diff_pct": pct(jc, old_e[ph]["j_per_token_net_counter"]),
                      "mean_w": (ne.get("mean_w") or {}).get("mean"), "old_mean_w": old_e[ph]["mean_w"]}
        items.append(("j_" + ph, dj, False))
    throttled_ok = (es or {}).get("sampler", {}).get("throttled_values") == ["0x0"]
    v = verdict(items, H["RB-E"]["tol_pct"])
    tps_items = [("t_" + ph, detail[ph]["tps_diff_pct"], False) for ph in ("pp", "tg")]
    vt = verdict(tps_items, H["RB-PP"]["tol_pct"])
    if v == "HOLDS" and vt == "FAILS":
        v = "FAILS"
    elif v == "HOLDS" and (vt != "HOLDS" or not throttled_ok):
        v = "INCOMPLETE"
    hyp["RB-E"] = {"verdict": v, "tol_pct": H["RB-E"]["tol_pct"], "tps_verdict": vt, "throttled_0x0": throttled_ok,
                   "run": e["dir"] if e else None, "phases": detail}
    s["hypotheses"] = hyp

    energy = {}
    for k in range(7, 12):
        x = st.get(k)
        if not x or not x["dir"]:
            continue
        p = os.path.join(ROOT, x["dir"], "score.json")
        if not os.path.exists(p):
            energy[x["name"]] = {"run": x["dir"], "score": None}
            continue
        sc = json.load(open(p))
        energy[x["name"]] = {"run": x["dir"], "checks_failed": [c["check"] for c in sc["checks"] if c["pass"] is False],
                             "checks_na": [c["check"] for c in sc["checks"] if c["pass"] is None],
                             "throttled": sc.get("sampler", {}).get("throttled_values"),
                             "phases": {ph: {f: (sc["phases"][ph].get(f) or {}).get("mean")
                                             for f in ("tps", "mean_w", "j_per_token_net_trap", "j_per_token_net_counter",
                                                       "j_per_token_gross_trap", "j_per_token_gross_counter")}
                                        for ph in ("pp", "tg")},
                             "idle_before_w": (sc.get("idle_before") or {}).get("mean_w")}
    s["energy"] = energy

    json.dump(s, open(os.path.join(a.session, "score.json"), "w"), indent=1)
    L = ["replacement Pi 5 (serial %s) against the old board (serial %s); registered %s" % (
        reg["new_board"]["serial"], reg["old_board"]["serial"], reg["registered_at"]),
        "code changed since registration: %s" % (s["code_changed_since_registration"] or "none"), ""]
    for name in ("RB-PP", "RB-TG", "RB-MEM", "RB-E"):
        h = hyp[name]
        extra = ""
        if name in ("RB-PP", "RB-TG"):
            extra = "%d/%d rows compared, max |diff| %s %%, mean |diff| %s %%%s" % (
                h["compared"], h["rows"], "%.2f" % h["max_abs_diff_pct"] if h["max_abs_diff_pct"] is not None else "?",
                "%.2f" % h["mean_abs_diff_pct"] if h["mean_abs_diff_pct"] is not None else "?",
                ("; outside: " + ", ".join(h["outside"])) if h["outside"] else "")
        elif name == "RB-MEM":
            extra = "old %s MiB, new %s, diff %s %%" % (h["old_mib"], (h["new"] or {}).get("mem_available_mib"),
                                                         "%.2f" % h["diff_pct"] if h["diff_pct"] is not None else "?")
        else:
            extra = "; ".join("%s J/token %s vs %.4f (%s %%), t/s %s %%" % (
                ph, "%.4f" % d["j_per_token_net_trap"] if d["j_per_token_net_trap"] is not None else "?", d["old"],
                "%+.2f" % d["diff_pct"] if d["diff_pct"] is not None else "?",
                "%+.2f" % d["tps_diff_pct"] if d["tps_diff_pct"] is not None else "?") for ph, d in h["phases"].items())
        L.append("%-7s %-10s (tol %g %%)  %s" % (name, h["verdict"], h["tol_pct"], extra))
    L += ["", "%-55s %10s %10s %8s  %s" % ("row (device|model|test)", "old t/s", "new t/s", "diff %", "flags")]
    for r in rows:
        L.append("%-55s %10.3f %10s %8s  %s" % (r["key"][:55], r["old_ts"], "%.3f" % r["new_ts"] if r["new_ts"] is not None else "MISSING",
                                                 "%+.2f" % r["diff_pct"] if r["diff_pct"] is not None else "", r["new_flags"]))
    L += ["", "new rows without an old counterpart:"] + ["  %-55s %10.3f  %s" % (x["key"][:55], x["mean_ts"], x["flags"])
                                                         for x in s["extra_rows"]]
    L += ["", "energy per token (net, trapezoid / counter; J):"]
    fmt = lambda v, f: f % v if v is not None else "?"
    for k, x in energy.items():
        ph = x.get("phases") or {}
        cell = lambda p: "%s t/s %s / %s J, %s W" % (
            fmt(ph.get(p, {}).get("tps"), "%.2f"), fmt(ph.get(p, {}).get("j_per_token_net_trap"), "%.4f"),
            fmt(ph.get(p, {}).get("j_per_token_net_counter"), "%.4f"), fmt(ph.get(p, {}).get("mean_w"), "%.2f"))
        L.append("  %-12s pp %s   tg %s   failed checks: %s" % (k, cell("pp"), cell("tg"), x.get("checks_failed")))
    open(os.path.join(a.session, "score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
