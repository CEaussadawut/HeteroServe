#!/usr/bin/env python3
"""Score the engine-against-prima.cpp four-node session (prima4_session_20261021.sh) by its registration's hypotheses.

    python3 testbed/scripts/score_prima4_20261021.py testbed/results/prima4_<stamp> \\
        --registration testbed/dataset/prima4_predictions_20261021.json

As score_prima_energy_20261020.py, on four nodes: per request (each arm-run's qr3 log), client TPOT samples (qr3_tpot.samples),
TTFT, and the gross energy over its decode window -- first token's event to its last -- at all four outlets (the whole cluster:
laptop, Pi 5, Pi 4, Jetson), with the session's idle-before window (full_eval.window_energy). Per arm, pooled over its runs:
client TPOT P50 / P95, the server's mean and P50 where llama-server gives per-token timings (arm A only), TTFT P50, and J per
decoded token (gen - 1 gaps per request). Then the registered hypotheses:
  P4-ALL      every B arm-run's prima.cpp windows give each of the four nodes a layer, inside its partial copy
  P4-TPOT     the registered faster arm has the lower client TPOT P50
  P4-ENERGY   the registered lower-energy arm has the lower J per decoded token (all four outlets)
  P4-FOLLOW   B/A in J per token within the registered tolerance of B/A in client TPOT P50
  P4-ACC      A's server mean TPOT within the registered tolerance of its prediction
  P4-CORRECT  request 0's first 16 tokens of text are the same in every A and B run (greedy; every request recorded)
Writes score.json and score.txt into the session folder. Nothing is filled in. Since 2026-10-23: an arm is complete with the
registration's min_runs arm-runs' worth of requests where it has one (a deadline may end the session early); each arm-run
carries the laptop battery's state over its requests (battery.tsv), and the session's DEADLINE / charger notes are listed.
"""
import argparse, datetime, hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import qr3_tpot as Q                                        # noqa: E402
import full_eval as FE                                      # noqa: E402


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def records(log, kind):
    pat = re.compile(r"^\S+ %s (\{.*\})\s*$" % re.escape(kind))
    out = []
    for ln in open(log, encoding="utf-8", errors="replace"):
        m = pat.match(ln)
        if m:
            out.append(json.loads(m.group(1)))
    return out


def reqs(log):
    pat = re.compile(r"^\S+ req \d+ (\{.*\})\s*$")
    return [json.loads(m.group(1)) for m in (pat.match(ln) for ln in open(log, encoding="utf-8", errors="replace")) if m]


def battery(session, windows):
    """The laptop's battery over an arm-run's request windows, from the session's battery.tsv (2026-10-23; absent before):
    {"samples", "status": {status: count}, "ac_off": count, "energy_change_wh": the battery's energy_now, last minus first}.
    The wall energy holds the laptop's whole draw only while the battery neither charges nor discharges; this says whether it
    did. None when there is no log or no sample inside the windows."""
    p = os.path.join(session, "battery.tsv")
    if not os.path.exists(p) or not windows:
        return None
    rows = []
    for ln in open(p, errors="replace"):
        f = ln.split()
        if len(f) == 6 and f[0].isdigit():
            t = int(f[0])
            if any(a <= t <= b for a, b in windows):
                rows.append((t, f[1], f[2], int(f[5])))
    if not rows:
        return None
    st = {}
    for _, _, s, _ in rows:
        st[s] = st.get(s, 0) + 1
    return {"samples": len(rows), "status": st, "ac_off": sum(1 for _, ac, _, _ in rows if ac != "1"),
            "energy_change_wh": round((rows[-1][3] - rows[0][3]) / 1e6, 3)}


def text16(x):
    """The first 16 tokens' text: llama-server's events carry it in field 5; prima.cpp's server one token's text per event."""
    ev = [e for e in x["events"] if e[1]][:16]
    return "".join((e[5] if len(e) > 5 and e[5] else (e[1][0] if isinstance(e[1][0], str) else "")) for e in ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("--registration", required=True)
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    outlets = reg["outlets"]
    text = open(os.path.join(a.session, "session.log"), encoding="utf-8", errors="replace").read()
    t = lambda k: float(re.search(r"^%s (\d+\.\d+)$" % k, text, re.M).group(1))
    idle = (t("IDLE_BEFORE start"), t("IDLE_BEFORE end"))
    runs, arms, texts, windows = [], {}, {}, []
    for ln in open(os.path.join(a.session, "arms.tsv")):
        i, arm, rc, lg = ln.rstrip("\n").split("\t")
        path = os.path.join(ROOT, lg) if lg else None
        rr = reqs(path) if path else []
        if arm == "B" and path:
            w = records(path, "prima windows")
            windows.append({"run": int(i), "windows": w[-1] if w else None})
        cg_all, sg_all, ttft, j_all, gaps, per_board = [], [], [], 0.0, 0, {}
        for x in rr:
            cg, sg, merged, tc, ts, ntok = Q.samples(x)
            ev = [e for e in x["events"] if e[1]]
            if not ev:
                continue
            t0, t1 = x["sent_wall"] + ev[0][0], x["sent_wall"] + x["done_s"]
            en = FE.window_energy(a.session, t0, t1, idle, list(outlets.values()))
            g = max(1, ntok - 1)
            j_all += sum(v[0] for v in en.values()); gaps += g
            for node, o in outlets.items():
                per_board[node] = per_board.get(node, 0.0) + en[o][0]
            cg_all += cg; sg_all += sg; ttft.append(tc)
            texts.setdefault(x.get("i"), {}).setdefault(arm, []).append(text16(x))
        bat = battery(a.session, [(x["sent_wall"], x["sent_wall"] + x["done_s"]) for x in rr if x.get("done_s")])
        run = {"i": int(i), "arm": arm, "rc": int(rc), "log": lg, "requests": len(rr), "laptop_battery": bat,
               "client_tpot": Q.stats(cg_all),
               "server_tpot": Q.stats(sg_all), "ttft_ms_p50": round(Q.pct([x for x in ttft if x], 50), 1) if ttft else None,
               "j_per_token_all": round(j_all / gaps, 4) if gaps else None,
               "j_per_token_by_board": {n: round(v / gaps, 4) for n, v in per_board.items()} if gaps else None}
        runs.append(run)
        d = arms.setdefault(arm, {"cg": [], "sg": [], "ttft": [], "j_all": 0.0, "gaps": 0, "requests": 0, "runs": [], "by": {}})
        d["cg"] += cg_all; d["sg"] += sg_all; d["ttft"] += [x for x in ttft if x]; d["j_all"] += j_all
        d["gaps"] += gaps; d["requests"] += len(rr); d["runs"].append(int(i))
        for n, v in per_board.items():
            d["by"][n] = d["by"].get(n, 0.0) + v
    # requests per arm, from the registered order; a registration with min_runs (2026-10-23: a deadline may end the session
    # before its last arm-runs) asks that many arm-runs' worth instead
    want = {k: reg["reqs"] * reg.get("min_runs", {}).get(k, reg["order"].count(k)) for k in ("A", "B")}
    res = {}
    for arm, d in arms.items():
        res[arm] = {"runs": d["runs"], "requests": d["requests"], "complete": d["requests"] >= want.get(arm, 0),
                    "client_tpot": Q.stats(d["cg"]), "server_tpot": Q.stats(d["sg"]),
                    "ttft_ms_p50": round(Q.pct(d["ttft"], 50), 1) if d["ttft"] else None,
                    "j_per_token_all": round(d["j_all"] / d["gaps"], 4) if d["gaps"] else None,
                    "j_per_token_by_board": {n: round(v / d["gaps"], 4) for n, v in d["by"].items()} if d["gaps"] else None}
    H = reg["hypotheses"]
    ok = all(res.get(k, {}).get("complete") for k in ("A", "B"))
    A, B = res.get("A", {}), res.get("B", {})
    pa, pb = A.get("client_tpot", {}).get("p50"), B.get("client_tpot", {}).get("p50")
    ja, jb = A.get("j_per_token_all"), B.get("j_per_token_all")
    tr = pb / pa if pa and pb else None
    jr = jb / ja if ja and jb else None
    v = lambda c: ("HOLDS" if c else "FAILS") if ok else "INCOMPLETE"
    fast, cheap = H["P4-TPOT"]["predicted_lower"], H["P4-ENERGY"]["predicted_lower"]
    p = {"A": pa, "B": pb}; j = {"A": ja, "B": jb}
    other = lambda k: "B" if k == "A" else "A"
    all_ok = bool(windows) and all(w["windows"] and all(x > 0 for x in w["windows"]["windows"].values())
                                   and not w["windows"]["outside_copies"] for w in windows)
    a_mean, a_pred = A.get("server_tpot", {}).get("mean"), reg["arms"]["A"]["predicted"]["decode_ms"]
    same = {str(k): {arm: v for arm, v in d.items()} for k, d in sorted(texts.items(), key=lambda kv: kv[0])}
    match = {k: (len({x for arm in d.values() for x in arm}) == 1) for k, d in same.items() if "A" in d and "B" in d}
    hyp = {"P4-ALL": {"verdict": v(all_ok), "windows": windows},
           "P4-TPOT": {"verdict": v(p[fast] is not None and p[other(fast)] is not None and p[fast] < p[other(fast)]),
                       "predicted_lower": fast, "A_p50": pa, "B_p50": pb, "B_over_A": round(tr, 4) if tr else None},
           "P4-ENERGY": {"verdict": v(j[cheap] is not None and j[other(cheap)] is not None and j[cheap] < j[other(cheap)]),
                         "predicted_lower": cheap, "A_j": ja, "B_j": jb, "B_over_A": round(jr, 4) if jr else None},
           "P4-FOLLOW": {"verdict": v(tr is not None and jr is not None and abs(jr - tr) <= H["P4-FOLLOW"]["tol"]),
                         "tpot_ratio": round(tr, 4) if tr else None, "energy_ratio": round(jr, 4) if jr else None},
           "P4-ACC": {"verdict": v(a_mean is not None and abs(a_mean / a_pred - 1) <= H["P4-ACC"]["tol"]),
                      "A_server_mean": a_mean, "A_predicted": a_pred,
                      "err_pct": round(100 * (a_mean / a_pred - 1), 1) if a_mean else None},
           "P4-CORRECT": {"verdict": v(match.get("0") is True), "on": "request 0 (the others recorded)",
                          "match_by_request": match, "texts": same}}
    s = {"session": os.path.relpath(os.path.abspath(a.session), ROOT),
         "registration": os.path.relpath(os.path.abspath(a.registration), ROOT), "registration_sha256": sha256(a.registration),
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and sha256(os.path.join(ROOT, f)) != h),
         "arms": res, "runs": runs, "hypotheses": hyp,
         "session_notes": re.findall(r"^((?:DEADLINE|NOT ON THE CHARGER|NOT QUIET|FATAL|STOPPED_BY_USER).*)$", text, re.M)}
    json.dump(s, open(os.path.join(a.session, "score.json"), "w"), indent=1)
    L = ["the engine (A) against prima.cpp + Halda (B), Qwen3-32B Q8_0 on laptop + Pi 5 + Pi 4 + Jetson, ctx %d; code changed "
         "since registration: %s" % (reg["ctx"], s["code_changed_since_registration"] or "none"), ""]
    for k, h in hyp.items():
        L.append("%-11s %-10s %s" % (k, h["verdict"], {x: y for x, y in h.items() if x not in ("verdict", "texts", "windows")}))
    for w in windows:
        L.append("B run %d windows: %s" % (w["run"], (w["windows"] or {}).get("windows")))
    L.append("")
    for r in runs:
        L.append("run %d %s rc %d: %d requests, client TPOT P50 %s ms, server mean %s, TTFT P50 %s ms, J/token all %s %s" % (
            r["i"], r["arm"], r["rc"], r["requests"], r["client_tpot"].get("p50"), r["server_tpot"].get("mean"), r["ttft_ms_p50"],
            r["j_per_token_all"], r["j_per_token_by_board"]))
        if r["laptop_battery"]:
            L.append("        laptop battery over its requests: %s" % r["laptop_battery"])
    L += ["session: %s" % n for n in s["session_notes"]]
    open(os.path.join(a.session, "score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
