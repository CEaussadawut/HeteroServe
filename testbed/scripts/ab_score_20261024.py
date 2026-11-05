#!/usr/bin/env python3
"""Score a session of ab_session_20261024.sh against its registration, with the hypotheses the registration declares.

    python3 testbed/scripts/ab_score_20261024.py testbed/results/<out_prefix>_<stamp> --registration <registration.json>

Per arm-run, from its qr3 log (the measurement code of score_spec4b_20261023.py, unchanged in substance): per request the
decode window over the tokens after the first event, the client gaps between single-token events (P50, P95), the server's
own gaps (mean), TTFT, and the gross energy at all four outlets over the decode window (full_eval.window_energy with the
session's idle-before window), also per outlet; request 0's first 16 tokens of text; a draft's acceptance from the host's
server log when the arm serves a draft. Pooled per arm over its arm-runs.

Hypotheses (registration "hypotheses": {name: {"type": ..., ...}}), every verdict HOLDS / FAILS, or INCOMPLETE when an
arm it names is incomplete:
  lower     {"metric", "a", "b"}                 arm a's metric below arm b's
  order     {"metric", "arms": [x, y, z]}         strictly increasing in that order
  best      {"metric", "predicted"}               the predicted arm has the lowest metric
  acc       {"metric", "tol", "arms"?}            each arm within tol of its registered prediction (arms[k].predicted[metric])
  ratio     {"metric", "a", "b", "lo", "hi"}      lo <= a / b <= hi
  diff      {"metric", "a", "b", "lo", "hi"}      lo <= a - b <= hi (the metric's own unit)
  bias      {"metric", "lo", "hi", "arms"?}       lo <= measured / predicted - 1 <= hi for each arm named (default all)
  spread    {"metric", "arm", "tol"}              the arm's arm-runs within tol of each other: (max - min) / mean <= tol
  text_within  {}                                request 0's text the same in every arm-run of each arm
  text_across  {"arms"?}                          request 0's text the same in every arm-run of the named arms (default all)
Metrics: mean_ms_per_token, client_p50_ms, client_p95_ms, server_mean_ms, ttft_ms_p50, j_per_token_all.
A registration's "scorer_extra" (a module path with a function extra(session, reg, result)) may add study-specific
results (a fit, say) after these. Writes score.json and score.txt into the session folder.
"""
import argparse, datetime, glob, importlib.util, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import qr3_tpot as Q                                        # noqa: E402
import full_eval as FE                                      # noqa: E402
import score_prima4_20261021 as S4                          # noqa: E402

ACC = re.compile(r"task +(\d+) \| draft acceptance = [\d.]+ \( *(\d+) accepted / *(\d+) generated\), mean len = +([\d.]+)")
METRICS = ("mean_ms_per_token", "client_p50_ms", "client_p95_ms", "server_mean_ms", "ttft_ms_p50", "j_per_token_all")


def server_log(qr3_log):
    m = re.search(r"_(\d{8}_\d{6})\.log$", qr3_log or "")
    if not m:
        return None
    later = sorted(p for p in glob.glob(os.path.expanduser("~/logs/llama-server_serve_*.log"))
                   if re.search(r"_(\d{8}_\d{6})\.log$", p).group(1) >= m.group(1))
    return later[0] if later else None


def acceptance(path):
    if not path:
        return None
    rows = [(int(t), int(a), int(g), float(ml)) for t, a, g, ml in ACC.findall(open(path, errors="replace").read())]
    rows = rows[1:]                                          # the warm-up request
    if not rows:
        return None
    a, g = sum(r[1] for r in rows), sum(r[2] for r in rows)
    return {"server_log": path, "requests": len(rows), "accepted": a, "drafted": g, "rate": round(a / g, 4) if g else None}


def measure(session, reg):
    """Per arm-run and per arm metrics; texts[request][arm] = [text16 per arm-run]."""
    outlets = reg["outlets"]
    text = open(os.path.join(session, "session.log"), encoding="utf-8", errors="replace").read()
    t = lambda k: float(re.search(r"^%s (\d+\.\d+)$" % k, text, re.M).group(1))
    idle = (t("IDLE_BEFORE start"), t("IDLE_BEFORE end"))
    runs, arms, texts = [], {}, {}
    for ln in open(os.path.join(session, "arms.tsv")):
        i, arm, rc, lg = ln.rstrip("\n").split("\t")
        path = os.path.join(ROOT, lg) if lg else None
        rr = S4.reqs(path) if path else []
        win, tok, ttft, j_all, gaps, cg_all, sg_all = 0.0, 0, [], 0.0, 0, [], []
        per_outlet = {}
        for x in rr:
            ev = [e for e in x["events"] if e[1]]
            if len(ev) < 2:
                continue
            cg, sg, merged, tc, ts, ntok = Q.samples(x)
            w, k = ev[-1][0] - ev[0][0], ntok - len(ev[0][1])
            if k < 1:
                continue
            win += w; tok += k; cg_all += cg; sg_all += sg
            en = FE.window_energy(session, x["sent_wall"] + ev[0][0], x["sent_wall"] + x["done_s"], idle, list(outlets.values()))
            j_all += sum(v[0] for v in en.values()); gaps += max(1, ntok - 1); ttft.append(tc)
            for node, o in outlets.items():
                per_outlet[node] = per_outlet.get(node, 0.0) + en.get(o, en.get(str(o), (0.0,)))[0]
            texts.setdefault(str(x.get("i")), {}).setdefault(arm, []).append(S4.text16(x))
        serve = reg["arms"].get(arm, {}).get("serve_env", {})
        acc = acceptance(server_log(lg)) if serve.get("SERVE_DRAFT") else None
        run = {"i": int(i), "arm": arm, "rc": rc, "log": lg, "requests": len(rr),
               "mean_ms_per_token": round(1000 * win / tok, 2) if tok else None,
               "client_p50_ms": round(Q.pct(cg_all, 50), 2) if cg_all else None,
               "server_mean_ms": round(sum(sg_all) / len(sg_all), 2) if sg_all else None,
               "ttft_ms_p50": round(Q.pct([x for x in ttft if x], 50), 1) if ttft else None,
               "j_per_token_all": round(j_all / gaps, 4) if gaps else None, "draft": acc,
               "laptop_battery": S4.battery(session, [(x["sent_wall"], x["sent_wall"] + x["done_s"]) for x in rr if x.get("done_s")])}
        runs.append(run)
        d = arms.setdefault(arm, {"win": 0.0, "tok": 0, "ttft": [], "j": 0.0, "gaps": 0, "requests": 0, "runs": [],
                                  "acc": [0, 0], "cg": [], "sg": [], "outlet": {}})
        d["win"] += win; d["tok"] += tok; d["ttft"] += [x for x in ttft if x]; d["j"] += j_all; d["gaps"] += gaps
        d["requests"] += len(rr); d["runs"].append(int(i)); d["cg"] += cg_all; d["sg"] += sg_all
        for node, v in per_outlet.items():
            d["outlet"][node] = d["outlet"].get(node, 0.0) + v
        if acc:
            d["acc"][0] += acc["accepted"]; d["acc"][1] += acc["drafted"]
    names = list(reg["arms"])
    want = {k: reg["reqs"] * reg.get("min_runs", {}).get(k, reg["order"].count(k)) for k in names}
    res = {}
    for arm, d in arms.items():
        res[arm] = {"runs": d["runs"], "requests": d["requests"], "complete": d["requests"] >= want.get(arm, 0),
                    "mean_ms_per_token": round(1000 * d["win"] / d["tok"], 2) if d["tok"] else None,
                    "client_p50_ms": round(Q.pct(d["cg"], 50), 2) if d["cg"] else None,
                    "client_p95_ms": round(Q.pct(d["cg"], 95), 2) if d["cg"] else None,
                    "server_mean_ms": round(sum(d["sg"]) / len(d["sg"]), 2) if d["sg"] else None,
                    "ttft_ms_p50": round(Q.pct(d["ttft"], 50), 1) if d["ttft"] else None,
                    "j_per_token_all": round(d["j"] / d["gaps"], 4) if d["gaps"] else None,
                    "j_per_token_by_outlet": {n: round(v / d["gaps"], 4) for n, v in d["outlet"].items()} if d["gaps"] else None,
                    "draft_acceptance_rate": round(d["acc"][0] / d["acc"][1], 4) if d["acc"][1] else None,
                    "predicted": reg["arms"].get(arm, {}).get("predicted", {})}
    return runs, res, texts, text


def verdicts(reg, res, texts, runs=()):
    names = list(reg["arms"])
    out = {}
    for name, h in reg["hypotheses"].items():
        kind, m = h.get("type"), h.get("metric", "mean_ms_per_token")
        involved = (h.get("arms") or [x for x in (h.get("a"), h.get("b")) if x]
                    or ([h["predicted"]] + names if kind == "best" else names))
        ok = all(res.get(k, {}).get("complete") for k in (names if kind in ("best", "text_within") else involved))
        val = lambda k: res.get(k, {}).get(m)
        have = ok and all(val(k) is not None for k in (names if kind == "best" else involved)) if kind not in (
            "text_within", "text_across") else ok
        c, info = None, {}
        if kind == "lower":
            c = have and val(h["a"]) < val(h["b"]); info = {h["a"]: val(h["a"]), h["b"]: val(h["b"])}
        elif kind == "order":
            vs = [val(k) for k in h["arms"]]; c = have and all(x < y for x, y in zip(vs, vs[1:])); info = dict(zip(h["arms"], vs))
        elif kind == "best":
            vals = {k: val(k) for k in names}
            fastest = min(vals, key=lambda k: vals[k]) if have else None
            c = have and fastest == h["predicted"]; info = {"lowest": fastest, "predicted": h["predicted"], "values": vals}
        elif kind == "acc":
            arms = h.get("arms") or names
            err = {k: round(100 * (val(k) / res[k]["predicted"][m] - 1), 2) if val(k) and res.get(k, {}).get("predicted", {}).get(m)
                   else None for k in arms}
            c = have and all(e is not None and abs(e) <= 100 * h["tol"] for e in err.values()); info = {"err_pct": err, "tol": h["tol"]}
        elif kind == "ratio":
            r = val(h["a"]) / val(h["b"]) if have else None
            c = have and h["lo"] <= r <= h["hi"]; info = {"ratio": round(r, 4) if r else None, "lo": h["lo"], "hi": h["hi"]}
        elif kind == "diff":
            dd = val(h["a"]) - val(h["b"]) if have else None
            c = have and h["lo"] <= dd <= h["hi"]; info = {"diff": round(dd, 3) if dd is not None else None, "lo": h["lo"], "hi": h["hi"]}
        elif kind == "bias":
            arms = h.get("arms") or names
            b = {k: round(val(k) / res[k]["predicted"][m] - 1, 4) if val(k) and res.get(k, {}).get("predicted", {}).get(m)
                 else None for k in arms}
            c = have and all(x is not None and h["lo"] <= x <= h["hi"] for x in b.values()); info = {"bias": b, "lo": h["lo"], "hi": h["hi"]}
        elif kind == "spread":
            vs = [r[m] for r in runs if r["arm"] == h["arm"] and r.get(m) is not None]
            sp = (max(vs) - min(vs)) / (sum(vs) / len(vs)) if len(vs) >= 2 else None
            c = sp is not None and sp <= h["tol"]; info = {"runs": vs, "spread": round(sp, 4) if sp is not None else None, "tol": h["tol"]}
        elif kind == "text_within":
            within = {k: len(set(texts.get("0", {}).get(k, []))) == 1 and len(texts.get("0", {}).get(k, [])) >= 2 for k in names}
            c = all(within.values()); info = {"within_arm": within}
        elif kind == "text_across":
            arms = h.get("arms") or names
            vals = {x for k in arms for x in texts.get("0", {}).get(k, [])}
            c = len(vals) == 1 and all(texts.get("0", {}).get(k) for k in arms); info = {"distinct_texts": len(vals)}
        else:
            raise SystemExit("unknown hypothesis type %r in %s" % (kind, name))
        out[name] = dict({"verdict": ("HOLDS" if c else "FAILS") if ok else "INCOMPLETE", "type": kind, "metric": m}, **info)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("--registration", required=True)
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    runs, res, texts, text = measure(a.session, reg)
    hyp = verdicts(reg, res, texts, runs)
    s = {"session": os.path.relpath(os.path.abspath(a.session), ROOT),
         "registration": os.path.relpath(os.path.abspath(a.registration), ROOT), "registration_sha256": S4.sha256(a.registration),
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and S4.sha256(os.path.join(ROOT, f)) != h),
         "arms": res, "runs": runs, "hypotheses": hyp, "texts": texts,
         "session_notes": re.findall(r"^((?:DEADLINE|NOT ON THE CHARGER|NOT QUIET|FATAL|PRE FAILED|STOPPED_BY_USER|!!!!!).*)$",
                                     text, re.M)}
    if reg.get("scorer_extra"):
        spec = importlib.util.spec_from_file_location("extra", os.path.join(ROOT, reg["scorer_extra"]))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        s["extra"] = mod.extra(a.session, reg, s)
    json.dump(s, open(os.path.join(a.session, "score.json"), "w"), indent=1)
    L = ["%s: %s; code changed since registration: %s" % (reg.get("out_prefix"), reg.get("what", ""),
                                                         s["code_changed_since_registration"] or "none"), ""]
    for k, h in hyp.items():
        L.append("%-12s %-10s %s" % (k, h["verdict"], {x: y for x, y in h.items() if x not in ("verdict",)}))
    L.append("")
    for k in reg["arms"]:
        L.append("arm %s: %s" % (k, res.get(k)))
    for r in runs:
        L.append("run %d %s rc %s: %d requests, mean %s ms/token, client P50 %s, server mean %s, TTFT P50 %s ms, J/token all %s%s" % (
            r["i"], r["arm"], r["rc"], r["requests"], r["mean_ms_per_token"], r["client_p50_ms"], r["server_mean_ms"],
            r["ttft_ms_p50"], r["j_per_token_all"], (", draft %s" % {x: y for x, y in r["draft"].items() if x != "server_log"})
            if r["draft"] else ""))
    if s.get("extra"):
        L += ["", "extra: %s" % json.dumps(s["extra"])[:4000]]
    L += ["session: %s" % n for n in s["session_notes"]]
    open(os.path.join(a.session, "score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
