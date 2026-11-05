#!/usr/bin/env python3
"""Read an item-10 exploration (explore_spec_20261102.py's folder): what a check costs on each plan, and what speculation with
no draft model did there (2026-11-03). Reads only the folder; nothing is fitted to a registered result.

    python3 testbed/scripts/spec_cost_20261103.py testbed/results/explspec_<stamp> [--json out.json]

Per config:
  scan   the prompt-length scan with no speculation: per m, the median of the server's prompt_ms over its repetitions, the
         raw values and their spread; T(m) / T(1); the marginal cost of one more token in a batch as a share of one, (T(m) - T(1)) / ((m - 1) T(1)).
  specs  per speculation setting and prompt: the time per decoded token (the server's predicted_per_token_ms, the mean over
         the requests), against no speculation (speedup > 1 is faster); the drafted and accepted tokens; the passes, i.e.
         the server's decode steps (each yields one token and the drafts it accepted: generated - 1 - accepted) and the time
         of one (predicted_ms / passes); and the server log's own per-request "mean len" where it printed one.
"""
import argparse, json, os, re, statistics, sys

ACC = re.compile(r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def spread_pct(xs):
    xs = [x for x in xs if x]
    return round(100 * (max(xs) / min(xs) - 1), 2) if len(xs) >= 2 else None


def scan(rec):
    by = {}
    for r in rec["scan"]:
        by.setdefault(r["m"], []).append((r.get("timings") or {}).get("prompt_ms"))
    med = lambda xs: statistics.median([x for x in xs if x is not None]) if any(x is not None for x in xs) else None
    t1 = med(by.get(1, []))
    out = []
    for m in sorted(by):
        t = med(by[m])         # the median of the repetitions: one stalled batch (A, m 3: 4,434 against 491 and 477 ms) is not a cost
        out.append({"m": m, "prompt_ms": round(t, 2) if t else None, "raw": by[m], "spread_pct": spread_pct(by[m]),
                    "ratio": round(t / t1, 3) if (t and t1) else None,
                    "marginal_share": round((t - t1) / ((m - 1) * t1), 3) if (t and t1 and m > 1) else None})
    return out


def server_lines(folder, c, sp):
    p = os.path.join(folder, "server_%s_%s.log" % (c, sp))
    if not os.path.exists(p):
        return []
    return [{"ratio": float(a), "accepted": int(b), "generated": int(g), "mean_len": float(L)}
            for a, b, g, L in ACC.findall(open(p, encoding="utf-8", errors="replace").read())]


def specs(folder, recs):
    base = {}
    for rec in recs:
        if rec["spec"] == "none":
            for pk in ("free", "copy"):
                base[pk] = mean([(r.get("timings") or {}).get("predicted_per_token_ms") for r in rec["requests"] if r.get("prompt") == pk])
    out = []
    for rec in recs:
        lines = server_lines(folder, rec["config"], rec["spec"])
        for pk in ("free", "copy"):
            rs = [r for r in rec["requests"] if r.get("prompt") == pk and r.get("timings")]
            if not rs:
                out.append({"spec": rec["spec"], "prompt": pk, "start_rc": rec["start_rc"], "error": rec.get("error"), "requests": 0})
                continue
            t = [r["timings"] for r in rs]
            ms = mean([x.get("predicted_per_token_ms") for x in t])
            dn = sum(x.get("draft_n") or 0 for x in t)
            da = sum(x.get("draft_n_accepted") or 0 for x in t)
            passes = sum((x.get("predicted_n") or 0) - 1 - (x.get("draft_n_accepted") or 0) for x in t)
            pms = sum(x.get("predicted_ms") or 0.0 for x in t)
            out.append({"spec": rec["spec"], "prompt": pk, "requests": len(rs), "ms_per_token": round(ms, 2) if ms else None,
                        "spread_pct": spread_pct([x.get("predicted_per_token_ms") for x in t]),
                        "speedup": round(base[pk] / ms, 3) if (base.get(pk) and ms) else None,
                        "drafted": dn, "accepted": da, "acceptance": round(da / dn, 3) if dn else None,
                        "passes": passes, "ms_per_pass": round(pms / passes, 1) if passes else None,
                        "tokens_per_pass": round(sum((x.get("predicted_n") or 0) - 1 for x in t) / passes, 3) if passes else None})
        if lines:      # every request of the serve that drafted, in order (the warm-up, then free, then copy)
            out[-1]["server_mean_len_every_request"] = [x["mean_len"] for x in lines]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--json")
    a = ap.parse_args()
    recs = json.load(open(os.path.join(a.folder, "results.json")))
    res = {}
    for c in sorted({r["config"] for r in recs}):
        rc = [r for r in recs if r["config"] == c]
        none = next((r for r in rc if r["spec"] == "none"), None)
        res[c] = {"model": rc[0]["model"], "place": rc[0]["place"], "head": rc[0]["head"],
                  "scan": scan(none) if none else [], "specs": specs(a.folder, rc)}
    for c, x in res.items():
        print("config %s: %s on %s (head %s)" % (c, x["model"], x["place"], x["head"]))
        for s in x["scan"]:
            print("  scan m %3d: %9s ms (spread %s %%), x%s of m 1, an extra token %s of one" % (
                s["m"], s["prompt_ms"], s["spread_pct"], s["ratio"], s["marginal_share"]))
        for s in x["specs"]:
            print("  %-6s %-4s %s" % (s["spec"], s["prompt"], {k: v for k, v in s.items() if k not in ("spec", "prompt")}))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
