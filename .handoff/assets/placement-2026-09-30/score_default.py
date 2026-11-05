#!/usr/bin/env python3
"""Score DEFAULT against testbed/dataset/pool_predictions_20260930default.json, applying its registered rules as written:

    python3 .handoff/assets/placement-2026-09-30/score_default.py

For each cell, the newest testbed/results/pool_<cell>_<stamp>.log: its llama-bench jsonl (pp64 = prefill, tg16 = decode), and, from
-v, the devices llama.cpp used ("using device RPCk (endpoint)") and the device of every layer ("layer N assigned to device RPCk"), so
the split llama.cpp actually made is read, not assumed. DEFAULT, ACCURACY, SPLIT and STEADY are then applied as registered.
Writes testbed/dataset/pool_results_20260930default.json.
"""
import glob, json, os, re, statistics, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REG = os.path.join(ROOT, "testbed", "dataset", "pool_predictions_20260930default.json")
OUT = os.path.join(ROOT, "testbed", "dataset", "pool_results_20260930default.json")
NODE_OF = {"192.168.10.11:50052": "pi5", "192.168.10.12:50052": "pi4", "192.168.100.111:50052": "jetson"}
N_LAYER = 40


def parse(path):
    txt = open(path, errors="replace").read().splitlines()
    dev, layers, bench = {}, {}, {}
    for line in txt:
        m = re.search(r"using device (RPC\d+) \((\S+)\)", line)
        if m:
            dev[m.group(1)] = NODE_OF.get(m.group(2), m.group(2))
        m = re.search(r"load_tensors: layer +(\d+) assigned to device (\S+?),", line)
        if m:
            layers[int(m.group(1))] = m.group(2)
        if line.startswith("{") and '"samples_ts"' in line:
            r = json.loads(line)
            s = r["samples_ts"]
            ph = "prefill" if r["n_prompt"] else "decode"
            bench[ph] = {"ms_per_token": round(1000 / statistics.fmean(s), 2), "tok_s": round(statistics.fmean(s), 4),
                         "samples_ts": s, "spread_pct": round(100 * (max(s) - min(s)) / statistics.fmean(s), 2)}
    # blocks per device, in --rpc order; the host's own CPU keeps layers below the offload point
    order, split = [], {}
    for il in sorted(layers):
        d = layers[il]
        name = dev.get(d, "host-cpu" if d == "CPU" else d)
        if il < N_LAYER:
            split[name] = split.get(name, 0) + 1
            if name not in order:
                order.append(name)
    head = dev.get(layers.get(N_LAYER), layers.get(N_LAYER))
    exit_ok = any(l.strip() == "EXIT_BENCH=0" for l in txt)
    return {"devices": dev, "split": [(n, split[n]) for n in order], "head_on": head, "bench": bench, "exit_ok": exit_ok}


def main():
    reg = json.load(open(REG))
    out = {"registration": os.path.relpath(REG, ROOT), "cells": {}, "rules": {}}
    for cid, c in reg["configs"].items():
        logs = sorted(glob.glob(os.path.join(ROOT, "testbed", "results", "pool_%s_2*.log" % cid)))
        if not logs:
            out["cells"][cid] = {"missing": True}; continue
        r = parse(logs[-1])
        r["log"] = os.path.relpath(logs[-1], ROOT)
        blocks = [b for _, b in r["split"]]
        pred, split_ok = c["predicted"], True
        if "variants" in c:                                         # the default split: find the variant llama.cpp made
            key = "/".join(str(b) for _, b in r["split"] if _ in ("pi5", "pi4", "jetson"))
            split_ok = key in c["variants"]
            if split_ok:
                pred = c["variants"][key]["predicted"]
            r["variant"] = key
        elif "possible_splits" in c:
            got = [b for n, b in r["split"] if n != "host-cpu"]
            r["split_in_possible"] = got in [list(s) for s in c["possible_splits"]]
            priced = [p["blocks"] for p in c["pipeline"] if p["blocks"]]
            split_ok = got == priced                               # only the priced split has a registered prediction
            if r["split_in_possible"] and not split_ok:
                r["note"] = "a registered possible split with no registered prediction: reported, accuracy not scored"
        else:                                                       # the rule's plan: the registered pipeline itself
            want = [(p["node"], p["blocks"]) for p in c["pipeline"]]
            got = [("pi5" if n == "host-cpu" else n, b) for n, b in r["split"]]
            split_ok = got == want
        r["split_as_registered"] = split_ok
        for ph, key in (("decode", "decode_ms_per_token"), ("prefill", "prefill_ms_per_token")):
            m = r["bench"].get(ph)
            if m and split_ok:
                m["predicted_ms"] = pred[key]
                m["error_pct"] = round(100 * (pred[key] - m["ms_per_token"]) / m["ms_per_token"], 2)
                m["accuracy_holds"] = abs(m["error_pct"]) <= 10
                m["steady"] = m["spread_pct"] <= 5
        out["cells"][cid] = r
    C = out["cells"]
    ok = lambda cid, ph: C.get(cid, {}).get("bench", {}).get(ph, {}).get("ms_per_token")
    if all(ok(x, "decode") for x in ("d.q3_14b.rule", "d.q3_14b.default", "d.q3_14b.readme")):
        rd, dd, md = ok("d.q3_14b.rule", "decode"), ok("d.q3_14b.default", "decode"), ok("d.q3_14b.readme", "decode")
        rp, dp, mp_ = ok("d.q3_14b.rule", "prefill"), ok("d.q3_14b.default", "prefill"), ok("d.q3_14b.readme", "prefill")
        out["rules"]["DEFAULT"] = {"holds": rd < dd and rd < md,
                                   "decode_ms": {"rule": rd, "default": dd, "readme": md},
                                   "decode_rule_less_than_pct": {"default": round(100 * (1 - rd / dd), 1), "readme": round(100 * (1 - rd / md), 1)},
                                   "prefill_ms": {"rule": rp, "default": dp, "readme": mp_},
                                   "prefill_rule_less_than_pct": {"default": round(100 * (1 - rp / dp), 1), "readme": round(100 * (1 - rp / mp_), 1)}}
    out["rules"]["ACCURACY"] = {cid: {ph: (r.get("bench", {}).get(ph) or {}).get("error_pct") for ph in ("decode", "prefill")}
                                for cid, r in C.items()}
    out["rules"]["SPLIT"] = {cid: {"actual": r.get("split"), "head_on": r.get("head_on"), "as_registered": r.get("split_as_registered")}
                             for cid, r in C.items()}
    out["rules"]["STEADY"] = {cid: {ph: (r.get("bench", {}).get(ph) or {}).get("spread_pct") for ph in ("decode", "prefill")}
                              for cid, r in C.items()}
    json.dump(out, open(OUT, "w"), indent=1)
    for cid, r in C.items():
        if r.get("missing"):
            print(cid, "no log"); continue
        b = r["bench"]
        print("%-18s exit %s split %-38s head %-7s decode %s  prefill %s" % (
            cid, "0" if r["exit_ok"] else "!=0", " + ".join("%s %d" % x for x in r["split"]), r["head_on"],
            "%.1f ms (pred %s, %+.1f %%, spread %.1f %%)" % (b["decode"]["ms_per_token"], b["decode"].get("predicted_ms"),
                                                              b["decode"].get("error_pct") or 0, b["decode"]["spread_pct"]) if "decode" in b else "n/a",
            "%.1f ms (pred %s, %+.1f %%, spread %.1f %%)" % (b["prefill"]["ms_per_token"], b["prefill"].get("predicted_ms"),
                                                              b["prefill"].get("error_pct") or 0, b["prefill"]["spread_pct"]) if "prefill" in b else "n/a"))
    for k, v in out["rules"].items():
        print(k, json.dumps(v))
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
