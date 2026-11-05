#!/usr/bin/env python3
"""Score the pool campaign against its pre-registered predictions. Read-only.

    python3 testbed/scripts/compare_pool.py [--pred REGISTRATION] [--out RESULTS] [LOG ...]
                                                          (default: the newest log of every registered cell)

--pred defaults to testbed/dataset/pool_predictions.json (the 2026-09-27 campaign), --out to
testbed/dataset/pool_results.json; a later campaign names its own pair (pool_predictions_20260928.json ->
pool_results_20260928.json). Cells registered after a campaign sit in <REGISTRATION>_addendum.json.

For each cell, from its run_pool.sh log alone:
  measured    pp64 / tg16 t/s from the jsonl samples (mean, min-max) -- those two tests only: a test a cell's
              registered protocol adds (e.g. an unscored pp512 warm-up) is not scored -- against the prediction in
              testbed/dataset/pool_predictions.json -- which was written before the cell ran (its registration time
              and sha256 are checked against the log's own start time)
  split       llama.cpp's own per-device model buffer sizes (-v) against the bytes the plan put on each board, from
              the tensor table: proof the -ngl / -ts arguments realised the planned pipeline
  memory      the memory probe: each board's MemAvailable drop during the cell (the footprint the kernel actually
              had to find) and the process's peak RssAnon / RssFile, against the engine's assumed need
  evidence    exit status, get_throttled after, the Jetson's guard trips and its eth0 byte counters
  page-in     the host's mapped weights (RssFile, from its memory probe log beside the cell's) that grew DURING the scored
              tests' timed repetitions (windows from llama-bench's test_time and samples_ns): a cold first repetition reads
              weights from the SD card inside a timed sample (2026-09-28's "first prefill repetition is cold"; +446 MiB and
              a 43 % slower repetition on 2026-10-07). Flagged over PAGE_IN_MIB; normal cells grew at most 23 MiB.
  thermal     each Pi's 1 Hz throttle poll through the cell (run_pool.sh's THROTTLE lines, from 2026-10-07 night): the
              verdict is recomputed here by throttle_summary.verdict -- clean, THROTTLED (a now-bit set, or a sticky bit
              come on, during the cell) or UNKNOWN (the poll does not cover the cell) -- and a cell takes the worst of its
              Pis. Logs from before the poll read "not polled": a get_throttled snapshot cannot certify a cell.
Writes --out.
"""
import datetime, glob, hashlib, json, os, re, statistics, sys

import throttle_summary

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(ROOT, "testbed", "dataset")
MB = 1024 * 1024
IP = {"192.168.10.11": "pi5", "192.168.10.12": "pi4", "192.168.100.111": "jetson", "192.168.1.109": "jetson", "192.168.1.114": "jetson"}   # old logs keep the old address


def parse(path):
    c = {"log": os.path.relpath(path, ROOT), "records": [], "buffers": {}, "mem": {}, "state": {}, "throttle": {},
         "refused": None}
    for ln in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"########## (\S+) cell (\S+):", ln)
        if m:
            c["started"], c["cell"] = m.group(1), m.group(2)
        if ln.startswith('{"build_commit"'):
            c["records"].append(json.loads(ln))
        m = re.search(r"load_tensors: +(\S+) model buffer size = +([\d.]+) MiB", ln)
        if m:
            dev = m.group(1)
            ip = re.search(r"\[([\d.]+):\d+\]", dev)
            c["buffers"][IP.get(ip.group(1), dev) if ip else dev] = float(m.group(2))
        m = re.match(r"MEMPROBE (\S+) (.*)", ln)
        if m:
            c["mem"][m.group(1)] = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", m.group(2))}
        m = re.match(r"THROTTLE (\S+) (.*)", ln)
        if m:
            c["throttle"][m.group(1)] = dict(re.findall(r"(\w+)=(\S+)", m.group(2)))
        m = re.match(r"STATE (before|after) (\S+) (.*)", ln)
        if m:
            c["state"].setdefault(m.group(2), {})[m.group(1)] = m.group(3)
        m = re.match(r"EXIT_BENCH=(\d+)", ln)
        if m:
            c["exit"] = int(m.group(1))
        if ln.startswith("REFUSED"):
            c["refused"] = ln.strip()
        if ln.startswith("CELL DONE"):
            c["done"] = True
    return c


SCORED = {"pp": (64, 0), "tg": (0, 16)}           # (n_prompt, n_gen) of the pooled protocol's two scored tests
PAGE_IN_MIB = 64                                   # survey of every probed cell, 2026-10-08: normal <= 23 MiB, page-ins >= 169


def page_in(c, path, host):
    """{"pp": MiB, "tg": MiB, "flag": bool} of the host's RssFile growth inside each scored test's timed repetitions, or
    None without the memory probe log or the timings (cells before 2026-09-28 night, or a refused cell)."""
    m = re.match(r"(.*)/pool_(.+)_(\d{8}_\d{6})\.log$", path)
    after = re.search(r"(\S+)", c["state"].get(host, {}).get("after", ""))
    if not m or not after or not c["records"]:
        return None
    probe = "%s/memprobe_pool_%s_%s_%s.log" % (m.group(1), m.group(2), m.group(3), host)
    if not os.path.exists(probe):
        return None
    rss = []
    for ln in open(probe, errors="replace"):
        t, f = re.match(r"(\d+) ", ln), re.search(r"RssFile:(\d+)", ln)
        if t and f:
            rss.append((int(t.group(1)), int(f.group(1)) // 1024))
    ep = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    recs = c["records"]
    out = {}
    for i, r in enumerate(recs):
        ph = next((k for k, v in SCORED.items() if (r.get("n_prompt"), r.get("n_gen")) == v), None)
        if ph is None or not r.get("samples_ns"):
            continue
        end = ep(recs[i + 1]["test_time"]) if i + 1 < len(recs) else ep(after.group(1)) - 1
        start = end - sum(r["samples_ns"]) / 1e9
        w = [x for x in rss if start <= x[0] <= end]
        if len(w) >= 2:
            out[ph] = max(x[1] for x in w) - w[0][1]
    if not out:
        return None
    out["flag"] = any(v > PAGE_IN_MIB for k, v in out.items() if k in SCORED)
    return out


def stats(c, phase):
    rs = [r for r in c["records"] if (r.get("n_prompt"), r.get("n_gen")) == SCORED[phase]]
    s = [x for r in rs for x in (r.get("samples_ts") or [])]
    if len(s) < 3:
        return None
    m = statistics.fmean(s)
    return {"tok_s": round(m, 4), "min": round(min(s), 4), "max": round(max(s), 4), "ms_per_token": round(1000 / m, 1),
            "reps": len(s), "commit": rs[0]["build_commit"]}


def thermal(c, nodes):
    """Each Pi's throttle verdict, recomputed from its THROTTLE fields, and the cell's: the worst of its Pis."""
    per = {}
    for n in nodes:
        if not n.startswith("pi"):
            continue
        f = c["throttle"].get(n)
        per[n] = dict(f, verdict=throttle_summary.verdict(f)) if f is not None else {"verdict": "not polled"}
    vs = {x["verdict"] for x in per.values()}
    if not per:
        cell = "n/a"
    elif "THROTTLED" in vs:
        cell = "THROTTLED"
    elif vs == {"not polled"}:
        cell = "not polled"
    elif vs == {"clean"}:
        cell = "clean"
    else:
        cell = "UNKNOWN"
    return {"verdict": cell, "per_node": per}


def thermal_str(t):
    parts = []
    for n, f in t["per_node"].items():
        if f["verdict"] == "THROTTLED":
            parts.append("%s %s s, bits %s, sticky new %s, min %s MHz, peak %s C" % (
                n, f.get("active_s"), f.get("now_bits"), f.get("sticky_new"), f.get("active_min_clock_mhz"), f.get("max_temp_c")))
        elif f["verdict"] == "clean":
            parts.append("%s peak %s C" % (n, f.get("max_temp_c")))
        elif f["verdict"] == "UNKNOWN":
            parts.append(n)
    return t["verdict"] + ("[%s]" % "; ".join(parts) if parts else "")


def main(argv):
    pred_path, out_path = os.path.join(DS, "pool_predictions.json"), os.path.join(DS, "pool_results.json")
    for flag in ("--pred", "--out"):
        if flag in argv:
            i = argv.index(flag)
            if flag == "--pred":
                pred_path = argv[i + 1]
            else:
                out_path = argv[i + 1]
            del argv[i:i + 2]
    pred = json.load(open(pred_path))
    pred_sha = hashlib.sha256(open(pred_path, "rb").read()).hexdigest()
    add_path = pred_path[:-len(".json")] + "_addendum.json"         # cells registered after the campaign
    add_reg = {}
    if os.path.exists(add_path):
        add = json.load(open(add_path))
        for cid, c in add["configs"].items():
            pred["configs"][cid] = c
            add_reg[cid] = datetime.datetime.fromisoformat(add["registered"])
    shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
    reg = datetime.datetime.fromisoformat(pred["registered"])
    logs = argv or [sorted(glob.glob(os.path.join(ROOT, "testbed", "results", "pool_%s_2*.log" % cid)))[-1]
                    for cid in pred["configs"] if glob.glob(os.path.join(ROOT, "testbed", "results", "pool_%s_2*.log" % cid))]
    out = {"predictions_sha256": pred_sha, "registered": pred["registered"], "cells": {}}
    for path in logs:
        c = parse(path)
        cid = c.get("cell")
        p = pred["configs"].get(cid)
        if not p or "pipeline" not in p:
            continue
        r = {"log": c["log"], "started": c.get("started"), "pipeline": p["pipeline"], "head_on": p["head_on"],
             "registered_before_run": datetime.datetime.fromisoformat(c["started"]) > add_reg.get(cid, reg) if c.get("started") else None}
        if c["refused"]:
            r["refused"] = c["refused"]; out["cells"][cid] = r; continue
        r["exit"], r["finished"] = c.get("exit"), bool(c.get("done"))
        for ph, key in (("tg", "decode"), ("pp", "prefill")):
            st = stats(c, ph)
            want = p["predicted"]["%s_ms_per_token" % key]
            r[key] = {"measured": st, "predicted_ms": want,
                      "error_pct": round(100 * (want - st["ms_per_token"]) / st["ms_per_token"], 1) if st else None}
            # A registration may carry alternative predictors, scored beside the primary one (2026-09-28 night:
            # "v3s", the Pi 5 host's model-size correction); older registrations have none, so they score as before.
            for alt, ap in sorted((p.get("predicted_alt") or {}).items()):
                aw = ap["%s_ms_per_token" % key]
                r[key].setdefault("alt", {})[alt] = {"predicted_ms": aw, "error_pct": round(
                    100 * (aw - st["ms_per_token"]) / st["ms_per_token"], 1) if st else None}
        # split: planned bytes per board against llama.cpp's buffers (workers: RPC buffer; host: CPU_REPACK = its blocks)
        q = shapes[p["model"]]
        lb, start, planned = q["layer_bytes"], 0, {}
        for i, st in enumerate(p["pipeline"]):
            # the board holding the head: the registration's head_on (2026-10-07: the host, with -ot), else the last board
            holder = p.get("head_on") or p["pipeline"][-1]["node"]
            b = sum(lb[start:start + st["blocks"]]) + (q["head_bytes"] if st["node"] == holder else 0)
            planned[st["node"]] = round(b / MB, 1); start += st["blocks"]
        seen = {n: c["buffers"].get(n) for n in planned}
        host = p["pipeline"][0]["node"]
        # a repacking host (the Pi 5) holds its blocks in CPU_REPACK; one that does not (the Pi 4) serves them from
        # the mmap, CPU_Mapped, which also holds the token embedding -- so the plan's figure for it adds that
        if "CPU_REPACK" in c["buffers"]:
            seen[host] = c["buffers"]["CPU_REPACK"]
        else:   # mapped (CPU_Mapped), or read into memory with -lm none (CPU, 2026-10-08)
            seen[host] = sum(v for k, v in c["buffers"].items() if k in ("CPU_Mapped", "CPU"))
            planned[host] = round(planned[host] + q["embd_bytes"] / MB, 1)
        r["split"] = {"planned_mib": planned, "llama_cpp_buffer_mib": seen}
        # memory: MemAvailable drop during the cell, and the process's peak resident set
        r["memory"] = {}
        for n, m in c["mem"].items():
            before = re.search(r"memavail_mib=(\d+)", c["state"].get(n, {}).get("before", ""))
            drop = int(before.group(1)) - m.get("min_avail_mib", 0) if before else None
            r["memory"][n] = {"memavail_before_mib": int(before.group(1)) if before else None,
                              "min_avail_mib": m.get("min_avail_mib"), "drop_mib": drop,
                              "peak_rss_anon_mib": m.get("max_RssAnon_mib"), "peak_rss_file_mib": m.get("max_RssFile_mib"),
                              "engine_need_mib": p["memory"][n]["need_mib"]}
        ev = {}
        for n, s in c["state"].items():
            if n == "jetson":
                b, a = s.get("before", ""), s.get("after", "")
                g = lambda k, t: int(re.search(k + r"=(\d+)", t).group(1)) if re.search(k + r"=(\d+)", t) else None
                ev[n] = {"trips": [g("trips", b), g("trips", a)],
                         "eth0_rx_mb": round((g("eth0_rx", a) - g("eth0_rx", b)) / 1e6, 1) if g("eth0_rx", b) else None,
                         "eth0_tx_mb": round((g("eth0_tx", a) - g("eth0_tx", b)) / 1e6, 1) if g("eth0_tx", b) else None,
                         "pmode_after": (re.search(r"(pmode:\d+)", a) or [None, None])[1] if a else None}
            else:
                t = re.search(r"throttled=(0x[0-9a-fA-F]+)", s.get("after", ""))
                ev[n] = {"throttled_after": t.group(1) if t else None}
        r["evidence"] = ev
        r["thermal"] = thermal(c, [s["node"] for s in p["pipeline"]])
        r["page_in"] = page_in(c, os.path.abspath(path), p["pipeline"][0]["node"])
        out["cells"][cid] = r
    json.dump(out, open(out_path, "w"), indent=1)
    print("%-24s %-30s %-7s %22s %22s  %s" % ("cell", "pipeline", "head", "decode ms meas/pred", "prefill ms meas/pred", "split / memory / evidence"))
    for cid, r in out["cells"].items():
        pl = " + ".join("%s %d" % (s["node"], s["blocks"]) for s in r["pipeline"])
        if r.get("refused"):
            print("%-24s %-30s %-7s REFUSED: %s" % (cid, pl, r["head_on"], r["refused"])); continue
        d, f = r["decode"], r["prefill"]
        fmt = lambda x: ("%7.1f / %7.1f (%+5.1f%%)" % (x["measured"]["ms_per_token"], x["predicted_ms"], x["error_pct"])) if x["measured"] else "      n/a"
        split = " ".join("%s %s/%s" % (n, r["split"]["llama_cpp_buffer_mib"].get(n), v) for n, v in r["split"]["planned_mib"].items())
        mem = " ".join("%s drop %s anon %s" % (n, m["drop_mib"], m["peak_rss_anon_mib"]) for n, m in r["memory"].items())
        pi = r["page_in"]
        pis = ("PAGE-IN[%s]" % " ".join("%s +%d MiB" % (k, v) for k, v in pi.items() if k in SCORED) if pi["flag"]
               else "none") if pi else "n/a"
        print("%-24s %-30s %-7s %s %s  split[%s] mem[%s] exit=%s pre-registered=%s thermal=%s page-in=%s %s"
              % (cid, pl, r["head_on"], fmt(d), fmt(f), split, mem, r["exit"], r["registered_before_run"],
                 thermal_str(r["thermal"]), pis, r["evidence"]))
    print("wrote %s" % os.path.relpath(out_path, ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
