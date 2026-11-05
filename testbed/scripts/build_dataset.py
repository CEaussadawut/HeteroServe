#!/usr/bin/env python3
"""Build the device characterisation dataset from raw llama-bench logs. Stdlib only.

    python3 testbed/scripts/build_dataset.py [--out testbed/dataset] RUN_DIR [RUN_DIR ...]

A RUN_DIR is either a quant campaign (testbed/results/quant_campaign_*: pi5.log / pi4.log from
run_quant_campaign.sh, t580.log from run_t580_rows.sh (the laptop, same format), plus jetson/<run>/run.log), a worker-role run (testbed/results/worker_role_*: the
same pi logs from bench_worker_role.sh, whose rows are device cpu-rpc-worker -- the board serving as a
ggml-rpc-server over loopback, driven by llama-bench on the same board), or a Jetson standalone run dir.
Every figure comes from the `-oe jsonl` records' own samples; nothing is typed in.

Writes, under --out:
  characterisation.csv  one row per node x device x model x test, with provenance (source log
                        and its sha256), the start temperature and the validity flags. Beside
                        the raw throughput it carries two model-independent columns: the effective
                        decode bandwidth (model bytes / time per token) and the effective prefill
                        compute (2 x params x tokens/s). Those are what make one node's cost
                        transferable to a model it never ran alone.
  results.csv           the same rows in the schema ThinkPad/control/profiles.py reads
                        (node_id, phase, model, n_threads, ms_per_token, flags), restricted to each
                        node's SERVING configuration: CPU for the Pis, the Vulkan GPU for the Jetson --
                        and to the characterisation protocol's tests (pp512 / tg128): profiles.py keys a
                        row by phase, so the pooled-protocol rows (pp64 / tg16, 2026-09-28) stay in
                        characterisation.csv only, where predict_pool.py --rows pool reads them.

Flags (a flagged row is kept and shown, never silently dropped):
  throttled   a Pi's get_throttled showed a current-state bit (0-3) after the run, or a sticky
              bit (16-19) that appeared during it
  warm-start  the CPU was at or above the cold threshold when the case started
  exit-N      llama-bench exited non-zero
  n<3         fewer than 3 repetitions -- not reportable (house rule)
A row counts as provisional for placement only through results.csv's flags column, which the
loader reads (analysis.invalidates_throughput decides which flags void a throughput figure).
"""
import csv, hashlib, json, os, re, statistics, sys

JETSON_COLD_C = 40
PI_COLD_C = {"pi5": 50, "pi4": 57, "t580": 65}
SERVING = {"pi5": "cpu", "pi4": "cpu", "jetson": "vulkan", "t580": "cpu"}
PROFILE_TESTS = ("pp512", "tg128")          # results.csv: the tests profiles.py was built on


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def quant_of(model_filename):
    m = re.search(r"(q\d_k_[ms]|q\d_\d|q\dk[ms]|q\d_k|f16|bf16)", os.path.basename(model_filename).lower())   # q\d_k: Q6_K (2026-10-11)
    q = m.group(1) if m else "?"
    short = re.fullmatch(r"q(\d)k([ms])", q)       # the boards' short names: tinyllama-1.1b-q4km, -q3km
    return "Q%s_K_%s" % (short.group(1), short.group(2).upper()) if short else q.upper()


def row_from_record(rec, node, device, source, flags, start_temp, when):
    s = rec.get("samples_ts") or []
    test = ("pp%d" % rec["n_prompt"]) if rec.get("n_prompt") else ("tg%d" % rec["n_gen"])
    mean = statistics.fmean(s) if s else None
    fl = list(flags)
    if len(s) < 3:
        fl.append("n<3")
    row = {
        "node_id": node, "device": device, "model": os.path.basename(rec["model_filename"]),
        "quant": quant_of(rec["model_filename"]), "n_params": rec.get("model_n_params"),
        "model_bytes": rec.get("model_size"), "test": test, "phase": test[:2],
        "n_threads": rec.get("n_threads"), "n_gpu_layers": rec.get("n_gpu_layers"),
        "reps": len(s), "mean_ts": round(mean, 4) if mean else None,
        "min_ts": round(min(s), 4) if s else None, "max_ts": round(max(s), 4) if s else None,
        # (max-min)/mean of the repetitions of ONE invocation: a within-run range, not the
        # across-run spread (the same config minutes apart drifts ~2%).
        "rep_range_pct": round(100 * (max(s) - min(s)) / mean, 2) if mean else None,
        "ms_per_token": round(1000.0 / mean, 3) if mean else None,
        "start_temp_c": start_temp, "flags": ";".join(fl), "date": when or rec.get("test_time", ""),
        "build_commit": rec.get("build_commit"), "source": source, "source_sha256": sha256(source),
    }
    if mean and row["model_bytes"] and row["phase"] == "tg":
        row["eff_decode_GBps"] = round(row["model_bytes"] / (1.0 / mean) / 1e9, 3)
    if mean and row["n_params"] and row["phase"] == "pp":
        row["eff_prefill_GFLOPs"] = round(2 * row["n_params"] * mean / 1e9, 2)
    return row


def throttle_flags(before, after):
    def bits(line):
        m = re.search(r"throttled=0x([0-9a-fA-F]+)", line or "")
        return int(m.group(1), 16) if m else None
    b, a = bits(before), bits(after)
    if a is None:
        return ["throttle-unknown"]
    if a & 0xF or (b is not None and (a & 0xF0000) & ~(b & 0xF0000)):
        return ["throttled"]
    return []


def intel_throttle_flags(before, after):
    """The T580 has no vcgencmd. Its STATE lines carry cpu0's Intel thermal-throttle event counters (core/package,
    run_t580_rows.sh); a rise during the row means the CPU throttled for heat. No counters, no verdict."""
    def ev(line):
        m = re.search(r"throttle_events=(\d+)/(\d+)", line or "")
        return (int(m.group(1)), int(m.group(2))) if m else None
    b, a = ev(before), ev(after)
    if a is None or b is None:
        return ["throttle-unknown"]
    return ["throttled"] if a[0] > b[0] or a[1] > b[1] else []


def parse_pi_log(path, node):
    rows, lines = [], open(path, encoding="utf-8", errors="replace").read().splitlines()
    cur, pending, end_line = None, [], None
    for ln in lines:
        m = re.match(r"##### (\S+) (\S+)\s+(?:role=(\S+)\s+)?cold gate: waited (\d+)s, now (\d+) C", ln)
        if m:
            cur = {"model": m.group(2), "temp": int(m.group(5)), "recs": [], "before": "", "after": "", "exit": None,
                   "device": "cpu" if not m.group(3) else "cpu-" + m.group(3)}
            continue
        if cur is None:
            continue
        if ln.startswith("STATE_BEFORE"):
            cur["before"] = ln
        elif ln.startswith('{"build_commit"'):
            cur["recs"].append(json.loads(ln))
        elif ln.startswith("EXIT_"):
            cur["exit"] = int(ln.rsplit("=", 1)[1])
        elif ln.startswith("STATE_AFTER"):
            cur["after"] = ln
            fl = [] if node == "t580" else throttle_flags(cur["before"], cur["after"])   # t580: flagged below
            if cur["temp"] >= PI_COLD_C.get(node, 99):
                fl.append("warm-start")
            if cur["exit"] not in (0, None):
                fl.append("exit-%d" % cur["exit"])
            when = cur["before"].split()[1] if len(cur["before"].split()) > 1 else ""
            for rec in cur["recs"]:
                rows.append(row_from_record(rec, node, cur["device"], path, fl, cur["temp"], when))
            if node == "t580":
                pending.append((cur["before"], cur["after"], len(rows) - len(cur["recs"]), len(rows)))
            cur = None
        elif ln.startswith("STATE_END"):
            end_line = ln
    # The T580's Intel counters register a throttle a moment AFTER the load that caused it (2026-09-29: 24 -> 52 and
    # 59 -> 80/85 both appeared in the idle cold-gate wait right after the hottest rows, never between a row's own
    # STATE_BEFORE and STATE_AFTER). So a row's window runs to the next row's STATE_BEFORE, or for the last row to the
    # log's STATE_END (a reading taken 30 s after it, run_t580_rows.sh) or its own STATE_AFTER. Conservative: the
    # gap between rows is idle, so this can only add flags.
    for i, (before, after, a0, a1) in enumerate(pending):
        until = pending[i + 1][0] if i + 1 < len(pending) else (end_line or after)
        for r in rows[a0:a1]:
            r["flags"] = ";".join(filter(None, [r["flags"]] + intel_throttle_flags(before, until)))
    return rows


def parse_jetson_run(path):
    rows, case, starts, exits, recs = [], None, {}, {}, {}
    for ln in open(path, encoding="utf-8", errors="replace").read().splitlines():
        m = re.match(r"##### case (\w+):", ln)
        if m:
            case = m.group(1); recs.setdefault(case, []); continue
        m = re.match(r"\[before (\w+)\] (\S+).*CPU-therm=(\d+)C", ln)
        if m:
            starts[m.group(1)] = (int(m.group(3)), m.group(2)); continue
        m = re.match(r"EXIT_(\w+)=(\d+)", ln)
        if m:
            exits[m.group(1)] = int(m.group(2)); continue
        if ln.startswith('{"build_commit"') and case:
            recs[case].append(json.loads(ln))
    for c, rs in recs.items():
        temp, when = starts.get(c, (None, ""))
        fl = []
        if temp is None or temp >= JETSON_COLD_C:
            fl.append("warm-start")
        if exits.get(c, None) not in (0,):
            fl.append("exit-%s" % exits.get(c, "missing"))
        for rec in rs:
            # the GPU backend from the record itself (2026-09-29: the Jetson now has a CUDA build at the pin, and a CUDA row
            # labelled vulkan would be priced as the other backend); older records carry "Vulkan"
            gpu = "cuda" if "CUDA" in str(rec.get("backends", "")) else "vulkan"
            dev = gpu if (rec.get("n_gpu_layers") or 0) > 0 and rec.get("devices") != "none" else "cpu"
            rows.append(row_from_record(rec, "jetson", dev, path, fl, temp, when))
    return rows


def collect(run_dirs):
    rows = []
    for d in run_dirs:
        for node in ("pi5", "pi4", "t580"):
            p = os.path.join(d, node + ".log")
            if os.path.exists(p):
                rows += parse_pi_log(p, node)
        for root, dirs, files in os.walk(d):
            dirs.sort(key=lambda s: (s.upper(), s))   # NTFS's order, which the published files were built in, on any disk
            if "run.log" in files and "jetson" in root:
                rows += parse_jetson_run(os.path.join(root, "run.log"))
    return rows


COLS = ["node_id", "device", "model", "quant", "n_params", "model_bytes", "test", "phase", "n_threads",
        "n_gpu_layers", "reps", "mean_ts", "min_ts", "max_ts", "rep_range_pct", "ms_per_token",
        "eff_decode_GBps", "eff_prefill_GFLOPs", "start_temp_c", "flags", "date", "build_commit",
        "source", "source_sha256"]


def main(argv):
    out = "testbed/dataset"
    if "--out" in argv:
        i = argv.index("--out"); out = argv[i + 1]; del argv[i:i + 2]
    if not argv:
        sys.exit(__doc__)
    rows = collect(argv)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "characterisation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore"); w.writeheader()
        for r in sorted(rows, key=lambda r: (r["node_id"], r["device"], r["model"], r["test"])):
            w.writerow(r)
    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["node_id", "phase", "model", "n_threads", "ms_per_token", "flags"])
        for r in rows:
            if SERVING.get(r["node_id"]) == r["device"] and r["ms_per_token"] and r["test"] in PROFILE_TESTS:
                w.writerow([r["node_id"], r["phase"], r["model"], r["n_threads"], r["ms_per_token"], r["flags"]])
    print("wrote %d rows -> %s/characterisation.csv (+ results.csv)" % (len(rows), out))
    for r in sorted(rows, key=lambda r: (r["node_id"], r["device"], r["model"], r["test"])):
        print("  %-7s %-6s %-26s %-6s %8s t/s  [%s-%s] n=%s  %s" % (
            r["node_id"], r["device"], r["model"], r["test"], r["mean_ts"], r["min_ts"], r["max_ts"],
            r["reps"], r["flags"] or "ok"))


if __name__ == "__main__":
    main(sys.argv[1:])
