#!/usr/bin/env python3
"""Score the TCP diagnostic (register_tcpdiag_20261009.py) against its registered hypothesis, as written.

    python3 testbed/scripts/score_tcpdiag_20261009.py testbed/results/tcpdiag_<stamp>

Reads <dir>/pool_<cell>_<stamp>.log (run_pool.sh, copied by the campaign: llama-bench's JSONL gives the decode ms/token
and the decode test's start, test_time), <dir>/ss_<cell>_<board>.log (ss_sample.sh) and <dir>/link_<setting>.log
(measure_link.sh). The decode window of a cell: from the decode test's test_time for (repetitions + 1 warm-up) x n_gen x
its ms/token, + 10 % + 2 s. Over that window:
  TD-SSAI  (tcp.tiny.last.b20) the decode ms/token is below the Jetson's median RTO, and no Jetson sample with data in flight
           (unacked > 0) shows cwnd <= 10
Exploratory, per cell: the Jetson's retransmissions in the window, cwnd and smoothed RTT in flight and idle, delivery rate;
the Pi 4's rcv_space, rcv_ssthresh and rcv_wnd; the decode time against the prediction re-priced at the measured link
(predict_pool.predict, rtt_trips 2 and 1). Writes <dir>/score.json.
"""
import calendar, glob, json, os, re, statistics, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
REG = os.path.join(ROOT, "testbed", "dataset", "pool_predictions_20261009tcpdiag.json")
NUM = {"rto": r"\brto:([\d.]+)", "cwnd": r"\bcwnd:(\d+)", "ssthresh": r"\bssthresh:(\d+)", "unacked": r"\bunacked:(\d+)",
       "lastsnd": r"\blastsnd:(\d+)", "rcv_space": r"\brcv_space:(\d+)", "rcv_ssthresh": r"\brcv_ssthresh:(\d+)",
       "rcv_wnd": r"\brcv_wnd:(\d+)", "snd_wnd": r"\bsnd_wnd:(\d+)", "bytes_acked": r"\bbytes_acked:(\d+)"}
UNIT = {"bps": 1e-6, "Kbps": 1e-3, "Mbps": 1.0, "Gbps": 1e3}


def parse_ss(lines):
    """[{"t", field: value, ...}] -- one dict per sample that had the connection; "srtt"/"rttvar" in ms, "retrans_total",
    "delivery_mbps"; unacked and retrans_total 0 when ss leaves them out (it prints them only when non-zero)."""
    out, t, sock = [], None, False
    for ln in lines:
        if ln.startswith("T "):
            t, sock = float(ln.split()[1]), False
            continue
        if t is None:
            continue
        if not ln[:1].isspace():
            sock = ln.split()[:1] != ["Recv-Q"] and ln.strip() != ""
            continue
        if not sock:
            continue
        d = {"t": t, "unacked": 0, "retrans_total": 0}
        for k, rx in NUM.items():
            m = re.search(rx, ln)
            if m:
                d[k] = float(m.group(1))
        m = re.search(r"\brtt:([\d.]+)/([\d.]+)", ln)
        if m:
            d["srtt"], d["rttvar"] = float(m.group(1)), float(m.group(2))
        m = re.search(r"\bretrans:(\d+)/(\d+)", ln)
        if m:
            d["retrans_total"] = int(m.group(2))
        m = re.search(r"\bdelivery_rate ([\d.]+)([KMG]?bps)", ln)
        if m:
            d["delivery_mbps"] = float(m.group(1)) * UNIT[m.group(2)]
        out.append(d)
        sock = False
    return out


def bench(lines):
    """(decode ms/token, decode test start epoch, repetitions, n_gen) from a run_pool log's llama-bench JSONL."""
    for ln in lines:
        if ln.startswith('{"build_commit"'):
            d = json.loads(ln)
            if d["n_gen"] > 0 and d["n_prompt"] == 0:
                ms = statistics.fmean(d["samples_ns"]) / 1e6 / d["n_gen"]
                t0 = calendar.timegm(time.strptime(d["test_time"], "%Y-%m-%dT%H:%M:%SZ"))
                return ms, float(t0), len(d["samples_ns"]), d["n_gen"]
    return None


def window(ms, t0, reps, n_gen):
    return t0, t0 + (reps + 1) * n_gen * ms / 1000.0 * 1.10 + 2.0


def med(xs):
    return round(statistics.median(xs), 3) if xs else None


def summarize(jet, pi, ms, w):
    j = [s for s in jet if w[0] <= s["t"] <= w[1]]
    p = [s for s in pi if w[0] <= s["t"] <= w[1]]
    fl = [s for s in j if s["unacked"] > 0]
    idle = [s for s in j if s["unacked"] == 0]
    rt = [s["retrans_total"] for s in j]
    return {"jetson_samples": len(j), "pi4_samples": len(p), "decode_ms": round(ms, 2),
            "jetson_rto_ms_median": med([s["rto"] for s in j if "rto" in s]),
            "jetson_cwnd_in_flight": {"min": min((s["cwnd"] for s in fl if "cwnd" in s), default=None),
                                      "median": med([s["cwnd"] for s in fl if "cwnd" in s]),
                                      "max": max((s["cwnd"] for s in fl if "cwnd" in s), default=None), "n": len(fl)},
            "jetson_cwnd_idle_median": med([s["cwnd"] for s in idle if "cwnd" in s]),
            "jetson_srtt_ms": {"in_flight": med([s["srtt"] for s in fl if "srtt" in s]),
                               "idle": med([s["srtt"] for s in idle if "srtt" in s])},
            "jetson_retransmissions": (max(rt) - min(rt)) if rt else None,
            "jetson_delivery_mbps_median": med([s["delivery_mbps"] for s in fl if "delivery_mbps" in s]),
            "pi4_rcv_space_median": med([s["rcv_space"] for s in p if "rcv_space" in s]),
            "pi4_rcv_ssthresh": {"min": min((s["rcv_ssthresh"] for s in p if "rcv_ssthresh" in s), default=None),
                                 "median": med([s["rcv_ssthresh"] for s in p if "rcv_ssthresh" in s])},
            "pi4_rcv_wnd_median": med([s["rcv_wnd"] for s in p if "rcv_wnd" in s])}


def td_ssai(cell):
    """The registered hypothesis on one cell's summary."""
    if not cell or not cell["jetson_samples"] or cell["jetson_rto_ms_median"] is None or not cell["jetson_cwnd_in_flight"]["n"]:
        return {"verdict": "INCOMPLETE"}
    a = cell["decode_ms"] < cell["jetson_rto_ms_median"]
    b = cell["jetson_cwnd_in_flight"]["min"] > 10
    return {"verdict": "HOLDS" if (a and b) else "FAILS", "token_below_rto": a, "no_restart_window_in_flight": b}


def main(argv):
    if not argv:
        sys.exit(__doc__)
    d = argv[0]
    import predict_pool as pp
    import link_to_json as lj
    reg = json.load(open(REG))
    shapes = json.load(open(os.path.join(ROOT, "testbed", "dataset", "model_shapes.json")))
    R = pp.rows(os.path.join(ROOT, "testbed", "dataset", "characterisation.csv"), "char")
    cells = {}
    for cid in reg["run_order"]:
        c = reg["configs"][cid]
        logs = sorted(glob.glob(os.path.join(d, "pool_%s_2*.log" % cid)))
        b = bench(open(logs[-1], encoding="utf-8", errors="replace").read().splitlines()) if logs else None
        if not b:
            cells[cid] = None
            continue
        ms, t0, reps, ng = b
        w = window(ms, t0, reps, ng)
        rd = lambda n: parse_ss(open(os.path.join(d, "ss_%s_%s.log" % (cid, n)), encoding="utf-8", errors="replace").read()
                                .splitlines()) if os.path.exists(os.path.join(d, "ss_%s_%s.log" % (cid, n))) else []
        s = summarize(rd("jetson"), rd("pi4"), ms, w)
        lp = os.path.join(d, "link_%s.log" % c["setting"])
        if os.path.exists(lp):
            _, _, r = lj.parse(lp)
            q = shapes[c["model"]]
            C, _ = pp.costs(R, shapes, c["quant"])
            assign = [(x["node"], x["blocks"]) for x in c["pipeline"]]
            L = {"pi4": {"mbps": 1e6, "rtt_ms": 0.0}, "jetson": {"mbps": r["mbps"], "rtt_ms": r["rtt_ms"]}}
            s["link_measured"] = {"mbps": r["mbps"], "rtt_ms": r["rtt_ms"]}
            for t in (2, 1):
                p = pp.predict(assign, q, C, L, head_at=c["head_at"], rtt_trips=t)[0]
                s["repriced_rtt%d_ms" % t] = round(p, 1)
                s["err_rtt%d_pct" % t] = round(100 * (p / ms - 1), 1)
        s["window"] = [round(w[0], 1), round(w[1], 1)]
        cells[cid] = s
    hyp = {"TD-SSAI": td_ssai(cells.get("tcp.tiny.last.b20"))}
    out = {"run": os.path.relpath(d, ROOT), "cells": cells, "hypotheses": hyp}
    json.dump(out, open(os.path.join(d, "score.json"), "w"), indent=1)
    for cid, s in cells.items():
        print("%-18s %s" % (cid, json.dumps(s)[:900] if s else "no result"))
    print("TD-SSAI %s" % hyp["TD-SSAI"])
    print("wrote %s" % os.path.relpath(os.path.join(d, "score.json"), ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
