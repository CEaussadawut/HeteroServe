#!/usr/bin/env python3
"""Scope 4 on today's boards: two users, and where the prefill/decode hand-off stops paying as the link slows (study pd4,
2026-10-24; the user: "complete scope 4"). pd_twouser.py's design (2026-09-29, the original Pi 5), driven here so that the
hand-off's state transfer can be shaped per arm; pd_twouser.py and pd_handoff.py are imported, not changed.

    python3 testbed/scripts/pd4_cross_20261024.py run --reg testbed/dataset/<registration>.json
    python3 testbed/scripts/pd4_cross_20261024.py score <log> --reg <registration>

The registration names the arms and the order (repeated `reps` times, interleaved): "shared_nr" (the Pi 5 serves both users,
--no-repack), "shared" (the same, repacked), "handoff" (the Pi 5 prefills user 2 and sends its state straight into the Pi 4's
RAM, the Pi 4 decodes user 2, the Pi 5 serves user 1), and "handoff@R": the hand-off with the Pi 5 -> Pi 4 state transfer
shaped to R Mbit/s on the Pi 4 (netshape.sh, peer the Pi 5's overlay 192.168.10.11, port 50091: only the transfer, never the
users' HTTP streams or ssh), its netem counters read and the shaping cleared after every arm-run. Two users of n tokens each
want gen tokens back, greedy; every token is streamed and timestamped on the laptop (pd_twouser.stream). The laptop computes
nothing. The strip logs all four outlets at 5 Hz for the whole run (an observation: energy is not a registered figure here).
Every arm-run's line is "<stamp> r<k> <arm> result {...}", as pd_twouser.py writes it, plus the shaping's own lines.
"""
import argparse, datetime, hashlib, json, os, re, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import pd_handoff as H  # noqa: E402
import pd_twouser as T  # noqa: E402

PEER, PORT = H.OVERLAY["pi5"], H.XPORT          # the Pi 5's overlay address, the transfer's port (both from pd_handoff)
NS = ["env", "NETSHAPE_PEER=%s" % PEER, "NETSHAPE_PORT=%d" % PORT, "bash", os.path.join(HERE, "netshape.sh")]
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "LogLevel=ERROR"]


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def netem_counters():
    raw = subprocess.run(SSH + ["pi4", "sudo -n /usr/sbin/tc -s qdisc show dev eth0; echo '== ifb0'; "
                                        "sudo -n /usr/sbin/tc -s qdisc show dev ifb0 2>/dev/null"], capture_output=True, text=True).stdout
    eth, _, ifb = raw.partition("== ifb0")

    def netem(txt):
        m = re.search(r"qdisc netem .*?Sent (\d+) bytes (\d+) pkt", txt, re.S)
        return (int(m.group(1)), int(m.group(2))) if m else (None, None)
    (eb, ep), (ib, ip) = netem(eth), netem(ifb)
    return {"egress_bytes": eb, "egress_pkts": ep, "ingress_bytes": ib, "ingress_pkts": ip}


def shape(rate, log, tag):
    p = subprocess.run(NS + ["apply", str(rate), "0"], capture_output=True, text=True)
    ok = p.returncode == 0
    log(tag + " shape", {"rate_mbit": rate, "ok": ok, "out": (p.stdout + p.stderr).strip()[-300:], "applied_epoch": time.time()})
    if not ok:
        raise RuntimeError("netshape.sh apply %s failed" % rate)


def unshape(log, tag):
    c = netem_counters()
    p = subprocess.run(NS + ["clear"], capture_output=True, text=True)
    q = subprocess.run(NS + ["check"], capture_output=True, text=True)
    log(tag + " unshape", {"counters": c, "cleared": p.returncode == 0, "check_rc": q.returncode})
    return c


def run(a):
    reg = json.load(open(a.reg))
    if os.access(a.reg, os.W_OK):
        raise SystemExit("refused: the registration is writable")
    if subprocess.run(NS + ["check"], capture_output=True).returncode != 0:
        raise SystemExit("refused: shaping is installed on the Pi 4")
    log = H.Log("pd4cross")
    raw = open(H.GPL3, "rb").read()
    n, gen, reps, arms = reg["n"], reg["gen"], reg["reps"], reg["arms_order"]
    log("start", {"registration": os.path.relpath(os.path.abspath(a.reg), ROOT), "registration_sha256": sha(os.path.relpath(os.path.abspath(a.reg), ROOT)),
                  "arms": arms, "n": n, "gen": gen, "reps": reps, "script_sha256": sha("testbed/scripts/pd4_cross_20261024.py"),
                  "pd_twouser_sha256": sha("testbed/scripts/pd_twouser.py"), "pd_handoff_sha256": sha("testbed/scripts/pd_handoff.py")})
    strip_out = os.path.join(ROOT, "testbed", "results", os.path.basename(log.path)[:-4] + "_strip.csv")
    lp = subprocess.Popen([sys.executable, os.path.join(HERE, "shelly_strip.py"), "log", "--out", strip_out, "--hz", "5"],
                          stdout=subprocess.DEVNULL, stderr=open(strip_out + ".stderr", "w"))
    for node in ("pi5", "pi4"):
        H.ssh(node, "mkdir -p %s" % H.TMP_SLOTS)
    try:
        for node in ("pi5", "pi4"):
            T.start(node, log, T.NP[node])
        head = H.http("pi5", "/tokenize", {"content": "", "add_special": True})["tokens"]
        doc = H.http("pi5", "/tokenize", {"content": raw.decode("utf-8"), "add_special": False})["tokens"]
        q = H.http("pi5", "/tokenize", {"content": H.QUESTION, "add_special": False})["tokens"]
        body = n - len(head) - len(q)
        p1, p2 = head + doc[:body] + q, head + doc[2000:2000 + body] + q
        assert len(p1) == len(p2) == n
        log("prompts", {"n": n, "p1_sha256": hashlib.sha256(json.dumps(p1).encode()).hexdigest(),
                        "p2_sha256": hashlib.sha256(json.dumps(p2).encode()).hexdigest()})
        mode = ""
        for r in range(1, reps + 1):
            log("rep %d health" % r, {nd: H.health(nd) for nd in ("pi5", "pi4")})
            for arm in arms:
                tag = "r%d %s" % (r, arm)
                base, _, rate = arm.partition("@")
                want = T.PI5_MODE.get(base, "")
                if want != mode:
                    H.stop("pi5", log)
                    mode = want
                    T.start("pi5", log, T.NP["pi5"], mode)
                counters = None
                try:
                    if rate:
                        shape(int(rate), log, tag)
                    u1, u2, extra = T.ARMS[base](p1, p2, gen, 0, log, tag)
                except Exception as e:
                    log(tag + " FAILED", repr(e)); u1 = None
                finally:
                    if rate:
                        counters = unshape(log, tag)
                if u1 is None:
                    continue
                done = [u["done_s"] for u in (u1, u2) if u["done_s"] is not None]
                log(tag + " result", {"user1": u1, "user2": u2, "makespan_s": round(max(done), 3) if len(done) == 2 else None,
                                      **extra, **({"rate_mbit": int(rate), "netem": counters} if rate else {})})
        log("end", {"health": {nd: H.health(nd) for nd in ("pi5", "pi4")}})
    finally:
        subprocess.run(NS + ["clear"], capture_output=True)
        for node in ("pi5", "pi4"):
            H.stop(node, log)
            with open(os.path.join(ROOT, "testbed", "results", "%s_%s_server.log" % (os.path.basename(log.path)[:-4], node)), "w") as f:
                f.write(H.ssh(node, "cat ~/logs/llama-server_pd2_%s.log" % os.path.basename(log.path)[:-4], check=False))
            H.ssh(node, "rm -f %s/twouser.bin" % H.TMP_SLOTS, check=False)
        lp.send_signal(2)
        try:
            lp.wait(timeout=30)
        except subprocess.TimeoutExpired:
            lp.kill()
    print("log:", os.path.relpath(log.path, ROOT))


def results(path):
    res = {}
    for line in open(path):
        m = re.match(r"(\S+) r(\d+) (\S+) result (\{.*\})$", line.strip())
        if m:
            res.setdefault(m.group(3), []).append(dict(json.loads(m.group(4)), rep=int(m.group(2))))
    return res


def mean_done(r):
    return (r["user1"]["done_s"] + r["user2"]["done_s"]) / 2 if r["user1"]["done_s"] is not None and r["user2"]["done_s"] is not None else None


def score(a):
    reg = json.load(open(a.reg))
    res = results(a.log)
    per = {}
    for arm, rs in res.items():
        md = [mean_done(r) for r in rs if mean_done(r) is not None]
        per[arm] = {"runs": len(rs), "mean_done_s": round(statistics.fmean(md), 3) if md else None,
                    "spread_pct": round(100 * (max(md) - min(md)) / statistics.fmean(md), 2) if len(md) > 1 else None,
                    "user1_done_s": round(statistics.fmean(r["user1"]["done_s"] for r in rs), 3),
                    "user2_done_s": round(statistics.fmean(r["user2"]["done_s"] for r in rs), 3),
                    "user2_ttft_s": round(statistics.fmean(r["user2"]["ttft_s"] for r in rs), 3),
                    "makespan_s": round(statistics.fmean(r["makespan_s"] for r in rs if r["makespan_s"]), 3)}
        hx = [r for r in rs if r.get("transfer_s") is not None]
        if hx:
            per[arm].update({"transfer_s": round(statistics.fmean(r["transfer_s"] for r in hx), 3),
                             "state_bytes": sorted({r["state_bytes"] for r in hx}),
                             "prompt_processed_after_restore": sorted({r["u2_decode_prompt_processed"] for r in hx}),
                             "restored": sorted({r["restored"] for r in hx})})
        if any(r.get("netem") for r in rs):
            per[arm]["netem_min_pkts"] = min(min(r["netem"]["egress_pkts"] or 0, r["netem"]["ingress_pkts"] or 0) for r in rs if r.get("netem"))
    v = {}
    complete = all(per.get(arm, {}).get("runs", 0) >= reg["reps"] for arm in set(reg["arms_order"]))
    for k, h in reg["hypotheses"].items():
        t = h["type"]
        try:
            if t == "lower":
                x, y = per[h["a"]]["mean_done_s"], per[h["b"]]["mean_done_s"]
                v[k] = {"verdict": "HOLDS" if x < y else "FAILS", h["a"]: x, h["b"]: y}
            elif t == "higher":
                x, y = per[h["a"]]["mean_done_s"], per[h["b"]]["mean_done_s"]
                v[k] = {"verdict": "HOLDS" if x > y else "FAILS", h["a"]: x, h["b"]: y}
            elif t == "acc":
                errs = {arm: round(100 * (per[arm]["mean_done_s"] / p - 1), 2) for arm, p in h["predicted"].items()}
                v[k] = {"verdict": "HOLDS" if all(abs(e) <= 100 * h["tol"] for e in errs.values()) else "FAILS", "err_pct": errs}
            elif t == "xfer":
                errs = {arm: round(100 * (per[arm]["transfer_s"] / p - 1), 2) for arm, p in h["predicted_s"].items()}
                v[k] = {"verdict": "HOLDS" if all(abs(e) <= 100 * h["tol"] for e in errs.values()) else "FAILS", "err_pct": errs,
                        "measured_s": {arm: per[arm]["transfer_s"] for arm in h["predicted_s"]}}
            elif t == "state":
                got = sorted({b for arm in per if "state_bytes" in per[arm] for b in per[arm]["state_bytes"]})
                pp = sorted({x for arm in per if "prompt_processed_after_restore" in per[arm] for x in per[arm]["prompt_processed_after_restore"]})
                v[k] = {"verdict": "HOLDS" if got == [h["bytes"]] and pp == [1] else "FAILS", "state_bytes": got, "prompt_processed": pp}
            elif t == "shaped":
                m = {arm: per[arm].get("netem_min_pkts") for arm in per if "@" in arm}
                v[k] = {"verdict": "HOLDS" if m and all((x or 0) >= h["min_pkts"] for x in m.values()) else "FAILS", "min_pkts": m}
            elif t == "steady":
                s = {arm: per[arm]["spread_pct"] for arm in per}
                v[k] = {"verdict": "HOLDS" if all(x is not None and x <= 100 * h["tol"] for x in s.values()) else "FAILS", "spread_pct": s}
            if not complete and v[k]["verdict"] != "FAILS":
                v[k]["verdict"] = "INCOMPLETE"
        except (KeyError, TypeError) as e:
            v[k] = {"verdict": "INCOMPLETE", "why": repr(e)[:120]}
    reg_rel = os.path.relpath(os.path.abspath(a.reg), ROOT)
    changed = [f for f, hh in reg.get("inputs_sha256", {}).items() if os.path.exists(os.path.join(ROOT, f)) and sha(f) != hh]
    out = {"log": os.path.relpath(os.path.abspath(a.log), ROOT), "registration": reg_rel, "registration_sha256": sha(reg_rel),
           "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "code_changed_since_registration": changed,
           "complete": complete, "arms": per, "hypotheses": v}
    dst = a.out or os.path.join(ROOT, "testbed", "results", os.path.basename(a.log)[:-4] + "_score.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("%s: %s; code changed since registration: %s" % (reg.get("name", reg_rel), "complete" if complete else "INCOMPLETE", ", ".join(changed) or "none"))
    for k, x in v.items():
        print("%-12s %-11s %s" % (k, x["verdict"], {a2: b for a2, b in x.items() if a2 != "verdict"}))
    for arm, x in per.items():
        print("arm %-12s %s" % (arm, x))
    print("wrote", os.path.relpath(dst, ROOT))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--reg", required=True)
    s = sp.add_parser("score"); s.add_argument("log"); s.add_argument("--reg", required=True); s.add_argument("--out", default=None)
    a = ap.parse_args()
    {"run": run, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
