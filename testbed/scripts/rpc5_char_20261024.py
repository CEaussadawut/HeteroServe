#!/usr/bin/env python3
"""The RPC messages' size against the layers a worker holds, and in a two-worker ring (study rpc5, 2026-10-24), registered
before it runs. rpc4 (rpc4_char_20261024.py, rpc4plan) measured them at one split: the laptop sends 11,260 B a token beyond
the 8,192-byte hidden state to a Pi 4 holding 11 of TinyLlama's 22 blocks, and with them counted the predictor priced shaped
links within 5.8 %. The planner prices every split and ring, so the term needs its shape: the messages carry the worker's
graph, which should grow with its blocks.

    python3 testbed/scripts/rpc5_char_20261024.py register --memfile testbed/dataset/memory_snapshot_<stamp>.json [--conditions TEXT]
    python3 testbed/scripts/rpc5_char_20261024.py run           # -> testbed/results/rpc5char_<stamp>/ (session.log, measurements.json)
    python3 testbed/scripts/rpc5_char_20261024.py score <dir>   # -> <dir>/score.json, score.txt

Splits, each TinyLlama Q4_0 with the head on the host, served by serve_pool.sh, the links unshaped:
  H6 laptop 16 + Pi 4 6;  H11 laptop 11 + Pi 4 11 (rpc4's H, repeated);  H16 laptop 6 + Pi 4 16;
  R2 laptop 8 + Pi 5 7 + Pi 4 7 (two workers; probed by hand, see below);
  R3 laptop 4 + Pi 5 6 + Jetson 6 + Pi 4 6 (three workers, the Jetson on Vulkan).
Disclosed before registering: a hand probe of R2 at ~17:05 (the instrument check for two endpoints) read 19,134 B a token to
the Pi 5 (the first worker) and 10,946 B to the Pi 4, both returning 8,200. With rpc4's single worker (also a first worker,
19,452 B at 11 blocks) it gives the model registered here:
  to a worker = 8,192 (the hidden state at its split) + 8,192 more if it is the FIRST worker + m(blocks),
  m(k) = A + B k, A = 2,199 B and B = 79 B per block (from H11's 3,068 B and R2's 2,752 B at 7 blocks, the two workers' mean).
H11 and R2 are therefore repeats; H6, H16 and every worker of R3 are blind.
As rpc4: around each greedy /completion the laptop's TCP counters (ss -tin) are read for each worker's RPC connection (the
--rpc endpoints in llama-server's command line, each counted on its own); three repetitions of a 16- and a 144-token
completion; a token's bytes are the difference / 128.
Hypotheses (registered):
  R5-REPEAT  H11 reproduces rpc4's H (19,452 B to the Pi 4) and R2 the probe (19,134 to the Pi 5, 10,946 to the Pi 4),
             each within 0.5 %
  R5-MODEL   every blind figure (H6, H16, and R3's three workers) within 2 % of the model
  R5-SLOPE   the messages grow with the blocks: (H16 - H6) / 10 between 40 and 120 B per block (the model: 79)
  R5-BACK    in every split every worker returns 8,200 B a token within 0.5 % (the hidden state and an 8-byte header)
Writes testbed/dataset/rpc5char_predictions_20261024.json, read-only.
"""
import argparse, datetime, hashlib, json, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REG = os.path.join(ROOT, "testbed", "dataset", "rpc5char_predictions_20261024.json")
KEY, CTX = "tinyllama_q4_0", 2048
SPLITS = {"H6": "laptop:16,pi4:6", "H11": "laptop:11,pi4:11", "H16": "laptop:6,pi4:16", "R2": "laptop:8,pi5:7,pi4:7",
          "R3": "laptop:4,pi5:6,jetson:6,pi4:6"}
MODEL_A, MODEL_B = 2199, 79                                 # m(k) = A + B k, from H11 (rpc4) and the R2 probe; see the docstring
PROBE_R2 = {"pi5": 19134, "pi4": 10946}
WIP = {"192.168.10.11": "pi5", "192.168.10.12": "pi4", "192.168.1.114": "jetson"}       # serve_pool.sh's worker addresses
URL = "http://192.168.1.108:8080/completion"
PROMPT = "The quick brown fox jumps over the lazy dog. Summarise this sentence in five words."
SHORT, LONG, REPS = 16, 144, 3
HIDDEN = 8192
RPC4 = {"to_pi4": 19452, "back": 8200}
COUNTERS = ("bytes_acked", "bytes_received", "data_segs_out", "data_segs_in")
INPUTS = ("testbed/scripts/rpc5_char_20261024.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
          "ThinkPad/config/cluster.json")


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def register(a):
    if os.path.exists(REG):
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(REG, ROOT))
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/rpc5_char_20261024.py", "out_prefix": "rpc5char",
           "what": "the RPC messages' bytes per decoded token against the blocks a worker holds (TinyLlama Q4_0, the Pi 4 holding "
                   "6, 11 or 16 blocks) and in two- and three-worker rings, the head on the host, unshaped",
           "model": {"hidden_B": HIDDEN, "first_worker_extra_B": HIDDEN, "m_A_B": MODEL_A, "m_B_per_block": MODEL_B,
                     "from": "rpc4's H11 (19,452 B) and the R2 hand probe at ~17:05 (19,134 / 10,946 B)"},
           "probe_disclosed": {"split": "R2", "to_worker_B": PROBE_R2, "back_B": 8200},
           "model": KEY, "ctx": CTX, "splits": SPLITS, "prompt": PROMPT, "short": SHORT, "long": LONG, "reps": REPS,
           "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), "rpc4_h": RPC4,
           **({"conditions": a.conditions} if a.conditions else {}),
           "hypotheses": {
               "R5-REPEAT": {"tol": 0.005, "claim": "H11 reproduces rpc4's H and R2 the probe, each within 0.5 %"},
               "R5-MODEL": {"tol": 0.02, "claim": "every blind figure (H6, H16, R3's three workers) within 2 % of the model"},
               "R5-SLOPE": {"lo": 40, "hi": 120, "claim": "(H16 - H6) / 10 between 40 and 120 B per block (the model: 79)"},
               "R5-BACK": {"tol": 0.005, "claim": "every worker returns 8,200 B a token within 0.5 %, in every split"}},
           "inputs_sha256": {f: sha(f) for f in INPUTS + (os.path.relpath(os.path.abspath(a.memfile), ROOT),)}}
    json.dump(reg, open(REG, "w"), indent=1)
    os.chmod(REG, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(REG, ROOT), sha(os.path.relpath(REG, ROOT))[:16]))


def ss(ip, port):
    t = subprocess.run(["ss", "-tinH", "state", "established", "dst", ip, "( dport = :%d )" % port],
                       capture_output=True, text=True).stdout
    out = {k: sum(int(x) for x in re.findall(r"\b%s:(\d+)" % k, t)) for k in COUNTERS}
    out["sockets"] = len(re.findall(r"%s:%d" % (re.escape(ip), port), t))
    return out


def complete(n):
    body = json.dumps({"prompt": PROMPT, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": False,
                       "ignore_eos": True}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=900))
    t = r["timings"]
    if t["predicted_n"] != n:
        raise RuntimeError("decoded %d tokens, not %d" % (t["predicted_n"], n))
    return {"predicted_n": t["predicted_n"], "predicted_per_token_ms": t["predicted_per_token_ms"], "prompt_n": t["prompt_n"]}


def serve(cmd, env, log):
    e = dict(os.environ, MODEL=KEY, CTX=str(CTX), **env)
    p = subprocess.run(["bash", "testbed/scripts/serve_pool.sh", cmd], cwd=ROOT, env=e, capture_output=True, text=True)
    with open(log, "a") as f:
        f.write(p.stdout + p.stderr)
    return p.returncode, p.stdout + p.stderr


GUARD_START = 'sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo guard started'
GUARD_STOP = ('for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); '
              'do sudo -n kill $p && echo "guard pid $p stopped"; done; echo "TRIP lines: $(grep -c TRIP ~/heteroserve/logs/power-guard.log)"')


def jetson(cmd):
    """ab_session_20261024.sh's guard commands: serve_pool.sh refuses the Jetson without a live, un-tripped power guard."""
    p = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "jetson", cmd], capture_output=True, text=True,
                       timeout=60)
    return (p.stdout + p.stderr).strip()


def run(a):
    reg = json.load(open(REG))
    if os.access(REG, os.W_OK):
        raise SystemExit("refused: the registration is writable")
    if os.path.exists(os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")):
        raise SystemExit("refused: serve_pool.sh state exists")
    out = os.path.join(ROOT, "testbed", "results", "rpc5char_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out)
    log = os.path.join(out, "session.log")
    say = lambda s: (print(s, flush=True), open(log, "a").write(s + "\n"))
    say("########## %s rpc5char -> %s (registration %s)" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                                                           os.path.relpath(out, ROOT), sha(os.path.relpath(REG, ROOT))[:16]))
    res = {"registration": os.path.relpath(REG, ROOT), "splits": {}}
    if any("jetson" in pl for pl in SPLITS.values()):
        say("jetson guard: %s" % jetson(GUARD_START)); time.sleep(8)
    try:
        _run_splits(reg, res, log, say)
    finally:
        if any("jetson" in pl for pl in SPLITS.values()):
            say("jetson guard: %s" % jetson(GUARD_STOP))
    json.dump(res, open(os.path.join(out, "measurements.json"), "w"), indent=1)
    say("RUN DONE %s" % datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
    print(out)


def _run_splits(reg, res, log, say):
    for name, place in SPLITS.items():
        nodes = " ".join(x.split(":")[0] for x in place.split(","))
        env = {"MEMFILE": reg["memfile"], "CLOCKS": "locked", "NODES": nodes, "PLACE": place, "PLACE_HEAD": "host",
               "SERVE_DRAFT": "none"}
        rc, txt = serve("start", env, log)
        line = next((x for x in txt.splitlines() if x.startswith("llama-server ")), "")
        m = re.search(r"--rpc (\S+)", line)
        peers = [(x.rsplit(":", 1)[0], int(x.rsplit(":", 1)[1])) for x in m.group(1).split(",")] if m else []
        rec = {"place": place, "serve_rc": rc, "llama_server": line, "peers": {WIP.get(i, i): "%s:%d" % (i, p) for i, p in peers},
               "runs": []}
        res["splits"][name] = rec
        if rc != 0 or not peers:
            say("%s: serve_pool.sh start failed (rc %d) or no --rpc endpoints" % (name, rc)); serve("stop", {}, log); continue
        try:
            complete(SHORT)                                         # warm-up
            for rep in range(1, REPS + 1):
                for n in (SHORT, LONG):
                    b = {WIP.get(i, i): ss(i, p) for i, p in peers}
                    t = complete(n)
                    e = {WIP.get(i, i): ss(i, p) for i, p in peers}
                    d = {w: {k: e[w][k] - b[w][k] for k in COUNTERS} for w in b}
                    socks = {w: [b[w]["sockets"], e[w]["sockets"]] for w in b}
                    rec["runs"].append({"rep": rep, "n": n, "sockets": socks, "delta": d, "timings": t})
                    say("%s rep %d n %d: %s; decode %.2f ms/token" % (name, rep, n, d, t["predicted_per_token_ms"]))
        except Exception as ex:
            rec["error"] = repr(ex)[:300]; say("%s: %r" % (name, ex))
        rc2, _ = serve("stop", {}, log)
        rec["stop_rc"] = rc2


def per_token(rec):
    """{worker: {"complete", "per_token": {counter: mean}, "reps"}} from a split's runs."""
    reps = {}
    for r in rec.get("runs", []):
        reps.setdefault(r["rep"], {})[r["n"]] = r
    out = {}
    for w in rec.get("peers", {}):
        rows = []
        for rep, x in sorted(reps.items()):
            if SHORT in x and LONG in x and all(s == 1 for s in x[SHORT]["sockets"][w] + x[LONG]["sockets"][w]):
                rows.append({k: (x[LONG]["delta"][w][k] - x[SHORT]["delta"][w][k]) / (LONG - SHORT) for k in COUNTERS})
        mean = {k: round(sum(r[k] for r in rows) / len(rows), 2) for k in COUNTERS} if rows else {}
        out[w] = {"complete": len(rows) == REPS, "per_token": mean, "reps": rows}
    return out


def score(a):
    d = os.path.abspath(a.dir)
    m = json.load(open(os.path.join(d, "measurements.json")))
    reg = json.load(open(os.path.join(ROOT, m["registration"])))
    changed = [f for f, h in reg["inputs_sha256"].items() if os.path.exists(os.path.join(ROOT, f)) and sha(f) != h]
    per = {s: per_token(rec) for s, rec in m["splits"].items()}
    ok = lambda s, w: per.get(s, {}).get(w, {}).get("complete")
    blocks = {s: {x.split(":")[0]: int(x.split(":")[1]) for x in SPLITS[s].split(",")} for s in SPLITS}
    extra = lambda s, w: per[s][w]["per_token"]["bytes_acked"] - HIDDEN
    hy, v = reg["hypotheses"], {}
    order = lambda sp: [x.split(":")[0] for x in SPLITS[sp].split(",")][1:]          # the workers, in ring order
    model = lambda sp, w: HIDDEN + (HIDDEN if order(sp)[0] == w else 0) + MODEL_A + MODEL_B * blocks[sp][w]
    to = lambda sp, w: per[sp][w]["per_token"]["bytes_acked"]
    rep_ok = ok("H11", "pi4") and ok("R2", "pi5") and ok("R2", "pi4")
    if rep_ok:
        got = {"H11/pi4": to("H11", "pi4"), "R2/pi5": to("R2", "pi5"), "R2/pi4": to("R2", "pi4")}
        ref = {"H11/pi4": RPC4["to_pi4"], "R2/pi5": PROBE_R2["pi5"], "R2/pi4": PROBE_R2["pi4"]}
        v["R5-REPEAT"] = {"verdict": "HOLDS" if all(abs(got[k] / ref[k] - 1) <= hy["R5-REPEAT"]["tol"] for k in got) else "FAILS",
                          "measured": got, "reference": ref}
    else:
        v["R5-REPEAT"] = {"verdict": "INCOMPLETE"}
    blind = [("H6", "pi4"), ("H16", "pi4")] + [("R3", w) for w in order("R3")]
    if all(ok(sp, w) for sp, w in blind):
        rows = {"%s/%s" % (sp, w): {"measured": to(sp, w), "model": model(sp, w), "err_pct": round(100 * (to(sp, w) / model(sp, w) - 1), 2)}
                for sp, w in blind}
        v["R5-MODEL"] = {"verdict": "HOLDS" if all(abs(x["err_pct"]) <= 100 * hy["R5-MODEL"]["tol"] for x in rows.values()) else "FAILS",
                         "workers": rows}
    else:
        v["R5-MODEL"] = {"verdict": "INCOMPLETE"}
    if ok("H6", "pi4") and ok("H16", "pi4"):
        slope = (to("H16", "pi4") - to("H6", "pi4")) / (blocks["H16"]["pi4"] - blocks["H6"]["pi4"])
        v["R5-SLOPE"] = {"verdict": "HOLDS" if hy["R5-SLOPE"]["lo"] <= slope <= hy["R5-SLOPE"]["hi"] else "FAILS",
                         "per_block_B": round(slope, 2)}
    else:
        v["R5-SLOPE"] = {"verdict": "INCOMPLETE"}
    line = None
    backs = {"%s/%s" % (s, w): x["per_token"].get("bytes_received") for s, ws in per.items() for w, x in ws.items()}
    allc = all(x.get("complete") for ws in per.values() for x in ws.values()) and len(per) == len(SPLITS)
    v["R5-BACK"] = ({"verdict": "HOLDS" if all(b and abs(b / 8200 - 1) <= hy["R5-BACK"]["tol"] for b in backs.values()) else "FAILS",
                     "back_B": backs} if allc else {"verdict": "INCOMPLETE", "back_B": backs})
    sc = {"session": os.path.relpath(d, ROOT), "registration": m["registration"], "registration_sha256": sha(m["registration"]),
          "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "code_changed_since_registration": changed,
          "splits": per, "line": line, "hypotheses": v}
    json.dump(sc, open(os.path.join(d, "score.json"), "w"), indent=1)
    lines = ["rpc5char: %s; code changed since registration: %s" % (reg["what"], ", ".join(changed) or "none"), ""]
    lines += ["%-10s %-11s %s" % (k, x["verdict"], {a: b for a, b in x.items() if a != "verdict"}) for k, x in v.items()]
    lines.append("")
    for s, ws in per.items():
        lines.append("split %s (%s): %s" % (s, SPLITS[s], {w: x["per_token"] for w, x in ws.items()}))
    open(os.path.join(d, "score.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("register"); r.add_argument("--memfile", required=True); r.add_argument("--conditions", default="")
    sp.add_parser("run")
    s = sp.add_parser("score"); s.add_argument("dir")
    a = ap.parse_args()
    {"register": register, "run": run, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
