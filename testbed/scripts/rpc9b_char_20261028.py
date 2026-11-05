#!/usr/bin/env python3
"""rpc9b (2026-10-28; the user: "let's do 4 and 1"): rpc9 again, with a DEEP warm-up, registered before it runs.

    python3 testbed/scripts/rpc9b_char_20261028.py register --memfile testbed/dataset/memory_snapshot_<stamp>.json [--conditions TEXT]
    python3 testbed/scripts/rpc9b_char_20261028.py run          # -> testbed/results/rpc9bchar_<stamp>/
    python3 testbed/scripts/rpc9b_char_20261028.py score <dir>

rpc9 (testbed/results/rpc9char_20261025_031402) held R9-MID, R9-ALL and R9-BACK, and failed R9-REPEAT at 1.24 %: the first
long-prompt (D3) repetition read low on every worker (the Jetson 12,061.8 B a token against 12,164 in repetitions 2 and 3),
because one-off messages of the first completion at that depth landed in its 16-token completion; its only warm-up used the
short prompt. rpc9b changes one thing: a 16-token completion of the LONG prompt follows the short warm-up, before any
measurement. Everything else -- the plan, the arms, the predictions, the four hypotheses and their tolerances -- is rpc9's.

rpc9's description follows.

rpc6 measured the attention mask's growth with the cache's depth (2 B per padded cell: +1,536 B a token at ~850 cells) on a
ONE-worker ring, and predict_pool.RPC_MSG charges it to every worker; whether a middle worker gets it too was assumed, not
measured (benchmarks.md, rpc6: "Whether a middle worker's mask grows with the cache too is assumed, not measured"). Every
worker that holds an attention layer reads the mask, so the term says it does.
TinyLlama Q4_0 on rpc5's three-worker ring, laptop 4 + Pi 5 6 + Jetson 6 + Pi 4 6 (workers: the Pi 5 first, the Jetson in the
middle, the Pi 4 last), the head on the host, ctx 2,048, unshaped, served once for both arms:
  S3  rpc6's short prompt (22 tokens): the cache holds ~38-166 cells while the tokens are measured (256 padded)
  D3  rpc6's long prompt (778 tokens): ~794-922 cells (1,024 padded)
As rpc6: the laptop's TCP counters per worker connection around each greedy completion (ignore_eos); three repetitions of a
16- and a 144-token completion; a token's bytes = the difference / 128. Predictions: predict_pool.rpc_bytes() at depth 100
(S3) and 850 (D3), per worker and its place. The Jetson's power guard runs for the whole run; the Pis' throttle flags are
read before and after.
Hypotheses (registered):
  R9-MID     the middle worker (the Jetson) receives its predicted bytes a token within 0.5 % in both arms (D3: 12,164 B;
             without the mask's growth it would be 10,628, -12.6 %)
  R9-ALL     all six figures (three workers x two depths) within 0.5 %
  R9-BACK    every worker returns the hidden state plus 8 B within 0.5 %
  R9-REPEAT  each figure repeats within 1 % across the three repetitions
rpc9b writes testbed/dataset/rpc9bchar_predictions_20261028.json, read-only.
"""
import argparse, datetime, hashlib, json, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P  # noqa: E402

REG = os.path.join(ROOT, "testbed", "dataset", "rpc9bchar_predictions_20261028.json")
KEY, PLACE, CTX = "tinyllama_q4_0", "laptop:4,pi5:6,jetson:6,pi4:6", 2048
SHORT_PROMPT = "The quick brown fox jumps over the lazy dog. Summarise this sentence in five words."
LONG_PROMPT = " ".join(["The quick brown fox jumps over the lazy dog."] * 64) + " Summarise this text in five words."
ARMS = {"S3": {"prompt": SHORT_PROMPT, "depth": 100}, "D3": {"prompt": LONG_PROMPT, "depth": 850}}
SHORT, LONG, REPS = 16, 144, 3
WIP = {"192.168.10.11": "pi5", "192.168.10.12": "pi4", "192.168.1.114": "jetson"}
URL = "http://192.168.1.108:8080/completion"
COUNTERS = ("bytes_acked", "bytes_received", "data_segs_out", "data_segs_in")
INPUTS = ("testbed/scripts/rpc9b_char_20261028.py", "testbed/scripts/rpc9_char_20261025.py",
          "testbed/results/rpc9char_20261025_031402/score.json", "testbed/scripts/predict_pool.py", "testbed/scripts/serve_pool.sh",
          "testbed/scripts/serve_defaults.sh", "ThinkPad/config/cluster.json", "testbed/dataset/model_shapes.json",
          "testbed/results/rpc6char_20261024_171408/score.json")
GUARD_START = 'sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo guard started'
GUARD_STOP = ('for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); '
              'do sudo -n kill $p && echo "guard pid $p stopped"; done; echo "TRIP lines: $(grep -c TRIP ~/heteroserve/logs/power-guard.log)"')


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def predictions():
    q = json.load(open(os.path.join(ROOT, "testbed", "dataset", "model_shapes.json")))[P.MODELS[KEY][0]]
    workers = [x.split(":")[0] for x in PLACE.split(",")][1:]
    out = {}
    for arm, a in ARMS.items():
        out[arm] = {}
        for i, w in enumerate(workers, 1):
            inb, ob, _ = P.rpc_bytes(q, i, len(workers), q["activation_bytes"], a["depth"], P.RPC_MSG, False)
            out[arm][w] = {"to": inb, "back": ob, "place": i, "of": len(workers), "depth": a["depth"]}
    return out


def register(a):
    if os.path.exists(REG):
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(REG, ROOT))
    pred = predictions()
    if pred["D3"]["jetson"]["to"] - pred["S3"]["jetson"]["to"] != 1536:
        raise SystemExit("refused: the term does not charge the middle worker the mask's growth: %s" % pred)
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/rpc9b_char_20261028.py", "out_prefix": "rpc9bchar",
           "follows": "rpc9 (rpc9char_predictions_20261025.json): R9-REPEAT failed at 1.24 %, the first deep repetition low; rpc9b adds a 16-token warm-up completion of the long prompt before any measurement, nothing else changed",
           "what": "the RPC term's attention mask on a middle worker: TinyLlama Q4_0 on laptop 4 + Pi 5 6 + Jetson 6 + Pi 4 6, a "
                   "short and a long prompt, bytes per decoded token per worker against predict_pool.rpc_bytes()",
           "model": KEY, "place": PLACE, "ctx": CTX, "arms": ARMS, "short": SHORT, "long": LONG, "reps": REPS,
           "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), "rpc_msg": P.RPC_MSG, "predicted": pred,
           **({"conditions": a.conditions} if a.conditions else {}),
           "hypotheses": {
               "R9-MID": {"worker": "jetson", "tol": 0.005, "claim": "the middle worker receives its predicted bytes within 0.5 % "
                                                                    "in both arms (D3 12,164 B; without the growth 10,628)"},
               "R9-ALL": {"tol": 0.005, "claim": "all six figures within 0.5 %"},
               "R9-BACK": {"tol": 0.005, "claim": "every worker returns the hidden state plus 8 B within 0.5 %"},
               "R9-REPEAT": {"tol": 0.01, "claim": "each figure repeats within 1 % across the three repetitions"}},
           "inputs_sha256": {f: sha(f) for f in INPUTS + (os.path.relpath(os.path.abspath(a.memfile), ROOT),)}}
    json.dump(reg, open(REG, "w"), indent=1)
    os.chmod(REG, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(REG, ROOT), sha(os.path.relpath(REG, ROOT))[:16]))
    print(json.dumps({arm: {w: x["to"] for w, x in ws.items()} for arm, ws in pred.items()}))


def ss(ip, port):
    t = subprocess.run(["ss", "-tinH", "state", "established", "dst", ip, "( dport = :%d )" % port], capture_output=True, text=True).stdout
    out = {k: sum(int(x) for x in re.findall(r"\b%s:(\d+)" % k, t)) for k in COUNTERS}
    out["sockets"] = len(re.findall(r"%s:%d" % (re.escape(ip), port), t))
    return out


def complete(prompt, n):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": False,
                       "ignore_eos": True}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=1800))
    t = r["timings"]
    if t["predicted_n"] != n:
        raise RuntimeError("decoded %d tokens, not %d" % (t["predicted_n"], n))
    return {"predicted_n": t["predicted_n"], "prompt_n": t["prompt_n"], "predicted_per_token_ms": t["predicted_per_token_ms"]}


def serve(cmd, env, log):
    e = dict(os.environ, MODEL=KEY, CTX=str(CTX), **env)
    p = subprocess.run(["bash", "testbed/scripts/serve_pool.sh", cmd], cwd=ROOT, env=e, capture_output=True, text=True)
    with open(log, "a") as f:
        f.write(p.stdout + p.stderr)
    return p.returncode, p.stdout + p.stderr


def on(node, cmd, timeout=60):
    p = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "LogLevel=ERROR", node, cmd],
                       capture_output=True, text=True, timeout=timeout)
    return (p.stdout + p.stderr).strip()


def run(a):
    reg = json.load(open(REG))
    if os.access(REG, os.W_OK):
        raise SystemExit("refused: the registration is writable")
    if os.path.exists(os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")):
        raise SystemExit("refused: serve_pool.sh state exists")
    out = os.path.join(ROOT, "testbed", "results", "rpc9bchar_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out)
    log = os.path.join(out, "session.log")
    say = lambda s: (print(s, flush=True), open(log, "a").write(s + "\n"))
    say("########## %s rpc9bchar -> %s (registration %s)" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                                                           os.path.relpath(out, ROOT), sha(os.path.relpath(REG, ROOT))[:16]))
    res = {"registration": os.path.relpath(REG, ROOT), "arms": {},
           "throttled_before": {n: on(n, "vcgencmd get_throttled") for n in ("pi5", "pi4")}}
    say("throttled before: %s" % res["throttled_before"])
    say("jetson guard: %s" % on("jetson", GUARD_START)); time.sleep(8)
    try:
        env = {"MEMFILE": reg["memfile"], "CLOCKS": "locked", "NODES": "laptop pi5 jetson pi4", "PLACE": PLACE,
               "PLACE_HEAD": "host", "SERVE_DRAFT": "none"}
        rc, txt = serve("start", env, log)
        line = next((x for x in txt.splitlines() if x.startswith("llama-server ")), "")
        m = re.search(r"--rpc (\S+)", line)
        peers = [(x.rsplit(":", 1)[0], int(x.rsplit(":", 1)[1])) for x in m.group(1).split(",")] if m else []
        res.update({"serve_rc": rc, "llama_server": line, "peers": {WIP.get(i, i): "%s:%d" % (i, p) for i, p in peers}})
        if rc != 0 or not peers:
            say("serve_pool.sh start failed (rc %d) or no --rpc endpoints" % rc)
        else:
            try:
                complete(SHORT_PROMPT, SHORT)                               # warm-up
                complete(LONG_PROMPT, SHORT)                                # rpc9b: the deep warm-up
                say("warm-ups done: the short prompt, then the long prompt (16 tokens each)")
                for arm, spec in ARMS.items():
                    rec = {"runs": []}
                    res["arms"][arm] = rec
                    for rep in range(1, REPS + 1):
                        for n in (SHORT, LONG):
                            b = {WIP.get(i, i): ss(i, p) for i, p in peers}
                            t = complete(spec["prompt"], n)
                            e = {WIP.get(i, i): ss(i, p) for i, p in peers}
                            d = {w: {k: e[w][k] - b[w][k] for k in COUNTERS} for w in b}
                            rec["runs"].append({"rep": rep, "n": n, "sockets": {w: [b[w]["sockets"], e[w]["sockets"]] for w in b},
                                                "delta": d, "timings": t})
                            say("%s rep %d n %d (prompt %d): %s" % (arm, rep, n, t["prompt_n"],
                                                                    {w: x["bytes_acked"] for w, x in d.items()}))
            except Exception as ex:
                res["error"] = repr(ex)[:300]; say("error: %r" % ex)
        res["stop_rc"] = serve("stop", {}, log)[0]
    finally:
        say("jetson guard: %s" % on("jetson", GUARD_STOP))
        res["throttled_after"] = {n: on(n, "vcgencmd get_throttled") for n in ("pi5", "pi4")}
        say("throttled after: %s" % res["throttled_after"])
        json.dump(res, open(os.path.join(out, "measurements.json"), "w"), indent=1)
    say("RUN DONE")
    print(out)


def score(a):
    d = os.path.abspath(a.dir)
    m = json.load(open(os.path.join(d, "measurements.json")))
    reg = json.load(open(os.path.join(ROOT, m["registration"])))
    changed = [f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh")) and sha(f) != h]
    per, spreads = {}, {}
    for arm, rec in m.get("arms", {}).items():
        reps = {}
        for r in rec.get("runs", []):
            reps.setdefault(r["rep"], {})[r["n"]] = r
        per[arm] = {}
        for w in m.get("peers", {}):
            rows = [{k: (x[LONG]["delta"][w][k] - x[SHORT]["delta"][w][k]) / (LONG - SHORT) for k in COUNTERS}
                    for _, x in sorted(reps.items()) if SHORT in x and LONG in x
                    and all(s == 1 for s in x[SHORT]["sockets"][w] + x[LONG]["sockets"][w])]
            mean = {k: round(sum(r[k] for r in rows) / len(rows), 2) for k in COUNTERS} if rows else {}
            per[arm][w] = {"complete": len(rows) == REPS, "per_token": mean, "reps": rows, "predicted": reg["predicted"][arm][w]}
            for k in ("bytes_acked", "bytes_received"):
                v = [r[k] for r in rows]
                if v and min(v) > 0:
                    spreads["%s/%s/%s" % (arm, w, k)] = max(v) / min(v) - 1
    hy, v = reg["hypotheses"], {}
    cells = [(arm, w, x) for arm, ws in per.items() for w, x in ws.items()]
    complete_all = len(cells) == 6 and all(x["complete"] for _, _, x in cells)
    err = {"%s/%s" % (arm, w): {"measured": x["per_token"].get("bytes_acked"), "predicted": x["predicted"]["to"],
                                "err_pct": round(100 * (x["per_token"]["bytes_acked"] / x["predicted"]["to"] - 1), 2)}
           for arm, w, x in cells if x["per_token"]}
    if complete_all:
        mid = {k: e for k, e in err.items() if k.endswith("/jetson")}
        v["R9-MID"] = {"verdict": "HOLDS" if all(abs(e["err_pct"]) <= 100 * hy["R9-MID"]["tol"] for e in mid.values()) else "FAILS",
                       "workers": mid}
        v["R9-ALL"] = {"verdict": "HOLDS" if all(abs(e["err_pct"]) <= 100 * hy["R9-ALL"]["tol"] for e in err.values()) else "FAILS",
                       "workers": err}
        backs = {"%s/%s" % (arm, w): {"measured": x["per_token"]["bytes_received"], "predicted": x["predicted"]["back"]}
                 for arm, w, x in cells}
        v["R9-BACK"] = {"verdict": "HOLDS" if all(abs(b["measured"] / b["predicted"] - 1) <= hy["R9-BACK"]["tol"] for b in backs.values())
                        else "FAILS", "workers": backs}
        v["R9-REPEAT"] = {"verdict": "HOLDS" if all(s <= hy["R9-REPEAT"]["tol"] for s in spreads.values()) else "FAILS",
                          "max_spread_pct": round(100 * max(spreads.values()), 3) if spreads else None}
    else:
        v = {h: {"verdict": "INCOMPLETE", "workers": err} for h in hy}
    sc = {"session": os.path.relpath(d, ROOT), "registration": m["registration"], "registration_sha256": sha(m["registration"]),
          "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "code_changed_since_registration": changed,
          "throttled": {"before": m.get("throttled_before"), "after": m.get("throttled_after")}, "arms": per, "hypotheses": v}
    json.dump(sc, open(os.path.join(d, "score.json"), "w"), indent=1)
    lines = ["rpc9bchar: %s; code changed since registration: %s" % (reg["what"], ", ".join(changed) or "none"), ""]
    lines += ["%-10s %-11s %s" % (k, x["verdict"], {a2: b for a2, b in x.items() if a2 != "verdict"}) for k, x in v.items()]
    lines += ["", "throttled: %s" % sc["throttled"]]
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
