#!/usr/bin/env python3
"""The RPC message term on Qwen3 architectures (study rpc7, 2026-10-24; the user: "continue what's left for project 2"),
registered before it runs. predict_pool.RPC_MSG / rpc_bytes() were measured on TinyLlama only (rpc4-rpc6). This asks whether
the same shape holds for a dense Qwen3 (attention with q/k norms) and for a hybrid MoE (recurrent gated delta-net layers and
experts), whose per-token graphs differ from TinyLlama's.

    python3 testbed/scripts/rpc7_char_20261024.py register --memfile testbed/dataset/memory_snapshot_<stamp>.json [--conditions TEXT]
    python3 testbed/scripts/rpc7_char_20261024.py run           # -> testbed/results/rpc7char_<stamp>/
    python3 testbed/scripts/rpc7_char_20261024.py score <dir>

Arms, the head on the host, unshaped, each a plan that was served before:
  D32  Qwen3-32B Q6_K on laptop 45 + Pi 5 18 + Jetson 1 (so4's plan; the laptop's reserve 7,168 MiB); workers Pi 5, Jetson
  M35  Qwen3.6-35B-A3B Q6_K on laptop 25 + Jetson 3 + Pi 5 10 + Pi 4 3 (c35's split A); workers Jetson, Pi 5, Pi 4
As rpc5: the laptop's TCP counters (ss -tin) per worker connection, read around each greedy completion (ignore_eos); two
repetitions of a 16- and a 48-token completion; a token's bytes = the difference / 32. The Jetson's power guard runs for the
whole run (serve_pool.sh requires it). Predictions: predict_pool.rpc_bytes() at depth 0, per worker and its place.
Hypotheses (registered):
  R7-DENSE   each D32 worker receives its predicted bytes a token within 2 %
  R7-HYBRID  each M35 worker likewise
  R7-BACK    every worker returns its predicted bytes (the hidden state plus 8 B) within 0.5 %
  R7-REPEAT  each figure repeats within 1 % across the two repetitions
Writes testbed/dataset/rpc7char_predictions_20261024.json, read-only.
"""
import argparse, datetime, hashlib, json, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P  # noqa: E402

REG = os.path.join(ROOT, "testbed", "dataset", "rpc7char_predictions_20261024.json")
CTX = 2048
ARMS = {"D32": {"model": "qwen3_32b_q6k", "place": "laptop:45,pi5:18,jetson:1",
                "env": {"HETEROSERVE_RESERVE_MB": "laptop=7168"}},
        "M35": {"model": "qwen36moe_q6k", "place": "laptop:25,jetson:3,pi5:10,pi4:3", "env": {}}}
WIP = {"192.168.10.11": "pi5", "192.168.10.12": "pi4", "192.168.1.114": "jetson"}
URL = "http://192.168.1.108:8080/completion"
PROMPT = "The quick brown fox jumps over the lazy dog. Summarise this sentence in five words."
SHORT, LONG, REPS = 16, 48, 2
COUNTERS = ("bytes_acked", "bytes_received", "data_segs_out", "data_segs_in")
INPUTS = ("testbed/scripts/rpc7_char_20261024.py", "testbed/scripts/predict_pool.py", "testbed/scripts/serve_pool.sh",
          "testbed/scripts/serve_defaults.sh", "ThinkPad/config/cluster.json", "testbed/dataset/model_shapes.json")
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
    shapes = json.load(open(os.path.join(ROOT, "testbed", "dataset", "model_shapes.json")))
    out = {}
    for arm, a in ARMS.items():
        q = shapes[P.MODELS[a["model"]][0]]
        workers = [x.split(":")[0] for x in a["place"].split(",")][1:]
        out[arm] = {}
        for i, w in enumerate(workers, 1):
            inb, ob, _ = P.rpc_bytes(q, i, len(workers), q["activation_bytes"], 0, P.RPC_MSG, False)
            out[arm][w] = {"to": inb, "back": ob, "place": i, "of": len(workers)}
    return out


def register(a):
    if os.path.exists(REG):
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(REG, ROOT))
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/rpc7_char_20261024.py", "out_prefix": "rpc7char",
           "what": "the RPC message term on Qwen3 architectures: bytes per decoded token per worker for Qwen3-32B Q6_K (dense) and "
                   "Qwen3.6-35B-A3B Q6_K (hybrid MoE), the head on the host, unshaped, against predict_pool.rpc_bytes()",
           "ctx": CTX, "arms": ARMS, "prompt": PROMPT, "short": SHORT, "long": LONG, "reps": REPS,
           "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), "rpc_msg": P.RPC_MSG, "predicted": predictions(),
           **({"conditions": a.conditions} if a.conditions else {}),
           "hypotheses": {
               "R7-DENSE": {"arm": "D32", "tol": 0.02, "claim": "each D32 worker receives its predicted bytes a token within 2 %"},
               "R7-HYBRID": {"arm": "M35", "tol": 0.02, "claim": "each M35 worker receives its predicted bytes a token within 2 %"},
               "R7-BACK": {"tol": 0.005, "claim": "every worker returns its predicted bytes a token within 0.5 %"},
               "R7-REPEAT": {"tol": 0.01, "claim": "each figure repeats within 1 % across the two repetitions"}},
           "inputs_sha256": {f: sha(f) for f in INPUTS + (os.path.relpath(os.path.abspath(a.memfile), ROOT),)}}
    json.dump(reg, open(REG, "w"), indent=1)
    os.chmod(REG, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(REG, ROOT), sha(os.path.relpath(REG, ROOT))[:16]))
    print(json.dumps(reg["predicted"]))


def ss(ip, port):
    t = subprocess.run(["ss", "-tinH", "state", "established", "dst", ip, "( dport = :%d )" % port], capture_output=True, text=True).stdout
    out = {k: sum(int(x) for x in re.findall(r"\b%s:(\d+)" % k, t)) for k in COUNTERS}
    out["sockets"] = len(re.findall(r"%s:%d" % (re.escape(ip), port), t))
    return out


def complete(n):
    body = json.dumps({"prompt": PROMPT, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": False,
                       "ignore_eos": True}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=1800))
    t = r["timings"]
    if t["predicted_n"] != n:
        raise RuntimeError("decoded %d tokens, not %d" % (t["predicted_n"], n))
    return {"predicted_n": t["predicted_n"], "predicted_per_token_ms": t["predicted_per_token_ms"], "prompt_n": t["prompt_n"]}


def serve(cmd, model, env, log):
    e = dict(os.environ, MODEL=model, CTX=str(CTX), **env)
    p = subprocess.run(["bash", "testbed/scripts/serve_pool.sh", cmd], cwd=ROOT, env=e, capture_output=True, text=True)
    with open(log, "a") as f:
        f.write(p.stdout + p.stderr)
    return p.returncode, p.stdout + p.stderr


def jetson(cmd):
    p = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "LogLevel=ERROR", "jetson", cmd],
                       capture_output=True, text=True, timeout=60)
    return (p.stdout + p.stderr).strip()


def run(a):
    reg = json.load(open(REG))
    if os.access(REG, os.W_OK):
        raise SystemExit("refused: the registration is writable")
    if os.path.exists(os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")):
        raise SystemExit("refused: serve_pool.sh state exists")
    out = os.path.join(ROOT, "testbed", "results", "rpc7char_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out)
    log = os.path.join(out, "session.log")
    say = lambda s: (print(s, flush=True), open(log, "a").write(s + "\n"))
    say("########## %s rpc7char -> %s (registration %s)" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                                                           os.path.relpath(out, ROOT), sha(os.path.relpath(REG, ROOT))[:16]))
    res = {"registration": os.path.relpath(REG, ROOT), "arms": {}}
    say("jetson guard: %s" % jetson(GUARD_START)); time.sleep(8)
    try:
        for name, arm in ARMS.items():
            nodes = " ".join(x.split(":")[0] for x in arm["place"].split(","))
            env = dict({"MEMFILE": reg["memfile"], "CLOCKS": "locked", "NODES": nodes, "PLACE": arm["place"], "PLACE_HEAD": "host",
                        "SERVE_DRAFT": "none"}, **arm["env"])
            rc, txt = serve("start", arm["model"], env, log)
            line = next((x for x in txt.splitlines() if x.startswith("llama-server ")), "")
            m = re.search(r"--rpc (\S+)", line)
            peers = [(x.rsplit(":", 1)[0], int(x.rsplit(":", 1)[1])) for x in m.group(1).split(",")] if m else []
            rec = {"model": arm["model"], "place": arm["place"], "serve_rc": rc, "llama_server": line,
                   "peers": {WIP.get(i, i): "%s:%d" % (i, p) for i, p in peers}, "runs": []}
            res["arms"][name] = rec
            if rc != 0 or not peers:
                say("%s: serve_pool.sh start failed (rc %d) or no --rpc endpoints" % (name, rc)); serve("stop", arm["model"], {}, log); continue
            try:
                complete(SHORT)                                     # warm-up
                for rep in range(1, REPS + 1):
                    for n in (SHORT, LONG):
                        b = {WIP.get(i, i): ss(i, p) for i, p in peers}
                        t = complete(n)
                        e = {WIP.get(i, i): ss(i, p) for i, p in peers}
                        d = {w: {k: e[w][k] - b[w][k] for k in COUNTERS} for w in b}
                        rec["runs"].append({"rep": rep, "n": n, "sockets": {w: [b[w]["sockets"], e[w]["sockets"]] for w in b},
                                            "delta": d, "timings": t})
                        say("%s rep %d n %d: %s; decode %.1f ms/token" % (name, rep, n, d, t["predicted_per_token_ms"]))
            except Exception as ex:
                rec["error"] = repr(ex)[:300]; say("%s: %r" % (name, ex))
            rc2, _ = serve("stop", arm["model"], {}, log)
            rec["stop_rc"] = rc2
    finally:
        say("jetson guard: %s" % jetson(GUARD_STOP))
    json.dump(res, open(os.path.join(out, "measurements.json"), "w"), indent=1)
    say("RUN DONE %s" % datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
    print(out)


def score(a):
    d = os.path.abspath(a.dir)
    m = json.load(open(os.path.join(d, "measurements.json")))
    reg = json.load(open(os.path.join(ROOT, m["registration"])))
    changed = [f for f, h in reg["inputs_sha256"].items() if os.path.exists(os.path.join(ROOT, f)) and sha(f) != h]
    per, spreads = {}, {}
    for arm, rec in m["arms"].items():
        reps = {}
        for r in rec.get("runs", []):
            reps.setdefault(r["rep"], {})[r["n"]] = r
        per[arm] = {}
        for w in rec.get("peers", {}):
            rows = [{k: (x[LONG]["delta"][w][k] - x[SHORT]["delta"][w][k]) / (LONG - SHORT) for k in COUNTERS}
                    for _, x in sorted(reps.items()) if SHORT in x and LONG in x
                    and all(s == 1 for s in x[SHORT]["sockets"][w] + x[LONG]["sockets"][w])]
            mean = {k: round(sum(r[k] for r in rows) / len(rows), 2) for k in COUNTERS} if rows else {}
            per[arm][w] = {"complete": len(rows) == REPS, "per_token": mean, "reps": rows,
                           "predicted": reg["predicted"][arm][w]}
            for k in ("bytes_acked", "bytes_received"):
                v = [r[k] for r in rows]
                if v and min(v) > 0:
                    spreads["%s/%s/%s" % (arm, w, k)] = max(v) / min(v) - 1
    hy, v = reg["hypotheses"], {}
    for hk in ("R7-DENSE", "R7-HYBRID"):
        arm = hy[hk]["arm"]
        ws = per.get(arm, {})
        if ws and all(x["complete"] for x in ws.values()) and len(ws) == len(reg["predicted"][arm]):
            rows = {w: {"measured": x["per_token"]["bytes_acked"], "predicted": x["predicted"]["to"],
                        "err_pct": round(100 * (x["per_token"]["bytes_acked"] / x["predicted"]["to"] - 1), 2)} for w, x in ws.items()}
            v[hk] = {"verdict": "HOLDS" if all(abs(r["err_pct"]) <= 100 * hy[hk]["tol"] for r in rows.values()) else "FAILS", "workers": rows}
        else:
            v[hk] = {"verdict": "INCOMPLETE"}
    allw = [(arm, w, x) for arm, ws in per.items() for w, x in ws.items()]
    complete = allw and all(x["complete"] for _, _, x in allw) and all(len(per.get(arm, {})) == len(reg["predicted"][arm]) for arm in ARMS)
    if complete:
        backs = {"%s/%s" % (arm, w): {"measured": x["per_token"]["bytes_received"], "predicted": x["predicted"]["back"]} for arm, w, x in allw}
        v["R7-BACK"] = {"verdict": "HOLDS" if all(abs(b["measured"] / b["predicted"] - 1) <= hy["R7-BACK"]["tol"] for b in backs.values())
                        else "FAILS", "workers": backs}
        v["R7-REPEAT"] = {"verdict": "HOLDS" if all(s <= hy["R7-REPEAT"]["tol"] for s in spreads.values()) else "FAILS",
                          "max_spread_pct": round(100 * max(spreads.values()), 3) if spreads else None}
    else:
        v["R7-BACK"] = {"verdict": "INCOMPLETE"}; v["R7-REPEAT"] = {"verdict": "INCOMPLETE"}
    sc = {"session": os.path.relpath(d, ROOT), "registration": m["registration"], "registration_sha256": sha(m["registration"]),
          "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "code_changed_since_registration": changed,
          "arms": per, "hypotheses": v}
    json.dump(sc, open(os.path.join(d, "score.json"), "w"), indent=1)
    lines = ["rpc7char: %s; code changed since registration: %s" % (reg["what"], ", ".join(changed) or "none"), ""]
    lines += ["%-10s %-11s %s" % (k, x["verdict"], {a2: b for a2, b in x.items() if a2 != "verdict"}) for k, x in v.items()]
    lines += [""] + ["arm %s worker %s: %s (predicted %s)" % (arm, w, x["per_token"], x["predicted"]) for arm, ws in per.items() for w, x in ws.items()]
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
