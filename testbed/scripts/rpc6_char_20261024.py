#!/usr/bin/env python3
"""Does the first RPC worker's extra ~8.5 KB a token depend on the context? (study rpc6, 2026-10-24), registered before it
runs. rpc5 found the RPC's messages independent of the blocks a worker holds, and the FIRST worker receiving 8,506 B a token
more than a later one: 8,192 + 314, which is TinyLlama's hidden state (n_embd 2,048 x 4 B) but equally n_ctx x 4 B at the
2,048-token context every rpc4/rpc5 run used. If it is a context-sized input (an attention mask over the cache), the RPC
term grows with the context or the cache depth; if it is a hidden-state copy, it does not.

    python3 testbed/scripts/rpc6_char_20261024.py register --memfile testbed/dataset/memory_snapshot_<stamp>.json [--conditions TEXT]
    python3 testbed/scripts/rpc6_char_20261024.py run           # -> testbed/results/rpc6char_<stamp>/
    python3 testbed/scripts/rpc6_char_20261024.py score <dir>

TinyLlama Q4_0 on laptop 11 + Pi 4 11, the head on the host (rpc4's H), unshaped, three ways:
  C2  ctx 2,048, rpc5's short prompt (the repeat: 19,452 B a token to the Pi 4);
  C4  ctx 4,096, the same prompt;
  D2  ctx 2,048, a ~700-token prompt (the same text repeated), so the cache holds ~700-850 cells during the measured tokens.
As rpc5: the laptop's TCP counters, three repetitions of a 16- and a 144-token completion, a token's bytes = difference / 128.
Hypotheses (registered, the null first):
  R6-REPEAT  C2 reproduces 19,452 B to the Pi 4 and 8,200 B back, within 0.5 %
  R6-CTX     the bytes to the Pi 4 do not depend on the context size: C4 within 0.5 % of C2
  R6-DEPTH   nor on the cache's depth: D2 within 0.5 % of C2
Writes testbed/dataset/rpc6char_predictions_20261024.json, read-only.
"""
import argparse, datetime, hashlib, json, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REG = os.path.join(ROOT, "testbed", "dataset", "rpc6char_predictions_20261024.json")
KEY, PLACE = "tinyllama_q4_0", "laptop:11,pi4:11"
PEER, PORT = "192.168.10.12", 50052
URL = "http://192.168.1.108:8080/completion"
SHORT_PROMPT = "The quick brown fox jumps over the lazy dog. Summarise this sentence in five words."
LONG_PROMPT = " ".join(["The quick brown fox jumps over the lazy dog."] * 64) + " Summarise this text in five words."
ARMS = {"C2": {"ctx": 2048, "prompt": SHORT_PROMPT}, "C4": {"ctx": 4096, "prompt": SHORT_PROMPT},
        "D2": {"ctx": 2048, "prompt": LONG_PROMPT}}
SHORT, LONG, REPS = 16, 144, 3
REF = {"to": 19452, "back": 8200}
COUNTERS = ("bytes_acked", "bytes_received", "data_segs_out", "data_segs_in")
INPUTS = ("testbed/scripts/rpc6_char_20261024.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
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
           "script": "testbed/scripts/rpc6_char_20261024.py", "out_prefix": "rpc6char",
           "what": "the first RPC worker's bytes per decoded token against the context size and the cache's depth (TinyLlama "
                   "Q4_0, laptop 11 + Pi 4 11, the head on the host, unshaped)",
           "model": KEY, "place": PLACE, "arms": ARMS, "short": SHORT, "long": LONG, "reps": REPS, "reference": REF,
           "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), **({"conditions": a.conditions} if a.conditions else {}),
           "hypotheses": {
               "R6-REPEAT": {"tol": 0.005, "claim": "C2 reproduces 19,452 B to the Pi 4 and 8,200 B back, within 0.5 %"},
               "R6-CTX": {"tol": 0.005, "claim": "the bytes to the Pi 4 do not depend on the context size: C4 within 0.5 % of C2"},
               "R6-DEPTH": {"tol": 0.005, "claim": "nor on the cache's depth: D2 within 0.5 % of C2"}},
           "inputs_sha256": {f: sha(f) for f in INPUTS + (os.path.relpath(os.path.abspath(a.memfile), ROOT),)}}
    json.dump(reg, open(REG, "w"), indent=1)
    os.chmod(REG, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(REG, ROOT), sha(os.path.relpath(REG, ROOT))[:16]))


def ss():
    t = subprocess.run(["ss", "-tinH", "state", "established", "dst", PEER, "( dport = :%d )" % PORT], capture_output=True, text=True).stdout
    out = {k: sum(int(x) for x in re.findall(r"\b%s:(\d+)" % k, t)) for k in COUNTERS}
    out["sockets"] = len(re.findall(r"%s:%d" % (re.escape(PEER), PORT), t))
    return out


def complete(prompt, n):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": False,
                       "ignore_eos": True}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=900))
    t = r["timings"]
    if t["predicted_n"] != n:
        raise RuntimeError("decoded %d tokens, not %d" % (t["predicted_n"], n))
    return {"predicted_n": t["predicted_n"], "predicted_per_token_ms": t["predicted_per_token_ms"], "prompt_n": t["prompt_n"]}


def serve(cmd, ctx, env, log):
    e = dict(os.environ, MODEL=KEY, CTX=str(ctx), **env)
    p = subprocess.run(["bash", "testbed/scripts/serve_pool.sh", cmd], cwd=ROOT, env=e, capture_output=True, text=True)
    with open(log, "a") as f:
        f.write(p.stdout + p.stderr)
    return p.returncode, p.stdout + p.stderr


def run(a):
    reg = json.load(open(REG))
    if os.access(REG, os.W_OK):
        raise SystemExit("refused: the registration is writable")
    if os.path.exists(os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")):
        raise SystemExit("refused: serve_pool.sh state exists")
    out = os.path.join(ROOT, "testbed", "results", "rpc6char_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out)
    log = os.path.join(out, "session.log")
    say = lambda s: (print(s, flush=True), open(log, "a").write(s + "\n"))
    say("########## %s rpc6char -> %s (registration %s)" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                                                           os.path.relpath(out, ROOT), sha(os.path.relpath(REG, ROOT))[:16]))
    res = {"registration": os.path.relpath(REG, ROOT), "arms": {}}
    for name, arm in ARMS.items():
        env = {"MEMFILE": reg["memfile"], "CLOCKS": "locked", "NODES": "laptop pi4", "PLACE": PLACE, "PLACE_HEAD": "host",
               "SERVE_DRAFT": "none"}
        rc, txt = serve("start", arm["ctx"], env, log)
        line = next((x for x in txt.splitlines() if x.startswith("llama-server ")), "")
        rec = {"ctx": arm["ctx"], "serve_rc": rc, "llama_server": line, "runs": []}
        res["arms"][name] = rec
        if rc != 0 or not line:
            say("%s: serve_pool.sh start failed (rc %d)" % (name, rc)); serve("stop", arm["ctx"], {}, log); continue
        try:
            complete(arm["prompt"], SHORT)
            for rep in range(1, REPS + 1):
                for n in (SHORT, LONG):
                    b = ss(); t = complete(arm["prompt"], n); e = ss()
                    d = {k: e[k] - b[k] for k in COUNTERS}
                    rec["runs"].append({"rep": rep, "n": n, "sockets": [b["sockets"], e["sockets"]], "delta": d, "timings": t})
                    say("%s rep %d n %d: %s; prompt %d, decode %.2f ms/token" % (name, rep, n, d, t["prompt_n"], t["predicted_per_token_ms"]))
        except Exception as ex:
            rec["error"] = repr(ex)[:300]; say("%s: %r" % (name, ex))
        rc2, _ = serve("stop", arm["ctx"], {}, log)
        rec["stop_rc"] = rc2
    json.dump(res, open(os.path.join(out, "measurements.json"), "w"), indent=1)
    say("RUN DONE %s" % datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
    print(out)


def score(a):
    d = os.path.abspath(a.dir)
    m = json.load(open(os.path.join(d, "measurements.json")))
    reg = json.load(open(os.path.join(ROOT, m["registration"])))
    changed = [f for f, h in reg["inputs_sha256"].items() if os.path.exists(os.path.join(ROOT, f)) and sha(f) != h]
    per = {}
    for name, rec in m["arms"].items():
        reps = {}
        for r in rec.get("runs", []):
            reps.setdefault(r["rep"], {})[r["n"]] = r
        rows = [{k: (x[LONG]["delta"][k] - x[SHORT]["delta"][k]) / (LONG - SHORT) for k in COUNTERS}
                for _, x in sorted(reps.items()) if SHORT in x and LONG in x
                and all(s == 1 for s in x[SHORT]["sockets"] + x[LONG]["sockets"])]
        per[name] = {"complete": len(rows) == REPS, "per_token": {k: round(sum(r[k] for r in rows) / len(rows), 2) for k in COUNTERS} if rows else {},
                     "prompt_n": next((r["timings"]["prompt_n"] for r in rec.get("runs", [])), None), "error": rec.get("error")}
    hy, v = reg["hypotheses"], {}
    ok = lambda k: per.get(k, {}).get("complete")
    to = lambda k: per[k]["per_token"]["bytes_acked"]
    if ok("C2"):
        v["R6-REPEAT"] = {"verdict": "HOLDS" if abs(to("C2") / REF["to"] - 1) <= hy["R6-REPEAT"]["tol"] and
                          abs(per["C2"]["per_token"]["bytes_received"] / REF["back"] - 1) <= hy["R6-REPEAT"]["tol"] else "FAILS",
                          "to": to("C2"), "back": per["C2"]["per_token"]["bytes_received"]}
    else:
        v["R6-REPEAT"] = {"verdict": "INCOMPLETE"}
    for hk, arm in (("R6-CTX", "C4"), ("R6-DEPTH", "D2")):
        v[hk] = ({"verdict": "HOLDS" if abs(to(arm) / to("C2") - 1) <= hy[hk]["tol"] else "FAILS", arm: to(arm), "C2": to("C2"),
                  "diff_B": round(to(arm) - to("C2"), 1)} if ok(arm) and ok("C2") else {"verdict": "INCOMPLETE"})
    sc = {"session": os.path.relpath(d, ROOT), "registration": m["registration"], "registration_sha256": sha(m["registration"]),
          "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "code_changed_since_registration": changed,
          "arms": per, "hypotheses": v}
    json.dump(sc, open(os.path.join(d, "score.json"), "w"), indent=1)
    lines = ["rpc6char: %s; code changed since registration: %s" % (reg["what"], ", ".join(changed) or "none"), ""]
    lines += ["%-10s %-11s %s" % (k, x["verdict"], {a: b for a, b in x.items() if a != "verdict"}) for k, x in v.items()]
    lines += [""] + ["arm %s (ctx %d, prompt %s tokens): %s" % (k, ARMS[k]["ctx"], x["prompt_n"], x["per_token"]) for k, x in per.items()]
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
