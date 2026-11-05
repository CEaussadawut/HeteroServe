#!/usr/bin/env python3
"""The RPC link's bytes per decoded token, measured on the four-node cluster (study rpc4, step 1; 2026-10-24), registered
before it runs.

    python3 testbed/scripts/rpc4_char_20261024.py register --memfile testbed/dataset/memory_snapshot_<stamp>.json [--conditions TEXT]
    python3 testbed/scripts/rpc4_char_20261024.py run           # -> testbed/results/rpc4char_<stamp>/ (session.log, measurements.json)
    python3 testbed/scripts/rpc4_char_20261024.py score <dir>   # -> <dir>/score.json, score.txt

Why. ns4c's head-on-host arm at 5 Mbit/s ran 21.2 % over its prediction and bs4's backend-sampling arm 27.3 %. Their slopes
implied ~15-22 KB a token over the Pi 4's RPC link beyond the tensors predict_pool counts: the 8,192-byte hidden state each
way, and with the head last the 128,000 bytes of logits (TinyLlama: n_vocab 32,000 x 4) instead of the hidden state back.
This step measures the link's bytes directly, so that step 2 (register_rpc4plan_20261024.py) can price shaped links from
them, at rates not measured before.

How. TinyLlama Q4_0 on laptop 11 + Pi 4 11 (ns4c's split) is served by serve_pool.sh three ways: H (the head on the host),
L (the head last, on the Pi 4) and LB (the head last, with llama-server's --backend-sampling: bs4's arm). The link is
UNSHAPED. Around each greedy /completion (ignore_eos, the same prompt) the laptop's own TCP counters for the RPC connection
are read with `ss -tin` (dst 192.168.10.12, dport 50052; summed over sockets): bytes_acked (payload, laptop -> Pi 4),
bytes_received (payload, Pi 4 -> laptop), data_segs_out / data_segs_in (MSS-sized data segments). Each plan: a warm-up,
then three repetitions of a 16-token and a 144-token completion; per decoded token = (144-token delta - 16-token delta) /
128, so the prompt's traffic cancels. Nothing is shaped and nothing on the Pi 4 is touched; the server's decode timings are
logged as found (informational).

Disclosed before registering: one hand probe of plan H at 14:20 (the instrument check this project now runs before
queuing anything) read 19,452 B a token laptop -> Pi 4 and 8,200 B back, identical across two repetitions. Plan H here is a
repeat; L and LB are measured for the first time. The blind test is step 2's shaped timings.

Hypotheses (registered):
  RC-REPEAT  each plan's per-token payload in each direction is the same within 1 % across its three repetitions
  RC-LOGITS  L's Pi 4 -> laptop payload a token exceeds H's by the logits less the hidden state, 119,808 B, within 1 %
  RC-LB      LB's Pi 4 -> laptop payload a token is below 1,000 B (the sampled token, not the logits)
  RC-EXTRA   in every plan the laptop -> Pi 4 payload a token exceeds the 8,192-byte hidden state by 8-16 KB
Writes testbed/dataset/rpc4char_predictions_20261024.json, read-only.
"""
import argparse, datetime, hashlib, json, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REG = os.path.join(ROOT, "testbed", "dataset", "rpc4char_predictions_20261024.json")
KEY, CTX = "tinyllama_q4_0", 2048
PLACE = "laptop:11,pi4:11"
PLANS = {"H": {"PLACE_HEAD": "host"}, "L": {"PLACE_HEAD": "last"},
         "LB": {"PLACE_HEAD": "last", "SERVE_EXTRA": "--backend-sampling"}}
PEER, PORT = "192.168.10.12", 50052
URL = "http://192.168.1.108:8080/completion"
PROMPT = "The quick brown fox jumps over the lazy dog. Summarise this sentence in five words."
SHORT, LONG, REPS = 16, 144, 3
HIDDEN, LOGITS = 8192, 4 * 32000                        # TinyLlama's n_embd x 4 and n_vocab x 4 (model_shapes.json)
COUNTERS = ("bytes_acked", "bytes_received", "data_segs_out", "data_segs_in")
INPUTS = ("testbed/scripts/rpc4_char_20261024.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
          "testbed/scripts/netshape.sh", "ThinkPad/config/cluster.json")


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
           "script": "testbed/scripts/rpc4_char_20261024.py", "out_prefix": "rpc4char",
           "what": "the Pi 4's RPC link, bytes per decoded token by direction, measured with the laptop's TCP counters, for "
                   "TinyLlama Q4_0 on laptop 11 + Pi 4 11 with the head on the host (H), last (L), and last with backend "
                   "sampling (LB); unshaped",
           "model": KEY, "ctx": CTX, "place": PLACE, "plans": PLANS, "peer": "%s:%d" % (PEER, PORT), "prompt": PROMPT,
           "short": SHORT, "long": LONG, "reps": REPS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT),
           "counted_by_predictor": {"H": {"in": HIDDEN, "out": HIDDEN}, "L": {"in": HIDDEN, "out": LOGITS},
                                    "LB": {"in": HIDDEN, "out": 4}},
           "probe_disclosed": {"when": "2026-10-24T14:20+07:00", "plan": "H", "laptop_to_pi4_B_per_token": 19452,
                               "pi4_to_laptop_B_per_token": 8200, "reps": 2},
           **({"conditions": a.conditions} if a.conditions else {}),
           "hypotheses": {
               "RC-REPEAT": {"tol": 0.01, "claim": "each plan's per-token payload in each direction is the same within 1 % "
                                                   "across its three repetitions"},
               "RC-LOGITS": {"expected": LOGITS - HIDDEN, "tol": 0.01,
                             "claim": "L's Pi 4 -> laptop payload a token exceeds H's by %d B within 1 %%" % (LOGITS - HIDDEN)},
               "RC-LB": {"below": 1000, "claim": "LB's Pi 4 -> laptop payload a token is below 1,000 B"},
               "RC-EXTRA": {"lo": 8000, "hi": 16000, "claim": "in every plan the laptop -> Pi 4 payload a token exceeds the "
                                                             "8,192-byte hidden state by 8-16 KB"}},
           "inputs_sha256": {f: sha(f) for f in INPUTS + (os.path.relpath(os.path.abspath(a.memfile), ROOT),)}}
    json.dump(reg, open(REG, "w"), indent=1)
    os.chmod(REG, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(REG, ROOT), sha(os.path.relpath(REG, ROOT))[:16]))


def ss():
    t = subprocess.run(["ss", "-tinH", "state", "established", "dst", PEER, "( dport = :%d )" % PORT],
                       capture_output=True, text=True).stdout
    out = {k: sum(int(x) for x in re.findall(r"\b%s:(\d+)" % k, t)) for k in COUNTERS}
    out["sockets"] = len(re.findall(r"%s:%d" % (re.escape(PEER), PORT), t))
    return out


def complete(n):
    body = json.dumps({"prompt": PROMPT, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": False,
                       "ignore_eos": True}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    t = r["timings"]
    if t["predicted_n"] != n:
        raise RuntimeError("decoded %d tokens, not %d" % (t["predicted_n"], n))
    return {"predicted_n": t["predicted_n"], "predicted_per_token_ms": t["predicted_per_token_ms"],
            "prompt_n": t["prompt_n"], "prompt_ms": t["prompt_ms"], "text": r.get("content", "")[:80]}


def serve(cmd, env, log):
    e = dict(os.environ, MODEL=KEY, CTX=str(CTX), **env)
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
    out = os.path.join(ROOT, "testbed", "results", "rpc4char_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out)
    log = os.path.join(out, "session.log")
    say = lambda s: (print(s, flush=True), open(log, "a").write(s + "\n"))
    say("########## %s rpc4char -> %s (registration %s)" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                                                           os.path.relpath(out, ROOT), sha(os.path.relpath(REG, ROOT))[:16]))
    res = {"registration": os.path.relpath(REG, ROOT), "plans": {}}
    for plan, env in PLANS.items():
        base = {"MEMFILE": reg["memfile"], "CLOCKS": "locked", "NODES": "laptop pi4", "PLACE": PLACE, "SERVE_DRAFT": "none"}
        rc, txt = serve("start", dict(base, **env), log)
        line = next((x for x in txt.splitlines() if x.startswith("llama-server ")), "")
        rec = {"env": env, "serve_rc": rc, "llama_server": line, "runs": []}
        res["plans"][plan] = rec
        if rc != 0 or not line:
            say("%s: serve_pool.sh start failed (rc %d)" % (plan, rc)); serve("stop", {}, log); continue
        if ("--backend-sampling" in line) != ("SERVE_EXTRA" in env):
            say("%s: the server's command line does not match the plan's flags; skipped" % plan)
            rec["skipped"] = "flag mismatch"; serve("stop", {}, log); continue
        try:
            complete(SHORT)                                         # warm-up
            for rep in range(1, REPS + 1):
                for n in (SHORT, LONG):
                    b = ss(); t = complete(n); e = ss()
                    d = {k: e[k] - b[k] for k in COUNTERS}
                    rec["runs"].append({"rep": rep, "n": n, "sockets": [b["sockets"], e["sockets"]], "delta": d, "timings": t})
                    say("%s rep %d n %d: %s; decode %.2f ms/token" % (plan, rep, n, d, t["predicted_per_token_ms"]))
        except Exception as ex:                                     # recorded as found, the plan left incomplete
            rec["error"] = repr(ex)[:300]; say("%s: %r" % (plan, ex))
        rc2, _ = serve("stop", {}, log)
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
    for plan, rec in m["plans"].items():
        reps = {}
        for r in rec.get("runs", []):
            reps.setdefault(r["rep"], {})[r["n"]] = r
        rows = []
        for rep, x in sorted(reps.items()):
            if SHORT in x and LONG in x and all(s == 1 for s in x[SHORT]["sockets"] + x[LONG]["sockets"]):
                rows.append({k: (x[LONG]["delta"][k] - x[SHORT]["delta"][k]) / (LONG - SHORT) for k in COUNTERS})
        complete_ = len(rows) == REPS
        mean = {k: sum(r[k] for r in rows) / len(rows) for k in COUNTERS} if rows else {}
        spread = {k: (max(r[k] for r in rows) / min(r[k] for r in rows) - 1) if rows and min(r[k] for r in rows) > 0 else None
                  for k in COUNTERS}
        dec = [r["timings"]["predicted_per_token_ms"] for r in rec.get("runs", []) if r["n"] == LONG]
        per[plan] = {"complete": complete_, "reps": rows, "per_token": {k: round(v, 2) for k, v in mean.items()},
                     "rep_spread": {k: (round(v, 5) if v is not None else None) for k, v in spread.items()},
                     "wire_per_token": ({"in": round(mean["bytes_acked"] + 66 * mean["data_segs_out"], 1),
                                         "out": round(mean["bytes_received"] + 66 * mean["data_segs_in"], 1)} if mean else {}),
                     "decode_ms_long_unshaped": [round(x, 2) for x in dec], "error": rec.get("error"), "skipped": rec.get("skipped")}
    H, L, LB = per.get("H", {}), per.get("L", {}), per.get("LB", {})
    ok = lambda *p: all(x.get("complete") for x in p)
    hy = reg["hypotheses"]
    v = {}
    v["RC-REPEAT"] = ({"verdict": "HOLDS" if all(s is not None and s <= hy["RC-REPEAT"]["tol"] for p in per.values()
                                                  for k, s in p["rep_spread"].items() if k in ("bytes_acked", "bytes_received"))
                       else "FAILS", "spread": {p: {k: per[p]["rep_spread"][k] for k in ("bytes_acked", "bytes_received")} for p in per}}
                      if ok(H, L, LB) else {"verdict": "INCOMPLETE"})
    if ok(H, L):
        diff = L["per_token"]["bytes_received"] - H["per_token"]["bytes_received"]
        v["RC-LOGITS"] = {"verdict": "HOLDS" if abs(diff / hy["RC-LOGITS"]["expected"] - 1) <= hy["RC-LOGITS"]["tol"] else "FAILS",
                          "diff_B": round(diff, 1), "expected_B": hy["RC-LOGITS"]["expected"]}
    else:
        v["RC-LOGITS"] = {"verdict": "INCOMPLETE"}
    v["RC-LB"] = ({"verdict": "HOLDS" if LB["per_token"]["bytes_received"] < hy["RC-LB"]["below"] else "FAILS",
                   "pi4_to_laptop_B": LB["per_token"]["bytes_received"]} if ok(LB) else {"verdict": "INCOMPLETE"})
    if ok(H, L, LB):
        ex = {p: round(per[p]["per_token"]["bytes_acked"] - HIDDEN, 1) for p in per}
        v["RC-EXTRA"] = {"verdict": "HOLDS" if all(hy["RC-EXTRA"]["lo"] <= x <= hy["RC-EXTRA"]["hi"] for x in ex.values())
                         else "FAILS", "extra_B": ex}
    else:
        v["RC-EXTRA"] = {"verdict": "INCOMPLETE"}
    sc = {"session": os.path.relpath(d, ROOT), "registration": m["registration"],
          "registration_sha256": sha(m["registration"]), "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
          "code_changed_since_registration": changed, "plans": per, "hypotheses": v}
    json.dump(sc, open(os.path.join(d, "score.json"), "w"), indent=1)
    lines = ["rpc4char: %s; code changed since registration: %s" % (reg["what"], ", ".join(changed) or "none"), ""]
    lines += ["%-10s %-11s %s" % (k, x["verdict"], {a: b for a, b in x.items() if a != "verdict"}) for k, x in v.items()]
    lines.append("")
    for p, x in per.items():
        lines.append("plan %s: per token %s; wire %s; spread %s; decode (144-token runs, unshaped) %s%s" % (
            p, x["per_token"], x["wire_per_token"], x["rep_spread"], x["decode_ms_long_unshaped"],
            "; " + str(x["error"] or x["skipped"]) if (x["error"] or x["skipped"]) else ""))
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
