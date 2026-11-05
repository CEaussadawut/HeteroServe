#!/usr/bin/env python3
"""The RPC message term's first-worker rule, tested blind (study rpc8, 2026-10-24; the user: "continue what's left for
project 2"), registered before it runs.

    python3 testbed/scripts/rpc8_char_20261024.py register --memfile testbed/dataset/memory_snapshot_<stamp>.json [--conditions TEXT]
    python3 testbed/scripts/rpc8_char_20261024.py run           # -> testbed/results/rpc8char_<stamp>/
    python3 testbed/scripts/rpc8_char_20261024.py score <dir>

rpc7 (testbed/results/rpc7char_20261024_195615) found predict_pool.rpc_bytes() exact to the byte on Qwen3-32B (dense) and
wrong on Qwen3.6-35B-A3B (a hybrid MoE): every worker received 330 B more, and the first worker 17,330 B more. A reading of
llama.cpp at the pin, made after those data, fits both to the byte:
  1. ggml_backend_sched expands each RPC worker's backend BACKWARDS from its first pinned op (the layer's "norm", which
     llama_context::graph_get_cb pins to the layer's device while a batch has < 32 tokens) through the weightless ops before
     it, and stops at the first op on the host's CPU (ggml-backend.cpp, "expand gpu up"). So the host's last block's trailing
     weightless ops run on the first worker, and every tensor they read crosses the link, each with a 314 B set_tensor header.
     A dense block ends in one add (FFN output + residual): one hidden state more than a later worker gets, rpc_bytes()'s
     "first" term. The 35B's block ends in routed experts + sigmoid-gated shared expert, then + residual: three hidden states
     and the one-value gate, 2 x (8,192 + 314) + (4 + 314) = 17,330 B more than a later worker.
  2. A qwen35 hybrid layer reads two inputs a dense one does not have in that size: the recurrent state's copy index (one
     int32 tensor, 4 + 314 B) and IMROPE's four positions a token instead of one (12 B more): 330 B on every worker.
The reading predicts two models it was not fitted on:
  M30  Qwen3-30B-A3B Q3_K_M (qwen3moe: experts without a shared expert, NEOX RoPE). Its block ends in build_moe_ffn's
       weighting and sum (ggml_mul(experts, weights), seven adds of views) then + residual, with no CPU op between them, so
       the first worker receives the eight experts' outputs before weighting (n_embd x 8 x 4 = 65,536 B), the router weights
       (8 x 4 B) and the residual: 57,690 B more than rpc_bytes() says. A later worker: rpc_bytes() exactly (no hybrid inputs).
  H9   Qwen3.5-9B Q4_K_M (qwen35: a dense hybrid, IMROPE). Its block ends in one add, as a dense one: rpc_bytes() + 330 B on
       every worker, the first worker's extra the dense one.
Arms, the head on the host, unshaped, ctx 2,048, on the laptop + Pi 5 (first worker) + Pi 4 (last); every worker holds both
an attention and a recurrent layer of H9. As rpc7: the laptop's TCP counters (ss -tin) per worker connection, read around
each greedy completion (ignore_eos); two repetitions of a 16- and a 48-token completion; a token's bytes = the difference / 32.
Each model's sha256 is checked on the laptop before it is served (testbed/models.md).
Hypotheses (registered):
  R8-MOE-FIRST  M30's first worker receives the reading's bytes a token within 2 % (rpc_bytes() as it is: -75 %)
  R8-MOE-LAST   M30's last worker receives rpc_bytes()'s bytes within 0.5 % (the reading adds nothing there)
  R8-HYB        each H9 worker receives rpc_bytes() + 330 B within 0.5 % (rpc_bytes() as it is: -0.9 and -1.7 %)
  R8-BACK       every worker returns the hidden state plus 8 B within 0.5 %
  R8-REPEAT     each figure repeats within 1 % across the two repetitions
Writes testbed/dataset/rpc8char_predictions_20261024.json, read-only.
"""
import argparse, datetime, hashlib, json, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P  # noqa: E402
import qr3_tpot as Q      # noqa: E402

REG = os.path.join(ROOT, "testbed", "dataset", "rpc8char_predictions_20261024.json")
CTX = 2048
ARMS = {"M30": {"model": "qwen3moe", "place": "laptop:36,pi5:6,pi4:6", "env": {}, "kind": "moe"},
        "H9": {"model": "qwen35_9b", "place": "laptop:25,pi5:4,pi4:4", "env": {}, "kind": "hybrid"}}
HDR = 314                    # one set_tensor's bytes besides its data (rpc4-rpc6: the first worker's "+314", the last's 4 + 314)
HYBRID_EXTRA = (4 + HDR) + 12          # the recurrent state's copy index, and IMROPE's three more positions a token
WIP = {"192.168.10.11": "pi5", "192.168.10.12": "pi4", "192.168.1.114": "jetson"}
URL = "http://192.168.1.108:8080/completion"
PROMPT = "The quick brown fox jumps over the lazy dog. Summarise this sentence in five words."
SHORT, LONG, REPS = 16, 48, 2
COUNTERS = ("bytes_acked", "bytes_received", "data_segs_out", "data_segs_in")
INPUTS = ("testbed/scripts/rpc8_char_20261024.py", "testbed/scripts/predict_pool.py", "testbed/scripts/serve_pool.sh",
          "testbed/scripts/serve_defaults.sh", "testbed/scripts/qr3_tpot.py", "ThinkPad/config/cluster.json",
          "testbed/dataset/model_shapes.json", "testbed/results/rpc7char_20261024_195615/score.json")


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p) if not os.path.isabs(p) else p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def predictions():
    """Per arm and worker: rpc_bytes() as it is, and the reading's figure with its parts."""
    shapes = json.load(open(os.path.join(ROOT, "testbed", "dataset", "model_shapes.json")))
    out = {}
    for arm, a in ARMS.items():
        q = shapes[P.MODELS[a["model"]][0]]
        h = q["activation_bytes"]
        workers = [x.split(":")[0] for x in a["place"].split(",")][1:]
        out[arm] = {}
        for i, w in enumerate(workers, 1):
            inb, ob, _ = P.rpc_bytes(q, i, len(workers), h, 0, P.RPC_MSG, False)
            parts = {}
            if a["kind"] == "moe" and i == 1:
                k = q["expert_used_count"]
                parts = {"minus the FFN output rpc_bytes() counts": -(h + HDR),
                         "the experts' outputs before weighting (n_embd x %d x 4)" % k: q["n_embd"] * k * 4 + HDR,
                         "the router weights (%d x 4)" % k: k * 4 + HDR}
            if a["kind"] == "hybrid":
                parts = {"the recurrent state's copy index and IMROPE's positions": HYBRID_EXTRA}
            out[arm][w] = {"to": inb + sum(parts.values()), "to_rpc_bytes": inb, "parts": parts, "back": ob,
                           "place": i, "of": len(workers)}
    return out


def fits(memfile):
    mem = json.load(open(memfile))
    out = {}
    for arm, a in ARMS.items():
        assign = [(x.split(":")[0], int(x.split(":")[1])) for x in a["place"].split(",")]
        e = P.plan_fixed(a["model"], assign, "host", CTX, mem)
        m = P.memory(assign, e["q"], mem, e["devices"], e["model"], head_at=0)
        out[arm] = {n: v["margin_mib"] for n, v in m.items()}
        if min(out[arm].values()) < 256:
            raise SystemExit("refused: %s does not fit with 256 MiB to spare: %s" % (arm, out[arm]))
    return out


def register(a):
    if os.path.exists(REG):
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(REG, ROOT))
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/rpc8_char_20261024.py", "out_prefix": "rpc8char",
           "what": "the RPC term's first-worker rule (the scheduler's backward expansion) and the hybrid inputs, read from "
                   "llama.cpp after rpc7, tested blind on Qwen3-30B-A3B Q3_K_M (MoE, no shared expert) and Qwen3.5-9B Q4_K_M "
                   "(dense hybrid): bytes per decoded token per worker, the head on the host, unshaped",
           "ctx": CTX, "arms": ARMS, "prompt": PROMPT, "short": SHORT, "long": LONG, "reps": REPS,
           "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), "margins_mib": fits(a.memfile),
           "rpc_msg": P.RPC_MSG, "header_bytes": HDR, "hybrid_extra": HYBRID_EXTRA, "predicted": predictions(),
           "models": {arm: {"file": Q.MODEL_FILE[x["model"]][0], "sha256": Q.MODEL_FILE[x["model"]][1]} for arm, x in ARMS.items()},
           **({"conditions": a.conditions} if a.conditions else {}),
           "hypotheses": {
               "R8-MOE-FIRST": {"arm": "M30", "place": 1, "tol": 0.02,
                                "claim": "M30's first worker receives the reading's bytes a token within 2 %"},
               "R8-MOE-LAST": {"arm": "M30", "place": 2, "against": "to_rpc_bytes", "tol": 0.005,
                               "claim": "M30's last worker receives rpc_bytes()'s bytes a token within 0.5 %"},
               "R8-HYB": {"arm": "H9", "tol": 0.005, "claim": "each H9 worker receives rpc_bytes() + 330 B a token within 0.5 %"},
               "R8-BACK": {"tol": 0.005, "claim": "every worker returns the hidden state plus 8 B a token within 0.5 %"},
               "R8-REPEAT": {"tol": 0.01, "claim": "each figure repeats within 1 % across the two repetitions"}},
           "inputs_sha256": {f: sha(f) for f in INPUTS + (os.path.relpath(os.path.abspath(a.memfile), ROOT),)}}
    if a.dry_run:
        print(json.dumps({k: reg[k] for k in ("margins_mib", "predicted", "models")}, indent=1)); return
    json.dump(reg, open(REG, "w"), indent=1)
    os.chmod(REG, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(REG, ROOT), sha(os.path.relpath(REG, ROOT))[:16]))
    print(json.dumps({arm: {w: x["to"] for w, x in ws.items()} for arm, ws in reg["predicted"].items()}))


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


def pi_throttled():
    out = {}
    for n in ("pi5", "pi4"):
        p = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "LogLevel=ERROR", n,
                            "vcgencmd get_throttled"], capture_output=True, text=True, timeout=30)
        out[n] = p.stdout.strip() or ("ssh exit %d" % p.returncode)
    return out


def run(a):
    reg = json.load(open(REG))
    if os.access(REG, os.W_OK):
        raise SystemExit("refused: the registration is writable")
    if os.path.exists(os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")):
        raise SystemExit("refused: serve_pool.sh state exists")
    out = os.path.join(ROOT, "testbed", "results", "rpc8char_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out)
    log = os.path.join(out, "session.log")
    say = lambda s: (print(s, flush=True), open(log, "a").write(s + "\n"))
    say("########## %s rpc8char -> %s (registration %s)" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                                                           os.path.relpath(out, ROOT), sha(os.path.relpath(REG, ROOT))[:16]))
    res = {"registration": os.path.relpath(REG, ROOT), "arms": {}, "throttled_before": pi_throttled()}
    say("throttled before: %s" % res["throttled_before"])
    for name, arm in ARMS.items():
        f, want = reg["models"][name]["file"], reg["models"][name]["sha256"]
        got = sha(os.path.realpath(os.path.expanduser(f)))
        rec = {"model": arm["model"], "place": arm["place"], "model_sha256": got, "runs": []}
        res["arms"][name] = rec
        say("%s: %s sha256 %s (%s)" % (name, f, got[:16], "pinned" if got == want else "NOT THE PINNED BYTES"))
        if got != want:
            rec["error"] = "model sha256 mismatch"; continue
        nodes = " ".join(x.split(":")[0] for x in arm["place"].split(","))
        env = dict({"MEMFILE": reg["memfile"], "CLOCKS": "locked", "NODES": nodes, "PLACE": arm["place"], "PLACE_HEAD": "host",
                    "SERVE_DRAFT": "none"}, **arm["env"])
        rc, txt = serve("start", arm["model"], env, log)
        line = next((x for x in txt.splitlines() if x.startswith("llama-server ")), "")
        m = re.search(r"--rpc (\S+)", line)
        peers = [(x.rsplit(":", 1)[0], int(x.rsplit(":", 1)[1])) for x in m.group(1).split(",")] if m else []
        rec.update({"serve_rc": rc, "llama_server": line, "peers": {WIP.get(i, i): "%s:%d" % (i, p) for i, p in peers}})
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
    res["throttled_after"] = pi_throttled()
    say("throttled after: %s" % res["throttled_after"])
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
    for hk in ("R8-MOE-FIRST", "R8-MOE-LAST", "R8-HYB"):
        h = hy[hk]
        ws = {w: x for w, x in per.get(h["arm"], {}).items() if "place" not in h or x["predicted"]["place"] == h["place"]}
        want = [w for w, x in reg["predicted"][h["arm"]].items() if "place" not in h or x["place"] == h["place"]]
        if ws and sorted(ws) == sorted(want) and all(x["complete"] for x in ws.values()):
            key = h.get("against", "to")
            rows = {w: {"measured": x["per_token"]["bytes_acked"], "predicted": x["predicted"][key],
                        "rpc_bytes_as_is": x["predicted"]["to_rpc_bytes"],
                        "err_pct": round(100 * (x["per_token"]["bytes_acked"] / x["predicted"][key] - 1), 2)} for w, x in ws.items()}
            v[hk] = {"verdict": "HOLDS" if all(abs(r["err_pct"]) <= 100 * h["tol"] for r in rows.values()) else "FAILS", "workers": rows}
        else:
            v[hk] = {"verdict": "INCOMPLETE"}
    allw = [(arm, w, x) for arm, ws in per.items() for w, x in ws.items()]
    complete = allw and all(x["complete"] for _, _, x in allw) and all(len(per.get(arm, {})) == len(reg["predicted"][arm]) for arm in ARMS)
    if complete:
        backs = {"%s/%s" % (arm, w): {"measured": x["per_token"]["bytes_received"], "predicted": x["predicted"]["back"]} for arm, w, x in allw}
        v["R8-BACK"] = {"verdict": "HOLDS" if all(abs(b["measured"] / b["predicted"] - 1) <= hy["R8-BACK"]["tol"] for b in backs.values())
                        else "FAILS", "workers": backs}
        v["R8-REPEAT"] = {"verdict": "HOLDS" if all(s <= hy["R8-REPEAT"]["tol"] for s in spreads.values()) else "FAILS",
                          "max_spread_pct": round(100 * max(spreads.values()), 3) if spreads else None}
    else:
        v["R8-BACK"] = {"verdict": "INCOMPLETE"}; v["R8-REPEAT"] = {"verdict": "INCOMPLETE"}
    sc = {"session": os.path.relpath(d, ROOT), "registration": m["registration"], "registration_sha256": sha(m["registration"]),
          "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "code_changed_since_registration": changed,
          "throttled": {"before": m.get("throttled_before"), "after": m.get("throttled_after")}, "arms": per, "hypotheses": v}
    json.dump(sc, open(os.path.join(d, "score.json"), "w"), indent=1)
    lines = ["rpc8char: %s; code changed since registration: %s" % (reg["what"], ", ".join(changed) or "none"), ""]
    lines += ["%-13s %-11s %s" % (k, x["verdict"], {a2: b for a2, b in x.items() if a2 != "verdict"}) for k, x in v.items()]
    lines += [""] + ["arm %s worker %s: %s (predicted %s)" % (arm, w, x["per_token"], x["predicted"]) for arm, ws in per.items() for w, x in ws.items()]
    lines += ["", "throttled: %s" % sc["throttled"]]
    open(os.path.join(d, "score.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("register"); r.add_argument("--memfile", required=True); r.add_argument("--conditions", default="")
    r.add_argument("--dry-run", action="store_true")
    sp.add_parser("run")
    s = sp.add_parser("score"); s.add_argument("dir")
    a = ap.parse_args()
    {"register": register, "run": run, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
