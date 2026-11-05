#!/usr/bin/env python3
"""Scheduler upgrade 1, step 2: the draft model's acceptance, on the laptop, exactly as registered
(register_spec_accept_20261010.py).

    python3 testbed/scripts/spec_accept_20261010.py run testbed/dataset/spec_accept_predictions_20261010.json [--smoke]
    python3 testbed/scripts/spec_accept_20261010.py score testbed/results/spec_accept_<stamp>

run: checks both models' sha256 in the laptop's cache against the registration; starts the pinned llama-server on
127.0.0.1 with the registered arguments (its log: <dir>/server.log); sends the 50 prompts in order, one at a time,
greedy, in Qwen3's non-thinking chat format; writes every response (text, timings with draft_n and draft_n_accepted) to
<dir>/run.jsonl as it goes; stops the server. --smoke: the first prompt only, 16 tokens, into spec_accept_smoke_<stamp>/
(the plumbing only: never scored).
score: reads every "accepted n/m draft tokens" line of server.log (one per check, in order) and assigns them to the
requests by each request's draft_n; over the checks that drafted the full 4 tokens, L = tokens accepted:
  SA-PAY1  alpha1 = share of checks with L >= 1, at least the registered bar (0.60)
INCOMPLETE if fewer than the 50 prompts were answered, or the checks cannot be assigned to the requests. Exploratory: the
acceptance per position P(L >= i | L >= i - 1); tau(d) = 1 + E[min(L, d)]; the ceiling speedup tau(d) x T(1) / T(d + 1)
with step 1's measured T(k), and the drafter time per drafted token that would still reach 1.3x; English against Thai;
the laptop's own speeds. Writes <dir>/score.json.
"""
import datetime, hashlib, json, os, re, signal, statistics, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.expanduser("~/.cache/heteroserve-models")
ACC = re.compile(r"accepted\s+(\d+)/\s*(\d+) draft tokens")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def healthy(url):
    try:
        with urllib.request.urlopen(url + "/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def run(reg_path, smoke=False):
    reg = json.load(open(reg_path))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", ("spec_accept_smoke_%s" if smoke else "spec_accept_%s") % stamp)
    os.makedirs(d)
    prog = open(os.path.join(d, "progress.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); prog.write(line + "\n"); prog.flush()
    say("registration %s (%s)%s" % (os.path.relpath(reg_path, ROOT), sha(reg_path)[:16], " SMOKE: never scored" if smoke else ""))
    for m in (reg["target"], reg["draft"]):
        got = sha(os.path.join(CACHE, m["file"]))
        if got != m["sha256"]:
            say("FATAL: %s sha256 %s, registered %s" % (m["file"], got[:16], m["sha256"][:16])); sys.exit(1)
        say("model %s sha256 OK" % m["file"])
    port = int(reg["server_args"][reg["server_args"].index("--port") + 1])
    url = "http://127.0.0.1:%d" % port
    if healthy(url):
        say("FATAL: something already answers on %s" % url); sys.exit(1)
    log = open(os.path.join(d, "server.log"), "w")
    env = dict(os.environ, **reg.get("server_env", {}))
    srv = subprocess.Popen([reg["server"]] + reg["server_args"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                           env=env)
    say("server pid %d: %s %s %s" % (srv.pid, " ".join("%s=%s" % kv for kv in reg.get("server_env", {}).items()),
                                     reg["server"], " ".join(reg["server_args"])))
    try:
        t0 = time.time()
        while not healthy(url):
            if srv.poll() is not None:
                say("FATAL: the server exited (%s) before it was ready; see server.log" % srv.returncode); sys.exit(1)
            if time.time() - t0 > 900:
                say("FATAL: the server was not ready in 900 s"); sys.exit(1)
            time.sleep(1)
        say("server ready after %.0f s" % (time.time() - t0))
        prompts = reg["prompts"][:1] if smoke else reg["prompts"]
        n = 16 if smoke else reg["n_predict"]
        with open(os.path.join(d, "run.jsonl"), "a") as out:
            for i, p in enumerate(prompts, 1):
                body = {"prompt": reg["chat_format"] % p["text"], "n_predict": n, **reg["sampling"]}
                t = time.time()
                try:
                    r = post(url + "/completion", body, timeout=3600)
                except Exception as e:
                    say("!!!!! %s failed: %s" % (p["id"], e)); continue
                row = {"id": p["id"], "lang": p["lang"], "category": p["category"], "wall_s": round(time.time() - t, 3),
                       "tokens_predicted": r.get("tokens_predicted"), "stop_type": r.get("stop_type"),
                       "timings": r.get("timings"), "content": r.get("content")}
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                tm = r.get("timings") or {}
                say("%2d/%d %-5s %3s tokens, %5.1f s, draft %s accepted %s" % (i, len(prompts), p["id"], row["tokens_predicted"],
                                                                           row["wall_s"], tm.get("draft_n"), tm.get("draft_n_accepted")))
    finally:
        os.killpg(srv.pid, signal.SIGTERM)
        try:
            srv.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(srv.pid, signal.SIGKILL); srv.wait()
        say("server stopped (exit %s)" % srv.returncode)
    say("RUN DONE %s" % os.path.relpath(d, ROOT))


def score(d, reg_path=None):
    rows = [json.loads(l) for l in open(os.path.join(d, "run.jsonl"))]
    reg_path = reg_path or os.path.join(ROOT, "testbed", "dataset", "spec_accept_predictions_20261010.json")
    reg = json.load(open(reg_path))
    D = reg["draft_n_max"]
    checks = [(int(a), int(b)) for a, b in ACC.findall(open(os.path.join(d, "server.log"), encoding="utf-8", errors="replace").read())]
    # assign the checks to the requests, in order, by each request's draft_n
    per_req, i, ok = {}, 0, True
    for r in rows:
        want, got = (r.get("timings") or {}).get("draft_n", 0) or 0, []
        while i < len(checks) and sum(b for _, b in got) < want:
            got.append(checks[i]); i += 1
        if sum(b for _, b in got) != want:
            ok = False
        per_req[r["id"]] = got
    ok = ok and i == len(checks)
    full = [a for a, b in checks if b == D]

    def stats(Ls):
        if not Ls:
            return None
        pos, alive = [], len(Ls)
        for k in range(1, D + 1):
            nxt = sum(1 for L in Ls if L >= k)
            pos.append(round(nxt / alive, 4) if alive else None)
            alive = nxt
        tau = {str(dd): round(1 + statistics.fmean(min(L, dd) for L in Ls), 4) for dd in range(1, D + 1)}
        return {"checks": len(Ls), "alpha1": round(sum(1 for L in Ls if L >= 1) / len(Ls), 4), "per_position": pos,
                "tau": tau, "L_hist": {str(k): sum(1 for L in Ls if L == k) for k in range(D + 1)}}
    allst = stats(full)
    by_lang = {}
    if ok:
        for lang in sorted({r["lang"] for r in rows}):
            by_lang[lang] = stats([a for r in rows if r["lang"] == lang for a, b in per_req[r["id"]] if b == D])
    complete = len(rows) == len(reg["prompts"]) and ok and allst is not None
    bar = reg["predicted"]["alpha1_bar"]
    hyp = {"SA-PAY1": {"verdict": ("HOLDS" if allst["alpha1"] >= bar else "FAILS") if complete else "INCOMPLETE",
                       "alpha1": allst["alpha1"] if allst else None, "bar": bar}}
    T = {int(k): v for k, v in reg["step1"]["T_ms"].items()}
    ceil = {}
    if allst:
        for dd in range(1, D + 1):
            tau = allst["tau"][str(dd)]
            if dd + 1 in T:
                ceil[str(dd)] = {"tau": tau, "speedup_free_draft": round(tau * T[1] / T[dd + 1], 3),
                                 "max_draft_ms_per_token_for_1.3x": round((tau * T[1] / 1.3 - T[dd + 1]) / dd, 1)}
    tm = [r["timings"] for r in rows if r.get("timings")]
    lap = {"target_with_draft_ms_per_token": round(sum(t["predicted_ms"] for t in tm) / max(1, sum(t["predicted_n"] for t in tm)), 2),
           "prompt_ms_per_token": round(sum(t["prompt_ms"] for t in tm) / max(1, sum(t["prompt_n"] for t in tm)), 2),
           "draft_accept_overall": round(sum(t.get("draft_n_accepted", 0) for t in tm) / max(1, sum(t.get("draft_n", 0) for t in tm)), 4)}
    code = [f for f, h in reg["inputs_sha256"].items() if sha(os.path.join(HERE, f)) != h]
    s = {"registration": os.path.relpath(reg_path, ROOT), "registration_sha256": sha(reg_path), "run": os.path.relpath(d, ROOT),
         "code_changed_since_registration": code, "requests": len(rows), "checks_logged": len(checks),
         "checks_assigned": ok, "hypotheses": hyp, "all": allst, "by_lang": by_lang, "ceiling_by_draft": ceil,
         "laptop": lap, "tokens_predicted": sum(r.get("tokens_predicted") or 0 for r in rows),
         "stop_types": {k: sum(1 for r in rows if r.get("stop_type") == k) for k in sorted({str(r.get("stop_type")) for r in rows})}}
    json.dump(s, open(os.path.join(d, "score.json"), "w"), indent=1)
    return s


def main(argv):
    if len(argv) < 2 or argv[0] not in ("run", "score"):
        sys.exit(__doc__)
    if argv[0] == "run":
        run(argv[1], smoke="--smoke" in argv)
    else:
        s = score(argv[1], argv[argv.index("--reg") + 1] if "--reg" in argv else None)
        print(json.dumps({k: s[k] for k in ("requests", "checks_logged", "checks_assigned", "hypotheses", "all", "by_lang",
                                            "ceiling_by_draft", "laptop", "stop_types", "code_changed_since_registration")},
                         indent=1, ensure_ascii=False))
        print("wrote %s" % os.path.relpath(os.path.join(argv[1], "score.json"), ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
