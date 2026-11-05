#!/usr/bin/env python3
"""Run sp10 (register_sp10_20261103.py; the scheduler review's item 10): speculation with no draft model on plans the checking
cost was not fitted on, then its scorer.

    python3 testbed/scripts/sp10_session_20261103.py --registration testbed/dataset/sp10_predictions_20261103.json

As explore_spec_20261102.py serves and asks (its helpers: the same prompts, scan, requests and Jetson guard), but every
config, speculation setting, prompt and size comes from the registration, in its order. Refuses to start when a registered
input's sha256 has changed, or when serve_pool.sh's state exists. Per config: the Jetson's guard while the plan holds the
Jetson; per setting: serve_pool.sh start (SERVE_PLANNER=v4, SERVE_DRAFT=none, SERVE_SWITCH=off, the setting's flags in
SERVE_EXTRA), a warm-up, the prompt-length scan when the registration asks for one, the requests, serve_pool.sh stop, the
llama-server log copied in; get_throttled and the temperature of each Pi in the plan before and after.
Raw output: testbed/results/sp10_<stamp>/ (run.log, start_ / stop_ / server_<config>_<spec>.log, results.json), then
score.json and score.txt from the registration's scorer.
"""
import argparse, datetime, glob, hashlib, json, os, shutil, socket, subprocess, sys, time, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import explore_spec_20261102 as X                           # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def complete(prompt, n, timeout):
    """explore_spec's complete(), with the registration's request timeout (2026-11-03: sp10's two hung serves)."""
    t0 = time.time()
    r = X.post("/completion", {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": False,
                               "ignore_eos": True}, timeout=timeout)
    t = r.get("timings", {})
    return {"wall_s": round(time.time() - t0, 3), "timings": t, "tokens_predicted": r.get("tokens_predicted"),
            "tokens_evaluated": r.get("tokens_evaluated"), "text": r.get("content", "")[:240]}


def timed_out(ex):
    return isinstance(ex, (socket.timeout, TimeoutError)) or (isinstance(ex, urllib.error.URLError)
                                                               and isinstance(ex.reason, (socket.timeout, TimeoutError)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", required=True)
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    tmo = reg.get("request_timeout_s", 1800)       # sp10b on (2026-11-03): a request that times out ends its arm's requests
    changed = sorted(f for f, h in reg["inputs_sha256"].items() if sha(f) != h)
    if changed:
        raise SystemExit("refused: registered inputs changed: %s" % changed)
    if os.path.exists(X.STATE):
        raise SystemExit("refused: serve_pool.sh state exists")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", "%s_%s" % (reg["out_prefix"], stamp))
    os.makedirs(d)
    log = open(os.path.join(d, "run.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); log.write(line + "\n"); log.flush()
    say("########## %s -> %s (registration %s)" % (reg["out_prefix"], os.path.relpath(d, ROOT), sha(os.path.relpath(a.registration, ROOT))[:16]))
    res = []
    for c in reg["order"]:
        cfg = reg["configs"][c]
        guarded = "jetson:" in cfg["PLACE"]
        if guarded:
            say("guard: %s" % X.guard_start())
        try:
            for sp in cfg["specs"]:
                env = dict(os.environ, MODEL=cfg["MODEL"], CTX=cfg["CTX"], PLACE=cfg["PLACE"], PLACE_HEAD=cfg["PLACE_HEAD"],
                           SERVE_PLANNER="v4", SERVE_DRAFT="none", SERVE_SWITCH="off", HETEROSERVE_RESERVE_MB="",
                           SERVE_EXTRA=reg["spec_args"][sp])
                env.pop("MEMFILE", None)
                say("%s %s before: %s" % (c, sp, X.pi_state(cfg["PLACE"])))
                t0 = time.time()
                p = subprocess.run(["bash", os.path.join(HERE, "serve_pool.sh"), "start"], cwd=ROOT, env=env, capture_output=True,
                                   text=True, timeout=3600)
                open(os.path.join(d, "start_%s_%s.log" % (c, sp)), "w").write(p.stdout + p.stderr)
                say("%s %s start rc %d in %.0f s" % (c, sp, p.returncode, time.time() - t0))
                logs = sorted(glob.glob(os.path.expanduser("~/logs/llama-server_serve_*.log")), key=os.path.getmtime)
                slog = logs[-1] if logs and os.path.getmtime(logs[-1]) >= t0 else None
                rec = {"config": c, "spec": sp, "spec_args": reg["spec_args"][sp], "model": cfg["MODEL"], "place": cfg["PLACE"],
                       "head": cfg["PLACE_HEAD"], "start_rc": p.returncode, "server_log": slog, "scan": [], "requests": []}
                if p.returncode == 0:
                    try:
                        complete("Hello.", 8, tmo)                            # a warm-up, not recorded
                        if sp == "none" and reg.get("scan"):
                            ids = X.post("/tokenize", {"content": reg["prompts"]["copy"] + " " + reg["prompts"]["free"]})["tokens"]
                            for m in reg["scan"]:
                                for i in range(reg["scan_reps"]):
                                    r = complete(ids[:m], 1, tmo)
                                    r.update(m=m, i=i)
                                    rec["scan"].append(r)
                                say("%s scan m %d: prompt_ms %s" % (c, m, [x["timings"].get("prompt_ms") for x in rec["scan"] if x["m"] == m]))
                        for pk in reg["prompt_order"]:
                            for i in range(reg["reqs"]):
                                if rec.get("timed_out"):
                                    break
                                try:
                                    r = complete(reg["prompts"][pk], reg["gen"], tmo)
                                except Exception as ex:
                                    r = {"error": repr(ex)[:200]}
                                    if timed_out(ex):
                                        rec["timed_out"] = "%s %d after %d s" % (pk, i, tmo)
                                        say("%s %s TIMED OUT (%s): the arm's other requests are skipped" % (c, sp, rec["timed_out"]))
                                r.update(prompt=pk, i=i)
                                rec["requests"].append(r)
                                t = r.get("timings", {})
                                say("%s %s %s %d: %s" % (c, sp, pk, i, {k: t.get(k) for k in ("predicted_per_token_ms", "draft_n",
                                                                                            "draft_n_accepted")} if t else r.get("error")))
                    except Exception as ex:
                        rec["error"] = repr(ex)[:300]
                        say("%s %s ERROR %s" % (c, sp, rec["error"]))
                q = subprocess.run(["bash", os.path.join(HERE, "serve_pool.sh"), "stop"], cwd=ROOT, env=env, capture_output=True,
                                   text=True, timeout=600)
                open(os.path.join(d, "stop_%s_%s.log" % (c, sp)), "w").write(q.stdout + q.stderr)
                say("%s %s stop rc %d; after: %s" % (c, sp, q.returncode, X.pi_state(cfg["PLACE"])))
                if slog:
                    shutil.copy2(slog, os.path.join(d, "server_%s_%s.log" % (c, sp)))
                res.append(rec)
                json.dump(res, open(os.path.join(d, "results.json"), "w"), indent=1)
        finally:
            if guarded:
                say("guard: %s" % X.guard_stop().replace("\n", "; "))
    say("RUN DONE")
    s = subprocess.run([sys.executable, os.path.join(ROOT, reg["scorer"]), d, "--registration", a.registration], cwd=ROOT,
                       capture_output=True, text=True)
    say("scorer rc %d\n%s%s" % (s.returncode, s.stdout, s.stderr))


if __name__ == "__main__":
    main()
