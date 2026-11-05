#!/usr/bin/env python3
"""Run sp11 (register_sp11_20261104.py; the user, 2026-11-04: "do all four follow-up studies"): the MTP 1 default across many
prompts, and why a plan's acceptance on free text moves. Then its scorer.

    python3 testbed/scripts/sp11_session_20261104.py --registration testbed/dataset/sp11_predictions_20261104.json

sp10_session_20261103.py's flow (serve_pool.sh per setting, the Jetson's guard while a plan holds the Jetson, get_throttled
and the temperature of each Pi around every serve, a warm-up, the server log copied in), with what sp11 needs:
- every prompt of the registration, one request each (greedy, gen tokens, ignore_eos), the FULL text and the token ids kept;
- on a config marked "nprobs", after its "none" requests: each prompt once more, NOT timed, with n_probs 2 and the
  probabilities taken before sampling (post_sampling_probs false: the softmax of the raw logits), so the scorer can read how
  close the top two tokens were at every position of the plain text;
- no batch scan (the checking-cost model's numbers are registered).
Refuses to start when a registered input's sha256 has changed, or when serve_pool.sh's state exists. A request that times out
ends its arm's requests (request_timeout_s). Raw output: testbed/results/sp11_<stamp>/ (run.log, start_ / stop_ /
server_<config>_<spec>.log, results.json), then score.json and score.txt from the registration's scorer.
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


def complete(prompt, n, timeout, nprobs=0):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": False,
            "ignore_eos": True, "return_tokens": True}
    if nprobs:
        body.update(n_probs=nprobs, post_sampling_probs=False)
    t0 = time.time()
    r = X.post("/completion", body, timeout=timeout)
    out = {"wall_s": round(time.time() - t0, 3), "timings": r.get("timings", {}), "tokens_predicted": r.get("tokens_predicted"),
           "tokens_evaluated": r.get("tokens_evaluated"), "text": r.get("content", ""), "tokens": r.get("tokens")}
    if nprobs:
        out["probs"] = [{"id": p.get("id"), "logprob": p.get("logprob"),
                         "top": [[t.get("id"), t.get("logprob")] for t in (p.get("top_logprobs") or [])]}
                        for p in (r.get("completion_probabilities") or [])]
    return out


def timed_out(ex):
    return isinstance(ex, (socket.timeout, TimeoutError)) or (isinstance(ex, urllib.error.URLError)
                                                               and isinstance(ex.reason, (socket.timeout, TimeoutError)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", required=True)
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    tmo = reg.get("request_timeout_s", 600)
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
                       "head": cfg["PLACE_HEAD"], "start_rc": p.returncode, "server_log": slog, "requests": [], "probs_pass": []}
                if p.returncode == 0:
                    try:
                        complete("Hello.", 8, tmo)                            # a warm-up, not recorded
                        for pk in reg["prompt_order"]:
                            if rec.get("timed_out"):
                                break
                            try:
                                r = complete(reg["prompts"][pk], reg["gen"], tmo)
                            except Exception as ex:
                                r = {"error": repr(ex)[:200]}
                                if timed_out(ex):
                                    rec["timed_out"] = "%s after %d s" % (pk, tmo)
                                    say("%s %s TIMED OUT (%s): the arm's other requests are skipped" % (c, sp, rec["timed_out"]))
                            r.update(prompt=pk)
                            rec["requests"].append(r)
                            t = r.get("timings", {})
                            say("%s %s %s: %s" % (c, sp, pk, {k: t.get(k) for k in ("predicted_per_token_ms", "draft_n",
                                                                                     "draft_n_accepted")} if t else r.get("error")))
                        if sp == "none" and cfg.get("nprobs") and not rec.get("timed_out"):
                            for pk in reg["prompt_order"]:
                                try:
                                    r = complete(reg["prompts"][pk], reg["gen"], tmo, nprobs=2)
                                except Exception as ex:
                                    r = {"error": repr(ex)[:200]}
                                r.update(prompt=pk)
                                rec["probs_pass"].append(r)
                                say("%s %s probs %s: %s" % (c, sp, pk, "%d positions" % len(r.get("probs") or []) if "probs" in r
                                                            else r.get("error")))
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
