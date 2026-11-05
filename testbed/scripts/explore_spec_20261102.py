#!/usr/bin/env python3
"""Exploration for the scheduler review's item 10 (2026-11-02; the user: "... then item 9-10"): speculative decoding with no
draft model -- the model's own multi-token-prediction head (llama.cpp `--spec-type draft-mtp`, Qwen3.5/3.6's nextn block) and
n-gram drafting (`--spec-type ngram-mod`, the pin's `--spec-default` settings) -- and what a check costs on a plan. Not a
registered test: it sizes the effect and the checking cost before one is registered.

    python3 testbed/scripts/explore_spec_20261102.py [--only A,B] [--reqs 3] [--gen 128]

Each config is served by serve_pool.sh as a harness serves (a fixed PLACE, SERVE_PLANNER=v4 so the gate charges a hybrid's
memory as llama.cpp allocates it, SERVE_DRAFT=none, SERVE_SWITCH=off, the speculation flags through SERVE_EXTRA). Per serve:
  - with no speculation, first the checking cost: prompts of the first m tokens of one text (the server's own /tokenize),
    m = 1, 2, 3, 4, 6, 8, 16, 32 and 64, one token generated, three times each. The server's prompt_ms is one batch of m
    tokens through the plan: what a check of m - 1 drafted tokens does, but for m - 1 more rows of the output head;
  - then `reqs` greedy completions of `gen` tokens (ignore_eos) of two prompts, free-form (an explanation: little to copy)
    and copying (repeat a passage it is given: much to copy, n-gram drafting's case; n-gram drafts only after 24 matching
    tokens, so `gen` is 128). The server's timings give the time per decoded token and the drafted and accepted tokens.
Then serve_pool.sh stop.
  A  Qwen3.6-35B-A3B Q6_K at ctx 4,096 on the four-node plan the planner served in so6 (laptop 24 + Jetson 2 + Pi 5 9 +
     Pi 4 6, head on the laptop): none, MTP 1 / 2 / 3 drafted tokens, ngram-mod
  B  Qwen3.5-9B Q4_K_M at ctx 4,096 on the laptop alone: the same five
The Jetson's power guard runs while a config uses the Jetson, by its NOTE's procedure (as spec4_session_20261023.sh: the
clock synced and the guard's sha256 the repository's, started before the load, stopped after).
Raw output: testbed/results/explspec_<stamp>/ (run.log; start_ / stop_<config>_<spec>.log, serve_pool.sh's output;
server_<config>_<spec>.log, the llama-server log; results.json, every response's timings).
"""
import argparse, datetime, glob, hashlib, json, os, shutil, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
STATE = os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")
BASE = "http://192.168.1.108:8080"
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
PROMPTS = {
    "free": "Explain, in plain words, why a cluster of slow boards can run a language model that none of them can hold "
            "alone, and what limits how fast it answers.",
    "copy": "Here is a passage. The cluster has four nodes: a ThinkPad T580 laptop, a Raspberry Pi 5, a Raspberry Pi 4 and a "
            "Jetson Nano. Each node is metered at the wall by a Shelly power strip. The laptop decodes a block two to six "
            "times faster than any board, so the planner puts as many layers on it as its memory allows, and uses the boards "
            "only when the model does not fit. Repeat the passage above word for word, then summarise it in one sentence.",
}
SCAN = [1, 2, 3, 4, 6, 8, 16, 32, 64]
SPEC = {"none": "", "mtp1": "--spec-type draft-mtp --spec-draft-n-max 1", "mtp2": "--spec-type draft-mtp --spec-draft-n-max 2",
        "mtp3": "--spec-type draft-mtp --spec-draft-n-max 3", "ngram": "--spec-type ngram-mod"}
CONFIGS = {
    "A": {"MODEL": "qwen36moe_q6k", "CTX": "4096", "PLACE": "laptop:24,jetson:2,pi5:9,pi4:6", "PLACE_HEAD": "host",
          "specs": ["none", "mtp1", "mtp2", "mtp3", "ngram"]},
    "B": {"MODEL": "qwen35_9b", "CTX": "4096", "PLACE": "laptop:33", "PLACE_HEAD": "last",
          "specs": ["none", "mtp1", "mtp2", "mtp3", "ngram"]},
}


def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def complete(prompt, n):
    t0 = time.time()
    r = post("/completion", {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": False,
                             "ignore_eos": True})
    t = r.get("timings", {})
    return {"wall_s": round(time.time() - t0, 3), "timings": t, "tokens_predicted": r.get("tokens_predicted"),
            "tokens_evaluated": r.get("tokens_evaluated"), "text": r.get("content", "")[:240]}


def jetson(cmd):
    p = subprocess.run(SSH + ["jetson", cmd], capture_output=True, text=True, timeout=60)
    return p.stdout.strip()


def pi_state(place):
    """get_throttled and the temperature of each Pi in the plan (logged with every Pi run, the house rule)."""
    out = []
    for n in ("pi5", "pi4"):
        if n + ":" in place:
            p = subprocess.run(SSH + [n, "vcgencmd get_throttled; vcgencmd measure_temp"], capture_output=True, text=True, timeout=60)
            out.append("%s %s" % (n, " ".join(p.stdout.split())))
    return "; ".join(out) or "no Pi in the plan"


def guard_start():
    """As spec4_session_20261023.sh: the Jetson's clock synced and its guard's sha256 the repository's, then started."""
    want = hashlib.sha256(open(os.path.join(HERE, "jetson_power_guard.sh"), "rb").read()).hexdigest()
    got = jetson('timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum ~/heteroserve/tools/jetson_power_guard.sh '
                 '| cut -d" " -f1').split()
    if got != ["synced", want]:
        raise SystemExit("refused: Jetson guard precheck %s" % got)
    return jetson("sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo guard started")


def guard_stop():
    return jetson('for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do '
                  'sudo -n kill $p && echo "guard pid $p stopped"; done; echo "TRIP lines: $(grep -c TRIP ~/heteroserve/logs/power-guard.log)"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="A,B"); ap.add_argument("--reqs", type=int, default=3); ap.add_argument("--gen", type=int, default=128)
    a = ap.parse_args()
    if os.path.exists(STATE):
        raise SystemExit("refused: serve_pool.sh state exists")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", "explspec_%s" % stamp)
    os.makedirs(d)
    log = open(os.path.join(d, "run.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); log.write(line + "\n"); log.flush()
    say("########## explspec: speculative decoding with no draft model, and the checking cost (not registered); %s, %d x %d"
        % (a.only, a.reqs, a.gen))
    res = []
    for c in a.only.split(","):
        cfg = CONFIGS[c]
        guarded = "jetson:" in cfg["PLACE"]
        if guarded:
            say("guard: %s" % guard_start())
        try:
            for sp in cfg["specs"]:
                env = dict(os.environ, MODEL=cfg["MODEL"], CTX=cfg["CTX"], PLACE=cfg["PLACE"], PLACE_HEAD=cfg["PLACE_HEAD"],
                           SERVE_PLANNER="v4", SERVE_DRAFT="none", SERVE_SWITCH="off", HETEROSERVE_RESERVE_MB="", SERVE_EXTRA=SPEC[sp])
                env.pop("MEMFILE", None)
                say("%s %s before: %s" % (c, sp, pi_state(cfg["PLACE"])))
                t0 = time.time()
                p = subprocess.run(["bash", os.path.join(HERE, "serve_pool.sh"), "start"], cwd=ROOT, env=env, capture_output=True,
                                   text=True, timeout=3600)
                open(os.path.join(d, "start_%s_%s.log" % (c, sp)), "w").write(p.stdout + p.stderr)
                say("%s %s start rc %d in %.0f s" % (c, sp, p.returncode, time.time() - t0))
                logs = sorted(glob.glob(os.path.expanduser("~/logs/llama-server_serve_*.log")), key=os.path.getmtime)
                slog = logs[-1] if logs and os.path.getmtime(logs[-1]) >= t0 else None    # the host is this laptop in A and B
                rec = {"config": c, "spec": sp, "spec_args": SPEC[sp], "model": cfg["MODEL"], "place": cfg["PLACE"],
                       "head": cfg["PLACE_HEAD"], "start_rc": p.returncode, "server_log": slog, "scan": [], "requests": []}
                if p.returncode == 0:
                    try:
                        complete("Hello.", 8)                                 # a warm-up, not recorded
                        if sp == "none":
                            ids = post("/tokenize", {"content": PROMPTS["copy"] + " " + PROMPTS["free"]})["tokens"]
                            for m in SCAN:
                                for i in range(3):
                                    r = complete(ids[:m], 1)
                                    r.update(m=m, i=i)
                                    rec["scan"].append(r)
                                say("%s scan m %d: prompt_ms %s" % (c, m, [x["timings"].get("prompt_ms") for x in rec["scan"] if x["m"] == m]))
                        for pk, pr in PROMPTS.items():
                            for i in range(a.reqs):
                                try:
                                    r = complete(pr, a.gen)
                                except Exception as ex:
                                    r = {"error": repr(ex)[:200]}
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
                say("%s %s stop rc %d; after: %s" % (c, sp, q.returncode, pi_state(cfg["PLACE"])))
                if slog:      # the server's own log, raw: its per-request "draft acceptance ... mean len" lines count the checks
                    shutil.copy2(slog, os.path.join(d, "server_%s_%s.log" % (c, sp)))
                res.append(rec)
                json.dump(res, open(os.path.join(d, "results.json"), "w"), indent=1)
        finally:
            if guarded:
                say("guard: %s" % guard_stop().replace("\n", "; "))
    say("DONE %s" % os.path.relpath(d, ROOT))


if __name__ == "__main__":
    main()
