#!/usr/bin/env python3
"""Does turning the Pi 5's hardware TSO off end its network stall? (2026-11-03; the owner ran `sudo ethtool -K eth0 tso off`
on the Pi 5 at about 22:00, the fix testbed/results/pi5_tx_stall_20261103/NOTE.md proposed.) Not a registered test: a check
of a fix, against the stall's own counters.

    python3 testbed/scripts/check_pi5_tso_20261103.py [--rounds 3]

Before the fix, the 9B on laptop 24 + Pi 5 9 (head on the laptop) with MTP 2 hung on its first request after the warm-up,
every time it was tried (sp10's P1, the 19:35 reproduction). Here that serve runs `rounds` times (SERVE_MTP_PI5=1, past
serve_pool.sh's guard). Each round: the Pi 5's TSO state and its counters (TcpExtTCPSpuriousRtxHostQueues, TCPTimeouts,
TcpRetransSegs, TCPAbortOnTimeout) before; serve_pool.sh start; a warm-up; the scan of tiny prompts that hung sp10d's P5
(m 1, 2, 3, 4, 6, 8, two each); the exploration's free and copy prompts, two of each, 128 tokens; serve_pool.sh stop; the
counters after. Every request has a 120 s timeout. A round holds if every request answers and the "still in the host queue"
counter does not move. Raw output: testbed/results/pi5_tso_check_<stamp>/ (run.log, start/stop logs, results.json).
"""
import argparse, datetime, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import explore_spec_20261102 as X                           # noqa: E402

ENV = {"MODEL": "qwen35_9b", "CTX": "4096", "PLACE": "laptop:24,pi5:9", "PLACE_HEAD": "host", "SERVE_PLANNER": "v4",
       "SERVE_DRAFT": "none", "SERVE_SWITCH": "off", "HETEROSERVE_RESERVE_MB": "", "SERVE_MTP_PI5": "1",
       "SERVE_EXTRA": X.SPEC["mtp2"]}
COUNTERS = ("TcpExtTCPSpuriousRtxHostQueues", "TcpExtTCPTimeouts", "TcpRetransSegs", "TcpExtTCPAbortOnTimeout")
TMO = 120


def pi5():
    out = subprocess.run(X.SSH + ["pi5", "/usr/sbin/ethtool -k eth0 | grep -E '^tcp-segmentation-offload:'; nstat -az %s"
                                  % " ".join(COUNTERS)], capture_output=True, text=True, timeout=60).stdout
    c = {}
    for ln in out.splitlines():
        p = ln.split()
        if len(p) >= 2 and p[0] in COUNTERS:
            c[p[0]] = int(p[1])
        elif ln.startswith("tcp-segmentation-offload:"):
            c["tso"] = ln.split(":")[1].strip()
    return c


def ask(prompt, n):
    t0 = time.time()
    try:
        r = X.post("/completion", {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 1,
                                   "cache_prompt": False, "ignore_eos": True}, timeout=TMO)
        t = r.get("timings", {})
        return {"ok": True, "wall_s": round(time.time() - t0, 2), "predicted_per_token_ms": t.get("predicted_per_token_ms"),
                "prompt_ms": t.get("prompt_ms"), "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted")}
    except Exception as ex:
        return {"ok": False, "wall_s": round(time.time() - t0, 2), "error": repr(ex)[:160]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    if os.path.exists(X.STATE):
        raise SystemExit("refused: serve_pool.sh state exists")
    d = os.path.join(ROOT, "testbed", "results", "pi5_tso_check_%s" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(d)
    log = open(os.path.join(d, "run.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); log.write(line + "\n"); log.flush()
    say("########## pi5_tso_check: %d rounds of %s with %s" % (a.rounds, ENV["PLACE"], ENV["SERVE_EXTRA"]))
    rounds = []
    for i in range(1, a.rounds + 1):
        before = pi5()
        say("round %d before: %s" % (i, before))
        env = dict(os.environ, **ENV); env.pop("MEMFILE", None)
        p = subprocess.run(["bash", os.path.join(HERE, "serve_pool.sh"), "start"], cwd=ROOT, env=env, capture_output=True,
                           text=True, timeout=1800)
        open(os.path.join(d, "start_%d.log" % i), "w").write(p.stdout + p.stderr)
        say("round %d start rc %d" % (i, p.returncode))
        reqs = []
        if p.returncode == 0:
            reqs.append(dict(ask("Hello.", 8), what="warm-up"))
            ids = X.post("/tokenize", {"content": X.PROMPTS["copy"] + " " + X.PROMPTS["free"]})["tokens"]
            for m in (1, 2, 3, 4, 6, 8):
                for k in range(2):
                    reqs.append(dict(ask(ids[:m], 1), what="scan m %d" % m))
            for pk in ("free", "copy"):
                for k in range(2):
                    reqs.append(dict(ask(X.PROMPTS[pk], 128), what="%s %d" % (pk, k)))
                    say("round %d %s %d: %s" % (i, pk, k, {x: reqs[-1].get(x) for x in ("ok", "wall_s", "predicted_per_token_ms",
                                                                                          "draft_n", "draft_n_accepted", "error")}))
        q = subprocess.run(["bash", os.path.join(HERE, "serve_pool.sh"), "stop"], cwd=ROOT, env=env, capture_output=True,
                           text=True, timeout=600)
        open(os.path.join(d, "stop_%d.log" % i), "w").write(q.stdout + q.stderr)
        after = pi5()
        moved = {k: after.get(k, 0) - before.get(k, 0) for k in COUNTERS}
        ok = p.returncode == 0 and reqs and all(r["ok"] for r in reqs) and moved["TcpExtTCPSpuriousRtxHostQueues"] == 0
        say("round %d stop rc %d; after: %s; moved %s; %s" % (i, q.returncode, after, moved, "HOLDS" if ok else "FAILS"))
        rounds.append({"round": i, "start_rc": p.returncode, "before": before, "after": after, "moved": moved, "requests": reqs,
                       "holds": ok})
        json.dump(rounds, open(os.path.join(d, "results.json"), "w"), indent=1)
    say("DONE: %d of %d rounds hold" % (sum(r["holds"] for r in rounds), len(rounds)))


if __name__ == "__main__":
    main()
