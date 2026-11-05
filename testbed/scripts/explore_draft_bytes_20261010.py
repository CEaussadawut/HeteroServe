#!/usr/bin/env python3
"""Exploratory (2026-10-10, after spec_shape's SH-SLOPE failed at c = 3.06): how many bytes cross the laptop draft's RPC
link per check, measured on the socket rather than modeled.

    python3 testbed/scripts/explore_draft_bytes_20261010.py

The registered model charged one hidden state up and one logits row down (612 KB) per check. The pinned server
re-decodes every verified batch on the draft (server-context.cpp:3741-3747, its own TODO TAG_SPEC_AVOID_DRAFT_REEVAL),
and draft-simple's process() leaves the batch's logits null, which outputs the last token (llama-batch.cpp:120-128). So
each check runs two draft decodes, each bringing back a logits row and each uploading its graph and inputs.

The run:
- the Jetson's guard started by its NOTE's procedure;
- serve_pool.sh start with SERVE_DRAFT=default (the laptop draft, the serving default's own path), ctx 1,024, step 3's
  plan; one warm-up;
- per request, the laptop RPC socket's byte counters (ss -tin on the laptop: bytes_acked, bytes_received) read before and
  after, beside the server's timings (draft_n, draft_n_accepted). Three requests: 64 tokens of step 3's e01 and c01
  prompts, then e01 again with 16 tokens, so the per-check cost separates from the per-request one;
- serve_pool.sh stop, and the guard stopped.
Nothing here is registered. Everything goes to testbed/results/draft_bytes_<stamp>/ (run.log tee'd, results.json).
"""
import datetime, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import spec_serve_20261010 as SS                            # noqa: E402

SERVE = os.path.join(HERE, "serve_pool.sh")
REG = json.load(open(os.path.join(ROOT, "testbed", "dataset", "spec_serve_predictions_20261010.json")))
LAPTOP_RPC = "192.168.1.108:50053"


def sockets():
    """[(peer, bytes_acked, bytes_received)] of the established connections on the laptop's draft RPC port."""
    p = subprocess.run(["ss", "-tinH", "state", "established", "( sport = :50053 )"], capture_output=True, text=True)
    out, peer = [], None
    for ln in p.stdout.splitlines():
        m = re.match(r"^\s*\d+\s+\d+\s+(\S+)\s+(\S+)", ln)
        if m:
            peer = m.group(2); continue
        a = re.search(r"bytes_acked:(\d+)", ln); r = re.search(r"bytes_received:(\d+)", ln)
        if peer and (a or r):
            out.append((peer, int(a.group(1)) if a else 0, int(r.group(1)) if r else 0)); peer = None
    return out


def totals():
    s = sockets()
    return sum(x[1] for x in s), sum(x[2] for x in s), len(s)


def main():
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", "draft_bytes_%s" % stamp)
    os.makedirs(d)
    log = open(os.path.join(d, "run.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); log.write(line + "\n"); log.flush()
    rc, g = SS.ssh("jetson", 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum '
                             '~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1')
    import hashlib
    want = hashlib.sha256(open(os.path.join(HERE, "jetson_power_guard.sh"), "rb").read()).hexdigest()
    if g.split() != ["synced", want]:
        say("FATAL: guard precheck %r" % g); sys.exit(1)
    rc, n = SS.ssh("jetson", SS.GUARD_PROCS + " | wc -l")
    started_guard = n.strip() == "0"
    if started_guard:
        SS.ssh("jetson", "sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo started")
        time.sleep(5)
    say("guard %s" % ("started" if started_guard else "already running"))
    res = {"requests": []}
    try:
        env = dict(os.environ, MODEL=REG["model"], CTX=str(REG["ctx"]), PLACE=REG["place"], PLACE_HEAD=REG["place_head"],
                   NODES="pi4 jetson", SERVE_DRAFT="default")
        r = subprocess.run(["bash", SERVE, "start"], env=env, capture_output=True, text=True, timeout=2400)
        open(os.path.join(d, "serve.console"), "w").write(r.stdout + r.stderr)
        if r.returncode or "READY" not in r.stdout:
            say("FATAL: serve_pool.sh start failed (exit %d)" % r.returncode); sys.exit(1)
        say("ready: %s" % " | ".join(l for l in r.stdout.splitlines() if l.startswith(("draft", "laptop", "mem", "split"))))
        res["serve_lines"] = [l for l in r.stdout.splitlines() if l.startswith(("draft", "laptop", "mem", "split", "llama-server"))]
        SS.post(REG["host_url"] + "/completion", dict(REG["sampling"], prompt=REG["chat_format"] % REG["warmup_prompt"], n_predict=16),
                timeout=3600)
        a0, r0, n0 = totals(); time.sleep(20); a1, r1, n1 = totals()
        res["idle_20s"] = {"sent": a1 - a0, "received": r1 - r0, "sockets": n1}
        say("idle 20 s: laptop sent %d B, received %d B (%d sockets)" % (a1 - a0, r1 - r0, n1))
        prompts = {p["id"]: p for p in REG["prompts"]}
        for pid, n_predict in (("e01", 64), ("c01", 64), ("e01", 16)):
            a0, r0, _ = totals(); t0 = time.time()
            resp = SS.post(REG["host_url"] + "/completion", dict(REG["sampling"], prompt=REG["chat_format"] % prompts[pid]["text"],
                                                                 n_predict=n_predict), timeout=3600)
            t1 = time.time(); a1, r1, _ = totals()
            tm = resp.get("timings") or {}
            row = {"id": pid, "n_predict": n_predict, "laptop_sent_bytes": a1 - a0, "laptop_received_bytes": r1 - r0,
                   "seconds": round(t1 - t0, 2), "timings": tm}
            res["requests"].append(row)
            say("%s %d: laptop sent %.1f KB, received %.1f KB; prompt_n %s, predicted_n %s, draft_n %s, accepted %s" % (
                pid, n_predict, (a1 - a0) / 1024, (r1 - r0) / 1024, tm.get("prompt_n"), tm.get("predicted_n"),
                tm.get("draft_n"), tm.get("draft_n_accepted")))
    finally:
        r = subprocess.run(["bash", SERVE, "stop"], capture_output=True, text=True, timeout=600)
        open(os.path.join(d, "serve.console"), "a").write(r.stdout + r.stderr)
        say("stopped: %s" % " | ".join(l for l in r.stdout.splitlines() if "stop" in l))
        if started_guard:
            rc, o = SS.ssh("jetson", 'for p in $(%s | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p '
                                     'stopped"; done; read -t 2 <> <(:); echo "guard procs: $(%s | wc -l)"' % (SS.GUARD_PROCS, SS.GUARD_PROCS))
            say(o.strip().replace("\n", "; "))
        json.dump(res, open(os.path.join(d, "results.json"), "w"), indent=1)
        say("DONE %s" % os.path.relpath(d, ROOT))


if __name__ == "__main__":
    main()
