#!/usr/bin/env python3
"""Exploratory (2026-10-10, the user: "try the backend sampling option then"): does --spec-draft-backend-sampling cut the
laptop draft's link traffic?

    python3 testbed/scripts/explore_draft_bs_20261010.py

The pinned source says no for our draft type. llama_set_sampler, which moves a draft's sampling onto its backend, is
called only by the EAGLE-3, DFlash and MTP implementations (common/speculative.cpp:499, 1031, 1406), never by
draft-simple (179-425). The option also defaults to on (common/common.h:332). So every run so far already had it, and
still brought back full logits rows.

The test: two sessions, identical but for the flag, explicitly on and then explicitly off. Each session:
- the laptop's draft RPC server started here, then serve_pool.sh start with step 3's laptop arm plus the flag
  (ctx 1,024, step 3's plan), and one warm-up;
- one 64-token request of step 3's e01 prompt;
- the laptop RPC socket's byte counters read before and after it, beside the server's timings and its trace lines about
  the draft (the implementation it added, and any backend-sampling message).
If the flag does nothing for draft-simple, both sessions move the same bytes per draft decode. Nothing here is registered.
Everything goes to testbed/results/draft_bs_<stamp>/.
"""
import datetime, hashlib, json, os, re, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import spec_serve_20261010 as SS                            # noqa: E402
import explore_draft_bytes_20261010 as XB                   # noqa: E402  totals()

SERVE = os.path.join(HERE, "serve_pool.sh")
REG = json.load(open(os.path.join(ROOT, "testbed", "dataset", "spec_serve_predictions_20261010.json")))
ARM = REG["arms"]["lapd1"]["env"]
FLAGS = [("on", "--spec-draft-backend-sampling"), ("off", "--no-spec-draft-backend-sampling")]


def main():
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", "draft_bs_%s" % stamp)
    os.makedirs(d)
    log = open(os.path.join(d, "run.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); log.write(line + "\n"); log.flush()
    rc, g = SS.ssh("jetson", 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum '
                             '~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1')
    want = hashlib.sha256(open(os.path.join(HERE, "jetson_power_guard.sh"), "rb").read()).hexdigest()
    if g.split() != ["synced", want]:
        say("FATAL: guard precheck %r" % g); sys.exit(1)
    rc, n = SS.ssh("jetson", SS.GUARD_PROCS + " | wc -l")
    started_guard = n.strip() == "0"
    if started_guard:
        SS.ssh("jetson", "sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo started")
        time.sleep(5)
    say("guard %s" % ("started" if started_guard else "already running"))
    res, lap = {"sessions": []}, None
    prompt = next(p for p in REG["prompts"] if p["id"] == "e01")
    try:
        for name, flag in FLAGS:
            ip, port = ARM["SERVE_RPC_EXTRA"].split(":")
            lap = subprocess.Popen([REG["laptop_rpc_server"], "-H", ip, "-p", port, "-t", "4"],
                                   stdout=open(os.path.join(d, "rpc_laptop_%s.log" % name), "w"), stderr=subprocess.STDOUT,
                                   start_new_session=True)
            for _ in range(60):
                if SS.listening(ARM["SERVE_RPC_EXTRA"]):
                    break
                time.sleep(0.5)
            env = dict(os.environ, MODEL=REG["model"], CTX=str(REG["ctx"]), PLACE=REG["place"], PLACE_HEAD=REG["place_head"],
                       NODES="pi4 jetson", SERVE_DRAFT=ARM["SERVE_DRAFT"] + " " + flag, DRAFT_HOST_MIB=ARM["DRAFT_HOST_MIB"],
                       SERVE_RPC_EXTRA=ARM["SERVE_RPC_EXTRA"])
            r = subprocess.run(["bash", SERVE, "start"], env=env, capture_output=True, text=True, timeout=2400)
            open(os.path.join(d, "serve_%s.console" % name), "w").write(r.stdout + r.stderr)
            if r.returncode or "READY" not in r.stdout:
                say("FATAL: serve_pool.sh start failed for %s (exit %d)" % (name, r.returncode)); sys.exit(1)
            say("%s: ready (%s)" % (name, flag))
            SS.post(REG["host_url"] + "/completion", dict(REG["sampling"], prompt=REG["chat_format"] % REG["warmup_prompt"],
                                                           n_predict=16), timeout=3600)
            a0, r0, _ = XB.totals()
            resp = SS.post(REG["host_url"] + "/completion", dict(REG["sampling"], prompt=REG["chat_format"] % prompt["text"],
                                                                 n_predict=64), timeout=3600)
            a1, r1, _ = XB.totals()
            tm = resp.get("timings") or {}
            decodes = (tm["predicted_n"] - tm["draft_n_accepted"]) + tm["draft_n"] + 1
            rc, srv = SS.ssh("pi4", "ls -t ~/logs/llama-server_serve_*.log | head -1")
            subprocess.run(["scp", "-q", "pi4:%s" % srv.strip(), os.path.join(d, "server_%s.log" % name)])
            text = open(os.path.join(d, "server_%s.log" % name), errors="replace").read()
            trace = [l.strip()[:200] for l in text.splitlines() if re.search(r"speculative implementation|backend_sampling|backend offload|"
                                                                             r"set_sampler|backend sampl", l)]
            row = {"flag": flag, "laptop_sent_bytes": a1 - a0, "laptop_received_bytes": r1 - r0, "draft_decodes": decodes,
                   "sent_per_decode": round((a1 - a0) / decodes), "received_per_decode": round((r1 - r0) / decodes),
                   "ms_per_token": round(tm["predicted_ms"] / tm["predicted_n"], 1), "draft_n": tm.get("draft_n"),
                   "accepted": tm.get("draft_n_accepted"), "content": resp.get("content"), "trace": trace}
            res["sessions"].append(row)
            say("%s: sent %.1f KB (%.1f KB per draft decode), received %.1f KB (%.1f KB per decode); %.1f ms/token; draft %s, "
                "accepted %s" % (name, (a1 - a0) / 1024, row["sent_per_decode"] / 1024, (r1 - r0) / 1024, row["received_per_decode"] / 1024,
                                 row["ms_per_token"], tm.get("draft_n"), tm.get("draft_n_accepted")))
            say("%s trace: %s" % (name, " | ".join(trace) or "(none)"))
            r = subprocess.run(["bash", SERVE, "stop"], capture_output=True, text=True, timeout=600)
            say("%s stopped: %s" % (name, " | ".join(l for l in r.stdout.splitlines() if "stop" in l)))
            os.killpg(lap.pid, signal.SIGTERM); lap.wait(timeout=30); lap = None
        a, b = res["sessions"]
        res["same_output"] = a["content"] == b["content"]
        res["sent_per_decode_ratio_on_over_off"] = round(a["sent_per_decode"] / b["sent_per_decode"], 4)
        say("same output: %s; sent per decode on/off = %.4f" % (res["same_output"], res["sent_per_decode_ratio_on_over_off"]))
    finally:
        if os.path.exists(os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")):
            subprocess.run(["bash", SERVE, "stop"], capture_output=True, text=True, timeout=600)
        if lap:
            os.killpg(lap.pid, signal.SIGTERM)
        if started_guard:
            rc, o = SS.ssh("jetson", 'for p in $(%s | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p '
                                     'stopped"; done; read -t 2 <> <(:); echo "guard procs: $(%s | wc -l)"' % (SS.GUARD_PROCS, SS.GUARD_PROCS))
            say(o.strip().replace("\n", "; "))
        json.dump(res, open(os.path.join(d, "results.json"), "w"), indent=1, ensure_ascii=False)
        say("DONE %s" % os.path.relpath(d, ROOT))


if __name__ == "__main__":
    main()
