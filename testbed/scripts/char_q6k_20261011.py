#!/usr/bin/env python3
"""The Q6_K characterisation rows on the boards (2026-10-11): TinyLlama Q6_K on the Pi 4 (CPU) and the Jetson (Vulkan), the
rows the planner needs to price Qwen3-30B-A3B at Q6_K (the user: "Qwen3-30B-A3B at Q6_K", a model that needs the pool even
with the laptop as a node). The laptop's own rows are taken by run_t580_rows.sh and laptop_worker_quants_20261011.py.

    python3 testbed/scripts/char_q6k_20261011.py        # run under systemd-run --user ... systemd-inhibit

- run_quant_campaign.sh, unchanged but for its Q6_K entries: the file copied from the laptop's cache and re-hashed on each
  board, every tree at the pin, -r 3 on the Pi 4 with get_throttled and temperatures around it, -r 5 on the Jetson under its
  guard with its clocks locked by its own bench script (the characterisation protocol, pp512 / tg128, both boards at once,
  each alone on its board);
- then the Pi 4 again at the pooled protocol (PP=64 TG=16), as every other row has both;
- the Jetson's guard started by its NOTE's procedure (as longctx_20261011.py), stopped at the end if this started it, and its
  state checked equal to as found.
"""
import datetime, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import spec_serve_20261010 as SS                            # noqa: E402  ssh, sha, GUARD_PROCS
import power_modes_char_20261010 as PC                      # noqa: E402  STATE_CMD, parse_state

STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = os.path.join(ROOT, "testbed", "results", "char_q6k_%s" % STAMP)
MODEL = "tinyllama-1.1b-q6_k.gguf"
STEPS = 'step tinyllama-q6_k MODEL=$HOME/models/tinyllama-1.1b-q6_k.gguf CASES=vulkan'


def say(msg):
    line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), msg)
    print(line, flush=True)
    open(os.path.join(OUT, "progress.log"), "a").write(line + "\n")


def campaign(env):
    p = subprocess.run(["bash", os.path.join(HERE, "run_quant_campaign.sh")], cwd=ROOT, env=dict(os.environ, **env),
                       capture_output=True, text=True)
    out = p.stdout + p.stderr
    d = next((ln.split("-> ")[1].split()[0] for ln in out.splitlines() if "quant campaign -> " in ln), None)
    return p.returncode, d, out


def main():
    os.makedirs(OUT)
    rc, asfound = SS.ssh("jetson", PC.STATE_CMD)
    asfound = asfound.strip()
    k = PC.parse_state(asfound)
    if not (k["pmode"] == "pmode:0000" and k["online"] == "0-3" and k["railgate"] == "1" and k["emcov"] == "0"):
        say("FATAL: the Jetson is not as found: %s" % asfound); sys.exit(1)
    rc, g = SS.ssh("jetson", 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum '
                             '~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1')
    if g.split() != ["synced", SS.sha(os.path.join(HERE, "jetson_power_guard.sh"))]:
        say("FATAL: guard precheck %r" % g); sys.exit(1)
    for n in ("pi4", "jetson"):
        rc, b = SS.ssh(n, "for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p; done | wc -l")
        if b.strip() != "0":
            say("FATAL: %s is busy" % n); sys.exit(1)
    say("prechecks passed: the Jetson as found (%s), guard script synced, boards idle" % asfound)
    started = False
    t_start = time.time()
    try:
        rc, n = SS.ssh("jetson", SS.GUARD_PROCS + " | wc -l")
        if n.strip() == "0":
            SS.ssh("jetson", "sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo started")
            started = True
            time.sleep(5)
        say("guard %s" % ("started" if started else "already running"))
        rc, d, out = campaign({"MODELS": MODEL, "NODES": "pi4 jetson", "JETSON_STEPS": STEPS})
        open(os.path.join(OUT, "campaign_char.out"), "w").write(out)
        say("characterisation protocol: exit %d, %s; %s" % (rc, d, " | ".join(l for l in out.splitlines()
                                                                             if l.startswith(("summary", "pi4     done", "guard   trips", "CAMPAIGN")))))
        rc2, d2, out2 = campaign({"MODELS": MODEL, "NODES": "pi4", "PP": "64", "TG": "16"})
        open(os.path.join(OUT, "campaign_pool.out"), "w").write(out2)
        say("pooled protocol (Pi 4): exit %d, %s; %s" % (rc2, d2, " | ".join(l for l in out2.splitlines()
                                                                             if l.startswith(("pi4     done", "CAMPAIGN")))))
    finally:
        rc, o = SS.ssh("jetson", "tail -n %d ~/heteroserve/logs/power-guard.log" % (int(time.time() - t_start) + 120))
        open(os.path.join(OUT, "power_guard_tail.log"), "w").write(o)
        say("guard: %d TRIP lines in the run's window" % sum(1 for l in o.splitlines() if "TRIP" in l))
        if started:
            rc, o = SS.ssh("jetson", 'for p in $(%s | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p '
                                     'stopped"; done; read -t 2 <> <(:); echo "guard procs: $(%s | wc -l)"' % (SS.GUARD_PROCS, SS.GUARD_PROCS))
            say(o.strip().replace("\n", "; "))
        rc, now = SS.ssh("jetson", PC.STATE_CMD)
        say("jetson now: %s -- %s to as found" % (now.strip(), "EQUAL" if now.strip() == asfound else "NOT EQUAL"))
        say("RUN DONE %s" % os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
