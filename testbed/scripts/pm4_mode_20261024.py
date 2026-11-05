#!/usr/bin/env python3
"""The Jetson's power mode for the four-node power-mode study (register_pm4_20261024.py), run from the laptop.

    python3 testbed/scripts/pm4_mode_20261024.py store                  # once, before anything: the as-found clocks
    python3 testbed/scripts/pm4_mode_20261024.py set MODE               # locked | unlocked | 5w, switched and verified
    python3 testbed/scripts/pm4_mode_20261024.py set-when-served MODE --log PATH [--timeout S]
    python3 testbed/scripts/pm4_mode_20261024.py restore                # back to as found (MAXN, the stored clocks), verified

The switch commands, the sysfs state line and its checks are power_modes_char_20261010.py's (mode_cmd, STATE_CMD,
mode_ok), unchanged: every switch goes through the as-found state (MAXN, then the stored clocks), then into the mode.
store refuses unless the Jetson is as found (MAXN, clocks free: CPU and GPU min below max) and idle of llama.cpp; it keeps
the state line in ~/.cache/heteroserve-wf/pm4_asfound.json and the clocks on the Jetson in
~/heteroserve/run/pm4_asfound.conf (jetson_clocks --store).
set-when-served exists because serve_pool.sh starts only in MAXN (its GATE-GUARD): it waits until the laptop's llama-server
answers /health "ok", switches, verifies, and appends one JSON line to --log with the epoch of the verified switch, which
the study's scorer compares with the first scored request. It writes its pid to <log>.pid while it waits.
Exit status: 0 when the mode is verified, 1 otherwise.
"""
import argparse, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import power_modes_char_20261010 as PMC                    # noqa: E402  mode_cmd, STATE_CMD, mode_ok, parse_state
import spec_serve_20261010 as SS                            # noqa: E402  ssh

ASFOUND = os.path.expanduser("~/.cache/heteroserve-wf/pm4_asfound.json")
STORED = "~/heteroserve/run/pm4_asfound.conf"
HEALTH = "http://192.168.1.108:8080/health"                 # serve_pool.sh's llama-server on the laptop host


def state():
    rc, out = SS.ssh("jetson", PMC.STATE_CMD, timeout=60)
    return out.strip() if rc == 0 else None


def store():
    rc, busy = SS.ssh("jetson", "for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p; done | wc -l", 60)
    s = state()
    if rc or busy.strip() != "0" or not s:
        print("refused: the Jetson is busy (%s) or unreadable" % busy.strip()); return 1
    k = PMC.parse_state(s)
    cpu, gpu = k["cpu"].split("-"), k["gpu"].split("-")
    if k["pmode"] != "pmode:0000" or cpu[0] == cpu[1] or gpu[0] == gpu[1]:
        print("refused: not as found (MAXN with free clocks): %s" % s); return 1
    rc, out = SS.ssh("jetson", "mkdir -p ~/heteroserve/run && sudo -n jetson_clocks --store %s && echo stored" % STORED, 60)
    if rc or "stored" not in out:
        print("refused: jetson_clocks --store failed: %s" % out); return 1
    os.makedirs(os.path.dirname(ASFOUND), exist_ok=True)
    json.dump({"state": s, "stored": STORED, "at": time.time()}, open(ASFOUND, "w"))
    print("as found: %s" % s); return 0


def set_mode(mode):
    a = json.load(open(ASFOUND))
    ok, s, said = PMC.set_mode(mode, a["stored"], a["state"])
    print("%s %s: %s%s" % ("OK" if ok else "NOT VERIFIED", mode, s, (" (%s)" % said.replace("\n", " | ")) if said else ""))
    return ok, s


def when_served(mode, log, timeout):
    open(log + ".pid", "w").write(str(os.getpid()))
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(HEALTH, timeout=5) as r:
                if '"ok"' in r.read().decode():
                    break
        except Exception:
            pass
        time.sleep(1)
    else:
        open(log, "a").write(json.dumps({"mode": mode, "ok": False, "why": "no healthy server in %d s" % timeout}) + "\n")
        return 1
    ok, s = set_mode(mode)
    open(log, "a").write(json.dumps({"mode": mode, "ok": ok, "verified_epoch": time.time(), "state": s}) + "\n")
    os.remove(log + ".pid")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("store", "set", "set-when-served", "restore"))
    ap.add_argument("mode", nargs="?", choices=("locked", "unlocked", "5w"))
    ap.add_argument("--log"); ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    if a.cmd == "store":
        return store()
    if a.cmd == "restore":
        return 0 if set_mode("unlocked")[0] else 1
    if not a.mode:
        raise SystemExit("give the mode")
    if a.cmd == "set":
        return 0 if set_mode(a.mode)[0] else 1
    if not a.log:
        raise SystemExit("set-when-served needs --log")
    return when_served(a.mode, a.log, a.timeout)


if __name__ == "__main__":
    sys.exit(main())
