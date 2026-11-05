#!/usr/bin/env python3
"""Switch the boards' strip outlets off and on, cleanly: every board the plan can leave out (2026-11-01, the scheduler review's
item 7; the user: "do item 1 to 4 then 6-7"). strip_switch_20261024.py switched only the Pi 4 (so4, so5); this one may switch
the Pi 4, the Pi 5 and the Jetson, never an outlet strip.json lists under do_not_switch_off (the laptop's). shelly_strip.py
stays read-only by construction.

    python3 testbed/scripts/strip_switch_20261101.py status pi4 [pi5 jetson]
    python3 testbed/scripts/strip_switch_20261101.py off-node pi5 [jetson ...] [--log F]
    python3 testbed/scripts/strip_switch_20261101.py on-node pi5 [jetson ...] [--log F]
    python3 testbed/scripts/strip_switch_20261101.py guard jetson [--log F]

status:   one JSON line per board: its outlet's output and apower.
off-node: per board, in parallel: its boot id is read, it is shut down over ssh -- the Pis `sudo -n systemctl poweroff`, the
          Jetson `sudo -n systemctl halt` (a clean poweroff made this Nano boot straight back up 20 s later, 2026-09-26;
          shutdown_all.sh halts it too) -- the tool waits until none of its addresses has answered a ping for 15 s and 10 s
          more, and only then switches its outlet off and reads it back. A board that does not go down is left powered.
on-node:  per board, in parallel: the outlet is switched on and read back; the tool waits for ssh (up to 300 s), then for the
          board's clock to be synchronized and its address to be up -- a Pi's overlay address, the Jetson's router-LAN one
          (it has no overlay) -- up to 300 s more. The Jetson then gets its power guard (guard, below). Exit 0 only when every
          board holds all of that.
guard:    the Jetson's power guard, by its NOTE's procedure (as spec4_session_20261023.sh and jetson_proc_20261028.py start it):
          if no guard process runs, its clock must be synchronized and its copy of jetson_power_guard.sh must match the
          repo's, then it is started; then the power mode must be MAXN, the guard alive with no TRIP since it started, and
          its log fresh. serve_pool.sh's Jetson gate checks the same; a rebooted Jetson has no guard until this starts it.
Each action appends one JSON line per board to --log. Refuses: a board not in ALLOWED; an outlet under do_not_switch_off; the
device at strip.json's host not being the strip it names (MAC); the outlet's name on the strip not being the board's."""
import argparse, hashlib, json, os, subprocess, sys, threading, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CFG = os.path.join(ROOT, "testbed", "energy", "strip.json")
ALLOWED = {"pi4": {"lan": "192.168.1.110", "overlay": "192.168.10.12", "cmd": "poweroff", "guard": False},
           "pi5": {"lan": "192.168.1.122", "overlay": "192.168.10.11", "cmd": "poweroff", "guard": False},
           "jetson": {"lan": "192.168.1.114", "overlay": None, "cmd": "halt", "guard": True}}
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "LogLevel=ERROR"]
GUARD = "jetson_power_guard.sh"
_lock = threading.Lock()


def rpc(host, method, **params):
    url = "http://%s/rpc/%s%s" % (host, method, ("?" + urllib.parse.urlencode(params)) if params else "")
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.load(r)


def outlet(board):
    cfg = json.load(open(CFG))
    if board not in ALLOWED:
        raise SystemExit("refused: %s is not a board this tool may switch (%s)" % (board, sorted(ALLOWED)))
    ids = [k for k, v in cfg["outlets"].items() if v == board]
    if len(ids) != 1:
        raise SystemExit("refused: strip.json maps %s to %s outlets" % (board, len(ids)))
    i = ids[0]
    if i in (cfg.get("do_not_switch_off") or {}):
        raise SystemExit("refused: outlet %s is under do_not_switch_off: %s" % (i, cfg["do_not_switch_off"][i]))
    host = cfg["host"]
    info = rpc(host, "Shelly.GetDeviceInfo")
    if str(info.get("mac", "")).upper().replace(":", "") != cfg["mac"].upper():
        raise SystemExit("refused: the device at %s has MAC %s, strip.json names %s" % (host, info.get("mac"), cfg["mac"]))
    name = (rpc(host, "Switch.GetConfig", id=int(i)) or {}).get("name")
    if name != board:
        raise SystemExit("refused: outlet %s is named %r on the strip, not %r" % (i, name, board))
    return host, int(i)


def status(host, i):
    s = rpc(host, "Switch.GetStatus", id=i)
    return {"output": s.get("output"), "apower": s.get("apower"), "source": s.get("source")}


def ping(ip):
    return bool(ip) and subprocess.run(["ping", "-c", "1", "-W", "1", ip], capture_output=True).returncode == 0


def ssh(board, cmd, timeout=15):
    try:
        p = subprocess.run(SSH + [board, cmd], capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip()
    except subprocess.TimeoutExpired:
        return 124, ""


def log_line(path, rec):
    rec = dict(rec, epoch=time.time(), at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with _lock:
        print(json.dumps(rec), flush=True)
        if path:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")


def address(a):
    """The address a board's network is up on once it is back: a Pi's overlay address, else (the Jetson) its router-LAN one."""
    return a["overlay"] or a["lan"]


def off_node(board, path):
    host, i = outlet(board)
    a = ALLOWED[board]
    st = status(host, i)
    if st["output"] is False:
        down = not ping(a["lan"]) and not ping(a["overlay"])
        log_line(path, {"action": "off-node", "board": board, "outlet": i, "result": "already off", "board_down": down, "status": st})
        return 0 if down else 1
    rc, boot = ssh(board, "cat /proc/sys/kernel/random/boot_id")
    if rc != 0 or not boot:
        log_line(path, {"action": "off-node", "board": board, "outlet": i, "result": "refused: the board does not answer ssh, so "
                        "it cannot be shut down cleanly; the outlet is left on", "status": st})
        return 1
    ssh(board, "sudo -n systemctl %s" % a["cmd"], timeout=20)            # the connection drops as it goes down
    t0, quiet_since = time.time(), None
    while time.time() - t0 < 180:
        up = ping(a["lan"]) or ping(a["overlay"])
        quiet_since = None if up else (quiet_since or time.time())
        if quiet_since and time.time() - quiet_since >= 15:
            break
        time.sleep(1)
    else:
        log_line(path, {"action": "off-node", "board": board, "outlet": i, "boot_id": boot, "cmd": a["cmd"],
                        "result": "the board still answers 180 s after its %s; the outlet is left on" % a["cmd"],
                        "status": status(host, i)})
        return 1
    time.sleep(10)                                                       # the board's own power-down, after its network stops
    before = status(host, i)
    rpc(host, "Switch.Set", id=i, on="false")
    time.sleep(1)
    after = status(host, i)
    ok = after["output"] is False
    log_line(path, {"action": "off-node", "board": board, "outlet": i, "boot_id": boot, "cmd": a["cmd"],
                    "down_after_s": round(time.time() - t0, 1), "apower_before_switch_w": before["apower"], "status": after,
                    "result": "outlet off" if ok else "the outlet did not read back off"})
    return 0 if ok else 1


def guard(board, path, repo_sha=None):
    """The Jetson's power guard running, MAXN, un-tripped and its log fresh (0), else 1; started here if none runs."""
    a = ALLOWED.get(board)
    if not a or not a["guard"]:
        raise SystemExit("refused: %s has no power guard" % board)
    repo_sha = repo_sha or hashlib.sha256(open(os.path.join(HERE, GUARD), "rb").read()).hexdigest()
    started = False
    rc, n = ssh(board, 'grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l')
    if rc != 0:
        log_line(path, {"action": "guard", "board": board, "result": "the board does not answer ssh"})
        return 1
    if n.strip() == "0":
        rc, c = ssh(board, 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum ~/heteroserve/tools/%s | cut -d" " -f1' % GUARD)
        if c.split() != ["synced", repo_sha]:
            log_line(path, {"action": "guard", "board": board, "result": "precheck failed (clock synchronized and the guard "
                            "script's sha256 against the repo's): %r" % c.split()})
            return 1
        ssh(board, "sudo -n setsid nohup bash ~/heteroserve/tools/%s </dev/null >/dev/null 2>&1 & echo started" % GUARD)
        started = True
        time.sleep(8)
    rc, g = ssh(board, 'm=$(cut -d" " -f1 /var/lib/nvpmodel/status); G=~/heteroserve/logs/power-guard.log; '
                       'p=$(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l); '
                       't=$(awk "/guard start/ {t=0} /TRIP/ {t++} END {print t+0}" $G); a=$(( $(date +%s) - $(stat -c %Y $G) )); '
                       'echo "$m $p $t $a"')
    f = g.split()
    ok = len(f) == 4 and f[0] == "pmode:0000" and int(f[1]) >= 1 and f[2] == "0" and int(f[3]) <= 5
    log_line(path, {"action": "guard", "board": board, "started": started, "pmode": f[0] if f else None,
                    "guard_processes": f[1] if len(f) > 1 else None, "trips_since_start": f[2] if len(f) > 2 else None,
                    "log_age_s": f[3] if len(f) > 3 else None, "result": "ok" if ok else "NOT OK"})
    return 0 if ok else 1


def on_node(board, path):
    host, i = outlet(board)
    a = ALLOWED[board]
    st = status(host, i)
    if st["output"] is True and ssh(board, "true")[0] == 0:
        log_line(path, {"action": "on-node", "board": board, "outlet": i, "result": "already on and answering", "status": st})
        return guard(board, path) if a["guard"] else 0
    rpc(host, "Switch.Set", id=i, on="true")
    time.sleep(1)
    st = status(host, i)
    t0 = time.time()
    boot = None
    while time.time() - t0 < 300:
        rc, out = ssh(board, "cat /proc/sys/kernel/random/boot_id", timeout=10)
        if rc == 0 and out:
            boot = out
            break
        time.sleep(5)
    t_ssh = round(time.time() - t0, 1) if boot else None
    sync = up = False
    t1 = time.time()
    while boot and time.time() - t1 < 300:
        _, s = ssh(board, "timedatectl show -p NTPSynchronized --value 2>/dev/null || "
                          "(timedatectl | grep -q 'synchronized: yes' && echo yes)", timeout=10)
        _, ip = ssh(board, "ip -4 -o addr show", timeout=10)
        sync, up = s.strip().splitlines()[-1:] == ["yes"], (" %s/" % address(a)) in ip
        if sync and up:
            break
        time.sleep(5)
    ok = bool(st["output"] is True and boot and sync and up)
    log_line(path, {"action": "on-node", "board": board, "outlet": i, "status": st, "boot_id": boot, "ssh_after_s": t_ssh,
                    "synced_after_s": round(time.time() - t0, 1) if ok else None, "ntp_synchronized": sync,
                    "address_up": up, "result": "back" if ok else "NOT BACK"})
    if ok and a["guard"]:
        return guard(board, path)
    return 0 if ok else 1


def each(fn, boards, path):
    """fn on every board at once (each in its own thread); 0 only when every board's returned 0."""
    rcs, ts = {}, []
    for b in boards:
        t = threading.Thread(target=lambda b=b: rcs.__setitem__(b, _safe(fn, b, path)))
        t.start(); ts.append(t)
    for t in ts:
        t.join()
    return 0 if all(v == 0 for v in rcs.values()) else 1


def _safe(fn, b, path):
    try:
        return fn(b, path)
    except SystemExit as e:
        log_line(path, {"action": fn.__name__.replace("_", "-"), "board": b, "result": str(e)})
        return 1
    except Exception as e:                                               # a strip that does not answer, a board's ssh
        log_line(path, {"action": fn.__name__.replace("_", "-"), "board": b, "result": "error: %s: %s" % (type(e).__name__, e)})
        return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("status", "off-node", "on-node", "guard"))
    ap.add_argument("boards", nargs="+")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    bad = [b for b in a.boards if b not in ALLOWED]
    if bad:
        raise SystemExit("refused: %s not boards this tool may switch (%s)" % (bad, sorted(ALLOWED)))
    if a.action == "status":
        for b in a.boards:
            host, i = outlet(b)
            print(json.dumps(dict(status(host, i), outlet=i, board=b)))
        return 0
    fn = {"off-node": off_node, "on-node": on_node, "guard": guard}[a.action]
    return each(fn, a.boards, a.log)


if __name__ == "__main__":
    sys.exit(main())
