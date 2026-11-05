#!/usr/bin/env python3
"""Switch one board's strip outlet off and on, cleanly (the switch-off study so4, 2026-10-24; the user, 14:44: "do the
switch-off after this"). shelly_strip.py never switches an outlet (read-only by construction); this is the one tool that does,
and it can switch only a board in ALLOWED (the Pi 4), never an outlet strip.json lists under do_not_switch_off (the laptop's).

    python3 testbed/scripts/strip_switch_20261024.py status pi4
    python3 testbed/scripts/strip_switch_20261024.py off-node pi4 [--log F]
    python3 testbed/scripts/strip_switch_20261024.py on-node pi4 [--log F]

off-node: the board's boot id is read, it is powered off over ssh (`sudo -n systemctl poweroff`, as shutdown_all.sh), the
          tool waits until neither of its addresses has answered a ping for 15 s and 10 s more, and only then switches its
          outlet off and reads the outlet back. A board that does not go down is left powered (exit 1).
on-node:  the outlet is switched on and read back; the tool waits for ssh (up to 300 s), then for the board's clock to be
          synchronized and its overlay address to be up (up to 300 s more). Exit 0 only when all three hold.
Each action appends one JSON line to --log. Refuses: a board not in ALLOWED; an outlet under do_not_switch_off; the device at
strip.json's host not being the strip it names (MAC); the outlet's name on the strip not being the board's."""
import argparse, json, os, subprocess, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CFG = os.path.join(ROOT, "testbed", "energy", "strip.json")
ALLOWED = {"pi4": {"lan": "192.168.1.110", "overlay": "192.168.10.12", "cmd": "poweroff"}}
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "LogLevel=ERROR"]


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
    return subprocess.run(["ping", "-c", "1", "-W", "1", ip], capture_output=True).returncode == 0


def ssh(board, cmd, timeout=15):
    try:
        p = subprocess.run(SSH + [board, cmd], capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip()
    except subprocess.TimeoutExpired:
        return 124, ""


def log_line(path, rec):
    rec = dict(rec, epoch=time.time(), at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print(json.dumps(rec))
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


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
        log_line(path, {"action": "off-node", "board": board, "outlet": i, "boot_id": boot,
                        "result": "the board still answers 180 s after its poweroff; the outlet is left on", "status": status(host, i)})
        return 1
    time.sleep(10)                                                       # the board's own power-down, after its network stops
    before = status(host, i)
    rpc(host, "Switch.Set", id=i, on="false")
    time.sleep(1)
    after = status(host, i)
    ok = after["output"] is False
    log_line(path, {"action": "off-node", "board": board, "outlet": i, "boot_id": boot, "down_after_s": round(time.time() - t0, 1),
                    "apower_before_switch_w": before["apower"], "status": after,
                    "result": "outlet off" if ok else "the outlet did not read back off"})
    return 0 if ok else 1


def on_node(board, path):
    host, i = outlet(board)
    a = ALLOWED[board]
    st = status(host, i)
    if st["output"] is True and ssh(board, "true")[0] == 0:
        log_line(path, {"action": "on-node", "board": board, "outlet": i, "result": "already on and answering", "status": st})
        return 0
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
    sync = overlay = False
    t1 = time.time()
    while boot and time.time() - t1 < 300:
        _, s = ssh(board, "timedatectl show -p NTPSynchronized --value", timeout=10)
        _, ip = ssh(board, "ip -4 -o addr show", timeout=10)
        sync, overlay = s == "yes", a["overlay"] in ip
        if sync and overlay:
            break
        time.sleep(5)
    ok = bool(st["output"] is True and boot and sync and overlay)
    log_line(path, {"action": "on-node", "board": board, "outlet": i, "status": st, "boot_id": boot, "ssh_after_s": t_ssh,
                    "synced_after_s": round(time.time() - t0, 1) if ok else None, "ntp_synchronized": sync,
                    "overlay_up": overlay, "result": "back" if ok else "NOT BACK"})
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("status", "off-node", "on-node"))
    ap.add_argument("board")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    if a.action == "status":
        host, i = outlet(a.board)
        print(json.dumps(dict(status(host, i), outlet=i, board=a.board)))
        return 0
    return (off_node if a.action == "off-node" else on_node)(a.board, a.log)


if __name__ == "__main__":
    sys.exit(main())
