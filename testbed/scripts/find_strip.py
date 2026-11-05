#!/usr/bin/env python3
"""Find the Shelly strip when its DHCP lease has moved (it is not reserved: .34 until 10-07, .33 since). Read-only towards
the strip: it pings, reads the neighbour table and calls Shelly.GetDeviceInfo through shelly_strip.py; it never switches.

    python3 testbed/scripts/find_strip.py [--fix]

Exit 0 when the strip answers at testbed/energy/strip.json's host. Otherwise it ping-sweeps every /24 the laptop is
directly connected to (the wired overlay 192.168.10.0/24 excepted; since 2026-10-08 that is the home router's 192.168.1.0/24),
looks for strip.json's MAC in the neighbour table, and confirms the device there with
`shelly_strip.py info --host IP --json` (no DEVICE_MISMATCH: the MAC and the device id must both match). With --fix it then
corrects strip.json's host (a text edit of that one value plus a dated note, so the file's layout is kept) and exits 0.
Exit 1 when the strip was not found or not confirmed.
"""
import datetime, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CFG = os.path.join(ROOT, "testbed", "energy", "strip.json")
STRIP = os.path.join(HERE, "shelly_strip.py")


def answers(host=None):
    cmd = [sys.executable, STRIP, "status"] + (["--host", host] if host else [])
    return subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0


def mac_colons(mac):
    m = re.sub(r"[^0-9a-f]", "", mac.lower())
    return ":".join(m[i:i + 2] for i in range(0, 12, 2))


def pools():
    """'a.b.c.' for every directly connected /24 but the wired overlay."""
    out = set()
    for ln in subprocess.run(["ip", "-4", "route", "show", "scope", "link"], capture_output=True, text=True).stdout.splitlines():
        m = re.match(r"^(\d+\.\d+\.\d+)\.0/24 ", ln)
        if m and m.group(1) != "192.168.10":
            out.add(m.group(1) + ".")
    return sorted(out)


def sweep(mac):
    for pre in pools():
        procs = [subprocess.Popen(["ping", "-c", "1", "-W", "1", "%s%d" % (pre, i)], stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL) for i in range(2, 255)]
        for p in procs:
            p.wait()
        neigh = subprocess.run(["ip", "neigh", "show"], capture_output=True, text=True).stdout
        for ln in neigh.splitlines():
            f = ln.split()
            if f and f[0].startswith(pre) and mac_colons(mac) in ln.lower() and "FAILED" not in ln:
                return f[0]
    return None


def confirm(ip):
    p = subprocess.run([sys.executable, STRIP, "info", "--host", ip, "--json"], capture_output=True, text=True, timeout=30)
    if p.returncode:
        return False, "info exit %d" % p.returncode
    checks = json.loads(p.stdout).get("checks", [])
    bad = [c for c in checks if "DEVICE_MISMATCH" in c]
    return (not bad), "; ".join(bad) or "MAC and device id match"


def fix(ip, how):
    s = open(CFG).read()
    old = json.loads(s)["host"]
    s2 = s.replace('"host": "%s"' % old, '"host": "%s"' % ip, 1)
    m = re.search(r'^( "_host_note": ".*)(",?)$', s2, re.M)
    note = (" %s: the lease had moved from %s to %s (found by its MAC after a ping sweep, testbed/scripts/find_strip.py; %s); "
            "this field was corrected. Still not reserved." % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), old, ip, how))
    if m:
        s2 = s2[:m.end(1)] + note + s2[m.end(1):]
    if json.loads(s2)["host"] != ip:
        raise SystemExit("refused: the host edit did not take")
    open(CFG, "w").write(s2)


def main(argv):
    cfg = json.load(open(CFG))
    if answers():
        print("strip OK at %s" % cfg["host"]); return 0
    ip = sweep(cfg["mac"])
    if not ip:
        print("strip not at %s and its MAC %s is not on %s" % (cfg["host"], cfg["mac"], ", ".join(p + "0/24" for p in pools()))); return 1
    ok, how = confirm(ip)
    if not ok:
        print("a device with the strip's MAC is at %s but not confirmed: %s" % (ip, how)); return 1
    if ip == cfg["host"]:
        print("strip at %s answers info but not status; not fixed" % ip); return 1
    if "--fix" not in argv:
        print("strip found at %s (strip.json says %s); rerun with --fix to correct it" % (ip, cfg["host"])); return 1
    fix(ip, how)
    print("strip moved %s -> %s; strip.json corrected" % (cfg["host"], ip))
    return 0 if answers() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
