#!/usr/bin/env python3
"""Read-only tool for the Shelly Power Strip Gen4 (S4PL-00416EU), HeteroServe's energy instrument for Project 1.
Python 3 standard library only. From the repo root:

    python3 testbed/scripts/shelly_strip.py info      [--host H] [--json]
    python3 testbed/scripts/shelly_strip.py status    [--host H] [--json]
    python3 testbed/scripts/shelly_strip.py log       --out CSV [--host H] [--hz 2] [--duration S] [--timeout 2]
    python3 testbed/scripts/shelly_strip.py summarise CSV --window START END [--idle START END] [--tokens N] [--json]

Common options: --config F (default testbed/energy/strip.json), --auth-file F (or the env SHELLY_AUTH_FILE).
Method, plan and the instrument's measured limits: testbed/energy.md.

READ-ONLY BY CONSTRUCTION. Every request this file can make is an HTTP GET of one of the exact paths in READ_ONLY_PATHS:
/shelly and four RPC methods, each of them a Get* method. ShellyClient.check_path() refuses any other path, and any query
string, before a socket is opened. testbed/tests/test_shelly_strip.py fails if this source names an RPC method that is not
a Get*, uses another HTTP verb, or makes a request outside the allow-list. The tool never switches an outlet, never changes
a setting, never restarts or updates the strip. Outlet 3 powers the router: see "do_not_switch_off" in strip.json.

HOST. "host" in testbed/energy/strip.json; --host overrides it. The strip moves between networks (the user's phone hotspot
first, the T628L router's Wi-Fi later), so no address is hard-coded here. status and log refuse to run when the device at
the host is not the one strip.json names (MAC): a wrong device would get the wrong outlet labels.

AUTH. Off on the strip today (auth_en false). If the user enables it, the password goes on one line in a file OUTSIDE the
repo, e.g. ~/.config/heteroserve/shelly.auth, chmod 600, passed with --auth-file or SHELLY_AUTH_FILE. The tool answers
the strip's HTTP digest challenge (RFC 7616, SHA-256 or MD5, user "admin"). It refuses an auth file inside the repo or
readable by group/others, and it never prints or writes the password.

WHAT THE STRIP MEASURES: AC wall power per outlet. That is the board PLUS its power adapter's conversion loss, unlike the
planned DC INA226 rail measurement. The strip refreshes its readings only about 1-2 times a second (measured 2026-10-04,
testbed/energy.md), so polling faster than ~2 Hz mostly re-reads the same values.

log  polls Shelly.GetStatus at --hz (at most 5) until --duration seconds pass, or SIGINT / SIGTERM. One request at a
     time; a slow request makes the next slot wait, never a burst. One CSV row per request:
       wall_time    laptop time.time() when the request started (unix s)
       mono_time    laptop time.monotonic() at the same instant
       rtt_s        request round trip (s)
       ok, error    1, or 0 and the error text
       dev_unixtime, dev_uptime          the strip's sys.unixtime (empty when its clock is not synced) and sys.uptime
       o<i>_output, _apower (W), _voltage (V), _current (A), _pf, _freq (Hz), _aenergy_total (Wh),
       _aenergy_minute_ts, _by_minute_0.._2 (mWh), _source (what last switched it), _errors   for outlets i = 0..3
     Numbers are written exactly as the strip sent them (its decimals are kept). A FAILED REQUEST IS A ROW WITH ok=0 AND
     EMPTY MEASUREMENT FIELDS: nothing is carried forward, nothing is filled in. Every row is flushed as it is written.
     <CSV>.meta.json is written at the start (device id, model, MAC, firmware, host, requested Hz, start time, outlet map,
     the outlets' names on the device, tool version and sha256) and rewritten at the end (end time, stop reason, row
     counts, firmware at the end). A request in flight when a signal arrives finishes and its row is written.

summarise  reads a log and reports, per outlet, over --window START END (unix seconds on the laptop clock):
     samples in the window; the achieved request rate; the FRESH-reading rate (how often the outlet's reported readings
     actually changed, overall and for apower / voltage / current separately); the largest gap between successful
     samples; the mean (of samples) and peak apower; energy by trapezoidal integration of apower (J); energy from the
     strip's own counter (aenergy_total delta, Wh x 3600 -> J); and their disagreement, 100 x (trapezoid - counter) /
     counter. With --idle START END: the idle mean power (by each method) and the idle-subtracted (net) energy. With
     --tokens N: J/token gross and net, by each method, and Wh per 1,000 tokens.
     Sample time = wall_time + rtt_s / 2. Both energies need the log to bracket the window (a successful sample at or before
     START and one at or after END): apower and the counter are linearly interpolated at START and END, never extrapolated.
     A freshness count is a LOWER bound: a fresh reading equal to the one before it cannot be seen.
     FLAGS are printed, never hidden: RESTART (sys.uptime went back, or advanced less than the elapsed time), COUNTER_RESET
     (aenergy_total decreased; the counter energy is then withheld), OUTPUT_FALSE, OUTLET_ERRORS, REQUEST_ERRORS, GAP
     (> 3 s between successful samples), TOO_FEW_SAMPLES (< 10 samples or < 10 fresh readings), NOT_COVERED (the log does
     not bracket the window; energies withheld), COUNTER_COARSE (the counter did not move, or one observed counter step at
     each end exceeds 5 % of the window's counter energy; the observed step is reported), FRESH_BELOW_1HZ,
     METHODS_DISAGREE (> 5 %), CLOCK_STEP (the laptop's wall clock moved against its monotonic clock by > 0.5 s, or ran
     backwards: energies withheld), NET_NEGATIVE, IDLE_OVERLAP, and from the meta file NO_META, LOG_NOT_CLOSED,
     FW_CHANGED, DEVICE_MISMATCH, NAMES_MISMATCH.

     NO FIGURE HERE IS A PHASE-ALIGNED ENERGY PER TOKEN unless the window is a single-phase run defined by the run protocol
     (testbed/energy.md): one long prefill-only or decode-only invocation, its start and end marked on the laptop clock,
     its tokens counted from its own timed repetitions. A window over a mixed run (load, warm-up, prefill and decode
     together) gives the run's energy, which cannot be attributed to either phase.
"""
import argparse, csv, datetime, decimal, hashlib, http.client, json, math, os, re, signal, socket, stat, statistics, sys, time

TOOL_VERSION = "1.0.0"
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_CONFIG = os.path.join(ROOT, "testbed", "energy", "strip.json")
# The ONLY requests this tool can make: HTTP GET of exactly these paths (no query strings). All are Get* methods.
READ_ONLY_PATHS = ("/shelly", "/rpc/Shelly.GetStatus", "/rpc/Shelly.GetConfig", "/rpc/Shelly.GetDeviceInfo",
                   "/rpc/Sys.GetStatus")
AUTH_USER = "admin"                 # the only user name a Shelly Gen2+ device accepts
AUTH_ENV = "SHELLY_AUTH_FILE"
N_OUTLETS = 4
MAX_HZ = 5.0                        # polling cap: the strip refreshes at ~1-2 Hz, faster polling buys nothing
GAP_FLAG_S = 3.0
MIN_SAMPLES = 10
FRESH_FLAG_HZ = 1.0
COUNTER_FLAG_PCT = 5.0              # one observed counter step at each end vs the window's counter energy (QR6's 5 %)
DISAGREE_FLAG_PCT = 5.0
CLOCK_STEP_FLAG_S = 0.5
RESTART_SLACK_S = 3.0               # sys.uptime is whole seconds
SECRETISH = re.compile(r"pass|psk|secret|token|key|pwd|cred", re.I)

ROW_FIELDS = ("wall_time", "mono_time", "rtt_s", "ok", "error", "dev_unixtime", "dev_uptime")
OUTLET_FIELDS = ("output", "apower", "voltage", "current", "pf", "freq", "aenergy_total", "aenergy_minute_ts",
                 "by_minute_0", "by_minute_1", "by_minute_2", "source", "errors")
FRESH_SIGNALS = {"any": ("apower", "voltage", "current", "pf"), "apower": ("apower",), "voltage": ("voltage",),
                 "current": ("current",)}


def csv_header():
    return list(ROW_FIELDS) + ["o%d_%s" % (i, f) for i in range(N_OUTLETS) for f in OUTLET_FIELDS]


# --------------------------------------------------------------------------------------------------------- HTTP client

class ShellyError(Exception):
    pass


def parse_digest_challenge(header):
    """The fields of a 'WWW-Authenticate: Digest ...' header, lower-case keys."""
    if not header or not header.strip().lower().startswith("digest "):
        raise ShellyError("the strip answered 401 without a Digest challenge: %r" % (header or "")[:120])
    fields = {}
    for k, quoted, bare in re.findall(r'([A-Za-z0-9_-]+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^\s,]*))', header.strip()[7:]):
        fields[k.lower()] = quoted if quoted or not bare else bare
    for need in ("realm", "nonce"):
        if need not in fields:
            raise ShellyError("Digest challenge without %s: %r" % (need, header[:120]))
    return fields


def digest_response(algorithm, user, realm, password, method, uri, nonce, nc, cnonce, qop):
    """RFC 7616 request-digest (MD5 or SHA-256, qop auth or none)."""
    impl = {"MD5": hashlib.md5, "SHA-256": hashlib.sha256}.get((algorithm or "MD5").upper())
    if impl is None:
        raise ShellyError("unsupported digest algorithm %r" % algorithm)
    h = lambda s: impl(s.encode("utf-8")).hexdigest()
    ha1 = h("%s:%s:%s" % (user, realm, password))
    ha2 = h("%s:%s" % (method, uri))
    if qop:
        return h("%s:%s:%s:%s:%s:%s" % (ha1, nonce, nc, cnonce, qop, ha2))
    return h("%s:%s:%s" % (ha1, nonce, ha2))


class ShellyClient:
    """HTTP GET of the allow-listed paths only. One connection per request."""

    def __init__(self, host, timeout=2.0, password=None):
        if not host or "/" in host or "?" in host or "@" in host or any(c.isspace() for c in host):
            raise ShellyError("bad host %r: give an address or name (optionally :port), no scheme or path" % host)
        self.host, self.timeout, self._password = host, timeout, password
        self._challenge, self._nc = None, 0

    @staticmethod
    def check_path(path):
        if path not in READ_ONLY_PATHS:
            raise ShellyError("refused: %r is not one of the read-only paths %s" % (path, ", ".join(READ_ONLY_PATHS)))
        return path

    def get(self, path):
        self.check_path(path)
        status, headers, body = self._get(path, self._authorization(path))
        if status == 401:
            if self._password is None:
                raise ShellyError("HTTP 401: the strip asks for a password; give --auth-file or set %s" % AUTH_ENV)
            self._challenge, self._nc = parse_digest_challenge(headers.get("WWW-Authenticate", "")), 0
            status, headers, body = self._get(path, self._authorization(path))
            if status == 401:
                raise ShellyError("HTTP 401 after answering the digest challenge: wrong password?")
        if status != 200:
            raise ShellyError("HTTP %d: %s" % (status, body[:200].decode("utf-8", "replace").strip()))
        try:
            return json.loads(body.decode("utf-8"), parse_float=decimal.Decimal)
        except ValueError as e:
            raise ShellyError("not JSON: %s" % e)

    def _get(self, path, authorization):
        conn = http.client.HTTPConnection(self.host, timeout=self.timeout)
        try:
            headers = {"Accept": "application/json", "Connection": "close"}
            if authorization:
                headers["Authorization"] = authorization
            conn.request("GET", path, headers=headers)
            r = conn.getresponse()
            return r.status, r.headers, r.read()
        finally:
            conn.close()

    def _authorization(self, path):
        if self._challenge is None:
            return None
        c = self._challenge
        qops = [q.strip() for q in c.get("qop", "").split(",") if q.strip()]
        if qops and "auth" not in qops:
            raise ShellyError("the strip's digest challenge offers qop %r only" % c.get("qop"))
        qop = "auth" if qops else ""
        self._nc += 1
        nc, cnonce = "%08x" % self._nc, os.urandom(12).hex()
        alg = c.get("algorithm", "MD5")
        resp = digest_response(alg, AUTH_USER, c["realm"], self._password, "GET", path, c["nonce"], nc, cnonce, qop)
        parts = ['username="%s"' % AUTH_USER, 'realm="%s"' % c["realm"], 'nonce="%s"' % c["nonce"], 'uri="%s"' % path,
                 "algorithm=%s" % alg, 'response="%s"' % resp]
        if qop:
            parts += ["qop=%s" % qop, "nc=%s" % nc, 'cnonce="%s"' % cnonce]
        if "opaque" in c:
            parts.append('opaque="%s"' % c["opaque"])
        return "Digest " + ", ".join(parts)


def read_password(path):
    """The password from an auth file outside the repo, chmod 600. Never printed."""
    real = os.path.realpath(os.path.expanduser(path))
    if real == ROOT or real.startswith(ROOT + os.sep):
        raise SystemExit("refused: the auth file %s is inside the repo; keep it in ~/.config/heteroserve/" % path)
    try:
        st = os.stat(real)
    except OSError as e:
        raise SystemExit("cannot read the auth file %s: %s" % (path, e.strerror))
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise SystemExit("refused: the auth file %s is readable by group or others; chmod 600 it" % path)
    with open(real, encoding="utf-8") as f:
        pw = f.readline().rstrip("\r\n")
    if not pw:
        raise SystemExit("refused: the auth file %s is empty" % path)
    return pw


# ------------------------------------------------------------------------------------------------- config and identity

def load_config(path, required):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        if required:
            raise SystemExit("no config at %s: create it (testbed/energy/strip.json) or give --config" % path)
        return {}


def outlet_map(cfg):
    m = cfg.get("outlets") or {}
    return {str(i): str(m.get(str(i), "?")) for i in range(N_OUTLETS)}


def norm_mac(s):
    return re.sub(r"[^0-9a-f]", "", str(s or "").lower())


def make_client(a, cfg):
    host = a.host or cfg.get("host")
    if not host:
        raise SystemExit("no host: give --host or set \"host\" in %s" % a.config)
    auth = a.auth_file or os.environ.get(AUTH_ENV)
    return ShellyClient(host, timeout=a.timeout, password=read_password(auth) if auth else None)


def identity(dev):
    return {k: _plain(dev.get(k)) for k in ("id", "mac", "model", "gen", "app", "ver", "fw_id", "name", "auth_en",
                                            "auth_domain")}


def identity_problems(dev, cfg):
    out = []
    want = norm_mac(cfg.get("mac"))
    if want and norm_mac(dev.get("mac")) != want:
        out.append("DEVICE_MISMATCH: the device at this host has MAC %s, strip.json names %s"
                   % (dev.get("mac"), cfg.get("mac")))
    if cfg.get("device_id") and dev.get("id") != cfg.get("device_id"):
        out.append("DEVICE_MISMATCH: the device at this host is %s, strip.json names %s" % (dev.get("id"), cfg.get("device_id")))
    return out


def outlet_names(conf):
    return {str(i): _plain(((conf or {}).get("switch:%d" % i) or {}).get("name")) for i in range(N_OUTLETS)}


def name_problems(names, omap):
    return ["NAMES_MISMATCH: outlet %s is named %r on the strip but mapped to %r in strip.json" % (i, names[i], omap[i])
            for i in sorted(names) if names[i] is not None and names[i] != omap.get(i)]


def _plain(v):
    """JSON-safe copy: Decimals become floats, password-like keys are dropped."""
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items() if not SECRETISH.search(str(k))}
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return v


def _now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------------------------- info

SWITCH_CONFIG_KEYS = ("name", "initial_state", "auto_on", "auto_on_delay", "auto_off", "auto_off_delay", "in_mode",
                      "power_limit", "current_limit", "voltage_limit", "undervoltage_limit")


def control_paths(conf):
    """Which other ways in to the strip are switched on (each of them could switch an outlet)."""
    out = {}
    for k in ("cloud", "mqtt", "ws", "matter", "knx"):
        if isinstance(conf.get(k), dict) and "enable" in conf[k]:
            out[k] = conf[k]["enable"]
    ble = conf.get("ble")
    if isinstance(ble, dict):
        out["ble_rpc"] = ble["enable"] if "enable" in ble else (ble.get("rpc") or {}).get("enable")
    return out


def build_info(dev, conf, st, cfg, host):
    sysst, syscf = st.get("sys") or {}, conf.get("sys") or {}
    wcf, wst = conf.get("wifi") or {}, st.get("wifi") or {}
    omap = outlet_map(cfg)
    outlets = []
    for i in range(N_OUTLETS):
        sc, ss = conf.get("switch:%d" % i) or {}, st.get("switch:%d" % i) or {}
        o = {"outlet": i, "map": omap[str(i)]}
        o.update({k: _plain(sc[k]) for k in SWITCH_CONFIG_KEYS if k in sc})
        o.update({"output_now": ss.get("output"), "source": ss.get("source")})
        outlets.append(o)
    updates = {ch: _plain((v or {}).get("version")) for ch, v in (sysst.get("available_updates") or {}).items()}
    info = {
        "host": host, "read_at": _now_iso(),
        "device": identity(dev),
        "available_updates": updates,
        "outlets": outlets,
        "wifi": {"sta_ssid": (wcf.get("sta") or {}).get("ssid"), "sta_enable": (wcf.get("sta") or {}).get("enable"),
                 "sta1_ssid": (wcf.get("sta1") or {}).get("ssid"), "sta1_enable": (wcf.get("sta1") or {}).get("enable"),
                 "ap_enable": (wcf.get("ap") or {}).get("enable"),
                 "now": {k: _plain(wst.get(k)) for k in ("status", "ssid", "sta_ip", "rssi")}},
        "sntp_server": (syscf.get("sntp") or {}).get("server"),
        "device_clock": {"unixtime": _plain(sysst.get("unixtime")), "time": sysst.get("time"),
                         "uptime_s": _plain(sysst.get("uptime"))},
        "control_paths": control_paths(conf),
        "do_not_switch_off": cfg.get("do_not_switch_off") or {},
    }
    checks = identity_problems(dev, cfg) + (name_problems(outlet_names(conf), omap) if cfg.get("outlets") else
                                            ["no strip.json outlet map: outlets unlabelled"])
    if dev.get("auth_en") is False:
        checks.append("auth is OFF: anyone on this network can switch any outlet, the router's included")
    for o in outlets:
        if o.get("initial_state") == "off":
            checks.append("outlet %d (%s) initial_state is off: after a power cut it stays off" % (o["outlet"], o["map"]))
        if o.get("auto_off"):
            checks.append("outlet %d (%s) has auto_off on (%s s)" % (o["outlet"], o["map"], o.get("auto_off_delay")))
    if sysst.get("unixtime") is None:
        checks.append("the strip's clock is not synced (sys.unixtime null); the logger timestamps on the laptop anyway")
    info["checks"] = checks
    return info


def print_info(info):
    d = info["device"]
    print("host        %s   (read %s)" % (info["host"], info["read_at"]))
    print("device      %s  model %s  gen %s  app %s  MAC %s" % (d["id"], d["model"], d["gen"], d["app"], d["mac"]))
    print("firmware    %s  (%s)" % (d["ver"], d["fw_id"]))
    print("updates     %s" % (", ".join("%s %s" % kv for kv in sorted(info["available_updates"].items())) or "none offered"))
    print("auth_en     %s" % d["auth_en"])
    print("wifi        sta %r (enable %s)  sta1 %r (enable %s)  ap enable %s" % (
        info["wifi"]["sta_ssid"], info["wifi"]["sta_enable"], info["wifi"]["sta1_ssid"], info["wifi"]["sta1_enable"],
        info["wifi"]["ap_enable"]))
    now = info["wifi"]["now"]
    print("            now: %s on %r, ip %s, rssi %s dBm" % (now.get("status"), now.get("ssid"), now.get("sta_ip"), now.get("rssi")))
    print("sntp        %s   device clock %s, uptime %s s" % (info["sntp_server"], info["device_clock"]["time"],
                                                              info["device_clock"]["uptime_s"]))
    print("control     %s" % ", ".join("%s %s" % kv for kv in info["control_paths"].items()))
    print("outlets")
    for o in info["outlets"]:
        extra = ", ".join("%s %s" % (k, o[k]) for k in SWITCH_CONFIG_KEYS[2:] if k in o)
        print("  %d %-7s name %-9r initial_state %-8s output now %-5s (source %s)%s" % (
            o["outlet"], o["map"], o.get("name"), o.get("initial_state"), o["output_now"], o["source"],
            ("\n             " + extra) if extra else ""))
    for k, v in sorted(info["do_not_switch_off"].items()):
        print("  DO NOT SWITCH OFF outlet %s: %s" % (k, v))
    for c in info["checks"]:
        print("CHECK  " + c)


def cmd_info(a):
    cfg = load_config(a.config, required=False)
    cli = make_client(a, cfg)
    try:
        dev = cli.get("/shelly")
        conf = cli.get("/rpc/Shelly.GetConfig")
        st = cli.get("/rpc/Shelly.GetStatus")
    except (ShellyError, OSError, http.client.HTTPException) as e:
        print("UNREACHABLE %s: %s: %s" % (cli.host, type(e).__name__, e), file=sys.stderr)
        return 2
    info = build_info(dev, conf, st, cfg, cli.host)
    if a.json:
        print(json.dumps(info, indent=1))
    else:
        print_info(info)
    return 0


# -------------------------------------------------------------------------------------------------------------- status

def _txt(v):
    """A device value as CSV text: '' for absent or null, booleans 1/0, numbers exactly as the strip sent them."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (list, tuple)):
        return ";".join(_txt(x) for x in v)
    return str(v)


def status_fields(st):
    """CSV fields of one Shelly.GetStatus answer. Raises ValueError if an outlet is missing (the row then fails)."""
    sysst = st.get("sys")
    if not isinstance(sysst, dict):
        raise ValueError("sys missing from Shelly.GetStatus")
    out = {"dev_unixtime": _txt(sysst.get("unixtime")), "dev_uptime": _txt(sysst.get("uptime"))}
    for i in range(N_OUTLETS):
        sw = st.get("switch:%d" % i)
        if not isinstance(sw, dict):
            raise ValueError("switch:%d missing from Shelly.GetStatus" % i)
        ae = sw.get("aenergy") if isinstance(sw.get("aenergy"), dict) else {}
        bm = ae.get("by_minute") if isinstance(ae.get("by_minute"), list) else []
        vals = {"output": sw.get("output"), "apower": sw.get("apower"), "voltage": sw.get("voltage"),
                "current": sw.get("current"), "pf": sw.get("pf"), "freq": sw.get("freq"),
                "aenergy_total": ae.get("total"), "aenergy_minute_ts": ae.get("minute_ts"),
                "by_minute_0": bm[0] if len(bm) > 0 else None, "by_minute_1": bm[1] if len(bm) > 1 else None,
                "by_minute_2": bm[2] if len(bm) > 2 else None, "source": sw.get("source"), "errors": sw.get("errors")}
        for k in OUTLET_FIELDS:
            out["o%d_%s" % (i, k)] = _txt(vals[k])
    return out


def cmd_status(a):
    cfg = load_config(a.config, required=False)
    cli = make_client(a, cfg)
    try:
        dev = cli.get("/shelly")
        st = cli.get("/rpc/Shelly.GetStatus")
    except (ShellyError, OSError, http.client.HTTPException) as e:
        print("UNREACHABLE %s: %s: %s" % (cli.host, type(e).__name__, e), file=sys.stderr)
        return 2
    bad = identity_problems(dev, cfg)
    if bad:
        print("\n".join(bad) + "\nrefused: fix strip.json (or --host) so the outlet labels are right", file=sys.stderr)
        return 3
    omap, sysst, w = outlet_map(cfg), st.get("sys") or {}, st.get("wifi") or {}
    rows = []
    for i in range(N_OUTLETS):
        sw = st.get("switch:%d" % i) or {}
        rows.append({"outlet": i, "map": omap[str(i)], "output": sw.get("output"), "apower_w": _plain(sw.get("apower")),
                     "voltage_v": _plain(sw.get("voltage")), "current_a": _plain(sw.get("current")), "pf": _plain(sw.get("pf")),
                     "aenergy_total_wh": _plain((sw.get("aenergy") or {}).get("total")), "source": sw.get("source"),
                     "errors": sw.get("errors") or []})
    head = {"host": cli.host, "device": dev.get("id"), "fw": dev.get("ver"), "read_at": _now_iso(),
            "device_unixtime": _plain(sysst.get("unixtime")), "uptime_s": _plain(sysst.get("uptime")),
            "wifi": "%s rssi %s" % (w.get("ssid"), w.get("rssi"))}
    if a.json:
        print(json.dumps({"head": head, "outlets": rows}, indent=1))
        return 0
    print("%(host)s  %(device)s  fw %(fw)s  read %(read_at)s  device unixtime %(device_unixtime)s  uptime %(uptime_s)s s  "
          "wifi %(wifi)s" % head)
    print("%-3s %-7s %-6s %9s %9s %9s %6s %14s  %-12s %s" % ("out", "map", "output", "apower W", "voltage V", "current A",
                                                             "pf", "aenergy.tot Wh", "source", "errors"))
    for r in rows:
        print("%-3d %-7s %-6s %9s %9s %9s %6s %14s  %-12s %s" % (
            r["outlet"], r["map"], r["output"], r["apower_w"], r["voltage_v"], r["current_a"], r["pf"],
            r["aenergy_total_wh"], r["source"], ";".join(r["errors"])))
    return 0


# ----------------------------------------------------------------------------------------------------------------- log

def make_row(wall, mono, rtt, ok, error="", fields=None):
    """One CSV row. A failed request keeps every measurement field empty: nothing is carried forward."""
    row = dict.fromkeys(csv_header(), "")
    row.update(wall_time="%.6f" % wall, mono_time="%.6f" % mono, rtt_s="%.6f" % rtt, ok="1" if ok else "0",
               error=("" if ok else " ".join(str(error).split())[:300]))
    if ok:
        row.update(fields)
    return row


def poll(cli, write, hz, duration, stopped, clock=time.monotonic, wall=time.time, sleep=time.sleep):
    """Request Shelly.GetStatus on a fixed schedule of 1/hz slots; one request at a time, late slots are skipped."""
    period = 1.0 / hz
    t0, k, n, n_ok = clock(), 0, 0, 0
    while not stopped():
        now = clock()
        if duration is not None and now - t0 >= duration:
            break
        target = t0 + k * period
        if now < target:
            sleep(min(target - now, 0.1))
            continue
        w, m = wall(), clock()
        try:
            fields = status_fields(cli.get("/rpc/Shelly.GetStatus"))
            row = make_row(w, m, clock() - m, True, fields=fields)
        except (ShellyError, OSError, http.client.HTTPException, ValueError) as e:
            row = make_row(w, m, clock() - m, False, "%s: %s" % (type(e).__name__, e))
        write(row)
        n += 1
        n_ok += row["ok"] == "1"
        k = max(k + 1, int(math.floor((clock() - t0) / period)) + 1)
    return n, n_ok


def write_meta(path, meta):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def _sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def cmd_log(a):
    if not (0 < a.hz <= MAX_HZ):
        raise SystemExit("refused: --hz must be in (0, %g]; the strip refreshes at ~1-2 Hz" % MAX_HZ)
    if a.duration is not None and a.duration <= 0:
        raise SystemExit("refused: --duration must be positive")
    out = os.path.abspath(a.out)
    meta_path = out + ".meta.json"
    if not os.path.isdir(os.path.dirname(out)):
        raise SystemExit("refused: the directory of %s does not exist" % out)
    for p in (out, meta_path):
        if os.path.exists(p):
            raise SystemExit("refused: %s exists; a log is never overwritten" % p)
    cfg = load_config(a.config, required=False)
    cli = make_client(a, cfg)
    try:
        dev = cli.get("/shelly")
        conf = cli.get("/rpc/Shelly.GetConfig")
    except (ShellyError, OSError, http.client.HTTPException) as e:
        print("UNREACHABLE %s: %s: %s" % (cli.host, type(e).__name__, e), file=sys.stderr)
        return 2
    bad = identity_problems(dev, cfg)
    if bad:
        print("\n".join(bad) + "\nrefused: fix strip.json (or --host) so the outlet labels are right", file=sys.stderr)
        return 3
    omap, names = outlet_map(cfg), outlet_names(conf)
    start_wall, start_mono = time.time(), time.monotonic()
    meta = {
        "tool": "testbed/scripts/shelly_strip.py", "tool_version": TOOL_VERSION,
        "tool_sha256": _sha256_file(os.path.abspath(__file__)),
        "csv": os.path.basename(out), "host": cli.host,
        "config": os.path.relpath(os.path.abspath(a.config), ROOT) if cfg else None,
        "device": identity(dev), "expected_mac": cfg.get("mac"),
        "outlet_map": omap if cfg else None, "outlet_names_on_device": names,
        "outlet_initial_state": {str(i): ((conf.get("switch:%d" % i) or {}).get("initial_state")) for i in range(N_OUTLETS)},
        "do_not_switch_off": cfg.get("do_not_switch_off"),
        "problems_at_start": name_problems(names, omap) if cfg else ["NO_CONFIG: no strip.json, outlets unlabelled"],
        "requested_hz": a.hz, "timeout_s": a.timeout, "duration_requested_s": a.duration,
        "start_wall": start_wall, "start_iso": datetime.datetime.fromtimestamp(start_wall).astimezone().isoformat(),
        "start_mono": start_mono, "end_wall": None, "end_iso": None, "stop_reason": "running",
        "rows": None, "ok_rows": None, "error_rows": None, "device_at_end": None, "firmware_changed": None,
        "sample_time": "wall_time + rtt_s / 2, laptop clock",
        "laptop": {"hostname": socket.gethostname(), "python": sys.version.split()[0]},
    }
    write_meta(meta_path, meta)
    stop = {"why": None}

    def on_signal(signum, frame):
        stop["why"] = signal.Signals(signum).name

    old = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    print("logging %s every %.3f s%s -> %s  (Ctrl-C to stop)" % (
        cli.host, 1.0 / a.hz, (" for %g s" % a.duration) if a.duration else "", out), file=sys.stderr)
    try:
        with open(out, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=csv_header())
            wr.writeheader()
            f.flush()

            def write(row):
                wr.writerow(row)
                f.flush()
                if row["ok"] != "1":
                    print("request failed: %s" % row["error"], file=sys.stderr)

            n, n_ok = poll(cli, write, a.hz, a.duration, lambda: stop["why"] is not None)
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    end_wall = time.time()
    try:
        dev_end = identity(cli.get("/shelly"))
    except (ShellyError, OSError, http.client.HTTPException) as e:
        dev_end = {"unreachable": "%s: %s" % (type(e).__name__, e)}
    meta.update(end_wall=end_wall, end_iso=datetime.datetime.fromtimestamp(end_wall).astimezone().isoformat(),
                stop_reason=stop["why"] or "duration", rows=n, ok_rows=n_ok, error_rows=n - n_ok, device_at_end=dev_end,
                firmware_changed=(None if "unreachable" in dev_end else
                                  (dev_end.get("fw_id") != meta["device"].get("fw_id"))))
    write_meta(meta_path, meta)
    span = end_wall - start_wall
    print("done: %d requests in %.1f s (%.2f/s), %d failed; stop: %s; meta %s" % (
        n, span, n / span if span > 0 else 0.0, n - n_ok, meta["stop_reason"], meta_path), file=sys.stderr)
    return 0


# ----------------------------------------------------------------------------------------------------------- summarise

def load_log(path):
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        missing = [c for c in csv_header() if c not in (rd.fieldnames or [])]
        if missing:
            raise SystemExit("%s is not a shelly_strip.py log (missing %s)" % (path, ", ".join(missing[:4])))
        rows = []
        for n, d in enumerate(rd, start=2):
            try:
                if None in d or None in d.values():
                    raise ValueError("%s fields, the header has %d" % ("too many" if None in d else "too few", len(rd.fieldnames)))
                if d["ok"] not in ("0", "1"):
                    raise ValueError("ok is %r" % d["ok"])
                wall, rtt = float(d["wall_time"]), float(d["rtt_s"])
                rows.append({"t": wall + rtt / 2.0, "wall": wall, "mono": float(d["mono_time"]), "ok": d["ok"] == "1",
                             "error": d["error"], "uptime": _num(d["dev_uptime"]), "raw": d})
            except (TypeError, ValueError) as e:
                raise SystemExit("%s line %d is malformed (%s). A truncated LAST line means the logger was killed while "
                                 "writing it: delete that line by hand and re-run; anything else, do not use the log"
                                 % (path, n, e))
    return rows


def _num(s):
    return float(s) if s not in ("", None) else None


def _key(i, f):
    return "o%d_%s" % (i, f)


def interp(points, x):
    """Linear interpolation of (t, v) points (t non-decreasing) at x; None if x is outside them (no extrapolation)."""
    if not points or x < points[0][0] or x > points[-1][0]:
        return None
    for (t1, v1), (t2, v2) in zip(points, points[1:]):
        if t1 <= x <= t2:
            return v1 if t2 == t1 else v1 + (v2 - v1) * (x - t1) / (t2 - t1)
    return points[-1][1]


def integrate(points, a, b):
    """Trapezoidal integral of (t, v) points over [a, b], interpolating at a and b; None unless the points bracket it."""
    va, vb = interp(points, a), interp(points, b)
    if va is None or vb is None:
        return None
    pts = [(a, va)] + [(t, v) for t, v in points if a < t < b] + [(b, vb)]
    return sum(0.5 * (v1 + v2) * (t2 - t1) for (t1, v1), (t2, v2) in zip(pts, pts[1:]))


def _decimals(s):
    return len(s.split(".", 1)[1]) if "." in s else 0


def analyse(rows, i, a, b):
    """Every per-outlet figure and flag for outlet i over [a, b]. Flags are (code, message) tuples."""
    T = b - a
    oks = [r for r in rows if r["ok"]]
    win = [r for r in rows if a <= r["t"] <= b]
    win_ok = [r for r in win if r["ok"]]
    flags, notes = [], []
    # the span the energies rest on: last ok row at/before a .. first ok row at/after b
    lo = max((k for k, r in enumerate(oks) if r["t"] <= a), default=None)
    hi = min((k for k, r in enumerate(oks) if r["t"] >= b), default=None)
    covered = lo is not None and hi is not None
    span = oks[lo:hi + 1] if covered else win_ok
    res = {"window_s": T, "requests": len(win), "samples_ok": len(win_ok), "request_rate_hz": len(win) / T,
           "ok_rate_hz": len(win_ok) / T, "covered": covered}
    if not covered:
        flags.append(("NOT_COVERED", "the log does not bracket the window (first ok sample %s, last %s); energies "
                      "withheld" % (_fmt_t(oks[0]["t"]) if oks else "-", _fmt_t(oks[-1]["t"]) if oks else "-")))
    # time axis: the laptop's wall clock against its monotonic clock, and ordering
    rng = [r for r in rows if (span[0]["t"] if span else a) <= r["t"] <= (span[-1]["t"] if span else b)] or win
    clock_bad = False
    if rng:
        off0 = rows[0]["wall"] - rows[0]["mono"]
        worst = max(abs((r["wall"] - r["mono"]) - off0) for r in rng)
        backwards = any(r2["t"] < r1["t"] for r1, r2 in zip(rng, rng[1:]))
        if worst > CLOCK_STEP_FLAG_S or backwards:
            clock_bad = True
            flags.append(("CLOCK_STEP", "the laptop's wall clock moved %.3f s against its monotonic clock%s; energies "
                          "withheld" % (worst, " and ran backwards" if backwards else "")))
    # device restarts
    for r1, r2 in zip(span, span[1:]):
        if r1["uptime"] is None or r2["uptime"] is None:
            continue
        if r2["uptime"] < r1["uptime"] or (r2["uptime"] - r1["uptime"]) < (r2["mono"] - r1["mono"]) - RESTART_SLACK_S:
            flags.append(("RESTART", "the strip restarted between %s and %s (uptime %g -> %g s)"
                          % (_fmt_t(r1["t"]), _fmt_t(r2["t"]), r1["uptime"], r2["uptime"])))
    # requests and gaps
    failed = [r for r in win if not r["ok"]]
    if failed:
        kinds = sorted(set(r["error"].split(":")[0] for r in failed))
        flags.append(("REQUEST_ERRORS", "%d of %d requests failed (%s); first: %s"
                      % (len(failed), len(win), ", ".join(kinds), failed[0]["error"][:120])))
    ts = [a] + [r["t"] for r in win_ok] + [b]
    gaps = [t2 - t1 for t1, t2 in zip(ts, ts[1:])]
    res["max_gap_s"] = max(gaps)
    big = [g for g in gaps if g > GAP_FLAG_S]
    if big:
        flags.append(("GAP", "%d gap(s) > %g s between successful samples, the largest %.2f s" % (len(big), GAP_FLAG_S, max(big))))
    # output state and the outlet's own errors
    off = [r for r in win_ok if r["raw"][_key(i, "output")] != "1"]
    if off:
        flags.append(("OUTPUT_FALSE", "output was not on in %d of %d samples" % (len(off), len(win_ok))))
    errs = sorted(set(r["raw"][_key(i, "errors")] for r in win_ok if r["raw"][_key(i, "errors")]))
    if errs:
        flags.append(("OUTLET_ERRORS", "the strip reported errors on this outlet: %s" % "; ".join(errs)))
    # freshness: a sample whose reading differs from the previous successful sample's (a lower bound)
    for name, fields in FRESH_SIGNALS.items():
        prev, times = None, []
        for r in oks:
            v = tuple(r["raw"][_key(i, f)] for f in fields)
            if prev is not None and v != prev and a <= r["t"] <= b:
                times.append(r["t"])
            prev = v
        res["fresh_%s_n" % name] = len(times)
        res["fresh_%s_hz" % name] = len(times) / T
        iv = [t2 - t1 for t1, t2 in zip(times, times[1:])]
        res["fresh_%s_median_interval_s" % name] = statistics.median(iv) if iv else None
    if res["fresh_any_hz"] < FRESH_FLAG_HZ:
        flags.append(("FRESH_BELOW_1HZ", "the readings changed %.2f times a second (%d changes, median interval %s s)"
                      % (res["fresh_any_hz"], res["fresh_any_n"], _g(res["fresh_any_median_interval_s"]))))
    if len(win_ok) < MIN_SAMPLES or res["fresh_any_n"] < MIN_SAMPLES:
        flags.append(("TOO_FEW_SAMPLES", "%d successful samples and %d fresh readings in the window (want >= %d of each)"
                      % (len(win_ok), res["fresh_any_n"], MIN_SAMPLES)))
    apw = sorted(set(r["raw"][_key(i, "apower")] for r in win_ok))
    if len(apw) == 1:
        notes.append("apower never changed in the window (%s W): its freshness cannot be seen; the overall fresh rate "
                     "here comes from voltage/current/pf" % apw[0])
    # power
    P = [(r["t"], float(r["raw"][_key(i, "apower")])) for r in oks if r["raw"][_key(i, "apower")] != ""]
    Pw = [v for t, v in P if a <= t <= b]
    res["mean_w"] = statistics.fmean(Pw) if Pw else None
    res["peak_w"] = max(Pw) if Pw else None
    res["energy_trap_j"] = None if clock_bad else integrate(P, a, b)
    # the counter
    C = [(r["t"], float(r["raw"][_key(i, "aenergy_total")])) for r in oks if r["raw"][_key(i, "aenergy_total")] != ""]
    Cs = [(r["t"], float(r["raw"][_key(i, "aenergy_total")]), r["raw"][_key(i, "aenergy_total")]) for r in span
          if r["raw"][_key(i, "aenergy_total")] != ""]
    reset = [(c1, c2) for (_, c1, _), (_, c2, _) in zip(Cs, Cs[1:]) if c2 < c1]
    if reset:
        flags.append(("COUNTER_RESET", "aenergy_total decreased %d time(s) (first %g -> %g Wh); the counter energy is "
                      "withheld" % (len(reset), reset[0][0], reset[0][1])))
    ca, cb = interp(C, a), interp(C, b)
    res["energy_counter_j"] = None if (reset or clock_bad or ca is None or cb is None) else (cb - ca) * 3600.0
    incs = [c2 - c1 for (_, c1, _), (_, c2, _) in zip(Cs, Cs[1:]) if c2 > c1]
    chg = [t2 for (t1, c1, _), (t2, c2, _) in zip(Cs, Cs[1:]) if c2 != c1]
    res["counter_step_wh"] = statistics.median(incs) if incs else None
    res["counter_step_j"] = res["counter_step_wh"] * 3600.0 if incs else None
    res["counter_steps_n"] = len(incs)
    civ = [t2 - t1 for t1, t2 in zip(chg, chg[1:])]
    res["counter_update_interval_s"] = statistics.median(civ) if civ else None
    res["counter_display_resolution_wh"] = 10.0 ** -max(_decimals(s) for _, _, s in Cs) if Cs else None
    Ec, Et = res["energy_counter_j"], res["energy_trap_j"]
    res["counter_quantisation_pct"] = None
    if Ec is not None:
        if not incs or Ec <= 0:
            flags.append(("COUNTER_COARSE", "the counter did not move in the window: its energy is below one counter step "
                          "(display resolution %s Wh = %s J); no step observed%s" % (
                              _g(res["counter_display_resolution_wh"]),
                              _g(res["counter_display_resolution_wh"] * 3600.0) if Cs else "?",
                              "" if not Et else "; the trapezoid gives %.1f J" % Et)))
        else:
            q = 100.0 * 2.0 * res["counter_step_j"] / Ec
            res["counter_quantisation_pct"] = q
            if q > COUNTER_FLAG_PCT:
                flags.append(("COUNTER_COARSE", "one observed counter step (%.4g Wh = %.1f J, every ~%s s) at each end is "
                              "%.1f %% of the window's counter energy %.1f J (flag above %g %%)" % (
                                  res["counter_step_wh"], res["counter_step_j"],
                                  _g(res["counter_update_interval_s"]), q, Ec, COUNTER_FLAG_PCT)))
    res["disagreement_pct"] = 100.0 * (Et - Ec) / Ec if (Et is not None and Ec) else None
    if res["disagreement_pct"] is not None and abs(res["disagreement_pct"]) > DISAGREE_FLAG_PCT:
        flags.append(("METHODS_DISAGREE", "trapezoid %.1f J vs counter %.1f J: %+.2f %%" % (Et, Ec, res["disagreement_pct"])))
    return res, flags, notes


def _g(v):
    return "-" if v is None else ("%.4g" % v)


def _fmt_t(t):
    return datetime.datetime.fromtimestamp(t).astimezone().strftime("%H:%M:%S.%f")[:-3]


def summarise(rows, meta, window, idle=None, tokens=None, cfg_map=None):
    a, b = window
    if not b > a:
        raise SystemExit("refused: the window must have END > START")
    if idle and not idle[1] > idle[0]:
        raise SystemExit("refused: the idle window must have END > START")
    if tokens is not None and tokens <= 0:
        raise SystemExit("refused: --tokens must be positive")
    flags, notes = [], []
    if meta is None:
        flags.append({"outlet": None, "code": "NO_META", "message": "no <CSV>.meta.json: device, firmware and outlet map "
                      "unverified; outlet labels from strip.json"})
        omap = cfg_map or {str(i): "?" for i in range(N_OUTLETS)}
    else:
        omap = meta.get("outlet_map") or cfg_map or {str(i): "?" for i in range(N_OUTLETS)}
        if meta.get("end_wall") is None:
            flags.append({"outlet": None, "code": "LOG_NOT_CLOSED", "message": "the meta file has no end: the logger is "
                          "still running, or it was killed"})
        if meta.get("firmware_changed"):
            flags.append({"outlet": None, "code": "FW_CHANGED", "message": "firmware changed during the log: %s -> %s" % (
                meta["device"].get("fw_id"), (meta.get("device_at_end") or {}).get("fw_id"))})
        dm = norm_mac((meta.get("device") or {}).get("mac"))
        if meta.get("expected_mac") and dm != norm_mac(meta["expected_mac"]):
            flags.append({"outlet": None, "code": "DEVICE_MISMATCH", "message": "logged device MAC %s, strip.json %s" % (
                meta["device"].get("mac"), meta["expected_mac"])})
        for p in meta.get("problems_at_start") or []:
            flags.append({"outlet": None, "code": p.split(":")[0], "message": p.split(":", 1)[-1].strip()})
    if idle and idle[0] < b and a < idle[1]:
        flags.append({"outlet": None, "code": "IDLE_OVERLAP", "message": "the idle window overlaps the measured window"})
    out = {"window": [a, b], "window_s": b - a, "idle": list(idle) if idle else None, "tokens": tokens, "outlets": {},
           "device": (meta or {}).get("device"), "host": (meta or {}).get("host"),
           "requested_hz": (meta or {}).get("requested_hz"), "sample_time": "wall_time + rtt_s / 2, laptop clock"}
    for i in range(N_OUTLETS):
        label = omap.get(str(i), "?")
        res, fl, nt = analyse(rows, i, a, b)
        res["label"] = label
        flags += [{"outlet": i, "code": c, "message": m} for c, m in fl]
        notes += ["outlet %d (%s): %s" % (i, label, n) for n in nt]
        if idle:
            ires, ifl, _ = analyse(rows, i, idle[0], idle[1])
            flags += [{"outlet": i, "code": "IDLE_" + c, "message": "idle window: " + m} for c, m in ifl]
            Ti = idle[1] - idle[0]
            for m in ("trap", "counter"):
                Ei, E = ires["energy_%s_j" % m], res["energy_%s_j" % m]
                pi = Ei / Ti if Ei is not None else None
                res["idle_mean_w_%s" % m] = pi
                res["net_energy_%s_j" % m] = (E - pi * (b - a)) if (E is not None and pi is not None) else None
                if res["net_energy_%s_j" % m] is not None and res["net_energy_%s_j" % m] < 0:
                    flags.append({"outlet": i, "code": "NET_NEGATIVE", "message": "net energy by %s is %.1f J: the idle "
                                  "floor exceeds the window's mean power" % (m, res["net_energy_%s_j" % m])})
            res["idle_samples_ok"] = ires["samples_ok"]
        if tokens:
            for m in ("trap", "counter"):
                E = res["energy_%s_j" % m]
                res["j_per_token_gross_%s" % m] = E / tokens if E is not None else None
                res["wh_per_1000_tokens_gross_%s" % m] = E / tokens * 1000.0 / 3600.0 if E is not None else None
                if idle:
                    N = res["net_energy_%s_j" % m]
                    res["j_per_token_net_%s" % m] = N / tokens if N is not None else None
                    res["wh_per_1000_tokens_net_%s" % m] = N / tokens * 1000.0 / 3600.0 if N is not None else None
        out["outlets"][str(i)] = res
    out["flags"], out["notes"] = collapse_flags(flags), notes
    return out


def collapse_flags(flags):
    """A flag raised with the same message for every outlet (a request error, a restart, a gap) is printed once."""
    owners = {}
    for f in flags:
        if isinstance(f["outlet"], int):
            owners.setdefault((f["code"], f["message"]), set()).add(f["outlet"])
    out, done = [], set()
    for f in flags:
        key = (f["code"], f["message"])
        if isinstance(f["outlet"], int) and owners[key] == set(range(N_OUTLETS)):
            if key not in done:
                done.add(key)
                out.append({"outlet": "all", "code": f["code"], "message": f["message"]})
        else:
            out.append(f)
    return out


TEXT_ROWS = [
    ("samples ok", "samples_ok", "%d"), ("requests / s", "request_rate_hz", "%.2f"), ("ok samples / s", "ok_rate_hz", "%.2f"),
    ("FRESH / s (any)", "fresh_any_hz", "%.2f"), ("  fresh / s apower", "fresh_apower_hz", "%.2f"),
    ("  fresh / s voltage", "fresh_voltage_hz", "%.2f"), ("  fresh / s current", "fresh_current_hz", "%.2f"),
    ("  median fresh interval s", "fresh_any_median_interval_s", "%.3f"), ("max gap s", "max_gap_s", "%.2f"),
    ("mean apower W", "mean_w", "%.3f"), ("peak apower W", "peak_w", "%.3f"),
    ("E trapezoid J", "energy_trap_j", "%.2f"), ("E counter J", "energy_counter_j", "%.2f"),
    ("disagreement %", "disagreement_pct", "%+.2f"), ("counter step Wh (observed)", "counter_step_wh", "%.4g"),
    ("counter update interval s", "counter_update_interval_s", "%.2f"),
    ("counter display res. Wh", "counter_display_resolution_wh", "%g"),
    ("counter quantisation %", "counter_quantisation_pct", "%.2f"),
    ("idle mean W (trap)", "idle_mean_w_trap", "%.3f"), ("idle mean W (counter)", "idle_mean_w_counter", "%.3f"),
    ("net E trapezoid J", "net_energy_trap_j", "%.2f"), ("net E counter J", "net_energy_counter_j", "%.2f"),
    ("J/token gross (trap)", "j_per_token_gross_trap", "%.4g"), ("J/token gross (counter)", "j_per_token_gross_counter", "%.4g"),
    ("J/token net (trap)", "j_per_token_net_trap", "%.4g"), ("J/token net (counter)", "j_per_token_net_counter", "%.4g"),
    ("Wh/1k tok gross (counter)", "wh_per_1000_tokens_gross_counter", "%.4g"),
    ("Wh/1k tok net (counter)", "wh_per_1000_tokens_net_counter", "%.4g"),
]


def print_summary(path, s):
    d = s.get("device") or {}
    print("log       %s" % path)
    print("device    %s  fw %s  host %s  requested %s Hz" % (d.get("id"), d.get("ver"), s.get("host"), s.get("requested_hz")))
    print("window    %.3f .. %.3f  (%s .. %s, %.1f s)%s%s" % (
        s["window"][0], s["window"][1], _fmt_t(s["window"][0]), _fmt_t(s["window"][1]), s["window_s"],
        ("  idle %.3f .. %.3f" % tuple(s["idle"])) if s["idle"] else "", ("  tokens %d" % s["tokens"]) if s["tokens"] else ""))
    print("sample time = %s" % s["sample_time"])
    cols = [s["outlets"][str(i)] for i in range(N_OUTLETS)]
    print("%-27s" % "outlet" + "".join("%14s" % ("%d %s" % (i, c["label"])) for i, c in enumerate(cols)))
    for label, k, fmt in TEXT_ROWS:
        if all(k not in c for c in cols):
            continue
        print("%-27s" % label + "".join("%14s" % ("-" if c.get(k) is None else fmt % c[k]) for c in cols))
    print("FLAGS (%d)" % len(s["flags"]))
    for f in s["flags"]:
        where = {None: "", "all": "all outlets: "}.get(f["outlet"]) if not isinstance(f["outlet"], int) else \
            "outlet %d %s: " % (f["outlet"], s["outlets"][str(f["outlet"])]["label"])
        print("  %-17s %s%s" % (f["code"], where, f["message"]))
    for n in s["notes"]:
        print("NOTE  " + n)
    print("No figure here is a phase-aligned energy per token unless the window is a single-phase run defined by the run "
          "protocol (testbed/energy.md).")


def cmd_summarise(a):
    rows = load_log(a.csv)
    if not rows:
        raise SystemExit("%s has no rows" % a.csv)
    meta = None
    if os.path.exists(a.csv + ".meta.json"):
        with open(a.csv + ".meta.json", encoding="utf-8") as f:
            meta = json.load(f)
    cfg = load_config(a.config, required=False)
    s = summarise(rows, meta, tuple(a.window), tuple(a.idle) if a.idle else None, a.tokens,
                  outlet_map(cfg) if cfg else None)
    if a.json:
        print(json.dumps(s, indent=1))
    else:
        print_summary(a.csv, s)
    return 1 if (a.fail_on_flags and s["flags"]) else 0


# ----------------------------------------------------------------------------------------------------------------- main

def build_parser():
    ap = argparse.ArgumentParser(description="Read-only tool for the Shelly Power Strip Gen4 (see the module docstring).")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=DEFAULT_CONFIG, help="strip.json (host, MAC, outlet map)")
    common.add_argument("--host", help="overrides strip.json's host")
    common.add_argument("--auth-file", help="password file outside the repo, chmod 600 (or env %s)" % AUTH_ENV)
    common.add_argument("--timeout", type=float, default=2.0, help="per-request socket timeout, s")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", parents=[common], help="identity, firmware, auth, outlet config, Wi-Fi, SNTP")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_info)
    p = sub.add_parser("status", parents=[common], help="one reading per outlet")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("log", parents=[common], help="poll Shelly.GetStatus into a CSV")
    p.add_argument("--out", required=True)
    p.add_argument("--hz", type=float, default=2.0, help="requested request rate, at most %g" % MAX_HZ)
    p.add_argument("--duration", type=float, help="seconds; default: until SIGINT/SIGTERM")
    p.set_defaults(func=cmd_log)
    p = sub.add_parser("summarise", parents=[common], help="energy, rates and flags over a window of a log")
    p.add_argument("csv")
    p.add_argument("--window", nargs=2, type=float, required=True, metavar=("START", "END"), help="unix s, laptop clock")
    p.add_argument("--idle", nargs=2, type=float, metavar=("START", "END"))
    p.add_argument("--tokens", type=int)
    p.add_argument("--json", action="store_true")
    p.add_argument("--fail-on-flags", action="store_true", help="exit 1 when any flag is raised")
    p.set_defaults(func=cmd_summarise)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except ShellyError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
