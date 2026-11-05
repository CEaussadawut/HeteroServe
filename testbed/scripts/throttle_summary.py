#!/usr/bin/env python3
"""Certify a cell from its 1 Hz throttle poll (throttle_poll.sh v2, or v1's lines): the THROTTLE fields and the verdict.

    python3 testbed/scripts/throttle_summary.py LOG [--from EPOCH] [--to EPOCH] [--alive 0|1]

run_pool.sh prints `THROTTLE <node> <fields>` from this for every Pi in a cell; compare_pool.py reads the fields back and
recomputes the verdict with verdict() below, so the rule lives here once.

get_throttled bits 0-3 are the state NOW (under-voltage, ARM frequency capped, throttled, soft temperature limit active);
bits 16-19 are the same four, sticky until reboot. A snapshot cannot certify a cell -- the sticky bits may predate it, and
the board cools before the after-snapshot (2026-10-07: about 68 C after, 83 C at the peak) -- so the verdict reads the poll:
  THROTTLED  a now-bit was set in some second, or a sticky bit came on between two seconds (an event between polls);
  UNKNOWN    the poll does not cover the cell: no samples, a failed read, a gap over 3 s, an unreadable line, or a poller
             that was no longer running when the runner stopped it;
  clean      otherwise.
The clock is reported, never judged. Under the ondemand governor it drops to 600 MHz whenever the board waits -- loading a
model from the SD card, the warm-up's page-in, a pipeline gap while the other board computes: on 2026-10-07, 31-86 s of
each 14B cell sat below 1.75 GHz at 52-70 C with no bit set. A thermal cap shows as a bit; active_min_clock_mhz says how
deep it went (the measured clock, in the seconds a bit was set).
Fields: poll samples span_s max_gap_s na_s bad_lines max_temp_c active_s now_bits sticky_first sticky_new active_min_clock_mhz
min_clock_mhz below_1750mhz_s clock [alive] verdict. poll is v2 when the log has v2's header line, else v1; clock says which
clock the two before it read: "measured" (v1 logs: the firmware's, every second) or "governor" (v2 logs: the clock cpu0's
governor asks for).
"""
import sys

MAX_GAP_S = 3


def parse(ln):
    """(epoch, {key: value}) for a sample line; None for a comment or a blank line; False for an unreadable one."""
    ln = ln.strip()
    if not ln or ln.startswith("#"):
        return None
    tok = ln.split()
    kv = dict(t.split("=", 1) for t in tok[1:] if "=" in t)
    if not tok[0].isdigit() or "throttled" not in kv or "temp" not in kv:
        return False
    return int(tok[0]), kv


def _num(v, conv):
    try:
        return conv(v)
    except (TypeError, ValueError):
        return None


def summarize(lines, t0=None, t1=None):
    """The THROTTLE fields of a poll log's lines over [t0, t1] (epoch seconds, inclusive; None = open)."""
    s, bad, poll = [], 0, "v1"
    for ln in lines:
        if ln.startswith("# throttle_poll v2"):
            poll = "v2"
        p = parse(ln)
        if p is None:
            continue
        if p is False:
            bad += 1
            continue
        t, kv = p
        if (t0 is not None and t < t0) or (t1 is not None and t > t1):
            continue
        fk = next((k for k in kv if k.startswith("frequency(")), None)
        x = {"t": t,
             "bits": _num(kv["throttled"], lambda v: int(v, 16) if v.startswith("0x") else None),
             "temp": _num(kv["temp"], lambda v: float(v.rstrip("C").rstrip("'"))),
             "meas": _num(kv[fk], int) / 1e6 if fk and _num(kv[fk], int) is not None else None,
             "gov": _num(kv["gov_khz"], int) / 1e3 if _num(kv.get("gov_khz"), int) is not None else None}
        x["na"] = (x["bits"] is None or x["temp"] is None or (fk is not None and x["meas"] is None)
                   or ("gov_khz" in kv and x["gov"] is None))
        x["v2"] = "gov_khz" in kv
        s.append(x)
    bits = [x for x in s if x["bits"] is not None]
    active = [x for x in bits if x["bits"] & 0xF]
    now = sticky_new = 0
    for x in active:
        now |= x["bits"] & 0xF
    for a, b in zip(bits, bits[1:]):
        sticky_new |= b["bits"] & ~a["bits"] & 0xF0000
    temps = [x["temp"] for x in s if x["temp"] is not None]
    act_clk = [x["meas"] for x in active if x["meas"] is not None]
    clock = "governor" if any(x["v2"] for x in s) else "measured"
    clk = [x["gov" if clock == "governor" else "meas"] for x in s]
    clk = [c for c in clk if c is not None]
    return {"poll": poll, "samples": len(s), "span_s": s[-1]["t"] - s[0]["t"] if s else 0,
            "max_gap_s": max((b["t"] - a["t"] for a, b in zip(s, s[1:])), default=0),
            "na_s": sum(1 for x in s if x["na"]), "bad_lines": bad,
            "max_temp_c": "%.1f" % max(temps) if temps else "NA",
            "active_s": len(active), "now_bits": "0x%x" % now,
            "sticky_first": "0x%x" % (bits[0]["bits"] & 0xF0000) if bits else "NA", "sticky_new": "0x%x" % sticky_new,
            "active_min_clock_mhz": ("%.0f" % min(act_clk)) if act_clk else ("-" if not active else "NA"),
            "min_clock_mhz": "%.0f" % min(clk) if clk else "NA", "below_1750mhz_s": sum(1 for c in clk if c < 1750),
            "clock": clock}


def verdict(f):
    """'THROTTLED', 'UNKNOWN' or 'clean' from THROTTLE fields: summarize()'s, or the strings of a THROTTLE line."""
    def g(k):
        try:
            return int(str(f.get(k, 0)), 0)
        except ValueError:
            return None
    vals = [g(k) for k in ("samples", "na_s", "bad_lines", "max_gap_s", "active_s", "sticky_new")]
    if None in vals or "active_s" not in f or str(f.get("alive", "1")) != "1":
        return "UNKNOWN"
    samples, na, bad, gap, active, sticky_new = vals
    if samples == 0 or na or bad or gap > MAX_GAP_S:
        return "UNKNOWN"
    return "THROTTLED" if active or sticky_new else "clean"


def main(argv):
    if not argv or argv[0].startswith("-"):
        sys.exit(__doc__)
    opts = dict(zip(argv[1::2], argv[2::2]))
    f = summarize(open(argv[0], encoding="utf-8", errors="replace"),
                  int(opts["--from"]) if "--from" in opts else None, int(opts["--to"]) if "--to" in opts else None)
    if "--alive" in opts:
        f["alive"] = opts["--alive"]
    f["verdict"] = verdict(f)
    print(" ".join("%s=%s" % kv for kv in f.items()))


if __name__ == "__main__":
    main(sys.argv[1:])
