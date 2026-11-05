#!/usr/bin/env python3
"""Where the prefill/decode hand-off stops paying as the network slows (topic-form scopes 4 and 6; 2026-10-08). MODELED from
measured parts; derived, not a registration. Reads files only.

    python3 testbed/scripts/pd_network_crossover.py [--netshape testbed/results/netshape_<stamp>] [--out FILE]

The measured parts (Qwen3.5-9B Q4_K_M, the 2026-09-29 hand-off runs, all on gigabit):
- each hand-off's state bytes and its transfer seconds, from the run logs (pd_sweep_*, pd_twouser_*);
- the single-user sweeps (testbed/results/pd_sweep_20260929_010837.log, Pi 5 -> Pi 4; pd_sweep_20260929_171413.log,
  laptop -> Pi 4): per prompt length, each arm's prefill and the hand-off's save and restore;
- the two-user runs (testbed/dataset/twouser_results_20260929.json, 32 tokens back; ..._gen128b.json, 128 back): each arm's
  mean completion time, and the hand-off arm's per-user times.
The model: a hand-off's transfer takes S / B at a link's bulk throughput B; nothing else depends on the link.
- single user, the Pi 4 decoding: the hand-off has the Pi 4 decoding after prefill_fast + save + S/B + restore, against
  prefill_pi4 for the Pi 4 alone, so it pays above B* = S / (prefill_pi4 - prefill_fast - save - restore);
- two users: only user 2's decode waits for the transfer. User 1's first token in the hand-off arm (27.8-28.5 s) equals the
  shared arm's (27.8 s), so the transfer overlapped user 1's prefill. Mean completion rises by (S/B - t0)/2 from the
  measured mean (t0 the measured transfer), and the hand-off pays against an arm that never uses the network above
  B* = S / (t0 + 2 (mean_arm - mean_handoff)); none, if the hand-off loses even at gigabit.
B* is a TCP goodput. With --netshape, each shaped setting's measured goodput against its tc rate (the network study's link
logs) converts it to the tc rate a shaped link would need; without it, B* is reported as goodput only.
"""
import glob, hashlib, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
RES = os.path.join(ROOT, "testbed", "results")
DS = os.path.join(ROOT, "testbed", "dataset")
SWEEPS = {"pi5->pi4": ("pd_sweep_20260929_010837.log", "pi5-alone"), "t580->pi4": ("pd_sweep_20260929_171413.log", "t580-alone")}
TWOUSER = {32: "twouser_results_20260929.json", 128: "twouser_results_20260929_gen128b.json"}
OUT = os.path.join(DS, "pd_network_crossover_20261008.json")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def results(log):
    """[(n_tokens, rep, arm, {result})] from a pd_*.log's ' result ' lines."""
    out = []
    for ln in open(log, encoding="utf-8", errors="replace"):
        m = re.match(r"^\S+ n(\d+) r(\d+) (\S+) result (\{.*\})\s*$", ln)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), m.group(3), json.loads(m.group(4))))
    return out


def handoff_results(log):
    """The two-user log's hand-off result lines: [{state_bytes, transfer_s, user1, user2, ...}]."""
    out = []
    for ln in open(log, encoding="utf-8", errors="replace"):
        m = re.match(r"^\S+ r\d+ handoff result (\{.*\})\s*$", ln)
        if m:
            out.append(json.loads(m.group(1)))
    return out


def crossover_single(S, prefill_slow_s, prefill_fast_s, save_s, restore_s):
    """Goodput (Mbit/s) above which the hand-off has the slow board decoding sooner than its own prefill; None if never."""
    room = prefill_slow_s - prefill_fast_s - save_s - restore_s
    return None if room <= 0 else S * 8 / room / 1e6


def crossover_two(S, t0, mean_handoff, mean_arm):
    """Goodput (Mbit/s) above which the hand-off's mean completion beats an arm that does not use the network; None if the
    hand-off does not beat it even at the measured link."""
    room = t0 + 2 * (mean_arm - mean_handoff)
    return None if mean_arm <= mean_handoff else S * 8 / room / 1e6


def goodput_ratio(netshape_dir):
    """{rate_mbit: measured goodput / tc rate} from the network study's link logs, for the rate-limited settings."""
    import link_to_json as lj
    reg = json.load(open(os.path.join(DS, "pool_predictions_20261008netshape_r2.json")))
    out = {}
    for s in reg["settings"]:
        p = os.path.join(netshape_dir, "link_%s.log" % s["id"])
        if s["rate_mbit"] and os.path.exists(p):
            out[s["rate_mbit"]] = round(lj.parse(p)[2]["mbps"] / s["rate_mbit"], 3)
    return out


def main(argv):
    out = argv[argv.index("--out") + 1] if "--out" in argv else OUT
    ns = argv[argv.index("--netshape") + 1] if "--netshape" in argv else None
    inputs, single, two = {}, {}, {}
    for route, (log, fast_arm) in SWEEPS.items():
        p = os.path.join(RES, log); inputs[log] = sha(p)
        rows = results(p)
        slow = {}
        if route == "t580->pi4":       # the laptop sweep reused the 01:08 sweep's Pi 4-alone arms (identical prompts)
            p0 = os.path.join(RES, SWEEPS["pi5->pi4"][0])
            slow_rows = [r for r in results(p0) if r[2] == "pi4-alone"]
        else:
            slow_rows = [r for r in rows if r[2] == "pi4-alone"]
        for n, _, _, r in slow_rows:
            slow.setdefault(n, []).append(r["prefill_ms"] / 1000)
        for n in sorted({r[0] for r in rows}):
            fast = [r["prefill_ms"] / 1000 for k, _, a, r in rows if k == n and a == fast_arm]
            hand = [r for k, _, a, r in rows if k == n and a.startswith("handoff")]
            if not (fast and hand and slow.get(n)):
                continue
            S = statistics.fmean(h["state_bytes"] for h in hand)
            t0 = statistics.fmean(h["transfer_s"] for h in hand)
            sv = statistics.fmean(h["save_ms"] for h in hand) / 1000
            rs = statistics.fmean(h["restore_ms"] for h in hand) / 1000
            b = crossover_single(S, statistics.fmean(slow[n]), statistics.fmean(fast), sv, rs)
            single.setdefault(route, {})[n] = {
                "state_bytes": round(S), "transfer_s_at_gigabit": round(t0, 3), "goodput_at_gigabit_mbps": round(S * 8 / t0 / 1e6, 1),
                "prefill_pi4_s": round(statistics.fmean(slow[n]), 2), "prefill_fast_s": round(statistics.fmean(fast), 2),
                "save_s": round(sv, 3), "restore_s": round(rs, 3), "crossover_goodput_mbps": round(b, 2) if b else None}
    for gen, f in TWOUSER.items():
        p = os.path.join(DS, f); inputs[f] = sha(p)
        r = json.load(open(p))
        lg = os.path.join(ROOT, r["log"]); inputs[os.path.basename(lg)] = sha(lg)
        hr = handoff_results(lg)
        S = statistics.fmean(h["state_bytes"] for h in hr)
        t0 = statistics.fmean(h["transfer_s"] for h in hr)
        mh = r["arms"]["handoff"]["mean_done_s"]
        two[gen] = {"state_bytes": round(S), "transfer_s_at_gigabit": round(t0, 3), "handoff_mean_done_s": mh, "against": {}}
        for arm in ("shared", "shared_nr", "static"):
            ma = r["arms"][arm]["mean_done_s"]
            b = crossover_two(S, t0, mh, ma) if arm != "static" else None
            two[gen]["against"][arm] = {"arm_mean_done_s": ma,
                                        "crossover_goodput_mbps": round(b, 1) if b else None,
                                        "note": ("the static arm uses no hand-off and its users never share a board; not a "
                                                 "network crossover") if arm == "static" else
                                                ("the hand-off loses even at gigabit" if not b else "")}
    gp = goodput_ratio(ns) if ns else None
    res = {"made": "testbed/scripts/pd_network_crossover.py", "status": "MODELED from measured parts, not measured end to end",
           "model": __doc__.split("The model:")[1].split("B* is")[0].strip(), "inputs_sha256": inputs,
           "single_user_pi4_decoding": single, "two_users": two,
           "netshape_goodput_over_tc_rate": gp}
    json.dump(res, open(out, "w"), indent=1)
    print("single user, the Pi 4 decoding: the hand-off beats the Pi 4's own prefill above (goodput, Mbit/s)")
    for route, d in single.items():
        print("  %-10s %s" % (route, ", ".join("n=%d: %s" % (n, v["crossover_goodput_mbps"]) for n, v in sorted(d.items()))))
    print("two users, 128-token prompts: the hand-off's mean completion beats the arm above (goodput, Mbit/s)")
    for gen, d in two.items():
        print("  %3d tokens back: hand-off %.2f s; %s" % (gen, d["handoff_mean_done_s"], "; ".join(
            "%s %.2f s -> %s" % (a, v["arm_mean_done_s"], v["crossover_goodput_mbps"] or v["note"] or "-")
            for a, v in d["against"].items() if a != "static")))
    if gp:
        print("netshape goodput / tc rate: %s" % gp)
    print("wrote %s" % os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
