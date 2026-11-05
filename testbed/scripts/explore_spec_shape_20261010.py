#!/usr/bin/env python3
"""Exploratory, beside the registered score (spec_shape_predictions_20261010.json judges only time): energy, the laptop's
sync passes, and the Pi 4's thermal state over the shaping run.

    python3 testbed/scripts/explore_spec_shape_20261010.py testbed/results/spec_shape_<stamp>

Per arm and setting:
- gross J per decoded token at the Pi 4 and Jetson outlets, over each request's decode window [t0 + prompt_ms, t1]
  (full_eval.window_energy on the run's strip log, the run's idle window as the idle reference). The laptop's outlet is
  not clean, since it also feeds the controller, so it is reported apart;
- the Pi 4's thermal verdict over each request (full_eval.thermal on the 1 Hz poll).
And every pass of the user's aunsync timer during the run (from the user journal), with the requests it overlapped. The
registration says these passes are reported beside the laptop arm's settings. Writes <dir>/explore.json; it never
touches score.json.
"""
import json, os, re, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))


def aunsync_passes(t_from, t_to):
    """[(start, end)] of aunsync.service runs in [t_from, t_to], from the user journal (epoch seconds)."""
    p = subprocess.run(["journalctl", "--user", "-u", "aunsync.service", "--since", "@%d" % int(t_from), "--until",
                        "@%d" % int(t_to + 60), "-o", "short-unix", "--no-pager"], capture_output=True, text=True)
    starts, out = None, []
    for ln in p.stdout.splitlines():
        m = re.match(r"^([\d.]+) \S+ \S+: (.*)$", ln)
        if not m:
            continue
        t, msg = float(m.group(1)), m.group(2)
        if msg.startswith("Starting aunsync.service"):
            starts = t
        elif starts is not None and (msg.startswith("Finished aunsync.service") or msg.startswith("aunsync.service: Deactivated")
                                     or "Failed with result" in msg):
            out.append((starts, t)); starts = None
    return out


def main(argv):
    import full_eval as F
    d = argv[0]
    reg = json.load(open(os.path.join(ROOT, "testbed", "dataset", "spec_shape_predictions_20261010.json")))
    lines = [json.loads(l) for l in open(os.path.join(d, "run.jsonl"))]
    idle = next((l["idle_before"] for l in lines if "idle_before" in l), None)
    rows = [l for l in lines if l.get("arm") and l.get("timings") and l["timings"].get("predicted_n")]
    poll_p = os.path.join(d, "throttle_pi4.log")
    poll = open(poll_p, errors="replace").readlines() if os.path.exists(poll_p) else None
    outlets = reg["outlets"]                                  # {node: outlet}
    pool = [outlets["pi4"], outlets["jetson"]]
    groups = {}
    for r in rows:
        groups.setdefault((r["arm"], r.get("setting", "s0")), []).append(r)
    passes = aunsync_passes(min(r["t0"] for r in rows), max(r["t1"] for r in rows)) if rows else []
    res = {}
    for (arm, st), rs in sorted(groups.items()):
        j_pool = j_lap = 0.0; tokens = 0; th = []; overlaps = []
        for r in rs:
            t0 = r["t0"] + r["timings"].get("prompt_ms", 0.0) / 1000.0
            if idle:
                w = F.window_energy(d, t0, r["t1"], idle, pool + [outlets["laptop"]])
                j_pool += sum(w[o][0] for o in pool); j_lap += w[outlets["laptop"]][0]
                tokens += r["timings"]["predicted_n"]
            if poll:
                th.append(F.thermal(poll, r["t0"], r["t1"])["verdict"])
            overlaps += ["%s@%d" % (r["id"], a) for a, b in passes if a < r["t1"] and b > r["t0"]]
        res["%s.%s" % (arm, st)] = {"requests": len(rs), "j_per_token_pool": round(j_pool / tokens, 3) if tokens else None,
                                    "j_per_token_laptop_outlet": round(j_lap / tokens, 3) if tokens else None,
                                    "thermal": sorted(set(th)), "aunsync_overlaps": overlaps}
    out = {"run": os.path.relpath(d, ROOT), "aunsync_passes": [[round(a, 1), round(b, 1)] for a, b in passes], "groups": res,
           "note": "exploratory; the registered hypotheses are in score.json"}
    json.dump(out, open(os.path.join(d, "explore.json"), "w"), indent=1)
    for k, v in res.items():
        print("%-14s %s" % (k, v))
    print("aunsync passes during the requests: %d" % len(passes))
    print("wrote %s" % os.path.relpath(os.path.join(d, "explore.json"), ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
