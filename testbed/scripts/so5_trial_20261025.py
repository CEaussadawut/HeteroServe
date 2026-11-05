#!/usr/bin/env python3
"""The switch-off automation's first live trial (study so5, 2026-10-25; the user: "Do the switch-off automation"), registered
before it runs. so4 measured the saving by hand (3.41 J per token, 5.7 %); serve_switch_20261024.py automates it and had run
only as dry runs. This trial runs it as a user would, both of its paths:

    python3 testbed/scripts/so5_trial_20261025.py register --memfile testbed/dataset/memory_snapshot_<stamp>.json
    python3 testbed/scripts/so5_trial_20261025.py run            # -> testbed/results/so5_<stamp>/
    python3 testbed/scripts/so5_trial_20261025.py score <dir>

  A  the planner's own pick for Qwen3-30B-A3B Q6_K on the laptop, the Pi 5 and the Pi 4 (NODES="laptop pi5 pi4": the
     Jetson is left out so that no guard is needed), which leaves the Pi 4 out. `serve_switch_20261024.py start` serves it,
     then shuts the Pi 4 down and cuts its outlet. Three greedy 32-token completions (ignore_eos) are served with the Pi 4
     off; then `serve_switch_20261024.py stop`, the Pi 4 left off.
  B  TinyLlama Q4_0 on laptop 11 + Pi 4 11 (PLACE; head on the host), which needs the Pi 4 while it is off.
     `serve_switch_20261024.py start` powers it up first (strip_switch_20261024.py on-node: ssh, clock, overlay), then
     serves; one 32-token completion; `serve_switch_20261024.py stop`. The cluster ends as it began, the Pi 4 on.
The strip logs all four outlets at 5 Hz for the whole trial (shelly_strip.py log). Only the Pi 4 can be switched: the
wrapper names no other board, and strip_switch_20261024.py refuses any other and the laptop's outlet.
Hypotheses (registered):
  SO5-PLAN    A's served plan is the registered one and leaves the Pi 4 out; B's served plan holds the Pi 4
  SO5-OFF     after A's start the switch log reads "outlet off", and every strip sample of outlet 0 during A's completions
              reads the outlet off at <= 0.1 W
  SO5-SERVE   all three of A's completions return their 32 tokens with the Pi 4 off, and B's one
  SO5-UP      B's start powers the Pi 4 up before serving: the switch log reads "back" (ssh, clock synchronized, overlay up)
  SO5-ONLY    outlets 1-3 (Jetson, Pi 5, laptop) read on in every strip sample of the trial
Reported, not tested: the time from poweroff to the outlet cut, and from the outlet on to the board back.
Writes testbed/dataset/so5_predictions_20261025.json, read-only.
"""
import argparse, csv, datetime, hashlib, json, os, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import serve_switch_20261024 as SW  # noqa: E402

REG = os.path.join(ROOT, "testbed", "dataset", "so5_predictions_20261025.json")
WRAP = [sys.executable, os.path.join(HERE, "serve_switch_20261024.py")]
URL = "http://192.168.1.108:8080/completion"
PROMPT = "The quick brown fox jumps over the lazy dog. Summarise this sentence in five words."
N_TOK, A_REQS = 32, 3
ARMS = {"A": {"MODEL": "qwen3moe_q6k", "CTX": "4096", "NODES": "laptop pi5 pi4", "SERVE_DRAFT": "none"},
        "B": {"MODEL": "tinyllama_q4_0", "CTX": "2048", "PLACE": "laptop:11,pi4:11", "PLACE_HEAD": "host", "SERVE_DRAFT": "none"}}
OUTLET = {"0": "pi4", "1": "jetson", "2": "pi5", "3": "laptop"}
INPUTS = ("testbed/scripts/so5_trial_20261025.py", "testbed/scripts/serve_switch_20261024.py",
          "testbed/scripts/strip_switch_20261024.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
          "testbed/scripts/predict_pool.py", "testbed/scripts/shelly_strip.py", "testbed/energy/strip.json",
          "ThinkPad/config/cluster.json")


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def register(a):
    if os.path.exists(REG):
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(REG, ROOT))
    mem = os.path.relpath(os.path.abspath(a.memfile), ROOT)
    plans = {}
    for k, env in ARMS.items():
        members, r = SW.planned_members(dict(env, MEMFILE=mem))
        plans[k] = {"members": members, "assign": r["assign"], "head_at": r["head_at"], "decode_ms": round(r["decode_ms"], 2)}
    if "pi4" in plans["A"]["members"] or "pi4" not in plans["B"]["members"]:
        raise SystemExit("refused: A must leave the Pi 4 out and B must hold it: %s" % plans)
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/so5_trial_20261025.py", "out_prefix": "so5",
           "what": "the switch-off automation (serve_switch_20261024.py) run live for the first time: a plan without the Pi 4 "
                   "(switched off after serving), then a plan that needs it (powered up before serving)",
           "memfile": mem, "arms": ARMS, "plans": plans, "prompt": PROMPT, "n_tokens": N_TOK, "a_requests": A_REQS,
           **({"conditions": a.conditions} if a.conditions else {}),
           "hypotheses": {
               "SO5-PLAN": {"claim": "A's served plan is the registered one and leaves the Pi 4 out; B's served plan holds the Pi 4"},
               "SO5-OFF": {"max_w": 0.1, "claim": "after A's start the switch log reads 'outlet off', and every strip sample of "
                                                 "outlet 0 during A's completions reads off at <= 0.1 W"},
               "SO5-SERVE": {"claim": "all three of A's completions return 32 tokens with the Pi 4 off, and B's one"},
               "SO5-UP": {"claim": "B's start powers the Pi 4 up before serving: the switch log reads 'back'"},
               "SO5-ONLY": {"claim": "outlets 1-3 read on in every strip sample of the trial"}},
           "inputs_sha256": {f: sha(f) for f in INPUTS + (mem,)}}
    json.dump(reg, open(REG, "w"), indent=1)
    os.chmod(REG, 0o444)
    print("registered %s (sha256 %s): A %s; B %s" % (os.path.relpath(REG, ROOT), sha(os.path.relpath(REG, ROOT))[:16],
                                                    plans["A"]["assign"], plans["B"]["assign"]))


def complete(n):
    body = json.dumps({"prompt": PROMPT, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": False,
                       "ignore_eos": True}).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=900))
    return {"start": t0, "end": time.time(), "predicted_n": r["timings"]["predicted_n"],
            "predicted_per_token_ms": r["timings"]["predicted_per_token_ms"], "text": r.get("content", "")[:120]}


def switch_log_since(t0):
    p = SW.LOG
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("epoch", 0) >= t0:
            out.append(d)
    return out


def run(a):
    reg = json.load(open(REG))
    if os.access(REG, os.W_OK):
        raise SystemExit("refused: the registration is writable")
    if SW.served_members() is not None:
        raise SystemExit("refused: serve_pool.sh state exists")
    out = os.path.join(ROOT, "testbed", "results", "so5_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out)
    log = open(os.path.join(out, "session.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); log.write(line + "\n"); log.flush()
    say("########## so5 -> %s (registration %s)" % (os.path.relpath(out, ROOT), sha(os.path.relpath(REG, ROOT))[:16]))
    res = {"registration": os.path.relpath(REG, ROOT), "arms": {}}
    lp = subprocess.Popen([sys.executable, os.path.join(HERE, "shelly_strip.py"), "log", "--out", os.path.join(out, "strip.csv"),
                           "--hz", "5"], stdout=open(os.path.join(out, "strip_logger.stdout"), "w"),
                          stderr=open(os.path.join(out, "strip_logger.stderr"), "w"))
    time.sleep(10)
    try:
        for k in ("A", "B"):
            env = dict(os.environ, **reg["arms"][k], MEMFILE=reg["memfile"])
            rec = {"start_wall": time.time()}
            res["arms"][k] = rec
            p = subprocess.run(WRAP + ["start"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=3600)
            rec.update({"start_rc": p.returncode, "start_out": p.stdout[-6000:], "start_err": p.stderr[-3000:],
                        "served": SW.served_members(), "switch_log": switch_log_since(rec["start_wall"])})
            say("%s start rc %d, served %s; switch log %s" % (k, p.returncode, rec["served"],
                                                             [(d.get("action"), d.get("result")) for d in rec["switch_log"]]))
            if p.returncode == 0 and rec["served"]:
                rec["requests"] = []
                for i in range(A_REQS if k == "A" else 1):
                    try:
                        c = complete(N_TOK)
                    except Exception as ex:
                        c = {"error": repr(ex)[:200]}
                    rec["requests"].append(c)
                    say("%s request %d: %s" % (k, i, {x: c.get(x) for x in ("predicted_n", "predicted_per_token_ms", "error")}))
            q = subprocess.run(WRAP + ["stop"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
            rec.update({"stop_rc": q.returncode, "stop_out": q.stdout[-3000:], "end_wall": time.time()})
            say("%s stop rc %d" % (k, q.returncode))
            if p.returncode != 0:
                say("%s did not start; the trial stops here" % k)
                break
    finally:
        time.sleep(10)
        lp.terminate(); lp.wait(timeout=30)
        st = subprocess.run([sys.executable, os.path.join(HERE, "strip_switch_20261024.py"), "status", "pi4"],
                            capture_output=True, text=True)
        res["pi4_outlet_at_end"] = st.stdout.strip()
        say("pi4 outlet at the end: %s" % res["pi4_outlet_at_end"])
        json.dump(res, open(os.path.join(out, "measurements.json"), "w"), indent=1)
    say("RUN DONE")
    print(out)


def strip_rows(d):
    """(wall time, {outlet: (output, apower W)}) for every successful poll in shelly_strip.py's CSV (columns wall_time, ok,
    o<i>_output, o<i>_apower)."""
    rows = []
    with open(os.path.join(d, "strip.csv")) as f:
        for r in csv.DictReader(f):
            if r.get("ok") != "1":
                continue
            o = {}
            for i in OUTLET:
                on, w = r.get("o%s_output" % i), r.get("o%s_apower" % i)
                if on not in (None, "") and w not in (None, ""):
                    o[i] = (on.strip().lower() in ("1", "true"), float(w))
            rows.append((float(r["wall_time"]), o))
    return rows


def score(a):
    d = os.path.abspath(a.dir)
    m = json.load(open(os.path.join(d, "measurements.json")))
    reg = json.load(open(os.path.join(ROOT, m["registration"])))
    A, B = m["arms"].get("A", {}), m["arms"].get("B", {})
    rows = strip_rows(d)
    hyp = {}
    ok_plan = (A.get("served") == reg["plans"]["A"]["members"] and "pi4" not in (A.get("served") or ["pi4"])
               and B.get("served") is not None and "pi4" in B.get("served"))
    hyp["SO5-PLAN"] = {"verdict": "HOLDS" if ok_plan else "FAILS", "A": A.get("served"), "B": B.get("served")}
    offs = [x for x in A.get("switch_log", []) if x.get("action") == "off-node"]
    win = [(r["start"], r["end"]) for r in A.get("requests", []) if "start" in r]
    s0 = [o["0"] for t, o in rows if "0" in o and any(a0 <= t <= b0 for a0, b0 in win)]
    hyp["SO5-OFF"] = {"verdict": "HOLDS" if offs and offs[-1].get("result") == "outlet off" and s0
                      and all((not on) and w <= reg["hypotheses"]["SO5-OFF"]["max_w"] for on, w in s0) else "FAILS",
                      "switch": [(x.get("result"), x.get("down_after_s")) for x in offs], "samples": len(s0),
                      "max_w": max((w for _, w in s0), default=None)}
    ra, rb = A.get("requests", []), B.get("requests", [])
    hyp["SO5-SERVE"] = {"verdict": "HOLDS" if len(ra) == 3 and len(rb) == 1 and all(r.get("predicted_n") == N_TOK for r in ra + rb)
                        else "FAILS", "A": [r.get("predicted_n") for r in ra], "B": [r.get("predicted_n") for r in rb]}
    ons = [x for x in B.get("switch_log", []) if x.get("action") == "on-node"]
    hyp["SO5-UP"] = {"verdict": "HOLDS" if ons and ons[-1].get("result") == "back" else "FAILS",
                     "switch": [(x.get("result"), x.get("ssh_after_s"), x.get("synced_after_s")) for x in ons]}
    t0, t1 = A.get("start_wall", 0), B.get("end_wall", A.get("end_wall", 0))
    others = [(i, o[i]) for t, o in rows if t0 <= t <= t1 for i in ("1", "2", "3") if i in o]
    hyp["SO5-ONLY"] = {"verdict": "HOLDS" if others and all(on for _, (on, _) in others) else "FAILS", "samples": len(others),
                       "off_samples": sum(1 for _, (on, _) in others if not on)}
    if not rows:
        for h in ("SO5-OFF", "SO5-ONLY"):
            hyp[h] = {"verdict": "INCOMPLETE", "why": "no strip samples"}
    sc = {"session": os.path.relpath(d, ROOT), "registration": m["registration"], "registration_sha256": sha(m["registration"]),
          "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
          "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items()
                                                    if f.endswith((".py", ".sh")) and sha(f) != h),
          "requests": {"A": ra, "B": rb}, "pi4_outlet_at_end": m.get("pi4_outlet_at_end"), "hypotheses": hyp}
    json.dump(sc, open(os.path.join(d, "score.json"), "w"), indent=1)
    print(json.dumps({k: v["verdict"] for k, v in hyp.items()}))
    print(json.dumps(hyp, indent=1)[:3000])


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("register"); r.add_argument("--memfile", required=True); r.add_argument("--conditions", default="")
    sp.add_parser("run")
    s = sp.add_parser("score"); s.add_argument("dir")
    a = ap.parse_args()
    {"register": register, "run": run, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
