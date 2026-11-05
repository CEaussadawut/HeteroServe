#!/usr/bin/env python3
"""Every idle board switched off, live (study so6, 2026-11-01; the scheduler review's item 7, the user: "do item 1 to 4 then
6-7"), registered before it runs. so5 ran the Pi 4's switch-off live; serve_pool.sh now switches the Pi 4, the Pi 5 and the
Jetson (strip_switch_20261101.py: several at once, the Jetson halted rather than powered off, its power guard started after it
boots). This trial runs serve_pool.sh as a person's serve runs (SERVE_SWITCH=on, SERVE_PLANNER=v4, the four nodes, the boards'
memory read live), both of its paths:

    python3 testbed/scripts/so6_trial_20261101.py register
    python3 testbed/scripts/so6_trial_20261101.py run            # -> testbed/results/so6_<stamp>/
    python3 testbed/scripts/so6_trial_20261101.py score <dir>

  A  serve_pool.sh's default model, Qwen3-30B-A3B Q3_K_M, at ctx 4,096: the planner's pick is the laptop alone, so after
     READY the Pi 4, the Pi 5 and the Jetson are shut down and their outlets cut. Three greedy 32-token completions
     (ignore_eos) are served with all three off; then serve_pool.sh stop, the boards left off.
  B  Qwen3.6-35B-A3B Q6_K at ctx 4,096, which needs every node: serve_pool.sh powers the three boards up first, in parallel
     (ssh, clock synchronized, address up; the Jetson's guard), then plans on their live memory and serves; one 32-token
     completion; stop. The cluster ends as it began, every board on.
The strip logs all four outlets at 5 Hz for the whole trial (shelly_strip.py log). The laptop's outlet is never switched:
strip_switch_20261101.py refuses it (strip.json's do_not_switch_off).
Hypotheses (registered):
  SO6-PLAN    A's served plan is the laptop alone; B's served plan holds all four nodes
  SO6-OFF     after A's start the switch log reads "outlet off" for the Pi 4, the Pi 5 and the Jetson, and every strip sample of
              outlets 0, 1 and 2 during A's completions reads the outlet off at <= 0.1 W
  SO6-SERVE   all three of A's completions return their 32 tokens with the three boards off, and B's one
  SO6-UP      B's start powers all three up before serving: the switch log reads "back" for each, and the Jetson's guard "ok"
  SO6-LAPTOP  outlet 3 (the laptop) reads on in every strip sample of the trial
Reported, not tested: each board's time from shutdown to the outlet cut and from the outlet on to the board back, and the
wall power during A's completions against the boards' idle.
Writes testbed/dataset/so6_predictions_20261101.json, read-only.
"""
import argparse, csv, datetime, hashlib, json, os, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]

REG = os.path.join(ROOT, "testbed", "dataset", "so6_predictions_20261101.json")
SERVE = ["bash", os.path.join(HERE, "serve_pool.sh")]
STATE = os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")
SWITCH_LOG = os.path.join(ROOT, "testbed", "results", "serve_switch.jsonl")
URL = "http://192.168.1.108:8080/completion"
PROMPT = "The quick brown fox jumps over the lazy dog. Summarise this sentence in five words."
N_TOK, A_REQS = 32, 3
BOARDS = ("pi4", "pi5", "jetson")
COMMON = {"SERVE_SWITCH": "on", "SERVE_PLANNER": "v4", "NODES": "laptop pi5 pi4 jetson", "SERVE_DRAFT": "none",
          "HETEROSERVE_RESERVE_MB": "", "CTX": "4096", "MEMFILE": "", "PLACE": "", "PLAN_DEPTH": ""}
ARMS = {"A": dict(COMMON, MODEL="qwen3moe"), "B": dict(COMMON, MODEL="qwen36moe_q6k")}
OUTLET = {"0": "pi4", "1": "jetson", "2": "pi5", "3": "laptop"}
INPUTS = ("testbed/scripts/so6_trial_20261101.py", "testbed/scripts/strip_switch_20261101.py", "testbed/scripts/serve_pool.sh",
          "testbed/scripts/serve_defaults.sh", "testbed/scripts/serve_plan_20261101.py", "testbed/scripts/predict_pool.py",
          "testbed/scripts/shelly_strip.py", "testbed/scripts/jetson_power_guard.sh", "testbed/energy/strip.json",
          "ThinkPad/config/cluster.json")


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def planned(env):
    """The members serve_pool.sh would plan on for this arm now, by the same call (serve_plan_20261101: live memory where a
    board answers, the newest snapshot where it does not, the grid, the depth, PLANNER_V4)."""
    import predict_pool as P, serve_plan_20261101 as SP
    os.environ["HETEROSERVE_RESERVE_MB"] = ""
    nodes = env["NODES"].split()
    mem, src = SP.live_memory(nodes, json.load(open(SP.newest_snapshot())))
    ctx = int(env["CTX"])
    r, why = SP.choose(env["MODEL"], nodes, ctx, SP.on_grid(mem), SP.plan_depth(ctx, {}), None, **P.PLANNER_V4)
    return [n for n, _ in r["assign"]], r, src


def register(a):
    if os.path.exists(REG):
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(REG, ROOT))
    plans = {}
    for k, env in ARMS.items():
        members, r, src = planned(env)
        plans[k] = {"members": members, "assign": r["assign"], "head_at": r["head_at"], "decode_ms": round(r["decode_ms"], 2),
                    "memory_source": src}
    if plans["A"]["members"] != ["laptop"] or sorted(plans["B"]["members"]) != sorted(["laptop"] + list(BOARDS)):
        raise SystemExit("refused: A must be the laptop alone and B must hold every node: %s" % plans)
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/so6_trial_20261101.py", "out_prefix": "so6",
           "what": "every idle board switched off, live: serve_pool.sh as a person serves (SERVE_SWITCH=on, SERVE_PLANNER=v4), a "
                   "plan on the laptop alone (the three boards shut down and cut after serving), then a plan that needs all "
                   "four (the three powered up first, the Jetson's guard started)",
           "arms": ARMS, "plans_at_registration": plans, "prompt": PROMPT, "n_tokens": N_TOK, "a_requests": A_REQS,
           **({"conditions": a.conditions} if a.conditions else {}),
           "hypotheses": {
               "SO6-PLAN": {"claim": "A's served plan is the laptop alone; B's served plan holds all four nodes"},
               "SO6-OFF": {"max_w": 0.1, "claim": "after A's start the switch log reads 'outlet off' for the Pi 4, the Pi 5 and the "
                                                 "Jetson, and every strip sample of outlets 0-2 during A's completions reads "
                                                 "off at <= 0.1 W"},
               "SO6-SERVE": {"claim": "all three of A's completions return 32 tokens with the three boards off, and B's one"},
               "SO6-UP": {"claim": "B's start powers all three up before serving: 'back' for each, and the Jetson's guard 'ok'"},
               "SO6-LAPTOP": {"claim": "outlet 3 reads on in every strip sample of the trial"}},
           "inputs_sha256": {f: sha(f) for f in INPUTS}}
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
    if not os.path.exists(SWITCH_LOG):
        return []
    out = []
    for line in open(SWITCH_LOG):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("epoch", 0) >= t0:
            out.append(d)
    return out


def served():
    """[HOST] + WORKERS from serve_pool.sh's state file, or None."""
    if not os.path.exists(STATE):
        return None
    import shlex
    kv = {}
    for line in open(STATE):
        k, _, v = line.rstrip("\n").partition("=")
        if k in ("HOST", "WORKERS"):
            kv[k] = " ".join(shlex.split(v)) if v else ""
    return [kv.get("HOST")] + kv.get("WORKERS", "").split() if kv.get("HOST") else None


def outlets_now():
    p = subprocess.run([sys.executable, os.path.join(HERE, "strip_switch_20261101.py"), "status"] + list(BOARDS),
                       capture_output=True, text=True)
    return p.stdout.strip()


def run(a):
    reg = json.load(open(REG))
    if os.access(REG, os.W_OK):
        raise SystemExit("refused: the registration is writable")
    if served() is not None:
        raise SystemExit("refused: serve_pool.sh state exists")
    out = os.path.join(ROOT, "testbed", "results", "so6_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out)
    log = open(os.path.join(out, "session.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); log.write(line + "\n"); log.flush()
    say("########## so6 -> %s (registration %s)" % (os.path.relpath(out, ROOT), sha(os.path.relpath(REG, ROOT))[:16]))
    res = {"registration": os.path.relpath(REG, ROOT), "arms": {}, "outlets_at_start": outlets_now()}
    say("outlets at the start: %s" % res["outlets_at_start"].replace("\n", " | "))
    lp = subprocess.Popen([sys.executable, os.path.join(HERE, "shelly_strip.py"), "log", "--out", os.path.join(out, "strip.csv"),
                           "--hz", "5"], stdout=open(os.path.join(out, "strip_logger.stdout"), "w"),
                          stderr=open(os.path.join(out, "strip_logger.stderr"), "w"))
    time.sleep(10)
    try:
        for k in ("A", "B"):
            env = dict(os.environ, **reg["arms"][k])
            for x in ("MEMFILE", "PLACE", "PLAN_DEPTH"):          # empty in the registration: unset here, as a person's shell
                env.pop(x, None)
            rec = {"start_wall": time.time()}
            res["arms"][k] = rec
            p = subprocess.run(SERVE + ["start"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=3600)
            rec.update({"start_rc": p.returncode, "start_out": p.stdout[-8000:], "start_err": p.stderr[-3000:],
                        "served": served(), "switch_log": switch_log_since(rec["start_wall"])})
            say("%s start rc %d, served %s; switch log %s" % (k, p.returncode, rec["served"],
                                                             [(d.get("action"), d.get("board"), d.get("result")) for d in rec["switch_log"]]))
            if p.returncode == 0 and rec["served"]:
                rec["requests"] = []
                for i in range(A_REQS if k == "A" else 1):
                    try:
                        c = complete(N_TOK)
                    except Exception as ex:
                        c = {"error": repr(ex)[:200]}
                    rec["requests"].append(c)
                    say("%s request %d: %s" % (k, i, {x: c.get(x) for x in ("predicted_n", "predicted_per_token_ms", "error")}))
            q = subprocess.run(SERVE + ["stop"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
            rec.update({"stop_rc": q.returncode, "stop_out": q.stdout[-3000:], "end_wall": time.time()})
            say("%s stop rc %d" % (k, q.returncode))
            if p.returncode != 0:
                say("%s did not start; the trial stops here" % k)
                break
    finally:
        time.sleep(10)
        lp.terminate(); lp.wait(timeout=30)
        res["outlets_at_end"] = outlets_now()
        say("outlets at the end: %s" % res["outlets_at_end"].replace("\n", " | "))
        json.dump(res, open(os.path.join(out, "measurements.json"), "w"), indent=1)
    say("RUN DONE")
    print(out)


def strip_rows(d):
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
    ok_plan = A.get("served") == ["laptop"] and B.get("served") is not None and sorted(B["served"]) == sorted(["laptop"] + list(BOARDS))
    hyp["SO6-PLAN"] = {"verdict": "HOLDS" if ok_plan else "FAILS", "A": A.get("served"), "B": B.get("served")}
    offs = {x.get("board"): x for x in A.get("switch_log", []) if x.get("action") == "off-node"}
    win = [(r["start"], r["end"]) for r in A.get("requests", []) if "start" in r]
    samples = {i: [o[i] for t, o in rows if i in o and any(a0 <= t <= b0 for a0, b0 in win)] for i in ("0", "1", "2")}
    mw = reg["hypotheses"]["SO6-OFF"]["max_w"]
    hyp["SO6-OFF"] = {"verdict": "HOLDS" if all(offs.get(b, {}).get("result") == "outlet off" for b in BOARDS)
                      and all(samples[i] and all((not on) and w <= mw for on, w in samples[i]) for i in samples) else "FAILS",
                      "switch": {b: (offs.get(b, {}).get("result"), offs.get(b, {}).get("down_after_s")) for b in BOARDS},
                      "samples": {OUTLET[i]: len(v) for i, v in samples.items()},
                      "max_w": {OUTLET[i]: max((w for _, w in v), default=None) for i, v in samples.items()}}
    ra, rb = A.get("requests", []), B.get("requests", [])
    hyp["SO6-SERVE"] = {"verdict": "HOLDS" if len(ra) == A_REQS and len(rb) == 1 and all(r.get("predicted_n") == N_TOK for r in ra + rb)
                        else "FAILS", "A": [r.get("predicted_n") for r in ra], "B": [r.get("predicted_n") for r in rb]}
    ons = {x.get("board"): x for x in B.get("switch_log", []) if x.get("action") == "on-node"}
    gd = [x for x in B.get("switch_log", []) if x.get("action") == "guard"]
    hyp["SO6-UP"] = {"verdict": "HOLDS" if all(ons.get(b, {}).get("result") == "back" for b in BOARDS)
                     and gd and gd[-1].get("result") == "ok" else "FAILS",
                     "switch": {b: (ons.get(b, {}).get("result"), ons.get(b, {}).get("ssh_after_s"), ons.get(b, {}).get("synced_after_s"))
                                for b in BOARDS}, "guard": [(x.get("started"), x.get("result")) for x in gd]}
    t0, t1 = A.get("start_wall", 0), B.get("end_wall", A.get("end_wall", 0))
    lap = [o["3"] for t, o in rows if t0 <= t <= t1 and "3" in o]
    hyp["SO6-LAPTOP"] = {"verdict": "HOLDS" if lap and all(on for on, _ in lap) else "FAILS", "samples": len(lap),
                         "off_samples": sum(1 for on, _ in lap if not on)}
    if not rows:
        for h in ("SO6-OFF", "SO6-LAPTOP"):
            hyp[h] = {"verdict": "INCOMPLETE", "why": "no strip samples"}
    wall = [sum(w for on, w in o.values()) for t, o in rows if any(a0 <= t <= b0 for a0, b0 in win) and len(o) == 4]
    sc = {"session": os.path.relpath(d, ROOT), "registration": m["registration"], "registration_sha256": sha(m["registration"]),
          "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
          "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items()
                                                    if f.endswith((".py", ".sh")) and sha(f) != h),
          "requests": {"A": ra, "B": rb}, "outlets_at_end": m.get("outlets_at_end"),
          "reported": {"a_wall_w_mean": round(sum(wall) / len(wall), 2) if wall else None,
                       "off_times_s": {b: offs.get(b, {}).get("down_after_s") for b in BOARDS},
                       "on_times_s": {b: ons.get(b, {}).get("synced_after_s") for b in BOARDS}},
          "hypotheses": hyp}
    json.dump(sc, open(os.path.join(d, "score.json"), "w"), indent=1)
    print(json.dumps({k: v["verdict"] for k, v in hyp.items()}))
    print(json.dumps(sc["reported"]))
    print(json.dumps(hyp, indent=1)[:3500])


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("register"); r.add_argument("--conditions", default="")
    sp.add_parser("run")
    s = sp.add_parser("score"); s.add_argument("dir")
    a = ap.parse_args()
    {"register": register, "run": run, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
