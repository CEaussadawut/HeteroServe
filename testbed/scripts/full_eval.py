#!/usr/bin/env python3
"""The full evaluation's one-command harness (2026-10-08): several placements of one model, each SERVED and timed per
token, in interleaved rounds, with the strip logging energy throughout.

    python3 testbed/scripts/full_eval.py register --name NAME --model KEY --nodes "pi4 jetson" \\
        --placements "engine,pooled=pi4:1+jetson:21/last,halda=pi4:1+jetson:21/host,even=pi4:11+jetson:11/last" \\
        [--rounds 5] [--reqs 4] [--n 128] [--gen 96] [--ctx 4096] [--rtt-trips 1|2] [--warm-n 512] [--warm-gen 32] [--note TEXT]
        [--below-headroom] [--dry-run]
    python3 testbed/scripts/full_eval.py run testbed/dataset/full_eval_<name>.json --outlets pi4=0,jetson=2 \\
        [--idle-before 180] [--idle-after 60]
    python3 testbed/scripts/full_eval.py summary testbed/results/full_eval_<name>_<stamp>

register writes the plan, read-only, before anything runs:
- each placement, with its predicted decode and prefill ms/token (predict_pool) and gross J/token (energy_model);
- "engine" is the engine's own pick (predict_pool.plan_host_choice: the host among the boards holding the file);
- a named placement is "label=node:blocks+node:blocks/head", the host first, head "host" or "last";
- the order: round r takes the placements rotated by r, so a drift through the session does not fall on one placement;
- --rtt-trips: network round trips per worker per token in the predictions (2, every registration before 2026-10-09;
  1, the registered rule RTT1, testbed/dataset/rule_rtt1_20261009.json). The other value's decode and prefill are
  recorded beside each prediction (predicted_rtt_alt), so the rule can be scored on the same cells;
- each placement's memory on every board (predict_pool.memory), charged as serve_pool.sh will gate it -- the weights read
  into memory, so the host pays its token embedding (load_mode "none", 2026-10-09) -- against memory_at_rest.json. A
  placement that does not fit is refused; one that fits with less than the engine's default headroom on a board
  (placement.EXACT_FILL_HEADROOM_MB, 256 MiB) is refused unless --below-headroom is given, which the registration records.
  2026-10-09's p32host was registered 159 MiB inside that headroom by the old charge and OOM-killed in all five rounds;
- --warm-n / --warm-gen: the warm-up request's prompt and reply (default qr3_tpot's 512 / 32; a 14B model on the Pi 4
  prefills 512 tokens in about 7 minutes, and serve_pool.sh reads the weights into memory, so nothing is left to page in).
run:
- the model's sha256 on every host first (testbed/models.md);
- the strip logged at 5 Hz for the whole session (shelly_strip.py, read-only), with an idle window first and last;
- the Jetson's clocks locked for the WHOLE session (stored and restored), so every window waits at the same floor;
- every Pi in the pool polled at 1 Hz for the whole session (throttle_poll.sh v2, as run_pool.sh does), its log copied into
  the run directory (throttle_<node>.log);
- the laptop, when it is a node (2026-10-11): refused unless it is on AC with no battery charging (outlet 1 would meter the
  charge), and polled at 1 Hz here (laptop_poll.sh -> laptop_poll.log: load, clock, package temperature, throttle counters,
  free memory, batteries);
- each (round, placement): serve_pool.sh start with PLACE/PLACE_HEAD (or the engine's own plan), one warm-up request
  (qr3_tpot.py's 512/32, never scored), then --reqs requests, each its own --n-token prompt (qr3_tpot.py's GPL-3 text),
  --gen tokens, greedy, ignore_eos, streamed with timings_per_token; then serve_pool.sh stop.
summary, per placement:
- server and client TPOT P50 / P95 / P99 over every round, with 95 % CIs of P50 and P95 from a two-level bootstrap
  (rounds, then requests within them: a round is the independent unit, one load of the model);
- TTFT P50;
- gross J per decoded token over each request's decode window, summed over EVERY pool board's outlet (a board the
  placement leaves out still draws its idle power) and over the placement's own boards (as if the rest were off);
- the laptop's state over each request (when it is a node): its hottest package reading, the throttle counters' rise, any
  window with a battery charging, its highest load (the desktop's activity shows there);
- a thermal verdict per placement: throttle_summary.py's verdict over each request's whole window, on each Pi the placement
  uses, the worst one kept (THROTTLED, then UNKNOWN, then clean; "n/a" for a placement without a Pi). A window the poll
  does not reach within 3 s of either edge is UNKNOWN.
"""
import argparse, datetime, hashlib, json, math, os, random, re, signal, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                    # noqa: E402
from control.placement import EXACT_FILL_HEADROOM_MB as HEADROOM_MB   # noqa: E402
import energy_model as EM                                   # noqa: E402
import qr3_tpot as Q                                        # noqa: E402
import throttle_summary as TS                               # noqa: E402

STRIP = os.path.join(HERE, "shelly_strip.py")
LOAD_MODE = "none"        # serve_pool.sh serves with -lm none; registrations price it so from 2026-10-09
SERVE = os.path.join(HERE, "serve_pool.sh")
POLL = os.path.join(HERE, "throttle_poll.sh")
LPOLL = os.path.join(HERE, "laptop_poll.sh")     # the laptop's own 1 Hz poll, when it is a node (2026-10-11)
WORST = {"THROTTLED": 3, "UNKNOWN": 2, "clean": 1}
SHA = {"tinyllama": "9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0",
       **{k: v[1] for k, v in Q.MODEL_FILE.items()}}
BOOT, BOOT_SEED = 4000, 20261008


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def parse_placement(spec):
    """'label=pi4:1+jetson:21/last' -> (label, [(node, blocks)], head); 'engine' -> ('engine', None, None)."""
    if spec == "engine":
        return "engine", None, None
    label, rest = spec.split("=", 1)
    body, head = rest.split("/")
    return label, [(n, int(k)) for n, k in (x.split(":") for x in body.split("+"))], head


def thermal(lines, t0, t1):
    """throttle_summary's fields and verdict over [t0, t1] (laptop epoch seconds; the boards keep NTP time from it), with
    one check the runner's whole-cell use never needed: the poll must reach within MAX_GAP_S of both edges, or a poll that
    died (or started late) inside the window would pass on the seconds it did cover."""
    a, b = math.floor(t0), math.ceil(t1)
    f = TS.summarize(lines, a, b)
    ts = [p[0] for p in (TS.parse(ln) for ln in lines) if p and a <= p[0] <= b]
    f["verdict"] = TS.verdict(f)
    if f["verdict"] != "UNKNOWN" and (not ts or ts[0] - t0 > TS.MAX_GAP_S or t1 - ts[-1] > TS.MAX_GAP_S):
        f["verdict"] = "UNKNOWN"
    return f


# ------------------------------------------------------------------ register
def register(a):
    out = os.path.join(ROOT, "testbed", "dataset", "full_eval_%s.json" % a.name)
    if os.path.exists(out) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(out, ROOT))
    nodes = a.nodes.split()
    # --memfile (2026-10-21): plan against a dated snapshot (memory_snapshot.py), not the at-rest file; run then serves with
    # serve_pool.sh MEMFILE= the same file, so the engine's own plan is the registered one (the 30B Q6_K's first run planned on
    # the laptop's at-rest 27,785 MiB and was refused at 26,495)
    memfile = getattr(a, "memfile", None)
    mem = json.load(open(memfile or os.path.join(ROOT, "testbed", "dataset", "memory_at_rest.json")))
    # --reserve (2026-10-21): this registration's own reserve for named nodes (predict_pool.reserve_overrides), e.g. the user's
    # 7 GiB laptop reserve for the Qwen3-32B runs only; recorded, and run serves with the same
    reserve = getattr(a, "reserve", None) or ""
    os.environ["HETEROSERVE_RESERVE_MB"] = reserve
    reserve_mb = P.reserve_overrides()
    power = EM.power_table()
    shapes, pairs, C = EM._inputs(P.MODELS[a.model][1])
    configs, unfit, tight = {}, [], []
    for spec in a.placements.split(","):
        label, assign, head = parse_placement(spec)
        r = (P.plan_host_choice(a.model, nodes, a.ctx, mem, rtt_trips=a.rtt_trips, load_mode=LOAD_MODE) if assign is None
             else P.plan_fixed(a.model, assign, head, a.ctx, mem, rtt_trips=a.rtt_trips, load_mode=LOAD_MODE))
        memo = P.memory(r["assign"], r["q"], mem, r["devices"], r["model"], head_at=r["head_at"])
        for nd, m in memo.items():
            if m["margin_mib"] < 0:
                unfit.append("%s: %s needs %d + %d MiB of %d" % (label, nd, m["need_mib"], m["reserve_mib"], m["available_mib"]))
            elif m["margin_mib"] < HEADROOM_MB:
                tight.append("%s: %s keeps %d MiB" % (label, nd, m["margin_mib"]))
        e = {ph: EM.energy(r["assign"], r["q"], r["C"], r["links"], power, ph, head_at=r["head_at"], rtt_trips=a.rtt_trips)
             for ph in ("tg", "pp")}
        alt = 3 - a.rtt_trips
        d_alt, p_alt = P.predict(r["assign"], r["q"], r["C"], r["links"], head_at=r["head_at"], rtt_trips=alt)
        idle_on = [n for n in nodes if n not in dict(r["assign"])]
        g = e["tg"]["gross_j"]
        configs[label] = {"assign": r["assign"], "head": "host" if r["head_at"] == 0 else "last", "host": r["host"],
                          "place_env": None if assign is None else ",".join("%s:%d" % x for x in r["assign"]),
                          "predicted": {"decode_ms": round(r["decode_ms"], 1), "prefill_ms": round(r["prefill_ms"], 1),
                                        "decode_gross_j_plan_boards": round(g, 4),
                                        "decode_gross_j_all_boards": round(g + sum(power[n]["idle"] for n in idle_on)
                                                                           * e["tg"]["T_ms"] / 1000, 4)},
                          "predicted_rtt_alt": {"rtt_trips": alt, "decode_ms": round(d_alt, 1), "prefill_ms": round(p_alt, 1)},
                          "memory": memo,
                          "considered": r["considered"]}
    if unfit:
        raise SystemExit("refused: does not fit at rest: %s" % "; ".join(unfit))
    if tight and not a.below_headroom:
        raise SystemExit("refused: inside the engine's %d MiB headroom: %s; --below-headroom registers it anyway"
                         % (HEADROOM_MB, "; ".join(tight)))
    labels = list(configs)
    order = [labels[r % len(labels):] + labels[:r % len(labels)] for r in range(a.rounds)]
    reg = {"registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "name": a.name,
           "model": a.model, "nodes": nodes, "ctx": a.ctx, "rounds": a.rounds, "reqs": a.reqs, "n": a.n, "gen": a.gen,
           "rtt_trips": a.rtt_trips, "warm_n": a.warm_n, "warm_gen": a.warm_gen, "load_mode": LOAD_MODE,
           **({"below_headroom": True} if a.below_headroom else {}),
           **({"memfile": os.path.relpath(os.path.abspath(memfile), ROOT)} if memfile else {}),
           **({"reserve_mb": reserve_mb} if reserve_mb else {}),
           **({"note": a.note} if a.note else {}),
           "order": order, "configs": configs, "power_table": power,
           "inputs_sha256": {f: sha(os.path.join(HERE, f)) for f in ("full_eval.py", "serve_pool.sh", "predict_pool.py",
                                                                      "energy_model.py", "qr3_tpot.py", "shelly_strip.py",
                                                                      "laptop_poll.sh")},
           "hypotheses": [
               {"id": "FE-TPOT", "claim": "the placement with the lowest predicted decode has the lowest server TPOT P50 and P95, "
                                          "its 95 % CI clear of every other placement's"},
               {"id": "FE-ENERGY", "claim": "the placement with the lowest predicted gross J/token (every pool board powered) "
                                            "measures the lowest"},
               {"id": "FE-ACC", "claim": "every placement's measured mean server TPOT within 10 % of its predicted decode ms"}]}
    for lab, c in configs.items():
        print("%-8s %-28s head %-4s host %-6s decode %6.1f ms  prefill %5.1f ms  J/token %.3f (all boards %.3f)" % (
            lab, " + ".join("%s %d" % x for x in c["assign"]), c["head"], c["host"], c["predicted"]["decode_ms"],
            c["predicted"]["prefill_ms"], c["predicted"]["decode_gross_j_plan_boards"], c["predicted"]["decode_gross_j_all_boards"]))
        print("         memory: %s" % ", ".join("%s needs %d + %d of %d MiB, margin %d" % (
            n, m["need_mib"], m["reserve_mib"], m["available_mib"], m["margin_mib"]) for n, m in c["memory"].items()))
    print("order:", order)
    if a.dry_run:
        print("dry run: nothing written")
        return reg
    json.dump(reg, open(out, "w"), indent=1)
    os.chmod(out, 0o444)
    print("registered %s at %s, sha256 %s (read-only)" % (os.path.relpath(out, ROOT), reg["registered"], sha(out)[:16]))


# ------------------------------------------------------------------ run
def laptop_power_state():
    """{"ac": "1"|"0", "bat0": "<status>/<capacity>", ...} of this laptop now."""
    ps = "/sys/class/power_supply"
    st = {}
    for d in sorted(os.listdir(ps)) if os.path.isdir(ps) else []:
        f = lambda k: open(os.path.join(ps, d, k)).read().strip() if os.path.exists(os.path.join(ps, d, k)) else "-"
        if d.startswith("AC"):
            st["ac"] = f("online")
        elif d.startswith("BAT"):
            st[d.lower()] = "%s/%s" % (f("status").replace(" ", "_"), f("capacity"))
    return st


def laptop_window(lines, t0, t1):
    """The laptop poll over [t0, t1]: hottest package C, throttle-counter rise, highest load, any battery charging."""
    rows = []
    for ln in lines:
        kv = dict(x.split("=", 1) for x in ln.split() if "=" in x)
        try:
            t = float(kv["t"])
        except (KeyError, ValueError):
            continue
        if t0 - 1.5 <= t <= t1 + 1.5:
            rows.append(kv)
    if not rows:
        return {"samples": 0}
    num = lambda k: [float(r[k]) for r in rows if r.get(k, "-").replace(".", "", 1).isdigit()]
    thr = lambda k: (max(num(k)) - min(num(k))) if num(k) else None
    return {"samples": len(rows), "max_pkg_c": max(num("pkg")) if num("pkg") else None, "max_load": max(num("load")),
            "throttle_rise": (thr("core_thr") or 0) + (thr("pkg_thr") or 0),
            "charging": any("Charging" in r.get(b, "") for r in rows for b in ("bat0", "bat1"))}


def run(a):
    reg = json.load(open(a.registration))
    outlets = {k: int(v) for k, v in (x.split("=") for x in a.outlets.split(","))}
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(ROOT, "testbed", "results", "full_eval_%s_%s" % (reg["name"], stamp))
    os.makedirs(out)
    rec = open(os.path.join(out, "run.jsonl"), "a")

    def log(kind, obj):
        rec.write(json.dumps({"t": time.time(), "kind": kind, "data": obj}) + "\n"); rec.flush()
        print("%s %s %s" % (datetime.datetime.now().strftime("%H:%M:%S"), kind, json.dumps(obj)[:200]), flush=True)

    log("start", {"registration": os.path.relpath(a.registration, ROOT), "registration_sha256": sha(a.registration),
                  "outlets": outlets, "args": vars(a)})
    path = P.MODELS[reg["model"]][2]
    for h in sorted({c["host"] for c in reg["configs"].values()}):
        got = Q.ssh(h, "sha256sum %s | cut -d' ' -f1" % path).strip()
        log("model_sha256", {"host": h, "sha256": got, "ok": got == SHA.get(reg["model"])})
        if got != SHA.get(reg["model"]):
            raise SystemExit("FATAL: %s on %s is not the pinned bytes" % (path, h))
    lock = "jetson" in reg["nodes"]
    logger, polls, lpoll = None, {}, None
    try:
        if "laptop" in reg["nodes"]:      # a node since 2026-10-11: on AC, no battery charging, and its own poll
            st = laptop_power_state()
            log("laptop power", st)
            if st.get("ac") != "1" or any(v.startswith("Charging") for k, v in st.items() if k.startswith("bat")):
                raise SystemExit("FATAL: the laptop must be on AC with no battery charging (outlet 1 would meter it): %s" % st)
            lpoll = subprocess.Popen(["bash", LPOLL, os.path.join(out, "laptop_poll.log")], stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            log("laptop poll start", {"pid": lpoll.pid})
        if lock:
            r = Q.ssh("jetson", "m=$(cut -d' ' -f1 /var/lib/nvpmodel/status); [ \"$m\" = pmode:0000 ] || { echo NOT-MAXN; exit 0; }; "
                                "mkdir -p ~/heteroserve/run && sudo -n jetson_clocks --store ~/heteroserve/run/jetson_clocks_fe_%s.stored "
                                "&& sudo -n jetson_clocks && echo LOCKED" % stamp, check=False).strip()
            log("jetson clocks", r)
            if r != "LOCKED":
                raise SystemExit("FATAL: could not lock the Jetson's clocks: %s" % r)
        logger = subprocess.Popen([sys.executable, STRIP, "log", "--out", os.path.join(out, "strip.csv"), "--hz", "5"],
                                  stdout=open(os.path.join(out, "strip_logger.stdout"), "w"),
                                  stderr=open(os.path.join(out, "strip_logger.stderr"), "w"))
        time.sleep(20)
        if logger.poll() is not None:
            raise SystemExit("the strip logger exited early")
        for n in [n for n in reg["nodes"] if n.startswith("pi")]:      # run_pool.sh's poll, for the whole session
            if subprocess.run(["scp", "-q", POLL, "%s:throttle_poll.sh" % n], capture_output=True).returncode:
                raise SystemExit("FATAL: could not copy throttle_poll.sh to %s" % n)
            pid = Q.ssh(n, "mkdir -p ~/logs; setsid nohup bash ~/throttle_poll.sh ~/logs/throttle_fe_%s.log > /dev/null 2>&1 "
                           "< /dev/null & echo $!" % stamp).strip()
            polls[n] = pid
            log("throttle poll start", {"node": n, "pid": pid})
        log("idle-before start", {}); time.sleep(a.idle_before); log("idle-before end", {})
        raw = open(Q.GPL3, "rb").read().decode("utf-8")
        for r_i, labels in enumerate(reg["order"], 1):
            for lab in labels:
                c = reg["configs"][lab]
                env = dict(os.environ, MODEL=reg["model"], CTX=str(reg["ctx"]), NODES=" ".join(reg["nodes"]), CLOCKS="locked",
                           SERVE_DRAFT="none")   # the registered placements, never serve_pool.sh's default draft (2026-10-10)
                if c["place_env"]:
                    env.update(PLACE=c["place_env"], PLACE_HEAD=c["head"])
                if reg.get("memfile"):               # the snapshot the registration planned against (2026-10-21)
                    env.update(MEMFILE=os.path.join(ROOT, reg["memfile"]))
                env.update(HETEROSERVE_RESERVE_MB=",".join("%s=%d" % (n, v) for n, v in sorted(reg.get("reserve_mb", {}).items())))
                log("serve start", {"round": r_i, "placement": lab})
                p = subprocess.run(["bash", SERVE, "start"], cwd=ROOT, env=env, capture_output=True, text=True)
                open(os.path.join(out, "serve_%02d_%s.log" % (r_i, lab)), "w").write(p.stdout + p.stderr)
                if p.returncode != 0:
                    log("serve FAILED", {"round": r_i, "placement": lab, "rc": p.returncode, "tail": (p.stdout + p.stderr)[-600:]})
                    subprocess.run(["bash", SERVE, "stop"], cwd=ROOT, env=env, capture_output=True, text=True)
                    continue
                st = Q.read_state()
                host = st["HOST"]
                try:
                    head = Q.http(host, "/tokenize", {"content": "", "add_special": True})["tokens"]
                    doc = Q.http(host, "/tokenize", {"content": raw, "add_special": False})["tokens"]
                    qn = Q.http(host, "/tokenize", {"content": Q.QUESTION, "add_special": False})["tokens"]
                    body, wbody = reg["n"] - len(head) - len(qn), reg.get("warm_n", Q.WARM_N) - len(head) - len(qn)
                    woff = len(doc) - wbody
                    prompts = [head + doc[i * Q.OFFSET:i * Q.OFFSET + body] + qn for i in range(reg["reqs"])]
                    warm = head + doc[woff:woff + wbody] + qn
                    log("served", {"round": r_i, "placement": lab, "host": host, "workers": st["WORKERS"], "args": st["ARGS"]})
                    w = Q.stream(host, warm, reg.get("warm_gen", Q.WARM_GEN))
                    log("warm", {"round": r_i, "placement": lab, "done_s": w["done_s"]})
                    for i, pr in enumerate(prompts):
                        x = Q.stream(host, pr, reg["gen"])
                        x.update({"i": i, "round": r_i, "placement": lab})
                        log("req", x)
                finally:
                    s = subprocess.run(["bash", SERVE, "stop"], cwd=ROOT, env=env, capture_output=True, text=True)
                    log("serve stop", {"round": r_i, "placement": lab, "out": s.stdout.strip()[-300:]})
                time.sleep(10)
        log("idle-after start", {}); time.sleep(a.idle_after); log("idle-after end", {})
    finally:
        if lpoll is not None:
            try:
                os.killpg(lpoll.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            log("laptop poll stop", {"lines": sum(1 for _ in open(os.path.join(out, "laptop_poll.log")))
                                     if os.path.exists(os.path.join(out, "laptop_poll.log")) else 0})
        for n, pid in polls.items():          # alive 0: the poll had died, so the windows after its last line are UNKNOWN
            alive = Q.ssh(n, "kill %s 2>/dev/null && echo 1 || echo 0" % pid, check=False).strip()
            dst = os.path.join(out, "throttle_%s.log" % n)
            ok = subprocess.run(["scp", "-q", "%s:logs/throttle_fe_%s.log" % (n, stamp), dst], capture_output=True).returncode == 0
            whole = TS.summarize(open(dst, encoding="utf-8", errors="replace").readlines()) if ok else {}
            if whole:
                whole["alive"] = alive
                whole["verdict"] = TS.verdict(whole)
            log("throttle poll stop", {"node": n, "alive": alive, "copied": ok, "session": whole})
        if logger is not None and logger.poll() is None:
            logger.send_signal(signal.SIGINT)
            try:
                logger.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.kill()
        if lock:
            r = Q.ssh("jetson", "m=$(cut -d' ' -f1 /var/lib/nvpmodel/status); [ \"$m\" = pmode:0000 ] && sudo -n jetson_clocks --restore "
                                "~/heteroserve/run/jetson_clocks_fe_%s.stored && echo restored" % stamp, check=False).strip()
            log("jetson clocks restore", r)
        log("done", {"out": os.path.relpath(out, ROOT)})


# ------------------------------------------------------------------ summary
def boot2(rounds, f, B=BOOT, seed=BOOT_SEED):
    """95 % CI of statistic f over the pooled samples, resampling rounds, then requests within each chosen round."""
    rng, out = random.Random(seed), []
    keys = list(rounds)
    for _ in range(B):
        pool = []
        for _ in keys:
            reqs = rounds[keys[rng.randrange(len(keys))]]
            for _ in reqs:
                pool.extend(reqs[rng.randrange(len(reqs))])
        out.append(f(pool))
    return [round(Q.pct(out, 2.5), 2), round(Q.pct(out, 97.5), 2)]


def window_energy(run, t0, t1, idle, outlets):
    """{outlet: (energy_trap_j, energy_counter_j)} over [t0, t1] (shelly_strip.py summarise)."""
    p = subprocess.run([sys.executable, STRIP, "summarise", os.path.join(run, "strip.csv"), "--window", "%.6f" % t0, "%.6f" % t1,
                        "--idle", "%.6f" % idle[0], "%.6f" % idle[1], "--tokens", "1", "--json"], capture_output=True, text=True)
    s = json.loads(p.stdout)
    return {o: (s["outlets"][str(o)]["energy_trap_j"], s["outlets"][str(o)]["energy_counter_j"]) for o in outlets}


def summary(a):
    run = a.run
    rows = [json.loads(l) for l in open(os.path.join(run, "run.jsonl"))]
    start = next(r for r in rows if r["kind"] == "start")["data"]
    reg = json.load(open(os.path.join(ROOT, start["registration"])))
    outlets = start["outlets"]
    idle = (next(r["t"] for r in rows if r["kind"] == "idle-before start"), next(r["t"] for r in rows if r["kind"] == "idle-before end"))
    polled = {n: open(os.path.join(run, "throttle_%s.log" % n), encoding="utf-8", errors="replace").readlines()
              for n in reg["nodes"] if n.startswith("pi") and os.path.exists(os.path.join(run, "throttle_%s.log" % n))}
    lpath = os.path.join(run, "laptop_poll.log")
    lpolled = open(lpath, encoding="utf-8", errors="replace").readlines() if os.path.exists(lpath) else None
    per = {}
    for r in rows:
        if r["kind"] != "req":
            continue
        x = r["data"]
        cg, sg, merged, tc, ts, ntok = Q.samples(x)
        ev = [e for e in x["events"] if e[1]]
        t_first, t_done = x["sent_wall"] + ev[0][0], x["sent_wall"] + x["done_s"]
        en = window_energy(run, t_first, t_done, idle, outlets.values())
        own = [outlets[n] for n, _ in reg["configs"][x["placement"]]["assign"]]
        gaps = max(1, ntok - 1)
        d = per.setdefault(x["placement"], {"rounds": {}, "ttft": [], "e_all": [], "e_own": [], "tok": 0, "thermal": {},
                                            "laptop": []})
        if lpolled is not None:
            d["laptop"].append(laptop_window(lpolled, x["sent_wall"], t_done))
        d["rounds"].setdefault(x["round"], []).append(sg)
        d["ttft"].append(tc)
        d["e_all"].append((sum(v[0] for v in en.values()), gaps))
        d["e_own"].append((sum(en[o][0] for o in own), gaps))
        for n in [n for n, _ in reg["configs"][x["placement"]]["assign"] if n.startswith("pi")]:   # the whole request
            v = thermal(polled[n], x["sent_wall"], t_done)["verdict"] if n in polled else "UNKNOWN"
            d["thermal"][v] = d["thermal"].get(v, 0) + 1
    res = {}
    for lab, d in per.items():
        allg = [g for reqs in d["rounds"].values() for req in reqs for g in req]
        c = reg["configs"][lab]
        per_round_j = {}
        for (j, n), rd in zip(d["e_all"], [x["data"]["round"] for x in rows if x["kind"] == "req" and x["data"]["placement"] == lab]):
            per_round_j.setdefault(rd, [0.0, 0])
            per_round_j[rd][0] += j; per_round_j[rd][1] += n
        rj = [v[0] / v[1] for v in per_round_j.values()]
        res[lab] = {"assign": c["assign"], "head": c["head"], "predicted": c["predicted"],
                    "server_tpot": Q.stats(allg), "p50_ci95": boot2(d["rounds"], lambda s: Q.pct(s, 50)),
                    "p95_ci95": boot2(d["rounds"], lambda s: Q.pct(s, 95)),
                    "mean_tpot_err_pct": round(100 * (c["predicted"]["decode_ms"] / statistics.fmean(allg) - 1), 1) if allg else None,
                    "ttft_ms_p50": round(Q.pct([t for t in d["ttft"] if t], 50), 1),
                    "j_per_token_all_boards": round(sum(j for j, _ in d["e_all"]) / sum(n for _, n in d["e_all"]), 4),
                    "j_per_token_own_boards": round(sum(j for j, _ in d["e_own"]) / sum(n for _, n in d["e_own"]), 4),
                    "j_per_token_all_boards_rounds": [round(x, 4) for x in rj], "rounds": len(d["rounds"]),
                    "thermal": max(d["thermal"], key=WORST.get) if d["thermal"] else "n/a",
                    "thermal_windows": d["thermal"],
                    **({"laptop": {"requests": len(d["laptop"]),
                                   "max_pkg_c": max((w.get("max_pkg_c") or 0) for w in d["laptop"]),
                                   "max_load": max((w.get("max_load") or 0) for w in d["laptop"]),
                                   "throttle_rise": sum((w.get("throttle_rise") or 0) for w in d["laptop"]),
                                   "charging_requests": sum(1 for w in d["laptop"] if w.get("charging")),
                                   "unpolled_requests": sum(1 for w in d["laptop"] if not w.get("samples"))}}
                       if d["laptop"] else {})}
    json.dump(res, open(os.path.join(run, "summary.json"), "w"), indent=1)
    print("%-8s %-24s %-4s %8s %8s %8s %18s %18s %7s %9s %9s  %s" % ("place", "plan", "head", "P50", "P95", "P99", "P50 CI95",
                                                                     "P95 CI95", "err%", "J/tok all", "J/tok own", "thermal"))
    for lab, r in sorted(res.items(), key=lambda kv: kv[1]["server_tpot"].get("p50", 1e9)):
        s = r["server_tpot"]
        print("%-8s %-24s %-4s %8.1f %8.1f %8.1f %18s %18s %+6.1f%% %9.4f %9.4f  %s %s" % (
            lab, " + ".join("%s %d" % tuple(x) for x in r["assign"]), r["head"], s["p50"], s["p95"], s["p99"], r["p50_ci95"],
            r["p95_ci95"], r["mean_tpot_err_pct"], r["j_per_token_all_boards"], r["j_per_token_own_boards"], r["thermal"],
            r["thermal_windows"] or ""))
    print("wrote %s" % os.path.relpath(os.path.join(run, "summary.json"), ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("register")
    g.add_argument("--name", required=True); g.add_argument("--model", required=True); g.add_argument("--nodes", required=True)
    g.add_argument("--placements", required=True); g.add_argument("--rounds", type=int, default=5)
    g.add_argument("--reqs", type=int, default=4); g.add_argument("--n", type=int, default=128)
    g.add_argument("--gen", type=int, default=96); g.add_argument("--ctx", type=int, default=4096)
    g.add_argument("--rtt-trips", type=int, choices=(1, 2), default=2)
    g.add_argument("--warm-n", type=int, default=Q.WARM_N); g.add_argument("--warm-gen", type=int, default=Q.WARM_GEN)
    g.add_argument("--note", default="", help="registered with the plan, before anything runs")
    g.add_argument("--below-headroom", action="store_true",
                   help="register a placement that fits with less than the engine's default headroom (recorded)")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--memfile", help="plan against this memory snapshot (memory_snapshot.py) instead of memory_at_rest.json")
    g.add_argument("--reserve", help='this registration\'s own reserve for named nodes, "node=MiB[,...]", in place of '
                                     "cluster.json's memory_reserve_mb (recorded; run serves with it)")
    r = sub.add_parser("run")
    r.add_argument("registration"); r.add_argument("--outlets", required=True)
    r.add_argument("--idle-before", type=float, default=180.0); r.add_argument("--idle-after", type=float, default=60.0)
    s = sub.add_parser("summary")
    s.add_argument("run")
    a = ap.parse_args()
    {"register": register, "run": run, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main()
