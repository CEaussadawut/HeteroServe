#!/usr/bin/env python3
"""Register the engine against prima.cpp + Halda on all four nodes (prima4_session_20261021.sh), before anything is measured.

    python3 testbed/scripts/register_prima4_20261021.py --memfile testbed/dataset/memory_snapshot_<stamp>.json \\
        --jetson-bench testbed/results/jetson_prima_cpu_bench_<stamp>.log --decision testbed/results/prima_run4_q32decide2048_<stamp>.log

Writes testbed/dataset/prima4_predictions_20261021.json, read-only; refuses to overwrite one. Reads files only.
  Arm A  the engine's own pick (predict_pool.plan_host_choice, ctx 2048, the memory snapshot, cluster.json's reserves: the
         laptop's 8 GiB desktop reserve, so the user's 7 GiB permission for these runs is not used), served as registered
         (PLACE / PLACE_HEAD), priced by the registrations' predictor.
  Arm B  prima.cpp + Halda as it runs: Halda decides at each start from its own profile. The windows its decision run logged
         (--decision) are registered as the expected split, priced by the same predictor with one change: the Jetson computes on
         its CPU under prima.cpp (no Vulkan), at the cost prima.cpp's own llama-bench measured there (--jetson-bench: three runs
         of TinyLlama Q8_0 under its llama-cli, 128 decoded tokens and a ~512-token prompt, the median run; extra_costs), the other nodes at the dataset's costs, as the
         2026-09-30 comparison priced Halda's split (within 2 % of the measured, Q4_K on Pi 5 + Pi 4).
Energy: energy_model.py's decode gross J/token with every node powered; for B the Jetson's power-table figures are its Vulkan
ones (no CPU figure exists), an approximation stated in the file.
"""
import argparse, csv, datetime, hashlib, io, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import qr3_tpot as Q                                        # noqa: E402
import prima_partial_20261021 as PP                         # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "prima4_predictions_20261021.json")
KEY, CTX, NODES = "qwen3_32b_q8", 2048, ["pi5", "pi4", "jetson", "laptop"]
RING = ["laptop", "pi5", "pi4", "jetson"]
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}
COPIES = "testbed/dataset/prima_copies_qwen3-32b-q8_0.json"


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def bench(log):
    """Per run of jetson_prima_cpu_bench_20261021.sh: (decode ms/token, prompt ms/token) from its RESULT lines (prima.cpp's
    llama-cli llama_perf report: "eval time ... ms per token" over 127 runs, "prompt eval time" over the ~512-token prompt)."""
    out = []
    for m in re.finditer(r"^RESULT run \d+ pp_tokens_ms (\d+) ([\d.]+) tg_runs_ms (\d+) ([\d.]+)", open(log, errors="replace").read(), re.M):
        out.append((float(m.group(4)), float(m.group(2))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--jetson-bench", required=True)
    ap.add_argument("--decision", required=True); ap.add_argument("--reqs", type=int, default=5)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=OUT, help="the registration file (2026-10-22: prima4b supersedes the 2026-10-21 one)")
    ap.add_argument("--supersedes", default="", help="why, recorded in the file")
    # 2026-10-23 (the laptop leaves at 08:00): the arm-runs' order, a deadline no arm-run may run past (each allowed its minutes;
    # an arm is then complete with one arm-run's worth of requests), a laptop reserve (the user's 7 GiB for Qwen3-32B runs,
    # 2026-10-21) and the night's conditions, all recorded in the file and read by the session
    ap.add_argument("--order", default="A B A B"); ap.add_argument("--deadline", default="", help="ISO time, e.g. 2026-10-23T07:30:00+07:00")
    ap.add_argument("--arm-min-a", type=int, default=55); ap.add_argument("--arm-min-b", type=int, default=70)
    ap.add_argument("--reserve", default="", help='HETEROSERVE_RESERVE_MB for arm A, e.g. "laptop=7168"; empty: cluster.json')
    ap.add_argument("--conditions", default="", help="the night's conditions, recorded")
    ap.add_argument("--reserve-why", default="the user's 2026-10-21 permission: a 7 GiB laptop reserve for the Qwen3-32B runs",
                    help="why the reserve, recorded (2026-10-23: prima4c, the user's OK for Halda's laptop budget)")
    ap.add_argument("--note", default="", help="what this registration adds, recorded")
    a = ap.parse_args()
    globals()["OUT"] = os.path.abspath(a.out)
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    order = a.order.split()
    if sorted(set(order)) != ["A", "B"]:
        raise SystemExit("refused: the order needs both arms: %s" % order)
    deadline = None
    if a.deadline:
        t = datetime.datetime.fromisoformat(a.deadline)
        deadline = {"at": t.isoformat(), "epoch": int(t.timestamp()), "arm_min": {"A": a.arm_min_a, "B": a.arm_min_b},
                    "rule": "an arm-run starts only if, at its allowed minutes, it ends before the deadline; otherwise the session "
                            "ends there as at its normal end; an arm is complete with one arm-run's worth of requests (min_runs)"}
    os.environ["HETEROSERVE_RESERVE_MB"] = a.reserve                # "" keeps cluster.json's reserves (the laptop's 8 GiB)
    mem = json.load(open(a.memfile))
    power = EM.power_table()
    # A: the engine's pick
    e = P.plan_host_choice(KEY, NODES, CTX, mem, host="laptop", load_mode="none")
    mA = P.memory(e["assign"], e["q"], mem, e["devices"], e["model"], head_at=e["head_at"])
    if min(v["margin_mib"] for v in mA.values()) < 0:
        raise SystemExit("refused: A does not fit at rest: %s" % mA)
    headA = "host" if e["head_at"] == 0 else "last"
    eA = EM.energy(e["assign"], e["q"], e["C"], e["links"], power, "tg", head_at=e["head_at"], rtt_trips=P.RTT_TRIPS_DEFAULT)
    # B: Halda's windows from its decision run, rank order laptop, pi5, pi4, jetson
    ws = {}
    for r, n in enumerate(RING):
        f = a.decision[:-4] + "_rank%d.log" % r if a.decision.endswith(".log") else None
        m = re.findall(r"Using window size: (\d+)", open(f, errors="replace").read()) if f and os.path.exists(f) else []
        ws[n] = int(m[-1]) if m else None
    if None in ws.values() or sum(ws.values()) != e["q"]["n_layers"]:
        raise SystemExit("refused: the decision run's windows are incomplete: %s" % ws)
    runs = bench(a.jetson_bench)
    if len(runs) < 3:
        raise SystemExit("refused: %d llama-bench runs in %s, three needed" % (len(runs), a.jetson_bench))
    tg_ms = statistics.median(x for x, _ in runs); pp_ms = statistics.median(y for _, y in runs)
    shapes = json.load(open(os.path.join(ROOT, "testbed", "dataset", "model_shapes.json")))
    cpu = P.unit(tg_ms, pp_ms, shapes[P.TINY["Q8_0"]])
    extra = {"jetson": {"host": cpu, "worker": cpu}}
    assignB = [(n, ws[n]) for n in RING]                              # the ring from the head: the same links as prima.cpp's
    b = P.plan_fixed(KEY, assignB, "host", CTX, mem, load_mode="none", extra_costs=extra)
    eB = EM.energy(b["assign"], b["q"], b["C"], b["links"], power, "tg", head_at=b["head_at"], rtt_trips=P.RTT_TRIPS_DEFAULT)
    bv = P.plan_fixed(KEY, assignB, "host", CTX, mem, load_mode="none")      # the same split with the Jetson on Vulkan
    jA, jB = eA["gross_j"], eB["gross_j"]
    fast = "A" if e["decode_ms"] < b["decode_ms"] else "B"
    cheap = "A" if jA < jB else "B"
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           **({"supersedes": a.supersedes} if a.supersedes else {}),
           "script": "testbed/scripts/register_prima4_20261021.py", "session": "testbed/scripts/prima4_session_20261021.sh",
           "what": "the engine (A) against prima.cpp + Halda (B) on all four nodes (user, 2026-10-21: 'I want a model that will do "
                   "with prima cpp and halda to our engine and it use full nodes'); energy at the wall, all four outlets",
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1], "ctx": CTX,
           "ctx_why": "prima.cpp's Halda has no in-memory plan for this model at ctx 4096 (its budgets, MemAvailable x 0.8, hold 62 "
                      "of 64 blocks) and its disk-offload path cannot engage: every board's disk is under its 100 MB/s floor, "
                      "so all are forced into M4, and its first solve's rollback asserts (testbed/results/"
                      "prima_run4_q32decide_20261021_210111_rank0.log, common.cpp:1435). At 2048 it decided 39/11/10/4 on "
                      "2026-10-21; this registration's decision run is decision_log, its windows expected_windows.",
           "decision_log": os.path.relpath(os.path.abspath(a.decision), ROOT),
           "reserve": ("cluster.json's: the laptop keeps its 8 GiB desktop reserve; " if not a.reserve else
                       "HETEROSERVE_RESERVE_MB=%s for arm A's plan and serve_pool.sh's gate (%s); " % (a.reserve, a.reserve_why))
                      + "prima.cpp keeps its own 20 % of MemAvailable on every node",
           "reserve_mb": a.reserve,
           **({"deadline": deadline, "min_runs": {"A": 1, "B": 1}} if deadline else {}),
           **({"conditions": a.conditions} if a.conditions else {}),
           **({"note": a.note} if a.note else {}),
           "nodes": NODES, "outlets": OUTLETS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT),
           "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN], "order": order,
           "cold_c": {"pi5": 50, "pi4": 55},
           "arms": {"A": {"what": "the engine's pick, llama.cpp at the pin", "assign": e["assign"], "head": headA,
                          "place_env": {"PLACE": ",".join("%s:%d" % tuple(x) for x in e["assign"]), "PLACE_HEAD": headA},
                          "predicted": {"decode_ms": round(e["decode_ms"], 1), "prefill_ms": round(e["prefill_ms"], 1),
                                        "decode_gross_j_all": round(jA, 4)},
                          "memory": mA, "considered": e["considered"]},
                    "B": {"what": "prima.cpp @ 288df6a2 + the Qwen3 port (+ the gcc-8 NEON patch on the Jetson), Halda deciding at "
                                  "start, ring laptop (rank 0) -> pi5 -> pi4 -> jetson, the boards on partial copies",
                          "copies": COPIES, "copies_sha256": sha(COPIES),
                          "expected_windows": ws,
                          "expected_layers": {RING[r]: [min(v), max(v)] for r, v in
                                              PP.prima_layers([ws[n] for n in RING], e["q"]["n_layers"]).items() if v},
                          "predicted": {"decode_ms": round(b["decode_ms"], 1), "prefill_ms": round(b["prefill_ms"], 1),
                                        "decode_gross_j_all": round(jB, 4),
                                        "decode_ms_if_the_jetson_used_vulkan": round(bv["decode_ms"], 1)},
                          "jetson_cpu": {"bench": os.path.relpath(os.path.abspath(a.jetson_bench), ROOT),
                                         "runs_tg_pp_ms_per_token": [[round(x, 2), round(y, 2)] for x, y in runs],
                                         "median_tg_ms": round(tg_ms, 2), "median_pp_ms": round(pp_ms, 2),
                                         "unit_per_byte_per_param": cpu},
                          "energy_note": "the Jetson's power-table figures are its Vulkan ones; no CPU-only power figure exists"}},
           "prior": {"20260930": "Pi 5 + Pi 4, Qwen3-14B: B/A client TPOT 1.127 (A faster); Halda memory-bound",
                     "20261020": "prima_energy_predictions_20261020.json, the same pair with energy",
                     "20261021": "prima4_predictions_20261021.json: arm-run A1 (laptop 32 + Pi 4 13 + Pi 5 14 + Jetson 5) client "
                                 "TPOT P50 3,547.7 ms; arm-run B2 ended at start: Halda gave the Jetson no layers and the "
                                 "harness stopped (fixed 2026-10-22: a rank Halda removes is accepted and recorded)",
                     "20261022": "sched2_predictions_20261022.json: the same A plan, arm-runs A1 / A3 client TPOT P50 3,468.7 / "
                                 "3,311.3 ms"},
           "hypotheses": {
               "P4-ALL": {"claim": "in every B arm-run Halda gives each of the four nodes a window, every rank's layers inside its "
                                   "partial copy"},
               "P4-TPOT": {"predicted_lower": fast, "claim": "the arm predicted faster (%s: A %.1f, B %.1f ms) has the lower client "
                                                             "TPOT P50" % (fast, e["decode_ms"], b["decode_ms"])},
               "P4-ENERGY": {"predicted_lower": cheap, "claim": "the arm predicted cheaper (%s: A %.2f, B %.2f J/token) has the lower "
                                                               "gross J per decoded token at all four outlets" % (cheap, jA, jB)},
               "P4-FOLLOW": {"tol": 0.05, "claim": "B/A in J per token within 0.05 of B/A in client TPOT P50"},
               "P4-ACC": {"tol": 0.10, "claim": "A's server mean TPOT within 10 % of its predicted decode"},
               "P4-CORRECT": {"claim": "request 0's first 16 tokens of text (greedy) are the same in every A and B run; the other "
                                       "requests recorded. Caveat (2026-10-21 TinyLlama check): on this x86 + ARM ring a different "
                                       "split can change greedy text, so a FAIL here is read with the runs' windows"}},
           "energy_definition": "per request, the gross energy at all four outlets over its decode window (first token to last), "
                                "with the session's idle-before window; summed over requests and divided by gen - 1 per request",
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/predict_pool.py",
               "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py", "testbed/scripts/full_eval.py",
               "testbed/scripts/prima4_session_20261021.sh", "testbed/scripts/score_prima4_20261021.py",
               "testbed/scripts/prima_partial_20261021.py", "testbed/scripts/register_prima4_20261021.py",
               "testbed/patches/prima-288df6a-qwen3-v1.patch", "testbed/patches/prima-288df6a-gcc8-neon-v1.patch",
               "testbed/dataset/model_shapes.json", "testbed/dataset/characterisation.csv", "ThinkPad/config/cluster.json",
               COPIES, os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    if a.dry_run:
        print(json.dumps({k: reg[k] for k in ("arms", "hypotheses")}, indent=1)[:6000]); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))
    print("A %s head %s: %.1f ms/token, %.2f J/token | B %s: %.1f ms/token (Vulkan Jetson %.1f), %.2f J/token | faster %s, cheaper %s"
          % (reg["arms"]["A"]["place_env"]["PLACE"], headA, e["decode_ms"], jA, ws, b["decode_ms"], bv["decode_ms"], jB, fast, cheap))


if __name__ == "__main__":
    main()
