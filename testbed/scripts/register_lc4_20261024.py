#!/usr/bin/env python3
"""Register the four-node long-context study (scheduler upgrade 4 again; 2026-10-24), in two steps, each before it runs.

    python3 testbed/scripts/register_lc4_20261024.py char [--dry-run]                       # step 1
    python3 testbed/scripts/register_lc4_20261024.py plan --char testbed/results/lc4char_<stamp> --memfile <snapshot> [--dry-run]

The 2026-10-10/11 term (attention_cost_20261011b.json: laws per board, measured on Qwen3-14B and Qwen3-0.6B with the Pi 4,
the Jetson and the laptop; the Pi 5 was away) priced decode at context depth, and on 10-11 a long-context plan was served
on the Pi 4 + Jetson + laptop. The user's rule of 2026-10-21 puts it back on the four nodes. Priced with those laws, the
planner keeps TinyLlama on the laptop at short context but moves it to the Jetson's GPU at long context (laptop 1 + Jetson
21 at 8,192 tokens of depth: 111.8 against 234.2 ms per token): the best placement would change with the context. Both
steps use TinyLlama Q4_K_M, on every board and byte-identical (testbed/models.md), so no law has to transfer between
models.

STEP 1 (lc4char), the cost of depth on each candidate board: llama-bench -p 0 -n 32 -d 0,2048,4096,8192 -r 3, TinyLlama
whole on the laptop (-t 4), the Pi 5 (-t 4) and the Jetson (-ngl 99, Vulkan, clocks locked), the three at once (they share
nothing). The Pi 4 is left out: its law is the dearest (3.9 us per block per token of depth, 3.5 x the laptop's) and its
prefill of 8,192 tokens three times would take about 70 minutes. Predicted slopes: the laws (laptop 0.54681 ns per
query-head dimension; Jetson 0.070251 ns per KV byte; the Pi 5, which has no law, the Pi 4's 1.89395 ns scaled by the
two boards' Q4_K_M weight-byte costs, PROVISIONAL). Hypothesis LC4C-LAWS: each board's measured slope within 30 % of its
prediction.
STEP 2 (lc4plan), registered from step 1's measured slopes: the planner's pick for TinyLlama without the depth term (A)
and with it at 8,192 tokens of depth (B, the term priced per block from the measured slopes, attn = slope / 22 / 1,000),
each a laptop-hosted llama-bench at -d 0,8192 -r 3, order A B B A. Hypotheses: LC4P-SHORT A faster at depth 0;
LC4P-LONG B faster at depth 8,192; LC4P-ACC every bench within 15 % of its prediction at both depths. If the measured
slopes leave B equal to A, step 2 is not registered (the term changes nothing on these boards), and that is the answer.
"""
import argparse, copy, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import full_eval as F                                       # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")
KEY = "tinyllama"
MODEL_PATH, MODEL_SHA = P.MODELS[KEY][2], F.SHA[KEY]
DEPTHS = (0, 2048, 4096, 8192)
LAWS = os.path.join(DS, "attention_cost_20261011b.json")
PI5_PROVISIONAL = 1.89395 * 0.0866 / 0.2807                 # the Pi 4's law x (Pi 5 / Pi 4 Q4_K_M weight-byte cost)
JETSON = {"node": "jetson", "ip": "192.168.1.114", "port": 50052, "extra": "-d Vulkan0"}


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def inputs(extra=()):
    fs = ["testbed/scripts/lc4_20261024.py", "testbed/scripts/register_lc4_20261024.py", "testbed/scripts/predict_pool.py",
          "testbed/scripts/jetson_power_guard.sh", "testbed/scripts/throttle_poll.sh", "testbed/dataset/model_shapes.json",
          "testbed/dataset/characterisation.csv", "testbed/dataset/attention_cost_20261011b.json", "ThinkPad/config/cluster.json"]
    return {f: sha(f) for f in fs + list(extra)}


def write(out, reg, dry):
    if dry:
        print(json.dumps({k: reg[k] for k in ("benches", "hypotheses")}, indent=1)[:5000]); return
    if os.path.exists(out):
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(out, ROOT))
    json.dump(reg, open(out, "w"), indent=1)
    os.chmod(out, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(out, ROOT), sha(os.path.relpath(out, ROOT))[:16]))


def char(dry):
    shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
    q = shapes[P.MODELS[KEY][0]]
    tab = copy.deepcopy(P.attention_table(LAWS))
    tab["boards"]["pi5"] = {"law": "query_heads", "ns_per_qhead_dim": PI5_PROVISIONAL}
    args = "-p 0 -n 32 -d %s -r 3" % ",".join(map(str, DEPTHS))
    benches = {}
    for label, node, extra in (("lap", "laptop", "-t 4"), ("pi5", "pi5", "-t 4"), ("jet", "jetson", "-ngl 99")):
        per_block = P.attn_ms(tab, node, q)
        benches[label] = {"host": node, "model_path": MODEL_PATH, "model_sha256": MODEL_SHA, "args": "%s %s" % (extra, args),
                          "predicted": {"slope_ms_per_1k": round(per_block * q["n_layers"] * 1000, 3),
                                        "law": tab["boards"][node], "provisional": node == "pi5"}}
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_lc4_20261024.py", "runner": "testbed/scripts/lc4_20261024.py",
           "out_prefix": "lc4char", "what": "the cost of context depth for TinyLlama Q4_K_M on the laptop, the Pi 5 and the Jetson",
           "model": KEY, "cold_c": {"pi5": 50, "pi4": 55}, "benches": benches, "order": [["jet", "pi5", "lap"]],
           "hypotheses": {"LC4C-LAWS": {"type": "slope_within", "labels": ["lap", "pi5", "jet"], "tol": 0.30,
                                        "claim": "each board's measured slope within 30 % of its law's prediction"}},
           "inputs_sha256": inputs()}
    write(os.path.join(DS, "lc4char_predictions_20261024.json"), reg, dry)


def plan(char_dir, memfile, dry):
    s1 = json.load(open(os.path.join(char_dir, "score.json")))
    if s1["code_changed_since_registration"]:
        raise SystemExit("step 1's code changed since its registration: %s" % s1["code_changed_since_registration"])
    mem = json.load(open(memfile))
    nodes = ["laptop", "pi5", "jetson"]
    e0 = P.plan_host_choice(KEY, nodes, 8192 + 512, mem, host="laptop", load_mode="none", order_check=True)
    q = e0["q"]
    meas = {"laptop": s1["benches"]["lap"]["slope_ms_per_1k"], "pi5": s1["benches"]["pi5"]["slope_ms_per_1k"],
            "jetson": s1["benches"]["jet"]["slope_ms_per_1k"]}
    attn = {n: v / q["n_layers"] / 1000 for n, v in meas.items()}
    e1 = P.plan_host_choice(KEY, nodes, 8192 + 512, mem, host="laptop", load_mode="none", depth=8192, attn=attn, order_check=True)
    if e1["assign"] == e0["assign"] and e1["head_at"] == e0["head_at"]:
        print("the measured slopes leave the plan unchanged (%s): step 2 is not registered" % e0["assign"]); return
    benches = {}
    for label, e in (("A", e0), ("B", e1)):
        ba = P.bench_args(e["assign"], KEY, head_at=e["head_at"])
        pred = {}
        for dp in (0, 8192):
            dd, _ = P.predict(e["assign"], q, e["C"], e["links"], head_at=e["head_at"], depth=dp, attn=attn, rtt_trips=1)
            pred[str(dp)] = round(dd, 2)
        workers = [n for n, _ in e["assign"][1:]]
        if [w for w in workers if w != "jetson"]:
            raise SystemExit("plan %s uses %s; this runner knows only the Jetson's server" % (label, workers))
        args = ba["args"].replace("-m %s " % MODEL_PATH, "")
        benches[label] = {"host": "laptop", "model_path": MODEL_PATH, "model_sha256": MODEL_SHA,
                          "args": "%s -t 4 -p 0 -n 32 -d 0,8192 -r 3" % args, "assign": e["assign"], "head_at": e["head_at"],
                          "servers": [JETSON] if workers else [],
                          "predicted": {"ms_per_token_by_depth": pred}}
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_lc4_20261024.py", "runner": "testbed/scripts/lc4_20261024.py",
           "out_prefix": "lc4plan", "what": "TinyLlama's plan without and with the depth term, served at depth 0 and 8,192",
           "model": KEY, "cold_c": {"pi5": 50, "pi4": 55}, "memfile": os.path.relpath(os.path.abspath(memfile), ROOT),
           "measured_slopes_ms_per_1k": meas, "attn_ms_per_block_per_token": attn, "step1": s1["run"],
           "benches": benches, "order": [["A"], ["B"], ["B"], ["A"]],
           "hypotheses": {"LC4P-SHORT": {"type": "faster_at", "a": "A", "b": "B", "depth": 0,
                                         "claim": "without depth, the plan without the term is faster (predicted %s against %s ms)"
                                                  % (benches["A"]["predicted"]["ms_per_token_by_depth"]["0"],
                                                     benches["B"]["predicted"]["ms_per_token_by_depth"]["0"])},
                          "LC4P-LONG": {"type": "faster_at", "a": "B", "b": "A", "depth": 8192,
                                        "claim": "at 8,192 tokens of depth, the plan with the term is faster (predicted %s against %s ms)"
                                                 % (benches["B"]["predicted"]["ms_per_token_by_depth"]["8192"],
                                                    benches["A"]["predicted"]["ms_per_token_by_depth"]["8192"])},
                          "LC4P-ACC": {"type": "acc_at", "labels": ["A", "B"], "tol": 0.15,
                                       "claim": "every bench within 15 % of its prediction at both depths"}},
           "inputs_sha256": inputs([os.path.join(s1["run"], "score.json"), os.path.relpath(os.path.abspath(memfile), ROOT)])}
    write(os.path.join(DS, "lc4plan_predictions_20261024.json"), reg, dry)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("char", "plan")); ap.add_argument("--char"); ap.add_argument("--memfile")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    char(a.dry_run) if a.step == "char" else plan(a.char, a.memfile, a.dry_run)
