#!/usr/bin/env python3
"""Version c of the laptop's worker rows (2026-10-11, the laptop as a node): every quantisation the planner prices, from
laptop_worker_quants_20261011.py's run. Writes, read-only, testbed/dataset/laptop_worker_20261011c.json.

    python3 testbed/scripts/laptop_worker_c_20261011.py [--dry-run]

As b (attention_cost_20261011b.py): each row's decode is taken net of the network -- one round trip and the hidden state
out and the logits back per token, at the pi4-laptop link measured the same night (the newest link_pi4-laptop_* log before
the run) -- and prefill is left as measured (its network share is under 0.1 %). TinyLlama's shapes are the same at every
quantisation, so one network figure serves them all. predict_pool.laptop_worker_row() reads the newest file that has a
quantisation, so c's Q4_K_M row (taken again here) replaces b's for new registrations; b stays as written.
"""
import glob, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import link_to_json as LJ                                   # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "laptop_worker_20261011c.json")
TINY = {"Q4_K_M": "tinyllama-1.1b-q4km", "Q3_K_M": "tinyllama-1.1b-q3km", "Q4_0": "tinyllama-1.1b-q4_0",
        "Q8_0": "tinyllama-1.1b-q8_0", "Q6_K": "tinyllama-1.1b-q6_k"}


def sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def build(run_dir, link_log):
    shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
    t = shapes["tinyllama-1.1b-q4km"]
    n_vocab = (t["head_params"] - t["n_embd"]) / t["n_embd"]
    _, _, lk = LJ.parse(link_log)
    net_ms = lk["rtt_ms"] + (t["activation_bytes"] + 4 * n_vocab) / (lk["mbps"] * 1e6 / 8 / 1000.0)
    rows = {}
    for qn in TINY:
        inv = [[json.loads(l) for l in open(p, errors="replace") if l.startswith("{")]
               for p in sorted(glob.glob(os.path.join(run_dir, "%s.[0-9].log" % qn)))]
        inv = [i for i in inv if any(r["n_gen"] for r in i) and any(r["n_prompt"] for r in i)]
        if len(inv) < 3:
            continue                                    # three invocations, or no row (house rule: never one sample)
        tg = [1000.0 / [r for r in i if r["n_gen"]][0]["avg_ts"] for i in inv]
        pp = [1000.0 / [r for r in i if r["n_prompt"]][0]["avg_ts"] for i in inv]
        rows[qn] = {"model": TINY[qn], "tg_ms": round(statistics.fmean(tg) - net_ms, 3), "pp_ms": round(statistics.fmean(pp), 3),
                    "tg_ms_raw_each": [round(x, 2) for x in tg], "pp_ms_each": [round(x, 2) for x in pp],
                    "tg_spread_pct": round(100 * (max(tg) - min(tg)) / statistics.fmean(tg), 1),
                    "pp_spread_pct": round(100 * (max(pp) - min(pp)) / statistics.fmean(pp), 1)}
    src = sorted(glob.glob(os.path.join(run_dir, "*.log"))) + [link_log]
    return {"_source": "testbed/scripts/laptop_worker_c_20261011.py, from %s" % os.path.relpath(run_dir, ROOT),
            "node": "laptop", "cluster_json_id": "t580", "role": "rpc-worker (ggml-rpc-server -t 4), Pi 4 host over the link",
            "link": {"log": os.path.relpath(link_log, ROOT), "mbps": lk["mbps"], "rtt_ms": lk["rtt_ms"]},
            "network_ms_per_token_taken_off_tg": round(net_ms, 3), "rows": rows,
            "inputs_sha256": {os.path.relpath(p, ROOT): sha(p) for p in src}}


def newest_inputs():
    run = sorted(glob.glob(os.path.join(ROOT, "testbed", "results", "laptop_worker_quants_2*")))[-1]
    stamp = os.path.basename(run).split("_", 3)[-1]
    links = [p for p in sorted(glob.glob(os.path.join(ROOT, "testbed", "results", "link_pi4-laptop_2*.log")))
             if os.path.basename(p)[len("link_pi4-laptop_"):-4] <= stamp]
    return run, links[-1]


def main(argv):
    run, link = newest_inputs()
    obj = build(run, link)
    print(json.dumps({k: obj[k] for k in ("link", "network_ms_per_token_taken_off_tg", "rows")}, indent=1))
    if "--dry-run" in argv:
        print("(dry run: nothing written)")
        return
    if os.path.exists(OUT):
        sys.exit("%s exists: never rewritten" % OUT)
    json.dump(obj, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("wrote %s (%s)" % (os.path.relpath(OUT, ROOT), sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
