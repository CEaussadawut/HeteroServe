#!/usr/bin/env python3
"""Version b of the planner's measured costs (2026-10-11, 00:45): the laptop's worker row and attention cost re-measured
with a remote host. Writes, read-only, testbed/dataset/attention_cost_20261011b.json and laptop_worker_20261011b.json.
The a files stay as written.

    python3 testbed/scripts/attention_cost_20261011b.py [--dry-run]

Why: the a cell (laptop_worker_char_20261010.sh) ran llama-bench and the worker on the same 4-core laptop. The laptop's
build waits by polling ("poll": 50, ggml's thread pool), so the host's spinning threads ran against the worker's. The
long-context smoke then decoded the plan about 2x faster on the laptop than a priced it. The b cell
(laptop_worker_remote_20261011.py) hosts on the Pi 4, over the measured link.
- The worker row (TinyLlama Q4_K_M) is taken net of the network: each decoded token pays one round trip, the hidden state
  out and the logits back. The time at the measured link (933.9 Mbit/s, RTT 0.846 ms) is taken off tg; pp's network
  share is under 0.1 % and is left in.
- The laptop's attention cost is the 0.6B's slope at the remote cell. The network is the same at every depth, so it
  cancels.
- The Pi 4's and the Jetson's figures are a's, unchanged.
"""
import glob, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import attention_cost_20261011 as A                         # noqa: E402
import link_to_json as LJ                                   # noqa: E402

OUT_ATTN = os.path.join(A.DS, "attention_cost_20261011b.json")
OUT_LAP = os.path.join(A.DS, "laptop_worker_20261011b.json")
LINK = os.path.join(ROOT, "testbed", "results", "spec_shape_20261010_125542", "link_s0.log")


def build(remote_dir):
    shapes = json.load(open(os.path.join(A.DS, "model_shapes.json")))
    t = shapes["tinyllama-1.1b-q4km"]
    n_vocab = (t["head_params"] - t["n_embd"]) / t["n_embd"]
    _, _, lk = LJ.parse(LINK)
    net_ms = lk["rtt_ms"] + (t["activation_bytes"] + 4 * n_vocab) / (lk["mbps"] * 1e6 / 8 / 1000.0)
    ld = lambda tag: [[json.loads(l) for l in open(os.path.join(remote_dir, "%s.%d.log" % (tag, i)), errors="replace") if l.startswith("{")]
                      for i in (1, 2, 3)]
    tiny = ld("tiny_q4km")
    tg = [1000.0 / [r for r in inv if r["n_gen"]][0]["avg_ts"] for inv in tiny]
    pp = [1000.0 / [r for r in inv if r["n_prompt"]][0]["avg_ts"] for inv in tiny]
    sl = [A.slope(sorted((r["n_depth"], 1000.0 / r["avg_ts"]) for r in inv)) for inv in ld("q06_depth")]
    sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
    from control import gguf
    q06 = gguf.shape(os.path.expanduser("~/.cache/heteroserve-models/Qwen3-0.6B-Q8_0.gguf"))
    lap = statistics.fmean(sl) / (1000.0 * q06["n_layers"]) / (q06["n_head"] * q06["head_dim"]) * 1e6
    a_attn = json.load(open(A.OUT_ATTN))
    src = {os.path.relpath(p, ROOT): A.sha(p) for p in sorted(glob.glob(os.path.join(remote_dir, "*.log"))) + [LINK, A.OUT_ATTN]}
    attn = dict(a_attn, _source="testbed/scripts/attention_cost_20261011b.py (b: the laptop re-measured with a remote host)",
                inputs_sha256=src)
    attn["boards"] = dict(a_attn["boards"])
    attn["boards"]["laptop"] = {"law": "query_heads", "ns_per_qhead_dim": round(lap, 5),
                                "from": "laptop_worker_remote: Qwen3-0.6B, Pi 4 host, every layer on the laptop's worker, three runs",
                                "slope_ms_per_1k_each": [round(x, 2) for x in sl],
                                "spread_pct": round(100 * (max(sl) - min(sl)) / statistics.fmean(sl), 1),
                                "a_value_loopback": a_attn["boards"]["laptop"]["ns_per_qhead_dim"]}
    lapw = {"_source": "testbed/scripts/attention_cost_20261011b.py, from %s" % os.path.relpath(remote_dir, ROOT),
            "node": "laptop", "cluster_json_id": "t580", "role": "rpc-worker (ggml-rpc-server -t 4), Pi 4 host over the link",
            "network_ms_per_token_taken_off_tg": round(net_ms, 3),
            "rows": {"Q4_K_M": {"model": "tinyllama-1.1b-q4km", "tg_ms": round(statistics.fmean(tg) - net_ms, 3),
                                "pp_ms": round(statistics.fmean(pp), 3), "tg_ms_raw_each": [round(x, 2) for x in tg],
                                "pp_ms_each": [round(x, 2) for x in pp],
                                "tg_spread_pct": round(100 * (max(tg) - min(tg)) / statistics.fmean(tg), 1)}},
            "inputs_sha256": {k: v for k, v in src.items() if "laptop_worker_remote" in k or "link_s0" in k}}
    return attn, lapw


def main(argv):
    remote = sorted(glob.glob(os.path.join(ROOT, "testbed", "results", "laptop_worker_remote_2*")))[-1]
    attn, lapw = build(remote)
    print(json.dumps({"laptop_attention": attn["boards"]["laptop"], "laptop_worker": lapw["rows"],
                      "network_ms": lapw["network_ms_per_token_taken_off_tg"]}, indent=1))
    if "--dry-run" in argv:
        print("(dry run: nothing written)")
        return
    for path, obj in ((OUT_ATTN, attn), (OUT_LAP, lapw)):
        if os.path.exists(path):
            sys.exit("%s exists: never rewritten" % path)
        json.dump(obj, open(path, "w"), indent=1)
        os.chmod(path, 0o444)
        print("wrote %s (%s)" % (os.path.relpath(path, ROOT), A.sha(path)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
