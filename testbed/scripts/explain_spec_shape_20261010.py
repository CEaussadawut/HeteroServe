#!/usr/bin/env python3
"""Exploratory, post hoc (2026-10-10): why the laptop draft's link cost 3x the registered model in spec_shape (SH-SLOPE
c = 3.06). Reads the shaping run and the socket byte count, writes <run>/explain.json.

    python3 testbed/scripts/explain_spec_shape_20261010.py [testbed/results/spec_shape_<stamp> testbed/results/draft_bytes_<stamp>]

The pinned source (b23701f7) says each check runs TWO draft decodes:
- the server re-decodes every verified batch on the draft (tools/server/server-context.cpp:3741-3747, its own TODO
  TAG_SPEC_AVOID_DRAFT_REEVAL), and draft-simple's process() leaves the batch's logits null, so the last token's logits
  come back (src/llama-batch.cpp:120-128);
- draft() then decodes the last token, with logits.
Each decode uploads its whole serialized graph. ggml-rpc re-sends a graph unless its uid equals the device's last one
(ggml-rpc.cpp:1031-1050), and the two decodes alternate between 2-token and 1-token graphs, so the cache never hits.
The model therefore needs 2 x (G + V + H) bytes per check, not H + V:
- G is the upload per draft decode, measured on the laptop's socket (draft_bytes run);
- V is the logits row, n_vocab x 4;
- H is the hidden state, n_embd x 4.
For each shaped setting this script compares the measured extra ms per check (over s0) with the registered model and with
the corrected count. It also recomputes the crossover from the corrected bytes and the arms' measured gap at gigabit.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import link_to_json as LJ                                   # noqa: E402

RUN = os.path.join(ROOT, "testbed", "results", "spec_shape_20261010_125542")
BYTES = os.path.join(ROOT, "testbed", "results", "draft_bytes_20261010_152629")


def main(argv):
    run, byt = (argv[0], argv[1]) if len(argv) >= 2 else (RUN, BYTES)
    reg = json.load(open(os.path.join(ROOT, "testbed", "dataset", "spec_shape_predictions_20261010.json")))
    m = reg["model_parts"]
    rows = [json.loads(l) for l in open(os.path.join(run, "run.jsonl"))]
    rows = [r for r in rows if r.get("arm") and r.get("timings") and r["timings"].get("predicted_n")]
    b = json.load(open(os.path.join(byt, "results.json")))
    dec = lambda t: (t["predicted_n"] - t["draft_n_accepted"]) + t["draft_n"] + 1   # process() per target step, draft(), the prompt
    g = [r["laptop_received_bytes"] / dec(r["timings"]) for r in b["requests"] if r["n_predict"] == 64]
    G, V, H = sum(g) / len(g), m["V"], m["H"]
    links = {}
    for s in reg["settings"]:
        _, _, rec = LJ.parse(os.path.join(run, "link_%s.log" % s["id"]))
        links[s["id"]] = rec["mbps"] * 125.0                     # bytes per ms
    per_step = lambda rs: sum(r["timings"]["predicted_ms"] for r in rs) / sum(
        r["timings"]["predicted_n"] - r["timings"]["draft_n_accepted"] for r in rs)
    lap = {s: per_step([r for r in rows if r["arm"] == "lapd1" and r["setting"] == s]) for s in links}
    pi4 = per_step([r for r in rows if r["arm"] == "pi4d1"])
    g0 = links["s0"]
    out = {"G_upload_per_decode_bytes": [round(x) for x in g], "V_logits_bytes": V, "H_hidden_bytes": H,
           "ms_per_check": {"pi4d1": round(pi4, 1), "lapd1": {s: round(v, 1) for s, v in lap.items()}}, "settings": {}}
    for s in links:
        if s in ("s0", "s0r"):
            continue
        bw = links[s]
        extra = lap[s] - lap["s0"]
        reg_m = (H + V) / bw - (H + V) / g0
        cor = 2 * (G + V + H) / bw - 2 * (G + V + H) / g0
        out["settings"][s] = {"goodput_mbit": round(bw * 8 / 1000, 2), "measured_extra_ms": round(extra, 1),
                              "registered_model_ms": round(reg_m, 1), "corrected_ms": round(cor, 1),
                              "measured_over_registered": round(extra / reg_m, 3), "measured_over_corrected": round(extra / cor, 3)}
    gap0 = pi4 - lap["s0"]
    per_chk = 2 * (G + V + H)
    bw_x = per_chk / (gap0 + per_chk / g0)
    out["crossover_mbit"] = {"corrected_from_measured_gigabit_gap": round(bw_x * 8 / 1000, 1),
                             "measured_score_json": json.load(open(os.path.join(run, "score.json")))["hypotheses"]["SH-CROSS"]["measured_mbit"],
                             "registered": reg["predicted_crossover_goodput_mbit"]}
    out["note"] = "exploratory and post hoc; the registered verdicts stand as scored in score.json"
    json.dump(out, open(os.path.join(run, "explain.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
