#!/usr/bin/env python3
"""Register the predictor rule RTT1 (2026-10-09), before the cells that test it run. Writes, read-only,
testbed/dataset/rule_rtt1_20261009.json; refuses to overwrite it.

    python3 testbed/scripts/register_rule_rtt1_20261009.py

RTT1: an RPC worker costs ONE network round trip per decoded token (and per prompt batch), not two. predict_pool.predict,
moe_prefill, plan_host_choice, plan_fixed and energy_model take rtt_trips (default 2, so that every earlier registration
and scorer reproduces unchanged); full_eval.py register takes --rtt-trips. The rule is adopted only if its registered
test holds (below); until then the default stays 2.
"""
import datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "testbed", "dataset", "rule_rtt1_20261009.json")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if os.path.exists(OUT):
        sys.exit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    reg = {
        "registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "rule": "RTT1",
        "definition": "predict_pool's link term per RPC worker: rtt_trips x RTT + (hidden state out + its output back) / "
                      "bandwidth, per decoded token and per prompt batch, with rtt_trips = 1 (was 2).",
        "code_sha256": {f: sha(os.path.join(HERE, f)) for f in ("predict_pool.py", "energy_model.py", "full_eval.py")},
        "evidence": {
            "code": "llama.cpp b23701f7 ggml/src/ggml-rpc/ggml-rpc.cpp:945-1047: SET_TENSOR and GRAPH_COMPUTE are sent "
                    "one-way (send_async, no reply); only GET_TENSOR waits. Per worker visit: the request and the reply each "
                    "cross the link once, so one round trip, plus the transfers.",
            "measured": "the network study (testbed/results/netshape_20261008_182208/score.json, NS-RTT): decode rose 0.94 "
                        "(head last) and 1.09 (head on the host) ms per ms of added RTT, one worker; the registered "
                        "prediction was 2. Re-pricing those cells with one trip (exploratory, post hoc) moved the +10 ms "
                        "cells from +9.0 / +9.2 % to -1.6 / -0.8 %.",
            "caveat": "at gigabit the second trip had been absorbing real per-visit overhead: the 2026-10-09 pilot's pooled "
                      "placements measured 4.3-4.4 % slower than the 2-trip predictions, and RTT1 makes them about 1 % more "
                      "optimistic. RTT1 corrects the round-trip COEFFICIENT; a per-visit overhead term is a separate "
                      "question, measured (exploratory) by the same study at its unshaped setting."},
        "test": {
            "study": "the ring-lap study (testbed/dataset/ring_laps_predictions_20261009.json, registered after this file): "
                     "TinyLlama with the Pi 4 holding 4 blocks and the Jetson 18, in 1, 2, 3 and 4 laps (1-4 Jetson visits "
                     "per token), unshaped and with +5 ms RTT (netshape.sh, the Pi 4 <-> Jetson path only).",
            "statistic": "c = the slope, through the origin, of [decode ms/token at +5 ms - unshaped] over the number of "
                         "Jetson visits k = 1..4, divided by the RTT increase measured by ping in the same round; the mean "
                         "over rounds.",
            "decision": "c in [0.75, 1.25]: RTT1 holds and is adopted (new registrations pass rtt_trips=1; the default stays "
                        "2 for reproducibility). c in [1.75, 2.25]: the 2-trip model stands. Otherwise neither holds, and "
                        "both are reported."},
        "status": "registered, not adopted",
    }
    json.dump(reg, open(OUT, "w"), indent=1, ensure_ascii=False)
    os.chmod(OUT, 0o444)
    print("registered %s at %s, sha256 %s (read-only)" % (os.path.relpath(OUT, ROOT), reg["registered"], sha(OUT)[:16]))


if __name__ == "__main__":
    main()
