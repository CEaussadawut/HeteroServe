#!/usr/bin/env python3
"""The scheduler's phase placement for one request (2026-10-28; item 3 of the improvements left, "the prefill-decode
hand-off inside the scheduler"): prefill and decode on the laptop (L), both on the Jetson (J), or prefill on the laptop and
decode on the Jetson (H). phase_serve_20261028.py runs the plan.

    python3 testbed/scripts/phase_plan_20261028.py --n 4096 --gen 256 [--objective energy|time] [--costs <json>]

Why these two nodes and this model: the Jetson's GPU decodes a long context much more cheaply than the laptop's CPU (lc4n:
its attention grows ~8 ms per 1,000 tokens of depth on Qwen3-0.6B, a CPU's far more), and the cluster draws far less with
the Jetson busy than with the laptop busy (the laptop idles at 14 W and computes at ~30 W; the Jetson computes at ~9-11 W).
Qwen3-0.6B Q8_0 fits on both and is trained to 32,768 tokens (TinyLlama's 2,048 rules it out for long prompts).
The cost of each plan (costs from phase_calib_20261028.py's runs through the serving path, --costs):
  prefill_s(node, n)  = (a + b n + c n^2) / 1000   llama-server's prompt_ms for an n-token prompt, least squares
  decode_ms(node, d)  = e + f d                    ms per generated token at context depth d, least squares
  the hand-off: save, transfer and restore seconds, each linear in the state's bytes (kv_bytes_per_token x tokens + a
  fixed part), least squares over the calibration's hand-offs
  L, J: ttft = prefill(n); total = ttft + (gen - 1) x decode(n + gen / 2)
  H:    ttft = prefill_laptop(n); total = ttft + save + transfer + restore + (gen - 1) x decode_jetson(n + gen / 2)
Energy (all four outlets, as every strip figure; the Pis are idle in every plan): each phase's time x (the working node's
measured power in that phase + every other outlet's idle power), energy_model.power_table(): pp while prefilling, tg while
decoding; during the hand-off's save, transfer and restore every outlet is counted at its idle power (an assumption: the
copy is light work; the registered test measures it). choose() returns every plan's figures and the pick for the objective.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
COSTS = os.path.join(ROOT, "testbed", "dataset", "phase_costs_20261028.json")
OUTLETS = ("laptop", "jetson", "pi5", "pi4")
PLANS = {"L": ("laptop", "laptop"), "J": ("jetson", "jetson"), "H": ("laptop", "jetson")}


def load(path=COSTS):
    return json.load(open(path))


def prefill_s(c, node, n):
    a, b, q = c["prefill"][node]["coef"]
    return (a + b * n + q * n * n) / 1000.0


def decode_ms(c, node, d):
    e, f = c["decode"][node]["coef"]
    return e + f * d


def handoff_s(c, n):
    nbytes = c["state"]["fixed_bytes"] + c["state"]["bytes_per_token"] * (n + 1)
    parts = {}
    for k in ("save", "transfer", "restore"):
        i, s = c["handoff"][k]["coef"]
        parts[k] = i + s * nbytes
    return nbytes, parts


def plan(c, name, n, gen, power):
    P, D = PLANS[name]
    idle = {o: power[o]["idle"] for o in OUTLETS}
    others = lambda busy: sum(v for o, v in idle.items() if o != busy)
    t_pre = prefill_s(c, P, n)
    per_tok = decode_ms(c, D, n + gen / 2.0) / 1000.0
    t_dec = (gen - 1) * per_tok
    e = t_pre * (power[P]["pp"] + others(P)) + t_dec * (power[D]["tg"] + others(D))
    out = {"prefill_node": P, "decode_node": D, "ttft_s": t_pre, "decode_s": t_dec, "decode_ms_per_token": per_tok * 1000}
    t_h = 0.0
    if P != D:
        nbytes, parts = handoff_s(c, n)
        t_h = sum(parts.values())
        e += t_h * sum(idle.values())
        out.update({"state_bytes": nbytes, "handoff_s": t_h, "handoff_parts_s": parts})
    out["total_s"] = t_pre + t_h + t_dec
    out["energy_j"] = e
    return out


# The hand-off's memory check (2026-11-04, hm1; the user: "do all four follow-up studies"). The receiver keeps the slot file in
# its tmpfs (RAM) from the transfer until the next request, on top of its loaded server; Qwen3-4B's hand-off to the 4 GB
# Jetson thrashed it from 1,024 tokens (phase_calib_20261104_004942: a 152.8 MB state; 77.3 MB at 512 was clean) and needed a
# power cycle, while Qwen3-0.6B's 941 MB state at 8,204 tokens never did. So H is offered only when the state fits the
# receiver's MemAvailable with its server loaded, less HEADROOM_BYTES; phase_serve_20261028.run() checks it again once the file
# has landed (its guard). The headroom, from the receiver's measured memory (handoff_mem_20261104.py measure,
# testbed/results/hmem_20261104_052145: the 4B's server at 2,560 cells leaves 401 MiB): the 4B's hand-off thrashed with 255 MiB
# left once its state landed (1,036 tokens) and ran clean with 327 MiB left (524 tokens); the 0.6B's ran clean with 794 MiB
# left. 320 MiB sits near the clean side, since a thrash needs a power cycle and a refusal costs only the hand-off's saving.
HEADROOM_BYTES = 320 * 1024 * 1024


def handoff_fits(c, n, receiver_avail, headroom=HEADROOM_BYTES):
    """(fits, state bytes, margin bytes): the hand-off of an n-token prompt fits when its state plus the headroom is within
    the receiver's MemAvailable (bytes, read with its server loaded)."""
    nbytes = c["state"]["fixed_bytes"] + c["state"]["bytes_per_token"] * (n + 1)
    margin = receiver_avail - nbytes - headroom
    return margin >= 0, nbytes, margin


def choose(n, gen, objective="energy", costs=None, power=None, rows=None, receiver_avail=None, headroom=HEADROOM_BYTES):
    """rows (2026-10-28, after ph2): per-model power rows (phase_power_20261028.load()["rows"]); with them each plan's energy
    is phase_power_20261028.energy() (the laptop's turbo burst, its serving idle, the hand-off's own draw), which ph2 found
    within 2.1 % on held-out prompts where the table's was 5-13 % low. None keeps the table, as ph1 registered.
    receiver_avail (2026-11-04, hm1): the decoding node's MemAvailable in bytes with its server loaded; given, H is left out of
    the pick when its state does not fit (handoff_fits), and its plan says why. None prices every plan, as before."""
    c = costs or load()
    if power is None:
        import energy_model as EM
        power = EM.power_table()
    plans = {k: plan(c, k, n, gen, power) for k in PLANS}
    if rows:
        import phase_power_20261028 as W                     # imported here: it imports this module
        plans = {k: W.energy(c, k, n, gen, rows) for k in PLANS}
    key = "energy_j" if objective == "energy" else "total_s"
    allowed = list(plans)
    if receiver_avail is not None:
        ok, nbytes, margin = handoff_fits(c, n, receiver_avail, headroom)
        plans["H"]["memory"] = {"state_bytes": nbytes, "receiver_avail": receiver_avail, "headroom": headroom,
                                "margin_bytes": margin, "fits": ok}
        if not ok:
            allowed.remove("H")
    pick = min(allowed, key=lambda k: plans[k][key])
    return {"n": n, "gen": gen, "objective": objective, "pick": pick, "plans": plans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True); ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--objective", choices=("energy", "time"), default="energy"); ap.add_argument("--costs", default=COSTS)
    a = ap.parse_args()
    r = choose(a.n, a.gen, a.objective, load(a.costs))
    for k, p in r["plans"].items():
        print("%s %-6s->%-6s ttft %7.2f s  total %8.2f s  energy %8.1f J%s" % (
            k, p["prefill_node"], p["decode_node"], p["ttft_s"], p["total_s"], p["energy_j"],
            ("  (hand-off %.2f s, %.0f MB)" % (p["handoff_s"], p["state_bytes"] / 1e6)) if "handoff_s" in p else ""))
    print("pick (%s): %s" % (a.objective, r["pick"]))


if __name__ == "__main__":
    main()
