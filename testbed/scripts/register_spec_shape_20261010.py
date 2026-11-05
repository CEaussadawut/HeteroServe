#!/usr/bin/env python3
"""Register the draft's placement under shaping (2026-10-10; the user chose "Draft under shaping") BEFORE it runs. Writes,
read-only, testbed/dataset/spec_shape_predictions_20261010.json.

    python3 testbed/scripts/register_spec_shape_20261010.py [--dry-run] [--out PATH]
        --out writes the same registration elsewhere (a smoke run's plumbing; never the dataset's file)

The question: scheduler upgrade 1 measured the Qwen3-0.6B draft faster on the laptop than on the Pi 4 at gigabit (1.40x
against 1.25x; spec_serve_predictions_20261010.json). On the laptop each draft step sends one hidden state (n_embd x 4
bytes) and brings back the logits (n_vocab x 4 = 608 KB) over the Pi 4 <-> laptop link. So as that link slows, the
laptop's draft step grows, and below some bandwidth the Pi 4, where the draft costs no link, wins. The engine's draft
placement rule must flip there. The model puts the crossover near 29 Mbit/s of goodput.

The setup: step 3's plan and server, exactly. Qwen3-14B with Pi 4 27 + Jetson 13 and the head on the host, ctx 1,024,
greedy, one drafted token per check. The prompts: every other one of step 3's eight (e01, c01, w01, th01), 64 tokens each.
- The shaping: netshape.sh on the Pi 4, NETSHAPE_PEER=192.168.1.108 and NETSHAPE_PORT=50053,5202.
  - Only the draft's RPC connection (the laptop's port 50053) and a measurement port share the shaped link; the ssh and
    HTTP from the laptop are not shaped.
  - The settings, as nominal netem rates: s0 unshaped, b100, b50, b35, b25, b15, and s0r unshaped again (the drift).
  - Each setting's link is MEASURED on port 5202 (measure_link.sh, PAIR="pi4 laptop", receiver-timed) before its requests.
  - Its predictions are re-priced at that measured link. The nominal predictions are registered too.
- The sessions: base, pi4d1, lapd1 through every setting in order (one server load; the shaping changes between
  settings), then pi4d1 and base again. Greedy decoding on the same prompts accepts the same tokens in every arm, so
  between arms only the draft's costs differ.

The model (step 3's, from its registered parts, with the hidden-state upload added: under shaping it is no longer
negligible):
  t_lap(bw)  = c_lap + rtt + (H + V) / bw     c_lap = the laptop's Q8_0 per-byte cost x the bytes a draft step reads
  i_lap(bw)  = q_lap P + rtt + (P H + V) / bw   the draft reading each prompt once (P tokens) before it can draft
  t_pi4      = step 3's 172.84 ms;   i_pi4 = q_pi4 P
  S(b)       = T(1) / ((T(2) + t_b) / tau(1) + i_b / (n - 1))   n = 64 tokens per request
H = n_embd x 4 (4,096 bytes), V = n_vocab x 4 (607,748 bytes), bw = the measured goodput in bytes per ms (the slower
direction), rtt = the measured ping. ICMP is not shaped, so the ping stays near the unshaped value.

Hypotheses, decided here before any data:
  SH-FLIP   at every setting where the re-priced predictions of lapd1 and pi4d1 differ by >= 3 %, the arm predicted faster
            measures faster (ms per token over the four prompts; pi4d1 pooled over its two sessions); at least two settings
            must qualify
  SH-CROSS  the crossover goodput, interpolated from the measured gaps (lapd1 / pi4d1 - 1, linear in 1 / bw between the two
            settings that bracket the sign change), lies within a factor 1.5 of the re-priced prediction's crossover
  SH-ACC    every lapd1 setting's measured speedup (base pooled / lapd1) within 10 % of its re-priced prediction
  SH-SLOPE  c = the slope through the origin of the measured ms/token increase of lapd1 over s0 against the predicted
            increase, over the shaped settings; c in [0.75, 1.25] (the link term priced right)
The rule is confirmed under shaping when SH-FLIP and SH-CROSS hold.
"""
import datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
from control import gguf                                    # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "spec_shape_predictions_20261010.json")
STEP3_REG = os.path.join(DS, "spec_serve_predictions_20261010.json")
STEP3_RUN = os.path.join(ROOT, "testbed", "results", "spec_serve_20261010_024045", "run.jsonl")
DRAFT_LOCAL = os.path.expanduser("~/.cache/heteroserve-models/Qwen3-0.6B-Q8_0.gguf")
LAPTOP = "192.168.1.108"
SETTINGS = [("s0", 0), ("b100", 100), ("b50", 50), ("b35", 35), ("b25", 25), ("b15", 15), ("s0r", 0)]
PORTS = "50053,5202"                  # the draft's RPC server, and the port its shaped link is measured on
MEASURE_PORT = 5202
PAYLOAD = 1448.0 / 1514.0             # TCP payload per Ethernet frame: netem's rate counts whole frames (measured 2026-10-09)
ORDER = ["base", "pi4d1", "lapd1", "pi4d1", "base"]
FLIP_MIN, CROSS_FACTOR, ACC_PCT, SLOPE_BAND = 0.03, 1.5, 10.0, (0.75, 1.25)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def parts():
    """The model's parts: step 3's registered ones, the draft's width from its GGUF, and the four prompts' measured lengths."""
    s3 = json.load(open(STEP3_REG))
    mp = s3["model_parts"]
    shp = gguf.shape(DRAFT_LOCAL)
    ids = [p["id"] for p in s3["prompts"]][0::2]
    pn = {}
    for line in open(STEP3_RUN):
        r = json.loads(line)
        if r.get("timings") and r.get("id") in ids:
            pn.setdefault(r["id"], set()).add(r["timings"]["prompt_n"])
    assert all(len(pn[i]) == 1 for i in ids), pn                 # one prompt length per prompt, every session
    P = sum(next(iter(pn[i])) for i in ids) / float(len(ids))
    params = mp["draft_ingest_ms"]["pi4"] / (mp["per_param_ms_q8"]["pi4"] * mp["mean_prompt_tokens"])   # step 3's draft params
    return {"T1": mp["T_ms"]["1"], "T2": mp["T_ms"]["2"], "tau1": mp["tau"]["1"], "n": s3["n_predict"],
            "c_lap": mp["per_byte_ms_q8"]["laptop"] * mp["draft_step_bytes"], "t_pi4": mp["draft_step_ms"]["pi4"],
            "q_lap": mp["per_param_ms_q8"]["laptop"] * params, "q_pi4": mp["per_param_ms_q8"]["pi4"] * params,
            "H": 4.0 * shp["n_embd"], "V": 4.0 * mp["n_vocab"], "P": P, "prompt_ids": ids,
            "prompt_n": {i: next(iter(pn[i])) for i in ids}, "rtt_registered": mp["laptop_rtt_ms_from_pi4"],
            "bw_gigabit": mp["bw_bytes_per_ms"], "draft_params": params}


def speedups(m, bw, rtt):
    """{lapd1, pi4d1}: predicted speedup over base at goodput bw (bytes per ms) and ping rtt (ms)."""
    n1 = m["n"] - 1
    t_lap = m["c_lap"] + rtt + (m["H"] + m["V"]) / bw
    i_lap = m["q_lap"] * m["P"] + rtt + (m["P"] * m["H"] + m["V"]) / bw
    i_pi4 = m["q_pi4"] * m["P"]
    S = lambda t, i: m["T1"] / ((m["T2"] + t) / m["tau1"] + i / n1)
    return {"lapd1": S(t_lap, i_lap), "pi4d1": S(m["t_pi4"], i_pi4), "t_lap_ms": t_lap, "i_lap_ms": i_lap}


def crossover(m, rtt):
    """The goodput (bytes per ms) where lapd1 and pi4d1 are predicted equal (bisection on 1..1e6)."""
    f = lambda bw: speedups(m, bw, rtt)["lapd1"] - speedups(m, bw, rtt)["pi4d1"]
    lo, hi = 1.0, 1e6
    if f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(200):
        mid = (lo * hi) ** 0.5
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo * hi) ** 0.5


def mbit(bw):
    return bw * 8.0 / 1000.0


def build():
    s3 = json.load(open(STEP3_REG))
    m = parts()
    nominal = {}
    for sid, rate in SETTINGS:
        bw = m["bw_gigabit"] if rate == 0 else rate * 1e6 * PAYLOAD / 8.0 / 1000.0
        s = speedups(m, bw, m["rtt_registered"])
        nominal[sid] = {"rate_mbit_nominal": rate or None, "goodput_mbit": round(mbit(bw), 2), "lapd1": round(s["lapd1"], 4),
                        "pi4d1": round(s["pi4d1"], 4), "gap_pct": round(100 * (s["pi4d1"] / s["lapd1"] - 1), 2),
                        "faster": "lapd1" if s["lapd1"] > s["pi4d1"] else "pi4d1", "t_lap_ms": round(s["t_lap_ms"], 2)}
    x = crossover(m, m["rtt_registered"])
    prompts = [p for p in s3["prompts"] if p["id"] in m["prompt_ids"]]
    return {"registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "name": "spec_shape_20261010",
            "question": "where the draft belongs as the Pi 4 <-> laptop link slows (the user: \"Draft under shaping\")",
            "model": s3["model"], "ctx": s3["ctx"], "place": s3["place"], "place_head": s3["place_head"],
            "n_predict": s3["n_predict"], "host_url": s3["host_url"], "outlets": s3["outlets"], "chat_format": s3["chat_format"],
            "sampling": s3["sampling"], "warmup_prompt": s3["warmup_prompt"], "prompts": prompts,
            "idle_before_s": s3["idle_before_s"], "idle_after_s": s3["idle_after_s"],
            "laptop_rpc_server": s3["laptop_rpc_server"], "models_on_host": s3["models_on_host"],
            "arms": {"base": {"env": {"SERVE_DRAFT": "none"}}, "pi4d1": s3["arms"]["pi4d1"],
                     "lapd1": dict(s3["arms"]["lapd1"], laptop_rpc="%s:50053" % LAPTOP)},
            "order": ORDER, "shaped_arm": "lapd1",
            "settings": [{"id": sid, "rate_mbit": rate or None} for sid, rate in SETTINGS],
            "shaping": {"peer": LAPTOP, "ports": PORTS, "measure_port": MEASURE_PORT,
                        "netshape": "NETSHAPE_PEER=%s NETSHAPE_PORT=%s netshape.sh apply <rate> 0" % (LAPTOP, PORTS),
                        "measure": "PAIR=\"pi4 laptop\" MEASURE_PORT=%d MIB=max(16, rate) REPS=3 measure_link.sh" % MEASURE_PORT},
            "model_parts": {k: v for k, v in m.items()},
            "predicted_nominal": nominal,
            "predicted_crossover_goodput_mbit": round(mbit(x), 2) if x else None,
            "hypotheses": [
                {"id": "SH-FLIP", "claim": "at every setting where the re-priced lapd1 and pi4d1 predictions differ by >= 3 %, the "
                                           "arm predicted faster measures faster; at least two settings qualify"},
                {"id": "SH-CROSS", "claim": "the measured crossover goodput (linear in 1/bw between the settings bracketing the sign "
                                            "change of lapd1/pi4d1 - 1) within a factor 1.5 of the re-priced predicted crossover"},
                {"id": "SH-ACC", "claim": "every lapd1 setting's measured speedup (base pooled / lapd1) within 10 % of its re-priced "
                                          "prediction"},
                {"id": "SH-SLOPE", "claim": "the slope through the origin of lapd1's measured ms/token increase over s0 against the "
                                            "predicted increase, over the shaped settings, in [0.75, 1.25]"}],
            "confirmed_when": "SH-FLIP and SH-CROSS hold",
            "notes": ["the links are measured in a pre-phase with nothing loaded (measure_link.sh refuses a busy board); each "
                      "setting's shaping is then reapplied for its requests (netem's rate is deterministic)",
                      "the laptop is the draft's board; the user's aunsync timer (a Google Drive sync pass every 15 min) is left "
                      "as it is, and its pass times are read from the user journal afterwards and reported beside the laptop "
                      "arm's settings",
                      "energy is logged at 5 Hz (the Pi 4 and Jetson outlets) and reported, not judged",
                      "the smoke (spec_shape_smoke_20261010_124238, never scored) ran the plumbing on a scratch copy of this "
                      "registration first"],
            "thresholds": {"flip_min": FLIP_MIN, "cross_factor": CROSS_FACTOR, "acc_pct": ACC_PCT, "slope_band": list(SLOPE_BAND)},
            "inputs_sha256": {**{os.path.relpath(p, ROOT): sha(p) for p in (STEP3_REG, STEP3_RUN, DRAFT_LOCAL)},
                              **{"testbed/scripts/" + f: sha(os.path.join(HERE, f))
                                 for f in ("register_spec_shape_20261010.py", "spec_shape_20261010.py", "netshape.sh",
                                           "measure_link.sh", "link_to_json.py", "serve_pool.sh", "serve_defaults.sh")
                                 if os.path.exists(os.path.join(HERE, f))}}}


def main(argv):
    out = OUT
    if "--out" in argv:                 # a smoke run's plumbing only, never the dataset's file
        out = os.path.abspath(argv[argv.index("--out") + 1])
        if os.path.dirname(out) == DS:
            sys.exit("refused: --out is for a smoke run, never testbed/dataset")
    if os.path.exists(out) and "--dry-run" not in argv:
        sys.exit("refused: %s exists; a registration is never overwritten" % os.path.relpath(out, ROOT))
    reg = build()
    m = reg["model_parts"]
    print("parts: c_lap %.2f ms, t_pi4 %.2f ms, q_lap %.3f / q_pi4 %.3f ms per prompt token, P %.2f, H %d, V %d" % (
        m["c_lap"], m["t_pi4"], m["q_lap"], m["q_pi4"], m["P"], m["H"], m["V"]))
    for sid, v in reg["predicted_nominal"].items():
        print("%-4s goodput %7.2f Mbit/s  lapd1 %.4f  pi4d1 %.4f  gap %+6.2f %%  faster %s" % (
            sid, v["goodput_mbit"], v["lapd1"], v["pi4d1"], v["gap_pct"], v["faster"]))
    print("predicted crossover: %s Mbit/s of goodput" % reg["predicted_crossover_goodput_mbit"])
    if "--dry-run" in argv:
        print("dry run: nothing written")
        return reg
    json.dump(reg, open(out, "w"), indent=1, ensure_ascii=False)
    os.chmod(out, 0o444)
    print("registered %s at %s, sha256 %s (read-only)" % (os.path.relpath(out, ROOT), reg["registered"], sha(out)[:16]))
    return reg


if __name__ == "__main__":
    main(sys.argv[1:])
