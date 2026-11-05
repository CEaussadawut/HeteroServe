#!/usr/bin/env python3
"""Register hm1 (2026-11-04; the user: "do all four follow-up studies": a memory check before the scheduler plans a
prefill/decode hand-off).

    PHASE_MODEL=qwen3_0.6b python3 testbed/scripts/register_hm1_20261104.py --mem <0.6B's mem.json> --mem <4B's mem.json> [--dry-run]

The check (phase_plan_20261028.handoff_fits, HEADROOM_BYTES 256 MiB): the hand-off of an n-token prompt is planned only when its
state (the calibration's fixed + per-token bytes) plus the headroom is within the receiver's MemAvailable with its server loaded;
phase_serve_20261028.run()'s guard checks the same once the file has landed (below the headroom: the file is deleted and the
laptop decodes the rest). Its inputs, measured before this registration (handoff_mem_20261104.py measure, mem.json): the
Jetson's MemAvailable after a warm-up with Qwen3-0.6B's server at 9,216 cells and Qwen3-4B's at 2,560.
Reported, not tested (the calibration the headroom must agree with): the check's verdict on the eight hand-offs measured before,
against what happened (0.6B: 1,036 / 4,108 / 8,204 tokens twice each, clean; 4B: 524 clean, 1,036 thrashed).
The test, at a context the Jetson has not served: Qwen3-0.6B at 16,384 cells.
  HM-READY  the Jetson's MemAvailable after the warm-up within 64 MiB of the prediction: the 9,216-cell figure less the KV
            cache's growth (7,168 cells x 114,688 bytes; nothing else assumed to grow)
  HM-AGREE  the live check's verdict on every case (at the live MemAvailable) is the one registered here
  HM-ALLOW  every case the live check allows completes its hand-off: the restore within 5 s and the Jetson's SwapFree down by
            less than 32 MiB over the request
  HM-GUARD  every case the guard refuses goes to the laptop, its tokens equal plan L's on the same prompt, and the Jetson's SwapFree
            falls by less than 32 MiB
Cases (N* = the largest prompt the check allows at the predicted MemAvailable):
  a1  n = 0.5 N*, allowed            a2  n = 0.85 N*, allowed
  g1  n = 1.05 N*, refused: the planner leaves H out; run as H to exercise the guard at its line (after landing, about 0.05 N*
      of state short of the headroom: no thrash), but only while the live margin is no more than 50 MiB past the headroom (the
      landing then leaves at least 270 MiB, above the 255 MiB the 4B thrashed at; otherwise it is not landed); compared with L
  g2  n = 1,024 with the guard's headroom forced to 8 GiB, refused whatever the memory; compared with L
17 tokens generated per request, as the calibration's hand-offs. Run by handoff_mem_20261104.py run (PHASE_MODEL=qwen3_0.6b).
Writes testbed/dataset/hm1_predictions_20261104.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import phase_plan_20261028 as PP                           # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "hm1_predictions_20261104.json")
COSTS06 = "testbed/dataset/phase_costs_20261028.json"
COSTS4B = "testbed/dataset/phase_costs_qwen3_4b_20261103.json"
KV06 = 28 * 8 * 128 * 2 * 2                                  # Qwen3-0.6B: bytes of f16 K and V per cache cell
CTX_NEW, MIB, GEN, QUESTION = 16384, 1024 * 1024, 17, 12
# the hand-offs measured before: model, context, prompt tokens, the state's bytes as saved (None: the 0.6B's state model, which
# its calibration fitted; the 4B's calibration fitted none, so its two saved sizes are used), what happened
PAST = [("qwen3_0.6b", 9216, 1036, None, "clean"), ("qwen3_0.6b", 9216, 4108, None, "clean"), ("qwen3_0.6b", 9216, 8204, None, "clean"),
        ("qwen3_4b", 2560, 524, 77276236, "clean"), ("qwen3_4b", 2560, 1036, 152781900, "thrashed")]


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem", required=True, action="append"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    mem = {(r["phase_model"], r["ctx"]): r for f in a.mem for r in json.load(open(f))}
    costs = {"qwen3_0.6b": PP.load(os.path.join(ROOT, COSTS06)), "qwen3_4b": PP.load(os.path.join(ROOT, COSTS4B))}
    hr = PP.HEADROOM_BYTES
    past = []
    for model, ctx, n, saved, outcome in PAST:
        avail = mem[(model, ctx)]["warm"]["MemAvailable"]
        if saved is None:
            ok, nbytes, margin = PP.handoff_fits(costs[model], n, avail, hr)
        else:
            nbytes, margin = saved, avail - saved - hr
            ok = margin >= 0
        past.append({"model": model, "ctx": ctx, "prompt_n": n, "outcome": outcome, "avail_mib": round(avail / MIB, 1),
                     "state_mib": round(nbytes / MIB, 1), "margin_mib": round(margin / MIB, 1), "check": "allow" if ok else "refuse",
                     "agrees": ok == (outcome == "clean")})
    c = costs["qwen3_0.6b"]
    pred_avail = mem[("qwen3_0.6b", 9216)]["warm"]["MemAvailable"] - (CTX_NEW - 9216) * KV06
    nstar = int((pred_avail - hr - c["state"]["fixed_bytes"]) / c["state"]["bytes_per_token"]) - 1     # the largest prompt_n
    cases = []
    for cid, frac, ghr in (("a1", 0.5, hr), ("a2", 0.85, hr), ("g1", 1.05, hr)):
        pn = int(frac * nstar)
        ok, nbytes, margin = PP.handoff_fits(c, pn, pred_avail, hr)
        cases.append({"id": cid, "ctx": CTX_NEW, "n": pn - QUESTION, "prompt_n": pn, "gen": GEN, "guard_headroom": ghr,
                      "registered_verdict": "allow" if ok else "refuse", "state_mib": round(nbytes / MIB, 1),
                      "margin_mib_at_predicted": round(margin / MIB, 1), "compare_L": not ok,
                      **({} if ok else {"max_overrun_mib": 50})})
    cases.append({"id": "g2", "ctx": CTX_NEW, "n": 1024, "prompt_n": 1024 + QUESTION, "gen": GEN, "guard_headroom": 8 * 1024 * MIB,
                  "registered_verdict": "allow", "forced_refusal": True, "compare_L": True})
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_hm1_20261104.py", "runner": "testbed/scripts/handoff_mem_20261104.py",
           "scorer": "testbed/scripts/score_hm1_20261104.py", "out_prefix": "hm1", "phase_model": "qwen3_0.6b", "costs": COSTS06, "headroom_bytes": hr,
           "what": "the hand-off's memory check (phase_plan.handoff_fits and phase_serve's guard) at a context the Jetson has not served",
           "measured_before": [os.path.relpath(os.path.abspath(f), ROOT) for f in a.mem],
           "past_handoffs": past, "past_all_agree": all(p["agrees"] for p in past),
           "prediction": {"ctx": CTX_NEW, "avail_mib": round(pred_avail / MIB, 1), "kv_growth_mib": round((CTX_NEW - 9216) * KV06 / MIB, 1),
                          "nstar_prompt_n": nstar, "tol_mib": 64},
           "cases": cases,
           "hypotheses": {
               "HM-READY": {"tol_mib": 64, "claim": "MemAvailable after the warm-up at 16,384 cells within 64 MiB of %.0f MiB" % (pred_avail / MIB)},
               "HM-AGREE": {"claim": "the live check's verdict on every case is the registered one"},
               "HM-ALLOW": {"restore_s": 5.0, "swap_drop_mib": 32,
                            "claim": "every case the live check allows restores within 5 s, SwapFree down < 32 MiB"},
               "HM-GUARD": {"swap_drop_mib": 32,
                            "claim": "every guarded refusal decodes on the laptop with plan L's tokens, SwapFree down < 32 MiB"}},
           "conditions": "Zen may stay open (QUIET_ALLOW_APPS=zen, the owner, 4 Nov); no energy measured",
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/register_hm1_20261104.py", "testbed/scripts/handoff_mem_20261104.py",
               "testbed/scripts/score_hm1_20261104.py",
               "testbed/scripts/phase_plan_20261028.py", "testbed/scripts/phase_serve_20261028.py", "testbed/scripts/pd_handoff.py",
               "testbed/scripts/jetson_proc_20261028.py", "testbed/scripts/jetson_power_guard.sh", COSTS06, COSTS4B,
               *[os.path.relpath(os.path.abspath(f), ROOT) for f in a.mem])}}
    for p in past:
        print("past %-10s ctx %5d n %5d: %-8s check %-6s (avail %.0f, state %.0f, margin %+.0f MiB) %s" % (
            p["model"], p["ctx"], p["prompt_n"], p["outcome"], p["check"], p["avail_mib"], p["state_mib"], p["margin_mib"],
            "agrees" if p["agrees"] else "DISAGREES"))
    print("ctx 16,384: predicted MemAvailable %.0f MiB, N* %d tokens" % (pred_avail / MIB, nstar))
    for cs in cases:
        print("case %s: %s" % (cs["id"], cs))
    if a.dry_run:
        print("dry run: nothing written"); return
    if not reg["past_all_agree"]:
        raise SystemExit("refused: the headroom disagrees with a past hand-off; set it from the evidence first")
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))


if __name__ == "__main__":
    main()
