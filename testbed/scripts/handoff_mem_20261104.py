#!/usr/bin/env python3
"""The hand-off's memory check (hm1, 2026-11-04; the user: "do all four follow-up studies": a memory check before the scheduler
plans a prefill/decode hand-off).

    PHASE_MODEL=<model> python3 testbed/scripts/handoff_mem_20261104.py measure --ctx 9216 [--ctx ...]
    PHASE_MODEL=qwen3_0.6b python3 testbed/scripts/handoff_mem_20261104.py run --reg testbed/dataset/hm1_predictions_20261104.json

Why: the hand-off (plan H) streams the laptop's slot file into the Jetson's tmpfs, which is RAM, and the file stays there while
the Jetson decodes. Qwen3-4B's hand-off thrashed the 4 GB Jetson from 1,024 tokens (a 152.8 MB state; it stopped answering
ssh and needed a power cycle) while Qwen3-0.6B's 941 MB state at 8,204 tokens never did. phase_plan_20261028.handoff_fits()
now leaves H out when the state does not fit the receiver's MemAvailable less a headroom, and phase_serve_20261028.run()
checks again once the file has landed (its guard: below the headroom, the file is deleted and the laptop decodes the rest).
measure: the Jetson's memory with its phase server loaded, for the model in PHASE_MODEL at each --ctx: the Jetson's procedure
  (jetson_proc_20261028: its guard, its clocks), /proc/meminfo before the server starts, at READY, and after a warm-up request
  (512 tokens, 8 generated: a server's first request builds its compute graphs). Nothing else runs on the Jetson.
run: the registration's cases, one context at a time: the Jetson's procedure, both phase servers at that context, the prompt
  tokenized on both and compared; per case: the Jetson's MemAvailable now, the planner's verdict (handoff_fits with the
  registered headroom), then plan H with the case's guard headroom, the Jetson's SwapFree before and after; where the case
  says so, plan L on the same prompt to compare the tokens.
Raw output: testbed/results/hmem_<stamp>/ or hm1_<stamp>/ (run.log, mem.json or results.json), and the phase servers' logs.
"""
import argparse, datetime, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import phase_serve_20261028 as S                           # noqa: E402
import phase_plan_20261028 as PP                           # noqa: E402
import jetson_proc_20261028 as JP                           # noqa: E402

MIB = 1024 * 1024


def folder(prefix):
    d = os.path.join(ROOT, "testbed", "results", "%s_%s" % (prefix, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    os.makedirs(d)
    return d


def measure(ctxs):
    d = folder("hmem")
    log = S.Log("hmem", os.path.join(d, "run.log"))
    out = []
    for ctx in ctxs:
        S.CTX = ctx
        tag = "hmem_%s_%d" % (S.PHASE_MODEL, ctx)
        st = JP.up(lambda s: log("jetson", s), tag)
        rec = {"model": S.MODEL["name"], "phase_model": S.PHASE_MODEL, "ctx": ctx}
        try:
            rec["before"] = S.meminfo("jetson")
            S.start("jetson", log, tag)
            rec["ready"] = S.meminfo("jetson")
            ids = S.prompt_ids("jetson", 512)
            S.complete("jetson", ids, 8, log, tag + " warm-up")
            rec["warm"] = S.meminfo("jetson")
        finally:
            S.stop("jetson", log)
            JP.down(lambda s: log("jetson", s), st)
        log("mem", rec)
        out.append(rec)
        json.dump(out, open(os.path.join(d, "mem.json"), "w"), indent=1)
    for r in out:
        print("%s ctx %d: MemAvailable before %.0f MiB, ready %.0f, after the warm-up %.0f; SwapFree %.0f" % (
            r["phase_model"], r["ctx"], r["before"]["MemAvailable"] / MIB, r["ready"]["MemAvailable"] / MIB,
            r["warm"]["MemAvailable"] / MIB, r["warm"]["SwapFree"] / MIB))


def run(regpath):
    reg = json.load(open(regpath))
    if reg["phase_model"] != S.PHASE_MODEL:
        raise SystemExit("FATAL: the registration's model is %s; run with PHASE_MODEL=%s" % (reg["phase_model"], reg["phase_model"]))
    import hashlib
    sha = lambda p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()
    changed = sorted(f for f, h in reg["inputs_sha256"].items() if sha(f) != h)
    if changed:
        raise SystemExit("refused: registered inputs changed: %s" % changed)
    costs = PP.load(os.path.join(ROOT, reg["costs"]))
    d = folder(reg["out_prefix"])
    log = S.Log(reg["out_prefix"], os.path.join(d, "run.log"))
    log("start", {"registration": os.path.relpath(regpath, ROOT), "sha256": sha(os.path.relpath(regpath, ROOT))[:16]})
    res = []
    for ctx in sorted({c["ctx"] for c in reg["cases"]}):
        S.CTX = ctx
        tag = "%s_ctx%d" % (os.path.basename(d), ctx)
        st = JP.up(lambda s: log("jetson", s), tag)
        try:
            S.start("laptop", log, tag); S.start("jetson", log, tag)
            S.complete("jetson", S.prompt_ids("jetson", 256), 4, log, tag + " warm-up jetson")
            S.complete("laptop", S.prompt_ids("laptop", 256), 4, log, tag + " warm-up laptop")
            ready = S.meminfo("jetson")
            log("ready", {"ctx": ctx, "jetson": ready})
            for case in [c for c in reg["cases"] if c["ctx"] == ctx]:
                ids = S.prompt_ids("laptop", case["n"])
                if S.prompt_ids("jetson", case["n"]) != ids:
                    raise SystemExit("FATAL: the two servers tokenize the prompt differently")
                before = S.meminfo("jetson")
                ok, nbytes, margin = PP.handoff_fits(costs, len(ids), before["MemAvailable"], reg["headroom_bytes"])
                verdict = {"fits": ok, "state_bytes_model": nbytes, "margin_bytes": margin, "avail": before["MemAvailable"]}
                log("check", {"case": case["id"], **verdict})
                rec = {"case": case["id"], "ctx": ctx, "n": case["n"], "prompt_n": len(ids), "verdict": verdict, "before": before}
                # a refused case runs as H only to exercise the guard, and only while the landing stays clear of the level the 4B
                # thrashed at (its max_overrun_mib below the headroom); past that it is not landed at all
                if not ok and "max_overrun_mib" in case and margin < -case["max_overrun_mib"] * MIB:
                    rec["skipped"] = "the live margin %.0f MiB is past the case's %d MiB: not landed" % (margin / MIB, case["max_overrun_mib"])
                    log("case", rec)
                    res.append(rec)
                    json.dump(res, open(os.path.join(d, "results.json"), "w"), indent=1)
                    continue
                try:
                    toks, r = S.run("H", ids, case["gen"], log, "%s H" % case["id"], guard_headroom=case["guard_headroom"])
                    rec.update({"H": r, "tokens_H": toks})
                except Exception as ex:
                    rec["error"] = repr(ex)[:300]
                    log("ERROR", {"case": case["id"], "error": rec["error"]})
                rec["after"] = S.meminfo("jetson")
                if case.get("compare_L") and "error" not in rec:
                    toksL, rL = S.run("L", ids, case["gen"], log, "%s L" % case["id"])
                    rec.update({"L": rL, "tokens_L": toksL})
                log("case", {k: v for k, v in rec.items() if not k.startswith("tokens")})
                res.append(rec)
                json.dump(res, open(os.path.join(d, "results.json"), "w"), indent=1)
        finally:
            S.stop("laptop", log); S.stop("jetson", log)
            JP.down(lambda s: log("jetson", s), st)
    log("RUN DONE", {"cases": len(res)})
    import subprocess
    s = subprocess.run([sys.executable, os.path.join(ROOT, reg["scorer"]), d, "--registration", regpath], cwd=ROOT,
                       capture_output=True, text=True)
    log("scorer", "rc %d\n%s%s" % (s.returncode, s.stdout, s.stderr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("measure", "run"))
    ap.add_argument("--ctx", type=int, action="append"); ap.add_argument("--reg")
    a = ap.parse_args()
    if a.cmd == "measure":
        measure(a.ctx or [S.CTX])
    else:
        run(a.reg)


if __name__ == "__main__":
    main()
