#!/usr/bin/env python3
"""Register the two-user test's predictions (TWOUSER) before any of its runs. From the repo root:

    python3 .handoff/assets/pd-2026-09-29/register_twouser.py [--dry-run]              # TWOUSER: 32-token replies
    python3 .handoff/assets/pd-2026-09-29/register_twouser.py --gen 128 [--dry-run]    # TWOUSER128: the reply-length reversal

The question (remaining work item 6): does the prefill/decode hand-off earn its place? "First, cheaply: batched decode on each
board; then: the Pi 5 alone with 2 slots, against Pi 5 prefill -> Pi 4 decode for the second user." User, 2026-09-29 evening:
"... then prefill/decode separation". The test is testbed/scripts/pd_twouser.py run --reps 3 --n 128 --gen 32 --stagger 0:
two users, 128-token prompts, 32 tokens each, both waiting at t = 0; arms shared, handoff, static.

Every prediction is priced from rows measured before this registration, read from their logs here:
- single-user llama-server figures at 128 tokens, 3 repetitions each (the 01:08 sweep, pd_sweep_20260929_010837.log): the Pi 5's
  and the Pi 4's prefill and decode alone; the hand-off's save, transfer (wall time with its two ssh logins, which is what
  pd_twouser.py pays), restore, and the Pi 4's first step after a restore;
- what a second sequence costs the Pi 5 (llama-batched-bench, batched_decode_20260929_202511, 3 rounds): the ratio of its
  2-sequence to its 1-sequence prefill and decode-step times, applied to the sweep's figures.
The shared arm is priced two ways, because which of llama-server's two orders happens is not controlled: both prompts in one
batch, or user 1's first and user 2's with user 1's next decode token. Its aggregates barely depend on it; its TTFTs do.

Writes testbed/dataset/twouser_predictions_20260929.json, refuses if it exists, leaves it read-only; --dry-run writes nothing.
"""
import datetime, hashlib, json, os, re, stat, statistics, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
GEN_ARG = int(sys.argv[sys.argv.index("--gen") + 1]) if "--gen" in sys.argv else 32
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
OUT = os.path.join(ROOT, "testbed", "dataset", "twouser_predictions_20260929%s%s.json" % ("" if GEN_ARG == 32 else "_gen%d" % GEN_ARG, TAG))
# --tag b (TWOUSER128b): the same predictions for the tool that sets ignore_eos, superseding the gen-128 registration whose run was
# stopped because user 1's answer ended at 61 tokens (EOS), so the arms did work the predictions did not price.
SUPERSEDES = {"b": {"file": "testbed/dataset/twouser_predictions_20260929_gen128.json",
                    "why": "its run (testbed/results/pd_twouser_20260929_224035.log) was stopped at 22:48 after two arms: user 1's reply "
                           "ended at 61 tokens (stop_type eos), user 2's at 128, so the arms did work the predictions did not price. "
                           "pd_twouser.py now sets ignore_eos (checked on the laptop with user 1's own prompt: 60 predicted tokens and "
                           "eos without it, 80 and limit with it)."}}
TWOUSER_RESULTS = os.path.join(ROOT, "testbed", "dataset", "twouser_results_20260929.json")   # the gen-32 outcome (gen != 32 only)
SWEEP = os.path.join(ROOT, "testbed", "results", "pd_sweep_20260929_010837.log")
BATCH = os.path.join(ROOT, "testbed", "results", "batched_decode_20260929_202511", "pi5.log")
BATCH_NR = os.path.join(ROOT, "testbed", "results", "batched_decode_20260929_211529", "pi5.log")    # the Pi 5 --no-repack
TOOL = os.path.join(ROOT, "testbed", "scripts", "pd_twouser.py")
N, GEN = 128, GEN_ARG


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sweep_rows():
    res, cont = {"pi5-alone": [], "pi4-alone": [], "handoff": []}, []
    for line in open(SWEEP):
        m = re.match(r"\S+ n%d r\d (pi5-alone|pi4-alone|handoff) result (\{.*\})$" % N, line.strip())
        if m:
            res[m.group(1)].append(json.loads(m.group(2)))
        m = re.match(r"\S+ n%d r\d handoff decode-pi4 (\{.*\})$" % N, line.strip())
        if m:
            cont.append(json.loads(m.group(1))["timings"])
    return res, cont


def batch_rows(path):
    rows = {}
    for line in open(path):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 10 and c[0].isdigit() and int(c[0]) == N:
            rows.setdefault(int(c[2]), []).append((float(c[4]), float(c[6]) / int(c[1])))   # T_PP s, s per decode step
    return {b: (statistics.fmean(t for t, _ in v), statistics.fmean(x for _, x in v), len(v)) for b, v in rows.items()}


def batch_ratios():
    rows = {}
    for line in open(BATCH):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 10 and c[0].isdigit() and int(c[0]) == N:
            rows.setdefault(int(c[2]), []).append((float(c[4]), float(c[6]) / int(c[1])))   # T_PP s, s per decode step
    one, two = rows[1], rows[2]
    return (statistics.fmean(t for t, _ in two) / (2 * statistics.fmean(t for t, _ in one)),
            statistics.fmean(s for _, s in two) / statistics.fmean(s for _, s in one), len(one), len(two))


def main(argv):
    dry = "--dry-run" in argv
    if os.path.exists(OUT) and not dry:
        sys.exit("%s exists: a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    res, cont = sweep_rows()
    assert all(len(v) == 3 for v in res.values()) and len(cont) == 3, {k: len(v) for k, v in res.items()}
    m = lambda arm, k: statistics.fmean(r[k] for r in res[arm])
    p5_pre, p5_dec = m("pi5-alone", "prefill_ms") / 1000, m("pi5-alone", "decode_ms_per_token") / 1000
    p4_pre, p4_dec = m("pi4-alone", "prefill_ms") / 1000, m("pi4-alone", "decode_ms_per_token") / 1000
    save, xfer, restore = m("handoff", "save_ms") / 1000, m("handoff", "transfer_wall_s_with_ssh"), m("handoff", "restore_ms") / 1000
    p4_first = statistics.fmean(t["prompt_ms"] for t in cont) / 1000             # the Pi 4 processes t1: its first step
    p4_after = m("handoff", "decode_ms_per_token") / 1000
    k_pp2, k_tg2, n1, n2 = batch_ratios()
    # shared_nr: the same arm with the Pi 5's server --no-repack, priced from llama-batched-bench's no-repack rows, turned into
    # llama-server terms by the ratio the repacked configuration shows between the two tools at one sequence.
    rep, nr = batch_rows(BATCH), batch_rows(BATCH_NR)
    srv_pp, srv_tg = p5_pre / rep[1][0], p5_dec / rep[1][1]
    nr_pre1, nr_pre2, nr_dec1, nr_step2 = nr[1][0] * srv_pp, nr[2][0] * srv_pp, nr[1][1] * srv_tg, nr[2][1] * srv_tg
    na_t1 = nr_pre2
    na_done = na_t1 + (GEN - 1) * nr_step2
    nb_t1_1 = nr_pre1
    nb_t1_2 = nb_t1_1 + nr_pre1 * (N + 1) / N
    nb_done1 = nb_t1_2 + (GEN - 2) * nr_step2
    nb_done2 = nb_done1 + nr_dec1
    step2 = p5_dec * k_tg2
    # shared, order A: both prompts in one batch; order B: user 1's prompt, then user 2's with user 1's next token
    a_t1 = 2 * p5_pre * k_pp2
    a_done = a_t1 + (GEN - 1) * step2
    b_t1_1 = p5_pre
    b_t1_2 = b_t1_1 + p5_pre * (N + 1) / N
    b_done1 = b_t1_2 + (GEN - 2) * step2
    b_done2 = b_done1 + p5_dec                                                     # its last token alone
    # handoff: user 2 prefilled first and handed off; user 1 served whole by the Pi 5 after user 2's save
    h_t1_2 = p5_pre
    h_done2 = h_t1_2 + save + xfer + restore + p4_first + (GEN - 2) * p4_after
    h_t1_1 = p5_pre + save + p5_pre
    h_done1 = h_t1_1 + (GEN - 1) * p5_dec
    # static: each board serves its own user
    s_done1, s_done2 = p5_pre + (GEN - 1) * p5_dec, p4_pre + (GEN - 1) * p4_dec
    r = lambda x: round(x, 2)
    pred = {
        "shared": {"order_A_one_batch": {"user1": {"ttft_s": r(a_t1), "done_s": r(a_done), "tpot_ms": r(1000 * step2)},
                                         "user2": {"ttft_s": r(a_t1), "done_s": r(a_done), "tpot_ms": r(1000 * step2)}},
                   "order_B_user1_first": {"user1": {"ttft_s": r(b_t1_1), "done_s": r(b_done1)},
                                           "user2": {"ttft_s": r(b_t1_2), "done_s": r(b_done2)}},
                   "mean_done_s": r((a_done + (b_done1 + b_done2) / 2) / 2), "mean_done_range_s": [r(min(a_done, (b_done1 + b_done2) / 2)),
                                                                                                 r(max(a_done, (b_done1 + b_done2) / 2))],
                   "makespan_s": r(max(a_done, b_done2))},
        "handoff": {"user1": {"ttft_s": r(h_t1_1), "done_s": r(h_done1), "tpot_ms": r(1000 * p5_dec)},
                    "user2": {"ttft_s": r(h_t1_2), "done_s": r(h_done2),
                              "tpot_ms": r(1000 * (save + xfer + restore + p4_first + (GEN - 2) * p4_after) / (GEN - 1))},
                    "mean_done_s": r((h_done1 + h_done2) / 2), "makespan_s": r(max(h_done1, h_done2)),
                    "mean_ttft_s": r((h_t1_1 + h_t1_2) / 2)},
        "shared_nr": {"order_A_one_batch": {"user1": {"ttft_s": r(na_t1), "done_s": r(na_done), "tpot_ms": r(1000 * nr_step2)},
                                            "user2": {"ttft_s": r(na_t1), "done_s": r(na_done), "tpot_ms": r(1000 * nr_step2)}},
                      "order_B_user1_first": {"user1": {"ttft_s": r(nb_t1_1), "done_s": r(nb_done1)},
                                              "user2": {"ttft_s": r(nb_t1_2), "done_s": r(nb_done2)}},
                      "mean_done_s": r((na_done + (nb_done1 + nb_done2) / 2) / 2),
                      "mean_done_range_s": [r(min(na_done, (nb_done1 + nb_done2) / 2)), r(max(na_done, (nb_done1 + nb_done2) / 2))],
                      "makespan_s": r(max(na_done, nb_done2))},
        "static": {"user1": {"ttft_s": r(p5_pre), "done_s": r(s_done1), "tpot_ms": r(1000 * p5_dec)},
                   "user2": {"ttft_s": r(p4_pre), "done_s": r(s_done2), "tpot_ms": r(1000 * p4_dec)},
                   "mean_done_s": r((s_done1 + s_done2) / 2), "makespan_s": r(max(s_done1, s_done2)),
                   "mean_ttft_s": r((p5_pre + p4_pre) / 2)}}
    reg = {"registered": None,
           "campaign": "2026-09-29 evening: two users at once on Qwen3.5-9B (TWOUSER) -- does the hand-off earn its place?",
           "test": "python3 testbed/scripts/pd_twouser.py run --reps 3 --n %d --gen %d --arms shared,handoff,static,shared_nr --stagger 0"
                   % (N, GEN),
           "inputs_sha256": {os.path.relpath(p, ROOT): sha(p) for p in (SWEEP, BATCH, BATCH_NR, TOOL)},
           "rows": {"pi5_alone": {"prefill_s": r(p5_pre), "decode_s_per_token": round(p5_dec, 4)},
                    "pi4_alone": {"prefill_s": r(p4_pre), "decode_s_per_token": round(p4_dec, 4)},
                    "handoff": {"save_s": round(save, 4), "transfer_wall_s_with_ssh": round(xfer, 3), "restore_s": round(restore, 4),
                                "pi4_first_step_s": round(p4_first, 3), "pi4_decode_s_per_token": round(p4_after, 4)},
                    "pi5_second_sequence": {"prefill_ratio": round(k_pp2, 4), "decode_step_ratio": round(k_tg2, 4),
                                            "rounds": [n1, n2], "source": "llama-batched-bench -npp 128 -ntg 32 -npl 1,2 (B=2 / B=1)"},
                    "pi5_no_repack_llama_server_terms": {"prefill_1_s": r(nr_pre1), "prefill_2_s": r(nr_pre2),
                                                         "decode_1_s_per_token": round(nr_dec1, 4), "decode_step_2_s": round(nr_step2, 4),
                                                         "server_over_bench": {"prefill": round(srv_pp, 4), "decode": round(srv_tg, 4)},
                                                         "source": "llama-batched-bench --no-repack, batched_decode_20260929_211529"}},
           "predicted": pred,
           "rules": [
               "HANDOFF: the hand-off earns its place for two users arriving together if its mean completion time (the mean of the two "
               "users' done, averaged over the repetitions) is lower than BOTH shared arms' -- the Pi 5 alone as the sweep ran it "
               "(repacked) and at its best two-user configuration (--no-repack). Predicted %.1f s against %.1f and %.1f s."
               % (pred["handoff"]["mean_done_s"], pred["shared"]["mean_done_s"], pred["shared_nr"]["mean_done_s"]),
               "NOREPACK: the Pi 5's two-user decode step is cheaper without repacking, in llama-server as in llama-batched-bench, if "
               "shared_nr's measured mean TPOT is below shared's. Predicted %.0f against %.0f ms."
               % (1000 * nr_step2, 1000 * step2),
               "ACCURACY: an arm's prediction holds if its measured mean completion time is within +-10 %% of the predicted one (the "
               "shared arm: of the nearer end of its range). Each user's TTFT, done and TPOT are reported against theirs either way.",
               "STATIC-CHECK: both boards run 2-slot servers (a 1-slot server refuses a 2-slot server's saved state: 'n_stream "
               "mismatch', the smoke run of 21:47), so the static arm's two users should match the sweep's 1-slot pi5-alone and "
               "pi4-alone within 5 %% in both prefill and decode (the servers' own timings). A miss is reported and the other arms "
               "still scored.",
               "STEADY: an arm is steady if its repetitions' mean completion times agree within 5 %; one that is not is reported as found "
               "and still scored."],
           "not_measured": "energy: the rig does not exist yet; the laptop is only the orchestrator here (it timestamps every token)."}
    if GEN != 32:
        # REVERSAL (the question of this registration): the model says a longer reply lets the Pi 5's cheap no-repack two-user
        # decode outweigh its slow prefill, and makes the Pi 4's slow decode of user 2 dominate the hand-off.
        reg["campaign"] = ("2026-09-29 night: two users at once on Qwen3.5-9B with %d-token replies (TWOUSER%d) -- does the answer of "
                           "TWOUSER (32-token replies: the hand-off pays) reverse?" % (GEN, GEN))
        reg["rules"].insert(0,
            "REVERSAL: the answer reverses at %d-token replies if the shared_nr arm's mean completion time is lower than the "
            "hand-off's. Predicted %.1f s against %.1f s. (HANDOFF below is then predicted NOT to hold: the hand-off beats the "
            "repacked shared arm, %.1f s, but not shared_nr.)" % (GEN, pred["shared_nr"]["mean_done_s"], pred["handoff"]["mean_done_s"],
                                                               pred["shared"]["mean_done_s"]))
        # an alternative, not the registered prediction: each arm's gen-32 measured/predicted ratio carried to this length
        tw = json.load(open(TWOUSER_RESULTS))["arms"]
        reg["predicted_alt"] = {"calibrated_on_twouser": {
            arm: {"mean_done_s": r(pred[arm]["mean_done_s"] * tw[arm]["mean_done_s"] / tw[arm]["predicted_mean_done_s"]),
                  "factor": round(tw[arm]["mean_done_s"] / tw[arm]["predicted_mean_done_s"], 4)} for arm in pred}}
        reg["inputs_sha256"][os.path.relpath(TWOUSER_RESULTS, ROOT)] = sha(TWOUSER_RESULTS)
        print("calibrated alt", json.dumps(reg["predicted_alt"]))
    if TAG in SUPERSEDES:
        sup = dict(SUPERSEDES[TAG]); sup["sha256"] = sha(os.path.join(ROOT, sup["file"]))
        reg["supersedes"] = sup
        reg["campaign"] += " (TWOUSER%d%s: the tool now sets ignore_eos)" % (GEN, TAG)
    for arm, v in pred.items():
        print("%-8s %s" % (arm, json.dumps(v)))
    if dry:
        if os.path.exists(OUT):                                    # a dry run against an existing registration: does it reproduce?
            same = json.load(open(OUT))["predicted"] == json.loads(json.dumps(pred))
            print("dry run: %s exists; these predictions %s it" % (os.path.relpath(OUT, ROOT), "REPRODUCE" if same else "DIFFER FROM"))
            return
        print("dry run: nothing written (would have written %s)" % os.path.relpath(OUT, ROOT)); return
    reg["registered"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    with open(OUT, "x") as f:
        json.dump(reg, f, indent=1)
    os.chmod(OUT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print("registered %s at %s, sha256 %s" % (os.path.relpath(OUT, ROOT), reg["registered"], sha(OUT)))


if __name__ == "__main__":
    main(sys.argv[1:])
