#!/usr/bin/env python3
"""The reproducibility check and package (topic-form scope 8, 2026-10-08). It reads the record and never writes into it:
every scorer is re-run on scratch copies, so no published file and no run directory is touched.

    python3 testbed/scripts/reproduce.py verify [--only FAMILY,...] [--keep DIR]
    python3 testbed/scripts/reproduce.py export [--out testbed/dataset/export]
    python3 testbed/scripts/reproduce.py manifest [--out testbed/dataset/MANIFEST.json]

verify re-derives every scored result from its raw data with the scripts on disk, and compares it with the published file:
  dataset   build_dataset.py --out <tmp> on the run dirs of testbed/dataset/README.md's "Rebuild:" line; characterisation.csv
            and results.csv compared byte for byte
  pool      compare_pool.py --pred <its registration> --out <tmp> <the cells' own logs>, for every pool_results_*.json
  energy    score_energy_mapcheck_idle.py / score_energy_phase_run.py on a COPY of each energy_results_*.json's run dir, and
            score_energy_pool_run.py on a copy of every testbed/results/energy_pool_*/ that has a score.json
  twouser   .handoff/assets/pd-2026-09-29/score_twouser.py <log> --reg <registration> --out <tmp>
  pd4       pd4_cross_20261024.py score <log> --reg <registration> --out <tmp>, for every pd_pd4cross_*_score.json (scope 4,
            2026-10-24)
  0930      .handoff/assets/{placement,prima,qr3}-2026-09-30/score_*.py, imported with OUT pointed at <tmp>; qr3_tpot.summary
            is handed a copy of each log, so its <log>_summary.json lands in <tmp>
  rundirs   the scorers that write into their run directory, each re-run on a COPY with the files it writes removed first
            (scored_run_dirs): score_ring_laps_20261009.py, or score_visit_20261009.py where the score has an "adoption";
            score_tcpdiag_20261009.py; score_netshape_20261008.py; full_eval.py summary, then score_full_eval.py where a
            score.json was published; score_energy_h2h_20261008.py. "run" and "code_changed_since_registration" are not
            compared: the first is the copy's path, the second depends on when it is scored. The energy head-to-head
            prices its model with today's link.json, re-measured after it was scored (2026-10-08 18:05), so its
            model_gross, model_err_pct and worst_pct are not compared either; its measured figures and verdicts are
  sessions  the four-node sessions' scorers (2026-10-24; they take the session and its registration, which each published
            score names), each re-run on a COPY the same way (scored_sessions): the replacement Pi 5's characterisation
            (score_pi5b_20261019.py, then score_pi5b_fw_20261020.py for the bootloader session), prima.cpp's energy on
            the replacement board, prima.cpp + Halda on four nodes, scheduler v2 (with its quality log), speculative
            decoding and the draft-length studies, the order check, the network, sampling, RPC-byte and switch-off studies
            (rpc4char/plan, rpc5char, rpc6char, so4; 24 Oct evening), tonight's rpc7char, rpc8char, lc4j, lc4k and c30, so5
            (25 Oct), rpc9b and the planner defaults' def1 and def2 (28 Oct), lc4n (28 Oct: run by lc4, scored by
            score_lc4n_20261028.py), ph1 (28 Oct, the phase scheduler: phase_study_20261028.py score), dl1 (3 Nov: run
            and scored by lc4), jmoe (3 Nov: run by lc4, its prefill scored by score_jmoe_20261103.py), dl2 (4 Nov, lc4),
            hm1 and sp11 (4 Nov, their own scorers), and
            the correction layer's re-test (imported, with
            OUTD pointed at the copy, as it has no --out). The same keys as rundirs are not compared
  regs      every registration (*predictions*.json, full_eval_*.json, rule_*.json) is read-only, and every results file's
            quoted sha256 is its registration's
  costs     the planner's measured attention and laptop-worker costs (2026-10-11): attention_cost_20261011{,b}.py's build(),
            imported and run in memory on the raw logs, against attention_cost_20261011{,b}.json and
            laptop_worker_20261011{,b}.json; and (28 Oct) the Jetson's attention table (attention_cost_20261028.py), the
            phase scheduler's calibration (phase_calib_20261028.py fit) and its power rows (phase_power_20261028.py fit),
            each run with its output pointed at the scratch copy and its content fields compared; and (4 Nov) the
            laptop's attention op per shape (dl2_fa_cpu_20261104.py table, from fa_bench_cpu's raw log) and the planner's
            V6 table built on it (dl2_fa_cpu_20261104.py v6)
Compared leaf by leaf, as JSON values (numbers equal within 1e-9): DIFF is a value that changed; PUBLISHED-ONLY is a key the
re-derivation does not write (a wrapper scorer added it, or the scorer has since changed), listed but not failed; keys only the
re-derivation writes (fields added since) are counted. Timestamps of scoring and scratch paths are ignored. Exit 0 only when no
family has a DIFF or an error.
export writes, from the published files only:
  pool_cells.csv   one row per scored pool cell: campaign, registration and its sha256, model, quant, pipeline, head, host,
                   llama.cpp commit, repetitions, protocol, measured and predicted decode and prefill, errors, exit, thermal
                   verdict and peak Pi temperature (the 1 Hz poll, since 2026-10-07), every board's temperature before and
                   after the cell and the Pis' throttled flags after it (the log's STATE lines), page-in flag, memory drops,
                   the start time and the raw log
  all_figures.csv  every numeric or string leaf of every results file in testbed/dataset, as (file, json path, value)
manifest records the sha256 and size of every file in testbed/dataset, of every raw input a results file names, of every
file in the run directories rundirs re-scores, and (since 2026-10-11) of every testbed/results file that a dataset file's
inputs_sha256 quotes (the raw logs behind the cost files, for one).
"""
import csv, glob, hashlib, importlib.util, json, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(ROOT, "testbed", "dataset")
RESD = os.path.join(ROOT, "testbed", "results")
ASSETS = os.path.join(ROOT, ".handoff", "assets")
IGNORE_KEYS = {"scored", "scored_at", "written", "generated", "out"}
RUNDIR_IGNORE = ("run", "code_changed_since_registration")
REG_GLOBS = ("*predictions*.json", "full_eval_*.json", "rule_*.json")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, ROOT)


def leaves(x, path=""):
    """{json path: leaf} of a JSON value; lists indexed."""
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            if k not in IGNORE_KEYS:
                out.update(leaves(v, "%s/%s" % (path, k)))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out.update(leaves(v, "%s[%d]" % (path, i)))
    else:
        out[path] = x
    return out


HOME_PATH = re.compile(r"^/home/[^/]+/")


def home_relative(p):
    """A path under a home folder as "~/...": the published scores name the laptop's /home/<user>/logs/..., a re-derivation
    the same file under whatever HOME it runs with (2026-10-24: the release, verified from another HOME). Else None."""
    if not isinstance(p, str):
        return None
    home = os.path.expanduser("~").rstrip("/") + "/"
    if p.startswith(home):
        return "~/" + p[len(home):]
    m = HOME_PATH.match(p)
    return "~/" + p[m.end():] if m else None


def compare(published, derived, tmp):
    """(diffs, published_only, derived_only) between two JSON values; scratch paths are not compared, and a path under a
    home folder is compared relative to it (home_relative), so the same file under another HOME is not a difference."""
    a, b = leaves(published), leaves(derived)
    diffs, only_pub = [], []
    for k, v in a.items():
        if k not in b:
            only_pub.append(k)
            continue
        w = b[k]
        if isinstance(w, str) and tmp and (tmp in w or tmp in os.path.normpath(os.path.join(ROOT, w))):
            continue   # a scratch path, also when written relative to a ROOT under the scratch's own parent (a clone in /tmp)
        if home_relative(v) is not None and home_relative(v) == home_relative(w):
            continue
        if isinstance(v, (int, float)) and isinstance(w, (int, float)) and not isinstance(v, bool):
            if abs(v - w) > 1e-9 * max(1.0, abs(v)):
                diffs.append((k, v, w))
        elif v != w:
            diffs.append((k, v, w))
    return diffs, only_pub, sum(1 for k in b if k not in a)


def run(cmd, **kw):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)
    return p.returncode, (p.stdout + p.stderr)[-1500:]


def registrations():
    return {sha(p): p for g in REG_GLOBS for p in glob.glob(os.path.join(DS, g))}


def scored_run_dirs():
    """[(run dir, [argv, ...], [file, ...], (key, ...))] for every run directory holding a score its scorer wrote there:
    the commands to run in order, "{d}" standing for the copy; the files they write, each compared with the published one;
    and the keys, at any depth, that are not compared (see the module docstring)."""
    S = lambda f: os.path.join(HERE, f)
    jobs = []
    for rd in sorted(glob.glob(os.path.join(RESD, "ring_laps_2*"))):
        sp = os.path.join(rd, "score.json")
        if os.path.exists(sp):
            scorer = "score_visit_20261009.py" if "adoption" in json.load(open(sp)) else "score_ring_laps_20261009.py"
            jobs.append((rd, [[S(scorer), "{d}"]], ["score.json"], ()))
    for pat, scorer in (("tcpdiag_*", "score_tcpdiag_20261009.py"), ("netshape_*", "score_netshape_20261008.py")):
        for rd in sorted(glob.glob(os.path.join(RESD, pat))):
            if os.path.exists(os.path.join(rd, "score.json")):
                jobs.append((rd, [[S(scorer), "{d}"]], ["score.json"], ()))
    for rd in sorted(glob.glob(os.path.join(RESD, "full_eval_*"))):
        if os.path.isdir(rd) and os.path.exists(os.path.join(rd, "summary.json")):
            scored = os.path.exists(os.path.join(rd, "score.json"))
            jobs.append((rd, [[S("full_eval.py"), "summary", "{d}"]] + ([[S("score_full_eval.py"), "{d}"]] if scored else []),
                         ["summary.json"] + (["score.json"] if scored else []), ()))
    for rd in sorted(glob.glob(os.path.join(RESD, "energy_pool_*"))):
        if os.path.exists(os.path.join(rd, "h2h_score.json")):
            jobs.append((rd, [[S("score_energy_h2h_20261008.py"), "{d}", "--out", os.path.join("{d}", "h2h_score.json")]],
                         ["h2h_score.json"], ("model_gross", "model_err_pct", "worst_pct")))
    cd = os.path.join(RESD, "correction_20261010")         # upgrade 2 (2026-10-10): it fits the rows its registration embeds
    if os.path.exists(os.path.join(cd, "score.json")):
        jobs.append((cd, [[S("score_correction_20261010.py"), "--out", "{d}"]], ["score.json"], ()))
    for rd in sorted(glob.glob(os.path.join(RESD, "spec_shape_2*"))):   # the draft under shaping (2026-10-10); not the smoke
        if os.path.exists(os.path.join(rd, "score.json")):
            jobs.append((rd, [[S("spec_shape_20261010.py"), "score", "{d}"]], ["score.json"], ()))
    # upgrade 5 (2026-10-10): the Jetson's power modes, characterised (laptop host) and served; not the smokes
    for rd in sorted(glob.glob(os.path.join(RESD, "pm_char_2*"))):
        if os.path.exists(os.path.join(rd, "analysis.json")):
            jobs.append((rd, [[S("power_modes_char_20261010.py"), "analyse", "{d}"]], ["analysis.json"], ()))
    for rd in sorted(glob.glob(os.path.join(RESD, "power_modes_2*"))):
        if os.path.exists(os.path.join(rd, "score.json")):
            jobs.append((rd, [[S("power_modes_20261010.py"), "score", "{d}"]], ["score.json"], ()))
    for rd in sorted(glob.glob(os.path.join(RESD, "kv_depth_2*"))):   # upgrade 4's KV term (2026-10-10)
        if os.path.exists(os.path.join(rd, "score.json")):
            jobs.append((rd, [[S("kv_depth_20261010.py"), "score", "{d}"]], ["score.json"], ()))
    for rd in sorted(glob.glob(os.path.join(RESD, "kv14_2*"))):       # the KV term on the served 14B (2026-10-10): kv14 and kv14b
        if os.path.exists(os.path.join(rd, "score.json")):
            reg = json.load(open(os.path.join(rd, "score.json")))["registration"]
            jobs.append((rd, [[S("kv14_20261010.py"), "score", "{d}", "--reg", os.path.join(ROOT, reg)]], ["score.json"], ()))
    for rd in sorted(glob.glob(os.path.join(RESD, "longctx_2*"))):    # the long-context plan with the laptop (2026-10-11)
        if os.path.exists(os.path.join(rd, "score.json")):
            reg = json.load(open(os.path.join(rd, "score.json")))["registration"]   # b supersedes a: score what the run named
            jobs.append((rd, [[S("longctx_20261011.py"), "score", "{d}", "--reg", os.path.join(ROOT, reg)]], ["score.json"], ()))
    for rd in sorted(glob.glob(os.path.join(RESD, "head_bs_2*"))):    # upgrade 6 (2026-10-10); not the smoke
        if os.path.exists(os.path.join(rd, "score.json")):
            reg = json.load(open(os.path.join(rd, "score.json")))["registration"]   # b supersedes a: score what the run named
            jobs.append((rd, [[S("head_bs_20261010.py"), "score", "{d}", "--reg", os.path.join(ROOT, reg)]], ["score.json"], ()))
    # upgrade 1 (2026-10-10): the pass time (step 1), the draft's acceptance (step 2) and the served test (step 3)
    for rd in sorted(glob.glob(os.path.join(RESD, "spec_pass_2*"))):
        logs = sorted(glob.glob(os.path.join(rd, "pool_sp.*_2*.log")))
        if os.path.exists(os.path.join(rd, "score.json")) and len(logs) == 1:
            jobs.append((rd, [[S("score_spec_pass_20261010.py"), os.path.join("{d}", os.path.basename(logs[0])),
                               "--out", os.path.join("{d}", "score.json")]], ["score.json"], ()))
    for pat, script in (("spec_accept_2*", "spec_accept_20261010.py"), ("spec_serve_2*", "spec_serve_20261010.py")):
        for rd in sorted(glob.glob(os.path.join(RESD, pat))):
            if os.path.exists(os.path.join(rd, "score.json")):
                jobs.append((rd, [[S(script), "score", "{d}"]], ["score.json"], ()))
    return jobs


def scored_sessions():
    """The four-node sessions (2026-10-19 on), in scored_run_dirs()'s shape. Each scorer takes the session folder and the
    registration that its published score names; sched2 also takes its quality log, prima's energy its continuation
    folders. Only folders holding a published score are re-scored (an aborted start has none)."""
    S = lambda f: os.path.join(HERE, f)
    named = lambda rd, f="score.json": json.load(open(os.path.join(rd, f)))
    jobs = []
    for pat, scorer in (("pi5b_session_2*", "score_pi5b_20261019.py"), ("prima4_2*", "score_prima4_20261021.py"),
                        ("spec4_2*", "score_spec4_20261023.py"), ("spec4b_2*", "score_spec4b_20261023.py"),
                        ("spec4c_2*", "score_spec4c_20261024.py"), ("spec4d_2*", "score_spec4d_20261024.py"),
                        ("order35_2*", "score_order35_20261023.py"),
                        ("def1_2*", "score_def1_20261028.py"), ("def2_2*", "score_def2_20261028.py"),   # 28 Oct: the defaults
                        ("v4a_2*", "score_v4a_20261101.py"),       # 1 Nov: PLANNER_V4's hybrid memory and the power refit
                        ("e2_2*", "score_e2_20261102.py"),         # 2 Nov: the power refit with the session's own idle
                        ("q9_2*", "score_q9_20261102.py"),         # 2 Nov: the request priced as prefill and decode
                        ("q9b_2*", "score_q9_20261102.py"),        # 3 Nov: q9 again, complete
                        ("sp10_2*", "score_sp10_20261103.py"),     # 3 Nov: the planner's checking cost (item 10)
                        ("sp10c_2*", "score_sp10_20261103.py"),    # 3 Nov: the same, the MTP block off the Pi 5
                        ("sp10d_2*", "score_sp10_20261103.py"),    # 3 Nov: per-node batch curves on the 9B
                        ("sp10e_2*", "score_sp10_20261103.py"),    # 3 Nov: sp10d again, the Pi 5's TSO off
                        ("hm1_2*", "score_hm1_20261104.py"),       # 4 Nov: the hand-off's memory check
                        ("sp11_2*", "score_sp11_20261104.py"),     # 4 Nov: MTP 1 across 12 prompts, the texts behind E
                        ("sp12_2*", "score_sp11_20261104.py"),     # 4 Nov: the same on the 35B
                        # the 24 October chain and its follow-ons: the A/B sessions (each registration names its
                        # scorer_extra, which ab_score imports) and the ring laps
                        ("c35_2*", "ab_score_20261024.py"), ("pm4_2*", "ab_score_20261024.py"),
                        ("ns4_2*", "ab_score_20261024.py"), ("ns4b_2*", "ab_score_20261024.py"),
                        ("ns4c_2*", "ab_score_20261024.py"), ("bs4_2*", "ab_score_20261024.py"),
                        ("rpc4plan_2*", "ab_score_20261024.py"), ("so4_2*", "ab_score_20261024.py"),
                        ("c30_2*", "ab_score_20261024.py"),        # 24 Oct night: the 35B's multipliers on the 30B
                        ("ring4_2*", "score_ring4_20261024.py")):
        for rd in sorted(glob.glob(os.path.join(RESD, pat))):
            if os.path.exists(os.path.join(rd, "score.json")):
                reg = os.path.join(ROOT, named(rd)["registration"])
                jobs.append((rd, [[S(scorer), "{d}", "--registration", reg]], ["score.json"], ()))
    for rd in sorted(glob.glob(os.path.join(RESD, "pi5b_fw_session_2*"))):   # the as-found scorer first: fw reads its score
        if os.path.exists(os.path.join(rd, "fw_score.json")):
            reg = os.path.join(ROOT, named(rd, "fw_score.json")["registration"])
            jobs.append((rd, [[S("score_pi5b_20261019.py"), "{d}", "--registration", reg],
                              [S("score_pi5b_fw_20261020.py"), "{d}", "--registration", reg]],
                         ["score.json", "fw_score.json"], ()))
    for rd in sorted(glob.glob(os.path.join(RESD, "prima_energy_2*"))):
        if os.path.exists(os.path.join(rd, "score.json")):
            s = named(rd)
            jobs.append((rd, [[S("score_prima_energy_20261020.py"), "{d}", "--registration", os.path.join(ROOT, s["registration"])]
                              + (["--also"] + [os.path.join(ROOT, x) for x in s["also"]] if s.get("also") else [])],
                         ["score.json"], ()))
    for rd in sorted(glob.glob(os.path.join(RESD, "sched2_2*"))):
        if os.path.isdir(rd) and os.path.exists(os.path.join(rd, "score.json")):
            s = named(rd)
            jobs.append((rd, [[S("score_sched2_20261022.py"), "{d}", "--registration", os.path.join(ROOT, s["registration"]),
                               "--quality", os.path.join(ROOT, s["quality_log"])]], ["score.json"], ()))
    for pat in ("lc4charb_2*", "lc4planb_2*", "lc4j_2*", "lc4k_2*", "lc4m_2*", "v4d_2*",   # the context-depth study (and the
                "dl1_2*", "dl2_2*", "dl3_2*"):                # Jetson's law, 24-25 Oct; v4d, depth priced by default, 1 Nov;
                                                              # dl1, the bent depth law, 3 Nov; dl2, four depth laws, 4 Nov;
                                                              # dl3, the Pis' per-shape tables, 4 Nov evening):
                                                              # lc4 reads its registration from run.log
        for rd in sorted(glob.glob(os.path.join(RESD, pat))):
            if os.path.exists(os.path.join(rd, "score.json")):
                jobs.append((rd, [[S("lc4_20261024.py"), "score", "{d}"]], ["score.json"], ()))
    for rd in sorted(glob.glob(os.path.join(RESD, "lc4n_2*"))):   # 28 Oct, the Jetson's depth cost against fa_bench: run by lc4
        if os.path.exists(os.path.join(rd, "score_lc4n.json")):     # (its own scorer gets no hypotheses), scored by lc4n's
            jobs.append((rd, [[S("lc4_20261024.py"), "score", "{d}"], [S("score_lc4n_20261028.py"), "{d}"]],
                         ["score.json", "score_lc4n.json"], ()))
    for rd in sorted(glob.glob(os.path.join(RESD, "jmoe_2*"))):   # 3 Nov, the Jetson prefilling MoE blocks: run by lc4 (its
        if os.path.exists(os.path.join(rd, "score_prefill.json")):  # score holds decode), prefill scored by jmoe's own scorer
            jobs.append((rd, [[S("lc4_20261024.py"), "score", "{d}"], [S("score_jmoe_20261103.py"), "{d}"]],
                         ["score.json", "score_prefill.json"], ()))
    for rd in sorted(glob.glob(os.path.join(RESD, "ph[0-9]_2*"))):   # 28 Oct, the phase scheduler (ph1, ph2: L, J or the
        if os.path.exists(os.path.join(rd, "score.json")):          # hand-off), energy from the run's own strip.csv
            jobs.append((rd, [[S("phase_study_20261028.py"), "score", "{d}"]], ["score.json"], ()))
    for pat, script in (("rpc4char_2*", "rpc4_char_20261024.py"), ("rpc5char_2*", "rpc5_char_20261024.py"),
                        ("rpc6char_2*", "rpc6_char_20261024.py"), ("rpc7char_2*", "rpc7_char_20261024.py"),
                        ("rpc8char_2*", "rpc8_char_20261024.py"),    # the RPC byte counts: each reads its registration
                        ("so5_2*", "so5_trial_20261025.py"),         # (and so5, the switch-off automation's live trial)
                        ("so6_2*", "so6_trial_20261101.py"),         # (so6, 1 Nov: every idle board switched off, live)
                        ("rpc9char_2*", "rpc9_char_20261025.py"), ("rpc9bchar_2*", "rpc9b_char_20261028.py")):
        for rd in sorted(glob.glob(os.path.join(RESD, pat))):          # from measurements.json
            if os.path.exists(os.path.join(rd, "score.json")):
                jobs.append((rd, [[S(script), "score", "{d}"]], ["score.json"], ()))
    cd = os.path.join(RESD, "correction4_20261024")
    if os.path.exists(os.path.join(cd, "score.json")):
        code = ("import sys; sys.path.insert(0, %r); import correction4_20261024 as m; m.OUTD = '{d}'; m.score()" % HERE)
        jobs.append((cd, [["-c", code]], ["score.json"], ()))
    return jobs


def drop(x, names):
    """x without the dict keys in names, at any depth."""
    if isinstance(x, dict):
        return {k: drop(v, names) for k, v in x.items() if k not in names}
    if isinstance(x, list):
        return [drop(v, names) for v in x]
    return x


# ------------------------------------------------------------------ verify families
def v_dataset(tmp):
    readme = open(os.path.join(DS, "README.md")).read()
    m = re.search(r"^Rebuild: `python3 testbed/scripts/build_dataset\.py ([^`]+)`", readme, re.M)
    if not m:
        return [("dataset", "ERROR", "no Rebuild: line in testbed/dataset/README.md")]
    out = os.path.join(tmp, "dataset"); os.makedirs(out)
    rc, txt = run([sys.executable, os.path.join(HERE, "build_dataset.py"), "--out", out] + m.group(1).split())
    if rc:
        return [("dataset", "ERROR", "build_dataset.py exit %d: %s" % (rc, txt[-300:]))]
    res = []
    for f in ("characterisation.csv", "results.csv"):
        same = os.path.exists(os.path.join(out, f)) and sha(os.path.join(out, f)) == sha(os.path.join(DS, f))
        res.append(("dataset/" + f, "MATCH" if same else "DIFF", "byte for byte" if same else
                    "rebuilt %s differs from the published file" % f))
    return res


def v_pool(tmp):
    regs, res = registrations(), []
    for p in sorted(glob.glob(os.path.join(DS, "pool_results*.json"))):
        pub = json.load(open(p))
        reg = regs.get(pub.get("predictions_sha256")) or (os.path.join(ROOT, pub["registration"]) if "registration" in pub else None)
        logs = [os.path.join(ROOT, c["log"]) for c in pub.get("cells", {}).values() if isinstance(c, dict) and c.get("log")]
        if not reg or not logs:
            res.append((rel(p), "SKIP", "no registration or no logs named")); continue
        out = os.path.join(tmp, "pool_" + os.path.basename(p))
        rc, txt = run([sys.executable, os.path.join(HERE, "compare_pool.py"), "--pred", reg, "--out", out] + logs)
        if rc or not os.path.exists(out):
            res.append((rel(p), "ERROR", "compare_pool.py exit %d: %s" % (rc, txt[-300:]))); continue
        d, po, do = compare({"cells": pub["cells"]}, {"cells": json.load(open(out))["cells"]}, tmp)
        res.append((rel(p), "DIFF" if d else "MATCH", summary_note(d, po, do)))
    return res


def summary_note(d, po, do):
    s = "%d differing, %d published-only, %d new fields" % (len(d), len(po), do)
    if d:
        s += "; e.g. " + "; ".join("%s: %r -> %r" % x for x in d[:4])
    if po:
        s += "; published-only e.g. " + ", ".join(po[:4])
    return s


def v_energy(tmp):
    res = []
    for p in sorted(glob.glob(os.path.join(DS, "energy_results_*.json"))):
        pub = json.load(open(p))
        rd, reg = os.path.join(ROOT, pub["run"]), os.path.join(ROOT, pub["registration"])
        if not (os.path.isdir(rd) and os.path.exists(reg)):
            res.append((rel(p), "SKIP", "run dir or registration missing")); continue
        cp = os.path.join(tmp, "energy", os.path.basename(rd)); shutil.copytree(rd, cp)
        out = os.path.join(tmp, "energy_" + os.path.basename(p))
        scorer = "score_energy_mapcheck_idle.py" if os.path.basename(p) == "energy_results_20261005.json" else "score_energy_phase_run.py"
        rc, txt = run([sys.executable, os.path.join(HERE, scorer), cp, "--registration", reg, "--out", out])
        if rc or not os.path.exists(out):
            res.append((rel(p), "ERROR", "%s exit %d: %s" % (scorer, rc, txt[-300:]))); continue
        pub2, der = dict(pub), json.load(open(out))
        for k in ("run",):                    # the copy's path, by construction
            pub2.pop(k, None); der.pop(k, None)
        d, po, do = compare(pub2, der, tmp)
        res.append((rel(p), "DIFF" if d else "MATCH", summary_note(d, po, do)))
    for rd in sorted(glob.glob(os.path.join(RESD, "energy_pool_*"))):
        sp = os.path.join(rd, "score.json")
        if not os.path.exists(sp):
            continue
        cp = os.path.join(tmp, "energy_pool", os.path.basename(rd)); shutil.copytree(rd, cp)
        out = os.path.join(tmp, "energy_pool_%s.json" % os.path.basename(rd))
        rc, txt = run([sys.executable, os.path.join(HERE, "score_energy_pool_run.py"), cp, "--out", out])
        if rc or not os.path.exists(out):
            res.append((rel(sp), "ERROR", "score_energy_pool_run.py exit %d: %s" % (rc, txt[-300:]))); continue
        pub2, der = json.load(open(sp)), json.load(open(out))
        for k in ("run",):
            pub2.pop(k, None); der.pop(k, None)
        d, po, do = compare(pub2, der, tmp)
        res.append((rel(sp), "DIFF" if d else "MATCH", summary_note(d, po, do)))
    return res


def v_twouser(tmp):
    res, regs = [], registrations()
    scorer = os.path.join(ASSETS, "pd-2026-09-29", "score_twouser.py")
    for p in sorted(glob.glob(os.path.join(DS, "twouser_results_*.json"))):
        pub = json.load(open(p))
        reg = regs.get(pub.get("registration_sha256"))
        if not reg:
            res.append((rel(p), "SKIP", "registration not found by sha256")); continue
        out = os.path.join(tmp, "twouser_" + os.path.basename(p))
        rc, txt = run([sys.executable, scorer, os.path.join(ROOT, pub["log"]), "--reg", reg, "--out", out])
        if rc or not os.path.exists(out):
            res.append((rel(p), "ERROR", "score_twouser.py exit %d: %s" % (rc, txt[-300:]))); continue
        d, po, do = compare(pub, json.load(open(out)), tmp)
        res.append((rel(p), "DIFF" if d else "MATCH", summary_note(d, po, do)))
    return res


def v_pd4(tmp):
    """Scope 4 on today's boards (2026-10-24, pd4base / pd4cross): pd4_cross_20261024.py score on each run's own log, against
    the registration its published *_score.json names, the score written into tmp."""
    res = []
    for p in sorted(glob.glob(os.path.join(RESD, "pd_pd4cross_*_score.json"))):
        pub = json.load(open(p))
        out = os.path.join(tmp, "pd4_" + os.path.basename(p))
        rc, txt = run([sys.executable, os.path.join(HERE, "pd4_cross_20261024.py"), "score", os.path.join(ROOT, pub["log"]),
                       "--reg", os.path.join(ROOT, pub["registration"]), "--out", out])
        if rc or not os.path.exists(out):
            res.append((rel(p), "ERROR", "pd4_cross_20261024.py score exit %d: %s" % (rc, txt[-300:]))); continue
        d, po, do = compare(pub, json.load(open(out)), tmp)
        res.append((rel(p), "DIFF" if d else "MATCH", summary_note(d, po, do)))
    return res


def v_0930(tmp):
    res = []
    sys.path.insert(0, HERE)
    import qr3_tpot as Q                                  # noqa: E402
    orig = Q.summary

    def summary_on_copy(path):                            # its <log>_summary.json must land in tmp, not testbed/results
        cp = os.path.join(tmp, "qr3_logs", os.path.basename(path))
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        shutil.copy2(path, cp)
        return orig(cp)
    for sub, name, pubf in (("placement-2026-09-30", "score_default.py", "pool_results_20260930default.json"),
                            ("prima-2026-09-30", "score_prima.py", "prima_results_20260930.json"),
                            ("qr3-2026-09-30", "score_qr3.py", "qr3_results_20260930.json")):
        path = os.path.join(ASSETS, sub, name)
        out = os.path.join(tmp, "0930_" + pubf)
        try:
            spec = importlib.util.spec_from_file_location("repro_" + name[:-3], path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.OUT = out
            if hasattr(mod, "Q"):
                mod.Q.summary = summary_on_copy
            mod.main()
        except SystemExit as e:
            if e.code not in (0, None):
                res.append((rel(os.path.join(DS, pubf)), "ERROR", "%s exited %s" % (name, e.code))); continue
        except Exception as e:                            # reported, not raised: one family's error must not hide the rest
            res.append((rel(os.path.join(DS, pubf)), "ERROR", "%s: %s: %s" % (name, type(e).__name__, e))); continue
        finally:
            Q.summary = orig
        if not os.path.exists(out):
            res.append((rel(os.path.join(DS, pubf)), "ERROR", "%s wrote nothing" % name)); continue
        d, po, do = compare(json.load(open(os.path.join(DS, pubf))), json.load(open(out)), tmp)
        res.append((rel(os.path.join(DS, pubf)), "DIFF" if d else "MATCH", summary_note(d, po, do)))
    return res


def v_rundirs(tmp):
    return rescore(tmp, scored_run_dirs(), "rundirs")


def v_sessions(tmp):
    return rescore(tmp, scored_sessions(), "sessions")


def rescore(tmp, jobs, sub):
    res = []
    for rd, steps, outs, skip in jobs:
        cp = os.path.join(tmp, sub, os.path.basename(rd)); shutil.copytree(rd, cp)
        for f in outs:                        # a scorer that writes nothing must not leave the published file to compare
            os.remove(os.path.join(cp, f))
        err = None
        for st in steps:
            rc, txt = run([sys.executable] + [a.replace("{d}", cp) for a in st])
            if rc:
                err = "%s exit %d: %s" % (os.path.basename(st[0]), rc, txt[-300:])
                break
        for f in outs:
            item = rel(os.path.join(rd, f))
            if err or not os.path.exists(os.path.join(cp, f)):
                res.append((item, "ERROR", err or "%s was not written" % f)); continue
            pub, der = json.load(open(os.path.join(rd, f))), json.load(open(os.path.join(cp, f)))
            for k in RUNDIR_IGNORE:
                pub.pop(k, None); der.pop(k, None)
            d, po, do = compare(drop(pub, skip), drop(der, skip), tmp)
            res.append((item, "DIFF" if d else "MATCH",
                        summary_note(d, po, do) + ("; not compared: %s" % ", ".join(skip) if skip else "")))
    return res


def v_regs(tmp):
    res, regs = [], registrations()
    for s, p in sorted(regs.items(), key=lambda kv: kv[1]):
        writable = bool(os.stat(p).st_mode & 0o222)
        res.append((rel(p), "WRITABLE" if writable else "MATCH", "sha256 %s, %s" % (s[:16], "writable!" if writable else "read-only")))
    for p in sorted(glob.glob(os.path.join(DS, "*results*.json"))):
        pub = json.load(open(p))
        q = pub.get("predictions_sha256") or pub.get("registration_sha256")
        if q and q not in regs:
            res.append((rel(p), "DIFF", "quotes registration sha256 %s, which no registration file has" % q[:16]))
    return res


def v_costs(tmp):
    # The long-context planner's inputs (2026-10-11): each build() reads only raw logs and published scores, and returns
    # what its script wrote; nothing is written here.
    sys.path.insert(0, HERE)
    import attention_cost_20261011 as ca, attention_cost_20261011b as cb
    remote = sorted(glob.glob(os.path.join(RESD, "laptop_worker_remote_2*")))[-1]
    res = []
    for built, paths in ((ca.build(), (ca.OUT_ATTN, ca.OUT_LAP)), (cb.build(remote), (cb.OUT_ATTN, cb.OUT_LAP))):
        for der, p in zip(built, paths):
            d, po, do = compare(json.load(open(p)), der, tmp)
            res.append((rel(p), "DIFF" if d else "MATCH", summary_note(d, po, do)))
    # 28 Oct: the Jetson's attention table (lc4n), the phase scheduler's calibration and its power rows (ph1, ph2). Each
    # script writes one file from raw data; its output path is pointed at the scratch copy (as correction4's OUTD), its
    # content fields compared (their "made"/"at" stamps are not results).
    import contextlib, io
    import attention_cost_20261028 as ac, phase_calib_20261028 as pc, phase_power_20261028 as pw
    # 1 Nov (the scheduler review's items 2-4): the hybrids' memory per block (from their GGUF headers, checked against the
    # 35B's logged buffers), the attention table with every node's law, and the laptop's power refitted from the four-node
    # evaluation's strip data
    import hybrid_shapes_20261101 as hs, attention_cost_20261101 as a4, energy_refit_20261101 as er
    calib = os.path.join(ROOT, json.load(open(pc.OUT))["from"]) if os.path.exists(pc.OUT) else None   # before OUT moves
    # 4 Nov (item 4): Qwen3-4B's calibration and power rows, each re-derived from its own raw data into a scratch copy
    c4b, w4b = os.path.join(DS, "phase_costs_qwen3_4b_20261103.json"), os.path.join(DS, "phase_power_qwen3_4b_20261103.json")
    calib4b = os.path.join(ROOT, json.load(open(c4b))["from"]) if os.path.exists(c4b) else None
    run4b = os.path.join(ROOT, json.load(open(w4b))["from"]) if os.path.exists(w4b) else None
    # 4 Nov (dl2): the laptop's attention op per shape, from fa_bench_cpu's raw log, and the planner's table built on it (V6,
    # from the published CPU table, which the job before re-derives)
    import dl2_fa_cpu_20261104 as dl
    fa_runs = sorted(glob.glob(os.path.join(RESD, "dl2fa_2*")))
    cpu_pub = dl.CPU_TABLE
    jobs = ((ac, lambda: ac.main(), ("boards",), None),
            (pc, lambda: pc.fit(calib), ("prefill", "decode", "handoff", "state"), None),
            # out_path given: fit's default is bound when the module loads, so it would write the published file (3 Nov)
            (pw, lambda: pw.fit(out_path=pw.OUT), ("rows", "fit_points"), None),
            (pc, lambda: pc.fit(calib4b), ("prefill", "decode", "handoff", "state", "handoff_not_fitted"), c4b),
            (pw, lambda: pw.fit(run=run4b, out_path=pw.OUT, model="Qwen3-4B Q4_K_M"), ("rows", "fit_points"), w4b),
            (hs, lambda: hs.main(), ("rule", "shapes", "check_35b"), None),
            (a4, lambda: a4.main(), ("boards",), None),
            (er, lambda: er.main(), ("laptop", "boards", "fit", "points"), None))
    if fa_runs:
        jobs += ((dl, lambda: dl.table(fa_runs[-1]), ("board", "law", "threads", "shapes"), cpu_pub),
                 (dl, lambda: dl.v6(out=dl.OUT, cpu=cpu_pub), ("boards",), dl.V6))
    # 4 Nov evening (dl3): the Pis' attention op per shape, from each board's fa_bench_cpu raw log
    import dl3_fa_pi_20261104 as dp
    if os.path.exists(dp.OUT):
        pi_runs = {b: os.path.join(ROOT, e["from"]) for b, e in json.load(open(dp.OUT))["boards"].items()}
        jobs += ((dp, lambda: dp.table(pi_runs["pi5"], pi_runs["pi4"]), ("law", "threads", "boards"), None),
                 (dp, lambda: dp.v7(out=dp.OUT, pi=dp.PI_TABLE), ("boards",), dp.V7))   # V7, from the published Pi table
    # the hybrids' GGUF files are named, not shipped (a release): without them that one file is skipped, not failed
    for mod, call, keys, published in jobs:
        path, orig = published or mod.OUT, mod.OUT
        if not os.path.exists(path):
            continue
        if mod is hs:
            shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
            missing = [shapes[k]["path"] for k in json.load(open(path))["shapes"]
                       if not os.path.exists(os.path.expanduser(shapes[k]["path"].split(".gguf")[0] + ".gguf"))]
            if missing:
                res.append((rel(path), "SKIP", "the hybrids' GGUF files are not here (named, not shipped): %s" % missing))
                continue
        mod.OUT, argv = os.path.join(tmp, os.path.basename(path)), sys.argv
        try:
            sys.argv = [mod.__file__]
            with contextlib.redirect_stdout(io.StringIO()):
                call()
            pub, der = json.load(open(path)), json.load(open(mod.OUT))
            d, po, do = compare({k: pub.get(k) for k in keys}, {k: der.get(k) for k in keys}, tmp)
            res.append((rel(path), "DIFF" if d else "MATCH", summary_note(d, po, do)))
        except Exception as e:
            res.append((rel(path), "ERROR", "%s: %s" % (type(e).__name__, e)))
        finally:
            mod.OUT, sys.argv = orig, argv
    return res


FAMILIES = {"dataset": v_dataset, "pool": v_pool, "energy": v_energy, "twouser": v_twouser, "pd4": v_pd4, "0930": v_0930,
            "rundirs": v_rundirs, "sessions": v_sessions, "regs": v_regs, "costs": v_costs}


def verify(argv):
    only = argv[argv.index("--only") + 1].split(",") if "--only" in argv else list(FAMILIES)
    keep = argv[argv.index("--keep") + 1] if "--keep" in argv else None
    tmp = keep or tempfile.mkdtemp(prefix="heteroserve-repro-")
    os.makedirs(tmp, exist_ok=True)
    before = {p: os.stat(p).st_mtime_ns for p in glob.glob(os.path.join(DS, "*")) + glob.glob(os.path.join(RESD, "**"), recursive=True)
              if os.path.isfile(p)}
    rows, bad = [], 0
    for fam in only:
        try:
            r = FAMILIES[fam](tmp)
        except Exception as e:
            r = [(fam, "ERROR", "%s: %s" % (type(e).__name__, e))]
        for item, verdict, note in r:
            rows.append((fam, item, verdict, note))
            bad += verdict in ("DIFF", "ERROR", "WRITABLE")
    touched = [p for p, t in before.items() if not os.path.exists(p) or os.stat(p).st_mtime_ns != t]
    for fam, item, verdict, note in rows:
        print("%-8s %-9s %-62s %s" % (fam, verdict, item, note[:220]))
    print("record untouched: %s" % ("yes" if not touched else "NO: %s" % [rel(p) for p in touched][:10]))
    print("%d checks, %d failing; scratch %s%s" % (len(rows), bad, tmp, "" if keep else " (removed)"))
    if not keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if (bad or touched) else 0


# ------------------------------------------------------------------ export
def state_temps(log):
    """'pi5:34.5>46.1;jetson:28/28>31/31' from a pool log's STATE before/after lines (Pi: temp; Jetson: CPU/GPU, C), and
    the Pis' throttled flags after the cell; ('', '') when the log has none."""
    t, thr = {}, {}
    for ln in open(log, encoding="utf-8", errors="replace"):
        m = re.match(r"^STATE (before|after) (\S+) \S+ (.*)$", ln)
        if not m:
            continue
        when, node, rest = m.groups()
        pi = re.search(r"temp=([\d.]+)'C", rest)
        je = re.search(r"CPU=(\d+)C GPU=(\d+)C", rest)
        v = pi.group(1) if pi else ("%s/%s" % je.groups() if je else None)
        if v:
            t.setdefault(node, {})[when] = v
        th = re.search(r"throttled=(0x[0-9a-fA-F]+)", rest)
        if th and when == "after":
            thr[node] = th.group(1)
    temps = ";".join("%s:%s>%s" % (n, d.get("before", "?"), d.get("after", "?")) for n, d in t.items())
    return temps, ";".join("%s:%s" % kv for kv in thr.items())


def export(argv):
    out = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(DS, "export")
    os.makedirs(out, exist_ok=True)
    regs = registrations()
    cols = ["campaign", "registration", "registration_sha256", "cell", "started", "model", "quant", "pipeline", "head_on", "host",
            "commit", "reps", "protocol", "decode_ms", "decode_tok_s_min", "decode_tok_s_max", "decode_ms_predicted",
            "decode_err_pct", "prefill_ms", "prefill_ms_predicted", "prefill_err_pct", "exit", "finished", "thermal",
            "pi_max_temp_c", "temps_before_after_c", "throttled_after", "page_in_flag", "memory_drop_mib", "log"]
    n = 0
    with open(os.path.join(out, "pool_cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for p in sorted(glob.glob(os.path.join(DS, "pool_results*.json"))):
            pub = json.load(open(p))
            reg = regs.get(pub.get("predictions_sha256")) or (os.path.join(ROOT, pub["registration"]) if "registration" in pub else None)
            R = json.load(open(reg)) if reg and os.path.exists(reg) else {}
            for cid, c in pub.get("cells", {}).items():
                if not isinstance(c, dict):
                    continue
                rc = R.get("configs", {}).get(cid, {})
                dec, pre = c.get("decode") or {}, c.get("prefill") or {}
                dm, pm = dec.get("measured") or {}, pre.get("measured") or {}
                th = c.get("thermal") if isinstance(c.get("thermal"), dict) else {}
                temps = [float(v.get("max_temp_c")) for v in (th.get("per_node") or {}).values()
                         if str(v.get("max_temp_c", "NA")).replace(".", "", 1).isdigit()]
                lp = os.path.join(ROOT, c.get("log", ""))
                snap, thr = state_temps(lp) if c.get("log") and os.path.isfile(lp) else ("", "")
                w.writerow({"campaign": os.path.basename(p), "registration": rel(reg) if reg else "",
                            "registration_sha256": sha(reg)[:16] if reg and os.path.exists(reg) else "", "cell": cid,
                            "started": c.get("started", ""), "model": rc.get("model", ""), "quant": rc.get("quant", ""),
                            "pipeline": "+".join("%s:%s" % (s["node"], s["blocks"]) for s in c.get("pipeline", [])),
                            "head_on": c.get("head_on", ""), "host": (rc.get("bench") or {}).get("host", ""),
                            "commit": dm.get("commit", ""), "reps": dm.get("reps", ""),
                            "protocol": (rc.get("bench") or {}).get("protocol") or R.get("protocol", ""),
                            "decode_ms": dm.get("ms_per_token", ""), "decode_tok_s_min": dm.get("min", ""),
                            "decode_tok_s_max": dm.get("max", ""), "decode_ms_predicted": dec.get("predicted_ms", ""),
                            "decode_err_pct": dec.get("error_pct", ""), "prefill_ms": pm.get("ms_per_token", ""),
                            "prefill_ms_predicted": pre.get("predicted_ms", ""), "prefill_err_pct": pre.get("error_pct", ""),
                            "exit": c.get("exit", ""), "finished": c.get("finished", ""), "thermal": th.get("verdict", "not polled"),
                            "pi_max_temp_c": max(temps) if temps else "",
                            "temps_before_after_c": snap, "throttled_after": thr, "page_in_flag": (c.get("page_in") or {}).get("flag", ""),
                            "memory_drop_mib": ";".join("%s:%s" % (k, v.get("drop_mib")) for k, v in (c.get("memory") or {}).items()),
                            "log": c.get("log", "")})
                n += 1
    m = 0
    with open(os.path.join(out, "all_figures.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "json_path", "value"])
        for p in sorted(glob.glob(os.path.join(DS, "*results*.json"))):
            for k, v in leaves(json.load(open(p))).items():
                w.writerow([os.path.basename(p), k, v]); m += 1
    print("wrote %s (%d cells) and %s (%d figures)" % (rel(os.path.join(out, "pool_cells.csv")), n,
                                                       rel(os.path.join(out, "all_figures.csv")), m))
    return 0


# ------------------------------------------------------------------ manifest
def manifest(argv):
    out = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(DS, "MANIFEST.json")
    files = set(p for p in glob.glob(os.path.join(DS, "**"), recursive=True) if os.path.isfile(p) and os.path.abspath(p) != os.path.abspath(out))
    for p in glob.glob(os.path.join(DS, "*results*.json")):
        pub = json.load(open(p))
        for c in (pub.get("cells") or {}).values():
            if isinstance(c, dict) and c.get("log"):
                files.add(os.path.join(ROOT, c["log"]))
        for k in ("log",):
            if isinstance(pub.get(k), str):
                files.add(os.path.join(ROOT, pub[k]))
        if isinstance(pub.get("run"), str) and os.path.isdir(os.path.join(ROOT, pub["run"])):
            files.update(q for q in glob.glob(os.path.join(ROOT, pub["run"], "**"), recursive=True) if os.path.isfile(q))
    for rd, *_ in scored_run_dirs() + scored_sessions():
        files.update(q for q in glob.glob(os.path.join(rd, "**"), recursive=True) if os.path.isfile(q))
    for p in glob.glob(os.path.join(DS, "*.json")):     # raw inputs a dataset file quotes by sha256 (2026-10-11)
        try:
            quoted = json.load(open(p)).get("inputs_sha256") or {}
        except (ValueError, AttributeError):
            continue
        files.update(os.path.join(ROOT, k) for k in quoted if k.startswith("testbed/results/"))
    files = sorted(f for f in files if os.path.exists(f))
    man = {"made": "testbed/scripts/reproduce.py manifest", "files": {rel(f): {"sha256": sha(f), "bytes": os.path.getsize(f)} for f in files}}
    json.dump(man, open(out, "w"), indent=1)
    print("wrote %s: %d files" % (rel(out), len(files)))
    return 0


def main(argv):
    if not argv or argv[0] not in ("verify", "export", "manifest"):
        sys.exit(__doc__)
    sys.exit({"verify": verify, "export": export, "manifest": manifest}[argv[0]](argv[1:]))


if __name__ == "__main__":
    main(sys.argv[1:])
