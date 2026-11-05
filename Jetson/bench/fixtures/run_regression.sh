#!/usr/bin/env bash
# Off-hardware regression for the bench driver and the energy analysis.
#
# Proves audit finding #3 is fixed and stays fixed: no power window may carry
# more than one phase, and no row may pair a throughput measured on one phase
# with energy measured over a different scope.
#
#   ./bench/fixtures/run_regression.sh
#
# Needs no Jetson. Budget ~3 min: each driver run sleeps 30 s collecting an idle
# baseline.
#
# This NEVER touches config/node.env. That file is live, gitignored machine
# config -- the node's paths and its llama.cpp pin -- and a clobber is silent and
# unrecoverable. The fixture env lives in a tmpdir and reaches the driver through
# --env-file.
set -u

cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
FIX=$(mktemp -d)
PASS=0; FAIL=0

ok()   { PASS=$((PASS+1)); printf '  \033[32mok\033[0m   %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); printf '  \033[31mFAIL\033[0m %s\n' "$1"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$1"; }
cleanup() { rm -rf "$FIX" "$ROOT/bench/__pycache__" "$ROOT/telemetry/__pycache__"; }
trap cleanup EXIT

mkdir -p "$FIX/bin" "$FIX/models" "$FIX/results"
cp bench/fixtures/fake_tegrastats.py  "$FIX/bin/tegrastats"
cp bench/fixtures/fake_llama_bench.py "$FIX/bin/llama-bench"
chmod +x "$FIX/bin/tegrastats" "$FIX/bin/llama-bench"
for m in Qwen2.5-0.5B-Instruct-Q4_K_M.gguf Qwen2.5-0.5B-Instruct-Q8_0.gguf \
         Qwen2.5-1.5B-Instruct-Q4_K_M.gguf Qwen2.5-3B-Instruct-Q4_K_M.gguf; do
  head -c 8000000 /dev/urandom > "$FIX/models/$m"
done
cat > "$FIX/node.env" <<ENV
NODE_ID=fixture
LLAMA_BIN="$FIX/bin"
MODEL_DIR="$FIX/models"
RESULTS_DIR="$FIX/results"
TELEMETRY_HZ=5
ENV
export PATH="$FIX/bin:$PATH" HS_FAKE_PHASE="$FIX/phase"
echo idle > "$HS_FAKE_PHASE"

# Fingerprint the live machine config. Nothing this script runs may touch it --
# it is gitignored, it holds this node's paths and its llama.cpp pin, and it has
# been silently clobbered by a fixture before. Checked again at the end.
REAL_ENV="$ROOT/config/node.env"
if [ -f "$REAL_ENV" ]; then ENV_BEFORE=$(md5sum < "$REAL_ENV"); else ENV_BEFORE=absent; fi

head_ "syntax"
for f in scripts/*.sh scripts/lib/*.sh bench/fixtures/*.sh; do
  bash -n "$f" || bad "bash -n $f"
done
python3 - <<'PY' && ok "python + json parse" || bad "python + json parse"
import ast, json, sys
for f in ["telemetry/tegrastats_logger.py", "telemetry/energy_window.py",
          "telemetry/make_synthetic_power_csv.py", "bench/run_matrix.py",
          "bench/check_finding3_invariants.py",
          "bench/fixtures/fake_tegrastats.py", "bench/fixtures/fake_llama_bench.py"]:
    ast.parse(open(f, encoding="utf-8").read())
for f in ["bench/matrix.json", "bench/fixtures/matrix_mixed.json",
          "bench/fixtures/matrix_split.json"]:
    json.load(open(f))
PY

head_ "unit: helpers"
python3 - <<'PY' && ok "helper assertions" || bad "helper assertions"
import csv, sys
sys.path.insert(0, "bench"); sys.path.insert(0, "telemetry")
from run_matrix import (bench_csv_text, row_tps, compute_seconds, row_compute_seconds,
                        tokens_of, phase_of, phase_of_counts, per_token_status,
                        phase_rows, bench_records, expand)
from energy_window import pick, bench_tokens, bench_compute_s

banner = 'ggml_init_cublas: found 1 CUDA devices:\n"build_commit","avg_ts"\n"a1b","10.0"\n'
assert bench_csv_text(banner).splitlines()[0].startswith('"build_commit"')
assert bench_csv_text("no csv here\n") == ""
assert row_tps(list(csv.DictReader(bench_csv_text(banner).splitlines()))[0]) == 10.0

# finding #7: "0" must not read as a token count
assert pick({"n_prompt": "0", "n_gen": "128"}, "n_prompt", "n_gen", skip_zero=True) == "128"
assert bench_tokens({"n_prompt": "0", "n_gen": "128"}, 3)[0] == 385.0
assert bench_tokens({"n_prompt": "512", "n_gen": "0"}, 3)[0] == 2048.0
# a fused row is refused, with a reason, NOT summed
tok, why = bench_tokens({"n_prompt": "512", "n_gen": "128"}, 3)
assert tok is None and why, (tok, why)

# finding #3: the row's own counts drive the divisor and the phase
assert phase_of_counts(512, 0) == "prefill"
assert phase_of_counts(0, 128) == "decode"
assert phase_of_counts(512, 128) == "combined"
assert tokens_of({"bench_n_prompt": 0, "bench_n_gen": 128, "repetitions": 3}) == 385
assert tokens_of({"bench_n_prompt": 512, "bench_n_gen": 0, "repetitions": 3}) == 2048
# upstream warms a -pg instance with a full prompt + test_gen(ctx, 1), i.e.
# ONE token -- so a fused row uses the same formula as the split halves
assert tokens_of({"bench_n_prompt": 512, "bench_n_gen": 128, "repetitions": 3}) == 2433
# the split changes no measured work
for r in (2, 3, 5):
    assert tokens_of({"bench_n_prompt": 512, "bench_n_gen": 0, "repetitions": r}) \
         + tokens_of({"bench_n_prompt": 0, "bench_n_gen": 128, "repetitions": r}) \
         == tokens_of({"bench_n_prompt": 512, "bench_n_gen": 128, "repetitions": r}), r

# the guard counts rows that did WORK, not rows -- a ref emitting a row for the
# zero axis must not blank the whole sweep
assert len(phase_rows([{"n_prompt": "512", "n_gen": "0"}, {"n_prompt": "0", "n_gen": "0"}])) == 1
rows = [{"n_prompt": "512", "n_gen": "0", "avg_ts": "1", "avg_ns": "1e9", "test": "pp512"},
        {"n_prompt": "0", "n_gen": "128", "avg_ts": "2", "avg_ns": "1e9", "test": "tg128"}]
recs = bench_records({"repetitions": 3, "n_prompt": 512, "n_gen": 0},
                     rows, {"status": "ok"})
assert len(recs) == 2, len(recs)
assert all(r["per_token_status"] == "refused_multi_phase" for r in recs)
# zero bench rows still yields exactly ONE record, or failures vanish
assert len(bench_records({"repetitions": 3, "n_prompt": 512, "n_gen": 0}, [], {"status": "ok"})) == 1

assert row_compute_seconds({"avg_ns": "1000000000", "n_prompt": "0", "n_gen": "128"}, 3) == 3.008
assert compute_seconds([], 3) is None

# A fused row that SHARES its window is refused as multi-phase, not
# re-denominated per request -- the window's joules cover work it does not
# describe. This is what a ref emitting pp+tg+fused for one -pg would do.
three = [{"n_prompt": "512", "n_gen": "0", "test": "pp512", "avg_ts": "1", "avg_ns": "1e9"},
         {"n_prompt": "0", "n_gen": "128", "test": "tg128", "avg_ts": "2", "avg_ns": "1e9"},
         {"n_prompt": "512", "n_gen": "128", "test": "pp512+tg128", "avg_ts": "3", "avg_ns": "2e9"}]
recs = bench_records({"repetitions": 3, "n_prompt": 0, "n_gen": 0}, three, {"status": "ok"})
assert [r["per_token_status"] for r in recs] == ["refused_multi_phase"] * 3, \
    [r["per_token_status"] for r in recs]
PY

head_ "unit: expand() refuses a mixed case"
python3 - <<'PY' && ok "expand refusal" || bad "expand refusal"
import sys
sys.path.insert(0, "bench")
from run_matrix import expand
for case in ({"id": "m1", "model": "x", "n_prompt": [512], "n_gen": [128]},
             {"id": "m2", "model": "x", "n_prompt": [512, 0], "n_gen": [0, 128]},
             {"id": "m3", "model": "x", "pg": [512, 128], "n_prompt": [512]},
             {"id": "m4", "model": "x", "pg": [512]}):
    try:
        expand(case, {})
    except SystemExit as exc:
        assert case["id"] in str(exc), str(exc)
    else:
        raise AssertionError("case %s was NOT refused" % case["id"])
# the legal shapes still expand
assert len(expand({"id": "p", "model": "x", "n_prompt": [128, 512], "n_gen": [0]}, {})) == 2
assert len(expand({"id": "g", "model": "x", "pg": [512, 128]}, {})) == 1
PY

head_ "the shipped matrix"
python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 --dry-run > "$FIX/dry.txt" 2>&1
# 33, not 32: the sweep now closes with an unchanged repeat of case 1, whose
# delta against the original IS the thermal drift over the sweep (finding #10).
grep -q "phases    : combined 2, decode 11, prefill 20" "$FIX/dry.txt" \
  && ok "33 runs (32 + drift control), zero mixed" \
  || { bad "run/phase counts"; sed -n '1,6p' "$FIX/dry.txt"; }
grep -q "runs      : 33" "$FIX/dry.txt" \
  && ok "the drift-control repeat is in the plan, not bolted on after" \
  || bad "drift control missing from the run plan"
python3 - <<'PY' && ok "no case mixes phases" || bad "a case mixes phases"
import json, sys
m = json.load(open("bench/matrix.json"))
for c in m["cases"]:
    p = c.get("n_prompt", [0]); g = c.get("n_gen", [0])
    p = p if isinstance(p, list) else [p]; g = g if isinstance(g, list) else [g]
    assert not (any(p) and any(g)), c["id"]
PY
python3 - "$FIX/dry.txt" <<'PY' && ok "split pairs interleave" || bad "split pairs do not interleave"
import re, sys
rows = [l.split() for l in open(sys.argv[1]) if re.match(r"\s+\d+\s+\w", l)]
seq = [(r[1], r[2].split("|")[0]) for r in rows]
pairs = [s for s in seq if s[1].startswith("partial_offload")]
phases = [p for p, _ in pairs]
assert phases == ["prefill", "decode"] * 5, phases
PY

head_ "driver: mixed case is refused before any hardware time"
python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 --matrix bench/fixtures/matrix_mixed.json --dry-run \
  > "$FIX/refuse.txt" 2>&1
if [ $? -ne 0 ] && grep -q "mixed_case" "$FIX/refuse.txt"; then
  ok "refused by name, exit 1"
else
  bad "mixed case not refused"; cat "$FIX/refuse.txt"
fi

head_ "driver: post-fix sweep end to end"
rm -f "$FIX"/results/*
python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 --matrix bench/fixtures/matrix_split.json --tag split \
  > "$FIX/split.txt" 2>&1
grep -q "refused_combined_request   1" "$FIX/split.txt" \
  && ok "fused case refuses per-token" || { bad "fused case status"; tail -6 "$FIX/split.txt"; }
python3 bench/check_finding3_invariants.py "$FIX"/results/*split_bench_energy.csv \
  > "$FIX/inv.txt" 2>&1 && ok "invariants hold" || { bad "invariants"; cat "$FIX/inv.txt"; }
python3 - "$FIX" <<'EOPY' && ok "J/request matches hand-computed physics" || bad "J/request physics"
import csv, glob, sys
# The fake burns 7400 mW for PP_NS=0.5 s and 4800 mW for TG_NS=1.0 s per timed
# repetition, so one whole request is 0.5*7.4 + 1.0*4.8 = 8.5 J. The per-request
# window must isolate the reps and exclude the load and the warm-up entirely.
rows = [r for r in csv.DictReader(open(glob.glob(sys.argv[1] + "/results/*split_bench_energy.csv")[0]))
        if r["row_phase"] == "combined"]
assert len(rows) == 1, len(rows)
r = rows[0]
assert r["per_token_status"] == "refused_combined_request", r["per_token_status"]
assert not r.get("joules_per_token"), "a fused row published J/token"
assert abs(float(r["request_window_s"]) - 4.5) < 0.01, r["request_window_s"]
jpr = float(r["joules_per_request"])
# 3%: what is left after interpolating the counter to the slice bounds is the
# residual from anchoring on the last MARKED sample, which can sit up to one
# sampling interval after compute actually ended.
assert abs(jpr - 8.5) / 8.5 < 0.03, "joules_per_request=%s, expected ~8.5" % jpr
# and it must be self-consistent with its own stated basis
assert r["energy_basis"] == "timed requests", r["energy_basis"]
aj, ajr = float(r["active_joules"]), float(r["active_joules_per_request"])
assert abs(ajr * float(r["requests"]) - aj) < 1e-3, (ajr, r["requests"], aj)
EOPY

head_ "driver: a ref that returns an extra row is caught"
rm -f "$FIX"/results/*
cat > "$FIX/one_prefill.json" <<'JSON'
{"defaults": {"repetitions": 3, "threads": 4, "cooldown_s": 1},
 "cases": [{"id": "rogue_ref", "model": "qwen2.5-1.5b-instruct-q4_k_m",
            "n_gpu_layers": 99, "n_prompt": [512], "n_gen": [0]}]}
JSON
HS_FAKE_EXTRA_ROW=1 python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 --matrix "$FIX/one_prefill.json" \
  --tag rogue > "$FIX/rogue.txt" 2>&1
grep -q "refused_multi_phase        2" "$FIX/rogue.txt" \
  && ok "both rows refused" || { bad "rogue rows not refused"; tail -6 "$FIX/rogue.txt"; }
python3 - "$FIX" <<'PY' && ok "no per-token column written at all" || bad "per-token column leaked"
import csv, glob, sys
f = glob.glob(sys.argv[1] + "/results/*rogue_bench_energy.csv")[0]
rows = list(csv.DictReader(open(f)))
assert len(rows) == 2, len(rows)
for col in ("joules_per_token", "wh_per_mtok", "active_joules_per_token"):
    assert not any(r.get(col) for r in rows), col
PY
python3 bench/check_finding3_invariants.py "$FIX"/results/*rogue_bench_energy.csv \
  > /dev/null 2>&1 && ok "invariants hold" || bad "invariants (rogue)"

head_ "driver: a ref that reports the wrong phase cannot publish a number"
rm -f "$FIX"/results/*
cat > "$FIX/pg.json" <<'JSON'
{"defaults": {"repetitions": 3, "threads": 4, "cooldown_s": 1},
 "cases": [{"id": "end_to_end_request", "model": "qwen2.5-1.5b-instruct-q4_k_m",
            "n_gpu_layers": 99, "pg": [512, 128]}]}
JSON
HS_FAKE_HIDE_PROMPT=1 python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 \
  --matrix "$FIX/pg.json" --tag mismatch > "$FIX/mm.txt" 2>&1
grep -q "refused_phase_mismatch     1" "$FIX/mm.txt" \
  && ok "-pg answered as decode is refused" || { bad "phase mismatch not refused"; tail -5 "$FIX/mm.txt"; }
python3 - "$FIX" <<'EOPY' && ok "no per-token or per-request figure leaked" || bad "a figure leaked"
import csv, glob, sys
r = list(csv.DictReader(open(glob.glob(sys.argv[1] + "/results/*mismatch_bench_energy.csv")[0])))[0]
assert r["phase"] == "combined" and r["row_phase"] == "decode", (r["phase"], r["row_phase"])
for col in ("joules_per_token", "wh_per_mtok", "active_joules_per_token",
            "joules_per_request", "wh_per_krequest"):
    assert not r.get(col), (col, r.get(col))
EOPY
python3 bench/check_finding3_invariants.py "$FIX"/results/*mismatch_bench_energy.csv \
  > "$FIX/mmi.txt" 2>&1 && ok "invariants hold, with the ref flagged as suspect" \
  || { bad "invariants (mismatch)"; cat "$FIX/mmi.txt"; }
grep -q "correctly refused" "$FIX/mmi.txt" \
  && ok "checker reports the suspect ref rather than passing silently" \
  || bad "checker did not flag the suspect ref"

head_ "driver: a ref with no count columns cannot publish a number either"
rm -f "$FIX"/results/*
HS_FAKE_NO_COUNTS=1 python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 \
  --matrix "$FIX/one_prefill.json" --tag nocounts > "$FIX/nc.txt" 2>&1
HS_FAKE_NO_COUNTS=1 HS_FAKE_EXTRA_ROW=1 python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 \
  --matrix "$FIX/one_prefill.json" --tag nc2 > "$FIX/nc2.txt" 2>&1
python3 - "$FIX" <<'EOPY' && ok "two rows, zero identifiable => refused_multi_phase" || bad "zero-working-rows guard"
import csv, glob, sys
rows = list(csv.DictReader(open(glob.glob(sys.argv[1] + "/results/*nc2_bench_energy.csv")[0])))
assert len(rows) == 2, len(rows)
for r in rows:
    assert r["per_token_status"] == "refused_multi_phase", r["per_token_status"]
    assert r["window_shared"] == "1", r["window_shared"]   # flag is row-count driven
    assert not r.get("joules_per_token")
EOPY

head_ "the driver and the analysis tool agree, and both track the physics"
rm -f "$FIX"/results/*
python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 \
  --matrix bench/fixtures/matrix_split.json --tag agree > /dev/null 2>&1
python3 - "$FIX" <<'EOPY' && ok "both tools agree exactly and match hand-computed truth" || bad "tool agreement / accuracy"
import csv, glob, json, subprocess, sys
F = sys.argv[1]
joined = glob.glob(F + "/results/*agree_bench_energy.csv")[0]
power = glob.glob(F + "/results/*agree_power.csv")[0]
# The fake burns 7400 mW for a 2.0 s prefill compute slice over 2048 tokens, and
# 4800 mW for a 3.008 s decode slice over 385 tokens. docs/10 names
# energy_window.py as the way to re-derive a published figure by hand, so the two
# must not merely be close -- they must be the same number.
TRUTH = {"prefill": 2.0 * 7.4 / 2048, "decode": 3.008 * 4.8 / 385}
n = 0
for r in csv.DictReader(open(joined)):
    if r["per_token_status"] != "ok":
        continue
    out = subprocess.run([sys.executable, "telemetry/energy_window.py", power,
                          "--marker", r["marker"], "--bench", F + "/results/" + r["bench_csv"],
                          "--reps", r["repetitions"], "--idle-mw", r["idle_mw"], "--json"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         universal_newlines=True)
    w = json.loads(out.stdout)[0]
    a, b = float(r["joules_per_token"]), w["joules_per_token"]
    assert abs(a - b) < 1e-9, (r["marker"], a, b)
    t = TRUTH[r["row_phase"]]
    assert abs(a - t) / t < 0.03, (r["marker"], a, t, abs(a - t) / t)
    # and the row must be self-consistent: mean power x span == energy
    cj, cs, mw = float(r["compute_joules"]), float(r["compute_s"]), float(r["compute_mean_mw"])
    assert abs(mw * cs / 1000.0 - cj) / cj < 0.05, (r["marker"], mw, cs, cj)
    n += 1
# 5, not 4: the drift-control repeat of case 1 is a real measured row
assert n == 5, n
EOPY

head_ "driver: counts that cannot be trusted cannot be divided by"
rm -f "$FIX"/results/*
cat > "$FIX/pre.json" <<'JSON'
{"defaults": {"repetitions": 3, "threads": 4, "cooldown_s": 1},
 "cases": [{"id": "pre", "model": "qwen2.5-1.5b-instruct-q4_k_m",
            "n_gpu_layers": 99, "n_prompt": [512], "n_gen": [0]}]}
JSON
HS_FAKE_WRONG_COUNTS=1 python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 \
  --matrix "$FIX/pre.json" --tag wrongcounts > /dev/null 2>&1
HS_FAKE_NO_COUNTS=1 python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 \
  --matrix "$FIX/pre.json" --tag nocounts1 > /dev/null 2>&1
python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 \
  --matrix "$FIX/pre.json" --tag honest > /dev/null 2>&1
python3 - "$FIX" <<'EOPY' && ok "wrong / unverifiable counts refused, honest ref ok" || bad "counts verification"
import csv, glob, sys
def row(tag):
    return list(csv.DictReader(open(glob.glob(sys.argv[1] + "/results/*%s_bench_energy.csv" % tag)[0])))[0]
# a ref answering -p 512 with n_prompt=128: same phase, same label, divisor 4x too small
w = row("wrongcounts")
assert w["per_token_status"] == "refused_counts_mismatch", w["per_token_status"]
assert not w.get("joules_per_token")
# a ref reporting no counts at all: row_phase was derived from the request, so the
# phase check compared the request with itself and could never fail
n = row("nocounts1")
assert n["counts_source"] == "matrix", n["counts_source"]
assert n["per_token_status"] == "refused_counts_unverified", n["per_token_status"]
assert not n.get("joules_per_token")
# and an honest ref still gets its number
h = row("honest")
assert h["per_token_status"] == "ok", h["per_token_status"]
assert float(h["joules_per_token"]) > 0
EOPY
for t in wrongcounts nocounts1 honest; do
  python3 bench/check_finding3_invariants.py "$FIX"/results/*${t}_bench_energy.csv > /dev/null 2>&1 \
    || { bad "invariants ($t)"; break; }
done && ok "invariants hold for all three"

head_ "driver: a ref that ran a different configuration is refused"
rm -f "$FIX"/results/*
HS_FAKE_IGNORE_NGL=1 python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 \
  --matrix "$FIX/pre.json" --tag ignorengl > /dev/null 2>&1
python3 - "$FIX" <<'EOPY' && ok "a ref that ignores -ngl publishes nothing" || bad "config cross-check"
import csv, glob, sys
r = list(csv.DictReader(open(glob.glob(sys.argv[1] + "/results/*ignorengl_bench_energy.csv")[0])))[0]
assert r["n_gpu_layers"] == "99" and r["bench_n_gpu_layers"] == "0", (r["n_gpu_layers"], r["bench_n_gpu_layers"])
assert r["config_mismatch"] == "1", r["config_mismatch"]
assert r["per_token_status"] == "refused_config_mismatch", r["per_token_status"]
assert not r.get("joules_per_token")
EOPY

head_ "checker: a refusal that still publishes is a violation, not a note"
python3 - "$FIX" <<'EOPY'
import csv, io, sys
row = {"case_id": "x", "test": "tg128", "status": "ok",
       "per_token_status": "refused_counts_mismatch",
       "n_prompt": "0", "n_gen": "128", "bench_n_prompt": "0", "bench_n_gen": "128",
       "tokens": "385", "joules": "10", "joules_per_token": "0.026"}
with io.open(sys.argv[1] + "/sneaky.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(row)); w.writeheader(); w.writerow(row)
EOPY
python3 bench/check_finding3_invariants.py "$FIX/sneaky.csv" > /dev/null 2>&1 \
  && bad "a refused row publishing a number exited 0" || ok "caught as a violation"

head_ "a run whose model is missing still appears in the CSV"
rm -f "$FIX"/results/*
cat > "$FIX/missing.json" <<'JSON'
{"defaults": {"repetitions": 3, "threads": 4, "cooldown_s": 1},
 "cases": [{"id": "has_model", "model": "qwen2.5-1.5b-instruct-q4_k_m",
            "n_gpu_layers": 0, "n_prompt": [512], "n_gen": [0]},
           {"id": "no_model", "model": "qwen2.5-3b-instruct-q4_k_m",
            "n_gpu_layers": 0, "n_prompt": [512], "n_gen": [0]}]}
JSON
rm -f "$FIX/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 --matrix "$FIX/missing.json" \
  --tag nomodel --no-power > "$FIX/nm.txt" 2>&1
# ...and again WITH power logging, which is the only mode the Jetson ever uses.
# A skipped record carrying no timestamps once made the join raise KeyError and
# the joined CSV -- the archived deliverable -- was never written at all. The
# --no-power run above cannot see that, because the join is skipped entirely.
python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 --matrix "$FIX/missing.json" \
  --tag nomodelpwr > "$FIX/nmp.txt" 2>&1
ls "$FIX"/results/*nomodelpwr_bench_energy.csv > /dev/null 2>&1 \
  && ok "joined CSV survives a missing model with power on" \
  || { bad "the sweep lost its joined CSV"; tail -6 "$FIX/nmp.txt"; }
python3 - "$FIX" <<'EOPY' && ok "skipped run is a distinct invocation" || bad "skipped invocation count"
import csv, glob, sys
rows = list(csv.DictReader(open(glob.glob(sys.argv[1] + "/results/*nomodelpwr_bench_energy.csv")[0])))
# 3, not 2: one good run, one skipped run, plus the drift-control repeat
assert len(rows) == 3, len(rows)
assert len({r["marker"] for r in rows}) == 3, [r["marker"] for r in rows]
sk = [r for r in rows if r["status"] == "skipped"][0]
assert sk["per_token_status"] == "model_missing"
assert not sk.get("joules_per_token")
EOPY
python3 - "$FIX" <<'EOPY' && ok "skipped run recorded; --no-power is not 'ok'" || bad "skip / no-power status"
import csv, glob, sys
rows = list(csv.DictReader(open(glob.glob(sys.argv[1] + "/results/*nomodel_bench_energy.csv")[0])))
assert len(rows) == 3, len(rows)          # + the drift-control repeat
by = {r["case_id"]: r for r in rows}
assert by["no_model"]["status"] == "skipped", by["no_model"]["status"]
assert by["no_model"]["per_token_status"] == "model_missing"
# --no-power means no energy at all -- "ok" would be a lie
assert by["has_model"]["per_token_status"] == "no_power_window", by["has_model"]["per_token_status"]
EOPY

head_ "the CSV is machine-readable even when runs were skipped"
python3 - "$FIX" <<'EOPY' && ok "no column is sometimes-blank; skipped runs counted apart" || bad "column consistency"
import csv, glob, sys
f = glob.glob(sys.argv[1] + "/results/*nomodelpwr_bench_energy.csv")[0]
rows = list(csv.DictReader(open(f)))
# window_shared must be int()-able on every row, including skipped ones
for r in rows:
    assert r["window_shared"] in ("0", "1"), (r["case_id"], repr(r["window_shared"]))
    assert r["marker"], r["case_id"]          # unique per run, never blank
assert len({r["marker"] for r in rows}) == len(rows)
EOPY

head_ "text I/O does not depend on the locale"
python3 - <<'EOPY' && ok "every text open() pins utf-8" || bad "a bare open() is back"
import re, sys
bad = []
for f in ("bench/run_matrix.py", "bench/check_finding3_invariants.py",
          "telemetry/energy_window.py", "telemetry/tegrastats_logger.py",
          "telemetry/make_synthetic_power_csv.py",
          "bench/fixtures/fake_llama_bench.py", "bench/fixtures/fake_tegrastats.py"):
    for i, line in enumerate(open(f, encoding="utf-8"), 1):
        if re.search(r"\bopen\(", line) and "encoding=" not in line \
           and '"rb"' not in line and "subprocess" not in line:
            bad.append("%s:%d %s" % (f, i, line.strip()))
assert not bad, bad
# subprocess text pipes decode with the LOCALE encoding under universal_newlines
# alone, which is ASCII on a C-locale 3.6 -- one stray byte in a model name, a
# CUDA banner or a tegrastats line would kill the run mid-sweep.
import subprocess as _sp
for f in ("bench/run_matrix.py", "telemetry/tegrastats_logger.py"):
    src = open(f, encoding="utf-8").read()
    n_pipes = src.count("universal_newlines=True") + src.count("text=True")
    n_enc = sum(1 for i in range(len(src))
                for tok in ("universal_newlines=True", "text=True")
                if src.startswith(tok, i) and 'encoding="utf-8"' in src[i:i + 400])
    assert n_pipes == n_enc, (f, n_pipes, n_enc)
# config the driver reads at runtime must be plain ASCII too: a C-locale 3.6
# decodes with ASCII, and models.tsv used to carry a U+2014
for f in ("config/models.tsv", "bench/matrix.json"):
    b = open(f, "rb").read()
    assert all(c < 128 for c in b), f
EOPY

head_ "driver: a failed run still produces exactly one row"
rm -f "$FIX"/results/*
cat > "$FIX/bin/llama-bench" <<'SH'
#!/bin/sh
echo "cudaMalloc failed: out of memory" >&2
exit 1
SH
chmod +x "$FIX/bin/llama-bench"
python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 --matrix "$FIX/one_prefill.json" --tag oom \
  > "$FIX/oom.txt" 2>&1
python3 - "$FIX" <<'PY' && ok "one row, status=failed, run_failed" || bad "failed-run record"
import csv, glob, sys
rows = list(csv.DictReader(open(glob.glob(sys.argv[1] + "/results/*oom_bench_energy.csv")[0])))
assert len(rows) == 1, len(rows)
assert rows[0]["status"] == "failed" and rows[0]["per_token_status"] == "run_failed"
assert not rows[0].get("joules_per_token")
PY
cp bench/fixtures/fake_llama_bench.py "$FIX/bin/llama-bench"; chmod +x "$FIX/bin/llama-bench"

head_ "analysis tool"
python3 telemetry/make_synthetic_power_csv.py "$FIX/p.csv" > /dev/null
python3 telemetry/energy_window.py "$FIX/p.csv" --marker decode --idle-mw 1900 \
  | grep -q "55.2 J" && ok "documented window reproduces" || bad "synthetic window"
python3 - <<'PY' && ok "parse_time: all forms agree" || bad "parse_time"
import sys
sys.path.insert(0, "telemetry")
from energy_window import parse_time
v = [parse_time(x) for x in ["1789226343", "2026-09-12T15:19:03Z",
                             "2026-09-12T15:19:03+00:00", "2026-09-12T22:19:03+07:00"]]
assert len(set(v)) == 1, v
PY
H='"build_commit","n_prompt","n_gen","test","avg_ns","avg_ts"'
printf '%s\n"a","0","128","tg128","12800000000","10.00"\n' "$H" > "$FIX/one.csv"
printf '%s\n"a","512","0","pp512","3200000000","160.00"\n"a","0","128","tg128","12800000000","10.00"\n' "$H" > "$FIX/two.csv"
printf '%s\n"a","512","128","pp512+tg128","16000000000","40.00"\n' "$H" > "$FIX/fused.csv"
python3 telemetry/energy_window.py "$FIX/p.csv" --marker decode --bench "$FIX/one.csv" --reps 3 \
  2>/dev/null | grep -q "J/tok" && ok "single-row join works" || bad "single-row join"
head_ "analysis tool: an estimated divisor is never reported as ok"
printf '%s\n%s\n' '"build_commit","test","avg_ts"' '"a","tg128","10.00"' > "$FIX/notok.csv"
python3 telemetry/energy_window.py "$FIX/p.csv" --marker decode --bench "$FIX/notok.csv" \
  --reps 3 --json 2>/dev/null > "$FIX/est.json"
python3 - "$FIX" <<'EOPY' && ok "flagged estimated_from_throughput, divisor uses the right span" || bad "throughput fallback"
import json, sys
w = json.load(open(sys.argv[1] + "/est.json"))[0]
assert w["per_token_status"] == "estimated_from_throughput", w["per_token_status"]
# the divisor must cover the same interval as the numerator
span = w.get("compute_s") or w["duration_s"]
assert abs(w["tokens"] - w["bench_tps"] * span) < 1e-6, (w["tokens"], w["bench_tps"], span)
EOPY

python3 telemetry/energy_window.py "$FIX/p.csv" --marker decode --bench "$FIX/two.csv" --reps 3 \
  > /dev/null 2>&1 && bad "two-row join was NOT refused" || ok "two-row join refused"
python3 telemetry/energy_window.py "$FIX/p.csv" --bench "$FIX/one.csv" --reps 3 \
  > /dev/null 2>&1 && bad "whole-file join was NOT refused" || ok "no-window join refused"
# a window FLAG is not a window: --from/--to spanning the file satisfied the check
# above and reproduced the very number it exists to prevent
python3 telemetry/energy_window.py "$FIX/p.csv" --from 0 --to 9999999999 \
  --bench "$FIX/one.csv" --reps 3 > /dev/null 2>&1 \
  && bad "--from/--to spanning idle was NOT refused" || ok "idle-contaminated window refused"
python3 telemetry/energy_window.py "$FIX/p.csv" --marker decode:x:001 --bench "$FIX/one.csv" \
  --reps 3 > /dev/null 2>&1; [ $? -ne 0 ] && ok "unknown marker refused" || bad "unknown marker accepted"
python3 telemetry/energy_window.py "$FIX/p.csv" --marker combined --bench "$FIX/fused.csv" --reps 3 \
  2>/dev/null | grep -q "refused_combined_request" && ok "fused row refused" || bad "fused row"
# ...and a fused row joined to a DECODE window is caught as a mismatched pairing,
# not silently accepted because the counts happen to line up
python3 telemetry/energy_window.py "$FIX/p.csv" --marker decode --bench "$FIX/fused.csv" --reps 3 \
  > /dev/null 2>&1 && bad "fused row joined to a decode window" || ok "mismatched window/row pairing refused"

head_ "model fetcher: a truncated GGUF is never mistaken for a complete one"
# Runs against a COPY of the kit: 30_fetch_models.sh sources config/node.env from
# its own tree, and the real one is live machine config.
KIT="$FIX/kit"
mkdir -p "$KIT" "$FIX/serve" "$FIX/dl"
cp -a "$ROOT" "$KIT/Jetson"
cp bench/fixtures/fake_curl.py "$FIX/bin/curl"; chmod +x "$FIX/bin/curl"
head -c 5000 /dev/urandom > "$FIX/serve/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
head -c 9000 /dev/urandom > "$FIX/serve/Qwen2.5-0.5B-Instruct-Q8_0.gguf"
python3 - "$KIT" "$FIX" <<'EOPY'
import io, re, sys
kit, fix = sys.argv[1], sys.argv[2]
p = kit + "/Jetson/config/models.tsv"
s = io.open(p, encoding="utf-8").read()
s = re.sub(r"https://huggingface\.co/\S+?/resolve/main/", "http://fake/", s)
io.open(p, "w", encoding="utf-8").write(s)
io.open(kit + "/Jetson/config/node.env", "w", encoding="utf-8").write(
    'NODE_ID=fetchtest\nMODEL_DIR="%s/dl"\nHS_ROOT="%s"\nRESULTS_DIR="%s/results"\n'
    'LLAMA_BIN="/nonexistent"\n'
    'MODELS="qwen2.5-0.5b-instruct-q4_k_m qwen2.5-0.5b-instruct-q8_0"\n' % (fix, fix, fix))
EOPY
export HS_SERVE="$FIX/serve"
FETCH="$KIT/Jetson/scripts/30_fetch_models.sh"

bash "$FETCH" > "$FIX/f1.txt" 2>&1 \
  && ok "fresh download of both models" || { bad "fresh download"; tail -5 "$FIX/f1.txt"; }

# THE BUG: 3000 of 5000 bytes on disk. The old presence gate was `[ -s "$dest" ]`,
# true of any non-empty file, so this was reported "already present" and its
# hash written into SHA256SUMS as ground truth.
head -c 3000 "$FIX/serve/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" > "$FIX/dl/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
rm -f "$FIX/dl/SHA256SUMS"
bash "$FETCH" qwen2.5-0.5b-instruct-q4_k_m > "$FIX/f2.txt" 2>&1
if cmp -s "$FIX/dl/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" "$FIX/serve/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"; then
  ok "truncated file detected and completed"
else
  bad "truncated file was accepted or not repaired"; tail -5 "$FIX/f2.txt"
fi
grep -q "TRUNCATED" "$FIX/f2.txt" && ok "and it said so" || bad "truncation was silent"

# a size-preserving tamper: only re-verifying SHA256SUMS can catch it. The old
# code skipped the hash step entirely whenever a line already existed.
python3 - "$FIX" <<'EOPY'
import sys
f = sys.argv[1] + "/dl/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
d = bytearray(open(f, "rb").read()); d[0] ^= 0xFF
open(f, "wb").write(bytes(d))
EOPY
bash "$FETCH" qwen2.5-0.5b-instruct-q4_k_m > "$FIX/f3.txt" 2>&1 \
  && bad "a tampered file exited 0" || ok "tampered file caught, non-zero exit"
grep -q "sha256 CHANGED" "$FIX/f3.txt" && ok "and named the mismatch" || bad "mismatch not reported"

# one dead URL must not skip every model after it
rm -f "$FIX/dl"/*.gguf "$FIX/dl"/*.part "$FIX/dl"/*.bad "$FIX/dl/SHA256SUMS"
HS_FAIL_URL=Q4_K_M bash "$FETCH" > "$FIX/f4.txt" 2>&1
[ "$(ls "$FIX/dl"/*.gguf 2>/dev/null | wc -l)" = "1" ] \
  && ok "a dead URL does not skip the models after it" \
  || { bad "one bad URL aborted the loop"; tail -5 "$FIX/f4.txt"; }
# and a partial is never left where a complete file would be looked for
[ -f "$FIX/dl/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" ] \
  && bad "a failed download left a file at the real path" \
  || ok "failed download left only a .part sidecar"

head_ "fact collection survives a tegrastats that never exits"
# `timeout` returns 124 by design here; unguarded under pipefail it killed the
# script before its only output, and `make facts | tee` then reported success
# over a zero-byte JSON file.
PATH="$FIX/bin:$PATH" bash "$ROOT/scripts/60_collect_node_facts.sh" \
  > "$FIX/facts.json" 2>/dev/null
python3 - "$FIX" <<'EOPY' && ok "emits valid JSON with the idle baseline in it" || bad "facts script"
import json, sys
d = json.load(open(sys.argv[1] + "/facts.json"))
assert d["power"]["idle_board_mw"] == 1900, d["power"]["idle_board_mw"]
assert d["power"]["external_meter"] is False
EOPY
bash "$ROOT/scripts/60_collect_node_facts.sh" > "$FIX/facts_off.json" 2>/dev/null \
  && ok "and still works off-device, with no tegrastats at all" || bad "facts off-device"
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$FIX/facts_off.json" \
  && ok "off-device output is valid JSON too" || bad "off-device JSON"

head_ "bring-up: a missing host compiler fails loudly, not silently"
# `-DCMAKE_CUDA_HOST_COMPILER="$(command -v g++-8)"` inside an array assignment
# returns 1 and prints NOTHING when g++-8 is absent, and under `set -e` that took
# the whole build script down with no output -- the normal outcome on a fresh
# JetPack box, since CUDA 10.2 refuses gcc > 8.
BK="$FIX/buildkit"
mkdir -p "$BK/fakebin" "$BK/ws/llama.cpp"
cp -a "$ROOT" "$BK/Jetson"
printf '#!/bin/sh\necho "Cuda compilation tools, release 10.2, V10.2.300"\n' > "$BK/fakebin/nvcc"
chmod +x "$BK/fakebin/nvcc"
cat > "$BK/Jetson/config/node.env" <<ENV
NODE_ID=buildtest
HS_ROOT="$BK/ws"
LLAMA_SRC="$BK/ws/llama.cpp"
LLAMA_BUILD="$BK/ws/llama.cpp/build"
LLAMA_BIN="$BK/ws/llama.cpp/build/bin"
MODEL_DIR="$BK/ws/models"
RESULTS_DIR="$BK/ws/results"
BUILD_CUDA=1
CUDA_ARCH=53
BUILD_JOBS=1
HOST_CXX=g++-nonexistent-99
LLAMA_CPP_URL="https://example.invalid/llama.cpp.git"
LLAMA_CPP_REF=""
CUDA_UNIFIED_MEMORY=0
MODELS=""
ENV
( cd "$BK/ws/llama.cpp" && git init -q . && git config user.email t@t \
  && git config user.name t && printf 'option(GGML_CUDA "" OFF)\n' > CMakeLists.txt \
  && git add -A && git commit -qm init ) > /dev/null 2>&1
PATH="$BK/fakebin:$PATH" bash "$BK/Jetson/scripts/20_build_llama_cpp.sh" \
  > "$FIX/build.txt" 2>&1
if grep -q "g++-nonexistent-99 not found" "$FIX/build.txt" \
   && grep -q "apt-get install -y g++-8" "$FIX/build.txt"; then
  ok "names the missing compiler and how to install it"
else
  bad "missing compiler failed silently again"; tail -4 "$FIX/build.txt"
fi

head_ "bring-up: an interrupted swapfile does not wedge the box"
# `[ ! -f "$SWAPFILE" ]` was the old guard. A Ctrl-C during the multi-minute SD
# allocation leaves a partial file, so that test was false, BOTH the allocation
# and the mkswap were skipped, `swapon` failed, and `set -e` killed the script --
# identically on every later run.
mkdir -p "$FIX/swapbin"
printf '#!/bin/sh\nexec "$@"\n' > "$FIX/swapbin/sudo"
cat > "$FIX/swapbin/swaplabel" <<'SH'
#!/bin/sh
tail -c 10 "$1" 2>/dev/null | grep -q SWAPSPACE2 || exit 1
SH
chmod +x "$FIX/swapbin/sudo" "$FIX/swapbin/swaplabel"
python3 - "$FIX" <<'EOPY'
import io, re, sys
# extract the predicate from the real script so this test tracks the code
s = io.open("scripts/00_system_prep.sh", encoding="utf-8").read()
m = re.search(r"swapfile_ready\(\) \{.*?\n\}\n", s, re.S)
assert m, "swapfile_ready() not found in 00_system_prep.sh"
io.open(sys.argv[1] + "/pred.sh", "w", encoding="utf-8").write("set -uo pipefail\n" + m.group(0))
EOPY
swap_case() {
  local want="$1"; shift
  SWAPFILE="$FIX/sw"; SWAP_BYTES=1048576
  rm -f "$SWAPFILE"; eval "$1"
  local got
  got=$( PATH="$FIX/swapbin:$PATH" bash -c \
    ". '$FIX/pred.sh'; SWAPFILE='$SWAPFILE' SWAP_BYTES=$SWAP_BYTES; \
     swapfile_ready && echo ready || echo rebuild" )
  [ "$got" = "$want" ] || { echo "       expected $want, got $got"; return 1; }
}
SWAPOK=1
swap_case rebuild 'true'                                                         || SWAPOK=0
swap_case rebuild 'head -c 500000 /dev/zero > "$SWAPFILE"'                       || SWAPOK=0
swap_case rebuild 'head -c 1048576 /dev/zero > "$SWAPFILE"'                      || SWAPOK=0
swap_case rebuild 'head -c 500000 /dev/zero > "$SWAPFILE"; printf SWAPSPACE2 >> "$SWAPFILE"' || SWAPOK=0
swap_case ready   'head -c 1048566 /dev/zero > "$SWAPFILE"; printf SWAPSPACE2 >> "$SWAPFILE"' || SWAPOK=0
[ "$SWAPOK" = 1 ] && ok "a partial or unformatted swapfile is rebuilt, a good one reused" \
                  || bad "swapfile readiness predicate"

head_ "cross-node: the shared 40-column projection"
python3 -m unittest discover -s tests -q > "$FIX/schema.txt" 2>&1 \
  && ok "shared-schema tests pass (incl. live drift checks against Pi4)" \
  || { bad "shared-schema tests"; tail -12 "$FIX/schema.txt"; }
# the sweep above already wrote one; check the driver emits it automatically
ls "$FIX"/results/*_results.csv > /dev/null 2>&1 \
  && ok "the driver emits it automatically" || bad "no _results.csv from the sweep"
python3 - "$FIX" <<'EOPY' && ok "header is Pi4's 40 columns in order, rows are pp/tg only" || bad "projection shape"
import csv, glob, os, re, sys
FIX = sys.argv[1]
f = sorted(glob.glob(FIX + "/results/*_results.csv"))[-1]
rows = list(csv.DictReader(open(f)))
header = next(csv.reader(open(f)))
pi4 = os.path.join(os.path.dirname(os.path.abspath("bench")), "..", "Pi4", "node", "bench_parse.py")
if os.path.exists(pi4):
    src = open(pi4, encoding="utf-8").read()
    cols = re.findall(r'"([^"]+)"', re.search(r"^COLUMNS = \[(.*?)\n\]", src, re.S | re.M).group(1))
    assert header == cols, "header drifted from Pi4"
assert all(r["phase"] in ("pp", "tg") for r in rows), {r["phase"] for r in rows}
assert all(r["node_id"] for r in rows), "a blank node_id cannot be grouped or power-joined"
# samples is the repetition count the control node multiplies tokens by
assert all(int(r["samples"]) == 3 for r in rows), [r["samples"] for r in rows]
# The throttle state is stated in the COLUMN, never as a flag: "" would read as
# checked-and-clean under Pi 4's fail-closed convention, while the flag token
# `throttle-unknown` is in the control node's THROUGHPUT_INVALIDATING tuple and
# would delete every one of this node's rows from the placement rule.
assert all(r["throttled_end"] == "unavailable" for r in rows), [r["throttled_end"] for r in rows]
assert not any("throttle" in r["flags"] for r in rows), [r["flags"] for r in rows]
EOPY

head_ "cross-node: the control node actually accepts these rows"
# The check that was missing when a node-deleting bug shipped green: every other
# assertion stops at Pi 4's header and never asks the consumer what it does with
# the row. `throttle-unknown` on every row satisfied the schema perfectly and was
# discarded in full by the placement rule.
rm -f "$FIX"/results/*
cat > "$FIX/mix.json" <<'JSON'
{"defaults": {"repetitions": 3, "threads": 4, "cooldown_s": 1},
 "cases": [{"id": "good_pp", "model": "qwen2.5-1.5b-instruct-q4_k_m",
            "n_gpu_layers": 99, "n_prompt": [512], "n_gen": [0]},
           {"id": "good_tg", "model": "qwen2.5-1.5b-instruct-q4_k_m",
            "n_gpu_layers": 99, "n_prompt": [0], "n_gen": [128]},
           {"id": "gone", "model": "qwen2.5-3b-instruct-q4_k_m",
            "n_gpu_layers": 0, "n_prompt": [512], "n_gen": [0]}]}
JSON
grep -q CLUSTER_NODE_ID "$FIX/node.env" || echo 'CLUSTER_NODE_ID=jetson' >> "$FIX/node.env"
rm -f "$FIX/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 --matrix "$FIX/mix.json" \
  --tag xnode > "$FIX/xnode.txt" 2>&1
[ -f "$FIX/results/results.csv" ] \
  && ok "wrote results.csv under the exact name the joiner opens" \
  || { bad "no results.csv"; tail -4 "$FIX/xnode.txt"; }
python3 - "$FIX" <<'EOPY' && ok "a skipped run keeps a row instead of vanishing" || bad "skipped run dropped"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1] + "/results/results.csv")))
assert len(rows) == 4, [r["flags"] for r in rows]   # + the drift-control repeat
gone = [r for r in rows if "model-missing" in r["flags"]]
assert len(gone) == 1, rows
assert all(r["throttled_end"] == "unavailable" for r in rows)
assert not any("throttle" in r["flags"] for r in rows)
EOPY
python3 - "$FIX" "$ROOT" <<'EOPY' && ok "the node reaches the placement rule, nothing dropped" || bad "control node rejects the rows"
import os, sys
FIX, ROOT = sys.argv[1], sys.argv[2]
ctl = os.path.join(os.path.dirname(ROOT), "ThinkPad")
if not os.path.exists(os.path.join(ctl, "control", "analysis.py")):
    print("   (ThinkPad/control absent -- consumer check skipped)")
    raise SystemExit(0)
sys.path.insert(0, ctl)
from control import profiles, analysis
p = FIX + "/results/results.csv"
t = profiles.per_layer_times(p, {"qwen2.5-1.5b-instruct-q4_k_m": 28}, threads=4)
assert t.get("_dropped_flagged_rows") == 0, t
assert "jetson" in t, t
assert t["jetson"].get("prefill") and t["jetson"].get("decode"), t
# one row per (node, model, quant, phase, threads) -- keying on phase alone
# silently keeps whichever model sorted last, which here is the SKIPPED one
good = "qwen2.5-1.5b-instruct-q4_k_m"
tbl = {(r["model"], r["phase"]): r for r in analysis.characterisation_table(p)}
for ph in ("pp", "tg"):
    row = tbl[(good, ph)]
    assert row["throughput_runs"] >= 1 and row["flagged"] == 0, row
# the skipped model is present and correctly marked unusable, not absent
assert tbl[("qwen2.5-3b-instruct-q4_k_m", "pp")]["flagged"] == 1
EOPY

head_ "the idle floor is local, and nothing is clamped"
rm -f "$FIX"/results/*
python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 \
  --matrix bench/fixtures/matrix_split.json --tag idle > "$FIX/idle.txt" 2>&1
python3 - "$FIX" <<'EOPY' && ok "each row corrected against the interval before IT" || bad "local idle floor"
import csv, glob, sys
F = sys.argv[1]
rows = list(csv.DictReader(open(glob.glob(F + "/results/*idle_bench_energy.csv")[0])))
# every row carries the floor it used, the interval it came from, and both of the
# figures it did NOT use -- so the subtraction is auditable from the CSV alone
for r in rows:
    for col in ("idle_mw", "idle_mw_source", "idle_mw_pooled", "idle_mw_baseline"):
        assert r.get(col) not in (None, ""), (col, r["case_id"])
    assert r["idle_mw_source"].startswith(("idle[", "baseline-fallback", "pooled-fallback"))
# local, not one scalar: consecutive cases must cite DIFFERENT intervals
srcs = [r["idle_mw_source"] for r in rows if r["idle_mw_source"].startswith("idle[")]
assert len(set(srcs)) > 1, "every row cites the same interval -- the floor is not local: %s" % set(srcs)
EOPY
python3 - "$FIX" <<'EOPY' && ok "the model pre-read is no longer counted as idle" || bad "precache marker"
import csv, glob, re, sys
# Structural, not timing-dependent: the fixture's model is small enough that the
# pre-read can finish inside one 200 ms sample, so "a precache-marked sample
# exists" is a property of the fixture, not of the code. What must hold is that
# the driver brackets precache() with its own marker -- on real hardware that
# read is 0.4-1.9 GB off an SD card, drawing several watts for tens of seconds,
# and it used to happen while the marker still said "idle", so the floor that is
# subtracted from every active-energy figure was measured partly over a file read.
lines = open("bench/run_matrix.py", encoding="utf-8").read().splitlines()
call = [i for i, l in enumerate(lines) if "cached = precache(mpath)" in l]
assert len(call) == 1, "expected exactly one precache() call site, found %d" % len(call)
i = call[0]
assert lines[i - 1].strip().startswith('set_marker("precache:'), \
    "precache() is not preceded by its own marker: %r" % lines[i - 1].strip()
assert lines[i + 1].strip() == 'set_marker("idle")', \
    "precache() is not followed by a return to idle: %r" % lines[i + 1].strip()
# and opportunistically: if any sample DID land in it, none of them is idle
rows = list(csv.DictReader(open(glob.glob(sys.argv[1] + "/results/*idle_power.csv")[0])))
pre = [r for r in rows if (r["marker"] or "").startswith("precache:")]
assert all((r["marker"] or "").strip() != "idle" for r in pre), pre[:2]
EOPY
python3 - <<'EOPY' && ok "a floor above the window's power yields NEGATIVE active energy" || bad "clamp still present"
import re
# max(x, 0.0) published a plausible 0.0 and threw away the evidence that the
# floor was wrong. Assert the clamps are gone from both tools, by source.
for f in ("bench/run_matrix.py", "telemetry/energy_window.py"):
    src = open(f, encoding="utf-8").read()
    bad = re.findall(r"max\(\s*(?:act|active|energy_j)[^)]*,\s*0\.0\s*\)", src)
    assert not bad, (f, bad)
EOPY

head_ "the analysis tool's idle floor agrees with the driver's"
python3 - "$FIX" <<'EOPY' && ok "--idle-mw auto reproduces the row's own idle_mw" || bad "floor disagreement"
import csv, glob, json, subprocess, sys
F = sys.argv[1]
joined = glob.glob(F + "/results/*idle_bench_energy.csv")[0]
power = glob.glob(F + "/results/*idle_power.csv")[0]
n = 0
for r in csv.DictReader(open(joined)):
    if r["per_token_status"] != "ok":
        continue
    out = subprocess.run([sys.executable, "telemetry/energy_window.py", power,
                          "--marker", r["marker"], "--json"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         universal_newlines=True)
    w = json.loads(out.stdout)[0]
    assert abs(w["idle_mw"] - float(r["idle_mw"])) < 0.05, (r["marker"], w["idle_mw"], r["idle_mw"])
    n += 1
assert n >= 2, n
EOPY
python3 telemetry/energy_window.py "$FIX/p.csv" --marker decode --idle-mw banana > /dev/null 2>&1 \
  && bad "a bad --idle-mw was accepted" || ok "a bad --idle-mw is refused, not silently zero"
python3 telemetry/energy_window.py "$FIX/p.csv" --marker decode --idle-mw none 2>/dev/null \
  | grep -q "active" && bad "--idle-mw none still subtracted a floor" \
  || ok "--idle-mw none reports total energy only"

head_ "fact collection records the clock cap in force, not the silicon ceiling"
# nvpmodel caps the CPU through scaling_max_freq and the GPU through its devfreq
# node; cpuinfo_max_freq and the top of available_frequencies do NOT move with
# the mode. A 5 W tree must therefore not read like a 10 W one -- that is the
# only thing that makes the two conditions separable in the dataset. The facts
# script takes its sysfs root from HS_SYSFS_ROOT, so this needs no Jetson.
mk_sys() {  # $1 root  $2 cpu scaling khz  $3 gpu max hz  $4 pmode
  mkdir -p "$1/sys/devices/system/cpu/cpu0/cpufreq" \
           "$1/sys/devices/57000000.gpu/devfreq/57000000.gpu" "$1/var/lib/nvpmodel"
  echo "$2" > "$1/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"
  echo 1479000 > "$1/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
  echo "$3" > "$1/sys/devices/57000000.gpu/devfreq/57000000.gpu/max_freq"
  echo "76800000 307200000 614400000 921600000" \
    > "$1/sys/devices/57000000.gpu/devfreq/57000000.gpu/available_frequencies"
  printf 'pmode:%04d fmode:quiet\n' "$4" > "$1/var/lib/nvpmodel/status"
}
mk_sys "$FIX/mode0" 1479000 921600000 0
mk_sys "$FIX/mode1"  918000 640000000 1
HS_SYSFS_ROOT="$FIX/mode0" bash "$ROOT/scripts/60_collect_node_facts.sh" \
  > "$FIX/m0.json" 2>/dev/null
HS_SYSFS_ROOT="$FIX/mode1" bash "$ROOT/scripts/60_collect_node_facts.sh" \
  > "$FIX/m1.json" 2>/dev/null
python3 - "$FIX" <<'EOPY' && ok "5 W and 10 W are distinguishable in the dataset" || bad "clock caps"
import json, sys
F = sys.argv[1]
a = json.load(open(F + "/m0.json")); b = json.load(open(F + "/m1.json"))
# the field that used to be recorded is the SAME in both modes -- that is the bug
assert a["cpu"]["hw_max_freq_khz"] == b["cpu"]["hw_max_freq_khz"], "test tree is wrong"
assert a["gpu"]["hw_max_freq_hz"] == b["gpu"]["hw_max_freq_hz"], "test tree is wrong"
# the field recorded now separates them
assert a["cpu"]["max_freq_khz"] == 1479000 and b["cpu"]["max_freq_khz"] == 918000, \
    (a["cpu"], b["cpu"])
assert a["gpu"]["max_freq_hz"] == 921600000 and b["gpu"]["max_freq_hz"] == 640000000, \
    (a["gpu"], b["gpu"])
# the GPU lookup found a real path, so a 0 would be diagnosable rather than mute
assert a["gpu"]["max_freq_source"].endswith("/max_freq"), a["gpu"]["max_freq_source"]
# and the live mode is read from the board, not echoed from node.env
assert a["power"]["nvpmodel_active"] == "0" and b["power"]["nvpmodel_active"] == "1", \
    (a["power"]["nvpmodel_active"], b["power"]["nvpmodel_active"])
EOPY
python3 - "$FIX" <<'EOPY' && ok "an absent GPU node yields 0 with an empty source, not a crash" || bad "gpu fallback"
import json, subprocess, sys, os, tempfile
F = sys.argv[1]
empty = tempfile.mkdtemp()
env = dict(os.environ, HS_SYSFS_ROOT=empty)
out = subprocess.run(["bash", "scripts/60_collect_node_facts.sh"], env=env,
                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                     universal_newlines=True)
assert out.returncode == 0, out.returncode
d = json.loads(out.stdout)
assert d["gpu"]["max_freq_hz"] == 0 and d["gpu"]["max_freq_source"] == "", d["gpu"]
EOPY

head_ "rpc contention guard sees every name the rpc server ships under"
# `pgrep -x rpc-server` matched one of the three names the kit resolves, and
# /proc/<pid>/comm truncates the longest to 15 chars, so the guard went quiet
# exactly when a sweep was contended. Real processes with real comm values -- a
# regex test against a string would not have caught the truncation.
mkdir -p "$FIX/rpcnames"
RPCPIDS=""
for n in rpc-server ggml-rpc-server llama-rpc-server; do
  printf '#!/bin/sh\nsleep 30\n' > "$FIX/rpcnames/$n"
  chmod +x "$FIX/rpcnames/$n"
done
python3 - "$FIX" <<'EOPY' && ok "the truncated name is the one that mattered" || bad "comm truncation"
import subprocess, sys, time, os
F = sys.argv[1]
p = subprocess.Popen([F + "/rpcnames/llama-rpc-server"])
try:
    time.sleep(0.3)
    comm = open("/proc/%d/comm" % p.pid).read().strip()
    # 16 chars in, 15 out: this is why a plain `pgrep -x llama-rpc-server` fails
    assert comm == "llama-rpc-serve", comm
finally:
    p.kill(); p.wait()
EOPY
for n in rpc-server ggml-rpc-server llama-rpc-server; do
  "$FIX/rpcnames/$n" & RPCPIDS="$RPCPIDS $!"
done
sleep 1
( . "$ROOT/scripts/lib/common.sh" 2>/dev/null
  RPC_PORT=""
  rpc_running ) && ok "rpc_running sees a worker under any of the three names" \
                || bad "rpc_running missed a running worker"
for pid in $RPCPIDS; do kill "$pid" 2>/dev/null || true; done
wait $RPCPIDS 2>/dev/null || true
sleep 1
( . "$ROOT/scripts/lib/common.sh" 2>/dev/null
  RPC_PORT=""
  rpc_running ) && bad "rpc_running reported a worker when none runs" \
                || ok "and reports none when none runs"
python3 - "$ROOT" <<'EOPY' && ok "one name list, read by both consumers" || bad "duplicate name list"
import re, sys
root = sys.argv[1]
common = open(root + "/scripts/lib/common.sh", encoding="utf-8").read()
bench = open(root + "/scripts/50_bench_device.sh", encoding="utf-8").read()
assert "RPC_BIN_NAMES=" in common
# bin_rpc and rpc_comm_re must both read the list, not restate it
assert "find_bin \"$LLAMA_BIN\" $RPC_BIN_NAMES" in common, "bin_rpc has its own list"
assert "for n in $RPC_BIN_NAMES" in common, "rpc_comm_re has its own list"
# and the consumer must not carry a second hand-maintained pattern
# executable lines only -- the comment above the call deliberately keeps the
# history of why `pgrep -x rpc-server` was wrong, and that is worth keeping
code = [l for l in bench.splitlines() if l.strip() and not l.strip().startswith("#")]
assert not any("pgrep" in l for l in code), \
    "50_bench_device.sh still greps for process names itself: %s" % \
    [l for l in code if "pgrep" in l]
assert "rpc_running" in bench
EOPY

head_ "parked cores do not halve the reported CPU utilisation"
python3 - <<'EOPY' && ok "utilisation is averaged over the cores that exist" || bad "parked cores"
import sys
sys.path.insert(0, "telemetry")
from tegrastats_logger import parse
mk = lambda cpu: ("RAM 1/2MB SWAP 0/1MB CPU [%s] EMC_FREQ 0%%@1600 GR3D_FREQ 0%%@76 "
                  "thermal@46.2C POM_5V_IN 1900/1900 POM_5V_GPU 0/0 POM_5V_CPU 238/238" % cpu)
# 10 W: four cores, all saturated
d = parse(mk("99%@1479,99%@1479,99%@1479,99%@1479"))
assert d["cpu_util_pct"] == 99.0 and d["cpu_cores_online"] == 4, d
# 5 W: two cores PARKED and the other two saturated. Counting "off" as 0% busy
# and dividing by 4 reported 49.5% for a fully loaded board.
d = parse(mk("99%@918,99%@918,off,off"))
assert d["cpu_util_pct"] == 99.0, d["cpu_util_pct"]
assert d["cpu_cores_online"] == 2 and d["cpu_cores_seen"] == 4, d
# the power mode stays legible in the row, not only in `raw`
assert d["cpu_util_per_core"] == "99|99|off|off", d["cpu_util_per_core"]
# every core parked: the absence of a measurement, not a measured zero
d = parse(mk("off,off,off,off"))
assert d["cpu_util_pct"] is None and d["cpu_cores_online"] == 0, d
EOPY

head_ "the pre-flight thermal gate reads a real zone, not the PMIC dummy"
TZ="$FIX/tz"
mkz() { mkdir -p "$1/sys/devices/virtual/thermal/thermal_zone$2"
        echo "$3" > "$1/sys/devices/virtual/thermal/thermal_zone$2/type"
        echo "$4" > "$1/sys/devices/virtual/thermal/thermal_zone$2/temp"; }
mkz "$TZ/nano" 0 AO 40500; mkz "$TZ/nano" 1 CPU-therm 38500
mkz "$TZ/nano" 2 GPU-therm 37000; mkz "$TZ/nano" 3 PLL 39000
mkz "$TZ/nano" 4 PMIC-Die 100000; mkz "$TZ/nano" 5 thermal-fan-est 38000
mkz "$TZ/nofan" 0 CPU-therm 41000; mkz "$TZ/nofan" 1 GPU-therm 44000
mkz "$TZ/nofan" 2 PMIC-Die 100000
mkdir -p "$TZ/none"
GOT_NANO="$( . "$ROOT/scripts/lib/common.sh" 2>/dev/null
             HS_SYSFS_ROOT="$TZ/nano" thermal_zone_c )"
GOT_NOFAN="$( . "$ROOT/scripts/lib/common.sh" 2>/dev/null
              HS_SYSFS_ROOT="$TZ/nofan" thermal_zone_c )"
GOT_NONE="$( . "$ROOT/scripts/lib/common.sh" 2>/dev/null
             HS_SYSFS_ROOT="$TZ/none" thermal_zone_c )"
# the OLD code took the max over all zones, i.e. PMIC-Die's constant 100 C, so
# the "let it settle below 45 C" warning fired on a stone-cold board every time
OLD_MAX="$( set +o pipefail
  cat "$TZ"/nano/sys/devices/virtual/thermal/thermal_zone*/temp | sort -n | tail -1 || true )"
[ "$GOT_NANO" = "thermal-fan-est 38" ] \
  && ok "prefers thermal-fan-est (= tegrastats thermal@, = every row's temp_thermal_c)" \
  || bad "zone selection: got '$GOT_NANO'"
[ "$GOT_NOFAN" = "GPU-therm 44" ] \
  && ok "falls back to the hotter of CPU/GPU, never PMIC-Die" \
  || bad "fallback: got '$GOT_NOFAN'"
[ -z "$GOT_NONE" ] && ok "no zones -> empty, and does not abort the caller" \
                   || bad "no-zone case returned '$GOT_NONE'"
[ "$((OLD_MAX/1000))" = "100" ] \
  && ok "(the old max-of-all-zones rule would have said 100 C here)" \
  || bad "test tree does not reproduce the PMIC dummy"

head_ "a second tegrastats aborts the sweep before it starts, not after"
# Two instances cannot both read the INA3221: the second silences the first, so
# the sweep used to complete with every energy column empty. A REAL binary is
# needed -- /proc/<pid>/comm of a #!python script is "python3", not the filename.
mkdir -p "$FIX/realbin"
cp "$(command -v sleep)" "$FIX/realbin/tegrastats"
"$FIX/realbin/tegrastats" 120 & SQUAT=$!
sleep 1
python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 \
  --matrix bench/fixtures/matrix_split.json --tag squat > "$FIX/squat.txt" 2>&1
RC=$?
kill $SQUAT 2>/dev/null || true; wait $SQUAT 2>/dev/null || true
[ "$RC" -ne 0 ] && grep -q "already running" "$FIX/squat.txt" \
  && ok "aborts with a named cause, exit $RC" \
  || { bad "the sweep ran anyway (rc=$RC)"; tail -3 "$FIX/squat.txt"; }
ls "$FIX"/results/*squat_bench_energy.csv > /dev/null 2>&1 \
  && bad "it produced a results CSV despite aborting" \
  || ok "and produced no results CSV to be mistaken for a measurement"
sleep 1
python3 bench/run_matrix.py --env-file "$FIX/node.env" --baseline-s 6 \
  --matrix bench/fixtures/matrix_split.json --tag rate > "$FIX/rate.txt" 2>&1
grep -qE "power logging: [0-9]+ samples, [0-9.]+ Hz achieved" "$FIX/rate.txt" \
  && ok "and reports the ACHIEVED sample rate, which is what bounds the integral" \
  || { bad "achieved rate not reported"; grep -i "power logging" "$FIX/rate.txt"; }

head_ "a cluster run can mark its own power windows"
# In the distributed experiment this node is an RPC worker: it is handed tensors,
# never phases, so nothing here knows when a prefill starts. The orchestrator
# marks over ssh. What must hold is that the result is readable by exactly the
# same commands as a sweep CSV -- otherwise the cluster capture is a second,
# divergent format and the analysis has to special-case the node again.
CAP="$FIX/cap"; mkdir -p "$CAP/results"
cp -a "$ROOT" "$CAP/kit"
cat > "$CAP/kit/config/node.env" <<ENV
NODE_ID=jetson-nano-01
CLUSTER_NODE_ID=jetson
HS_ROOT="$CAP"
RESULTS_DIR="$CAP/results"
MODEL_DIR="$CAP/models"
LLAMA_BIN="$FIX/bin"
TELEMETRY_HZ=5
ENV
CW="$CAP/kit/scripts/45_capture_window.sh"
bash "$CW" start > /dev/null 2>&1 || bad "capture would not start"
bash "$CW" mark prefill > /dev/null 2>&1
echo prefill > "$HS_FAKE_PHASE"; sleep 2
bash "$CW" idle > /dev/null 2>&1; echo idle > "$HS_FAKE_PHASE"; sleep 1
bash "$CW" mark decode > /dev/null 2>&1
echo decode > "$HS_FAKE_PHASE"; sleep 3
CAPCSV="$(bash "$CW" stop 2>/dev/null | tail -1)"
echo idle > "$HS_FAKE_PHASE"
[ -s "$CAPCSV" ] && ok "capture wrote a CSV under the house naming" \
                 || bad "no capture CSV at '$CAPCSV'"
case "$(basename "$CAPCSV")" in
  jetson-nano-01_cluster_power_*.csv) ok "named so 90_collect_results.sh bundles it" ;;
  *) bad "unexpected name: $(basename "$CAPCSV")" ;;
esac
for ph in prefill decode; do
  python3 telemetry/energy_window.py "$CAPCSV" --marker "$ph" > /dev/null 2>&1 \
    || { bad "energy_window could not read the $ph window"; break; }
done && ok "read by the same energy_window commands as a sweep CSV"
python3 - "$CAPCSV" <<'EOPY' && ok "the phases carry different power, as marked" || bad "phase marking"
import csv, collections, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
by = collections.defaultdict(list)
for r in rows:
    m = (r.get("marker") or "").strip()
    try: by[m].append(float(r["pow_in_mw"]))
    except (KeyError, TypeError, ValueError): pass
assert {"idle", "prefill", "decode"} <= set(by), sorted(by)
mean = lambda k: sum(by[k]) / len(by[k])
# the fake draws 7400 mW in prefill, 4800 in decode, 1900 idle -- if the marker
# were not tracking, all three would average the same
assert mean("prefill") > mean("decode") > mean("idle"), \
    {k: round(mean(k)) for k in ("prefill", "decode", "idle")}
EOPY
# a second capture must refuse rather than silently fight for the rail
bash "$CW" start > /dev/null 2>&1
bash "$CW" start > "$CAP/second.txt" 2>&1 \
  && bad "a second capture started anyway" || ok "a second capture is refused"
bash "$CW" stop > /dev/null 2>&1

head_ "every row carries the sampling bound its integral rests on"
python3 - "$FIX" <<'EOPY' && ok "gap, plateau and the bookkeeping check are on the row" || bad "sampling bound"
import csv, glob, sys
rows = [r for r in csv.DictReader(open(glob.glob(sys.argv[1] + "/results/*idle_bench_energy.csv")[0]))
        if r.get("joules")]
assert rows, "no rows with energy"
for r in rows:
    # the longest hole the trapezoid had to cross on one straight line
    g = float(r["sample_gap_max_s"])
    assert 0 < g < 1.0, (r["case_id"], g)
    # how often the rail repeated -- a bound on effective bandwidth, not on aliasing
    pf = float(r["pow_in_plateau_frac"])
    assert 0.0 <= pf <= 1.0, pf
    # running integral vs mean x duration: two arithmetics over the SAME samples,
    # so this catches a bookkeeping error and nothing else
    assert abs(float(r["joules_vs_mean_pct"])) < 10.0, (r["case_id"], r["joules_vs_mean_pct"])
EOPY
python3 - <<'EOPY' && ok "the shipped default rate is 10 Hz and says why" || bad "shipped rate"
src = open("config/node.env.example", encoding="utf-8").read()
assert "TELEMETRY_HZ=10" in src, "shipped default is not 10 Hz"
# and the reason travels with it: repetitions cannot expose an insufficient rate
assert "repetitions cannot expose" in src
EOPY
python3 - <<'EOPY' && ok "the unvalidated-accuracy limit is stated, not implied" || bad "accuracy statement"
import re
# Normalise whitespace before matching. These claims are prose in wrapped
# comments, so a line break inside the phrase is a formatting detail, not a
# missing statement -- asserting on the raw text made this fail the moment the
# paragraph rewrapped. A test that breaks on reflow gets deleted, not read.
flat = lambda t: re.sub(r"\s+", " ", t)
log = flat(open("telemetry/tegrastats_logger.py", encoding="utf-8").read())
doc = flat(open("docs/10_measurement_protocol.md", encoding="utf-8").read())
for src, name in ((log, "tegrastats_logger.py"), (doc, "docs/10")):
    assert "not a second instrument" in src, name
assert "as accurate as that chip and no better" in log
EOPY
python3 - "$FIX" <<'EOPY' && ok "the power CSV is parsed once, not once per record" || bad "_rows memoisation"
import sys, glob, time
sys.path.insert(0, "bench")
import run_matrix as R
p = glob.glob(sys.argv[1] + "/results/*idle_power.csv")[0]
R._ROWS_CACHE.clear()
t0 = time.time(); first = R._rows(p); t_first = time.time() - t0
t0 = time.time()
for _ in range(50):
    again = R._rows(p)
t_cached = (time.time() - t0) / 50
assert again is first, "memoisation is not returning the same object"
# the join makes up to 3 energy_between() calls per record plus the idle scan;
# without this the cost is O(records x rows) and grows with the sample rate,
# which is exactly what #29 raises
assert t_cached < t_first / 5 or t_first < 0.005, (t_first, t_cached)
EOPY

head_ "the cooldown is a closed loop, and drift is measured not assumed"
# A fixed 20 s sleep recovers almost nothing on a fanless Nano (heatsink time
# constant is minutes), so the board ratchets upward across a 2-3 h sweep and
# position in the run order becomes a hidden variable. The gate makes cooldown_s
# a FLOOR and then waits for the board. HS_FAKE_TEMP_RAMP gives the fixture a
# board that actually heats and cools -- with a constant temperature the gate
# passes on its first poll and proves nothing.
rm -f "$FIX"/results/*
cat > "$FIX/hot.json" <<'JSON'
{"defaults": {"repetitions": 3, "threads": 4, "cooldown_s": 1,
              "cooldown_target_c": 3, "max_cooldown_s": 90},
 "cases": [{"id": "hot", "model": "qwen2.5-1.5b-instruct-q4_k_m",
            "n_gpu_layers": 99, "n_prompt": [512], "n_gen": [0]}]}
JSON
HS_FAKE_TEMP_RAMP=2.0 python3 bench/run_matrix.py --env-file "$FIX/node.env" \
  --baseline-s 6 --matrix "$FIX/hot.json" --tag hot > "$FIX/hot.txt" 2>&1
python3 - "$FIX" <<'EOPY' && ok "the gate waits past the floor until the board comes down" || bad "cooldown gate"
import csv, glob, sys
rows = list(csv.DictReader(open(glob.glob(sys.argv[1] + "/results/*hot_bench_energy.csv")[0])))
assert rows, "no rows"
for r in rows:
    # cooldown_s is 1 s here; a gate that engaged waited much longer than that
    assert float(r["cooldown_actual_s"]) > 3.0, (r["case_id"], r["cooldown_actual_s"])
    assert r["cooldown_reached"] == "1", (r["case_id"], r["cooldown_reached"])
    # the BETWEEN-run quantity temp_rise_c structurally cannot express
    assert r["temp_baseline_c"] and r["temp_drift_c"], r
    assert float(r["temp_drift_c"]) > 0, r["temp_drift_c"]
EOPY
# the cap must be honoured, and a capped wait must NOT claim it reached target
cat > "$FIX/capped.json" <<'JSON'
{"defaults": {"repetitions": 3, "threads": 4, "cooldown_s": 1,
              "cooldown_target_c": 0.01, "max_cooldown_s": 6},
 "cases": [{"id": "capped", "model": "qwen2.5-1.5b-instruct-q4_k_m",
            "n_gpu_layers": 99, "n_prompt": [512], "n_gen": [0]}]}
JSON
rm -f "$FIX"/results/*
HS_FAKE_TEMP_RAMP=2.0 python3 bench/run_matrix.py --env-file "$FIX/node.env" \
  --baseline-s 6 --matrix "$FIX/capped.json" --tag capped > "$FIX/capped.txt" 2>&1
python3 - "$FIX" <<'EOPY' && ok "a capped wait reports reached=0, never a gate it did not pass" || bad "cooldown cap"
import csv, glob, sys
r = list(csv.DictReader(open(glob.glob(sys.argv[1] + "/results/*capped_bench_energy.csv")[0])))[0]
assert r["cooldown_reached"] == "0", r["cooldown_reached"]
assert 5.0 <= float(r["cooldown_actual_s"]) <= 20.0, r["cooldown_actual_s"]
EOPY
# with no power logging there is no gate, and the row must not claim one
rm -f "$FIX"/results/*
python3 bench/run_matrix.py --env-file "$FIX/node.env" --no-power \
  --matrix "$FIX/hot.json" --tag nogate > /dev/null 2>&1
python3 - "$FIX" <<'EOPY' && ok "--no-power leaves cooldown_reached blank, not 1" || bad "open-loop claim"
import csv, glob, sys
r = list(csv.DictReader(open(glob.glob(sys.argv[1] + "/results/*nogate_bench_energy.csv")[0])))[0]
assert r["cooldown_reached"] == "", repr(r["cooldown_reached"])
EOPY
python3 - "$FIX" <<'EOPY' && ok "case 1 is repeated last, so drift is a measured pair" || bad "drift control"
import csv, glob, subprocess, sys, os
F = sys.argv[1]
for f in glob.glob(F + "/results/*"):
    os.remove(f)
env = dict(os.environ, HS_FAKE_TEMP_RAMP="2.0")
subprocess.run([sys.executable, "bench/run_matrix.py", "--env-file", F + "/node.env",
                "--baseline-s", "6", "--matrix", "bench/fixtures/matrix_split.json",
                "--tag", "ctrl"], env=env, stdout=subprocess.DEVNULL,
               stderr=subprocess.DEVNULL)
rows = list(csv.DictReader(open(glob.glob(F + "/results/*ctrl_bench_energy.csv")[0])))
ctrl = [r for r in rows if r["drift_control"] == "1"]
assert len(ctrl) == 1, [r["drift_control"] for r in rows]
first = rows[0]
# same case, same phase, hours apart on real hardware: the delta IS drift
assert ctrl[0]["case_id"] == first["case_id"] and ctrl[0]["row_phase"] == first["row_phase"]
assert float(ctrl[0]["temp_drift_c"]) >= float(first["temp_drift_c"])
# and opting out must actually opt out
out = subprocess.run([sys.executable, "bench/run_matrix.py", "--env-file", F + "/node.env",
                      "--baseline-s", "6", "--matrix", F + "/hot.json",
                      "--no-drift-control", "--tag", "noctrl"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
rows = list(csv.DictReader(open(glob.glob(F + "/results/*noctrl_bench_energy.csv")[0])))
# the column must EXIST on every row even when the control is skipped: one that
# appears only when a flag is on is one a consumer cannot rely on
assert all("drift_control" in r for r in rows), list(rows[0])
assert not any(r["drift_control"] == "1" for r in rows), rows
EOPY

head_ "regression baseline"
python3 bench/check_finding3_invariants.py bench/fixtures/BEFORE_mixed_bench_energy.csv \
  > /dev/null 2>&1 && bad "BEFORE baseline should FAIL" || ok "BEFORE baseline still fails"

head_ "the live machine config was not touched"
if [ -f "$REAL_ENV" ]; then ENV_AFTER=$(md5sum < "$REAL_ENV"); else ENV_AFTER=absent; fi
if [ "$ENV_BEFORE" = "$ENV_AFTER" ]; then
  ok "config/node.env unchanged ($ENV_AFTER)"
else
  bad "config/node.env was MODIFIED by this run -- a fixture is writing the real
       path instead of passing --env-file. It is machine config, gitignored, and
       the clobber is silent. was=$ENV_BEFORE now=$ENV_AFTER"
fi

printf '\n\033[1m%d passed, %d failed\033[0m\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
