#!/usr/bin/env python3
"""Fake llama-bench. Banner on STDERR, CSV on STDOUT, a load phase before the
timed loops, and -- the point of this fixture -- TWO CSV rows when both -p and -n
are non-zero, exactly as the real llama-bench does for a mixed invocation."""
import os, sys, time

def opt(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default

n_p, n_g = int(opt("-p", "0")), int(opt("-n", "0"))
pg = opt("-pg", "")
pg_p, pg_g = (int(x) for x in pg.split(",")) if pg else (0, 0)
reps = int(opt("-r", "1"))
model = os.path.basename(opt("-m", "model.gguf"))
ngl, th = opt("-ngl", "0"), opt("-t", "4")
# HS_FAKE_IGNORE_NGL reports ngl=0 whatever was asked -- a CPU-only build, or
# a ref that clamps to the model's layer count. The numbers would then be
# published under the requested ngl, and the GPU-offload result is exactly a
# comparison between two ngl settings.
if os.environ.get("HS_FAKE_IGNORE_NGL"):
    ngl = "0"
phase_file = os.environ["HS_FAKE_PHASE"]
LOAD_S = float(os.environ.get("HS_FAKE_LOAD_S", "3.0"))
PP_NS, TG_NS = 0.5e9, 1.0e9      # per-repetition nanoseconds

def phase(p):
    with open(phase_file, "w", encoding="utf-8") as fh:
        fh.write(p)

for line in ("ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no",
             "ggml_init_cublas: found 1 CUDA devices:",
             "  Device 0: NVIDIA Tegra X1, compute capability 5.3"):
    sys.stderr.write(line + "\n")
sys.stderr.flush()

phase("load"); time.sleep(LOAD_S)                  # SD read + CUDA context

rows = []
if pg_p and pg_g:
    # -pg builds ONE fused instance and emits ONE row -- the whole point of
    # using it for the combined case instead of -p N -n M.
    #
    # Faithful to upstream's shape: the warm-up is a full prompt pass plus
    # test_gen(ctx, 1), i.e. ONE token -- not a full generation -- and then each
    # timed repetition is a whole prompt+generate. So the trailing avg_ns x reps
    # seconds of this window are exactly the timed requests, which is what
    # run_matrix.request_window_s() relies on.
    ns = PP_NS + TG_NS
    phase("prefill"); time.sleep(PP_NS / 1e9)                    # warm-up prompt
    phase("decode");  time.sleep(TG_NS / 1e9 / pg_g)             # warm-up: 1 token
    for _ in range(reps):
        phase("prefill"); time.sleep(PP_NS / 1e9)
        phase("decode");  time.sleep(TG_NS / 1e9)
    rows.append((pg_p, pg_g, "pp%d+tg%d" % (pg_p, pg_g), ns,
                 (pg_p + pg_g) / (ns / 1e9)))
else:
    if n_p:                                        # pp test: warm-up is a full pass
        phase("prefill"); time.sleep((PP_NS * reps + PP_NS) / 1e9)
        rows.append((n_p, 0, "pp%d" % n_p, PP_NS, n_p / (PP_NS / 1e9)))
    if n_g:                                        # tg test: warm-up is one token
        phase("decode"); time.sleep((TG_NS * reps + TG_NS / n_g) / 1e9)
        rows.append((0, n_g, "tg%d" % n_g, TG_NS, n_g / (TG_NS / 1e9)))

# HS_FAKE_EXTRA_ROW simulates a ref that emits a row for the zero-valued axis,
# or otherwise puts two tests in one window despite a single-phase request. It
# is how the driver's last-line-of-defence refusal gets exercised, since
# expand() now blocks the matrix route into that state.
if os.environ.get("HS_FAKE_EXTRA_ROW") and len(rows) == 1:
    if rows[0][0]:
        rows.append((0, 128, "tg128", TG_NS, 128 / (TG_NS / 1e9)))
    else:
        rows.append((512, 0, "pp512", PP_NS, 512 / (PP_NS / 1e9)))

phase("idle")

# HS_FAKE_HIDE_PROMPT simulates a ref that reports a fused row's prompt count in
# a column this driver does not read, leaving n_prompt=0. The row then looks like
# decode while the window's energy covers prompt AND generation -- audit finding
# #3 arriving through the ref instead of through the matrix.
if os.environ.get("HS_FAKE_HIDE_PROMPT"):
    rows = [(0, g, test, ns, tps) for p, g, test, ns, tps in rows]
# HS_FAKE_NO_COUNTS drops the count columns entirely, so phase_rows() can
# identify no working row at all.
DROP_COUNTS = bool(os.environ.get("HS_FAKE_NO_COUNTS"))
# HS_FAKE_WRONG_COUNTS reports a prompt count other than the one that ran: same
# phase, same label, a divisor four times too small against the energy of the
# work actually done. The phase check cannot see this -- only comparing the
# reported counts with the requested ones can.
if os.environ.get("HS_FAKE_WRONG_COUNTS"):
    rows = [(128 if p else p, g, test, ns, tps) for p, g, test, ns, tps in rows]

# Upstream's get_fields() has NO "test" column -- that label is built by the
# markdown printer, not the CSV one. The driver's test_label() reconstructs it
# from the counts, which is why this fixture must not hand it one for free: doing
# so hid the fact that on real hardware the `test` label is DERIVED from the very
# counts the invariant checker compares it against, so that cross-check has no
# independent power there.
hdr = ("build_commit,build_number,cuda,model_filename,model_type,model_size,"
       "model_n_params,n_batch,n_gpu_layers,"
       "n_threads,n_prompt,n_gen,avg_ns,stddev_ns,avg_ts,stddev_ts")
if DROP_COUNTS:
    hdr = hdr.replace("n_prompt,n_gen,", "")
print(",".join('"%s"' % c for c in hdr.split(",")))
for p, g, test, ns, tps in rows:
    cells = ["a1b2c3d", "4321", "1", model, "qwen2 0.5B Q4_K_M",
             "397000000", "494032768", "2048", ngl, th]
    if not DROP_COUNTS:
        cells += [p, g]
    cells += [int(ns), 41000000, "%.2f" % tps, "0.03"]
    print(",".join('"%s"' % c for c in cells))
