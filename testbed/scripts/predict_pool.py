#!/usr/bin/env python3
"""Pre-register the placement engine's predictions for the pool campaign. Reads files only.

    python3 testbed/scripts/predict_pool.py [--dataset testbed/dataset] [--out testbed/dataset/pool_predictions.json]
                                            [--rows char|pool] [--size] [--moe-batch] [--rtt1] [--models k1,k2] [--dry-run]

Every configuration the campaign runs is planned HERE, before any of it runs, and written with a timestamp and
the sha256 of every input and of the engine itself -- so a later measurement is compared with a prediction that
could not have seen it.

Per board, per role, per quantisation, the cost comes from the TinyLlama row at the TARGET model's quantisation
(testbed/dataset/README.md, finding 1):
  decode   ms/token per byte read per token (blocks + output head)
  prefill  ms/token per block parameter
  host     the board running llama.cpp itself: device cpu (Pis), vulkan (Jetson)
  worker   the board as an RPC worker: cpu-rpc-worker rows (loopback, bench_worker_role.sh); the Jetson's Vulkan
           row with the RPC cost the smoke test measured (+1.2 % prefill, +2.5 % decode); a Pi quantisation with no
           worker row uses its host row (the Pis without dotprod never repack: the Pi 4 measured within 1 %)
An assignment is a pipeline [(host, blocks), (worker, blocks), ...]: the host keeps the first blocks and the token
embedding, each worker the next ones, and the last worker the output head (llama.cpp at the pin, -ngl >= 1). Its
predicted time per token is each board's work at its cost, plus per worker two round trips (rtt_trips; RTT1, below) and
the hidden state
(logits back from the last), on that worker's own measured pair with the host (link.json "pairs": every pooled byte
passes through the host; a pool whose host has no measured pair to a member is not planned -- a link is never
borrowed from another pair). Memory is checked with the real block sizes from the tensor table.
"""
import csv, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
from control.placement import DeviceProfile, ModelProfile, plan_even, plan_optimal, memory_need_mb, MB  # noqa: E402

TINY = {"Q4_K_M": "tinyllama-1.1b-q4km", "Q8_0": "tinyllama-1.1b-q8_0", "Q4_0": "tinyllama-1.1b-q4_0",
        "Q3_K_M": "tinyllama-1.1b-q3km", "Q6_K": "tinyllama-1.1b-q6_k"}   # Q6_K: 2026-10-11, for the 30B MoE at Q6_K
HOST_DEV = {"pi5": "cpu", "pi4": "cpu", "jetson": "vulkan", "laptop": "cpu"}
# The laptop as a node (the user, 2026-10-11: "make it as one of the node"). The scripts name it "laptop"; the
# characterisation rows and cluster.json call it t580. Its host rows are the t580 cpu rows (its AVX2 build repacks Q4_K
# as a host). Its WORKER rows are its own files, taken with the Pi 4 as host over the link (an RPC worker never repacks,
# and a host on the laptop itself polls on the worker's cores: laptop_worker_20261011.json's loopback rows were 37 % slow);
# the newest file holding a quantisation wins. Without a worker row at a quantisation the laptop is left out, as the Pi 5.
CSV_NODE = {"laptop": "t580"}
LAPTOP_WORKER_FILES = ["laptop_worker_20261011b.json", "laptop_worker_20261011c.json"]
JETSON_RPC = {"pp": 1.012, "tg": 1.025}          # rpc_smoke_pi5_jetson 2026-09-27: ngl=99 vs local, -1.23 % / -2.49 % t/s
# RTT1 (registered 2026-10-09, testbed/dataset/rule_rtt1_20261009.json): a worker costs ONE network round trip per token at
# the pin, not two. The RPC client sends SET_TENSOR and GRAPH_COMPUTE one-way and waits only on GET_TENSOR
# (ggml-rpc.cpp:945-1047), and the network study measured 0.94 and 1.09 ms of decode per ms of added RTT (NS-RTT,
# 2026-10-08). Applied only with rtt_trips=1 (--rtt1), so that every earlier registration reproduces unchanged.
RTT_TRIPS_DEFAULT = 2
# VISIT (registered 2026-10-09, under test; register_visit_20261009.py): each worker visit costs a fixed ~1.59 ms of decode
# per token beyond RTT1's link terms -- the ring-lap study's 3.173 ms per lap of two visits (a Jetson stage and a Pi 4
# loopback stage), split evenly. Applied only with visit_ms=VISIT_MS, so that every earlier registration reproduces.
VISIT_MS = 1.587
# The Pi 5 host's prefill model-size factor on the characterisation rows: a large Q4_K_M model prefills 8.06 % slower per
# parameter than TinyLlama's row predicts (Mistral-Nemo 12B at pp64, testbed/results/protocol_ab_20260928_174753, condition
# P). Adopted by the registered rule SIZE2 (pool_predictions_20260928c.json): prefill mean |error| 1.08 % against 5.12 % on
# five fresh cells. Applied only with size=True (--size), so that every earlier registration reproduces unchanged.
PI5_HOST_PREFILL_SIZE = {"Q4_K_M": 1.0806}
ENDPOINT = {"pi5": "192.168.10.11:50052", "pi4": "192.168.10.12:50052", "jetson": "192.168.1.114:50052"}
# The laptop as an RPC worker (2026-10-11, the long-context plan): cluster.json's t580, named "laptop" as measure_link.sh and
# serve_defaults.sh name it, on its router-LAN address and the port its draft server uses (50053).
ENDPOINT["laptop"] = "192.168.1.108:50053"   # the Jetson: .100.111 until 2026-10-08, .1.109 until 2026-10-09
OVERHEAD_MB = 350.0                               # the stand-in, for a board role cluster.json does not measure
CLUSTER = os.path.join(ROOT, "ThinkPad", "config", "cluster.json")
MODELS = {  # key -> (shape name, quant, files on the host, n_prompt, n_gen)
    "qwen14b": ("qwen2.5-14b-instruct-q4_k_m", "Q4_K_M", "~/models/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"),
    "tinyllama": ("tinyllama-1.1b-q4km", "Q4_K_M", "~/models/tinyllama-1.1b-q4km.gguf"),
    "mistral7b": ("mistral-7b-q8_0", "Q8_0", "~/models/mistral-7b-q8_0.gguf"),
    # The 2026-09-28 models (testbed/dataset/model_manifest.json); on the boards under these short names.
    "nemo12b": ("mistral-nemo-12b-q4_k_m", "Q4_K_M", "~/models/mistral-nemo-12b-q4_k_m.gguf"),
    "qwen3_14b": ("qwen3-14b-q4_k_m", "Q4_K_M", "~/models/qwen3-14b-q4_k_m.gguf"),
    "qwen3moe": ("qwen3-30b-a3b-q3_k_m", "Q3_K_M", "~/models/qwen3-30b-a3b-q3_k_m.gguf"),
    # 2026-10-11 (user: "Qwen3-30B-A3B at Q6_K"): the same model at Q6_K, 23.4 GiB: too big for the laptop alone with its
    # desktop reserve, so it needs the pool even with the laptop as a node. On the laptop only.
    "qwen3moe_q6k": ("qwen3-30b-a3b-q6_k", "Q6_K", "~/models/qwen3-30b-a3b-q6_k.gguf"),
    # 2026-09-28 night (user: "let's download the 27B and let's benchmark it out"): a qwen35 hybrid, on the Pi 4 only.
    "qwen38_27b": ("qwen3.8-27b-q3_k_m", "Q3_K_M", "~/models/qwen3.8-27b-q3_k_m.gguf"),
    # 2026-09-28 night (user: "ok let's go with Qwen3.5-9B"): the prefill / decode separation model, a qwen35 hybrid.
    "qwen35_9b": ("qwen3.5-9b-q4_k_m", "Q4_K_M", "~/models/qwen3.5-9b-q4_k_m.gguf"),
    # 2026-10-20 (the full evaluation on four nodes, the rest of the model list): TinyLlama at its other quantisations,
    # the characterisation rows' files, on every node under these short names
    "tinyllama_q8_0": ("tinyllama-1.1b-q8_0", "Q8_0", "~/models/tinyllama-1.1b-q8_0.gguf"),
    "tinyllama_q4_0": ("tinyllama-1.1b-q4_0", "Q4_0", "~/models/tinyllama-1.1b-q4_0.gguf"),
    "tinyllama_q3km": ("tinyllama-1.1b-q3km", "Q3_K_M", "~/models/tinyllama-1.1b-q3km.gguf"),
    "tinyllama_q6k": ("tinyllama-1.1b-q6_k", "Q6_K", "~/models/tinyllama-1.1b-q6_k.gguf"),
    # 2026-10-21 (user: "Q6_K, 31.0 GB"): Qwen3.6-35B-A3B, a qwen35moe hybrid that needs every node; on the laptop only.
    "qwen36moe_q6k": ("qwen3.6-35b-a3b-q6_k", "Q6_K", "~/models/qwen3.6-35b-a3b-q6_k.gguf"),
    # 2026-10-21 (user: "Yes, download it"): Qwen3-32B Q8_0, the newest Qwen prima.cpp can run, for prima.cpp + Halda against
    # the engine on all four nodes (the laptop's reserve 7 GiB for these runs only: full_eval.py --reserve). On the laptop only.
    "qwen3_32b_q8": ("qwen3-32b-q8_0", "Q8_0", "~/models/qwen3-32b-q8_0.gguf"),
    # 2026-10-21 (user: "Both files"; scheduler v2, precision-aware placement): the same model's official Q6_K and Q4_K_M,
    # Qwen/Qwen3-32B-GGUF @ 938a7432, on the laptop's data drive (~/models links into /mnt/data/heteroserve-models)
    "qwen3_32b_q6k": ("qwen3-32b-q6_k", "Q6_K", "~/models/qwen3-32b-q6_k.gguf"),
    "qwen3_32b_q4km": ("qwen3-32b-q4_k_m", "Q4_K_M", "~/models/qwen3-32b-q4_k_m.gguf"),
}
# The board that holds each model's file, and so the only one that can run llama.cpp as the host (the storage plan
# in model_manifest.json: the MoE lives on the Pi 4, whose disk has the room).
HOST = {"qwen3moe": "pi4", "qwen38_27b": "pi4", "qwen3moe_q6k": "laptop", "qwen36moe_q6k": "laptop", "qwen3_32b_q8": "laptop",
        "qwen3_32b_q6k": "laptop", "qwen3_32b_q4km": "laptop"}
PP, TG = 64, 16                                   # the pooled protocol (run_14b_pooled.sh)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


ROW_TESTS = {"char": ("pp512", "tg128"),      # the characterisation protocol: every registration before 2026-09-28 night
             "pool": ("pp%d" % PP, "tg%d" % TG)}  # rows taken the way a pooled cell runs (protocol_ab_pi5.sh: the
                                             # characterisation rows mispriced a cell's decode by the probe and length)


def rows(char_csv, which="char"):
    out = {}
    for r in csv.DictReader(open(char_csv)):
        if r["flags"] or "tinyllama" not in r["model"] or r["test"] not in ROW_TESTS[which]:
            continue
        out[(r["node_id"], r["device"], r["quant"], r["phase"])] = float(r["ms_per_token"])
    return out


# Per-model, per-board multipliers on the predicted decode cost per byte, fitted where the model runs (2026-10-24, c35:
# Qwen3.6-35B-A3B Q6_K in five splits on all four nodes, testbed/results/c35_20261024_025725/score.json "k"; registered
# c35_predictions_20261024.json). Fitted on splits A-D, they predicted the held-out split E within 0.04 % and 23 October's
# order35b runs within 1.0 and 2.2 % (C35-HOLDOUT, C35-PAST); without them the predictor is 19.5-21.7 % optimistic on this
# model. Opt-in: plan_host_choice / plan_fixed(node_mult="measured"); None, the default, reproduces every plan before.
NODE_MULT = {"qwen36moe_q6k": {"laptop": 1.1717, "jetson": 1.2782, "pi5": 1.3436, "pi4": 1.1517}}


def apply_node_mult(C, key, node_mult):
    """C with each board's decode cost per byte multiplied by its factor, in both roles. node_mult: "measured" (NODE_MULT
    for this model key, if any), a {node: factor} dict, or None (C itself, unchanged). Prefill is not scaled: the factors
    were fitted on decode. "measured_blocks" (2026-10-28, after def1): the same factors on the BLOCKS only; the output head
    keeps the board's unscaled rate, carried as C[n]["host_head"] / C[n]["worker_head"] and read by predict(). def1's H3
    (the head on the Pi 4) came out 6.3 % below "measured", and -1.6 % with the head unscaled, found after the data."""
    blocks_only = node_mult == "measured_blocks"
    m = NODE_MULT.get(key) if node_mult in ("measured", "measured_blocks") else node_mult
    if not m:
        return C
    out = {}
    for n, c in C.items():
        if n not in m:
            out[n] = c
            continue
        out[n] = {r: (pb * m[n], pp) for r, (pb, pp) in c.items()}
        if blocks_only:
            out[n].update({r + "_head": c[r] for r in ("host", "worker") if r in c})
    return out


# 2026-11-03 (jmoe, testbed/results/jmoe_20261103_195639; the user: "... Measure the Jetson's prefill speed on mixture-of-experts
# models"): the Jetson's GPU prefilling Qwen3-30B-A3B Q6_K's MoE blocks as a worker. Each block moved from the Pi 5 to the
# Jetson saved 111.9 ms of a 512-token prefill (laptop 36 fixed, the Jetson at 1, 3 and 5 blocks), where the planner's rows
# (TinyLlama's Vulkan rows, a dense model) priced the move at 717 ms. With the Pi 5's price kept, the Jetson's prefill per
# parameter is 2.3692x its row on this model; prefill_mult="measured" reads it, None (the default) leaves every plan as before.
PREFILL_MULT = {"qwen3moe_q6k": {"jetson": 2.3692}}


def apply_prefill_mult(C, key, prefill_mult):
    """C with a board's prefill cost per parameter multiplied by its factor in both roles (the head's entries untouched).
    prefill_mult: "measured" (PREFILL_MULT for this model key, if any), a {node: factor} dict, or None (C unchanged)."""
    m = PREFILL_MULT.get(key) if prefill_mult == "measured" else prefill_mult
    if not m:
        return C
    return {n: ({r: ((v[0], v[1] * m[n]) if r in ("host", "worker") else v) for r, v in c.items()} if n in m else c)
            for n, c in C.items()}


def unit(ms_tg, ms_pp, tiny):
    """(ms per byte read per decoded token, ms per block parameter per prompt token)."""
    return ms_tg / (sum(tiny["layer_bytes"]) + tiny["head_bytes"]), ms_pp / (tiny["params_per_layer"] * tiny["n_layers"])


def costs(R, shapes, quant, size=False):
    """{node: {"host": (per_byte, per_param), "worker": (...)}} for one quantisation, with each figure's source."""
    tiny = shapes[TINY[quant]]
    out, src = {}, {}
    for n, dev in HOST_DEV.items():
        cn = CSV_NODE.get(n, n)
        tg, pp = R.get((cn, dev, quant, "tg")), R.get((cn, dev, quant, "pp"))
        if tg is None or pp is None:
            continue
        host = unit(tg, pp, tiny)
        if n == "laptop":
            w = laptop_worker_row(quant)
            if w is None:
                continue              # no measured worker row: no stand-in (its host row is repacked)
            worker, wsrc = unit(w[0]["tg_ms"], w[0]["pp_ms"], tiny), "worker row, %s (Pi 4 host)" % w[1]
        elif n == "jetson":
            worker, wsrc = (host[0] * JETSON_RPC["tg"], host[1] * JETSON_RPC["pp"]), "vulkan row + smoke-test RPC cost"
        elif (n, "cpu-rpc-worker", quant, "tg") in R:
            worker = unit(R[(n, "cpu-rpc-worker", quant, "tg")], R[(n, "cpu-rpc-worker", quant, "pp")], tiny)
            wsrc = "cpu-rpc-worker row"
        else:
            worker, wsrc = host, "host row (no worker row at %s; this board never repacks)" % quant
            if n == "pi5":
                continue              # the Pi 5 does repack: no stand-in for its worker cost
        if size and n == "pi5" and quant in PI5_HOST_PREFILL_SIZE:
            host = (host[0], host[1] * PI5_HOST_PREFILL_SIZE[quant])
            src_note = " x %.4f prefill (SIZE2)" % PI5_HOST_PREFILL_SIZE[quant]
        else:
            src_note = ""
        out[n] = {"host": host, "worker": worker}
        src[n] = {"host": "%s%s row%s" % ("" if cn == n else cn + " ", dev, src_note), "worker": wsrc}
    return out, src


def recurrent_blocks(q):
    """The blocks a hybrid computes as recurrent layers, marked as llama.cpp at the pin marks them (qwen3next.cpp,
    qwen35moe.cpp: block i < n_layer() is recurrent unless (i + 1) % full_attention_interval == 0; the MTP block is
    neither). A shape without full_attention_interval has none."""
    fai = int(q.get("full_attention_interval") or 0)
    main = len(q["layer_bytes"]) - int(q.get("nextn_layers") or 0)
    return {i for i in range(main) if (i + 1) % fai != 0} if fai else set()


ATTENTION = os.path.join(ROOT, "testbed", "dataset", "attention_cost_20261011.json")
# 2026-10-28 (lc4n): the same boards, with the Jetson priced by its own attention op per shape (law "fa_table"). Opt-in:
# attention_table(ATTENTION_FA); the default above stays, so every registration that priced depth reproduces.
ATTENTION_FA = os.path.join(ROOT, "testbed", "dataset", "attention_cost_20261028.json")


def attention_table(path=ATTENTION):
    """The measured attention cost per board (attention_cost_20261011.py writes it): {"boards": {node: {"law", constant}}}."""
    return json.load(open(path))


def fa_curve_us(curve, n_kv):
    """A per-shape curve's time per attention op (us) at n_kv cache cells, linear between its measured cells (past the last,
    the last two cells' line). curve: {cells (str or int): us}."""
    pts = sorted((int(k), v) for k, v in curve.items())
    i = 1
    while i < len(pts) - 1 and pts[i][0] < n_kv:
        i += 1
    (x0, y0), (x1, y1) = pts[i - 1], pts[i]
    return y0 + (y1 - y0) * (n_kv - x0) / (x1 - x0)


def attn_ms(table, node, q, depth=None):
    """ms per block per token of context depth for model shape q on `node`, by the board's measured law:
    "query_heads" (a CPU board: the work grows with the query heads, n_head x head_dim x ns), "kv_bytes" (the Jetson's GPU:
    the KV bytes read, kv_bytes_per_layer_per_token x ns), "fa_table" (2026-10-28, lc4n: the Jetson's GPU by its own
    flash-attention op, timed with no model loaded by fa_bench for each attention shape, head_dim x n_head x n_head_kv: ms
    per attention layer per token of depth over the long range, from 1,280 cache cells; a hybrid's blocks carry its share of
    attention layers, hybrid_attention_fraction). None for a board with no measured law, or a shape the table lacks.
    hybrid_attention_fraction applies to any law that carries it (2026-11-01, attention_cost_20261101.json: every board); the
    tables before carry it on the Jetson's fa_table alone, so every registration that priced depth reproduces."""
    b = table["boards"].get(node)
    if b is None:
        return None
    f = b.get("hybrid_attention_fraction", {}).get(q.get("architecture"), 1.0)
    if b["law"] == "query_heads":
        return q["n_head"] * q["head_dim"] * b["ns_per_qhead_dim"] * 1e-6 * f
    if b["law"] == "kv_bytes":
        return q["kv_bytes_per_layer_per_token"] * b["ns_per_kv_byte"] * 1e-6 * f
    if b["law"] == "fa_table":
        e = b["table"].get("%dx%dx%d" % (q["head_dim"], q["n_head"], q["n_head_kv"]))
        if e is None:
            return None
        return e["ms_per_layer_per_token_long"] * f
    if b["law"] == "fa_curve":
        # 2026-11-04 (dl2): a CPU board by its own attention op per shape, timed with no model loaded at 256-8,192 cache cells
        # (fa_bench_cpu, dl2_fa_cpu_20261104.py). A token at depth d attends over 256 x ceil((d + 1) / 256) cells; per attention
        # layer its depth costs the op's time there less at 256 cells, and the law is that over d (ms per block per token of
        # depth at this depth). It needs the depth; at depth 0 there is no depth cost.
        e = b["shapes"].get("%dx%dx%d" % (q["head_dim"], q["n_head"], q["n_head_kv"]))
        if e is None or depth is None:
            return None
        if depth <= 0:
            return 0.0
        n_kv = 256 * -(-(int(depth) + 1) // 256)
        return (fa_curve_us(e, n_kv) - fa_curve_us(e, 256)) / 1000.0 / depth * f
    raise ValueError("unknown attention law %r for %s" % (b["law"], node))


# 2026-11-01 (the scheduler review's item 3): a hybrid's memory per block, as llama.cpp at the pin allocates it -- a KV cache
# on its full-attention blocks only, a fixed recurrent state on the rest, nothing on its MTP block (hybrid_shapes_20261101.py,
# which reproduces every device's KV and state buffer of the 35B's 28 Oct serve within llama.cpp's rounding). Opt-in:
# plan_host_choice / plan_fixed(hybrid_memory=True); False charges every block a full-attention KV cache, as every plan before.
HYBRID_SHAPES = os.path.join(ROOT, "testbed", "dataset", "hybrid_shapes_20261101.json")


def with_hybrid(q, shape, path=HYBRID_SHAPES):
    """q with its hybrid fields (full_attention_interval, rs_bytes_per_layer) from hybrid_shapes_20261101.json; q itself for
    a shape the file does not list (every dense model)."""
    h = json.load(open(path))["shapes"].get(shape)
    return dict(q, full_attention_interval=h["full_attention_interval"], rs_bytes_per_layer=h["rs_bytes_per_layer"]) if h else q


def state_bytes(q, ctx):
    """Per block, the memory llama.cpp allocates beside its weights at context ctx: a full-attention block's KV cache
    (kv_bytes_per_layer_per_token x ctx), a recurrent block's state (rs_bytes_per_layer, whatever the context), nothing for an
    MTP block. None for a shape without full_attention_interval: every block then pays its KV, as before."""
    if not q.get("full_attention_interval"):
        return None
    rb = recurrent_blocks(q)
    main = len(q["layer_bytes"]) - int(q.get("nextn_layers") or 0)
    kv = q["kv_bytes_per_layer_per_token"] * ctx
    return [0.0 if i >= main else (float(q["rs_bytes_per_layer"]) if i in rb else kv) for i in range(len(q["layer_bytes"]))]


def hybrid_profile(mp, q):
    """The ModelProfile with a hybrid's state folded into its blocks (q's "state_bytes", from state_bytes()): each block's
    real bytes plus its state, and no per-block KV term, so the exact fill and the memory gate charge what llama.cpp
    allocates. The Jetson's weight cap (max_weights_mb) then counts that state too, at most one attention block's KV (64 MiB
    at 32k cells) beyond its weights."""
    import dataclasses
    st = q.get("state_bytes")
    if not st:
        return mp
    lb = tuple(w + s for w, s in zip(q["layer_bytes"], st))
    return dataclasses.replace(mp, layer_bytes=lb, bytes_per_layer=sum(lb) / len(lb), kv_bytes_per_layer_per_token=0.0)


def planner_inputs(q, shape, ctx, hybrid_memory=False, depth=0, attn=None, attn_table=None, nodes=()):
    """(q, attn) as plan_host_choice and plan_fixed price them: with hybrid_memory, q carries its hybrid fields and
    "state_bytes" (hybrid_profile folds them into the model's blocks); with attn_table and a depth, attn is that table's
    law for every node (a node without one is refused: profiles() would multiply the depth by None)."""
    if hybrid_memory:
        q = with_hybrid(q, shape)
        st = state_bytes(q, ctx)
        if st:
            q = dict(q, state_bytes=st)
    if attn_table and depth and attn is None:
        tab = attention_table(attn_table)
        attn = {n: attn_ms(tab, n, q, depth) for n in nodes}
        missing = sorted(n for n, v in attn.items() if v is None)
        if missing:
            raise ValueError("no attention law for %s in %s" % (missing, os.path.basename(attn_table)))
        # 2026-11-03 (attention_bend_20261103.py): a board whose law bends with depth carries bend_d1; its law at this depth is
        # a (1 + depth / d1). Tables without it (every one before attention_cost_20261103.json) price as before.
        attn = {n: v * (1 + depth / tab["boards"][n]["bend_d1"]) if "bend_d1" in tab["boards"][n] else v for n, v in attn.items()}
    return q, attn


def laptop_worker_row(quant, ds=None):
    """(row, file) of the laptop's measured worker row at `quant`: the newest of LAPTOP_WORKER_FILES that has one, or None."""
    ds = ds or os.path.join(ROOT, "testbed", "dataset")
    for f in reversed(LAPTOP_WORKER_FILES):
        p = os.path.join(ds, f)
        if os.path.exists(p):
            r = json.load(open(p))["rows"].get(quant)
            if r:
                return r, f
    return None


def laptop_costs(R, shapes, quant, worker_json):
    """{"host": ..., "worker": ...} for the laptop (cluster.json t580), as costs() gives the boards: the host from the
    characterisation's t580 cpu rows at `quant` (its CPU repacks Q4_K for AVX2), the worker from the measured worker rows in
    `worker_json` (laptop_worker_20261011.json: an RPC worker never repacks)."""
    tiny = shapes[TINY[quant]]
    w = json.load(open(worker_json))["rows"][quant]
    return {"host": unit(R[("t580", "cpu", quant, "tg")], R[("t580", "cpu", quant, "pp")], tiny),
            "worker": unit(w["tg_ms"], w["pp_ms"], tiny)}


# The RPC's own bytes per decoded token, measured with the host's TCP counters (2026-10-24; TinyLlama Q4_0, gigabit):
# rpc4 (testbed/results/rpc4char_20261024_142530), rpc5 (rpc5char_20261024_170616), rpc6 (rpc6char_20261024_171408).
# The host sends EVERY worker its input hidden state plus `base` bytes plus `mask` bytes per cache cell (the cells padded
# to `kv_pad`); the FIRST worker one more hidden state plus `first`; the LAST worker `last` more. Each worker returns its
# output (the hidden state, or the logits with the head on it) plus `back` bytes. On the wire every payload byte carries
# `hdr` per `mss` of headers, plus `hdr` for each of `msgs` small segments a token. With backend sampling on the head's
# worker, `bs_out` bytes come back instead of the logits, `bs_in` more go there, and `bs_msgs` more small segments.
# They do NOT depend on the blocks a worker holds (rpc5: 19,452 B at 6, 11 and 16 blocks) or on the context size (rpc6).
# Registered test: priced from rpc4's bytes, six shaped arms at 10 and 3 Mbit/s came within 1.5-5.8 % (rpc4plan), where the
# predictor without them missed by up to 40.8 %. ASSUMED: other architectures and quantisations send the same messages
# (the mask scales with the cache, not the model), and every worker's mask grows with the cache (rpc6 measured one worker).
# Opt-in: predict(..., rpc_msg=RPC_MSG); None reproduces every plan before.
RPC_MSG = {"base": 1924, "mask": 2, "kv_pad": 256, "first": 314, "last": 318, "back": 8, "hdr": 66, "mss": 1448,
           "msgs": 10.0, "bs_in": 1917, "bs_out": 48, "bs_msgs": 17.4,
           # 2026-10-24 night (rpc7, rpc8): the first worker's extra and the hybrid layers' inputs by architecture
           # (ARCH_TAIL, HYBRID_ARCH below). "first" is one set_tensor's bytes besides its data, as rpc4-rpc6 measured.
           "arch": True, "rs_copy": 4 + 314, "pos_extra": 12}
RPC_MSG_DENSE = {k: v for k, v in RPC_MSG.items() if k not in ("arch", "rs_copy", "pos_extra")}   # the 24 Oct evening term
# The host's last block ends in weightless ops that llama.cpp's scheduler runs on the FIRST worker (ggml-backend.cpp, "expand
# gpu up", from the worker's first op pinned by llama_context::graph_get_cb), so every tensor they read crosses the link:
#   dense (llama, qwen3, qwen35 ...): the FFN output and the residual -> one hidden state more than a later worker;
#   "experts" (qwen3moe, no shared expert): build_moe_ffn's weighting and sum -> the eight experts' outputs before weighting
#       (n_embd x n_expert_used x 4 B) and the router weights (n_expert_used x 4 B) instead of the FFN output;
#   "shexp_gated" (qwen35moe): the routed output, the shared expert's output and its one-value gate -> two hidden states and
#       4 B. Measured: dense on TinyLlama (rpc4-rpc6) and Qwen3-32B (rpc7); shexp_gated fitted on Qwen3.6-35B-A3B (rpc7,
#       found after the data); experts on Qwen3-30B-A3B and the hybrid inputs on Qwen3.5-9B tested blind (rpc8), all to the byte.
ARCH_TAIL = {"qwen3moe": "experts", "qwen35moe": "shexp_gated"}
# qwen35-family layers also read the recurrent state's copy index (one int32 tensor) and IMROPE's four positions a token
# instead of one: rs_copy + pos_extra on every worker. Assumes each worker holds an attention and a recurrent block (true of
# any four consecutive blocks at full_attention_interval 4, and of every plan measured).
HYBRID_ARCH = ("qwen35", "qwen35moe")

# The planner's defaults since 2026-10-28 (def1, def2): what new registrations and interactive serving pass to
# plan_host_choice / plan_fixed, as **PLANNER_V3. order_check (measured by order35b), the RPC term (bytes to the byte,
# rpc4-rpc9b; on slow links rpc4plan) and the measured per-board multipliers where a model has them, on the BLOCKS only
# (NODE_MULT: Qwen3.6-35B-A3B only; c30 found they do not carry to another model, and the mode leaves every other model
# unchanged). def1 (registered): the multipliers on blocks and head predicted two new head-on-host placements within 0.6 %
# but the head-on-the-Pi-4 one 6.3 % slow (D1-V3 failed; the shipped predictor 11.6-20.4 % fast); def2 (registered): on the
# blocks only, three new head-last placements within 1.2-2.0 %, better than blocks-and-head on each. Backend sampling stays
# off: serving samples on the head's board only when SERVE_EXTRA asks for it. The functions' own defaults stay as they were,
# so every registration on file reproduces.
PLANNER_V3 = {"order_check": True, "rpc_msg": RPC_MSG, "node_mult": "measured_blocks"}
# Interactive serving's planner from 2026-11-01 (the scheduler review's items 3 and 4; the user: "do item 1 to 4 then 6-7"):
# PLANNER_V3, a hybrid's memory charged as llama.cpp allocates it (hybrid_memory), and the cost of context depth priced from
# attention_cost_20261101.json, which has a law for every node (the Pi 5's from lc4charb) and the hybrids' attention fraction
# on every board, at the depth the caller passes. serve_pool.sh passes none unless PLAN_DEPTH asks (v4d, 1 Nov: at half the
# context the laptop's query-head law priced the default model's attention 34.5 % high at 4,096 tokens, and the plan it
# chose was 4.0 % slower). At depth 0 attn_table is not read. A harness keeps PLANNER_V3.
ATTENTION_V4 = os.path.join(ROOT, "testbed", "dataset", "attention_cost_20261101.json")
# 2026-11-03 (attention_bend_20261103.py): the laptop's and the Pi 5's laws bend with depth, a (1 + d / d1), fitted on lc4charb's
# curves; the other boards as in ATTENTION_V4. Opt-in: pass attn_table=ATTENTION_V5 with a depth.
ATTENTION_V5 = os.path.join(ROOT, "testbed", "dataset", "attention_cost_20261103.json")
# 2026-11-04 (dl2): the laptop priced per attention shape by its own attention op, timed with no model loaded (law "fa_curve",
# dl2_fa_cpu_20261104.py v6); the other boards as in ATTENTION_V5. Registered against V5 and two fitted laws on three models no
# depth curve had been measured on, it was the only law within 10 % on all three (0.6-4.9 %). Opt-in: attn_table=ATTENTION_V6.
ATTENTION_V6 = os.path.join(ROOT, "testbed", "dataset", "attention_cost_20261104.json")
# 2026-11-04 evening (dl3): the Pis priced per attention shape too, by their own attention op timed on each board (law
# "fa_curve", dl3_fa_pi_20261104.py v7); the laptop as in V6, the Jetson as in V5. Registered against V5 on four benches: within
# 2.4 % on the three dense ones (V5: 6-199 % off), 38 % high on the hybrid Qwen3.5-9B on the Pi 5 (V5: 56 % low). Opt-in:
# attn_table=ATTENTION_V7.
ATTENTION_V7 = os.path.join(ROOT, "testbed", "dataset", "attention_cost_20261105.json")
PLANNER_V4 = dict(PLANNER_V3, hybrid_memory=True, attn_table=ATTENTION_V4)


def first_tail_bytes(q, m):
    """The first worker's extra bytes beyond a later worker's, per decoded token (see ARCH_TAIL)."""
    a, h = q["activation_bytes"], m["first"]
    kind = ARCH_TAIL.get(q.get("architecture")) if m.get("arch") else None
    if kind == "experts":
        k = int(q["expert_used_count"])
        return q["n_embd"] * k * 4 + h + k * 4 + h
    if kind == "shexp_gated":
        return 2 * (a + h) + (4 + h)
    return a + h


def rpc_bytes(q, i, n_workers, out_bytes, depth=0, rpc_msg=None, head_samples=False):
    """(bytes the host sends worker i, bytes it returns, small segments) per decoded token, i = 1..n_workers in ring order;
    out_bytes is what the worker's compute hands back (the hidden state, or the logits when the head is on it). rpc_msg:
    RPC_MSG (the default; by architecture since 2026-10-24 night) or RPC_MSG_DENSE (the evening's term, every model as dense;
    the two differ only for ARCH_TAIL and HYBRID_ARCH models)."""
    m = rpc_msg or RPC_MSG
    cells = max(m["kv_pad"], -(-max(int(depth), 1) // m["kv_pad"]) * m["kv_pad"])
    a = q["activation_bytes"]
    inb = a + m["base"] + m["mask"] * cells + (first_tail_bytes(q, m) if i == 1 else 0) + (m["last"] if i == n_workers else 0)
    if m.get("arch") and q.get("architecture") in HYBRID_ARCH:
        inb += m["rs_copy"] + m["pos_extra"]
    out, msgs = out_bytes + m["back"], m["msgs"]
    if head_samples:
        out, inb, msgs = m["bs_out"], inb + m["bs_in"], msgs + m["bs_msgs"]
    return inb, out, msgs


def predict(assign, q, C, links, recur=None, recur_worker=None, head_at=None, rtt_trips=RTT_TRIPS_DEFAULT, visit_ms=0.0,
            depth=0, attn=None, rpc_msg=None, backend_sampling=False):
    """(decode ms/token, prefill ms/token) for a pipeline [(node, blocks), ...]; head on the last node, or on the node
    at index head_at (prima.cpp's Halda keeps it on the host, index 0: the last worker then returns its hidden state,
    not the logits, and the host computes the head; at the pin, -ot ^output.weight=CPU, anchored). recur (RECUR,
    2026-09-29): {node: {"dec": ms per recurrent block per token, "pre": ms per recurrent block per prompt token}}, added
    for each recurrent block a node holds; None reproduces every earlier registration. recur_worker (RECUR2, 2026-09-29
    evening): the same for a node in the WORKER role, so that recur prices the host alone. A term may carry "dec", "pre"
    or both. The registered rule ROLE-PRE adopted the role-aware PREFILL term; ROLE-DEC did not adopt a decode term
    (pool_predictions_20260929role.json). The adopted form: recurrent_terms(), then pass only each term's "pre".
    rtt_trips: network round trips per worker per token (and per prompt batch); 2 reproduces every registration before
    2026-10-09, 1 is the registered rule RTT1. visit_ms: a fixed decode cost per worker visit per token beyond its link
    terms (the rule VISIT under test, VISIT_MS; register_visit_20261009.py); 0 reproduces every registration before it.
    depth, attn (2026-10-11, the KV term): a decoded token at context depth `depth` also costs attn[node] ms per block it
    computes per token of depth (attn_ms(), from the measured attention_cost table); 0 / None reproduces every
    registration before it. rpc_msg (2026-10-24, RPC_MSG): each worker's link carries the RPC's own bytes too (rpc_bytes(),
    at this depth); backend_sampling: the head's worker samples, so its logits do not cross. None / False reproduce every
    plan before."""
    lb = q.get("layer_active_bytes") or q["layer_bytes"]          # bytes a decoded token reads (MoE: active experts)
    ppl = q.get("active_params_per_layer") or q["params_per_layer"]
    # A multi-token-prediction block (model_shapes nextn_layers: the GGUF's <arch>.nextn_predict_layers) is the last
    # block. It is held in memory but not computed in normal decoding (qwen35.cpp at the pin: the main loop covers
    # n_layer(), not n_layer_all), so it costs no time here; memory() still charges its bytes.
    nx = int(q.get("nextn_layers") or 0)
    main = len(q["layer_bytes"]) - nx
    if nx:
        lb = list(lb[:main]) + [0.0] * nx
    n_vocab = (q["head_params"] - q["n_embd"]) / q["n_embd"]
    dec = pre = 0.0
    start = 0
    last = len(assign) - 1
    h = last if head_at is None else head_at
    for i, (n, k) in enumerate(assign):
        role = "host" if i == 0 else "worker"
        pb, pp = C[n][role]
        dec += pb * sum(lb[start:start + k])
        pre += pp * ppl * max(0, min(start + k, main) - start)   # the blocks it computes
        if i == h:
            dec += C[n].get(role + "_head", (pb, pp))[0] * q["head_bytes"]   # node_mult "measured_blocks": unscaled
            pre += pp * q["head_params"] / PP
        if i > 0:
            bw = links[n]["mbps"] * 1e6 / 8 / 1000                  # bytes per ms
            out = 4 * n_vocab if (i == last and h == last) else q["activation_bytes"]
            if rpc_msg:
                inb, outb, msgs = rpc_bytes(q, i, last, out, depth, rpc_msg, backend_sampling and i == last and h == last)
                wire = (inb + outb) * (1 + rpc_msg["hdr"] / rpc_msg["mss"]) + rpc_msg["hdr"] * msgs
                dec += rtt_trips * links[n]["rtt_ms"] + wire / bw + visit_ms
            else:
                dec += rtt_trips * links[n]["rtt_ms"] + (q["activation_bytes"] + out) / bw + visit_ms
            out_pp = 4 * n_vocab if (i == last and h == last) else PP * q["activation_bytes"]
            pre += (rtt_trips * links[n]["rtt_ms"] + (PP * q["activation_bytes"] + out_pp) / bw) / PP
        if depth and attn:
            dec += depth * attn[n] * max(0, min(start + k, main) - start)
        t = recur_worker if (recur_worker is not None and i > 0) else recur
        if t and n in t:
            nr = len(recurrent_blocks(q) & set(range(start, start + k)))
            dec += t[n].get("dec", 0.0) * nr
            pre += t[n].get("pre", 0.0) * nr
        start += k
    return dec, pre


def recurrent_terms(char_csv, C, q, model_file, boards=("pi5", "pi4")):
    """The per-recurrent-block terms of a qwen35 hybrid, by role (RECUR2, 2026-09-29): for each board, (its unflagged pp64 /
    tg16 row of the model alone - this predictor's price of that board holding every block and the head, in that role, with
    no link) / the recurrent blocks. Host: device cpu. Worker: device cpu-rpc-worker (loopback, so no link cost is in the
    row). Returns ({board: {"dec", "pre"}} as host, the same as worker); a board without both rows is left out."""
    rows = {}
    for r in csv.DictReader(open(char_csv)):
        if r["model"] == model_file and not r["flags"] and r["test"] in ("pp64", "tg16"):
            rows[(r["node_id"], r["device"], r["test"])] = float(r["ms_per_token"])
    rb = recurrent_blocks(q)
    nx = int(q.get("nextn_layers") or 0)
    main = len(q["layer_bytes"]) - nx
    lb = list(q["layer_bytes"][:main]) + [0.0] * nx
    out = ({}, {})
    for b in boards:
        for j, (role, dev) in enumerate((("host", "cpu"), ("worker", "cpu-rpc-worker"))):
            if b not in C or (b, dev, "tg16") not in rows or (b, dev, "pp64") not in rows or not rb:
                continue
            pb, pp = C[b][role]
            d = pb * (sum(lb) + q["head_bytes"])
            p = pp * q["params_per_layer"] * main + pp * q["head_params"] / PP
            out[j][b] = {"dec": (rows[(b, dev, "tg16")] - d) / len(rb), "pre": (rows[(b, dev, "pp64")] - p) / len(rb)}
    return out


def rows_by_test(char_csv):
    """Every unflagged TinyLlama row, keyed (node, device, quant, test) -- test as llama-bench names it (pp1, pp64, tg16...)."""
    out = {}
    for r in csv.DictReader(open(char_csv)):
        if not r["flags"] and "tinyllama" in r["model"]:
            out[(r["node_id"], r["device"], r["quant"], r["test"])] = float(r["ms_per_token"])
    return out


def pp_curve(Rt, shapes, quant, node, role):
    """{prompt length: ms per block parameter per prompt token} for one node in one role, from every TinyLlama pp row
    at that quantisation: the host's own device row (cpu, or the Jetson's vulkan); as a worker, a Pi's cpu-rpc-worker
    row where it was measured at that length, else its host row (the Pis without dotprod never repack), and the
    Jetson's vulkan row with the smoke test's RPC cost -- the same sources costs() uses."""
    tiny = shapes[TINY[quant]]
    per = float(tiny["params_per_layer"] * tiny["n_layers"])
    dev = HOST_DEV[node]
    cur = {}
    for (n, d, qn, test), ms in Rt.items():
        if n != node or qn != quant or not test.startswith("pp"):
            continue
        x = int(test[2:])
        if d == dev and (role == "host" or node == "jetson" or x not in cur):
            cur[x] = ms / per * (JETSON_RPC["pp"] if (role == "worker" and node == "jetson") else 1.0)
        if role == "worker" and d == "cpu-rpc-worker":
            cur[x] = ms / per                       # a measured worker row wins over the host row at that length
    return cur


def rate_at(curve, x):
    """The curve's rate at prompt length x: log-log interpolation between measured lengths, flat beyond them."""
    import math
    xs = sorted(curve)
    if not xs:
        raise KeyError("no pp rows")
    if x <= xs[0]:
        return curve[xs[0]]
    if x >= xs[-1]:
        return curve[xs[-1]]
    for a, b in zip(xs, xs[1:]):
        if a <= x <= b:
            t = (math.log(x) - math.log(a)) / (math.log(b) - math.log(a))
            return math.exp(math.log(curve[a]) + t * (math.log(curve[b]) - math.log(curve[a])))


def moe_prefill(assign, q, curves, links, n, batch_aware=True, rtt_trips=RTT_TRIPS_DEFAULT):
    """Prefill ms per prompt token of an n-token prompt for a pipeline [(node, blocks), ...], head on the last node.
    A layer's dense parameters (attention, norms, the router) run at batch n. Its active expert parameters run, with
    batch_aware, at the batch each routed expert actually sees -- n x expert_used_count / expert_count tokens (about 4
    of a 64-token prompt on a 128-expert, 8-active model) -- and otherwise at batch n: the active-parameter pricing
    predict() uses. curves: {node: {"host": pp_curve, "worker": pp_curve}}. Links as in predict()."""
    E, k = q["expert_count"], q["expert_used_count"]
    X = (q["params_per_layer"] - q["active_params_per_layer"]) / (1.0 - float(k) / E)   # every routed expert's params
    dense, act = q["params_per_layer"] - X, X * float(k) / E
    b = max(1.0, n * float(k) / E) if batch_aware else float(n)
    n_vocab = (q["head_params"] - q["n_embd"]) / q["n_embd"]
    pre, last = 0.0, len(assign) - 1
    for i, (node, blocks) in enumerate(assign):
        cv = curves[node]["host" if i == 0 else "worker"]
        pre += blocks * (dense * rate_at(cv, n) + act * rate_at(cv, b))
        if i == last:
            pre += rate_at(cv, n) * q["head_params"] / n
        if i > 0:
            bw = links[node]["mbps"] * 1e6 / 8 / 1000
            out_pp = 4 * n_vocab if i == last else n * q["activation_bytes"]
            pre += (rtt_trips * links[node]["rtt_ms"] + (n * q["activation_bytes"] + out_pp) / bw) / n
    return pre


def link_view(pairs, host, members):
    """Per-node link figures for a pool `host` hosts: each other member gets its own measured pair with the host, and
    the host the best figures among all its measured pairs, so placement.hop_ms (the slower endpoint) reproduces each
    pair's own figure. For the Pi 5 this is link.json's "nodes" exactly. Returns (view, pairs that were never
    measured)."""
    def pair(a, b):
        k = "%s-%s" % (a, b) if "%s-%s" % (a, b) in pairs else "%s-%s" % (b, a)
        return k, pairs.get(k)
    own = [r for k, r in pairs.items() if host in k.split("-")]
    # a host with no measured pair (the laptop before its links were measured, 2026-10-11) can still plan alone: its own
    # figures then enter no hop, and any other member is reported missing below
    view = {host: {"mbps": max(r["mbps"] for r in own), "rtt_ms": min(r["rtt_ms"] for r in own), "pair": "host"} if own
            else {"mbps": 1000.0, "rtt_ms": 0.0, "pair": "host (no measured pair: plans alone only)"}}
    missing = []
    for n in members:
        if n != host:
            k, r = pair(host, n)
            if r is None:
                missing.append("%s <-> %s" % (host, n))
            else:
                view[n] = {"mbps": r["mbps"], "rtt_ms": r["rtt_ms"], "pair": k}
    return view, missing


def board_memory():
    """Each board's measured memory overheads by role, whether it repays received tensors, and the reserve it must
    keep free, from cluster.json (memory_overhead_mb, worker_repays_received_tensor: validated out of sample
    2026-09-28; memory_reserve_mb: the Jetson's, after the MoE cell's Vulkan allocation failed, 2026-09-28), and the most
    MiB of weights it may hold (max_weights_mb: the Jetson's, registered rule GPU_CAP, 2026-09-29)."""
    nodes = {n["id"]: n for n in json.load(open(CLUSTER))["nodes"]}
    nodes.update({n["planner_id"]: n for n in list(nodes.values()) if n.get("planner_id")})   # t580 is "laptop" (2026-10-11)
    over = reserve_overrides()
    unknown = [k for k in over if k not in nodes]
    if unknown:
        raise ValueError("HETEROSERVE_RESERVE_MB names no node: %s" % unknown)
    over = {nid: over[k] for k in over for nid, n in nodes.items() if n is nodes[k]}        # by id and planner_id alike
    return {nid: ((n.get("memory_overhead_mb") or {}).get("host"), (n.get("memory_overhead_mb") or {}).get("worker"),
                  bool(n.get("worker_repays_received_tensor", False)),
                  over.get(nid, float(n.get("memory_reserve_mb") or 0)),
                  bool(n.get("host_weights_mapped", False)),
                  float(n["max_weights_mb"]) if n.get("max_weights_mb") is not None else None)
            for nid, n in nodes.items()}


def reserve_overrides():
    """HETEROSERVE_RESERVE_MB="laptop=7168[,node=MiB...]": a registration's own reserve for the named nodes, in place of
    cluster.json's memory_reserve_mb (2026-10-21: the user's 7 GiB laptop desktop reserve for the Qwen3-32B runs only; every
    other run keeps cluster.json's 8 GiB). full_eval.py register records it and run sets it for itself and serve_pool.sh.
    Unset or empty: cluster.json's figures stand."""
    out = {}
    for part in filter(None, (x.strip() for x in os.environ.get("HETEROSERVE_RESERVE_MB", "").split(","))):
        node, _, mib = part.partition("=")
        if not node.strip() or not mib.strip():
            raise ValueError("HETEROSERVE_RESERVE_MB: %r is not node=MiB" % part)
        out[node.strip()] = float(mib)
    return out


def memory(assign, q, mem, devs, mp, head_at=None):
    """Per node: MiB the assignment needs in its role -- real block sizes from the tensor table, charged by the
    engine's own accounting (placement.memory_need_mb: board-aware where cluster.json measures the board, the
    stand-in otherwise), so the plan and the run_pool.sh memory gate agree -- the reserve the board must keep free
    on top (run_pool.sh gates on need + reserve), and the margin left after both. A q carrying "state_bytes" (hybrid_memory,
    2026-11-01) charges each block's own state, with mp's per-block KV term then 0 (hybrid_profile)."""
    lb, out, start = q["layer_bytes"], {}, 0
    st = q.get("state_bytes")
    h = len(assign) - 1 if head_at is None else head_at          # the node holding the output head
    for i, (n, k) in enumerate(assign):
        host = i == 0
        held = sum(lb[start:start + k]) + (sum(st[start:start + k]) if st else 0.0)
        need = memory_need_mb(devs[n], mp, held, k, holds_head=(i == h),
                              holds_embd=host, role=("host" if host else "worker"), uploads=(host and len(assign) > 1))
        avail, reserve = mem[n]["mem_available_mib"], devs[n].memory_reserve_mb
        out[n] = {"blocks": k, "need_mib": round(need), "reserve_mib": round(reserve), "available_mib": avail,
                  "margin_mib": round(avail - need - reserve)}
        cap = devs[n].max_weights_mb
        if cap is not None:          # real block sizes here; the planner's memory_cap prices the average block
            w = (sum(lb[start:start + k]) + (q["head_bytes"] if i == h else 0.0)) / MB
            out[n].update({"weights_mib": round(w, 1), "weights_cap_mib": round(cap), "weights_margin_mib": round(cap - w)})
        start += k
    return out


def planned_buffers(q, assign, head_node, mtp=False):
    """{node: bytes} of each node's model buffer as llama.cpp loads the plan: its blocks, and the output head on head_node
    (serve_pool.sh's split check). At the pin a multi-token-prediction (nextn) block is skipped unless MTP speculation is
    asked for (2026-10-21: the board holding the last blocks reported exactly that block less); with --spec-type draft-mtp
    it is loaded (load_mtp; qwen35*.cpp's mtp_flags) on the device of the last layer, the node holding the last blocks
    (2026-11-02, item 10). Every value None when the shapes carry no per-layer table."""
    if not q.get("layer_bytes"):
        return {n: None for n, _ in assign}
    main = len(q["layer_bytes"]) - (0 if mtp else int(q.get("nextn_layers") or 0))
    out, start = {}, 0
    for n, k in assign:
        out[n] = sum(q["layer_bytes"][start:min(start + k, main)]) + (q["head_bytes"] if n == head_node else 0)
        start += k
    return out


def bench_args(assign, key, head_at=None, load_mode=None):
    """The llama-bench flags that realise the pipeline at the pin: -ngl counts the output layer; -ts puts each
    worker's boundary half a layer inside its share so float rounding cannot move a block.
    head_at=0 (the output head on the host, as Halda places it): -ot "^output\\.weight=CPU" overrides the output weight
    onto the host's CPU buffer. -ngl still counts the output layer, so every block lands as before, and llama.cpp's
    scheduler runs the head matmul where its weight is. The pattern is anchored because llama.cpp matches it with
    std::regex_search. An override to CPU still goes through select_weight_buft over the CPU's extra buffer types
    (llama-model-loader.cpp 1240-1242 at the pin), so a repacking host gets its repacked layout. None (the default)
    leaves every registration made before 2026-10-07 unchanged."""
    if load_mode:      # 2026-10-08 (the load-mode A/B): new Pi 4-hosted registrations pass "none", the weights in memory
        return dict(bench_args(assign, key, head_at), args=bench_args(assign, key, head_at)["args"] + " -lm %s" % load_mode)
    workers = assign[1:]
    if not workers:
        # one board: -ngl 0 keeps a CPU board on its CPU; a GPU board (the Jetson, priced from its Vulkan row) takes every
        # layer and the head onto its GPU, -ngl 99 (2026-10-08: the host choice can now pick it alone)
        gpu = HOST_DEV.get(assign[0][0]) not in (None, "cpu")
        return {"host": assign[0][0], "args": "-m %s -ngl %d" % (MODELS[key][2], 99 if gpu else 0)}
    n = [k for _, k in workers]
    ngl = sum(n) + 1
    args = "-m %s -rpc %s -ngl %d" % (MODELS[key][2], ",".join(ENDPOINT[w] for w, _ in workers), ngl)
    if len(n) > 1:
        ts = [n[0] - 0.5] + n[1:-1] + [n[-1] + 1.5]
        args += " -ts " + "/".join(("%g" % t) for t in ts)
    if head_at == 0:
        # ANCHORED: llama.cpp matches -ot with std::regex_search (llama-model-loader.cpp:1239 at the pin), so an
        # unanchored "output\\.weight" also matches every blk.N.attn_output.weight -- 2026-10-07's first head-on-host
        # cells moved each block's attention projection to the host that way (pool_predictions_20261007h2h.json).
        args += ' -ot "^output\\.weight=CPU"'
    return {"host": assign[0][0], "args": args}


def profiles(C, q, mem, links, nodes, power=None, depth=0, attn=None):
    # A mixture of experts decodes from its active experts, and computes its prompt with them: its per-layer cost
    # is priced from active bytes and parameters (gguf.shape); a dense model's active figures are its own.
    # power (2026-10-08): {node: {"tg": W, "pp": W, "idle": W}}, the strip's measured wall power per board
    # (energy_model.power_table()), for energy-weighted planning; None leaves every profile unpowered, as before.
    bpl = q.get("active_bytes_per_layer") or q["bytes_per_layer"]
    ppl = q.get("active_params_per_layer") or q["params_per_layer"]
    # depth, attn (2026-10-11, the KV term): each block's decode cost also carries depth x attn[node] ms (attn_ms()), in
    # both roles, so the planner sees the attention a long context costs on each board; 0 / None leaves every profile as
    # before. A board cluster.json does not describe (the laptop as a worker) gets no measured overheads (the stand-in).
    bm = board_memory()
    nobm = (None, None, False, 0.0, False, None)
    pw = lambda n, k: power[n][k] if power and n in power else None
    ka = lambda n: depth * attn[n] if (depth and attn) else 0.0
    bm = {n: bm.get(n, nobm) for n in nodes}
    return {n: DeviceProfile(n, C[n]["host"][0] * bpl + ka(n), C[n]["host"][1] * ppl, mem[n]["mem_available_mib"],
                             link_mbps=links[n]["mbps"], rtt_ms=links[n]["rtt_ms"], is_straggler=(n == "pi4"),
                             ms_per_layer_decode_worker=C[n]["worker"][0] * bpl + ka(n),
                             ms_per_layer_prefill_worker=C[n]["worker"][1] * ppl,
                             host_overhead_mb=bm[n][0], worker_overhead_mb=bm[n][1],
                             worker_repays_received=bm[n][2], memory_reserve_mb=bm[n][3],
                             host_weights_mapped=bm[n][4], max_weights_mb=bm[n][5],
                             power_active_w=pw(n, "tg"), power_idle_w=pw(n, "idle"),
                             power_active_prefill_w=pw(n, "pp")) for n in nodes}


def newest_inventory(ds=None):
    """The newest testbed/dataset/model_inventory_*.json (model_inventory.py), or None."""
    import glob
    fs = sorted(glob.glob(os.path.join(ds or os.path.join(ROOT, "testbed", "dataset"), "model_inventory_*.json")))
    return json.load(open(fs[-1])) if fs else None


def holders(key, inventory=None):
    """The boards holding model `key`'s file in the inventory (newest_inventory() by default); None without one."""
    inv = inventory if inventory is not None else newest_inventory()
    if not inv:
        return None
    return list(inv.get("holders", {}).get(key, []))


def objective_ms(phase, dec, pre):
    """The time a plan is chosen by: decode ms per token, prefill ms per prompt token, or, for a request workload
    {"prefill": P, "decode": G}, the request's whole time P x prefill + G x decode (2026-11-02, the scheduler review's item 9:
    plan_host_choice and best_by_predictor scored a workload on its prefill alone, as if it were "prefill"). Prefill is
    predict()'s per-token figure, linear in the prompt, as the planner prices it."""
    if isinstance(phase, dict):
        return phase.get("prefill", 0) * pre + phase.get("decode", 0) * dec
    return dec if phase == "decode" else pre


def plan_host_choice(key, nodes, ctx, mem, host=None, inventory=None, ds=None, phase="decode",
                     rtt_trips=RTT_TRIPS_DEFAULT, load_mode=None, depth=0, attn=None, extra_costs=None, extra_pairs=None,
                     order_check=False, node_mult=None, rpc_msg=None, backend_sampling=False, hybrid_memory=False,
                     attn_table=None, headroom_mb=None, prefill_mult=None):
    """The plan for model `key` on `nodes`, with the HOST chosen too (2026-10-08). Candidates: `host` if given; else every
    board in `nodes` holding the model's file (holders(), from the newest model inventory); else HOST's default, as before.
    plan_optimal plans each candidate as host (with that host's measured links), and the lowest predicted per-token time in
    `phase` wins. A GPU board hosts only a plan on itself alone: with RPC workers its own GPU and theirs need a device split
    that bench_args does not write, so such a plan is not a candidate (reported in "considered").
    load_mode "none" (2026-10-09): the host reads its weights into memory, as serve_pool.sh serves, and is charged its
    token embedding (placement.ModelProfile.host_loads_weights); None prices the mapped launch, as every registration on
    file did.
    depth, attn (2026-10-11): plan and price decode at context depth `depth` with the per-board attention costs attn
    ({node: ms per block per token of depth}, attn_ms()). extra_costs ({node: {"host", "worker"}}, e.g. laptop_costs()) and
    extra_pairs ({"a-b": link}) add a board and its measured links that the dataset files do not hold. All None / 0
    reproduce every plan before.
    Returns {"host", "assign", "head_at", "plan", "links", "devices", "model", "q", "C", "decode_ms", "prefill_ms",
    "considered": [(host, summary)]}, or raises ValueError when no candidate gives a feasible plan.
    order_check (2026-10-23): plan_optimal chooses the order, the head and the layers by its own linear objective, and
    predict() then prices that one plan; with order_check the candidate is instead the fastest plan BY predict() over every
    order of every subset that starts at the host, both head positions, with plan_optimal's exact fill (best_by_predictor).
    order_audit_20261023.py found one model where the two differ (Qwen3.6-35B-A3B, the pick 4.7 % slower by the
    predictor); order35b measured the predictor's order 3.8 % faster (P/F 1.038, testbed/results/order35_20261023_154835).
    False (the default) reproduces every plan before. The result then carries "order_check": what it replaced, or "kept".
    node_mult (2026-10-24): "measured" prices the model with its fitted per-board decode multipliers (NODE_MULT, from c35),
    or a {node: factor} dict; None (the default) reproduces every plan before. The result's "C" is the scaled table.
    rpc_msg, backend_sampling (2026-10-25): every plan is priced with the RPC term (RPC_MSG) and, for a head-last plan, with
    its board sampling (the logits do not cross); predict()'s options. The head's position follows them only with
    order_check (best_by_predictor prices both positions by predict(); plan_optimal's own objective does not know them).
    None / False reproduce every plan before.
    hybrid_memory, attn_table (2026-11-01, PLANNER_V4): a hybrid's blocks charged the KV or recurrent state llama.cpp gives
    each (hybrid_profile; the result's "q" carries "state_bytes", which memory() reads), and with a depth, attn from that
    table for every node (planner_inputs). headroom_mb: MiB every board keeps free on top of its reserve, a number or
    {node: MiB}, for plan_optimal and the order check; None keeps placement.EXACT_FILL_HEADROOM_MB. False / None reproduce
    every plan before."""
    import dataclasses
    ds = ds or os.path.join(ROOT, "testbed", "dataset")
    shapes = json.load(open(os.path.join(ds, "model_shapes.json")))
    pairs = json.load(open(os.path.join(ds, "link.json")))["pairs"]
    R = rows(os.path.join(ds, "characterisation.csv"), "char")
    shape, quant, _ = MODELS[key]
    q = shapes[shape]
    C, _ = costs(R, shapes, quant)
    if extra_costs:
        C.update(extra_costs)
    C = apply_prefill_mult(apply_node_mult(C, key, node_mult), key, prefill_mult)   # prefill_mult: jmoe (2026-11-03)
    if extra_pairs:
        pairs = dict(pairs, **extra_pairs)
    nodes = [n for n in nodes if n in C]
    q, attn = planner_inputs(q, shape, ctx, hybrid_memory, depth, attn, attn_table, nodes)
    if host:
        cands = [host]
    else:
        h = holders(key, inventory)
        cands = [n for n in (h or []) if n in nodes] or [HOST.get(key, "pi5")]
    best, considered = None, []
    for hst in cands:
        if hst not in nodes:
            considered.append((hst, "not among the boards %s" % nodes))
            continue
        gpu = HOST_DEV.get(hst) not in (None, "cpu")
        members = [hst] if gpu else nodes
        L, missing = link_view(pairs, hst, members)
        if missing:
            considered.append((hst, "no measured link %s" % missing))
            continue
        devs = profiles(C, q, mem, L, members, depth=depth, attn=attn)
        mp = hybrid_profile(dataclasses.replace(model_profile(q, shape, load_mode), context_len=ctx), q)
        plan = plan_optimal(devs, mp, phase, host_candidates=[hst], headroom_mb=headroom_mb)
        if not plan.feasible:
            considered.append((hst, "infeasible: %s" % plan.reason[:80]))
            continue
        assign = [(n, plan.layers[n]) for n in plan.order if plan.layers.get(n, 0) > 0]
        head_at = 0 if plan.head_node and plan.head_node == assign[0][0] and len(assign) > 1 else None
        dec, pre = predict(assign, q, C, L, head_at=head_at, rtt_trips=rtt_trips, depth=depth, attn=attn, rpc_msg=rpc_msg,
                           backend_sampling=backend_sampling)
        oc = None
        if order_check:
            alt = best_by_predictor(hst, devs, mp, q, C, L, phase, rtt_trips, depth, attn, rpc_msg, backend_sampling,
                                    headroom_mb=headroom_mb)
            if alt is not None and alt["t"] < objective_ms(phase, dec, pre) - 1e-9:
                oc = {"replaced": [list(x) for x in assign], "head_at": head_at, "decode_ms": round(dec, 1),
                      "prefill_ms": round(pre, 1)}
                assign, head_at, dec, pre = alt["assign"], alt["head_at"], alt["decode_ms"], alt["prefill_ms"]
            else:
                oc = "kept"
        t = objective_ms(phase, dec, pre)
        considered.append((hst, "%s: decode %.1f ms, prefill %.1f ms" % (" + ".join("%s %d" % a for a in assign), dec, pre)))
        if best is None or t < best["t"]:
            best = {"t": t, "host": hst, "assign": assign, "head_at": head_at, "plan": plan, "links": L, "devices": devs,
                    "model": mp, "q": q, "C": C, "decode_ms": dec, "prefill_ms": pre, "rtt_trips": rtt_trips,
                    **({"order_check": oc} if order_check else {})}
    if best is None:
        raise ValueError("no host gives a feasible plan for %s on %s: %s" % (key, nodes, considered))
    best["considered"] = considered
    return best


# Scheduler v2, precision as a plan variable (2026-10-21). A model's official quantisations of the same weights, each a MODELS
# key, with its bits per weight (Q8_0 8.5, Q6_K 6.5625, Q4_K_M as the file averages). The user's quality rule (2026-10-21
# ~23:40, "near-lossless only"): a plan may serve Q6_K or better, MIN_BITS = 6.5; the lower variants are priced and reported,
# never chosen, unless a caller lowers min_bits on purpose.
VARIANTS = {"qwen3_32b": [("qwen3_32b_q8", 8.5), ("qwen3_32b_q6k", 6.5625), ("qwen3_32b_q4km", 4.85)]}
MIN_BITS = 6.5


def variant_family(key):
    """The VARIANTS family a MODELS key belongs to, or None."""
    for fam, vs in VARIANTS.items():
        if any(k == key for k, _ in vs):
            return fam
    return None


def best_by_predictor(host, devs, mp, q, C, L, phase="decode", rtt_trips=RTT_TRIPS_DEFAULT, depth=0, attn=None, rpc_msg=None,
                      backend_sampling=False, headroom_mb=None):
    """The fastest plan by predict() over every order of every subset of `devs` that starts at `host`, both head positions,
    every listed node holding at least one block, with plan_optimal's exact fill and budgets (placement._fill_exact on
    with_headroom(devs, EXACT_FILL_HEADROOM_MB)) -- order_audit_20261023.py's enumeration, as plan_host_choice(order_check=True)
    uses it (2026-10-23). {"t", "assign", "head_at", "decode_ms", "prefill_ms"}, or None when nothing fits or the model has no
    block sizes (the average fill is not covered). rpc_msg and backend_sampling (2026-10-25) price every plan as predict() does
    with them, so a head-last plan whose board samples is priced without its logits crossing. headroom_mb (2026-11-01): a
    number or {node: MiB} in place of placement.EXACT_FILL_HEADROOM_MB, as plan_optimal takes it; None keeps that."""
    import itertools
    from control import placement as PL
    if not mp.layer_bytes:
        return None
    hd = PL.with_headroom(devs, PL.EXACT_FILL_HEADROOM_MB if headroom_mb is None else headroom_mb)
    others = [n for n in devs if n != host]
    best = None
    for size in range(0, len(others) + 1):
        for sub in itertools.combinations(others, size):
            for perm in itertools.permutations(sub):
                order = [host] + list(perm)
                for hn in ([None, host] if len(order) > 1 else [None]):
                    layers, _ = PL._fill_exact(order, hd, mp, phase, 0.0, 0.0, head_node=hn)
                    if layers is None:
                        continue
                    assign = [(n, layers[n]) for n in order if layers.get(n, 0) > 0]
                    if len(assign) < len(order):
                        continue
                    head_at = 0 if hn and len(assign) > 1 else None
                    dec, pre = predict(assign, q, C, L, head_at=head_at, rtt_trips=rtt_trips, depth=depth, attn=attn,
                                       rpc_msg=rpc_msg, backend_sampling=backend_sampling)
                    t = objective_ms(phase, dec, pre)
                    if best is None or t < best["t"]:
                        best = {"t": t, "assign": assign, "head_at": head_at, "decode_ms": dec, "prefill_ms": pre}
    return best


def plan_precision(family, nodes, ctx, mem, min_bits=MIN_BITS, **kw):
    """The fastest plan over a family's variants: plan_host_choice for every variant with at least min_bits, the lowest
    predicted decode wins (ties to the more precise). Returns (best, considered): best = {"variant", "bits", "plan"}; each
    considered entry carries its plan or why it is infeasible, and the variants below min_bits are priced too, marked
    "below_floor", so a report can say what the rule costs. Raises ValueError when no allowed variant has a feasible plan."""
    out = []
    for key, bits in sorted(VARIANTS[family], key=lambda x: -x[1]):
        e = {"variant": key, "bits": bits, "below_floor": bits < min_bits}
        try:
            e["plan"] = plan_host_choice(key, nodes, ctx, mem, **kw)
        except (ValueError, KeyError) as ex:
            e["infeasible"] = str(ex)[:300]
        out.append(e)
    ok = [e for e in out if "plan" in e and not e["below_floor"]]
    if not ok:
        raise ValueError("no variant of %s with >= %.2f bits has a feasible plan on %s: %s" % (
            family, min_bits, nodes, [(e["variant"], e.get("infeasible", "below the floor")) for e in out]))
    best = min(ok, key=lambda e: (e["plan"]["decode_ms"], -e["bits"]))
    return best, out


def plan_fixed(key, assign, head, ctx, mem, ds=None, rtt_trips=RTT_TRIPS_DEFAULT, load_mode=None, depth=0, attn=None,
               extra_costs=None, extra_pairs=None, node_mult=None, rpc_msg=None, backend_sampling=False, hybrid_memory=False,
               attn_table=None, order_check=None, headroom_mb=None, prefill_mult=None):
    """A GIVEN placement, priced the way plan_host_choice prices its pick (the full evaluation serves several placements of
    one model, 2026-10-08): assign = [(node, blocks), ...], the first board the host; head = "host" or "last". Returns the
    same dict as plan_host_choice ("plan" None, "considered" naming the placement); load_mode, depth, attn, extra_costs,
    extra_pairs, node_mult, rpc_msg, backend_sampling, hybrid_memory and attn_table as there. order_check and headroom_mb
    choose nothing for a given placement: they are accepted so that **PLANNER_V3 / **PLANNER_V4 can be passed here too."""
    import dataclasses
    ds = ds or os.path.join(ROOT, "testbed", "dataset")
    shapes = json.load(open(os.path.join(ds, "model_shapes.json")))
    pairs = json.load(open(os.path.join(ds, "link.json")))["pairs"]
    R = rows(os.path.join(ds, "characterisation.csv"), "char")
    shape, quant, _ = MODELS[key]
    q = shapes[shape]
    C, _ = costs(R, shapes, quant)
    if extra_costs:
        C.update(extra_costs)
    C = apply_prefill_mult(apply_node_mult(C, key, node_mult), key, prefill_mult)   # prefill_mult: jmoe (2026-11-03)
    if extra_pairs:
        pairs = dict(pairs, **extra_pairs)
    if sum(k for _, k in assign) != q["n_layers"]:
        raise ValueError("the placement holds %d blocks, the model has %d" % (sum(k for _, k in assign), q["n_layers"]))
    if head not in ("host", "last"):
        raise ValueError("head must be 'host' or 'last', not %r" % (head,))
    members = [n for n, _ in assign]
    if HOST_DEV.get(members[0]) not in (None, "cpu") and len(members) > 1:
        raise ValueError("a GPU board hosts only a plan on itself alone (bench_args)")
    L, missing = link_view(pairs, members[0], members)
    if missing:
        raise ValueError("no measured link %s" % missing)
    q, attn = planner_inputs(q, shape, ctx, hybrid_memory, depth, attn, attn_table, members)
    devs = profiles(C, q, mem, L, members, depth=depth, attn=attn)
    mp = hybrid_profile(dataclasses.replace(model_profile(q, shape, load_mode), context_len=ctx), q)
    head_at = 0 if head == "host" and len(assign) > 1 else None
    dec, pre = predict(assign, q, C, L, head_at=head_at, rtt_trips=rtt_trips, depth=depth, attn=attn, rpc_msg=rpc_msg,
                       backend_sampling=backend_sampling)
    return {"t": dec, "host": members[0], "assign": list(assign), "head_at": head_at, "plan": None, "links": L,
            "devices": devs, "model": mp, "q": q, "C": C, "decode_ms": dec, "prefill_ms": pre, "rtt_trips": rtt_trips,
            "considered": [(members[0], "fixed: %s, head %s" % (" + ".join("%s %d" % a for a in assign), head))]}


def model_profile(q, name, load_mode=None):
    """The engine's ModelProfile of shape q. load_mode "none": the host reads its weights into memory (-lm none) and pays
    its token embedding (2026-10-09); None, the mapped launch every registration on file priced."""
    return ModelProfile(name, q["n_layers"], q["bytes_per_layer"], q["kv_bytes_per_layer_per_token"],
                        q["activation_bytes"], context_len=PP + TG, head_bytes=q["head_bytes"],
                        embd_bytes=q["embd_bytes"], host_in_pipeline=True,
                        max_block_tensor_bytes=q.get("max_block_tensor_bytes", 0.0),
                        active_bytes_per_layer=q.get("active_bytes_per_layer", 0.0),
                        # real block sizes: from 2026-10-07 the rule plans with the exact fill and its default headroom
                        # (placement.EXACT_FILL_HEADROOM_MB); registrations made before then used the average fill
                        layer_bytes=tuple(q["layer_bytes"]) if q.get("layer_bytes") else None,
                        host_loads_weights=(load_mode == "none"))


def main(argv):
    ds = argv[argv.index("--dataset") + 1] if "--dataset" in argv else os.path.join(ROOT, "testbed", "dataset")
    out = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(ds, "pool_predictions.json")
    keys = argv[argv.index("--models") + 1].split(",") if "--models" in argv else ["qwen14b", "tinyllama", "mistral7b"]
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        raise SystemExit("unknown model key(s) %s; known: %s" % (", ".join(unknown), ", ".join(MODELS)))
    if os.path.exists(out) and "--dry-run" not in argv:
        raise SystemExit("%s exists: a registration is never overwritten (write a new file, or --dry-run)" % out)
    inputs = {f: os.path.join(ds, f) for f in ("characterisation.csv", "model_shapes.json", "link.json", "memory_at_rest.json")}
    shapes = json.load(open(inputs["model_shapes.json"]))
    pairs = json.load(open(inputs["link.json"]))["pairs"]
    mem = json.load(open(inputs["memory_at_rest.json"]))
    which = argv[argv.index("--rows") + 1] if "--rows" in argv else "char"
    size = "--size" in argv                         # SIZE2: the Pi 5 host's prefill size factor
    moe_batch = "--moe-batch" in argv               # MOEPRE: an MoE's prefill priced at the per-expert batch
    rtt = 1 if "--rtt1" in argv else RTT_TRIPS_DEFAULT   # RTT1: one round trip per worker per token
    R = rows(inputs["characterisation.csv"], which)
    Rt = rows_by_test(inputs["characterisation.csv"])
    res = {"registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "inputs_sha256": {**{f: sha(p) for f, p in inputs.items()}, "cluster.json": sha(CLUSTER)},
           "engine_sha256": {f: sha(os.path.join(ROOT, "ThinkPad", "control", f)) for f in ("placement.py", "gguf.py")},
           "protocol": "llama-bench -t 4 -p %d -n %d -r 3 -oe jsonl" % (PP, TG), "overhead_mb": OVERHEAD_MB,
           **({"row_tests": list(ROW_TESTS[which])} if which != "char" else {}),
           **({"pi5_host_prefill_size": PI5_HOST_PREFILL_SIZE} if size else {}),
           **({"rtt_trips": rtt, "rtt_rule": "RTT1 (testbed/dataset/rule_rtt1_20261009.json)"} if rtt == 1 else {}),
           **({"moe_prefill": "batch-aware (moe_prefill; registered rule MOEPRE, pool_predictions_20260928moe.json)"}
              if moe_batch else {}),
           "memory_accounting": "board-aware where cluster.json measures a board's role overheads (memory_overhead_mb, "
                                "worker_repays_received_tensor), the stand-in overhead_mb otherwise (placement.memory_need_mb)",
           "configs": {}}

    def add(cid, key, assign, why, plan=None):
        shape, quant, _ = MODELS[key]
        q = shapes[shape]
        C, src = costs(R, shapes, quant, size)
        members = [n for n, _ in assign]
        L, missing = link_view(pairs, assign[0][0], members)
        assert not missing, missing
        # The plan's own head placement (plan_optimal's default is head_placement="best" since 2026-10-08): priced,
        # charged and launched (-ot) where the plan put it. A plan without one (plan_even, a fixed split) keeps it last.
        head_at = 0 if plan is not None and plan.head_node and plan.head_node == assign[0][0] else None
        dec, pre = predict(assign, q, C, L, head_at=head_at, rtt_trips=rtt)
        if moe_batch and q.get("expert_count"):
            if head_at == 0:
                raise SystemExit("%s: --moe-batch prices the head on the last board only; this plan keeps it on the "
                                 "host (refused rather than registered with a wrong prefill)" % cid)
            curves = {n: {r: pp_curve(Rt, shapes, quant, n, r) for r in ("host", "worker")} for n in members}
            pre = moe_prefill(assign, q, curves, L, PP, True, rtt_trips=rtt)
        devs_a, mp_a = profiles(C, q, mem, L, members), model_profile(q, shape)
        res["configs"][cid] = {
            "model": shape, "quant": quant, "why": why, "pipeline": [{"node": n, "blocks": k} for n, k in assign],
            "head_on": assign[0][0] if head_at == 0 else assign[-1][0], "head_at": head_at,
            "bench": bench_args(assign, key, head_at=head_at), "memory": memory(assign, q, mem, devs_a, mp_a, head_at=head_at),
            "predicted": {"decode_ms_per_token": round(dec, 1), "decode_tok_s": round(1000 / dec, 3),
                          "prefill_ms_per_token": round(pre, 1), "prefill_tok_s": round(1000 / pre, 2)},
            "cost_sources": {n: src[n] for n, _ in assign},
            "link_pairs": {n: L[n]["pair"] for n in members[1:]}}
        if plan is not None:
            res["configs"][cid]["engine"] = {"order": plan.order, "layers": plan.layers, "head_node": plan.head_node,
                                             "predicted_ms": plan.predicted.get("ms_per_token")}

    def pipeline(plan):
        return [(n, plan.layers[n]) for n in plan.order if plan.layers.get(n, 0) > 0]

    for key in keys:
        shape, quant, _ = MODELS[key]
        host = HOST.get(key, "pi5")
        if shape not in shapes or shapes[shape].get("bytes_per_layer_source") != "tensor table":
            res["configs"]["%s.*" % key] = {"skipped": "no tensor-table shape for %s in model_shapes.json" % shape}
            continue
        q = shapes[shape]
        C, _ = costs(R, shapes, quant, size)
        nodes = [n for n in ("pi5", "pi4", "jetson") if n in C]
        L, missing = link_view(pairs, host, nodes)
        if missing:
            res["configs"]["%s.*" % key] = {"skipped": "no measured link %s in link.json (a link is never borrowed "
                                                       "from another pair)" % ", ".join(missing)}
            continue
        devs = profiles(C, q, mem, L, nodes)
        mp = model_profile(q, shape)
        rule = plan_optimal(devs, mp, "decode", host_candidates=[host])
        add("%s.rule" % key, key, pipeline(rule), "the rule, all three boards, hosted on the %s" % host, rule)
        rule_pre = plan_optimal(devs, mp, "prefill", host_candidates=[host])
        res["configs"]["%s.rule" % key]["engine"]["prefill_pick"] = {"order": rule_pre.order, "layers": rule_pre.layers,
                                                                    "head_node": rule_pre.head_node}
        if key != "tinyllama" or len(pipeline(rule)) > 1:
            if len(pipeline(rule)) > 1 or key in ("qwen14b", "mistral7b"):
                add("%s.%salone" % (key, host), key, [(host, q["n_layers"])],
                    "%s alone (baseline; infeasible = do not run)" % host)
            else:
                res["configs"]["%s.rule" % key]["why"] += "; the rule's pick IS the %s alone" % host
        else:
            res["configs"]["%s.rule" % key]["why"] += "; the rule's pick IS the Pi 5 alone"
        # The even baseline: naive, so in the cluster's own worker order (cluster.json), with the output head on
        # the last board. Only where that order leaves a board under 300 MiB of margin (the 14B's head on the 4 GB
        # Jetson) is the other order used instead, and the reason recorded.
        # The host runs llama.cpp, so it comes first; the others follow in the cluster's order.
        rest = [n for n in ("pi5", "pi4", "jetson") if n != host]
        order, why = [host] + rest, "the cluster's own order" + ("" if host == "pi5" else ", the host first")
        ev = plan_even(devs, mp, node_order=order)
        worst = min(min(v["margin_mib"], v.get("weights_margin_mib", v["margin_mib"]))
                    for v in memory(pipeline(ev), q, mem, devs, mp).values())
        if not ev.feasible or worst < 300:
            order = [host] + rest[::-1]
            why = "not the cluster's order, which leaves %d MiB on the tightest board" % worst
            ev = plan_even(devs, mp, node_order=order)
        add("%s.even" % key, key, pipeline(ev), "even split over all three boards, order %s (%s)"
            % ("/".join(order), why), ev)
        if key == "qwen14b":
            forced = plan_optimal(devs, mp, "decode", force_include=["pi4"], host_candidates=["pi5"])
            add("qwen14b.rule_pi4_forced", key, pipeline(forced), "the rule with the Pi 4 forced in (straggler study)", forced)
            add("qwen14b.pi5_pi4_best", key, [("pi5", 43), ("pi4", 5)],
                "the 2026-09-25 sweep's fastest split (ngl=6), re-run with the memory probe and -v")
            L4, _ = link_view(pairs, "pi4", ["pi4", "pi5"])
            d2 = profiles(C, q, mem, L4, ["pi4", "pi5"])
            p4 = plan_optimal(d2, mp, "decode", host_candidates=["pi4"])
            add("qwen14b.pi4_host", key, pipeline(p4), "the Pi 4 hosts, the Pi 5 is its worker (straggler study: the "
                "head moves off the Pi 4; the Pi 5 loses its repacked kernels)", p4)
    for cid, c in res["configs"].items():
        if "skipped" in c:
            print("%-26s SKIPPED: %s" % (cid, c["skipped"])); continue
        pl = " + ".join("%s %d" % (p["node"], p["blocks"]) for p in c["pipeline"])
        worst = min(v["margin_mib"] for v in c["memory"].values())
        print("%-26s %-34s head:%-6s decode %7.1f ms (%.3f t/s)  prefill %6.1f ms  min margin %5d MiB   %s %s"
              % (cid, pl, c["head_on"], c["predicted"]["decode_ms_per_token"], c["predicted"]["decode_tok_s"],
                 c["predicted"]["prefill_ms_per_token"], worst, c["bench"]["host"], c["bench"]["args"]))
    if "--dry-run" in argv:
        print("dry run: nothing written (would have written %s)" % os.path.relpath(out, ROOT))
        return
    json.dump(res, open(out, "w"), indent=1)
    print("wrote %s  (registered %s)" % (os.path.relpath(out, ROOT), res["registered"]))


if __name__ == "__main__":
    main(sys.argv[1:])
