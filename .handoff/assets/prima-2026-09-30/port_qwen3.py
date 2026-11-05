#!/usr/bin/env python3
"""Port Qwen3 (dense) support into prima.cpp @ 288df6a2 (user, 2026-09-30: "i insist to go with qwen3-14B"). From the prima.cpp tree:

    python3 /mnt/data/KMITL/Final/HeteroServe/.handoff/assets/prima-2026-09-30/port_qwen3.py src/llama.cpp

prima.cpp's late-2024 llama.cpp base has no qwen3 architecture, and its distributed path supports only llama and qwen2
(src/llama.cpp: GGML_ASSERT(model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_QWEN2)). Qwen3 dense is qwen2 without the
Q/K/V biases and with a per-head RMS norm of Q and K before RoPE (upstream llama.cpp at the cluster pin b23701f7,
src/models/qwen3.cpp). So every edit here copies prima.cpp's own qwen2 code and makes only those changes:
  1. the architecture (enum, name "qwen3") and its tensor names (qwen2's + attn_q_norm / attn_k_norm);
  2. hyperparameters: qwen2's read (the RMS eps); the model type is informational;
  3. llm_load_qwen3_tensors: prima.cpp's llm_load_qwen2_tensors (only this rank's layer window) with the Q/K norms, no biases,
     and Q / O sized n_embd_head_k * n_head, as upstream;
  4. both loader call sites: the model load, and the Halda scheduler's profiler (llama_model_n_flops);
  5. build_qwen3: prima.cpp's distributed build_qwen2 (sub-graphs per layer window) with each head's Q and K RMS-normed
     between the reshape and RoPE, as upstream;
  6. the graph dispatch and the distributed-support assertion;
  7. RoPE type NEOX, as qwen2 (upstream lists QWEN3 with the NEOX architectures);
  8. the profiler's per-tensor FLOP count: attn_q_norm / attn_k_norm counted like attn_norm (an RMS norm and its weights).
     Without this it logs "Uncaught tensor" and RETURNS EARLY, handing Halda an incomplete profile.
Not changed: the F32 KQ precision list (upstream does not force it for qwen3), and everything else. Every anchor must match
exactly once, or nothing is written.
"""
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "src/llama.cpp"
s = open(p).read()
TAG = "qwen3 port (HeteroServe, 2026-09-30)"


def ins_after(anchor, add):
    global s
    assert s.count(anchor) == 1, (anchor[:70], s.count(anchor))
    s = s.replace(anchor, anchor + add, 1)


def ins_before(anchor, add):
    global s
    assert s.count(anchor) == 1, (anchor[:70], s.count(anchor))
    s = s.replace(anchor, add + anchor, 1)


def repl(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new, 1)


# 1. the architecture and its tensor names
ins_after("\n    LLM_ARCH_QWEN2,\n", "    LLM_ARCH_QWEN3,  // %s\n" % TAG)
ins_after('    { LLM_ARCH_QWEN2,           "qwen2"        },\n', '    { LLM_ARCH_QWEN3,           "qwen3"        },\n')
q2map = """    {
        LLM_ARCH_QWEN2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
"""
q3map = q2map.replace("LLM_ARCH_QWEN2,", "LLM_ARCH_QWEN3,  // %s" % TAG).replace(
    """            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
""", """            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
""")
ins_after(q2map, q3map)

# 2. hyperparameters
ins_before("""        case LLM_ARCH_QWEN2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 24:""", """        case LLM_ARCH_QWEN3:  // %s
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_14B; break;
                    case 64: model.type = e_model::MODEL_32B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
""" % TAG)

# 3. the tensor loader
loader = '''// %s: llm_load_qwen2_tensors with upstream's qwen3 tensors -- attn_q_norm / attn_k_norm per head, no Q/K/V biases,
// and Q / O sized n_embd_head_k * n_head (upstream llama.cpp b23701f7, src/models/qwen3.cpp)
static void llm_load_qwen3_tensors(
    llama_model_loader & ml,
    llama_model        & model,
    std::map<ggml_backend_buffer_type_t, ggml_context *> & ctx_map,
    uint32_t             n_world,
    uint32_t             my_rank,
    const uint32_t     * n_layer_window,
    bool                 set_needed) {
    const auto tn = LLM_TN(model.arch);

    ggml_context * ctx_input        = nullptr;
    ggml_context * ctx_output       = nullptr;
    ggml_context * ctx_output_split = nullptr;

    if (my_rank == 0) {
        ctx_input        = ctx_map.at(model.buft_input.buft);
        ctx_output       = ctx_map.at(model.buft_output.buft);
        ctx_output_split = ctx_map.at(model.buft_output.buft_matrix);
    }

    auto ctx_for_layer       = [&](int i) { return ctx_map.at(model.buft_layer[i].buft); };
    auto ctx_for_layer_split = [&](int i) { return ctx_map.at(model.buft_layer[i].buft_matrix); };

    const llama_hparams hparams = model.hparams;
    const int64_t n_embd        = hparams.n_embd;
    const int64_t n_embd_head_k = hparams.n_embd_head_k;
    const int64_t n_head        = hparams.n_head();
    const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa();
    const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa();
    const int64_t n_ff          = hparams.n_ff();
    const int64_t n_vocab       = hparams.n_vocab;
    const int64_t n_layer       = hparams.n_layer;

    if (my_rank == 0) {
        // token embedding
        model.tok_embd    = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0, set_needed);

        // output
        model.output_norm = ml.create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0, set_needed);
        model.output      = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED, set_needed);

        // if output is NULL, init from the input tok embed
        if (model.output == NULL) {
            model.output = ml.create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED, set_needed);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        if (!this_layer_is_mine(i, n_world, my_rank, n_layer_window)) {
            continue;
        }

        int local_i = map_layer_to_local_id(i, n_world, my_rank, n_layer_window);
        ggml_context * ctx_layer = ctx_for_layer(local_i);
        ggml_context * ctx_split = ctx_for_layer_split(local_i);

        auto & layer = model.layers[local_i];

        // attention norm
        layer.attn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0, set_needed);

        // attention
        layer.wq = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0, set_needed);
        layer.wk = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0, set_needed);
        layer.wv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0, set_needed);
        layer.wo = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0, set_needed);

        // per-head Q and K norms
        layer.attn_k_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0, set_needed);
        layer.attn_q_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0, set_needed);

        // feed-forward
        layer.ffn_norm = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0, set_needed);

        layer.ffn_gate = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0, set_needed);
        layer.ffn_down = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0, set_needed);
        layer.ffn_up   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0, set_needed);
    }
}

''' % TAG
ins_before("// Returns false if cancelled by progress_callback\nstatic bool llm_load_tensors_impl(", loader)

# 4. both loader call sites
ins_after("""            case LLM_ARCH_QWEN2:
                llm_load_qwen2_tensors(ml, model, ctx_map, n_world, my_rank, n_layer_window, true);
                break;
""", """            case LLM_ARCH_QWEN3:  // %s
                llm_load_qwen3_tensors(ml, model, ctx_map, n_world, my_rank, n_layer_window, true);
                break;
""" % TAG)
ins_after("""            case LLM_ARCH_QWEN2:
                llm_load_qwen2_tensors(*ml, *model, ctx_map, 1, 0, n_layer_window, false);
                break;
""", """            case LLM_ARCH_QWEN3:  // %s
                llm_load_qwen3_tensors(*ml, *model, ctx_map, 1, 0, n_layer_window, false);
                break;
""" % TAG)

# 5. the distributed graph
START = "    std::vector<ggml_cgraph *> build_qwen2() {"
NEXT = "\n    struct ggml_cgraph * build_qwen2moe() {"          # the builder that follows build_qwen2 at 288df6a2
assert s.count(START) == 1 and s.count(NEXT) == 1
a, b = s.index(START), s.index(NEXT)
q2 = s[a:b + 1]
old_qk = """                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);
"""
new_qk = """                // qwen3: each head's Q and K RMS-normed before RoPE (upstream llama.cpp b23701f7, src/models/qwen3.cpp)
                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[local_il].attn_q_norm, NULL, LLM_NORM_RMS, cb, il);
                cb(Qcur, "Qcur_normed", il);

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[local_il].attn_k_norm, NULL, LLM_NORM_RMS, cb, il);
                cb(Kcur, "Kcur_normed", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);
"""
assert q2.count(old_qk) == 1
q3 = q2.replace(START, "    // %s: build_qwen2 with qwen3's per-head Q/K norms; its loader creates no Q/K/V biases\n"
                       "    std::vector<ggml_cgraph *> build_qwen3() {" % TAG).replace(old_qk, new_qk)
s = s[:b + 1] + "\n" + q3 + s[b + 1:]

# 6. dispatch and the distributed-support assertion
repl('GGML_ASSERT((model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_QWEN2) && "this model is currently not supported.\\n");',
     'GGML_ASSERT((model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_QWEN2 || model.arch == LLM_ARCH_QWEN3) && "this model is currently not supported.\\n");')
ins_after("""        case LLM_ARCH_QWEN2:
            {
                result = llm.build_qwen2();
            } break;
""", """        case LLM_ARCH_QWEN3:  // %s
            {
                result = llm.build_qwen3();
            } break;
""" % TAG)

# 7. RoPE type
ins_after("""        case LLM_ARCH_QWEN:
        case LLM_ARCH_QWEN2:
""", """        case LLM_ARCH_QWEN3:  // %s
""" % TAG)

# 8. the profiler's per-tensor count
ins_before("""            } else if (blk_suffix == "attn_q.bias" || blk_suffix == "attn_k.bias" || blk_suffix == "attn_v.bias" ||""",
"""            } else if (blk_suffix == "attn_q_norm.weight") { // %s: per-head RMS norm of Q
                count_n_flops (n_flops,  GGML_TYPE_F32, PROFILER_LAYER_BACKEND, 4 * n_embd_head_k * n_head + n_head); // rms norm
                count_n_flops (n_flops,  cur->type,     PROFILER_LAYER_BACKEND, n_embd_head_k * n_head); // norm weights
            } else if (blk_suffix == "attn_k_norm.weight") { // %s: per-head RMS norm of K
                count_n_flops (n_flops,  GGML_TYPE_F32, PROFILER_LAYER_BACKEND, 4 * n_embd_head_k * n_head_kv + n_head_kv); // rms norm
                count_n_flops (n_flops,  cur->type,     PROFILER_LAYER_BACKEND, n_embd_head_k * n_head_kv); // norm weights
""" % (TAG, TAG))

open(p, "w").write(s)
print("ported: %s" % p)
