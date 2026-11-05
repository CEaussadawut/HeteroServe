// fa_bench: time ggml's FLASH_ATTN_EXT for one decoded token of a given attention shape, called as llama.cpp at the pin
// calls it (src/llama-graph.cpp, build_attn_mha), with no model loaded (2026-10-28, lc4n: a long-context law for the Jetson).
//
//   fa_bench <head_dim> <n_head> <n_head_kv> <layers> <reps> <n_kv>[,<n_kv>...]
//
// One graph holds <layers> attention ops, each on its own K and V cache, as one token's graph holds one per layer, and
// CHAINED as a model's are: each op's query is its own q plus the previous op's output (one ADD a layer), so the ops run
// one after another (ggml-vulkan puts no barrier between independent ops, and on this GPU unchained ops overlapped: v1
// of this file, 2026-10-28 08:09, priced Qwen3-0.6B's slope 4 % under the in-model timer's). Per op: q [head_dim, n_head, 1] F32 permuted (0,2,1,3); K and V cache views [head_dim, n_head_kv, n_kv] F16 over
// a [head_dim * n_head_kv, n_kv] cache, permuted (0,2,1,3) (the cache's layout with flash attention: V not transposed); an
// F16 mask [n_kv, 1] whose last 128 cells are -inf (a token mid-way through the cache's last 256-cell pad, as llama-bench's
// decode sees it); scale 1/sqrt(head_dim); F32 accumulation (ggml_prec_set_acc); the sparse hint n_kv_max = the finite
// cells, as llama.cpp passes. Backend: Vulkan device 0. Per n_kv: 3 untimed graphs, then <reps> timed ones, each
// ggml_backend_graph_compute (submit and wait). Prints "n_kv <n> us_per_graph <t> us_per_op <t/layers>" per n_kv; the
// wall time includes one submit and wait per graph, as a decoded token's graph has.
// Build on the node against its pinned tree (instrumentation only; nothing of llama.cpp is changed):
//   g++ -O2 -std=c++17 fa_bench_20261028.cpp -I ~/llama.cpp/ggml/include -L ~/llama.cpp/build/bin \
//       -lggml -lggml-base -lggml-vulkan -Wl,-rpath,$HOME/llama.cpp/build/bin -o fa_bench
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void fill_f32(ggml_tensor * t, unsigned seed) {
    std::vector<float> v(ggml_nelements(t));
    for (size_t i = 0; i < v.size(); i++) {
        seed = seed * 1664525u + 1013904223u;
        v[i] = ((seed >> 9) & 0xffff) / 65536.0f - 0.5f;
    }
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
}

static void fill_f16(ggml_tensor * t, unsigned seed) {
    std::vector<ggml_fp16_t> v(ggml_nelements(t));
    for (size_t i = 0; i < v.size(); i++) {
        seed = seed * 1664525u + 1013904223u;
        v[i] = ggml_fp32_to_fp16(((seed >> 9) & 0xffff) / 65536.0f - 0.5f);
    }
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(ggml_fp16_t));
}

int main(int argc, char ** argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s <head_dim> <n_head> <n_head_kv> <layers> <reps> <n_kv>[,<n_kv>...]\n", argv[0]);
        return 2;
    }
    const int hd = atoi(argv[1]), nh = atoi(argv[2]), nkvh = atoi(argv[3]), layers = atoi(argv[4]), reps = atoi(argv[5]);
    std::vector<int> kvs;
    for (char * p = strtok(argv[6], ","); p; p = strtok(nullptr, ",")) {
        kvs.push_back(atoi(p));
    }
    ggml_backend_t be = ggml_backend_vk_init(0);
    if (!be) {
        fprintf(stderr, "no Vulkan device\n");
        return 1;
    }
    printf("shape head_dim %d n_head %d n_head_kv %d layers %d reps %d\n", hd, nh, nkvh, layers, reps);
    for (int n_kv : kvs) {
        const size_t graph_size = 64 + 16 * (size_t) layers;
        ggml_init_params ip = { ggml_tensor_overhead() * (16 + 12 * (size_t) layers) + ggml_graph_overhead_custom(graph_size, false),
                                nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, 1, 1, 1);
        std::vector<ggml_tensor *> qs, kcs, vcs, outs;
    ggml_tensor * prev = nullptr;
        const int n_finite = n_kv > 128 ? n_kv - 128 : n_kv;
        for (int l = 0; l < layers; l++) {
            ggml_tensor * q  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, 1);
            ggml_tensor * kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, (int64_t) hd * nkvh, n_kv);
            ggml_tensor * vc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, (int64_t) hd * nkvh, n_kv);
            ggml_tensor * k  = ggml_view_3d(ctx, kc, hd, nkvh, n_kv, ggml_row_size(GGML_TYPE_F16, hd),
                                            ggml_row_size(GGML_TYPE_F16, (int64_t) hd * nkvh), 0);
            ggml_tensor * v  = ggml_view_3d(ctx, vc, hd, nkvh, n_kv, ggml_row_size(GGML_TYPE_F16, hd),
                                            ggml_row_size(GGML_TYPE_F16, (int64_t) hd * nkvh), 0);
            ggml_tensor * qin = prev ? ggml_add(ctx, q, prev) : q;      // [hd, n_head, 1] both: the FA output's shape
            ggml_tensor * o  = ggml_flash_attn_ext(ctx, ggml_permute(ctx, qin, 0, 2, 1, 3), ggml_permute(ctx, k, 0, 2, 1, 3),
                                                   ggml_permute(ctx, v, 0, 2, 1, 3), mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
            ggml_flash_attn_ext_set_n_kv_max(o, n_finite);
            ggml_prec_set_acc(o, GGML_PREC_F32);
            qs.push_back(q); kcs.push_back(kc); vcs.push_back(vc); outs.push_back(o);
            prev = o;
        }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(gf, outs.back());              // the chain pulls in every layer
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        if (!buf) {
            fprintf(stderr, "n_kv %d: could not allocate\n", n_kv);
            return 1;
        }
        std::vector<ggml_fp16_t> m(n_kv, ggml_fp32_to_fp16(0.0f));
        for (int i = n_finite; i < n_kv; i++) {
            m[i] = ggml_fp32_to_fp16(-INFINITY);
        }
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(ggml_fp16_t));
        for (int l = 0; l < layers; l++) {
            fill_f32(qs[l], 1u + l); fill_f16(kcs[l], 1000u + l); fill_f16(vcs[l], 2000u + l);
        }
        for (int i = 0; i < 3; i++) {
            ggml_backend_graph_compute(be, gf);
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; r++) {
            ggml_backend_graph_compute(be, gf);
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        printf("n_kv %d us_per_graph %.1f us_per_op %.2f\n", n_kv, us, us / layers);
        fflush(stdout);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(be);
    return 0;
}
