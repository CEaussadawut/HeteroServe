// fa_bench_cpu: fa_bench_20261028.cpp's graph on the CPU backend (2026-11-04, dl2: a depth law per attention shape for the
// CPU boards; the user: "do all four follow-up studies"). Times ggml's FLASH_ATTN_EXT for one decoded token of a given
// attention shape, as llama.cpp at the pin calls it (src/llama-graph.cpp, build_attn_mha; flash attention is on "auto" in
// llama-bench and llama-server, which the CPU backend resolves to on), with no model loaded.
//
//   fa_bench_cpu <head_dim> <n_head> <n_head_kv> <layers> <reps> <threads> <n_kv>[,<n_kv>...]
//
// The graph is fa_bench's: <layers> chained attention ops, each on its own K and V cache (F16, the cache's layout with flash
// attention), an F16 mask whose last 128 cells are -inf, scale 1/sqrt(head_dim), F32 accumulation, n_kv_max the finite
// cells. What differs, for a CPU:
// - the backend is ggml-cpu with <threads> threads (the boards' -t 4);
// - each repetition is timed alone, and between repetitions a 64 MiB buffer is read through, outside the timing, so the
//   caches start each graph as a decoded token's attention finds them: the model's weights have just streamed through
//   them. The timing is the median over <reps>.
// Prints "n_kv <n> us_per_graph <median> us_per_op <median/layers> min <min> max <max>" per n_kv.
// Build on the node against its pinned tree (instrumentation only; nothing of llama.cpp is changed):
//   g++ -O2 -std=c++17 fa_bench_cpu_20261104.cpp -I /opt/heteroserve/llama.cpp/ggml/include \
//       -L /opt/heteroserve/llama.cpp/build/bin -lggml -lggml-base -lggml-cpu \
//       -Wl,-rpath,/opt/heteroserve/llama.cpp/build/bin -o fa_bench_cpu
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
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

static volatile unsigned long long g_sink = 0;

static void flush_caches(std::vector<unsigned long long> & buf) {
    unsigned long long s = 0;
    for (size_t i = 0; i < buf.size(); i += 8) {      // one read per 64-byte line
        s += buf[i];
    }
    g_sink += s;
}

int main(int argc, char ** argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: %s <head_dim> <n_head> <n_head_kv> <layers> <reps> <threads> <n_kv>[,<n_kv>...]\n", argv[0]);
        return 2;
    }
    const int hd = atoi(argv[1]), nh = atoi(argv[2]), nkvh = atoi(argv[3]), layers = atoi(argv[4]), reps = atoi(argv[5]);
    const int threads = atoi(argv[6]);
    std::vector<int> kvs;
    for (char * p = strtok(argv[7], ","); p; p = strtok(nullptr, ",")) {
        kvs.push_back(atoi(p));
    }
    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) {
        fprintf(stderr, "no CPU backend\n");
        return 1;
    }
    ggml_backend_cpu_set_n_threads(be, threads);
    std::vector<unsigned long long> flush(64ull * 1024 * 1024 / sizeof(unsigned long long), 1ull);
    printf("shape head_dim %d n_head %d n_head_kv %d layers %d reps %d threads %d flush 64MiB\n", hd, nh, nkvh, layers, reps, threads);
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
        std::vector<double> t;
        for (int r = 0; r < reps; r++) {
            flush_caches(flush);
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(be, gf);
            t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(t.begin(), t.end());
        const double med = reps % 2 ? t[reps / 2] : 0.5 * (t[reps / 2 - 1] + t[reps / 2]);
        printf("n_kv %d us_per_graph %.1f us_per_op %.2f min %.1f max %.1f\n", n_kv, med, med / layers, t.front(), t.back());
        fflush(stdout);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(be);
    return 0;
}
