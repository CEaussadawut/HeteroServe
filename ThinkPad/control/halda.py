"""Halda, prima.cpp's layer-to-device scheduler, re-implemented. Its decisions can then run on our runtime
(llama.cpp at the pin) and be priced by our cost model: one scheduler against the other, on the same executor.

Source, read line by line: OpenCPIL/prima.cpp @ 288df6a2 (testbed/prima.md).
  common/common.cpp   assign_layers_to_device            lines 860-1592
                      assign_layers_and_select_devices   lines 1595-1650
  src/llama.cpp       llama_model_compute_buf_size       lines 22243-22410

What Halda decides, for a ring of devices in a FIXED order with rank 0, the head, first:
  w_m   the layers per round on device m (its window); n_m of them on its GPU; k rounds, with L = k * sum(w_m).
  minimise   k * sum_m (a_m * w_m + b_m * n_m) + kappa, with a_0 divided by master_priority (1.01).
  - a_m is per-layer compute time from FLOPS micro-benchmarks, plus a KV-copy delay. Its RAM-read term is commented
    out (common.cpp:921), so decode is priced as compute.
  - The network term xi_m is 0 (common.cpp:971).
  subject to each device's memory: its layers, each with a full n_ctx KV cache, plus its compute buffer, and on the
  head also the output layer and one embedding row, all within the memory available when it profiled the device.
  A device left with exactly one layer is removed, and the rest are re-solved.

This port covers the case every model here falls in:
  - every device's share fits its memory (Halda's set M4);
  - no device has a CUDA or Metal GPU. prima.cpp has no Vulkan backend, and the Jetson cannot build it at all.
In that case k does not change the objective: each k gives the same per-device totals, and prima.cpp's own log
shows the ties (testbed/results/qr3_primacmpB_qwen3_14b_20260930_061646_prima_rank0.log, lines 253-259). Halda
keeps the first, k = 1, and its integer program reduces to a box-bounded fill: every device gets 1 layer, and the
rest go to the cheapest device first, up to its cap. When the shares cannot fit, Halda moves devices into its
disk-offload sets (M1-M3). That needs prima.cpp's prefetching runtime, so this port reports the case instead of
planning it.

Validated against the one Halda decision measured here: fed prima.cpp's 2026-09-30 profile of the Pi 5 and Pi 4 on
Qwen3-14B, it reproduces the windows [24, 16] (ThinkPad/tests/test_halda.py).
"""
from __future__ import annotations

GIB = 1024 ** 3          # prima.cpp's GIGABYTE (common.cpp:79)
F32 = 4
EPS = 1e-9
MASTER_PRIORITY = 1.01   # common/common.h:172, the default


def compute_buffer_bytes(n_embd: int, n_ff: int, n_head: int, n_ctx: int, n_ubatch: int = 512) -> int:
    """llama_model_compute_buf_size for the CPU backend, on a device that holds layers (src/llama.cpp:22381-22391).
    There, the offload branch overwrites the head's input/output term, so that term is not added here either."""
    n_inp_pos = n_ubatch
    n_kq_mask = n_ctx * n_ubatch
    n_bak_embd = n_norm = n_qcur = n_embd * n_ubatch
    n_ffn = 2 * n_ff * n_ubatch                       # gate + up
    n_kq = n_ctx * n_ubatch * n_head
    return F32 * (n_inp_pos + n_kq_mask + n_bak_embd + n_norm) + F32 * max(n_ffn, 2 * n_qcur + n_kq)


def kv_bytes_per_layer(n_head_kv: int, head_dim: int, n_ctx: int) -> int:
    """b' - b at common.cpp:887: K and V at F16 for the whole context, 2 * (n_embd_k_gqa + n_embd_v_gqa) * n_ctx."""
    gqa = n_head_kv * head_dim
    return 2 * (gqa + gqa) * n_ctx


def layer_ms_from_profile(layer_flops: dict, gflops: dict, kv_copy_ms: float) -> float:
    """a_m at common.cpp:906-923: over the weight types, the FLOPs of one layer of that type divided by the
    device's profiled FLOPS for it, in ms, plus the KV-copy delay. A type with no FLOPs contributes nothing."""
    t = sum(fl / (gflops[typ] * 1e9 + EPS) for typ, fl in layer_flops.items() if fl)
    return t * 1000.0 + kv_copy_ms


def memory_caps(budgets_gib: list, layer_bytes: float, kv_layer_bytes: float, buffer_bytes: list,
                head_bytes: float, embd_row_bytes: float) -> list:
    """The most layers each device of the ring may hold under Halda's M4 row, for k = 1 (common.cpp:1087-1089,
    1221-1235, 1303-1304): (budget - compute buffer - on the head the embedding row and the output layer) divided
    by (one layer plus its KV cache), rounded down. buffer_bytes is per device."""
    caps = []
    for m, g in enumerate(budgets_gib):
        cio = buffer_bytes[m] + ((embd_row_bytes + head_bytes) if m == 0 else 0.0)
        caps.append(int((g * GIB - cio) // (layer_bytes + kv_layer_bytes)))
    return caps


def solve(costs: list, caps: list, n_layers: int, master_priority: float = MASTER_PRIORITY):
    """Halda's integer program for k = 1 with every share fitting: minimise sum a'_m * w_m, subject to
    sum w_m = L and 1 <= w_m <= cap_m, with a'_0 = a_0 / master_priority (common.cpp:1280). The objective is linear
    over a box with one equality, so every device gets its lower bound and the rest go to the cheapest device first,
    ties to the earlier one. Returns (windows, objective_ms excluding kappa), or None when the caps cannot hold L:
    Halda would then offload to disk."""
    a = [c / master_priority if m == 0 else c for m, c in enumerate(costs)]
    if any(cap < 1 for cap in caps) or sum(caps) < n_layers or len(caps) > n_layers:
        return None
    w = [1] * len(caps)
    left = n_layers - len(caps)
    for m in sorted(range(len(caps)), key=lambda i: (a[i], i)):
        take = min(left, caps[m] - 1)
        w[m] += take
        left -= take
    if left:
        return None
    return w, sum(a[m] * w[m] for m in range(len(w)))


def plan(ring: list, costs: dict, caps: dict, n_layers: int, master_priority: float = MASTER_PRIORITY) -> dict:
    """assign_layers_and_select_devices (common.cpp:1595-1650). Solve, remove every device except the head that is
    left with exactly one layer, then re-solve with the rest, until no device is removed. The ring order is kept
    throughout, and the head (ring[0]) is never removed.

    Returns {"feasible", "ring", "layers": {node: w}, "removed": [...], "k": 1, "objective_ms"}. objective_ms
    excludes kappa, a constant for a given head. When infeasible it carries a "reason" instead."""
    live, removed = list(ring), []
    while True:
        res = solve([costs[n] for n in live], [caps[n] for n in live], n_layers, master_priority)
        if res is None:
            return {"feasible": False, "ring": list(ring), "removed": removed,
                    "reason": "the shares cannot fit in memory (%s): Halda would move devices into its disk-offload "
                              "sets M1-M3, which need prima.cpp's prefetching runtime, outside this port"
                              % ", ".join("%s cap %d" % (n, caps[n]) for n in live)}
        w, obj = res
        drop = [n for n, x in zip(live, w) if x == 1 and n != live[0]]
        if not drop:
            return {"feasible": True, "ring": list(ring), "layers": dict(zip(live, w)), "removed": removed,
                    "k": 1, "objective_ms": round(obj, 3)}
        removed += drop
        live = [n for n in live if n not in drop]
