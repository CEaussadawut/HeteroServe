# 04 — The placement engine

The project's core contribution: a rule that decides where layers go, computed from
measured device characteristics rather than guessed, and compared honestly against an even
split.

## The cost model

For one token of decode through a layer-pipelined cluster:

```
T = Σᵢ nᵢ·cᵢ + C                         C = Σ hops (activation_bytes / bw + rtt)
E = Σᵢ [ nᵢ·cᵢ·P_actᵢ + (T − nᵢ·cᵢ)·P_idleᵢ ]
J = T + λ·E          subject to   memᵢ(n) ≤ Mᵢ
```

- `nᵢ` — layers on node *i*
- `cᵢ` — measured per-layer time on node *i* (ms), separately for prefill and decode
- `P_actᵢ`, `P_idleᵢ` — measured rail power, active and idle
- `λ` — energy weight, in seconds per joule; sweeping it traces the Pareto front

The second term of `E` is the interesting one. A node holding one layer still burns idle
power for the *whole* token period, so adding a slow node taxes every other node's energy
as well as its own. That is the formal statement of the straggler question, and it is why
the rule sometimes drops a node that could have helped.

## Why it is exact rather than greedy

Substituting `T` into `E` makes the objective linear in the layer counts:

```
J = Σᵢ nᵢ·cᵢ·[1 + λ(ΔPᵢ + P_idle_S)] + C·[1 + λ·P_idle_S]
```

For a fixed participating set `S` and pipeline order, the optimum is reached by filling
layers into the smallest weights first, subject to each node's memory cap — no search.
The only combinatorial part is choosing `S` and its order, and with at most five nodes
that is enumerated exhaustively.

Complexity is `O(2^N · N! · N log N)` in the worst case: a few thousand cheap operations
for N ≤ 6, run once at cluster assembly. The proposal quotes `O(N·L)`; that describes the
layer fill, and `control/placement.py` is the honest version. Serving-time overhead is
zero — the plan is computed before the run, not per request.

## Memory constraint

```
per_layer_footprint = bytes_per_layer + kv_bytes_per_layer_per_token × context_len
cap_i = (usable_mem_mb − runtime_overhead_mb) / per_layer_footprint
```

`bytes_per_layer`, `kv_bytes_per_layer_per_token` and `activation_bytes` come from the
model's own GGUF header (`control/gguf.py`), not from a table someone typed. A wrong layer
count silently produces a wrong memory constraint and a wrong per-layer time.

Two honest approximations, stated where they are made:

- Weights per layer is derived from file size ÷ layer count, which folds in the embedding
  and output tensors. Within a few percent for these models; the alternative needs the
  layout of every quant type.
- KV is sized at full context. A run that never reaches full context leaves memory unused,
  so the rule is conservative rather than optimistic — the right direction to be wrong in
  when the failure mode is an OOM mid-measurement.

## Inputs are measured or the rule refuses

`control/profiles.py` builds `DeviceProfile`s from:

| Input | Source |
|---|---|
| `ms_per_layer_decode` / `_prefill` | Phase 2 isolated benchmark rows ÷ layer count |
| `power_active_w` / `power_idle_w` | the energy collector's power summary |
| `link_mbps`, `rtt_ms` | `heteroctl survey` (iperf3 + ping), re-run after every tier change |
| `usable_mem_mb` | each node's agent inventory |

With `strict=True` (the default) a missing input is an error naming the node and the field,
not a default. `--loose` exists for planning sketches and the output must be labelled as
one: a placement computed from invented numbers looks exactly like a placement computed
from measured ones.

Rows carrying validity flags (throttled, no energy window) are dropped before the medians
are taken. A flagged row is not a measurement, and averaging it in would quietly poison
every decision downstream.

## Strategies

| Name | What it does |
|---|---|
| `rule` | the heterogeneity-aware optimum |
| `even` | equal layers across every available node — the baseline to beat |
| `rule-with-straggler` | the best plan that is *forced* to use the straggler |
| `rule-no-straggler` | the best plan that may not use it |
| `single:<node>` | everything on one node — the "just buy more RAM" alternative |

`heteroctl compare` runs them all and reports the two numbers the success criteria name:
TPOT improvement over even split (MVP ≥ 10 %, stretch ≥ 25 %) and the energy improvement.

The straggler contrast is the answer to QR8 expressed as a number: forcing the Pi 4 in
costs *x* % latency and *y* % energy against the best plan without it. Expect the answer to
flip with model size — at 14B the straggler's memory may be the only reason the model runs
at all, and then its cost is not a cost but a price.

## From plan to command

A placement that cannot be expressed as a llama.cpp invocation is a paper exercise, so the
rule emits the flags:

```bash
heteroctl plan /opt/heteroserve/models/qwen2.5-7b-q4km.gguf
# ...
#   llama.cpp flags:
#     -m ... --rpc 192.168.50.12:50052,192.168.50.13:50052 -ngl 99 --tensor-split 0.64,0.36
```

`--tensor-split` carries the layer proportions in the order of the `--rpc` list. llama.cpp
rounds proportions to whole layers itself, so the realised split can differ by a layer from
the plan; the harness records the plan and the measured result separately so that
difference is visible rather than assumed away.

## Closing the loop

`experiment.prediction_error()` reports `(predicted − measured) / measured` per strategy and
tier. This is the feedback the proposal asks for — *measure → model → place*. A consistent
sign means the cost model has a systematic bias worth fixing before its output is trusted;
scatter around zero means the model is unbiased and merely imprecise.

Fix a biased model before reporting placement wins. A rule that beats the baseline while
mispredicting the physics is only accidentally right, and the next model size will expose it.

## Known limits

- **Single-stream decode only.** Each hop counts one RTT, which is right for a synchronous
  hand-off and wrong for a pipelined batch. Concurrent requests would overlap hops.
- **Contiguous layer blocks per node.** llama.cpp's `--tensor-split` works that way.
- **Per-layer time treated as model-independent per node.** Attention width varies between
  models; the spread is small next to the between-node spread the rule decides on, but it
  is an approximation, and the per-model rows are kept so it can be checked.
- **Prefill/decode disaggregation is not expressed here.** The rule places layers. The
  disaggregation experiment is exploratory scope and rides on top of it.
