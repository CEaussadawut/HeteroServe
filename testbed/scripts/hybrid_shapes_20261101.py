#!/usr/bin/env python3
"""The hybrid models' memory beyond their weights, per block (2026-11-01, the scheduler review's item 3; the user: "do item 1
to 4 then 6-7"). Reads GGUF headers and one serve's logs; contacts nothing.

    python3 testbed/scripts/hybrid_shapes_20261101.py

The qwen35 / qwen35moe hybrids compute three of every four blocks as gated delta-net recurrences. llama.cpp at the pin gives a
block a KV cache only when (i + 1) % full_attention_interval == 0, every other block below n_layer a fixed recurrent state
whatever the context, and the MTP block neither. model_shapes.json carries no full_attention_interval, so the planner charged
every block a full-attention KV cache: the 35B's 41 blocks at 4,096 cells, 328 MiB, where llama.cpp allocated 80 MiB of KV
(10 layers) and 62.81 MiB of recurrent state (2026-10-28's serve). Per hybrid shape this writes, from its GGUF header (the
laptop's copy, the file model_shapes.json was read from):
  full_attention_interval   <arch>.full_attention_interval
  rs_bytes_per_layer        4 x (n_embd_r + n_embd_s): f32, llama.cpp's default type_r / type_s, with llama-hparams.cpp at
                            the pin: n_embd_r = (ssm_d_conv - 1) x (ssm_d_inner + 2 x ssm_n_group x ssm_d_state),
                            n_embd_s = ssm_d_state x ssm_d_inner
and checks both against that serve: every device's KV and recurrent-state buffer, as llama.cpp logged them, against the blocks
the plan gave it (laptop 25 + Jetson 3 + Pi 4 4 + Pi 5 9). The serve's full server log lives in the laptop's ~/logs; it is
copied into testbed/results/hybrid_shapes_20261101/ the first time, so the check reads only the record.
Writes testbed/dataset/hybrid_shapes_20261101.json (never overwritten).
"""
import datetime, hashlib, json, os, re, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
from control import gguf                                    # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "hybrid_shapes_20261101.json")
EVID = os.path.join(ROOT, "testbed", "results", "hybrid_shapes_20261101")
SERVE = os.path.join(ROOT, "testbed", "results", "serve_qwen36moe_q6k_20261028_043905.log")   # serve_pool.sh's own log
SERVER = "llama-server_serve_20261028_043905.log"                                             # the host's server log
PLAN = [("laptop", 25), ("jetson", 3), ("pi4", 4), ("pi5", 9)]                                # that serve's plan, host first
DEVICE = {"CPU": "laptop", "192.168.1.114:50052": "jetson", "192.168.10.12:50052": "pi4", "192.168.10.11:50052": "pi5"}
MIB = 1048576.0


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def hybrid(path):
    """The GGUF header's hybrid fields, and the bytes llama.cpp allocates per recurrent block."""
    kv = gguf.read_metadata(os.path.expanduser(path.split(".gguf")[0] + ".gguf"))   # "... .gguf (on pi4)": the laptop's copy
    kv = kv.get("kv", kv)
    a = kv["general.architecture"]
    fai = kv.get(a + ".full_attention_interval")
    if not fai:
        return None
    d_conv, d_inner = kv[a + ".ssm.conv_kernel"], kv[a + ".ssm.inner_size"]
    n_group, d_state = kv[a + ".ssm.group_count"], kv[a + ".ssm.state_size"]
    n_embd_r = (d_conv - 1) * (d_inner + 2 * n_group * d_state)
    n_embd_s = d_state * d_inner
    return {"architecture": a, "full_attention_interval": int(fai), "rs_bytes_per_layer": 4 * (n_embd_r + n_embd_s),
            "n_embd_r": n_embd_r, "n_embd_s": n_embd_s,
            "gguf": {k: kv[a + "." + k] for k in ("ssm.conv_kernel", "ssm.inner_size", "ssm.group_count", "ssm.state_size",
                                                   "ssm.time_step_rank", "block_count", "nextn_predict_layers",
                                                   "attention.head_count_kv", "attention.key_length",
                                                   "attention.value_length")}}


def per_block(q, h, ctx):
    """(KV bytes, recurrent-state bytes) of each block at context ctx."""
    main = len(q["layer_bytes"]) - int(q.get("nextn_layers") or 0)
    fai = h["full_attention_interval"]
    out = []
    for i in range(len(q["layer_bytes"])):
        if i >= main:
            out.append((0.0, 0.0))
        elif (i + 1) % fai == 0:
            out.append((q["kv_bytes_per_layer_per_token"] * ctx, 0.0))
        else:
            out.append((0.0, float(h["rs_bytes_per_layer"])))
    return out


def logged(path, what):
    """{device: MiB} of one buffer kind as llama.cpp logged it ("KV" or "RS"), and the logged total line."""
    dev, total = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"(llama_kv_cache|llama_memory_recurrent):\s+(.*?) %s buffer size =\s+([0-9.]+) MiB" % what, line)
        if m:
            name = m.group(2).strip()
            key = "CPU" if name == "CPU" else re.sub(r"^RPC\d+\[(.*)\]$", r"\1", name)
            dev[DEVICE[key]] = float(m.group(3))
        m = re.search(r"(llama_kv_cache|llama_memory_recurrent): size =\s+([0-9.]+) MiB \(\s*(\d+) cells,\s+(\d+) layers", line)
        if m and (m.group(1) == "llama_kv_cache") == (what == "KV"):
            total = {"mib": float(m.group(2)), "cells": int(m.group(3)), "layers": int(m.group(4))}
    return dev, total


def main():
    if os.path.exists(OUT):
        raise SystemExit("%s exists: never overwritten" % os.path.relpath(OUT, ROOT))
    shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
    res = {"made": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "why": "the scheduler review's item 3 (testbed/JOURNAL.md, 2026-11-01): the planner charged a hybrid's every block "
                  "a full-attention KV cache", "rule": "a block i < n_layer is full-attention when (i + 1) % "
           "full_attention_interval == 0 and then holds kv_bytes_per_layer_per_token x context; every other block below "
           "n_layer holds rs_bytes_per_layer whatever the context; the MTP block (nextn) holds neither", "shapes": {}}
    for name, q in shapes.items():
        if not isinstance(q, dict) or q.get("architecture") not in ("qwen35", "qwen35moe"):
            continue
        h = hybrid(q["path"])
        if h is None:
            raise SystemExit("%s: a %s shape whose GGUF has no full_attention_interval" % (name, q["architecture"]))
        g = h["gguf"]
        kv_tok = 2 * g["attention.head_count_kv"] * g["attention.key_length"] * 2      # K and V in f16, both heads 256 wide
        if g["attention.key_length"] != g["attention.value_length"] or kv_tok != q["kv_bytes_per_layer_per_token"]:
            raise SystemExit("%s: the header's KV (%d B per token) is not model_shapes.json's" % (name, kv_tok))
        if g["block_count"] != q["n_layers"] or int(g.get("nextn_predict_layers") or 0) != int(q.get("nextn_layers") or 0):
            raise SystemExit("%s: the header's blocks are not model_shapes.json's" % name)
        h["from"] = "%s (%s)" % (q["path"], q.get("read_from", "")[:120])
        res["shapes"][name] = h
        blocks = per_block(q, h, 4096)
        print("%-24s %-9s interval %d: %2d attention + %2d recurrent blocks; %.4f MiB recurrent state each; at 4,096 cells "
              "KV %.2f + state %.2f MiB, where the planner charged %.2f" % (
                  name, h["architecture"], h["full_attention_interval"], sum(1 for k, _ in blocks if k),
                  sum(1 for _, s in blocks if s), h["rs_bytes_per_layer"] / MIB, sum(k for k, _ in blocks) / MIB,
                  sum(s for _, s in blocks) / MIB, q["kv_bytes_per_layer_per_token"] * 4096 * q["n_layers"] / MIB))
    # the check against the 35B's serve (2026-10-28 04:39, laptop-hosted; the full server log copied into the record once)
    os.makedirs(EVID, exist_ok=True)
    ev = os.path.join(EVID, SERVER)
    if not os.path.exists(ev):
        shutil.copy2(os.path.expanduser("~/logs/" + SERVER), ev)
    q, h = shapes["qwen3.6-35b-a3b-q6_k"], res["shapes"]["qwen3.6-35b-a3b-q6_k"]
    blocks = per_block(q, h, 4096)
    kv_log, kv_tot = logged(ev, "KV")
    rs_log, rs_tot = logged(ev, "RS")
    rows, start, worst = [], 0, 0.0
    for n, k in PLAN:
        want_kv = sum(b[0] for b in blocks[start:start + k]) / MIB
        want_rs = sum(b[1] for b in blocks[start:start + k]) / MIB
        rows.append({"device": n, "blocks": k, "kv_mib": {"logged": kv_log[n], "rule": round(want_kv, 3)},
                     "rs_mib": {"logged": rs_log[n], "rule": round(want_rs, 3)}})
        worst = max(worst, abs(kv_log[n] - want_kv), abs(rs_log[n] - want_rs))
        start += k
    res["check_35b"] = {"serve_log": os.path.relpath(SERVE, ROOT), "server_log": os.path.relpath(ev, ROOT),
                        "server_log_sha256": sha(ev), "plan": PLAN, "kv_total": kv_tot, "rs_total": rs_tot,
                        "devices": rows, "worst_abs_mib": round(worst, 3),
                        "holds": worst < 0.01, "note": "llama.cpp prints two decimals: 0.005 MiB is its rounding"}
    for r in rows:
        print("  35B on %-6s %2d blocks: KV logged %6.2f, rule %6.2f MiB; recurrent state logged %6.2f, rule %6.2f MiB" % (
            r["device"], r["blocks"], r["kv_mib"]["logged"], r["kv_mib"]["rule"], r["rs_mib"]["logged"], r["rs_mib"]["rule"]))
    print("  logged totals: KV %s; recurrent %s; worst device difference %.3f MiB -> %s" % (
        kv_tot, rs_tot, worst, "HOLDS" if res["check_35b"]["holds"] else "FAILS"))
    if not res["check_35b"]["holds"]:
        raise SystemExit("the rule does not reproduce the 35B's serve: nothing written")
    res["inputs_sha256"] = {"model_shapes.json": sha(os.path.join(DS, "model_shapes.json")),
                            os.path.relpath(SERVE, ROOT): sha(SERVE), os.path.relpath(ev, ROOT): sha(ev),
                            "hybrid_shapes_20261101.py": sha(os.path.abspath(__file__))}
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote %s" % os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
