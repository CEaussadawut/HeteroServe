#!/usr/bin/env python3
"""The two models downloaded on 2026-10-21, into the dataset's model_shapes.json (as laptop_node_datasets_20261011.py shapes):
  qwen3.6-35b-a3b-q6_k  bartowski/Qwen_Qwen3.6-35B-A3B-GGUF @ 5c2410d7 (user, ~17:15: "Q6_K, 31.0 GB"): the four-node
                        evaluation's model that needs every node; a qwen35moe hybrid with one MTP block (nextn, not loaded at the
                        pin). As the 27B's qwen35 entry: kv_bytes_per_layer_per_token is the full-attention figure applied to every
                        block (it over-states the KV), activation n_embd x 4; the recurrent state is not charged.
  qwen3-32b-q8_0        Qwen/Qwen3-32B-GGUF @ 938a7432 (user, ~20:45: "Yes, download it"): the newest Qwen prima.cpp can run, for
                        prima.cpp + Halda against the engine on all four nodes.
Both read from the laptop's copies, sha256-verified against the LFS oid Hugging Face publishes at the pinned revision (the
fetch logs in ~/.cache/heteroserve-models; testbed/models.md). The file keeps its own format (one line), the old entries
byte-identical; the previous file is kept read-only beside it first.

    python3 testbed/scripts/models_20261021.py
"""
import json, os, shutil, stat, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(ROOT, "testbed", "dataset")
CACHE = os.path.expanduser("~/.cache/heteroserve-models")
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
from control import gguf                                   # noqa: E402

SHAPES = {"qwen3.6-35b-a3b-q6_k": ("Qwen_Qwen3.6-35B-A3B-Q6_K.gguf", "~/models/qwen3.6-35b-a3b-q6_k.gguf (on the laptop)",
                                   "a97ef79439194d8ec722bc0b6c316f0c41c0b477b35bbae8177c772ecbb0eb91"),
          "qwen3-32b-q8_0": ("Qwen3-32B-Q8_0.gguf", "~/models/qwen3-32b-q8_0.gguf (on the laptop; prima.cpp's partial copies on "
                             "the boards: testbed/dataset/prima_copies_qwen3-32b-q8_0.json)",
                             "2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169")}


def main():
    p = os.path.join(DS, "model_shapes.json")
    keep = os.path.join(DS, "model_shapes.20261021_before_35b_32b.json")
    if not os.path.exists(keep):
        shutil.copy2(p, keep); os.chmod(keep, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        print("kept model_shapes.json as %s (read-only)" % os.path.basename(keep))
    d = json.load(open(p))
    for key, (f, path, sha) in SHAPES.items():
        s = gguf.shape(os.path.join(CACHE, f))
        if s["architecture"] == "qwen35moe":
            kv = gguf.read_metadata(os.path.join(CACHE, f))
            s.update(kv_bytes_per_layer_per_token=2 * s["n_head_kv"] * s["head_dim"] * 2, activation_bytes=s["n_embd"] * 4,
                     nextn_layers=int(kv["qwen35moe.nextn_predict_layers"]))
        s.update(name=key, path=path, read_from="the laptop copy %s, sha256 %s..., verified against the Hugging Face LFS oid at its "
                                                 "pinned revision (testbed/models.md)" % (f, sha[:16]))
        d[key] = s
        print("%s: %s, %d blocks, %.1f MiB per block, head %.1f MiB, kv %d B/token/block" % (
            key, s["architecture"], s["n_layers"], s["bytes_per_layer"] / 2 ** 20, s["head_bytes"] / 2 ** 20,
            s["kv_bytes_per_layer_per_token"]))
    json.dump(d, open(p, "w"))


if __name__ == "__main__":
    main()
