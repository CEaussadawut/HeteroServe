#!/usr/bin/env python3
"""Qwen3-32B at Q6_K and Q4_K_M, into the dataset's model_shapes.json (as models_20261021.py): scheduler v2's precision
choice (the user, 2026-10-21 ~23:40: "Both files", "near-lossless only"). Qwen/Qwen3-32B-GGUF @ 938a7432, the Q8_0's source
and revision, read from the laptop's data drive (/mnt/data/heteroserve-models; ~/models links to them), each sha256-verified
against the LFS oid Hugging Face publishes at that revision (fetch_32b_q6k_q4km_20261021.log there; testbed/models.md). The
file keeps its own format (one line), the old entries byte-identical; the previous file is kept read-only beside it first.

    python3 testbed/scripts/models_20261022.py
"""
import json, os, shutil, stat, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(ROOT, "testbed", "dataset")
DIR = "/mnt/data/heteroserve-models"
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
from control import gguf                                   # noqa: E402

SHAPES = {"qwen3-32b-q6_k": ("Qwen3-32B-Q6_K.gguf", "c4c7c3cb6da260df1fe1d3cfd090a32dc7cc348f1278158be18e301f390d6f6e"),
          "qwen3-32b-q4_k_m": ("Qwen3-32B-Q4_K_M.gguf", "efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689")}


def main():
    p = os.path.join(DS, "model_shapes.json")
    keep = os.path.join(DS, "model_shapes.20261022_before_32b_q6k_q4km.json")
    if not os.path.exists(keep):
        shutil.copy2(p, keep); os.chmod(keep, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        print("kept model_shapes.json as %s (read-only)" % os.path.basename(keep))
    d = json.load(open(p))
    for key, (f, sha) in SHAPES.items():
        s = gguf.shape(os.path.join(DIR, f))
        s.update(name=key, path="~/models/%s.gguf (on the laptop; the file on its data drive, %s)" % (key, DIR),
                 read_from="the laptop copy %s/%s, sha256 %s..., verified against the Hugging Face LFS oid at its pinned revision "
                           "(testbed/models.md)" % (DIR, f, sha[:16]))
        d[key] = s
        print("%s: %s, %d blocks, %.1f MiB per block (min %.1f max %.1f), head %.1f MiB, embd %.1f MiB" % (
            key, s["architecture"], s["n_layers"], s["bytes_per_layer"] / 2 ** 20, min(s["layer_bytes"]) / 2 ** 20,
            max(s["layer_bytes"]) / 2 ** 20, s["head_bytes"] / 2 ** 20, s["embd_bytes"] / 2 ** 20))
    json.dump(d, open(p, "w"))


if __name__ == "__main__":
    main()
