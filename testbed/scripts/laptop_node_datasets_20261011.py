#!/usr/bin/env python3
"""The dataset side of making the laptop a node (2026-10-11, the user: "make it as one of the node"). Each step keeps the
file it changes as a dated, read-only copy first, as link.20261008_t628l.json was kept: every earlier registration quotes
the old bytes.

    python3 testbed/scripts/laptop_node_datasets_20261011.py memory      # the laptop's MemAvailable now, into memory_at_rest.json
    python3 testbed/scripts/laptop_node_datasets_20261011.py links LOG LOG   # the pi4-laptop and laptop-jetson logs, into link.json
    python3 testbed/scripts/laptop_node_datasets_20261011.py shapes      # the two Q6_K files, into model_shapes.json

memory: MemAvailable of THIS laptop now, read when it is quiet (no apps but the session), and its MemTotal. The planner keeps
  cluster.json's memory_reserve_mb (8 GiB, for the desktop) on top of it; the boards' figures are unchanged (2026-09-27).
links: link_to_json.py over the three logs link.json was built from plus the two new ones. The three old pairs come out
  byte-identical (checked 2026-10-11 03:56: only the _source text, updated 2026-10-09, differs).
shapes: control/gguf.shape() of the laptop's copies -- it reproduces the published Q3_K_M MoE entry exactly (checked
  2026-10-11) -- with the name, path and read_from fields the other entries carry.
"""
import json, os, shutil, stat, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(ROOT, "testbed", "dataset")
CACHE = os.path.expanduser("~/.cache/heteroserve-models")
OLD_LINK_LOGS = ["testbed/results/link_20260927_195141.log", "testbed/results/link_pi5-jetson_20260927_203819.log",
                 "testbed/results/link_pi4-jetson_20261008_180505.log"]
SHAPES = {"tinyllama-1.1b-q6_k": ("tinyllama-1.1b-chat-v1.0.Q6_K.gguf", "~/models/tinyllama-1.1b-q6_k.gguf (on pi4, jetson, laptop)"),
          "qwen3-30b-a3b-q6_k": ("Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf", "~/models/qwen3-30b-a3b-q6_k.gguf (on the laptop)")}


def keep(name, copy_name):
    src, dst = os.path.join(DS, name), os.path.join(DS, copy_name)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
        os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        print("kept %s as %s (read-only)" % (name, copy_name))


def memory():
    keep("memory_at_rest.json", "memory_at_rest.20260927.json")
    p = os.path.join(DS, "memory_at_rest.json")
    d = json.load(open(p))
    mi = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in open("/proc/meminfo")}
    apps = subprocess.run(["bash", "-c", "ps -eo rss,comm --sort=-rss | head -8"], capture_output=True, text=True).stdout
    d["laptop"] = {"mem_total_mib": mi["MemTotal"], "mem_available_mib": mi["MemAvailable"],
                   "note": "the T580 as a node, 2026-10-11 (%s): read here with the desktop session up and no apps open but the "
                           "Claude app; the planner keeps cluster.json's 8 GiB desktop reserve on top. The largest processes then: %s"
                           % (subprocess.run(["date", "-Is"], capture_output=True, text=True).stdout.strip(),
                              "; ".join(" ".join(l.split()) for l in apps.splitlines()[1:6]))}
    json.dump(d, open(p, "w"), indent=1)
    print("memory_at_rest.json: laptop %s" % d["laptop"])


def links(new_logs):
    keep("link.json", "link.20261011_before_laptop.json")
    r = subprocess.run([sys.executable, os.path.join(HERE, "link_to_json.py")] + OLD_LINK_LOGS + list(new_logs)
                       + ["--out", os.path.join(DS, "link.json")], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout + r.stderr)
    old = json.load(open(os.path.join(DS, "link.20261011_before_laptop.json")))
    new = json.load(open(os.path.join(DS, "link.json")))
    assert all(new["pairs"][k] == v for k, v in old["pairs"].items()) and new["nodes"] == old["nodes"], "an old pair changed"
    print("the three old pairs and the nodes view are unchanged; new: %s" % sorted(set(new["pairs"]) - set(old["pairs"])))


def shapes():
    sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
    from control import gguf
    keep("model_shapes.json", "model_shapes.20261011_before_q6k.json")
    p = os.path.join(DS, "model_shapes.json")
    d = json.load(open(p))
    for key, (f, path) in SHAPES.items():
        s = gguf.shape(os.path.join(CACHE, f))
        s.update(name=key, path=path, read_from="the laptop copy %s, sha256-verified against the Hugging Face LFS oid at its pinned "
                                                 "revision (testbed/models.md)" % f)
        d[key] = s
        print("%s: %d layers, %.1f MiB per block, head %.1f MiB" % (key, s["n_layers"], s["bytes_per_layer"] / 2**20, s["head_bytes"] / 2**20))
    json.dump(d, open(p, "w"))            # the file's own format: one line, no indent, the old entries byte-identical


if __name__ == "__main__":
    a = sys.argv[1:]
    {"memory": lambda: memory(), "links": lambda: links(a[1:]), "shapes": lambda: shapes()}[a[0]]()
