#!/usr/bin/env python3
"""The cluster's topology as a figure (the advisor, 2026-10-16: "topology visualization"). Laptop only; every label is read
from a dataset or config file, nothing is typed in.

    python3 testbed/scripts/topology_figure_20261020.py [--plans FILE]

Left: the four nodes on the home router's switch -- each node's processor, memory and serving device (cluster.json and
memory_at_rest.json / the newest memory_snapshot), its strip outlet (testbed/energy/strip.json), and each measured pair's
throughput and RTT (testbed/dataset/link.json). Right: layer maps of served plans -- which board holds which blocks, where the
output head is -- from full-evaluation registrations (testbed/dataset/full_eval_*.json, their "configs"), the newest first,
or from --plans (a JSON list of {"label", "assign": [[node, blocks], ...], "head": "host"|"last", "n_layers"}).
Writes testbed/results/topology_<stamp>/topology.{png,svg} and topology.json (what was drawn and from where).
"""
import argparse, datetime, glob, json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DS = os.path.join(ROOT, "testbed", "dataset")
NODE = {"pi5": ("Raspberry Pi 5", "4x Cortex-A76 2.4 GHz", "CPU"),
        "pi4": ("Raspberry Pi 4", "4x Cortex-A72 1.8 GHz", "CPU"),
        "jetson": ("Jetson Nano", "128-core Maxwell GPU", "Vulkan GPU worker"),
        "laptop": ("ThinkPad T580", "i5-8350U, 4C/8T", "control + CPU node")}
COL = {"pi5": "#2a78d6", "pi4": "#e8590c", "jetson": "#2f9e44", "laptop": "#7048e8"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(ROOT, "testbed", "results", "topology_%s" % stamp)
    os.makedirs(out)
    link = json.load(open(os.path.join(DS, "link.json")))["pairs"]
    strip = json.load(open(os.path.join(ROOT, "testbed", "energy", "strip.json")))["outlets"]
    outlet = {v: k for k, v in strip.items()}
    snaps = sorted(glob.glob(os.path.join(DS, "memory_snapshot_2*.json")))
    snap = json.load(open(snaps[-1])) if snaps else {}
    rest = json.load(open(os.path.join(DS, "memory_at_rest.json")))
    mem = {}
    for n in NODE:
        src = snap.get(n) or rest.get(n)
        mem[n] = (src or {}).get("mem_available_mib"), (src or {}).get("mem_total_mib")
    if a.plans:
        plans = json.load(open(a.plans))
    else:
        plans = []
        for f in sorted(glob.glob(os.path.join(DS, "full_eval_*.json")), reverse=True):
            r = json.load(open(f))
            if "smoke" in f or "tiny" in f:
                continue
            shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
            for lab, c in r["configs"].items():
                plans.append({"label": "%s %s" % (r["model"], lab), "assign": c["assign"], "head": c["head"],
                              "n_layers": sum(k for _, k in c["assign"]), "source": os.path.relpath(f, ROOT)})
            if len(plans) >= 6:
                break
    fig = plt.figure(figsize=(13, 6.2))
    ax = fig.add_axes([0.0, 0.0, 0.55, 1.0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    pos = {"laptop": (0.20, 0.74), "jetson": (0.80, 0.74), "pi5": (0.20, 0.36), "pi4": (0.80, 0.36)}
    sw = (0.50, 0.55)
    ax.add_patch(FancyBboxPatch((sw[0] - 0.10, sw[1] - 0.045), 0.20, 0.09, boxstyle="round,pad=0.01", fc="#f1f0ec", ec="#52514e"))
    ax.text(sw[0], sw[1] + 0.010, "home router", ha="center", fontsize=9, weight="bold")
    ax.text(sw[0], sw[1] - 0.022, "gigabit switch, 192.168.1.0/24", ha="center", fontsize=6.5, color="#52514e")
    drawn = {}
    for n, (x, y) in pos.items():
        ax.plot([x, sw[0]], [y, sw[1]], color="#b5b3ad", lw=2, zorder=0)
        ax.add_patch(FancyBboxPatch((x - 0.17, y - 0.085), 0.34, 0.17, boxstyle="round,pad=0.01", fc="white", ec=COL[n], lw=2))
        t, cpu, role = NODE[n]
        av, tot = mem[n]
        ax.text(x, y + 0.050, t, ha="center", fontsize=10, weight="bold", color=COL[n])
        ax.text(x, y + 0.017, "%s; %s" % (cpu, role), ha="center", fontsize=6.8)
        ax.text(x, y - 0.015, ("%s of %s MiB free at rest" % ("{:,}".format(av), "{:,}".format(tot))) if av else
                "free memory at rest: not yet measured", ha="center", fontsize=6.8)
        ax.text(x, y - 0.048, "strip outlet %s" % outlet.get(n, "?"), ha="center", fontsize=6.8, color="#52514e")
        drawn[n] = {"title": t, "cpu": cpu, "role": role, "mem_available_mib": av, "mem_total_mib": tot, "outlet": outlet.get(n)}
    pairs = []
    rows = [(k, v) for k, v in sorted(link.items()) if all(x in pos for x in k.split("-"))]
    ax.text(0.04, 0.205, "Measured pairs (measure_link.sh: throughput of the slower direction; RTT both ways)", fontsize=7.5,
            weight="bold")
    for i, (k, v) in enumerate(rows):
        cx, cy = 0.04 + 0.47 * (i % 2), 0.165 - 0.042 * (i // 2)
        ax.text(cx, cy, "%-14s %5.0f Mbit/s   %4.2f ms   (%s)" % (k.replace("-", " – "), v["mbps"], v["rtt_ms"], v["measured"][:10]),
                fontsize=7, family="DejaVu Sans Mono")
        pairs.append({"pair": k, "mbps": v["mbps"], "rtt_ms": v["rtt_ms"], "measured": v["measured"]})
    ax.text(0.04, 0.03, "RPC between two boards passes through the host; the Pi 5, Pi 4 and laptop also share the 192.168.10.0/24 "
            "overlay on the same cables", fontsize=6.5, color="#52514e")
    ax.text(0.02, 0.975, "HeteroServe cluster, %s" % datetime.date.today().isoformat(), fontsize=12, weight="bold", va="top")
    ax2 = fig.add_axes([0.70, 0.14, 0.28, 0.74])
    for i, p in enumerate(plans[:6]):
        y = len(plans[:6]) - 1 - i
        x0 = 0
        for nd, k in p["assign"]:
            ax2.barh(y, k, left=x0, color=COL.get(nd, "#999"), edgecolor="white", height=0.6)
            ax2.text(x0 + k / 2, y, "%s %d" % (nd, k), ha="center", va="center", fontsize=7, color="white", weight="bold")
            x0 += k
        hx = 0 if p["head"] == "host" else x0
        ax2.plot([hx], [y + 0.38], marker="v", color="#0b0b0b", ms=6)
    ax2.set_yticks(range(len(plans[:6]))); ax2.set_yticklabels([p["label"] for p in plans[:6]][::-1], fontsize=8)
    ax2.set_xlabel("blocks, in pipeline order (the host first); ▼ = the output head", fontsize=8)
    ax2.set_title("Layer maps of registered plans", loc="left", fontsize=10, weight="bold")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    fig.savefig(os.path.join(out, "topology.png"), dpi=200); fig.savefig(os.path.join(out, "topology.svg"))
    json.dump({"made": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "nodes": drawn, "pairs": pairs,
               "plans": plans[:6], "memory_from": os.path.relpath(snaps[-1], ROOT) if snaps else "testbed/dataset/memory_at_rest.json",
               "link_from": "testbed/dataset/link.json", "outlets_from": "testbed/energy/strip.json"},
              open(os.path.join(out, "topology.json"), "w"), indent=1)
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
