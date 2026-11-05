#!/usr/bin/env python3
"""FLOPs per token, and what each board achieves (the advisor, 2026-10-16: "flops"). Laptop only; reads the dataset.

    python3 testbed/scripts/flops_per_token_20261020.py

Per model (testbed/dataset/model_shapes.json), counted from its tensor table:
  decode, per token:  matmul ops = 2 x (n_layers x active params per block + output-head params): a multiply and an add per
                      weight (the token embedding is a row read, no arithmetic); attention at context depth c adds
                      4 x n_layers x n_head x head_dim x c (Q.K and the weighted sum of V), for full-attention blocks only (a hybrid's
                      attention blocks are not marked in model_shapes.json, so its term is not given);
                      bytes read = n_layers x active bytes per block + head bytes (+ the KV cache: kv bytes x n_layers x c);
                      arithmetic intensity = ops / bytes.
  prefill, per prompt of n tokens: 2 x params x n + attention 2 x n_layers x n_head x head_dim x n^2 (causal: half of 4 x n^2).
Per board (testbed/dataset/characterisation.csv, host rows, both protocols), MEASURED rates turned into work:
  prefill GOP/s = matmul ops per token x tokens/s;  decode GB/s = bytes read per token x tokens/s, and its GOP/s.
Nominal DRAM bandwidth (bus width x transfer rate, from each board's specification -- NOT measured here) gives decode's
fraction of the memory roof. No compute peak is used: the kernels differ by board and quantisation (int8 dot products on the
Pi 5, none on the Pi 4, FP16 on the Jetson's GPU), so a single peak would mislead.
Writes testbed/results/flops_<stamp>/{flops.json, report.txt, roofline.png}.
"""
import csv, datetime, json, os, statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DS = os.path.join(ROOT, "testbed", "dataset")
# (bus bits, MT/s, memory type): each board's specification
DRAM = {"pi5": (32, 4267, "LPDDR4X-4267, 32-bit (Raspberry Pi 5 product brief)"),
        "pi4": (32, 3200, "LPDDR4-3200, 32-bit (Raspberry Pi 4 product brief)"),
        "jetson": (64, 3200, "LPDDR4 at 1600 MHz, 64-bit (Jetson Nano module data sheet), shared by the CPU and the GPU"),
        "t580": (128, 2400, "DDR4-2400, two 64-bit channels if both slots are filled (32 GB installed; not verified)")}
DEVICE = {"pi5": "cpu", "pi4": "cpu", "jetson": "vulkan", "t580": "cpu"}      # each board's serving device
FILE_TO_SHAPE = {"tinyllama-1.1b-q4km.gguf": "tinyllama-1.1b-q4km", "tinyllama-1.1b-q8_0.gguf": "tinyllama-1.1b-q8_0",
                 "tinyllama-1.1b-q4_0.gguf": "tinyllama-1.1b-q4_0", "tinyllama-1.1b-q3km.gguf": "tinyllama-1.1b-q3km",
                 "tinyllama-1.1b-q6_k.gguf": "tinyllama-1.1b-q6_k", "qwen3.5-9b-q4_k_m.gguf": "qwen3.5-9b-q4_k_m"}
DENSE = ("llama", "qwen2", "qwen3", "qwen3moe")   # full attention in every block (the MoE: its FFN only is experts)


def per_token(q):
    L = q["n_layers"]
    p = q.get("active_params_per_layer") or q["params_per_layer"]
    b = q.get("active_bytes_per_layer") or q["bytes_per_layer"]
    ops = 2.0 * (L * p + q["head_params"])
    byt = L * b + q["head_bytes"]
    att = (4.0 * L * q["n_head"] * q["head_dim"]) if q["architecture"] in DENSE else None
    return ops, byt, att


def main():
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(ROOT, "testbed", "results", "flops_%s" % stamp)
    os.makedirs(out)
    shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
    models = {}
    for k, q in shapes.items():
        if k.startswith("_") or "qwen2.5" in k or "mistral-7b" in k:      # Qwen3 and later, and TinyLlama, as documents show
            continue
        ops, byt, att = per_token(q)
        kv = q["kv_bytes_per_layer_per_token"] * q["n_layers"]
        models[k] = {"architecture": q["architecture"], "n_layers": q["n_layers"], "params": q["n_params"],
                     "active_params_per_token": round(ops / 2), "decode_gop_per_token": round(ops / 1e9, 3),
                     "decode_bytes_per_token_mb": round(byt / 1e6, 1), "intensity_ops_per_byte": round(ops / byt, 2),
                     "attention_gop_per_token_at": {str(c): (round(att * c / 1e9, 4) if att else None) for c in (128, 1024, 4096)},
                     "kv_bytes_per_token_at_4096_mb": round(kv * 4096 / 1e6, 1),
                     "prefill_gop_per_512_prompt": round((ops * 512 + (att / 2 * 512 * 512 if att else 0)) / 1e9, 1)}
    rows = list(csv.DictReader(open(os.path.join(DS, "characterisation.csv"))))
    boards = {}
    for r in rows:
        n, dev = r["node_id"], r["device"]
        if DEVICE.get(n) != dev or r["model"] not in FILE_TO_SHAPE:
            continue
        q = shapes[FILE_TO_SHAPE[r["model"]]]
        ops, byt, _ = per_token(q)
        ts = float(r["mean_ts"])
        key = "%s|%s|%s" % (r["model"], r["test"], r["source"].split("/")[-2])
        e = {"model": r["model"], "quant": r["quant"], "test": r["test"], "phase": r["phase"], "tokens_per_s": ts,
             "gop_per_s": round(ops * ts / 1e9, 2), "source": r["source"], "date": r["date"][:10]}
        if r["phase"] == "tg":
            gbs = byt * ts / 1e9
            bits, mts, note = DRAM[n]
            peak = bits / 8 * mts / 1e3
            e.update({"gb_per_s": round(gbs, 2), "nominal_dram_gb_per_s": round(peak, 1), "fraction_of_dram_pct": round(100 * gbs / peak, 1)})
        boards.setdefault(n, {"device": dev, "dram": DRAM[n][2], "nominal_dram_gb_per_s": round(DRAM[n][0] / 8 * DRAM[n][1] / 1e3, 1),
                              "rows": {}})["rows"][key] = e
    res = {"made": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/flops_per_token_20261020.py", "models": models, "boards": boards,
           "note": "ops count a multiply and an add per weight per token; measured rates are the dataset's tokens/s; the DRAM "
                   "roof is each board's specification, not measured. The Pi 5 rows are the OLD board's (2026-09-27..29) until "
                   "the replacement's characterisation is merged."}
    json.dump(res, open(os.path.join(out, "flops.json"), "w"), indent=1)
    L = ["FLOPs per token (counted from model_shapes.json; Qwen3 and later, and TinyLlama)", "",
         "%-24s %8s %10s %10s %9s %22s %12s" % ("model", "layers", "GOP/token", "MB/token", "ops/byte", "attention GOP @128/1k/4k",
                                               "prefill 512")]
    for k, m in models.items():
        a = m["attention_gop_per_token_at"]
        L.append("%-24s %8d %10.3f %10.1f %9.2f %22s %10.1f G" % (k, m["n_layers"], m["decode_gop_per_token"], m["decode_bytes_per_token_mb"],
                                                                 m["intensity_ops_per_byte"],
                                                                 "/".join("%.3f" % a[c] if a[c] is not None else "n/a" for c in ("128", "1024", "4096")),
                                                                 m["prefill_gop_per_512_prompt"]))
    L += ["", "Achieved, per board (serving device; host rows): prefill GOP/s; decode GB/s and its share of the DRAM roof"]
    for n, b in boards.items():
        L.append("%s (%s; DRAM roof %.1f GB/s, %s)" % (n, b["device"], b["nominal_dram_gb_per_s"], b["dram"]))
        for key, e in sorted(b["rows"].items()):
            if e["phase"] == "pp":
                L.append("    %-26s %-6s %7.2f t/s  %8.1f GOP/s" % (e["model"], e["test"], e["tokens_per_s"], e["gop_per_s"]))
            else:
                L.append("    %-26s %-6s %7.2f t/s  %8.1f GOP/s  %6.2f GB/s = %5.1f %% of the roof" % (
                    e["model"], e["test"], e["tokens_per_s"], e["gop_per_s"], e["gb_per_s"], e["fraction_of_dram_pct"]))
    open(os.path.join(out, "report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(boards), figsize=(3.2 * len(boards), 3.2), sharey=True)
        for ax, (n, b) in zip(axs, boards.items()):
            roof = b["nominal_dram_gb_per_s"]
            xs = [0.5, 2000]
            ax.loglog(xs, [roof * x for x in xs], "--", color="#52514e", lw=1, label="DRAM roof (spec)")
            for e in b["rows"].values():
                if not e["test"] in ("pp512", "tg128"):
                    continue
                q = shapes[FILE_TO_SHAPE[e["model"]]]
                ops, byt, _ = per_token(q)
                ai = ops / byt * (512 if e["phase"] == "pp" else 1)        # a 512-token batch reuses each weight 512 times
                ax.loglog([ai], [e["gop_per_s"]], "o" if e["phase"] == "tg" else "s",
                          color="#2a78d6" if e["phase"] == "tg" else "#e8590c", ms=4)
            ax.set_title("%s (%s)" % (n, b["device"]), loc="left", fontsize=9)
            ax.set_xlabel("ops per byte of weights", fontsize=8); ax.grid(True, color="#e4e3df", lw=0.5)
        axs[0].set_ylabel("achieved GOP/s"); axs[0].plot([], [], "o", color="#2a78d6", label="decode (tg128)")
        axs[0].plot([], [], "s", color="#e8590c", label="prefill (pp512)"); axs[0].legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, "roofline.png"), dpi=200)
    except Exception as ex:
        print("no figure:", ex)
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
