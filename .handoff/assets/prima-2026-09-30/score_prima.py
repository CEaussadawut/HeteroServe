#!/usr/bin/env python3
"""Score the prima.cpp comparison against testbed/dataset/prima_predictions_20260930.json, its rules applied as written:

    python3 .handoff/assets/prima-2026-09-30/score_prima.py

Arms A and B: the newest log of each arm's registered glob, summarised by qr3_tpot.summary() (the code that printed each run's
summary). D: QR3's measured figures, as registered. CORRECTNESS reads the token texts: prima.cpp streams text per token;
llama.cpp's token ids are turned into text with the model's own vocabulary (GPT-2 byte-level BPE, read from the GGUF).
Writes testbed/dataset/prima_results_20260930.json.
"""
import glob, json, os, re, sys, types

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
import qr3_tpot as Q  # noqa: E402

REG = os.path.join(ROOT, "testbed", "dataset", "prima_predictions_20260930.json")
OUT = os.path.join(ROOT, "testbed", "dataset", "prima_results_20260930.json")
GGUF = os.path.expanduser("~/.cache/heteroserve-models/Qwen3-14B-Q4_K_M.gguf")   # the pinned bytes (models.md)
SMOKE = "testbed/results/qr3_primasmoke_qwen3_14b_2*.log"


def vocab():
    sys.modules.setdefault("sentencepiece", types.SimpleNamespace(SentencePieceProcessor=None))   # the reader never uses it
    # 2026-10-08: the pinned llama.cpp's gguf-py, not prima.cpp's vendored copy. prima.cpp's calls ndarray.newbyteorder, which
    # NumPy 2.0 removed, and the laptop now has NumPy 2.3. Both read the same tokenizer strings from the same pinned file;
    # reproduce.py verify re-derives prima_results_20260930.json with this reader.
    sys.path.insert(0, "/opt/heteroserve/llama.cpp/gguf-py")
    from gguf import GGUFReader
    f = GGUFReader(GGUF).fields["tokenizer.ggml.tokens"]
    toks = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    dec = {chr(c): b for b, c in zip(bs, cs)}
    return lambda i: bytes(dec[ch] for ch in toks[i]).decode("utf-8", "replace")


def req(path, i=0):
    for line in open(path):
        p = line.split(" ", 3)
        if len(p) > 3 and p[1] == "req" and p[2] == str(i):
            return json.loads(p[3])


def newest(pattern):
    logs = [l for l in sorted(glob.glob(os.path.join(ROOT, pattern))) if not re.search(r"_(summary|server|prima_rank\d|memprobe_\w+)\.(log|json)$", l)]
    return logs[-1] if logs else None


def windows(path):
    """Halda's split, from the 'prima ready' lines of a prima.cpp run's log: {node: window}."""
    out = {}
    for line in open(path):
        p = line.split(" ", 3)
        if len(p) > 3 and p[1] == "prima" and p[2] == "ready":
            o = json.loads(p[3])
            m = re.search(r"Using window size: (\d+)", o.get("window", ""))
            out[o["node"]] = int(m.group(1)) if m else o.get("window")
    return out


def main():
    reg = json.load(open(REG))
    out = {"registration": os.path.relpath(REG, ROOT), "arms": {}, "rules": {}}
    A, B = reg["arms"]["A_ours_pis"], reg["arms"]["B_prima_halda"]
    la, lb = newest(A["log_glob"]), newest(B["log_glob"])
    for name, log in (("A_ours_pis", la), ("B_prima_halda", lb)):
        if not log:
            out["arms"][name] = {"missing": True}; continue
        s = Q.summary(log)
        full = [r for r in s["requests"] if r["tokens"] == 128]
        e = {"log": s["log"], "client": s["pooled"]["client"], "server": s["pooled"]["server"],
             "ttft_client_ms": s["ttft_client_ms"], "per_request_mean_tpot": s["per_request_mean_tpot"],
             "requests": len(s["requests"]), "complete_requests": len(full), "merged_events": s["merged_events"]}
        if name == "A_ours_pis":
            served = s["meta"].get("served", {})
            e["served_args"] = "%s -c %s -np 1 -t 4" % (served.get("args"), served.get("ctx"))
            e["plan_as_registered"] = e["served_args"] == A["server_args"]
        else:
            e["halda_split"] = windows(log)
            tp = [req(log, i).get("tokens_predicted") for i in range(10) if req(log, i)]
            e["tokens_predicted"] = tp
        out["arms"][name] = e
    out["arms"]["D_ours_full_reference"] = reg["arms"]["D_ours_full_reference"]["measured"]
    a, b = out["arms"]["A_ours_pis"], out["arms"]["B_prima_halda"]
    R = out["rules"]
    # CORRECTNESS: prima.cpp's first 8 greedy tokens on request 0 against arm A's request 0, as text
    if la:
        detok = vocab()
        at = [detok(t) for e in req(la)["events"] for t in e[1]]
        cmp = {}
        for tag, path in (("smoke", newest(SMOKE)), ("B", lb)):
            if path:
                pt = [t for e in req(path)["events"] for t in e[1]]
                same = 0
                for x, y in zip(at, pt):
                    if x != y:
                        break
                    same += 1
                cmp[tag] = {"identical_leading_tokens": same, "of": min(len(at), len(pt)), "holds": same >= 8,
                            "llama_first_16": "".join(at[:16]), "prima_first_16": "".join(pt[:16])}
        R["CORRECTNESS"] = cmp
    if a.get("client") and b.get("client"):
        am, bm = a["client"]["mean"], b["client"]["mean"]
        R["DECODE"] = {"A_mean_ms": am, "B_mean_ms": bm, "faster": "A" if am < bm else "B",
                       "A_less_than_B_pct": round(100 * (1 - am / bm), 1), "B_over_A": round(bm / am, 3), "predicted": "A",
                       "as_predicted": am < bm}
        at_, bt_ = a["ttft_client_ms"]["mean"], b["ttft_client_ms"]["mean"]
        R["TTFT"] = {"A_mean_ms": at_, "B_mean_ms": bt_, "faster": "A" if at_ < bt_ else "B",
                     "A_less_than_B_pct": round(100 * (1 - at_ / bt_), 1), "predicted": "A", "as_predicted": at_ < bt_}
        R["QR3"] = {k: {"p95": v["client"]["p95"], "ci": v["client"].get("p95_ci95"),
                        "MVP": "met" if v["client"]["p95"] <= 500 else "not met",
                        "STRETCH": "met" if v["client"]["p95"] <= 250 else "not met"} for k, v in (("A", a), ("B", b))}
        R["QR3"]["predicted"] = "neither meets 500"
    if b.get("halda_split"):
        w5 = b["halda_split"].get("pi5")
        R["HALDA-SPLIT"] = {"split": b["halda_split"], "pi5_blocks": w5, "predicted": "pi5 <= 30",
                            "holds": isinstance(w5, int) and w5 <= 30}
        if newest(SMOKE):
            R["HALDA-SPLIT"]["smoke_split"] = windows(newest(SMOKE))
    if a.get("server"):
        pred = A["predicted"]["decode_ms_per_token"]
        R["ACCURACY"] = {"A_server_mean_ms": a["server"]["mean"], "predicted_ms": pred,
                         "error_pct": round(100 * (pred - a["server"]["mean"]) / a["server"]["mean"], 2),
                         "holds": abs(100 * (pred - a["server"]["mean"]) / a["server"]["mean"]) <= 10,
                         "plan_as_registered": a.get("plan_as_registered")}
    R["COMPLETE"] = {"A": a.get("complete_requests") == a.get("requests") == 10,
                     "B": b.get("complete_requests") == b.get("requests") == 10 and all(t == 128 for t in b.get("tokens_predicted", [])),
                     "B_tokens_predicted": b.get("tokens_predicted")}
    R["REFERENCE_D"] = out["arms"]["D_ours_full_reference"]
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(R, indent=1))
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
