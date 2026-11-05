#!/usr/bin/env python3
"""Energy per token of a pooled plan, modeled from single-board measurements. Read-only.

    python3 testbed/scripts/energy_model.py validate     # the model against every measured pooled cell (2026-10-05/06)
    python3 testbed/scripts/energy_model.py prima        # the 2026-09-30 head-to-head (Qwen3-14B, Pi 5 + Pi 4), modeled
    python3 testbed/scripts/energy_model.py offline      # the engine against Halda on today's boards (Pi 4 + Jetson)
    python3 testbed/scripts/energy_model.py power        # the per-board power table the model reads

The model is the planner's own (placement.evaluate): each board draws its measured ACTIVE power while it computes its share
of a token, and its measured IDLE power while it waits for the others. Per token:
    gross J = sum over the boards of  P_active x busy + P_wait x (T - busy)
    net J   = gross J - sum over the boards of  idle floor x T       (the strip scorers' "net": above each board's floor)
- P_active per board and PHASE (prefill, decode), and the idle power: the strip's single-board runs of TinyLlama Q4_K_M
  (energy_results_*.json; apower, the trapezoid method). The power a board draws at full load does not depend much on the
  model; that is an assumption, checked below against the pooled 14B cell.
- busy per board: predict_pool.predict's own terms, split by board (per_node(), which is checked to sum to predict()).
  Transfers count as waiting for both boards.
- T: the MEASURED time per token where a run exists (then only the busy FRACTIONS come from the cost model), else the
  predicted one.
- P_wait: the board's idle power in the state it waits in. A Pi waits at its idle floor. The Jetson's clocks are locked in
  every run_pool.sh cell, so it waits at its locked idle (3.14 W), while a strip session's idle window, taken before the
  cells, sees it unlocked (about 2.1 W); "net" subtracts the floor the session measured, as its scorer did.
Everything printed is MODELED unless a column says measured.
"""
import glob, hashlib, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(ROOT, "testbed", "dataset")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as pp                                   # noqa: E402

# The single-board energy runs the power table is read from: TinyLlama Q4_K_M on each board (energy.md, 2026-10-05/06).
# 2026-10-20: the Pi 5 is a replacement board (serial b8ed9049ada12a03) after its bootloader update; its own Q4_K_M run
# replaces the returned board's (energy_results_20261005b.json: 2.31 W idle; this board idles at 3.0 W). Registrations
# before then quote the power table they used, so they are unchanged.
POWER_RUNS = {"pi5": "energy_results_20261020_pi5fw_q4km.json", "pi4": "energy_results_20261005_pi4_q4km.json",
              "jetson": "energy_results_20261005_jetson_vk_q4km.json", "laptop": "energy_results_20261006_laptop_q4km.json"}
POOL_SCORES = ["testbed/results/energy_pool_20261005_215722/score.json",
               "testbed/results/energy_pool_20261006_024600/score.json"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def power_table():
    """{board: {"idle": W, "pp": W, "tg": W, "source": file, "sha256": ...}} from the scored single-board runs (apower)."""
    out = {}
    for b, f in POWER_RUNS.items():
        p = os.path.join(DS, f)
        d = json.load(open(p))
        out[b] = {"idle": d["idle_before"]["mean_w"], "pp": d["phases"]["pp"]["mean_w"]["mean"],
                  "tg": d["phases"]["tg"]["mean_w"]["mean"], "source": "testbed/dataset/" + f, "sha256": sha(p)[:16]}
    return out


def per_node(assign, q, C, links, head_at=None, rtt_trips=pp.RTT_TRIPS_DEFAULT, rpc_msg=None, backend_sampling=False,
             depth=0, attn=None):
    """({node: {"tg": busy ms, "pp": busy ms}}, {"tg": transfer ms, "pp": transfer ms}) per token: predict()'s terms by
    board (no recurrent terms: the dense models only). rtt_trips as in predict() (RTT1, 2026-10-09).
    2026-11-01 (the scheduler review): the head is priced at the rate predict() gives it -- with node_mult "measured_blocks"
    the board's unscaled rate (C[n][role + "_head"]), where this function used the scaled one and its check against
    predict() failed on every such plan (PLANNER_V3's 35B) -- and rpc_msg, backend_sampling, depth and attn are predict()'s
    options: the RPC's wire time counts as transfer, the attention at depth as the board's busy time. Their defaults give
    the terms as before."""
    lb = q.get("layer_active_bytes") or q["layer_bytes"]
    ppl = q.get("active_params_per_layer") or q["params_per_layer"]
    nx = int(q.get("nextn_layers") or 0)
    main = len(q["layer_bytes"]) - nx
    if nx:
        lb = list(lb[:main]) + [0.0] * nx
    n_vocab = (q["head_params"] - q["n_embd"]) / q["n_embd"]
    busy, xfer = {}, {"tg": 0.0, "pp": 0.0}
    start, last = 0, len(assign) - 1
    h = last if head_at is None else head_at
    for i, (n, k) in enumerate(assign):
        role = "host" if i == 0 else "worker"
        pb, ppar = C[n][role]
        tg = pb * sum(lb[start:start + k])
        pre = ppar * ppl * max(0, min(start + k, main) - start)
        if i == h:
            tg += C[n].get(role + "_head", (pb, ppar))[0] * q["head_bytes"]
            pre += ppar * q["head_params"] / pp.PP
        if i > 0:
            bw = links[n]["mbps"] * 1e6 / 8 / 1000
            out = 4 * n_vocab if (i == last and h == last) else q["activation_bytes"]
            if rpc_msg:
                inb, outb, msgs = pp.rpc_bytes(q, i, last, out, depth, rpc_msg, backend_sampling and i == last and h == last)
                wire = (inb + outb) * (1 + rpc_msg["hdr"] / rpc_msg["mss"]) + rpc_msg["hdr"] * msgs
                xfer["tg"] += rtt_trips * links[n]["rtt_ms"] + wire / bw
            else:
                xfer["tg"] += rtt_trips * links[n]["rtt_ms"] + (q["activation_bytes"] + out) / bw
            out_pp = 4 * n_vocab if (i == last and h == last) else pp.PP * q["activation_bytes"]
            xfer["pp"] += (rtt_trips * links[n]["rtt_ms"] + (pp.PP * q["activation_bytes"] + out_pp) / bw) / pp.PP
        if depth and attn:
            tg += depth * attn[n] * max(0, min(start + k, main) - start)
        busy[n] = {"tg": busy.get(n, {}).get("tg", 0.0) + tg, "pp": busy.get(n, {}).get("pp", 0.0) + pre}
        start += k
    dec, pre = pp.predict(assign, q, C, links, head_at=head_at, rtt_trips=rtt_trips, depth=depth, attn=attn,
                          rpc_msg=rpc_msg, backend_sampling=backend_sampling)
    assert abs(sum(b["tg"] for b in busy.values()) + xfer["tg"] - dec) < 1e-6 * max(1.0, dec)
    assert abs(sum(b["pp"] for b in busy.values()) + xfer["pp"] - pre) < 1e-6 * max(1.0, pre)
    return busy, xfer


def energy(assign, q, C, links, power, phase, head_at=None, t_ms=None, wait=None, floor=None, rtt_trips=pp.RTT_TRIPS_DEFAULT):
    """Modeled {"T_ms", "gross_j", "net_j", "board_w": {node: mean W}} per token for one phase ("tg" or "pp")."""
    busy, xfer = per_node(assign, q, C, links, head_at, rtt_trips)
    t_pred = sum(b[phase] for b in busy.values()) + xfer[phase]
    T = t_ms or t_pred
    gross, net, board_w = 0.0, 0.0, {}
    for n, b in busy.items():
        f = min(1.0, b[phase] / t_pred)                          # the busy FRACTION of a token, from the cost model
        pw = (wait or {}).get(n, power[n]["idle"])
        w = power[n][phase] * f + pw * (1 - f)
        board_w[n] = w
        gross += w * T / 1000
        net += (w - (floor or {}).get(n, power[n]["idle"])) * T / 1000
    return {"T_ms": T, "T_pred_ms": t_pred, "gross_j": gross, "net_j": net, "board_w": board_w,
            "busy_frac": {n: busy[n][phase] / t_pred for n in busy}}


# The laptop's power refitted from the four-node evaluation's own strip data (2026-11-01, the scheduler review's item 2;
# energy_refit_20261101.py): the evaluation's 52 placements, all four outlets, within a median -0.1 % and at worst -8.0 % --
# the same left one model out -- where power_table()'s laptop row (a single-board run of 2026-10-06) was 21.6 % low. The
# boards keep power_table(). Opt-in: energy_refit(); energy() above is unchanged, so every registration reproduces.
REFIT = os.path.join(DS, "power_refit_20261101.json")
ALL_NODES = ("laptop", "pi5", "pi4", "jetson")


def refit_table(path=REFIT):
    """power_refit_20261101.json: {"laptop": {"idle_w", "active_w", "per_token_j", "pool_host_w"}, "boards": {...}}."""
    return json.load(open(path))


def laptop_refit_w(lap, f, T_s, role, pool):
    """The laptop's mean W in a decoded token of T_s seconds: outside the plan (role "none") idle_w; in it idle_w + (active_w
    - idle_w) x f, and as the host its serving work per token (per_token_j / T_s) and, hosting workers, pool_host_w."""
    if role == "none":
        return lap["idle_w"]
    w = lap["idle_w"] + (lap["active_w"] - lap["idle_w"]) * f
    if role == "host":
        w += lap["per_token_j"] / T_s + (lap["pool_host_w"] if pool else 0.0)
    return w


def energy_refit(assign, q, C, links, refit=None, head_at=None, t_ms=None, rtt_trips=pp.RTT_TRIPS_DEFAULT, rpc_msg=None,
                 backend_sampling=False, depth=0, attn=None, nodes=ALL_NODES, laptop_idle_w=None):
    """Modeled decode energy per token with the refit: {"T_ms", "T_pred_ms", "plan_j" (the plan's boards), "all_j" (every
    node in `nodes` powered, those outside the plan at idle), "board_w", "busy_frac"}. T: t_ms (a measured time per token)
    or predict()'s; the busy fractions from per_node with the same options as the plan's prediction.
    laptop_idle_w (2026-11-01, after v4a): the laptop's idle as measured at rest before the run; its idle and busy rows move
    by the difference from the refit's (the baseline -- display, desktop -- adds to both). v4a's E-REFIT failed with the
    evaluation's 20.47 W while that session idled at 17.1; over 24 engine arms of 10 sessions the refit was a median +10.4 %
    high, and with each session's own idle -1.2 % (21 within 10 %; energy_holdout_20261101.py, post hoc). None: as fitted."""
    refit = refit or refit_table()
    lap, boards = refit["laptop"], refit["boards"]
    if laptop_idle_w is not None:
        lap = dict(lap, idle_w=laptop_idle_w, active_w=lap["active_w"] + laptop_idle_w - lap["idle_w"])
    busy, xfer = per_node(assign, q, C, links, head_at, rtt_trips, rpc_msg, backend_sampling, depth, attn)
    t_pred = sum(b["tg"] for b in busy.values()) + xfer["tg"]
    T = t_ms or t_pred
    members = [n for n, _ in assign]
    board_w = {}
    for n in nodes:
        f = min(1.0, busy[n]["tg"] / t_pred) if n in busy else None
        if n == "laptop":
            role = "none" if f is None else ("host" if members[0] == "laptop" else "worker")
            board_w[n] = laptop_refit_w(lap, f or 0.0, T / 1000.0, role, role == "host" and len(members) > 1)
        else:
            b = boards[n]
            board_w[n] = b["idle"] if f is None else b["tg"] * f + b["idle"] * (1 - f)
    return {"T_ms": T, "T_pred_ms": t_pred, "plan_j": sum(board_w[n] for n in members) * T / 1000.0,
            "all_j": sum(board_w.values()) * T / 1000.0, "board_w": board_w,
            "busy_frac": {n: busy[n]["tg"] / t_pred for n in busy}}


def _inputs(quant):
    shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
    pairs = json.load(open(os.path.join(DS, "link.json")))["pairs"]
    R = pp.rows(os.path.join(DS, "characterisation.csv"), "char")
    C, _ = pp.costs(R, shapes, quant)
    return shapes, pairs, C


def validate():
    """Modeled against measured, for every pooled energy run that exited 0 (trapezoid = apower, as the power table)."""
    power = power_table()
    rows = []
    for f in POOL_SCORES:
        sc = json.load(open(os.path.join(ROOT, f)))
        reg = json.load(open(os.path.join(ROOT, sc["registration"])))
        for r in sc["runs"]:
            if r.get("exit") != 0:
                continue
            c = reg["configs"][r["cell"]]
            key = {"tinyllama-1.1b-q4km": "tinyllama", "qwen3-14b-q4_k_m": "qwen3_14b"}[c["model"]]
            shape, quant, _ = pp.MODELS[key]
            shapes, pairs, C = _inputs(quant)
            q = shapes[shape]
            assign = [(s["node"], s["blocks"]) for s in c["pipeline"]]
            L, _ = pp.link_view(pairs, assign[0][0], [n for n, _ in assign])
            for ph in ("pp", "tg"):
                m = r["phases"][ph]
                T = 1000.0 / m["bench"]["avg_ts"]
                floor = {n: m["per_board"][n]["idle_mean_w_trap"] for n in m["per_board"]}
                e = energy(assign, q, C, L, power, ph, t_ms=T, wait={"jetson": power["jetson"]["idle"]}, floor=floor)
                rows.append({"run": "%s %s" % (os.path.basename(os.path.dirname(f))[-15:], r["label"]), "phase": ph,
                             "measured_gross": m["pool"]["j_per_token_gross_trap"], "model_gross": e["gross_j"],
                             "measured_net": m["pool"]["j_per_token_net_trap"], "model_net": e["net_j"],
                             "board_w": {n: (m["per_board"][n]["mean_w"], e["board_w"][n]) for n in e["board_w"]}})
    print("%-40s %-3s %22s %22s   board mean W measured/modeled" % ("run", "ph", "gross J/token meas/model", "net J/token meas/model"))
    errs = {"gross": [], "net": []}
    for x in rows:
        eg = 100 * (x["model_gross"] / x["measured_gross"] - 1)
        en = 100 * (x["model_net"] / x["measured_net"] - 1)
        errs["gross"].append(eg); errs["net"].append(en)
        print("%-40s %-3s %8.3f / %6.3f %+6.1f%% %8.3f / %6.3f %+6.1f%%   %s" % (
            x["run"], x["phase"], x["measured_gross"], x["model_gross"], eg, x["measured_net"], x["model_net"], en,
            " ".join("%s %.2f/%.2f" % (n, a, b) for n, (a, b) in x["board_w"].items())))
    for k, v in errs.items():
        print("%s: mean |error| %.1f %%, worst %+.1f %%" % (k, statistics.fmean(abs(e) for e in v), max(v, key=abs)))
    return rows


def _assign(pipeline):
    """[(node, blocks), ...] from "pi5 38 + jetson 10"."""
    out = []
    for part in pipeline.split(" + "):
        n, k = part.split()
        out.append((n, int(k)))
    return out


def prima():
    """The 2026-09-30 head-to-head (Qwen3-14B Q4_K_M, Pi 5 + Pi 4, QR3 serving protocol), modeled at the MEASURED time
    per token and time to first token. prima.cpp runs the Pi 5 without the repacked kernels (its llama.cpp base predates
    them), so its Pi 5 is priced at the Pi 5's unrepacked (RPC-worker) row; the power per board is the same table."""
    power = power_table()
    shapes, pairs, C = _inputs("Q4_K_M")
    q = shapes["qwen3-14b-q4_k_m"]
    L, _ = pp.link_view(pairs, "pi5", ["pi5", "pi4"])
    Cb = {n: dict(v) for n, v in C.items()}
    Cb["pi5"]["host"] = C["pi5"]["worker"]                       # unrepacked, as prima.cpp runs it
    arms = {"engine + llama.cpp": ([("pi5", 32), ("pi4", 8)], None, C, 1188.6, 45.2),
            "prima.cpp + Halda": ([("pi5", 24), ("pi4", 16)], 0, Cb, 1339.3, 81.6)}
    res = {}
    for name, (assign, head_at, CC, tpot, ttft) in arms.items():
        d = energy(assign, q, CC, L, power, "tg", head_at=head_at, t_ms=tpot)
        f = energy(assign, q, CC, L, power, "pp", head_at=head_at, t_ms=1000.0 * ttft / 128)
        res[name] = {"decode_gross_j": d["gross_j"], "decode_net_j": d["net_j"], "board_w_decode": d["board_w"],
                     "busy_decode": d["busy_frac"], "prompt128_gross_j": f["gross_j"] * 128,
                     "prompt128_net_j": f["net_j"] * 128, "tpot_ms": tpot, "ttft_s": ttft}
    a, b = res["engine + llama.cpp"], res["prima.cpp + Halda"]
    print("MODELED energy, Qwen3-14B on Pi 5 + Pi 4 at the 2026-09-30 measured TPOT / TTFT (QR3 protocol):")
    for name, r in res.items():
        print("  %-20s decode %.2f J/token gross (%.2f net), %s; a 128-token prompt %.0f J gross (%.0f net)" % (
            name, r["decode_gross_j"], r["decode_net_j"],
            ", ".join("%s %.2f W busy %.0f %%" % (n, w, 100 * r["busy_decode"][n]) for n, w in r["board_w_decode"].items()),
            r["prompt128_gross_j"], r["prompt128_net_j"]))
    for k in ("decode_gross_j", "decode_net_j", "prompt128_gross_j", "prompt128_net_j"):
        print("  engine against prima.cpp, %-17s %+.1f %%" % (k, 100 * (a[k] / b[k] - 1)))
    return res


def offline(boards_filter=None):
    """Every model x board set of halda_offline_20261007.json: the engine's default plan, Halda's plan (each ring order it
    tried, the fastest by our cost model that fits our memory accounting) and llama.cpp's even split are priced in energy
    at their PREDICTED time per token; plus the plan our planner picks for minimum energy once it has the measured power
    (placement.plan_optimal with energy_weight, on DeviceProfiles carrying the strip's power)."""
    power = power_table()
    d = json.load(open(os.path.join(DS, "halda_offline_20261007.json")))
    rows = []
    for c in d["configs"]:
        if boards_filter and sorted(c["boards"]) != sorted(boards_filter):
            continue
        key = c["model"]
        shape, quant, _ = pp.MODELS[key]
        shapes, pairs, C = _inputs(quant)
        q = shapes[shape]
        if q.get("expert_count"):
            continue                                              # MoE: dense per-byte power assumption not checked
        host = c["host"]
        L, missing = pp.link_view(pairs, host, c["boards"])
        if missing:
            continue
        row = {"model": key, "boards": "+".join(c["boards"])}
        e = c.get("engine_default")
        if e and e.get("fits_engine_accounting"):
            a = _assign(e["pipeline"])
            ha = 0 if e["head_on"] == a[0][0] and len(a) > 1 else None
            row["engine"] = (e["pipeline"] + " head " + e["head_on"],
                             energy(a, q, C, L, power, "tg", head_at=ha), energy(a, q, C, L, power, "pp", head_at=ha))
        hs = [h for h in c.get("halda", []) if h.get("pipeline") and h.get("fits_engine_accounting")]
        if hs:
            h = min(hs, key=lambda x: x["decode_ms"])
            a = _assign(h["pipeline"])
            ha = 0 if h["head_on"] == a[0][0] and len(a) > 1 else None
            row["halda"] = (h["pipeline"] + " head " + h["head_on"],
                            energy(a, q, C, L, power, "tg", head_at=ha), energy(a, q, C, L, power, "pp", head_at=ha))
        elif c.get("halda"):
            row["halda"] = ("no ring fits our memory accounting", None, None)
        rows.append(row)
    print("MODELED, predicted time per token; decode / prefill gross J per token (net in brackets):")
    for r in rows:
        line = "%-11s %-17s" % (r["model"], r["boards"])
        for k in ("engine", "halda"):
            if k in r and r[k][1]:
                line += "  %s %s: %.3f (%.3f) / %.4f" % (k, r[k][0], r[k][1]["gross_j"], r[k][1]["net_j"], r[k][2]["gross_j"])
            elif k in r:
                line += "  %s: %s" % (k, r[k][0])
        if "engine" in r and "halda" in r and r["halda"][1] and r["engine"][1]:
            line += "  | engine vs Halda decode %+.1f %%, prefill %+.1f %%" % (
                100 * (r["engine"][1]["gross_j"] / r["halda"][1]["gross_j"] - 1),
                100 * (r["engine"][2]["gross_j"] / r["halda"][2]["gross_j"] - 1))
        print(line)
    return rows


def techniques(models=("tinyllama", "qwen35_9b", "qwen3_14b"), nodes=("pi4", "jetson"), host="pi4"):
    """On today's boards, what energy-weighted planning (placement.plan_optimal with the strip's power) buys over the
    fastest plan: with every board left powered (unused="idle"), and with the boards a plan leaves out switched off
    (unused="off": the strip can; Halda and prima.cpp never do). Each plan is priced by energy() at its predicted time;
    a board left powered is charged its idle power for the whole token."""
    import dataclasses
    from control.placement import plan_optimal
    power = power_table()
    ds = DS
    mem = json.load(open(os.path.join(ds, "memory_at_rest.json")))
    rows = []
    for key in models:
        shape, quant, _ = pp.MODELS[key]
        shapes, pairs, C = _inputs(quant)
        q = shapes[shape]
        L, _ = pp.link_view(pairs, host, list(nodes))
        devs = pp.profiles(C, q, mem, L, list(nodes), power=power)
        mp = dataclasses.replace(pp.model_profile(q, shape), context_len=4096)
        out = {"model": key}
        for name, kw in (("fastest", {}), ("least energy, both on", {"energy_weight": 1000, "unused": "idle"}),
                         ("least energy, rest off", {"energy_weight": 1000, "unused": "off"})):
            pl = plan_optimal(devs, mp, "decode", host_candidates=[host], **kw)
            if not pl.feasible:
                out[name] = None
                continue
            a = [(n, pl.layers[n]) for n in pl.order if pl.layers.get(n, 0) > 0]
            ha = 0 if pl.head_node and pl.head_node == a[0][0] and len(a) > 1 else None
            e = energy(a, q, C, L, power, "tg", head_at=ha)
            on = [n for n in nodes if n not in dict(a)] if kw.get("unused") != "off" else []
            gross = e["gross_j"] + sum(power[n]["idle"] for n in on) * e["T_ms"] / 1000
            out[name] = {"plan": " + ".join("%s %d" % x for x in a) + " head " + (pl.head_node or a[-1][0]),
                         "ms": e["T_ms"], "gross_j": gross, "off": [n for n in nodes if n not in dict(a)] if not on else []}
        rows.append(out)
    for r in rows:
        base = r["fastest"]
        print("%s:" % r["model"])
        for name in ("fastest", "least energy, both on", "least energy, rest off"):
            x = r[name]
            if x is None:
                print("   %-24s infeasible" % name); continue
            print("   %-24s %-30s %8.1f ms/token  %7.3f J/token gross (%+.1f %% energy, %+.1f %% time)%s" % (
                name, x["plan"], x["ms"], x["gross_j"], 100 * (x["gross_j"] / base["gross_j"] - 1),
                100 * (x["ms"] / base["ms"] - 1), ("  [%s switched off]" % ", ".join(x["off"])) if x["off"] else ""))
    return rows


def main(argv):
    what = argv[0] if argv else "validate"
    if what == "power":
        print(json.dumps(power_table(), indent=1))
    elif what == "validate":
        validate()
    elif what == "prima":
        prima()
    elif what == "techniques":
        techniques()
    elif what == "offline":
        offline(argv[1].split(",") if len(argv) > 1 else None)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
