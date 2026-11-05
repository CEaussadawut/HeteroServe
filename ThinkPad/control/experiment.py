"""Sweep runner: one experiment spec in, a directory of recorded runs out.

An experiment is a JSON file in config/experiments/. Keeping it declarative matters for
more than tidiness - the spec file is the thing that gets attached to the report and that
someone else re-runs. A sweep that only exists as the arguments somebody typed is not
reproducible, and reproducibility is a stated deliverable.

Order of operations per sweep, and why:
  1. preflight            - a blocked cluster produces rows that look fine and are wrong
  2. idle baseline        - taken with the cluster up and idle, before any load
  3. per tier: apply, then re-survey the link, then rebuild the device profiles
     (the rule must be fed the link that exists, not the tier that was requested)
  4. per strategy: compute the placement, run it `repeats` times
  5. clear shaping, always, even on failure
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from . import netem, nodes as nodeclient, placement as plc, profiles


class ExperimentError(Exception):
    pass


def load_spec(path) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("name", "model", "strategies"):
        if key not in spec:
            raise ExperimentError("experiment spec %s is missing '%s'" % (path, key))
    spec.setdefault("tiers", ["gbe"])
    spec.setdefault("repeats", 3)
    spec.setdefault("n_predict", 128)
    spec.setdefault("energy_weight", 0.0)
    spec.setdefault("idle_baseline_s", 120)
    spec.setdefault("prompt", "Explain what a transformer is, briefly.")
    spec["_path"] = str(path)
    return spec


def build_placement(strategy: str, devices, model, energy_weight=0.0):
    """Map a strategy name from the spec onto a Placement."""
    if strategy == "rule":
        return plc.plan_optimal(devices, model, energy_weight=energy_weight)
    if strategy == "even":
        return plc.plan_even(devices, model)
    if strategy == "rule-with-straggler":
        stragglers = [nid for nid, d in devices.items() if d.is_straggler]
        return plc.plan_optimal(devices, model, energy_weight=energy_weight, force_include=stragglers)
    if strategy == "rule-no-straggler":
        remaining = {nid: d for nid, d in devices.items() if not d.is_straggler}
        if not remaining:
            raise ExperimentError("every node is marked a straggler")
        return plc.plan_optimal(remaining, model, energy_weight=energy_weight)
    if strategy.startswith("single:"):
        return plc.plan_single(devices, model, strategy.split(":", 1)[1])
    raise ExperimentError("unknown strategy '%s'" % strategy)


def run_sweep(cluster, harness, spec, devices, model, out_dir=None, skip_preflight=False) -> dict:
    """Execute one experiment spec end to end."""
    sweep_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + spec["name"]
    out_dir = Path(out_dir or harness.data_dir / "sweeps" / sweep_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    result = {
        "sweep_id": sweep_id,
        "spec": spec,
        "started_utc": time.time(),
        "runs": [],
        "placements": {},
        "link_surveys": {},
        "aborted": None,
    }

    if not skip_preflight:
        report = nodeclient.preflight(cluster)
        (out_dir / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not report["ready"]:
            result["aborted"] = "preflight blocked: " + "; ".join(report["blocking"])
            _save(out_dir, result)
            return result

    harness.rpc_up()
    try:
        if spec["idle_baseline_s"]:
            result["idle_baseline"] = harness.idle_baseline(
                spec["idle_baseline_s"], note=sweep_id)

        for tier in spec["tiers"]:
            # The return value used to be discarded, so a tier that reached only some nodes
            # was never even printed -- and every run in this sweep would still be labelled
            # with `tier`. apply_tier now raises on a partial application; record what was
            # actually applied so the sweep's own output carries the evidence.
            result.setdefault("tier_application", {})[tier] = netem.apply_tier(cluster, tier)
            time.sleep(3)
            survey = netem.survey(cluster, seconds=spec.get("survey_s", 10))
            result["link_surveys"][tier] = survey
            devices_tier = _apply_survey(devices, survey)

            # Every placement is built BEFORE any run, so the repeat loop can be the outer
            # one below. They all derive from `devices_tier`, which is fixed for this tier,
            # so building them up front changes no placement -- only when they are built.
            runnable = []
            for strategy in spec["strategies"]:
                try:
                    placement = build_placement(strategy, devices_tier, model, spec["energy_weight"])
                except ExperimentError as exc:
                    result["runs"].append({"strategy": strategy, "tier": tier, "error": str(exc)})
                    continue

                key = "%s@%s" % (strategy, tier)
                result["placements"][key] = {
                    "strategy": placement.strategy, "feasible": placement.feasible,
                    "reason": placement.reason, "layers": placement.layers,
                    "order": placement.order, "predicted": placement.predicted,
                }
                if not placement.feasible:
                    # An infeasible placement is a result: it is the memory wall the project
                    # exists to demonstrate. Record it and move on rather than failing.
                    result["runs"].append({"strategy": strategy, "tier": tier,
                                           "infeasible": placement.reason})
                    continue

                runnable.append((strategy, placement))

            # Repeat is the OUTER loop and the strategies are interleaved inside it. When
            # repeats were innermost, every repeat of the first strategy ran from a cold
            # cluster and every repeat of the last ran after minutes of sustained load, so
            # thermal drift was perfectly confounded with the variable under test: for
            # placement_vs_even_7b (the >=10 % MVP criterion) all 5 `rule` repeats ran
            # cold and all 5 `even` repeats started ~450 s later. Any Pi4/Pi5 throttling
            # therefore entered the rule-vs-even delta with the sign that favours the
            # hypothesis, and -- because the sweep path records no throttle flag -- nothing
            # in the output could distinguish that from a real placement win.
            #
            # Interleaving spreads the drift across every arm instead of loading it onto
            # one. It also makes the per-repeat comparison honest: the seed is a function
            # of `rep` alone, so `rule` r0 and `even` r0 already share seed 1234, and now
            # they also run within one cooldown of each other and at nearly the same board
            # temperature. `run_id` still encodes strategy and rep, so nothing downstream
            # depends on the order these are emitted in.
            for rep in range(spec["repeats"]):
                for strategy, placement in runnable:
                    record = harness.run(
                        placement, spec["model"], prompt=spec["prompt"],
                        n_predict=spec["n_predict"], tier=tier,
                        run_id="%s_%s_%s_r%d" % (sweep_id, strategy.replace(":", "-"), tier, rep),
                        seed=spec.get("seed", 1234) + rep,
                        context=getattr(model, "context_len", None))   # what the plan charged KV for
                    record["sweep_id"] = sweep_id
                    record["repeat"] = rep
                    result["runs"].append(record)
                    _save(out_dir, result)
                    time.sleep(spec.get("cooldown_s", 20))
    finally:
        netem.clear(cluster)
        result["ended_utc"] = time.time()
        _save(out_dir, result)

    return result


def _apply_survey(devices, survey):
    """Feed the freshly measured link into copies of the device profiles."""
    out = {}
    for nid, dev in devices.items():
        entry = (survey.get("nodes") or {}).get(nid) or {}
        clone = plc.DeviceProfile(**{**dev.__dict__})
        if entry.get("mbps"):
            clone.link_mbps = entry["mbps"]
        if entry.get("rtt_ms"):
            clone.rtt_ms = entry["rtt_ms"]
        out[nid] = clone
    return out


def _save(out_dir, result):
    (Path(out_dir) / "sweep.json").write_text(json.dumps(result, indent=2, default=str),
                                              encoding="utf-8")


def _temps(snapshot):
    """(hottest, contributed, missing) from one conditions snapshot.

    Never returns a bare maximum. A max over a SUBSET of the participants, labelled as the
    hottest board, is a complete and plausible wrong number -- and it was the real
    behaviour here: `harness.conditions` reads only top-level /metrics keys, the Pi 5's
    agent wraps its body in `{"sample": ...}` (Pi5/node/agent.py:309), so the Pi 5 answered
    200 OK with a valid temperature and was dropped. On a sweep where the Pi 5 sat at
    78.4 C and the Pi 4 at 61.0 C, this column read 61.0 C. The Pi 5 is the strong node
    that carries most layers and runs hottest, so the error is both large and one-signed,
    and it defeated the interleaving above, whose only purpose is making drift visible.

    So the coverage travels with the figure. A reader can see 'one of three boards
    reported' and discount it; they cannot see that in a bare number.
    """
    snap = snapshot or {}
    got, missing = {}, []
    for nid, v in snap.items():
        temp = v.get("soc_temp_c") if isinstance(v, dict) else None
        if isinstance(temp, (int, float)):
            got[nid] = temp
        else:
            missing.append(nid)
    return (max(got.values()) if got else None, sorted(got), sorted(missing))


def _hottest(snapshot) -> float | None:
    """The hottest board that reported. Prefer `_temps` -- this hides the coverage."""
    return _temps(snapshot)[0]


def summarise(sweep_result: dict) -> list:
    """Flatten a sweep into comparable rows: strategy x tier x repeat."""
    rows = []
    for run in sweep_result.get("runs", []):
        if "error" in run or "infeasible" in run:
            rows.append({
                "strategy": run.get("strategy"), "tier": run.get("tier"),
                "status": run.get("error") or "infeasible: " + run.get("infeasible", ""),
            })
            continue
        t = run.get("timings", {})
        start_t, end_t = _temps(run.get("conditions_start")), _temps(run.get("conditions_end"))
        rows.append({
            "strategy": run.get("strategy"),
            "tier": run.get("tier"),
            "repeat": run.get("repeat"),
            "participants": ",".join(run.get("participants", [])),
            "layers": run.get("layers"),
            "ttft_ms": t.get("ttft_ms_approx"),
            "tpot_ms": t.get("tpot_ms"),
            "decode_tok_s": t.get("decode_tokens_per_s"),
            "prefill_tok_s": t.get("prefill_tokens_per_s"),
            "predicted_ms_per_token": (run.get("predicted") or {}).get("ms_per_token"),
            "wall_s": run.get("wall_s"),
            # Emitted so the interleaved run order can be audited rather than trusted: if
            # these still climb monotonically across an arm, the ordering fix did not
            # deliver a thermally level field and the comparison is still contaminated.
            # `temp_nodes`/`temp_missing` ship WITH the maximum, never separately: the
            # figure is only as good as its coverage, and a max over one of three boards
            # read as cluster-wide for as long as the coverage was not written down.
            "temp_start_max_c": start_t[0],
            "temp_end_max_c": end_t[0],
            "temp_nodes": ",".join(end_t[1]),
            "temp_missing": ",".join(sorted(set(start_t[2]) | set(end_t[2]))),
            "status": "ok" if run.get("ok") else run.get("failure_hint", "failed"),
        })
    return rows


def prediction_error(sweep_result: dict) -> dict:
    """How well the rule's model predicted reality.

    This is the feedback loop the proposal calls for (measure -> model -> place): if the
    predicted per-token time is consistently off, the cost model is wrong and the placement
    it produces is only accidentally good. Reported as a signed percentage so a systematic
    bias is visible rather than averaged away.
    """
    errors = {}
    for run in sweep_result.get("runs", []):
        if not run.get("ok"):
            continue
        predicted = (run.get("predicted") or {}).get("ms_per_token")
        actual = (run.get("timings") or {}).get("tpot_ms")
        if not predicted or not actual:
            continue
        key = "%s@%s" % (run.get("strategy"), run.get("tier"))
        errors.setdefault(key, []).append(round(100.0 * (predicted - actual) / actual, 2))
    return {k: {"runs": len(v), "mean_pct": round(sum(v) / len(v), 2), "values": v}
            for k, v in errors.items()}
