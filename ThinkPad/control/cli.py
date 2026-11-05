#!/usr/bin/env python3
"""heteroctl - the control node's command line.

    heteroctl status                     what the cluster is and whether it is up
    heteroctl preflight                  readiness gate across every node
    heteroctl inventory --save           collect device_info from all agents
    heteroctl rpc up|down                start/stop rpc-server everywhere
    heteroctl tier gbe|wifi|shaped-100   set the link tier on every node
    heteroctl survey --save              measure real bandwidth and RTT
    heteroctl daq sync|record|status     the power DAQ
    heteroctl idle-baseline              record the marked idle window
    heteroctl profiles --build           measured data -> device profiles
    heteroctl plan MODEL                 compute a placement
    heteroctl compare MODEL              rule vs even vs single vs straggler
    heteroctl pareto MODEL               sweep lambda for the energy Pareto front
    heteroctl run MODEL --strategy rule  one measured run
    heteroctl sweep EXPERIMENT.json      a full experiment
    heteroctl join RUN_DIR               attach energy to benchmark rows
    heteroctl report ...                 tables for the write-up
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import analysis, config, energy, experiment, netem, nodes as nodeclient, profiles, report
from .harness import Harness


def _cluster(args):
    return config.load(args.config)


def _data(cluster, *parts) -> Path:
    root = Path(cluster.paths.get("data", "/opt/heteroserve/data"))
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _print_json(obj):
    print(json.dumps(obj, indent=2, default=str))


# ------------------------------------------------------------------ commands
def cmd_status(args):
    cluster = _cluster(args)
    print(cluster.summary())
    print()
    workers = [n for n in cluster.workers() if n.agent_port]
    results = nodeclient.fanout(workers, nodeclient.metrics)
    for node in workers:
        entry = results.get(node.id, {})
        if not entry.get("ok"):
            print("  %-13s unreachable (%s)" % (node.id, entry.get("error", "?")))
            continue
        m = entry["data"]
        # `"up" if m.get("rpc_active") else "down"` collapsed an ABSENT key into an
        # affirmative "down". The Pi 5's /metrics does not carry the key at all (it wraps
        # its body in {"sample": ...}), so this line asserted that a running RPC server was
        # stopped. An unknown state must read as unknown: a blank is honest, a wrong word
        # is not. Same reason the other columns show "?" rather than the string "None".
        rpc = m.get("rpc_active")
        print("  %-13s %5s C  %6s MHz  gov=%-11s free=%5s MB  rpc=%s" %
              (node.id, _or_q(m.get("soc_temp_c")), _or_q(m.get("cpu_mhz", m.get("arm_mhz"))),
               _or_q(m.get("governor")), _or_q(m.get("mem_available_mb")),
               "?" if rpc is None else ("up" if rpc else "down")))
    return 0


def _or_q(value):
    """Render a missing reading as "?" rather than the string "None"."""
    return "?" if value is None else value


def cmd_preflight(args):
    cluster = _cluster(args)
    rep = nodeclient.preflight(cluster, require_pin=not args.no_pin_check,
                               require_clock=not args.no_clock_check)
    print(nodeclient.format_preflight(rep))
    if args.save:
        path = _data(cluster, "inventory", "preflight_%s.json" % time.strftime("%Y%m%dT%H%M%SZ"))
        path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print("\nsaved -> %s" % path)
    return 0 if rep["ready"] else 1


def cmd_inventory(args):
    cluster = _cluster(args)
    inv = nodeclient.inventory(cluster)
    if args.save:
        path = _data(cluster, "inventory", "cluster_inventory.json")
        path.write_text(json.dumps(inv, indent=2), encoding="utf-8")
        print("saved -> %s" % path)
    else:
        _print_json(inv)
    return 0


def cmd_rpc(args):
    cluster = _cluster(args)
    harness = Harness(cluster)
    result = harness.rpc_up() if args.action == "up" else harness.rpc_down()
    for nid, entry in sorted(result.items()):
        print("  %-13s %s" % (nid, "ok" if entry.get("ok") else entry.get("error")))
    return 0


def cmd_tier(args):
    cluster = _cluster(args)
    if args.tier == "clear":
        _print_json(netem.clear(cluster))
        return 0
    if args.tier == "verify":
        _print_json(netem.verify(cluster))
        return 0
    try:
        report = netem.apply_tier(cluster, args.tier, strict=not args.allow_partial)
    except netem.NetemError as exc:
        # Exit non-zero: a tier that reached only some nodes must never look like success,
        # because the runs that follow are still labelled with it.
        print("tier NOT applied: %s" % exc, file=sys.stderr)
        return 1
    _print_json(report)
    if not report["ok"]:
        print("WARNING: proceeding with a partially shaped cluster; runs labelled '%s' are "
              "not that condition on %d node(s)." % (args.tier, len(report["failed"])),
              file=sys.stderr)
        return 1
    return 0


def cmd_survey(args):
    cluster = _cluster(args)
    result = netem.survey(cluster, seconds=args.seconds)
    _print_json(result)
    if args.save:
        path = _data(cluster, "inventory", "link_survey.json")
        netem.save_survey(result, path)
        print("saved -> %s" % path)
    return 0


def cmd_daq(args):
    cluster = _cluster(args)
    daq = energy.open_daq(cluster)
    with daq:
        sync = daq.sync_clock()
        print("clock offset %.6f s (sd %.6f, %d rounds)" %
              (sync["offset_s"], sync["offset_stdev_s"] or 0, sync["rounds"]))

        if args.action == "sync":
            return 0

        if args.action == "status":
            daq.start(cluster.daq.get("sample_hz"))
            seen = {}
            deadline = time.time() + args.seconds
            while time.time() < deadline:
                sample = daq.read_sample()
                if not sample:
                    continue
                _, ch, node, bus_v, current_a, power_w, _ = sample
                seen.setdefault(ch, []).append((bus_v, current_a, power_w))
            daq.stop()
            for ch in sorted(seen):
                rows = seen[ch]
                node = daq.channels.get(ch, "?")
                print("  ch%d %-13s %5.2f V  %6.3f A  %6.2f W   (%d samples)" %
                      (ch, node, rows[-1][0], rows[-1][1],
                       sum(r[2] for r in rows) / len(rows), len(rows)))
            if not seen:
                print("  no samples - check the DAQ wiring and firmware/pico2/main.py")
            return 0 if seen else 1

        if args.action == "record":
            out = _data(cluster, "power", "power_%s.csv" % time.strftime("%Y%m%dT%H%M%SZ"))
            sync_path = out.with_suffix(".sync.json")
            sync_path.write_text(json.dumps(sync, indent=2), encoding="utf-8")
            rec = energy.Recorder(daq, out).start()
            print("recording to %s - Ctrl-C to stop" % out)
            try:
                while True:
                    time.sleep(5)
                    status = rec.status()
                    print("  %d samples, %.1f Hz total, %d errors" %
                          (status["samples"], status["effective_hz_all_channels"] or 0,
                           status["errors"]), end="\r", flush=True)
            except KeyboardInterrupt:
                print()
            _print_json(rec.stop())
            return 0
    return 0


def cmd_calibrate(args):
    """Turn measured (reported, reference) current pairs into a channel calibration."""
    points = [tuple(float(x) for x in p.split(":")) for p in args.point]
    gain, offset = energy.fit_calibration(points)
    cal = {
        "channel": args.channel,
        "gain": round(gain, 6),
        "offset_a": round(offset, 8),
        "shunt_ohms": args.shunt,
        "points": [{"reported_a": p[0], "reference_a": p[1]} for p in points],
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": args.note or "reference current from I = V/R, V measured on the multimeter",
    }
    out = Path(args.out or Path(__file__).resolve().parent.parent / "calibration" / ("ch%d.json" % args.channel))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print("gain=%.6f  offset=%.8f A  ->  %s" % (gain, offset, out))
    if len(points) < 2:
        print("warning: one point sets the gain only; that is a scale check, not a calibration")
    return 0


def cmd_idle_baseline(args):
    cluster = _cluster(args)
    harness = Harness(cluster)
    harness.rpc_up()
    print("recording a %d s idle baseline with the cluster up and idle..." % args.seconds)
    result = harness.idle_baseline(args.seconds, note=args.note or "")
    _print_json(result)
    return 0


def cmd_profiles(args):
    cluster = _cluster(args)
    n_layers = json.loads(Path(args.layers).read_text()) if args.layers else {}
    devices = profiles.build_devices(
        cluster,
        results_csv=args.results, power_summary=args.power,
        link_json=args.links, inventory_json=args.inventory,
        n_layers_by_model=n_layers, threads=args.threads, strict=not args.loose,
        model_dir=cluster.paths.get("models"))
    out = args.out or _data(cluster, "inventory", "device_profiles.json")
    profiles.save_devices(devices, out)
    print("wrote %d device profiles -> %s" % (len(devices), out))
    for nid, d in devices.items():
        power = ("%.2f W active / %.2f W idle" % (d.power_active_w, d.power_idle_w)
                 if d.has_power else "power not measured (latency-only)")
        print("  %-13s decode %.3f ms/layer  prefill %.3f  %s  %s Mbps" %
              (nid, d.ms_per_layer_decode, d.ms_per_layer_prefill, power, d.link_mbps))
    return 0


def _load_model_and_devices(args, cluster):
    devices = profiles.load_devices(args.profiles or _data(cluster, "inventory", "device_profiles.json"))
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path(cluster.paths.get("models", ".")) / model_path
    model = profiles.model_profile(model_path, context_len=args.context)
    unpriced = sorted(n for n, d in devices.items() if not d.ms_per_byte_decode)
    if unpriced:
        print("note: %s carry no per-byte costs (an older profile file): their per-layer figures are the profiled "
              "model's, not %s's" % (", ".join(unpriced), model.name), file=sys.stderr)
    return profiles.for_model(devices, model), model, model_path


def cmd_plan(args):
    cluster = _cluster(args)
    devices, model, _ = _load_model_and_devices(args, cluster)
    from . import placement as plc
    p = plc.plan_optimal(devices, model, phase=args.phase, energy_weight=args.energy_weight)
    print(p.describe())
    if p.feasible:
        print("\n  llama.cpp flags:")
        print("    " + " ".join(p.llama_args(cluster, str(args.model))))
    return 0 if p.feasible else 1


def cmd_compare(args):
    cluster = _cluster(args)
    devices, model, _ = _load_model_and_devices(args, cluster)
    from . import placement as plc
    comparison = plc.compare(devices, model, phase=args.phase, energy_weight=args.energy_weight)
    text = report.placement_comparison(comparison, out_md=args.out)
    print(text)
    return 0


def cmd_pareto(args):
    cluster = _cluster(args)
    devices, model, _ = _load_model_and_devices(args, cluster)
    from . import placement as plc
    front = plc.pareto(devices, model, phase=args.phase)
    for point in front:
        print("  lambda=%-6g %-40s %7.1f ms/tok  %6.3f J/tok  %5.2f W" %
              (point["energy_weight"],
               ", ".join("%s:%d" % (n, point["layers"][n]) for n in point["order"]),
               point["ms_per_token"], point["joules_per_token"], point["mean_power_w"]))
    if args.out:
        report.pareto_csv(front, args.out)
        png = report.plot_pareto(front, str(Path(args.out).with_suffix(".png")))
        print("\nwrote %s%s" % (args.out, (" and " + png) if png else " (matplotlib absent, no plot)"))
    return 0


def cmd_run(args):
    cluster = _cluster(args)
    devices, model, model_path = _load_model_and_devices(args, cluster)
    placement = experiment.build_placement(args.strategy, devices, model, args.energy_weight)
    if not placement.feasible:
        print("infeasible placement: %s" % placement.reason)
        return 1
    print(placement.describe())

    if not args.skip_preflight:
        rep = nodeclient.preflight(cluster)
        if not rep["ready"]:
            print(nodeclient.format_preflight(rep))
            print("\nrefusing to run - fix the blocking items or pass --skip-preflight "
                  "(and mark the resulting rows as unverified)")
            return 1

    harness = Harness(cluster, dry_run=args.dry_run)
    harness.rpc_up()
    record = harness.run(placement, model_path, prompt=args.prompt, n_predict=args.n_predict,
                         tier=args.tier, context=args.context)
    _print_json(record.get("timings") or record)
    return 0 if record.get("ok") or args.dry_run else 1


def cmd_sweep(args):
    cluster = _cluster(args)
    spec = experiment.load_spec(args.spec)
    devices, model, _ = _load_model_and_devices(
        argparse.Namespace(model=spec["model"], profiles=args.profiles, context=args.context), cluster)
    harness = Harness(cluster, dry_run=args.dry_run)
    result = experiment.run_sweep(cluster, harness, spec, devices, model,
                                  skip_preflight=args.skip_preflight)
    if result.get("aborted"):
        print("sweep aborted: %s" % result["aborted"])
        return 1
    rows = experiment.summarise(result)
    for row in rows:
        print("  %-22s %-12s %s" % (row.get("strategy"), row.get("tier"), row.get("status")))
    print("\nsweep %s: %d runs" % (result["sweep_id"], len(result["runs"])))
    return 0


def cmd_join(args):
    cluster = _cluster(args)
    node_channel = {n.id: n.ina_channel for n in cluster.nodes if n.ina_channel}
    calibrated = set(energy.Calibration.load_all(
        Path(__file__).resolve().parent.parent / "calibration"))
    result = analysis.join_run(args.run_dir, args.power,
                               sample_hz=cluster.daq.get("sample_hz", 28),
                               calibrated_channels=calibrated, node_channel=node_channel)
    _print_json(result)
    return 0


def cmd_power_summary(args):
    cluster = _cluster(args)
    summary = analysis.node_power_summary(args.power, events_jsonl=args.events,
                                          phase_windows_jsonl=args.windows)
    out = args.out or _data(cluster, "inventory", "power_summary.json")
    Path(out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_json(summary)
    print("saved -> %s" % out)
    return 0


def cmd_report(args):
    if args.kind == "characterisation":
        print(report.characterisation(args.input, out_md=args.out,
                                      out_csv=args.out and str(Path(args.out).with_suffix(".csv"))))
    elif args.kind == "sweep":
        print(report.sweep_report(args.input, out_md=args.out))
    return 0


# --------------------------------------------------------------------- parser
def build_parser():
    p = argparse.ArgumentParser(prog="heteroctl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to cluster.json")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="cluster summary and live node metrics").set_defaults(func=cmd_status)

    sp = sub.add_parser("preflight", help="readiness gate across every node")
    sp.add_argument("--save", action="store_true")
    sp.add_argument("--no-clock-check", action="store_true",
                    help="skip the clock-sync gate (a skewed clock misattributes phases)")
    sp.add_argument("--no-pin-check", action="store_true",
                    help="skip the llama.cpp revision check (bring-up only)")
    sp.set_defaults(func=cmd_preflight)

    sp = sub.add_parser("inventory", help="collect device_info from every agent")
    sp.add_argument("--save", action="store_true")
    sp.set_defaults(func=cmd_inventory)

    sp = sub.add_parser("rpc", help="start/stop rpc-server on every worker")
    sp.add_argument("action", choices=["up", "down"])
    sp.set_defaults(func=cmd_rpc)

    sp = sub.add_parser("tier", help="set the link tier everywhere")
    sp.add_argument("tier", help="a tier id from cluster.json, or 'clear' / 'verify'")
    sp.add_argument("--allow-partial", action="store_true",
                    help="apply to whatever nodes respond instead of refusing; the run "
                         "is then labelled with a tier it is not on, so label it as such")
    sp.set_defaults(func=cmd_tier)

    sp = sub.add_parser("survey", help="measure bandwidth and RTT to every node")
    sp.add_argument("--seconds", type=int, default=10)
    sp.add_argument("--save", action="store_true")
    sp.set_defaults(func=cmd_survey)

    sp = sub.add_parser("daq", help="power DAQ: sync, status or record")
    sp.add_argument("action", choices=["sync", "status", "record"])
    sp.add_argument("--seconds", type=int, default=10, help="for 'status'")
    sp.set_defaults(func=cmd_daq)

    sp = sub.add_parser("calibrate", help="write a channel calibration from measured points")
    sp.add_argument("--channel", type=int, required=True)
    sp.add_argument("--point", action="append", required=True, metavar="REPORTED:REFERENCE",
                    help="amps as reported by the INA226 : amps from I=V/R, repeatable")
    sp.add_argument("--shunt", type=float, default=0.01)
    sp.add_argument("--note", default=None)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("idle-baseline", help="record a marked idle window on every node")
    sp.add_argument("--seconds", type=int, default=120)
    sp.add_argument("--note", default=None)
    sp.set_defaults(func=cmd_idle_baseline)

    sp = sub.add_parser("profiles", help="build device profiles from measured data")
    sp.add_argument("--results", required=True, help="joined results_energy.csv (all nodes)")
    sp.add_argument("--power", help="power_summary.json")
    sp.add_argument("--links", help="link_survey.json")
    sp.add_argument("--inventory", help="cluster_inventory.json")
    sp.add_argument("--layers", help="json: {model_name: n_layers}")
    sp.add_argument("--threads", type=int, default=None,
                    help="thread count the cluster run uses; required when the results span "
                         "a thread sweep, because a median across it inflates c_i")
    sp.add_argument("--loose", action="store_true",
                    help="allow unmeasured inputs (planning sketch only - label it as one)")
    sp.add_argument("--out", default=None)
    sp.add_argument("--build", action="store_true", help="accepted for readability; always builds")
    sp.set_defaults(func=cmd_profiles)

    for name, func, extra in (("plan", cmd_plan, False), ("compare", cmd_compare, True),
                              ("pareto", cmd_pareto, True)):
        sp = sub.add_parser(name, help="placement: %s" % name)
        sp.add_argument("model", help="GGUF path (absolute, or relative to paths.models)")
        sp.add_argument("--profiles", default=None)
        sp.add_argument("--phase", choices=["decode", "prefill"], default="decode")
        sp.add_argument("--energy-weight", type=float, default=0.0, dest="energy_weight")
        sp.add_argument("--context", type=int, default=2048)
        if extra:
            sp.add_argument("--out", default=None)
        sp.set_defaults(func=func)

    sp = sub.add_parser("run", help="one measured run on the cluster")
    sp.add_argument("model")
    sp.add_argument("--strategy", default="rule",
                    help="rule | even | rule-with-straggler | rule-no-straggler | single:<node>")
    sp.add_argument("--profiles", default=None)
    sp.add_argument("--phase", choices=["decode", "prefill"], default="decode")
    sp.add_argument("--energy-weight", type=float, default=0.0, dest="energy_weight")
    sp.add_argument("--context", type=int, default=2048)
    sp.add_argument("--prompt", default="Explain what a transformer is, briefly.")
    sp.add_argument("--n-predict", type=int, default=128, dest="n_predict")
    sp.add_argument("--tier", default="gbe")
    sp.add_argument("--skip-preflight", action="store_true")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("sweep", help="run an experiment spec")
    sp.add_argument("spec")
    sp.add_argument("--profiles", default=None)
    sp.add_argument("--context", type=int, default=2048)
    sp.add_argument("--skip-preflight", action="store_true")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("join", help="attach energy to a run's benchmark rows")
    sp.add_argument("run_dir")
    sp.add_argument("--power", required=True, help="power_*.csv from the collector")
    sp.set_defaults(func=cmd_join)

    sp = sub.add_parser("power-summary", help="active/idle power per node from the trace")
    sp.add_argument("--power", required=True)
    sp.add_argument("--events", help="events.jsonl with idle_baseline markers")
    sp.add_argument("--windows", default=None,
                    help="runs/<id>/windows.jsonl - the recorded load/prefill/decode split. "
                         "Without it the active power is left unset, because the whole-run "
                         "mean includes model load and is not P_act")
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_power_summary)

    sp = sub.add_parser("report", help="generate a markdown table")
    sp.add_argument("kind", choices=["characterisation", "sweep"])
    sp.add_argument("input")
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_report)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (config.ConfigError, profiles.ProfileError, energy.DaqError,
            netem.NetemError, experiment.ExperimentError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
