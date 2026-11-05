"""Benchmark harness: run a placement on the cluster and record what happened.

One run produces four artefacts, and the run is not considered done until all four exist:
  * the raw stdout/stderr of llama.cpp, kept verbatim
  * parsed timings (prefill and decode, separately)
  * a window record with the wall-clock bracket the energy collector integrates over
  * markers on every participating node, so each node's own log agrees with the harness

Prefill and decode are separated from llama.cpp's own timing report ("prompt eval time"
and "eval time"), which is the boundary the engine itself sees. That is more trustworthy
than inferring the boundary from the power trace, and it is what makes the phase-separated
joules-per-token figure defensible.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from . import nodes as nodeclient


class HarnessError(Exception):
    pass


TIMING_PATTERNS = {
    "load_ms": re.compile(r"load time\s*=\s*([\d.]+)\s*ms"),
    "prefill": re.compile(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second", re.S),
    "decode": re.compile(r"\beval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs.*?([\d.]+)\s*tokens per second", re.S),
    "total_ms": re.compile(r"total time\s*=\s*([\d.]+)\s*ms"),
}


def parse_timings(text: str) -> dict:
    """Pull llama.cpp's own timing report out of the run log."""
    out = {}
    m = TIMING_PATTERNS["load_ms"].search(text)
    if m:
        out["load_ms"] = float(m.group(1))

    m = TIMING_PATTERNS["prefill"].search(text)
    if m:
        ms, tokens, tps = float(m.group(1)), int(m.group(2)), float(m.group(3))
        out["prefill_ms"] = ms
        out["prefill_tokens"] = tokens
        out["prefill_tokens_per_s"] = tps
        # TTFT is the prefill time plus model-independent overhead; llama-cli does not
        # report first-token latency directly, so this is named for what it is.
        out["ttft_ms_approx"] = ms

    m = TIMING_PATTERNS["decode"].search(text)
    if m:
        ms, runs, tps = float(m.group(1)), int(m.group(2)), float(m.group(3))
        out["decode_ms"] = ms
        out["decode_tokens"] = runs
        out["decode_tokens_per_s"] = tps
        out["tpot_ms"] = round(ms / runs, 3) if runs else None

    m = TIMING_PATTERNS["total_ms"].search(text)
    if m:
        out["total_ms"] = float(m.group(1))
    return out


# The keys this folder recognises in a node agent's /metrics body.
#
# An earlier revision of this comment claimed `soc_temp_c` is "common to both" agents. That
# is true of Pi5/node/telemetry.py sample(), and FALSE of what /metrics actually returns:
# only Pi 4's agent answers a flat body (Pi4/node/node_agent.py:68-84), while the Pi 5's
# wraps it -- `{"sample": {...}, "run": tag}` at Pi5/node/agent.py:309 -- so NOT ONE of
# these keys appears at the top level there, `soc_temp_c` included. The cluster has no
# portable /metrics contract yet, and the two agents also disagree on spelling
# (`throttled_word` vs `throttled_hex`, `arm_mhz` vs `cpu_mhz`).
#
# Unwrapping the Pi 5's envelope here would paper over that rather than settle it, and the
# canonical key set is a cross-node contract that belongs to those folders. So `conditions`
# records what it actually received -- including the top-level keys it did NOT recognise,
# which is the evidence those sessions need -- and the caller is told which nodes
# contributed rather than being handed a silent subset.
CONDITION_KEYS = ("soc_temp_c", "throttled_word", "throttled_hex", "throttled_now",
                  "arm_mhz", "governor")


def conditions(nodes) -> dict:
    """Per-node thermal and clock state, so drift across a sweep is visible afterwards.

    A sweep used to record no thermal state at any point: `preflight` samples `temp_c`
    once, before `rpc_up()`, so it captures only the cold cluster and cannot show drift
    BETWEEN arms. The throttle taxonomy in analysis.py exists but lives entirely on the
    per-node llama-bench path and never sees a cluster run. So a run whose board had
    throttled produced a complete, plausible, wrong TPOT with nothing to mark it -- the
    exact defect class this project is built to avoid.

    Best-effort by construction: `/metrics` carries a 5 s timeout and `fanout` captures
    every error, so an unreachable node costs a bounded pause and records why. It never
    raises. A run is the measurement; refusing to record one because a temperature could
    not be read would discard the measurement to protect the annotation.
    """
    if not nodes:
        return {}
    out = {}
    for nid, entry in nodeclient.fanout(nodes, nodeclient.metrics).items():
        if not entry.get("ok"):
            out[nid] = {"error": entry.get("error")}
            continue
        data = entry.get("data") or {}
        got = {k: data[k] for k in CONDITION_KEYS if data.get(k) is not None}
        if not got and data:
            # The agent answered 200 with a body this folder cannot read. Recording the
            # top-level keys turns a silent drop into evidence: against a Pi 5 this writes
            # ["run", "sample"] into every sweep, which names the envelope outright.
            got = {"unrecognised_payload": sorted(data)[:12]}
        out[nid] = got
    return out


class Harness:
    def __init__(self, cluster, data_dir=None, dry_run: bool = False):
        self.cluster = cluster
        self.data_dir = Path(data_dir or cluster.paths.get("data", "/opt/heteroserve/data"))
        self.bin = Path(cluster.paths.get("llama_bin", "/opt/heteroserve/llama.cpp/build/bin"))
        self.dry_run = dry_run

    # -- cluster lifecycle ---------------------------------------------------
    def rpc_up(self, node_ids=None):
        targets = [self.cluster.node(n) for n in node_ids] if node_ids else \
                  [n for n in self.cluster.workers() if n.agent_port]
        results = nodeclient.fanout(targets, nodeclient.rpc_start)
        time.sleep(2)                      # let the servers bind before a client connects
        return results

    def rpc_down(self, node_ids=None):
        targets = [self.cluster.node(n) for n in node_ids] if node_ids else \
                  [n for n in self.cluster.workers() if n.agent_port]
        return nodeclient.fanout(targets, nodeclient.rpc_stop)

    # -- a single run --------------------------------------------------------
    def run(self, placement, model_path, prompt="Explain what a transformer is, briefly.",
            n_predict=128, run_id=None, tier="gbe", extra_args=None, timeout=1800,
            seed=1234, threads=None, context=None) -> dict:
        """Execute one placement and return its record.

        The wall-clock bracket recorded here is deliberately wider than the inference
        itself: it starts before the process launches and ends after it exits, so model
        load is inside the window. The parsed timings then let the analysis attribute
        energy to load, prefill and decode separately, instead of pretending load is free.
        """
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = self.data_dir / "runs" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        participants = [self.cluster.node(nid) for nid in placement.participants]
        marked = [n for n in participants if n.agent_port]

        # llama-completion, not llama-cli. At the pin (b23701f) llama-cli is the chat front end
        # and rejects -no-cnv ("error: invalid argument: -no-cnv", checked 2026-09-29), while
        # llama-completion keeps the one-shot completion path and prints the timing lines
        # parse_timings reads.
        cmd = [str(self.bin / "llama-completion")]
        cmd += placement.llama_args(self.cluster, str(model_path))
        cmd += ["-p", prompt, "-n", str(n_predict), "--seed", str(seed), "-no-cnv"]
        # The context the plan charged KV cache for (placement: kv_bytes_per_layer_per_token x
        # context_len). Left unset, llama.cpp at the pin sizes the cache for the model's whole
        # training context ("0 = loaded from model": 40960 tokens for Qwen3-14B, ~6.7 GB), and its
        # own fitting is off because -ngl is set -- a plan made for 2048 tokens OOM-killed the
        # Pi 5's rpc-server (2026-09-29, the first bring-up on the real cluster).
        if context:
            cmd += ["-c", str(context)]
        if threads:
            cmd += ["-t", str(threads)]
        cmd += list(extra_args or [])

        record = {
            "run_id": run_id,
            "strategy": placement.strategy,
            "model": placement.model,
            "model_path": str(model_path),
            "tier": tier,
            "layers": placement.layers,
            "order": placement.order,
            "participants": placement.participants,
            "tensor_split": placement.tensor_split(),
            "predicted": placement.predicted,
            "n_predict": n_predict,
            "context": context,
            "prompt_chars": len(prompt),
            "seed": seed,
            "command": cmd,
            "started_utc": None,
            "ended_utc": None,
        }

        if self.dry_run:
            record["dry_run"] = True
            (out_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            return record

        # Sampled before the marks so the run_begin/run_end pair stays tight around the
        # timing bracket; the snapshot's own latency sits outside the measured window.
        record["conditions_start"] = conditions(marked)
        nodeclient.mark_all(marked, "run_begin", "%s %s tier=%s" % (run_id, placement.strategy, tier))
        t0 = time.time()
        try:
            # stdin closed, not inherited. llama-completion (llama-cli before the split) drops
            # into interactive conversation mode when the model carries a chat template and
            # `-no-cnv` is absent or
            # unsupported -- which is exactly what happens on a ref older than b4475, the
            # pin's lower bound. With an inherited stdin it then sits at a prompt and burns
            # the FULL timeout (1800 s) before failing, instead of exiting at once. Costs
            # nothing on a correct run; turns a 30-minute hang into an immediate error on a
            # wrong one, which matters while the pin is still being chosen.
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                  stdin=subprocess.DEVNULL)
            stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, rc = (exc.stdout or ""), (exc.stderr or "") + "\nTIMEOUT", 124
        t1 = time.time()
        nodeclient.mark_all(marked, "run_end", run_id)
        record["conditions_end"] = conditions(marked)

        log = (stdout or "") + "\n" + (stderr or "")
        (out_dir / "llama.log").write_text(log, encoding="utf-8")

        record["started_utc"] = t0
        record["ended_utc"] = t1
        record["wall_s"] = round(t1 - t0, 3)
        record["returncode"] = rc
        record["timings"] = parse_timings(log)
        record["ok"] = rc == 0 and bool(record["timings"].get("decode_tokens"))
        if not record["ok"]:
            record["failure_hint"] = _failure_hint(log, rc)

        (out_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        self._append_window(run_id, record)
        return record

    def _append_window(self, run_id, record):
        """One window line per participating node, so the energy join is per node.

        Energy is per rail, and a distributed run consumes on several rails at once. The
        cluster figure is the sum over participants; keeping them separate first is what
        makes the per-node contribution visible.

        `t_start`/`t_end` are the WHOLE llama-cli process, and that meaning is fixed --
        this is the primary artefact and silently redefining a field in it is its own
        defect. But the whole process is the wrong numerator for any per-token figure:
        it contains model load and, for a decode metric, the entire prefill as well. The
        same mistake on the per-node bench path overstated joules-per-token by tens of
        percent (Pi4 C4), and it is worse here, because pairing this window with
        `decode_tokens` charges load AND prefill to decode alone.

        So the load/prefill/decode split goes in the record too, as `phases`, and the
        residual the split does not explain is carried on every sub-window as
        `unaccounted_ms`. Whoever writes the Phase 3 energy join gets the decomposition
        already recorded and does not have to re-run the cluster to obtain it.
        """
        path = self.data_dir / "runs" / run_id / "windows.jsonl"
        timings = record.get("timings", {})
        phases = self.phase_windows(record)
        with path.open("a", encoding="utf-8") as fh:
            for nid in record["participants"]:
                fh.write(json.dumps({
                    "run": run_id,
                    "node": nid,
                    "strategy": record["strategy"],
                    "model": record["model"],
                    "tier": record["tier"],
                    "layers": record["layers"].get(nid),
                    "t_start": record["started_utc"],
                    "t_end": record["ended_utc"],
                    "phase": "full",
                    "decode_tokens": timings.get("decode_tokens"),
                    "prefill_tokens": timings.get("prefill_tokens"),
                    # Additive; "full" above keeps its meaning. Empty when llama.cpp did
                    # not report the durations (a failed run), which is the honest answer
                    # -- never a guessed split.
                    "phases": phases,
                    "phase_split": "reconstructed-from-reported-durations" if phases else "",
                }) + "\n")

    # -- phase-split windows -------------------------------------------------
    def phase_windows(self, record) -> list:
        """Split a run's wall-clock bracket into load / prefill / decode sub-windows.

        llama.cpp reports the duration of each phase but not its absolute start, so the
        sub-windows are reconstructed by laying the reported durations end to end from the
        process start. That is exact to the extent the reported durations account for the
        whole run; the residual is reported as `unaccounted_ms` so a reader can see how
        much of the window the split does not explain.
        """
        t = record.get("timings") or {}
        t0, t1 = record.get("started_utc"), record.get("ended_utc")
        if not (t0 and t1 and t.get("prefill_ms") and t.get("decode_ms")):
            return []

        load_s = (t.get("load_ms") or 0) / 1000.0
        prefill_s = t["prefill_ms"] / 1000.0
        decode_s = t["decode_ms"] / 1000.0
        accounted = load_s + prefill_s + decode_s
        unaccounted = (t1 - t0) - accounted

        cursor = t0
        windows = []
        for phase, dur, tokens in (("load", load_s, None),
                                   ("prefill", prefill_s, t.get("prefill_tokens")),
                                   ("decode", decode_s, t.get("decode_tokens"))):
            if dur <= 0:
                continue
            windows.append({
                "run": record["run_id"], "phase": phase,
                "t_start": cursor, "t_end": cursor + dur,
                "tokens": tokens,
                "unaccounted_ms": round(unaccounted * 1000, 1),
            })
            cursor += dur
        return windows

    # -- idle baseline -------------------------------------------------------
    def idle_baseline(self, seconds=120, note="") -> dict:
        """Record a marked quiet window on every node - the baseline for marginal energy.

        Must be taken with the cluster up and idle, not powered down: an rpc-server waiting
        for work costs more than a node at rest, and it is the waiting cost that a placement
        decision actually trades against.
        """
        marked = [n for n in self.cluster.workers() if n.agent_port]
        began = nodeclient.mark_all(marked, "idle_baseline_begin", note)
        t0 = time.time()
        time.sleep(seconds)
        ended = nodeclient.mark_all(marked, "idle_baseline_end", note)
        # `nodes` used to be [n.id for n in marked] -- the nodes we ATTEMPTED to mark. The
        # Jetson has agent_port 9410 in cluster.json and ships no agent at all, so that list
        # named a node whose POST cannot have succeeded: the primary artefact asserted a
        # marker that does not exist. `mark_all` returns per-node ok/error via `fanout` and
        # both return values were discarded. Report what actually landed, and name the rest.
        ok = sorted(nid for nid in began
                    if began.get(nid, {}).get("ok") and ended.get(nid, {}).get("ok"))
        failed = sorted(set(n.id for n in marked) - set(ok))
        return {"t_start": t0, "t_end": time.time(), "seconds": seconds,
                "nodes": ok, "unmarked": failed}


def _failure_hint(log: str, rc: int) -> str:
    lowered = log.lower()
    if "connection refused" in lowered or "failed to connect" in lowered:
        return ("a worker's rpc-server is not listening - check `heteroctl rpc up` and the "
                "node's firewall")
    if "out of memory" in lowered or "cannot allocate" in lowered:
        return ("a worker ran out of memory - lower its RPC_MEM_MB or its layer share; the "
                "placement believed a memory budget the node could not honour")
    if "unknown argument" in lowered:
        return "llama.cpp flags have drifted from this harness - check `llama-completion --help`"
    if rc == 124:
        return "run exceeded its timeout - the pipeline may be stalled on a slow link"
    return "see llama.log"
