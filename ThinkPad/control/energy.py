"""Energy collector: the Pico 2 DAQ link, clock alignment, calibration and recording.

The Pico has no real-time clock. It timestamps samples with its own microsecond counter,
and this module maps that counter onto host UTC so power samples share a time base with
the benchmark harness and with every node's phase markers. Everything in the project that
says "joules per token" rests on that mapping being right, so it is measured with a
round-trip handshake and its residual uncertainty is recorded alongside the data rather
than assumed away.

Wire protocol (see firmware/pico2/main.py - the two must be changed together):

    host -> pico    PING            ->  PONG,<pico_us>
                    START           ->  OK,START
                    STOP            ->  OK,STOP
                    RATE,<hz>       ->  OK,RATE,<hz>
                    INFO            ->  INFO,<json>
    pico -> host    D,<pico_us>,<ch>,<bus_mV>,<current_uA>,<power_uW>
                    E,<message>     (error, e.g. a channel that stopped responding)
"""
from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import serial  # pyserial
except ImportError:  # pragma: no cover - import error is reported where it matters
    serial = None


class DaqError(Exception):
    pass


POWER_CSV_HEADER = ["t_utc", "channel", "node", "bus_v", "current_a", "power_w", "pico_us"]


# --------------------------------------------------------------- calibration
@dataclass
class Calibration:
    """Per-channel correction from the known-resistive-load procedure.

    current_measured = gain * current_reported + offset_a

    Both terms come from measuring a real resistor: reference current is I = V/R with V
    read on the multimeter and R the resistor's marked value, at two or three points across
    the node's operating range. A calibration that exists only as a note in a lab book is
    not reproducible, so these live in calibration/*.json next to the data.
    """
    channel: int
    gain: float = 1.0
    offset_a: float = 0.0
    shunt_ohms: float = 0.01
    points: list = None
    calibrated_at: str = ""
    note: str = ""

    def apply(self, current_a: float) -> float:
        return self.gain * current_a + self.offset_a

    @classmethod
    def load(cls, path):
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not str(raw.get("calibrated_at", "")).strip():
            # A calibration with no date is not a measurement, and applying it is worse
            # than having none: gain and offset silently rescale every joule on the
            # channel, and the channel then counts as calibrated, so analysis stops
            # flagging it. Refuse, consistent with the rest of the pipeline.
            raise DaqError(
                "%s has an empty calibrated_at - refusing to use an undated calibration. "
                "Run `heteroctl calibrate` and record the date, or delete the file."
                % path.name)
        return cls(channel=int(raw["channel"]), gain=float(raw.get("gain", 1.0)),
                   offset_a=float(raw.get("offset_a", 0.0)),
                   shunt_ohms=float(raw.get("shunt_ohms", 0.01)),
                   points=raw.get("points"), calibrated_at=raw["calibrated_at"],
                   note=raw.get("note", ""))

    @classmethod
    def load_all(cls, directory) -> dict:
        """Only `ch<N>.json` counts. The glob is deliberately strict: `ch*.json` also
        matched the shipped `ch2.example.json`, which handed the Pi 4 an invented gain of
        0.994 and, worse, marked its channel calibrated so `analysis` never raised the
        `uncalibrated-channel` flag. The Pi 4 was the only node with an example file, so
        the bias landed on exactly the axis this study measures.
        """
        directory = Path(directory)
        out = {}
        if not directory.exists():
            return out
        for path in sorted(directory.glob("ch[0-9].json")):
            cal = cls.load(path)
            out[cal.channel] = cal
        return out


def fit_calibration(points: list) -> tuple:
    """Least-squares gain and offset from (reported_a, reference_a) pairs.

    Two points give an exact line; three or more are fitted. With one point only the gain
    is set and the offset is held at zero - stated explicitly because a single-point
    "calibration" is really a scale check.
    """
    if not points:
        raise DaqError("no calibration points")
    if len(points) == 1:
        reported, reference = points[0]
        if reported == 0:
            raise DaqError("cannot calibrate from a zero-current point")
        return reference / reported, 0.0

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        raise DaqError("all calibration points are at the same current")
    gain = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    return gain, mean_y - gain * mean_x


# ------------------------------------------------------------------- the DAQ
class Daq:
    """Serial link to the Pico 2, with clock alignment and calibrated decoding."""

    def __init__(self, port: str, baud: int = 921600, channels: dict | None = None,
                 calibrations: dict | None = None, timeout: float = 2.0):
        if serial is None:
            raise DaqError("pyserial is not installed - pip install -r requirements.txt")
        self.port_name = port
        self.baud = baud
        self.channels = channels or {}       # channel int -> node id
        self.calibrations = calibrations or {}
        self.timeout = timeout
        self.ser = None
        self.offset_s = None                 # host_utc = pico_us/1e6 + offset_s
        self.offset_uncertainty_s = None

    # -- connection ---------------------------------------------------------
    def open(self):
        self.ser = serial.Serial(self.port_name, self.baud, timeout=self.timeout)
        time.sleep(0.3)                      # USB CDC needs a moment before it talks
        self.ser.reset_input_buffer()
        return self

    def close(self):
        if self.ser:
            try:
                self.command("STOP")
            except Exception:  # noqa: BLE001
                pass
            self.ser.close()
            self.ser = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def command(self, text: str, expect_reply: bool = True, tries: int = 3):
        self.ser.write((text + "\n").encode())
        self.ser.flush()
        if not expect_reply:
            return None
        for _ in range(tries * 50):
            line = self.ser.readline().decode(errors="replace").strip()
            if not line:
                continue
            if line.startswith("D,"):
                continue                     # a data line racing the reply
            return line
        raise DaqError("no reply to %r" % text)

    # -- clock alignment ----------------------------------------------------
    def sync_clock(self, rounds: int = 21) -> dict:
        """Map the Pico's microsecond counter onto host UTC.

        Each round brackets a PING/PONG with host timestamps and assumes the reply lands at
        the midpoint of the round trip - the same assumption NTP makes. The median of many
        rounds rejects the occasional USB scheduling delay; the spread across rounds is
        reported as the residual uncertainty and belongs in the error budget.
        """
        offsets, rtts = [], []
        for _ in range(rounds):
            t0 = time.time()
            reply = self.command("PING")
            t1 = time.time()
            if not reply.startswith("PONG,"):
                continue
            pico_us = int(reply.split(",")[1])
            midpoint = (t0 + t1) / 2.0
            offsets.append(midpoint - pico_us / 1e6)
            rtts.append(t1 - t0)

        if len(offsets) < 3:
            raise DaqError("clock sync failed - the DAQ answered %d of %d pings" % (len(offsets), rounds))

        self.offset_s = statistics.median(offsets)
        self.offset_uncertainty_s = statistics.pstdev(offsets) if len(offsets) > 1 else None
        return {
            "offset_s": self.offset_s,
            "offset_stdev_s": self.offset_uncertainty_s,
            "rtt_median_s": statistics.median(rtts),
            "rounds": len(offsets),
            "synced_at_utc": time.time(),
        }

    def host_time(self, pico_us: int) -> float:
        if self.offset_s is None:
            raise DaqError("clock not synced - call sync_clock() before recording")
        return pico_us / 1e6 + self.offset_s

    # -- sampling -----------------------------------------------------------
    def info(self) -> str:
        return self.command("INFO")

    def start(self, rate_hz: int | None = None):
        if rate_hz:
            self.command("RATE,%d" % int(rate_hz))
        return self.command("START")

    def stop(self):
        return self.command("STOP")

    def read_sample(self):
        """One decoded, calibrated sample, or None for a non-data line.

        Returns (t_utc, channel, node, bus_v, current_a, power_w, pico_us).

        Power is recomputed as V*I after the calibration correction rather than taken from
        the INA226's on-chip power register: the correction applies to current, and an
        uncorrected power register would disagree with the corrected current by exactly the
        correction. One consistent number is worth more than two nearly-equal ones.
        """
        raw = self.ser.readline().decode(errors="replace").strip()
        if not raw:
            return None
        if raw.startswith("E,"):
            raise DaqError("DAQ reported: " + raw[2:])
        if not raw.startswith("D,"):
            return None

        try:
            _, pico_us, ch, bus_mv, cur_ua, _pow_uw = raw.split(",")
            pico_us, ch = int(pico_us), int(ch)
            bus_v = int(bus_mv) / 1000.0
            current_a = int(cur_ua) / 1e6
        except ValueError:
            return None

        cal = self.calibrations.get(ch)
        if cal:
            current_a = cal.apply(current_a)
        return (self.host_time(pico_us), ch, self.channels.get(ch), bus_v,
                current_a, bus_v * current_a, pico_us)


# ------------------------------------------------------------------ recorder
class Recorder:
    """Streams the DAQ to a CSV for the duration of a session, in a background thread.

    One file per session, appended continuously. Experiment windows are cut out of it
    afterwards by timestamp (see analysis.py), so the recorder never needs to know what
    the harness is doing - it just has to be running, and to be honest about gaps.
    """

    def __init__(self, daq: Daq, out_path, flush_every: int = 200):
        self.daq = daq
        self.out_path = Path(out_path)
        self.flush_every = flush_every
        self._thread = None
        self._stop = threading.Event()
        self.samples = 0
        self.errors = 0
        self.last_sample = None
        self.started_at = None

    def _run(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.out_path.exists()
        with self.out_path.open("a", encoding="utf-8") as fh:
            if new_file:
                fh.write(",".join(POWER_CSV_HEADER) + "\n")
            since_flush = 0
            while not self._stop.is_set():
                try:
                    sample = self.daq.read_sample()
                except DaqError as exc:
                    self.errors += 1
                    fh.write("# error,%s,%s\n" % (time.time(), str(exc).replace(",", ";")))
                    continue
                except Exception as exc:  # noqa: BLE001
                    self.errors += 1
                    fh.write("# error,%s,%s\n" % (time.time(), str(exc).replace(",", ";")))
                    time.sleep(0.1)
                    continue
                if sample is None:
                    continue
                t_utc, ch, node, bus_v, current_a, power_w, pico_us = sample
                fh.write("%.6f,%d,%s,%.4f,%.6f,%.5f,%d\n" %
                         (t_utc, ch, node or "", bus_v, current_a, power_w, pico_us))
                self.samples += 1
                self.last_sample = sample
                since_flush += 1
                if since_flush >= self.flush_every:
                    fh.flush()
                    since_flush = 0

    def start(self):
        if self.daq.offset_s is None:
            raise DaqError("refusing to record before the clock is synced")
        self.started_at = time.time()
        self.daq.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="daq-recorder", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        try:
            self.daq.stop()
        except Exception:  # noqa: BLE001
            pass
        return {
            "samples": self.samples,
            "errors": self.errors,
            "duration_s": time.time() - (self.started_at or time.time()),
            "path": str(self.out_path),
        }

    def status(self) -> dict:
        rate = None
        if self.started_at and self.samples:
            rate = self.samples / max(1e-6, time.time() - self.started_at)
        return {
            "samples": self.samples,
            "errors": self.errors,
            "effective_hz_all_channels": round(rate, 2) if rate else None,
            "last": self.last_sample,
            "path": str(self.out_path),
        }


def open_daq(cluster, calibration_dir=None) -> Daq:
    """Build a Daq from the cluster config: port, baud, channel->node map, calibrations."""
    daq_cfg = cluster.daq or {}
    channels = {int(ch): spec["node"] for ch, spec in (daq_cfg.get("channels") or {}).items()}
    cal_dir = Path(calibration_dir) if calibration_dir else Path(cluster.path).parent.parent / "calibration"
    calibrations = Calibration.load_all(cal_dir)

    missing = sorted(set(channels) - set(calibrations))
    if missing:
        # Not fatal: an uncalibrated channel still records, and the raw current is still
        # useful. But the analysis marks its rows, because an uncalibrated channel cannot
        # carry the +-5% accuracy claim the success criteria make.
        print("warning: no calibration for channel(s) %s - their rows will be flagged"
              % ", ".join(str(c) for c in missing))

    return Daq(port=daq_cfg.get("port", "/dev/ttyACM0"),
               baud=int(daq_cfg.get("baud", 921600)),
               channels=channels, calibrations=calibrations)
