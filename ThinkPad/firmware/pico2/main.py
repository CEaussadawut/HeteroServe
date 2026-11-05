"""HeteroServe DAQ firmware - Raspberry Pi Pico 2, MicroPython.

Polls four INA226 sensors over one I2C bus and streams timestamped samples to the control
node over USB serial. The Pico does one job and does it without interruption: it never
blocks on the host, never buffers more than a line, and never tries to interpret what it
is measuring.

Wire protocol (control/energy.py is the other half - change both together):

    host -> pico    PING        ->  PONG,<pico_us>
                    START       ->  OK,START
                    STOP        ->  OK,STOP
                    RATE,<hz>   ->  OK,RATE,<hz>
                    INFO        ->  INFO,<json>
                    SCAN        ->  SCAN,<addr>,<addr>,...
    pico -> host    D,<pico_us>,<ch>,<bus_mV>,<current_uA>,<power_uW>
                    E,<message>

Why the Pico timestamps rather than the host: USB CDC delivery jitters by milliseconds
under host load, so a host-side timestamp would record when Linux got round to reading the
line, not when the sample was taken. The Pico's counter is steady; control/energy.py maps
it onto host UTC once, with a measured offset and a stated uncertainty.

Install: copy ina226.py and this file to the Pico as /ina226.py and /main.py.
"""

import json
import sys
import time

from machine import I2C, Pin

import ina226

# ---------------------------------------------------------------- wiring
# GP4 = SDA, GP5 = SCL on I2C0. One bus, four sensors, distinct addresses.
# 4.7k pull-ups belong on the bus once, not per sensor.
I2C_ID = 0
SDA_PIN = 4
SCL_PIN = 5
I2C_FREQ = 400_000

# channel -> (i2c address, node label). The labels are for humans reading a raw capture;
# the authoritative channel->node map lives in the control node's cluster.json.
CHANNELS = (
    (1, 0x40, "pi5"),
    (2, 0x41, "pi4"),
    (3, 0x44, "jetson"),
    (4, 0x45, "orangepi-rv2"),
)

SHUNT_OHMS = 0.01
MAX_CURRENT_A = 3.2768          # CAL = 5120 with a 0.01 ohm shunt; see ina226.py
DEFAULT_RATE_HZ = 28            # matches AVG_16 + 1.1 ms conversions

LED = Pin("LED", Pin.OUT)


class Daq:
    def __init__(self):
        self.i2c = I2C(I2C_ID, sda=Pin(SDA_PIN), scl=Pin(SCL_PIN), freq=I2C_FREQ)
        self.sensors = []
        self.rate_hz = DEFAULT_RATE_HZ
        self.running = False
        self.errors = 0
        self.samples = 0

    def probe(self):
        """Attach to whichever sensors are actually present.

        A missing sensor is reported once and then skipped, rather than aborting: three
        good channels are worth having, and a silent channel that looks alive would be far
        worse than one that is openly absent.
        """
        found = []
        for channel, address, label in CHANNELS:
            sensor = ina226.INA226(self.i2c, address, SHUNT_OHMS, MAX_CURRENT_A)
            if sensor.present():
                sensor.configure()
                self.sensors.append((channel, sensor, label))
                found.append((channel, address, label))
            else:
                emit("E,channel %d (0x%02x, %s) did not answer" % (channel, address, label))
        return found

    def sample_once(self):
        now = time.ticks_us()
        for channel, sensor, _ in self.sensors:
            try:
                bus_mv, current_ua, power_uw = sensor.read_all()
            except OSError:
                self.errors += 1
                emit("E,i2c read failed on channel %d" % channel)
                continue
            # Integers on the wire: no float formatting cost on the Pico, and no decimal
            # rounding surprises on the host. Scaling back to volts/amps is the host's job.
            emit("D,%d,%d,%d,%d,%d" % (now, channel, int(bus_mv), int(current_ua), int(power_uw)))
            self.samples += 1

    def info(self):
        return json.dumps({
            "firmware": "heteroserve-daq/1",
            "channels": [{"channel": c, "label": l} for c, _, l in self.sensors],
            "rate_hz": self.rate_hz,
            "shunt_ohms": SHUNT_OHMS,
            "max_current_a": MAX_CURRENT_A,
            "samples": self.samples,
            "errors": self.errors,
            "uptime_us": time.ticks_us(),
        })


def emit(line):
    sys.stdout.write(line + "\n")


def read_command():
    """Non-blocking single-line read from USB serial.

    select.poll on sys.stdin keeps the sample loop free-running: a DAQ that stalls waiting
    for a command it may never receive would leave gaps in the power trace, and a gap is
    exactly what makes an energy integral unusable.
    """
    try:
        import select
        poller = select.poll()
        poller.register(sys.stdin, select.POLLIN)
        if not poller.poll(0):
            return None
        return sys.stdin.readline().strip()
    except Exception:
        return None


def main():
    daq = Daq()
    found = daq.probe()
    emit("E,booted with %d of %d channels" % (len(found), len(CHANNELS)))

    period_us = int(1_000_000 / daq.rate_hz)
    next_due = time.ticks_us()

    while True:
        command = read_command()
        if command:
            parts = command.split(",")
            verb = parts[0].upper()
            if verb == "PING":
                emit("PONG,%d" % time.ticks_us())
            elif verb == "START":
                daq.running = True
                next_due = time.ticks_us()
                LED.on()
                emit("OK,START")
            elif verb == "STOP":
                daq.running = False
                LED.off()
                emit("OK,STOP")
            elif verb == "RATE" and len(parts) > 1:
                try:
                    daq.rate_hz = max(1, min(200, int(parts[1])))
                    period_us = int(1_000_000 / daq.rate_hz)
                    emit("OK,RATE,%d" % daq.rate_hz)
                except ValueError:
                    emit("E,bad rate")
            elif verb == "INFO":
                emit("INFO," + daq.info())
            elif verb == "SCAN":
                emit("SCAN," + ",".join("0x%02x" % a for a in daq.i2c.scan()))
            else:
                emit("E,unknown command: " + command)

        if daq.running:
            now = time.ticks_us()
            if time.ticks_diff(now, next_due) >= 0:
                daq.sample_once()
                # Advance by whole periods so a slow cycle does not shift the whole
                # schedule: the sample rate stays nominal instead of drifting slower.
                next_due = time.ticks_add(next_due, period_us)
                if time.ticks_diff(time.ticks_us(), next_due) > 0:
                    next_due = time.ticks_us()
        else:
            time.sleep_ms(5)


if __name__ == "__main__":
    main()
