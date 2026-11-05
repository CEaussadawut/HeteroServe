#!/usr/bin/env bash
# Copy the DAQ firmware onto a Pico 2 already running MicroPython.
# Flashing MicroPython itself is a one-time manual step - see docs/03_energy_collector.md.
source "$(dirname "$0")/lib.sh"

PORT="${1:-$(cfg daq port)}"
have mpremote || { log "installing mpremote into the venv"; "$VENV/bin/pip" install --quiet mpremote; }
MPREMOTE="$VENV/bin/mpremote"
[ -x "$MPREMOTE" ] || MPREMOTE="$(command -v mpremote)"

log "copying firmware to $PORT"
"$MPREMOTE" connect "$PORT" fs cp "$HS_ROOT/firmware/pico2/ina226.py" :ina226.py
"$MPREMOTE" connect "$PORT" fs cp "$HS_ROOT/firmware/pico2/main.py" :main.py
log "resetting"
"$MPREMOTE" connect "$PORT" reset || true
sleep 2

log "sensor scan (expect 0x40 0x41 0x44 0x45):"
py -m control.cli daq sync || warn "DAQ did not answer - check the port and the wiring"
