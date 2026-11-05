#!/usr/bin/env bash
# Guided per-channel calibration against a known resistive load.
# Usage: scripts/05_calibrate.sh <channel>
source "$(dirname "$0")/lib.sh"
CH="${1:?usage: 05_calibrate.sh <channel 1-4>}"

cat <<TEXT

Calibration, channel $CH
------------------------
For each point:
  1. Disconnect the node. Put a known power resistor across the channel in its place.
  2. Measure the voltage ACROSS THE RESISTOR with the multimeter. Do not assume 5.00 V.
  3. Reference current is I = V / R, from those two measured values.
  4. Read what the INA226 reports for the same load (this script shows it).

Use two or three resistors spanning the node's real range (roughly 0.5 A to 1.5 A).
A single point sets the gain only, which is a scale check rather than a calibration.

TEXT

POINTS=()
while true; do
  read -rp "Resistor value in ohms (blank to finish): " R
  [ -z "$R" ] && break
  read -rp "Measured voltage across the resistor (V): " V
  log "reading the INA226 for 10 s - keep the load connected"
  py -m control.cli daq status --seconds 10 | sed -n "s/.*ch$CH /  channel $CH: /p"
  read -rp "Current the INA226 reported (A): " REPORTED
  REF="$(python3 -c "print($V/$R)")"
  log "reference I = $V / $R = $REF A"
  POINTS+=("--point" "$REPORTED:$REF")
done

[ ${#POINTS[@]} -eq 0 ] && die "no points collected"
py -m control.cli calibrate --channel "$CH" "${POINTS[@]}"
log "calibration written. Re-run the idle reading and confirm it moved the way you expect."
