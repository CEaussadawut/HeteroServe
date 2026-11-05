#!/usr/bin/env bash
# Run the energy collector in the foreground (systemd unit does the same thing).
source "$(dirname "$0")/lib.sh"
exec py -m control.cli daq record
