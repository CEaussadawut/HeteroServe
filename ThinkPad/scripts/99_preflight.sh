#!/usr/bin/env bash
# Everything that must be true before a measurement session, in one command.
source "$(dirname "$0")/lib.sh"
fail=0

log "1/5 cluster readiness"
py -m control.cli preflight || fail=1

log "2/5 link tiers clear of leftover shaping"
py -m control.cli tier verify | grep -q '"shaped": true' && { warn "shaping is still applied somewhere"; fail=1; } || true

log "3/5 DAQ clock sync"
py -m control.cli daq sync || fail=1

log "4/5 DAQ channels reporting"
py -m control.cli daq status --seconds 5 || fail=1

log "5/5 calibration present for every channel"
for ch in 1 2 3 4; do
  if [ -f "$HS_ROOT/calibration/ch$ch.json" ]; then
    printf '  ch%s calibrated\n' "$ch"
  else
    warn "  ch$ch has no calibration - its rows cannot carry the +-5%% accuracy claim"
  fi
done

[ "$fail" -eq 0 ] && log "READY" || die "NOT READY - see above"
