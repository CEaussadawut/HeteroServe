#!/bin/bash
# Call the user to this PC (the user, 2026-10-11: "can you trigger alarm of this pc when you need an answer or when everything
# done?"): a critical desktop notification and the freedesktop alarm sound, REPEAT times. Never during a measurement on the
# laptop -- the sound and the notification would land in it.
# Louder since 2026-10-23 (the user: "i didn't wake up and don't know if alarm rang can you change the sound of alarm to be
# louder?"; it had played at the desktop's 31 % volume): the sound goes to the laptop's own speakers, since the default output
# can point at a Bluetooth headset, unmuted at 100 % for the alarm only (the old volume and mute are put back afterwards), and
# it rings for about a minute by default (10 x the 6-second alarm clock).
#     bash testbed/scripts/alarm_user.sh "the 14B evaluation is done" [REPEAT]
# Stopping it early (the user, 2026-10-24: "alarms sometime too long i want to have a way to close it"):
#   - press "Stop alarm" on the notification, or close the notification;
#   - or run, from any terminal:   bash testbed/scripts/alarm_user.sh stop
# Either way the sound stops within a second and the volume is put back. `stop` also closes the notification, and the
# notification of every earlier alarm that is still open. An alarm that rings out on its own leaves its notification up (the
# message, for when you come back) until you close it or run `stop`.
# 2026-10-25 (the user: "i don't see where to close the alarm previous rang ... your wait is not noticed it too"): the
# notification is alarm_notify.py's, not `notify-send --wait`'s. Ubuntu's AppArmor profile for notify-send has no rule to
# receive the notification server's ActionInvoked / NotificationClosed signals or any signal, so `notify-send --wait` never
# saw its button or its closing, waited for the whole session and could not be killed -- and, holding the caller's output
# open, kept the caller waiting after the sound had stopped. The notification's process now writes nothing to the caller's
# output.
# The volume to put back is kept in VOLF from the first alarm of a session until one restores it, so an alarm cut off
# before it could restore (SIGKILL) does not make the next one "restore" 100 %.
# ALARM_SOUND overrides the sound file and ALARM_SINK the sink's wpctl id (both for tests).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PIDF=${XDG_RUNTIME_DIR:-/tmp}/heteroserve-alarm.pid
VOLF=${PIDF%.pid}.volume
if [ "${1:-}" = stop ]; then
  said=""
  p=$(cat "$PIDF" 2>/dev/null)
  if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then kill -TERM "$p" && said="alarm stopped"; sleep 0.3; fi
  # only the helper itself (python3 .../alarm_notify.py HeteroServe ...), never an editor or a shell that names the file
  NPAT='^python3 [^ ]*alarm_notify[.]py HeteroServe'
  n=$(pgrep -f "$NPAT" | wc -l)
  if [ "$n" -gt 0 ]; then pkill -TERM -f "$NPAT"; said="${said:+$said; }$n notification(s) closed"; fi
  echo "${said:-no alarm is ringing}"
  exit 0
fi
MSG=${1:-HeteroServe needs you}; REPEAT=${2:-10}
SND=${ALARM_SOUND:-/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga}
ME=$$
echo "$ME" > "$PIDF"
# the built-in speakers' sink (its id changes across boots); the default sink when there is none
SINK=${ALARM_SINK:-$(wpctl status 2>/dev/null | awk '/Sinks:/ {s = 1; next} /Sources:/ {s = 0} s && /Built-in Audio/ {gsub(/[^0-9.]/, " "); print $1 + 0; exit}')}
[ -n "$SINK" ] && [ "$SINK" != 0 ] || SINK=@DEFAULT_AUDIO_SINK@
if [ -s "$VOLF" ]; then OLD=$(cat "$VOLF")           # an earlier alarm's, not yet put back
else
  OLD=$(wpctl get-volume "$SINK" 2>/dev/null)         # "Volume: 0.31", or "Volume: 0.31 [MUTED]"
  [ -n "$OLD" ] && printf '%s\n' "$OLD" > "$VOLF"
fi
PLAY=""; NSUB=""; STOPPED=0
restore() {
  [ -n "$OLD" ] || return 0
  wpctl set-volume "$SINK" "$(echo "$OLD" | awk '{print $2}')" 2>/dev/null
  case "$OLD" in *MUTED*) wpctl set-mute "$SINK" 1 2>/dev/null ;; esac
  rm -f "$VOLF"
}
cleanup() {
  [ -n "$PLAY" ] && { pkill -TERM -P "$PLAY" 2>/dev/null; kill "$PLAY" 2>/dev/null; }
  restore
  # stopped (by command, or by the notification's button): close the notification too; rung out: leave it up
  [ "$STOPPED" = 1 ] && [ -n "$NSUB" ] && pkill -TERM -P "$NSUB" 2>/dev/null
  [ "$(cat "$PIDF" 2>/dev/null)" = "$ME" ] && rm -f "$PIDF"
}
trap cleanup EXIT
trap 'STOPPED=1; exit 130' INT TERM
# The notification, with a Stop button: when it is closed or Stop is pressed, alarm_notify.py exits 0 and the alarm is stopped
# -- only while this same alarm still rings (its pid in PIDF), so a notification closed later stops nothing else. When the
# notification service is not there (exit 2), a plain notification is shown and the alarm rings as before (stop by command).
NIDF=${PIDF%.pid}.notification      # alarm_notify.py: the notification's id, then what happened to it
HLOG=${PIDF%.pid}.log               # one line per alarm: how its notification ended (for when an alarm stops unexpectedly)
( python3 "$HERE/alarm_notify.py" "HeteroServe" "$MSG" > "$NIDF" 2>/dev/null; rc=$?
  echo "$(date -Is) alarm $ME notification rc $rc: $(tr '\n' ' ' < "$NIDF")" >> "$HLOG"
  if [ "$rc" = 0 ]; then
    [ "$(cat "$PIDF" 2>/dev/null)" = "$ME" ] && kill -TERM "$ME" 2>/dev/null
  elif [ "$rc" = 2 ]; then
    notify-send -u critical -a HeteroServe "HeteroServe" "$MSG" 2>/dev/null
  fi ) < /dev/null > /dev/null 2>&1 &
NSUB=$!
wpctl set-mute "$SINK" 0 2>/dev/null; wpctl set-volume "$SINK" 1.0 2>/dev/null
TARGET=$(wpctl inspect "$SINK" 2>/dev/null | awk -F'"' '/node.name/ {print $2; exit}')
for i in $(seq 1 "$REPEAT"); do
  # in the background and waited for, so that a stop interrupts the sound at once rather than after the current 6 seconds
  { pw-play ${TARGET:+--target "$TARGET"} "$SND" 2>/dev/null || canberra-gtk-play -f "$SND" 2>/dev/null || printf '\a'; } &
  PLAY=$!
  wait "$PLAY" 2>/dev/null
  PLAY=""
done
