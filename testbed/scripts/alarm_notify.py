#!/usr/bin/env python3
"""The alarm's desktop notification, with a "Stop alarm" button, for alarm_user.sh (2026-10-25).

    python3 testbed/scripts/alarm_notify.py "<title>" "<message>"

Why not `notify-send --wait`: Ubuntu's AppArmor profile for notify-send (/etc/apparmor.d/notify-send, "notify-send (enforce)")
lets it send Notify but has no rule to receive the server's ActionInvoked or NotificationClosed signals, nor any signal: so
`notify-send --wait` never learns that its button was pressed or its notification closed, waits until the session ends, and
cannot be killed (found 2026-10-25 01:50, five such processes left by the 24 October alarms). Python is not confined by it.

This shows one critical notification (app "HeteroServe", a "stop" action labelled "Stop alarm", no timeout), prints its id,
and waits:
  - the button pressed (any action) or the notification dismissed by the user (reason 2): prints "stop" or "closed 2 by
    the user", exit 0;
  - closed otherwise (reason 1, expired: GNOME keeps three notifications per app and drops the oldest when a fourth comes;
    found 2026-10-25 when a test alarm stopped itself): prints "closed <reason>, not by the user", exit 4 -- the alarm keeps
    ringing (stop it by command);
  - SIGTERM or SIGINT (the alarm stopped by command, or it rang out and alarm_user.sh stop clears it): it closes its own
    notification (GNOME lets only the sender close one) and exits 3;
  - no notification service, or Notify refused: exit 2 (alarm_user.sh then shows a plain notify-send without --wait).
"""
import signal, sys

try:
    import gi
    gi.require_version("Gio", "2.0")
    from gi.repository import Gio, GLib
except Exception as ex:                                   # no PyGObject: the caller falls back
    print("no gi: %r" % ex, file=sys.stderr); sys.exit(2)

BUS, PATH, IFACE = "org.freedesktop.Notifications", "/org/freedesktop/Notifications", "org.freedesktop.Notifications"


def main():
    title, body = (sys.argv[1:3] + ["HeteroServe", ""])[:2] if len(sys.argv) > 1 else ("HeteroServe", "")
    try:
        con = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        r = con.call_sync(BUS, PATH, IFACE, "Notify",
                          GLib.Variant("(susssasa{sv}i)", ("HeteroServe", 0, "", title, body, ["stop", "Stop alarm"],
                                                           {"urgency": GLib.Variant("y", 2)}, 0)),
                          GLib.VariantType("(u)"), Gio.DBusCallFlags.NONE, 5000, None)
    except Exception as ex:
        print("notify failed: %r" % ex, file=sys.stderr); sys.exit(2)
    nid = r.unpack()[0]
    print(nid, flush=True)
    loop, result = GLib.MainLoop(), {"rc": 3, "closing": False}

    def on_signal(_con, _sender, _path, _iface, member, params):
        args = params.unpack()                              # (GNOME sends these to the notification's sender only)
        if args[0] != nid:
            return
        if result["closing"]:                               # our own CloseNotification took effect (reason 3)
            print("closed %s by stop" % args[1], flush=True)
        elif member == "ActionInvoked" or args[1] == 2:     # the button, or dismissed by the user: stop the alarm
            print("stop" if member == "ActionInvoked" else "closed 2 by the user", flush=True)
            result["rc"] = 0
        else:                                               # expired (1: GNOME keeps 3 a source and drops the oldest),
            print("closed %s, not by the user" % args[1], flush=True)   # or closed by another program: keep ringing
            result["rc"] = 4
        loop.quit()
    for member in ("ActionInvoked", "NotificationClosed"):
        con.signal_subscribe(BUS, IFACE, member, PATH, None, Gio.DBusSignalFlags.NONE, on_signal)

    def on_term():
        if result["closing"]:
            return False
        result["closing"] = True
        try:                                                # our own notification: GNOME allows its sender to close it
            con.call_sync(BUS, PATH, IFACE, "CloseNotification", GLib.Variant("(u)", (nid,)), None,
                          Gio.DBusCallFlags.NONE, 3000, None)
        except Exception as ex:
            print("close failed: %r" % ex, file=sys.stderr, flush=True)
            loop.quit()
            return False
        GLib.timeout_add(1500, lambda: loop.quit() or False)   # wait up to 1.5 s for the server's NotificationClosed
        return False
    try:                                                    # GLib >= 2.84 moved it to GLibUnix
        gi.require_version("GLibUnix", "2.0")
        from gi.repository import GLibUnix
        add = GLibUnix.signal_add
    except (ImportError, ValueError, AttributeError):
        add = GLib.unix_signal_add
    for s in (signal.SIGTERM, signal.SIGINT):
        add(GLib.PRIORITY_HIGH, s, on_term)
    loop.run()
    sys.exit(result["rc"])


if __name__ == "__main__":
    main()
