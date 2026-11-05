#!/usr/bin/env python3
"""The Jetson's procedure before and after a load, as lc4_20261024.py does it inline (2026-10-28, for the phase scripts):
its power guard started by its NOTE's procedure (or found live and un-tripped), MAXN, the clocks locked for the run and
restored after, and the guard stopped if this run started it. Import and use up() / down():

    st = jetson_proc_20261028.up(say, "phase_smoke_<stamp>")
    try: ...
    finally: jetson_proc_20261028.down(say, st)
"""
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lc4_20261024 as L                                     # noqa: E402


def up(say, tag):
    st = {"guard_started": False, "locked": False, "stored": "~/heteroserve/run/jetson_clocks_%s.stored" % tag}
    rc, g = L.sh("jetson", 'grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l')
    if g.strip() == "0":
        rc, c = L.sh("jetson", 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum ~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1')
        if c.split() != ["synced", L.sha(os.path.join(HERE, "jetson_power_guard.sh"))]:
            raise SystemExit("FATAL: Jetson guard precheck %r" % c.split())
        L.sh("jetson", "sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo started")
        st["guard_started"] = True; time.sleep(5)
    rc, g = L.sh("jetson", 'm=$(cut -d" " -f1 /var/lib/nvpmodel/status); G=~/heteroserve/logs/power-guard.log; '
                           't=$(awk "/guard start/ {t=0} /TRIP/ {t++} END {print t+0}" $G); echo "$m $t"')
    if g.split() != ["pmode:0000", "0"]:
        down(say, st)
        raise SystemExit("FATAL: Jetson not MAXN with an un-tripped guard: %r" % g)
    rc, o = L.sh("jetson", "c=/sys/devices/system/cpu/cpu0/cpufreq; g=/sys/devices/57000000.gpu/devfreq/57000000.gpu; "
                           "if [ $(cat $c/scaling_min_freq) = $(cat $c/scaling_max_freq) ] && [ $(cat $g/min_freq) = $(cat $g/max_freq) ]; "
                           "then echo ALREADY_LOCKED; else mkdir -p ~/heteroserve/run && sudo -n jetson_clocks --store %s && sudo -n jetson_clocks && echo LOCKED; fi" % st["stored"])
    st["locked"] = o.strip() == "LOCKED"
    if o.strip() not in ("LOCKED", "ALREADY_LOCKED"):
        down(say, st)
        raise SystemExit("FATAL: could not lock the Jetson's clocks: %r" % o)
    say("jetson guard %s, clocks %s" % ("started" if st["guard_started"] else "found live", o.strip()))
    return st


def down(say, st):
    if st.get("locked"):
        rc, o = L.sh("jetson", "m=$(cut -d' ' -f1 /var/lib/nvpmodel/status); [ \"$m\" = pmode:0000 ] && sudo -n jetson_clocks --restore %s && echo RESTORED" % st["stored"])
        say("jetson clocks %s" % o.strip())
    if st.get("guard_started"):
        L.sh("jetson", 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p; done')
        say("jetson guard stopped")
    rc, o = L.sh("jetson", 'G=~/heteroserve/logs/power-guard.log; awk "/guard start/ {t=0} /TRIP/ {t++} END {print t+0}" $G')
    say("jetson guard trips this run: %s" % o.strip())
