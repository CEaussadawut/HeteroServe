#!/bin/bash
# Serve a model on the pooled cluster for interactive use: llama-server on the host board, ggml-rpc-server on the
# workers, the split planned by the engine for a SERVING context (the benchmark cells plan for 80 tokens). Not a
# measurement: nothing here is a reported figure. From the repo root on the laptop:
#     bash testbed/scripts/serve_pool.sh start     # plan, gate, start the workers and the server, wait until healthy
#     bash testbed/scripts/serve_pool.sh status
#     bash testbed/scripts/serve_pool.sh stop      # stops exactly what start started (exact argv), and checks
#     MODEL=qwen3moe CTX=4096                      # (defaults) a predict_pool.py model key; the context in tokens
#     MEMFILE=testbed/dataset/memory_snapshot_<stamp>.json   # plan against a dated snapshot of the boards' free memory
#                                                  # (memory_snapshot.py) instead of memory_at_rest.json (the default)
#     CLOCKS=locked                                # (default) lock the Jetson's clocks for the session; asfound leaves them
#     NODES="laptop pi5 pi4 jetson"                # (default from a terminal since 2026-11-01; a harness that names no NODES
#                                                  # keeps "pi5 pi4 jetson", serve_defaults.sh default_nodes) the boards the
#                                                  # engine may plan on (2026-09-30: "pi5 pi4" for the prima.cpp comparison,
#                                                  # whose runtime cannot use the Jetson)
#     PLACE="pi4:1,jetson:21" PLACE_HEAD=last      # serve a GIVEN placement instead of the engine's (the full evaluation)
#     NODES="pi4 jetson laptop"                    # the laptop is a node since 2026-10-11 (the user: "make it as one of the
#                                                  # node"): an RPC worker on 192.168.1.108:50053 or the host. It has no sshd,
#                                                  # so its commands run here (on()); its tree is /opt/heteroserve/llama.cpp
#     SERVE_HOST=pi4                               # the board that runs llama-server. Default (2026-10-08): every board
#                                                  # holding the model file (the newest model_inventory_*.json) is planned
#                                                  # as host and the fastest plan wins: TinyLlama goes on the Jetson alone
# The output head goes where the engine puts it (plan_optimal's default head_placement="best", 2026-10-08): on the host
# when that is cheaper, realised with -ot "^output\.weight=CPU" (predict_pool.bench_args(head_at=0), as the 2026-10-07
# head-to-head ran it), and the gate and the prediction charge it there. After loading, each worker's model buffer, as
# llama.cpp reports it, must match the plan within 1 MiB, or the server is stopped: a head left on the wrong board would
# show as its ~600 MiB (llama-server does not print the "buffer type overridden" lines, which are debug-level).
# Carries run_pool.sh's rules: nothing loads a busy board or a tree off the pin; each board's free memory NOW must
# cover the engine's charge plus its reserve; the Jetson must be MAXN with a live, un-tripped guard and eth0 its only
# router-LAN interface; worker readiness via `ss` on the worker itself (a bare TCP connect-then-close kills
# ggml-rpc-server); processes found by exact argv, never pkill -f. The server binds the host's router-LAN address only
# (the router has no WAN).
# The Jetson's clocks, when it serves: LOCKED for the session with run_pool.sh's own commands -- stored first, locked, and
# restored by `stop` only if this start locked them and the power mode is still MAXN (already locked: left, and not restored).
# Decided 2026-09-30 (user: "yes, lock the Jetson clocks in serve_pool.sh"): QR3 measured that its governors (nvhost_podgov,
# schedutil) cost 24 % of the 14B's mean time per token and 37 % at the P95 (benchmarks.md, "QR3"). Until then this script
# left them as found; CLOCKS=asfound still does.
# Speculative decoding (scheduler upgrade 1, 2026-10-10). THE DEFAULT (the user, 2026-10-10: "Laptop draft"): a person
# serving Qwen3-14B from a terminal gets the Qwen3-0.6B draft on this laptop, whose ggml-rpc-server this script starts and
# stops. A harness (stdout not a terminal: full_eval.py, spec_serve_20261010.py) serves exactly what its registration names.
# MTP (the user, 2026-11-04: "Turn MTP 1 on"): a person serving a pooled plan of Qwen3.5-9B or Qwen3.6-35B-A3B from a terminal
# gets speculation with the model's own MTP head, MTP 1 (serve_defaults.sh mtp_mode; SERVE_MTP=off turns it off, SERVE_MTP=on
# asks for it from anywhere); without it when the Pi 5 is in the plan with its TSO on; a harness never.
# The boards' switch-off (the user, 2026-10-25: "Interactive default"; every board since 2026-11-01, the scheduler review's
# item 7): a person serving from a terminal gets each of the Pi 4, the Pi 5 and the Jetson shut down (the Jetson halted) and
# its outlet cut once the served plan leaves it out, and powered up first, all at once, when a plan needs it while it is off
# (the Jetson's power guard started after it boots); a harness never switches. SERVE_SWITCH=off turns it off,
# SERVE_SWITCH=on asks for it from anywhere (serve_defaults.sh, strip_switch_20261101.py).
# The rules and settings are in serve_defaults.sh. SERVE_DRAFT=default forces the laptop draft; SERVE_DRAFT=none turns it off;
# otherwise, as before:
#     SERVE_DRAFT="-md ~/models/qwen3-0.6b-q8_0.gguf --spec-type draft-simple --spec-draft-n-max 1 ..."
#         extra llama-server arguments; the server then runs with LLAMA_TRACE=1, so its log has one
#         "accepted n/m draft tokens" line per check
#     DRAFT_HOST_MIB=800   what the draft adds on the host (it inherits -lm none: its weights, its KV cache at CTX, its
#                          compute buffer), added to the host's gate and charge
#     SERVE_RPC_EXTRA=192.168.1.108:50053   one more ggml-rpc-server, started by the caller (the draft's board): appended
#         to --rpc, with the main model pinned to the planned workers' devices (-dev RPC0..RPC<n-1>), so that only the
#         draft goes there (the caller's -devd RPC<n>, n the planned workers' count)
set -u
cd "$(dirname "$0")/../.."
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=6"
STATE=$HOME/.cache/heteroserve-wf/serve_pool.state
MODEL=${MODEL:-qwen3moe}; CTX=${CTX:-4096}; PORT=50052; HTTP=8080; CLOCKS=${CLOCKS:-locked}
# llama-server's prompt cache OFF by default (the user, 2026-10-21, "Cache off"): at the pin it keeps up to 8,192 MiB of
# earlier prompts' state in the HOST's RAM, which the engine does not charge. Qwen3.8-27B on Pi 5 28 + Jetson 15 + Pi 4 22 (the
# Pi 5 at 671 MiB free at READY) answered every second request with no tokens and was OOM-killed in round 5 while saving
# 162 MiB of state (full_eval_q27b_fe4_20261020_20261021_141403); the cache also copied a worker's KV back at every new request
# (2026-10-10). SERVE_CACHE_RAM=<MiB> restores it.
CACHE_RAM=${SERVE_CACHE_RAM:-0}
# SERVE_EXTRA (2026-10-24): further llama-server arguments, appended after the draft's, e.g. "--backend-sampling" for the
# head-board sampling study. Unset (every registration before it), the command line is exactly as before.
EXTRA=${SERVE_EXTRA:-}
. testbed/scripts/serve_defaults.sh              # the serving defaults: draft_mode, laptop_draft_args, laptop_rpc_pid
DRAFT_GIVEN=${SERVE_DRAFT-__unset__}; DRAFT_HOST_MIB=${DRAFT_HOST_MIB:-}; RPC_EXTRA=${SERVE_RPC_EXTRA:-}; DRAFT=""; LAPTOP_RPC=0
SWITCH_GIVEN=${SERVE_SWITCH-__unset__}             # the Pi 4's switch-off default (serve_defaults.sh, 2026-10-25)
INTERACTIVE=0; [ -t 1 ] && INTERACTIVE=1           # read before start's tee takes stdout: is a person at this terminal?
NODES=${NODES:-$(default_nodes "$INTERACTIVE")}; export NODES   # serve_defaults.sh: a person gets the laptop too (2026-11-01)
SERVE_PLANNER_MODE=$(planner_mode "${SERVE_PLANNER-__unset__}" "$INTERACTIVE"); export SERVE_PLANNER_MODE   # v4 | v3 | shipped
SERVE_MTP_MODE=$(mtp_mode "${SERVE_MTP-__unset__}" "$MODEL" "$INTERACTIVE" "$DRAFT_GIVEN" "$EXTRA"); export SERVE_MTP_MODE   # on | off
[ "$SERVE_PLANNER_MODE" = invalid ] && { echo "FATAL: SERVE_PLANNER=$SERVE_PLANNER (v4, v3 or shipped)"; exit 2; }
case "$CLOCKS" in locked|asfound) ;; *) echo "FATAL: CLOCKS=$CLOCKS (locked or asfound)"; exit 2 ;; esac
declare -A WIP=([pi5]=192.168.10.11 [pi4]=192.168.10.12 [jetson]=192.168.1.114)   # the Jetson: 192.168.100.111 until 2026-10-08, .1.109 until 2026-10-09
declare -A LAN=([pi5]=192.168.1.122 [pi4]=192.168.1.110 [jetson]=192.168.1.114)   # 2026-10-08: the home router (192.168.1.x); the Pi 5 back 2026-10-19 (a replacement board), .100.107 before
declare -A WEXTRA=([pi5]="-t 4" [pi4]="-t 4" [jetson]="-d Vulkan0")
WIP[laptop]=192.168.1.108; LAN[laptop]=192.168.1.108; WEXTRA[laptop]="-t 4"     # the laptop as a node (2026-10-11)
declare -A WPORT=([laptop]=50053) LDIR=([laptop]=/opt/heteroserve/llama.cpp)   # its worker port (never 50052) and its tree
on() { if [ "$1" = laptop ]; then bash -c "$2" < /dev/null; else $SSH $1 "$2"; fi; }   # a command on a node: here, or over ssh
wport() { echo "${WPORT[$1]:-$PORT}"; }
ldir() { if [ -n "${LDIR[$1]:-}" ]; then echo "${LDIR[$1]}"; else echo '~/llama.cpp'; fi; }   # the ~ expands on the node

wpid() { on $1 "for p in \$(pgrep -x ggml-rpc-server); do tr '\\0' ' ' 2>/dev/null < /proc/\$p/cmdline | grep -qF -- '-H ${WIP[$1]} -p $(wport $1) ' && echo \$p; done"; }
spid() { on $1 "for p in \$(pgrep -x llama-server); do tr '\\0' ' ' 2>/dev/null < /proc/\$p/cmdline | grep -qF -- '--host ${LAN[$1]} --port $HTTP' && echo \$p; done"; }

stop() {
  [ -f "$STATE" ] || { echo "nothing started by serve_pool.sh (no $STATE)"; return 0; }
  . "$STATE"
  local n p i
  for n in $HOST; do
    p=$(spid $n); [ -n "$p" ] && on $n "kill $p"
    for i in $(seq 1 20); do [ -z "$(spid $n)" ] && break; sleep 1; done
    # llama-server takes the first SIGTERM as a graceful shutdown, which hung with its RPC worker still up (2026-10-08, twice:
    # still running 90 s on); a second signal makes it exit at once ("Received second interrupt, terminating immediately")
    p=$(spid $n); [ -n "$p" ] && { echo "stop    llama-server on $n still up after 20 s: a second TERM"; on $n "kill $p"; }
    for i in $(seq 1 30); do [ -z "$(spid $n)" ] && break; sleep 1; done
    [ -z "$(spid $n)" ] && echo "stop    llama-server on $n stopped" || echo "!!!!! stop: llama-server on $n STILL RUNNING (pid $(spid $n))"
  done
  for n in $WORKERS; do
    p=$(wpid $n); [ -n "$p" ] && on $n "kill $p"
    for i in $(seq 1 10); do [ -z "$(wpid $n)" ] && break; sleep 1; done
    [ -z "$(wpid $n)" ] && echo "stop    worker on $n stopped" || echo "!!!!! stop: worker on $n STILL RUNNING (pid $(wpid $n))"
  done
  if [ "${LAPTOP_RPC_STARTED:-0}" = 1 ]; then    # only the laptop draft server this start started
    p=$(laptop_rpc_pid); [ -n "$p" ] && kill $p
    for i in $(seq 1 10); do [ -z "$(laptop_rpc_pid)" ] && break; sleep 1; done
    [ -z "$(laptop_rpc_pid)" ] && echo "stop    the laptop's draft RPC server stopped" || echo "!!!!! stop: the laptop's draft RPC server STILL RUNNING (pid $(laptop_rpc_pid))"
  fi
  if [ "${LOCKED:-0}" = 1 ]; then   # only what this start locked (run_pool.sh's cleanup, verbatim but the file name)
    $SSH jetson "m=\$(cut -d' ' -f1 /var/lib/nvpmodel/status); if [ \"\$m\" = pmode:0000 ]; then sudo -n jetson_clocks --restore ~/heteroserve/run/jetson_clocks_serve_$CLOCKSTAMP.stored && echo 'stop    jetson clocks restored'; else echo \"!!!!! stop: jetson power mode is \$m -- clocks NOT restored\"; fi"
  fi
  rm -f "$STATE"
}

status() {
  [ -f "$STATE" ] && { . "$STATE"; echo "started $STARTED: host $HOST, workers [$WORKERS], $ARGS -c $CTX"; } || echo "no serve_pool.sh state"
  local n
  for n in pi5 pi4 jetson laptop; do echo "$n: server [$(spid $n)] worker [$(wpid $n)]"; done
  echo "laptop: draft RPC server [$(laptop_rpc_pid)]"
  $SSH jetson "c=/sys/devices/system/cpu/cpu0/cpufreq; g=/sys/devices/57000000.gpu/devfreq/57000000.gpu; echo jetson clocks: cpu \$(cat \$c/scaling_min_freq)-\$(cat \$c/scaling_max_freq) kHz, gpu \$(cat \$g/min_freq)-\$(cat \$g/max_freq) Hz"
  curl -s -m 5 "http://${LAN[${HOST:-pi4}]}:$HTTP/health"; echo
}

start() {
  [ -f "$STATE" ] && { echo "FATAL: already started (state $STATE); stop first"; exit 1; }
  STAMP=$(date +%Y%m%d_%H%M%S); OUT=testbed/results/serve_${MODEL}_$STAMP.log
  exec > >(trap '' INT TERM HUP QUIT PIPE; exec tee -i -a "$OUT") 2>&1
  PIN=$(python3 -c "import json;print(json.load(open('ThinkPad/config/cluster.json'))['llama']['pin_sha'])")
  PLANNED=$(python3 - "$MODEL" "$CTX" <<'PY'
import dataclasses, json, os, shlex, sys
sys.path.insert(0, "testbed/scripts")
import predict_pool as P
from control.placement import plan_optimal
key, ctx = sys.argv[1], int(sys.argv[2])
ds = "testbed/dataset"
mem = json.load(open(os.environ.get("MEMFILE") or ds + "/memory_at_rest.json"))
nodes = (os.environ.get("NODES") or "pi5 pi4 jetson").split()
# The host as well as the split (2026-10-08): SERVE_HOST if given, else every board holding the model's file (the newest
# model_inventory_*.json), each planned as host, the fastest predicted decode winning (predict_pool.plan_host_choice).
place = os.environ.get("PLACE")   # a fixed placement, "pi4:1,jetson:21" (the host first), with PLACE_HEAD=host|last
mode, depth, why, pend = os.environ.get("SERVE_PLANNER_MODE", "v3"), 0, "", ""
if place:
    # SERVE_PLANNER=v4 (2026-11-01): a given placement priced and charged as PLANNER_V4 does it (a hybrid's memory as
    # llama.cpp allocates it), at depth 0; a harness without it, as before
    r = P.plan_fixed(key, [(n, int(k)) for n, k in (x.split(":") for x in place.split(","))],
                     os.environ.get("PLACE_HEAD") or "last", ctx, mem, load_mode="none", **(P.PLANNER_V4 if mode == "v4" else {}))
elif mode == "v4":
    # 2026-11-01 (the scheduler review's items 3, 4 and 6; serve_defaults.sh planner_mode, serve_plan_20261101.py): a person at
    # a terminal. Planned on each node's memory now (a board that does not answer: the newest snapshot's figure; MEMFILE, if
    # given, instead), on a 512 MiB grid, priced at PLAN_DEPTH (default 0: v4d), the last served plan kept while it
    # fits and is within 2 %.
    import serve_plan_20261101 as SP
    if not os.environ.get("MEMFILE"):
        mem, src = SP.live_memory(nodes, json.load(open(SP.newest_snapshot())))
        print("MEMSRC=%s" % shlex.quote(", ".join("%s %d MiB (%s)" % (n, mem[n]["mem_available_mib"], s) for n, s in src.items())))
    mem = SP.on_grid(mem)
    depth = SP.plan_depth(ctx)
    r, why = SP.choose(key, nodes, ctx, mem, depth, SP.recall(key, ctx, nodes), host=os.environ.get("SERVE_HOST") or None,
                       **P.PLANNER_V4)
    pend = os.path.join(SP.STATE_DIR, "serve_plan.pending.json")
    json.dump(SP.record(key, ctx, nodes, depth, r), open(pend, "w"))
else:
    # 2026-10-28: the planner's own choice uses its defaults, predict_pool.PLANNER_V3 (def1, def2); SERVE_PLANNER=shipped
    # plans as before. A fixed PLACE (every harness) is priced as before either way.
    v3 = {} if mode == "shipped" else P.PLANNER_V3
    r = P.plan_host_choice(key, nodes, ctx, mem, host=os.environ.get("SERVE_HOST") or None, load_mode="none", **v3)
print("WHY=%s" % shlex.quote(why)); print("PLAN_PENDING=%s" % shlex.quote(pend))
# load_mode "none" (2026-10-09): the server reads its weights into memory (-lm none, below), so the host is charged its
# token embedding as well -- 417 MiB on Qwen3-14B, the gap that let the full evaluation's p32host through this gate
# and into the Pi 4's OOM killer in all five rounds (testbed/benchmarks.md, "do all 5", section 4).
q, C, L, devs, mp = r["q"], r["C"], r["links"], r["devices"], r["model"]
assign, head_at = r["assign"], r["head_at"]
hn = assign[0][0] if head_at == 0 else assign[-1][0]
m = P.memory(assign, q, mem, devs, mp, head_at=head_at); dec = r["decode_ms"]
a = P.bench_args(assign, key, head_at=head_at)["args"]
print("CONSIDERED=%s" % shlex.quote("; ".join("%s: %s" % c for c in r["considered"])))
a = a.replace(" -rpc ", " --rpc ")
if " -ts " in a:
    head, ts = a.split(" -ts "); a = head + " -ts " + ts.replace("/", ",")
print("HOST=%s" % assign[0][0]); print("WORKERS=%s" % shlex.quote(" ".join(n for n, _ in assign[1:])))
print("ARGS=%s" % shlex.quote(a))
print("PLAN=%s" % shlex.quote(" + ".join("%s %d" % (n, k) for n, k in assign) + ", head on " + hn
                              + ("; predicted decode %.1f ms/token (%.2f tok/s)" % (dec, 1000 / dec))
                              + (((" at a depth of %d tokens" % depth) if depth else "") + " (planner v4)" if mode == "v4" and not place else "")))
print("HEAD=%s" % hn)
# Each worker's model buffer as planned: its blocks (+ the head). 2026-10-21: llama.cpp at the pin loads no MTP (nextn) block
# onto a device, so the board holding the last blocks reported exactly that block less (Qwen3.8-27B: 227.9 MiB, on the Pi 4 and
# on the Jetson); the memory gate still charges it. 2026-11-02 (item 10): with --spec-type draft-mtp, in SERVE_EXTRA or a
# custom SERVE_DRAFT, it loads it there (predict_pool.planned_buffers).
mtp = "draft-mtp" in os.environ.get("SERVE_EXTRA", "") + " " + os.environ.get("SERVE_DRAFT", "")
# 2026-11-04 (the owner: "Turn MTP 1 on"; serve_defaults.sh mtp_mode): a pooled plan of a model MTP was measured with gets MTP 1,
# and so its MTP block too. WANT_MIB_NOMTP is the split without it, for when the Pi 5's TSO turns the default off below.
auto = os.environ.get("SERVE_MTP_MODE") == "on" and not mtp and len(assign) > 1 and int(q.get("nextn_layers") or 0) > 0
print("MTP_AUTO=%d" % auto)
if auto:
    nomtp = P.planned_buffers(q, assign, hn)
    print("declare -A WANT_MIB_NOMTP=(%s)" % " ".join("[%s]=%.1f" % (n, nomtp[n] / 1048576) for n, _ in assign[1:] if nomtp[n] is not None))
mtp = mtp or auto
want = P.planned_buffers(q, assign, hn, mtp=mtp)
print("declare -A WANT_MIB=(%s)" % " ".join("[%s]=%.1f" % (n, want[n] / 1048576) for n, _ in assign[1:] if want[n] is not None))
print("declare -A GATE=(%s)" % " ".join("[%s]=%d" % (n, v["need_mib"] + v["reserve_mib"]) for n, v in m.items()))
print("declare -A MARGIN=(%s)" % " ".join("[%s]=%d" % (n, v["margin_mib"]) for n, v in m.items()))
print("declare -A NEED=(%s)" % " ".join("[%s]=%d" % (n, v["need_mib"]) for n, v in m.items()))
PY
) || { echo "FATAL: the engine could not plan $MODEL at ctx $CTX"; exit 1; }   # (2026-10-11: eval "$(...)" hid a failed plan)
  eval "$PLANNED"
  # 2026-11-04 (the owner: "Turn MTP 1 on", after sp10e; serve_defaults.sh mtp_mode): the planner gave a pooled plan of a model
  # MTP was measured with, so it is served with MTP 1, as sp10e measured it. Unless the Pi 5 is in the plan with its hardware
  # TSO on: a default never takes the stall the guard below refuses; it serves without MTP instead.
  if [ "$MTP_AUTO" = 1 ]; then
    if [[ " $WORKERS " == *" pi5 "* ]] && ! on pi5 "/usr/sbin/ethtool -k eth0 2>/dev/null | grep -q '^tcp-segmentation-offload: off'"; then
      echo "mtp     off: the Pi 5 is in the plan with its hardware TSO on (testbed/results/pi5_tx_stall_20261103)"
      for k in "${!WANT_MIB[@]}"; do unset "WANT_MIB[$k]"; done
      for k in "${!WANT_MIB_NOMTP[@]}"; do WANT_MIB[$k]=${WANT_MIB_NOMTP[$k]}; done
    else
      EXTRA="$MTP_DEFAULT_ARGS${EXTRA:+ $EXTRA}"
      echo "mtp     MTP 1, the default for a pooled plan of $MODEL (the owner, 2026-11-04; SERVE_MTP=off serves without it)"
    fi
  fi
  # 2026-11-03 (item 10): MTP speculation with the Pi 5 in the plan hangs while the Pi 5's hardware TSO is on. Its network
  # driver (macb) holds a reply, and TCP will not resend it (TCPSpuriousRtxHostQueues; testbed/results/pi5_tx_stall_20261103):
  # three times out of three with the MTP block on the Pi 5, twice in sp10d with the Pi 5 a middle worker. With its TSO off
  # (the owner, 22:00) the serve that always hung held three rounds out of three, the counter unmoved (pi5_tso_check_*).
  # So MTP is allowed when the Pi 5 reports TSO off (it comes back on at a reboot unless made persistent), refused otherwise;
  # SERVE_MTP_PI5=1 tries anyway.
  if [[ "$EXTRA $DRAFT_GIVEN" == *draft-mtp* ]] && [[ " $WORKERS " == *" pi5 "* ]] && [ "${SERVE_MTP_PI5:-0}" != 1 ]; then
    if on pi5 "/usr/sbin/ethtool -k eth0 2>/dev/null | grep -q '^tcp-segmentation-offload: off'"; then
      echo "mtp     the Pi 5's hardware TSO is off (pi5_tx_stall_20261103's fix): MTP with the Pi 5 allowed"
    else
      echo "FATAL: MTP with the Pi 5 in the plan ($PLAN) hangs while the Pi 5's hardware TSO is on: its network driver holds a"
      echo "       reply (testbed/results/pi5_tx_stall_20261103). Turn it off (ssh pi5 'sudo ethtool -K eth0 tso off'), plan"
      echo "       without the Pi 5, or SERVE_MTP_PI5=1 to try anyway."
      exit 2
    fi
  fi
  case "$(draft_mode "$DRAFT_GIVEN" "$MODEL" "$INTERACTIVE")" in     # serve_defaults.sh
    laptop) DRAFT=$(laptop_draft_args $(wc -w <<< "$WORKERS")); DRAFT_HOST_MIB=${DRAFT_HOST_MIB:-$LAPTOP_DRAFT_HOST_MIB}
            RPC_EXTRA=${RPC_EXTRA:-$LAPTOP_LAN:$LAPTOP_RPC_PORT}; LAPTOP_RPC=1 ;;
    custom) DRAFT=$DRAFT_GIVEN ;;
    incompatible) echo "FATAL: SERVE_DRAFT=default: the draft is measured only with $DRAFT_MODELS, not $MODEL"; exit 2 ;;
    *) DRAFT="" ;;
  esac
  if [[ " $HOST $WORKERS " == *" laptop "* ]] && { [ "$LAPTOP_RPC" = 1 ] || [ "$RPC_EXTRA" = "$LAPTOP_LAN:$LAPTOP_RPC_PORT" ]; }; then
    echo "draft   off: the laptop is in the plan, and its port $LAPTOP_RPC_PORT is the laptop worker's (the laptop draft was measured only beside the Pi 4 + Jetson plan)"
    DRAFT=""; DRAFT_HOST_MIB=""; RPC_EXTRA=""; LAPTOP_RPC=0
  fi
  DRAFT_HOST_MIB=${DRAFT_HOST_MIB:-0}
  if [ -n "$RPC_EXTRA" ]; then        # the draft's own RPC server: the main model stays on the planned workers
    local eps devs
    eps=$(printf '%s' "$ARGS" | sed -nE 's/.*--rpc ([^ ]+).*/\1/p')
    [ -n "$eps" ] || { echo "FATAL: SERVE_RPC_EXTRA needs a plan with workers"; exit 2; }
    # device names count across every --rpc server in order (ggml-rpc.cpp ggml_backend_rpc_add_server: "RPC" + a global
    # dev_id; "RPC0[host:port]" names a buffer type, and -dev refuses it): the planned workers are RPC0..RPC<n-1>, and the
    # extra server's device is RPC<n>, which the caller's -devd names
    devs=$(seq -s, -f 'RPC%g' 0 $(( $(printf '%s\n' ${eps//,/ } | wc -l) - 1 )))
    ARGS="${ARGS/--rpc $eps/--rpc $eps,$RPC_EXTRA} -dev '$devs'"
  fi
  if [ "$DRAFT_HOST_MIB" != 0 ]; then
    GATE[$HOST]=$(( ${GATE[$HOST]} + DRAFT_HOST_MIB )); NEED[$HOST]=$(( ${NEED[$HOST]} + DRAFT_HOST_MIB ))
    MARGIN[$HOST]=$(( ${MARGIN[$HOST]} - DRAFT_HOST_MIB ))
  fi
  local against=${MEMFILE:-testbed/dataset/memory_at_rest.json}
  [ -n "${MEMSRC:-}" ] && against="the boards' memory now, on a 512 MiB grid"
  echo "########## $(date -Is) serve $MODEL, ctx $CTX, clocks $CLOCKS, boards $NODES, planned against $against: $PLAN"
  [ -n "${MEMSRC:-}" ] && echo "memory  $MEMSRC"
  [ -n "$WHY" ] && echo "plan    $WHY"
  echo "hosts   $CONSIDERED"
  [ -n "$DRAFT" ] && echo "draft   $DRAFT (host charge +$DRAFT_HOST_MIB MiB${RPC_EXTRA:+, its RPC server $RPC_EXTRA})"
  if [ -n "$DRAFT" ]; then             # the draft model must be on the host, which loads it
    local md; md=$(sed -nE 's/.*-md ([^ ]+).*/\1/p' <<< "$DRAFT")
    [ -z "$md" ] || on $HOST "test -f $md" || { echo "FATAL: the draft $md is not on $HOST"; exit 2; }
  fi
  SWITCH=$(switch_mode "$SWITCH_GIVEN" "$INTERACTIVE")   # serve_defaults.sh: a person at a terminal -> on
  if [ "$SWITCH" = on ]; then   # every board the plan needs whose outlet is off, powered up at once (2026-11-01: every board)
    local up="" b
    for b in $SWITCH_BOARDS; do
      [ "$(switch_before "$HOST $WORKERS" "$b" "$(switch_outlet "$b")")" = power-up ] && up+=" $b"
    done
    if [ -n "$up" ]; then
      echo "switch  the plan needs${up} and the outlet is off: powering up first, all at once (about a minute; SERVE_SWITCH=off skips this)"
      python3 "$SWITCH_TOOL" on-node $up --log "$SWITCH_LOG" || { echo "FATAL: not every board came back:${up} (see $SWITCH_LOG)"; exit 1; }
    fi
    if [[ " $HOST $WORKERS " == *" jetson "* ]]; then   # its power guard, started if none runs (a reboot stops it); the gate needs it
      python3 "$SWITCH_TOOL" guard jetson --log "$SWITCH_LOG" > /dev/null || { echo "FATAL: the Jetson's power guard is not ok (see $SWITCH_LOG)"; exit 1; }
    fi
  fi
  echo "llama-server $ARGS $DRAFT${EXTRA:+ $EXTRA} -c $CTX -np 1 -t 4 -lv 4 -lm none --cache-ram $CACHE_RAM --host ${LAN[$HOST]} --port $HTTP   pin $PIN"
  local n b h g w a
  local -A REST=()
  for n in $HOST $WORKERS; do
    # stdout only (2026-10-05): the laptop's OpenSSH 10.2 prints a post-quantum key-exchange warning on stderr for the
    # Jetson's old sshd, which 2>&1 made read as "busy"; an unreachable board still fails by ssh's exit code
    b=$(on $n 'for p in llama-bench llama-cli llama-server ggml-rpc-server cc1plus cmake; do pgrep -x $p >/dev/null && printf "%s " $p; done; true') || b+=" (ssh exit $?)"
    [ -z "$b" ] || { echo "FATAL: $n is busy or unreachable: $b"; exit 1; }
    h=$(on $n "git -C $(ldir $n) rev-parse HEAD; git -C $(ldir $n) status --porcelain --untracked-files=no | wc -l" | tr '\n' ' ')
    [ "$h" = "$PIN 0 " ] || { echo "FATAL: $n tree is '$h', not the pin $PIN clean"; exit 2; }
    a=$(on $n "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo")
    echo "gate    $n: MemAvailable $a MiB, the plan needs ${GATE[$n]} MiB with its reserve (margin at rest ${MARGIN[$n]} MiB)"
    REST[$n]=$a
    [ "${a:-0}" -ge "${GATE[$n]}" ] || { echo "REFUSED: $n has $a MiB free"; exit 8; }
  done
  if [[ " $HOST $WORKERS " == *" jetson "* ]]; then     # host or worker
    g=$($SSH jetson 'm=$(cut -d" " -f1 /var/lib/nvpmodel/status); [ "$m" = pmode:0000 ] || { echo "GATE-GUARD pmode $m"; exit 6; }
      [ "$(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l)" -ge 1 ] || { echo "GATE-GUARD no guard process"; exit 6; }
      G=~/heteroserve/logs/power-guard.log; t=$(awk "/guard start/ {t=0} /TRIP/ {t++} END {print t+0}" $G); [ "$t" = 0 ] || { echo "GATE-GUARD tripped $t since start"; exit 6; }
      a=$(( $(date +%s) - $(stat -c %Y $G) )); [ $a -le 5 ] || { echo "GATE-GUARD log $a s stale"; exit 6; }
      [ "$(ip -4 -o addr show | grep -cF " 192.168.1.")" = 1 ] && ip -4 -o addr show eth0 | grep -qF " 192.168.1.114/" || { echo "GATE-NET not eth0-only on 192.168.1.x"; exit 7; }
      echo "GATE ok: $m, guard live and un-tripped, eth0 is the only interface on the router LAN"')   # stdout only, as the busy check
    echo "jetson  $g"; [[ "$g" == "GATE ok"* ]] || { echo "FATAL: Jetson gate failed"; exit 3; }
  fi
  printf 'STARTED=%q\nHOST=%q\nWORKERS=%q\nARGS=%q\nCTX=%q\nLOG=%q\n' "$(date -Is)" "$HOST" "$WORKERS" "$ARGS" "$CTX" "$OUT" > "$STATE"
  if [[ " $HOST $WORKERS " == *" jetson "* ]] && [ "$CLOCKS" = locked ]; then   # run_pool.sh's lock, verbatim but the file name
    c=$($SSH jetson "c=/sys/devices/system/cpu/cpu0/cpufreq; g=/sys/devices/57000000.gpu/devfreq/57000000.gpu; if [ \$(cat \$c/scaling_min_freq) = \$(cat \$c/scaling_max_freq) ] && [ \$(cat \$g/min_freq) = \$(cat \$g/max_freq) ]; then echo ALREADY_LOCKED; else mkdir -p ~/heteroserve/run && sudo -n jetson_clocks --store ~/heteroserve/run/jetson_clocks_serve_$STAMP.stored && sudo -n jetson_clocks && echo LOCKED; fi")
    echo "jetson  clocks: $c"
    [ "$c" = LOCKED ] && printf 'LOCKED=1\nCLOCKSTAMP=%q\n' "$STAMP" >> "$STATE"
    [ "$c" = LOCKED ] || [ "$c" = ALREADY_LOCKED ] || { echo "FATAL: could not lock the Jetson's clocks"; stop; exit 1; }
  elif [[ " $HOST $WORKERS " == *" jetson "* ]]; then
    echo "jetson  clocks left as found (CLOCKS=$CLOCKS)"
  fi
  for n in $WORKERS; do
    on $n "mkdir -p ~/logs; setsid nohup $(ldir $n)/build/bin/ggml-rpc-server -H ${WIP[$n]} -p $(wport $n) ${WEXTRA[$n]} > ~/logs/rpc-server_serve_$STAMP.log 2>&1 < /dev/null &"
    for i in $(seq 1 30); do on $n "ss -ltn | grep -q ' ${WIP[$n]}:$(wport $n) '" && break; sleep 1; done
    on $n "ss -ltn | grep -q ' ${WIP[$n]}:$(wport $n) '" || { echo "FATAL: worker on $n never listened"; stop; exit 1; }
    echo "worker  $n listening on ${WIP[$n]}:$(wport $n) (pid $(wpid $n))"
  done
  # No "cd ~ &&" in front: "a && b &" backgrounds a subshell that keeps the ssh channel open until the server exits
  # (2026-09-28: the first start hung here, with the server up). Every path in the command is absolute.
  # -lv 4: at the pin the library's INFO lines (each device's "model buffer size", the KV and compute buffers) count as
  # TRACE (common_log_get_verbosity), above llama-server's default threshold of 3, so without it the log has none of them
  # and the split check below cannot run (2026-10-08 00:52: every serve log since 09-30 lacks them).
  # -lm none: the weights in the server's own memory, not memory-mapped. Measured 2026-10-08 01:00, Qwen3-14B with the Pi 4
  # as host (27 blocks + head, 6,089 MiB mapped) at -c 4096: the page cache did not keep the weights resident beside the
  # server's 0.7 GB (KV, compute), and every token re-read them from the SD card -- 19-31 MB/s at 65-75 % iowait, 4 tokens in
  # 5.5 min. The engine charges these weights as memory the board must hold, so resident is what it planned; a host that
  # repacks (the Pi 5) copies them out of the mapping anyway. mlock is not an option (ulimit -l is 8 MiB on the Pis).
  if [ "$LAPTOP_RPC" = 1 ]; then      # the draft's board, this laptop (serve_defaults.sh): started here, stopped by stop
    ip -4 -o addr show | grep -qF " $LAPTOP_LAN/" || { echo "FATAL: this laptop has no $LAPTOP_LAN, the draft's RPC address"; stop; exit 1; }
    if [ -z "$(laptop_rpc_pid)" ]; then
      ss -ltn | grep -q " $LAPTOP_LAN:$LAPTOP_RPC_PORT " && { echo "FATAL: $LAPTOP_LAN:$LAPTOP_RPC_PORT is taken by another process"; stop; exit 1; }
      setsid nohup "$LAPTOP_RPC_BIN" -H $LAPTOP_LAN -p $LAPTOP_RPC_PORT -t 4 > "${OUT%.log}_laptop_rpc.log" 2>&1 < /dev/null &
      for i in $(seq 1 30); do ss -ltn | grep -q " $LAPTOP_LAN:$LAPTOP_RPC_PORT " && break; sleep 1; done
      [ -n "$(laptop_rpc_pid)" ] || { echo "FATAL: the laptop's draft RPC server did not start"; stop; exit 1; }
      printf 'LAPTOP_RPC_STARTED=1\n' >> "$STATE"
    fi
    echo "laptop  draft RPC server on $LAPTOP_LAN:$LAPTOP_RPC_PORT (pid $(laptop_rpc_pid))"
  fi
  on $HOST "mkdir -p ~/logs; ${DRAFT:+LLAMA_TRACE=1 }setsid nohup $(ldir $HOST)/build/bin/llama-server $ARGS $DRAFT${EXTRA:+ $EXTRA} -c $CTX -np 1 -t 4 -lv 4 -lm none --cache-ram $CACHE_RAM --host ${LAN[$HOST]} --port $HTTP > ~/logs/llama-server_serve_$STAMP.log 2>&1 < /dev/null &"
  echo "server  starting on $HOST (pid $(spid $HOST)); loading streams the model from the SD card to the workers"
  for i in $(seq 1 180); do   # up to 15 min
    r=$(curl -s -m 5 "http://${LAN[$HOST]}:$HTTP/health"); [[ "$r" == *'"ok"'* ]] && break
    [ -z "$(spid $HOST)" ] && { echo "FATAL: llama-server exited while loading:"; on $HOST "tail -25 ~/logs/llama-server_serve_$STAMP.log"; stop; exit 1; }
    sleep 5
  done
  [[ "$r" == *'"ok"'* ]] || { echo "FATAL: not healthy after 15 min: $r"; stop; exit 1; }
  echo "READY   $(date -Is): http://${LAN[$HOST]}:$HTTP/ (web chat) -- health $r"
  on $HOST "grep -a -E 'model buffer size|KV self size|llama_kv_cache.*size|compute buffer size|flash_attn|n_ctx ' ~/logs/llama-server_serve_$STAMP.log | head -20"
  bufs=$(on $HOST "grep -a 'model buffer size' ~/logs/llama-server_serve_$STAMP.log")
  local bad=0 got
  for n in $WORKERS; do      # the layout llama.cpp loaded, against the plan (head on $HEAD)
    got=$(printf '%s\n' "$bufs" | grep -F "[${WIP[$n]}:$(wport $n)]" | sed -E 's/.*= +([0-9.]+) MiB.*/\1/' | head -1)
    echo "split   $n: llama.cpp ${got:-?} MiB, planned ${WANT_MIB[$n]:-?} MiB (head on $HEAD)"
    [ -n "$got" ] && [ -n "${WANT_MIB[$n]:-}" ] && awk -v g="$got" -v w="${WANT_MIB[$n]}" 'BEGIN {exit !(g - w < 1 && w - g < 1)}' || bad=1
  done
  [ "$bad" = 0 ] || { echo "FATAL: the loaded split is not the plan; stopping"; stop; exit 1; }
  if [ -n "$PLAN_PENDING" ] && [ -f "$PLAN_PENDING" ]; then   # planner v4: the plan served, kept for the next start
    python3 -c 'import json, sys; sys.path.insert(0, "testbed/scripts"); import serve_plan_20261101 as SP; SP.remember(json.load(open(sys.argv[1])))' "$PLAN_PENDING" \
      && rm -f "$PLAN_PENDING" && echo "plan    remembered for the next start (serve_plan_20261101.py)"
  fi
  for n in $HOST $WORKERS; do   # what the serve holds, read once at READY, before any request (2026-10-09: the full
    # evaluation logged buffers but no memory, so the 14B's footprint had to be rebuilt from its buffer lines)
    a=$(on $n "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo")
    echo "mem     $n: MemAvailable $a MiB at READY, ${REST[$n]} at the gate: the serve holds $(( ${REST[$n]} - ${a:-0} )) MiB, charged ${NEED[$n]} MiB"
  done
  if [ "$SWITCH" = on ]; then   # served without them: every idle board shut down and its outlet cut, at once (serve_defaults.sh)
    local so busy down="" b
    for b in $SWITCH_BOARDS; do
      so=$(switch_outlet "$b"); busy=""
      if [ "$so" = on ] && [[ " $HOST $WORKERS " != *" $b "* ]]; then
        busy=$(on $b 'for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null && printf "%s " $p; done; true' 2>/dev/null)
      fi
      case "$(switch_after "$HOST $WORKERS" "$b" "$so" "$busy")" in
        switch-off) down+=" $b" ;;
        leave-on)   echo "switch  the $b runs $busy: left on" ;;
      esac
    done
    if [ -n "$down" ]; then
      echo "switch  the plan leaves${down} out: shutting down and cutting the outlets, all at once (SERVE_SWITCH=off skips this)"
      python3 "$SWITCH_TOOL" off-node $down --log "$SWITCH_LOG" \
        || echo "!!!!! switch: not every board was switched off:${down} (the model is served; see $SWITCH_LOG)"
    fi
  fi
}

case "${1:-}" in
  start) start ;; stop) stop ;; status) status ;;
  *) echo "usage: $0 start|status|stop"; exit 2 ;;
esac
