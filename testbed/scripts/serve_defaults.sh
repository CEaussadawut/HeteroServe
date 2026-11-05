# The serving defaults of serve_pool.sh (2026-10-10). Sourced by it; defines settings and functions, starts nothing.
#
# The user's decision of 2026-10-10 ("Laptop draft"): serving uses speculative decoding by default, with the Qwen3-0.6B
# draft on this laptop. Scheduler upgrade 1 measured it at 1.400x the draft-free decode and 22 % less energy per token at
# the boards' outlets (Qwen3-14B on Pi 4 27 + Jetson 13, head on the host, ctx 1,024; testbed/benchmarks.md, the three
# "2026-10-10 - scheduler upgrade 1" sections). The laptop then joins the pool as the draft's board.
#
# Who gets it:
# - A PERSON serving from a terminal: the draft, for the targets it was measured with (DRAFT_MODELS).
# - A HARNESS keeps serving exactly what its registration names, so every registered campaign keeps its meaning. A harness
#   is any caller whose stdout is not a terminal: full_eval.py, spec_serve_20261010.py and the like.
#
# The SERVE_DRAFT settings:
#     unset            the default above (a terminal and a measured target), else no draft
#     default          the laptop draft, from anywhere (a harness that wants it says so)
#     none, or ""      no draft
#     <arguments>      those llama-server arguments, as before, with DRAFT_HOST_MIB and SERVE_RPC_EXTRA
#
# serve_pool.sh starts and stops the laptop's ggml-rpc-server itself when it serves the laptop draft. It reuses one already
# listening on the same address and port, found by its exact argv, and then leaves it running at stop.

DRAFT_MODELS="qwen3_14b"                                   # the targets measured with this draft (Qwen3's tokenizer)
DRAFT_MODEL_PATH='~/models/qwen3-0.6b-q8_0.gguf'           # on the host, expanded there (testbed/models.md: sha256 9465e63a...)
LAPTOP_LAN=192.168.1.108                                   # the laptop on the router LAN, where the boards reach it
LAPTOP_RPC_PORT=50053                                      # never 50052, the boards' RPC port
LAPTOP_RPC_BIN=/opt/heteroserve/llama.cpp/build/bin/ggml-rpc-server   # the pinned build, b23701f7
LAPTOP_DRAFT_HOST_MIB=160                                  # what the laptop draft adds on the host: its CPU-side tensors

draft_mode() {   # $1 = SERVE_DRAFT as given ("__unset__" when unset), $2 = the model key, $3 = 1 when stdout is a terminal
  local ok=0
  [[ " $DRAFT_MODELS " == *" $2 "* ]] && ok=1
  case "$1" in
    __unset__) if [ "$3" = 1 ] && [ $ok = 1 ]; then echo laptop; else echo none; fi ;;
    default)   if [ $ok = 1 ]; then echo laptop; else echo incompatible; fi ;;
    none|"")   echo none ;;
    *)         echo custom ;;
  esac
}

laptop_draft_args() {   # $1 = the planned workers' count: the laptop's server comes after them, so its device is RPC<n>
  echo "-md $DRAFT_MODEL_PATH --spec-type draft-simple --spec-draft-n-max 1 --spec-draft-n-min 0 --spec-draft-p-min 0 -devd RPC$1 -ngld 99"
}

laptop_rpc_pid() {   # the laptop's draft ggml-rpc-server, found by its exact argv; empty when none
  local p
  for p in $(pgrep -x ggml-rpc-server); do
    tr '\0' ' ' 2>/dev/null < /proc/$p/cmdline | grep -qF -- "-H $LAPTOP_LAN -p $LAPTOP_RPC_PORT " && echo $p
  done
}

# The user's decision of 2026-11-04 ("Turn MTP 1 on", once the Pi 5 no longer stalled): a PERSON serving a POOLED plan from a
# terminal gets speculation with the model's own multi-token-prediction head, MTP 1, for the models it was measured with
# (MTP_MODELS). sp10e: MTP 1, the setting the checking-cost model picks, was within 2 % of the best on both of the 9B's
# three-node plans and both prompts, gaining 6-14 % on free text and 16-29 % on copying; sp10c: 12-43 % on the 35B's pooled
# plans. Not on a one-board plan: on the laptop alone the 9B lost on free text (0.86-0.99x). Qwen3.8-27B has the head but was
# never measured with it. A HARNESS keeps what its registration names. serve_pool.sh decides "pooled" once it has the plan,
# and serves without MTP when the Pi 5 is in the plan with its hardware TSO on (testbed/results/pi5_tx_stall_20261103).
# The SERVE_MTP settings:
#     unset            the default above (a terminal: on; a harness: off)
#     on               MTP 1 on a pooled plan of a measured model, from anywhere
#     off              never by default
# Speculation asked for another way (any SERVE_DRAFT given, "none" included, or --spec-type in SERVE_EXTRA) is served as given.
MTP_MODELS="qwen35_9b qwen36moe_q6k"                       # the models MTP was measured with (sp10, sp10c, sp10e)
MTP_DEFAULT_ARGS="--spec-type draft-mtp --spec-draft-n-max 1"   # sp10e's mtp1 arm

mtp_mode() {   # $1 = SERVE_MTP as given ("__unset__" when unset), $2 = the model key, $3 = 1 when stdout is a terminal,
               # $4 = SERVE_DRAFT as given ("__unset__" when unset), $5 = SERVE_EXTRA
  [[ " $MTP_MODELS " == *" $2 "* ]] || { echo off; return; }
  { [ "$4" = __unset__ ] && [[ "$5" != *--spec-type* ]]; } || { echo off; return; }
  case "$1" in
    __unset__) if [ "$3" = 1 ]; then echo on; else echo off; fi ;;
    on)        echo on ;;
    *)         echo off ;;
  esac
}

# The user's decision of 2026-10-25 ("Interactive default", after so4 and so5): when a PERSON serves from a terminal, the
# Pi 4 -- the one board the strip may switch (strip_switch_20261024.py; the laptop's outlet is do_not_switch_off) -- is
# shut down and its outlet cut once the served plan leaves it out and it runs nothing, and powered up first when a plan
# needs it while its outlet is off. so4 measured the saving: 3.41 J per token (5.7 %), the time per token unchanged; so5 ran
# it live (cut 38 s after the poweroff, back in 59 s). A HARNESS never switches anything: every registered study assumed all
# four boards on, and a chain's memory snapshot reads every board. The SERVE_SWITCH settings:
#     unset            the default above (a terminal: on; a harness: off)
#     on               switch, from anywhere (a caller that wants it says so)
#     off              never switch
# 2026-11-01 (the scheduler review's item 7; the user: "do item 1 to 4 then 6-7"): every board, not just the Pi 4 -- on 10 of
# the evaluation's 11 picks the Pi 5 and the Jetson sit idle too, about 10-13 % of the energy per token. Several boards are
# powered up, or shut down and cut, at once (strip_switch_20261101.py, which halts the Jetson rather than powering it off, and
# starts its power guard after it boots); the decisions below are per board.
SWITCH_BOARDS="pi4 pi5 jetson"
SWITCH_TOOL=testbed/scripts/strip_switch_20261101.py
SWITCH_LOG=testbed/results/serve_switch.jsonl             # the tool's own lines, one per board and action

# The boards the engine may plan on when NODES is unset (2026-11-01, the scheduler review's item 1): a PERSON at a terminal
# gets all four nodes, the laptop included -- a node since 2026-10-11, and the default "pi5 pi4 jetson" was written before
# that (the default model was predicted 2.85x slower without it: 284.0 against 99.8 ms a token). A HARNESS that does not
# name NODES keeps that old default, so every registered campaign keeps its meaning.
default_nodes() {   # $1 = 1 when stdout is a terminal -> the boards, space-separated
  if [ "$1" = 1 ]; then echo "laptop pi5 pi4 jetson"; else echo "pi5 pi4 jetson"; fi
}

# The planner (2026-11-01, the scheduler review's items 3, 4 and 6): a PERSON at a terminal gets predict_pool.PLANNER_V4 --
# a hybrid's memory as llama.cpp allocates it, the cost of context depth only at a PLAN_DEPTH asked for (v4d), planned on
# the boards' memory now on a 512 MiB grid, and the plan served last kept while it fits and is within 2 %
# (serve_plan_20261101.py). A HARNESS keeps PLANNER_V3, the default since 2026-10-28. The SERVE_PLANNER settings:
#     unset            the default above (a terminal: v4; a harness: v3)
#     v4, v3           that planner, from anywhere
#     shipped          the planner before 2026-10-28 (as before)
planner_mode() {   # $1 = SERVE_PLANNER as given ("__unset__" when unset), $2 = 1 when stdout is a terminal
  case "$1" in
    __unset__)       if [ "$2" = 1 ]; then echo v4; else echo v3; fi ;;
    v4|v3|shipped)   echo "$1" ;;
    *)               echo invalid ;;
  esac
}

switch_mode() {   # $1 = SERVE_SWITCH as given ("__unset__" when unset), $2 = 1 when stdout is a terminal -> on | off
  case "$1" in
    __unset__) if [ "$2" = 1 ]; then echo on; else echo off; fi ;;
    on)        echo on ;;
    *)         echo off ;;
  esac
}

switch_before() {   # $1 = the plan's boards, $2 = the board, $3 = its outlet (on | off | ?) -> power-up, or nothing
  if [[ " $1 " == *" $2 "* ]] && [ "$3" = off ]; then echo power-up; fi
}

switch_after() {   # $1 = the served boards, $2 = the board, $3 = its outlet (on | off | ?), $4 = the llama.cpp it runs ("" when none)
  if [[ " $1 " == *" $2 "* ]] || [ "$3" != on ]; then return 0; fi
  if [ -n "$4" ]; then echo leave-on; else echo switch-off; fi
}

switch_outlet() {   # $1 = the board -> its outlet as the strip reads it: on | off | ? (the tool's status)
  python3 "$SWITCH_TOOL" status "$1" 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.loads(sys.stdin.read().strip().splitlines()[-1])
    print({True: "on", False: "off"}.get(d.get("output"), "?") if d.get("board") == sys.argv[1] else "?")
except Exception:
    print("?")' "$1"
}
