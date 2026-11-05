#!/bin/bash
# Runs ON the Jetson. Stops between steps if the guard tripped or the power mode left MAXN.
T=$HOME/heteroserve/tools/jetson_bench_tinyllama.sh; R=$HOME/heteroserve/results; L=$R/quant_chain.log
t0=$(grep -c TRIP $HOME/heteroserve/logs/power-guard.log 2>/dev/null; true)
ok() { local t; t=$(grep -c TRIP $HOME/heteroserve/logs/power-guard.log 2>/dev/null; true)
       [ "$t" = "$t0" ] && [ "$(cut -d' ' -f1 /var/lib/nvpmodel/status)" = pmode:0000 ]; }
step() { local tag=$1; shift; ok || { echo "STOPPED before $tag: guard trip or pmode change" >> $L; return 1; }
         env "$@" TAG=$tag bash $T; echo "$tag rc=$? $(date -Is)" >> $L; }
rm -f $R/quant_chain.done; : > $L
step tinyllama-q4km-cold MODEL=$HOME/models/tinyllama-1.1b-q4km.gguf CASES=cpu &&
step tinyllama-q8_0      MODEL=$HOME/models/tinyllama-1.1b-q8_0.gguf "CASES=vulkan cpu" &&
step tinyllama-q4_0      MODEL=$HOME/models/tinyllama-1.1b-q4_0.gguf "CASES=vulkan cpu"
echo DONE > $R/quant_chain.done
