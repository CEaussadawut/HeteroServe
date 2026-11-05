# Experiment log — template

Copy one block per experiment into your lab notebook. The point is that six
months later you can tell why two runs disagree.

```
## EXP-<nnn>  <one-line question this run answers>

Date (UTC)      :
Operator        :
Node(s)         : jetson-nano-01 [+ rpi5-01, rpi4-01, laptop-i5-8350u]

### Conditions
llama.cpp ref   :            (build_info.json -> llama_cpp_ref)
Backend         : cuda-sm53 | cpu-neon
Model           :            (+ sha256 prefix)
Power mode      : 0 (MAXN/10W) | 1 (5W)
Clocks locked   : yes/no
Fan             : pwm 255 | none
Link            : wired GbE | Wi-Fi 2.4 | Wi-Fi 5
Room temp       :        C
Idle floor      :        mW     (from the sweep's baseline)

### Command
    <exact command line>

### Artefacts
Power CSV          : results/..._power.csv
Joined CSV         : results/..._bench_energy.csv
Per-run bench CSVs : results/..._NNN.csv   (one per run; NNN is the run index)
Run log            : results/..._run.log

### Result
The prefill and decode numbers for one placement point come from TWO rows of the
same sweep -- two invocations, two power windows, interleaved so they sit at
nearly the same board temperature. Record both rows' run_index.

tokens/s          : prefill ......   decode ......
J/token (prefill) : total ......     active ......      run_index ...
J/token (decode)  : total ......     active ......      run_index ...
Wh per Mtok       : prefill ......   decode ......
J/request         :                  (combined case only)
Peak temp         :        C   (rise ...... C)
Swap peak         :        MB

### Observations


### Anomalies / things that invalidate this run


### What this changes about the next experiment

```

## Numbering

`EXP-001` upward, never reused. If you re-run an experiment after fixing a
condition, it gets a new number and a line in the old one saying it was
superseded. Deleting a bad run is how you end up unable to explain a figure.

## What counts as an anomaly worth recording

- `temp_rise_c` above ~8 °C in any window
- any `status=failed` row
- any `per_token_status` other than `ok` or `refused_combined_request` — in
  particular `refused_multi_phase`, which means a power window held two
  `llama-bench` tests and its per-token columns are blank by design. The shipped
  matrix cannot produce that; it means a case was edited to mix phases, or the
  installed ref emitted an unexpected extra row
- a placement or quantisation point whose prefill half completed but whose decode
  half did not, or the reverse — the pair is the result, not either half
- `swap_peak_mb` > 0 on a case that was supposed to fit
- repetitions disagreeing by more than a few percent
- a link that did not negotiate 1 Gbit
- anything in `kernel_warnings.txt` from `90_collect_results.sh`
