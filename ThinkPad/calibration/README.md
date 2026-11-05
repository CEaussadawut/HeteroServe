# Channel calibrations

One file per DAQ channel: `ch1.json` … `ch4.json`. These are **data**, not configuration —
they are produced by measurement, they belong in version control alongside the results, and
every energy figure in the project depends on them.

Produce one with the guided script:

    scripts/05_calibrate.sh 2

or directly, from points you measured yourself:

    heteroctl calibrate --channel 2 --point 0.812:0.806 --point 1.204:1.197

Each `--point` is `reported_amps:reference_amps`, where the reference comes from `I = V/R`
with *both* V and R known by measurement, not by assumption. See
`docs/03_energy_collector.md` for the procedure and the error budget.

A channel with no file here still records, but every row derived from it is flagged
`uncalibrated-channel` and cannot carry the ±5 % accuracy the success criteria claim.

Only `ch1.json` … `ch9.json` are loaded, and only if `calibrated_at` is filled in. Both
rules exist because this directory once shipped a `ch2.example.json` full of invented
constants: `load_all` globbed `ch*.json`, matched it, silently rescaled every Pi 4 current
sample by a made-up gain, and — because the channel then counted as calibrated — stopped
flagging the very rows that were wrong. Anything that is not a real measurement must be
named so the glob cannot reach it, e.g. `ch2.json.example`.

Re-calibrate after changing a shunt, a sensor, a supply, or the wiring — and note the date
in the file's `calibrated_at` field, which the tool fills in.
