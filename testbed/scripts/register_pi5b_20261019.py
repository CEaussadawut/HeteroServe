#!/usr/bin/env python3
"""Register, before any row is taken, what the replacement Pi 5's characterisation must show for the old board's results to
stand for it, and how its rows join the dataset (2026-10-19).

    python3 testbed/scripts/register_pi5b_20261019.py

Writes, read-only:
  testbed/dataset/pi5b_predictions_20261019.json            the comparison (RB-PP, RB-TG, RB-MEM, RB-E) and the merge rule
  testbed/dataset/energy_predictions_20261019_pi5b_<q>.json  one per energy run, in score_energy_phase_run.py's format
Why: the Pi 5 came back on a replacement board (serial b8ed9049ada12a03, eth0 2c:cf:67:b2:af:52) running the old SD card;
every Pi 5 row in characterisation.csv (34) and its one energy run are the old board's (serial b266e1877252f91e). The user
(2026-10-19): "Replacement board" -- re-characterise it and compare before merging any data. testbed/scripts/
pi5b_char_20261019.sh takes the rows; score_pi5b_20261019.py scores them against this file. Refuses to overwrite.
"""
import csv, datetime, hashlib, json, os, stat, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "pi5b_predictions_20261019.json")
OLD_ENERGY = "testbed/results/energy_phase_pi5_20261005_145817"
SHA = {"q4km": ("tinyllama-1.1b-q4km.gguf", "9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0"),
       "q4_0": ("tinyllama-1.1b-q4_0.gguf", "da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556"),
       "q8_0": ("tinyllama-1.1b-q8_0.gguf", "a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59"),
       "q3km": ("tinyllama-1.1b-q3km.gguf", "ec461c4d2b60896152bca3978aa49cd70edad7298715e4222e12af1eefea2125"),
       "q6k": ("tinyllama-1.1b-q6_k.gguf", "9fb2be7c8bbc0ca358ae88cef6d9e6844b81929eaffa5016a543e6f56b22cb6d")}
# reps sized to ~120 s of timed repetitions at the old board's host rows (Q6_K: no old row; estimated), as on 2026-10-05
REPS = {"q4km": (20, 17, 300), "q4_0": (32, 18, 120), "q8_0": (15, 10, 120), "q3km": (10, 21, 120), "q6k": (13, 12, 120)}
CODE = ["testbed/scripts/pi5b_char_20261019.sh", "testbed/scripts/score_pi5b_20261019.py",
        "testbed/scripts/register_pi5b_20261019.py", "testbed/scripts/run_quant_campaign.sh",
        "testbed/scripts/bench_worker_role.sh", "testbed/scripts/build_dataset.py", "testbed/scripts/memory_snapshot.py",
        "testbed/scripts/energy_phase_run.py", "testbed/scripts/score_energy_phase_run.py", "testbed/scripts/shelly_strip.py"]
DATA = ["testbed/dataset/characterisation.csv", "testbed/dataset/memory_at_rest.json", "testbed/energy/strip.json",
        OLD_ENERGY + "/score.json", "testbed/results/pi5_final_collect_20261005/system/pi5_system_snapshot.txt",
        "testbed/results/pi5b_models_sha_20261019_234724.log"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_ro(path, obj):
    if os.path.exists(path):
        sys.exit("refusing to overwrite %s" % os.path.relpath(path, ROOT))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False); f.write("\n")
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print("wrote", os.path.relpath(path, ROOT), sha256(path))


def band(x, pct):
    return [round(x * (1 - pct / 100), 4), round(x * (1 + pct / 100), 4)]


def main():
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    old_rows = {}
    for r in csv.DictReader(open(os.path.join(DS, "characterisation.csv"))):
        if r["node_id"] == "pi5":
            old_rows["%s|%s|%s" % (r["device"], r["model"], r["test"])] = {
                "phase": r["phase"], "quant": r["quant"], "mean_ts": float(r["mean_ts"]), "rep_range_pct": float(r["rep_range_pct"]),
                "start_temp_c": r["start_temp_c"], "flags": r["flags"], "date": r["date"], "source": r["source"]}
    assert len(old_rows) == 34, len(old_rows)
    host = {}
    for k, v in old_rows.items():
        dev, model, test = k.split("|")
        if dev == "cpu" and test in ("pp512", "tg128"):
            host[(model, test)] = v["mean_ts"]
    mem = json.load(open(os.path.join(DS, "memory_at_rest.json")))["pi5"]
    es = json.load(open(os.path.join(ROOT, OLD_ENERGY, "score.json")))
    old_energy = {"run": OLD_ENERGY, "protocol": "--pp 512:20 --tg 128:17 --cycles 3 --idle-before 300 --idle-after 300 "
                                                 "--settle 30 --gap 10, outlet 1 (2026-10-05)",
                  "idle_before_w": (es.get("idle_before") or {}).get("mean_w"),
                  "phases": {ph: {f: es["phases"][ph][f]["mean"] for f in ("tps", "mean_w", "j_per_token_net_trap",
                                                                            "j_per_token_net_counter")} for ph in ("pp", "tg")}}

    energy_runs = {}
    for q, (f, h) in SHA.items():
        rp, rt, ia = REPS[q]
        reg_path = os.path.join(DS, "energy_predictions_20261019_pi5b_%s.json" % q)
        if q == "q4km":
            ph = old_energy["phases"]
            pred = {ph_: {"tps": band(ph[ph_]["tps"], 5), "mean_w": band(ph[ph_]["mean_w"], 10),
                          "j_per_token_net": {"trap": band(ph[ph_]["j_per_token_net_trap"], 10),
                                              "counter": band(ph[ph_]["j_per_token_net_counter"], 10)}} for ph_ in ("pp", "tg")}
            basis = ("the old board's run %s, same protocol: tokens/s +/- 5 %%, mean power and net J/token +/- 10 %% (RB-E in "
                     "pi5b_predictions_20261019.json is the comparison itself)" % OLD_ENERGY)
        else:
            pred = {}
            for ph_, test, wb in (("pp", "pp512", [7.5, 11.5]), ("tg", "tg128", [6.5, 10.0])):
                t = host.get((f, test))
                tb = band(t, 5) if t else ([35.0, 90.0] if ph_ == "pp" else [9.0, 18.0])
                tp = t if t else sum(tb) / 2
                trap = [round((wb[0] - 2.31) / tp, 4), round((wb[1] - 2.31) / tp, 4)]
                pred[ph_] = {"tps": tb, "mean_w": wb,
                             "j_per_token_net": {"trap": trap, "counter": [round(trap[0] * 1.03, 4), round(trap[1] * 1.08, 4)]}}
            basis = ("tokens/s: the old board's host row at this quant +/- 5 % (Q6_K: no old row, a wide guess); power: a guess, "
                     "wider than the old Q4_K_M run (9.46 / 8.26 W); net J/token trap = (power band - the old idle 2.31 W) / "
                     "the t/s point, counter = trap x [1.03, 1.08]. These are the energy scorer's sanity bands, not a test of "
                     "the board: no old energy run exists at this quant")
        pred.update({"spread_pct_max": 5.0, "gap_pct": [-7.0, -2.0], "throttled": "0x0 throughout"})
        cmd = ("python3 testbed/scripts/energy_phase_run.py --node pi5 --outlet 2 --outlet-name pi5 --model '~/models/%s' "
               "--sha256 %s --pp 512:%d --tg 128:%d --cycles 3 --threads 4 --idle-before 300 --idle-after %d --settle 30 --gap 10"
               % (f, h, rp, rt, ia))
        ereg = {"registered": now,
                "campaign": "2026-10-19: the replacement Pi 5's energy per token, TinyLlama-1.1B %s, prefill-only and decode-only "
                            "at the wall (one of five runs registered together with pi5b_predictions_20261019.json)" % q,
                "model": {"file_on_board": "~/models/" + f, "sha256": h,
                          "pin": "testbed/models.md (TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF @ 52e7645b)",
                          "verified": "re-hashed on the Pi 5 by the runner's preflight; it refuses on a mismatch"},
                "build": "llama.cpp b23701f7 (clean tree, checked by run_quant_campaign.sh earlier in the session), "
                         "~/llama.cpp/build/bin/llama-bench on the Pi 5",
                "command": cmd,
                "plan": {"order": "settle 30 s; idle 300 s; pp1, tg1, pp2, tg2, pp3, tg3 (10 s gaps); idle %d s" % ia,
                         "strip": "outlet 2 = pi5 (load-tested and renamed 2026-10-19)"},
                "definitions": "as in energy_predictions_20261005b.json",
                "predictions": pred, "basis": basis,
                "scoring": "testbed/scripts/score_energy_phase_run.py; PASS / FAIL / N/A, reported as found",
                "inputs_sha256": {p: sha256(os.path.join(ROOT, p)) for p in
                                  ("testbed/scripts/energy_phase_run.py", "testbed/scripts/score_energy_phase_run.py",
                                   "testbed/scripts/shelly_strip.py", "testbed/energy/strip.json")}}
        write_ro(reg_path, ereg)
        energy_runs[q] = {"file": f, "sha256": h, "reps_pp": rp, "reps_tg": rt, "idle_after_s": ia, "command": cmd,
                          "registration": os.path.relpath(reg_path, ROOT), "registration_sha256": sha256(reg_path)}

    reg = {
        "registered_at": now, "script": "testbed/scripts/register_pi5b_20261019.py",
        "question": "the Pi 5 came back on a replacement board running the old SD card: do the old board's measured rows describe "
                    "it, so that every earlier Pi 5 result stands for the Pi 5, or only for the old unit?",
        "user_decision": "2026-10-19 ~23:35, \"Replacement board (Recommended)\": re-characterise it and compare with the old "
                         "Pi 5's rows before merging any data",
        "old_board": {"serial": "b266e1877252f91e", "revision": "d04171 (Rev 1.1, 8 GB)", "eth0_mac": "88:a2:9e:f0:e7:8a",
                      "bootloader_firmware": "2026/01/21 13:38:47, version ab8a9dde (pi5_final_collect_20261005 snapshot)",
                      "eeprom_config": "BOOT_UART=1 BOOT_ORDER=0xf461 NET_INSTALL_AT_POWER_ON=1",
                      "measured": "2026-09-19 .. 2026-10-05; returned 2026-10-05"},
        "new_board": {"serial": "b8ed9049ada12a03", "revision": "d04171 (Rev 1.1, 8 GB)", "eth0_mac": "2c:cf:67:b2:af:52",
                      "bootloader_firmware": "2024/09/23 14:02:56, version 26826259 (rpi-eeprom-update: update available, "
                                             "2026-05-26 on the card; not applied)",
                      "eeprom_config": "BOOT_UART=1 POWER_OFF_ON_HALT=0 BOOT_ORDER=0xf461",
                      "cooling": "Active Cooler present and spinning (2,800-4,800 rpm in a 24 s 4-core load, 2026-10-19 23:45); "
                                 "idle 48 C, just under the fan's 50 C trip",
                      "same_sd_card": "hostname hs-pi5, the host keys known since 2026-09-19, kernel 6.18.50+rpt-rpi-2712, "
                                      "llama.cpp b23701f7 clean (build and build-nodotprod), governor ondemand, cmdline "
                                      "numa=fake=8 numa_policy=interleave, zram off, the models (sha256 log in inputs)",
                      "network": "eth0 192.168.1.122 (DHCP) + overlay 192.168.10.11; Wi-Fi off since 2026-10-19 23:3x"},
        "old_rows": old_rows,
        "old_memory": {"mem_total_mib": mem["mem_total_mib"], "mem_available_mib": mem["mem_available_mib"],
                       "source": "testbed/dataset/memory_at_rest.json (2026-09-27 ~19:05)"},
        "old_energy": old_energy,
        "plan": {
            "runner": "testbed/scripts/pi5b_char_20261019.sh under systemd-run + systemd-inhibit; the rest of the cluster idle",
            "steps": ["1 memory_snapshot.py --reads 10",
                      "2 run_quant_campaign.sh NODES=pi5, host rows pp512/tg128: TinyLlama Q4_K_M, Q8_0, Q4_0, Q3_K_M, Q6_K, Qwen3.5-9B",
                      "3 the same at PP=64 TG=16",
                      "4 bench_worker_role.sh NODES=pi5, worker rows pp512/tg128: TinyLlama Q4_K_M, Q8_0, Q4_0, Q3_K_M, Q6_K",
                      "5 the same plus Qwen3.5-9B at PP=64 TG=16",
                      "6 the worker prefill-size sweep: Q3_K_M at PP=1,4,16,256 TG=0 (as 2026-09-28 22:03)",
                      "7-11 energy_phase_run.py on outlet 2: Q4_K_M at the 2026-10-05 protocol, then Q4_0, Q8_0, Q3_K_M, Q6_K"],
            "protocol": "the old rows' own: -t 4 -r 3 -oe jsonl, the cold gate below 50 C (max 10 min), get_throttled and the "
                        "temperature before and after every run, every model re-hashed on the board, the tree at the pin and clean"},
        "hypotheses": {
            "RB-PP": {"tol_pct": 3.0, "claim": "every matched prefill row (host and worker, both protocols, the sweep) is within "
                                               "3 % of the old board's tokens/s: the same Cortex-A76 cores at 2.4 GHz, the same build"},
            "RB-TG": {"tol_pct": 3.0, "claim": "every matched decode row is within 3 % of the old board's: the same LPDDR4X "
                                               "configuration (the house drift is ~2 %)"},
            "RB-MEM": {"tol_pct": 1.5, "claim": "MemAvailable at rest (the lowest of 10 reads) within 1.5 % of the old board's "
                                                "7,619 MiB (the firmware's reservations differ: MemTotal reads 8,050 MiB "
                                                "against 8,062)"},
            "RB-E": {"tol_pct": 10.0, "claim": "TinyLlama Q4_K_M at the 2026-10-05 energy protocol: net J/token (trapezoid) per "
                                               "phase within 10 % of the old board's 0.0883 / 0.3385, and its tokens/s within "
                                               "3 %; get_throttled 0x0 throughout"},
            "verdicts": "HOLDS / FAILS / INCOMPLETE: FAILS if any valid compared figure is outside its tolerance; else "
                        "INCOMPLETE if a registered figure is missing or its row is flagged throttled, exit-N or n<3; else HOLDS. "
                        "warm-start is reported, not voiding"},
        "diagnostic_stated_in_advance": "if decode rows fail while prefill rows hold, the bootloader is the first suspect: the "
                                        "new board runs a 2024-09-23 bootloader, the old one ran 2026-01-21, and later "
                                        "bootloaders changed the Pi 5's SDRAM settings. Updating it (rpi-eeprom-update -a, "
                                        "the project's own setup step, Pi5/docs/SETUP.md) is a firmware flash: only with "
                                        "the user's yes, and the decode rows re-taken as a follow-up registered then",
        "merge_rule": {
            "always": "the session's run dirs are appended to the dataset's Rebuild line (testbed/dataset/README.md) after "
                      "scoring, node_id pi5, so the planner, which keeps the last row per key, prices the board on the bench; "
                      "the rows before 2026-10-06 are the old board's (serial b266e187...), from 2026-10-19 the new board's",
            "if_RB-PP_and_RB-TG_hold": "every earlier Pi 5 result stands for the Pi 5 (two units, one population; the "
                                       "differences are reported as unit-to-unit repeatability)",
            "otherwise": "the earlier Pi 5 results are reported as the old board's, and any comparison across 2026-10-05 names "
                         "the unit and the measured difference"},
        "energy": {"outlet": 2, "outlet_name": "pi5", "runs": energy_runs},
        "inputs_sha256": {p: sha256(os.path.join(ROOT, p)) for p in CODE + DATA},
    }
    write_ro(OUT, reg)


if __name__ == "__main__":
    main()
