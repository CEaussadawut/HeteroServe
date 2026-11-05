#!/usr/bin/env python3
"""Register the bootloader follow-up for the replacement Pi 5 (2026-10-20), BEFORE the EEPROM update is installed.

    python3 testbed/scripts/register_pi5b_fw_20261020.py testbed/results/pi5b_session_<stamp>

Takes the as-found session (pi5b_char_20261019.sh, scored by score_pi5b_20261019.py) and writes, read-only:
  testbed/dataset/pi5b_fw_predictions_20261020.json        the follow-up: the old board's rows and figures as in the first
        registration (RB-PP, RB-TG, RB-MEM, RB-E against them, same tolerances), the as-found rows, the bootloader before and
        after, and the firmware hypotheses FW-ALL, FW-HALF-TG, FW-HALF-PP8, FW-NOWORSE (score_pi5b_fw_20261020.py)
  testbed/dataset/energy_predictions_20261020_pi5fw_<q>.json    the five energy runs' bands, the first registration's
Why: the first registration (pi5b_predictions_20261019.json) stated in advance that a slower new board points first to its
bootloader (2024-09-23 against the old board's 2026-01-21; later bootloaders changed the Pi 5's SDRAM settings), and that an
update needs the user's yes and a follow-up registered then. The user, 2026-10-20 ~00:20: "Yes, update it".
"""
import csv, datetime, hashlib, json, os, stat, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DS = os.path.join(ROOT, "testbed", "dataset")
FIRST = os.path.join(DS, "pi5b_predictions_20261019.json")
OUT = os.path.join(DS, "pi5b_fw_predictions_20261020.json")
EREG = os.path.join(DS, "energy_predictions_20261020_pi5fw_%s.json")    # one per quantisation, as the first session's
CODE = ["testbed/scripts/pi5b_fw_20261020.sh", "testbed/scripts/score_pi5b_20261019.py", "testbed/scripts/score_pi5b_fw_20261020.py",
        "testbed/scripts/register_pi5b_fw_20261020.py", "testbed/scripts/run_quant_campaign.sh", "testbed/scripts/bench_worker_role.sh",
        "testbed/scripts/build_dataset.py", "testbed/scripts/memory_snapshot.py", "testbed/scripts/energy_phase_run.py",
        "testbed/scripts/score_energy_phase_run.py", "testbed/scripts/shelly_strip.py"]


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


def main():
    sess = os.path.abspath(sys.argv[1])
    first = json.load(open(FIRST))
    sc = json.load(open(os.path.join(sess, "score.json")))
    assert sc["registration"] == os.path.relpath(FIRST, ROOT), sc["registration"]
    rows_csv = os.path.join(sess, "rows", "characterisation.csv")
    af = {}
    for r in csv.DictReader(open(rows_csv)):
        if r["node_id"] == "pi5":
            af["%s|%s|%s" % (r["device"], r["model"], r["test"])] = {"phase": r["phase"], "mean_ts": float(r["mean_ts"]),
                                                                    "flags": r["flags"], "source": r["source"]}
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    runs = {}
    for q, r0 in first["energy"]["runs"].items():
        e_first = json.load(open(os.path.join(ROOT, r0["registration"])))
        ereg = dict(e_first, registered=now,
                    campaign="2026-10-20: the replacement Pi 5's %s energy per token after its bootloader update (registered with "
                             "pi5b_fw_predictions_20261020.json); the bands are the first registration's (%s)" % (q, r0["registration"]))
        write_ro(EREG % q, ereg)
        runs[q] = dict(r0, registration=os.path.relpath(EREG % q, ROOT), registration_sha256=sha256(EREG % q))
    reg = {
        "registered_at": now, "script": "testbed/scripts/register_pi5b_fw_20261020.py",
        "question": "how much of the replacement Pi 5's gap to the old board is its bootloader? After the official update, are "
                    "its rows the old board's, within the first registration's tolerances?",
        "user_decision": "2026-10-20 ~00:20, \"Yes, update it (Recommended)\": install the official update on the card "
                         "(rpi-eeprom-update -a, the project's own setup step), reboot, re-take the rows; keep the as-found rows",
        "first_registration": {"path": os.path.relpath(FIRST, ROOT), "sha256": sha256(FIRST)},
        "old_board": first["old_board"], "new_board": first["new_board"],
        "bootloader_before": {"timestamp": "1727096576", "date": "2024-09-23 13:02:56 UTC", "version": "26826259"},
        "bootloader_after": {"timestamp": "1779807685", "date": "2026-05-26 15:01:25 UTC", "release": "default",
                             "source": "rpi-eeprom 28.31-1 on the card (/usr/lib/firmware/raspberrypi/bootloader-2712/default), "
                                       "as rpi-eeprom-update reported it on 2026-10-19"},
        "as_found": {"session": os.path.relpath(sess, ROOT), "score_sha256": sha256(os.path.join(sess, "score.json")),
                     "verdicts": {k: v["verdict"] for k, v in sc["hypotheses"].items()}},
        "as_found_rows": af,
        "old_rows": first["old_rows"], "old_memory": first["old_memory"], "old_energy": first["old_energy"],
        "hypotheses": first["hypotheses"],
        "fw_hypotheses": {
            "FW-ALL": {"claim": "after the update every matched row is within the first registration's tolerances of the old "
                                "board's (RB-PP and RB-TG hold): the bootloader was the whole gap"},
            "FW-HALF-TG": {"claim": "the decode rows' mean |gap to the old board| after the update is at most half its as-found "
                                    "value: the bootloader is most of the decode gap"},
            "FW-HALF-PP8": {"claim": "the same for the host Q8_0 pp512 row, the prefill row with the largest as-found gap"},
            "FW-NOWORSE": {"tol_pct": 3.0, "claim": "no matched row is slower after the update than as found by more than 3 %"},
            "verdicts": "HOLDS / FAILS / INCOMPLETE, as in the first registration"},
        "plan": {"runner": "testbed/scripts/pi5b_fw_20261020.sh: steps 1-11 of the first session (memory; host and worker rows at "
                           "both protocols; the worker sweep; the five energy runs), refused unless the board reports the "
                           "registered bootloader timestamp",
                 "before_it": "sudo rpi-eeprom-update -a, then sudo reboot; on the way back: the clock synchronized, Wi-Fi still "
                              "off, the serial the registered board's"},
        "merge_rule": {"amended": "the follow-up's run dirs go into the dataset's Rebuild line AFTER the as-found ones, so the "
                                  "planner prices the board with its post-update rows (the board as it runs from now on); the "
                                  "as-found rows stay in the dataset, dated, and are reported",
                       "first": first["merge_rule"]},
        "energy": {"outlet": 2, "outlet_name": "pi5", "runs": runs,
                   "as_found_stopped": "the as-found session was stopped after its step 7 (the Q4_K_M energy run, RB-E's): the "
                                       "user's yes to the update came after its registration, and its other four energy runs "
                                       "(sanity bands only, no hypothesis) would have described a firmware state the update ends; "
                                       "they run here instead"},
        "inputs_sha256": {**{p: sha256(os.path.join(ROOT, p)) for p in CODE},
                          **{os.path.relpath(p, ROOT): sha256(p) for p in (FIRST, os.path.join(sess, "score.json"),
                                                                            os.path.join(sess, "steps.tsv"), rows_csv)},
                          "testbed/dataset/characterisation.csv": sha256(os.path.join(DS, "characterisation.csv"))},
    }
    write_ro(OUT, reg)


if __name__ == "__main__":
    main()
