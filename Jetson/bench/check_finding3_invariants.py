#!/usr/bin/env python3
"""Assert the properties audit finding #3 requires, over a joined *_bench_energy.csv.

Finding #3: a case with both n_prompt and n_gen is ONE llama-bench invocation
running TWO tests inside ONE power window, so its row paired one phase's
throughput with both phases' energy. The fix splits such cases, emits one record
per bench row, and refuses a per-token figure for any window that still holds
more than one phase.

This checker is deliberately independent of how the fix is spelled: it does not
care what the cases are called, only that no row can pair a throughput measured
on one phase with energy measured over a different scope.

    check_finding3_invariants.py <joined.csv> [<joined.csv> ...]

Exits 1 and lists every violation, or prints the row counts it checked.
"""
import csv
import sys

PER_TOKEN = ("joules_per_token", "wh_per_mtok", "active_joules_per_token")
PER_REQUEST = ("joules_per_request", "wh_per_krequest", "active_joules_per_request")
KNOWN_STATUS = {
    "ok", "refused_multi_phase", "refused_combined_request", "refused_phase_mismatch",
    "run_failed", "no_bench_csv", "no_power_window", "no_token_count",
    "model_missing", "refused_counts_mismatch", "refused_counts_unverified",
    "refused_config_mismatch", "estimated_from_throughput",
}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def has(row, col):
    return row.get(col, "") not in ("", None)


def counts(row):
    """The row's OWN token counts. bench_n_* is what llama-bench reported for
    this test; n_* is what the matrix asked for, and the two differ precisely in
    the cases this checker exists to catch -- a -pg case has n_*=0 while the row
    carries both, and a mixed invocation has both in n_* while each row carries
    one."""
    if has(row, "bench_n_prompt") or has(row, "bench_n_gen"):
        return num(row.get("bench_n_prompt")) or 0, num(row.get("bench_n_gen")) or 0
    return num(row.get("n_prompt")) or 0, num(row.get("n_gen")) or 0


def check(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    fails, notes, checked = [], [], 0

    for i, r in enumerate(rows):
        tag = "row %d (%s / %s)" % (i, r.get("case_id"), r.get("test") or r.get("marker"))
        status = (r.get("per_token_status") or "").strip()
        p, g = counts(r)
        working = num(r.get("working_rows"))
        if working is None:
            working = num(r.get("bench_rows"))
        jpt = num(r.get("joules_per_token"))
        tok = num(r.get("tokens"))
        # A refusal is the CORRECT outcome for a ref whose row contradicts
        # itself, so such observations are routed to notes: the data is suspect,
        # but the driver handled it and published nothing. Only a PUBLISHED
        # number violates the invariant this checker exists to enforce.
        # A published per-unit figure is a violation no matter what the status
        # says. Routing to notes on the status string alone let a row that both
        # says "refused_" AND carries a number be downgraded to a note -- and the
        # column scans that would have caught it live in rules that need optional
        # columns, so a CSV missing those exits 0 with a real defect in it.
        # refused_combined_request is the one status that legitimately publishes,
        # and only the per-REQUEST columns.
        published = any(has(r, c) for c in PER_TOKEN) or (
            status != "refused_combined_request"
            and any(has(r, c) for c in PER_REQUEST))
        refused = status.startswith("refused_") and not published
        if status.startswith("refused_") and published:
            fails.append("%s: per_token_status=%s but the row publishes %s"
                         % (tag, status,
                            ", ".join(c for c in PER_TOKEN + PER_REQUEST if has(r, c))))

        # 0. Every row must say whether it carries a per-token figure and why.
        #    A blank cell in this CSV otherwise means at least five things.
        if "per_token_status" in (r.keys() if hasattr(r, "keys") else []):
            if status not in KNOWN_STATUS:
                fails.append("%s: per_token_status=%r is not a known status" % (tag, status))

        if r.get("status") in ("failed", "skipped"):
            if jpt is not None:
                fails.append("%s: run %s but carries joules_per_token=%s"
                             % (tag, r.get("status"), jpt))
            continue
        checked += 1

        # 1. A row describing BOTH phases may exist only as a deliberate fused
        #    request, and only with every per-token column withheld.
        if p and g:
            if (r.get("row_phase") or "").strip() != "combined":
                fails.append("%s: n_prompt=%g and n_gen=%g but row_phase=%r"
                             % (tag, p, g, r.get("row_phase")))
            if status and status != "refused_combined_request":
                fails.append("%s: fused row has per_token_status=%r, expected "
                             "refused_combined_request" % (tag, status))
            for col in PER_TOKEN:
                if has(r, col):
                    fails.append("%s: fused row publishes %s=%s -- its joules are "
                                 "not attributable to a phase" % (tag, col, r.get(col)))

        # 2. A per-token figure requires exactly one working bench row behind it.
        if jpt is not None and working is not None and working != 1:
            fails.append("%s: joules_per_token=%s but working_rows=%g" % (tag, jpt, working))

        # 3. A window shared by more than one phase must withhold every
        #    per-token column, and must say why.
        if working is not None and working > 1:
            for col in PER_TOKEN:
                if has(r, col):
                    fails.append("%s: working_rows=%g but %s=%s is populated"
                                 % (tag, working, col, r.get(col)))
            if not status and not (r.get("energy_basis") or "").strip():
                fails.append("%s: multi-phase window gives no reason the columns "
                             "are blank" % tag)

        # 4. The test label must agree with the counts it claims to describe.
        #    It is an INDEPENDENT signal: llama-bench writes it from the instance
        #    it actually built, so it catches a ref whose count columns lie.
        test = (r.get("test") or "").strip().lower()
        if test.startswith("pp") and "+" not in test and g:
            fails.append("%s: test=%s (prefill) but n_gen=%g" % (tag, test, g))
        if test.startswith("tg") and p:
            fails.append("%s: test=%s (decode) but n_prompt=%g" % (tag, test, p))
        if "+" in test and not (p and g):
            (notes if refused else fails).append(
                "%s: test=%s is a fused label but the row reports n_prompt=%g "
                "n_gen=%g -- the counts and the label disagree about what ran"
                % (tag, test, p, g))

        # 5. The invocation and the row must describe the same phase. The driver
        #    records what it ASKED for in `phase` and what came BACK in
        #    `row_phase`; a fused -pg request answered by a row that looks like
        #    decode is finding #3 arriving through the ref instead of the matrix.
        want, got = (r.get("phase") or "").strip(), (r.get("row_phase") or "").strip()
        if want and got and want != got:
            # ANY refusal is an acceptable response -- a window shared by two
            # tests is refused as multi_phase before the mismatch is even
            # reached, and that is the more fundamental reason. What must never
            # happen is a mismatched row carrying a number.
            if status and not refused:
                fails.append("%s: asked for phase=%s but row_phase=%s, and "
                             "per_token_status=%r is not a refusal"
                             % (tag, want, got, status))
            elif refused:
                notes.append("%s: asked for phase=%s but row_phase=%s (refused as %s)"
                             % (tag, want, got, status))
            for col in PER_TOKEN + PER_REQUEST:
                if has(r, col):
                    fails.append("%s: phase=%s but row_phase=%s, yet publishes %s=%s"
                                 % (tag, want, got, col, r.get(col)))

        # 6. A -pg invocation must come back as a fused row, and must never carry
        #    a per-token figure whatever its reported counts say.
        pg_p, pg_g = num(r.get("pg_n_prompt")) or 0, num(r.get("pg_n_gen")) or 0
        if pg_p and pg_g:
            if got and got != "combined":
                (notes if refused else fails).append(
                    "%s: -pg %g,%g was requested but row_phase=%s"
                    % (tag, pg_p, pg_g, got))
            for col in PER_TOKEN:
                if has(r, col):
                    fails.append("%s: -pg row publishes %s=%s -- a fused window's "
                                 "joules are not attributable to a phase"
                                 % (tag, col, r.get(col)))

        # 6b. A per-token figure requires counts the ref actually reported AND
        #     that match what was asked for. A divisor taken from the request
        #     rather than from the measurement is unverifiable; a divisor that
        #     contradicts the request is wrong by exactly that ratio.
        req_p, req_g = num(r.get("req_n_prompt")), num(r.get("req_n_gen"))
        if jpt is not None:
            if (r.get("counts_source") or "").strip() == "matrix":
                fails.append("%s: joules_per_token=%s but counts_source=matrix -- "
                             "the ref reported no counts, so the divisor is the "
                             "request, not the measurement" % (tag, jpt))
            if req_p is not None and req_g is not None and (req_p or req_g) \
               and (p, g) != (req_p, req_g):
                fails.append("%s: joules_per_token=%s over n_prompt=%g n_gen=%g, but "
                             "the invocation asked for %g/%g" % (tag, jpt, p, g, req_p, req_g))

        # 7. Energy must never be attributed to zero or missing tokens.
        if jpt is not None and not tok:
            fails.append("%s: joules_per_token=%s over tokens=%r" % (tag, jpt, r.get("tokens")))

        # 8. Each per-unit figure must equal the energy it names, over its own
        #    denominator -- this is what catches a divisor from the wrong row.
        basis = (r.get("energy_basis") or "").strip()
        src = "compute_joules" if basis.startswith("compute") else "joules"
        for col, den_col, e_col in (
                ("joules_per_token", "tokens", src),
                # a fused row is denominated over its TIMED loops alone, so its
                # numerator is request_joules, never the whole compute window
                ("joules_per_request", "requests", "request_joules")):
            v, den, e = num(r.get(col)), num(r.get(den_col)), num(r.get(e_col))
            if v is not None and e is not None and den:
                if abs(v - e / den) > max(1e-6, abs(v) * 0.001):
                    fails.append("%s: %s=%s != %s/%s=%.6f"
                                 % (tag, col, v, e_col, den_col, e / den))
            elif v is not None and e is None:
                fails.append("%s: %s=%s but %s is absent -- cannot be checked"
                             % (tag, col, v, e_col))

        # 9. "ok" must mean the number is really there, backed by real energy.
        #    A row with no power window at all has its own status and must not
        #    masquerade as ok.
        if status == "ok":
            if not has(r, "joules"):
                fails.append("%s: per_token_status=ok but no joules -- there was "
                             "no power window, so the status should say so" % tag)
            elif jpt is None and not (p and g):
                fails.append("%s: per_token_status=ok but no joules_per_token" % tag)

    n_skip = sum(1 for r in rows if r.get("status") == "skipped")
    if n_skip:
        notes.append("%d of %d planned run(s) never happened (status=skipped): %s"
                     % (n_skip, len(rows),
                        ", ".join(sorted({r["case_id"] for r in rows
                                          if r.get("status") == "skipped"}))))
    print("checked %d rows (%d total) in %s" % (checked, len(rows), path))
    if notes:
        print("  %d row(s) carry suspect data that the driver correctly refused:" % len(notes))
        for n in notes:
            print("    note: " + n)
        print("    -> no bad number was published, but that case produced NO usable")
        print("       energy figure. Check the ref and the run log before relying on")
        print("       the sweep as complete.")
    return fails


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    fails = []
    for path in sys.argv[1:]:
        fails.extend(check(path))
    if fails:
        print("\nFAIL (%d):" % len(fails))
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("all finding-#3 invariants hold")


if __name__ == "__main__":
    main()
