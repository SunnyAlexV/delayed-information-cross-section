"""
run_all.py - reproduce every number in both papers, in order.

    python run_all.py              run everything, print as it goes
    python run_all.py --check      run everything and DIFF against expected_output/
    python run_all.py lab05        run one lab by name

Each lab writes its console output to output/<lab>.txt.  With --check, that
output is compared line by line against the copy in expected_output/.  A clean
--check is the reproducibility claim the papers make; anything else is a real
difference and the diff tells you where.

Two lines are expected to differ and are ignored by the comparison: the line
naming the data folder, and the runtime line.  Nothing else is exempt.
"""

import os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def _dir(name, marker):
    """The folder called `name`, or this one if the files simply sit beside us.

    The tidy layout has labs/, data/ and expected_output/ as siblings of this
    script, and that is what the README describes.  It is not what always
    arrives.  GitHub's web uploader flattens a directory tree unless each
    folder is dragged in as a folder, so a repository uploaded by dragging
    files lands with all hundred-odd of them in one directory - and then this
    script stopped before running anything, which is a confusing way to tell
    someone their upload went sideways.  `marker` is a filename that must exist
    for the fallback to be believed, so a genuinely incomplete download still
    fails loudly instead of being mistaken for a flat layout.
    """
    tidy = os.path.join(HERE, name)
    if os.path.isdir(tidy):
        return tidy
    import glob as _g
    return HERE if _g.glob(os.path.join(HERE, marker)) else tidy


LABS = _dir("labs", "lab05_robustness.py")
DATA = _dir("data", "SPX_*.csv")


def _data_folder(argv):
    """Where the inputs are, which is not in this repository.

    Every vendor behind the inputs forbids redistributing their files, so the
    repository ships data/MANIFEST.tsv and data/README.md instead of the CSVs -
    see verify_data.py.  A reader who has obtained the files can either drop
    them into data/ or keep them anywhere and say so:

        python run_all.py --data /path/to/my/copy

    The manifest, the FHFA series and the provenance record stay in data/, so
    the folder is not empty and `--data` is a convenience rather than a
    requirement.
    """
    import glob as _g
    for i, a in enumerate(argv):
        if a == "--data" and i + 1 < len(argv):
            d = os.path.abspath(argv[i + 1])
            if not os.path.isdir(d):
                raise SystemExit(f"--data {d} is not a folder")
            return d
        if a.startswith("--data="):
            d = os.path.abspath(a.split("=", 1)[1])
            if not os.path.isdir(d):
                raise SystemExit(f"--data {d} is not a folder")
            return d
    if _g.glob(os.path.join(DATA, "SPX_*.csv")):
        return DATA
    raise SystemExit(
        "The input files are not in " + DATA + ".\n\n"
        "They are not distributed with this repository, because every vendor\n"
        "behind them prohibits it: Investing.com, Cboe and Yahoo all forbid\n"
        "redistribution, and the Case-Shiller series carry FRED's\n"
        "'Copyrighted: Pre-approval Required' label. What ships instead is\n"
        "data/MANIFEST.tsv, which records for every file its source, its date\n"
        "span, the SHA-256 of the exact bytes the paper used, and a digest of\n"
        "the derived series taken one year at a time.\n\n"
        "  1. data/README.md says where each file comes from and how to get it.\n"
        "  2. python verify_data.py <folder>   confirms your copy matches ours.\n"
        "  3. python run_all.py --check --data <folder>\n\n"
        "A reader who obtains the same exports reproduces every number in the\n"
        "paper bitwise; a reader with a fresher export is told so, per file and\n"
        "per year, rather than left to guess.")
OUT = os.path.join(HERE, "output")
EXPECTED = _dir("expected_output", "lab05_robustness.txt")

# (script, argument, what it establishes)
ORDER = [
    ("lab01_ceiling.py", None,
     "the ceiling formula, checked against an explicit Kalman filter"),
    ("lab02_delay_curve.py", None,
     "the decay curve on one market, with honest baselines"),
    ("lab02b_threshold_ceiling.py", None,
     "how close the threshold rule gets to its own ceiling"),
    ("lab03_crosssection.py", DATA,
     "seam audit, time-zone admissibility, in-sample upper bound"),
    ("lab04_walkforward.py", DATA,
     "out-of-sample walk-forward, binary target"),
    ("lab05_robustness.py", DATA,
     "benchmark, five markets, continuous target"),
    ("lab06_inference.py", DATA,
     "Giacomini-White, Clark-West, simultaneous bands, placebo"),
    ("lab07_estimation_cost.py", DATA,
     "what breadth costs, and the gross-versus-net decomposition"),
    ("lab08_implied_vol.py", DATA,
     "the implied-volatility horse race - the objection, tested"),
    ("lab09_nonlinearity.py", DATA,
     "does a non-linear map find more than the linear one?"),
    ("lab10_loss_scale.py", DATA,
     "which scale each loss lives on, and full-ratio intervals"),
    ("lab11_markets_continuous.py", DATA,
     "the five-market check, on the continuous target"),
    ("lab12_appendix.py", DATA,
     "settings, ridge-grid sensitivity, block-length sensitivity"),
    ("lab13_origin_median.py", DATA,
     "does the trailing median leak the future into the target?"),
    ("lab14_appraisal_smoothing.py", DATA,
     "is a clean lag the right model of a stale mark?"),
    ("lab15_nonlinear_given_iv.py", DATA,
     "is redundancy given implied volatility an artefact of the linear estimator?"),
    ("lab16_clark_west_shrinkage.py", DATA,
     "does our own Clark-West column survive the companion paper's result?"),
    ("lab17_horizons.py", DATA,
     "is the substitution rate a property of the information or of the horizon?"),
    ("lab18_economic_reading.py", DATA,
     "what the R-squared column means to someone sizing a position"),
    ("lab19_vstoxx_third_series.py", DATA,
     "would a fuller implied-volatility block change the redundancy conclusion?"),
    ("lab20_ceiling_is_not_a_ceiling.py", None,
     "two referee corrections to the companion note, tested rather than conceded"),
    ("lab21_stronger_inference.py", DATA,
     "a better null for the cost of breadth, and a Fieller interval for the rate"),
    ("lab22_factor_benchmark.py", DATA,
     "fit the cheaper representation instead of inferring it from surrogates"),
    ("lab23_compressed_everywhere.py", DATA,
     "the compressed block under every loss, and redundancy with one regressor"),
    ("lab24_proper_scores.py", None,
     "the note's forecasters under proper scoring rules, not 0-1 accuracy"),
    ("lab25_window_sensitivity.py", None,
     "is the threshold rule's standing a property of the 252-day median?"),
    ("lab26_quantile_target.py", None,
     "the note's theory off the median: a tail target, and where the data runs out"),
    ("lab27_regime_conditioning.py", DATA,
     "does any of it hold in stress? a ladder of regime cuts, on a scale-free measure"),
    ("lab28_orthogonal_breadth.py", DATA,
     "is breadth a global factor by another name? the block split against its own PC1"),
    ("lab29_single_regime.py", DATA,
     "EXPLORATORY, cited by neither paper: one episode at a time, and one full arc"),
    ("lab30_information_bound.py", DATA,
     "EXPLORATORY: a ceiling from information theory, assuming no distribution"),
    ("lab31_mark_to_model.py", DATA,
     "EXPLORATORY: the practitioner's mark-to-model, against the paper's regression"),
    ("lab32_masked_training.py", DATA,
     "EXPLORATORY: one model for every delay, trained by masking, against ten"),
    ("lab33_ragged_edge.py", DATA,
     "a Kalman filter on the ragged edge, against the paper's regression"),
    ("lab34_data_snooping.py", DATA,
     "EXPLORATORY: Reality Check and SPA - what survives admitting how many models were tried"),
    ("lab35_purged_cv.py", DATA,
     "EXPLORATORY: is the purge this project already leaves big enough?"),
    ("lab36_variance_risk_premium.py", DATA,
     "EXPLORATORY: the variance risk premium, and which premium a delayed desk can form"),
    ("lab37_lead_lag.py", DATA,
     "propagation or a common factor, and what a stale foreign feed costs"),
    ("lab38_domestic_baseline.py", DATA,
     "is the substitution rate an artefact of a weak domestic control?"),
    ("lab39_temporal_stability.py", DATA,
     "is R(delta) stable over nineteen years, or an average of two eras?"),
    ("lab40_level_or_normaliser.py", DATA,
     "is the level result information, or the model undoing the normalisation?"),
    ("lab41_conditional_anatomy.py", DATA,
     "is the conditional rate about markets, or about where the ratio is measurable?"),
    ("lab42_factor_model.py", DATA,
     "the factor model the results assemble, and its three predictions"),
    ("lab43_official_vix.py", DATA,
     "is the paper's VIX the series Cboe published? the provenance audit"),
    ("lab44_horizon_matched_iv.py", DATA,
     "the horse race with the horizons matched - nine-day implied volatility"),
    ("lab45_effective_age_interval.py", DATA,
     "an interval for the effective-age headline, by re-interpolating each resample"),
    ("lab46_own_only_filter.py", DATA,
     "is the rate inflated by a weak control? the state-space objection, tested"),
    ("lab47_target_seam.py", DATA,
     "the one join that cannot be compared, audited against an independent vendor"),
    ("lab48_age_by_regime.py", DATA,
     "the calm-market cell, asked again in a coordinate with no denominator"),
    ("lab49_control_envelope.py", DATA,
     "the rate against eight domestic controls at once, and the lowest it reaches"),
    ("lab50_filtered_headline.py", DATA,
     "the headline recomputed inside the estimator Section 11 declined to adopt"),
    ("lab51_foreign_options.py", DATA,
     "whose options market? redundancy where the option chain is on another index"),
    ("lab52_compressed_foreign_options.py", DATA,
     "the same race with the block the paper recommends, and the target that did not fit"),
    ("lab53_no_options_targets.py", DATA,
     "four targets on which no volatility index is published at all"),
    ("lab54_monthly_feasibility.py", DATA,
     "does the measurement survive at the frequency an appraised asset reports?"),
    ("lab55_illiquid_measured.py", DATA,
     "the illiquid case measured on twenty Case-Shiller metros, not rehearsed"),
    ("lab56_seasonal_and_breadth.py", DATA,
     "is the illiquid rate a calendar effect? a placebo peer, and ragged breadth"),
    ("lab57_housing_overlap.py", DATA,
     "is the illiquid rate the index's own three-month average? "
     "non-overlapping quarters, and a de-smoothed panel"),
    ("lab58_ratio_inference.py", DATA,
     "does the interval cover the rate? a coverage study of four procedures, "
     "and the block length the real losses require"),
    ("lab59_session_timestamps.py", DATA,
     "is the admissibility rule true on real session clocks? daylight saving, "
     "holidays, and how much information the fixed table throws away"),
    ("lab60_outage_decomposition.py", DATA,
     "is delta a stale feature or a stale system? the outage split into "
     "information and estimation vintage"),
    ("lab61_block_sensitivity.py", None,
     "what was the companion note's block length worth? three block lengths "
     "and the stationary bootstrap beside each"),
    ("lab62_post_selection.py", None,
     "how many of the secondary grids' significant cells are chance? the "
     "post-selection audit and its false-discovery correction"),
]

# Lines that name a LOCATION rather than a result.  The reference outputs
# ship with these neutralised, because a replication package should not
# carry the author's own filesystem paths, and a reader whose data sits
# elsewhere should not see a diff for that reason alone.
IGNORE = (re.compile(r"^data folder:"), re.compile(r"^runtime "),
          re.compile(r"^panel: (?:/|[A-Za-z]:|<your data folder>)"))


def normalise(text):
    return [ln.rstrip() for ln in text.splitlines()
            if not any(p.match(ln) for p in IGNORE)]


def run(script, arg):
    cmd = [sys.executable, os.path.join(LABS, script)] + ([arg] if arg else [])
    t0 = time.time()
    # Pin the pipe to UTF-8 at both ends.  Without PYTHONIOENCODING the child
    # encodes its output with the Windows console codepage, and without the
    # explicit encoding= the parent decodes it with the locale - two chances for
    # a lab that prints correctly to be recorded wrongly.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       encoding="utf-8", errors="replace")
    return p.stdout + p.stderr, p.returncode, time.time() - t0


def preflight():
    """Fail with a sentence, not a WinError.

    The common way to get here is to double-click the .zip and run the script
    straight out of the archive preview.  Windows quietly extracts the single
    file you launched into Temp, so the script exists but labs/ and data/ next
    to it do not, and the first subprocess call dies with an unreadable
    "[WinError 267] The directory name is invalid".  Check first and say so.
    """
    missing = [n for n, d in (("labs", LABS), ("data", DATA)) if not os.path.isdir(d)]
    if not missing and LABS == HERE:
        print("note: labs/ and data/ are not present, but the scripts and CSVs are\n"
              "      beside this file, so everything runs from here.\n")
    if not missing:
        return
    zipped = ".zip" in HERE.lower()
    print("Cannot find " + " or ".join(missing) + f" next to this script.\n"
          f"  script is at : {HERE}")
    if zipped:
        print("\nThat path runs through a .zip, so this is being run from inside the\n"
              "archive.  Windows copies out only the file you launch, which is why the\n"
              "sibling folders are missing.\n"
              "\n  EXTRACT the archive to a real folder first - right-click the .zip,\n"
              "  'Extract All...', pick somewhere like your Desktop - then run\n"
              "  run_all.py from the extracted copy.")
    else:
        print("\nrun_all.py expects labs/ and data/ as sibling folders. Keep the\n"
              "extracted folder intact rather than moving the script out of it.")
    raise SystemExit(1)


def main(argv):
    preflight()
    check = "--check" in argv
    data = _data_folder(argv)
    skip = set()
    for i, a in enumerate(argv):
        if a == "--data":
            skip.add(i)
            skip.add(i + 1)
        elif a.startswith("--data="):
            skip.add(i)
    only = [a for i, a in enumerate(argv) if i and i not in skip
            and not a.startswith("--")]
    os.makedirs(OUT, exist_ok=True)
    todo = [r for r in ORDER if not only or any(o in r[0] for o in only)]
    if not todo:
        raise SystemExit(f"no lab matches {only}; known: {[r[0] for r in ORDER]}")

    failures = []
    for script, arg, what in todo:
        name = script.replace(".py", "")
        print(f"\n{'=' * 78}\n{name}  -  {what}\n{'=' * 78}", flush=True)
        text, code, secs = run(script, data if arg is not None else None)
        with open(os.path.join(OUT, name + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(text)
        if code != 0:
            print(text)
            failures.append((name, "exited non-zero"))
            continue
        if check:
            ref = os.path.join(EXPECTED, name + ".txt")
            if not os.path.exists(ref):
                failures.append((name, "no reference output to compare against"))
                print(f"  no reference output for {name}")
                continue
            # encoding is pinned for the same reason verify_paper.py pins it:
            # open() otherwise follows the locale, so a file written as UTF-8
            # here is read back as cp1252 on Windows and "differs" spuriously.
            with open(ref, encoding="utf-8") as fh:
                got, want = normalise(text), normalise(fh.read())
            if got == want:
                print(f"  MATCHES the reference output ({len(got)} lines, {secs:.0f}s)")
            else:
                bad = [(i, w, g) for i, (w, g) in
                       enumerate(zip(want, got), 1) if w != g]
                failures.append((name, f"{len(bad)} differing lines"))
                print(f"  DIFFERS from the reference output ({secs:.0f}s)")
                for i, w, g in bad[:5]:
                    print(f"    line {i}\n      expected: {w}\n      got:      {g}")
                if len(want) != len(got):
                    print(f"    line count {len(want)} expected, {len(got)} produced")
        else:
            print(text)

    print(f"\n{'=' * 78}")
    if failures:
        for n, why in failures:
            print(f"  {n}: {why}")
        raise SystemExit(f"{len(failures)} of {len(todo)} labs did not reproduce")
    print(f"all {len(todo)} labs "
          f"{'reproduced the reference output exactly' if check else 'ran'}")


if __name__ == "__main__":
    main(sys.argv)
