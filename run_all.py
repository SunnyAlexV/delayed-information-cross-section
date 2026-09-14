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
LABS = os.path.join(HERE, "labs")
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
EXPECTED = os.path.join(HERE, "expected_output")

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
]

IGNORE = (re.compile(r"^data folder:"), re.compile(r"^runtime "))


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
    only = [a for a in argv[1:] if not a.startswith("--")]
    os.makedirs(OUT, exist_ok=True)
    todo = [r for r in ORDER if not only or any(o in r[0] for o in only)]
    if not todo:
        raise SystemExit(f"no lab matches {only}; known: {[r[0] for r in ORDER]}")

    failures = []
    for script, arg, what in todo:
        name = script.replace(".py", "")
        print(f"\n{'=' * 78}\n{name}  -  {what}\n{'=' * 78}", flush=True)
        text, code, secs = run(script, arg)
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
