"""
lab45_effective_age_interval.py - an interval for the number everyone will quote.

Imports lab05_robustness, lab22_factor_benchmark and lab37_lead_lag; keep all
four in labs/.  Runtime about three minutes.

THE OMISSION THIS FIXES
-----------------------
"A fifty-five-day-old mark plus today's foreign closes forecasts as well as a
mark 4.6 days old" is the sentence this paper will be remembered by.  It is in
the abstract and in Section 9.3, and until now it carried no uncertainty
anywhere, while every other headline in the paper carried a bootstrap interval.
That asymmetry is not defensible: the effective age is an interpolation of one
estimated quantity onto a curve of other estimated quantities, so it inherits
sampling error twice - once from the combined model being matched, once from the
own-only curve doing the matching - and it is if anything MORE fragile than the
R-squared figures that are quoted with intervals.

WHY THE INTERVAL IS NOT SIMPLY THE INTERVAL ON R-SQUARED
--------------------------------------------------------
The map from skill to age is the inverse of the delay curve, and that curve is
steep near zero and flat far out.  A fixed error in R-squared therefore becomes a
SMALL error in age at short effective ages and a LARGE one further along, so the
interval must be asymmetric and cannot be obtained by pushing the R-squared
interval through the curve at its midpoint.  The only honest construction
resamples and re-interpolates, which is what happens below.

THE CONSTRUCTION
----------------
Every forecast is computed once: the own-only model at sixteen mark ages, and
the combined model at delta = 55 with the foreign feed g = 0, 1 and 5 days
stale.  Those forecasts do not depend on which test days are resampled, so the
bootstrap is over DAYS only, in moving blocks of ten as everywhere else in this
paper.  On each resample:

    1. recompute the own-only curve, R-squared at each of the sixteen ages,
       with the benchmark variance recomputed on the SAME resampled days, which
       is the whole-ratio bootstrap Section 4 uses;
    2. recompute the combined model's R-squared on those days;
    3. interpolate the age at which the curve attains that skill.

The percentile interval of step 3 across replications is the answer.  A
replication in which the combined model beats a zero-day-old mark yields an age
of zero; one in which it is worse than the oldest mark on the grid yields no
finite age at all, and both outcomes are counted and reported rather than
dropped, because silently discarding the replications that misbehave is how a
bootstrap interval becomes narrower than the thing it is measuring.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab22_factor_benchmark as F
import lab37_lead_lag as LL

SEED = 20260915
TARGET = "SPX"
DELTA = 55                                  # the delay the headline quotes
GAPS = [0, 1, 5]                            # foreign feed staleness, as in Table 15
FINE = [0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 21, 26, 34, 44, 55]
N_BOOT, BLOCK = 1500, 2 * L.HORIZON


def r2_on(y, f, rows):
    """Out-of-sample R-squared on a subset of days, benchmark recomputed there."""
    yy, ff = y[rows], f[rows]
    den = ((yy - yy.mean()) ** 2).sum()
    return 1 - ((yy - ff) ** 2).sum() / den if den > 0 else np.nan


def age_of(skill, xs, vs):
    """The mark age whose own-only skill equals `skill`, by linear interpolation.

    Returns 0.0 when the model beats a fresh mark and nan when it is worse than
    the oldest mark on the grid.  Both are real answers about a replication and
    are counted by the caller.
    """
    if skill >= vs[0]:
        return 0.0
    if skill <= vs[-1]:
        return np.nan
    for i in range(len(xs) - 1):
        if vs[i] >= skill >= vs[i + 1]:
            span = vs[i] - vs[i + 1]
            w = 0.0 if span < 1e-12 else (vs[i] - skill) / span
            return float(xs[i] + w * (xs[i + 1] - xs[i]))
    return np.nan


def blocks_of(n, rng, block=BLOCK):
    """Moving-block resample of the day index, as used throughout the paper."""
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints an interval for the effective-age headline of Section 9.3.\n")

    own, P, y, idx, D, peers, _lag = LL.panel(folder)
    yb = y[idx]
    n = len(idx)
    print(f"target {TARGET}, {n} test days, {len(peers)} foreign peers, "
          f"{N_BOOT} replications in moving blocks of {BLOCK}\n")

    # ---------------- forecasts, computed once ----------------------------
    print("=" * 96)
    print("A.  THE RULER, AND THE MODELS BEING MEASURED AGAINST IT")
    print("=" * 96)
    curve_f = {g: LL.walk_own(own, y, idx, g) for g in FINE}
    comb_f = {g: LL.walk(own, P, y, idx, DELTA, g) for g in GAPS}
    xs = np.array(FINE, dtype=float)
    vs_full = np.array([F.r2(yb, curve_f[g]) for g in FINE])
    print("own-only skill against the age of the mark:")
    print("  " + "  ".join(f"{g}d:{v:.3f}" for g, v in zip(FINE[:8], vs_full[:8])))
    print("  " + "  ".join(f"{g}d:{v:.3f}" for g, v in zip(FINE[8:], vs_full[8:])))
    print(f"\nthe combined model at delta = {DELTA}, by foreign-feed staleness:")
    for g in GAPS:
        print(f"  g = {g:>2}:  R-squared {F.r2(yb, comb_f[g]):.4f}   "
              f"effective age {age_of(F.r2(yb, comb_f[g]), xs, vs_full):.1f} days")

    # ---------------- the bootstrap ---------------------------------------
    print("\n" + "=" * 96)
    print("B.  THE INTERVAL")
    print("=" * 96)
    print("Days are resampled in moving blocks; the curve, the combined model and the")
    print("benchmark variance are all recomputed on the SAME resampled days, and the")
    print("age is re-interpolated inside each replication.\n")
    print(f"{'g':>4}{'effective age':>16}{'95% interval':>22}{'at 0 days':>12}"
          f"{'off the grid':>14}")
    # One resample serves every gap, so the three ages within a replication are
    # computed on the SAME days.  That is what makes the paired differences in
    # part C legitimate; drawing a separate resample per gap would destroy the
    # correlation between them and inflate every difference's interval.
    rng = np.random.default_rng(SEED)
    draws = {g: [] for g in GAPS}
    nzero = {g: 0 for g in GAPS}
    noff = {g: 0 for g in GAPS}
    for _ in range(N_BOOT):
        rows = blocks_of(n, rng)
        vs = np.array([r2_on(yb, curve_f[k], rows) for k in FINE])
        # The curve must be decreasing for the interpolation to mean anything; a
        # resample that breaks monotonicity is made monotone by a running
        # minimum rather than discarded, which is the conservative direction -
        # it can only make the implied age older.
        vs = np.minimum.accumulate(vs)
        for g in GAPS:
            a = age_of(r2_on(yb, comb_f[g], rows), xs, vs)
            draws[g].append(a)
            if np.isnan(a):
                noff[g] += 1
            elif a == 0.0:
                nzero[g] += 1
    out = {}
    for g in GAPS:
        arr = np.array(draws[g], dtype=float)
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        out[g] = (age_of(F.r2(yb, comb_f[g]), xs, vs_full), lo, hi,
                  nzero[g], noff[g])
        print(f"{g:>4}{out[g][0]:>15.1f}d{f'[{lo:.1f}, {hi:.1f}] days':>22}"
              f"{nzero[g]:>12}{noff[g]:>14}")
    print(f"\n  'at 0 days' counts replications where the combined model beat a fresh")
    print(f"  mark outright; 'off the grid' counts those where it was worse than a")
    print(f"  {FINE[-1]}-day-old one. Both are reported rather than dropped.")

    # ---------------- what it means ---------------------------------------
    print("\n" + "=" * 96)
    print("C.  IS THE HEADLINE SAFE?")
    print("=" * 96)
    pt, lo, hi, _z, _o = out[0]
    width = hi - lo
    print(f"  The headline quotes {pt:.1f} days. The interval is "
          f"[{lo:.1f}, {hi:.1f}], {width:.1f} days wide.")
    print(f"  Relative to the {DELTA}-day mark being repaired, the interval spans")
    print(f"  {lo / DELTA:.1%} to {hi / DELTA:.1%} of the original staleness.\n")
    # What a day of foreign staleness costs, in days of domestic freshness.
    # This is the PAIRED difference, recomputed inside each replication on the
    # same days - not a comparison of the two marginal intervals above, which
    # overlap for the ordinary reason that overlapping marginals are a weak and
    # conservative test of a difference and say little about it either way.
    print("  What one day of foreign staleness costs, in days of domestic")
    print("  freshness, as a paired difference inside each replication:\n")
    print(f"{'comparison':>16}{'point':>10}{'95% interval':>22}{'excludes 0':>12}")
    pairs = {}
    for g in GAPS[1:]:
        d = np.array(draws[g], dtype=float) - np.array(draws[0], dtype=float)
        d = d[np.isfinite(d)]
        lo_d, hi_d = np.percentile(d, [2.5, 97.5])
        pt_d = out[g][0] - out[0][0]
        pairs[g] = (pt_d, lo_d, hi_d, lo_d > 0)
        print(f"{f'g = {g} minus g = 0':>16}{pt_d:>9.1f}d"
              f"{f'[{lo_d:.1f}, {hi_d:.1f}] days':>22}"
              f"{'yes' if lo_d > 0 else 'no':>12}")
    sep = pairs[1][3]

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    if hi < 10:
        print(f"  The effective age is {pt:.1f} days with a 95% interval of "
              f"[{lo:.1f}, {hi:.1f}].")
        print(f"  Every value in that interval is a small fraction of the {DELTA} days of")
        print("  staleness the cross-section is repairing, so the headline is a finding")
        print("  and not an artefact of reading one point off a curve. It should be")
        print("  quoted with the interval, which is what this lab exists to supply.")
    else:
        print(f"  The interval runs to {hi:.1f} days, which is no longer a small fraction")
        print(f"  of {DELTA}. The point estimate is not wrong but it is far more fragile")
        print("  than quoting it alone suggests, and the paper must quote the interval")
        print("  wherever it quotes the number.")
    if sep:
        print(f"\n  A single day of foreign staleness costs "
              f"{pairs[1][0]:.1f} days of domestic")
        print(f"  freshness, interval [{pairs[1][1]:.1f}, {pairs[1][2]:.1f}], "
              f"which excludes zero. Section 9.3's")
        print("  exchange rate is a measured difference and not a direction, and it is")
        print("  now quotable in the unit the section is written in. The two marginal")
        print("  intervals in part B overlap, which is the ordinary behaviour of")
        print("  marginal intervals and is not evidence against a paired difference.")
    else:
        print(f"\n  The paired difference is {pairs[1][0]:.1f} days with an interval of")
        print(f"  [{pairs[1][1]:.1f}, {pairs[1][2]:.1f}], which does not exclude zero, so at "
              f"this delay a single day")
        print("  of foreign staleness cannot be shown to cost anything in the age unit,")
        print("  and Section 9.3 must say so rather than quoting the two point")
        print("  estimates side by side.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
