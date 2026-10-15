"""
lab74_coupling_stability.py - does the rate fall when the correlation structure
breaks down?

Imports lab05_robustness, lab08_implied_vol, lab21_stronger_inference and
lab41_conditional_anatomy; keep all five in labs/.  Runtime about two minutes.

THE LIMITATION THIS TESTS
-------------------------
The paper fits its cross-sectional model with a ridge on a fixed window and so
treats the covariance between the target and its peers as something that holds
still long enough to be estimated.  Its own limitations section says that a
constant covariance misses the co-movement a crisis produces.  The objection has
a literature behind it: the correlation structure of an equity cross-section is
not stable, it reorganises, and the reorganisation is itself informative
(Musmeci, Aste and Di Matteo, on the persistence and the restructuring of
correlation networks).

If that matters here, it has a signature: the substitution rate should be LOWER
on days when the structure has just reorganised, because the fitted loadings are
then describing a structure the market has left behind.

WHAT IS AND IS NOT A TEST OF IT
-------------------------------
Bucketing days by a measure of instability and reporting R in each bucket is a
conditional reading, and this project has learned three things about those.

  1. The ratio has a denominator that moves with the state.  A rate that is
     lower in unstable markets may mean the cross-section does less work there,
     or it may mean the delay costs less there, which is a statement about
     arithmetic.  So N and D are reported separately, as in lab41.
  2. A cut is real-time only if the BOUNDARIES are real-time too.  Both cuts are
     run: trailing boundaries from each day's own past, and whole-window
     boundaries, which are not a rule a forecaster could have applied.
  3. The buckets must not be a relabelling of a cut already in the paper.  If
     instability is just stress under another name, this tests nothing new, so
     the overlap with the VIX tercile is measured and printed before any rate
     is read.

The forecasts and the per-day losses are the paper's own.  Only the bucket
changes, so nothing here is refitted and nothing can differ from Section 5
except the grouping.

THE MEASURE
-----------
At each test day, the correlation matrix of the eight decision variables over
the trailing year ending the day before, against the same matrix a quarter
earlier:

    S(t) = corr( offdiag C[t-252, t-1] , offdiag C[t-315, t-64] )

One number per day, in [-1, 1], high when the structure the model was fitted on
is the structure in force.  Twenty-eight off-diagonal entries, both windows
strictly in the past, so the measure is available on the morning of day t.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab08_implied_vol as IV
import lab41_conditional_anatomy as CA
from lab21_stronger_inference import fieller

SEED = 20261015
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 5, 13, 21, 55]
WIN = 252                 # the trailing year the structure is read over
GAP = 63                  # a quarter between the two windows being compared
N_BOOT = 2000
CRISIS = ("2008-06-01", "2009-06-30")


def stability(D, idx):
    """S(t) for every row of D, by the definition in the docstring."""
    X = D.values
    n, k = X.shape
    iu = np.triu_indices(k, 1)
    out = np.full(n, np.nan)
    # rolling correlation of the whole panel, recomputed each day from the
    # trailing window; k is eight, so this is cheap enough to do directly
    def corr_at(end):
        lo = end - WIN
        if lo < 0:
            return None
        W = X[lo:end]
        W = W[np.isfinite(W).all(axis=1)]
        if len(W) < WIN // 2:
            return None
        C = np.corrcoef(W, rowvar=False)
        return C[iu]
    cache = {}
    for t in range(n):
        for e in (t, t - GAP):
            if e not in cache:
                cache[e] = corr_at(e)
        a, b = cache[t], cache[t - GAP]
        if a is None or b is None or not np.isfinite(a).all() \
                or not np.isfinite(b).all():
            continue
        out[t] = float(np.corrcoef(a, b)[0, 1])
    return out


def rate(num, den):
    return num.mean() / den.mean() if abs(den.mean()) > 1e-12 else np.nan


def boot_diff(num_a, den_a, num_b, den_b, rng, block):
    """Block bootstrap of R(a) - R(b) on matched days, resampled together."""
    na, nb = len(num_a), len(num_b)
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sa = _blocks(na, rng, block)
        sb = _blocks(nb, rng, block)
        out[i] = rate(num_a[sa], den_a[sa]) - rate(num_b[sb], den_b[sb])
    out = out[np.isfinite(out)]
    return np.percentile(out, [2.5, 97.5]) if len(out) else (np.nan, np.nan)


def _blocks(n, rng, block):
    nb = int(np.ceil(n / block))
    st = rng.integers(0, max(1, n - block + 1), size=nb)
    return (st[:, None] + np.arange(block)).ravel()[:n] % n


def main(folder=None):
    folder = L.find_folder(folder)
    rng = np.random.default_rng(SEED)
    print(f"data folder: {folder}")
    print("Prints the rate by the stability of the correlation structure.\n")

    own, P, y, idx, D, k = CA.panel(folder)
    yb = y[idx]
    print(f"target {TARGET}, {len(idx)} test days, {k} foreign peers")
    print(f"structure read over {WIN} trailing days against the same window "
          f"{GAP} days earlier\n")

    E = {}
    for d in DELAYS:
        E[(d, "own")] = (yb - CA.walk(own, P, y, idx, d, False)) ** 2
        E[(d, "cross")] = (yb - CA.walk(own, P, y, idx, d, True)) ** 2

    S_all = stability(D, idx)
    s = S_all[idx]
    fin = np.isfinite(s)

    # ---------------- A ----------------------------------------------------
    print("=" * 96)
    print("A.  THE MEASURE, AND WHETHER IT IS STRESS UNDER ANOTHER NAME")
    print("=" * 96)
    print(f"  days with a stability reading: {int(fin.sum())} of {len(idx)}")
    print(f"  S runs {np.nanmin(s):+.3f} to {np.nanmax(s):+.3f}, "
          f"median {np.nanmedian(s):+.3f}")
    import contextlib, io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        vix = IV.load_iv(folder, "VIX").reindex(D.index).ffill(limit=5).shift(1)
    v = vix.values[idx]
    both = fin & np.isfinite(v)
    rho = float(pd.Series(s[both]).corr(pd.Series(v[both]), method="spearman"))
    print(f"  rank correlation between S and the VIX level at t-1: {rho:+.3f}")
    q1, q2 = np.nanpercentile(s[fin], [33.3, 66.7])
    vq1, vq2 = np.nanpercentile(v[both], [33.3, 66.7])
    s_low = both & (s <= q1)
    v_high = both & (v > vq2)
    overlap = float((s_low & v_high).sum()) / max(1, int(s_low.sum()))
    print(f"  of the least stable third of days, {overlap:.0%} are also in the "
          f"most stressed third")
    print("  If that share were near one the two cuts would be the same cut and")
    print("  nothing below would be new.\n")

    # ---------------- B ----------------------------------------------------
    print("=" * 96)
    print("B.  THE RATE BY STABILITY, REAL-TIME CUT")
    print("=" * 96)
    print("Boundaries from each day's own trailing year, so the bucket is one a")
    print("forecaster could have assigned on the morning of the day.\n")
    ss = pd.Series(S_all).shift(1).rolling(WIN, min_periods=60)
    rq1 = ss.quantile(1.0 / 3.0).values[idx]
    rq2 = ss.quantile(2.0 / 3.0).values[idx]
    rt = fin & np.isfinite(rq1) & np.isfinite(rq2)
    buckets_rt = {"unstable": rt & (s <= rq1),
                  "middle":   rt & (s > rq1) & (s <= rq2),
                  "stable":   rt & (s > rq2)}
    _table(E, buckets_rt)

    # ---------------- C ----------------------------------------------------
    print("=" * 96)
    print("C.  THE SAME CUT WITH WHOLE-WINDOW BOUNDARIES")
    print("=" * 96)
    print("Not a rule anyone could have applied on the day; reported because a")
    print("fixed boundary keeps 'unstable' meaning one thing across the sample.\n")
    buckets_fw = {"unstable": fin & (s <= q1),
                  "middle":   fin & (s > q1) & (s <= q2),
                  "stable":   fin & (s > q2)}
    _table(E, buckets_fw)

    # ---------------- D ----------------------------------------------------
    print("=" * 96)
    print("D.  THE CRISIS THE LIMITATION NAMES")
    print("=" * 96)
    dates = pd.to_datetime(D.index[idx])
    cz = (dates >= CRISIS[0]) & (dates <= CRISIS[1])
    print(f"  {CRISIS[0]} to {CRISIS[1]}: {int(cz.sum())} test days")
    if cz.sum() and fin.sum():
        print(f"  mean stability inside that window {np.nanmean(s[cz]):+.3f}, "
              f"outside {np.nanmean(s[fin & ~cz]):+.3f}")
    _table(E, {"crisis": cz & fin, "rest": fin & ~cz})

    # ---------------- E ----------------------------------------------------
    print("=" * 96)
    print("E.  THE CONTRAST, AND WHAT IT LICENSES")
    print("=" * 96)
    print("Two marginal sets failing to overlap is not a test of a difference,")
    print("which is why the difference itself is resampled here, in blocks of")
    print(f"{L.block_for(max(DELAYS))} days with {N_BOOT} replications, both")
    print("buckets resampled inside the same replication.\n")
    print(f"{'delta':>6}{'R stable':>11}{'R unstable':>12}{'difference':>13}"
          f"{'95% block bootstrap':>24}")
    moved = []
    for d in DELAYS[1:]:
        a_i = np.where(buckets_rt["stable"])[0]
        b_i = np.where(buckets_rt["unstable"])[0]
        na = E[(d, "own")][a_i] - E[(d, "cross")][a_i]
        da = E[(d, "own")][a_i] - E[(0, "own")][a_i]
        nb = E[(d, "own")][b_i] - E[(d, "cross")][b_i]
        db = E[(d, "own")][b_i] - E[(0, "own")][b_i]
        ra, rb = rate(na, da), rate(nb, db)
        lo, hi = boot_diff(na, da, nb, db, rng, L.block_for(d))
        sig = "*" if (lo > 0 or hi < 0) else ""
        moved.append((d, ra - rb, lo, hi, bool(sig)))
        print(f"{d:>6}{ra:>11.0%}{rb:>12.0%}{ra - rb:>+13.0%}"
              + f"[{lo:>+7.0%},{hi:>+7.0%}]{sig}".rjust(24))
    nsig = sum(1 for _, _, _, _, g in moved if g)
    nneg = sum(1 for _, dd, _, _, _ in moved if dd < 0)
    print(f"\n  the stable-minus-unstable difference excludes zero at "
          f"{nsig} of {len(moved)} delays,")
    print(f"  and is negative at {nneg} of {len(moved)}.")
    print("  The limitation predicts a POSITIVE difference: the rate should be")
    print("  lower where the structure has just reorganised.")
    if nsig == 0:
        print("\n  No delay separates the two states on this measure.  That is a")
        print("  failure to detect, not a demonstration that stability does not")
        print("  matter: the unstable bucket is a third of the sample and the")
        print("  sets below are wide.")
    print()
    print("  Fieller sets for the two extreme buckets at the longest delay,")
    print("  which is where the rate is best identified:")
    for nm in ("stable", "unstable"):
        i = np.where(buckets_rt[nm])[0]
        d = DELAYS[-1]
        num = E[(d, "own")][i] - E[(d, "cross")][i]
        den = E[(d, "own")][i] - E[(0, "own")][i]
        kind, flo, fhi = fieller(num, den, lag=L.block_for(d))
        got = (f"[{flo:.0%}, {fhi:.0%}]" if kind == "interval"
               else f"two half-lines, outside [{flo:.0%}, {fhi:.0%}]"
               if kind == "exclusion" else "the whole line")
        print(f"    {nm:>9}  R = {rate(num, den):>5.0%}   {kind:>10}   {got}")
    print()


def _table(E, buckets):
    print(f"{'bucket':>10}{'delta':>7}{'N (removed)':>14}{'D (delay cost)':>16}"
          f"{'R = N/D':>10}{'days':>7}")
    for nm, m in buckets.items():
        i = np.where(m)[0]
        if len(i) < 50:
            print(f"{nm:>10}  too few days to read ({len(i)})")
            continue
        for d in DELAYS[1:]:
            num = E[(d, "own")][i] - E[(d, "cross")][i]
            den = E[(d, "own")][i] - E[(0, "own")][i]
            print(f"{nm:>10}{d:>7}{num.mean():>14.4f}{den.mean():>16.4f}"
                  f"{rate(num, den):>10.0%}{len(i):>7}")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
