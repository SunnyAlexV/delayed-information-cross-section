"""
lab06_inference.py - the formal tests the paper was missing.

Imports lab05_robustness, so keep both files in the same folder.
Runtime ~6 minutes.

WHY
---
Bootstrap confidence intervals on a loss difference are a DESCRIPTION, not a
test of the null that the cross-section adds nothing.  The forecasting
literature expects a formal test, and our own-only model is NESTED inside the
cross model, which rules out the naive choice.

  Giacomini-White (2006)   primary.  Valid for NESTED models, and valid under
                           ANY loss, so the same test covers R-squared and
                           QLIKE.  Requires a FIXED ROLLING estimation window,
                           which is what lab05 already uses.
  Clark-West (2007)        secondary.  Purpose-built for nested models, but
                           MSPE-only, so it cannot speak to QLIKE.
  Diebold-Mariano          reported for familiarity.  Diebold (2015) warns it
                           was built to compare FORECASTS, not MODELS, and
                           degenerates under a nested null.  Read it as a
                           reference point, not as evidence.

Three further things a referee would ask for, all here:
  * SIMULTANEOUS bands across the delta grid, not pointwise ones.  Ten delays
    with pointwise 95% intervals will breach somewhere by chance.
  * A PLACEBO: re-run with the foreign block circularly shifted by years, so
    it keeps its own distribution but loses its alignment.  The substitution
    rate should collapse to zero.
  * A TEST of the convergence claim, instead of asserting it from a chart.
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".")
import lab05_robustness as L

DELAYS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]
N_BOOT, SEED = 3000, 20260912


def hac_se(d, lag=None):
    """Newey-West standard error of a mean, at lab05's measured bandwidth.

    The default is L.HAC_LAG, not the automatic 4 (n/100)^(2/9) rule.  That
    rule gives 9 lags here and truncates the kernel while the loss
    differential still carries 0.52 autocorrelation, which understates every
    standard error in this project by up to 54% and every statistic formed
    from one in the paper's favour.  lab58 part 6 measures it.
    """
    n = len(d); d = d - d.mean()
    if lag is None:
        lag = L.HAC_LAG
    g0 = (d @ d) / n
    s = g0
    for k in range(1, lag + 1):
        gk = (d[k:] @ d[:-k]) / n
        s += 2 * (1 - k / (lag + 1)) * gk
    return np.sqrt(max(s, 1e-18) / n)


def norm_p(z):
    from math import erf, sqrt
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def forecasts(folder, target="SPX", continuous=True):
    """Walk-forward predictions from both models at every delay. One pass."""
    D, peers, lag, idx, yb, fitrun, a = L.run_target(folder, target, DELAYS,
                                                     np.random.default_rng(SEED),
                                                     continuous=continuous)
    out = {}
    for d in DELAYS:
        out[d] = (fitrun(d, False), fitrun(d, True))
    # The benchmark every skill in this file is scored against.  run_target
    # hands back the decision variable rather than the label series, so the
    # labels are rebuilt here exactly as it builds them, and the trailing-mean
    # benchmark is taken on the same days.  Until an audit of every scorer in
    # the package, this file divided by the mean of the target over the test
    # period at four separate places, including inside the bootstrap.
    n_all = len(D)
    y_full = np.full(n_all, np.nan)
    y_full[:-L.HORIZON] = a[L.HORIZON:]
    bench = L.bench_mean(y_full, idx)
    return yb, out, D, peers, idx, bench


def main(folder=None):
    rng = np.random.default_rng(SEED)
    folder = L.find_folder(folder)
    print(f"data folder: {folder}\nfitting both models at {len(DELAYS)} delays...\n")
    y, F, D, peers, idx, BENCH = forecasts(folder)
    n = len(y)
    sst = ((y - BENCH) ** 2).sum()

    # ---------------- 1. formal tests -----------------------------------
    print("=" * 82)
    print("1. FORMAL TESTS OF EQUAL PREDICTIVE ABILITY (continuous target, SPX)")
    print("=" * 82)
    print("H0: the foreign cross-section adds nothing.  Positive statistic favours cross.")
    print(f"{'delta':>6}{'GW (sq loss)':>16}{'GW (QLIKE)':>16}{'Clark-West':>16}{'DM (ref only)':>16}")
    tests = {}
    for d in DELAYS:
        po, pc = F[d]
        # squared-error loss differential
        d_sq = (y - po) ** 2 - (y - pc) ** 2
        # QLIKE on the variance scale
        yv, fo, fc = np.exp(y), np.exp(po), np.exp(pc)
        q = lambda f: yv / f - np.log(yv / f) - 1
        d_ql = q(fo) - q(fc)
        # Clark-West adjustment for nesting
        cw = (y - po) ** 2 - ((y - pc) ** 2 - (po - pc) ** 2)
        z_sq = d_sq.mean() / hac_se(d_sq)
        z_ql = d_ql.mean() / hac_se(d_ql)
        z_cw = cw.mean() / hac_se(cw)
        z_dm = d_sq.mean() / (d_sq.std(ddof=1) / np.sqrt(n))
        tests[d] = (z_sq, z_ql, z_cw, d_sq, d_ql)
        f = lambda z: f"{z:>8.2f} ({norm_p(z):.3f})"
        print(f"{d:>6}{f(z_sq):>16}{f(z_ql):>16}{f(z_cw):>16}{f(z_dm):>16}")
    print("\nGW and Clark-West are HAC-adjusted; p-values in brackets.")
    print("Diebold-Mariano uses the naive standard error and is shown only because")
    print("readers expect it; it is not valid for nested models.")

    # ---------------- 2. simultaneous bands ------------------------------
    print("\n" + "=" * 82)
    print("2. SIMULTANEOUS BANDS ACROSS THE DELTA GRID")
    print("=" * 82)
    print("Ten pointwise 95% intervals will breach somewhere by chance.  We bootstrap")
    print("the whole delta-vector jointly and take the sup-t critical value.\n")
    # The block was hardcoded at 10 here and did not follow lab05 when that was
    # corrected.  A simultaneous band is a statement about the joint
    # distribution of the whole delta-vector, so it is at least as sensitive to
    # under-blocking as any single interval, and it was the last place in the
    # package still resampling at 2h.
    blocks = L.BLOCK
    nb = int(np.ceil(n / blocks)); offs = np.arange(blocks)
    st = rng.integers(0, n - blocks + 1, size=(N_BOOT, nb))
    Tmax = np.empty(N_BOOT)
    boot = {d: np.empty(N_BOOT) for d in DELAYS}
    # One HAC standard error per delay, on the loss differential itself, fixed
    # before the resampling starts so every draw is studentised by the same
    # quantity the printed band uses.
    se_hac = {d: hac_se(tests[d][3]) for d in DELAYS}
    for i in range(N_BOOT):
        s = (st[i][:, None] + offs).ravel()[:n]
        ts = []
        for d in DELAYS:
            v = tests[d][3][s]
            m = v.mean()
            boot[d][i] = m
            # Studentise by the SAME standard error the band will use, which is
            # the HAC one on the original series.  An earlier version divided a
            # block-bootstrap deviation by an iid standard error computed on the
            # resampled days: the numerator carried the dependence the block was
            # there to preserve and the denominator assumed it away, so the
            # ratio was inflated and the critical value with it.  The error was
            # small while the block was 10 and large once the block was widened
            # to the measured 40, which is how the Bonferroni comparison below
            # found it - the bootstrap's critical value had risen ABOVE the
            # correction that ignores dependence entirely, which cannot be right
            # for ten overlapping windows of one series.
            ts.append(abs(m - tests[d][3].mean()) / max(se_hac[d], 1e-12))
        Tmax[i] = max(ts)
    crit = np.percentile(Tmax, 95)
    # Reporting the critical value alone hides everything that produced it, so
    # the sup-t statistic's own distribution is printed, and beside it the
    # Bonferroni value a reader would otherwise reach for.  Bonferroni is the
    # conservative comparison here: it ignores the dependence across delays,
    # and these ten columns are built from overlapping windows of the same
    # series, so it should sit ABOVE the bootstrap's value.  If it ever sits
    # below, the bootstrap is under-covering and the band is not a band.
    from math import erf, sqrt as _sqrt
    def _inv_norm(pp):                       # two-sided critical value, bisection
        lo_, hi_ = 0.0, 10.0
        for _ in range(200):
            mid = 0.5 * (lo_ + hi_)
            if 2 * (1 - 0.5 * (1 + erf(mid / _sqrt(2)))) > pp:
                lo_ = mid
            else:
                hi_ = mid
        return 0.5 * (lo_ + hi_)
    bonf = _inv_norm(0.05 / len(DELAYS))
    q50, q90, q99 = np.percentile(Tmax, [50, 90, 99])
    print(f"sup-t critical value {crit:.2f}  (pointwise 95% would use 1.96)")
    print(f"  bootstrap block {blocks} days, {N_BOOT} draws, {len(DELAYS)} delays")
    print(f"  distribution of the sup-t statistic: median {q50:.2f}, "
          f"90th {q90:.2f}, 95th {crit:.2f}, 99th {q99:.2f}")
    print(f"  Bonferroni would use {bonf:.2f} for {len(DELAYS)} delays at 5%")
    print(f"  the bootstrap value is {'below' if crit < bonf else 'ABOVE'} it, which is "
          f"{'expected' if crit < bonf else 'NOT expected'}: the delays share")
    print("  overlapping windows of one series, so treating them as independent")
    print("  overstates the correction.\n")
    print(f"{'delta':>6}{'dR2':>10}{'SE':>9}{'pointwise 95%':>22}"
          f"{'simultaneous 95%':>24}{'Bonferroni 95%':>24}")
    for d in DELAYS:
        mean = tests[d][3].mean() * n / sst
        # HAC, not iid: the five-day targets overlap, so the naive standard
        # error understates the dispersion of every column in this table.
        se = se_hac[d] * n / sst
        lo_p, hi_p = np.percentile(boot[d] * n / sst, [2.5, 97.5])
        lo_s, hi_s = mean - crit * se, mean + crit * se
        lo_b, hi_b = mean - bonf * se, mean + bonf * se
        mark = "*" if lo_s > 0 else " "
        print(f"{d:>6}{mean:>+10.4f}{se:>9.4f}  [{lo_p:+.4f},{hi_p:+.4f}]   "
              f"[{lo_s:+.4f},{hi_s:+.4f}]{mark}  [{lo_b:+.4f},{hi_b:+.4f}]")
    print("\n* = still excludes zero under the simultaneous band.")
    print("SE is the per-delay standard error of the dR2, on the same scale as the")
    print("column beside it; the pointwise interval is the bootstrap's own")
    print("percentiles and is not symmetric about the estimate, while the two")
    print("t-based bands are.")

    # ---------------- 3. placebo ----------------------------------------
    print("\n" + "=" * 82)
    print("3. PLACEBO: foreign block circularly shifted by five years")
    print("=" * 82)
    print("Same series, same distribution, same autocorrelation - alignment destroyed.")
    print("If the result is real the substitution rate must collapse.\n")
    shift = 1260
    Dp = D.copy()
    Dp[peers] = np.roll(Dp[peers].values, shift, axis=0)
    orig = L.pd.DataFrame
    # monkey-patch the panel builder for one run
    real_build = L.build
    def fake_build(folder, target):
        _, pe, lg = real_build(folder, target)
        return Dp, pe, lg
    L.build = fake_build
    try:
        yp, Fp, _, _, _, bp = forecasts(folder)
        sstp = ((yp - bp) ** 2).sum()
        print(f"{'delta':>6}{'R2 own':>10}{'R2 cross':>11}{'dR2':>10}{'rate':>9}")
        r0 = 1 - ((yp - Fp[0][0]) ** 2).sum() / sstp
        for d in [5, 21, 55]:
            po, pc = Fp[d]
            ro = 1 - ((yp - po) ** 2).sum() / sstp
            rc = 1 - ((yp - pc) ** 2).sum() / sstp
            rate = (rc - ro) / (r0 - ro) if r0 > ro else float("nan")
            print(f"{d:>6}{ro:>10.4f}{rc:>11.4f}{rc-ro:>+10.4f}{rate:>8.1%}")
    finally:
        L.build = real_build

    # ---------------- 4. does the rate converge? -------------------------
    print("\n" + "=" * 82)
    print("4. TESTING THE CONVERGENCE CLAIM")
    print("=" * 82)
    print("The paper says the substitution rate flattens. That is a hypothesis,")
    print("not an observation. H0: R(55) = R(d) for earlier d.\n")
    def rate_vec(sample):
        ss0 = ((y[sample] - BENCH[sample]) ** 2).sum()
        r0 = 1 - ((y[sample] - F[0][0][sample]) ** 2).sum() / ss0
        out = {}
        for d in DELAYS:
            po, pc = F[d][0][sample], F[d][1][sample]
            ss = ss0
            ro = 1 - ((y[sample] - po) ** 2).sum() / ss
            rc = 1 - ((y[sample] - pc) ** 2).sum() / ss
            out[d] = (rc - ro) / (r0 - ro) if r0 > ro else np.nan
        return out
    base = rate_vec(np.arange(n))
    diffs = {d: np.empty(N_BOOT) for d in [13, 21, 34]}
    for i in range(N_BOOT):
        s = (st[i][:, None] + offs).ravel()[:n]
        rv = rate_vec(s)
        for d in diffs: diffs[d][i] = rv[55] - rv[d]
    print(f"{'comparison':>16}{'R(55) - R(d)':>15}{'95% interval':>22}  verdict")
    for d in [13, 21, 34]:
        lo, hi = np.percentile(diffs[d], [2.5, 97.5])
        v = "still rising" if lo > 0 else "flat - cannot reject equality"
        print(f"{'R(55) vs R('+str(d)+')':>16}{base[55]-base[d]:>+15.3f}"
              f"   [{lo:+.3f},{hi:+.3f}]  {v}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
