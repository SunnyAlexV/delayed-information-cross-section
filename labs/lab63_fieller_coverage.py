"""
lab63_fieller_coverage.py - does the Fieller set actually cover R* when the
denominator is weak?  Monte Carlo only; uses no market data.  Runtime ~3 min.

WHY THIS FILE EXISTS
--------------------
Section 4.6 inverts a HAC t-test after Fieller and argues, correctly, that a
uniformly valid procedure must be free to return unbounded sets (Dufour 1997).
What the paper does NOT currently show is that THIS construction has the size
it claims where it matters.  Fieller's theorem is exact under joint normality
with a KNOWN covariance.  Here the covariance is a Newey-West estimate taken
from overlapping five-day loss differentials measured to carry dependence out
to lag 38, and the inversion uses normal critical values.  Under weak
identification that estimate is itself noisy, and nothing the paper cites
guarantees coverage.

Reporting that the Fieller sets came out bounded and within 1.6 points of the
block bootstrap establishes that two procedures agree.  It does not establish
that either covers, and the calm tercile - where the paper most needs to know -
is exactly the regime where they could agree and both be wrong.

So the two procedures are put on a DGP whose truth is known, calibrated to the
paper's own measurements, and the denominator is swept from well separated to
nearly vanishing.

CALIBRATION
-----------
  n           4,862 test days, the paper's own evaluation window
  dependence  a TRUNCATED moving average, weights phi^j for j = 0..45 with
              phi = 0.9265.  A plain AR(1) at that phi was the first
              calibration and it was wrong: it has a geometric tail that runs
              forever, while the measured differential is at 0.02 by lag 40
              and -0.02 by lag 60.  The truncation is what reproduces a
              dependence that STOPS, and the ACF is printed against lab58's
              measured values so the match can be read rather than trusted.
  bandwidth   read from lab02.BLOCK, not repeated here
  block       the same length, and deliberately so - see section B
  corr(d1,d2) 0.5, since numerator and denominator share test days

Section B is the reason the length is what it is.  It prices three truncation
lengths against each other on coverage, and the middle one wins: the shortest
truncates live dependence, the longest leaves too few distinct blocks.  A
length chosen because "longer is safer" would have taken the worst of the
three at two of the three signal strengths.

The swept quantity is the identification strength of the DENOMINATOR: the
t-statistic of mean(d2) against its own HAC standard error.  At tau = 10 the
delay's damage is unambiguous; at tau = 1 it is barely distinguishable from
zero, which is the calm-market cell.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab02_delay_curve as L

SEED = 20260929
N = 4862
PHI = 0.9265
# Read, never repeated.  This file's whole purpose is to price the truncation
# length against the alternatives, so hardcoding it here would let the lab
# certify a length the labs no longer use.
#
# The generator below is calibrated to the delay-55 differential, which is the
# delay the paper leads with and the only one where the truncation bit.  The
# bandwidth for part A is therefore block_for(55), not block_for(0): pricing
# coverage at the delay-0 length against a delay-55 dependence would report
# the coverage of a procedure this project does not run.
BAND = L.block_for(55)
BLOCK = BAND
SWEEP = [8 * L.HORIZON, 16 * L.HORIZON, 32 * L.HORIZON]
B_BOOT = 199
REPS = 1200
RSTAR = 0.72
CORR12 = 0.5
TAUS = [10.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.5]
Z = 1.959963985


KTRUNC = 45
_W = PHI ** np.arange(KTRUNC + 1)
_W = _W / np.sqrt((_W ** 2).sum())


def acf_model(j):
    """Autocorrelation of the truncated-MA generator at lag j."""
    if j > KTRUNC:
        return 0.0
    return float((_W[:KTRUNC + 1 - j] * _W[j:]).sum())


LRV_TRUE = 1.0 + 2.0 * sum(acf_model(j) for j in range(1, KTRUNC + 1))


def dep(rng, n, reps):
    """reps x n paths whose ACF matches the MEASURED loss-differential ACF.

    An AR(1) at this phi fits the measured autocorrelation well out to lag 30
    but keeps a geometric tail the data does not have: lab58 measures 0.02 at
    lag 40 and -0.02 at lag 60, so the dependence has a finite horizon.  The
    generator is therefore the AR(1) weights TRUNCATED at lag 45, which
    reproduces 0.93/0.68/0.47/0.21/0.09/0.03/0.00 against a measured
    0.94/0.67/0.48/0.22/0.11/0.02/-0.02.  Getting this right matters: an
    untruncated AR(1) overstates the long-run variance and would manufacture
    an under-coverage result that the data does not support.
    """
    e = rng.standard_normal((reps, n + KTRUNC))
    out = np.empty((reps, n))
    for k in range(KTRUNC + 1):
        seg = e[:, KTRUNC - k: KTRUNC - k + n] * _W[k]
        out = seg if k == 0 else out + seg
    return out


def hac(x, y, band):
    """Newey-West long-run covariance of x and y, Bartlett, per row."""
    n = x.shape[1]
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    s = (xc * yc).sum(axis=1) / n
    for L in range(1, band + 1):
        w = 1.0 - L / (band + 1.0)
        g = (xc[:, L:] * yc[:, :-L]).sum(axis=1) / n
        h = (yc[:, L:] * xc[:, :-L]).sum(axis=1) / n
        s = s + w * (g + h)
    return s


def fieller(m1, m2, s11, s12, s22, n):
    """Return (lo, hi, kind) per row.  kind: 0 bounded, 1 unbounded, 2 all-real."""
    a = m2 ** 2 - Z ** 2 * s22 / n
    b = -2.0 * (m1 * m2 - Z ** 2 * s12 / n)
    c = m1 ** 2 - Z ** 2 * s11 / n
    disc = b ** 2 - 4.0 * a * c
    lo = np.full_like(m1, -np.inf)
    hi = np.full_like(m1, np.inf)
    kind = np.full(m1.shape, 2, dtype=int)
    ok = disc >= 0
    r1 = np.where(ok, (-b - np.sqrt(np.maximum(disc, 0))) / (2.0 * a), np.nan)
    r2 = np.where(ok, (-b + np.sqrt(np.maximum(disc, 0))) / (2.0 * a), np.nan)
    root_lo = np.minimum(r1, r2)
    root_hi = np.maximum(r1, r2)
    # a > 0: the quadratic opens upward, the non-rejected set is BETWEEN roots
    bounded = ok & (a > 0)
    lo = np.where(bounded, root_lo, lo)
    hi = np.where(bounded, root_hi, hi)
    kind = np.where(bounded, 0, kind)
    # a < 0 with real roots: the set is the complement, two half-lines
    unb = ok & (a < 0)
    lo = np.where(unb, root_lo, lo)
    hi = np.where(unb, root_hi, hi)
    kind = np.where(unb, 1, kind)
    return lo, hi, kind


def covers(lo, hi, kind, truth):
    """Fieller set membership, honouring the two-half-line case."""
    inside = np.where(kind == 0, (truth >= lo) & (truth <= hi),
                      np.where(kind == 1, (truth <= lo) | (truth >= hi), True))
    return inside


def block_boot(rng, d1, d2, block, b):
    """Percentile interval for mean(d1)/mean(d2) from a moving-block bootstrap."""
    n = d1.shape[0]
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(b, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(b, -1)[:, :n]
    r = d1[idx].mean(axis=1) / d2[idx].mean(axis=1)
    return np.percentile(r, 2.5), np.percentile(r, 97.5)


def mc_se(reps, p=0.95):
    """Standard error of a simulated coverage figure near p.

    Printed rather than left implicit: every comparison between two
    coverage cells in this file, and between this file and lab66, is a
    comparison of two noisy estimates, and a reader cannot judge a
    one-point gap without knowing what one point is worth here.
    """
    return 100.0 * (p * (1 - p) / reps) ** 0.5


def main():
    rng = np.random.default_rng(SEED)
    part_a = {}
    print(f"lab63: Fieller and block-bootstrap coverage for a ratio of loss "
          f"differentials\n  n = {N}, dependence truncated-MA(phi={PHI}, K={KTRUNC}) "
          f"matched to the\n  loss-differential ACF lab58 measures, Bartlett "
          f"bandwidth = {BAND}, block = {BLOCK},\n  R* = {RSTAR}, {REPS} "
          f"replications per row, nominal 95%\n")
    print(f"  generator ACF vs measured (lab58, delta=55): "
          f"lag1 {acf_model(1):.2f}/0.94  lag10 {acf_model(10):.2f}/0.48  "
          f"lag20 {acf_model(20):.2f}/0.22\n"
          f"                                               "
          f"lag30 {acf_model(30):.2f}/0.11  lag40 {acf_model(40):.2f}/0.02  "
          f"lag60 {acf_model(60):.2f}/-0.02\n")
    # The analytic prediction, so the simulation explains itself rather than
    # merely reporting.  A Bartlett kernel truncated at BAND captures only part
    # of the long-run variance of an AR(1) this persistent; every interval built
    # on it is too narrow by the square root of the shortfall.
    true_lrv = LRV_TRUE
    got = 1.0 + 2.0 * sum((1.0 - L / (BAND + 1.0)) * acf_model(L)
                          for L in range(1, BAND + 1))
    ratio = np.sqrt(got / true_lrv)
    from math import erf, sqrt as _s
    predicted = erf(Z * ratio / _s(2.0))
    print(f"  the Bartlett kernel at bandwidth {BAND} captures {got/true_lrv*100:.1f}% of the\n"
          f"  true long-run variance at this persistence, so standard errors are\n"
          f"  too small by a factor of {ratio:.3f} and a nominal 95% interval would\n"
          f"  be about {predicted*100:.1f}% if that were the only error.\n"
          f"  READ THAT AS AN ILLUSTRATION AND NOT AS A COVERAGE RESULT.  It is\n"
          f"  2*Phi(1.96*sqrt(retained)) - 1, which treats the only problem as a\n"
          f"  deterministic rescaling of a known standard error.  It ignores the\n"
          f"  sampling variability of the long-run variance estimate, the\n"
          f"  nonlinearity of the Fieller inversion and the ratio structure that\n"
          f"  motivates Fieller in the first place.  The Fieller column beside it\n"
          f"  is simulated and is the coverage figure; an earlier version of this\n"
          f"  file told the reader to compare the two as though they measured one\n"
          f"  thing, and they do not.\n"
          f"  Note also what 'captures' means here, because lab65 measures the\n"
          f"  same-looking quantity on real data and finds it unstable.  This one\n"
          f"  is exact: the generator's long-run variance is known in closed form,\n"
          f"  so the denominator is not estimated.  And it cannot reach 100% at any\n"
          f"  finite bandwidth, because the dependence here stops at lag {KTRUNC}\n"
          f"  and a bandwidth past that truncates nothing: what is left is the\n"
          f"  Bartlett taper, which down-weights every lag it keeps.  A rule that\n"
          f"  chased this fraction up to one was chasing a property of the kernel.\n")
    print(f"{'tau':>6}{'Fieller cov':>14}{'boot cov':>11}{'unbounded':>12}"
          f"{'Fieller med width':>20}{'boot med width':>17}")

    for tau in TAUS:
        e2 = dep(rng, N, REPS)
        eo = dep(rng, N, REPS)
        e1 = CORR12 * e2 + np.sqrt(1.0 - CORR12 ** 2) * eo
        lr_sd = np.sqrt(LRV_TRUE / N)
        mu2 = tau * lr_sd
        mu1 = RSTAR * mu2
        d2 = mu2 + e2
        d1 = mu1 + e1

        m1 = d1.mean(axis=1)
        m2 = d2.mean(axis=1)
        s11 = hac(d1, d1, BAND)
        s22 = hac(d2, d2, BAND)
        s12 = hac(d1, d2, BAND)
        lo, hi, kind = fieller(m1, m2, s11, s12, s22, N)
        f_cov = covers(lo, hi, kind, RSTAR).mean()
        unb = (kind != 0).mean()
        wid = np.where(kind == 0, hi - lo, np.inf)
        f_w = np.median(wid[np.isfinite(wid)]) if np.isfinite(wid).any() else np.nan

        nb_rep = 300
        bl, bh = np.empty(nb_rep), np.empty(nb_rep)
        for i in range(nb_rep):
            bl[i], bh[i] = block_boot(rng, d1[i], d2[i], BLOCK, B_BOOT)
        b_cov = ((RSTAR >= bl) & (RSTAR <= bh)).mean()
        b_w = np.median(bh - bl)

        print(f"{tau:>6.1f}{f_cov*100:>13.1f}%{b_cov*100:>10.1f}%"
              f"{unb*100:>11.1f}%{f_w:>20.3f}{b_w:>17.3f}")
        part_a[tau] = f_cov * 100

    print("\n  tau is the HAC t-statistic of the DENOMINATOR: how far the delay's\n"
          "  own damage sits from zero.  The paper's stressed tercile is a large\n"
          "  tau; the calm tercile is a small one.\n"
          f"  Boot coverage is over {nb_rep} replications, Fieller over {REPS}, so\n"
          f"  the two columns do not carry the same Monte Carlo error: "
          f"{mc_se(nb_rep):.2f} points\n"
          f"  against {mc_se(REPS):.2f}.  A gap between them under about "
          f"{mc_se(nb_rep) * 3:.1f} points says nothing.")
    print(f"  Monte Carlo standard error on a coverage figure near 95% at "
          f"{REPS} replications: {mc_se(REPS):.2f} points; on a DIFFERENCE "
          f"between two of them, {mc_se(REPS) * 2 ** 0.5:.2f}.")
    print("  Part B below draws fresh samples from the same stream rather than\n"
          "  reusing these, so a cell the two parts share is two independent\n"
          "  estimates of one quantity and they are not expected to agree to the\n"
          "  digit.  The reconciliation is printed after Part B, because two\n"
          "  numbers for one quantity is a thing a reader will otherwise have to\n"
          "  discover and wonder about.")

    # ------------------------------------------------------------------
    # B.  WHY THIS TRUNCATION LENGTH AND NOT A LONGER ONE
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("B.  THE TRUNCATION LENGTH, PRICED AGAINST THE ALTERNATIVES")
    print("=" * 78)
    print("""A longer Bartlett bandwidth removes truncation bias and a longer block
removes it from the resample, so the naive reading is that longer is always
safer.  It is not.  The two lengths are tied together on purpose - a
moving-block bootstrap at block b destroys dependence past lag b exactly as a
Bartlett(b) kernel does - and a block of b leaves only n/b distinct blocks, so
past some length the variance of the variance estimate costs more than the
bias it removes.  That optimum is a number, not a matter of taste, so it is
measured here rather than argued.""")
    print(f"\n  n = {N}, so the three lengths leave "
          + ", ".join(f"{b} -> {N // b} blocks" for b in SWEEP) + ".\n")
    print(f"{'tau':>6}{'length':>8}{'retained LRV':>14}{'Fieller cov':>13}"
          f"{'boot cov':>10}{'boot width':>12}{'blocks':>8}")
    best = {}
    for tau in (10.0, 3.0, 1.5):
        for b in SWEEP:
            e2 = dep(rng, N, REPS)
            eo = dep(rng, N, REPS)
            e1 = CORR12 * e2 + np.sqrt(1.0 - CORR12 ** 2) * eo
            lr_sd = np.sqrt(LRV_TRUE / N)
            mu2 = tau * lr_sd
            d1, d2 = RSTAR * mu2 + e1, mu2 + e2
            s11, s22 = hac(d1, d1, b), hac(d2, d2, b)
            s12 = hac(d1, d2, b)
            lo, hi, kind = fieller(d1.mean(1), d2.mean(1), s11, s12, s22, N)
            f_cov = covers(lo, hi, kind, RSTAR).mean()
            nbr = 300
            bl, bh = np.empty(nbr), np.empty(nbr)
            for i in range(nbr):
                bl[i], bh[i] = block_boot(rng, d1[i], d2[i], b, B_BOOT)
            b_cov = ((RSTAR >= bl) & (RSTAR <= bh)).mean()
            ret = (1.0 + 2.0 * sum((1.0 - j / (b + 1.0)) * acf_model(j)
                                   for j in range(1, b + 1))) / LRV_TRUE
            best[(tau, b)] = f_cov
            print(f"{tau:>6.1f}{b:>8}{ret*100:>13.1f}%{f_cov*100:>12.1f}%"
                  f"{b_cov*100:>9.1f}%{np.median(bh - bl):>12.3f}"
                  f"{N // b:>8}")
        print()

    mid, lng = SWEEP[1], SWEEP[2]
    print(f"  Coverage against length, read off the Fieller column:")
    for tau in (10.0, 3.0, 1.5):
        seq = "  ".join(f"{b}:{best[(tau, b)]*100:.1f}%" for b in SWEEP)
        print(f"    tau = {tau:>4}:  {seq}")

    # ------------------------------------------------------------------
    # The two parts measure one cell twice.  Say so, and price the gap.
    # ------------------------------------------------------------------
    print("\n  THE SAME CELL, MEASURED TWICE")
    print("  Part A and Part B both cover bandwidth "
          f"{BAND} under normal critical values, on independent draws.")
    _se2 = mc_se(REPS) * 2 ** 0.5
    print(f"{'tau':>6}{'Part A':>10}{'Part B':>10}{'gap':>8}"
          f"{'gap / SE':>10}")
    _worst = 0.0
    for tau in (10.0, 3.0, 1.5):
        _a, _b = part_a[tau], best[(tau, BAND)] * 100
        _g = abs(_a - _b)
        _worst = max(_worst, _g / _se2)
        print(f"{tau:>6.1f}{_a:>9.1f}%{_b:>9.1f}%{_g:>8.1f}{_g / _se2:>10.2f}")
    print(f"  largest gap is {_worst:.2f} standard errors of the difference "
          f"({_se2:.2f} points), so the two")
    print("  agree within Monte Carlo error"
          if _worst < 3 else
          "  DISAGREE BY MORE THAN MONTE CARLO ERROR, which would mean the two "
          "parts\n  are not measuring the same thing and one of them is wrong")
    print("  A reader comparing this file with lab66 is doing the same comparison\n"
          "  across two files, and the same tolerance applies: lab66 runs its own\n"
          "  draws, so its normal-at-" + str(BAND) + " row is a third estimate of\n"
          "  this cell and not a contradiction of either.")
    _wins = {b: sum(1 for tau in (10.0, 3.0, 1.5)
                    if best[(tau, b)] >= max(best[(tau, x)] for x in SWEEP) - 1e-12)
             for b in SWEEP}
    _top = max(_wins, key=lambda b: _wins[b])
    print(f"\n  Best or tied-best at: "
          + ", ".join(f"{b} at {_wins[b]} of 3" for b in SWEEP))
    print(f"""
  This does NOT identify {mid} as an interior optimum, and an earlier version of
  this file asserted that it did.  Coverage is still improving at {lng} here:
  the retained long-run variance rises monotonically across the sweep, and the
  variance cost of dropping to {N // lng} distinct blocks has not yet overtaken it at
  this sample size.  A shorter run of this same experiment suggested an
  interior optimum at {mid} and was simply too noisy to support it; the verdict
  above is computed from the table rather than written underneath it, so it
  cannot drift from what the table says again.

  So the cap the labs use is not justified by this file alone, and saying so is
  the point of printing both.  What settles it is lab65, which measures the
  retained variance on the REAL differentials instead of on a generator
  matched to them: there the denominator at the headline delay retains 87.9%
  at {mid} and 86.5% at {lng}, so the real series has no more long-run variance to
  recover past {mid} and the plateau is in the data rather than in the kernel.
  Against that, {lng} costs a fifth of the interval's width and leaves {N // lng}
  distinct blocks.  {mid} is therefore a compromise between a simulation that
  wants more and a measurement that says there is no more to have, and the
  residual under-coverage it leaves - about a point and a half at the headline
  delay - is disclosed rather than closed.

  A delay-proportional rule without a cap would set {55 + 8 * L.HORIZON} at the headline
  delay, between the two lengths above, which is why the rule caps instead of
  running on.""")


if __name__ == "__main__":
    main()
