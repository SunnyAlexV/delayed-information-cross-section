"""
lab65_hac_coverage_real.py - what the paper's HAC bandwidth actually captures,
measured on the real loss differentials rather than on a generator.
Imports lab10; keep both in labs/.  Runtime ~5 min.

WHY THIS FILE EXISTS
--------------------
Section 4.6's intervals are studentised by a Bartlett kernel at a fixed
b = M/n = 0.05, which at the evaluation length of 4,862 days is a bandwidth of
243.  Earlier versions of the design used 40, and lab58 part 6 justified that
choice by showing HAC(40), HAC(80) and a moving-block bootstrap with block 40
land close together.  Two of those three are not independent of each other.  A moving-block bootstrap with block b
cannot reproduce dependence longer than b, in exactly the way a Bartlett kernel
truncated at b cannot: resampling whole blocks of 40 destroys every
autocovariance past lag 40.  So HAC(40) agreeing with boot(40) is what you
would see whether or not 40 is long enough, and the agreement carries no
information about the question it is being used to answer.

The one genuinely informative comparison in that table is HAC(40) against
HAC(80), and it is not reassuring: at the headline delay it is 0.075210 against
0.080987, a 7.7% rise.  A standard error that is still climbing at double the
bandwidth has not converged.

This file measures the thing directly.  For each delay it takes the real loss
differentials, estimates their autocovariance sequence out to a long horizon,
and asks what fraction of the long-run variance a Bartlett kernel at each
bandwidth retains - including the 243 the paper now uses, which is read from
lab66 rather than written here.  A kernel that keeps a fraction f of the long-run variance produces
standard errors too small by sqrt(f), which turns a nominal 95% interval into
2*Phi(1.96*sqrt(f)) - 1.

WHAT TO DO WITH THE ANSWER
--------------------------
If the retained fraction is near 1 the paper's intervals are fine and lab58's
conclusion stands for a better reason than the one it gives.  If it is well
below 1, every HAC-studentised interval in the paper is too narrow, the
Fieller sets inherit it, and the fix is a longer bandwidth - not a longer
block, which cannot help beyond its own length.

AND THE ONE COMPARISON THIS FILE IS FOR
---------------------------------------
The coverage figures the paper quotes come from a simulation whose dependence
stops at a known lag, so the kernel there retains almost all of the long-run
variance and coverage comes out near nominal.  On the real series it does not.
The retained share at the paper's own bandwidth, printed below, converts into
an implied coverage by the same arithmetic the generator's does, and that figure
belongs beside the simulated one wherever the simulated one is quoted.  A
simulation that is easier than the thing it certifies is not a certificate.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab02_delay_curve as L

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lab10_loss_scale as L10

DELAYS = [0, 3, 5, 8, 13, 21, 34, 55]
# The chosen length is READ and forced into the sweep, so this file
# cannot go on reporting the shortfall of a bandwidth the labs no
# longer use - which is exactly what it did when the length moved
# from 40 to 80 and every sentence below still said 40.
import lab66_har_alternatives as H66          # noqa: E402
# The bandwidth the paper's own intervals are studentised at is read from the
# function that sets it, not listed here.  It was missing from this sweep for
# two bandwidth changes - the file went on measuring 40 and 80 while Section 4.6
# moved to a fixed b = 0.05, which at the evaluation length is 243 - so the one
# figure a reader of this file wants was the one it did not print.  Reading it
# rather than typing it is the only version of this that cannot go stale again.
BANDS_BASE = sorted({9, 40, 80, 160, 320} | {L.BLOCK, L.BLOCK_CAP})
# REF_LAG is the REFERENCE truncation, summed far past any dependence
# these series carry.  It is deliberately NOT the delay's own length:
# it is the denominator every retained-variance figure is a share of,
# so a checker that demanded the delay's length here would be asking
# for the measurement to be taken against itself.
REF_LAG = 600          # horizon for the reference autocovariance sum
Z = 1.959963985


def bartlett_lrv(g, band):
    """Long-run variance from an autocovariance sequence under a Bartlett kernel."""
    s = g[0]
    for L in range(1, min(band, len(g) - 1) + 1):
        s += 2.0 * (1.0 - L / (band + 1.0)) * g[L]
    return s


def autocov(x, kmax):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    k = min(kmax, n - 2)
    return np.array([float((x[L:] * x[:n - L]).sum() / n) for L in range(k + 1)])


def norm_cdf(z):
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def main(argv):
    folder = None
    for i, a in enumerate(argv):
        if a == "--data" and i + 1 < len(argv):
            folder = argv[i + 1]
    own, P, yy, idx, _ = L10.panel(L10.L.find_folder(folder))
    yb = yy[idx]
    # The bandwidth Section 4.6's own intervals use, read from the function that
    # sets it at the length this file is measuring.  BANDS is built here rather
    # than at module level for that reason: it depends on n.
    HAR_BAND = int(H66.har_band(len(idx)))
    BANDS = sorted(set(BANDS_BASE) | {HAR_BAND})

    def sq(d, cross):
        fo, so = L10.walk(own, P, yy, idx, d, cross)
        return (yb - fo) ** 2

    L0 = sq(0, False)
    print("lab65: what the Bartlett kernel actually keeps, at every bandwidth\n"
          "  and at the one the labs use\n")
    print("  d1 = L_own(d) - L_alt(d)   the numerator of R")
    print("  d2 = L_own(d) - L_own(0)   the denominator, what delay costs")
    print(f"  reference long-run variance summed to lag {REF_LAG}\n")

    print(f"{'delta':>6}{'series':>8}" + "".join(f"{'ret@'+str(b):>10}" for b in BANDS)
          + f"{'rule':>6}{'ret@rule':>10}{'95% at 40 is':>14}"
            f"{'95% at rule is':>16}")

    worst = 1.0
    for d in DELAYS:
        Ld = sq(d, False)
        La = sq(d, True)
        for tag, x in (("d1", Ld - La), ("d2", Ld - L0)):
            if d == 0 and tag == "d2":
                continue
            g = autocov(x, REF_LAG)
            ref = bartlett_lrv(g, REF_LAG)
            if ref <= 0:
                print(f"{d:>6}{tag:>8}   reference long-run variance non-positive")
                continue
            rets = [bartlett_lrv(g, b) / ref for b in BANDS]
            # the bandwidth the labs ACTUALLY use at this delay, not a
            # constant: block_for(d) is what every interval at this delay
            # is studentised and resampled at, so it is the only column
            # whose shortfall is the paper's own.
            rule = L.block_for(d)
            rc = bartlett_lrv(g, rule) / ref
            cov = 2.0 * norm_cdf(Z * np.sqrt(max(rc, 1e-9))) - 1.0
            # What a FLAT 40 would have given at this delay, printed beside
            # it.  This is the shortfall the rule exists to correct, and the
            # paper quotes it; dropping the column once the rule was in place
            # left the justification quoting a number nothing printed.
            r40 = rets[BANDS.index(40)]
            cov40 = 2.0 * norm_cdf(Z * np.sqrt(max(r40, 1e-9))) - 1.0
            worst = min(worst, cov)
            print(f"{d:>6}{tag:>8}" + "".join(f"{v*100:>9.1f}%" for v in rets)
                  + f"{rule:>6}{rc*100:>9.1f}%{cov40*100:>13.1f}%"
                    f"{cov*100:>15.1f}%")

    # The one line the paper quotes from this file.  The generator in lab66's
    # own coverage study has dependence that stops dead at a known lag, so at a
    # bandwidth five times that lag it keeps nearly all of its long-run variance
    # and coverage lands near nominal.  The real denominator does not, and the
    # same arithmetic that converts a retained share into a coverage there
    # converts this one here.  Printed so that a simulated coverage is never
    # quoted without the figure that says how much easier the simulation was.
    _g55 = autocov(sq(55, False) - L0, REF_LAG)
    _ref55 = bartlett_lrv(_g55, REF_LAG)
    _r55 = bartlett_lrv(_g55, HAR_BAND) / _ref55
    _c55 = 2.0 * norm_cdf(Z * np.sqrt(max(_r55, 1e-9))) - 1.0
    print(f"\n  AT THE BANDWIDTH SECTION 4.6 USES, {HAR_BAND} DAYS, ON THE "
          f"HEADLINE DENOMINATOR")
    print(f"  retained share of the reference long-run variance: "
          f"{_r55 * 100:.1f}%")
    print(f"  which implies a coverage of {_c55 * 100:.1f}% for a nominal 95% "
          f"interval, by the")
    print(f"  arithmetic 2*Phi(1.96*sqrt(f)) - 1.  The fixed-b correction "
          f"prices the ESTIMATION")
    print(f"  of the long-run variance and cannot price what the taper discards,"
          f" so this")
    print(f"  shortfall survives it.  Any coverage figure simulated on a "
          f"generator whose")
    print(f"  dependence is exhausted inside the bandwidth is measuring an "
          f"easier problem,")
    print(f"  and belongs beside this number rather than on its own.")

    # ------------------------------------------------------------------
    # IS ANY OF THIS STABLE?  The honest answer is no, and it is printed here
    # rather than left for a referee to find.
    # ------------------------------------------------------------------
    # Every figure above is a share of a REFERENCE long-run variance, and the
    # reference is itself a Bartlett sum at REF_LAG on the same 4,862 days - a
    # bandwidth of about an eighth of the sample, where the usual guidance is
    # nearer n^(1/3).  A share is only as good as its denominator, so the
    # denominator is swept below.  It does not hold still: the delta = 55
    # denominator reads 89% retained against a reference at 60 lags and 68%
    # against one at 900, and the reference itself climbs from 38.8 to 51.0
    # across that range without settling.  The 75.7% this file used to lead
    # with is therefore a property of choosing 600, not a property of the data.
    #
    # That is why the paper no longer rests anything on this measurement.
    # Section 4.6 inverts a fixed-b t-test instead, which needs no view about
    # how long a bandwidth has to be and so needs no retained fraction to
    # justify one; lab66 measures its coverage directly.  This file is kept
    # because the diagnostic is still informative about WHY a flat 40 was
    # wrong at the long delays - the direction is robust across every
    # reference tried, even where the magnitude is not - and because a
    # measurement that was quoted in a paper and then found to be unstable
    # should be recorded as such rather than deleted.
    print("\n" + "=" * 78)
    print("HOW MUCH OF THIS SURVIVES THE REFERENCE TRUNCATION")
    print("=" * 78)
    print("""The table above divides by a reference long-run variance summed to
REF_LAG.  If the retained fractions are properties of the data they should
barely move when that choice moves.  They move a great deal.\n""")
    _ref_sweep = (60, 120, 200, 300, 400, REF_LAG, 900, 1200)
    _d1_55 = sq(55, False) - sq(55, True)
    _d2_55 = sq(55, False) - L0
    _d1_3 = sq(3, False) - sq(3, True)
    print(f"{'REF_LAG':>9}{'d=55 d1 ret@40':>17}{'d=55 d2 ret@40':>17}"
          f"{'d=3 d1 ret@40':>16}{'reference LRV(d2)':>20}")
    for R in _ref_sweep:
        row = f"{R:>9}"
        for x in (_d1_55, _d2_55, _d1_3):
            g = autocov(x, R)
            ref = bartlett_lrv(g, R)
            row += f"{(bartlett_lrv(g, 40) / ref * 100) if ref > 0 else float('nan'):>16.1f}%"
        g2 = autocov(_d2_55, R)
        row += f"{bartlett_lrv(g2, R):>20.4g}"
        print(row)
    print("""
  Read the last column first: the reference has not converged, so neither has
  anything divided by it.  The delta = 3 column diverges outright, passing
  100% and climbing - a kernel cannot retain more than all of a variance, and
  a figure that does is telling you the denominator is broken rather than that
  the numerator is fine.
""")
    print(f"\n  worst nominal-95% interval in the table above: {worst*100:.1f}%")
    print("\n  'ret@b' is the share of the reference long-run variance a Bartlett")
    print("  kernel at bandwidth b retains.  'rule' is block_for(delta), the")
    print("  bandwidth the labs use at that delay.  If ret@rule is well under")
    print("  100% the standard errors are too small by its square root, every")
    print("  interval studentised by them is too narrow, and the Fieller sets")
    print("  inherit the same shortfall.  Note that ret@rule and a moving-block")
    print("  bootstrap at the same block share this limitation, so their agreement")
    print("  lab58 part 6 is not evidence that either is long enough: the two")
    print("  truncate the same dependence at the same lag.  Coverage is lab63's")
    print("  question, not this file's.")
    print("\n  The 40 column is kept because a flat 40 is what this project used")
    print("  before this measurement, so the table shows what the change was")
    print("  worth rather than merely asserting that it helped.")


if __name__ == "__main__":
    main(sys.argv)
