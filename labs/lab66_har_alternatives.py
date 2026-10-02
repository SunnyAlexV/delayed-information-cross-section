"""
lab66_har_alternatives.py - can the ratio's confidence set actually cover at
95%, and does it need a bandwidth at all?  Monte Carlo only; uses no market
data.  Runtime ~15 min.

WHY THIS FILE EXISTS
--------------------
lab63 prices the truncation length and finds something the paper has to answer:
NO length reaches nominal coverage.  Fieller with normal critical values covers
89.4%, 92.6% and 92.6% at bandwidths 40, 80 and 160 when the nominal level is
95%.  Decomposing that shortfall is what motivates this file.

Write ret(M) for the share of the true long-run variance a Bartlett kernel at
bandwidth M retains.  If the ONLY error were that the standard error is too
small by sqrt(ret), coverage would be 2*Phi(1.96*sqrt(ret)) - 1:

    M      ret     predicted     observed     unexplained
    40    73.3%       90.7%        89.4%           -1.3
    80    86.5%       93.2%        92.6%           -0.6
   160    93.2%       94.2%        92.6%           -1.6

Truncation bias accounts for almost the whole shortfall at 40 and for NONE of
the residual at 160.  The residual does not shrink as the bandwidth grows, so
no choice of length removes it.  Two different errors are at work:

  (1) TRUNCATION BIAS.  The kernel misses long-run variance.  Lengthening the
      bandwidth fixes this, and lab02.block_for() does.
  (2) SOMETHING ELSE, worth about a point and a half.  The candidates are the
      sampling variability of the long-run variance estimate itself - a normal
      critical value treats Omega-hat as known when it is not - and the
      nonlinearity of inverting a ratio.

This file tests the standard answers to (2).  Both replace "make the bandwidth
long enough" - which nothing guarantees - with critical values that price the
estimation noise directly.

  FIXED-b (Kiefer and Vogelsang 2005; Sun 2013).  Hold b = M/n fixed rather
  than letting it vanish.  The studentised statistic then has a non-standard
  but PIVOTAL limit depending on the kernel and b, so a generous bandwidth
  becomes affordable: the extra noise it brings is priced in the critical
  value instead of being assumed away.

  SELF-NORMALISED (Shao 2010).  Studentise by a functional of the recursive
  subsample means rather than by a long-run variance estimate.  No bandwidth
  at all.  For a paper whose central difficulty is that the bandwidth matters,
  a procedure with no bandwidth is the sharpest available check.

HOW THE CRITICAL VALUES ARE OBTAINED, AND WHY NOT FROM A FORMULA
----------------------------------------------------------------
Kiefer and Vogelsang publish a polynomial approximation to the fixed-b Bartlett
critical values.  It is not used here.  A coefficient recalled from memory and
not checkable against anything in this repository is exactly the kind of
plausible-looking input this project has been bitten by, and the alternative is
cheap: the fixed-b and self-normalised limits are PIVOTAL, so their critical
values can be simulated from iid data and are then valid for dependent data.
Part A does that and checks the output against the two things it must satisfy -
the fixed-b value must approach 1.96 as b approaches zero, and must rise with
b.  A formula cannot be audited that way.

WHAT WOULD FALSIFY THE FIX
--------------------------
If fixed-b and self-normalisation still under-cover once the bandwidth is
generous, then the shortfall is not about critical values either, and the
honest reading is that inverting a ratio under weak identification is harder
than any of these procedures admits - which is a finding about the estimand
rather than a defect in the arithmetic.  Part C reports whichever it is.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab02_delay_curve as L                                    # noqa: E402
import lab63_fieller_coverage as G                               # noqa: E402

SEED = 20260930
N = G.N                       # 4,862 test days, the paper's own window
RSTAR = G.RSTAR
CORR12 = G.CORR12
LRV_TRUE = G.LRV_TRUE

CV_DRAWS = 3000               # iid draws behind each simulated critical value
REPS = 1200                   # replications per coverage cell
TAUS = (10.0, 3.0, 1.5)       # denominator identification strength
CHUNK = 400                   # rows per pass, to bound memory

# b = M/n.  The paper's current length and lab63's sweep sit at the small-b
# end; 0.05 and 0.10 are where the fixed-b correction starts to matter and are
# the region Kiefer and Vogelsang work in.
BANDS = (L.BLOCK, L.BLOCK_CAP, 160, int(round(0.05 * N)), int(round(0.10 * N)))
Z_NORMAL = 1.959963985


# ----------------------------------------------------------------------
# shared machinery
# ----------------------------------------------------------------------
def autocovs(x, y, maxlag):
    """g[:, L] = sample autocovariance of x and y at lag L, per row.

    Computed by FFT, which is exact here rather than approximate: zero-padding
    to at least 2n turns the circular convolution the transform computes into
    the linear one wanted, so this returns the same numbers the direct double
    loop returns and returns them about twenty times faster.  That matters
    because the critical value below needs 24,000 draws at bandwidth 243, and
    at direct-loop speed every lab that asked for it would pay four minutes.

    The symmetrised cross-covariance (both orderings, halved) is what the
    Bartlett sum below expects, so that the same routine serves the variance
    and the covariance terms.
    """
    n = x.shape[1]
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    size = 1
    while size < 2 * n:
        size *= 2
    fx = np.fft.rfft(xc, size, axis=1)
    fy = np.fft.rfft(yc, size, axis=1)
    fwd = np.fft.irfft(fx * np.conj(fy), size, axis=1)[:, :maxlag + 1]
    bwd = np.fft.irfft(fy * np.conj(fx), size, axis=1)[:, :maxlag + 1]
    return (fwd + bwd) / (2.0 * n)


def _autocovs_direct(x, y, maxlag):
    """The double loop the FFT replaces, kept so the FFT can be checked."""
    n = x.shape[1]
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    g = np.empty((x.shape[0], maxlag + 1))
    g[:, 0] = (xc * yc).sum(axis=1) / n
    for k in range(1, maxlag + 1):
        g[:, k] = ((xc[:, k:] * yc[:, :-k]).sum(axis=1)
                   + (yc[:, k:] * xc[:, :-k]).sum(axis=1)) / (2.0 * n)
    return g


def bartlett(g, band):
    """Bartlett long-run variance from an autocovariance sequence, per row."""
    w = 1.0 - np.arange(1, band + 1) / (band + 1.0)
    return g[:, 0] + 2.0 * (g[:, 1:band + 1] * w).sum(axis=1)


def invert(A, B, C):
    """The set {r : A r^2 + B r + C <= 0}, as (kind, lo, hi) per row.

    kind 0 an ordinary interval, 1 the complement of one (two half-lines),
    2 the whole line, 3 empty.  Inverting on a grid was the alternative and is
    both slower and less exact: every statistic here is a ratio of quadratics
    in r, so the set has closed form and the awkward cases - unbounded, empty -
    fall out of the discriminant instead of being missed between grid points.
    """
    A = np.asarray(A, float); B = np.asarray(B, float); C = np.asarray(C, float)
    kind = np.zeros(A.shape, int)
    lo = np.full(A.shape, -np.inf)
    hi = np.full(A.shape, np.inf)
    disc = B * B - 4.0 * A * C

    no_root = disc < 0
    kind[no_root & (A > 0)] = 3                  # always positive: empty
    kind[no_root & (A <= 0)] = 2                 # always negative: whole line

    ok = ~no_root
    sq = np.sqrt(np.where(ok, disc, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = (-B - sq) / (2.0 * A)
        r2 = (-B + sq) / (2.0 * A)
    a, b_ = np.minimum(r1, r2), np.maximum(r1, r2)

    inter = ok & (A > 0)
    kind[inter] = 0
    lo[inter], hi[inter] = a[inter], b_[inter]

    excl = ok & (A < 0)
    kind[excl] = 1
    lo[excl], hi[excl] = a[excl], b_[excl]

    lin = ok & (A == 0)                          # degenerate, treat as unbounded
    kind[lin] = 2
    return kind, lo, hi


def covers(kind, lo, hi, truth):
    out = np.zeros(kind.shape, bool)
    out[kind == 0] = (lo[kind == 0] <= truth) & (truth <= hi[kind == 0])
    out[kind == 1] = (truth <= lo[kind == 1]) | (truth >= hi[kind == 1])
    out[kind == 2] = True
    return out


def widths(kind, lo, hi):
    w = np.full(kind.shape, np.inf)
    w[kind == 0] = hi[kind == 0] - lo[kind == 0]
    return w


# ----------------------------------------------------------------------
# A - critical values, simulated rather than quoted
# ----------------------------------------------------------------------
def cv_fixed_b_n(rng, n, band, draws=CV_DRAWS):
    """Two-sided 95% critical value for the Bartlett HAR t-statistic at M=band.

    Simulated under iid data of length n.  That is legitimate precisely because
    the fixed-b limit is pivotal: it depends on the kernel and on b = M/n, not
    on the dependence in the series, so a value calibrated on iid draws is the
    right value to use on dependent ones.
    """
    keep = []
    done = 0
    while done < draws:
        m = min(CHUNK, draws - done)
        x = rng.standard_normal((m, n))
        g = autocovs(x, x, band)
        om = bartlett(g, band)
        t = np.sqrt(n) * x.mean(axis=1) / np.sqrt(np.maximum(om, 1e-300))
        keep.append(np.abs(t))
        done += m
    return float(np.percentile(np.concatenate(keep), 95.0))


def cv_fixed_b(rng, band, draws=CV_DRAWS):
    """This file's own sample length, kept for Part A's table."""
    return cv_fixed_b_n(rng, N, band, draws)


# ----------------------------------------------------------------------
# the interface the other labs use
# ----------------------------------------------------------------------
B_FRAC = 0.05                 # b = M/n, the length fixed-b is run at

# Cached per (n, band) and seeded from them, so two labs asking for the same
# critical value get the same number and a lab asking twice pays once.  The
# seed is derived from the arguments rather than drawn from a shared stream:
# otherwise the value would depend on how many other things a lab had
# simulated first, which is not a property a critical value should have.
_CV = {}


def har_band(n, b=B_FRAC):
    """The bandwidth fixed-b inference is run at, for a sample of length n."""
    return max(2, int(round(b * n)))


# More draws than Part A's table needs.  A critical value that published
# intervals are built on has to be precise enough that its own simulation
# noise does not move an endpoint: at 3,000 draws the 95th percentile of |t|
# carries a standard error near 0.06, which on the headline interval is
# nearly a point of width - visible at the precision the paper reports.
# At 12,000 it is about 0.03, which moves that interval by less than
# half a point and so disappears into the rounding.  Going further buys
# nothing a reader could see and costs every lab that asks for it.
CV_DRAWS_EXPORT = 12000


def har_cv(n, band=None, b=B_FRAC, draws=CV_DRAWS_EXPORT):
    """Fixed-b 95% two-sided critical value for a sample of length n.

    This is the one function the inference labs call.  lab66 owns it because
    lab66 is where it is validated: Part A checks that it approaches 1.96 as
    b approaches zero and rises with b, which is the only audit a simulated
    constant can be given.
    """
    band = har_band(n, b) if band is None else band
    key = (int(n), int(band), int(draws))
    if key not in _CV:
        rng = np.random.default_rng(SEED + 7919 * int(band) + int(n))
        _CV[key] = cv_fixed_b_n(rng, int(n), int(band), draws)
    return _CV[key]


def sn_parts(d1, d2):
    """The self-normaliser's quadratic coefficients, per row.

    Shao's statistic studentises by a functional of the RECURSIVE subsample
    means, so nothing here is a long-run variance and nothing takes a
    bandwidth.  Writing gbar_k(r) = a_k - r b_k makes the normaliser a
    quadratic in r, which lets the same inversion routine serve both
    procedures.
    """
    n = d1.shape[1]
    k = np.arange(1, n + 1, dtype=float)
    m1 = d1.mean(axis=1, keepdims=True)
    m2 = d2.mean(axis=1, keepdims=True)
    a = np.cumsum(d1, axis=1) / k - m1
    b = np.cumsum(d2, axis=1) / k - m2
    k2 = k * k
    S = (k2 * a * a).sum(axis=1) / n ** 2
    Q = (k2 * a * b).sum(axis=1) / n ** 2
    P = (k2 * b * b).sum(axis=1) / n ** 2
    return S, Q, P


def cv_sn(rng, draws=CV_DRAWS):
    """95% critical value for Shao's self-normalised statistic, simulated iid.

    Pivotal for the same reason as fixed-b, so iid calibration transfers.
    """
    keep = []
    done = 0
    while done < draws:
        m = min(CHUNK, draws - done)
        x = rng.standard_normal((m, N))
        n = N
        k = np.arange(1, n + 1, dtype=float)
        mx = x.mean(axis=1, keepdims=True)
        a = np.cumsum(x, axis=1) / k - mx
        D = ((k * k) * a * a).sum(axis=1) / n ** 2
        T = n * (mx.ravel() ** 2) / np.maximum(D, 1e-300)
        keep.append(T)
        done += m
    return float(np.percentile(np.concatenate(keep), 95.0))


def fieller_fixed_b(num, den, b=B_FRAC, draws=CV_DRAWS_EXPORT):
    """Fixed-b confidence set for mean(num)/mean(den).  Returns (kind, lo, hi).

    kind: "interval", "exclusion" (two half-lines) or "unbounded".

    This is the one implementation every lab calls, so there is exactly one
    place where the bandwidth, the critical value and the inversion live.  The
    project has been bitten four times by the same constant spelled out in two
    files, and a confidence set is the last thing to spell twice.

    The set inverts a test of E[num - r den] = 0 over r, which is the
    Anderson-Rubin form of Fieller's construction and is why the awkward cases
    appear at all: under weak identification the quadratic opens the other way
    and the honest answer is two half-lines or the whole line, not a bounded
    interval that happens to be reported.

    Why fixed-b rather than a normal critical value: lab63 shows the normal
    version under-covers at EVERY truncation length, and Part B above shows the
    shortfall is not about length - it is that a normal quantile treats the
    long-run variance as known when it is estimated.  Fixed-b prices that
    estimation, which makes a generous bandwidth affordable and removes the
    need for any rule about how long the bandwidth should be.
    """
    num = np.asarray(num, float).ravel()
    den = np.asarray(den, float).ravel()
    n = len(num)
    band = har_band(n, b)
    cv2 = har_cv(n, band, b, draws) ** 2
    g11 = autocovs(num[None, :], num[None, :], band)
    g22 = autocovs(den[None, :], den[None, :], band)
    g12 = autocovs(num[None, :], den[None, :], band)
    o11 = float(bartlett(g11, band)[0])
    o22 = float(bartlett(g22, band)[0])
    o12 = float(bartlett(g12, band)[0])
    m1, m2 = float(num.mean()), float(den.mean())
    A = np.array([m2 * m2 - cv2 * o22 / n])
    B = np.array([-2.0 * (m1 * m2 - cv2 * o12 / n)])
    C = np.array([m1 * m1 - cv2 * o11 / n])
    k, lo, hi = invert(A, B, C)
    name = {0: "interval", 1: "exclusion", 2: "unbounded", 3: "empty"}[int(k[0])]
    return name, float(lo[0]), float(hi[0])


# ----------------------------------------------------------------------
# B - the differentials, drawn as lab63 draws them
# ----------------------------------------------------------------------
def draw(rng, tau, reps, corr=None):
    corr = CORR12 if corr is None else corr
    e2 = G.dep(rng, N, reps)
    eo = G.dep(rng, N, reps)
    e1 = corr * e2 + np.sqrt(1.0 - corr ** 2) * eo
    mu2 = tau * np.sqrt(LRV_TRUE / N)
    return RSTAR * mu2 + e1, mu2 + e2


# The correlation between the two differentials is a PARAMETER of the generator
# above, and coverage of a ratio depends on it, so leaving it at one asserted
# value is leaving the result resting on that value.  lab21 measures it on the
# real differentials: 0.89 at the headline delay and 0.71 as a mean over the
# grid, against the 0.50 the generator was given.  Part D reruns the procedure
# the paper uses at the measured values, so the reader is told what the
# assumption is worth instead of being asked to accept it.
CORRS_TRIED = [CORR12, 0.71, 0.89]


def part_d(rng, cvs, rows=None):
    print("\n" + "=" * 78)
    print("D.  THE CORRELATION THE GENERATOR ASSUMES, PRICED")
    print("=" * 78)
    print(f"""The two differentials share test days, so they are correlated, and the
generator sets that correlation to {CORR12:.2f}.  lab21 measures it on the real
differentials and finds 0.89 at delta = 55 and 0.71 averaged over the delay
grid.  A coverage figure computed at {CORR12:.2f} is therefore a figure at a
correlation the data do not have.  Below is the procedure the paper uses,
b = {B_FRAC}, at each of the three.\n""")
    print(f"{'corr(d1,d2)':>13}" + "".join(f"{'tau=' + str(t):>12}" for t in TAUS))
    # the bandwidth the paper actually uses: b = B_FRAC of the sample
    band = har_band(N)
    out = {}
    for c in CORRS_TRIED:
        cells = []
        for tau in TAUS:
            d1, d2 = draw(rng, tau, REPS, corr=c)
            m1, m2 = d1.mean(axis=1), d2.mean(axis=1)
            o11 = bartlett(autocovs(d1, d1, band), band)
            o12 = bartlett(autocovs(d1, d2, band), band)
            o22 = bartlett(autocovs(d2, d2, band), band)
            cv2 = cvs[band] ** 2
            A = m2 * m2 - cv2 * o22 / N
            B = -2.0 * (m1 * m2 - cv2 * o12 / N)
            C = m1 * m1 - cv2 * o11 / N
            k_, lo_, hi_ = invert(A, B, C)
            cov = covers(k_, lo_, hi_, RSTAR).mean() * 100
            cells.append(cov)
            out[(c, tau)] = cov
        print(f"{c:>13.2f}" + "".join(f"{x:>11.1f}%" for x in cells))
    _base = [out[(CORR12, t)] for t in TAUS]
    _worst = max(abs(out[(c, t)] - out[(CORR12, t)])
                 for c in CORRS_TRIED for t in TAUS)
    print(f"\n  largest move in coverage between the assumed {CORR12:.2f} and the "
          f"measured values: {_worst:.1f} points,")
    print(f"  against a Monte Carlo standard error of "
          f"{G.mc_se(REPS) * 2 ** 0.5:.2f} on a difference.")
    print("  So the assumption is not load-bearing"
          if _worst < 3 * G.mc_se(REPS) * 2 ** 0.5 else
          "  So the assumption IS load-bearing and the generator should carry the\n"
          "  measured correlation rather than the assumed one")
    # This block draws fresh samples, so its corr = CORR12 row is a SECOND
    # estimate of Part B's fixed-b row at the same bandwidth, not a correction
    # of it.  lab63 makes the same disclosure about its own two parts, for the
    # same reason: two numbers for one quantity is something a reader finds and
    # wonders about, and the honest answer is the Monte Carlo error rather than
    # an explanation invented afterwards.
    if rows is not None:
        _se = G.mc_se(REPS) * 2 ** 0.5
        print("\n  THE SAME CELL, MEASURED TWICE")
        print("  This block's corr = "
              f"{CORR12:.2f} row and Part B's fixed-b M = {band} row estimate one")
        print("  quantity on independent draws.")
        print(f"{'tau':>6}{'Part B':>10}{'Part D':>10}{'gap':>8}{'gap / SE':>10}")
        _wd = 0.0
        for tau in TAUS:
            _b = rows[(tau, "fixed-b", band)][0] * 100
            _d = out[(CORR12, tau)]
            _g = abs(_b - _d)
            _wd = max(_wd, _g / _se)
            print(f"{tau:>6.1f}{_b:>9.1f}%{_d:>9.1f}%{_g:>8.1f}{_g / _se:>10.2f}")
        print(f"  largest gap is {_wd:.2f} standard errors of the difference "
              f"({_se:.2f} points), so the two")
        print("  agree within Monte Carlo error" if _wd < 3 else
              "  DISAGREE BY MORE THAN MONTE CARLO ERROR")


def main(argv=None):
    rng = np.random.default_rng(SEED)
    print("lab66: does the ratio's confidence set cover at 95%, and does it "
          "need a bandwidth?\n")
    print(f"  n = {N}, R* = {RSTAR}, corr(d1,d2) = {CORR12}, nominal 95%")
    print(f"  dependence: lab63's generator, matched to the measured "
          f"loss-differential ACF")
    print(f"  {CV_DRAWS} iid draws per critical value, {REPS} replications "
          f"per coverage cell\n")

    print("=" * 78)
    print("A.  CRITICAL VALUES, SIMULATED RATHER THAN QUOTED")
    print("=" * 78)
    print("""Kiefer and Vogelsang publish a polynomial for the fixed-b Bartlett values.
It is deliberately not used: a coefficient recalled from memory cannot be
audited against anything in this repository, and the limits are pivotal, so
simulating them from iid draws is both exact enough and checkable.  The two
checks are printed beneath the table - the value must approach the normal
1.96 as b approaches zero, and must rise with b.  A quoted constant would
satisfy neither test because neither test could be run on it.\n""")
    print(f"{'procedure':>26}{'M':>7}{'b = M/n':>10}{'95% two-sided cv':>19}")
    cvs = {}
    for band in BANDS:
        cv = cv_fixed_b(rng, band)
        cvs[band] = cv
        print(f"{'fixed-b Bartlett':>26}{band:>7}{band / N:>10.4f}{cv:>19.3f}")
    cv_s = cv_sn(rng)
    print(f"{'self-normalised (Shao)':>26}{'-':>7}{'-':>10}{cv_s:>19.2f}"
          f"   (on the SQUARED statistic)")
    print(f"{'normal':>26}{'-':>7}{'-':>10}{Z_NORMAL:>19.3f}")

    _small = cvs[min(BANDS)]
    _mono = all(cvs[a] <= cvs[b_] + 1e-9
                for a, b_ in zip(sorted(BANDS), sorted(BANDS)[1:]))
    # The value the inference labs actually use, printed here so there is
    # exactly one number in the repository for it.  Part A's sweep above runs
    # at fewer draws because it only has to show monotonicity; this row is the
    # one the documents quote, and a paper quoting the sweep's figure instead
    # would be quoting a number no lab used.
    _use_band = har_band(N)
    _use_cv = har_cv(N)
    print(f"\n  THE VALUE USED: b = {B_FRAC}, M = {_use_band} at n = {N}, "
          f"cv = {_use_cv:.2f} ({CV_DRAWS_EXPORT} draws)")
    print(f"\n  check 1, cv -> 1.96 as b -> 0: at b = {min(BANDS) / N:.4f} the "
          f"value is {_small:.3f}, "
          f"{'as it should be' if abs(_small - Z_NORMAL) < 0.15 else 'WHICH IS WRONG'}")
    print(f"  check 2, cv rises with b: "
          f"{'yes' if _mono else 'NO - the simulation is not behaving'}")
    print("  Both checks are on the simulation, not on the paper.  They are "
          "here because\n  a critical value that is wrong makes every coverage "
          "figure below wrong in the\n  same direction, and nothing else in "
          "this file would reveal it.")

    print("\n" + "=" * 78)
    print("B.  COVERAGE OF THE RATIO'S CONFIDENCE SET")
    print("=" * 78)
    print("""Every procedure inverts the same test of E[d1 - r d2] = 0 over r, so they
differ only in how the statistic is studentised and which critical value is
used.  'normal' at the paper's own bandwidth is the procedure the paper
currently runs.  Width is the median over bounded sets; sets that are a whole
line or two half-lines are counted separately, because a procedure can buy
coverage by declining to identify anything.\n""")
    maxband = max(BANDS)
    rows = {}
    for tau in TAUS:
        print(f"  tau = {tau}")
        print(f"{'procedure':>26}{'M':>7}{'b':>8}{'coverage':>11}"
              f"{'bounded only':>14}{'median width':>14}{'unbounded':>11}")
        d1, d2 = draw(rng, tau, REPS)
        m1, m2 = d1.mean(axis=1), d2.mean(axis=1)
        g11 = autocovs(d1, d1, maxband)
        g22 = autocovs(d2, d2, maxband)
        g12 = autocovs(d1, d2, maxband)
        S, Q, P = sn_parts(d1, d2)

        def report(name, band, bval, cv2, o11, o12, o22):
            A = m2 * m2 - cv2 * o22 / N
            B = -2.0 * (m1 * m2 - cv2 * o12 / N)
            C = m1 * m1 - cv2 * o11 / N
            k_, lo_, hi_ = invert(A, B, C)
            cov = covers(k_, lo_, hi_, RSTAR).mean()
            w = widths(k_, lo_, hi_)
            med = np.median(w[np.isfinite(w)]) if np.isfinite(w).any() else np.nan
            unb = (k_ != 0).mean()
            # Coverage among the BOUNDED sets, which is the number a reader
            # who quotes an interval actually needs.  A whole-line set covers
            # by construction, so total coverage mixes two different things,
            # and at small tau most of the sets are unbounded: reporting the
            # total alone lets a procedure look nominal while the intervals it
            # does produce are well short.  Two thirds of this file's warning
            # about buying coverage was unmeasured until this column existed.
            bnd = k_ == 0
            cov_b = covers(k_[bnd], lo_[bnd], hi_[bnd], RSTAR).mean() \
                if bnd.any() else np.nan
            rows[(tau, name, band)] = (cov, med, unb, cov_b)
            print(f"{name:>26}{band:>7}{bval:>8}{cov * 100:>10.1f}%"
                  f"{cov_b * 100:>13.1f}%{med:>14.3f}{unb * 100:>10.1f}%")

        # the paper's current procedure, and the same bandwidth under fixed-b
        for band in BANDS:
            o11 = bartlett(g11, band)
            o12 = bartlett(g12, band)
            o22 = bartlett(g22, band)
            if band == L.BLOCK_CAP:
                report("normal (the paper's)", band, f"{band / N:.3f}",
                       Z_NORMAL ** 2, o11, o12, o22)
            report("fixed-b", band, f"{band / N:.3f}", cvs[band] ** 2,
                   o11, o12, o22)
        # self-normalised: no bandwidth anywhere
        A = N * m2 * m2 - cv_s * P
        B = -2.0 * N * m1 * m2 + 2.0 * cv_s * Q
        C = N * m1 * m1 - cv_s * S
        k_, lo_, hi_ = invert(A, B, C)
        cov = covers(k_, lo_, hi_, RSTAR).mean()
        w = widths(k_, lo_, hi_)
        med = np.median(w[np.isfinite(w)]) if np.isfinite(w).any() else np.nan
        _bnd = k_ == 0
        _cb = covers(k_[_bnd], lo_[_bnd], hi_[_bnd], RSTAR).mean() \
            if _bnd.any() else np.nan
        rows[(tau, "self-normalised", 0)] = (cov, med, (k_ != 0).mean(), _cb)
        print(f"{'self-normalised':>26}{'none':>7}{'-':>8}{cov * 100:>10.1f}%"
              f"{_cb * 100:>13.1f}%{med:>14.3f}{(k_ != 0).mean() * 100:>10.1f}%")
        print()

    print("=" * 78)
    print("C.  WHAT THIS SETTLES")
    print("=" * 78)
    best = {}
    for tau in TAUS:
        cands = [(abs(v[0] - 0.95), k) for k, v in rows.items() if k[0] == tau]
        best[tau] = min(cands)[1]
    print("  closest to nominal 95% at each identification strength:")
    for tau in TAUS:
        k = best[tau]
        c, w, u, cb = rows[k]
        print(f"    tau = {tau:>4}: {k[1]}"
              + (f" at M = {k[2]}" if k[2] else "")
              + f"  -  {c * 100:.1f}% coverage ({cb * 100:.1f}% among bounded "
              + f"sets), median width {w:.3f}")
    _nrm = [rows[(t, "normal (the paper's)", L.BLOCK_CAP)][0] for t in TAUS]
    print(f"\n  the paper's current procedure covers "
          + ", ".join(f"{x * 100:.1f}%" for x in _nrm)
          + f" at tau = " + ", ".join(str(t) for t in TAUS) + ".")
    part_d(rng, cvs, rows)
    print(f"""
  Read the width column beside the coverage column.  A procedure that covers
  95% with an interval twice as wide has not solved the problem, it has
  restated it as a wider claim - which for this paper may still be the right
  trade, because the paper's contribution IS an interval and an interval that
  does not cover is worth nothing.  The choice between them is a judgement
  about how much width the reading can carry, and that judgement belongs in
  the paper rather than here.

  What this file cannot do is rescue a procedure that under-covers at every
  setting.  If that is what the table shows, the shortfall is a property of
  inverting a ratio under weak identification and not of any bandwidth or
  critical value, and Section 4.6 should say so in those terms.""")


if __name__ == "__main__":
    main(sys.argv)
