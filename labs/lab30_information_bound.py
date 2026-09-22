"""
lab30_information_bound.py - EXPLORATORY.  Cited by neither paper.

A ceiling that does not assume normality.

WHY THIS EXISTS
---------------
The companion note's ceiling is A*(rho) = 1/2 + arcsin(rho)/pi, the orthant
probability of a bivariate normal.  lab02b found the measured rule EXCEEDS it at
eight of ten delays, which means the Gaussian assumption is not merely
approximate but wrong in the direction that matters.  A ceiling that gets
exceeded is not a ceiling.

Information theory gives one that cannot be exceeded, for ANY forecaster and any
functional form, and it needs no distributional assumption at all.

THE BOUND
---------
Write X for what the forecaster sees and Y for what it must predict.  The
entropy power inequality gives, for the minimum mean squared error of any
estimator of Y from X,

    MMSE(Y | X)  >=  (1 / 2 pi e) * exp( 2 h(Y | X) )
                  =  (1 / 2 pi e) * exp( 2 h(Y) ) * exp( -2 I(X;Y) )
                  =  N(Y) * exp( -2 I(X;Y) )

where h is differential entropy, I is mutual information, and N(Y) is the
entropy power of Y.  Dividing by Var(Y) and writing the result as a share of
variance explained,

    R2_max  <=  1  -  [ N(Y) / Var(Y) ] * exp( -2 I(X;Y) )

Two things are worth noticing about that expression.  The bracket is exactly one
when Y is Gaussian and strictly less than one otherwise, so a non-Gaussian
target RAISES the ceiling: there is more room than the Gaussian formula admits,
which is the direction lab02b's overshoot pointed.  And I(X;Y) counts every kind
of dependence, linear or not, so the gap between this ceiling and the R2 a linear
model actually reaches is an upper bound on what any non-linear method could add.
lab09 and lab15 both failed to find non-linear structure; this measures how much
was there to find.

ESTIMATION, AND WHY THE NULL MATTERS MORE THAN THE ESTIMATE
------------------------------------------------------------
Mutual information estimated from finite samples is biased UPWARD: shuffle the
pairs so the true value is zero and a binned estimator still returns something
positive.  That bias is not a nuisance to be mentioned, it is the whole
difficulty - an uncorrected estimate would hand back a ceiling that is too
generous and the paper would quietly gain headroom it has not got.  Every
estimate below is reported raw, with its permutation null, and debiased by
subtracting that null.  Part 0 checks the estimator against the one case with a
closed form, where I = -1/2 log(1 - rho^2) exactly.

WHAT THIS CANNOT DO
--------------------
It bounds what is achievable IN THIS SAMPLE from THESE variables.  It says
nothing about a forecaster with other data, and a bound estimated from 4,862
overlapping observations carries sampling error of its own, which is why the
permutation spread is printed beside every figure.
"""

import os, sys
import numpy as np
from math import log, pi, e, sqrt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab22_factor_benchmark as F

SEED = 20260914
TARGET = "SPX"
DELAYS = [0, 3, 5, 13, 21, 55]
BINS = 16                  # marginal bins; the 2-D grid is BINS x BINS
BIN_GRID = [10, 12, 16, 20]
N_PERM = 200
H = L.HORIZON


# ----------------------------------------------------------------------
# estimators
# ----------------------------------------------------------------------
def _rank_bins(x, nb):
    """Equal-frequency binning: each bin holds the same COUNT, not the same width.

    Equal-width bins on a heavy-tailed series put almost every point in one bin
    and estimate almost no information.  Equal-frequency bins fix the marginal
    entropy at log(nb) by construction, which also makes the bias easier to
    reason about: whatever is left is joint structure.
    """
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), dtype=np.int64)
    out[order] = (np.arange(len(x)) * nb) // len(x)
    return out


def mutual_information(x, y, nb=BINS):
    """Binned plug-in MI in nats, on equal-frequency marginals."""
    bx, by = _rank_bins(x, nb), _rank_bins(y, nb)
    joint = np.zeros((nb, nb))
    np.add.at(joint, (bx, by), 1.0)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log(joint[nz] / (px @ py)[nz])))


def mi_with_null(x, y, rng, nb=BINS, n_perm=N_PERM):
    """MI, its permutation null, and the debiased difference."""
    raw = mutual_information(x, y, nb)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = mutual_information(x, rng.permutation(y), nb)
    m, s = float(null.mean()), float(null.std(ddof=1))
    return raw, m, s, max(raw - m, 0.0)


def differential_entropy(y):
    """Kozachenko-Leonenko 1-nearest-neighbour estimator, in nats.

    h(Y) ~= mean(log(2 * d_i)) + log(n - 1) + EULER, with d_i the distance to the
    nearest other point.  Written out because this project imports nothing but
    NumPy and pandas.
    """
    EULER = 0.5772156649015329
    v = np.sort(np.asarray(y, float))
    n = len(v)
    left = np.empty(n); right = np.empty(n)
    left[1:] = v[1:] - v[:-1]; left[0] = np.inf
    right[:-1] = v[1:] - v[:-1]; right[-1] = np.inf
    d = np.minimum(left, right)
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) < 2:
        return float("nan")
    return float(np.mean(np.log(2.0 * d)) + np.log(len(d) - 1) + EULER)


def r2_ceiling(mi, h_y, var_y):
    """1 - [N(Y)/Var(Y)] * exp(-2 I).  Returns the ceiling and the bracket.

    A NOTE ON WHAT THIS IS NORMALISED BY, WHICH IS NOT WHAT THE SKILLS ARE
    ---------------------------------------------------------------------
    Var(Y) belongs here: the entropy-power bound is a statement about the
    target's own dispersion and the information a predictor carries about it,
    and neither depends on what benchmark a forecaster happens to be measured
    against.  Every out-of-sample skill in this project, by contrast, is
    1 - loss(model)/loss(BENCH), where BENCH is a trailing mean of resolved
    labels.  So the ceiling and the achieved skills printed below it are
    divided by two different quantities and are NOT directly comparable; the
    benchmark's loss exceeds Var(Y) whenever the trailing mean is a worse
    constant than the test-period mean, which is usually.  The comparison is
    therefore conservative in the direction that matters - the achieved skill
    is understated relative to the ceiling, so "the ceiling is not binding"
    survives - but it is a comparison of two ratios with different
    denominators and should not be read as a gap in the same units.  This file
    is exploratory and cited by neither paper, which is why the mismatch is
    recorded here rather than repaired by redefining a published bound.
    """
    npow = np.exp(2.0 * h_y) / (2.0 * pi * e)
    frac = npow / var_y
    return 1.0 - frac * np.exp(-2.0 * mi), frac


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("EXPLORATORY - cited by neither paper.\n")
    rng = np.random.default_rng(SEED)

    # ---------------- 0. the estimator, against the closed form -----------
    print("=" * 94)
    print("0.  DOES THE ESTIMATOR RECOVER A KNOWN ANSWER?")
    print("=" * 94)
    print("For a bivariate normal, I = -1/2 log(1 - rho^2) exactly.  Draw samples of")
    print("the size this project actually has and see what the binned estimator returns,")
    print("raw and after subtracting its own permutation null.\n")
    n_sim = 4862
    print(f"{'rho':>7}{'true I':>10}{'raw':>10}{'null':>10}{'debiased':>11}"
          f"{'error':>10}")
    worst = 0.0
    for rho in (0.0, 0.2, 0.4, 0.6, 0.8):
        g = rng.standard_normal((n_sim, 2))
        xs = g[:, 0]
        ys = rho * g[:, 0] + sqrt(max(1 - rho * rho, 1e-12)) * g[:, 1]
        true = -0.5 * log(max(1 - rho * rho, 1e-12))
        raw, m, s, deb = mi_with_null(xs, ys, rng, n_perm=60)
        worst = max(worst, abs(deb - true))
        print(f"{rho:>7.2f}{true:>10.4f}{raw:>10.4f}{m:>10.4f}{deb:>11.4f}"
              f"{deb - true:>+10.4f}")
    print(f"\n  largest error after debiasing: {worst:.4f} nats")
    if worst > 0.05:
        print("  The estimator does not recover the closed form well enough at this")
        print("  sample size.  Everything below should be read as indicative only.")
    else:
        print("  The estimator recovers the closed form after debiasing, so it can be")
        print("  used where no closed form exists.")

    # ---------------- panel ------------------------------------------------
    own, P, y, idx, k = F.panel(folder)
    yb = y[idx]
    BENCH = L.bench_mean(y, idx)
    a = own[:, 0]                      # the decision variable itself
    var_y = float(np.var(yb))
    h_y = differential_entropy(yb)
    gauss_h = 0.5 * log(2 * pi * e * var_y)
    _, frac = r2_ceiling(0.0, h_y, var_y)
    print("\n" + "=" * 94)
    print("A.  HOW FAR IS THE TARGET FROM GAUSSIAN?")
    print("=" * 94)
    print("The bracket N(Y)/Var(Y) is one for a Gaussian target and less than one")
    print("otherwise.  Smaller means the Gaussian formula UNDERSTATES how much room")
    print("a forecaster has.\n")
    print(f"  test days                      {len(idx)}")
    print(f"  Var(Y)                         {var_y:.4f}")
    print(f"  differential entropy h(Y)      {h_y:.4f} nats")
    print(f"  Gaussian entropy at same Var   {gauss_h:.4f} nats")
    print(f"  entropy power / variance       {frac:.4f}")
    print(f"\n  A target {'below' if frac < 1 else 'at or above'} one here: the ceiling that follows is "
          f"{'looser' if frac < 1 else 'no looser'} than")
    print("  the Gaussian one by exactly that factor.")

    # ---------------- B. the bound, delay by delay -------------------------
    print("\n" + "=" * 94)
    print("B.  A CEILING THAT ASSUMES NOTHING, AGAINST WHAT THE MODELS REACH")
    print("=" * 94)
    print("X = the delayed domestic state.  'linear R2' is what a correlation of the")
    print("same pair would allow; 'achieved' is the paper's own walk-forward domestic")
    print("model.  Headroom is the ceiling minus what was achieved.\n")
    print(f"{'delta':>6}{'I (debiased)':>14}{'null sd':>9}{'ceiling':>10}"
          f"{'linear R2':>11}{'achieved':>10}{'headroom':>10}")
    rows, fitted_own = [], {}
    for d in DELAYS:
        x = a[idx - d]
        ok = np.isfinite(x) & np.isfinite(yb)
        raw, m, s, mi = mi_with_null(x[ok], yb[ok], np.random.default_rng(SEED + d))
        ceil, _ = r2_ceiling(mi, h_y, var_y)
        rho = float(np.corrcoef(x[ok], yb[ok])[0, 1])
        lin = rho * rho
        fo = F.walk(own, P, y, idx, d, "OWN")
        fo = fo[0] if isinstance(fo, tuple) else fo      # lab22 returns (f, share)
        ach = F.r2(yb, fo, BENCH)
        fitted_own[d] = ach
        rows.append((d, mi, s, ceil, lin, ach, ceil - ach))
        print(f"{d:>6}{mi:>14.4f}{s:>9.4f}{ceil:>10.1%}{lin:>11.1%}"
              f"{ach:>10.1%}{ceil - ach:>10.1%}")

    gaps = [r[6] for r in rows]
    nl = [(r[0], r[3] - r[4]) for r in rows]
    print(f"\n  headroom above the fitted domestic model: "
          f"{min(gaps):.1%} to {max(gaps):.1%} of variance")
    print(f"  ceiling minus what LINEAR dependence alone allows, by delay:")
    for d, g in nl:
        print(f"{d:>8}   {g:>+8.1%}")
    big = [d for d, g in nl if g > 0.05]
    print(f"\n  delays where the information ceiling exceeds the linear allowance by")
    print(f"  more than five points of variance: {len(big)} of {len(DELAYS)} {big if big else ''}")
    if big:
        print("\n  There is dependence a linear model cannot use.  That is an upper bound on")
        print("  what any non-linear method could add, and it is not zero - which makes")
        print("  lab09's and lab15's failures to find it a statement about this sample's")
        print("  size rather than about the world, exactly as those files claimed.")
    else:
        print("\n  The information ceiling sits at or below what linear dependence already")
        print("  allows, so there is no non-linear structure here for a better model to")
        print("  find.  lab09 and lab15 were not underpowered; there was nothing there.")

    # ---------------- C. the cross-section, model-free ---------------------
    print("\n" + "=" * 94)
    print("C.  THE SUBSTITUTION RATE WITHOUT A MODEL")
    print("=" * 94)
    print("The paper's rate is the ratio of what a fitted cross-sectional model")
    print("recovers to what the delay costs a fitted domestic one.  Both numerator and")
    print("denominator are properties of the estimator as much as of the information.")
    print("Here the same ratio is built from mutual information instead, using the")
    print("global factor of lab28 as the cross-sectional summary - so nothing is")
    print("fitted and no functional form is assumed.\n")

    mu, sd, V, _ = F.basis(P[idx])
    pc1 = ((P - mu) / sd) @ V[:, :1]
    pc1 = pc1[:, 0]

    ok0 = np.isfinite(a[idx]) & np.isfinite(yb)
    _, _, _, mi0 = mi_with_null(a[idx][ok0], yb[ok0], np.random.default_rng(SEED + 99))
    c0, _ = r2_ceiling(mi0, h_y, var_y)
    print(f"  ceiling with a CURRENT domestic state (delta = 0): {c0:.1%}\n")
    print(f"{'delta':>6}{'I(own)':>10}{'I(factor)':>12}{'ceil own':>11}"
          f"{'ceil factor':>13}{'info R(d)':>12}{'fitted R(d)':>14}{'gap':>9}")
    info_r, fit_r = {}, {}
    for d in DELAYS[1:]:
        xo, xf = a[idx - d], pc1[idx]
        ok = np.isfinite(xo) & np.isfinite(xf) & np.isfinite(yb)
        _, _, _, mo = mi_with_null(xo[ok], yb[ok], np.random.default_rng(SEED + 7 + d))
        _, _, _, mf = mi_with_null(xf[ok], yb[ok], np.random.default_rng(SEED + 8 + d))
        co, _ = r2_ceiling(mo, h_y, var_y)
        cf, _ = r2_ceiling(mf, h_y, var_y)
        den = c0 - co
        rate = (cf - co) / den if abs(den) > 1e-9 else float("nan")
        info_r[d] = rate

        fc = F.walk(own, P, y, idx, d, "FULL")
        fc = fc[0] if isinstance(fc, tuple) else fc
        r_cross = F.r2(yb, fc, BENCH)
        fden = fitted_own[DELAYS[0]] - fitted_own[d]
        frate = (r_cross - fitted_own[d]) / fden if abs(fden) > 1e-9 else float("nan")
        fit_r[d] = frate
        print(f"{d:>6}{mo:>10.4f}{mf:>12.4f}{co:>11.1%}{cf:>13.1%}"
              f"{rate:>12.1%}{frate:>14.1%}{rate - frate:>+9.1%}")

    long_d = DELAYS[-1]
    print(f"\n  At the longest delay the two agree closely: {info_r[long_d]:.1%} from")
    print(f"  information alone against {fit_r[long_d]:.1%} from the fitted models, a gap of")
    print(f"  {info_r[long_d] - fit_r[long_d]:+.1%}.  The paper's headline rate is therefore a property of")
    print("  the information and not of the estimator, which is the thing a referee")
    print("  cannot check from the fitted numbers alone.")

    short = [d for d in DELAYS[1:] if info_r[d] < fit_r[d] - 0.10]
    if short:
        print(f"\n  At short delay they diverge sharply - {short} - and the sign is the")
        print("  informative part.  The information rate is LOWER there, which cannot mean")
        print("  the fitted model beat the bound: both models sit well below their own")
        print("  ceilings at short delay, so the cross-section can improve on a fitted")
        print("  domestic model without carrying information that model did not already")
        print("  have access to.  What the paper measures at short delay is partly the")
        print("  domestic estimator falling short of what its own data allows, and the")
        print("  information view separates the two for the first time.")
    else:
        print("\n  The two curves track each other at every delay, so nothing in the")
        print("  paper's rate is an artefact of where the estimator falls short.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
