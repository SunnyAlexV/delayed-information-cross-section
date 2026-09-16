"""
lab26_quantile_target.py - the companion note splits its target at the MEDIAN.
A referee points out that volatility regimes are asymmetric and asks whether
the threshold rule still dominates when the target is an extreme quantile - a
spike rather than a typical day.  This file generalises the note's theory to an
arbitrary quantile, derives the rule that theory actually implies, and measures
where the answer runs out of data.

Imports lab02_delay_curve; keep both in labs/.  Runtime under a minute.

WHAT THE NOTE PROVES, AND WHY IT DOES NOT TRANSFER
---------------------------------------------------
Write a for the standardised delayed state and b for the standardised future
state, and take them approximately joint normal with correlation rho.  The
note's target is 1{b > 0} - the median split - and its ceiling is

    A*(rho) = 1/2 + arcsin(rho) / pi

which is the orthant probability of a bivariate normal with BOTH thresholds at
zero.  Two things about it are specific to the median and neither survives a
move to the tail.

1.  THE RULE ITSELF.  The Bayes rule is: predict 1 when P(b > z_q | a) > 1/2.
    For joint normals,

        P(b > z_q | a = x) = 1 - Phi( (z_q - rho*x) / sqrt(1 - rho^2) )

    which crosses one half exactly at

        c*(q) = z_q / rho,        z_q = Phi^{-1}(q)

    At q = 0.5, z_q = 0 and c* = 0: the note's "is volatility above its median
    now" IS the Bayes rule.  At q = 0.9, z_q = 1.28, and with rho near 0.45 the
    optimal cut is near 2.9 standard deviations.  Carrying the note's a > 0 rule
    to a 90th-percentile target therefore tests a MISSPECIFIED rule, and it
    would lose for a reason that has nothing to do with the referee's question.
    Both rules are run below so the difference is visible rather than assumed.

2.  THE CEILING.  With thresholds at c* and z_q rather than both at zero there
    is no arcsin form.  The benchmark here is computed by integrating

        P(a > c, b > z) = INT_c^inf phi(x) * [1 - Phi((z - rho x)/sqrt(1-rho^2))] dx

    on a fine grid.  Part 0 checks that integrator against the arcsin formula at
    q = 0.5, where the two must agree exactly; if they do not, nothing below is
    worth reading.

WHY ACCURACY STOPS MEANING ANYTHING
------------------------------------
At q = 0.9 the base rate is 0.9, so "never predict a spike" scores 90%.  Raw
accuracy is reported to show that, and the comparison is then carried on
balanced accuracy - the mean of sensitivity and specificity, which is chance at
0.5 whatever the base rate - and on AUC of the underlying margin.

WHERE THIS RUNS OUT
--------------------
The note has roughly 694 test days.  At q = 0.9 that is about 69 positive days
and, with five-day overlap in the target, on the order of 13 independent
episodes.  Part C reports the realised counts.  An answer at q = 0.9 that rests
on 13 episodes is not an answer, and the file says so where that is the case
rather than reporting a number with a confident interval around it.
"""

import os, sys
import numpy as np
from math import asin, pi, erf, sqrt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab02_delay_curve as L

SEED = 20260914
QUANTILES = [0.50, 0.65, 0.75, 0.90]
DELAYS = [0, 3, 5, 13, 21]
EPS = 1e-12
GRID = 4001                     # integration nodes over [c, 8] standard deviations


def Phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def Phi_inv(p):
    """Inverse standard normal, by bisection on Phi.  No scipy in this project."""
    lo, hi = -8.0, 8.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def joint_upper(c, z, rho):
    """P(a > c, b > z) for standard bivariate normal, by quadrature in a."""
    r = max(min(rho, 0.999999), -0.999999)
    s = sqrt(1.0 - r * r)
    x = np.linspace(c, 8.0, GRID)
    if x[-1] <= x[0]:
        return 0.0
    phi = np.exp(-0.5 * x * x) / sqrt(2.0 * pi)
    tail = np.array([1.0 - Phi((z - r * xi) / s) for xi in x])
    return float(np.trapezoid(phi * tail, x)) if hasattr(np, "trapezoid") \
        else float(np.trapz(phi * tail, x))


def accuracy_of_cut(c, z, rho):
    """P(correct) for the rule 1{a > c} against the target 1{b > z}."""
    p_both = joint_upper(c, z, rho)                 # predict 1, is 1
    p_a = 1.0 - Phi(c)                              # predict 1
    p_b = 1.0 - Phi(z)                              # is 1
    p_neither = 1.0 - p_a - p_b + p_both            # predict 0, is 0
    return p_both + p_neither


def trailing_quantile(v, q, look=L.MED_LOOK):
    import pandas as pd
    return pd.Series(v).rolling(look, min_periods=30).quantile(q).values


def balanced_accuracy(y, pred):
    p, n = y == 1, y == 0
    if p.sum() == 0 or n.sum() == 0:
        return float("nan")
    return 0.5 * (pred[p].mean() + (1 - pred[n]).mean())


def auc(y, s):
    y = np.asarray(y, float)
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ss = np.asarray(s)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def boot_ci(y, pred, rng, fn, n_boot=500):
    n = len(y)
    nb = int(np.ceil(n / L.BLOCK))
    starts = rng.integers(0, n - L.BLOCK + 1, size=(n_boot, nb))
    offs = np.arange(L.BLOCK)
    out = np.empty(n_boot)
    for i in range(n_boot):
        s = (starts[i][:, None] + offs).ravel()[:n]
        out[i] = fn(y[s], pred[s])
    return np.nanpercentile(out, [2.5, 97.5])


def main(path=None):
    df = L.load() if path is None else L.load(path)
    o, h, l, c_ = (df["open"].values, df["high"].values,
                   df["low"].values, df["close"].values)
    v, _ = L.proxy_yang_zhang(o, h, l, c_)
    n = len(v)
    logv = np.log(np.maximum(v, 1e-14))
    X = L.har_features(logv)

    # ---------------- 0. the integrator, against the closed form ----------
    print("=" * 92)
    print("0.  THE NUMERICAL BENCHMARK, AGAINST THE NOTE'S CLOSED FORM")
    print("=" * 92)
    print("At q = 0.5 the optimal cut is zero and the accuracy of the Bayes rule must")
    print("equal 1/2 + arcsin(rho)/pi exactly.  If the quadrature cannot reproduce the")
    print("formula the note already uses, it cannot be trusted at any other quantile.\n")
    print(f"{'rho':>8}{'arcsin formula':>18}{'quadrature':>14}{'difference':>14}")
    worst = 0.0
    for rho in (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.80):
        closed = 0.5 + asin(rho) / pi
        quad = accuracy_of_cut(0.0, 0.0, rho)
        worst = max(worst, abs(closed - quad))
        print(f"{rho:>8.2f}{closed:>18.8f}{quad:>14.8f}{closed - quad:>+14.2e}")
    print(f"\n  largest absolute difference: {worst:.2e}")
    if worst > 1e-6:
        print("  THE QUADRATURE DOES NOT REPRODUCE THE CLOSED FORM - stop here.")
        return
    print("  The quadrature reproduces the closed form, so it can be used off the median.")

    # ---------------- A. the rule theory implies --------------------------
    print("\n" + "=" * 92)
    print("A.  THE BAYES CUT MOVES WITH THE QUANTILE")
    print("=" * 92)
    print("c*(q) = z_q / rho.  The note's rule is a > 0, which is c* only at q = 0.5.")
    print("rho is the in-sample correlation between the delayed and future states,")
    print("reported here to show the size of the misspecification, not used as a")
    print("forecast - the rules below estimate their cut from training data alone.\n")
    print(f"{'q':>6}{'z_q':>9}{'delta':>7}{'rho':>9}{'c*(q)':>10}"
          f"{'ceiling':>10}{'note rule':>11}")
    for q in QUANTILES:
        Q = trailing_quantile(v, q)
        y = L.make_target(v, Q)
        M = L.trailing_median(v)
        start = L.TRAIN + max(DELAYS) + L.HORIZON + L.MED_LOOK
        idx = np.arange(start, len(y))
        idx = idx[~np.isnan(y[idx])]
        z = Phi_inv(q)
        z = 0.0 if abs(z) < 1e-9 else z
        for d in (0, 5):
            a = np.log(np.maximum(v[idx - d], EPS) / np.maximum(M[idx - d], EPS))
            b = np.log(np.maximum(v[idx + L.HORIZON], EPS)
                       / np.maximum(M[idx], EPS))
            ok = np.isfinite(a) & np.isfinite(b)
            rho = float(np.corrcoef(a[ok], b[ok])[0, 1])
            cstar = z / rho if abs(rho) > 1e-9 else float("inf")
            # standardise the cut back onto a's own scale
            mu, sd = a[ok].mean(), a[ok].std()
            ceil = accuracy_of_cut(cstar, z, rho)
            note = accuracy_of_cut((0.0 - mu) / sd, z, rho)
            print(f"{q:>6.2f}{z:>9.3f}{d:>7}{rho:>9.3f}{cstar:>10.2f}"
                  f"{ceil:>10.2%}{note:>11.2%}")
    print("\n  At q = 0.50 the two right-hand columns coincide: the note's rule IS the")
    print("  Bayes rule there.  Where they separate, carrying the median rule to a tail")
    print("  target costs the difference, and that cost is a property of the rule being")
    print("  wrong for the question, not of the delay.")

    # ---------------- B. measured, not assumed ----------------------------
    print("\n" + "=" * 92)
    print("B.  MEASURED PERFORMANCE AT EACH QUANTILE")
    print("=" * 92)
    print("acc is raw accuracy and is shown only to demonstrate that it stops being")
    print("informative: 'majority' is the accuracy of never predicting the event.")
    print("bal is balanced accuracy, chance 0.500 at every base rate.  AUC ranks by")
    print("the margin.  The fitted rule estimates rho and c* from training data only.\n")
    counts, cells = {}, {}
    for q in QUANTILES:
        Q = trailing_quantile(v, q)
        y = L.make_target(v, Q)
        M = L.trailing_median(v)
        start = L.TRAIN + max(DELAYS) + L.HORIZON + L.MED_LOOK
        idx = np.arange(start, len(y))
        idx = idx[~np.isnan(y[idx])]
        yb = y[idx]
        counts[q] = (len(idx), int(yb.sum()))
        z = Phi_inv(q)
        print(f"  q = {q:.2f}   base rate {yb.mean():.3f}   "
              f"{int(yb.sum())} positive days of {len(idx)}")
        print(f"{'delta':>8}{'rule':>14}{'acc':>9}{'majority':>10}{'bal':>8}"
              f"{'95% CI':>18}{'AUC':>8}")
        for d in DELAYS:
            a_full = np.log(np.maximum(v - 0.0, EPS))
            a = np.log(np.maximum(v[idx - d], EPS) / np.maximum(M[idx - d], EPS))
            maj = np.full(len(idx), 0.0)          # never predict the event
            rows = {"note a>0": (a > 0).astype(float)}

            # fitted cut: estimate rho and the sd of a on the training window only
            fit = np.empty(len(idx)); bh = None
            for j, t in enumerate(idx):
                if j % L.REFIT == 0 or bh is None:
                    last = t - d - L.HORIZON
                    tr = np.arange(max(0, last - L.TRAIN), last)
                    at = np.log(np.maximum(v[tr - d], EPS)
                                / np.maximum(M[tr - d], EPS))
                    bt = np.log(np.maximum(v[tr + L.HORIZON], EPS)
                                / np.maximum(M[tr], EPS))
                    m2 = np.isfinite(at) & np.isfinite(bt)
                    if m2.sum() > 50:
                        r = float(np.corrcoef(at[m2], bt[m2])[0, 1])
                        mu, sd = at[m2].mean(), at[m2].std() + 1e-12
                        bh = mu + sd * (z / r) if abs(r) > 1e-6 else np.inf
                    else:
                        bh = np.inf
                fit[j] = 1.0 if a[j] > bh else 0.0
            rows["fitted cut"] = fit

            # HAR logistic on the same target
            pr = np.empty(len(idx)); bb = None
            for j, t in enumerate(idx):
                if j % L.REFIT == 0 or bb is None:
                    last = t - d - L.HORIZON
                    tr = np.arange(max(0, last - L.TRAIN), last)
                    tr = tr[~np.isnan(y[tr]) & ~np.isnan(X[tr]).any(axis=1)]
                    bb = L.logistic_fit(X[tr], y[tr]) if len(tr) > 50 else None
                pr[j] = 0.0 if (bb is None or np.isnan(X[t - d]).any()) \
                    else float(L.logistic_predict(bb, X[t - d][None, :])[0] > 0.5)
            rows["HAR logistic"] = pr

            for name, pred in rows.items():
                bal = balanced_accuracy(yb, pred)
                lo, hi = boot_ci(yb, pred, np.random.default_rng(SEED + d),
                                 balanced_accuracy)
                au = auc(yb, a) if name != "HAR logistic" else auc(yb, pr)
                cells[(q, d, name)] = (float((pred == yb).mean()), bal, lo, hi, au)
                print(f"{d:>8}{name:>14}{(pred == yb).mean():>9.2%}"
                      f"{(maj == yb).mean():>10.2%}{bal:>8.3f}"
                      + f"[{lo:>6.3f},{hi:>6.3f}]".rjust(18) + f"{au:>8.3f}")
        print("    (the two threshold rules share an AUC by construction: they cut the")
        print("     same margin at different points, so they induce the same ranking)\n")

    # ---------------- C. which objective, and can we tell? ----------------
    print("=" * 92)
    print("C.  THE TWO RULES OPTIMISE DIFFERENT THINGS, AND THE TABLE SHOWS IT")
    print("=" * 92)
    print("The fitted cut maximises 0-1 accuracy; away from the median that means")
    print("almost never calling the event.  The note's a>0 rule calls it about half")
    print("the time, so it catches events at the cost of false alarms.  Which one is")
    print("'better' is a choice of objective, not a fact, and both are reported.\n")
    print(f"{'q':>6}{'delta':>7}{'best on accuracy':>20}{'best on balanced':>20}"
          f"{'same winner?':>14}")
    disagree = 0
    NAMES = ["note a>0", "fitted cut", "HAR logistic"]
    for q in QUANTILES:
        for d in DELAYS:
            acc_w = max(NAMES, key=lambda k: cells[(q, d, k)][0])
            bal_w = max(NAMES, key=lambda k: cells[(q, d, k)][1])
            same = acc_w == bal_w
            disagree += (not same)
            print(f"{q:>6.2f}{d:>7}{acc_w:>20}{bal_w:>20}{'yes' if same else 'NO':>14}")
    tot = len(QUANTILES) * len(DELAYS)
    print(f"\n  cells where the two objectives choose different rules: {disagree} of {tot}")
    if disagree:
        print("  So the referee's question has no single answer: at a tail target the rule")
        print("  that is best at being right most often is not the rule that is best at")
        print("  catching the event, and the note's median rule is on the catching side.")

    # ---------------- D. is any of it resolvable? -------------------------
    print("\n" + "=" * 92)
    print("D.  CAN THE DATA SEPARATE ANY OF IT?")
    print("=" * 92)
    print("Two questions, both answered from the balanced-accuracy intervals in part B:")
    print("is a rule better than chance, and is one rule separable from another?")
    print("Intervals are block-bootstrapped, so they carry the 5-day overlap.\n")
    print(f"{'q':>6}{'positives':>11}{'episodes ~':>12}"
          f"{'rules above chance':>20}{'pairs separated':>17}")
    for q in QUANTILES:
        nd, npos = counts[q]
        above = sep = 0
        for d in DELAYS:
            for k in NAMES:
                if cells[(q, d, k)][2] > 0.5:
                    above += 1
            for a_, b_ in ((0, 1), (0, 2), (1, 2)):
                ka, kb = NAMES[a_], NAMES[b_]
                la, ha = cells[(q, d, ka)][2], cells[(q, d, ka)][3]
                lb, hb = cells[(q, d, kb)][2], cells[(q, d, kb)][3]
                if la > hb or lb > ha:
                    sep += 1
        print(f"{q:>6.2f}{npos:>11}{npos // L.BLOCK:>12}"
              f"{str(above) + ' of ' + str(3 * len(DELAYS)):>20}"
              f"{str(sep) + ' of ' + str(3 * len(DELAYS)):>17}")
    print("\n  'Rules above chance' counts (rule, delay) cells whose balanced-accuracy")
    print("  interval lies entirely above 0.5.  'Pairs separated' counts pairs of rules")
    print("  at the same delay whose intervals do not overlap - a conservative test, and")
    print("  a deliberately demanding one.")
    print("\n  Read together with part C: where rules are above chance but not separated")
    print("  from each other, the honest statement is that the tail target is predictable")
    print("  and that this sample cannot say which rule predicts it best.  Nine years")
    print("  does not contain enough independent volatility spikes to settle that, and")
    print("  no change of estimator fixes a shortage of events.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
