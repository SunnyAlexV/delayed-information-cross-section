"""
lab18_economic_reading.py - what does a change in R-squared mean to someone
who has to act on the forecast?

Imports lab05_robustness; keep both in labs/.  Runtime about two minutes.

THE OBJECTION
-------------
The paper reports its results as R-squared and QLIKE, which are the right
quantities for establishing that something is there but say nothing about
whether the difference matters to anyone.  A referee will ask what an R-squared
of 0.347 against 0.112 buys in practice.

WHAT THIS FILE WILL NOT DO
--------------------------
It will not build a strategy.  A variance-targeting overlay, a hedging rule or
any statistic resembling a return would put a trading claim in a paper whose
sample is a retail end-of-day export with no point-in-time guarantee, whose
target is a five-day realised variance proxy rather than a traded quantity, and
which has no transaction costs anywhere in it.  Section 12 spends its length
arguing that what transfers is the apparatus rather than the numbers, and a
backtest would invite exactly the reading that section exists to prevent.  The
absence of a P&L figure here is a decision, not an omission.

WHAT IT DOES INSTEAD
--------------------
A variance forecast is used, in practice, to size something.  The quantity a
risk manager cares about is therefore not squared error on a log scale but how
far wrong the forecast is as a MULTIPLE of the truth - being out by a factor of
two matters, and it matters in a way that 0.235 of R-squared does not
communicate.  So report the distribution of the ratio

    forecast variance / realised variance

for the same two models the paper compares, at each delay, on the same days:
its median, the share of days outside a factor of two, and the share of days on
which the forecast is less than half the realised variance, which is the tail a
risk manager is actually exposed to.  Every one of those is a transformation of
forecasts the paper already produced.  Nothing new is estimated and no position
is taken.

A CORRECTION IS NEEDED FIRST, AND IT IS THE ONE lab10 ESTABLISHED
-----------------------------------------------------------------
exp() of a forecast made on a log scale is median-unbiased, not mean-unbiased,
so reading it as a variance forecast understates systematically.  The ratios
below therefore use Duan's smearing estimator computed from training residuals
at each refit, exactly as lab10 does.

One consequence is worth stating before the table rather than after it, because
it looks like a bug and is not: a smearing-corrected forecast is MEAN-unbiased,
and realised variance is strongly right-skewed, so the MEDIAN ratio sits above
one by construction.  A median of 1.26 at delta = 0 is a centred forecast of a
skewed quantity, not a forecast that is 26% too high.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L

SEED = 20260913
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 5, 21, 55]


def panel(folder):
    D, peers, lag = L.build(folder, TARGET)
    a = D[TARGET].values
    sa = L.pd.Series(a)
    own = np.column_stack([sa.values, sa.rolling(5).mean().values,
                           sa.rolling(22).mean().values])
    n = len(D)
    y = np.full(n, np.nan); y[:-H] = a[H:]
    start = L.MED + L.TRAIN + L.VAL + max(DELAYS) + H
    idx = np.arange(start, n - H)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])]
    return own, D[peers].values, y, idx


def walk(own, P, y, idx, delta, use_peers):
    """As lab05, but carrying the smearing factor out with the forecast."""
    out = np.empty(len(idx)); sm = np.ones(len(idx))
    b = mu = sd = None; s = 1.0
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), cut)
            X = np.column_stack([own[tr - delta]] + ([P[tr]] if use_peers else []))
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y[tr])
            X, yy = X[ok], y[tr][ok]
            if len(yy) < 200:
                b = None; continue
            mu, sd = X.mean(0), X.std(0) + 1e-9
            b = L.cv((X - mu) / sd, yy, "cont")
            s = float(np.mean(np.exp(yy - L.p_ridge(b, (X - mu) / sd))))
        xt = np.concatenate([own[t - delta]] + ([P[t]] if use_peers else []))
        out[j] = 0.0 if (b is None or not np.isfinite(xt).all()) \
            else L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
        sm[j] = s
    return out, sm


def report(yb, f, sm):
    """The ratio of forecast to realised variance, and what is in its tails."""
    r = np.exp(f) * sm / np.exp(yb)
    return dict(med=np.median(r),
                out2=float(np.mean((r > 2) | (r < 0.5))),
                under2=float(np.mean(r < 0.5)),
                p05=float(np.percentile(r, 5)),
                p95=float(np.percentile(r, 95)))


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    own, P, y, idx = panel(folder)
    yb = y[idx]
    print(f"target {TARGET}, {len(idx)} test days, horizon {H}\n")

    print("=" * 86)
    print("HOW FAR WRONG IS THE VARIANCE FORECAST, AS A MULTIPLE OF THE TRUTH?")
    print("=" * 86)
    print("'outside 2x' is the share of days on which the forecast is more than double")
    print("or less than half the realised variance.  'under by 2x' is the half of that")
    print("which understates - the side a risk manager is exposed to.  Smearing-")
    print("corrected as in lab10, which makes the forecast mean-unbiased; realised")
    print("variance is right-skewed, so a centred forecast sits ABOVE the median.\n")
    print(f"{'delta':>6}{'model':>9}{'median':>9}{'outside 2x':>12}"
          f"{'under by 2x':>13}{'5th pct':>10}{'95th pct':>10}")
    R = {}
    for d in DELAYS:
        for lbl, peers in (("domestic", False), ("cross", True)):
            f, sm = walk(own, P, y, idx, d, peers)
            R[(d, lbl)] = report(yb, f, sm)
            r = R[(d, lbl)]
            print(f"{d:>6}{lbl:>9}{r['med']:>9.2f}{r['out2']:>11.1%}"
                  f"{r['under2']:>13.1%}{r['p05']:>10.2f}{r['p95']:>10.2f}")
        print()

    print("=" * 86)
    print("WHAT THE CROSS-SECTION BUYS, IN THOSE TERMS")
    print("=" * 86)
    print(f"{'delta':>6}{'outside 2x: domestic':>22}{'cross':>9}{'change':>9}"
          f"{'   under by 2x: dom':>21}{'cross':>9}{'change':>9}")
    for d in DELAYS:
        a, b = R[(d, "domestic")], R[(d, "cross")]
        print(f"{d:>6}{a['out2']:>21.1%}{b['out2']:>9.1%}{b['out2']-a['out2']:>+9.1%}"
              f"{a['under2']:>20.1%}{b['under2']:>9.1%}{b['under2']-a['under2']:>+9.1%}")

    print("""
Read these as a translation of the paper's R-squared column, not as a new result:
they are the same forecasts, transformed.  Three things follow, and the third was
not what we expected to find.

The floor is high.  Even at delta = 0 roughly three days in ten sit outside a
factor of two, because five-day realised variance is a noisy object and no model
in this paper claims otherwise.  A reader expecting a volatility forecast to be
reliable within a factor of two on most days should revise that expectation
downward regardless of which model they use.

The size of the gain is real and large.  At eleven weeks the stale domestic model
is outside a factor of two on 59% of days and the cross-sectional model on 37%.
That is the same fact as the R-squared column, in units a risk manager can act
on, and it buys nothing at delta = 0 - the answer Section 5 gives on every loss.

But the gain is on the WRONG SIDE for risk management, and this is the part the
paper has to say plainly.  Splitting the tail shows the stale model does not fail
by under-forecasting; it fails by OVER-forecasting, with a median ratio drifting
from 1.26 at delta = 0 to 1.90 at eleven weeks.  Almost the entire 22-point
improvement is the cross-section pulling that overstatement back.  The
understating tail - the forecast less than half the truth, which is the one that
leaves a position too large - sits between 8.6% and 11.4% of days and is
essentially unmoved by breadth at every delay, changing by less than a point and
in the wrong direction at three of the four.  A practitioner who adopts the
cross-section to avoid being caught short by a volatility spike is adopting it
for something it does not do.  What it does is stop a stale forecaster from
systematically over-estimating risk, which is a real cost and a different one.

None of this is a trading result: there is no strategy here, no transaction cost,
and no claim that any of it survives contact with an execution desk.""")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
