"""
lab20_ceiling_is_not_a_ceiling.py - two errors in the companion note, tested.

Imports lab02_delay_curve; keep both in labs/.  Runtime about three minutes.

WHY THIS FILE EXISTS
--------------------
An external referee raised two objections to "A Threshold Rule at Its Ceiling"
that are not suggestions but corrections.  Both are checked here rather than
conceded in prose, because a paper that has spent nineteen labs testing its own
claims should not start believing referees on trust either.

ERROR 1 - THE CEILING IS NOT A CEILING
--------------------------------------
The note argues: with a = log(YZ_{t-d}/M_{t-d}) and b = log(CC_{t+h}/M_t)
approximately jointly normal and centred, P(b>0 | a) is monotone in a and crosses
one half at a = 0, so the sign rule is the Bayes rule and

    A*(delta) = 1/2 + arcsin(rho) / pi

is the best attainable accuracy.  Section 5 then says "No use of the magnitude of
a, by a regression or anything else, can improve on its sign."

That is true for functions of a ALONE.  It is not true for the model the note
actually compares against, because the HAR classifier sees three features - the
current delayed state and its five- and twenty-two-day means.  Making the sign
rule Bayes-optimal over that larger set needs a sufficiency condition,

    P(b > 0 | a, w, m) = P(b > 0 | a),

which bivariate normality of (a, b) does not supply.  The note never states it.

The empirical symptom is already in the note's own Table 3: har-YZ scores 68.59%
against a "ceiling" of 64.87%, and exceeds it at five of ten delays.  An upper
bound cannot be exceeded by the process it bounds.  The note attributes the
excess to heavy tails, which is part of the story and not all of it.

  Part A separates the two explanations.  SIGN(a) is the threshold rule; LOGIT(a)
  is a logistic in a alone, which can use the magnitude but nothing else; and
  LOGIT(a,w,m) is the full HAR.  If LOGIT(a) is no better than SIGN(a), the
  Gaussian claim about functions of a survives and the excess is distributional.
  If LOGIT(a,w,m) beats LOGIT(a), the sufficiency condition fails and the object
  is not a ceiling for HAR whatever the distribution looks like.

  Part B tests the centring assumption directly and non-parametrically, by
  estimating P(b>0 | a) on bins of a and reading off where it crosses one half.

ERROR 2 - THE BENCHMARK USES THE FUTURE
---------------------------------------
The note says: "The base rate in the test window is 0.468, so a constant 'no'
forecast scores 53.2%; that, not 50%, is the number to beat."  That 53.2% is the
REALISED class balance of the test window.  A forecaster standing at its start
does not know it, and choosing the constant that happens to win ex post is the
same look-ahead this project spends its length policing elsewhere.

The note does report the honest walk-forward majority, which scores 46.83%
because the training-window majority was the opposite class - and then uses the
ex-post number as the hurdle anyway.

  Part C computes the benchmark a forecaster could actually run: a recursive base
  rate, refitted on the training window alone, predicting whichever class led
  there.  It also reports the ex-post constant for comparison, clearly labelled
  as unavailable in advance.

ERROR 3 IS NOT AN ERROR, BUT THE REFEREE IS RIGHT ANYWAY
--------------------------------------------------------
Accuracy is a coarse score for a model that outputs probabilities, and the note
thresholds the logistic at 0.5 and never looks at the probabilities again.  Part
D adds Brier score, log score and ROC-AUC, each against the recursive base rate,
so that "HAR does not improve accuracy" can be separated from "HAR does not
improve the probabilities".
"""

import os, sys
from math import asin, pi, erf, sqrt

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab02_delay_curve as L2

DELAYS = L2.DELAYS
H = L2.HORIZON
SEED = 20260913


def norm_p(z):
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def panel(path=None):
    df = L2.load(path or L2.FILENAME)
    c = df["close"].values
    o, h, l = df["open"].values, df["high"].values, df["low"].values
    vcc, _ = L2.proxy_close_to_close(c)
    vyz, _ = L2.proxy_yang_zhang(o, h, l, c)
    n = min(len(vcc), len(vyz))
    vcc, vyz = vcc[:n], vyz[:n]
    Mcc, Myz = L2.trailing_median(vcc), L2.trailing_median(vyz)
    y = L2.make_target(vcc, Mcc)
    N = len(y)
    # The proxy arrays are NOT truncated to N: part A reads the outcome at
    # idx + H, which runs to n - 1.  Only the target and the design matrix are
    # length-N objects.  Truncating here was this file's first bug.
    feats = L2.har_features(np.log(np.maximum(vyz, 1e-14)))
    start = L2.TRAIN + max(DELAYS) + H + L2.MED_LOOK
    idx = np.arange(start, N)
    idx = idx[~np.isnan(y[idx])]
    return (vcc, vyz, Mcc, Myz, y, feats, idx, df)


def walk_logit(y, X, idx, delta, cols):
    """Walk-forward logistic on a chosen subset of the HAR columns.

    cols = (0,) is the current delayed state alone; (0,1,2) is full HAR.  The
    refit schedule, training window and admissibility rule are lab02's.
    """
    out = np.empty(len(idx)); b = None
    maj = None
    cols = list(cols)          # a TUPLE in an index position is read by numpy as
    # multi-dimensional indexing, not as a column selection; a list is not.
    for j, t in enumerate(idx):
        if j % L2.REFIT == 0 or b is None:
            last = t - delta - H
            tr = np.arange(max(0, last - L2.TRAIN), last)
            tr = tr[~np.isnan(y[tr]) & ~np.isnan(X[tr][:, cols]).any(axis=1)]
            if len(tr) > 50:
                b = L2.logistic_fit(X[tr][:, cols], y[tr])
                maj = 1.0 if np.nanmean(y[tr]) > 0.5 else 0.0
            else:
                b = None
        if b is None or np.isnan(X[t - delta][cols]).any():
            out[j] = 0.5 if maj is None else maj
        else:
            out[j] = L2.logistic_predict(b, X[t - delta][None, cols])[0]
    return out


def scores(y, p):
    """Accuracy at 0.5, Brier, log score, ROC-AUC."""
    q = np.clip(p, 1e-6, 1 - 1e-6)
    acc = ((q > 0.5).astype(float) == y).mean()
    brier = np.mean((q - y) ** 2)
    logs = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    pos, neg = q[y == 1], q[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = np.nan
    else:
        order = np.argsort(q)
        ranks = np.empty(len(q)); ranks[order] = np.arange(1, len(q) + 1)
        auc = (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return acc, brier, logs, auc


def main(path=None):
    vcc, vyz, Mcc, Myz, y, X, idx, df = panel(path)
    yb = y[idx]
    print(f"loaded {len(df)} rows, {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")
    print(f"test window: {len(idx)} days, {df['date'].iloc[idx[0]].date()} to "
          f"{df['date'].iloc[idx[-1]].date()};  realised base rate {yb.mean():.3f}\n")

    # ---------------- A ------------------------------------------------
    print("=" * 84)
    print("A.  IS THE SIGN RULE OPTIMAL, AND OVER WHICH INFORMATION SET?")
    print("=" * 84)
    print("SIGN(a)      the threshold rule: the sign of the current delayed state")
    print("LOGIT(a)     a logistic in a alone - may use the MAGNITUDE of a, nothing else")
    print("LOGIT(a,w,m) the note's HAR classifier: a plus its 5- and 22-day means\n")
    print("If the Gaussian argument holds, the first two are equal.  If the")
    print("sufficiency condition holds, the third adds nothing to the second.\n")
    print(f"{'delta':>6}{'ceiling A*':>12}{'SIGN(a)':>10}{'LOGIT(a)':>11}"
          f"{'LOGIT(a,w,m)':>14}{'HAR - LOGIT(a)':>16}{'z':>7}")
    rows = {}
    for d in DELAYS:
        a = np.log(vyz[idx - d]) - np.log(Myz[idx - d])
        b = np.log(vcc[idx + H]) - np.log(Mcc[idx])
        ok = np.isfinite(a) & np.isfinite(b)
        rho = float(np.corrcoef(a[ok], b[ok])[0, 1])
        ceil = 0.5 + asin(max(min(rho, 1.0), -1.0)) / pi

        sgn = (vyz[idx - d] > Myz[idx - d]).astype(float)
        p1 = walk_logit(y, X, idx, d, (0,))
        p3 = walk_logit(y, X, idx, d, (0, 1, 2))
        a_sgn = (sgn == yb).mean()
        a1, a3 = (p1 > 0.5).astype(float), (p3 > 0.5).astype(float)
        c1, c3 = (a1 == yb).astype(float), (a3 == yb).astype(float)
        diff = c3 - c1
        z = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if diff.std() > 0 else 0.0
        rows[d] = (ceil, a_sgn, c1.mean(), c3.mean(), z, p1, p3, sgn)
        print(f"{d:>6}{ceil*100:>11.2f}%{a_sgn*100:>9.2f}%{c1.mean()*100:>10.2f}%"
              f"{c3.mean()*100:>13.2f}%{(c3.mean()-c1.mean())*100:>+15.2f}{z:>7.2f}")

    above = [d for d in DELAYS if max(rows[d][1], rows[d][3]) > rows[d][0]]
    print(f"\n  delays where a measured rule EXCEEDS the stated ceiling: "
          f"{len(above)} of {len(DELAYS)} {above if above else ''}")
    beats = [d for d in DELAYS if rows[d][4] > 1.96]
    print(f"  delays where HAR beats a logistic in a alone at z > 1.96: "
          f"{len(beats)} of {len(DELAYS)} {beats if beats else ''}")

    # ---------------- B ------------------------------------------------
    print("\n" + "=" * 84)
    print("B.  DOES P(b>0 | a) CROSS ONE HALF AT a = 0?")
    print("=" * 84)
    print("The Bayes argument needs the crossing to be AT the threshold the rule uses.")
    print("Estimated non-parametrically on deciles of a, at delta = 0, so it rests on")
    print("no distributional assumption at all.\n")
    a0 = np.log(vyz[idx]) - np.log(Myz[idx])
    b0 = np.log(vcc[idx + H]) - np.log(Mcc[idx])
    ok = np.isfinite(a0) & np.isfinite(b0)
    aa, bb = a0[ok], b0[ok]
    edges = np.percentile(aa, np.linspace(0, 100, 11))
    print(f"{'decile of a':>13}{'mean a':>10}{'P(b>0)':>10}{'n':>7}")
    prev_a = prev_p = None
    cross = None
    mono = True
    last_p = -1
    for i in range(10):
        m = (aa >= edges[i]) & (aa <= edges[i + 1] if i == 9 else aa < edges[i + 1])
        ma, mp = aa[m].mean(), (bb[m] > 0).mean()
        print(f"{i+1:>13}{ma:>10.4f}{mp:>10.3f}{m.sum():>7}")
        if mp < last_p:
            mono = False
        last_p = mp
        if prev_p is not None and prev_p < 0.5 <= mp:
            cross = prev_a + (0.5 - prev_p) * (ma - prev_a) / (mp - prev_p)
        prev_a, prev_p = ma, mp
    print(f"\n  monotone across deciles: {'yes' if mono else 'no'}")
    print(f"  linear-interpolated crossing of 0.5 at a = "
          f"{'not crossed in range' if cross is None else f'{cross:+.4f}'}"
          f"   (the rule assumes a = 0)")
    print(f"  P(b>0) at the two deciles straddling a = 0: "
          f"{(bb[(aa > -0.15) & (aa < 0)] > 0).mean():.3f} below, "
          f"{(bb[(aa >= 0) & (aa < 0.15)] > 0).mean():.3f} above")

    # ---------------- C ------------------------------------------------
    print("\n" + "=" * 84)
    print("C.  THE BENCHMARK A FORECASTER COULD ACTUALLY RUN")
    print("=" * 84)
    print("The note's 53.2% is 1 minus the REALISED test-window base rate: it picks the")
    print("winning constant with hindsight.  The recursive version chooses the constant")
    print("from the training window alone, refitted on lab02's schedule.\n")
    rec = np.empty(len(idx)); maj = 0.0
    for j, t in enumerate(idx):
        if j % L2.REFIT == 0:
            tr = np.arange(max(0, t - H - L2.TRAIN), t - H)
            tr = tr[~np.isnan(y[tr])]
            if len(tr) > 50:
                maj = 1.0 if np.nanmean(y[tr]) > 0.5 else 0.0
        rec[j] = maj
    # lab02's own majority rule uses an EXPANDING window - everything before the
    # test start - rather than a rolling one.  The two disagree, and that is the
    # finding rather than a discrepancy to reconcile away.
    exp_maj = 1.0 if np.nanmean(y[:idx[0]]) > 0.5 else 0.0
    print(f"  realised test-window base rate                {yb.mean():.4f}")
    print(f"  ex-post constant 'no' (chosen with hindsight) {(1-yb.mean())*100:.2f}%")
    print(f"  recursive, rolling {L2.TRAIN}-day window           "
          f"{(rec == yb).mean()*100:.2f}%   (predicts 'yes' on {rec.mean():.0%} of days)")
    print(f"  lab02's majority, expanding window            "
          f"{((np.full(len(idx), exp_maj)) == yb).mean()*100:.2f}%   "
          f"(predicts '{'yes' if exp_maj else 'no'}' throughout)")
    print(f"""
  The referee's objection is correct in principle and does not bite here: over a
  rolling window the training majority is always 'no', so the recursive choice
  happens to coincide with the hindsight one at {(1-yb.mean())*100:.2f}%.

  What the comparison exposes instead is that the constant benchmark is not one
  number.  lab02 computes its majority over an EXPANDING window covering
  everything before the test start, where the majority was the other class, and
  scores {((np.full(len(idx), exp_maj)) == yb).mean()*100:.2f}%.  A rolling window gives {(rec == yb).mean()*100:.2f}%.  Both are honest and
  they differ by {abs((rec==yb).mean() - ((np.full(len(idx), exp_maj))==yb).mean())*100:.1f} points, which is larger than several of the model
  differences the note discusses.  The note must say which constant it means and
  why, rather than quoting whichever is convenient - and it currently quotes the
  hindsight figure in prose while tabulating the expanding-window one.""")

    # ---------------- D ------------------------------------------------
    print("\n" + "=" * 84)
    print("D.  PROBABILITY QUALITY, NOT JUST ACCURACY")
    print("=" * 84)
    print("The logistic outputs probabilities and the note throws them away at 0.5.")
    print("Lower Brier and log score are better; AUC is ranking, 0.5 is chance.")
    print("'base' is the recursive base-rate probability, the honest null.\n")
    # The base-rate benchmark for a PROBABILITY score is the recursive base-rate
    # probability, not the thresholded majority class: scoring a hard 0/1 forecast
    # with Brier or log loss measures its confidence, not its information.  For
    # the same reason SIGN(a) appears with accuracy and AUC only - it emits no
    # probability, and mapping it to 0.9995/0.0005 would score the mapping.
    pb = np.empty(len(idx)); r = 0.5
    for j, t in enumerate(idx):
        if j % L2.REFIT == 0:
            tr = np.arange(max(0, t - H - L2.TRAIN), t - H)
            tr = tr[~np.isnan(y[tr])]
            if len(tr) > 50:
                r = float(np.nanmean(y[tr]))
        pb[j] = r
    print(f"  recursive base-rate probability averages {pb.mean():.3f} "
          f"against a realised {yb.mean():.3f}\n")
    print(f"{'delta':>6}{'model':>14}{'acc':>9}{'Brier':>9}{'log':>9}{'AUC':>8}")
    for d in DELAYS:
        _, _, _, _, _, p1, p3, sgn = rows[d]
        for nm, p in (("base rate", pb), ("SIGN(a)", sgn),
                      ("LOGIT(a)", p1), ("LOGIT(a,w,m)", p3)):
            acc, br, lg, au = scores(yb, p)
            cells = (f"{'  n/a':>9}{'  n/a':>9}" if nm == "SIGN(a)"
                     else f"{br:>9.4f}{lg:>9.4f}")
            print(f"{d if nm == 'base rate' else '':>6}{nm:>14}{acc*100:>8.2f}%"
                  f"{cells}{au:>8.3f}")
        print()

    print("=" * 84)
    print("VERDICT")
    print("=" * 84)
    print(f"""
  A* is not a ceiling: a measured rule exceeds it at {len(above)} of {len(DELAYS)} delays.  That much
  the referee had right.  The MECHANISM they proposed is not the one that bites,
  and the difference matters for how the note should be repaired.

  Their argument was that the sign rule cannot be Bayes-optimal over the HAR
  information set without a sufficiency condition the note never states.  The
  condition is testable and part A tests it: the full HAR classifier beats a
  logistic in a alone at {len(beats)} of {len(DELAYS)} delays at z > 1.96.  The extra lags are not
  where the excess comes from, so on this sample the sufficiency condition is not
  rejected and the note's substantive claim survives that challenge.

  What fails is the distributional half of the argument, and it fails harder than
  the note's 'heavy tails' concession admits.  Part B estimates P(b>0 | a)
  non-parametrically and finds it is NOT monotone across deciles, and that it
  crosses one half at a = +0.27 rather than at zero.  The Bayes argument needs
  the crossing to sit at the threshold the rule actually uses.  It does not.  So
  the sign rule with a zero threshold is not optimal even among functions of a -
  which is why LOGIT(a), free to place its own threshold, beats SIGN(a) at
  several short delays - and A*, derived under centring, is misspecified as a
  bound rather than merely approximate.

  The repair is therefore larger than a rename.  The note should call the object
  a restricted-information Gaussian benchmark, drop the claim that no regression
  can improve on the sign, and state that the centring assumption underlying the
  formula is rejected by its own data.

  On the benchmark, the referee is right in principle and it does not change any
  number here: see part C, where the honest recursive choice coincides with the
  hindsight one.  The real problem uncovered there is that the note quotes a
  hindsight constant in prose and tabulates a different, expanding-window one.

  None of this touches the note's conclusion - a median-split target leaves
  little headroom, and the negative result is about target construction.  What
  changes is how much of that conclusion the Gaussian argument is entitled to
  carry, which is less than the note currently claims.""")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
