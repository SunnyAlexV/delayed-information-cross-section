"""
lab02b_threshold_ceiling.py - how close does the threshold rule get to its own
ceiling?  Imports lab02_delay_curve; keep both in labs/.  Runtime ~1 minute.

THE QUESTION
------------
lab02 measures how accurate a median-threshold rule is at each delay.  It does
not say whether that accuracy is *good*.  A rule scoring 67.9% could be leaving
a great deal on the table or almost none, and the difference decides whether the
companion note's conclusion - that breadth, not a better univariate rule, is the
only thing left to buy - is earned or asserted.

So compute the ceiling the rule is aiming at.

  a = log( YZ_{t-d} / M_{t-d} )     the decision variable, as of t-delta
  b = log( CC_{t+h} / M_t )         the outcome, on the scale the label uses

The label is 1{b > 0} and the rule is 1{a > 0}.  If (a, b) were jointly normal
with correlation rho, the probability that two mean-zero normals share a sign is
the orthant probability

    P(sign a = sign b) = 1/2 + arcsin(rho) / pi                            (*)

and - this is the part that matters - under joint normality the sign rule is
also the OPTIMAL rule, because P(b > 0 | a) is monotone in a and crosses 1/2
exactly at a = 0.  There is no threshold other than zero to tune and no
non-linear function of a that does better.  So (*) is not merely a benchmark the
rule happens to be compared against; it is the best any function of a alone can
achieve, and the gap between it and the measured accuracy is the entire value of
any further work on univariate timing.

WHY THIS IS A SEPARATE FILE FROM lab02
--------------------------------------
It was not, originally.  This began as a scratch script that read a CSV from the
author's Desktop and compared the ceiling against accuracy figures PASTED IN AS
LITERALS from an earlier run of lab02.  Both halves of that are disqualifying in
a repository whose claim is reproducibility: nobody else has the Desktop file,
and a comparison against hard-coded constants cannot detect that the thing it
compares against has changed.  It produced a table in the companion note that no
part of the repository could regenerate.

The accuracies below are therefore recomputed here, by calling lab02's own
evaluate(), on the CSV that ships in data/single_market/.  Nothing is pasted.

WHAT TO EXPECT
--------------
Normality is an approximation and the sample is finite, so measured accuracy can
land slightly ABOVE the normal-theory ceiling; readings just over 100% are
sampling noise plus non-normality, not a rule beating its own optimum.  What
would be meaningful is a persistent shortfall, and there is only one: the ceiling
stops falling after about two weeks while the rule keeps falling, so at three
weeks the rule reaches only about nine-tenths of what is available.  That gap is
the one place a better univariate rule could still pay, and it is small compared
with what the cross-section recovers over the same range.
"""

import os, sys
from math import asin, pi

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab02_delay_curve as L

DELAYS = L.DELAYS
H = L.HORIZON


def main(path=None):
    df = L.load(path or L.FILENAME)
    c = df["close"].values
    o, h, l = df["open"].values, df["high"].values, df["low"].values
    print(f"loaded {len(df)} rows, {df['date'].iloc[0].date()} "
          f"to {df['date'].iloc[-1].date()}")

    vcc, _ = L.proxy_close_to_close(c)
    vyz, _ = L.proxy_yang_zhang(o, h, l, c)
    n = min(len(vcc), len(vyz))
    vcc, vyz = vcc[:n], vyz[:n]

    Mcc, Myz = L.trailing_median(vcc), L.trailing_median(vyz)
    y = L.make_target(vcc, Mcc)                       # 1{CC_{t+h} > M_t}

    # --- accuracies, RECOMPUTED, not pasted -------------------------------
    # Only the YZ proxy is needed here, so evaluate() is given that alone;
    # the target is still lab02's, built from CC.
    N = len(y)
    acc, ybar, idx = L.evaluate(
        y,
        {"YZ": L.har_features(np.log(np.maximum(vyz[:N], 1e-14)))},
        {"YZ": Myz[:N]},
        {"YZ": vyz[:N]},
    )
    print(f"test window: {len(idx)} days, {df['date'].iloc[idx[0]].date()} "
          f"to {df['date'].iloc[idx[-1]].date()};  base rate {ybar.mean():.3f}\n")

    print("Decision variables:  a = log(YZ_{t-d}/M_{t-d})   b = log(CC_{t+h}/M_t)")
    print("Target is 1{b>0}; the threshold rule is 1{a>0}.  Under joint normality")
    print("the threshold rule IS the optimal rule, since P(b>0|a) crosses 1/2")
    print("exactly at a=0, so the ceiling below is what ANY function of a can reach.\n")

    print(f"{'delta':>6}{'corr(a,b)':>11}{'ceiling':>10}"
          f"{'persist-YZ':>12}{'har-YZ':>9}{'% of ceiling':>14}")
    for d in DELAYS:
        a = np.log(vyz[idx - d]) - np.log(Myz[idx - d])
        b = np.log(vcc[idx + H]) - np.log(Mcc[idx])
        ok = np.isfinite(a) & np.isfinite(b)
        rho = float(np.corrcoef(a[ok], b[ok])[0, 1])
        ceil = 0.5 + asin(max(min(rho, 1.0), -1.0)) / pi
        obs = (acc[("persist-YZ", d)] == ybar).mean() * 100
        har = (acc[("har-YZ", d)] == ybar).mean() * 100
        print(f"{d:>6}{rho:>11.4f}{ceil*100:>9.2f}%{obs:>11.2f}%{har:>8.2f}%"
              f"{obs/(ceil*100):>13.1%}")

    print("\nReadings over 100% are non-normality and sampling noise, not a rule")
    print("beating its own optimum.  The one real shortfall is at delta = 21,")
    print("where the ceiling has flattened but the rule has not.")
    print("\nDone.  Every number above came from this file.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
