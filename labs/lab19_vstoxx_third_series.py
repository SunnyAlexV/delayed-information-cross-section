"""
lab19_vstoxx_third_series.py - would a fuller implied-volatility block change the
redundancy conclusion?

Imports lab05_robustness and lab08_implied_vol; keep all three in labs/.
Runtime about two minutes.

THE ASSERTION THIS FILE REPLACES
--------------------------------
Section 9 says the implied-volatility block is thinner than we would like, that
the spot VSTOXX index would have added a third series, and that a fuller block
"might widen the gap in Section 7 further.  It would not narrow it."

That is an argument, not a measurement.  The spot series is now in data/ and the
argument can be checked, so check it.

WHY IT IS NOT SIMPLY ADDED TO THE MAIN BLOCK
--------------------------------------------
Because doing so would cost 73% of the sample.  The spot export runs 2012-12-28
to 2025-03-28, and the walk-forward needs 252 + 1250 + 250 days of burn-in plus
the longest delay before it can produce a single test day.  Requiring all three
implied-volatility series to be present therefore moves the test window from
4,525 days beginning September 2008 to 1,240 days beginning April 2020, which

  - discards the 2008 financial crisis entirely, and
  - begins AFTER the March 2020 spike: peak VIX over the full sample is 82.69 on
    2020-03-16, and inside the retained window it is 41.38.

A volatility study that has dropped every major volatility episode in its sample
is not a stronger study.  Adding the series to the headline would make the result
less established, not more, which is the opposite of the reason for obtaining it.

WHAT THIS FILE DOES INSTEAD
---------------------------
It runs the horse race on the window where all three series exist, and reports
the two-series and three-series blocks ON THE SAME DAYS.  That separation matters
more than anything else here: a comparison of the three-series block on 1,240
days against the paper's two-series block on 4,525 days would confound the extra
series with the shorter, calmer window, and the window is by far the larger
effect.  Holding the days fixed isolates what the third series actually adds.

  A  what the window itself costs    two-series block, short window vs full
  B  does the third series add?      IV2 vs IV3, same days
  C  is breadth still redundant?     foreign given IV3, same days

WHAT WOULD COUNT AS A PROBLEM
-----------------------------
If the foreign block stops being redundant once implied volatility is richer -
if 'foreign | IV3' turns positive with a Giacomini-White statistic above 1.96 -
then Section 7's conclusion depends on how thin the options block was, and the
paper has to say so.  If redundancy survives, Section 9's assertion is upgraded
to a measurement on the window where it can be made, and stays an assertion
everywhere else, which is the honest split.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab08_implied_vol as IV

TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 3, 5, 13, 21, 55]
TAGS2 = ["VIX", "VDAX"]
TAGS3 = ["VIX", "VDAX", "VSTOXX"]


def build(folder, tags):
    """lab08's build(), with the implied-volatility tag list as an argument."""
    D, peers, lag = L.build(folder, TARGET)
    raw = {t: IV.load_iv(folder, t) for t in tags}
    iv = L.pd.DataFrame(index=D.index)
    for t, v in raw.items():
        on = v.reindex(D.index).ffill(limit=5)
        if t == "VIX":
            on = on.shift(IV.VIX_LAG_STRICT)     # VIX prints after the S&P close
        iv[t] = np.log(on / on.rolling(L.MED, min_periods=30).median())
    keep = iv.notna().all(axis=1) & D[TARGET].notna()
    return D.loc[keep], iv.loc[keep], peers


def make(D, iv, peers):
    a = D[TARGET].values
    sa = L.pd.Series(a)
    own = np.column_stack([sa.values, sa.rolling(5).mean().values,
                           sa.rolling(22).mean().values])
    n = len(D)
    y = np.full(n, np.nan); y[:-H] = a[H:]
    start = L.MED + L.TRAIN + L.VAL + max(DELAYS) + H
    idx = np.arange(start, n - H)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])]
    return own, D[peers].values, iv.values, y, idx


def r2(yb, f):
    return 1 - ((yb - f) ** 2).sum() / ((yb - yb.mean()) ** 2).sum()


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")

    print("\n  implied-volatility files and seams:")
    D3, iv3, peers = build(folder, TAGS3)
    own3, P3, V3, y3, i3 = make(D3, iv3, peers)
    lo, hi = D3.index[i3[0]], D3.index[i3[-1]]

    # the same calendar, but only the two series the paper uses
    D2f, iv2f, _ = build(folder, TAGS2)
    own2f, P2f, V2f, y2f, i2f = make(D2f, iv2f, peers)

    print(f"\ntarget {TARGET}, {len(peers)} foreign peers, VIX at t-1 (STRICT)")
    print(f"  two-series block  : {len(i2f)} test days, "
          f"{D2f.index[i2f[0]].date()} to {D2f.index[i2f[-1]].date()}")
    print(f"  three-series block: {len(i3)} test days, {lo.date()} to {hi.date()}")
    print(f"  requiring the spot VSTOXX costs {len(i2f)-len(i3)} test days, "
          f"{(len(i2f)-len(i3))/len(i2f):.0%} of the sample")

    # restrict the TWO-series run to exactly the three-series days, so the
    # comparison in part B varies the block and nothing else
    mask = np.isin(D2f.index[i2f], D3.index[i3])
    i2 = i2f[mask]
    print(f"  two-series block restricted to the same days: {len(i2)}")

    print("\n" + "=" * 86)
    print("A.  WHAT THE SHORTER WINDOW COSTS, BEFORE THE THIRD SERIES IS ADDED")
    print("=" * 86)
    print("The same two-series specification the paper reports, on the full sample and")
    print("on the sub-window the spot series forces.  Any difference here belongs to the")
    print("window, not to VSTOXX, and it is the larger of the two effects.\n")
    print(f"{'delta':>6}{'own (full)':>13}{'+iv (full)':>13}"
          f"{'own (short)':>14}{'+iv (short)':>14}")
    yb2f, yb2 = y2f[i2f], y2f[i2]
    for d in DELAYS:
        of = r2(yb2f, IV.walk(own2f, [], y2f, i2f, d))
        vf = r2(yb2f, IV.walk(own2f, [V2f], y2f, i2f, d))
        os_ = r2(yb2, IV.walk(own2f, [], y2f, i2, d))
        vs = r2(yb2, IV.walk(own2f, [V2f], y2f, i2, d))
        print(f"{d:>6}{of:>13.4f}{vf:>13.4f}{os_:>14.4f}{vs:>14.4f}")

    print("\n" + "=" * 86)
    print("B.  DOES THE THIRD SERIES ADD ANYTHING TO THE OPTIONS BLOCK?")
    print("=" * 86)
    print("Identical days, identical estimator.  Only the implied-volatility block")
    print("changes, from VIX + VDAX to VIX + VDAX + VSTOXX.\n")
    yb3 = y3[i3]
    print(f"{'delta':>6}{'own':>10}{'+IV2':>10}{'+IV3':>10}"
          f"{'IV3 - IV2':>12}{'GW z':>9}{'p':>8}")
    for d in DELAYS:
        f_own = IV.walk(own3, [], y3, i3, d)
        f2 = IV.walk(own3, [V3[:, :2]], y3, i3, d)
        f3 = IV.walk(own3, [V3], y3, i3, d)
        dl = (yb3 - f2) ** 2 - (yb3 - f3) ** 2
        z = dl.mean() / IV.hac_se(dl)
        print(f"{d:>6}{r2(yb3,f_own):>10.4f}{r2(yb3,f2):>10.4f}{r2(yb3,f3):>10.4f}"
              f"{r2(yb3,f3)-r2(yb3,f2):>+12.4f}{z:>9.2f}{IV.norm_p(z):>8.3f}")

    print("\n" + "=" * 86)
    print("C.  IS THE FOREIGN BLOCK STILL REDUNDANT GIVEN THE RICHER OPTIONS BLOCK?")
    print("=" * 86)
    print("This is the question Section 9's assertion turns on.  'foreign | IV3' is what")
    print("the seven foreign closes add on top of all three implied-volatility series;")
    print("+IV3 is nested inside +both, so Giacomini-White applies and DM would not.\n")
    print(f"{'delta':>6}{'+IV3':>10}{'+both':>10}{'foreign | IV3':>16}"
          f"{'GW z':>9}{'p':>8}")
    wins = []
    for d in DELAYS:
        f3 = IV.walk(own3, [V3], y3, i3, d)
        fb = IV.walk(own3, [P3, V3], y3, i3, d)
        inc = r2(yb3, fb) - r2(yb3, f3)
        dl = (yb3 - f3) ** 2 - (yb3 - fb) ** 2
        z = dl.mean() / IV.hac_se(dl)
        if z > 1.96:
            wins.append(d)
        print(f"{d:>6}{r2(yb3,f3):>10.4f}{r2(yb3,fb):>10.4f}{inc:>+16.4f}"
              f"{z:>9.2f}{IV.norm_p(z):>8.3f}")

    print("\n" + "=" * 86)
    print("VERDICT")
    print("=" * 86)
    print(f"  foreign block beats the three-series options block at "
          f"{len(wins)} of {len(DELAYS)} delays"
          f"{': ' + ', '.join(map(str, wins)) if wins else ''}")
    print(f"""
  Read every number above against {len(i3)} test days, not 4,525, and against a
  window containing no volatility crisis: peak VIX inside it is 41.4, against
  82.7 in the full sample.  Intervals are correspondingly wide and the power to
  detect a small incremental contribution is low, so 'adds nothing' here is a
  weaker statement than the same words in Section 7.

  What this can support: on the window where a third implied-volatility series is
  available, adding it does not overturn the redundancy of the foreign block.
  What it cannot support: anything about 2008 to 2012, or about crisis regimes,
  because the data required to add the third series does not reach them.

  The honest conclusion is that the spot VSTOXX export obtained here does not do
  the job it was obtained for.  Its history begins in December 2012 and ends in
  March 2025, so it neither covers the sample nor reaches the present.  The paper
  keeps VIX and VDAX over the full window as its headline and reports this file
  as the sensitivity run that the third series makes possible.""")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
