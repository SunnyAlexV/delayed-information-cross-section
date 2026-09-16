"""
lab44_horizon_matched_iv.py - the horse race, with the horizons matched.

Imports lab05_robustness and lab08_implied_vol; keep all three in labs/.
Runtime about six minutes.

THE OBJECTION, IN ITS SHARPEST FORM
-----------------------------------
Section 8 races the foreign cross-section against implied volatility and finds
the cross-section adds nothing measurable once implied volatility is held.  A
referee raised the obvious asymmetry: the target is realised variance five
trading days ahead, and the instrument racing against it is VIX, which prices a
thirty-day horizon.  A thirty-day option-implied variance is not the natural
forecast of a five-day realised variance, so the race was run with a horse
suited to a different distance.

Cboe publishes VIX9D, a nine-day index on the same underlying, computed the
same way, free, from January 2011.  Nine days against a five-day target is not
exact either, but it is the closest listed instrument in existence and it is
four times closer than thirty.  The objection is therefore testable rather than
arguable, and this lab tests it.

WHAT IS AT STAKE, IN BOTH DIRECTIONS
------------------------------------
    If the foreign block is STILL redundant given nine-day implied volatility,
    Section 8's conclusion has survived the strongest version of the objection
    that can be put to it with listed data, and the paper can say so.

    If the redundancy WEAKENS once the horizons match, then part of what
    Section 8 reads as redundancy was horizon mismatch: the thirty-day
    instrument was leaving something on the table that the cross-section was
    picking up, and matching the horizon takes it away.  That is a correction to
    this paper's own central section and it is better found here than by a
    referee.

Either outcome is reportable.  Neither is the outcome this lab is looking for.

THE DESIGN, AND WHY THE COMPARISON IS FAIR
------------------------------------------
The nine-day and thirty-day blocks are given the SAME number of regressors -
one implied-volatility series each, plus VDAX in both - so the estimation cost
lab07 measures is identical on both sides and cancels in the difference.  That
is the equal-width design this paper uses wherever two representations are
compared, and it is what makes a Giacomini-White statistic on the difference
between two non-nested models readable.

THE COST, STATED BEFORE THE RESULT
----------------------------------
VIX9D begins in 2011, and the decision variable needs a 252-day trailing median
before it exists at all, and the walk-forward needs 1,500 days of labelled
history before it can fit.  The usable test window is therefore about half of
Section 8's, and - this is the part that matters - it begins after the 2008
crisis.  It contains the 2020 episode and not the 2008 one, and the 2008 one is
where being late costs a forecaster most.  So this lab sharpens the test on a
calmer sample; it does not replace Table 8, and part A0 measures how much of any
difference is the window rather than the horizon.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab08_implied_vol as IV

TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 5, 13, 21, 55]


def dvar(series, index, lag):
    """The paper's implied-volatility decision variable: log level over median."""
    on = series.reindex(index).ffill(limit=5)
    if lag:
        on = on.shift(lag)
    return np.log(on / on.rolling(L.MED, min_periods=30).median())


def blocks(folder):
    D, peers, lag = L.build(folder, TARGET)
    print("  implied-volatility files and seams:")
    raw = {t: IV.load_iv(folder, t) for t in ("VIX", "VIX9D", "VDAX")}
    col = pd.DataFrame(index=D.index)
    col["VIX"] = dvar(raw["VIX"], D.index, IV.VIX_LAG_STRICT)
    col["VIX9D"] = dvar(raw["VIX9D"], D.index, IV.VIX_LAG_STRICT)
    col["VDAX"] = dvar(raw["VDAX"], D.index, 0)
    # The term-structure slope, one column: how steep the curve is between nine
    # and thirty days, which is a different object from either level.
    on9 = raw["VIX9D"].reindex(D.index).ffill(limit=5).shift(IV.VIX_LAG_STRICT)
    on30 = raw["VIX"].reindex(D.index).ffill(limit=5).shift(IV.VIX_LAG_STRICT)
    col["SLOPE"] = np.log(on9 / on30)
    return D, peers, col


def run(D, peers, col, names, delays, label):
    """Walk-forward every arm in `names` at every delay. Returns R2 and errors."""
    own, P, _V, y, idx = IV.make(D, col, peers)
    yb = y[idx]
    print(f"  {label}: {len(idx)} test days, {D.index[idx[0]].date()} to "
          f"{D.index[idx[-1]].date()}")
    mats = {k: col[v].values for k, v in names.items()}
    out = {}
    for d in delays:
        out[d] = {}
        for k, M in mats.items():
            f = IV.walk(own, [M], y, idx, d)
            out[d][k] = f
        out[d]["own"] = IV.walk(own, [], y, idx, d)
        for k, M in mats.items():
            out[d]["F+" + k] = IV.walk(own, [P, M], y, idx, d)
        out[d]["F"] = IV.walk(own, [P], y, idx, d)
    return out, yb, len(idx), D.index[idx[0]], D.index[idx[-1]]


def z_of(yb, f_small, f_big):
    """GW on the paired loss differential; positive favours f_big."""
    dl = (yb - f_small) ** 2 - (yb - f_big) ** 2
    return dl.mean() / IV.hac_se(dl)


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints the horizon-matched implied-volatility race, Section 8.\n")

    D, peers, col = blocks(folder)
    ARMS = {"IV30": ["VIX", "VDAX"], "IV9": ["VIX9D", "VDAX"],
            "TERM": ["VIX9D", "VIX", "VDAX"], "SLOPE": ["SLOPE", "VDAX"]}

    # ---------------- A0 ---------------------------------------------------
    print("\n" + "=" * 100)
    print("A0.  HOW MUCH OF ANY DIFFERENCE IS THE WINDOW RATHER THAN THE HORIZON?")
    print("=" * 100)
    print("The thirty-day block alone, on the window Table 8 uses and on the shorter")
    print("one VIX9D forces. Nothing else changes, so the gap between these two")
    print("columns is the sample and nothing but the sample.\n")
    full = col[["VIX", "VDAX"]].notna().all(axis=1) & D[TARGET].notna()
    short = col.notna().all(axis=1) & D[TARGET].notna()
    wide, ybw, nw, aw, bw = run(D.loc[full], peers, col.loc[full],
                                {"IV30": ["VIX", "VDAX"]}, DELAYS, "full window")
    narrow, ybn, nn, an, bn = run(D.loc[short], peers, col.loc[short],
                                  ARMS, DELAYS, "VIX9D window")
    print(f"\n{'delta':>6}{'+IV30 full':>13}{'+IV30 short':>14}{'difference':>13}")
    for d in DELAYS:
        a = IV.r2(ybw, wide[d]["IV30"])
        b = IV.r2(ybn, narrow[d]["IV30"])
        print(f"{d:>6}{a:>13.4f}{b:>14.4f}{b - a:>+13.4f}")
    print(f"\n  full window   {nw} days, {aw.date()} to {bw.date()}")
    print(f"  VIX9D window  {nn} days, {an.date()} to {bn.date()}, "
          f"{nn / nw:.0%} of the days")
    print("  Every comparison below is inside the VIX9D window, so the sample is held")
    print("  fixed and the horizon is the only thing that varies.")

    # ---------------- A ----------------------------------------------------
    print("\n" + "=" * 100)
    print("A.  DOES MATCHING THE HORIZON MAKE THE INSTRUMENT BETTER?")
    print("=" * 100)
    print("Both blocks carry two series, so the estimation cost is identical and")
    print("cancels. 'IV9 - IV30' positive means nine-day implied volatility")
    print("forecasts five-day realised variance better than thirty-day does.\n")
    print(f"{'delta':>6}{'own':>9}{'+IV30':>9}{'+IV9':>9}{'+TERM':>9}"
          f"{'IV9 - IV30':>13}{'GW z':>8}{'TERM - IV30':>14}{'GW z':>8}")
    better = 0
    for d in DELAYS:
        r_own = IV.r2(ybn, narrow[d]["own"])
        r30 = IV.r2(ybn, narrow[d]["IV30"])
        r9 = IV.r2(ybn, narrow[d]["IV9"])
        rt = IV.r2(ybn, narrow[d]["TERM"])
        z9 = z_of(ybn, narrow[d]["IV30"], narrow[d]["IV9"])
        zt = z_of(ybn, narrow[d]["IV30"], narrow[d]["TERM"])
        better += int(z9 > 1.96)
        print(f"{d:>6}{r_own:>9.4f}{r30:>9.4f}{r9:>9.4f}{rt:>9.4f}"
              f"{r9 - r30:>+13.4f}{z9:>8.2f}{rt - r30:>+14.4f}{zt:>8.2f}")
    print(f"\n  nine-day beats thirty-day significantly at {better} of "
          f"{len(DELAYS)} delays")

    # ---------------- B ----------------------------------------------------
    print("\n" + "=" * 100)
    print("B.  THE QUESTION SECTION 8 ASKS, WITH THE HORIZON MATCHED")
    print("=" * 100)
    print("What the seven foreign closes add ON TOP of each implied-volatility block.")
    print("Each block is nested inside its own 'with foreign' counterpart, so GW is")
    print("the valid test and Diebold-Mariano would not be.\n")
    print(f"{'delta':>6}{'fgn | IV30':>12}{'GW z':>8}{'fgn | IV9':>12}{'GW z':>8}"
          f"{'fgn | TERM':>13}{'GW z':>8}")
    pos = {k: 0 for k in ("IV30", "IV9", "TERM")}
    inc = {}
    for d in DELAYS:
        row = f"{d:>6}"
        for k, w in (("IV30", 12), ("IV9", 12), ("TERM", 13)):
            g = IV.r2(ybn, narrow[d]["F+" + k]) - IV.r2(ybn, narrow[d][k])
            z = z_of(ybn, narrow[d][k], narrow[d]["F+" + k])
            inc[(d, k)] = (g, z)
            pos[k] += int(z > 1.96)
            row += f"{g:>+{w}.4f}{z:>8.2f}"
        print(row)
    print("\n  delays where the foreign block significantly HELPS, given each block:")
    for k in ("IV30", "IV9", "TERM"):
        print(f"    given {k:>5}: {pos[k]} of {len(DELAYS)}")
    print("\n  A positive, significant column here would overturn Section 8.")

    # ---------------- B2 ---------------------------------------------------
    # The window holds exactly one violent episode, and it is the episode where a
    # front-tenor index should separate from a back one.  A horizon result read
    # off a sample whose only stress is March 2020 could be a fact about 2020,
    # and nothing above would show it.  So the same comparison is run again with
    # that episode removed - not resampled around, removed - which is the
    # bluntest possible version of the test and the hardest to argue with.
    print("\n" + "=" * 100)
    print("B2.  IS THE HORIZON RESULT JUST MARCH 2020?")
    print("=" * 100)
    print("The one stress episode in this window is dropped outright: every test day")
    print("from 15 February to 30 June 2020 is deleted from the evaluation, the models")
    print("are untouched, and the two blocks are compared on what is left.\n")
    # Same call `run` made for the narrow window, so idxn indexes the same days
    # the `narrow` forecasts are aligned to.
    _o, _P, _V, _y, idxn = IV.make(D.loc[short], col.loc[short], peers)
    dts = D.loc[short].index[idxn]
    covid = (dts >= pd.Timestamp("2020-02-15")) & (dts <= pd.Timestamp("2020-06-30"))
    keep2 = ~covid
    print(f"  {int(covid.sum())} test days removed, {int(keep2.sum())} remain")
    print(f"\n{'delta':>6}{'+IV30':>9}{'+IV9':>9}{'IV9 - IV30':>13}{'GW z':>8}"
          f"{'fgn | IV9':>12}{'GW z':>8}")
    kept = 0
    for d in DELAYS:
        yk = ybn[keep2]
        r30 = IV.r2(yk, narrow[d]["IV30"][keep2])
        r9 = IV.r2(yk, narrow[d]["IV9"][keep2])
        z9 = z_of(yk, narrow[d]["IV30"][keep2], narrow[d]["IV9"][keep2])
        g9 = IV.r2(yk, narrow[d]["F+IV9"][keep2]) - r9
        zg = z_of(yk, narrow[d]["IV9"][keep2], narrow[d]["F+IV9"][keep2])
        kept += int(z9 > 1.96)
        print(f"{d:>6}{r30:>9.4f}{r9:>9.4f}{r9 - r30:>+13.4f}{z9:>8.2f}"
              f"{g9:>+12.4f}{zg:>8.2f}")
    print(f"\n  nine-day still beats thirty-day significantly at {kept} of "
          f"{len(DELAYS)} delays with 2020 removed")
    if kept == len(DELAYS):
        print("  The horizon result is not an artefact of one episode. It survives")
        print("  deleting the only violent stretch the window contains.")
    elif kept == 0:
        print("  The horizon result does not survive removing 2020, so what Table 12")
        print("  measures is the behaviour of the term structure in one crisis rather")
        print("  than a general property of the matched horizon.")
    else:
        print("  The horizon result survives at some delays and not others once 2020 is")
        print("  removed. The table above is the split and it belongs in the paper.")

    # ---------------- C ----------------------------------------------------
    print("\n" + "=" * 100)
    print("C.  THE SLOPE ON ITS OWN, AND WHAT EACH BLOCK RECOVERS")
    print("=" * 100)
    print("The slope column replaces the nine-day LEVEL with the nine-to-thirty-day")
    print("SLOPE, keeping the regressor count at two. If the slope carries what the")
    print("nine-day level carries, the information is in the shape of the curve")
    print("rather than in its near end.\n")
    print(f"{'delta':>6}{'+SLOPE':>9}{'+IV9':>9}{'SLOPE - IV9':>14}{'GW z':>8}")
    for d in DELAYS:
        rs = IV.r2(ybn, narrow[d]["SLOPE"])
        r9 = IV.r2(ybn, narrow[d]["IV9"])
        print(f"{d:>6}{rs:>9.4f}{r9:>9.4f}{rs - r9:>+14.4f}"
              f"{z_of(ybn, narrow[d]['IV9'], narrow[d]['SLOPE']):>8.2f}")

    print(f"\n  the substitution rate each block recovers, on this window")
    print(f"{'delta':>6}{'R(d) foreign':>14}{'R(d) IV30':>12}{'R(d) IV9':>11}"
          f"{'R(d) TERM':>12}")
    base = IV.r2(ybn, narrow[0]["own"])
    for d in DELAYS[1:]:
        o = IV.r2(ybn, narrow[d]["own"])
        den = base - o
        f = lambda key: (f"{(IV.r2(ybn, narrow[d][key]) - o) / den:.0%}"
                         if den > 1e-9 else "n/a")
        print(f"{d:>6}{f('F'):>14}{f('IV30'):>12}{f('IV9'):>11}{f('TERM'):>12}")

    # ---------------- verdict ----------------------------------------------
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    survives = pos["IV9"] == 0 and pos["TERM"] == 0
    if survives and better == 0:
        print("  Matching the horizon does not measurably improve the instrument, and")
        print("  the foreign cross-section adds nothing detectable on top of either")
        print("  block at any delay. Section 8's conclusion survives the sharpest")
        print("  version of the objection available with listed data, and the")
        print("  thirty-day index was not the reason it held.")
    elif survives:
        print(f"  Nine-day implied volatility is the better instrument at {better} of")
        print(f"  {len(DELAYS)} delays, which is the referee's point and it is correct.")
        print("  It does not change the conclusion: the foreign cross-section still")
        print("  adds nothing detectable on top of it. The horizon mismatch made the")
        print("  paper's implied-volatility control WEAKER than it should have been,")
        print("  and redundancy held even against the weaker control.")
    else:
        print(f"  The foreign block significantly helps at {pos['IV9']} delays given the")
        print("  nine-day block, against")
        print(f"  {pos['IV30']} given the thirty-day one. Part of what Section 8 reads as")
        print("  redundancy was horizon mismatch, and the section must be rewritten")
        print("  around the horizon-matched instrument rather than around VIX.")
    print("\n  The window is the standing caveat: it begins after 2008 and cannot")
    print("  speak to the episode where delay is most expensive. Table 8 remains the")
    print("  paper's headline for that reason, and this lab is the sharper test on")
    print("  the sample that exists.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
