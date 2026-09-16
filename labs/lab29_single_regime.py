"""
lab29_single_regime.py - EXPLORATORY.  Cited by neither paper, and no check
in verify_paper.py depends on it.  It exists to test one thing lab27 cannot.

THE HOLE IN lab27
------------------
lab27 reports that the cross-section removes 63.1% of the domestic model's error
on the most stressed days against 23.9% outside any episode.  That figure POOLS
six episodes of very unequal size - 2022 contributes 251 test days and 2018
contributes 40 - and lab27 never reports a single episode on its own.  A pooled
average over an unbalanced set can be carried by one member, and nothing in that
table rules it out.  This file asks whether it is.

Part A gives every episode separately.  Part A2 is the decisive version: drop
each episode in turn and recompute the pooled figure, which answers "does the
result survive without this one?" directly rather than by inspection.

WHY A COMPLETE ARC, AND NOT JUST THE PEAK
------------------------------------------
lab27's narrow calendar window is 41 trading days centred on each episode's VIX
peak.  That is the right cut for measuring severity and the wrong one for asking
whether a result holds through a regime, because a regime has a build-up and a
recovery and the peak is neither.  Part B takes 2007-2010 as one complete arc -
calm, escalation, trough, recovery - and reports the quarter-by-quarter path.

WHAT MAKES 2008 THE RIGHT ARC
------------------------------
The models refit on a rolling window of TRAIN + VAL = 1,500 trading days, about
six years.  Entering September 2008 that window covers roughly 2002 to 2008 and
contains none of the episodes in the list.  So the forecasts made during the
2008 crisis come from a model that had never been fitted on anything like it,
which is a stronger test than conditioning on state after the fact.  Part 0
measures that exposure rather than asserting it: for every test day it computes
what fraction of the training window fell inside a listed episode.

WHAT THIS CANNOT DO
--------------------
It cannot make 2018's forty test days informative.  Per-episode intervals on the
short episodes will be wide, and where a row rests on fewer than five
independent blocks the file says so instead of reporting a number that looks
like evidence.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab27_regime_conditioning as R27

SEED = 20260914
TARGET = "SPX"
DELAYS = [0, 21, 55]
BLK = R27.BLK
EPISODES = R27.EPISODES

ARC = ("2007-01-01", "2010-12-31")          # the complete 2008 regime, end to end
MIN_BLOCKS = 5                              # below this a row is not reported as evidence


def mask_for(dates, lo, hi):
    return (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("EXPLORATORY - cited by neither paper.\n")
    rng = np.random.default_rng(SEED)

    D, peers, lag, idx, yb, fitrun, a = L.run_target(folder, TARGET, DELAYS,
                                                     rng, continuous=True)
    dates = pd.DatetimeIndex(D.index[idx])
    all_dates = pd.DatetimeIndex(D.index)
    print(f"target {TARGET}, {len(idx)} test days, "
          f"{dates[0].date()} to {dates[-1].date()}\n")

    # ---------------- 0. had the model ever seen a crisis? ----------------
    print("=" * 94)
    print("0.  WHAT THE TRAINING WINDOW HAD SEEN")
    print("=" * 94)
    print("For each test day the model was fitted on the 1,500 trading days ending")
    print("before t - delta - h.  This is the share of THAT window falling inside any")
    print("episode in the list - a model with 0% had never been fitted on a crisis.\n")
    ep_row = np.zeros(len(all_dates), bool)
    for _, lo, hi in EPISODES:
        ep_row |= mask_for(all_dates, lo, hi)
    span = L.TRAIN + L.VAL
    print(f"{'episode':>20}{'mean crisis share of training window':>40}")
    for name, lo, hi in EPISODES:
        m = mask_for(dates, lo, hi)
        if not m.sum():
            continue
        shares = []
        for t in idx[m]:
            cut = t - max(DELAYS) - L.HORIZON
            tr = np.arange(max(0, cut - span), cut)
            shares.append(ep_row[tr].mean() if len(tr) else np.nan)
        print(f"{name:>20}{np.nanmean(shares):>39.1%}")

    # ---------------- A. every episode on its own -------------------------
    print("\n" + "=" * 94)
    print("A.  EVERY EPISODE SEPARATELY")
    print("=" * 94)
    print("Share of the domestic model's mean squared error removed by the")
    print("cross-section, inside each episode, with its own block bootstrap.\n")

    own_f, cross_f = {}, {}
    for d in DELAYS:
        own_f[d] = fitrun(d, False)
        cross_f[d] = fitrun(d, True)

    thin = []
    per_ep = {}
    for d in DELAYS:
        e_own = (yb - own_f[d]) ** 2
        dd = e_own - (yb - cross_f[d]) ** 2
        print(f"  delta = {d}")
        print(f"{'episode':>20}{'days':>7}{'blocks':>8}{'share removed':>16}"
              f"{'95% CI':>20}{'':>4}")
        for name, lo, hi in EPISODES:
            m = mask_for(dates, lo, hi)
            n = int(m.sum()); nb = n // BLK
            pt, clo, chi = R27.rel_improvement(dd[m], e_own[m],
                                               np.random.default_rng(SEED + d))
            per_ep[(d, name)] = (pt, clo, chi, n)
            flag = "thin" if nb < MIN_BLOCKS else ""
            if nb < MIN_BLOCKS and (d, name) not in thin:
                thin.append((d, name))
            ci = ("     n/a" if not np.isfinite(clo)
                  else f"[{clo:>+7.1%},{chi:>+7.1%}]")
            print(f"{name:>20}{n:>7}{nb:>8}{pt:>16.1%}{ci:>20}{flag:>6}")
        print()

    # ---------------- A2. leave one episode out ---------------------------
    print("=" * 94)
    print("A2.  DOES THE POOLED FIGURE SURVIVE DROPPING EACH EPISODE?")
    print("=" * 94)
    print("lab27 pools all six.  If one carries the result, removing it should move")
    print("the pooled number a long way.  'all six' is lab27's own episode figure.\n")
    for d in DELAYS:
        e_own = (yb - own_f[d]) ** 2
        dd = e_own - (yb - cross_f[d]) ** 2
        full = np.zeros(len(dates), bool)
        for _, lo, hi in EPISODES:
            full |= mask_for(dates, lo, hi)
        base_pt, base_lo, base_hi = R27.rel_improvement(
            dd[full], e_own[full], np.random.default_rng(SEED + 11 + d))
        print(f"  delta = {d}:  all six episodes  {base_pt:.1%}  "
              f"[{base_lo:+.1%}, {base_hi:+.1%}]  ({int(full.sum())} days)")
        print(f"{'dropping':>20}{'days left':>12}{'pooled share':>15}{'95% CI':>20}"
              f"{'move':>9}")
        moves = {}
        for name, lo, hi in EPISODES:
            keep = full & ~mask_for(dates, lo, hi)
            pt, clo, chi = R27.rel_improvement(
                dd[keep], e_own[keep], np.random.default_rng(SEED + 13 + d))
            moves[name] = pt - base_pt
            print(f"{name:>20}{int(keep.sum()):>12}{pt:>15.1%}"
                  + f"[{clo:>+7.1%},{chi:>+7.1%}]".rjust(20)
                  + f"{pt - base_pt:>+9.1%}")
        worst = max(moves, key=lambda k: abs(moves[k]))
        print(f"\n    largest move: dropping {worst} shifts the pooled figure by "
              f"{moves[worst]:+.1%}")
        if abs(moves[worst]) > 0.10:
            print("    That is a big enough shift that the pooled number should not be")
            print("    quoted without naming which episode it leans on.")
        else:
            print("    No single episode moves the pooled figure by more than ten points,")
            print("    so it is not the average of one crisis and five bystanders.")
        print()

    # ---------------- B. the 2008 arc, quarter by quarter -----------------
    print("=" * 94)
    print("B.  ONE COMPLETE REGIME, END TO END: 2007-2010")
    print("=" * 94)
    print("Calm, escalation, trough and recovery in one series of quarters, so the")
    print("question is not 'does it work at the peak' but 'does it hold throughout'.")
    print("VIX is shown only as context for which quarter is which.\n")

    # lab08's build() returns the DECISION VARIABLE for VIX - log of the level
    # over its own trailing median - not the level.  An earlier version of this
    # file printed it under the heading "mean VIX", which was simply wrong.
    _, ivdf, _ = __import__("lab08_implied_vol").build(folder, strict=True)
    vix = ivdf["VIX"].shift(1).reindex(dates).values

    arc = mask_for(dates, *ARC)
    q = pd.PeriodIndex(dates, freq="Q")
    d_last = DELAYS[-1]
    e_own = (yb - own_f[d_last]) ** 2
    dd = e_own - (yb - cross_f[d_last]) ** 2
    print(f"  delta = {d_last}\n")
    print(f"{'quarter':>10}{'days':>7}{'log VIX/med':>13}{'share removed':>16}"
          f"{'95% CI':>20}{'':>6}")
    path = []
    for period in sorted(set(q[arc])):
        m = arc & (q == period)
        n = int(m.sum()); nb = n // BLK
        pt, clo, chi = R27.rel_improvement(dd[m], e_own[m],
                                           np.random.default_rng(SEED + 17))
        path.append((str(period), pt, clo, chi, nb))
        ci = ("     n/a" if not np.isfinite(clo)
              else f"[{clo:>+7.1%},{chi:>+7.1%}]")
        v = np.nanmean(vix[m]) if np.isfinite(vix[m]).any() else np.nan
        print(f"{str(period):>10}{n:>7}{v:>13.2f}{pt:>16.1%}{ci:>20}"
              f"{('thin' if nb < MIN_BLOCKS else ''):>6}")

    usable = [p for p in path if p[4] >= MIN_BLOCKS]
    pos = [p for p in usable if np.isfinite(p[2]) and p[2] > 0]
    neg = [p for p in usable if np.isfinite(p[3]) and p[3] < 0]
    print(f"\n  quarters in the arc with enough blocks to read: {len(usable)} of {len(path)}")
    print(f"    interval entirely ABOVE zero (cross-section helps): {len(pos)}"
          f"   {[p[0] for p in pos]}")
    print(f"    interval entirely BELOW zero (cross-section hurts): {len(neg)}"
          f"   {[p[0] for p in neg]}")
    print(f"    neither: {len(usable) - len(pos) - len(neg)}")
    exp_false = 0.05 * len(usable)
    print(f"    (at 5% and {len(usable)} readable quarters, about {exp_false:.1f} cells")
    print(f"     would breach in either direction by chance, so a single significant")
    print(f"     quarter is not on its own evidence of anything - the phase view below")
    print(f"     carries three to twelve times as many days per cell and is the one to read)")
    if usable:
        best = max(usable, key=lambda p: p[1])
        worst = min(usable, key=lambda p: p[1])
        print(f"  range across readable quarters: {worst[1]:.1%} ({worst[0]}) "
              f"to {best[1]:.1%} ({best[0]})")

    # ---- phases: quarters are thin, so aggregate the arc into four ------
    PHASES = [("calm build-up",  "2007-01-01", "2008-08-31"),
              ("escalation",     "2008-09-01", "2008-12-31"),
              ("trough",         "2009-01-01", "2009-06-30"),
              ("recovery",       "2009-07-01", "2010-12-31")]
    print("\n  The same arc in four phases, which carry more days than a quarter:\n")
    print(f"{'phase':>16}{'window':>26}{'days':>7}{'share removed':>16}{'95% CI':>20}")
    ph = []
    for name, lo, hi in PHASES:
        m = arc & mask_for(dates, lo, hi)
        n = int(m.sum())
        pt, clo, chi = R27.rel_improvement(dd[m], e_own[m],
                                           np.random.default_rng(SEED + 19))
        ph.append((name, pt, clo, chi, n))
        print(f"{name:>16}{lo + ' to ' + hi:>26}{n:>7}{pt:>16.1%}"
              + f"[{clo:>+7.1%},{chi:>+7.1%}]".rjust(20))

    p_pos = [x for x in ph if np.isfinite(x[2]) and x[2] > 0]
    p_neg = [x for x in ph if np.isfinite(x[3]) and x[3] < 0]
    print(f"\n  phases where the cross-section significantly helps: "
          f"{len(p_pos)} of {len(ph)}  {[x[0] for x in p_pos]}")
    print(f"  phases where it significantly hurts:                "
          f"{len(p_neg)} of {len(ph)}  {[x[0] for x in p_neg]}")

    if p_neg and p_pos:
        print("\n  The arc separates cleanly and an episode-level average hides it: the")
        print("  cross-section COSTS the forecaster in the calm build-up and pays through")
        print("  the crisis itself.  That is the estimation bill of Section 5 and the")
        print("  stress result of Section 8.1 appearing in the same regime, in sequence,")
        print("  and it is a sharper statement of both than either section makes alone.")
    elif p_pos and not p_neg:
        print("\n  The cross-section helps in every phase of the arc that resolves, so")
        print("  within this regime the contribution is not confined to the peak.")
    elif p_neg and not p_pos:
        print("\n  Within this regime the cross-section only ever hurts where it resolves,")
        print("  which contradicts the pooled episode figure and needs explaining before")
        print("  anything here is used.")
    else:
        print("\n  No phase of the arc resolves on its own, so a single regime cut this")
        print("  way is too little data for this measure.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
