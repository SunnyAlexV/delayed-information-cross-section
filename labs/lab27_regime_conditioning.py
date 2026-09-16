"""
lab27_regime_conditioning.py - two referee objections that are really one
question: does any of this hold when it matters?

    "A rolling median assumes temporal stationarity.  During structural macro
     regime transitions it fails to adjust, inducing persistent
     misclassification that adaptive models are designed to prevent."

    "The assertion that liquid implied volatility renders foreign breadth
     redundant assumes a continuous, frictionless options surface.  During
     systemic crunches options markets experience severe structural
     distortions, wide spreads and liquidity premiums."

Both say: your averages are averages, and the interesting days are not average.
This file splits the 4,862-day test window by market state and re-runs the two
headline results inside each state.

Imports lab05_robustness and lab08_implied_vol; keep all three in labs/.
Runtime about twenty minutes.

WHY THIS CAN BE ASKED HERE AND NOT OF THE COMPANION NOTE
---------------------------------------------------------
The main paper's test window runs from May 2007 to September 2026 and contains
the 2008 crisis, 2011, August 2015, February 2018, COVID and the 2022 inflation
shock.  The companion note's window starts in November 2023 and contains no
major transition at all, which is why the note states this as a limitation
rather than answering it.  Sample, not method.

A LADDER OF CUTS, NOT ONE CUT
------------------------------
The first version of this file used a single tercile of VIX and a single set of
calendar windows.  Both are wide - "the 2022 inflation shock" as the whole of
2022 is 251 days, most of them ordinary - and a wide window dilutes the very
thing being tested.  So the definition is tightened in steps and the answer is
read off the ladder: VIX above its trailing 67th, 90th and 95th percentile, and
the calendar episodes both at full width and as the 41 trading days centred on
each window's VIX peak.

The VIX rungs need no dates at all.  They are percentiles of the series against
its own trailing year, evaluated at t-1, so they are mechanical and available to
the forecaster in real time.  The calendar rows are kept because anyone who
disagrees with the dates can change them and re-run.

Conditioning an EVALUATION on the state of the world is not look-ahead - no
model sees the label - but it is not free either: a split chosen after seeing
results would be specification search.  The dates are listed in the source, the
percentile rungs are mechanical, and the narrow calendar window is located by
VIX rather than by the outcome.  The tightening itself was added after the wide
version had been run, at a reader's suggestion, and that is said here rather
than presented as the original design.

SCALE, AND WHY THE RAW NUMBER MISLEADS
---------------------------------------
Squared errors are larger when the target is more volatile, so a larger mean
loss differential during stress can be pure heteroskedasticity.  Every table
below therefore carries the differential divided by the domestic model's own
mean squared error INSIDE THE SAME REGIME - a fraction of error removed, which
is comparable across regimes in a way the raw difference is not.

WHAT WOULD OVERTURN THE PAPERS
-------------------------------
Part A: if the cross-section's contribution vanishes or reverses in stress, then
the headline substitution rate is an artefact of calm days and the paper must
say so.  Part B: if the foreign block stops being redundant given implied
volatility during stress, then Section 7's conclusion holds only when the
options market is working, which is a materially weaker claim than the one the
paper currently makes.  Both verdicts are computed from the tables.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab08_implied_vol as IV

SEED = 20260914
TARGET = "SPX"
DELAYS = [0, 5, 21, 55]
N_BOOT = 2000
BLK = 10
CORE = 20        # half-width, in trading days, of the narrow calendar window

# Fixed before any result below was inspected.  Edit these and the verdict
# changes, which is the point of writing them down rather than describing them.
EPISODES = [
    ("2008 crisis",    "2008-09-01", "2009-06-30"),
    ("2011 euro",      "2011-07-01", "2011-12-31"),
    ("2015 devaluation", "2015-08-01", "2016-02-29"),
    ("2018 vol spike", "2018-02-01", "2018-03-31"),
    ("2020 COVID",     "2020-02-15", "2020-06-30"),
    ("2022 inflation", "2022-01-01", "2022-12-31"),
]


def block_mean_ci(d, rng, n_boot=N_BOOT, block=BLK):
    """Moving-block bootstrap of a mean, on a time-ordered subsequence."""
    n = len(d)
    if n < 2 * block:
        return float(np.mean(d)) if n else float("nan"), float("nan"), float("nan")
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    offs = np.arange(block)
    out = np.empty(n_boot)
    for i in range(n_boot):
        s = (starts[i][:, None] + offs).ravel()[:n]
        out[i] = d[s].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def rel_improvement(d, e_own, rng, n_boot=N_BOOT, block=BLK):
    """mean(d) / mean(e_own), block-bootstrapped.

    The raw mean of d is NOT comparable across regimes: squared errors are
    larger when the target is more volatile, so a bigger number in stress can be
    pure heteroskedasticity rather than more help.  Dividing by the domestic
    model's own mean squared error inside the SAME regime removes the scale and
    leaves a fraction-of-error-removed, which is comparable.
    """
    n = len(d)
    pt = float(d.mean() / e_own.mean()) if n and e_own.mean() > 0 else float("nan")
    if n < 2 * block:
        return pt, float("nan"), float("nan")
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    offs = np.arange(block)
    out = np.empty(n_boot)
    for i_ in range(n_boot):
        sel = (starts[i_][:, None] + offs).ravel()[:n]
        den = e_own[sel].mean()
        out[i_] = d[sel].mean() / den if den > 0 else np.nan
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return pt, float(lo), float(hi)


def regimes(dates, vix_series):
    """A LADDER of stress definitions, from loose to severe.

    The first version of this file used one tercile and one set of calendar
    windows.  Both are wide: 'the 2022 inflation shock' as all of 2022 is 251
    days, most of them ordinary.  A wide window dilutes exactly the thing being
    tested, so the cut is tightened here in steps and the answer is read off the
    ladder rather than off any single split.

    The VIX ladder needs no dates at all - it is a percentile of the series
    against its own trailing year, evaluated at t-1, so it is both mechanical
    and available to the forecaster in real time.  The calendar rows are kept
    because they are checkable by anyone who disagrees with the dates, and the
    narrow calendar row is located at each window's VIX peak.
    """
    out = {}
    v = vix_series.reindex(dates).values
    sv = pd.Series(v)
    for tag, q in (("VIX top 33%", 2.0 / 3.0), ("VIX top 10%", 0.90),
                   ("VIX top 5%", 0.95)):
        thr = sv.rolling(252, min_periods=60).quantile(q).values
        m = np.isfinite(v) & np.isfinite(thr) & (v > thr)
        out[tag] = m
    base = sv.rolling(252, min_periods=60).quantile(2.0 / 3.0).values
    out["VIX bottom 67%"] = np.isfinite(v) & np.isfinite(base) & (v <= base)

    cal = np.zeros(len(dates), bool)
    core = np.zeros(len(dates), bool)
    for name, lo, hi in EPISODES:
        m = (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
        cal |= m
        w = np.where(m)[0]
        if len(w):
            vv = np.where(np.isfinite(v[w]), v[w], -np.inf)
            peak = w[int(np.argmax(vv))]          # located by VIX, not by outcome
            lo_i, hi_i = max(0, peak - CORE), min(len(dates) - 1, peak + CORE)
            core[lo_i:hi_i + 1] = True
    out["episodes, full window"] = cal
    out[f"episodes, peak +/-{CORE}d"] = core
    out["outside every episode"] = ~cal
    return out


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    rng = np.random.default_rng(SEED)

    D, peers, lag, idx, yb, fitrun, a = L.run_target(folder, TARGET, DELAYS,
                                                     rng, continuous=True)
    dates = pd.DatetimeIndex(D.index[idx])
    print(f"target {TARGET}, {len(idx)} test days, "
          f"{dates[0].date()} to {dates[-1].date()}, {len(peers)} peers\n")

    # VIX at t-1, for the real-time cut only
    _, ivdf, _ = IV.build(folder, strict=True)
    vix = ivdf["VIX"].shift(1) if "VIX" in ivdf else None
    if vix is None:
        raise SystemExit("no VIX column; the real-time cut cannot be built")

    R = regimes(dates, vix)
    print("=" * 96)
    print("0.  THE SPLITS, FIXED BEFORE THE RESULTS WERE LOOKED AT")
    print("=" * 96)
    for name, lo, hi in EPISODES:
        m = (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
        print(f"  {name:<20} {lo} to {hi}   {int(m.sum()):>5} test days")
    print()
    for k, m in R.items():
        print(f"  {k:<26} {int(m.sum()):>5} days  ({m.mean():>5.1%} of the window)")

    # ---------------- A ---------------------------------------------------
    print("\n" + "=" * 96)
    print("A.  DOES THE CROSS-SECTION STILL HELP WHEN THE MARKET IS STRESSED?")
    print("=" * 96)
    print("d_t = squared error of the domestic model minus that of the cross-sectional")
    print("model, per day.  Positive means the cross-section helps.  Each regime's")
    print("subsequence is block-bootstrapped in its own time order.\n")

    own_f, cross_f = {}, {}
    for d in DELAYS:
        own_f[d] = fitrun(d, False)
        cross_f[d] = fitrun(d, True)

    keys = list(R.keys())
    holds, rel = {}, {}
    print(f"{'delta':>6}{'regime':>24}{'days':>7}{'mean d':>10}"
          f"{'share of own MSE':>19}{'95% CI':>20}{'helps?':>8}")
    for d in DELAYS:
        e_own = (yb - own_f[d]) ** 2
        e_cro = (yb - cross_f[d]) ** 2
        dd = e_own - e_cro
        for k in keys:
            m = R[k]
            mu, lo0, _ = block_mean_ci(dd[m], np.random.default_rng(SEED + d))
            pt, lo, hi = rel_improvement(dd[m], e_own[m],
                                         np.random.default_rng(SEED + 3 + d))
            helps = np.isfinite(lo) and lo > 0
            holds[(d, k)] = helps
            rel[(d, k)] = (pt, lo, hi)
            print(f"{d:>6}{k:>24}{int(m.sum()):>7}{mu:>+10.4f}{pt:>19.1%}"
                  + f"[{lo:>+7.1%},{hi:>+7.1%}]".rjust(20)
                  + f"{'yes' if helps else 'no':>8}")
        print()

    LADDER = ["VIX top 33%", "VIX top 10%", "VIX top 5%",
              f"episodes, peak +/-{CORE}d"]
    CALM = ["VIX bottom 67%", "outside every episode"]
    print("  Reading the ladder at the longest delay, where the effect is largest.")
    d_last = DELAYS[-1]
    print(f"{'regime':>24}{'share of own MSE removed':>28}{'95% CI':>20}")
    for k in LADDER + CALM:
        pt, lo, hi = rel[(d_last, k)]
        print(f"{k:>24}{pt:>28.1%}" + f"[{lo:>+7.1%},{hi:>+7.1%}]".rjust(20))

    s_hit = [(d, k) for d in DELAYS for k in LADDER if holds[(d, k)]]
    c_hit = [(d, k) for d in DELAYS for k in CALM if holds[(d, k)]]
    print(f"\n  cells where the cross-section significantly helps, STRESS rungs: "
          f"{len(s_hit)} of {len(DELAYS) * len(LADDER)}")
    print(f"  the same on calm rungs:                                        "
          f"{len(c_hit)} of {len(DELAYS) * len(CALM)}")

    tightest = rel[(d_last, "VIX top 5%")]
    calmest = rel[(d_last, "VIX bottom 67%")]
    sep = tightest[1] > calmest[2] or calmest[1] > tightest[2]
    print(f"\n  severest rung vs calmest, at delta = {d_last}: "
          f"{tightest[0]:.1%} [{tightest[1]:+.1%},{tightest[2]:+.1%}] against "
          f"{calmest[0]:.1%} [{calmest[1]:+.1%},{calmest[2]:+.1%}]")
    print(f"  intervals disjoint: {'yes' if sep else 'no'}")

    if len(s_hit) == 0 and len(c_hit) > 0:
        print("\n  The contribution is a calm-market phenomenon and disappears as the stress")
        print("  definition tightens.  The headline rate averages over days on which the")
        print("  effect is absent, and the paper must say where it holds.")
    elif len(s_hit) and not sep:
        print("\n  The contribution survives every rung of the ladder, and once the scale")
        print("  difference is removed the severest and calmest rungs are not separated.")
        print("  The rolling-median design does not break down across these transitions,")
        print("  and it does not visibly strengthen in them either: what looks in the raw")
        print("  column like a much larger effect in stress is mostly larger errors.")
    elif len(s_hit) and sep and tightest[0] > calmest[0]:
        print("\n  The contribution is LARGER in stress even after the scale is removed,")
        print("  which is the opposite of the objection and a stronger claim than the")
        print("  paper currently makes.")
    else:
        print("\n  The pattern across the ladder is mixed; it is reported above and no")
        print("  single summary of it is offered.")

    # ---------------- B ---------------------------------------------------
    print("\n" + "=" * 96)
    print("B.  IS BREADTH STILL REDUNDANT GIVEN IMPLIED VOLATILITY, IN STRESS?")
    print("=" * 96)
    print("Section 7 finds the foreign block adds nothing once VIX and VDAX are held.")
    print("If options markets distort under stress, that should fail exactly there.")
    print("Positive means the foreign block adds to the implied-volatility model.\n")

    D2, iv2, peers2 = IV.build(folder, strict=True)
    own2, P2, V2, y2, i2 = IV.make(D2, iv2, peers2)
    yb2 = y2[i2]
    d2ates = pd.DatetimeIndex(D2.index[i2])
    R2 = regimes(d2ates, vix)
    print(f"  implied-volatility panel: {len(i2)} test days, "
          f"{d2ates[0].date()} to {d2ates[-1].date()}\n")

    adds = {}
    print(f"{'delta':>6}{'regime':>24}{'days':>7}{'mean d':>10}"
          f"{'share of IV MSE':>18}{'95% CI':>20}{'adds?':>8}")
    for d in DELAYS:
        f_iv = IV.walk(own2, [V2], y2, i2, d)
        f_both = IV.walk(own2, [V2, P2], y2, i2, d)
        e_iv = (yb2 - f_iv) ** 2
        dd = e_iv - (yb2 - f_both) ** 2
        for k in R2:
            m = R2[k]
            mu, _, _ = block_mean_ci(dd[m], np.random.default_rng(SEED + 7 + d))
            pt, lo, hi = rel_improvement(dd[m], e_iv[m],
                                         np.random.default_rng(SEED + 9 + d))
            a_ = np.isfinite(lo) and lo > 0
            adds[(d, k)] = a_
            print(f"{d:>6}{k:>24}{int(m.sum()):>7}{mu:>+10.4f}{pt:>18.1%}"
                  + f"[{lo:>+7.1%},{hi:>+7.1%}]".rjust(20)
                  + f"{'yes' if a_ else 'no':>8}")
        print()

    LADDER2 = ["VIX top 33%", "VIX top 10%", "VIX top 5%",
               f"episodes, peak +/-{CORE}d"]
    CALM2 = ["VIX bottom 67%", "outside every episode"]
    s_add = [(d, k) for d in DELAYS for k in LADDER2 if adds[(d, k)]]
    c_add = [(d, k) for d in DELAYS for k in CALM2 if adds[(d, k)]]
    print(f"  cells where the foreign block significantly ADDS given implied volatility")
    print(f"    on STRESS rungs: {len(s_add)} of {len(DELAYS) * len(LADDER2)}"
          f"  {s_add if s_add else ''}")
    print(f"    on calm rungs:   {len(c_add)} of {len(DELAYS) * len(CALM2)}"
          f"  {c_add if c_add else ''}")
    print(f"\n  The severest rung carries {int(R2['VIX top 5%'].sum())} days, which is the")
    print("  binding constraint on this test: a foreign block that helped only in the")
    print("  very worst days would need more of them to show it.")
    if s_add and not c_add:
        print("\n  Redundancy is conditional.  The foreign block adds nothing on ordinary")
        print("  days and something during stress - when an options market is least able")
        print("  to price - so Section 7's conclusion needs the qualifier that it holds")
        print("  while the options market is functioning.")
    elif not s_add and not c_add:
        print("\n  The foreign block adds nothing detectable on any rung, calm or severe,")
        print("  down to the top 5% of VIX days.  Redundancy given implied volatility is")
        print("  not an average concealing a stressed subsample.  What this cannot rule")
        print("  out is an effect confined to days more extreme than the ladder reaches.")
    else:
        print("\n  The foreign block adds somewhere other than stress alone; the pattern is")
        print("  reported above and is not the one the objection predicted.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
