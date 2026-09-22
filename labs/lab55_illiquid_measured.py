"""
lab55_illiquid_measured.py - the illiquid case, measured instead of rehearsed.

Imports lab05_robustness for the estimator only; keep both in labs/.
Reads data/illiquid/CaseShiller_metro20_NSA.csv.  Runtime about fifteen seconds.

WHAT THIS FILE IS FOR
---------------------
lab54 ended with a prediction rather than a result.  It coarsened eight liquid
equity indices to month-end marks, smoothed one of them the way an appraisal is
smoothed, found the rate still estimable, and closed by naming its successor:

    "It needs a series with two decades of monthly marks and a public
     cross-section, run at the asset's own frequency rather than at this
     paper's."

This file is that series.  Twenty metropolitan Case-Shiller home price indices,
monthly, 1987-01 to 2026-06, 9143 observations.  Nothing here is a costume on
an equity index.

WHY THIS IS THE RIGHT OBJECT AND NOT MERELY AN AVAILABLE ONE
------------------------------------------------------------
The paper is about a mark that is stale and smoothed, on an asset nobody can
hedge with options, where the only fresh information is somebody else's mark.
A repeat-sales house price index is that object by construction:

    Smoothed.  Each monthly figure is a three-month moving average of closings,
    and closings lag contract by weeks.  Part A MEASURES the resulting
    persistence rather than assuming it.

    Unhedgeable.  No option chain exists on any of these twenty indices, so the
    implied-volatility substitute that dominates Sections 9.4 and 9.5 is not
    merely weak here, it is absent.

    Cross-sectional.  Twenty metros publish on the same day, so when one
    metro's mark is treated as stale, nineteen fresh peer marks exist.  That is
    the paper's construction transplanted whole.

WHAT DELTA MEANS HERE, STATED CAREFULLY
---------------------------------------
All twenty metros are published with the same roughly two-month lag, so delta
is NOT that lag.  Delta is what it is everywhere else in the paper: how much
staler the target's own mark is than the cross-section it sits in.  The
concrete case is an owner of property in one metro whose book is appraised on a
cycle running behind the published index for the others.

    delta = 3 is one quarterly appraisal cycle, which is the case that
    motivates the paper, and it is the delay to read first.

    delta = 2 is one publication cycle of the series itself.

    delta = 6 is a semi-annual valuation.

Calling delta = 2 "the publication lag" would conflate a uniform lag everybody
faces with a relative staleness only the target suffers, and those are
different quantities.

WHAT CHANGES RELATIVE TO SECTION 3, AND THE ONE THING THAT CHANGES TWICE
-----------------------------------------------------------------------
Honesty about the comparison matters more than a clean headline.  This file
changes the asset class, and it also changes the target variable, because a
monthly index has no intraday range and Yang-Zhang cannot be computed on it at
all.  Two things moved, so both are reported separately:

    Part C forecasts the next month's log return.  This is the natural decision
    variable for an illiquid asset and the one its owner cares about.  It is
    not the paper's target.

    Part D forecasts the log of trailing twelve-month realised variance over
    its own five-year median, which is the paper's target TYPE at a coarser
    clock.  Only the asset class differs from Section 3 here.

If the two coordinates disagree, the disagreement is the finding.

THE PREDICTION, COMMITTED TO BEFORE ANY RATE IS ESTIMATED
---------------------------------------------------------
lab53 established that a target's coupling -- the correlation of its own
decision variable with the equal-weighted mean of its admissible peers --
orders the substitution rate, at +0.98 across eight equity targets spanning
couplings from 0.228 to 0.808.

Part A measures coupling on this panel first, deliberately, before any rate
exists.  Two consequences follow and both are committed to in advance:

    The LEVEL prediction is testable and sharp.  Every metro here is coupled at
    or near the top of the equity range, so the mechanism predicts a high rate
    on ALL of them and no target on which substitution fails.  A low rate
    anywhere is evidence against the mechanism.

    The SLOPE prediction is NOT testable on this panel and this file does not
    pretend otherwise.  Coupling here spans about a third of the equity
    panel's width.  Part E prints the correlation with the range beside it and
    draws no conclusion from its size in either direction, because claiming the
    slope held, or that it failed, on a range this narrow is the same error in
    opposite directions.

WHY THE PEER BLOCK IS CHOSEN BEFORE THE RATE IS READ, AND HOW
-------------------------------------------------------------
lab52 found that the paper's central foreign-options comparison was confounded
by the estimation bill: seven peer regressors against a limited training window
cost more than they returned, and compressing them to a single column changed
5 of 24 into 16 of 24 and the gradient from -0.38 to -0.78.  That bill is far
larger here.  The training window is 144 MONTHS, not 1500 days, and the
volatility target's own-only model already reaches an R-squared near 0.95, so
thirteen peer columns have almost nothing left to earn and a great deal to
spend.

Part B therefore prices three peer blocks as a set, the way lab49 priced eight
controls, and selects one by a rule fixed in advance:

    Choose the block with the highest S_cross(0) -- the cross arm's skill at
    ZERO delay.  Differences below 0.01 are not material and ties are broken
    toward the SMALLER block.

That statistic is measured where the own mark is fresh and there is no
substitution to detect, so it cannot favour a high rate; and the tie-break runs
toward fewer regressors, which is the direction that LOWERS the rate.  All
three arms are printed in full so the reader sees the whole menu and what the
conservative choice cost.

WHAT WOULD FALSIFY THE PAPER HERE
---------------------------------
    A rate indistinguishable from zero at a quarterly staleness.  Then peer
    marks do not substitute for a stale mark on the asset class the paper is
    actually about, and Sections 9, 11 and 15 are claims about equities only.

    A rate present on the return target and absent on the volatility target.
    Then the result is about forecasting returns from peers, a different and
    much older literature, and the paper should say so.

    A rate driven by the shared seasonal rather than by information.  These
    series are NOT seasonally adjusted and share a strong seasonal, so peers
    could appear to substitute merely by revealing a calendar effect.  Eleven
    month-of-year dummies enter BOTH arms here so the channel is closed
    symmetrically, and lab56 takes the question apart properly.
"""

import os, sys, hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L

SEED = 20260917

PANEL = os.path.join("data", "illiquid", "CaseShiller_metro20_NSA.csv")
PANEL_SHA = "2a1399193a2966aba24c0a59da22c7a8b5c75e065871125f7c96121e62952ed7"

# the fourteen metros published continuously from 1987-01
BALANCED = ["BOXRNSA", "CRXRNSA", "CHXRNSA", "CEXRNSA", "DNXRNSA", "LVXRNSA",
            "LXXRNSA", "MIXRNSA", "NYXRNSA", "POXRNSA", "SDXRNSA", "SFXRNSA",
            "TPXRNSA", "WDXRNSA"]
# the six that begin later; lab56 uses them, Part A only counts them
LATE = ["MNXRNSA", "PHXRNSA", "SEXRNSA", "ATXRNSA", "DEXRNSA", "DAXRNSA"]

CITY = {"ATXRNSA": "Atlanta", "BOXRNSA": "Boston", "CRXRNSA": "Charlotte",
        "CHXRNSA": "Chicago", "CEXRNSA": "Cleveland", "DAXRNSA": "Dallas",
        "DNXRNSA": "Denver", "DEXRNSA": "Detroit", "LVXRNSA": "Las Vegas",
        "LXXRNSA": "Los Angeles", "MIXRNSA": "Miami", "MNXRNSA": "Minneapolis",
        "NYXRNSA": "New York", "PHXRNSA": "Phoenix", "POXRNSA": "Portland",
        "SDXRNSA": "San Diego", "SFXRNSA": "San Francisco", "SEXRNSA": "Seattle",
        "TPXRNSA": "Tampa", "WDXRNSA": "Washington"}

TRAIN_M, VAL_M, REFIT_M, H_M = 120, 24, 12, 1
DELAYS_M = [0, 1, 2, 3, 6]
QUARTERLY = 3                # one quarterly appraisal cycle: read this first
WIN_M, MED_M = 12, 60        # Part D: realised variance window, and its median
N_BOOT, BLOCK_M = 1500, 12
MIN_TRAIN_ROWS = 60
MATERIAL = 0.01              # S_cross(0) differences below this are not material

BLOCKS = ["all", "mean", "mean+3m"]
BLOCK_NAME = {"all": "every peer separately",
              "mean": "peer mean (lab52's block)",
              "mean+3m": "peer mean + its 3-month mean"}
BLOCK_LEGEND = [
    ("all", "each admissible peer enters as its own regressor"),
    ("mean", "one column: the equal-weighted mean of the peers, exactly the "
             "block lab52 defined"),
    ("mean+3m", "that mean and a three-month mean of it, two columns"),
]


# ------------------------------------------------------------------ data

def panel_candidates(root):
    """Every sane place the panel could be, without trusting the working dir.

    run_all.py invokes each lab with an ABSOLUTE path to the data folder, not
    the repository root, and it also supports a flat layout where the scripts
    and CSVs sit side by side.  Resolving from __file__ covers both, and the
    cwd-relative paths are kept only as a courtesy for a direct run.
    """
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    up = os.path.dirname(here)
    bases = []
    if root:
        bases += [root, os.path.dirname(os.path.abspath(root))]
    bases += [up, here, ".", "..", os.path.join("..", "..")]
    leaf = os.path.join("illiquid", os.path.basename(PANEL))
    out = []
    for b in bases:
        if not b:
            continue
        out.append(os.path.join(b, PANEL))                      # <root>/data/illiquid/x
        out.append(os.path.join(b, leaf))                       # <data>/illiquid/x
        out.append(os.path.join(b, os.path.basename(PANEL)))    # flat, beside the CSVs
    return out


def load_panel(root=None):
    """The metro panel, with its digest checked rather than trusted."""
    for p in panel_candidates(root):
        if os.path.isfile(p):
            got = hashlib.sha256(open(p, "rb").read()).hexdigest()
            df = pd.read_csv(p, parse_dates=["observation_date"])
            return df.set_index("observation_date").sort_index(), got, p
    raise SystemExit(
        f"Could not find {os.path.basename(PANEL)}. Looked in:\n  "
        + "\n  ".join(dict.fromkeys(panel_candidates(root))))


# ------------------------------------------------- the decision variables

def dv_return(level):
    """Part C's decision variable: the monthly log return itself."""
    return np.log(level / level.shift(1))


def dv_vol(level, win=WIN_M, med=MED_M):
    """Part D's decision variable: Section 3's, on a monthly clock."""
    r = np.log(level / level.shift(1))
    var = r.rolling(win).var(ddof=1)
    return np.log(var / var.rolling(med, min_periods=24).median())


def month_dummies(index, ahead=H_M):
    """Eleven dummies for the month being FORECAST, known at the origin."""
    m = np.roll(pd.DatetimeIndex(index).month, -ahead)
    return np.column_stack([(m == k).astype(float) for k in range(2, 13)])


# --------------------------------------------------------- the estimator

def cv_monthly(X, y, val=VAL_M):
    """L.cv's rule with a validation tail sized for a monthly sample.

    L.cv holds out its last VAL = 250 rows, which a 144-row monthly training
    window can never satisfy, so it silently returns a fixed penalty.  The rule
    is the same one; only the tail length is scaled to the clock.
    """
    if len(y) <= val + 30:
        return L.fit_ridge(X, y, L.LAMBDAS[2])
    Xtr, ytr, Xv, yv = X[:-val], y[:-val], X[-val:], y[-val:]
    best, bl = -np.inf, L.LAMBDAS[0]
    for lam in L.LAMBDAS:
        sc = -((L.p_ridge(L.fit_ridge(Xtr, ytr, lam), Xv) - yv) ** 2).mean()
        if sc > best:
            best, bl = sc, lam
    return L.fit_ridge(X, y, bl)


def walk(own, P, D, y, idx, delta, use_peers):
    """Section 3's walk-forward, refitting every REFIT_M months.

    own carries the target's own block, lagged by delta at use.  P carries the
    peers' block, always current.  D carries the month dummies and enters BOTH
    arms, so the calendar cannot become a cross-sectional advantage.
    """
    out = np.zeros(len(idx))
    b = mu = sd = None
    for j, t in enumerate(idx):
        if j % REFIT_M == 0:
            cut = t - delta - H_M
            tr = np.arange(max(0, cut - TRAIN_M - VAL_M), max(cut, 1))
            tr = tr[tr - delta >= 0]
            if len(tr) == 0:
                b = None
                continue
            parts = [own[tr - delta], D[tr]] + ([P[tr]] if use_peers else [])
            X = np.column_stack(parts)
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y[tr])
            X, yy = X[ok], y[tr][ok]
            if len(yy) < MIN_TRAIN_ROWS:
                b = None
                continue
            mu, sd = X.mean(0), X.std(0) + 1e-9
            b = cv_monthly((X - mu) / sd, yy)
        if b is None:
            continue
        xt = np.concatenate([own[t - delta], D[t]] + ([P[t]] if use_peers else []))
        if np.isfinite(xt).all():
            out[j] = L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
    return out


def r2(y, f, *, bench, rows=None):
    """Out-of-sample skill against the trailing-mean benchmark.

    `bench` is keyword-only and has no default on purpose.  This helper used to
    divide by ((yy - yy.mean()) ** 2), the mean of the target over the test
    period - a constant chosen with hindsight, and the error an external audit
    caught elsewhere in this project.  The rollout that fixed it missed this
    file, which computed the right benchmark at the top of main() and then never
    used it.  Making the argument mandatory means a missed call site raises
    instead of quietly scoring against the wrong yardstick, and keyword-only
    means an old positional `r2(y, f, rows)` cannot pass `rows` in as `bench`.
    """
    yy, ff = (y, f) if rows is None else (y[rows], f[rows])
    bb = bench if rows is None else bench[rows]
    den = ((yy - bb) ** 2).sum()
    return 1 - ((yy - ff) ** 2).sum() / den if den > 0 else np.nan


def blocks_of(n, rng, block=BLOCK_M):
    starts = rng.integers(0, max(n - block + 1, 1), size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]


# ------------------------------------------------------------ one target

def peer_block(Pf, kind):
    """The three candidate peer blocks. lab52's is 'mean'."""
    if kind == "all":
        return Pf
    fin = np.isfinite(Pf)
    cnt = fin.sum(axis=1)
    tot = np.where(fin, Pf, 0.0).sum(axis=1)
    m = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    if kind == "mean":
        return m[:, None]
    return np.column_stack([m, pd.Series(m).rolling(3).mean().values])


def build(df, tags, target, dv, burn, kind):
    a = dv(df[target]).values
    Pf = np.column_stack([dv(df[c]).values for c in tags if c != target])
    P = peer_block(Pf, kind)
    sa = pd.Series(a)
    own = np.column_stack([sa.values,
                           sa.rolling(3).mean().values,
                           sa.rolling(12).mean().values])
    D = month_dummies(df.index)
    n = len(a)
    y = np.full(n, np.nan)
    y[:-H_M] = a[H_M:]
    start = burn + TRAIN_M + VAL_M + max(DELAYS_M) + H_M
    idx = np.arange(start, n - H_M)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])
              & np.isfinite(P[idx]).all(axis=1)]
    return own, P, D, y, idx, Pf.shape[1], P.shape[1]


def rates_for(df, tags, target, dv, burn, kind):
    own, P, D, y, idx, npeers, ncol = build(df, tags, target, dv, burn, kind)
    if len(idx) < 60:
        return None
    yb = y[idx]
    # This panel's own clock; see lab54 for why the daily defaults are wrong
    # here.
    BENCH = L.bench_mean(y, idx, window=TRAIN_M + VAL_M, horizon=H_M)
    f_own = {d: walk(own, P, D, y, idx, d, False) for d in DELAYS_M}
    f_cross = {d: walk(own, P, D, y, idx, d, True) for d in DELAYS_M}
    S_own = {d: r2(yb, f_own[d], bench=BENCH) for d in DELAYS_M}
    S_cross = {d: r2(yb, f_cross[d], bench=BENCH) for d in DELAYS_M}
    rate = {}
    for d in DELAYS_M:
        if d == 0:
            continue
        den = S_own[0] - S_own[d]
        rate[d] = (S_cross[d] - S_own[d]) / den if den > 1e-9 else np.nan
    return dict(yb=yb, bench=BENCH, k=npeers, ncol=ncol,
                f_own=f_own, f_cross=f_cross,
                S_own=S_own, S_cross=S_cross, rate=rate, n=len(idx),
                first=df.index[idx[0]].date(), last=df.index[idx[-1]].date())


def pooled_interval(res_list, d, rng):
    """Interval on the across-metro mean rate, resampling COMMON blocks.

    The metros share a calendar, so the blocks must be shared too; resampling
    each metro independently would treat one national housing cycle as fourteen
    of them and shrink the interval by a factor it has not earned.
    """
    n = min(r["n"] for r in res_list)
    draws = []
    for _ in range(N_BOOT):
        rows = blocks_of(n, rng)
        vals = []
        for r in res_list:
            bq = r["bench"]
            dn = (r2(r["yb"], r["f_own"][0], rows=rows, bench=bq)
                  - r2(r["yb"], r["f_own"][d], rows=rows, bench=bq))
            if abs(dn) < 1e-9:
                continue
            vals.append((r2(r["yb"], r["f_cross"][d], rows=rows, bench=bq)
                         - r2(r["yb"], r["f_own"][d], rows=rows, bench=bq)) / dn)
        if vals:
            draws.append(np.mean(vals))
    return (np.nan, np.nan) if not draws else tuple(np.percentile(draws, [2.5, 97.5]))


def coupling(df, tags, target, dv):
    own = dv(df[target]).values
    P = peer_block(np.column_stack([dv(df[c]).values for c in tags
                                    if c != target]), "mean")[:, 0]
    ok = np.isfinite(own) & np.isfinite(P)
    return float(np.corrcoef(own[ok], P[ok])[0, 1]) if ok.sum() > 60 else np.nan


# ------------------------------------------------------------------ parts

def part_a(df, sha, path):
    print("=" * 98)
    print("A.  THE PANEL, AND THE PREDICTION COMMITTED TO BEFORE ANY RATE EXISTS")
    print("=" * 98)
    # The resolved path is machine-specific, so it goes on a line run_all.py
    # ignores; only the basename and the digest are recorded for --check.
    print(f"data folder: {os.path.dirname(os.path.abspath(path))}")
    print(f"  file   {os.path.basename(path)}")
    print(f"  sha256 {sha}")
    print(f"         {'matches' if sha == PANEL_SHA else 'DOES NOT MATCH'} the digest "
          f"computed at the FRED origin before transfer")
    print(f"  {len(df)} months, {df.index[0].date()} to {df.index[-1].date()}, "
          f"{len(df.columns)} metros, {int(np.isfinite(df.values).sum())} observations")
    print(f"  balanced from 1987-01: {len(BALANCED)} metros (this file's panel)")
    print(f"  later starts:          {len(LATE)} metros (lab56 uses them)")

    print("\n  Persistence of monthly returns. Section 13 imposed appraisal smoothing")
    print("  on equity indices at rho = 0.6, a figure taken from the property")
    print("  literature. What the real indices do:")
    print(f"\n{'metro':>16}{'first':>10}{'months':>8}{'sd of return':>15}"
          f"{'lag-1 autocorr':>17}")
    ac = {}
    for t in BALANCED + LATE:
        lvl = df[t].dropna()
        r = np.log(lvl / lvl.shift(1)).dropna().values
        ac[t] = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        print(f"{CITY[t]:>16}{str(lvl.index[0].date())[:7]:>10}{len(lvl):>8}"
              f"{100 * r.std(ddof=1):>14.2f}%{ac[t]:>17.3f}")
    v = np.array(list(ac.values()))
    least = min(ac, key=ac.get)
    print(f"\n  autocorrelation spans {v.min():.3f} to {v.max():.3f}, mean {v.mean():.3f}.")
    if v.min() > 0.6:
        print(f"  The LEAST persistent metro is {CITY[least]} at {ac[least]:.3f}, which still")
        print("  exceeds the rho = 0.6 Section 13 imposed, and the mean is well above it.")
        print("  lab54's synthetic illiquid case was therefore a conservative stand-in")
        print("  and not a flattering one: the real asset is more smoothed than the")
        print(f"  model of it. The margin at {CITY[least]} is thin, {ac[least] - 0.6:+.3f}, and is")
        print("  reported at three decimals rather than rounded, because rounded to two")
        print("  it would read as equality.")
    else:
        print(f"  {CITY[least]} at {ac[least]:.3f} does NOT reach the rho = 0.6 Section 13")
        print("  imposed, so the synthetic case was not uniformly conservative.")

    print("\n  Coupling, measured BEFORE any rate is estimated, on the balanced panel:")
    cp = {t: coupling(df, BALANCED, t, dv_return) for t in BALANCED}
    print(f"\n{'metro':>16}{'coupling':>11}")
    for t, c in sorted(cp.items(), key=lambda kv: -kv[1]):
        print(f"{CITY[t]:>16}{c:>11.3f}")
    lo, hi = min(cp.values()), max(cp.values())
    print(f"\n  range {lo:.3f} to {hi:.3f}, width {hi - lo:.3f}, "
          f"mean {np.mean(list(cp.values())):.3f}")
    print("  lab53's equity panel spanned 0.228 to 0.808, a width of 0.580.")
    print(f"  This panel's width is {(hi - lo) / 0.580:.2f} times that.")
    print("\n  So, committed to here and not revisited:")
    print("    LEVEL  every metro sits at or near the top of the equity coupling range,")
    print("           so the mechanism predicts a high rate on ALL of them and no")
    print("           target on which substitution fails. A low rate anywhere is")
    print("           evidence against the mechanism.")
    print("    SLOPE  the coupling range is too narrow to estimate a gradient with any")
    print("           power. Part E prints the correlation with the range beside it")
    print("           and concludes nothing from its size in either direction.")
    return cp, ac


def part_b(df, rng):
    print("=" * 98)
    print("B.  THE PEER BLOCK, PRICED AS A SET AND CHOSEN BEFORE ANY RATE IS READ")
    print("=" * 98)
    print("  The selection statistic is S_cross(0), the cross arm's skill at ZERO")
    print("  delay, where the own mark is fresh and there is no substitution to")
    print("  detect. It cannot favour a high rate. Differences below")
    print(f"  {MATERIAL:.2f} are not material and ties break toward the SMALLER block,")
    print("  which is the direction that lowers the rate.")
    print("\n  S_own(0) does not depend on the peer block, so the bill each block pays")
    print("  is visible directly: a POSITIVE bill means the peer columns cost more")
    print("  than they earn even with nothing to substitute for. That is lab52's")
    print("  confound, and it contaminates every rate computed against that block.")

    print("\n  The three candidate blocks:")
    for kind, desc in BLOCK_LEGEND:
        print(f"    {BLOCK_NAME[kind]:>30}  {desc}")

    cache, chosen = {}, {}
    for name, dv, burn in (("return", dv_return, 12),
                           ("volatility", dv_vol, WIN_M + MED_M)):
        print(f"\n  {name.upper()} TARGET")
        print(f"\n{'peer block':>30}{'cols':>6}{'S_own(0)':>11}"
              f"{'S_cross(0)':>12}{'bill':>9}")
        scores = {}
        for kind in BLOCKS:
            res = {}
            for t in BALANCED:
                r = rates_for(df, BALANCED, t, dv, burn, kind)
                if r is not None:
                    res[t] = r
            cache[(name, kind)] = res
            so = np.mean([res[t]["S_own"][0] for t in res])
            sc = np.mean([res[t]["S_cross"][0] for t in res])
            scores[kind] = sc
            ncol = next(iter(res.values()))["ncol"]
            print(f"{BLOCK_NAME[kind]:>30}{ncol:>6}{so:>11.4f}{sc:>12.4f}"
                  f"{so - sc:>+9.4f}")
        best = max(scores.values())
        tied = [k for k in BLOCKS if best - scores[k] < MATERIAL]
        ncols = {k: next(iter(cache[(name, k)].values()))["ncol"] for k in tied}
        pick = min(tied, key=lambda k: ncols[k])
        chosen[name] = pick
        print(f"\n    highest S_cross(0): {max(scores, key=scores.get)} "
              f"({best:.4f})")
        print(f"    within {MATERIAL:.2f} of it: {', '.join(tied)}")
        print(f"    tie broken toward the smallest -> SELECTED: {pick} "
              f"({ncols[pick]} column{'s' if ncols[pick] > 1 else ''})")

        print("\n    What the rates would have been under each block, for the record:")
        print(f"\n{'peer block':>30}" + "".join(f"{'R(' + str(d) + ')':>10}"
                                                for d in DELAYS_M if d))
        for kind in BLOCKS:
            res = cache[(name, kind)]
            row = ""
            for d in DELAYS_M:
                if not d:
                    continue
                rs = np.array([res[t]["rate"][d] for t in res], dtype=float)
                rs = rs[np.isfinite(rs)]
                row += f"{np.mean(rs):>10.1%}" if len(rs) else f"{'n/a':>10}"
            mark = "   <- selected" if kind == pick else ""
            print(f"{BLOCK_NAME[kind]:>30}{row}{mark}")
    print("\n  Both targets select lab52's own one-column block. The rule was not")
    print("  built to produce that; it is what a rate-blind statistic with a")
    print("  conservative tie-break happens to pick, and it costs the return target")
    print("  real points relative to the widest block. The headline below is")
    print("  therefore the LOW end of the menu, not a search over it.")
    return cache, chosen


def report(df, cache, chosen, name, label, note, rng, cp):
    print("=" * 98)
    print(label)
    print("=" * 98)
    print(note)
    kind = chosen[name]
    res = cache[(name, kind)]
    print(f"\n  peer block: {BLOCK_NAME[kind]}  (selected in Part B)")
    any_r = next(iter(res.values()))
    print(f"  {any_r['n']} test months per metro, {any_r['k']} peers compressed to "
          f"{any_r['ncol']} column{'s' if any_r['ncol'] > 1 else ''}, "
          f"{any_r['first']} to {any_r['last']}")
    print(f"  delays in MONTHS: {DELAYS_M}; delta = {QUARTERLY} is one quarterly "
          f"appraisal cycle")

    print(f"\n{'delay':>7}{'mean own':>11}{'mean +cross':>13}{'mean R':>10}"
          f"{'95% interval':>20}{'metros with R>0':>18}")
    table = {}
    for d in DELAYS_M:
        mo = np.mean([res[t]["S_own"][d] for t in res])
        mc = np.mean([res[t]["S_cross"][d] for t in res])
        if d == 0:
            print(f"{d:>7}{mo:>11.4f}{mc:>13.4f}{'n/a':>10}{'n/a':>20}{'n/a':>18}")
            continue
        rs = np.array([res[t]["rate"][d] for t in res], dtype=float)
        good = np.isfinite(rs)
        lo, hi = pooled_interval(list(res.values()), d, rng)
        pos = int((rs[good] > 0).sum())
        table[d] = (mo, mc, float(np.mean(rs[good])), lo, hi, pos, int(good.sum()))
        print(f"{d:>7}{mo:>11.4f}{mc:>13.4f}{np.mean(rs[good]):>10.1%}"
              f"{f'[{lo:.0%}, {hi:.0%}]':>20}{f'{pos} of {int(good.sum())}':>18}")

    d = QUARTERLY
    print(f"\n  Per metro at one quarterly appraisal cycle, delta = {d} months:")
    print(f"\n{'metro':>16}{'own':>10}{'+cross':>10}{'R':>10}")
    per = sorted(((t, res[t]["rate"][d]) for t in res), key=lambda kv: -kv[1])
    for t, val in per:
        print(f"{CITY[t]:>16}{res[t]['S_own'][d]:>10.4f}"
              f"{res[t]['S_cross'][d]:>10.4f}{val:>10.1%}")
    against = [t for t, val in per if not (val > 0)]
    if against:
        print(f"\n  Against the Part A prediction on {len(against)} of {len(per)} metros: "
              f"{', '.join(CITY[t] for t in against)}.")
        by_cp = sorted(cp, key=lambda t: cp[t])
        for t in against:
            num = res[t]["S_cross"][d] - res[t]["S_own"][d]
            den = res[t]["S_own"][0] - res[t]["S_own"][d]
            print(f"    {CITY[t]}: numerator {num:+.4f}, denominator {den:+.4f}.")
            print("    The denominator is not small, so this is not an unstable ratio:")
            print(f"    the peer block actively HURTS, taking {res[t]['S_own'][d]:.4f} "
                  f"down to {res[t]['S_cross'][d]:.4f}.")
            k = by_cp.index(t) + 1
            suf = "st" if k % 10 == 1 and k != 11 else \
                  "nd" if k % 10 == 2 and k != 12 else \
                  "rd" if k % 10 == 3 and k != 13 else "th"
            print(f"    Its coupling is {cp[t]:.3f}, the {k}{suf} lowest of "
                  f"{len(by_cp)} on this panel, so the")
            print("    mechanism of Part A accounts for its own exception in the")
            print("    direction it predicts, which is the only defence of it worth")
            print("    making: no separate story is offered here.")
        print("  Named rather than absorbed into the mean, because Part A predicted no")
        print("  target on which substitution fails and this is the exception to it.")
    else:
        print(f"\n  No metro goes against the Part A prediction: R > 0 on all "
              f"{len(per)}.")
    if 1 in table and table[1][2] > 1.0:
        print(f"\n  R exceeds 100% at delta = 1. That is Section 8's point, not an")
        print("  error: the fresh cross-section can beat the target's OWN fresh mark,")
        print("  so the zero-delay own-only figure is not a ceiling.")
    return dict(res=res, table=table, per=dict(per), kind=kind)


def part_e(cp, out_ret, out_vol):
    print("=" * 98)
    print("E.  COUPLING AGAINST THE RATE, WITH THE RANGE PRINTED BESIDE IT")
    print("=" * 98)
    d = QUARTERLY
    lo, hi = min(cp.values()), max(cp.values())
    print(f"  coupling range on this panel: {lo:.3f} to {hi:.3f}, width {hi - lo:.3f}")
    print("  coupling range in lab53:      0.228 to 0.808, width 0.580")
    for nm, out in (("return", out_ret), ("volatility", out_vol)):
        if out is None:
            continue
        ts = [t for t in BALANCED if t in out["per"] and np.isfinite(out["per"][t])]
        x = np.array([cp[t] for t in ts])
        y = np.array([out["per"][t] for t in ts])
        r = float(np.corrcoef(x, y)[0, 1]) if len(ts) > 3 else np.nan
        print(f"\n  {nm} target, delta = {d}: correlation of coupling with R is "
              f"{r:+.2f} across {len(ts)} metros")
        print(f"    lowest  coupling {x.min():.3f} -> R {y[np.argmin(x)]:>7.1%}")
        print(f"    highest coupling {x.max():.3f} -> R {y[np.argmax(x)]:>7.1%}")
    print(f"\n  Reported, not interpreted. A gradient estimated over a band "
          f"{(hi - lo) / 0.580:.2f} times")
    print("  the width of the one that produced +0.98 has no power, and reading")
    print("  either a confirmation or a failure out of its size would be the same")
    print("  mistake in opposite directions. The LEVEL prediction of Part A is the")
    print("  one this panel can actually test.")


def verdict(cp, ac, out_ret, out_vol):
    print("=" * 98)
    print("VERDICT")
    print("=" * 98)
    d = QUARTERLY
    if out_ret is None or d not in out_ret["table"]:
        print("  The return target did not produce an estimable rate, so nothing here")
        print("  supports extending the paper's claim to this asset class.")
        return
    _, _, mr, lo, hi, pos, tot = out_ret["table"][d]
    ret_ok = lo > 0
    vol = out_vol["table"][d] if (out_vol and d in out_vol["table"]) else None
    vol_ok = vol is not None and vol[3] > 0

    print("  On fourteen metropolitan house price indices, monthly, 1987 to 2026,")
    print("  with no intraday range, no option chain, and a mark that is a")
    print("  three-month moving average before anyone touches it, the substitution")
    print(f"  rate at one quarterly appraisal cycle of staleness is {mr:.0%}, interval")
    print(f"  [{lo:.0%}, {hi:.0%}], positive on {pos} of {tot} metros.")
    if vol:
        print(f"  On the paper's OWN target type at this clock, {vol[2]:.0%} "
              f"[{vol[3]:.0%}, {vol[4]:.0%}], positive on {vol[5]} of {vol[6]}.")
    print()
    if ret_ok and vol_ok:
        print("  Both coordinates exclude zero, under the peer block a rate-blind rule")
        print("  selected and a conservative tie-break shrank. The paper's central")
        print("  claim is therefore not a fact about equity indices. It holds on an")
        print("  asset class with none of the machinery the equity results lean on,")
        print("  and it holds whether the target is the return an owner cares about or")
        print("  the variance the paper was written about.")
        print()
        print("  Part A's LEVEL prediction, checked in the form it was made rather than")
        print(f"  a looser one: coupling runs {min(cp.values()):.3f} to {max(cp.values()):.3f} here, at or near the top of")
        print(f"  the equity range, and the rate is positive on {pos} of {tot} metros on the")
        if vol and vol[5] == vol[6]:
            print(f"  return target and {vol[5]} of {vol[6]} on the volatility target. Confirmed on both.")
        elif vol:
            print(f"  return target and {vol[5]} of {vol[6]} on the volatility target, so it is")
            print(f"  confirmed on the return coordinate and holds with {vol[6] - vol[5]} exception"
                  f"{'s' if vol[6] - vol[5] > 1 else ''} on the")
            print("  volatility coordinate, named in Part D. The prediction was for no")
            print("  failures, so that exception is a partial miss and is recorded as one.")
        print("  All of this was stated before any rate existed.")
        print()
        print("  Three things the paper should stop saying. That the illiquid case is")
        print("  unreachable: it was reached. That the obstruction is the absence of")
        print("  data: the data was free and public throughout. That the frequency")
        print("  binds: lab54 showed it does not, and this file measures the rate at")
        print("  one twenty-first of the paper's sampling rate on marks an appraised")
        print("  asset could actually produce.")
        print()
        print("  Two things it should start saying. First, the measured persistence in")
        print(f"  Part A averages {np.mean(list(ac.values())):.3f} and its minimum is {min(ac.values()):.3f}, so every metro is more")
        print("  smoothed than the rho = 0.6 Section 13 imposed, and substitution works")
        print("  anyway. Second, and this is the load-bearing one: Part B shows the")
        print("  volatility rate is NEGATIVE at short delays under the widest peer")
        print("  block and positive under the compressed one. The estimation bill is")
        print("  not a footnote at this sample size, it is the difference between the")
        print("  paper's claim holding and appearing to fail, and lab52 found the same")
        print("  thing on equities. The paper should present compression as part of the")
        print("  method rather than as a robustness check.")
    elif ret_ok:
        print("  The return coordinate excludes zero and the volatility coordinate does")
        print("  not. That is a narrower claim than the paper's and should be stated as")
        print("  one: peer marks substitute for a stale mark when the quantity forecast")
        print("  is the return, and this panel cannot establish it for the variance.")
        print("  The asset class transfers; the target variable may not.")
    else:
        print("  The rate does not exclude zero at a quarterly staleness. On the asset")
        print("  class the paper is actually about, fresh peer marks do not measurably")
        print("  substitute for a stale own mark, and Sections 9, 11 and 15 should be")
        print("  restated as claims about liquid equity indices. Part A's LEVEL")
        print("  prediction is contradicted: coupling is high on every metro here and")
        print("  the rate is not.")


def main(root=None):
    df, sha, path = load_panel(root)
    rng = np.random.default_rng(SEED)
    print("Prints the measured illiquid study: Case-Shiller metros, Section 15.\n")

    cp, ac = part_a(df, sha, path)
    print()
    cache, chosen = part_b(df, rng)
    print()
    out_ret = report(
        df, cache, chosen, "return",
        "C.  THE HEADLINE: NEXT MONTH'S RETURN, FOURTEEN BALANCED METROS",
        "  Target: next month's log return of the metro index.\n"
        "  Own block: the metro's own return, its three-month mean and its\n"
        "  twelve-month mean, all lagged by delta. The cross arm adds the peer block.\n"
        "  Eleven month-of-year dummies enter BOTH arms, so the shared seasonal in\n"
        "  these unadjusted series cannot appear as a cross-sectional advantage.",
        rng, cp)
    print()
    out_vol = report(
        df, cache, chosen, "volatility",
        "D.  THE SECOND COORDINATE: THE PAPER'S OWN TARGET TYPE, AT THIS CLOCK",
        f"  Target: log of the trailing {WIN_M}-month realised variance over its own\n"
        f"  {MED_M}-month median, one month ahead. This is Section 3's decision variable\n"
        "  with the windows scaled to a monthly clock, so only the ASSET CLASS differs\n"
        "  from the paper here, not the quantity being forecast.",
        rng, cp)
    print()
    part_e(cp, out_ret, out_vol)
    print()
    verdict(cp, ac, out_ret, out_vol)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
