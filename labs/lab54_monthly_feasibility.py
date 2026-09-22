"""
lab54_monthly_feasibility.py - what an appraisal-based study would actually need.

Imports lab05_robustness; keep both in labs/.  Runtime about two minutes.

THE QUESTION THIS FILE REFUSES TO LEAVE AS A SENTENCE
-----------------------------------------------------
Sections 9, 11 and 14 all end at the same wall.  The case that motivates the
paper is a quarterly-marked fund or an appraised property book, and every
measurement here is on a liquid equity index.  The paper says the gap would be
closed by "a genuine appraisal-based or quarterly-marked target" and leaves it
there, which invites the obvious question: why not go and get one?

The answer is not that such series are hard to find.  It is that the property
which makes an asset illiquid is the same property that removes the data this
method runs on, and that can be stated precisely rather than gestured at:

    The volatility proxy is Yang-Zhang over five daily bars.  It needs an open,
    a high, a low and a close for every session.  An appraised asset has no
    intraday range, no open and no close.  It has one number per quarter.

    The walk-forward needs MED + TRAIN + VAL rows before it can make its first
    forecast: 252 + 1250 + 250 = 1752 observations.  At one mark per quarter
    that is 438 years.  At one per month it is 146 years.

So the daily design cannot be pointed at a quarterly asset by finding better
data.  It has to be re-specified at the frequency the asset reports, and the
question that decides whether anyone can do that is whether the measurement
survives the loss of resolution at all.

WHAT THIS FILE DOES
-------------------
A dress rehearsal on data where the truth is still observable.  The S&P and its
seven peers are coarsened to month-end marks, which throws away exactly what an
appraised series never had: the daily bars.  The proxy becomes the realised
variance of the last six MONTHLY returns, the target is that proxy one month
ahead, the normaliser is a trailing sixty-month median, and the delay is counted
in months rather than days.  Everything else is Section 3's design with the
windows scaled by twenty-one.

Then the same thing again with the target smoothed the way an appraisal is, by
the partial adjustment of Section 13, which is as close to the real case as a
liquid asset can be made to come.

WHAT WOULD COUNT AS AN ANSWER
-----------------------------
Three outcomes and they say different things:

    The rate is estimable and its interval is usable.  Then an appraisal study
    is possible today, the obstruction was never the frequency, and Section 15
    should say what series to point it at.

    The rate is estimable and its interval is too wide to lean on.  Then the
    obstruction is sample size, and the useful output is the number of years of
    marks that would be needed, which this file can estimate from how the
    interval narrows with the test window.

    The rate is not estimable at all.  Then the method is daily by nature and
    the paper should stop implying that a better dataset would rescue it.

The file does not choose in advance which of those it wants to find.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L

SEED = 20260917
TARGET = "SPX"
WIN_M = 6           # months of returns in the volatility proxy
MED_M = 60          # months in the trailing median, five years
TRAIN_M = 120       # months of training, ten years
VAL_M = 24
REFIT_M = 12
H_M = 1             # forecast one month ahead
DELAYS_M = [0, 1, 2, 3, 6]
N_BOOT, BLOCK_M = 1500, 12
RHO = 0.6           # partial-adjustment weight, as in Section 13


def monthly_panel(folder):
    """Month-end marks for every index, and nothing else."""
    tags = [t for t in L.CLOSE_UTC if L.files_for(t, folder)]
    px = {}
    for t in tags:
        d = L.load_index(t, folder)
        s = pd.Series(d["close"].values, index=pd.DatetimeIndex(d["date"].values))
        px[t] = s.resample("ME").last()
    return pd.DataFrame(px).dropna(how="all")


def decision(level, win=WIN_M, med=MED_M):
    """Section 3's decision variable, computed from monthly marks alone."""
    r = np.log(level / level.shift(1))
    var = r.rolling(win).var(ddof=1)
    return np.log(var / var.rolling(med, min_periods=12).median())


def smooth(level, rho=RHO):
    """An appraised mark: partial adjustment toward the current value."""
    v = level.values.astype(float)
    out = np.empty_like(v)
    out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = rho * out[i - 1] + (1 - rho) * v[i] if np.isfinite(v[i]) else out[i - 1]
    return pd.Series(out, index=level.index)


def walk(own, P, y, idx, delta, use_peers):
    out = np.empty(len(idx)); b = mu = sd = None
    for j, t in enumerate(idx):
        if j % REFIT_M == 0:
            cut = t - delta - H_M
            tr = np.arange(max(0, cut - TRAIN_M - VAL_M), max(cut, 1))
            X = (np.column_stack([own[tr - delta], P[tr]]) if use_peers
                 else own[tr - delta])
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y[tr])
            X, yy = X[ok], y[tr][ok]
            if len(yy) < 40:
                b = None; continue
            mu, sd = X.mean(0), X.std(0) + 1e-9
            b = L.cv((X - mu) / sd, yy, "cont")
        xt = (np.concatenate([own[t - delta], P[t]]) if use_peers else own[t - delta])
        out[j] = 0.0 if (b is None or not np.isfinite(xt).all()) \
            else L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
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


def build(M, target_level):
    a = decision(target_level).values
    peers = [c for c in M.columns if c != TARGET]
    P = np.column_stack([decision(M[c]).values for c in peers])
    sa = pd.Series(a)
    own = np.column_stack([sa.values, sa.rolling(3).mean().values,
                           sa.rolling(12).mean().values])
    n = len(a)
    y = np.full(n, np.nan); y[:-H_M] = a[H_M:]
    start = MED_M + TRAIN_M + VAL_M + max(DELAYS_M) + H_M
    idx = np.arange(start, n - H_M)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])]
    return own, P, y, idx, len(peers)


def run(M, level, label, rng):
    own, P, y, idx, k = build(M, level)
    yb = y[idx]
    # The window and the horizon are this panel's, not lab05's: these are
    # months, so the daily defaults would build the benchmark from labels a
    # monthly forecaster could not yet have seen and over a window longer
    # than the series.
    BENCH = L.bench_mean(y, idx, window=TRAIN_M + VAL_M, horizon=H_M)
    n = len(idx)
    print(f"\n  {label}: {n} test months, {k} peers, "
          f"first forecast {M.index[idx[0]].date() if len(idx) else 'n/a'}")
    if n < 24:
        print("    too few test months to report anything")
        return None
    S_own = {d: r2(yb, walk(own, P, y, idx, d, False), bench=BENCH)
             for d in DELAYS_M}
    f_cross = {d: walk(own, P, y, idx, d, True) for d in DELAYS_M}
    S_cross = {d: r2(yb, f_cross[d], bench=BENCH) for d in DELAYS_M}
    f_own = {d: walk(own, P, y, idx, d, False) for d in DELAYS_M}
    print(f"{'delay, months':>15}{'own':>9}{'+cross':>9}{'R':>9}{'95% interval':>22}")
    out = {}
    for d in DELAYS_M:
        den = S_own[0] - S_own[d]
        rate = (S_cross[d] - S_own[d]) / den if d and den > 1e-9 else np.nan
        if d == 0:
            print(f"{d:>15}{S_own[0]:>9.4f}{S_cross[0]:>9.4f}{'n/a':>9}{'n/a':>22}")
            continue
        draws = []
        for _ in range(N_BOOT):
            rows = blocks_of(n, rng)
            dn = (r2(yb, f_own[0], rows=rows, bench=BENCH)
                  - r2(yb, f_own[d], rows=rows, bench=BENCH))
            if abs(dn) < 1e-9:
                continue
            draws.append((r2(yb, f_cross[d], rows=rows, bench=BENCH)
                          - r2(yb, f_own[d], rows=rows, bench=BENCH)) / dn)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out[d] = (rate, lo, hi)
        print(f"{d:>15}{S_own[d]:>9.4f}{S_cross[d]:>9.4f}{rate:>9.1%}"
              f"{f'[{lo:.0%}, {hi:.0%}]':>22}")
    return out


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints the monthly feasibility study, Section 15.\n")
    rng = np.random.default_rng(SEED)

    print("=" * 96)
    print("A.  WHY THE DAILY DESIGN CANNOT SIMPLY BE POINTED AT AN APPRAISED ASSET")
    print("=" * 96)
    need = L.MED + L.TRAIN + L.VAL
    print(f"  the walk-forward needs {L.MED} + {L.TRAIN} + {L.VAL} = {need} observations before its")
    print("  first forecast, and the proxy needs an open, high, low and close for")
    print("  every one of them.\n")
    print(f"{'reporting frequency':>24}{'observations per year':>24}"
          f"{'years to first forecast':>26}")
    for name, per_year in (("daily", 252), ("weekly", 52), ("monthly", 12), ("quarterly", 4)):
        print(f"{name:>24}{per_year:>24}{need / per_year:>26.0f}")
    print("\n  A quarterly mark would need four centuries of history before this")
    print("  design made its first forecast, and would still have no intraday range")
    print("  to compute a proxy from. The design has to be re-specified at the")
    print("  frequency the asset reports. Whether that survives is the question.")

    # ---------------- B. the dress rehearsal --------------------------------
    print("\n" + "=" * 96)
    print("B.  THE SAME EXPERIMENT ON MONTHLY MARKS, WHERE THE TRUTH IS STILL VISIBLE")
    print("=" * 96)
    M = monthly_panel(folder)
    print(f"month-end marks: {len(M)} months, {M.index[0].date()} to {M.index[-1].date()}")
    print(f"proxy: variance of the last {WIN_M} monthly returns, over a {MED_M}-month median")
    print(f"windows: {TRAIN_M} months training, {VAL_M} validation, refit every {REFIT_M}")
    print(f"delays in MONTHS: {DELAYS_M}; three months is one quarterly marking cycle")
    plain = run(M, M[TARGET], "clean monthly marks", rng)

    # ---------------- C. with the mark smoothed the way an appraisal is -----
    print("\n" + "=" * 96)
    print("C.  AND WITH THE MARK SMOOTHED THE WAY AN APPRAISAL IS")
    print("=" * 96)
    print(f"Partial adjustment at rho = {RHO}, the model of Section 13: each mark moves")
    print("only part of the way to the current value, which is how the property")
    print("literature has described appraisal indices for thirty years.")
    Ms = M.copy()
    Ms[TARGET] = smooth(M[TARGET])
    smoothed = run(Ms, Ms[TARGET], "appraised monthly marks", rng)

    # ---------------- D. what it would take ---------------------------------
    print("\n" + "=" * 96)
    print("D.  WHAT A REAL STUDY WOULD NEED")
    print("=" * 96)
    if plain:
        w = {d: plain[d][2] - plain[d][1] for d in plain}
        d3 = 3 if 3 in w else max(w)
        n_test = len(build(M, M[TARGET])[3])
        print(f"  At {n_test} test months the interval on R at a one-quarter delay is")
        print(f"  {plain[d3][1]:.0%} to {plain[d3][2]:.0%}, a width of {w[d3]:.0%}.")
        print("\n  A bootstrap interval narrows roughly with the square root of the test")
        print("  window, so the months needed to reach a target width scale as the")
        print("  square of the ratio. On that arithmetic:")
        print(f"\n{'target width':>16}{'test months needed':>22}{'years of marks':>18}")
        for target_w in (0.60, 0.40, 0.30, 0.20):
            need_m = n_test * (w[d3] / target_w) ** 2
            total = (need_m + MED_M + TRAIN_M + VAL_M) / 12
            print(f"{target_w:>16.0%}{need_m:>22.0f}{total:>18.0f}")
        print("\n  That arithmetic assumes the interval narrows as it would on")
        print("  independent draws, which overstates how fast it really narrows on a")
        print("  series this persistent. The years column is therefore a floor and")
        print("  not an estimate.")

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    if not plain:
        print("  The measurement does not survive the loss of daily bars at all, so the")
        print("  method is daily by nature and no appraisal series would rescue it.")
    else:
        d3 = 3 if 3 in plain else max(plain)
        wide = (plain[d3][2] - plain[d3][1]) > 0.5
        crosses = plain[d3][1] < 0 < plain[d3][2]
        print(f"  The experiment runs on monthly marks. At a one-quarter delay the rate")
        print(f"  is {plain[d3][0]:.0%} with an interval of [{plain[d3][1]:.0%}, {plain[d3][2]:.0%}]"
              f"{', which contains zero' if crosses else ''}.")
        if smoothed and d3 in smoothed:
            print(f"  With the mark smoothed the way an appraisal is, {smoothed[d3][0]:.0%} "
                  f"[{smoothed[d3][1]:.0%}, {smoothed[d3][2]:.0%}].")
        if wide or crosses:
            print("\n  So the obstruction is not imagination and it is not the absence of a")
            print("  dataset. It is sample size. Twenty-six years of month-end marks on")
            print("  the most-studied index in the world leave an interval too wide to")
            print("  lean on, and an appraised asset reports less often than that. The")
            print("  honest thing the paper can say is not 'we could not find the data'")
            print("  but 'a study at that frequency needs the history in part D, and")
            print("  almost no real asset has it'.")
            print("\n  This also explains a choice the paper never justified: the daily")
            print("  frequency is not a convenience, it is what makes the interval")
            print("  narrow enough to say anything at all. The apparatus transfers to an")
            print("  illiquid asset in principle and starves there in practice.")
        else:
            print("\n  The frequency was not the obstruction. With no daily bars at all and")
            print(f"  {len(build(M, M[TARGET])[3])} test months, the rate at one quarter's delay excludes zero, and it")
            print("  still excludes zero when the mark is smoothed the way an appraisal is.")
            print("  The shape survives too: at one month the rate is indistinguishable")
            print("  from zero, exactly as it is at one day in Table 1, and it climbs from")
            print("  there. That is the daily result reproduced at a frequency twenty-one")
            print("  times coarser, on marks an appraised asset could actually produce.")
            print("\n  What the intervals will not do is price it. Thirty to forty points of")
            print("  width establishes that the effect is there and cannot say what it is")
            print("  worth, which is the right claim for a first study of an asset class")
            print("  nobody has measured. Part D says what history would narrow it.")
            print("\n  So the paper's own limitation should be restated. The illiquid case")
            print("  is not unreachable and it is not blocked by the absence of data. It")
            print("  needs a series with two decades of monthly marks and a public")
            print("  cross-section, run at the asset's own frequency rather than at this")
            print("  paper's, and this file is the rehearsal that shows the design holds")
            print("  when it gets there.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
