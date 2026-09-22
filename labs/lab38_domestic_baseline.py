"""
lab38_domestic_baseline.py - is the substitution rate an artefact of a weak control?

Imports lab05_robustness; keep both in labs/.  Runtime about two minutes.

THE OBJECTION, WHICH IS THE SHARPEST ONE THIS PROJECT HAS FACED
---------------------------------------------------------------
The substitution rate is

    R(delta) = [ S_alt(delta) - S_own(delta) ] / [ S_own(0) - S_own(delta) ]

and S_own appears three times in it.  Every number the paper reports about what
breadth is worth is therefore a statement about how good the DOMESTIC model is.
If that model is weak, the denominator - "what delay destroys" - is inflated by
whatever the weak model failed to extract from fresh data, and R(delta) is
correspondingly flattered.

The paper is open to the charge because of where it spent its effort.  Section
5.1 compresses the foreign block into a real-time factor and reports the gain.
Section 5.3 splits that block into a factor and a remainder.  Section 7.1 tries
the whole cross-section as one regressor instead of seven.  All of that is work
on the TREATMENT.  The CONTROL has been a three-feature ridge - the level, the
five-day mean and the twenty-two-day mean of the decision variable - since the
first draft, and nothing has ever varied it.  A referee is entitled to say that
the paper optimised the treatment and left the control alone.

WHAT THIS FILE DOES
-------------------
Five stronger domestic blocks, each a superset of the paper's, each given the
same walk-forward, the same refit schedule and the same ridge:

    HAR3      the paper: a(t-d), mean over 5, mean over 22          3 features
    +LAGS     four further own lags, a(t-d-1) .. a(t-d-4)           7
    +LONG     quarterly and annual means, 66 and 252 days           5
    +LEV      the leverage terms: signed return, its negative part,
              and a five-day mean of that negative part            6
    +MEAS     a second realised measure (22-day Yang-Zhang) and the
              LEVEL of the trailing median, which the paper's
              normalisation hides from every model                  5
    ALL       all of the above at once                             14

For each, S_own is that block delta days stale and S_alt is the same block plus
the seven foreign closes, current.  R(delta) is recomputed inside each arm, so
every rate is measured against its own control rather than against the paper's.

WHAT WOULD COUNT AS A PROBLEM
-----------------------------
If a stronger control shrinks R(delta) materially, the headline is partly an
artefact of the baseline and the paper must requote it against the better model.
If R(delta) is stable across controls that differ in their own forecasting
power, the rate is a property of the information rather than of the estimator,
which is what the paper has been claiming without testing.

A third outcome is possible and would also be worth reporting: the richer blocks
may forecast WORSE, because lab07 established that regressors cost R-squared
whether or not they inform.  That would not vindicate the three-feature choice
by luck - it would show the paper's control is already at the point where adding
domestic information costs more than it returns, which is a stronger defence
than never having looked.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L

BENCH = None          # set in main(); read by the scorers below

SEED = 20260915
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 1, 3, 5, 8, 13, 21, 34, 55]
# The block is lab05's measured one, not 2 * HORIZON.  The h-day overlap in
# the target is not the only dependence in a loss difference: it also
# inherits the common factor's persistence, which lab58 measures at 0.48
# autocorrelation at lag 10 and not below 0.05 until lag 38.  A block
# shorter than the dependence leaves it inside the resample and the
# interval comes out too narrow.
N_BOOT, BLK = 1500, L.BLOCK
ARMS = ["HAR3", "+LAGS", "+LONG", "+LEV", "+MEAS", "ALL"]


def hac_se(d, lag=None):
    """Newey-West standard error of a mean, at lab05's measured bandwidth.

    The default is L.HAC_LAG, not the automatic 4 (n/100)^(2/9) rule.  That
    rule gives 9 lags here and truncates the kernel while the loss
    differential still carries 0.52 autocorrelation, which understates every
    standard error in this project by up to 54% and every statistic formed
    from one in the paper's favour.  lab58 part 6 measures it.
    """
    lag = L.HAC_LAG if lag is None else lag
    n = len(d); d = d - d.mean()
    s = (d @ d) / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * ((d[k:] @ d[:-k]) / n)
    return np.sqrt(max(s, 1e-18) / n)


def norm_p(z):
    from math import erf, sqrt
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def blocks(folder):
    """Every domestic feature this file considers, on the paper's own panel."""
    D, peers, lag = L.build(folder, TARGET)
    a = D[TARGET].values
    sa = pd.Series(a)

    raw = L.load_index(TARGET, folder)
    px = pd.Series(raw["close"].values, index=pd.DatetimeIndex(raw["date"].values))
    r = np.log(px / px.shift(1)).reindex(D.index).ffill(limit=5)
    neg = r.clip(upper=0.0)

    yz22 = L.yang_zhang(raw, n=22).reindex(D.index).ffill(limit=5)
    m22 = yz22.rolling(L.MED, min_periods=30).median()
    yz5 = L.yang_zhang(raw, n=L.WINDOW).reindex(D.index).ffill(limit=5)
    m5 = yz5.rolling(L.MED, min_periods=30).median()

    F = {
        "HAR3": np.column_stack([sa.values,
                                 sa.rolling(5).mean().values,
                                 sa.rolling(22).mean().values]),
        "LAGS": np.column_stack([sa.shift(k).values for k in (1, 2, 3, 4)]),
        "LONG": np.column_stack([sa.rolling(66).mean().values,
                                 sa.rolling(252).mean().values]),
        "LEV": np.column_stack([r.values, neg.values,
                                neg.rolling(5).mean().values]),
        # A second realised measure, and the LEVEL the paper's normalisation hides:
        # every model so far sees variance only relative to its own median, so it
        # cannot tell a quiet decade from a loud one.  This gives it that.
        "MEAS": np.column_stack([np.log(yz22.values / m22.values),
                                 np.log(m5.values)]),
    }
    P = D[peers].values
    n = len(D)
    y = np.full(n, np.nan); y[:-H] = a[H:]
    start = L.MED + L.TRAIN + L.VAL + max(DELAYS) + H
    idx = np.arange(start, n - H)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])]
    return F, P, y, idx, len(peers), D


def own_block(F, arm):
    if arm == "HAR3":
        return F["HAR3"]
    if arm == "ALL":
        return np.column_stack([F["HAR3"], F["LAGS"], F["LONG"], F["LEV"], F["MEAS"]])
    return np.column_stack([F["HAR3"], F[arm[1:]]])


def walk(own, P, y, idx, delta, use_peers):
    out = np.empty(len(idx)); b = mu = sd = None
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), max(cut, 1))
            X = (np.column_stack([own[tr - delta], P[tr]]) if use_peers
                 else own[tr - delta])
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y[tr])
            X, yy = X[ok], y[tr][ok]
            if len(yy) < 200:
                b = None; continue
            mu, sd = X.mean(0), X.std(0) + 1e-9
            b = L.cv((X - mu) / sd, yy, "cont")
        xt = (np.concatenate([own[t - delta], P[t]]) if use_peers
              else own[t - delta])
        out[j] = 0.0 if (b is None or not np.isfinite(xt).all()) \
            else L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
    return out


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints Table 18 and the normalisation result of Section 9.1.\n")

    F, P, y, idx, k, D = blocks(folder)
    yb = y[idx]
    global BENCH
    BENCH = L.bench_mean(y, idx)
    n = len(idx)
    print(f"target {TARGET}, {n} test days, "
          f"{D.index[idx[0]].date()} to {D.index[idx[-1]].date()}, {k} foreign peers")
    print(f"{'arm':>8}{'features':>10}")
    for arm in ARMS:
        print(f"{arm:>8}{own_block(F, arm).shape[1]:>10}")

    # per-day squared errors for every arm, both models, every delay
    err = {}
    for arm in ARMS:
        ob = own_block(F, arm)
        for d in DELAYS:
            err[(arm, d, "own")] = (yb - walk(ob, P, y, idx, d, False)) ** 2
            err[(arm, d, "cross")] = (yb - walk(ob, P, y, idx, d, True)) ** 2
    base = (yb - BENCH) ** 2

    def r2(arm, d, m, s=None):
        v = err[(arm, d, m)]
        return 1 - (v.sum() / base.sum() if s is None
                    else v[s].sum() / base[s].sum())

    # ---------------- A -----------------------------------------------------
    print("\n" + "=" * 98)
    print("A.  DOES A STRONGER DOMESTIC MODEL FORECAST BETTER AT ALL?")
    print("=" * 98)
    print("Out-of-sample R-squared of the domestic block alone, delta days stale.")
    print("If no arm beats HAR3, the control was never weak and the objection fails")
    print("at the first step.\n")
    print(f"{'delta':>6}" + "".join(f"{a:>11}" for a in ARMS))
    better = []
    for d in DELAYS:
        row = f"{d:>6}"
        for arm in ARMS:
            row += f"{r2(arm, d, 'own'):>11.4f}"
        print(row)
    print("\nAgainst HAR3, with Giacomini-White on the paired loss differential:\n")
    print(f"{'delta':>6}" + "".join(f"{a:>22}" for a in ARMS[1:]))
    for d in DELAYS:
        row = f"{d:>6}"
        for arm in ARMS[1:]:
            dd = err[("HAR3", d, "own")] - err[(arm, d, "own")]
            z = dd.mean() / hac_se(dd)
            if z > 1.96:
                better.append((arm, d))
            row += f"{r2(arm, d, 'own') - r2('HAR3', d, 'own'):>+13.4f}" \
                   f"{('*' if z > 1.96 else ('-' if z < -1.96 else ' ')):>2}" \
                   f"{'':>7}"
        print(row)
    print("\n  * = significantly better than HAR3   - = significantly worse")
    print(f"  cells where a richer domestic block significantly beats the paper's: "
          f"{len(better)} of {len(DELAYS) * (len(ARMS) - 1)}")

    # pick the strongest arm by MEAN own-only R2 across delays - not by R(delta),
    # which is the thing under test and must not choose its own control.
    mean_r2 = {a: float(np.mean([r2(a, d, "own") for d in DELAYS])) for a in ARMS}
    best = max(mean_r2, key=mean_r2.get)
    print(f"\n  mean own-only R-squared across delays: "
          + ", ".join(f"{a} {mean_r2[a]:.4f}" for a in ARMS))
    print(f"  strongest domestic block by that measure: {best}")
    print("  (chosen on own-only skill, never on the substitution rate, so the")
    print("   control is not selected by the quantity it is about to be used to test)")

    # ---------------- B -----------------------------------------------------
    print("\n" + "=" * 98)
    print("B.  THE SUBSTITUTION RATE, MEASURED AGAINST EACH CONTROL")
    print("=" * 98)
    print("R(delta) recomputed inside every arm: same foreign block, different")
    print("domestic baseline in all three places the rate uses it.\n")
    print(f"{'delta':>6}" + "".join(f"{a:>11}" for a in ARMS))
    rates = {}
    for d in DELAYS[1:]:
        row = f"{d:>6}"
        for arm in ARMS:
            den = r2(arm, 0, "own") - r2(arm, d, "own")
            v = (r2(arm, d, "cross") - r2(arm, d, "own")) / den if den > 1e-9 else np.nan
            rates[(arm, d)] = v
            row += (f"{v:>11.0%}" if np.isfinite(v) else f"{'n/a':>11}")
        print(row)

    # ---------------- C -----------------------------------------------------
    print("\n" + "=" * 98)
    print("C.  DOES THE HEADLINE MOVE?")
    print("=" * 98)
    print(f"The paper's rate against the strongest control ({best}), with a paired")
    print("bootstrap: both ratios recomputed on the SAME resampled days, numerator")
    print("and denominator together, so the interval is on the DIFFERENCE.\n")
    rb = np.random.default_rng(SEED)
    st = rb.integers(0, n - BLK + 1, size=(N_BOOT, int(np.ceil(n / BLK))))
    offs = np.arange(BLK)
    S = [(st[i][:, None] + offs).ravel()[:n] for i in range(N_BOOT)]

    print(f"{'delta':>6}{'R(d) HAR3':>12}{'R(d) ' + best:>14}{'difference':>13}"
          f"{'95% CI':>24}")
    moved = []
    for d in DELAYS[1:]:
        diffs = np.empty(N_BOOT)
        for i, s in enumerate(S):
            out = []
            for arm in ("HAR3", best):
                den = r2(arm, 0, "own", s) - r2(arm, d, "own", s)
                out.append((r2(arm, d, "cross", s) - r2(arm, d, "own", s)) / den
                           if abs(den) > 1e-9 else np.nan)
            diffs[i] = out[1] - out[0]
        good = diffs[np.isfinite(diffs)]
        lo, hi = np.percentile(good, [2.5, 97.5])
        pt = rates[(best, d)] - rates[("HAR3", d)]
        if lo > 0 or hi < 0:
            moved.append(d)
        print(f"{d:>6}{rates[('HAR3', d)]:>12.0%}{rates[(best, d)]:>14.0%}"
              f"{pt:>+13.1%}" + f"[{lo:>+9.1%},{hi:>+9.1%}]".rjust(24))

    print(f"\n  delays where the rate moves significantly with the control: "
          f"{len(moved)} of {len(DELAYS) - 1} {moved if moved else ''}")

    # ---------------- D -----------------------------------------------------
    print("\n" + "=" * 98)
    print("D.  WHICH OF THE TWO DOES IT, AND IS THE GAIN LEGITIMATE?")
    print("=" * 98)
    print("+MEAS carries two features and they are not the same kind of thing. One is a")
    print("second realised measure, a 22-day Yang-Zhang, which is simply less noisy than")
    print("the 5-day one. The other is log M, the LEVEL of the trailing median, which")
    print("every model in the paper is blind to because the normalisation divides it out.")
    print("The distinction matters: the target is log(variance / M at t+5), so a model")
    print("that sees log M at t-delta can partly reconstruct the target's own denominator,")
    print("M being slow. That is admissible - every feature is dated t-delta - but it is")
    print("a fact about how the target is built as much as about volatility, and it would")
    print("be wrong to report it as though the two features were interchangeable.\n")
    for nm, cols in (("+VAR22", [F["MEAS"][:, [0]]]), ("+LEVEL", [F["MEAS"][:, [1]]])):
        F[nm] = np.column_stack(cols)
    for nm in ("+VAR22", "+LEVEL"):
        ob = np.column_stack([F["HAR3"], F[nm]])
        for d in DELAYS:
            err[(nm, d, "own")] = (yb - walk(ob, P, y, idx, d, False)) ** 2
            err[(nm, d, "cross")] = (yb - walk(ob, P, y, idx, d, True)) ** 2
    print(f"{'delta':>6}{'HAR3':>10}{'+VAR22':>10}{'+LEVEL':>10}{'+MEAS':>10}"
          f"{'  level - var22':>17}")
    lvl = []
    for d in DELAYS:
        gv = r2("+VAR22", d, "own") - r2("HAR3", d, "own")
        gl = r2("+LEVEL", d, "own") - r2("HAR3", d, "own")
        if gl > gv:
            lvl.append(d)
        print(f"{d:>6}{r2('HAR3', d, 'own'):>10.4f}{r2('+VAR22', d, 'own'):>10.4f}"
              f"{r2('+LEVEL', d, 'own'):>10.4f}{r2('+MEAS', d, 'own'):>10.4f}"
              f"{gl - gv:>+17.4f}")
    print(f"\n  delays where the LEVEL contributes more than the second measure: "
          f"{len(lvl)} of {len(DELAYS)} {lvl if lvl else ''}")
    rl = {}
    print(f"\n{'delta':>6}{'R(d) HAR3':>12}{'R(d) +VAR22':>14}{'R(d) +LEVEL':>14}")
    for d in DELAYS[1:]:
        row = f"{d:>6}{rates[('HAR3', d)]:>12.0%}"
        for nm in ("+VAR22", "+LEVEL"):
            den = r2(nm, 0, "own") - r2(nm, d, "own")
            rl[(nm, d)] = ((r2(nm, d, "cross") - r2(nm, d, "own")) / den
                           if den > 1e-9 else np.nan)
            row += (f"{rl[(nm, d)]:>14.0%}" if np.isfinite(rl[(nm, d)])
                    else f"{'n/a':>14}")
        print(row)
    if len(lvl) >= len(DELAYS) - 1:
        print("\n  The level does the work almost everywhere. So the stronger control is")
        print("  strong mainly because it can see something the paper's normalisation")
        print("  removed, and the honest reading is narrower than 'the baseline was weak':")
        print("  a forecaster told how volatile the world has lately been, in absolute")
        print("  terms, needs less help from abroad. The paper should quote the rate")
        print("  against this control AND say where its strength comes from.")
    elif not lvl:
        print("\n  The second realised measure does the work, not the level, so the gain is")
        print("  ordinary: a less noisy estimate of the same quantity. No caveat about the")
        print("  target's construction is needed.")
    else:
        print("\n  The two contribute at different delays, so neither explanation stands")
        print("  alone and the table above is the finding.")

    # ---------------- verdict ----------------------------------------------
    print("\n" + "=" * 98)
    print("VERDICT")
    print("=" * 98)
    if not better:
        print("  No richer domestic block beats the paper's three features anywhere.")
        print("  The control is not weak, and the objection that R(delta) rests on an")
        print("  under-specified baseline does not survive: there is no better baseline")
        print("  to rest it on, among the five tried here.")
    else:
        print(f"  A richer domestic block beats the paper's in {len(better)} cells, so")
        print(f"  the control CAN be improved and {best} improves it.")
    if not moved:
        print("  The substitution rate does not move significantly with the control at")
        print("  any delay. R(delta) is a property of the information rather than of")
        print("  the domestic estimator, which is what the paper assumed and had not")
        print("  tested.")
    else:
        print(f"  The rate moves significantly at {moved}. The headline is partly a")
        print("  statement about the baseline, and the paper must quote it against the")
        print("  stronger control rather than the three-feature one.")
    worse = [a for a in ARMS[1:] if mean_r2[a] < mean_r2["HAR3"]]
    if worse:
        print(f"  Arms that forecast WORSE than the paper's on average: {worse}. That is")
        print("  lab07's estimation cost again, on the domestic side: the extra domestic")
        print("  features cost more to estimate than they return.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
