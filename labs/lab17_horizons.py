"""
lab17_horizons.py - does the substitution rate survive a change of forecast horizon?

Imports lab05_robustness; keep both in labs/.  Runtime about four minutes.

THE OBJECTION
-------------
Every number in this paper forecasts five trading days ahead.  A referee will ask
whether that is a result or a setting, and the question is fair: the whole story
is about how a delayed forecaster is rescued by contemporaneous information, and
how much rescuing is available could easily depend on how far ahead they are
being asked to see.  So run the same experiment at h = 1, 10 and 21 as well.

WHAT MAKES h = 1 DIFFERENT, AND WHY IT IS STILL THE RIGHT TEST
--------------------------------------------------------------
The volatility proxy is a Yang-Zhang estimator over a FIVE-day window.  That
window has nothing to do with the forecast horizon - it is how the latent
variance is measured - but it does mean the target and the features can overlap.
At h = 5 they do not: sigma^2 over (t+1 .. t+5) shares no day with sigma^2 over
(t-4 .. t).  At h = 1 the target shares four of its five days with the domestic
feature the model already holds, and at h = 10 and 21 there is no overlap at all.

That is not a leak - every feature is still dated t - delta or earlier for the
domestic block and t for the others, and the outcome is still strictly in the
future.  But it does change what the h = 1 column MEANS.  A domestic forecaster
at delta = 0 is being asked to predict an object four-fifths of which they can
already see, so their R-squared will be high for a reason that has nothing to do
with skill, and the room left for any substitute to recover is correspondingly
small.  The right way to read h = 1 is therefore as the hardest case for the
paper's claim rather than as a fifth equally-weighted data point, and part A
measures the overlap rather than leaving the reader to assume it.

WHAT WOULD COUNT AS A PROBLEM
-----------------------------
If the substitution rate at eleven weeks collapses at some horizon, or swings
far outside the [64%, 79%] interval the paper quotes at h = 5, then five days
was doing work the paper does not acknowledge and Section 4 has to be rewritten
around it.  If the rate is stable across a twenty-one-fold change in horizon,
the headline is a property of the information rather than of the setting.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L

SEED = 20260913
TARGET = "SPX"
HORIZONS = [1, 5, 10, 21]
DELAYS = [0, 3, 5, 13, 21, 55]
N_BOOT = 2000


def panel(folder):
    D, peers, lag = L.build(folder, TARGET)
    a = D[TARGET].values
    sa = L.pd.Series(a)
    own = np.column_stack([sa.values, sa.rolling(5).mean().values,
                           sa.rolling(22).mean().values])
    return own, D[peers].values, a, len(D), len(peers)


def walk(own, P, y, idx, delta, h, use_peers):
    """lab05's walk-forward with the horizon as an argument rather than a constant.

    The admissibility rule is the one lab04 states: a training pair is usable
    only once its own outcome was resolvable at t - delta, so the cut is
    t - delta - h and it moves with the horizon.
    """
    out = np.empty(len(idx)); b = mu = sd = None
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - h
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), cut)
            X = np.column_stack([own[tr - delta]] + ([P[tr]] if use_peers else []))
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y[tr])
            X, yy = X[ok], y[tr][ok]
            if len(yy) < 200:
                b = None; continue
            mu, sd = X.mean(0), X.std(0) + 1e-9
            b = L.cv((X - mu) / sd, yy, "cont")
        xt = np.concatenate([own[t - delta]] + ([P[t]] if use_peers else []))
        out[j] = 0.0 if (b is None or not np.isfinite(xt).all()) \
            else L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
    return out


def skills(yb, f, bench):
    """R-squared on the log target, and QLIKE on the variance scale.

    QLIKE depends only on the ratio of outcome to forecast, so the trailing
    median in the decision variable cancels and no smearing correction is
    needed - the point lab10 had to establish.
    """
    sst = ((yb - bench) ** 2).sum()
    r2 = 1 - ((yb - f) ** 2).sum() / sst
    yv, fv = np.exp(yb), np.exp(f)
    z = yv / fv
    return r2, float(np.mean(z - np.log(z) - 1))


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    own, P, a, n, k = panel(folder)
    rng = np.random.default_rng(SEED)

    # ---- 0. does the h = 5 column reproduce the paper? --------------------
    print("\n" + "=" * 84)
    print("0.  RECONCILIATION AT h = 5")
    print("=" * 84)
    print("walk() below is lab05's fitrun() with the horizon lifted out of a module")
    print("constant and into an argument.  That rewrite is where a silent difference")
    print("would hide, so lab05's own function is called at h = 5 and compared -")
    print("nothing pasted.  If these disagree, none of the other horizons mean anything.\n")
    _, _, _, idx5, yb5, fitrun, _ = L.run_target(
        folder, TARGET, DELAYS, np.random.default_rng(SEED), continuous=True)
    y5 = np.full(n, np.nan); y5[:-5] = a[5:]
    start5 = L.MED + L.TRAIN + L.VAL + max(DELAYS) + 5
    i5 = np.arange(start5, n - 5)
    i5 = i5[np.isfinite(y5[i5]) & np.isfinite(a[i5])]
    print(f"{'delta':>6}{'own here':>11}{'own lab05':>12}"
          f"{'cross here':>13}{'cross lab05':>13}")
    agree = len(i5) == len(idx5)
    for d in (0, 21, 55):
        _b5 = L.bench_mean(y5, i5)
        ro, _ = skills(y5[i5], walk(own, P, y5, i5, d, 5, False), _b5)
        rc, _ = skills(y5[i5], walk(own, P, y5, i5, d, 5, True), _b5)
        ro5, _ = skills(yb5, fitrun(d, False), _b5)
        rc5, _ = skills(yb5, fitrun(d, True), _b5)
        ok = abs(ro - ro5) < 5e-9 and abs(rc - rc5) < 5e-9
        agree &= ok
        print(f"{d:>6}{ro:>11.4f}{ro5:>12.4f}{rc:>13.4f}{rc5:>13.4f}"
              f"{'   ok' if ok else '   MISMATCH'}")
    print(f"\n  {'the rewrite matches lab05 exactly' if agree else 'RECONCILIATION FAILED'}")

    # ---- A. how much of the target can the model already see? -------------
    print("\n" + "=" * 84)
    print("A.  DOES THE TARGET OVERLAP THE FEATURES, AND BY HOW MUCH?")
    print("=" * 84)
    print("The Yang-Zhang proxy spans five days, so for h < 5 the outcome window and")
    print("the domestic feature's own window share days.  Nothing is leaked - the")
    print("outcome is still in the future - but a forecaster at h = 1 is predicting an")
    print("object they can largely already see, which inflates every arm at once.\n")
    print(f"{'h':>4}{'days shared with sigma^2_t':>28}{'corr(a_t, a_t+h)':>20}")
    fin = np.isfinite(a)
    for h in HORIZONS:
        v = a[fin]
        c = np.corrcoef(v[:-h], v[h:])[0, 1]
        print(f"{h:>4}{max(0, 5 - h):>28}{c:>20.4f}")

    # ---- B. the experiment, at each horizon --------------------------------
    print("\n" + "=" * 84)
    print("B.  THE SUBSTITUTION RATE AT EACH HORIZON")
    print("=" * 84)
    print("Identical features, identical estimator, identical delay grid.  Only the")
    print("horizon changes, and with it the admissibility cut and the benchmark's own")
    print("cut at t - h.  The bootstrap block is lab05's measured one at every")
    print("horizon, so the intervals below are comparable across the table.\n")
    summary = {}
    for h in HORIZONS:
        y = np.full(n, np.nan); y[:-h] = a[h:]
        start = L.MED + L.TRAIN + L.VAL + max(DELAYS) + h
        idx = np.arange(start, n - h)
        idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])]
        yb = y[idx]
        # The benchmark is cut at t - h, not at the module's HORIZON: at this
        # horizon that is where a forecaster's labels stop resolving.  It is
        # built once per horizon and used by both the point estimates and the
        # bootstrap below, so an interval and the estimate it surrounds are
        # always measured against the same constant.
        BENCH = L.bench_mean(y, idx, horizon=h)
        # The block was 2 * h, scaled with the horizon on the reasoning that the
        # overlap in the target was the only dependence to respect.  lab58 shows
        # it is not: the loss differences inherit the common factor's
        # persistence, which does not shrink when h does, so scaling the block
        # with h leaves the shortest horizons with the narrowest blocks and the
        # most dependence left inside the resample.  lab05's measured block is
        # used at every horizon instead, which also makes the intervals across
        # horizons comparable - the point of this table.
        blk = L.BLOCK
        m = len(idx)
        st = rng.integers(0, m - blk + 1, size=(N_BOOT, int(np.ceil(m / blk))))
        offs = np.arange(blk)
        S = [(st[i][:, None] + offs).ravel()[:m] for i in range(N_BOOT)]

        print(f"  --- h = {h} ({m} test days, bootstrap block {blk}) " + "-" * (34 - len(str(h))))
        print(f"{'delta':>6}{'own R2':>10}{'cross R2':>11}{'dR2':>10}"
              f"{'R(d)':>8}{'  95% CI':>18}{'R(d) QLIKE':>13}")
        err = {}
        for d in DELAYS:
            fo, fc = walk(own, P, y, idx, d, h, False), walk(own, P, y, idx, d, h, True)
            ro, qo = skills(yb, fo, BENCH)
            rc, qc = skills(yb, fc, BENCH)
            yv = np.exp(yb)
            ql = lambda f: (lambda z: z - np.log(z) - 1)(yv / np.exp(f))
            err[d] = {"own_r": (yb - fo) ** 2, "cross_r": (yb - fc) ** 2,
                      "own_q": ql(fo), "cross_q": ql(fc)}
            line = f"{d:>6}{ro:>10.4f}{rc:>11.4f}{rc-ro:>+10.4f}"
            if d == 0:
                print(line + f"{'-':>8}{'-':>18}{'-':>13}")
                continue

            def rate(kind, s):
                if kind == "r":
                    base = (yb[s] - BENCH[s]) ** 2
                    sk = lambda e: 1 - e[s].sum() / base.sum()
                else:
                    sk = lambda e: -e[s].mean()
                o0 = sk(err[0][f"own_{kind}"]); od = sk(err[d][f"own_{kind}"])
                cd = sk(err[d][f"cross_{kind}"])
                den = o0 - od
                return (cd - od) / den if abs(den) > 1e-12 else np.nan

            full = np.arange(m)
            pr, pq = rate("r", full), rate("q", full)
            vals = [v for v in (rate("r", s) for s in S) if np.isfinite(v)]
            lo, hi = np.percentile(vals, [2.5, 97.5])
            print(line + f"{pr:>8.0%}" + f"[{lo:>4.0%},{hi:>4.0%}]".rjust(18)
                  + f"{pq:>13.0%}")
            if d == DELAYS[-1]:
                summary[h] = (pr, lo, hi, pq)
        print()

    # ---- C. verdict --------------------------------------------------------
    print("=" * 84)
    print("C.  IS THE HEADLINE A PROPERTY OF THE INFORMATION OR OF THE SETTING?")
    print("=" * 84)
    print(f"{'h':>4}{'R(55) under R2':>18}{'  95% CI':>18}{'R(55) under QLIKE':>20}")
    for h in HORIZONS:
        pr, lo, hi, pq = summary[h]
        print(f"{h:>4}{pr:>18.0%}" + f"[{lo:>4.0%},{hi:>4.0%}]".rjust(18) + f"{pq:>20.0%}")
    r5 = summary[5]
    out = [h for h in HORIZONS if not (r5[1] <= summary[h][0] <= r5[2])]
    print(f"\n  horizons whose point estimate falls outside the h = 5 interval "
          f"[{r5[1]:.0%}, {r5[2]:.0%}]: {out if out else 'none'}")
    print("""
  Read h = 1 against part A before reading it against the others.  Four of its
  five target days are already inside the domestic feature, so both arms score
  high and the denominator of the rate - how much a delay costs a domestic-only
  forecaster - is small.  A rate computed against a small denominator is the
  unstable object this project has flagged twice before, and it is the reason
  h = 1 is reported here rather than folded into the headline.""")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
