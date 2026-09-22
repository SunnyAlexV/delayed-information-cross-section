"""
lab13_origin_median.py - does the trailing median leak the future into the target?

Imports lab05_robustness; keep both in labs/.  Runtime about three minutes.

THE OBJECTION
-------------
The decision variable is a_t = log(sigma^2_YZ,t / M_t), where M_t is the
trailing 252-day median AT t.  We forecast a_{t+5}.  So the target carries
M_{t+5}, a median whose window includes the five days between the forecast
origin and the outcome.  A forecaster standing at t does not know M_{t+5}, so
the quantity being predicted is not one they could convert back to a variance
forecast at the moment they make it.

This is not leakage into the FEATURES - every feature is dated t - delta or
earlier for the domestic block and t for the others, and nothing here changes
that.  It is a question about the target's own definition, and it has two
possible answers.  Either the normalisation is a device for defining a
comparable state variable, in which case the exercise is state prediction and
the point stands but should be said out loud; or the target should be
normalised at the FORECAST ORIGIN, which a practitioner can actually do.

The second is strictly better if it costs nothing, so measure what it costs.

  TARGET-DATED   y = log(sigma^2_{t+h} / M_{t+h})   the paper's definition
  ORIGIN-DATED   y = log(sigma^2_{t+h} / M_t)       implementable at t

The two differ by log(M_{t+h} / M_t), the drift in a 252-day median over five
days, which should be small - but "should be small" is the kind of claim this
project has learned to check rather than assert.  If the substitution rate is
unchanged, the paper reports the origin-dated version as a robustness run and
the objection is closed.  If it moves, the origin-dated version becomes the
headline, because it is the one a forecaster could act on.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".")
import lab05_robustness as L

SEED = 20260912
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 3, 5, 13, 21, 55]
N_BOOT, BLK = 2000, L.BLOCK


def panel(folder):
    """Rebuild the panel and keep the log median, which lab05 discards."""
    tags = [t for t in L.CLOSE_UTC if L.files_for(t, folder)]
    proxy = {t: L.yang_zhang(L.load_index(t, folder)) for t in tags}
    P = L.pd.DataFrame(proxy).sort_index()
    med = {t: P[t].rolling(L.MED, min_periods=30).median() for t in tags}
    D = L.pd.DataFrame({t: np.log(P[t] / med[t]) for t in tags})
    logM = np.log(med[TARGET])
    peers = [t for t in tags if t != TARGET]
    D[peers] = D[peers].ffill(limit=5)
    keep = D[TARGET].notna()
    D, logM = D.loc[keep], logM.loc[keep]
    lag = {p: (0 if L.CLOSE_UTC[p] < L.CLOSE_UTC[TARGET] else 1) for p in peers}
    for p in peers:
        if lag[p]:
            D[p] = D[p].shift(1)
    a = D[TARGET].values
    sa = L.pd.Series(a)
    own = np.column_stack([sa.values, sa.rolling(5).mean().values,
                           sa.rolling(22).mean().values])
    return own, D[peers].values, a, logM.values, len(D)


def targets(a, logM, n):
    """Both target definitions on the same index."""
    y_tgt = np.full(n, np.nan); y_tgt[:-H] = a[H:]
    y_org = np.full(n, np.nan)
    y_org[:-H] = a[H:] + (logM[H:] - logM[:-H])      # undo M_{t+h}, apply M_t
    return y_tgt, y_org


def walk(own, P, y, idx, delta, use_peers):
    out = np.empty(len(idx)); b = mu = sd = None
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
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


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    own, P, a, logM, n = panel(folder)
    y_tgt, y_org = targets(a, logM, n)
    start = L.MED + L.TRAIN + L.VAL + max(DELAYS) + H
    idx = np.arange(start, n - H)
    idx = idx[np.isfinite(y_tgt[idx]) & np.isfinite(y_org[idx]) & np.isfinite(a[idx])]
    print(f"{len(idx)} test days\n")

    drift = (logM[H:] - logM[:-H])[idx]
    print("=" * 82)
    print("A.  HOW BIG IS THE THING IN DISPUTE?")
    print("=" * 82)
    print("log(M_{t+h} / M_t) is the entire difference between the two targets.\n")
    print(f"  mean {drift.mean():+.5f}   sd {drift.std():.5f}   "
          f"|max| {np.abs(drift).max():.4f}")
    print(f"  sd of the target itself                {np.nanstd(y_tgt[idx]):.4f}")
    print(f"  ratio                                  {drift.std()/np.nanstd(y_tgt[idx]):.1%}")
    print(f"  correlation with the target            "
          f"{np.corrcoef(drift, y_tgt[idx])[0,1]:+.4f}")
    print("\nA median over 252 days moves slowly, so the two targets are close. Close")
    print("is not the same as irrelevant, and the correlation above is the reason to")
    print("check: the drift is not independent of what is being predicted.")

    print("\n" + "=" * 82)
    print("B.  DOES THE RESULT DEPEND ON WHICH TARGET IS USED?")
    print("=" * 82)
    print("Identical features, identical estimator, identical days. Only the")
    print("normalisation date of the outcome changes.\n")
    print(f"{'delta':>6}{'--- target-dated (paper) ---':>32}{'--- origin-dated ---':>30}")
    print(f"{'':>6}{'own':>10}{'cross':>10}{'dR2':>12}{'own':>10}{'cross':>10}{'dR2':>10}")
    res, err = {}, {}
    for d in DELAYS:
        row = f"{d:>6}"
        res[d] = {}
        for nm, y in (("tgt", y_tgt), ("org", y_org)):
            yb = y[idx]
            sst = ((yb - L.bench_mean(y, idx)) ** 2).sum()
            po, pc = walk(own, P, y, idx, d, False), walk(own, P, y, idx, d, True)
            ro = 1 - ((yb - po) ** 2).sum() / sst
            rc = 1 - ((yb - pc) ** 2).sum() / sst
            res[d][nm] = (ro, rc, (yb - po) ** 2 - (yb - pc) ** 2, sst)
            err.setdefault(d, {})[nm] = {"own": (yb - po) ** 2, "cross": (yb - pc) ** 2}
            row += f"{ro:>10.4f}{rc:>10.4f}{rc-ro:>+12.4f}" if nm == "tgt" \
                else f"{ro:>10.4f}{rc:>10.4f}{rc-ro:>+10.4f}"
        print(row)

    print("\n" + "=" * 82)
    print("C.  THE SUBSTITUTION RATE UNDER BOTH, WITH INTERVALS")
    print("=" * 82)
    print("Full-ratio bootstrap, as in lab10: numerator and denominator recomputed")
    print("on the same resampled days.\n")
    rng = np.random.default_rng(SEED + 17)
    m = len(idx)
    st = rng.integers(0, m - BLK + 1, size=(N_BOOT, int(np.ceil(m / BLK))))
    offs = np.arange(BLK)
    S = [(st[i][:, None] + offs).ravel()[:m] for i in range(N_BOOT)]

    print(f"{'delta':>6}{'R(d) target-dated':>24}{'R(d) origin-dated':>24}")
    for d in DELAYS[1:]:
        line = f"{d:>6}"
        for nm in ("tgt", "org"):
            _ys = y_tgt if nm == "tgt" else y_org
            yb = _ys[idx]
            _bn = L.bench_mean(_ys, idx)
            e_own0 = err[0][nm]["own"]
            e_ownd = err[d][nm]["own"]
            e_crod = err[d][nm]["cross"]

            def sk(e, s):
                base = (yb[s] - _bn[s]) ** 2
                return 1 - e[s].sum() / base.sum()

            pt = ((sk(e_crod, np.arange(m)) - sk(e_ownd, np.arange(m)))
                  / (sk(e_own0, np.arange(m)) - sk(e_ownd, np.arange(m))))
            vals = []
            for s in S:
                den = sk(e_own0, s) - sk(e_ownd, s)
                if abs(den) > 1e-9:
                    vals.append((sk(e_crod, s) - sk(e_ownd, s)) / den)
            lo, hi = np.percentile(vals, [2.5, 97.5])
            line += f"{pt:>8.0%} [{lo:>4.0%},{hi:>4.0%}]".rjust(24)
        print(line)

    print("\n" + "=" * 82)
    print("VERDICT")
    print("=" * 82)
    d55 = DELAYS[-1]
    for nm, lbl in (("tgt", "target-dated (paper)"), ("org", "origin-dated")):
        o0, od = res[0][nm], res[d55][nm]
        print(f"  {lbl:>22}: R({d55}) = {(od[1]-od[0])/(o0[0]-od[0]):.0%}, "
              f"dR2 = {od[1]-od[0]:+.4f}")
    print()
    print("If those two lines agree, the median's five-day drift is not doing any")
    print("work and the paper may keep its definition, reporting this run as the")
    print("check that says so.  If they do not, the origin-dated target is the one a")
    print("forecaster could act on and should become the headline.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
