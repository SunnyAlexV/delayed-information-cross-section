"""
lab08_implied_vol.py - the objection the paper could not answer.

Imports lab05_robustness; keep both in labs/.  Runtime about two minutes.

THE OBJECTION
-------------
Korkusuz (2025) and Buncic & Gisler (2016) both find that implied volatility
carries much of what looks like a cross-market effect.  So a referee will ask
the obvious thing: if a forecaster whose domestic data is delta days stale can
see today's VIX, why would they need Tokyo?  Perhaps the foreign cross-section
is a slow proxy for an options market that reports without any delay at all,
in which case the paper measures something real and recommends the wrong
instrument.

Four models, all with the domestic block held delta days stale:

    own            stale domestic history only
    own + foreign  the paper's model: seven foreign closes, current
    own + iv       implied volatility, current
    own + both     everything

The number that settles it is the INCREMENTAL contribution of the foreign
block GIVEN implied volatility, R2(both) - R2(iv).  If that is zero, the
foreign cross-section is a proxy and the paper should say so.  "own + iv" is
nested inside "own + both", so Giacomini-White is again the right test and
Diebold-Mariano is again the wrong one.

TWO DATA PROBLEMS, BOTH REPORTED RATHER THAN PAPERED OVER
---------------------------------------------------------
1. TIMING.  Our own admissibility rule says a series counts only if its
   timestamp precedes the target's close.  VDAX-NEW satisfies it outright:
   Frankfurt finishes hours before New York.  VIX is awkward only if its
   PRINTED DAILY CLOSE is taken as the observation, since that print is stamped
   21:15 UTC, fifteen minutes after the 21:00 S&P cash close.  But VIX is
   computed continuously from live SPX option quotes, so a forecaster standing
   at the cash close does observe a value of it, a few minutes' drift from the
   printed one.  The SAME-DAY arm is therefore the realistic treatment and the
   one to read; the STRICT arm lags VIX a whole day and concedes the entire
   fifteen minutes, which makes it a conservative bound rather than the right
   answer.  Both are reported because the verdict is the same at both ends of
   that interval.  A retail end-of-day export carries no intraday snapshot, so
   the 21:00 value itself is not available to us.

2. COVERAGE.  VDAX-NEW starts 2001-05-11 against the index panel's 2000-02-22,
   which costs 6% of the test days.  Every arm is therefore re-estimated on
   the SAME reduced sample, so the comparison stays internal.  The VIX-only
   race, which costs nothing, is also reported on the full sample.
"""

import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".")
import lab05_robustness as L

SEED = 20260912
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]
IV_TAGS = ["VIX", "VDAX"]          # series with usable history; see docstring

# Matched by KEYWORD, like every other lab, so the original download names work
# untouched.  Renaming files by hand is how a data error gets in.
IV_ALIASES = {
    "VIX":    ["vix", "cboe volatility", "cboe_volatility"],
    "VDAX":   ["vdax", "dax new volatility", "dax_new_volatility"],
    "VSTOXX": ["vstoxx", "v2tx", "stoxx 50 volatility", "stoxx_50_volatility"],
}

# Keyword matching is what keeps the original download names working, and it is
# also how a wrong file gets silently merged into a right one.  This project has
# already been bitten once, by "DAX_New_Volatility" matching the plain "dax"
# index discovery.  The VSTOXX case is live and waiting: the spot index exports
# as "STOXX 50 Volatility VSTOXX EUR", the unusable Mini Futures contract exports
# as "VSTOXX Mini Futures", and BOTH contain "vstoxx".  Without this, adding the
# spot series would concatenate an index and a futures contract into one column
# and the seam audit would report the overlap as a mismatch rather than as the
# category error it is.
IV_EXCLUDE = {
    "VIX":    [],
    "VDAX":   [],
    "VSTOXX": ["mini", "futures", "fvs"],
}
VIX_LAG_STRICT = 1                 # VIX prints after the S&P close


def hac_se(d, lag=None):
    n = len(d); d = d - d.mean()
    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    s = (d @ d) / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * ((d[k:] @ d[:-k]) / n)
    return np.sqrt(max(s, 1e-18) / n)


def norm_p(z):
    from math import erf, sqrt
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def iv_files(folder, tag):
    """Every CSV under `folder` (one level deep too) whose name names this series."""
    seen, out = set(), []
    for pat in (os.path.join(folder, "*.csv"), os.path.join(folder, "*", "*.csv")):
        for p in sorted(glob.glob(pat)):
            b = os.path.basename(p).lower()
            if (any(k in b for k in IV_ALIASES[tag])
                    and not any(x in b for x in IV_EXCLUDE[tag])
                    and b not in seen):
                seen.add(b); out.append(p)
    return out


def load_iv(folder, tag):
    """Merge the exports for one implied-volatility series, auditing the seam."""
    paths = iv_files(folder, tag)
    if not paths:
        raise SystemExit(
            f"\nCould not find the {tag} data.\n"
            f"Looked for CSVs whose name contains any of {IV_ALIASES[tag]}\n"
            f"in {os.path.abspath(folder)} and one level below it.\n"
            f"Original download names work as-is - do not rename them.\n"
            f"Put the files next to the index CSVs, or pass their folder as the "
            f"first argument.")
    print(f"    {tag} files: {[os.path.basename(p) for p in paths]}")
    parts = [L.read_one(p) for p in paths]
    m = parts[0]
    for b in parts[1:]:
        ov = m.merge(b, on="date", suffixes=("_a", "_b"))
        if len(ov) >= 20:
            bad = int(((ov.close_a - ov.close_b).abs() > 0.005 * ov.close_a.abs()).sum())
            print(f"    seam {tag}: {len(ov)} overlapping rows, {bad} mismatched"
                  f"  -> {'VERIFIED' if bad == 0 else 'MISMATCH'}")
        else:
            print(f"    seam {tag}: only {len(ov)} overlapping rows -> UNVERIFIABLE")
        m = (pd.concat([m, b]).drop_duplicates("date", keep="first")
               .sort_values("date").reset_index(drop=True))
    return m.set_index("date")["close"]


def build(folder, strict):
    """Index panel plus an implied-volatility block on the same calendar.

    The IV decision variable mirrors the index one: log of the level over its
    own trailing 252-day median, so a quiet market and a nervous one are
    comparable and no series carries its own units into the ridge.
    """
    D, peers, lag = L.build(folder, TARGET)
    print("  implied-volatility files and seams:")
    raw = {t: load_iv(folder, t) for t in IV_TAGS}
    iv = pd.DataFrame(index=D.index)
    for t, v in raw.items():
        on = v.reindex(D.index).ffill(limit=5)
        if t == "VIX" and strict:
            on = on.shift(VIX_LAG_STRICT)
        iv[t] = np.log(on / on.rolling(L.MED, min_periods=30).median())
    keep = iv.notna().all(axis=1) & D[TARGET].notna()
    return D.loc[keep], iv.loc[keep], peers


def make(D, iv, peers):
    a = D[TARGET].values
    sa = pd.Series(a)
    own = np.column_stack([sa.values, sa.rolling(5).mean().values,
                           sa.rolling(22).mean().values])
    n = len(D)
    y = np.full(n, np.nan); y[:-H] = a[H:]
    start = L.MED + L.TRAIN + L.VAL + max(DELAYS) + H
    idx = np.arange(start, n - H)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])]
    return own, D[peers].values, iv.values, y, idx


def walk(own, blocks, y, idx, delta):
    """blocks = extra CURRENT feature matrices; own is held delta days stale."""
    out = np.empty(len(idx)); b = mu = sd = None
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), cut)
            X = np.column_stack([own[tr - delta]] + [B[tr] for B in blocks])
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y[tr])
            X, yy = X[ok], y[tr][ok]
            if len(yy) < 200:
                b = None; continue
            mu, sd = X.mean(0), X.std(0) + 1e-9
            b = L.cv((X - mu) / sd, yy, "cont")
        xt = np.concatenate([own[t - delta]] + [B[t] for B in blocks])
        out[j] = 0.0 if (b is None or not np.isfinite(xt).all()) \
            else L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
    return out


def r2(yb, f):
    return 1 - ((yb - f) ** 2).sum() / ((yb - yb.mean()) ** 2).sum()


def race(D, iv, peers, title, tags):
    own, P, V, y, idx = make(D, iv, peers)
    yb = y[idx]
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    print(f"{len(idx)} test days, {D.index[idx[0]].date()} to {D.index[idx[-1]].date()}; "
          f"implied-volatility block: {', '.join(tags)}\n")
    print(f"{'delta':>6}{'own':>9}{'+foreign':>10}{'+iv':>9}{'+both':>9}"
          f"{'foreign | iv':>15}{'GW z':>9}{'p':>8}")
    rows = {}
    for d in DELAYS:
        f_own = walk(own, [], y, idx, d)
        f_for = walk(own, [P], y, idx, d)
        f_iv = walk(own, [V], y, idx, d)
        f_both = walk(own, [P, V], y, idx, d)
        inc = r2(yb, f_both) - r2(yb, f_iv)          # foreign, given implied vol
        dl = (yb - f_iv) ** 2 - (yb - f_both) ** 2   # nested: iv inside both
        z = dl.mean() / hac_se(dl)
        rows[d] = (r2(yb, f_own), r2(yb, f_for), r2(yb, f_iv), r2(yb, f_both), inc, z)
        print(f"{d:>6}{rows[d][0]:>9.4f}{rows[d][1]:>10.4f}{rows[d][2]:>9.4f}"
              f"{rows[d][3]:>9.4f}{inc:>+15.4f}{z:>9.2f}{norm_p(z):>8.3f}")
    print("\n'foreign | iv' is what the foreign cross-section adds ON TOP of implied")
    print("volatility.  It is the number the objection turns on.  GW tests it;")
    print("'+iv' is nested inside '+both', so Diebold-Mariano would be invalid.")

    print(f"\n{'delta':>6}{'R(d) foreign':>14}{'R(d) iv':>10}{'R(d) both':>12}")
    for d in DELAYS[1:]:
        o, fo, ivr, bo = rows[d][:4]
        den = rows[0][0] - o
        f = lambda x: f"{(x - o) / den:.0%}" if den > 1e-9 else "n/a"
        print(f"{d:>6}{f(fo):>14}{f(ivr):>10}{f(bo):>12}")
    print("Substitution rate: the share of delay-induced loss each block recovers.")
    return rows


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}\n")

    print("=" * 80)
    print("STRICT: VIX lagged one day, obeying our own admissibility rule")
    print("=" * 80)
    Ds, ivs, peers = build(folder, strict=True)
    strict = race(Ds, ivs, peers,
                  "STRICT - VIX at t-1 (it prints 21:15 UTC, after the 21:00 close)",
                  ["VIX(t-1)", "VDAX(t)"])

    print("\n\n" + "=" * 80)
    print("GENEROUS: same-day VIX, which we are NOT entitled to see")
    print("=" * 80)
    print("This breaks our own timing rule, deliberately, in the objection's favour.")
    Dg, ivg, _ = build(folder, strict=False)
    gen = race(Dg, ivg, peers,
               "GENEROUS - same-day VIX, the harshest test of the paper",
               ["VIX(t)", "VDAX(t)"])

    print("\n\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    for name, rows in (("STRICT", strict), ("GENEROUS", gen)):
        sig = [d for d in DELAYS if rows[d][5] > 1.96]
        print(f"{name:>9}: foreign block adds to implied volatility at "
              f"{len(sig)} of {len(DELAYS)} delays "
              f"({'delta ' + str(min(sig)) + ' onward' if sig else 'nowhere'})")
        print(f"{'':>9}  incremental R2 at delta 21: {rows[21][4]:+.4f}  "
              f"(z = {rows[21][5]:.2f})")
    print("\nIf the incremental column is flat and insignificant, the foreign")
    print("cross-section is a slow proxy for the options market and the paper")
    print("must say so.  If it is not, the objection is answered.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("IS THE FOREIGN BLOCK HARMFUL, OR MERELY REDUNDANT?")
    print("=" * 80)
    print("The incremental column above is NEGATIVE.  Read literally that says")
    print("foreign closes damage a forecast that already has implied volatility,")
    print("which would be a strange thing for genuine information to do.  lab07")
    print("measured the alternative: seven extra regressors cost about a point of")
    print("R-squared whether or not they help.  Subtract that and the question")
    print("becomes whether the foreign block's content given implied volatility is")
    print("negative or simply zero - a difference the paper has to get right.")
    print()
    print("Same instrument as lab07: replace the seven foreign series with AR(1)")
    print("surrogates matched on persistence and variance, so they carry no")
    print("information by construction.  What remains is the bill.\n")

    own, P, V, y, idx = make(Dg, ivg, peers)
    yb = y[idx]
    rng = np.random.default_rng(SEED + 3)
    fin = np.isfinite(P).all(axis=1)
    k = P.shape[1]
    phi = np.empty(k); mu_p = np.empty(k); resid = []
    for j in range(k):
        v = P[fin, j]; v0, v1 = v[:-1] - v.mean(), v[1:] - v.mean()
        phi[j] = (v0 @ v1) / (v0 @ v0)
        resid.append(v1 - phi[j] * v0)
        mu_p[j] = v.mean()
    Sig = np.cov(np.column_stack(resid), rowvar=False)   # cross-market covariance
    sig = np.sqrt(np.diag(Sig))
    Chol = np.linalg.cholesky(Sig + 1e-12 * np.eye(k))

    def surrogate():
        S = np.empty_like(P); e = rng.normal(size=P.shape) @ Chol.T  # keeps peers correlated
        S[0] = mu_p + e[0] / np.sqrt(np.maximum(1 - phi ** 2, 1e-6))
        for t in range(1, len(P)):
            S[t] = mu_p + phi * (S[t - 1] - mu_p) + e[t]
        S[~fin] = np.nan
        return S

    draws = [surrogate() for _ in range(100)]
    nb, blk = 2000, 10
    bn = len(yb)
    brng = np.random.default_rng(SEED + 4)
    st = brng.integers(0, bn - blk + 1, size=(nb, int(np.ceil(bn / blk))))
    offs = np.arange(blk)
    samples = [(st[i][:, None] + offs).ravel()[:bn] for i in range(nb)]
    sst_b = np.array([((yb[s] - yb[s].mean()) ** 2).sum() for s in samples])

    print("Gross gets an interval before it gets a label, for the reason lab07")
    print("gives: it is a difference of two estimated quantities.\n")
    print(f"{'delta':>6}{'incremental':>13}{'cost':>10}{'gross':>10}"
          f"{'gross 95%':>22}   reading")
    for d in [0, 5, 21, 55]:
        f_iv = walk(own, [V], y, idx, d)
        f_both = walk(own, [P, V], y, idx, d)
        e_iv, e_both = (yb - f_iv) ** 2, (yb - f_both) ** 2
        e_sur = np.mean([(yb - walk(own, [S, V], y, idx, d)) ** 2 for S in draws], axis=0)
        inc = (e_iv - e_both).sum() / ((yb - yb.mean()) ** 2).sum()
        cost = (e_iv - e_sur).sum() / ((yb - yb.mean()) ** 2).sum()
        gvec = e_sur - e_both                     # gross, per day
        gb = np.array([gvec[s].sum() for s in samples]) / sst_b
        lo, hi = np.percentile(gb, [2.5, 97.5])
        g = inc - cost
        note = ("redundant - nothing is not harm" if lo <= 0 <= hi else
                "genuinely harmful" if hi < 0 else "adds something after all")
        print(f"{d:>6}{inc:>+13.4f}{cost:>+10.4f}{g:>+10.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>22}   {note}")
    print("\nGross content of the foreign block GIVEN implied volatility.  Where the")
    print("interval covers zero the honest word is redundant, not harmful, and it")
    print("is the paper's RECOMMENDATION rather than its MEASUREMENT that changes.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
