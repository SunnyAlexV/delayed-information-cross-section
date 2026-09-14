"""
lab22_factor_benchmark.py - stop inferring the cheaper representation and fit it.

Imports lab05_robustness; keep both in labs/.  Runtime about four minutes.

THE ARGUMENT THIS FILE REPLACES WITH A MEASUREMENT
--------------------------------------------------
Section 5 makes two claims that sit next to each other without ever being joined.
First, that the foreign block is nearly one-dimensional: the first principal
component explains 66.5% of its variance and the first two 76.9%.  Second, that
carrying seven regressors costs about a point of R-squared at zero delay and more
as the delay grows, measured by replacing the block with surrogates that carry no
signal.  The paper then observes that a cheaper representation ought to recover
part of that cost.

Ought to.  A referee points out, correctly, that this is inferred from a
counterfactual generator rather than demonstrated, and that the demonstration is
available: compress the seven series into one or two real-time global volatility
factors and compare that model directly against the seven-coefficient ridge.  If
the factor model wins, the estimation-cost story stops being an argument about
hypothetical surrogates and becomes a constructive result about a model anyone
can fit.  If it loses, the paper has been overselling what a cheaper
parameterisation would buy, and should say so.

REAL-TIME MEANS REAL-TIME
-------------------------
The obvious way to do this is the wrong one.  A principal-component basis
computed on the whole sample and then used to forecast within it leaks: the
loadings are chosen knowing which directions mattered over the test period, and
"the first component of the foreign block" is not something a forecaster in 2009
could have written down using 2026 information.

So the eigenvectors are recomputed at every refit, from the covariance of the
training window ONLY, on standardised columns using training-window statistics
alone - the same discipline lab05 applies to its features.  The sign of each
eigenvector is pinned (first loading positive) so the factor does not flip
arbitrarily between refits, which would otherwise show up as spurious instability
rather than as anything about information.

  PC1     own(3) + one real-time global factor        4 regressors
  PC2     own(3) + two real-time global factors       5
  FULL    own(3) + all seven foreign series           10   - the paper's model
  MEAN    own(3) + the equal-weighted foreign mean    4    - the naive compression

MEAN is included because it is the version that needs no estimation at all.  If
PC1 does not beat an equal-weighted average, the compression is not buying
anything that a practitioner could not have had for free, and the honest summary
is that the dimension reduction matters less than the decision to reduce.

WHAT WOULD COUNT AS A PROBLEM
-----------------------------
If FULL beats the compressions at every delay, Section 5's suggestion that a
cheaper representation would recover part of the estimation cost is wrong and
must be withdrawn.  If a compression wins at the long delays where lab07 says the
cost is largest, the suggestion is confirmed constructively and the paper can
report a model rather than a counterfactual.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L

SEED = 20260913
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 3, 5, 13, 21, 55]
ARMS = ["OWN", "MEAN", "PC1", "PC2", "FULL"]


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


def panel(folder):
    D, peers, lag = L.build(folder, TARGET)
    a = D[TARGET].values
    sa = L.pd.Series(a)
    own = np.column_stack([sa.values, sa.rolling(5).mean().values,
                           sa.rolling(22).mean().values])
    P = D[peers].values
    n = len(D)
    y = np.full(n, np.nan); y[:-H] = a[H:]
    start = L.MED + L.TRAIN + L.VAL + max(DELAYS) + H
    idx = np.arange(start, n - H)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])]
    return own, P, y, idx, len(peers)


def basis(Ptr):
    """Eigenvectors of the TRAINING covariance, with signs pinned.

    Returns (mean, sd, V, explained) where V's columns are the loadings in
    descending eigenvalue order.  Sign pinning matters: an eigenvector is only
    defined up to sign, and an unpinned one flips between refits, which would
    enter the regression as noise that has nothing to do with the data.
    """
    ok = np.isfinite(Ptr).all(axis=1)
    Z = Ptr[ok]
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    Zs = (Z - mu) / sd
    C = np.cov(Zs, rowvar=False)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    for j in range(V.shape[1]):
        if V[0, j] < 0:
            V[:, j] = -V[:, j]
    return mu, sd, V, w / w.sum()


def walk(own, P, y, idx, delta, arm):
    out = np.empty(len(idx))
    b = fmu = fsd = None
    pm = psd = V = None
    share = []
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), cut)
            if arm in ("PC1", "PC2"):
                pm, psd, V, ev = basis(P[tr])
                share.append(ev[0] if arm == "PC1" else ev[0] + ev[1])

            def design(rows_own, rows_P):
                if arm == "OWN":
                    return rows_own
                if arm == "FULL":
                    return np.column_stack([rows_own, rows_P])
                if arm == "MEAN":
                    return np.column_stack([rows_own, rows_P.mean(axis=1)])
                k = 1 if arm == "PC1" else 2
                F = ((rows_P - pm) / psd) @ V[:, :k]
                return np.column_stack([rows_own, F])

            X = design(own[tr - delta], P[tr])
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y[tr])
            X, yy = X[ok], y[tr][ok]
            if len(yy) < 200:
                b = None; continue
            fmu, fsd = X.mean(0), X.std(0) + 1e-9
            b = L.cv((X - fmu) / fsd, yy, "cont")
            _design = design
        if b is None:
            out[j] = 0.0; continue
        xt = _design(own[t - delta][None, :], P[t][None, :])[0]
        out[j] = 0.0 if not np.isfinite(xt).all() \
            else L.p_ridge(b, ((xt - fmu) / fsd)[None, :])[0]
    return out, (float(np.mean(share)) if share else np.nan)


def r2(yb, f):
    return 1 - ((yb - f) ** 2).sum() / ((yb - yb.mean()) ** 2).sum()


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    own, P, y, idx, k = panel(folder)
    yb = y[idx]
    print(f"target {TARGET}, {len(idx)} test days, {k} foreign peers")
    print(f"regressors: OWN 3, MEAN 4, PC1 4, PC2 5, FULL {3+k}\n")

    print("=" * 88)
    print("A.  DOES A CHEAPER REPRESENTATION OF THE FOREIGN BLOCK FORECAST BETTER?")
    print("=" * 88)
    print("Every arm sees the same days, the same domestic block and the same estimator.")
    print("The factors are recomputed at every refit from the training window alone.\n")
    print(f"{'delta':>6}" + "".join(f"{a:>10}" for a in ARMS)
          + f"{'best':>8}{'PC1 - FULL':>13}{'GW z':>8}{'p':>8}")
    F, shares = {}, {}
    for d in DELAYS:
        row = f"{d:>6}"
        for a in ARMS:
            F[(a, d)], sh = walk(own, P, y, idx, d, a)
            if a == "PC1":
                shares[d] = sh
            row += f"{r2(yb, F[(a, d)]):>10.4f}"
        best = max(ARMS, key=lambda a: r2(yb, F[(a, d)]))
        dl = (yb - F[("FULL", d)]) ** 2 - (yb - F[("PC1", d)]) ** 2
        z = dl.mean() / hac_se(dl)
        row += f"{best:>8}{r2(yb,F[('PC1',d)])-r2(yb,F[('FULL',d)]):>+13.4f}{z:>8.2f}{norm_p(z):>8.3f}"
        print(row)

    print(f"\n  mean variance share of the real-time first component, by delay:")
    print("   " + "  ".join(f"d={d}: {shares[d]:.1%}" for d in DELAYS))

    print("\n" + "=" * 88)
    print("B.  AGAINST THE INFERRED COST")
    print("=" * 88)
    print("lab07 estimates what the seven regressors cost by replacing them with")
    print("surrogates carrying no signal.  If that cost is real and a compression avoids")
    print("part of it, PC1 minus FULL should be positive and should grow with delay,")
    print("because the inferred cost does.\n")
    print(f"{'delta':>6}{'FULL':>10}{'PC1':>10}{'PC1 - FULL':>13}"
          f"{'lab07 inferred cost':>22}")
    LAB07 = {0: -0.0131, 3: None, 5: None, 13: None, 21: -0.0536, 55: None}
    for d in DELAYS:
        c = LAB07.get(d)
        print(f"{d:>6}{r2(yb,F[('FULL',d)]):>10.4f}{r2(yb,F[('PC1',d)]):>10.4f}"
              f"{r2(yb,F[('PC1',d)])-r2(yb,F[('FULL',d)]):>+13.4f}"
              f"{('-' if c is None else f'{c:+.4f}'):>22}")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    wins = [d for d in DELAYS if r2(yb, F[("PC1", d)]) > r2(yb, F[("FULL", d)])]
    # dl = squared error of FULL minus squared error of PC1, so POSITIVE means
    # PC1 is the better model.  The first version of this line tested z < -1.96,
    # the tail that would mean the opposite, and reported 0 of 6 while part A was
    # printing z = 2.91.  A verdict that disagrees with its own table is the tell.
    sig = [d for d in DELAYS
           if (lambda dl: dl.mean() / hac_se(dl) > 1.96)(
               (yb - F[("FULL", d)]) ** 2 - (yb - F[("PC1", d)]) ** 2)]
    mean_wins = [d for d in DELAYS if r2(yb, F[("MEAN", d)]) > r2(yb, F[("PC1", d)])]
    print(f"  PC1 beats FULL at {len(wins)} of {len(DELAYS)} delays{': ' + str(wins) if wins else ''}")
    print(f"  and does so at z > 1.96 at {len(sig)}{': ' + str(sig) if sig else ''}")
    print(f"  the equal-weighted MEAN beats PC1 at {len(mean_wins)} of {len(DELAYS)}"
          f"{': ' + str(mean_wins) if mean_wins else ''}")
    print("""
  Read the MEAN column before celebrating any of this.  If an equal-weighted
  average of the seven series does as well as a real-time principal component,
  then what is being bought is the decision to compress rather than the machinery
  of compressing, and the paper should say the cheaper thing rather than the
  cleverer one.  Either way this replaces an inference from surrogates with a
  model a reader can fit, which is what the referee asked for and what Section 5
  should have done in the first place.""")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
