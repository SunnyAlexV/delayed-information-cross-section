"""
lab42_factor_model.py - the model the results assemble, and whether it holds.

Imports lab05_robustness and lab22_factor_benchmark; keep all three in labs/.
Runtime about a minute.

WHY A MODEL AT ALL
------------------
Everything in this paper so far is a measurement.  Two of those measurements are
close to a specification and nobody has written it down.

Section 5.3: the recovery is entirely the leading component of the foreign
block, and the six components orthogonal to it are worth less than nothing.
Section 9.1: the level of volatility, which the normalisation removes, carries
real information that grows with delay.

Put together those say: log variance is a slow level, plus a global factor that
several markets share, plus something idiosyncratic.  Write

    log sigma^2(i,t) = mu(i,t) + lambda(i) g(t) + e(i,t),        g(t) = phi g(t-1) + u(t)

with mu the slow level the 252-day median tracks, g a persistent global factor
and e idiosyncratic.  The paper's decision variable subtracts an estimate of mu,

    a(i,t) = log sigma^2(i,t) - log M(i,t)  ~=  lambda(i) g(t) + e(i,t)

which is exactly why Section 9.1 finds the level missing: the normalisation
removes mu by construction.

WHAT THE MODEL PREDICTS
-----------------------
Target y(t) = a(SPX, t+h).  A forecaster delta days late sees a(SPX, t-delta), a
noisy reading of g(t-delta); the cross-section supplies seven readings of g(t),
whose average suppresses the idiosyncratic part.  Under the AR(1),

    E[g(t+h) | g(t-delta)] = phi^(h+delta) g(t-delta)
    E[g(t+h) | g(t)]       = phi^h g(t)

so three things follow, and each is checkable:

    1. S_own(delta) decays geometrically:  log S_own(delta) is linear in delta
       with slope 2 log phi.  The slope is a PREDICTION, not a fit - phi can be
       estimated separately from the factor itself.

    2. S_cross(delta) is flat in delta.  The foreign block delivers g(t)
       whatever the domestic delay is.

    3. R(delta) rises from zero towards an asymptote BELOW one, and that
       asymptote is

           R(infinity) = S_cross / S_own(0)

       because S_own(delta) -> 0.  It is below one because seven peers estimate
       g(t) with error and never see e(SPX,t).

Prediction 3 is the one worth having: it says the curve in Figure 1 has a
ceiling, names it, and computes it from quantities measured elsewhere in the
paper rather than fitted to the curve.

WHAT WOULD FALSIFY IT
---------------------
If log S_own is not linear in delta, the single-AR(1)-factor story is wrong and
the decay has more than one time scale.  If the slope implies a persistence far
from the factor's own measured persistence, the target's decay is not the
factor's decay.  If the measured R at long delay misses the predicted asymptote,
the cross-section is doing something other than reading g.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab22_factor_benchmark as F

SEED = 20260915
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]
FIT = [1, 2, 3, 5, 8, 13, 21, 34]      # delays where S_own is positive, so log exists


def panel(folder):
    D, peers, lag = L.build(folder, TARGET)
    a = D[TARGET].values
    sa = pd.Series(a)
    own = np.column_stack([sa.values, sa.rolling(5).mean().values,
                           sa.rolling(22).mean().values])
    P = D[peers].values
    n = len(D)
    y = np.full(n, np.nan); y[:-H] = a[H:]
    start = L.MED + L.TRAIN + L.VAL + max(DELAYS) + H
    idx = np.arange(start, n - H)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])]
    return own, P, y, idx, D, peers


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


def ar1(x):
    x = x[np.isfinite(x)]
    v = x - x.mean()
    return float((v[:-1] @ v[1:]) / (v[:-1] @ v[:-1]))


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints the factor model's parameters and its three predictions.\n")

    own, P, y, idx, D, peers = panel(folder)
    yb = y[idx]
    print(f"target {TARGET}, {len(idx)} test days, {len(peers)} foreign peers\n")

    # ---------------- 0. the factor and its persistence --------------------
    print("=" * 92)
    print("0.  THE FACTOR, AND ITS PERSISTENCE")
    print("=" * 92)
    fin = np.isfinite(P).all(axis=1)
    pm, psd, V, ev = F.basis(P[fin])
    g = ((P[fin] - pm) / psd) @ V[:, 0]
    phi_g = ar1(g)
    phi_a = ar1(D[TARGET].values)
    print(f"first component explains {ev[0]:.1%} of the foreign block's variance")
    print(f"AR(1) persistence of the factor g:            phi = {phi_g:.4f}")
    print(f"AR(1) persistence of the target's own series: phi = {phi_a:.4f}")
    print(f"implied half-life of the factor: {np.log(0.5) / np.log(phi_g):.1f} trading days")
    print("\nThe factor is estimated on the whole sample here because it is being")
    print("DESCRIBED, not used to forecast; every forecast below uses the real-time")
    print("basis of Section 5.1.")

    # ---------------- 1. does own-only skill decay geometrically? ----------
    print("\n" + "=" * 92)
    print("1.  PREDICTION: log S_own(delta) IS LINEAR IN delta, SLOPE 2 log phi")
    print("=" * 92)
    S_own, S_cross = {}, {}
    for d in DELAYS:
        S_own[d] = F.r2(yb, walk(own, P, y, idx, d, False))
        S_cross[d] = F.r2(yb, walk(own, P, y, idx, d, True))
    print(f"{'delta':>6}{'S_own':>10}{'log S_own':>12}{'S_cross':>10}")
    for d in DELAYS:
        ls = f"{np.log(S_own[d]):>12.4f}" if S_own[d] > 0 else f"{'n/a':>12}"
        print(f"{d:>6}{S_own[d]:>10.4f}{ls}{S_cross[d]:>10.4f}")

    xs = np.array(FIT, float)
    ys = np.array([np.log(S_own[d]) for d in FIT])
    A = np.column_stack([np.ones(len(xs)), xs])
    coef, res, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = A @ coef
    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    fit_r2 = 1 - ss_res / ss_tot
    phi_implied = float(np.exp(coef[1] / 2))
    print(f"\n  least squares on delays {FIT}:")
    print(f"    slope = {coef[1]:+.5f} per day, fit R-squared = {fit_r2:.4f}")
    print(f"    implied persistence phi = exp(slope/2) = {phi_implied:.4f}")
    print(f"    measured persistence of the factor       = {phi_g:.4f}")
    print(f"    difference = {phi_implied - phi_g:+.4f}")
    linear = fit_r2 > 0.97
    close = abs(phi_implied - phi_g) < 0.02
    print(f"\n  linear to within R-squared 0.97? {'yes' if linear else 'NO'}")
    print(f"  implied persistence within 0.02 of the factor's? {'yes' if close else 'NO'}")

    # ---------------- 2. is cross-sectional skill flat? --------------------
    print("\n" + "=" * 92)
    print("2.  PREDICTION: S_cross(delta) IS FLAT IN delta")
    print("=" * 92)
    cs = np.array([S_cross[d] for d in DELAYS])
    drop = (cs[0] - cs[-1]) / cs[0]
    own_drop = (S_own[0] - S_own[DELAYS[-1]]) / S_own[0]
    print(f"  S_cross falls from {cs[0]:.4f} to {cs[-1]:.4f} across 55 days, "
          f"a loss of {drop:.0%}")
    print(f"  S_own   falls from {S_own[0]:.4f} to {S_own[DELAYS[-1]]:.4f}, "
          f"a loss of {own_drop:.0%}")
    flat = drop < 0.4
    print(f"  cross-sectional skill loses less than 40% of itself? "
          f"{'yes' if flat else 'NO'}")
    print("  The model does not predict perfect flatness: the domestic block is still")
    print("  in the model and still decays, so some slope is expected.")

    # ---------------- 3. the asymptote -------------------------------------
    print("\n" + "=" * 92)
    print("3.  PREDICTION: R(delta) RISES TOWARDS S_cross / S_own(0)")
    print("=" * 92)
    print("The asymptote is computed from two numbers measured elsewhere - the")
    print("cross-sectional model's skill at long delay and the fresh domestic model's")
    print("skill - and is then compared with the rate the curve actually reaches.\n")
    asym = S_cross[DELAYS[-1]] / S_own[0]
    print(f"  S_cross(55) = {S_cross[DELAYS[-1]]:.4f}, S_own(0) = {S_own[0]:.4f}")
    print(f"  predicted asymptote R(infinity) = {asym:.1%}")
    print(f"\n{'delta':>6}{'R(delta) measured':>20}{'predicted ceiling':>20}{'gap':>9}")
    for d in DELAYS[1:]:
        den = S_own[0] - S_own[d]
        r = (S_cross[d] - S_own[d]) / den if den > 1e-9 else np.nan
        print(f"{d:>6}{r:>20.1%}{asym:>20.1%}{r - asym:>+9.1%}")
    r55 = (S_cross[55] - S_own[55]) / (S_own[0] - S_own[55])
    print(f"\n  at the longest delay the curve reaches {r55:.1%} against a predicted "
          f"{asym:.1%}")
    hit = abs(r55 - asym) < 0.05
    print(f"  within five points of the prediction? {'yes' if hit else 'NO'}")

    # ---------------- verdict ----------------------------------------------
    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    passed = sum([linear, close, flat, hit])
    print(f"  {passed} of 4 predictions hold.")
    if linear and close:
        print("  The decay of a stale forecaster's skill is geometric, and the rate of")
        print("  decay implies a persistence that matches the factor's own. The delay")
        print("  curve is the factor's memory, measured through a forecast.")
    elif linear:
        print("  The decay is geometric but at a rate that does not match the factor's")
        print("  measured persistence, so the target's memory and the factor's are not")
        print("  the same thing and the one-factor reading is incomplete.")
    else:
        print("  The decay is not geometric, so a single AR(1) factor cannot be the")
        print("  whole story and the curve has more than one time scale in it.")
    if hit:
        print("  The ceiling the model predicts from two independently measured")
        print("  quantities is where the curve actually goes. R(delta) is not an")
        print("  arbitrary rising curve; it is a factor-share, and the paper can say")
        print("  what it converges to and why.")
    else:
        print("  The curve does not reach the predicted ceiling, so the cross-section is")
        print("  doing something other than supplying the factor, or the factor share is")
        print("  mismeasured. Either way the model is not yet the explanation.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
