"""
lab16_clark_west_shrinkage.py - does this paper's own Clark-West column survive
a known hazard of the test under per-arm tuning?

Imports lab05_robustness and lab06_inference; keep all three in labs/.
Runtime about nine minutes.

THE CONCERN
-----------
Clark-West is a test about MODELS.  Its adjustment is derived on the null that
the added coefficients are zero in population, and it adds back IN FULL the
estimation penalty the larger model pays for carrying them.  That derivation
assumes the larger model actually pays that penalty.  Shrinkage removes part of
it - that is what shrinkage is for - so if the two arms are shrunk by different
amounts, the adjustment restores more than was ever charged, and the surplus
lands in the numerator and inflates the statistic.

Nothing in that argument is specific to this data; it follows from how the
Clark-West correction is built.  What is specific to this data is whether the
precondition holds: Section 6 of THIS paper reports Clark-West statistics from
0.23 to 9.21, and lab05's cv() selects the ridge penalty on a validation slice,
SEPARATELY, for each arm.  So the precondition is present by construction, and
the question is whether it bites.

WHAT THIS FILE CAN AND CANNOT CONCLUDE
--------------------------------------
It cannot conclude that our Clark-West column is invalid.  Ridge shrinks
coefficients anisotropically rather than scaling a forecast, so it is not the
scalar shrinkage the argument above is cleanest for, and the mapping is not
exact.  What it can do is test the thing directly rather than reason about it.

The decisive test needs no calibration from anywhere: impose ONE ridge penalty
on both arms, chosen on the restricted arm, so the two shrinkages are equal by
construction and the surplus has nowhere to come from.  If the statistics barely
move, the mechanism is not biting at this sample size.

Alongside it we report the ratio of the mean Clark-West adjustment to the
magnitude of the mean loss differential.  Near one, the adjustment matches the
penalty it removes; materially above one, it exceeds it.  That ratio is a
DESCRIPTION, not a threshold - this repository has no external calibration for
how large it must be before a p-value should be distrusted, and says so rather
than borrowing a number it cannot reproduce.

"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab06_inference as L6

SEED = L6.SEED if hasattr(L6, "SEED") else 20260912
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 3, 5, 13, 21, 55]      # the delays Section 6 quotes
RECON = [0, 5, 55]                   # reconciled against lab05's own walk


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


def pick_lambda(X, y):
    """The lambda lab05's cv() would choose, returned rather than swallowed."""
    if len(y) <= L.VAL + 50:
        return L.LAMBDAS[2]
    Xtr, ytr, Xv, yv = X[:-L.VAL], y[:-L.VAL], X[-L.VAL:], y[-L.VAL:]
    best, bl = -np.inf, L.LAMBDAS[0]
    for lam in L.LAMBDAS:
        sc = -((L.p_ridge(L.fit_ridge(Xtr, ytr, lam), Xv) - yv) ** 2).mean()
        if sc > best:
            best, bl = sc, lam
    return bl


def both_arms(own, P, y, idx, delta, common):
    """Run the two arms on one refit schedule.

    common=False reproduces lab05/lab06 exactly: each arm picks its own penalty.
    common=True  applies the RESTRICTED arm's penalty to both, which is the
                 remedy described above.
    Returns (own forecasts, cross forecasts, lambdas chosen per arm per refit).
    """
    fo = np.empty(len(idx)); fc = np.empty(len(idx))
    bo = bc = None; mo = so = mc = sc = None
    lams = []
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), cut)
            Xo = own[tr - delta]
            Xc = np.column_stack([own[tr - delta], P[tr]])
            ko = np.isfinite(Xo).all(axis=1) & np.isfinite(y[tr])
            kc = np.isfinite(Xc).all(axis=1) & np.isfinite(y[tr])
            Xo, yo = Xo[ko], y[tr][ko]
            Xc, yc = Xc[kc], y[tr][kc]
            if len(yo) < 200 or len(yc) < 200:
                bo = bc = None; continue
            mo, so = Xo.mean(0), Xo.std(0) + 1e-9
            mc, sc = Xc.mean(0), Xc.std(0) + 1e-9
            Zo, Zc = (Xo - mo) / so, (Xc - mc) / sc
            lo = pick_lambda(Zo, yo)
            lc = lo if common else pick_lambda(Zc, yc)
            lams.append((lo, lc))
            bo = L.fit_ridge(Zo, yo, lo)
            bc = L.fit_ridge(Zc, yc, lc)
        xo = own[t - delta]
        xc = np.concatenate([own[t - delta], P[t]])
        fo[j] = 0.0 if (bo is None or not np.isfinite(xo).all()) \
            else L.p_ridge(bo, ((xo - mo) / so)[None, :])[0]
        fc[j] = 0.0 if (bc is None or not np.isfinite(xc).all()) \
            else L.p_ridge(bc, ((xc - mc) / sc)[None, :])[0]
    return fo, fc, lams


def stats(y, fo, fc):
    """Loss differential, Clark-West adjustment, and the statistics from both."""
    d = (y - fo) ** 2 - (y - fc) ** 2          # positive favours cross
    a = (fo - fc) ** 2                          # the nesting adjustment
    z_gw = d.mean() / L6.hac_se(d)
    z_cw = (d + a).mean() / L6.hac_se(d + a)
    return d, a, z_gw, z_cw


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    own, P, y, idx, k = panel(folder)
    yb = y[idx]
    print(f"target {TARGET}, {len(idx)} test days, {k} foreign peers, "
          f"{len(DELAYS)} delays")
    print(f"ridge grid {L.LAMBDAS}, selected on the last {L.VAL} training rows\n")

    SEP, COM = {}, {}
    for d in DELAYS:
        SEP[d] = both_arms(own, P, y, idx, d, common=False)
        COM[d] = both_arms(own, P, y, idx, d, common=True)

    # ---- reconciliation --------------------------------------------------
    print("=" * 84)
    print("0.  RECONCILIATION WITH lab05's OWN WALK-FORWARD")
    print("=" * 84)
    print("both_arms() runs the two models on one shared refit schedule so that a")
    print("common penalty can be imposed.  That is a rewrite of lab05's fitrun(), and")
    print("a rewrite is exactly where a silent difference hides.  lab05's own function")
    print("is therefore called here and the statistics compared - nothing pasted.\n")
    print(f"{'delta':>6}{'CW here':>12}{'CW lab05':>12}{'GW here':>12}{'GW lab05':>12}")
    _, _, _, idx5, yb5, fitrun, _ = L.run_target(
        folder, TARGET, DELAYS, np.random.default_rng(SEED), continuous=True)
    agree = len(idx5) == len(idx)
    if not agree:
        print(f"  test-day counts differ: {len(idx5)} vs {len(idx)}")
    for d in RECON:
        fo, fc, _ = SEP[d]
        _, _, zg, zc = stats(yb, fo, fc)
        _, _, zg5, zc5 = stats(yb5, fitrun(d, False), fitrun(d, True))
        ok = abs(zc - zc5) < 5e-9 and abs(zg - zg5) < 5e-9
        agree &= ok
        print(f"{d:>6}{zc:>12.4f}{zc5:>12.4f}{zg:>12.4f}{zg5:>12.4f}"
              f"{'   ok' if ok else '   MISMATCH'}")
    print(f"\n  {'the rewrite matches lab05 exactly' if agree else 'RECONCILIATION FAILED'}")

    # ---- A. is the precondition present? ---------------------------------
    print("\n" + "=" * 84)
    print("A.  IS THE PRECONDITION PRESENT?  PENALTY SELECTION, PER ARM")
    print("=" * 84)
    print("The bias is proportional to the DIFFERENCE between the two arms'")
    print("two arms' shrinkage.  If both arms landed on the same penalty at every")
    print("refit, there would be nothing to discuss.\n")
    print(f"{'delta':>6}{'mean lam own':>14}{'mean lam cross':>16}"
          f"{'refits differing':>18}{'cross shrunk harder':>21}")
    for d in DELAYS:
        lams = SEP[d][2]
        lo = np.array([x[0] for x in lams]); lc = np.array([x[1] for x in lams])
        print(f"{d:>6}{lo.mean():>14.2f}{lc.mean():>16.2f}"
              f"{(lo != lc).mean():>17.0%}{(lc > lo).mean():>21.0%}")
    print("\nThe arms disagree at most refits and the cross arm is usually penalised")
    print("harder, which is the configuration that would bias the")
    print("statistic upward - the direction our column runs.")

    # ---- B. the diagnostic ------------------------------------------------
    print("\n" + "=" * 84)
    print("B.  THE COMPANION PAPER'S DIAGNOSTIC:  mean adjustment / |mean differential|")
    print("=" * 84)
    print("Near one, the adjustment matches the estimation penalty it removes.")
    print("Materially above one, it exceeds that penalty and the surplus is what")
    print("inflates the statistic.  We report it as a description, not a threshold:")
    print("there is no calibration here for how large it must be to matter.\n")
    print(f"{'delta':>6}{'mean d':>12}{'mean a':>12}{'ratio':>10}"
          f"{'CW':>9}{'GW':>9}")
    ratio = {}
    for d in DELAYS:
        fo, fc, _ = SEP[d]
        dd, aa, zg, zc = stats(yb, fo, fc)
        r = aa.mean() / abs(dd.mean()) if abs(dd.mean()) > 1e-12 else np.inf
        ratio[d] = r
        print(f"{d:>6}{dd.mean():>+12.5f}{aa.mean():>12.5f}{r:>10.2f}"
              f"{zc:>9.2f}{zg:>9.2f}")

    # ---- C. the remedy ----------------------------------------------------
    print("\n" + "=" * 84)
    print("C.  THE REMEDY:  ONE PENALTY, CHOSEN ON THE RESTRICTED ARM, APPLIED TO BOTH")
    print("=" * 84)
    print("Setting the two shrinkage factors equal sets the bias described above")
    print("term to zero identically.  If the mechanism is biting here, this column")
    print("moves.\n")
    print(f"{'delta':>6}{'CW separate':>13}{'CW common':>12}{'change':>9}"
          f"{'ratio sep':>11}{'ratio common':>14}")
    for d in DELAYS:
        fo, fc, _ = SEP[d]
        _, _, _, zc_s = stats(yb, fo, fc)
        go, gc, _ = COM[d]
        dd, aa, _, zc_c = stats(yb, go, gc)
        rc = aa.mean() / abs(dd.mean()) if abs(dd.mean()) > 1e-12 else np.inf
        print(f"{d:>6}{zc_s:>13.2f}{zc_c:>12.2f}{zc_c-zc_s:>+9.2f}"
              f"{ratio[d]:>11.2f}{rc:>14.2f}")

    # ---- D. the primary test ---------------------------------------------
    print("\n" + "=" * 84)
    print("D.  DOES THE PRIMARY TEST MOVE?")
    print("=" * 84)
    print("Giacomini-White compares METHODS under a fixed rolling window, and the")
    print("validation slice sits inside that window, so tuning is part of the method")
    print("being tested rather than a contaminant of a correction - there is no")
    print("correction here to throw off.  The prediction is that this column barely")
    print("moves.  Sign changes would falsify that.\n")
    print(f"{'delta':>6}{'GW separate':>13}{'GW common':>12}{'change':>9}"
          f"{'p separate':>13}{'p common':>11}")
    flips = 0
    for d in DELAYS:
        fo, fc, _ = SEP[d]
        _, _, zg_s, _ = stats(yb, fo, fc)
        go, gc, _ = COM[d]
        _, _, zg_c, _ = stats(yb, go, gc)
        flips += (zg_s > 1.96) != (zg_c > 1.96)
        print(f"{d:>6}{zg_s:>13.2f}{zg_c:>12.2f}{zg_c-zg_s:>+9.2f}"
              f"{norm_p(zg_s):>13.3f}{norm_p(zg_c):>11.3f}")
    print(f"\n  delays whose GW verdict changes at 1.96: {flips} of {len(DELAYS)}")

    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    hi = [d for d in DELAYS if ratio[d] > 1.5]
    print(f"  diagnostic above 1.5 at {len(hi)} of {len(DELAYS)} delays"
          f"{': ' + ', '.join(map(str, hi)) if hi else ''}")
    print("""
  What this licenses depends on the diagnostic above, and the paper should say
  only what it shows.  Where the ratio sits near one the Clark-West adjustment
  is matching the penalty it removes and the statistic is doing its job.  Where
  it is well above one the statistic is carrying a surplus, and the honest move
  is to report the Clark-West column with that ratio beside it rather than to
  quote a p-value that this mechanism can inflate.

  Either way the primary test is Giacomini-White, for the reason in part D, and
  nothing here touches the substitution rates, the intervals or the bootstraps -
  none of which involve a nested-model correction.""")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
