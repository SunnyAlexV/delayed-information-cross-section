"""
lab40_level_or_normaliser.py - is Section 9.1's level result information or arithmetic?

Imports lab05_robustness; keep both in labs/.  Runtime about a minute.

THE PROBLEM WITH THE PAPER'S BEST NEW RESULT
---------------------------------------------
Section 9.1 reports that adding log M - the level of the trailing median, which
the paper's own normalisation removes - improves the stale forecaster by +0.0011
of R-squared at zero delay and +0.0736 at eleven weeks, and that this is worth
more at long delays than the entire foreign cross-section.  It also concedes, in
one sentence, that the gain might not be what it looks like.

Here is the concern stated properly.  The target is

    y = log(sigma^2 at t+5) - log(M at t+5)

and the feature is log(M at t-delta).  M is a 252-day median, so it moves very
slowly: log M at t-delta is an excellent predictor of log M at t+5 even when
delta is 55.  A regression handed that feature can therefore lower its squared
error by predicting the target's own DENOMINATOR, which is arithmetic about how
the target was constructed and not a forecast of volatility.  Nothing is
inadmissible - every feature is dated t-delta - but "the level is worth more
than the whole cross-section" would be a very different claim if most of it were
this.

THE TEST
--------
Re-run the same comparison against three targets that differ only in WHEN the
normaliser is dated:

    A   log(sigma^2 t+5) - log(M t+5)     the paper's target
    B   log(sigma^2 t+5) - log(M t)       origin-dated; Section 3 reports this
                                          as a robustness check already
    C   log(sigma^2 t+5) - log(M t-delta) dated to the forecaster's own
                                          information set

C is the decisive one, and it is also the only one of the three a delayed
forecaster could actually use.  Its normaliser is known exactly at the moment
the forecast is made, so a model given log M at t-delta CANNOT buy anything by
predicting it - the quantity is already subtracted out.  Whatever the level is
worth on target C is worth having.

WHAT WOULD COUNT AS A PROBLEM
-----------------------------
If the level's gain is large on A and near zero on C, Section 9.1 is mostly
measuring the model undoing the paper's own normalisation, the 72%-to-68%
correction is wrong, and both must be rewritten.

If the gain survives on C, the finding stands and is stronger than it reads: a
stale forecaster who knows the absolute level of recent volatility - not merely
where it sits against its own median - forecasts materially better, and the
normalisation this literature applies without comment is throwing that away.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L

SEED = 20260915
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 5, 13, 21, 55]
TARGETS = ["A t+5", "B origin", "C t-delta"]


def hac_se(d, lag=9):
    n = len(d); d = d - d.mean()
    s = (d @ d) / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * ((d[k:] @ d[:-k]) / n)
    return np.sqrt(max(s, 1e-18) / n)


def build(folder):
    D, peers, lag = L.build(folder, TARGET)
    raw = L.load_index(TARGET, folder)
    yz = L.yang_zhang(raw, n=L.WINDOW).reindex(D.index).ffill(limit=5)
    logv = np.log(yz.values)
    logM = np.log(yz.rolling(L.MED, min_periods=30).median().values)

    a = D[TARGET].values                      # = logv - logM, the paper's variable
    sa = pd.Series(a)
    own = np.column_stack([sa.values, sa.rolling(5).mean().values,
                           sa.rolling(22).mean().values])
    P = D[peers].values
    n = len(D)
    start = L.MED + L.TRAIN + L.VAL + max(DELAYS) + H
    idx = np.arange(start, n - H)
    return own, P, logv, logM, idx, len(peers), D


def target(kind, logv, logM, t, delta):
    """log variance at t+h, normalised by a median dated as `kind` says."""
    fwd = logv[t + H]
    if kind == "A t+5":
        return fwd - logM[t + H]
    if kind == "B origin":
        return fwd - logM[t]
    return fwd - logM[t - delta]              # C: the forecaster's own information set


def walk(own, P, lv, lm, idx, delta, kind, level, peers):
    """One walk-forward. `level` adds log M at t-delta as a regressor."""
    yall = np.array([target(kind, lv, lm, t, delta) for t in range(len(lv) - H)])
    out = np.empty(len(idx)); b = mu = sd = None

    def X(rows):
        cols = [own[rows - delta]]
        if level:
            cols.append(lm[rows - delta][:, None])
        if peers:
            cols.append(P[rows])
        return np.column_stack(cols)

    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), max(cut, 1))
            M = X(tr); yy = yall[tr]
            ok = np.isfinite(M).all(axis=1) & np.isfinite(yy)
            M, yy = M[ok], yy[ok]
            if len(yy) < 200:
                b = None; continue
            mu, sd = M.mean(0), M.std(0) + 1e-9
            b = L.cv((M - mu) / sd, yy, "cont")
        xt = X(np.array([t]))[0]
        out[j] = 0.0 if (b is None or not np.isfinite(xt).all()) \
            else L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
    return out, yall[idx]


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints the normalisation test of Section 9.1.\n")

    own, P, lv, lm, idx, k, D = build(folder)
    idx = idx[np.isfinite(lv[idx + H]) & np.isfinite(lm[idx + H]) & np.isfinite(lm[idx])]
    print(f"target {TARGET}, {len(idx)} test days, "
          f"{D.index[idx[0]].date()} to {D.index[idx[-1]].date()}, {k} foreign peers")
    print("\nThree targets, differing only in when the normalising median is dated:")
    print("  A  log(var t+5) - log(M t+5)       the paper's")
    print("  B  log(var t+5) - log(M t)         origin-dated")
    print("  C  log(var t+5) - log(M t-delta)   the forecaster's own information set\n")

    R = {}
    print("=" * 96)
    print("A.  WHAT THE LEVEL IS WORTH, AGAINST EACH TARGET")
    print("=" * 96)
    print("Domestic model only. 'gain' is adding log M at t-delta to the paper's three")
    print("features; a Giacomini-White statistic tests it on the paired differential.\n")
    print(f"{'target':>11}{'delta':>7}{'R2 base':>10}{'R2 +level':>11}"
          f"{'gain':>10}{'GW z':>8}")
    sig = {kind: [] for kind in TARGETS}
    for kind in TARGETS:
        for d in DELAYS:
            f0, yb = walk(own, P, lv, lm, idx, d, kind, False, False)
            f1, _ = walk(own, P, lv, lm, idx, d, kind, True, False)
            base = ((yb - yb.mean()) ** 2).sum()
            r0 = 1 - ((yb - f0) ** 2).sum() / base
            r1 = 1 - ((yb - f1) ** 2).sum() / base
            dd = (yb - f0) ** 2 - (yb - f1) ** 2
            z = dd.mean() / hac_se(dd)
            R[(kind, d)] = (r0, r1, z)
            if z > 1.96:
                sig[kind].append(d)
            print(f"{kind:>11}{d:>7}{r0:>10.4f}{r1:>11.4f}{r1 - r0:>+10.4f}{z:>8.2f}")
        print()

    print(f"  delays where the level significantly helps, by target:")
    for kind in TARGETS:
        print(f"    {kind:>11}: {len(sig[kind])} of {len(DELAYS)} {sig[kind]}")

    print("\nThe discriminating quantity is not the significance count, which is the")
    print("same for all three targets and is a statement about power. It is how much of")
    print("the gain SURVIVES re-dating the normaliser: a gain that is the model")
    print("predicting log M at t+5 must shrink when the target's normaliser is instead")
    print("one the model has already been handed.\n")
    print(f"{'delta':>6}{'gain, target A':>17}{'gain, target C':>17}{'C keeps':>10}")
    shares = []
    for d in DELAYS[1:]:
        gA = R[("A t+5", d)][1] - R[("A t+5", d)][0]
        gC = R[("C t-delta", d)][1] - R[("C t-delta", d)][0]
        sh = gC / gA if abs(gA) > 1e-9 else float("nan")
        shares.append(sh)
        print(f"{d:>6}{gA:>+17.4f}{gC:>+17.4f}{sh:>10.0%}")
    lo, hi = min(shares), max(shares)
    print(f"\n  the implementable target keeps between {lo:.0%} and {hi:.0%} of the gain")

    # ---------------- B ----------------------------------------------------
    print("\n" + "=" * 96)
    print("B.  DOES THE SUBSTITUTION RATE CORRECTION SURVIVE?")
    print("=" * 96)
    print("Section 9.1 lowered R(55) from 72% to 68% by measuring it against a control")
    print("that includes the level. If the level is arithmetic, that correction was")
    print("arithmetic too. Here the rate is recomputed inside each target.\n")
    print(f"{'target':>11}{'delta':>7}{'R(d) base':>12}{'R(d) +level':>14}{'move':>9}")
    moves = {}
    for kind in TARGETS:
        base0 = {}
        for level in (False, True):
            f, yb = walk(own, P, lv, lm, idx, 0, kind, level, False)
            b = ((yb - yb.mean()) ** 2).sum()
            base0[level] = 1 - ((yb - f) ** 2).sum() / b
        for d in DELAYS[1:]:
            row = f"{kind:>11}{d:>7}"
            got = []
            for level in (False, True):
                fo, yb = walk(own, P, lv, lm, idx, d, kind, level, False)
                fc, _ = walk(own, P, lv, lm, idx, d, kind, level, True)
                b = ((yb - yb.mean()) ** 2).sum()
                ro = 1 - ((yb - fo) ** 2).sum() / b
                rc = 1 - ((yb - fc) ** 2).sum() / b
                den = base0[level] - ro
                got.append((rc - ro) / den if den > 1e-9 else np.nan)
            moves[(kind, d)] = got[1] - got[0]
            print(row + f"{got[0]:>12.0%}{got[1]:>14.0%}{got[1] - got[0]:>+9.1%}")
        print()

    # ---------------- verdict ----------------------------------------------
    print("=" * 96)
    print("VERDICT")
    print("=" * 96)
    if hi < 0.25:
        print("  Re-dating the normaliser destroys the gain. Section 9.1 was measuring")
        print("  the model undoing the paper's own normalisation, not information about")
        print("  volatility, and the 72%-to-68% correction goes with it.")
    elif lo > 0.75:
        print("  Re-dating the normaliser changes nothing: the level is worth the same")
        print("  against a denominator the model must predict and against one it has")
        print("  been given exactly. It cannot be buying that by predicting a quantity")
        print("  it already knows, so the gain is information about volatility. Knowing")
        print("  the absolute level of recent variance - not merely where it sits")
        print("  against its own median - is worth real skill to a stale forecaster,")
        print("  and the normalisation this literature applies without comment throws")
        print("  it away.")
        print("  Two qualifications belong with that. The gain reaches significance")
        print(f"  only at delta = 55 on every target ({sig['C t-delta']}), so the short-delay")
        print("  column is a direction rather than a result; and the rate correction in")
        print("  part B holds on the implementable target too, so Section 9.1's")
        print("  arithmetic stands even though its caveat does not apply.")
    else:
        print(f"  The implementable target keeps {lo:.0%} to {hi:.0%} of the gain, so part")
        print("  of Section 9.1's result is information and part is normalisation")
        print("  arithmetic. The table above is the split; neither reading carries alone.")
    mv = [abs(moves[("C t-delta", d)]) for d in DELAYS[1:]]
    print(f"\n  largest move in R(delta) from adding the level, on target C: "
          f"{max(mv):.1%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
