"""
lab31_mark_to_model.py - EXPLORATORY.  Cited by neither paper.

The benchmark this project has never had to beat.

WHAT PRACTITIONERS ACTUALLY DO
-------------------------------
Faced with a stale mark, nobody puts foreign index closes into a forecasting
regression.  They estimate the asset's exposure to observable factors on
whatever history they have, then roll the stale mark forward using the factors'
CURRENT returns:

    a_hat(t)  =  a(t - delta)  +  beta' * [ f(t) - f(t - delta) ]

That is mark-to-model, and it is the standing method for private equity,
property and anything else valued between trades.  The paper's approach is a
different thing: it hands the foreign closes to the forecaster as extra
regressors and lets the fit decide what to do with them.

Nothing in either paper compares the two.  If the practitioner method wins, the
paper's contribution is smaller than it claims and the honest framing changes.
That is the point of running it.

THE FOUR ARMS
-------------
  STALE      the domestic model on data delta days old.  The thing to beat.
  PAPER      the domestic model plus the seven foreign closes as regressors.
  MARK       reconstruct a_hat(t) by the formula above, then run the ORDINARY
             domestic model on the reconstructed state as if it were observed.
  MARK+      the reconstruction, plus the domestic history, in one model.

The distinction that matters is between PAPER and MARK.  Both see exactly the
same data.  PAPER lets a regression choose how to use it; MARK imposes the
structure a practitioner would impose - update the level, then forecast as usual
- and buys a much smaller parameter count for that restriction.  Section 5 of the
paper shows parameter count is expensive here, so the restriction may pay.

HOW THE EXPOSURE IS ESTIMATED
------------------------------
beta comes from regressing changes in the domestic state on contemporaneous
changes in the foreign block, over the training window only, refitted on the
same schedule as everything else.  The factor used is the real-time global
component of lab28, because lab28 showed the remainder carries nothing - so
beta is one number, not seven, and the reconstruction costs almost nothing to
estimate.  A seven-beta version is run beside it to show what the extra
parameters cost.

WHAT WOULD OVERTURN THE PAPER
------------------------------
If MARK beats PAPER at the delays the paper reports, then the paper's
cross-sectional regression is a worse way of using the same information than the
method the industry already uses, and the contribution is the measurement rather
than the method.  The verdict is computed.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab22_factor_benchmark as F

SEED = 20260914
DELAYS = F.DELAYS
H = F.H
N_BOOT = 2000
BLK = 10


def gw_z(yb, fa, fb):
    """Giacomini-White on squared loss; positive favours fa."""
    d = (yb - fb) ** 2 - (yb - fa) ** 2
    return float(d.mean() / F.hac_se(d))


def boot_ci(yb, fa, fb, rng, n_boot=N_BOOT, block=BLK):
    d = (yb - fb) ** 2 - (yb - fa) ** 2
    n = len(d); nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    offs = np.arange(block)
    out = np.empty(n_boot)
    for i in range(n_boot):
        s = (starts[i][:, None] + offs).ravel()[:n]
        out[i] = d[s].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def walk(own, P, y, idx, delta, arm):
    """One arm of the horse race.  own[:,0] is the decision variable itself."""
    a = own[:, 0]
    out = np.empty(len(idx))
    b = fmu = fsd = None
    pm = psd = V = None
    beta1 = None
    beta7 = None
    _design = None

    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), cut)

            if arm in ("MARK", "MARK+", "MARK7"):
                pm, psd, V, _ = F.basis(P[tr])
                # exposure of the domestic state's CHANGE to the factor's change,
                # estimated on training rows only
                Ztr = (P[tr] - pm) / psd
                ftr = Ztr @ V[:, :1]
                da = a[tr] - a[tr - delta]
                df = ftr[:, 0] - (((P[tr - delta] - pm) / psd) @ V[:, :1])[:, 0]
                ok = np.isfinite(da) & np.isfinite(df)
                beta1 = float(np.dot(df[ok], da[ok]) / (np.dot(df[ok], df[ok]) + 1e-12)) \
                    if ok.sum() > 50 else 0.0
                if arm == "MARK7":
                    dP = P[tr] - P[tr - delta]
                    ok7 = np.isfinite(da) & np.isfinite(dP).all(axis=1)
                    if ok7.sum() > 50:
                        X7 = dP[ok7]
                        beta7 = np.linalg.solve(
                            X7.T @ X7 + 1e-6 * np.eye(X7.shape[1]), X7.T @ da[ok7])
                    else:
                        beta7 = np.zeros(P.shape[1])

            def design(rows, rows_P, rows_stale_P):
                """rows = own block held stale; rows_P = CURRENT foreign block."""
                if arm == "STALE":
                    return rows
                if arm == "PAPER":
                    return np.column_stack([rows, rows_P])
                Z = (rows_P - pm) / psd
                Zs = (rows_stale_P - pm) / psd
                if arm == "MARK7":
                    shift = (rows_P - rows_stale_P) @ beta7
                else:
                    shift = beta1 * ((Z @ V[:, :1])[:, 0] - (Zs @ V[:, :1])[:, 0])
                marked = rows[:, 0] + shift          # the reconstructed level
                if arm in ("MARK", "MARK7"):
                    return np.column_stack([marked, rows[:, 1], rows[:, 2]])
                return np.column_stack([marked, rows])      # MARK+

            Xtr = design(own[tr - delta], P[tr], P[tr - delta])
            ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(y[tr])
            Xtr, ytr = Xtr[ok], y[tr][ok]
            if len(ytr) < 200:
                b = None; continue
            fmu, fsd = Xtr.mean(0), Xtr.std(0) + 1e-9
            b = L.cv((Xtr - fmu) / fsd, ytr, "cont")
            _design = design

        if b is None or _design is None:
            out[j] = 0.0; continue
        xt = _design(own[t - delta][None, :], P[t][None, :], P[t - delta][None, :])[0]
        out[j] = 0.0 if not np.isfinite(xt).all() \
            else L.p_ridge(b, ((xt - fmu) / fsd)[None, :])[0]
    return out


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("EXPLORATORY - cited by neither paper.\n")
    own, P, y, idx, k = F.panel(folder)
    yb = y[idx]
    print(f"target {F.TARGET}, {len(idx)} test days, {k} foreign peers")
    print(f"block bootstrap, block {BLK}, {N_BOOT} draws\n")

    ARMS = ["STALE", "PAPER", "MARK", "MARK7", "MARK+"]
    print("=" * 96)
    print("A.  THE HORSE RACE")
    print("=" * 96)
    print("STALE  domestic model on delta-day-old data")
    print("PAPER  domestic model plus seven foreign closes as regressors")
    print("MARK   roll the stale level forward on ONE factor exposure, then forecast")
    print("MARK7  the same with seven exposures, to price the extra parameters")
    print("MARK+  the reconstruction alongside the domestic history\n")
    print(f"{'delta':>6}" + "".join(f"{arm:>10}" for arm in ARMS))
    R = {}
    for d in DELAYS:
        row = f"{d:>6}"
        for arm in ARMS:
            f = walk(own, P, y, idx, d, arm)
            R[(d, arm)] = f
            row += f"{F.r2(yb, f):>10.4f}"
        print(row)

    # ---------------- B -----------------------------------------------------
    print("\n" + "=" * 96)
    print("B.  DOES THE PRACTITIONER METHOD BEAT THE PAPER'S?")
    print("=" * 96)
    print("Positive favours MARK.  Both arms see identical data; they differ only in")
    print("whether the structure is imposed or estimated.\n")
    print(f"{'delta':>6}{'MARK - PAPER':>15}{'95% CI':>24}{'GW z':>8}{'winner':>10}")
    mark_wins, paper_wins = [], []
    for d in DELAYS:
        fa, fb = R[(d, "MARK")], R[(d, "PAPER")]
        dm, lo, hi = boot_ci(yb, fa, fb, np.random.default_rng(SEED + d))
        z = gw_z(yb, fa, fb)
        if lo > 0:
            mark_wins.append(d)
        if hi < 0:
            paper_wins.append(d)
        w = "MARK" if lo > 0 else ("PAPER" if hi < 0 else "neither")
        print(f"{d:>6}{F.r2(yb, fa) - F.r2(yb, fb):>+15.4f}"
              + f"[{lo:>+9.5f},{hi:>+9.5f}]".rjust(24) + f"{z:>8.2f}{w:>10}")

    print(f"\n  delays where mark-to-model significantly beats the paper's model: "
          f"{len(mark_wins)} of {len(DELAYS)} {mark_wins if mark_wins else ''}")
    print(f"  delays where the paper's model significantly beats mark-to-model: "
          f"{len(paper_wins)} of {len(DELAYS)} {paper_wins if paper_wins else ''}")

    if mark_wins and not paper_wins:
        print("\n  The practitioner method wins. The paper's regression is a worse way of")
        print("  using the same data than the restriction the industry already imposes,")
        print("  and the paper's contribution is the MEASUREMENT of the substitution rate")
        print("  rather than the method for exploiting it.  That is a real demotion and")
        print("  the paper should carry it.")
    elif paper_wins and not mark_wins:
        print("\n  The paper's model wins. Imposing the practitioner's restriction - update")
        print("  the level, then forecast as usual - throws away more than the parameters")
        print("  it saves, so letting the regression decide is not merely defensible but")
        print("  better than the standing method.")
    elif mark_wins and paper_wins:
        print("\n  Each wins somewhere. The delay at which they change places is the")
        print("  finding, and neither method should be recommended without naming it.")
    else:
        print("\n  Neither separates at any delay. Two quite different ways of using the")
        print("  same information reach the same place, which is worth knowing: the")
        print("  paper's regression is not buying anything the simple update does not,")
        print("  and is not losing anything either.")

    # ---------------- C -----------------------------------------------------
    print("\n" + "=" * 96)
    print("C.  WHAT DO THE EXTRA EXPOSURES COST?")
    print("=" * 96)
    print("MARK uses one factor exposure, MARK7 uses seven. Section 5 says parameters")
    print("are expensive here; this prices them inside the reconstruction itself.\n")
    print(f"{'delta':>6}{'MARK':>10}{'MARK7':>10}{'difference':>13}{'95% CI':>24}")
    one_better, seven_better = [], []
    for d in DELAYS:
        fa, fb = R[(d, "MARK")], R[(d, "MARK7")]
        dm, lo, hi = boot_ci(yb, fa, fb, np.random.default_rng(SEED + 5 + d))
        if lo > 0:
            one_better.append(d)
        if hi < 0:
            seven_better.append(d)
        print(f"{d:>6}{F.r2(yb, fa):>10.4f}{F.r2(yb, fb):>10.4f}"
              f"{F.r2(yb, fa) - F.r2(yb, fb):>+13.4f}"
              + f"[{lo:>+9.5f},{hi:>+9.5f}]".rjust(24))
    print(f"\n  delays where ONE exposure significantly beats seven: "
          f"{len(one_better)} of {len(DELAYS)} {one_better if one_better else ''}")
    print(f"  delays where SEVEN significantly beat one: "
          f"{len(seven_better)} of {len(DELAYS)} {seven_better if seven_better else ''}")
    print("  (delta = 0 is identically zero in both arms: with no delay there is no")
    print("   shift to compute, so the reconstruction is the stale level unchanged.)")
    if seven_better and not one_better:
        print("\n  This is the OPPOSITE of Section 5.1, and the difference is instructive.")
        print("  There, seven free coefficients were expensive because they sat inside the")
        print("  forecasting regression and had to be paid for out of forecast accuracy.")
        print("  Here they sit in a separate, much easier regression of changes on changes,")
        print("  are estimated from the whole training window, and are then used only to")
        print("  compute a shift.  Cheap parameters in the right place beat compression;")
        print("  expensive ones in the wrong place do not.  The lesson is about WHERE a")
        print("  parameter is estimated, not how many there are.")
    elif one_better and not seven_better:
        print("\n  The same lesson as Section 5.1 by another route: what is bought is the")
        print("  decision to compress, not the machinery of compressing.")
    else:
        print("\n  Neither exposure count dominates; the reconstruction is insensitive to")
        print("  how the factor is built, which is the least interesting of the three")
        print("  possible answers and is reported as such.")

    # ---------------- D -----------------------------------------------------
    print("\n" + "=" * 96)
    print("D.  DOES KEEPING THE DOMESTIC HISTORY HELP THE RECONSTRUCTION?")
    print("=" * 96)
    print("MARK replaces the stale level with the reconstructed one and keeps the")
    print("weekly and monthly averages. MARK+ keeps the stale level too, so the model")
    print("can weigh the reconstruction against what it replaced.\n")
    print(f"{'delta':>6}{'MARK':>10}{'MARK+':>10}{'difference':>13}{'95% CI':>24}")
    plus_better = []
    for d in DELAYS:
        fa, fb = R[(d, "MARK+")], R[(d, "MARK")]
        dm, lo, hi = boot_ci(yb, fa, fb, np.random.default_rng(SEED + 9 + d))
        if lo > 0:
            plus_better.append(d)
        print(f"{d:>6}{F.r2(yb, fb):>10.4f}{F.r2(yb, fa):>10.4f}"
              f"{F.r2(yb, fa) - F.r2(yb, fb):>+13.4f}"
              + f"[{lo:>+9.5f},{hi:>+9.5f}]".rjust(24))
    print(f"\n  delays where keeping the stale level alongside helps: "
          f"{len(plus_better)} of {len(DELAYS)} {plus_better if plus_better else ''}")
    if plus_better:
        print("  The reconstruction is not a sufficient statistic for the stale level:")
        print("  a forecaster should carry both rather than overwrite one with the other.")
    else:
        print("  The reconstruction absorbs what the stale level carried, which is what a")
        print("  practitioner assumes when they overwrite the mark rather than keep both.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
