"""
lab28_orthogonal_breadth.py - a referee raises omitted-variable bias:

    "Rather than capturing independent cross-border information diffusion,
     foreign breadth often proxies for global systemic risk factors.  Without
     orthogonalizing foreign breadth against global common components, the
     'transmission' mechanism may simply reflect simultaneous reactions to
     global news."

Section 5.1 already shows that ONE real-time factor beats all seven foreign
series at every delay, which is most of the way to saying the block IS a global
factor.  What it does not do is separate the two halves and ask which carries
the recovery.  This file does that.

Imports lab05_robustness and lab22_factor_benchmark; keep all three in labs/.
Runtime about eight minutes.

THE DECOMPOSITION
-----------------
At every refit the training covariance of the standardised foreign block is
eigendecomposed, exactly as in Section 5.1 - training rows only, signs pinned,
nothing from the future.  Write v1 for the leading eigenvector.  Each day's
standardised foreign row x then splits into

    x = (x . v1) v1   +   [ x - (x . v1) v1 ]
        global part        idiosyncratic remainder

and the remainder is spanned by components two through seven.  Three arms
follow: the global part alone (PC1), the remainder alone (PC2-7), and the two
together, which is just a rotation of the full block and should match it.

WHAT EACH OUTCOME WOULD MEAN
-----------------------------
If PC1 carries the recovery and the remainder adds nothing, the paper's
"breadth" is a global volatility factor observed early, and the paper should say
so plainly rather than implying bilateral diffusion.  That is not a flaw in the
result - the substitution rate is unchanged either way, and a forecaster who
wants the number can compute a cross-market average - but it changes what the
mechanism is called, and the referee is right that the current wording does not
settle it.

If the remainder DOES add, then something survives orthogonalisation against the
global component, and there is genuine cross-border idiosyncratic content.  That
would be the stronger and more surprising result.

WHAT THIS CANNOT SETTLE
------------------------
A first principal component estimated from seven equity indices is a proxy for
"global systemic risk", not a measurement of it.  Orthogonalising against it is
not the same as orthogonalising against a true global factor built from a wider
universe, and the remainder may still contain global content this basis cannot
see.  The direction of the test is informative; its magnitude is not a bound.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab22_factor_benchmark as F

SEED = 20260914
DELAYS = F.DELAYS
H = F.H if hasattr(F, "H") else L.HORIZON


def walk(own, P, y, idx, delta, arm):
    """As lab22's walk, plus a REST arm: the block orthogonal to its own PC1."""
    out = np.empty(len(idx))
    b = fmu = fsd = None
    pm = psd = V = None
    _design = None
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), cut)
            if arm in ("PC1", "REST", "PC1+REST"):
                pm, psd, V, ev = F.basis(P[tr])

            def design(rows_own, rows_P):
                if arm == "OWN":
                    return rows_own
                if arm == "FULL":
                    return np.column_stack([rows_own, rows_P])
                Z = (rows_P - pm) / psd
                if arm == "PC1":
                    return np.column_stack([rows_own, Z @ V[:, :1]])
                if arm == "REST":
                    return np.column_stack([rows_own, Z @ V[:, 1:]])
                return np.column_stack([rows_own, Z @ V])      # PC1+REST

            X = design(own[tr - delta], P[tr])
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y[tr])
            X, yy = X[ok], y[tr][ok]
            if len(yy) < 200:
                b = None; continue
            fmu, fsd = X.mean(0), X.std(0) + 1e-9
            b = L.cv((X - fmu) / fsd, yy, "cont")
            _design = design
        if b is None or _design is None:
            out[j] = 0.0; continue
        xt = _design(own[t - delta][None, :], P[t][None, :])[0]
        out[j] = 0.0 if not np.isfinite(xt).all() \
            else L.p_ridge(b, ((xt - fmu) / fsd)[None, :])[0]
    return out


def gw_z(yb, fa, fb):
    """Giacomini-White on squared loss; positive favours fa."""
    d = (yb - fb) ** 2 - (yb - fa) ** 2
    return float(d.mean() / F.hac_se(d))


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    own, P, y, idx, k = F.panel(folder)
    yb = y[idx]
    print(f"target {F.TARGET}, {len(idx)} test days, {k} foreign peers\n")

    print("=" * 96)
    print("0.  THE ROTATION IS A ROTATION")
    print("=" * 96)
    print("PC1+REST spans the same column space as the raw block, so the two arms must")
    print("agree up to the ridge penalty acting on a rotated basis.  A large gap would")
    print("mean the decomposition is wrong, not that rotation matters.\n")
    print(f"{'delta':>6}{'FULL':>12}{'PC1+REST':>12}{'difference':>13}")
    worst = 0.0
    for d in DELAYS:
        rf = F.r2(yb, walk(own, P, y, idx, d, "FULL"))
        rr = F.r2(yb, walk(own, P, y, idx, d, "PC1+REST"))
        worst = max(worst, abs(rf - rr))
        print(f"{d:>6}{rf:>12.4f}{rr:>12.4f}{rf - rr:>+13.4f}")
    print(f"\n  largest gap: {worst:.4f} of R2")
    if worst > 0.02:
        print("  The rotation does NOT reproduce the raw block.  Ridge is acting very")
        print("  differently on the two bases, so the split below is not clean and the")
        print("  numbers should be read as indicative only.")
    else:
        print("  The rotation reproduces the raw block, so splitting it is legitimate.")

    print("\n" + "=" * 96)
    print("A.  WHICH HALF CARRIES THE RECOVERY?")
    print("=" * 96)
    print("OWN is the domestic-only model.  PC1 adds the global component alone; REST")
    print("adds components two through seven, which are orthogonal to it by construction.")
    print("The last column tests PC1 against REST by Giacomini-White; positive favours")
    print("the global component.\n")
    print(f"{'delta':>6}{'OWN':>10}{'+PC1':>10}{'+REST':>10}{'+FULL':>10}"
          f"{'PC1-OWN':>10}{'REST-OWN':>11}{'GW z':>8}")
    rest_adds, pc1_adds = [], []
    for d in DELAYS:
        f_own = walk(own, P, y, idx, d, "OWN")
        f_pc1 = walk(own, P, y, idx, d, "PC1")
        f_rst = walk(own, P, y, idx, d, "REST")
        f_ful = walk(own, P, y, idx, d, "FULL")
        r_own, r_pc1 = F.r2(yb, f_own), F.r2(yb, f_pc1)
        r_rst, r_ful = F.r2(yb, f_rst), F.r2(yb, f_ful)
        z_pr = gw_z(yb, f_pc1, f_rst)
        z_ro = gw_z(yb, f_rst, f_own)
        z_po = gw_z(yb, f_pc1, f_own)
        if z_ro > 1.96:
            rest_adds.append(d)
        if z_po > 1.96:
            pc1_adds.append(d)
        print(f"{d:>6}{r_own:>10.4f}{r_pc1:>10.4f}{r_rst:>10.4f}{r_ful:>10.4f}"
              f"{r_pc1 - r_own:>+10.4f}{r_rst - r_own:>+11.4f}{z_pr:>8.2f}")

    print(f"\n  delays at which the GLOBAL component alone beats domestic-only "
          f"(GW z > 1.96): {len(pc1_adds)} of {len(DELAYS)} {pc1_adds}")
    print(f"  delays at which the ORTHOGONAL remainder beats domestic-only: "
          f"{len(rest_adds)} of {len(DELAYS)} {rest_adds}")

    if pc1_adds and not rest_adds:
        print("\n  The recovery is the global component.  Orthogonalised against it, the")
        print("  foreign block has nothing left that this design can detect, so what the")
        print("  paper calls breadth is one common factor observed while the domestic")
        print("  series is stale - not bilateral diffusion between markets.  The")
        print("  substitution rate is unaffected; what changes is the name of the")
        print("  mechanism, and the referee is right that the paper should give it.")
    elif rest_adds and pc1_adds:
        print("\n  Both halves carry something.  Orthogonalising against the global")
        print("  component does not remove the effect, so the block is not only a proxy")
        print("  for global risk and there is idiosyncratic cross-border content too.")
    elif rest_adds and not pc1_adds:
        print("\n  Only the orthogonal remainder helps, which is the reverse of what the")
        print("  factor structure in Section 5.1 would predict and should be treated as a")
        print("  finding in need of explanation rather than a confirmation.")
    else:
        print("\n  Neither half separates from the domestic-only model at these delays,")
        print("  so this decomposition cannot allocate the recovery and no claim is made.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
