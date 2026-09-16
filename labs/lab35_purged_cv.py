"""
lab35_purged_cv.py - is the purge big enough?  Section 9.4.

Is the gap this project already leaves big enough?

THE CONCERN
-----------
The target is realised variance over the NEXT FIVE DAYS, so consecutive
observations share four days of outcome.  Split such a series naively and the
training set contains the answer to part of the test set.  Lopez de Prado's
purged and embargoed cross-validation is the standard remedy: remove training
observations whose outcome window overlaps the test point, then remove a further
buffer on top.

WHAT THE PROJECT ALREADY DOES
------------------------------
Every lab here is walk-forward, and every refit cuts the training window at

    cut = t - delta - HORIZON

The `- HORIZON` is a purge: it drops exactly the rows whose five-day outcome
window would still be unresolved at the moment of forecasting.  So the design is
already purged; it has simply never been called that, and - more to the point -
the size of the gap has never been tested.  A purge of exactly H is the minimum
that removes overlap.  It leaves no embargo at all.

WHAT THIS FILE DOES
-------------------
Vary the gap and watch what happens to out-of-sample skill:

    NO PURGE     cut = t - delta            the leak, deliberately reintroduced
    CURRENT      cut = t - delta - H        what every other lab does
    +5, +10, +21 cut = t - delta - H - E    an embargo on top of the purge

HOW TO READ IT
--------------
If NO PURGE scores higher than CURRENT, the purge is doing real work and the
project needs it - which also means any study of this target that splits naively
is reporting inflated numbers.

If CURRENT, +5, +10 and +21 sit on top of each other, the existing gap is
sufficient and no embargo is needed.  If skill keeps falling as the embargo
grows, there is residual leakage the purge does not catch, and every number in
both papers is optimistic by whatever that decline is worth.

The second is the outcome that would matter, and it is the reason to run this
rather than assert that walk-forward is safe.
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
GAPS = [("no purge", -H), ("current (purge H)", 0),
        ("+5 embargo", 5), ("+10 embargo", 10), ("+21 embargo", 21)]
N_BOOT = 2000
BLK = 10


def walk(own, P, y, idx, delta, extra, use_peers=True):
    """The project's walk-forward, with the training cut moved by `extra` days."""
    out = np.empty(len(idx)); b = mu = sd = None
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H - extra
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


def boot_ci(yb, fa, fb, rng, n_boot=N_BOOT, block=BLK):
    d = (yb - fb) ** 2 - (yb - fa) ** 2
    n = len(d); nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    offs = np.arange(block)
    o = np.empty(n_boot)
    for i in range(n_boot):
        s = (starts[i][:, None] + offs).ravel()[:n]
        o[i] = d[s].mean()
    lo, hi = np.percentile(o, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints the purge and embargo results of Section 9.4.\n")
    own, P, y, idx, k = F.panel(folder)
    yb = y[idx]
    print(f"target {F.TARGET}, {len(idx)} test days, {k} foreign peers")
    print(f"target horizon H = {H} days, so consecutive targets share {H - 1}\n")

    # ---------------- A ---------------------------------------------------
    print("=" * 96)
    print("A.  OUT-OF-SAMPLE SKILL AS THE GAP GROWS")
    print("=" * 96)
    print("Cross-sectional model. 'no purge' puts the leak back deliberately; every")
    print("other column adds distance between training and the forecast date.\n")
    print(f"{'delta':>6}" + "".join(f"{nm:>20}" for nm, _ in GAPS))
    R = {}
    for d in DELAYS:
        row = f"{d:>6}"
        for nm, e in GAPS:
            f = walk(own, P, y, idx, d, e)
            R[(d, nm)] = f
            row += f"{F.r2(yb, f):>20.4f}"
        print(row)

    # ---------------- B ---------------------------------------------------
    print("\n" + "=" * 96)
    print("B.  IS THE PURGE DOING ANYTHING?")
    print("=" * 96)
    print("no purge minus current. Positive means removing the purge FLATTERS the")
    print("result, which is the definition of a leak.\n")
    print(f"{'delta':>6}{'no purge - current':>21}{'95% CI':>24}{'leak?':>8}")
    leaks = []
    for d in DELAYS:
        fa, fb = R[(d, "no purge")], R[(d, "current (purge H)")]
        dm, lo, hi = boot_ci(yb, fa, fb, np.random.default_rng(SEED + d))
        if lo > 0:
            leaks.append(d)
        print(f"{d:>6}{F.r2(yb, fa) - F.r2(yb, fb):>+21.4f}"
              + f"[{lo:>+9.5f},{hi:>+9.5f}]".rjust(24)
              + f"{'yes' if lo > 0 else 'no':>8}")
    print(f"\n  delays where dropping the purge significantly flatters the result: "
          f"{len(leaks)} of {len(DELAYS)} {leaks if leaks else ''}")
    if leaks:
        print("  The purge is load-bearing. Any study of a five-day-ahead target that")
        print("  splits without one is reporting numbers inflated by this much.")
    else:
        print("  Removing the purge changes nothing detectable, so the overlap in this")
        print("  target is not a route to leakage at this sample size - which is worth")
        print("  knowing, because the purge is usually justified by assertion.")

    # ---------------- C ---------------------------------------------------
    print("\n" + "=" * 96)
    print("C.  IS THE CURRENT GAP BIG ENOUGH?")
    print("=" * 96)
    print("Each embargo against the current protocol. A DECLINE that keeps going as")
    print("the embargo grows would mean residual leakage the purge does not catch,")
    print("and every number in both papers would be optimistic by that much.\n")
    print(f"{'delta':>6}" + "".join(f"{nm:>22}" for nm, _ in GAPS[2:]))
    resid = []
    for d in DELAYS:
        row = f"{d:>6}"
        for nm, _ in GAPS[2:]:
            fa, fb = R[(d, nm)], R[(d, "current (purge H)")]
            dm, lo, hi = boot_ci(yb, fa, fb, np.random.default_rng(SEED + 7 + d))
            if hi < 0:
                resid.append((d, nm))
            row += f"{F.r2(yb, fa) - F.r2(yb, fb):>+11.4f}" + \
                   f"{'*' if hi < 0 else ' ':>2}" + f"{'':>9}"
        print(row)
    print("\n  * = the embargo significantly REDUCES skill at that delay")
    print(f"  cells where a larger embargo significantly reduces skill: "
          f"{len(resid)} of {len(DELAYS) * 3}")

    mono = []
    for d in DELAYS:
        vals = [F.r2(yb, R[(d, nm)]) for nm, _ in GAPS[1:]]
        mono.append(all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1)))
    print(f"  delays where skill falls monotonically with the embargo: "
          f"{sum(mono)} of {len(DELAYS)}")

    if len(resid) == 0:
        print("\n  Adding an embargo on top of the purge costs nothing detectable at any")
        print("  delay. The gap this project already leaves is sufficient, and the")
        print("  walk-forward design was purged correctly before anyone called it that.")
    elif sum(mono) >= len(DELAYS) - 1:
        print("\n  Skill falls monotonically as the gap widens at nearly every delay, which")
        print("  is the signature of residual leakage rather than of noise. Both papers'")
        print("  numbers should be read as optimistic by roughly the size of that decline.")
    else:
        print("\n  Some embargoes reduce skill and the pattern is not monotone, which is")
        print("  what losing recent training data looks like rather than what removing a")
        print("  leak looks like. The two are separated by the monotonicity count above,")
        print("  and on this evidence it is the former.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
