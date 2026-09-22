"""
lab34_data_snooping.py - Reality Check and SPA; Section 9.3.

What survives once you admit how many models were tried.

WHY THIS FILE EXISTS
--------------------
Thirty-four labs have now been run against the same 4,862 days.  Every one of
them compared some model against the domestic baseline, and the ones that
separated were reported.  That is how research proceeds, and it is also how a
literature accumulates false positives: with enough specifications, the best of
them looks good whether or not anything is there.

The individual tests in this project are honest.  Giacomini-White is applied
correctly, the intervals are block-bootstrapped, Section 6 already carries a
simultaneous band across the DELAY grid with a critical value of 6.51 rather
than 1.96.  What none of that covers is the number of MODELS tried.  A band
across ten delays for one model is not a band across a dozen models.

White's Reality Check and Hansen's Superior Predictive Ability test are the
corrections for exactly that.  Both ask one question: is the BEST of k models
better than the benchmark, once you account for having picked the best of k?

THE TESTS
---------
Write d_j(t) for the per-day loss differential between model j and the
benchmark, positive when the model is better.  The statistic is

    T = max_j  sqrt(n) * mean(d_j)

and its null distribution comes from a stationary bootstrap of the d_j's,
recentred so the null holds:

    Reality Check   recentre every model by its own mean
    SPA             recentre only models that are not too far below zero,
                    which stops a hopeless model from inflating the critical
                    value and makes the test less conservative

Hansen's point is that the Reality Check is dragged around by bad models.  Both
are reported because the gap between them is itself informative: if the two
disagree, the benchmark is being flattered by the company it keeps.

WHAT IS IN THE FAMILY
---------------------
Every distinct forecasting model this project has built that runs on the main
panel and can be evaluated on identical days: the paper's own cross-sectional
model, the compressions from Section 5.1, the factor split from Section 5.3, the
practitioner's mark-to-model, the masked model, and the ragged-edge filter.  The
benchmark is the domestic-only model, which is what every one of them was
introduced to beat.

WHAT WOULD BE BAD NEWS
-----------------------
If the best model fails at conventional levels once the family is accounted for,
then the paper's headline is not safe from data snooping and the honest move is
to say so.  If it survives, the 72% has cleared a test no individual comparison
in either paper attempts.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab22_factor_benchmark as F
import lab31_mark_to_model as MM
import lab32_masked_training as MT
import lab33_ragged_edge as RE

SEED = 20260914
DELAYS = F.DELAYS
H = F.H
N_BOOT = 2000
Q_GEOM = 0.1          # stationary-bootstrap block parameter; mean block 1/q = 10


def stationary_indices(n, rng, q=Q_GEOM):
    """Politis-Romano stationary bootstrap: geometric blocks, wrapped."""
    idx = np.empty(n, dtype=np.int64)
    i = rng.integers(0, n)
    for t in range(n):
        idx[t] = i
        if rng.random() < q:
            i = rng.integers(0, n)
        else:
            i = (i + 1) % n
    return idx


def reality_check(D, rng, n_boot=N_BOOT):
    """White's Reality Check and Hansen's SPA on a (k, n) matrix of differentials.

    D[j] is the per-day loss differential of model j against the benchmark,
    positive when the model is better.  Returns both p-values and the observed
    statistic.
    """
    k, n = D.shape
    mu = D.mean(axis=1)
    se = np.array([F.hac_se(D[j]) * np.sqrt(n) for j in range(k)])
    se = np.maximum(se, 1e-12)
    T = float(np.max(np.sqrt(n) * mu))

    # Hansen's threshold: a model more than this far below zero is dropped from
    # the recentring, so hopeless models stop inflating the critical value.
    thr = -se * np.sqrt(2.0 * np.log(np.log(max(n, 3))))
    keep = mu >= thr / np.sqrt(n)

    boots_rc = np.empty(n_boot)
    boots_spa = np.empty(n_boot)
    for b in range(n_boot):
        sel = stationary_indices(n, rng)
        mb = D[:, sel].mean(axis=1)
        rc = np.sqrt(n) * (mb - mu)                    # recentre everything
        boots_rc[b] = float(np.max(rc))
        sp = np.where(keep, np.sqrt(n) * (mb - mu), np.sqrt(n) * mb)
        boots_spa[b] = float(np.max(sp))
    p_rc = float(np.mean(boots_rc >= T))
    p_spa = float(np.mean(boots_spa >= T))
    return T, p_rc, p_spa, int(keep.sum())


def build_family(own, P, y, idx, delta, M, par, st):
    """Every forecaster this project has built, on identical days."""
    fam = {}
    fam["paper: + seven closes"] = MM.walk(own, P, y, idx, delta, "PAPER")
    for arm, name in (("MEAN", "5.1: + equal-weighted mean"),
                      ("PC1", "5.1: + one real-time factor"),
                      ("PC2", "5.1: + two factors")):
        f = F.walk(own, P, y, idx, delta, arm)
        fam[name] = f[0] if isinstance(f, tuple) else f
    fam["5.3: + global component only"] = __import__(
        "lab28_orthogonal_breadth").walk(own, P, y, idx, delta, "PC1")
    fam["lab31: mark-to-model"] = MM.walk(own, P, y, idx, delta, "MARK")
    fam["lab31: mark-to-model +"] = MM.walk(own, P, y, idx, delta, "MARK+")
    fam["lab33: ragged-edge filter"] = RE.walk(own, P, y, idx, delta, "FILTER", st)
    fam["lab33: filter + history"] = RE.walk(own, P, y, idx, delta, "FILTER+", st)
    return fam


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints the Reality Check and SPA results of Section 9.3.\n")
    own, P, y, idx, k = F.panel(folder)
    yb = y[idx]
    BENCH = L.bench_mean(y, idx)
    a = own[:, 0]
    M = np.column_stack([a, P])
    par = RE.fit_state_space(M[:L.TRAIN + L.VAL])
    print(f"target {F.TARGET}, {len(idx)} test days, {k} foreign peers")
    print(f"stationary bootstrap, mean block {1 / Q_GEOM:.0f} days, {N_BOOT} draws\n")

    # ---------------- 0. does the test have the size it claims? -----------
    print("=" * 96)
    print("0.  DOES THE TEST REJECT AT THE RATE IT PROMISES?")
    print("=" * 96)
    print("Under a true null - every model a pure noise forecast of the same target -")
    print("a 5% test should reject about 5% of the time, and the Reality Check should")
    print("reject LESS often than SPA because it recentres hopeless models too.\n")
    rng = np.random.default_rng(SEED)
    n_sim, k_sim, reps = 800, 10, 200
    rej_rc = rej_spa = 0
    for _ in range(reps):
        Dn = rng.standard_normal((k_sim, n_sim)) * 0.1
        for j in range(k_sim):                       # give them realistic persistence
            for t in range(1, n_sim):
                Dn[j, t] += 0.5 * Dn[j, t - 1]
        _, prc, psp, _ = reality_check(Dn, rng, n_boot=200)
        rej_rc += (prc < 0.05); rej_spa += (psp < 0.05)
    print(f"  {reps} replications, {k_sim} null models, {n_sim} days each")
    print(f"  Reality Check rejects at 5%: {rej_rc / reps:.1%}")
    print(f"  SPA rejects at 5%:           {rej_spa / reps:.1%}")
    ok = rej_rc / reps < 0.12 and rej_spa / reps < 0.15
    print(f"\n  {'Sizes are usable.' if ok else 'SIZE IS OFF - read everything below with that in mind.'}")
    print(f"  SPA rejects {'more' if rej_spa >= rej_rc else 'less'} often than the Reality Check, "
          f"{'as Hansen predicts.' if rej_spa >= rej_rc else 'which is not what Hansen predicts.'}")

    # ---------------- A ---------------------------------------------------
    print("\n" + "=" * 96)
    print("A.  THE WHOLE FAMILY AGAINST THE DOMESTIC BENCHMARK")
    print("=" * 96)
    print("Nine models per delay, all introduced somewhere in this project to beat the")
    print("same domestic-only baseline.  p is for the BEST of the nine, corrected for")
    print("there being nine.\n")
    print(f"{'delta':>6}{'models':>8}{'best model':>30}{'best dR2':>10}"
          f"{'RC p':>8}{'SPA p':>8}{'kept':>6}")
    survive_rc, survive_spa, best_dr2 = [], [], {}
    for d in DELAYS:
        st = RE.ragged_state(par, M, d)
        base = MM.walk(own, P, y, idx, d, "STALE")
        e0 = (yb - base) ** 2
        fam = build_family(own, P, y, idx, d, M, par, st)
        names = list(fam)
        D = np.vstack([e0 - (yb - fam[nm]) ** 2 for nm in names])
        T, p_rc, p_spa, kept = reality_check(D, np.random.default_rng(SEED + d))
        j = int(np.argmax(D.mean(axis=1)))
        best = F.r2(yb, fam[names[j]], BENCH) - F.r2(yb, base, BENCH)
        best_dr2[d] = best
        if p_rc < 0.05:
            survive_rc.append(d)
        if p_spa < 0.05:
            survive_spa.append(d)
        print(f"{d:>6}{len(names):>8}{names[j]:>30}{best:>+10.4f}"
              f"{p_rc:>8.3f}{p_spa:>8.3f}{kept:>6}")

    print(f"\n  delays where the best of nine still beats the benchmark at 5%")
    print(f"    Reality Check: {len(survive_rc)} of {len(DELAYS)} {survive_rc}")
    print(f"    SPA:           {len(survive_spa)} of {len(DELAYS)} {survive_spa}")

    if len(survive_spa) == len(DELAYS):
        print("\n  Everything survives.  The project's central result is not an artefact of")
        print("  how many models were tried: correcting for a family of nine changes no")
        print("  verdict at any delay.")
    elif survive_spa:
        lost = [d for d in DELAYS if d not in survive_spa]
        print(f"\n  The result survives at {survive_spa} and does NOT at {lost}.")
        for d in lost:
            print(f"    delta = {d}: best margin {best_dr2[d]:+.4f} of R2.  "
                  + ("nothing beats the benchmark there at all, so this is an absence"
                     if abs(best_dr2[d]) < 0.01 else
                     "the margin is real but inside what picking the best of nine gives"))
        print("\n  Where the best margin is near zero the correction is not what killed the")
        print("  result - there was nothing to kill.  That is the delta = 0 row the paper")
        print("  already declines to interpret, and this file agrees with it by a route")
        print("  the paper does not use.")
    else:
        print("\n  Nothing survives the correction.  Every apparent win in this project is")
        print("  within what picking the best of nine models would produce by chance, and")
        print("  that is the headline finding of this file.")

    # ---------------- B ---------------------------------------------------
    print("\n" + "=" * 96)
    print("B.  WHAT DOES THE CORRECTION ACTUALLY COST?")
    print("=" * 96)
    print("The uncorrected p-value for the paper's own model, against the corrected one")
    print("for the best of nine.  The gap is the price of the search.\n")
    res = 1.0 / N_BOOT
    print(f"  bootstrap resolution is 1/{N_BOOT} = {res:.4f}; a p-value at or below it")
    print("  is reported as such rather than as a number the bootstrap cannot support.\n")
    print(f"{'delta':>6}{'paper model z':>15}{'uncorrected p':>17}{'SPA p (best of 9)':>20}")
    for d in DELAYS:
        st = RE.ragged_state(par, M, d)
        base = MM.walk(own, P, y, idx, d, "STALE")
        e0 = (yb - base) ** 2
        fam = build_family(own, P, y, idx, d, M, par, st)
        names = list(fam)
        dpaper = e0 - (yb - fam["paper: + seven closes"]) ** 2
        z = float(dpaper.mean() / F.hac_se(dpaper))
        p_un = F.norm_p(z) / 2.0 if z > 0 else 1.0
        D = np.vstack([e0 - (yb - fam[nm]) ** 2 for nm in names])
        _, _, p_spa, _ = reality_check(D, np.random.default_rng(SEED + 5 + d))
        us = f"{p_un:.2e}" if p_un < 1e-4 else f"{p_un:.4f}"
        ss = f"< {res:.4f}" if p_spa <= res else f"{p_spa:.4f}"
        print(f"{d:>6}{z:>15.2f}{us:>17}{ss:>20}")
    print("\n  Where both columns sit below the bootstrap's resolution the honest")
    print("  statement is that the correction demonstrably costs nothing, not that it")
    print("  costs some particular amount: a ratio of two numbers the bootstrap cannot")
    print("  resolve measures the bootstrap, not the search.  An earlier version of this")
    print("  file printed that ratio and one row read 65,913, which is a statement about")
    print("  2,000 draws and nothing else.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
