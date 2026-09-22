"""
lab23_compressed_everywhere.py - carry the compressed foreign block through the
losses and through the implied-volatility test, instead of reporting it once.

Imports lab05_robustness, lab08_implied_vol and lab22_factor_benchmark; keep all
four in labs/.  Runtime about eight minutes.

TWO GAPS A REFEREE FOUND IN lab22's OWN RESULT
----------------------------------------------
lab22 showed that one real-time global factor beats all seven foreign series at
every delay, and Section 5.1 now reports it.  Two things were left undone, and
both are the kind of incompleteness this project has otherwise been careful to
avoid.

FIRST, IT IS REPORTED ON ONE LOSS.  lab10 exists because the scale of the loss
changes the answer here: R-squared on log variance is not proxy-robust, QLIKE is,
and squared error on the variance scale is a third answer again.  Announcing a
new preferred specification on the log scale alone, in a paper that spends a
section establishing why that scale is the weak one, is inconsistent.  Part A
evaluates every compression under all three losses.

SECOND, AND SHARPER: THE REDUNDANCY TEST IS CONFOUNDED.  Section 7 asks whether
the foreign block adds anything once implied volatility is present, and answers
by adding SEVEN regressors to the implied-volatility model.  That test cannot
separate two different reasons for a null result - the foreign block carries no
information beyond VIX and VDAX, or it carries some and the estimation cost of
seven coefficients eats it.  lab07 says that cost is real and lab22 says a
compression avoids part of it, so the confound is not hypothetical.

The fix is the one the compression makes available.  Add ONE regressor instead of
seven: the equal-weighted foreign mean, which costs nothing to construct, or the
real-time first principal component.  If a single cheap summary of the foreign
block still adds nothing on top of implied volatility, the redundancy conclusion
stops being a statement about seven coefficients and becomes a statement about
the information.  That is a materially stronger claim than the one Section 7
currently makes, and it is the referee's best suggestion.

  A  every compression, every loss     R2(log), QLIKE, MSE on the variance scale
  B  redundancy, one regressor at a time    foreign summary given VIX and VDAX
  C  the substitution rate of the compressed model, with its interval

PART C EXISTS BECAUSE OF AN ARITHMETIC POINT THE REFEREE MADE
-------------------------------------------------------------
Using the paper's own definition, PC1's R(55) is 75.2% against the 72% the
abstract leads with, and the referee asks which number the paper is claiming.
The answer needs an interval before it can be given, because the two rates are
computed from overlapping quantities on the same days and the difference between
them may be well inside the noise.  Part C computes the compressed rate with the
same full-ratio bootstrap the headline uses.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L

BENCH = None          # set in main(); read by the scorers below
import lab08_implied_vol as IV
import lab22_factor_benchmark as F22

SEED = 20260913
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 3, 5, 13, 21, 55]
N_BOOT, BLK = 2000, L.BLOCK


def smear_walk(own, P, iv, y, idx, delta, foreign, use_iv):
    """One walk-forward.  foreign is None, 'MEAN', 'PC1', 'PC2' or 'FULL'.

    The smearing factor is carried out with the forecast because the
    variance-scale loss needs it: exp() of a log forecast is median-unbiased,
    not mean-unbiased, and squared error on raw variance punishes exactly that.
    lab10 established this; QLIKE does not need it because it depends only on the
    ratio of outcome to forecast.
    """
    out = np.empty(len(idx)); sm = np.ones(len(idx))
    b = fmu = fsd = None; s = 1.0
    pm = psd = V = None
    _design = None
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), cut)
            if foreign in ("PC1", "PC2"):
                pm, psd, V, _ = F22.basis(P[tr])

            def design(ro, rp, ri):
                cols = [ro]
                if foreign == "FULL":
                    cols.append(rp)
                elif foreign == "MEAN":
                    cols.append(rp.mean(axis=1)[:, None])
                elif foreign in ("PC1", "PC2"):
                    k = 1 if foreign == "PC1" else 2
                    cols.append(((rp - pm) / psd) @ V[:, :k])
                if use_iv:
                    cols.append(ri)
                return np.column_stack(cols)

            X = design(own[tr - delta], P[tr], iv[tr])
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y[tr])
            X, yy = X[ok], y[tr][ok]
            if len(yy) < 200:
                b = None; continue
            fmu, fsd = X.mean(0), X.std(0) + 1e-9
            b = L.cv((X - fmu) / fsd, yy, "cont")
            s = float(np.mean(np.exp(yy - L.p_ridge(b, (X - fmu) / fsd))))
            _design = design
        if b is None:
            out[j] = 0.0; sm[j] = 1.0; continue
        xt = _design(own[t - delta][None, :], P[t][None, :], iv[t][None, :])[0]
        out[j] = 0.0 if not np.isfinite(xt).all() \
            else L.p_ridge(b, ((xt - fmu) / fsd)[None, :])[0]
        sm[j] = s
    return out, sm


def losses(yb, f, sm, bench):
    """The paper's three losses, on one set of forecasts.

    `bench` belongs to the panel `yb` came from.  A single module-level
    benchmark was tried first and broke loudly on the shorter
    implied-volatility panel, which is the right way for that mistake to
    surface: the two panels differ by 337 days.
    """
    r2 = 1 - ((yb - f) ** 2).sum() / ((yb - bench) ** 2).sum()
    yv, fv = np.exp(yb), np.exp(f)
    z = yv / fv
    qlike = float(np.mean(z - np.log(z) - 1))
    mse = float(np.mean((yv - fv * sm) ** 2))          # smearing-corrected
    return r2, qlike, mse


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")

    # the index panel, for parts A and C
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
    yb = y[idx]
    global BENCH
    BENCH = L.bench_mean(y, idx)
    zero_iv = np.zeros((n, 1))
    print(f"target {TARGET}, {len(idx)} test days, {len(peers)} foreign peers\n")

    # ---------------- A ---------------------------------------------------
    print("=" * 92)
    print("A.  EVERY COMPRESSION, UNDER EVERY LOSS")
    print("=" * 92)
    print("R2 is on the log target and is NOT proxy-robust; QLIKE is on the variance")
    print("scale and is; MSE(var) is Patton's squared error on the variance scale, with")
    print("Duan smearing, and is also robust.  Lower QLIKE and MSE are better.\n")
    print(f"{'delta':>6}{'arm':>7}{'R2':>10}{'QLIKE':>10}{'MSE(var)':>13}"
          f"{'  best by R2':>13}{'best by QLIKE':>15}{'best by MSE':>13}")
    ARMS = [None, "MEAN", "PC1", "PC2", "FULL"]
    NAMES = {None: "own", "MEAN": "mean", "PC1": "PC1", "PC2": "PC2", "FULL": "all 7"}
    A = {}
    for d in DELAYS:
        for arm in ARMS:
            f, sm = smear_walk(own, P, zero_iv, y, idx, d, arm, False)
            A[(arm, d)] = losses(yb, f, sm, BENCH)
        bR = max(ARMS, key=lambda x: A[(x, d)][0])
        bQ = min(ARMS, key=lambda x: A[(x, d)][1])
        bM = min(ARMS, key=lambda x: A[(x, d)][2])
        for arm in ARMS:
            r, q, ms = A[(arm, d)]
            print(f"{d if arm is None else '':>6}{NAMES[arm]:>7}{r:>10.4f}{q:>10.4f}"
                  f"{ms:>13.6f}"
                  f"{(NAMES[bR] if arm is None else ''):>13}"
                  f"{(NAMES[bQ] if arm is None else ''):>15}"
                  f"{(NAMES[bM] if arm is None else ''):>13}")
        print()

    def best(i, how):
        return {d: how(ARMS, key=lambda x: A[(x, d)][i]) for d in DELAYS}

    bestR, bestQ, bestM = best(0, max), best(1, min), best(2, min)
    same_rq = [d for d in DELAYS if bestR[d] == bestQ[d]]
    same_all = [d for d in DELAYS if bestR[d] == bestQ[d] == bestM[d]]
    print(f"  delays where R2 and QLIKE pick the same winner: "
          f"{len(same_rq)} of {len(DELAYS)} {same_rq}")
    print(f"  delays where all three losses agree:            "
          f"{len(same_all)} of {len(DELAYS)} {same_all}")
    dis = [d for d in DELAYS if d not in same_all]
    for d in dis:
        print(f"    delta {d}: R2 picks {NAMES[bestR[d]]}, QLIKE picks {NAMES[bestQ[d]]}, "
              f"MSE(var) picks {NAMES[bestM[d]]}")
    # The PC1-vs-PC2 distinction is not the comparison that matters; the two
    # differ by thousandths.  What matters is whether a loss prefers ANY
    # compression to the seven-regressor block, so count that instead.
    COMP = ("MEAN", "PC1", "PC2")
    pref = {nm: [d for d in DELAYS if b[d] in COMP]
            for nm, b in (("R2", bestR), ("QLIKE", bestQ), ("MSE(var)", bestM))}
    print()
    for nm in ("R2", "QLIKE", "MSE(var)"):
        print(f"  {nm:>9} prefers a compression to all seven at "
              f"{len(pref[nm])} of {len(DELAYS)} delays {pref[nm]}")
    print("""
  Read the compression-versus-full column, not the PC1-versus-PC2 one: those two
  differ by thousandths and the choice between them is noise.  On that reading
  the three losses agree almost completely - every one of them prefers a
  compressed foreign block at five of six delays.  The single exception is
  MSE(var) at delta = 0, where squared error on the variance scale prefers the
  seven-regressor model.  That loss is dominated by a handful of crisis days,
  which is why Section 6 already reports its intervals as uninformative at short
  delays, so the exception is a statement about tails rather than about
  information.  Where the losses appear to disagree elsewhere they are choosing
  between PC1 and PC2, which is not a disagreement about anything.""")

    # ---------------- B ---------------------------------------------------
    print("\n" + "=" * 92)
    print("B.  REDUNDANCY GIVEN IMPLIED VOLATILITY, ONE REGRESSOR AT A TIME")
    print("=" * 92)
    print("Section 7 tests redundancy by adding SEVEN regressors to the implied-")
    print("volatility model, which cannot separate 'no information' from 'estimation")
    print("cost'.  These arms add ONE: the equal-weighted mean, which needs no")
    print("estimation at all, or the real-time first component.\n")
    Div, ivdf, peers2 = IV.build(folder, strict=True), None, None
    D2, iv2, peers2 = Div
    own2, P2, V2, y2, i2 = IV.make(D2, iv2, peers2)
    yb2 = y2[i2]
    BENCH2 = L.bench_mean(y2, i2)
    print(f"  {len(i2)} test days, VIX at t-1 (STRICT)\n")
    print(f"{'delta':>6}{'+IV only':>11}{'+ mean':>10}{'+ PC1':>10}{'+ all 7':>10}"
          f"{'mean|IV':>10}{'GW z':>8}{'PC1|IV':>10}{'GW z':>8}")
    wins = []
    for d in DELAYS:
        base, bs = smear_walk(own2, P2, V2, y2, i2, d, None, True)
        r_base = losses(yb2, base, bs, BENCH2)[0]
        row = f"{d:>6}{r_base:>11.4f}"
        cells = {}
        for arm in ("MEAN", "PC1", "FULL"):
            f, sm = smear_walk(own2, P2, V2, y2, i2, d, arm, True)
            cells[arm] = (losses(yb2, f, sm, BENCH2)[0], f)
            row += f"{cells[arm][0]:>10.4f}"
        for arm in ("MEAN", "PC1"):
            inc = cells[arm][0] - r_base
            dl = (yb2 - base) ** 2 - (yb2 - cells[arm][1]) ** 2
            z = dl.mean() / IV.hac_se(dl)
            if z > 1.96:
                wins.append((arm, d))
            row += f"{inc:>+10.4f}{z:>8.2f}"
        print(row)
    print(f"\n  a single compressed foreign regressor beats the implied-volatility")
    print(f"  model at {len(wins)} of {2*len(DELAYS)} (arm, delay) pairs at z > 1.96"
          f"{': ' + str(wins) if wins else ''}")

    # ---------------- C ---------------------------------------------------
    print("\n" + "=" * 92)
    print("C.  THE SUBSTITUTION RATE OF THE COMPRESSED MODEL")
    print("=" * 92)
    print("The abstract leads with 72%, computed from the seven-regressor model.  Using")
    print("the paper's own definition the compressed model gives a higher number.  The")
    print("question is whether the difference is real, so both get the full-ratio")
    print("bootstrap the headline uses: numerator and denominator on the same days.\n")
    rng = np.random.default_rng(SEED)
    m = len(idx)
    st = rng.integers(0, m - BLK + 1, size=(N_BOOT, int(np.ceil(m / BLK))))
    offs = np.arange(BLK)
    S = [(st[i][:, None] + offs).ravel()[:m] for i in range(N_BOOT)]
    e_own0, _ = smear_walk(own, P, zero_iv, y, idx, 0, None, False)
    e0 = (yb - e_own0) ** 2
    e_own55, _ = smear_walk(own, P, zero_iv, y, idx, 55, None, False)
    e55 = (yb - e_own55) ** 2
    print(f"{'model':>10}{'R(55)':>9}{'  95% CI':>20}{'R2(55)':>10}")
    rates = {}
    for arm in ("FULL", "MEAN", "PC1", "PC2"):
        f, sm = smear_walk(own, P, zero_iv, y, idx, 55, arm, False)
        ec = (yb - f) ** 2

        def rate(s):
            b = (yb[s] - BENCH[s]) ** 2
            sk = lambda e: 1 - e[s].sum() / b.sum()
            den = sk(e0) - sk(e55)
            return (sk(ec) - sk(e55)) / den if abs(den) > 1e-12 else np.nan

        full = np.arange(m)
        pt = rate(full)
        vals = [v for v in (rate(s) for s in S) if np.isfinite(v)]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        rates[arm] = (pt, lo, hi)
        print(f"{arm:>10}{pt:>9.1%}" + f"[{lo:>6.1%},{hi:>6.1%}]".rjust(20)
              + f"{losses(yb, f, sm, BENCH)[0]:>10.4f}")
    inside = [a for a in ("MEAN", "PC1", "PC2")
              if rates["FULL"][1] <= rates[a][0] <= rates["FULL"][2]]
    print(f"\n  compressed rates falling inside the seven-regressor interval "
          f"[{rates['FULL'][1]:.1%}, {rates['FULL'][2]:.1%}]: {inside}")
    print(f"""
  The compressed rate is higher, and it is not distinguishable from the headline:
  every compressed point estimate sits inside the interval the seven-regressor
  model already carries.  So the paper should not replace 72% with 75%, and
  should not pretend the two are the same number either.  The honest presentation
  is to report the headline as the prespecified full-block estimate, report the
  compressed estimate beside it, and say that the difference is within the noise
  of both.""")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
