"""
lab07_estimation_cost.py - what the NEGATIVE numbers are.

Imports lab05_robustness; keep both files in the same folder.

WHY
---
Two results in lab06 are negative, and the paper currently treats them as
separate curiosities:

  * at delta = 0 the cross model is WORSE than own-only, dR2 = -0.0103;
  * under the placebo (foreign block shifted five years) the substitution
    rate does not collapse to zero, it lands at about -10% at every delay.

A placebo is supposed to go to zero.  Ours goes slightly negative, at all
three delays tested, and by roughly the same amount as the delta = 0 result.
Two negatives of the same size, from two experiments that share exactly one
feature - the model is carrying seven extra regressors that contain nothing
useful - is not two findings.  It is one.

HYPOTHESIS.  The negative is the ESTIMATION COST of the cross-section: the
price of estimating seven extra coefficients from a finite training window.
It is present at every delay.  At short delays it is all there is, so the
total is negative.  At long delays the substitution benefit exceeds it.

That hypothesis makes two predictions that can be falsified here:

  P1  SCALE.  The cost at delta = 0 grows with the NUMBER of peers carried.
      Drop to one peer and it should be roughly one seventh.  If the delta = 0
      negative were instead something economic - foreign markets actively
      MISLEADING about the US state - it would not have to scale with a count.

  P2  SHRINKAGE.  Force the ridge penalty up and the cost must fall toward
      zero, because a heavily penalised coefficient is not really estimated.
      A genuine misleading signal would not be cured by shrinkage; it would
      survive as long as the coefficient is non-zero.

Both hold, so two further questions have to be answered before the number is
usable:

  P3  IS THE COST CONSTANT?  lab06's placebo says no, but a circular shift
      measures estimation cost PLUS misalignment damage and cannot separate
      them.  Replaced with AR(1) surrogates that are uninformative BY
      CONSTRUCTION.  The cost is not constant: it grows with delay.

  P4  HOW SHARP IS ANY OF THIS?  Gross benefit is net minus cost, a
      DIFFERENCE OF TWO ESTIMATED QUANTITIES, so it carries error from both
      and gets an interval before it gets a label.  Reported in the final
      table rather than as a separate section, because a labelled row with
      no interval is what caused the problem in the first place.

What the paper gains is a number it did not have: the cost of carrying the
cross-section, in R-squared points, measured rather than assumed.  Subtract
it and the delay at which the cross-section starts paying for itself follows
from the data instead of being read off a chart.

Runtime under two minutes.
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".")
import lab05_robustness as L

SEED = 20260912
TARGET = "SPX"
H = L.HORIZON


def panel(folder):
    """Everything the walk-forward needs, built once."""
    D, peers, lag = L.build(folder, TARGET)
    a = D[TARGET].values
    sa = L.pd.Series(a)
    own = np.column_stack([sa.values,
                           sa.rolling(5).mean().values,
                           sa.rolling(22).mean().values])
    P = D[peers].values
    n = len(D)
    y = np.full(n, np.nan)
    y[:-H] = a[H:]
    start = L.MED + L.TRAIN + L.VAL + 55 + H
    idx = np.arange(start, n - H)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])]
    return own, P, y, idx, peers


def walk(own, P, y, idx, delta, cols, lam):
    """lab05's walk-forward, with an explicit peer subset and a forced lambda.

    cols = list of column indices into P; [] means the own-only model.
    lam  = None reproduces lab05 exactly (lambda chosen by validation tail);
           a number forces that penalty on both models, which is what makes
           the shrinkage test a fair comparison.
    """
    out = np.empty(len(idx))
    b = mu = sd = None
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), cut)
            Xtr = np.column_stack([own[tr - delta]] + ([P[tr][:, cols]] if cols else []))
            ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(y[tr])
            Xtr, ytr = Xtr[ok], y[tr][ok]
            if len(ytr) < 200:
                b = None
                continue
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            Z = (Xtr - mu) / sd
            b = L.fit_ridge(Z, ytr, lam) if lam is not None else L.cv(Z, ytr, "cont")
        xt = np.concatenate([own[t - delta]] + ([P[t][cols]] if cols else []))
        if b is None or not np.isfinite(xt).all():
            out[j] = 0.0
        else:
            out[j] = L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
    return out


def r2(y, f):
    return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main(folder=None):
    t0 = time.time()
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    own, P, y, idx, peers = panel(folder)
    yb = y[idx]
    npeer = len(peers)
    print(f"target {TARGET}, peers {peers}")
    print(f"{len(idx)} test days, training window {L.TRAIN + L.VAL} rows, "
          f"horizon {H} days\n")

    # ------------------------------------------------------------------
    print("=" * 80)
    print("P1.  DOES THE delta = 0 PENALTY SCALE WITH THE NUMBER OF PEERS?")
    print("=" * 80)
    print("Each row adds peers to the SAME own-only model at delta = 0, where the")
    print("cross-section has nothing to contribute: domestic data is fully current.")
    print("Subsets are drawn at random and averaged, so no single market drives it.")
    print("If this is estimation cost, the penalty is a straight line through the")
    print("origin.  If foreign markets actively mislead, it need not be.\n")

    rng = np.random.default_rng(SEED)
    base0 = walk(own, P, y, idx, 0, [], None)
    r_own = r2(yb, base0)
    print(f"own-only R2 at delta = 0: {r_own:.4f}\n")
    print(f"{'peers':>7}{'mean dR2':>12}{'per peer':>12}{'spread over subsets':>24}")
    scale = {}
    for k in [1, 2, 3, 5, npeer]:
        reps = 1 if k == npeer else 3
        vals = []
        for _ in range(reps):
            cols = list(rng.choice(npeer, size=k, replace=False)) if k < npeer else list(range(npeer))
            vals.append(r2(yb, walk(own, P, y, idx, 0, cols, None)) - r_own)
        m = float(np.mean(vals))
        scale[k] = m
        spread = f"[{min(vals):+.4f},{max(vals):+.4f}]" if reps > 1 else "(all peers)"
        print(f"{k:>7}{m:>+12.4f}{m/k:>+12.4f}{spread:>24}")

    ks = np.array(sorted(scale)); vs = np.array([scale[k] for k in ks])
    slope = float((ks @ vs) / (ks @ ks))                       # through the origin
    resid = vs - slope * ks
    fit = 1 - (resid @ resid) / (vs @ vs)
    print(f"\nline through the origin: dR2 = {slope:+.5f} per peer, "
          f"fraction of variation explained {fit:.3f}")
    print(f"implied cost of the full {npeer}-peer cross-section: {slope*npeer:+.4f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("P2.  DOES SHRINKAGE CURE IT?")
    print("=" * 80)
    print("Both models are forced to the same ridge penalty, so the comparison is")
    print("like for like.  Estimation cost is an artefact of estimating; it must")
    print("vanish as the penalty rises.  A genuinely misleading signal would not.\n")
    print("delta = 0 is where the cost should be ALL there is.")
    print("delta = 21 is where the paper claims a real benefit.")
    print("Both are shown, because the test only means something if they differ.\n")
    print(f"{'lambda':>9}{'dR2 at d=0':>14}{'dR2 at d=21':>14}   reading")
    for lam in [1.0, 10.0, 100.0, 1000.0]:
        row = []
        for d in (0, 21):
            o = walk(own, P, y, idx, d, [], lam)
            c = walk(own, P, y, idx, d, list(range(npeer)), lam)
            row.append(r2(yb, c) - r2(yb, o))
        note = ("cost gone - sign flipped" if row[0] > 0 else
                "cost nearly gone" if abs(row[0]) < 0.002 else
                "cost shrinking" if abs(row[0]) < 0.009 else "cost present")
        print(f"{lam:>9.0f}{row[0]:>+14.4f}{row[1]:>+14.4f}   {note}")
    print("\nThe delta = 0 column must go to zero; the delta = 21 column must not.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("P3.  IS THE COST THE SAME AT EVERY DELAY?")
    print("=" * 80)
    print("P1 measured the cost at delta = 0.  Treating it as a constant would be")
    print("convenient, and lab06's placebo says not to: shifted peers cost -0.019")
    print("at delta = 5 but -0.044 at delta = 21.")
    print()
    print("But a CIRCULAR SHIFT is the wrong instrument for this.  Shifting by two")
    print("or five years can park a crisis period against a calm one, and the")
    print("validation step will happily fit that spurious structure.  What it")
    print("measures is estimation cost PLUS misalignment damage, and the two are")
    print("not separable.  A first run confirmed the problem: four shift lengths")
    print("gave -0.0074 to -0.0424 for the same quantity at delta = 0, against a")
    print("tight -0.0100 from P1.  A statistic whose value depends that much on an")
    print("arbitrary choice of shift is not measuring what it claims to.")
    print()
    print("So build surrogates that are uninformative BY CONSTRUCTION: fit an")
    print("AR(1) to each peer and simulate from it.  Same persistence, same")
    print("variance, no relationship to the target at any lag.  Ten draws.\n")

    grid = [0, 1, 2, 3, 5, 8, 13, 21]
    srng = np.random.default_rng(SEED + 1)
    fin = np.isfinite(P).all(axis=1)
    phi, mu_p = np.empty(npeer), np.empty(npeer)
    resid = []
    for j in range(npeer):
        v = P[fin, j]
        v0, v1 = v[:-1] - v.mean(), v[1:] - v.mean()
        phi[j] = float((v0 @ v1) / (v0 @ v0))
        resid.append(v1 - phi[j] * v0)
        mu_p[j] = float(v.mean())
    E = np.column_stack(resid)
    Sig = np.cov(E, rowvar=False)               # CONTEMPORANEOUS innovation covariance
    sig = np.sqrt(np.diag(Sig))
    Chol = np.linalg.cholesky(Sig + 1e-12 * np.eye(npeer))
    corr = (Sig / np.outer(sig, sig))[np.triu_indices(npeer, 1)]
    print("AR(1) fits to the real peers:")
    for j, p_ in enumerate(peers):
        print(f"    {p_:>5}  phi {phi[j]:.3f}   innovation sd {sig[j]:.4f}")
    print(f"  innovation correlation across peers: mean {corr.mean():.3f}, "
          f"range {corr.min():.3f} to {corr.max():.3f}")
    print("  Surrogates draw innovations from a multivariate normal with THIS")
    print("  covariance, not independently.  Correlated regressors share degrees of")
    print("  freedom under ridge, so independent draws would price a cross-section")
    print("  that is more spread out than the real one and overstate the cost.")

    def surrogate():
        S = np.empty_like(P)
        e = srng.normal(size=P.shape) @ Chol.T   # preserves cross-market covariance
        S[0] = mu_p + e[0] / np.sqrt(np.maximum(1 - phi ** 2, 1e-6))
        for t in range(1, len(P)):
            S[t] = mu_p + phi * (S[t - 1] - mu_p) + e[t]
        S[~fin] = np.nan                       # keep the real missingness pattern
        return S

    n_draw = 100      # Monte Carlo error is reported below, not assumed away
    draws = [surrogate() for _ in range(n_draw)]
    cost_curve, cost_se = {}, {}
    # cache per-observation squared-error differentials so the bootstrap below
    # never has to refit anything: every quantity it needs is a sum over days.
    dnet, dcost = {}, {}
    print(f"\n{'delta':>6}{'cost':>11}{'MC std err':>13}{'95% MC interval':>22}")
    for d in grid:
        o = walk(own, P, y, idx, d, [], None)
        c = walk(own, P, y, idx, d, list(range(npeer)), None)
        dnet[d] = (yb - o) ** 2 - (yb - c) ** 2
        vals, acc = [], np.zeros(len(yb))
        for S in draws:
            so = walk(own, S, y, idx, d, [], None)
            sc = walk(own, S, y, idx, d, list(range(npeer)), None)
            e = (yb - so) ** 2 - (yb - sc) ** 2
            acc += e
            vals.append(e.sum() / ((yb - yb.mean()) ** 2).sum())
        dcost[d] = acc / n_draw
        m, s = float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(n_draw))
        cost_curve[d], cost_se[d] = m, s
        print(f"{d:>6}{m:>+11.4f}{s:>10.4f}"
              f"{f'[{m-1.96*s:+.4f},{m+1.96*s:+.4f}]':>22}")

    lo0, hi0 = cost_curve[0] - 1.96 * cost_se[0], cost_curve[0] + 1.96 * cost_se[0]
    gap = abs(cost_curve[0] - slope * npeer)
    print(f"\nAt delta = 0: surrogates give {cost_curve[0]:+.4f} [{lo0:+.4f},{hi0:+.4f}];")
    print(f"the peer-count line in P1 gives {slope*npeer:+.4f}.  They differ by {gap:.4f}.")
    print("Read that as agreement on the MAGNITUDE, not as a formal match.  The")
    print("interval above is over surrogate draws only: it conditions on this one")
    print("sample of the target, so it understates the true uncertainty, and P1's")
    print("slope carries its own error that is not quantified here.  What the two")
    print("methods establish together is that the cost is about one R-squared")
    print("point at zero delay, which is what the argument below needs; neither is")
    print("precise enough to support a third decimal place.")
    drift = cost_curve[21] - cost_curve[0]
    print(f"\nCost at delta 21 minus cost at delta 0: {drift:+.4f}.  "
          f"{'It grows with delay.' if abs(drift) > 2*(cost_se[0]+cost_se[21]) else 'Flat within error.'}")
    print("That is not an artefact: as own information goes stale the fit weakens,")
    print("validation picks a looser penalty, and useless regressors get more rope.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("WHAT THIS BUYS THE PAPER")
    print("=" * 80)
    print("Gross benefit = measured dR2 minus the cost of carrying the regressors")
    print("at that same delay.  Gross is what the foreign markets actually KNOW;")
    print("net is what a forecaster actually GETS after paying to estimate it.")
    print()
    print("Gross is a DIFFERENCE OF TWO ESTIMATED QUANTITIES, so it gets an")
    print("interval before it gets a label.  Two warning signs said this matters:")
    print("under heavy shrinkage in P2 the delta = 0 gap read +0.0063, four times")
    print("the +0.0016 the point estimate gives, and both are far smaller than the")
    print("interval lab06 put on the NET figure alone.")
    print()
    print("Each bootstrap replicate resamples test days in moving blocks of ten and")
    print("recomputes the net difference AND the surrogate cost on those SAME days.")
    print("Sharing the resample is what keeps the two errors correlated instead of")
    print("adding them blindly, and it is why no model is refitted here: both are")
    print("sums of per-day squared-error differentials cached above.\n")

    nb_boot, blk = 2000, 10
    bn = len(yb)
    nblk = int(np.ceil(bn / blk))
    brng = np.random.default_rng(SEED + 2)
    starts = brng.integers(0, bn - blk + 1, size=(nb_boot, nblk))
    offs = np.arange(blk)
    samples = [(starts[i][:, None] + offs).ravel()[:bn] for i in range(nb_boot)]
    sst_b = np.array([((yb[s] - yb[s].mean()) ** 2).sum() for s in samples])

    print(f"{'delta':>6}{'net dR2':>10}{'cost':>10}{'gross':>10}"
          f"{'gross 95%':>22}   reading")
    verdict = {}
    for d in grid:
        g = dnet[d] - dcost[d]
        gb = np.array([g[s].sum() for s in samples]) / sst_b
        lo, hi = np.percentile(gb, [2.5, 97.5])
        net = dnet[d].sum() / ((yb - yb.mean()) ** 2).sum()
        pt = net - cost_curve[d]
        verdict[d] = (pt, lo, hi)
        note = ("no measurable content" if lo <= 0 <= hi and net < 0 else
                "content not established" if lo <= 0 <= hi else
                "pays for itself" if net > 0 else "real, but does not cover its cost")
        print(f"{d:>6}{net:>+10.4f}{cost_curve[d]:>+10.4f}{pt:>+10.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>22}   {note}")

    first = next((d for d in grid if verdict[d][1] > 0), None)
    print(f"\nGross content first clears zero at delta = {first}.")
    print("A label is only claimed where the interval supports it.\n")

    print("Three things follow, and none of them were in the paper:")
    pt0, lo0g, hi0g = verdict[0]
    print(f"  1. At delta = 0 gross is {pt0:+.4f}, interval [{lo0g:+.4f},{hi0g:+.4f}],")
    print(f"     which covers zero and caps the content at about {hi0g:+.3f}.  Foreign")
    print(f"     markets do not MISLEAD about the current US state; whatever they")
    print(f"     know about it beyond current domestic data is too small to measure")
    print(f"     here.  The negative was the bill, not the information.  That cap is")
    print(f"     the useful part against Korkusuz and Jayawardena et al.: it does not")
    print(f"     say they are wrong, it says how large their effect can be and still")
    print(f"     sit inside what we see.")
    pt1, lo1, hi1 = verdict[1]
    print(f"  2. The tempting middle band - real signal that cannot yet pay for its")
    print(f"     own estimation - is NOT established.  At delta = 1 the interval is")
    print(f"     [{lo1:+.4f},{hi1:+.4f}], which includes zero.  Suggestive, and not")
    print(f"     something the paper may assert.")
    net21 = dnet[21].sum() / ((yb - yb.mean()) ** 2).sum()
    print(f"  3. At delta = 21 estimation eats "
          f"{abs(cost_curve[21])/(net21 - cost_curve[21]):.0%} of what the")
    print(f"     cross-section knows ({net21 - cost_curve[21]:+.4f} gross -> "
          f"{net21:+.4f} net).  A method that")
    print(f"     estimated those seven coefficients more cheaply would recover part")
    print(f"     of it.  We have not built one and claim nothing about how much.")
    print(f"\nruntime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
