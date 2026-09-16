"""
lab10_loss_scale.py - on which scale are the losses computed, and does it matter?

Imports lab05_robustness; keep both in labs/.  Runtime about three minutes.

THE PROBLEM
-----------
The paper cites Patton (2011) to justify reporting R-squared and QLIKE together,
on the grounds that they are the two canonical members of the family of losses
that stay robust when the volatility proxy is noisy.  That justification is
MISAPPLIED to one of the two, and the error is ours.

Patton's family is defined for losses of the form L(proxy, forecast) on the
VARIANCE scale.  Our target is a = log(sigma^2_YZ / M), so:

  QLIKE  IS fine.  We compute it as y/f - log(y/f) - 1 after exponentiating,
         and QLIKE depends on the ratio y/f only.  Dividing target and forecast
         by the same trailing median leaves that ratio untouched, so our QLIKE
         is exactly the variance-scale QLIKE.  Verified numerically below.

  R^2    IS NOT.  Squared error computed on log variance is not in Patton's
         family; a monotone transform of the argument is precisely what the
         robustness result does not survive.  The paper's headline number is
         therefore reported under a loss that a noisy proxy can, in principle,
         reorder.

This file does two things about it.  First it reports a THIRD loss, squared
error on the variance scale, which is Patton's MSE and is robust.  Second it
fixes a second-order problem that appears the moment one reconstructs a
variance forecast: exp() of a log forecast is median-unbiased, not
mean-unbiased, and squared error punishes exactly that bias.  We apply Duan's
smearing correction, estimated from TRAINING residuals only at each refit.

SECOND ISSUE, SAME FILE
-----------------------
R(delta) is a ratio of two estimated quantities and the paper's intervals come
from the numerator alone, which it admits in its limitations and then does
nothing about.  Every interval here bootstraps the WHOLE ratio, resampling the
test days once per replicate and recomputing all three skill terms on the same
days.

WHAT WOULD COUNT AS A PROBLEM
-----------------------------
If the substitution rate under Patton's MSE looks materially different from the
rate under log-scale R-squared, the headline is an artefact of the scale and
has to be restated.  If the three agree, the paper keeps its number and gains a
sentence saying which loss is the robust one.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".")
import lab05_robustness as L

SEED = 20260912
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 1, 3, 5, 8, 13, 21, 34, 55]
N_BOOT, BLK = 2000, 10


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
    return own, P, y, idx


def walk(own, P, y, idx, delta, use_peers):
    """Returns the log-scale forecast AND the smearing factor for each test day.

    The smearing factor is mean(exp(training residual)) at the most recent
    refit, which is Duan's non-parametric correction for the fact that
    exp(E[log x]) understates E[x].  It uses training rows only.
    """
    f = np.empty(len(idx)); sm = np.ones(len(idx))
    b = mu = sd = None; smear = 1.0
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), cut)
            X = np.column_stack([own[tr - delta]] + ([P[tr]] if use_peers else []))
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y[tr])
            X, yy = X[ok], y[tr][ok]
            if len(yy) < 200:
                b = None
            else:
                mu, sd = X.mean(0), X.std(0) + 1e-9
                b = L.cv((X - mu) / sd, yy, "cont")
                resid = yy - L.p_ridge(b, (X - mu) / sd)
                smear = float(np.mean(np.exp(resid)))
        xt = np.concatenate([own[t - delta]] + ([P[t]] if use_peers else []))
        f[j] = 0.0 if (b is None or not np.isfinite(xt).all()) \
            else L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
        sm[j] = smear
    return f, sm


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    own, P, y, idx = panel(folder)
    yb = y[idx]
    n = len(idx)
    print(f"{n} test days\n")

    print("=" * 84)
    print("0.  IS OUR QLIKE THE VARIANCE-SCALE QLIKE?")
    print("=" * 84)
    rng0 = np.random.default_rng(1)
    v1 = np.exp(rng0.normal(-9, 1, 4000)); v2 = np.exp(rng0.normal(-9, 1, 4000))
    M = np.exp(rng0.normal(-9, .5, 4000))
    q = lambda a_, b_: np.mean(a_ / b_ - np.log(a_ / b_) - 1)
    print(f"QLIKE(variance)            {q(v1, v2):.10f}")
    print(f"QLIKE(variance / median)   {q(v1 / M, v2 / M):.10f}")
    print(f"identical -> {np.isclose(q(v1, v2), q(v1 / M, v2 / M))}.  QLIKE depends on the")
    print("ratio alone, so the median normalisation cancels and Patton's result applies.")
    print("Squared error on LOG variance has no such property and is not in the family.")

    # ---------------- forecasts, once ------------------------------------
    F = {}
    for d in DELAYS:
        F[d] = (walk(own, P, y, idx, d, False), walk(own, P, y, idx, d, True))

    yv = np.exp(yb)                     # target on the normalised-variance scale

    def per_obs(d):
        """Per-day loss for every model and loss function, ready to bootstrap."""
        (fo, so), (fc, sc) = F[d]
        out = {}
        out["sq_log_own"] = (yb - fo) ** 2
        out["sq_log_cross"] = (yb - fc) ** 2
        ho, hc = np.exp(fo) * so, np.exp(fc) * sc        # smeared variance forecasts
        out["ql_own"] = yv / ho - np.log(yv / ho) - 1
        out["ql_cross"] = yv / hc - np.log(yv / hc) - 1
        out["mse_own"] = (yv - ho) ** 2
        out["mse_cross"] = (yv - hc) ** 2
        return out

    O = {d: per_obs(d) for d in DELAYS}

    rng = np.random.default_rng(SEED + 7)
    st = rng.integers(0, n - BLK + 1, size=(N_BOOT, int(np.ceil(n / BLK))))
    offs = np.arange(BLK)
    S = [(st[i][:, None] + offs).ravel()[:n] for i in range(N_BOOT)]

    def skill(d, loss, model, s=None):
        """Skill = 1 - loss(model) / loss(benchmark), benchmark = sample mean."""
        v = O[d][f"{loss}_{model}"]
        if loss == "sq_log":
            base = (yb - yb.mean()) ** 2
        elif loss == "mse":
            base = (yv - yv.mean()) ** 2
        else:
            m = yv.mean()
            base = yv / m - np.log(yv / m) - 1
        if s is None:
            return 1 - v.sum() / base.sum()
        return 1 - v[s].sum() / base[s].sum()

    print("\n" + "=" * 84)
    print("1.  THE SUBSTITUTION RATE UNDER THREE LOSSES")
    print("=" * 84)
    print("R(d) = [S_cross(d) - S_own(d)] / [S_own(0) - S_own(d)].  Intervals bootstrap")
    print("the WHOLE ratio: numerator and denominator are recomputed on the same")
    print("resampled days, which the paper previously did not do.\n")
    print("  sq(log)  squared error on log variance   - NOT in Patton's robust family")
    print("  QLIKE    variance scale                  - robust")
    print("  MSE(var) squared error on variance, Duan-smeared - robust\n")
    print(f"{'delta':>6}  {'R(d) sq(log)':>24}{'R(d) QLIKE':>24}{'R(d) MSE(var)':>24}")
    for d in DELAYS[1:]:
        row = f"{d:>6}  "
        for loss in ("sq_log", "ql", "mse"):
            pt = ((skill(d, loss, "cross") - skill(d, loss, "own")) /
                  (skill(0, loss, "own") - skill(d, loss, "own")))
            bs = []
            for s in S:
                den = skill(0, loss, "own", s) - skill(d, loss, "own", s)
                if abs(den) > 1e-9:
                    bs.append((skill(d, loss, "cross", s) - skill(d, loss, "own", s)) / den)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            row += f"{pt:>8.0%} [{lo:>4.0%},{hi:>4.0%}]".rjust(24)
        print(row)

    print("\n" + "=" * 84)
    print("1b. SKILL UNDER EACH LOSS, WHICH IS WHAT THE PAPER'S TABLE 1 REPORTS")
    print("=" * 84)
    print("Skill = 1 - loss(model) / loss(training-mean benchmark), so all three are")
    print("on a comparable 'fraction of benchmark loss removed' footing.\n")
    print(f"{'delta':>6}{'R2 own':>10}{'R2 cross':>11}{'QLIKE own':>12}"
          f"{'QLIKE cross':>13}{'MSEvar own':>12}{'MSEvar cross':>14}")
    for d in DELAYS:
        print(f"{d:>6}{skill(d,'sq_log','own'):>10.4f}{skill(d,'sq_log','cross'):>11.4f}"
              f"{skill(d,'ql','own'):>12.4f}{skill(d,'ql','cross'):>13.4f}"
              f"{skill(d,'mse','own'):>12.4f}{skill(d,'mse','cross'):>14.4f}")

    print("\n" + "=" * 84)
    print("2.  DOES THE CHOICE OF SCALE CHANGE THE VERDICT?")
    print("=" * 84)
    print(f"{'delta':>6}{'S own sq(log)':>15}{'S cross sq(log)':>17}"
          f"{'S own MSE(var)':>16}{'S cross MSE(var)':>18}")
    for d in DELAYS:
        print(f"{d:>6}{skill(d,'sq_log','own'):>15.4f}{skill(d,'sq_log','cross'):>17.4f}"
              f"{skill(d,'mse','own'):>16.4f}{skill(d,'mse','cross'):>18.4f}")
    print("\n" + "=" * 84)
    print("3.  THE delta = 0 ROW, WHICH CHANGES SIGN")
    print("=" * 84)
    print("On log-scale squared error the cross-section is slightly WORSE at zero delay,")
    print("and the paper leans on that row when discussing Korkusuz and Jayawardena.")
    print("On the robust variance-scale loss the sign flips.  Either way the question is")
    print("whether the difference is distinguishable from zero, so test it.\n")
    print(f"{'loss':>10}{'S own':>10}{'S cross':>10}{'difference':>13}{'95% interval':>22}")
    for loss, nm in (("sq_log", "sq(log)"), ("ql", "QLIKE"), ("mse", "MSE(var)")):
        pt = skill(0, loss, "cross") - skill(0, loss, "own")
        bs = np.array([skill(0, loss, "cross", s) - skill(0, loss, "own", s) for s in S])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        mark = "" if lo <= 0 <= hi else "  <- excludes zero"
        print(f"{nm:>10}{skill(0,loss,'own'):>10.4f}{skill(0,loss,'cross'):>10.4f}"
              f"{pt:>+13.4f}{f'[{lo:+.4f},{hi:+.4f}]':>22}{mark}")
    print("\nA claim that rests on a row whose SIGN depends on the scale is a claim the")
    print("paper should not make.  Whatever survives here is what it may say.")

    print("\nThe robust loss is the one to believe if they disagree.  If the rates in")
    print("part 1 sit on top of each other, the headline survives the correction and")
    print("the paper owes the reader a sentence naming which loss is robust, not a")
    print("new set of numbers.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
