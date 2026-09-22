"""
lab10_loss_scale.py - on which scale are the losses computed, and does it matter?

Imports lab05_robustness; keep both in labs/.  Runtime about twenty seconds.

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
test days once per replicate and recomputing every skill term on the same
days.

WHAT WOULD COUNT AS A PROBLEM
-----------------------------
If the substitution rate under Patton's MSE looks materially different from the
rate under log-scale R-squared, the headline is an artefact of the scale and
has to be restated.  If they agree, the paper keeps its number and gains a
sentence saying which loss is the robust one.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".")
import lab05_robustness as L

BENCH = None          # set in main(); read by the scorers below

SEED = 20260912
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 1, 3, 5, 8, 13, 21, 34, 55]
N_BOOT, BLK = 2000, L.BLOCK


def panel(folder):
    D, peers, lag = L.build(folder, TARGET)
    # The trailing median the decision variable divides by.  build() returns
    # only the normalised series, so it is recomputed here from the same
    # estimator and the same window rather than approximated: M_t is what turns
    # a normalised variance back into a variance, and Part 4 needs it.
    _yz = L.yang_zhang(L.load_index(TARGET, folder))
    _M = _yz.rolling(L.MED, min_periods=30).median()
    Mser = _M.reindex(D.index).values
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
    return own, P, y, idx, Mser


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
    own, P, y, idx, Mser = panel(folder)
    yb = y[idx]
    global BENCH
    BENCH = L.bench_mean(y, idx)
    n = len(idx)
    print(f"{n} test days\n")

    print("=" * 84)
    print("0.  IS OUR QLIKE THE VARIANCE-SCALE QLIKE?")
    print("=" * 84)
    # NB: Part 0 uses M for a synthetic median.  The panel's real median is
    # Mser, deliberately named differently: the first version of this file
    # called both M and Part 0 silently shadowed the real one.
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
    BVAR = L.bench_var(y, idx)          # the same benchmark, variance scale
    M0 = Mser[idx]                      # the median at the forecast ORIGIN
    Mh = Mser[idx + H]                  # the median at the OUTCOME date

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
        # The SAME squared error on the natural variance scale, which is what
        # Patton's result is actually about.  Two things change and both are
        # forced by the definitions rather than chosen:
        #
        #   the outcome   y_t = log(sigma^2_{t+h} / M_{t+h}), so realised
        #                 variance is exp(y_t) * M_{t+h} and needs the median
        #                 at t+h, not at t
        #   the forecast  a forecaster at t does not know M_{t+h}, so the only
        #                 implementable reconstruction multiplies by M_t and
        #                 carries the median's h-day drift as forecast error
        #
        # Averaging (v - w)^2 over these is Patton's MSE.  Averaging it over the
        # normalised pair above is the same loss weighted by 1/M^2, which is not
        # the same thing and is not covered by his robustness result.
        vn = yv * Mh
        out["msen_own"] = (vn - ho * M0) ** 2
        out["msen_cross"] = (vn - hc * M0) ** 2
        # A third column that no forecaster could compute, and that is the point
        # of it.  Going from the normalised loss to the implementable natural one
        # changes TWO things at once - the 1/M^2 weight comes off, and the
        # forecast's median moves from M_{t+h} to M_t - so the gap between them
        # is not a measurement of the weight.  This column changes only the
        # first: it reconstructs BOTH sides with M_{t+h}, so per day it is
        # exactly M^2_{t+h} times the normalised loss and the weight is the only
        # thing separating it from the normalised column.  It is infeasible
        # because M_{t+h} is not known at t, so it is a decomposition device and
        # never a reported forecast.
        out["msei_own"] = (vn - ho * Mh) ** 2
        out["msei_cross"] = (vn - hc * Mh) ** 2
        return out

    O = {d: per_obs(d) for d in DELAYS}

    rng = np.random.default_rng(SEED + 7)
    st = rng.integers(0, n - BLK + 1, size=(N_BOOT, int(np.ceil(n / BLK))))
    offs = np.arange(BLK)
    S = [(st[i][:, None] + offs).ravel()[:n] for i in range(N_BOOT)]

    def skill(d, loss, model, s=None):
        """Skill = 1 - loss(model) / loss(benchmark).

        The benchmark is the trailing mean of resolved labels on every loss,
        carried onto the loss's own scale.  Two of these branches divided by the
        mean of the test period instead, so the log-scale column was scored
        against one yardstick and the two variance-scale columns against
        another - and Section 4 reads the three side by side as shares of a
        common benchmark loss.
        """
        v = O[d][f"{loss}_{model}"]
        # BENCH is the benchmark's forecast of the LOG target; BVAR is the same
        # forecaster carried onto the NORMALISED variance scale with Duan's
        # correction, the way every model forecast here is.  Using exp(BENCH)
        # unsmeared would deny the benchmark a correction the models get and
        # overstate variance-scale skill.
        #
        # The natural-scale branches multiply BVAR by the SAME median the model
        # they are scoring is given, and an audit caught this file using M_h for
        # the benchmark on both of them.  msen is the implementable column: the
        # model only has M_0 at the origin and carries the median's h-day drift
        # as forecast error, so a benchmark handed M_h is handed a quantity the
        # model is denied, which flatters the benchmark and understates skill.
        # msei is the infeasible decomposition column, where BOTH sides are
        # reconstructed with M_h on purpose, so M_h is right there.
        if loss == "sq_log":
            base = (yb - BENCH) ** 2
        elif loss == "mse":
            base = (yv - BVAR) ** 2
        elif loss == "msen":
            base = (yv * Mh - BVAR * M0) ** 2
        elif loss == "msei":
            base = (yv * Mh - BVAR * Mh) ** 2
        else:
            base = yv / BVAR - np.log(yv / BVAR) - 1
        if s is None:
            return 1 - v.sum() / base.sum()
        return 1 - v[s].sum() / base[s].sum()

    print("\n" + "=" * 84)
    print("1.  THE SUBSTITUTION RATE UNDER FOUR LOSSES")
    print("=" * 84)
    print("R(d) = [S_cross(d) - S_own(d)] / [S_own(0) - S_own(d)].  Intervals bootstrap")
    print("the WHOLE ratio: numerator and denominator are recomputed on the same")
    print("resampled days, which the paper previously did not do.\n")
    print("  sq(log)  squared error on log variance        - NOT in Patton's family")
    print("  QLIKE    variance scale; the median cancels   - robust, proved in Part 0")
    print("  MSE(nrm) squared error on NORMALISED variance - Patton's MSE weighted")
    print("           by 1/M^2, which his result does not cover")
    print("  MSE(nat) squared error on NATURAL variance    - Patton's MSE itself\n")
    print(f"{'delta':>6}  {'R(d) sq(log)':>22}{'R(d) QLIKE':>22}"
          f"{'R(d) MSE(nrm)':>22}{'R(d) MSE(nat)':>22}")
    for d in DELAYS[1:]:
        row = f"{d:>6}  "
        for loss in ("sq_log", "ql", "mse", "msen"):
            pt = ((skill(d, loss, "cross") - skill(d, loss, "own")) /
                  (skill(0, loss, "own") - skill(d, loss, "own")))
            bs = []
            for s in S:
                den = skill(0, loss, "own", s) - skill(d, loss, "own", s)
                if abs(den) > 1e-9:
                    bs.append((skill(d, loss, "cross", s) - skill(d, loss, "own", s)) / den)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            row += f"{pt:>7.0%} [{lo:>4.0%},{hi:>4.0%}]".rjust(22)
        print(row)

    print("\n" + "=" * 84)
    print("1b. SKILL UNDER EACH LOSS, WHICH IS WHAT THE PAPER'S TABLE 2 REPORTS")
    print("=" * 84)
    print("Skill = 1 - loss(model) / loss(training-mean benchmark), so all of them are")
    print("on a comparable 'fraction of benchmark loss removed' footing.\n")
    print("MSE below is the NATURAL-scale one, which is the loss the paper claims.\n")
    print(f"{'delta':>6}{'R2 own':>10}{'R2 cross':>11}{'QLIKE own':>12}"
          f"{'QLIKE cross':>13}{'MSEnat own':>12}{'MSEnat cross':>14}")
    for d in DELAYS:
        print(f"{d:>6}{skill(d,'sq_log','own'):>10.4f}{skill(d,'sq_log','cross'):>11.4f}"
              f"{skill(d,'ql','own'):>12.4f}{skill(d,'ql','cross'):>13.4f}"
              f"{skill(d,'msen','own'):>12.4f}{skill(d,'msen','cross'):>14.4f}")

    print("\n" + "=" * 84)
    print("2.  DOES THE CHOICE OF SCALE CHANGE THE VERDICT?")
    print("=" * 84)
    print(f"{'delta':>6}{'S own sq(log)':>15}{'S cross sq(log)':>17}"
          f"{'S own MSE(nat)':>16}{'S cross MSE(nat)':>18}")
    for d in DELAYS:
        print(f"{d:>6}{skill(d,'sq_log','own'):>15.4f}{skill(d,'sq_log','cross'):>17.4f}"
              f"{skill(d,'msen','own'):>16.4f}{skill(d,'msen','cross'):>18.4f}")
    print("\n" + "=" * 84)
    print("3.  THE delta = 0 ROW, WHICH CHANGES SIGN")
    print("=" * 84)
    print("On log-scale squared error the cross-section is slightly WORSE at zero delay,")
    print("and the paper leans on that row when discussing Korkusuz and Jayawardena.")
    print("On the robust variance-scale loss the sign flips.  Either way the question is")
    print("whether the difference is distinguishable from zero, so test it.\n")
    print(f"{'loss':>10}{'S own':>10}{'S cross':>10}{'difference':>13}{'95% interval':>22}")
    for loss, nm in (("sq_log", "sq(log)"), ("ql", "QLIKE"),
                     ("msen", "MSE(nat)"), ("mse", "MSE(nrm)")):
        pt = skill(0, loss, "cross") - skill(0, loss, "own")
        bs = np.array([skill(0, loss, "cross", s) - skill(0, loss, "own", s) for s in S])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        mark = "" if lo <= 0 <= hi else "  <- excludes zero"
        print(f"{nm:>10}{skill(0,loss,'own'):>10.4f}{skill(0,loss,'cross'):>10.4f}"
              f"{pt:>+13.4f}{f'[{lo:+.4f},{hi:+.4f}]':>22}{mark}")
    print("\nA claim that rests on a row whose SIGN depends on the scale is a claim the")
    print("paper should not make.  Whatever survives here is what it may say.")

    print("\n" + "=" * 84)
    print("4.  WHAT THE NORMALISATION WAS DOING TO PATTON'S MSE")
    print("=" * 84)
    print("This part exists because the paper described its MSE column as Patton's")
    print("MSE on the variance scale and computed it on the MEDIAN-NORMALISED")
    print("variance scale.  Two things separate those two columns, not one, and")
    print("reporting a single gap would credit the whole of it to the weight:")
    print()
    print("  MSE(nrm)   (exp y - g exp f)^2                 no median on either side")
    print("  MSE(inf)   (exp y * Mh - g exp f * Mh)^2       BOTH sides at t+h.  Per")
    print("             day this is exactly Mh^2 * MSE(nrm), so what separates it")
    print("             from the column above is the 1/M^2 weight ALONE.  It is")
    print("             infeasible: M_{t+h} is not known at t.")
    print("  MSE(nat)   (exp y * Mh - g exp f * M0)^2       the implementable one,")
    print("             which the paper reports.  What separates it from MSE(inf)")
    print("             is the median's h-day drift ALONE.")
    print()
    print("So gap 1 prices the weighting and gap 2 prices the drift, and the sum of")
    print("the two is the whole of the earlier mislabelling.\n")
    # The identity the middle column rests on, checked rather than asserted: if
    # this ever stopped holding, the decomposition below would be arithmetic
    # about nothing.
    _d = DELAYS[-1]
    _lhs, _rhs = O[_d]["msei_cross"], (Mh ** 2) * O[_d]["mse_cross"]
    _err = np.max(np.abs(_lhs - _rhs) / np.maximum(_rhs, 1e-300))
    print(f"  identity check at delta = {_d}: max |MSE(inf) - M^2 MSE(nrm)| / "
          f"M^2 MSE(nrm) = {_err:.2e}")
    print(f"  -> the middle column IS the normalised loss reweighted: "
          f"{np.allclose(_lhs, _rhs)}\n")
    print(f"{'delta':>6}{'MSE(nrm)':>11}{'MSE(inf)':>11}{'MSE(nat)':>11}"
          f"{'weight':>10}{'drift':>10}{'total':>10}")
    worst = worst_w = worst_d = 0.0
    def rate(d, loss):
        return ((skill(d, loss, "cross") - skill(d, loss, "own")) /
                (skill(0, loss, "own") - skill(d, loss, "own")))
    for d in DELAYS[1:]:
        r_n, r_i, r_t = rate(d, "mse"), rate(d, "msei"), rate(d, "msen")
        worst = max(worst, abs(r_n - r_t))
        worst_w = max(worst_w, abs(r_n - r_i))
        worst_d = max(worst_d, abs(r_i - r_t))
        print(f"{d:>6}{r_n:>11.1%}{r_i:>11.1%}{r_t:>11.1%}"
              f"{r_n - r_i:>+10.1%}{r_i - r_t:>+10.1%}{r_n - r_t:>+10.1%}")
    print(f"\n  largest total gap: {worst:.1%} of the rate;"
          f" largest weighting piece {worst_w:.1%},"
          f" largest drift piece {worst_d:.1%}.")
    print("  The paper reports MSE(nat), which is the one Patton's result covers and")
    print("  the only one of the three a forecaster could have computed.  The other")
    print("  two are kept so the size of the earlier mislabelling is on the record")
    print("  as two named effects rather than as one undifferentiated number.")

    # ---------------- 5. the benchmark on the variance scale ---------------
    print("\n" + "=" * 84)
    print("5.  WHICH BENCHMARK BELONGS ON THE VARIANCE SCALE")
    print("=" * 84)
    print("The benchmark forecasts the NORMALISED log target, y = log(var / M),")
    print("and is carried onto the variance scale exactly as the models are:")
    print("exponentiate, apply Duan's smearing factor, multiply by the median the")
    print("forecaster holds.  For a constant log forecast the smeared value is")
    print("mean_u exp(y_u), and an earlier version of this project called that")
    print("'the trailing mean of the variance'.  It is not.  It is the trailing")
    print("mean of the variance OVER ITS OWN MEDIAN, and the two are different")
    print("objects.  This part measures how different, and what turns on it.\n")

    nat_all = np.exp(y) * np.roll(Mser, -H)          # sigma^2_{u+h}, the natural one
    def bench_nat(ix, window=L.TRAIN + L.VAL):
        out = np.empty(len(ix))
        for j, t in enumerate(ix):
            cut = t - H
            w = nat_all[max(0, cut - window):cut]
            w = w[np.isfinite(w)]
            out[j] = w.mean() if len(w) else 1.0
        return out
    BNAT = bench_nat(idx)
    _recon = BVAR * M0                               # the reconstructed benchmark
    _rel = float(np.nanmean(np.abs(_recon - BNAT)) / np.nanmean(BNAT))
    _cor = float(np.corrcoef(_recon[np.isfinite(_recon) & np.isfinite(BNAT)],
                             BNAT[np.isfinite(_recon) & np.isfinite(BNAT)])[0, 1])
    print(f"  mean |M_t mean_u exp(y_u)  -  mean_u sigma^2_{{u+h}}| as a share of the")
    print(f"  mean natural variance: {_rel:.1%}; correlation between them: {_cor:.2f}")
    print("  So the collapse an earlier draft claimed does not hold, and the")
    print("  reconstructed benchmark has to be described as what it is.\n")

    print("  Both constructions are scored below on the implementable natural loss.")
    print("  The reconstructed one is what this file reports, because it is the")
    print("  same forecaster the log and QLIKE columns use and keeps Section 4's")
    print("  claim that the three columns share one benchmark.  The direct one is")
    print("  a different forecaster and is shown so the choice is visible.\n")
    print(f"{'delta':>6}{'S_own recon':>13}{'S_own direct':>14}"
          f"{'S_cross recon':>15}{'S_cross direct':>16}"
          f"{'R recon':>10}{'R direct':>10}")
    _vn = yv * Mh
    _b_re, _b_di = ((_vn - BVAR * M0) ** 2).sum(), ((_vn - BNAT) ** 2).sum()
    def _sk(d, cross, base):
        (fo, so), (fc, sc) = F[d]
        w = (np.exp(fc) * sc) if cross else (np.exp(fo) * so)
        return 1 - ((_vn - w * M0) ** 2).sum() / base
    for d in DELAYS[1:]:
        r = lambda base: ((_sk(d, True, base) - _sk(d, False, base)) /
                          (_sk(0, False, base) - _sk(d, False, base)))
        print(f"{d:>6}{_sk(d, False, _b_re):>13.4f}{_sk(d, False, _b_di):>14.4f}"
              f"{_sk(d, True, _b_re):>15.4f}{_sk(d, True, _b_di):>16.4f}"
              f"{r(_b_re):>10.1%}{r(_b_di):>10.1%}")
    _gap = max(abs(((_sk(d, True, _b_re) - _sk(d, False, _b_re)) /
                    (_sk(0, False, _b_re) - _sk(d, False, _b_re))) -
                   ((_sk(d, True, _b_di) - _sk(d, False, _b_di)) /
                    (_sk(0, False, _b_di) - _sk(d, False, _b_di))))
               for d in DELAYS[1:])
    print(f"\n  largest difference the choice makes to R(delta): {_gap:.2%}")
    print("  The levels move and the rate does not, for the reason part 1 of")
    print("  lab58 proves: the benchmark cancels from a ratio of skill")
    print("  differences. So the correction is a correction to how this paper")
    print("  describes and reports SKILL, and not to anything it concludes.")

    print("\nThe robust loss is the one to believe if they disagree.  If the rates in")
    print("part 1 sit on top of each other, the headline survives the correction and")
    print("the paper owes the reader a sentence naming which loss is robust, not a")
    print("new set of numbers.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
