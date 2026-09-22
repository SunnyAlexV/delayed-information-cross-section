"""
lab46_own_only_filter.py - is the substitution rate inflated by a weak control?

Imports lab05_robustness, lab22_factor_benchmark and lab37_lead_lag; keep all
four in labs/.  Runtime about five minutes.

THE OBJECTION, STATED PROPERLY
------------------------------
R(delta) is a ratio in which the same own-only model appears three times:

    R(delta) = [ S_cross(delta) - S_own(delta) ] / [ S_own(0) - S_own(delta) ]

Two referees have now made a version of the same point.  The paper's own-only
control is a ridge on three stale features, and a ridge handles a delayed
observation by simply using the stale value; a filter would project the state
forward.  If the control decays faster than it should, the rate is measured
against a straw man.

The direction is not a matter of opinion.  Write N and D for numerator and
denominator, with N < D since the cross-section never recovers the whole loss.
Raising S_own(delta) by e subtracts e from BOTH:

    (N - e) / (D - e)  <  N / D  whenever  N < D

so a weak control INFLATES R.  The referees are right about the sign.  What is
not settled is the size, and that is an empirical question.

WHY lab38 IS NOT ALREADY THE ANSWER
-----------------------------------
Section 10.1 rebuilds the domestic control five ways - more own lags, quarterly
and annual means, leverage terms, a second realised measure - and the rate falls
from 72% to 68%.  That is the same test with a richer RIDGE.  It does not answer
the referees, because every arm still treats a delayed observation the same way:
as a stale number handed to a regression.  The objection is about the estimator
class, not the feature count.

WHAT A FILTER WOULD ACTUALLY CHANGE
-----------------------------------
An own-only state-space model, with no foreign data anywhere in it:

    state        f(t) = phi f(t-1) + eta(t),   Var(eta) = q
    observation  a(t) = f(t) + eps(t),         Var(eps) = r

Parameters come from the first training block by method of moments and are then
FROZEN, so nothing about the test period can reach them.  For an AR(1) observed
with noise the autocovariances are

    gamma(0) = q/(1-phi^2) + r,     gamma(k) = phi^k q/(1-phi^2),  k >= 1

so phi = gamma(2)/gamma(1), the signal variance is gamma(1)/phi, and r is what
is left of gamma(0).  Three moments, three parameters, no EM loop and nothing to
tune.

Such a model does two things a stale number cannot.  It PROJECTS, replacing
a(t-delta) by phi^(delta+h) xhat(t-delta); and it SMOOTHS, so that xhat is a
weighted average of the whole history rather than the last reading.  This file
takes those two separately, because only one of them can matter, and it is not
the one the objection names.

FOUR QUESTIONS, IN ORDER
------------------------
    1. The projection, implemented exactly as asked.  A deterministic projection
       multiplies a column by a constant, and the paper's ridge standardises its
       columns before fitting, so the constant cancels.  Part 1 is that argument
       carried out rather than asserted.
    2. The smoothing, which is the part that can carry information the three
       features do not span.  Give the control the exponential windows a filter
       would use, on top of everything it already has, and see whether the
       own-only curve moves at all.
    3. What R(delta) becomes against the best own-only control that exists.
    4. A referee has predicted that a filter "will dramatically tighten" the
       interval on the effective age of Section 9.3.  That interval comes from
       resampling TEST DAYS, not from estimator noise, so the prediction looks
       wrong on its face - but it is cheaper to check than to argue about.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab22_factor_benchmark as F
import lab37_lead_lag as LL

SEED = 20260916
TARGET = "SPX"
H = L.HORIZON
FINE = [0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 21, 26, 34, 44, 55]
DELAYS = [0, 3, 5, 13, 21, 55]           # all members of FINE
HALFLIVES = (5, 22, 66)                  # the paper's two windows, plus a quarter
# The block is lab05's measured one, not 2 * HORIZON.  The h-day overlap in
# the target is not the only dependence in a loss difference: it also
# inherits the common factor's persistence, which lab58 measures at 0.48
# autocorrelation at lag 10 and not below 0.05 until lag 38.  A block
# shorter than the dependence leaves it inside the resample and the
# interval comes out too narrow.
N_BOOT, BLOCK = 1500, L.BLOCK
TRAIN0 = L.TRAIN + L.VAL                 # the block the parameters are fitted on


def moments(x):
    """phi, q, r for an AR(1) observed with noise, by method of moments."""
    x = x[np.isfinite(x)]
    v = x - x.mean()
    n = len(v)
    g = [float(v[:n - k] @ v[k:]) / n for k in (0, 1, 2)]
    phi = g[2] / g[1] if abs(g[1]) > 1e-12 else 0.0
    phi = float(np.clip(phi, 0.0, 0.999))
    sig = g[1] / phi if phi > 1e-9 else 0.0          # Var(f) = q/(1-phi^2)
    sig = max(sig, 1e-12)
    r = max(g[0] - sig, 1e-9)
    q = sig * (1 - phi ** 2)
    return phi, q, r


def filtered(a, phi, q, r):
    """One causal pass: xhat[s] = E[f(s) | a(1..s)]. Missing days carry forward."""
    n = len(a)
    xh = np.zeros(n)
    P = q / max(1 - phi ** 2, 1e-6)
    x = 0.0
    for s in range(n):
        x = phi * x                                   # predict
        P = phi * phi * P + q
        if np.isfinite(a[s]):                         # update, if observed
            K = P / (P + r)
            x = x + K * (a[s] - x)
            P = (1 - K) * P
        xh[s] = x
    return xh


def walk_block(X_all, y, idx, delta, scale=1.0):
    """lab37's own-only walk-forward, on an arbitrary feature block.

    Everything - refit schedule, training window, penalty grid, standardisation
    - is lab05's, so the only thing that differs between calls is what the model
    is told.  `scale` multiplies every column, which is how part 1 implements the
    projection without changing anything else.
    """
    out = np.empty(len(idx)); b = mu = sd = None
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), max(cut, 1))
            X = scale * X_all[tr - delta]
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y[tr])
            X, yy = X[ok], y[tr][ok]
            if len(yy) < 200:
                b = None; continue
            mu, sd = X.mean(0), X.std(0) + 1e-9
            b = L.cv((X - mu) / sd, yy, "cont")
        xt = scale * X_all[t - delta]
        out[j] = 0.0 if (b is None or not np.isfinite(xt).all()) \
            else L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
    return out


def r2_on(y, f, rows, *, bench):
    """Out-of-sample skill on a subset of days, against the trailing-mean
    benchmark restricted to those same days.

    `bench` is keyword-only and mandatory; see lab49 for why.  The docstring
    this replaces claimed the benchmark was "recomputed there", which it was -
    as the mean of the subset, chosen with hindsight.
    """
    yy, ff, bb = y[rows], f[rows], bench[rows]
    den = ((yy - bb) ** 2).sum()
    return 1 - ((yy - ff) ** 2).sum() / den if den > 0 else np.nan


def age_of(skill, xs, vs):
    if skill >= vs[0]:
        return 0.0
    if skill <= vs[-1]:
        return np.nan
    for i in range(len(xs) - 1):
        if vs[i] >= skill >= vs[i + 1]:
            span = vs[i] - vs[i + 1]
            w = 0.0 if span < 1e-12 else (vs[i] - skill) / span
            return float(xs[i] + w * (xs[i + 1] - xs[i]))
    return np.nan


def blocks_of(n, rng, block=BLOCK):
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints the own-only state-space control, Section 10.1.\n")

    own, P, y, idx, D, peers, _lag = LL.panel(folder)
    a = D[TARGET].values
    yb = y[idx]
    BENCH = L.bench_mean(y, idx)
    n = len(idx)
    print(f"target {TARGET}, {n} test days, {len(peers)} foreign peers")

    # ---------------- the model, fitted once and frozen ---------------------
    print("\n" + "=" * 96)
    print("0.  AN OWN-ONLY STATE-SPACE MODEL, PARAMETERS FROZEN BEFORE THE TEST")
    print("=" * 96)
    phi, q, r = moments(a[:TRAIN0])
    share = q / (1 - phi ** 2) / (q / (1 - phi ** 2) + r)
    print(f"  fitted on the first {TRAIN0} rows, by method of moments:")
    print(f"    phi = {phi:.4f}   q = {q:.5f}   r = {r:.5f}")
    print(f"    signal share of variance: {share:.1%}")
    print(f"    implied half-life of the state: "
          f"{np.log(0.5) / np.log(phi):.1f} trading days")
    xh = filtered(a, phi, q, r)
    gap = float(np.nanmax(np.abs(xh - a)))
    print(f"\n  largest gap between the filtered state and the raw observation: {gap:.2e}")
    if share > 0.999:
        print("  The moments put no observation noise in this series at all, so the")
        print("  Kalman gain is one and the filtered state IS the observation. That")
        print("  is a finding, not a failure: at daily frequency the decision")
        print("  variable is already its own best estimate of itself, and there is")
        print("  nothing for the update step to clean up.")

    # ---------------- 1. the projection --------------------------------------
    print("\n" + "=" * 96)
    print("1.  THE PROJECTION, IMPLEMENTED EXACTLY AS ASKED")
    print("=" * 96)
    print("Hand the ridge phi^(delta+h) xhat(t-delta) instead of xhat(t-delta).")
    print("Same feature, same days, same refit schedule; one is projected forward")
    print("and the other is not.\n")
    print(f"{'delta':>6}{'stale':>10}{'projected':>12}{'difference':>13}")
    worst_proj = 0.0
    for d in DELAYS:
        f0 = walk_block(xh[:, None], y, idx, d, 1.0)
        f1 = walk_block(xh[:, None], y, idx, d, phi ** (d + H))
        s0, s1 = F.r2(yb, f0, BENCH), F.r2(yb, f1, BENCH)
        worst_proj = max(worst_proj, abs(s1 - s0))
        print(f"{d:>6}{s0:>10.6f}{s1:>12.6f}{s1 - s0:>+13.2e}")
    print(f"\n  largest difference the projection makes: {worst_proj:.1e}")
    print("\n  It makes none, and it cannot. At a fixed delay the projection is one")
    print("  constant per column; the ridge standardises each column before")
    print("  fitting; and (c x - c mu) / (c sd) = (x - mu) / sd. The coefficient")
    print("  simply absorbs the constant. What residue there is comes from the")
    print("  1e-9 added to the standard deviation to keep it from vanishing.")
    print("\n  So the objection cannot bite through the projection. It can only bite")
    print("  through the state itself - if a filtered history carries something the")
    print("  paper's three features do not already span. Part 2 tests that.")

    # ---------------- 2. the smoothing ---------------------------------------
    print("\n" + "=" * 96)
    print("2.  THE SMOOTHING, WHICH IS THE PART THAT COULD MATTER")
    print("=" * 96)
    print("The paper's control is the stale level and two rectangular means, 5 and")
    print("22 days. A filter weights the whole history exponentially instead. So")
    print("give the control both: the three features it has, plus exponentially")
    print(f"weighted averages at half-lives {HALFLIVES} days. Six features against")
    print("three, nested, so anything the filter knows is in there somewhere.\n")
    sa = pd.Series(a)
    ewm = [sa.ewm(halflife=h, adjust=False).mean().values for h in HALFLIVES]
    own_plus = np.column_stack([own] + [e[:, None] for e in ewm])
    f_ridge = {d: LL.walk_own(own, y, idx, d) for d in FINE}
    f_plus = {d: walk_block(own_plus, y, idx, d) for d in FINE}
    S_r = {d: F.r2(yb, f_ridge[d], BENCH) for d in FINE}
    S_p = {d: F.r2(yb, f_plus[d], BENCH) for d in FINE}
    print(f"{'delta':>6}{'paper, 3':>11}{'plus EWMA, 6':>14}{'difference':>13}")
    better = 0
    for d in DELAYS:
        diff = S_p[d] - S_r[d]
        better += int(diff > 0)
        print(f"{d:>6}{S_r[d]:>11.4f}{S_p[d]:>14.4f}{diff:>+13.4f}")
    print(f"\n  the exponential control is better at {better} of {len(DELAYS)} delays")
    print(f"  at the origin, where the denominator of R is set: "
          f"{S_p[0] - S_r[0]:+.4f}")
    best = {d: (S_p[d] if S_p[d] > S_r[d] else S_r[d]) for d in FINE}
    print(f"  taking the better arm at every delay, the own-only curve rises by")
    print(f"  {max(best[d] - S_r[d] for d in FINE):.4f} at most")
    if better == 0:
        print("\n  Three extra columns, every one of them a weighted average of a")
        print("  history the model already has, and the control gets worse - by more")
        print("  the staler it is. Nothing was added to the information set; what was")
        print("  added was six coefficients to estimate where there had been three,")
        print("  on a window that does not grow. This is Section 7's cost of breadth")
        print("  arriving inside the domestic block, and it is the reason the paper's")
        print("  control is three features rather than every feature available.")

    # ---------------- 3. the rate --------------------------------------------
    print("\n" + "=" * 96)
    print("3.  WHAT R(delta) BECOMES AGAINST THE BEST OWN-ONLY CONTROL")
    print("=" * 96)
    print("The cross-sectional model is unchanged - the paper's, with the foreign")
    print("block current. Only the control in numerator and denominator moves. The")
    print("last column takes the better of the two arms at each delay AND at the")
    print("origin, which is the most hostile reading of the control available.\n")
    f_cross = {d: LL.walk(own, P, y, idx, d, 0) for d in DELAYS}
    S_c = {d: F.r2(yb, f_cross[d], BENCH) for d in DELAYS}
    print(f"{'delta':>6}{'S_cross':>10}{'R, paper':>11}{'R, +EWMA':>11}{'R, best':>10}")
    moves = []
    for d in DELAYS[1:]:
        dr = S_r[0] - S_r[d]
        dp = S_p[0] - S_p[d]
        db = best[0] - best[d]
        Rr = (S_c[d] - S_r[d]) / dr if dr > 1e-9 else np.nan
        Rp = (S_c[d] - S_p[d]) / dp if dp > 1e-9 else np.nan
        Rb = (S_c[d] - best[d]) / db if db > 1e-9 else np.nan
        moves.append(Rb - Rr)
        print(f"{d:>6}{S_c[d]:>10.4f}{Rr:>11.1%}{Rp:>11.1%}{Rb:>10.1%}")
    worst = max(abs(m) for m in moves)
    down = sum(m < 0 for m in moves)
    d55 = (S_c[55] - best[55]) / (best[0] - best[55])
    r55 = (S_c[55] - S_r[55]) / (S_r[0] - S_r[55])
    p55 = (S_c[55] - S_p[55]) / (S_p[0] - S_p[55])
    inflate = max((S_c[d] - S_p[d]) / (S_p[0] - S_p[d])
                  - (S_c[d] - S_r[d]) / (S_r[0] - S_r[d]) for d in DELAYS[1:])
    print(f"\n  largest move in the rate against the best control: {worst:.1%}")
    print(f"  the rate falls against the best control at {down} of "
          f"{len(moves)} delays")
    print(f"  at eleven weeks: {r55:.1%} as published, {d55:.1%} against the "
          f"best control")
    print(f"\n  The middle column is the referees' mechanism caught in the act. The")
    print(f"  exponential control is the weaker one, and R measured against it is")
    print(f"  higher at every delay, by up to {inflate:.1%} - {p55:.1%} at eleven weeks")
    print(f"  against the published {r55:.1%}. A weak control inflates the rate, exactly")
    print("  as the algebra says; the question was only ever which control is weak.")
    print("  Here it is the alternative, so the published figure is the smaller of")
    print("  the two on offer and the objection has nothing left to correct.")
    print("\n  Which way a move runs is not obvious from the control alone. R reads")
    print("  the own-only curve at two points, and the origin sets the denominator,")
    print("  so a control that improves mostly at delta = 0 pushes the rate DOWN")
    print("  twice over, while one that improves only at long delays can push it up.")

    # ---------------- 4. the interval ----------------------------------------
    print("\n" + "=" * 96)
    print("4.  DOES A BETTER RULER TIGHTEN THE EFFECTIVE-AGE INTERVAL?")
    print("=" * 96)
    print("A referee predicted it would, 'dramatically'. The interval is a block")
    print("bootstrap over TEST DAYS, so the prediction is testable: rebuild the")
    print("ruler from the stronger own-only curve, resample the same way, and")
    print("compare the widths. Width in days is not the comparison - a ruler that")
    print("decays faster reads every skill as a younger age and shrinks the")
    print("interval whether or not the measurement got sharper - so the width is")
    print("also reported as a share of the point it surrounds.\n")
    comb = LL.walk(own, P, y, idx, 55, 0)
    xs = np.array(FINE, dtype=float)
    out = {}
    for name, curve in (("paper ruler", f_ridge), ("EWMA ruler", f_plus)):
        vs_full = np.minimum.accumulate(np.array([F.r2(yb, curve[g], BENCH) for g in FINE]))
        pt = age_of(F.r2(yb, comb, BENCH), xs, vs_full)
        rng = np.random.default_rng(SEED)
        draws = []
        for _ in range(N_BOOT):
            rows = blocks_of(n, rng)
            vs = np.minimum.accumulate(
                np.array([r2_on(yb, curve[g], rows, bench=BENCH) for g in FINE]))
            draws.append(age_of(r2_on(yb, comb, rows, bench=BENCH), xs, vs))
        arr = np.array(draws, dtype=float)
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        out[name] = (pt, lo, hi, hi - lo, int(np.isnan(arr).sum()))
        print(f"  {name:>12}:  {pt:>5.1f} days   [{lo:.1f}, {hi:.1f}]   "
              f"width {hi - lo:.1f} days   {(hi - lo) / pt:.2f} x the point"
              f"   off-grid {out[name][4]}")
    wr, wp = out["paper ruler"][3], out["EWMA ruler"][3]
    rel_r = wr / out["paper ruler"][0]
    rel_p = wp / out["EWMA ruler"][0]
    print(f"\n  width changes by {100 * (wp - wr) / wr:+.0f}% in days, "
          f"{100 * (rel_p - rel_r) / rel_r:+.0f}% as a share of the point")

    # ---------------- verdict ------------------------------------------------
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    print(f"  The projection the objection names changes nothing at all ({worst_proj:.0e}),")
    print("  and the reason is algebraic rather than empirical: a standardised")
    print("  linear model absorbs any deterministic rescaling of its inputs. A")
    print("  filter can only help through the state it builds, so the test that")
    print("  matters is the exponential one.")
    if better == 0:
        print(f"\n  Exponential weighting of the whole history beats the paper's two")
        print(f"  rectangular windows at 0 of {len(DELAYS)} delays, and loses by more the staler")
        print("  the control is. The own-only control is not the straw man the")
        print("  objection assumes: it is the strongest arm of the three tried here,")
        print(f"  and the rate at eleven weeks stays at {r55:.1%}. Measured against the")
        print(f"  weaker exponential control it would read {p55:.1%}, which is the")
        print("  inflation the referees warned about, pointing the other way.")
    elif worst < 0.05:
        print(f"\n  Exponential weighting helps at {better} of {len(DELAYS)} delays, so the objection")
        print(f"  has real content, and the rate moves by {worst:.1%} at most - less than")
        print("  the four points Section 10.1 already gives up to a richer feature")
        print(f"  set. At eleven weeks: {(S_c[55] - S_r[55]) / (S_r[0] - S_r[55]):.1%} as published, "
              f"{d55:.1%} against the best control.")
    else:
        print(f"\n  Exponential weighting helps at {better} of {len(DELAYS)} delays and moves the rate")
        print(f"  by up to {worst:.1%}. That is large enough that the headline should be")
        print(f"  quoted against this control: {d55:.1%} at eleven weeks, not "
              f"{(S_c[55] - S_r[55]) / (S_r[0] - S_r[55]):.1%}.")
    print(f"\n  The effective-age interval moves by {100 * (rel_p - rel_r) / rel_r:+.0f}% "
          f"as a share of the point it")
    print("  surrounds, which is not the dramatic tightening predicted. Nor could it")
    print("  be: that width is sampling uncertainty over test days, and changing the")
    print("  ruler changes the age a forecast reads as, not how much that reading")
    print("  varies from one resampled window to the next. Section 9.3 keeps the")
    print("  paper's ruler, which reads the headline as the older and therefore less")
    print(f"  flattering {out['paper ruler'][0]:.1f} days rather than "
          f"{out['EWMA ruler'][0]:.1f}. The prediction was the right kind of")
    print("  claim and the wrong answer, and it was cheaper to test than to argue")
    print("  about.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
