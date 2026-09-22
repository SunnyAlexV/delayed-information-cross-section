"""
lab50_filtered_headline.py - the headline recomputed inside the better estimator.

Imports lab05_robustness, lab22_factor_benchmark and lab33_ragged_edge; keep all
four in labs/.  Runtime about six minutes.

THE LIMITATION THIS ADDRESSES
-----------------------------
Section 11 names the estimator as the largest limitation in the paper and prices
it rather than gesturing at it.  A two-step state-space model after Doz, Giannone
and Reichlin, with the parameters frozen on the initial training block and the
filtered state fed into the same forecasting ridge, beats the specification used
throughout this paper at delta = 21 by 0.0124 of R-squared [+0.00114, +0.02192]
and at delta = 55 by 0.0204 [+0.00728, +0.03109].  It loses at delta = 0 by
0.0233.  Imposing a one-factor structure costs skill when the mark is fresh and
pays when it is stale.

The paper then reports this instead of adopting it, and gives a scope reason:
R(delta) carries the same domestic control three times, so a better estimator
does not simply add its improvement to the rate, and recomputing every
conditional and every robustness check inside a filtered specification is a
different paper.

That defence invites one obvious reply, and it is the reply a referee will make:
then do the HEADLINE.  Not every conditional, not every robustness check.  One
number, the one the paper is quoted on, computed inside the estimator the paper
itself says is better.  This file does that, and does the effective age with it.

WHAT MOVES AND WHAT DOES NOT
----------------------------
The rate is

    R(delta) = [ S_cross(delta) - S_own(delta) ] / [ S_own(0) - S_own(delta) ]

and the filter enters only through S_cross, for a reason established elsewhere.
lab46 fitted an own-only state space and found nothing for it to do: the moments
put no observation noise in the domestic series, so the filtered state IS the
observation, and a deterministic projection of it is absorbed exactly by a
standardised ridge.  There is no filtered own-only control to build.  The
control therefore stays the paper's, the ruler stays the paper's, and the
question is what happens to the rate and to the age when the combined model is
allowed to use the estimator the paper declined.

The direction is worth predicting before reading the answer.  The filter raises
S_cross at long delays and leaves S_own alone, so it raises the numerator and
leaves the denominator: the rate should go UP.  If it does, the objection
inverts.  The paper is not hiding behind a weaker estimator to protect a large
number; it is quoting a smaller number than the better estimator would give it.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab22_factor_benchmark as F
import lab33_ragged_edge as RE

SEED = 20260917
H = L.HORIZON
DELAYS = [0, 3, 5, 13, 21, 55]
FINE = [0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 21, 26, 34, 44, 55]
HEAD = 55
# The block is lab05's measured one, not 2 * HORIZON.  The h-day overlap in
# the target is not the only dependence in a loss difference: it also
# inherits the common factor's persistence, which lab58 measures at 0.48
# autocorrelation at lag 10 and not below 0.05 until lag 38.  A block
# shorter than the dependence leaves it inside the resample and the
# interval comes out too narrow.
N_BOOT, BLOCK = 1500, L.BLOCK


def r2(y, f, *, bench, rows=None):
    """Out-of-sample skill against the trailing-mean benchmark.

    `bench` is keyword-only and has no default on purpose.  This helper used to
    divide by ((yy - yy.mean()) ** 2), the mean of the target over the test
    period - a constant chosen with hindsight, and the error an external audit
    caught elsewhere in this project.  The rollout that fixed it missed this
    file, which computed the right benchmark at the top of main() and then never
    used it.  Making the argument mandatory means a missed call site raises
    instead of quietly scoring against the wrong yardstick, and keyword-only
    means an old positional `r2(y, f, rows)` cannot pass `rows` in as `bench`.
    """
    yy, ff = (y, f) if rows is None else (y[rows], f[rows])
    bb = bench if rows is None else bench[rows]
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
    print("Prints the headline inside the filtered specification, Section 11.\n")

    own, P, y, idx, k = F.panel(folder)
    yb = y[idx]
    BENCH = L.bench_mean(y, idx)
    n = len(idx)
    M = np.column_stack([own[:, 0], P])
    print(f"target {F.TARGET}, {n} test days, {k} foreign peers")

    par = RE.fit_state_space(M[:L.TRAIN + L.VAL])
    if par is None:
        raise SystemExit("the state space could not be fitted on the training block")
    print(f"\nstate space fitted ONCE on the first {L.TRAIN + L.VAL} rows and frozen:")
    print(f"  factor persistence phi = {par['phi']:.4f}, innovation variance "
          f"q = {par['q']:.4f}")
    print(f"  loadings {np.array2string(par['lam'], precision=2, separator=' ')}")
    print("  Nothing about the test period enters these, and the forecasting ridge")
    print("  on top is refitted on the paper's own schedule.")

    # ---------------- A. the three specifications --------------------------
    print("\n" + "=" * 96)
    print("A.  THE SAME THREE MODELS THE RATE IS BUILT FROM, IN BOTH ESTIMATORS")
    print("=" * 96)
    print("own-only is the paper's ridge in both columns, because lab46 shows there")
    print("is no filtered own-only control to build: with no observation noise the")
    print("filtered state is the observation itself.\n")
    st = {d: RE.ragged_state(par, M, d) for d in DELAYS}
    f_own = {d: RE.walk(own, P, y, idx, d, "STALE") for d in DELAYS}
    f_pap = {d: RE.walk(own, P, y, idx, d, "PAPER") for d in DELAYS}
    f_fil = {d: RE.walk(own, P, y, idx, d, "FILTER", st[d]) for d in DELAYS}
    S_own = {d: r2(yb, f_own[d], bench=BENCH) for d in DELAYS}
    S_pap = {d: r2(yb, f_pap[d], bench=BENCH) for d in DELAYS}
    S_fil = {d: r2(yb, f_fil[d], bench=BENCH) for d in DELAYS}
    print(f"{'delta':>6}{'own only':>11}{'cross, ridge':>15}{'cross, filter':>15}"
          f"{'filter - ridge':>17}")
    for d in DELAYS:
        print(f"{d:>6}{S_own[d]:>11.4f}{S_pap[d]:>15.4f}{S_fil[d]:>15.4f}"
              f"{S_fil[d] - S_pap[d]:>+17.4f}")
    better = [d for d in DELAYS if S_fil[d] > S_pap[d]]
    print(f"\n  the filter is the better cross-sectional model at "
          f"{len(better)} of {len(DELAYS)} delays: {better}")

    # ---------------- B. the rate ------------------------------------------
    print("\n" + "=" * 96)
    print("B.  THE RATE, COMPUTED INSIDE EACH ESTIMATOR")
    print("=" * 96)
    print("Same control, same denominator, same days. The only thing that differs")
    print("is which model is allowed to carry the cross-section.\n")
    print(f"{'delta':>6}{'R, paper':>11}{'R, filtered':>14}{'move':>9}")
    Rp, Rf = {}, {}
    for d in DELAYS[1:]:
        den = S_own[0] - S_own[d]
        Rp[d] = (S_pap[d] - S_own[d]) / den
        Rf[d] = (S_fil[d] - S_own[d]) / den
        print(f"{d:>6}{Rp[d]:>11.1%}{Rf[d]:>14.1%}{Rf[d] - Rp[d]:>+9.1%}")

    rng = np.random.default_rng(SEED)
    diffs, both = [], []
    for _ in range(N_BOOT):
        rows = blocks_of(n, rng)
        den = (r2(yb, f_own[0], rows=rows, bench=BENCH)
               - r2(yb, f_own[HEAD], rows=rows, bench=BENCH))
        if den < 1e-9:
            continue
        sd_ = r2(yb, f_own[HEAD], rows=rows, bench=BENCH)
        rp = (r2(yb, f_pap[HEAD], rows=rows, bench=BENCH) - sd_) / den
        rf = (r2(yb, f_fil[HEAD], rows=rows, bench=BENCH) - sd_) / den
        diffs.append(rf - rp)
        both.append(rf)
    dlo, dhi = np.percentile(diffs, [2.5, 97.5])
    flo, fhi = np.percentile(both, [2.5, 97.5])
    print(f"\n  at eleven weeks: {Rp[HEAD]:.1%} as published, {Rf[HEAD]:.1%} filtered")
    print(f"  paired difference {Rf[HEAD] - Rp[HEAD]:+.1%} [{dlo:+.1%}, {dhi:+.1%}], "
          f"{len(diffs)} replications")
    print(f"  the filtered rate's own interval: [{flo:.1%}, {fhi:.1%}]")

    # ---------------- C. the age -------------------------------------------
    print("\n" + "=" * 96)
    print("C.  AND THE HEADLINE IN DAYS")
    print("=" * 96)
    print("The ruler is the paper's own-only delay curve over sixteen ages, which is")
    print("what the sentence means by 'a mark this old'. Only the model being")
    print("measured against it changes.\n")
    curve = {g: RE.walk(own, P, y, idx, g, "STALE") for g in FINE}
    xs = np.array(FINE, dtype=float)
    vs = np.minimum.accumulate(
        np.array([r2(yb, curve[g], bench=BENCH) for g in FINE]))
    age_p = age_of(S_pap[HEAD], xs, vs)
    age_f = age_of(S_fil[HEAD], xs, vs)
    rng = np.random.default_rng(SEED)
    ap, af, off = [], [], 0
    for _ in range(N_BOOT):
        rows = blocks_of(n, rng)
        v = np.minimum.accumulate(
            np.array([r2(yb, curve[g], rows=rows, bench=BENCH) for g in FINE]))
        x1 = age_of(r2(yb, f_pap[HEAD], rows=rows, bench=BENCH), xs, v)
        x2 = age_of(r2(yb, f_fil[HEAD], rows=rows, bench=BENCH), xs, v)
        if not np.isfinite(x1) or not np.isfinite(x2):
            off += 1
            continue
        ap.append(x1); af.append(x2)
    plo, phi_ = np.percentile(ap, [2.5, 97.5])
    filo, fihi = np.percentile(af, [2.5, 97.5])
    dif = np.array(af) - np.array(ap)
    glo, ghi = np.percentile(dif, [2.5, 97.5])
    print(f"  paper's cross-sectional model:  {age_p:>5.1f} days  "
          f"[{plo:.1f}, {phi_:.1f}]")
    print(f"  filtered cross-sectional model: {age_f:>5.1f} days  "
          f"[{filo:.1f}, {fihi:.1f}]")
    print(f"  paired difference: {age_f - age_p:+.1f} days [{glo:+.1f}, {ghi:+.1f}], "
          f"{off} replications off the grid")

    # ---------------- verdict ----------------------------------------------
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    up = Rf[HEAD] > Rp[HEAD]
    if up:
        print(f"  Inside the estimator the paper declined, the headline is LARGER:")
        print(f"  {Rf[HEAD]:.1%} against {Rp[HEAD]:.1%} at eleven weeks, a paired "
              f"{Rf[HEAD] - Rp[HEAD]:+.1%} [{dlo:+.1%}, {dhi:+.1%}],")
        print(f"  and the mark it is worth reads {age_f:.1f} days old rather than "
              f"{age_p:.1f}.")
        print("\n  So the estimator limitation does not run the way a limitation")
        print("  usually does. The filter improves the model that carries the")
        print("  cross-section and has nothing to add to the control, so it raises")
        print("  the numerator and leaves the denominator alone. The paper is not")
        print("  protecting a large number with a weak estimator. It is quoting a")
        print("  smaller number than the better estimator would give it, for the")
        print("  transparency of a fixed feature set, and this file prices that")
        print("  choice in the paper's own units instead of in R-squared.")
    else:
        print(f"  Inside the filtered specification the headline falls to {Rf[HEAD]:.1%} from")
        print(f"  {Rp[HEAD]:.1%}, a paired {Rf[HEAD] - Rp[HEAD]:+.1%} [{dlo:+.1%}, {dhi:+.1%}]. The estimator")
        print("  limitation therefore bears on the headline and not only on skill,")
        print("  and Section 11 has to report the filtered figure beside the")
        print("  published one rather than describing the cost in R-squared alone.")
    print("\n  What this does not do is recompute the conditionals, the regime")
    print("  ladder, the five markets or the horse race inside the filter. That is")
    print("  the different paper Section 11 declines to write, and it is still")
    print("  declined. What was owed was the headline, and the headline is here.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
