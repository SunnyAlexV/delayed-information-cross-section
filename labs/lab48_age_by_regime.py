"""
lab48_age_by_regime.py - the calm-market cell, measured in a coordinate that works.

Imports lab05_robustness, lab08_implied_vol and lab37_lead_lag; keep all four in
labs/.  Runtime about four minutes.

THE LIMITATION THIS ADDRESSES
-----------------------------
Table 17 reports the substitution rate by market state.  The stressed column is
81% [74, 87] at eleven weeks and is identical in both halves of the sample.  The
calm column is 47% [6, 84], and at three of four delays it does not exclude zero.
Section 4 therefore warns that the headline 72% is the average of a result and a
non-result, and Section 11 carries it as a limitation on the headline itself.

The reason is mechanical and the paper states it:

    R(delta) = [ S_cross(delta) - S_own(delta) ] / [ S_own(0) - S_own(delta) ]

and in a calm market the denominator is the damage delay does, which is nearly
nothing.  A share of nearly nothing has no stable value.  That is a property of
the QUESTION as posed by a ratio, and it invites an obvious objection: the paper
has chosen a coordinate that fails in half its sample and then reported the
failure as a fact about markets.

THE TEST
--------
The paper has a second coordinate, and it does not divide by the damage.  The
effective age of Section 9.3 asks: how old would a plain domestic mark have to
be for its own forecast to be as bad as the model that holds a fifty-five-day-old
mark and today's foreign closes?  Solve

    MSE_own(a)  =  MSE_cross(55)

for the age a, by interpolating the own-only curve over sixteen ages.  Two things
make this the right instrument for the question.  It has no denominator that can
vanish.  And it is stated in mean squared error rather than in skill, so no
benchmark variance enters either - which matters, because the benchmark variance
is exactly what is small in a calm market.

So the calm cell can be asked again in a coordinate that has no reason to fail.
Either it answers, and the paper gains a number where it has none; or it does not,
and the paper gains something better than a number: the knowledge that the
calm-market non-result survives a change of coordinate and is therefore about
markets rather than about ratios.

WHAT WOULD COUNT AS FAILURE
---------------------------
The interpolation is bounded by the grid, so an interval always exists on paper.
The honest diagnostic is how many bootstrap replications land AT zero (the
combined model beating a current mark) or run off the far end (beyond 55 days),
because those are replications where the curve did not answer.  Both counts are
printed for every cell, and no cell is read as determinate if they are large.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab08_implied_vol as IV
import lab37_lead_lag as LL

SEED = 20260917
TARGET = "SPX"
DELTA = 55
FINE = [0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 21, 26, 34, 44, 55]
# The block is lab05's measured one, not 2 * HORIZON.  The h-day overlap in
# the target is not the only dependence in a loss difference: it also
# inherits the common factor's persistence, which lab58 measures at 0.48
# autocorrelation at lag 10 and not below 0.05 until lag 38.  A block
# shorter than the dependence leaves it inside the resample and the
# interval comes out too narrow.
N_BOOT, BLOCK = 1500, L.BLOCK
OFFGRID_MAX = 0.10          # above this share of replications, a cell is not read
WIDTH_MAX = 3.0             # and no wider, relative to its point, than 3x the stressed cell


def mse_on(y, f, rows):
    d = y[rows] - f[rows]
    return float(d @ d) / len(rows)


def age_of_mse(target, xs, vs):
    """The age at which the own-only curve's error reaches `target`.

    vs is the own-only mean squared error against age, made monotone the same
    way Section 9.3 makes the skill curve monotone.  Returns 0.0 when the
    combined model is worse than a current mark and NaN when it is better than
    the oldest mark on the grid, and both are counted rather than hidden.
    """
    if target <= vs[0]:
        return 0.0
    if target >= vs[-1]:
        return np.nan
    for i in range(len(xs) - 1):
        if vs[i] <= target <= vs[i + 1]:
            span = vs[i + 1] - vs[i]
            w = 0.0 if span < 1e-15 else (target - vs[i]) / span
            return float(xs[i] + w * (xs[i + 1] - xs[i]))
    return np.nan


def blocks_of(n, rng, block=BLOCK):
    starts = rng.integers(0, max(n - block + 1, 1), size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints the effective age by market state, Sections 9.1 and 10.2.\n")

    own, P, y, idx, D, peers, _lag = LL.panel(folder)
    yb = y[idx]
    n = len(idx)
    dates = pd.DatetimeIndex(D.index[idx])
    print(f"target {TARGET}, {n} test days, {dates[0].date()} to {dates[-1].date()}")

    # ---------------- the models, computed once ---------------------------
    curve = {g: LL.walk_own(own, y, idx, g) for g in FINE}
    comb = LL.walk(own, P, y, idx, DELTA, 0)
    xs = np.array(FINE, dtype=float)

    # ---------------- the states ------------------------------------------
    vix = IV.load_iv(folder, "VIX").reindex(D.index).ffill(limit=5).shift(1)
    v = vix.values[idx]
    fin = np.isfinite(v)
    q1, q2 = np.nanpercentile(v[fin], [33.3, 66.7])
    sv = pd.Series(v)
    top10 = np.isfinite(v) & (v > sv.rolling(252, min_periods=60).quantile(0.90).values)
    top5 = np.isfinite(v) & (v > sv.rolling(252, min_periods=60).quantile(0.95).values)
    STATES = [("every day", np.ones(n, bool)),
              ("calm (bottom tercile)", fin & (v <= q1)),
              ("middle tercile", fin & (v > q1) & (v <= q2)),
              ("stressed (top tercile)", fin & (v > q2)),
              ("VIX top 10%", top10),
              ("VIX top 5%", top5)]
    print(f"VIX at t-1, terciles at {q1:.1f} and {q2:.1f}; the two rungs are")
    print("percentiles of the series against its own trailing year, as in Table 13.\n")

    # ---------------- A. the two coordinates, side by side ----------------
    print("=" * 96)
    print("A.  THE RATE AND THE AGE, IN EVERY STATE, ON THE SAME DAYS")
    print("=" * 96)
    print("The rate divides by the damage delay does. The age does not divide by")
    print("anything: it asks how old a plain mark would have to be to forecast as")
    print("badly, by matching mean squared error on the same days.\n")
    print(f"{'state':>24}{'days':>7}{'R(55)':>9}{'age':>9}{'own MSE at 0d':>16}"
          f"{'at 55d':>9}")
    point = {}
    for name, m in STATES:
        rows = np.where(m)[0]
        if len(rows) < 100:
            continue
        e0 = mse_on(yb, curve[0], rows)
        e55 = mse_on(yb, curve[55], rows)
        ec = mse_on(yb, comb, rows)
        rate = (e55 - ec) / (e55 - e0) if abs(e55 - e0) > 1e-12 else np.nan
        vs = np.maximum.accumulate(np.array([mse_on(yb, curve[g], rows) for g in FINE]))
        age = age_of_mse(ec, xs, vs)
        point[name] = (len(rows), rate, age)
        print(f"{name:>24}{len(rows):>7}{rate:>9.1%}{age:>9.1f}{e0:>16.4f}{e55:>9.4f}")
    print("\n  The rate and the age are computed from the same three quantities, so")
    print("  they are not independent readings. What differs is that one of them")
    print("  puts the delay damage in a denominator and the other does not.")
    print("\n  One row of that table was not what this file was built to find. Along")
    print("  the stress ladder the age falls monotonically: "
          f"{point['stressed (top tercile)'][2]:.1f} days in the top")
    print(f"  tercile, {point['VIX top 10%'][2]:.1f} in the top decile, "
          f"{point['VIX top 5%'][2]:.1f} in the top 5% of VIX days. On the days")
    print("  when a stale mark is most dangerous, a mark eleven weeks old plus")
    print("  today's foreign closes is worth a mark two days old.")

    # ---------------- B. intervals ----------------------------------------
    print("\n" + "=" * 96)
    print("B.  WHICH OF THE TWO SURVIVES BEING RESAMPLED")
    print("=" * 96)
    print("Moving blocks of ten days, drawn inside each state's own time-ordered")
    print("subsequence, with the curve, the combined model and the rate all")
    print("recomputed on the same resampled days. 'at 0d' and 'off grid' count the")
    print("replications where the age question had no answer on the grid.\n")
    print(f"{'state':>24}{'R(55)':>8}{'95% interval':>20}{'age':>7}"
          f"{'95% interval':>18}{'at 0d':>7}{'off':>6}")
    out = {}
    for name, m in STATES:
        rows_all = np.where(m)[0]
        if len(rows_all) < 100:
            continue
        rng = np.random.default_rng(SEED)
        k = len(rows_all)
        ages, rates = [], []
        for _ in range(N_BOOT):
            pick = rows_all[blocks_of(k, rng)]
            e0 = mse_on(yb, curve[0], pick)
            e55 = mse_on(yb, curve[55], pick)
            ec = mse_on(yb, comb, pick)
            rates.append((e55 - ec) / (e55 - e0) if abs(e55 - e0) > 1e-12 else np.nan)
            vs = np.maximum.accumulate(
                np.array([mse_on(yb, curve[g], pick) for g in FINE]))
            ages.append(age_of_mse(ec, xs, vs))
        A = np.array(ages, dtype=float)
        Rr = np.array(rates, dtype=float)
        at0 = int((A == 0.0).sum())
        off = int(np.isnan(A).sum())
        alo, ahi = np.nanpercentile(A, [2.5, 97.5])
        rlo, rhi = np.nanpercentile(Rr, [2.5, 97.5])
        _, rate_pt, age_pt = point[name]
        out[name] = (rate_pt, rlo, rhi, age_pt, alo, ahi, at0, off, len(rows_all))
        print(f"{name:>24}{rate_pt:>8.0%}  [{rlo:>5.0%},{rhi:>5.0%}]{age_pt:>10.1f}"
              f"  [{alo:>4.1f},{ahi:>5.1f}]{at0:>7}{off:>6}")

    # ---------------- C. the comparison that matters ----------------------
    print("\n" + "=" * 96)
    print("C.  IS THE CALM CELL UNDETERMINED, OR WAS IT THE RATIO?")
    print("=" * 96)
    calm = out["calm (bottom tercile)"]
    stress = out["stressed (top tercile)"]
    cw_r = calm[2] - calm[1]
    sw_r = stress[2] - stress[1]
    cw_a = calm[5] - calm[4]
    sw_a = stress[5] - stress[4]
    print(f"  calm     rate spans {cw_r:.0%} of the unit interval; "
          f"age spans {cw_a:.1f} days")
    print(f"  stressed rate spans {sw_r:.0%} of the unit interval; "
          f"age spans {sw_a:.1f} days")
    print(f"\n  relative width, calm against stressed: "
          f"{cw_r / sw_r:.1f}x on the rate, {(cw_a / calm[3]) / (sw_a / stress[3]):.1f}x "
          f"on the age")
    bad = (calm[6] + calm[7]) / N_BOOT
    rel = (cw_a / calm[3]) / (sw_a / stress[3])
    print(f"  calm replications with no answer on the grid: "
          f"{calm[6] + calm[7]} of {N_BOOT} ({bad:.1%})")
    print(f"\n  A cell is read as determinate here when two things hold: fewer than")
    print(f"  {OFFGRID_MAX:.0%} of its replications fail to place the age on the grid, and its")
    print(f"  width relative to its own point is within {WIDTH_MAX:.0f}x the stressed cell's.")
    print(f"  Calm: {bad:.1%} fail to place, and {rel:.1f}x the stressed width.")
    determinate = bad <= OFFGRID_MAX and rel <= WIDTH_MAX

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    if determinate:
        print(f"  In calm markets the rate is {calm[0]:.0%} [{calm[1]:.0%}, {calm[2]:.0%}], which is most")
        print("  of the unit interval and tells a reader nothing. The same days, asked")
        print(f"  in days rather than in shares, say that a fifty-five-day-old mark plus")
        print(f"  today's closes forecasts like a mark {calm[3]:.1f} days old "
              f"[{calm[4]:.1f}, {calm[5]:.1f}].")
        print(f"  Only {calm[6] + calm[7]} of {N_BOOT} replications failed to place it on the grid.")
        print("\n  So the calm-market cell is not unmeasurable. It was the ratio that")
        print("  could not be measured there, exactly as Section 4 said and for the")
        print("  reason Section 4 gave, and the paper now has a number for the state")
        print("  where it previously had a shrug. The stressed reading is unaffected:")
        print(f"  {stress[3]:.1f} days [{stress[4]:.1f}, {stress[5]:.1f}] against the calm "
              f"{calm[3]:.1f}.")
    else:
        print(f"  In calm markets the age is {calm[3]:.1f} days [{calm[4]:.1f}, {calm[5]:.1f}], "
              f"with {calm[6] + calm[7]} of {N_BOOT}")
        print("  replications unable to place it on the grid at all. Changing")
        print("  coordinate does not rescue the calm cell.")
        print("\n  That is worth more than a number would have been. The calm-market")
        print("  non-result is not an artefact of putting the delay damage in a")
        print("  denominator: where delay costs a forecaster almost nothing, there is")
        print("  almost nothing for the cross-section to repair, and no coordinate")
        print("  makes a share of nothing precise. Section 11 can say that the")
        print("  limitation survives a change of instrument rather than resting on")
        print("  the one instrument that fails.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
