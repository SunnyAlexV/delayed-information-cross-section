"""
lab41_conditional_anatomy.py - is the conditional result about markets or about the ratio?

Imports lab05_robustness and lab21_stronger_inference; keep all three in labs/.
Runtime about a minute.

THE OBJECTION
-------------
Section S5 reports the substitution rate by market state and finds it high and
well determined in stress, and unmeasurable in calm.  The paper reads that as a
statement about the cross-section.  Part of it may be a statement about the
measuring instrument instead, and the two are easy to confuse.

R(delta) is a ratio,

    R = [ SSE_own(delta) - SSE_cross(delta) ] / [ SSE_own(delta) - SSE_own(0) ]
      =            N(delta)                  /            D(delta)

with N the error the cross-section removes and D the error the delay adds.  Both
are means of squared errors on the same days, so both are in the units of the
target and both can be read directly.

Now: stressed days are, almost by definition, the days on which being late costs
a forecaster most.  So stress is exactly where D is largest - which is exactly
where a ratio with D underneath is best determined.  "The rate is high and tight
in stress" could therefore mean the cross-section does more work there, or it
could mean the denominator is big enough there for the ratio to behave.  The
first is a claim about markets.  The second is a claim about arithmetic.

THE TEST
--------
Report N and D separately by state, not just their ratio.

    If N is much larger in stress, the cross-section genuinely does more work
    when markets are frightened and the conditional headline is a finding.

    If N is flat across states and only D moves, the conditional headline is
    largely circular: it says the ratio is measurable where its denominator is
    large, which is true of every ratio and says nothing about breadth.

PART B: THE INTERVAL THE PAPER ADMITS IS WRONG
-----------------------------------------------
Section S5 prints [6, 84] for the calm-market rate at eleven weeks and then
says, in the text, that the percentile bootstrap which produced it cannot be
right - resampling a ratio whose denominator can approach zero cannot represent
an unbounded set, and an unbounded set is what a near-zero denominator implies.
Appendix C derives Fieller's construction for exactly this case and lab21
implements it.  It has never been applied to the conditional table.  It is here.
A set that comes back unbounded is not a failure; it is the honest answer, and
it is a stronger statement than a bounded interval nobody should believe.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab08_implied_vol as IV
from lab21_stronger_inference import fieller, den_tau

SEED = 20260915
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 5, 13, 21, 55]


def panel(folder):
    D, peers, lag = L.build(folder, TARGET)
    a = D[TARGET].values
    sa = pd.Series(a)
    own = np.column_stack([sa.values, sa.rolling(5).mean().values,
                           sa.rolling(22).mean().values])
    P = D[peers].values
    n = len(D)
    y = np.full(n, np.nan); y[:-H] = a[H:]
    start = L.MED + L.TRAIN + L.VAL + max(DELAYS) + H
    idx = np.arange(start, n - H)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])]
    return own, P, y, idx, D, len(peers)


def walk(own, P, y, idx, delta, use_peers):
    out = np.empty(len(idx)); b = mu = sd = None
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
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


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints the anatomy of the conditional rate, Section S5.\n")

    own, P, y, idx, D, k = panel(folder)
    yb = y[idx]
    n = len(idx)
    print(f"target {TARGET}, {n} test days, {k} foreign peers")

    E = {}
    for d in DELAYS:
        E[(d, "own")] = (yb - walk(own, P, y, idx, d, False)) ** 2
        E[(d, "cross")] = (yb - walk(own, P, y, idx, d, True)) ** 2

    vix = IV.load_iv(folder, "VIX").reindex(D.index).ffill(limit=5).shift(1)
    v = vix.values[idx]
    fin = np.isfinite(v)
    q1, q2 = np.nanpercentile(v[fin], [33.3, 66.7])
    terc = {"calm": v <= q1, "middle": (v > q1) & (v <= q2), "stressed": v > q2}
    print(f"VIX terciles at t-1: calm <= {q1:.1f}, stressed > {q2:.1f}")
    print("The VALUE is at t-1 and the BOUNDARIES are percentiles of the whole")
    print("test window, so a day's bucket is decided with the rest of the sample")
    print("in hand.  That is deliberate - a fixed boundary keeps 'stressed'")
    print("meaning the same thing in 2008 and in 2017, which a trailing boundary")
    print("does not - but it is NOT a rule a forecaster could have applied on the")
    print("day, and the paper called the cut a rolling one until this arm")
    print("existed.  Part E prices the difference instead of leaving the")
    print("word to carry it.\n")
    # The real-time counterpart: boundaries from each day's own trailing year,
    # lagged with the value, exactly as lab27 cuts its ladder.  The per-day
    # squared errors in E are the same object under either cut - only the mask
    # changes - so this costs no refitting.
    _sv = pd.Series(vix.values).rolling(252, min_periods=60)
    _rq1 = _sv.quantile(1.0 / 3.0).values[idx]
    _rq2 = _sv.quantile(2.0 / 3.0).values[idx]
    _rt = np.isfinite(v) & np.isfinite(_rq1) & np.isfinite(_rq2)
    terc_rt = {"calm": _rt & (v <= _rq1),
               "middle": _rt & (v > _rq1) & (v <= _rq2),
               "stressed": _rt & (v > _rq2)}

    # ---------------- A ----------------------------------------------------
    print("=" * 96)
    print("A.  THE RATIO, TAKEN APART")
    print("=" * 96)
    print("N is the mean squared error the cross-section REMOVES at that delay.")
    print("D is the mean squared error the delay ADDS. R = N / D. Both are in the")
    print("target's own units, so they can be compared across states directly.\n")
    print(f"{'state':>10}{'delta':>7}{'N (removed)':>14}{'D (delay cost)':>16}"
          f"{'R = N/D':>10}{'days':>7}")
    N, Dn = {}, {}
    for nm, m in terc.items():
        s = np.where(m)[0]
        for d in DELAYS[1:]:
            num = E[(d, "own")][s] - E[(d, "cross")][s]
            den = E[(d, "own")][s] - E[(0, "own")][s]
            N[(nm, d)], Dn[(nm, d)] = num, den
            r = num.mean() / den.mean() if abs(den.mean()) > 1e-12 else np.nan
            print(f"{nm:>10}{d:>7}{num.mean():>14.4f}{den.mean():>16.4f}"
                  f"{r:>10.0%}{len(s):>7}")
        print()

    print("The question is whether N moves with state, or only D.\n")
    print(f"{'delta':>6}{'N calm':>10}{'N stressed':>13}{'N ratio':>10}"
          f"{'D calm':>10}{'D stressed':>13}{'D ratio':>10}")
    nrat, drat = [], []
    for d in DELAYS[1:]:
        nc, ns = N[("calm", d)].mean(), N[("stressed", d)].mean()
        dc, ds = Dn[("calm", d)].mean(), Dn[("stressed", d)].mean()
        nrat.append(ns / nc if abs(nc) > 1e-12 else np.nan)
        drat.append(ds / dc if abs(dc) > 1e-12 else np.nan)
        print(f"{d:>6}{nc:>10.4f}{ns:>13.4f}{nrat[-1]:>10.1f}x"
              f"{dc:>10.4f}{ds:>13.4f}{drat[-1]:>10.1f}x")
    mn_n, mn_d = float(np.nanmean(nrat)), float(np.nanmean(drat))
    print(f"\n  averaged over delays, stress multiplies N by {mn_n:.1f} and D by {mn_d:.1f}")

    # ---------------- B ----------------------------------------------------
    print("\n" + "=" * 96)
    print("B.  A CONFIDENCE SET THAT CAN SAY 'NOT IDENTIFIED'")
    print("=" * 96)
    print("Fieller's construction on the same N and D, HAC-adjusted, from Appendix C.")
    print("'interval' is an ordinary two-sided set; 'exclusion' is the complement of")
    print("one, two half-lines; 'unbounded' excludes nothing at all. The last two are")
    print("answers a percentile bootstrap cannot give, and are the honest ones when a")
    print("denominator can approach zero.\n")
    # tau is printed beside every set, not left to be inferred.  lab63 and
    # lab66 index coverage by the denominator's t-statistic: the set reaches
    # its nominal level at large tau and falls short at small tau.  Without
    # tau here a reader cannot tell which row of those tables applies to the
    # cell in front of them, and the calm column is exactly where it matters.
    print(f"{'state':>10}{'delta':>7}{'R':>8}{'tau':>8}"
          f"{'Fieller 95% set':>40}")
    kinds = {}
    for nm in terc:
        for d in DELAYS[1:]:
            num, den = N[(nm, d)], Dn[(nm, d)]
            # fixed-b: fieller() chooses its own bandwidth from n
            kind, lo, hi = fieller(num, den)
            kinds[(nm, d)] = kind
            r = num.mean() / den.mean()
            if kind == "interval":
                txt = f"[{lo:+.0%}, {hi:+.0%}]"
            elif kind == "exclusion":
                txt = f"everything outside ({lo:+.0%}, {hi:+.0%})"
            else:
                txt = "unbounded - not identified"
            tau, _ = den_tau(den)
            print(f"{nm:>10}{d:>7}{r:>8.0%}{tau:>8.1f}{txt:>40}")
        print()

    bad = {nm: sum(1 for d in DELAYS[1:] if kinds[(nm, d)] != "interval")
           for nm in terc}
    for nm in terc:
        print(f"  {nm:>10}: {bad[nm]} of {len(DELAYS)-1} sets are not ordinary intervals")

    # ---------------- verdict ----------------------------------------------
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    if mn_n > 2.0 and mn_n > 0.5 * mn_d:
        print(f"  Stress multiplies the error the cross-section removes by {mn_n:.1f}, against")
        print(f"  {mn_d:.1f} for the error the delay adds. The numerator moves with the state,")
        print("  so the conditional result is a fact about markets and not an artefact of")
        print("  dividing by a larger denominator. Breadth does more work when markets")
        print("  are frightened, in absolute error and not merely as a share.")
    elif mn_n < 1.5:
        print(f"  Stress barely moves the error the cross-section removes ({mn_n:.1f}x) while")
        print(f"  multiplying the delay's cost by {mn_d:.1f}. The conditional headline is")
        print("  therefore mostly arithmetic: the ratio is measurable in stress because")
        print("  its denominator is large there, which is true of any ratio. The paper")
        print("  must report N alongside R and drop the claim that breadth works better")
        print("  in stress.")
    else:
        print(f"  Stress multiplies N by {mn_n:.1f} and D by {mn_d:.1f}. Both move, so the")
        print("  conditional result is part finding and part arithmetic, and the paper")
        print("  should quote N beside R rather than R alone.")
    if bad["calm"]:
        print(f"\n  {bad['calm']} of the calm-market sets are unbounded or exclusions, which is")
        print("  what the text already suspected and the percentile bootstrap could not")
        print("  express. The calm rate is not identified, and [6, 84] should go.")
    else:
        print("\n  Every calm-market set is an ordinary interval, so the percentile")
        print("  bootstrap was not misleading after all and the text's caveat can go.")

    # ---------------- E ----------------------------------------------------
    print("\n" + "=" * 96)
    print("E.  THE SAME CUT A FORECASTER COULD HAVE MADE ON THE DAY")
    print("=" * 96)
    print("Boundaries from each day's own trailing 252 sessions of VIX, taken at")
    print("t-1 with the value, instead of percentiles of the whole test window.")
    print("Same forecasts, same losses, same days: only which bucket a day falls")
    print("in changes. This is what the word 'rolling' in the paper has to mean")
    print("if it means anything, so the two cuts are printed side by side and")
    print("the paper quotes whichever it actually computed.\n")
    print(f"{'state':>10}{'delta':>7}{'R fixed':>10}{'days':>7}"
          f"{'R rolling':>12}{'days':>7}{'move':>8}")
    rt_rate = {}
    for nm in ("calm", "middle", "stressed"):
        s_rt = np.where(terc_rt[nm])[0]
        for d in DELAYS[1:]:
            num = E[(d, "own")][s_rt] - E[(d, "cross")][s_rt]
            den = E[(d, "own")][s_rt] - E[(0, "own")][s_rt]
            r_rt = (num.mean() / den.mean()
                    if abs(den.mean()) > 1e-12 else np.nan)
            rt_rate[(nm, d)] = r_rt
            r_fx = N[(nm, d)].mean() / Dn[(nm, d)].mean()
            print(f"{nm:>10}{d:>7}{r_fx:>10.0%}{len(N[(nm, d)]):>7}"
                  f"{r_rt:>12.0%}{len(s_rt):>7}{(r_rt - r_fx):>+8.0%}")
        print()
    _mv = [abs(rt_rate[(nm, d)] - N[(nm, d)].mean() / Dn[(nm, d)].mean())
           for nm in ("calm", "middle", "stressed") for d in DELAYS[1:]
           if np.isfinite(rt_rate[(nm, d)])]
    print(f"  largest move between the two cuts: {max(_mv):.0%} of the rate; "
          f"mean {float(np.mean(_mv)):.0%}")
    _hs = N[("stressed", DELAYS[-1])].mean() / Dn[("stressed", DELAYS[-1])].mean()
    print(f"  at the headline delta = {DELAYS[-1]}, stressed reads {_hs:.0%} on the "
          f"fixed cut and {rt_rate[('stressed', DELAYS[-1])]:.0%} on the rolling one")
    print("  Which one the paper may call real-time is the rolling column, and")
    print("  the fixed column is the one every table reports; the difference")
    print("  above is what the description was carrying for free.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
