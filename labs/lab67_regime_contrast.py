"""
lab67_regime_contrast.py - the state-dependence claim, estimated as ONE
parameter on the full sample instead of compared between two short slices.

THE DEFECT THIS REPLACES
------------------------
Table S7 slices 4,862 evaluation days into regimes as short as 246 days and runs
a HAC t-statistic inside each slice with b = M/n held at 0.05.  Fixed-b
asymptotics are asymptotic in n, and at n = 246 against dependence running to
delta + h = 60 the slice holds about four effective observations.  Simulated
coverage of a nominal 95% fixed-b set at those lengths, printed in part A below,
is 57% on the 246-day rung and 65% on the 367-day one, against 94% on the
3,996-day rung and 95% on the full 4,861-day window.  The pair the grade
actually rested on is 367 against 3,996, so the gap in reliability between the
two sets being compared is 65% against 94%, not an extreme of the list.

The paper's claim that the effect is larger in stressed markets was graded
ESTABLISHED on the non-overlap of two such marginal intervals.  Two objections.
The first is that non-overlap of marginal intervals is not a test of a difference:
the quantity the claim is about is the contrast, and neither of those two sets is a
set for the contrast.  Section S10 of the appendix now says so in its own words;
an earlier version of this docstring credited Section S12, which is about a stale
foreign feed and says nothing of the kind.  A citation nobody follows is a claim
nobody checks, which is how the wrong section number survived being written down.  The second is that one of the two
intervals has 66% coverage and the other 93%, so they are not even comparable
instruments.

THE REPAIR
----------
Indicator-weighted series, defined on EVERY day and zero outside their regime:

    ns_t = d1_t * 1{stressed}      vs_t = Lown_t * 1{stressed}
    nc_t = d1_t * 1{baseline}      vc_t = Lown_t * 1{baseline}

so the series stay contiguous and n stays at its full length.  The contrast is a
smooth function of four means whose joint long-run covariance is estimated at
full length, and the delta method gives ONE interval for ONE parameter.  A
moving-block bootstrap at a block of delta + h is run beside it as a second
instrument.

WHICH QUANTITY
--------------
Two, kept apart deliberately, because conflating them is what produced the grade
the audit overturned.  The MSE SHARE is mean(d1)/MSE_own inside the regime; the
SUBSTITUTION RATE is the paper's R(delta).  The claim "the effect is larger in
stressed markets" is about the first.  An interval for the first cannot grade a
claim about the second, and part C reports both so the distinction is visible.

THE LADDER
----------
lab27's rungs are a search family: it says itself that the tightening "was added
after the wide version had been run".  So the rungs are treated as ORDERED
robustness, not as five independent confirmations, and the headline statistic is
the MAXIMUM standardised contrast across the pre-registered ladder, calibrated by
the same joint block bootstrap.  That statistic prices the freedom to report
whichever rung looks strongest, which is the freedom the post-hoc extension
created.  Quoting it lets the paper concede the search and still make the claim.
"""
import math
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab08_implied_vol as IV
import lab66_har_alternatives as H66                             # noqa: E402

H = 5
DELTA = 55
TARGET = "SPX"
BLOCK = DELTA + H
NBOOT = 4000
EPISODES = [("2008 crisis", "2008-09-01", "2009-06-30"),
            ("2011 euro", "2011-07-01", "2011-12-31"),
            ("2015 devaluation", "2015-08-01", "2016-02-29"),
            ("2018 vol spike", "2018-02-01", "2018-03-31"),
            ("2020 COVID", "2020-02-15", "2020-06-30"),
            ("2022 inflation", "2022-01-01", "2022-12-31")]


def r2(y, f, bench):
    return 1 - np.sum((y - f) ** 2) / np.sum((y - bench) ** 2)


def bartlett_cov(X, M):
    """Long-run covariance OF THE COLUMN MEANS, at full length."""
    X = X - X.mean(0)
    T = len(X)
    S = X.T @ X / T
    for j in range(1, M + 1):
        w = 1 - j / (M + 1)
        G = X[j:].T @ X[:-j] / T
        S += w * (G + G.T)
    return S / T


def contrast(na, da, nb, db, M):
    X = np.column_stack([na, da, nb, db])
    mu = X.mean(0)
    V = bartlett_cov(X, M)
    ra, rb = mu[0] / mu[1], mu[2] / mu[3]
    g = np.array([1 / mu[1], -mu[0] / mu[1] ** 2, -1 / mu[3], mu[2] / mu[3] ** 2])
    return ra, rb, ra - rb, math.sqrt(max(float(g @ V @ g), 0.0))


def fixed_b_cv(n, M, rng, reps=3000):
    ts = np.empty(reps)
    for i in range(reps):
        z = rng.standard_normal(n)
        zc = z - z.mean()
        s = np.dot(zc, zc) / n
        for j in range(1, M + 1):
            s += 2 * (1 - j / (M + 1)) * np.dot(zc[j:], zc[:-j]) / n
        ts[i] = abs(z.mean()) / math.sqrt(max(s, 1e-18) / n)
    return float(np.quantile(ts, 0.95))


def coverage(n, M, q, rng, reps=1500):
    """Coverage of a nominal 95% fixed-b set at length n, dependence to lag q."""
    cv = fixed_b_cv(n, M, rng, reps=2000)
    hit = 0
    for _ in range(reps):
        e = rng.standard_normal(n + q)
        x = np.convolve(e, np.ones(q + 1), mode="valid")[:n]
        xc = x - x.mean()
        s = np.dot(xc, xc) / n
        for j in range(1, M + 1):
            s += 2 * (1 - j / (M + 1)) * np.dot(xc[j:], xc[:-j]) / n
        if abs(x.mean()) <= cv * math.sqrt(max(s, 1e-18) / n):
            hit += 1
    return cv, hit / reps


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("The state-dependence claim, as one parameter rather than two slices.\n")
    # TWO generators, deliberately.  Part A is a DIAGNOSTIC: it simulates
    # coverage at a list of sample lengths, and that list is read from
    # lab27 rather than typed, so it can change when a rung changes.  While
    # it shared one generator with parts B and C, lengthening the list moved
    # the bootstrap figures the paper quotes - the maximum standardised
    # contrast went from +4.92 to +5.00 because a coverage table grew three
    # rows.  A diagnostic must not be able to perturb a result, so part A
    # draws from its own stream and parts B and C from theirs.
    rng_a = np.random.default_rng(6701)
    rng = np.random.default_rng(67)

    print("=" * 92)
    print("A.  WHY THE SUBSAMPLE SETS CANNOT BE COMPARED")
    print("=" * 92)
    print("  Fixed-b coverage of a nominal 95% set, b = 0.05, dependence to")
    print(f"  lag delta + h = {BLOCK}, at the lengths Table S7 actually uses.\n")
    # Table S7's own rung lengths, read from lab27's printed panel rather than
    # typed: the list used to include 1,618, which is not one of them - the
    # nearest rung is 1,729 - so a table captioned "the lengths Table S7
    # actually uses" named a length it does not use.  The full evaluation
    # window is appended and LABELLED as such, because the comparison S10
    # withdrew was between two RUNGS and calling the longest rung "full
    # length" is what let the withdrawal be written up as 55% against 93%
    # when the compared pair is 367 days against 3,996.
    _l27 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "expected_output", "lab27_regime_conditioning.txt")
    _rungs, _l27txt = [], ""
    if os.path.isfile(_l27):
        _l27txt = open(_l27, encoding="utf-8").read()
        _rungs = sorted({int(m.group(1)) for m in re.finditer(
            r"^\s+\S.*?\s(\d{3,5}) days  \(", _l27txt, re.M)})
    if not _rungs:
        raise SystemExit("lab67 part A needs Table S7's rung lengths and "
                         "cannot find lab27's stored panel to read them from")
    print(f"  {'n':>7}{'M = 0.05n':>11}{'fixed-b cv':>13}{'coverage':>11}"
          f"   what it is")
    _cov = {}
    for n in _rungs:
        M = max(1, int(round(0.05 * n)))
        cv, cov = coverage(n, M, BLOCK, rng_a)
        _cov[n] = cov
        print(f"  {n:>7,}{M:>11}{cv:>13.2f}{cov*100:>10.1f}%   a Table S7 rung")
    for n in (len(_rungs) and 4861,):
        M = max(1, int(round(0.05 * n)))
        cv, cov = coverage(n, M, BLOCK, rng_a)
        _cov[n] = cov
        print(f"  {n:>7,}{M:>11}{cv:>13.2f}{cov*100:>10.1f}%   the full "
              f"evaluation window, which is NOT a rung")
    _sev, _calm = min(_rungs), max(_rungs)
    # The two rungs Section S10's grade rested on, identified BY NAME from
    # lab27's panel rather than typed: the severest rung it grades on and the
    # calmest.  Typing 367 and 3,996 here would survive a re-cut of the rungs
    # and quietly describe lengths the table no longer has.
    _named = {m.group(1).strip(): int(m.group(2)) for m in re.finditer(
        r"^\s{2,}(\S.*?)\s+(\d{3,5}) days  \(", _l27txt, re.M)}
    _sevk, _clmk = "VIX top 5%", "outside every episode"
    if _sevk not in _named or _clmk not in _named:
        raise SystemExit(f"lab27 no longer names the rungs Section S10 compares "
                         f"({_sevk!r}, {_clmk!r}); its panel lists "
                         f"{sorted(_named)}")
    _sev, _clm = _named[_sevk], _named[_clmk]
    print(f"\n  THE PAIR SECTION S10 COMPARED, and the reason it cannot be:")
    print(f"  the severest rung it grades on, '{_sevk}', holds {_sev:,} days "
          f"and the calmest,")
    print(f"  '{_clmk}', holds {_clm:,}.  They cover "
          f"{_cov[_sev] * 100:.0f}% and {_cov[_clm] * 100:.0f}% respectively, "
          f"so a")
    print(f"  {_cov[_sev] * 100:.0f}% set and a {_cov[_clm] * 100:.0f}% set are "
          f"being read off against each other.  The shortest rung in")
    print(f"  the table, {min(_rungs):,} days, covers "
          f"{_cov[min(_rungs)] * 100:.0f}%, and the full window "
          f"{_cov[4861] * 100:.0f}% - quoted so the")
    print(f"  gap is not read off the extremes of a list that includes lengths "
          f"the graded")
    print(f"  comparison never used.")
    # One rounding, in one place.  The table above prints a decimal and the
    # documents quote a whole number, and rounding the decimal is not the same
    # as rounding the figure it came from: 65.47% prints as 65.5% and then as
    # 66%, against the 65% the measurement supports.  The integers the
    # documents quote are emitted here, from the raw coverages, and the
    # verifier reads this line instead of re-rounding the table.
    print(f"\n  rounded for the prose: "
          + "  ".join(f"{n:,}:{_cov[n] * 100:.0f}%" for n in sorted(_cov)))

    # ---------------- the panel ------------------------------------------
    D, peers, lag = L.build(folder, TARGET)
    a = D[TARGET].values
    sa = pd.Series(a)
    own = np.column_stack([sa.values, sa.rolling(5).mean().values,
                           sa.rolling(22).mean().values])
    P = D[peers].values
    n_all = len(D)
    y = np.full(n_all, np.nan)
    y[:-H] = a[H:]
    # The + 1 that every other lab's start index does not have.  It is the one
    # day the VIX lag below costs: the ladder is cut on VIX at t-1, so the
    # first evaluation day needs a predecessor.  That is why this lab evaluates
    # 4,861 days where the equity tables evaluate 4,862, and the difference is
    # one day at the start of the window and nothing else.
    start = L.MED + L.TRAIN + L.VAL + DELTA + 1 + H
    idx = np.arange(start, n_all - H)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])]
    yb = y[idx]
    f_own_d = IV.walk(own, [], y, idx, DELTA)
    f_own_0 = IV.walk(own, [], y, idx, 0)
    f_alt_d = IV.walk(own, [P], y, idx, DELTA)
    Lown = (yb - f_own_d) ** 2
    d1 = Lown - (yb - f_alt_d) ** 2
    d2 = Lown - (yb - f_own_0) ** 2
    n = len(d1)
    dates = D.index[idx]

    # ---------------- the ladder, exactly as lab27 defines it -------------
    #
    # .shift(1) IS THE LADDER'S DEFINITION, not a refinement of it.  lab27,
    # which this comment claims to follow and which produces the market-state
    # rows the paper reports, does `vix = ivdf["VIX"].shift(1)` under the
    # comment "VIX at t-1, for the real-time cut only".  Both papers state the
    # same thing in four places - "cut on a rolling VIX tercile at t-1", "VIX
    # terciles taken at t-1", "available to a forecaster in real time", "the
    # strictly admissible specification of Section 2".
    #
    # This file did not do it.  It read VIX at t and cut the ladder on the
    # closing level of the very day whose loss differential it was weighting,
    # while its own comment said it matched lab27.  So the paper's central
    # design commitment - that nothing enters a specification before a
    # forecaster could have had it - was stated and then not implemented, in
    # the one lab behind an ESTABLISHED claim.
    #
    # Correcting it moves the contrast from +38.8 to +42.0 points and the
    # maximum standardised contrast from +4.94 to +5.49: the claim is
    # stronger, not weaker, which is the only reason it survived unnoticed.
    vix = IV.load_iv(folder, "VIX").reindex(D.index).ffill().shift(1)
    sv = pd.Series(vix.values[idx], index=dates)

    def rung(q):
        thr = sv.rolling(252, min_periods=60).quantile(q)
        return np.asarray((sv > thr).fillna(False), dtype=bool)

    base = np.asarray(
        (sv <= sv.rolling(252, min_periods=60).quantile(2 / 3)).fillna(False), bool)
    cal = np.zeros(n, bool)
    core = np.zeros(n, bool)
    for _nm, lo, hi in EPISODES:
        m = (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
        cal |= np.asarray(m)
        w = np.where(np.asarray(m))[0]
        if len(w):
            pk = w[int(np.argmax(sv.to_numpy()[w]))]
            core[max(0, pk - 20):min(n, pk + 21)] = True
    LADDER = [("VIX top 33%", rung(2 / 3), 1), ("calendar episodes", cal, 2),
              ("VIX top 10%", rung(0.90), 3), ("episodes, peak +/-20d", core, 4),
              ("VIX top 5%", rung(0.95), 5)]

    print()
    print("=" * 92)
    print("B.  THE CONTRAST, ONE PARAMETER ON THE FULL SAMPLE")
    print("=" * 92)
    M = int(round(0.05 * n))
    # The critical value from the function that simulates it, at THIS file's own
    # evaluation length, rather than the two-decimal figure the paper quotes.
    # Typing it here made a second spelling of one constant, and the two had
    # already drifted apart by the time anyone looked.
    z = H66.har_cv(n, M)
    print(f"  evaluation days {n:,}   baseline 'VIX bottom 67%' {base.sum():,}"
          f"   bandwidth M = {M}\n")
    print(f"  {'quantity':<28}{'stressed':>10}{'base':>9}{'contrast':>10}"
          f"{'se':>7}   fixed-b interval")
    top5 = LADDER[-1][1]
    for lab_, num, den in (("share of domestic MSE", d1, Lown),
                           ("substitution rate R(55)", d1, d2)):
        ra, rb, dl, se = contrast(num * top5, den * top5, num * base, den * base, M)
        print(f"  {lab_:<28}{ra*100:>9.1f}%{rb*100:>8.1f}%{dl*100:>10.1f}{se*100:>7.1f}"
              f"   [{(dl-z*se)*100:+6.1f}, {(dl+z*se)*100:+6.1f}]"
              f"{'   separated' if abs(dl/se)>z else '   NOT separated'}")
    print("\n  The claim is about the first. An interval for the first cannot")
    print("  grade a claim about the second, and the second does not separate.")

    print()
    print("=" * 92)
    print("C.  THE LADDER AS ORDERED ROBUSTNESS, JOINT BLOCK BOOTSTRAP")
    print("=" * 92)
    cols = np.column_stack([d1, Lown] + [m.astype(float) for _, m, _ in LADDER]
                           + [base.astype(float)])
    # ceil, and an inclusive upper end on the start.  Both halves of this were
    # wrong and both are the same defect lab68 carried: n // BLOCK draws too
    # few blocks, so the replication is shorter than the sample the observed
    # statistic was computed on, and integers(0, n - BLOCK) is exclusive, so
    # starts ran [0, n-BLOCK-1] and the LAST observation was never drawn.  A
    # sweep of every bootstrap in labs/ found these two sites and no others:
    # every other lab already used int(np.ceil(n / block)) and n - block + 1.
    # The effect here is one day in 4,861, which is why it survived - and the
    # reason to fix it is that a reader cannot tell a harmless off-by-one from
    # a harmful one without checking, and should not have to.
    nb = int(np.ceil(n / BLOCK))
    boot = np.full((NBOOT, len(LADDER)), np.nan)
    for b in range(NBOOT):
        st = rng.integers(0, n - BLOCK + 1, nb)
        S = np.concatenate([cols[s:s + BLOCK] for s in st])[:n]
        nu, de = S[:, 0], S[:, 1]
        bm = S[:, -1] > 0.5
        for k in range(len(LADDER)):
            mk = S[:, 2 + k] > 0.5
            if mk.sum() < 20 or bm.sum() < 20 or de[mk].sum() <= 0 or de[bm].sum() <= 0:
                continue
            boot[b, k] = nu[mk].sum() / de[mk].sum() - nu[bm].sum() / de[bm].sum()
    obs = np.array([float(d1[m].sum() / Lown[m].sum() - d1[base].sum() / Lown[base].sum())
                    for _, m, _ in LADDER])
    sds = np.nanstd(boot, axis=0, ddof=1)
    print(f"  {'rung':<24}{'rank':>5}{'days':>7}{'contrast':>10}{'boot sd':>9}"
          f"{'standardised':>14}")
    for (tag, m, rk), o, sg in zip(LADDER, obs, sds):
        print(f"  {tag:<24}{rk:>5}{int(m.sum()):>7,}{o*100:>9.1f}{sg*100:>9.1f}{o/sg:>14.2f}")
    good = ~np.isnan(boot).any(axis=1)
    Bv = boot[good]
    cen = Bv - obs
    ranks = np.array([r for _, _, r in LADDER], float)
    rho_o = float(np.corrcoef(ranks, obs)[0, 1])
    slope_o = float(np.polyfit(ranks, obs, 1)[0])
    tmax_o = float(np.max(obs / sds))
    rho_n = np.array([np.corrcoef(ranks, c)[0, 1] for c in cen])
    slope_n = np.array([np.polyfit(ranks, c, 1)[0] for c in cen])
    tmax_n = np.max(cen / sds, axis=1)
    # (1 + #{null >= observed}) / (B + 1).  When the count is zero this is the
    # SMALLEST value B replications can return, so it is the test's resolution
    # and not a measurement, and the helper below says which it is - the same
    # distinction lab68 draws, and for the same reason: "p = 0.0002" invites a
    # reader to treat a floor as an estimate.
    def pr(nullv, o):
        return (1 + int((nullv >= o).sum())) / (len(nullv) + 1)

    def prs(nullv, o):
        _h = int((nullv >= o).sum())
        _p = (1 + _h) / (len(nullv) + 1)
        return (f"<= {_p:.4f}  (no replication of {len(nullv)} reached it, so "
                f"this is the test's resolution)" if _h == 0 else f"= {_p:.4f}")
    d_prim = obs[-1]
    null_prim = cen[:, -1]
    print(f"\n  PRIMARY, pre-specified: share(VIX top 5%) - share(bottom 67%)")
    print(f"     contrast {d_prim*100:+.1f} points, bootstrap sd {sds[-1]*100:.1f},"
          f" percentile [{np.quantile(Bv[:,-1],0.025)*100:+.1f},"
          f" {np.quantile(Bv[:,-1],0.975)*100:+.1f}]")
    print(f"     one-sided resampling p for contrast > 0   "
          f"{prs(null_prim, d_prim)}")
    print(f"\n  SECONDARY, the ladder as ordered evidence")
    print(f"     Pearson(severity rank, share)   {rho_o:+.3f}   p = {pr(rho_n, rho_o):.4f}")
    print(f"     ordered-state slope             {slope_o*100:+.1f} pts/rank"
          f"   p = {pr(slope_n, slope_o):.4f}")
    print(f"     MAX standardised contrast       {tmax_o:+.2f}   "
          f"p {prs(tmax_n, tmax_o)}")
    print("\n  The last prices the freedom to report the strongest rung, which is")
    print("  the freedom lab27's post-hoc tightening created. It is the figure")
    print("  to quote: it concedes the search and still carries the claim.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
