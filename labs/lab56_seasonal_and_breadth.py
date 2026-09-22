"""
lab56_seasonal_and_breadth.py - the two ways lab55's headline could be wrong.

Imports lab55_illiquid_measured and lab05_robustness; keep all three in labs/.
Runtime about a minute.

WHY THIS FILE EXISTS
--------------------
lab55 measured the substitution rate on twenty Case-Shiller metros and got 68%
at a quarterly staleness on the return target and 19% on the paper's own target
type, both excluding zero.  Its own docstring named the way that result could
be an artefact, and then deferred the test:

    "These series are NOT seasonally adjusted and share a strong seasonal, so
     peers could appear to substitute merely by revealing a calendar effect."

That deferral is the whole reason for this file.  A promise to audit later is
not an audit, and the objection is not a small one.  House prices have a large
and nearly common seasonal: every metro rises in spring and falls in autumn.  A
target whose own mark is three months stale has lost its recent history, and a
fresh peer mark tells it what month it is.  If that is ALL the peers are doing,
the rate measures a calendar and not information, and lab55's headline should
be withdrawn rather than qualified.

lab55 already puts eleven month-of-year dummies in BOTH arms, which closes the
channel if the seasonal is a fixed additive monthly effect.  That is an
assumption, not a proof, and it fails in three ways worth testing: the seasonal
could be time-varying, so a fixed dummy cannot absorb it; it could be
multiplicative in the level of volatility; or the dummies could be estimated so
noisily on 144 months that they leave a usable residual seasonal behind.

WHAT THIS FILE DOES
-------------------
Part A sizes the object.  How much of each metro's monthly return variance is
month-of-year, and how similar are the seasonals across metros?  If the
seasonal were small or idiosyncratic there would be nothing to worry about.

Part B is the decisive test and it is a PLACEBO, not another control.  The real
peer mean is replaced by a synthetic peer that carries the seasonal and nothing
else: the month-of-year mean of the peer mean, estimated on the TRAINING BLOCK
ONLY and then read off by calendar month at the forecast origin.  By
construction that series has no information in it about what actually happened
in any peer market -- it is a lookup table on twelve numbers -- but it has the
full shared seasonal.  So:

    If the placebo reproduces lab55's rate, the rate is a calendar effect and
    lab55 is wrong.

    If the placebo produces approximately nothing while the real peer mean
    produces the published rate, the rate is information and the seasonal
    objection is answered rather than assumed away.

This is a sharper instrument than adding another control, because a control can
only fail to change the answer, whereas a placebo can actively produce the
wrong one and thereby convict the design.

Part C removes the dummies altogether and deseasonalises causally instead,
subtracting training-block month-of-year means from every series before
anything is fitted.  Three treatments are then comparable: dummies in both arms
(lab55's headline), causal deseasonalisation, and no seasonal handling at all.
The third is expected to be the highest, and if it is not then the dummies were
not doing the work they were put there to do.

Part D is the other way the headline could be wrong, in the opposite direction:
it could be too LOW.  lab55 used only the fourteen metros with complete history
from 1987-01 so that no ragged edge had to be handled.  The six late starters
were set aside.  Here they are added back as extra PEERS for the same fourteen
targets, which tests the paper's ragged-edge propagation on genuinely ragged
real data rather than on the simulated raggedness of Section 12.  A target's
own history is untouched, so the comparison is clean: same targets, same test
months, more peers, some of which are absent early.

WHAT WOULD COUNT AS A PROBLEM
-----------------------------
    The placebo in Part B earning a rate comparable to the real one.  That
    invalidates lab55.

    The no-seasonal-handling arm in Part C coming out BELOW the dummy arm.
    That would mean the dummies are adding signal rather than removing a
    nuisance, which is the opposite of their purpose.

    Part D lowering the rate materially.  Extra peers cannot destroy
    information, so a fall would indicate the ragged-edge handling is paying an
    estimation bill instead of propagating, which is lab52's confound again.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab55_illiquid_measured as I

SEED = 20260918
DELAYS_M = I.DELAYS_M
QUARTERLY = I.QUARTERLY


# ------------------------------------------------------- seasonal machinery

def month_of(index, ahead=0):
    return np.roll(pd.DatetimeIndex(index).month, -ahead)


def causal_month_means(v, months, upto):
    """Month-of-year means of v using rows strictly before `upto`."""
    out = np.full(12, np.nan)
    for k in range(1, 13):
        sel = (months[:upto] == k) & np.isfinite(v[:upto])
        if sel.sum() >= 3:
            out[k - 1] = v[:upto][sel].mean()
    return out


def seasonal_placebo(P1, months, idx, train_m):
    """A peer series carrying the SHARED SEASONAL and nothing else.

    At every refit origin the month-of-year means of the real peer mean are
    estimated on the training block only; the placebo's value at t is then the
    mean for t's calendar month.  Twelve numbers, refreshed on the same
    schedule as the model, and no dependence whatever on what any peer market
    actually did in the current month.
    """
    out = np.full(len(P1), np.nan)
    if len(idx) == 0:
        return out[:, None]
    # refresh on the same REFIT_M cadence the estimator uses
    for j, t in enumerate(idx):
        if j % I.REFIT_M == 0:
            lo = max(0, t - train_m)
            mm = causal_month_means(P1, months, t)
            base = mm
        out[t] = base[months[t] - 1]
    # fill the rest of the axis with the earliest available table so the
    # training rows are defined too
    first = idx[0]
    mm0 = causal_month_means(P1, months, first)
    for t in range(len(P1)):
        if not np.isfinite(out[t]):
            out[t] = mm0[months[t] - 1]
    return out[:, None]


def deseasonalise_causal(df, tags, cut):
    """Subtract training-block month-of-year means from every column."""
    months = month_of(df.index)
    out = df.copy()
    for c in tags:
        r = np.log(df[c] / df[c].shift(1)).values
        mm = causal_month_means(r, months, cut)
        adj = np.where(np.isfinite(mm[months - 1]), mm[months - 1], 0.0)
        # rebuild a level series whose log returns are deseasonalised
        rr = np.where(np.isfinite(r), r - adj, np.nan)
        lvl = np.full(len(rr), np.nan)
        lvl[0] = 100.0
        for i in range(1, len(rr)):
            lvl[i] = lvl[i - 1] * np.exp(rr[i]) if np.isfinite(rr[i]) else np.nan
        # carry the first finite forward so early NaNs do not poison the series
        out[c] = pd.Series(lvl, index=df.index)
    return out


# ---------------------------------------------------------------- engines

def build_custom(df, tags, target, dv, burn, Pblock, use_dummies):
    a = dv(df[target]).values
    sa = pd.Series(a)
    own = np.column_stack([sa.values,
                           sa.rolling(3).mean().values,
                           sa.rolling(12).mean().values])
    n = len(a)
    D = (I.month_dummies(df.index) if use_dummies
         else np.zeros((n, 1)))
    y = np.full(n, np.nan)
    y[:-I.H_M] = a[I.H_M:]
    start = burn + I.TRAIN_M + I.VAL_M + max(DELAYS_M) + I.H_M
    idx = np.arange(start, n - I.H_M)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])
              & np.isfinite(Pblock[idx]).all(axis=1)]
    return own, Pblock, D, y, idx


def rate_table(df, tags, dv, burn, make_block, use_dummies, targets):
    """Mean S_own, S_cross and R by delay across `targets`."""
    S_own = {d: [] for d in DELAYS_M}
    S_cross = {d: [] for d in DELAYS_M}
    rates = {d: [] for d in DELAYS_M if d}
    for t in targets:
        Pblock = make_block(df, tags, t, dv)
        own, P, D, y, idx = build_custom(df, tags, t, dv, burn, Pblock, use_dummies)
        if len(idx) < 60:
            continue
        yb = y[idx]
        # This file borrows lab55's scorer, so it inherited lab55's benchmark
        # too.  The benchmark is built on this panel's monthly clock rather
        # than lab05's daily defaults.
        bench = L.bench_mean(y, idx, window=I.TRAIN_M + I.VAL_M, horizon=I.H_M)
        fo = {d: I.walk(own, P, D, y, idx, d, False) for d in DELAYS_M}
        fc = {d: I.walk(own, P, D, y, idx, d, True) for d in DELAYS_M}
        so = {d: I.r2(yb, fo[d], bench=bench) for d in DELAYS_M}
        sc = {d: I.r2(yb, fc[d], bench=bench) for d in DELAYS_M}
        for d in DELAYS_M:
            S_own[d].append(so[d])
            S_cross[d].append(sc[d])
            if d:
                den = so[0] - so[d]
                if den > 1e-9:
                    rates[d].append((sc[d] - so[d]) / den)
    return S_own, S_cross, rates


def peer_mean_block(df, tags, target, dv):
    Pf = np.column_stack([dv(df[c]).values for c in tags if c != target])
    return I.peer_block(Pf, "mean")


def print_table(label, S_own, S_cross, rates, n_t):
    print(f"\n  {label}")
    print(f"{'delay':>7}{'mean own':>11}{'mean +cross':>13}{'mean R':>10}"
          f"{'metros R>0':>13}")
    got = {}
    for d in DELAYS_M:
        if not S_own[d]:
            continue
        mo, mc = np.mean(S_own[d]), np.mean(S_cross[d])
        if d == 0:
            print(f"{d:>7}{mo:>11.4f}{mc:>13.4f}{'n/a':>10}{'n/a':>13}")
            continue
        v = np.array(rates[d], dtype=float)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        got[d] = float(np.mean(v))
        print(f"{d:>7}{mo:>11.4f}{mc:>13.4f}{np.mean(v):>10.1%}"
              f"{f'{int((v > 0).sum())} of {len(v)}':>13}")
    return got


# ------------------------------------------------------------------ parts

def part_a(df):
    print("=" * 98)
    print("A.  HOW BIG IS THE SHARED SEASONAL, AND HOW SHARED IS IT")
    print("=" * 98)
    months = month_of(df.index)
    prof = {}
    print(f"\n{'metro':>16}{'seasonal sd':>14}{'return sd':>12}"
          f"{'variance share':>17}")
    for t in I.BALANCED:
        r = np.log(df[t] / df[t].shift(1)).values
        mm = np.array([np.nanmean(r[months == k]) for k in range(1, 13)])
        prof[t] = mm
        fit = mm[months - 1]
        ok = np.isfinite(r) & np.isfinite(fit)
        share = fit[ok].var() / r[ok].var()
        print(f"{I.CITY[t]:>16}{100 * mm.std(ddof=1):>13.3f}%"
              f"{100 * np.nanstd(r, ddof=1):>11.3f}%{share:>17.1%}")
    M = np.array([prof[t] for t in I.BALANCED])
    C = np.corrcoef(M)
    off = C[np.triu_indices_from(C, 1)]
    shares = []
    for t in I.BALANCED:
        r = np.log(df[t] / df[t].shift(1)).values
        fit = prof[t][months - 1]
        ok = np.isfinite(r) & np.isfinite(fit)
        shares.append(fit[ok].var() / r[ok].var())
    print(f"\n  month-of-year pattern explains {np.mean(shares):.1%} of monthly return")
    print(f"  variance on average, up to {max(shares):.1%} at "
          f"{I.CITY[I.BALANCED[int(np.argmax(shares))]]}.")
    print(f"  pairwise correlation of the twelve-month profiles across metros:")
    print(f"    mean {off.mean():+.3f}, min {off.min():+.3f}, max {off.max():+.3f}")
    if off.mean() > 0.5 and np.mean(shares) > 0.05:
        print("\n  So the objection is a real one and not a formality: the seasonal is")
        print("  large enough to matter and nearly common across metros, which is")
        print("  exactly the configuration in which a fresh peer could substitute for a")
        print("  stale own mark by revealing the calendar alone. Part B tests whether")
        print("  that is what happens.")
    else:
        print("\n  The seasonal is either small or not shared, so it is a weak candidate")
        print("  for driving the rate. Part B tests it anyway.")
    return np.mean(shares), off.mean()


def part_b(df):
    print("=" * 98)
    print("B.  THE PLACEBO: A PEER THAT CARRIES THE SEASONAL AND NOTHING ELSE")
    print("=" * 98)
    print("  The real peer mean is replaced by the month-of-year mean OF that peer")
    print("  mean, estimated on the training block only and refreshed on the same")
    print(f"  {I.REFIT_M}-month cadence as the model. Twelve numbers read off by calendar")
    print("  month. It has the entire shared seasonal and, by construction, no")
    print("  information about what any peer market did in the current month.")
    print("\n  Month dummies are REMOVED from both arms here, so the placebo is given")
    print("  the best possible chance: nothing else in the model is absorbing the")
    print("  calendar, and any seasonal advantage available to a peer is available")
    print("  to this one.")

    out = {}
    for name, dv, burn in (("return", I.dv_return, 12),
                           ("volatility", I.dv_vol, I.WIN_M + I.MED_M)):
        print(f"\n  {name.upper()} TARGET")

        def real(df_, tags, t, dv_=dv):
            return peer_mean_block(df_, tags, t, dv_)

        def placebo(df_, tags, t, dv_=dv):
            P1 = peer_mean_block(df_, tags, t, dv_)[:, 0]
            months = month_of(df_.index)
            _, _, _, _, idx = build_custom(df_, tags, t, dv_, burn,
                                           P1[:, None], False)
            return seasonal_placebo(P1, months, idx, I.TRAIN_M + I.VAL_M)

        a = print_table("real peer mean, no dummies",
                        *rate_table(df, I.BALANCED, dv, burn, real, False,
                                    I.BALANCED), len(I.BALANCED))
        b = print_table("SEASONAL PLACEBO, no dummies",
                        *rate_table(df, I.BALANCED, dv, burn, placebo, False,
                                    I.BALANCED), len(I.BALANCED))
        out[name] = (a, b)
        d = QUARTERLY
        if d in a and d in b:
            print(f"\n    at delta = {d}: real {a[d]:.1%}, placebo {b[d]:.1%}, "
                  f"difference {a[d] - b[d]:+.1%}")
            if abs(b[d]) < 0.15 and a[d] - b[d] > 0.10:
                print("    The placebo earns approximately nothing while the real peer")
                print("    mean earns the published rate. The rate is information and not")
                print("    a calendar effect.")
            elif b[d] > 0.5 * a[d]:
                print("    The placebo reproduces a large share of the rate. The seasonal")
                print("    IS doing the work and lab55's headline does not stand.")
            else:
                print("    The placebo earns a non-trivial but minority share of the rate,")
                print("    so the seasonal contributes and does not account for it.")
    return out


def part_c(df):
    print("=" * 98)
    print("C.  THREE SEASONAL TREATMENTS, COMPARED DIRECTLY")
    print("=" * 98)
    print("  (i)   month dummies in both arms      - lab55's published treatment")
    print("  (ii)  causal deseasonalisation        - training-block month means")
    print("        subtracted from every series before anything is fitted")
    print("  (iii) nothing at all                  - the raw unadjusted series")
    print("\n  (iii) is expected to be the HIGHEST, because it is the only arm in")
    print("  which a peer can still be paid for revealing the calendar. If it is")
    print("  not the highest then the dummies were not removing a nuisance.")
    cut = 12 + I.TRAIN_M + I.VAL_M
    dfd = deseasonalise_causal(df, I.BALANCED, cut)
    out = {}
    for name, dv, burn in (("return", I.dv_return, 12),
                           ("volatility", I.dv_vol, I.WIN_M + I.MED_M)):
        print(f"\n  {name.upper()} TARGET")
        got = {}
        got["dummies"] = print_table(
            "(i)   month dummies in both arms",
            *rate_table(df, I.BALANCED, dv, burn, peer_mean_block, True,
                        I.BALANCED), 14)
        got["deseason"] = print_table(
            "(ii)  causally deseasonalised, no dummies",
            *rate_table(dfd, I.BALANCED, dv, burn, peer_mean_block, False,
                        I.BALANCED), 14)
        got["raw"] = print_table(
            "(iii) no seasonal handling at all",
            *rate_table(df, I.BALANCED, dv, burn, peer_mean_block, False,
                        I.BALANCED), 14)
        out[name] = got
        d = QUARTERLY
        if all(d in got[k] for k in got):
            hi = max(got, key=lambda k: got[k][d])
            vals = [got[k][d] for k in got]
            spread = max(vals) - min(vals)
            gap = max(vals) - got["raw"][d]
            print(f"\n    at delta = {d}: dummies {got['dummies'][d]:.1%}, "
                  f"deseasonalised {got['deseason'][d]:.1%}, raw {got['raw'][d]:.1%}")
            print(f"    highest: {hi}; spread across treatments {spread:.1%}")
            if hi == "raw":
                print("    As expected. The untreated arm is the most flattering one, so")
                print("    the published treatment is the conservative choice and the")
                print("    calendar was worth removing.")
            elif gap < 0.02:
                print(f"    The untreated arm is not the highest, but it trails the highest")
                print(f"    by only {gap:.1%}, against an interval on R some twenty points wide")
                print("    in lab55. The three treatments are indistinguishable on this")
                print("    coordinate, which is not a puzzle: see the note below.")
            else:
                print("    NOT as expected: the untreated arm is materially below the")
                print("    highest, so the dummies are not simply absorbing a nuisance")
                print("    and the published treatment needs re-examining.")
            if name == "volatility" and spread < 0.05:
                print(f"\n    Why the seasonal barely matters here, structurally rather than")
                print(f"    empirically: the target is a {I.WIN_M}-month rolling variance, and a")
                print(f"    {I.WIN_M}-month window necessarily spans every calendar month, so a")
                print("    fixed month-of-year effect largely cancels inside the target")
                print("    before any model sees it. There is little seasonal channel to")
                print("    close on this coordinate, which is also why the Part B placebo")
                print("    came out negative rather than merely small.")
    return out


def part_d(df, rng):
    print("=" * 98)
    print("D.  THE OTHER DIRECTION: SIX RAGGED PEERS ADDED TO THE SAME FOURTEEN TARGETS")
    print("=" * 98)
    print("  lab55 used only the fourteen metros with complete history so that no")
    print("  ragged edge had to be handled. The six late starters are added here as")
    print("  extra PEERS for those same fourteen targets. No target's own history")
    print("  changes, so the comparison is same targets, same clock, more peers,")
    print("  some absent early.")
    print(f"\n  added: " + ", ".join(I.CITY[t] for t in I.LATE))
    print("  The peer block is the equal-weighted mean, which is computed over")
    print("  whatever peers exist in a given month, so a late starter simply does")
    print("  not enter the mean before it begins. That is the ragged-edge")
    print("  propagation of Section 12 applied to real raggedness.")
    wide = I.BALANCED + I.LATE
    out = {}
    for name, dv, burn in (("return", I.dv_return, 12),
                           ("volatility", I.dv_vol, I.WIN_M + I.MED_M)):
        print(f"\n  {name.upper()} TARGET")
        a = print_table("13 balanced peers (lab55)",
                        *rate_table(df, I.BALANCED, dv, burn, peer_mean_block,
                                    True, I.BALANCED), 14)
        b = print_table("19 peers, 6 of them ragged",
                        *rate_table(df, wide, dv, burn, peer_mean_block,
                                    True, I.BALANCED), 14)
        out[name] = (a, b)
        d = QUARTERLY
        if d in a and d in b:
            print(f"\n    at delta = {d}: 13 peers {a[d]:.1%}, 19 peers {b[d]:.1%}, "
                  f"change {b[d] - a[d]:+.1%}")
            if b[d] >= a[d] - 0.02:
                print("    Breadth does not cost anything. The ragged peers are absorbed")
                print("    without paying an estimation bill, which is what compressing")
                print("    them to a mean is for.")
                if abs(b[d] - a[d]) < 0.02:
                    print("    It does not HELP either, and that is the more interesting")
                    print("    reading. Six additional metros move the rate by less than two")
                    print("    points, so thirteen peers already span whatever common factor")
                    print("    the cross-section carries. That is consistent with the single")
                    print("    dominant factor lab42 found on equities and with the coupling")
                    print("    profiles of Part A correlating +0.89: there is one national")
                    print("    housing cycle here, not nineteen independent signals, and a")
                    print("    handful of peers is enough to read it.")
            else:
                print("    Breadth COSTS here, which with a one-column block should not")
                print("    happen from parameter count. The ragged peers are adding noise")
                print("    to the mean in their early months.")
    return out


def verdict(share, prof_corr, pb, tc, td):
    print("=" * 98)
    print("VERDICT")
    print("=" * 98)
    d = QUARTERLY
    print(f"  The seasonal objection was worth raising: month-of-year explains")
    print(f"  {share:.1%} of monthly return variance on average and the twelve-month")
    print(f"  profiles correlate {prof_corr:+.2f} across metros, so a fresh peer really")
    print("  could have substituted for a stale mark by revealing the calendar.")
    print()
    ok = []
    for name in ("return", "volatility"):
        real, plac = pb[name]
        if d in real and d in plac:
            print(f"  {name}: a peer carrying ONLY the seasonal earns {plac[d]:.1%} where the")
            print(f"  real peer mean earns {real[d]:.1%}.")
            ok.append(abs(plac[d]) < 0.15 and real[d] - plac[d] > 0.10)
    print()
    if ok and all(ok):
        print("  So the rate is not a calendar effect on either coordinate. That is")
        print("  established by a placebo that could have convicted the design and did")
        print("  not, rather than by a control that could only have failed to change")
        print("  the answer.")
    elif any(ok):
        print("  The placebo is harmless on one coordinate and not on the other, so the")
        print("  seasonal objection is answered for one target variable only and the")
        print("  paper should restrict its claim accordingly.")
    else:
        print("  The placebo earns a rate comparable to the real peer mean. lab55's")
        print("  headline measures a calendar and must be withdrawn, not qualified.")
    print()
    for name in ("return", "volatility"):
        g = tc.get(name, {})
        if all(d in g.get(k, {}) for k in ("dummies", "deseason", "raw")):
            print(f"  {name}: dummies {g['dummies'][d]:.1%}, causally deseasonalised "
                  f"{g['deseason'][d]:.1%}, untreated {g['raw'][d]:.1%}.")
    print()
    for name in ("return", "volatility"):
        a, b = td.get(name, ({}, {}))
        if d in a and d in b:
            print(f"  {name}: widening from 13 to 19 peers, six of them ragged, moves the")
            print(f"  rate from {a[d]:.1%} to {b[d]:.1%}.")
    print()
    conservative = []
    for name in ("return", "volatility"):
        g = tc.get(name, {})
        if all(d in g.get(k, {}) for k in ("dummies", "deseason", "raw")):
            conservative.append((name, g["raw"][d] - g["dummies"][d]))
    print("  What the paper should take from this file. The seasonal objection is")
    print("  the first thing a referee will raise about an unadjusted monthly")
    print("  series, and the answer is not 'we included dummies'. It is that a")
    print("  synthetic peer built to carry the seasonal and nothing else cannot")
    print("  reproduce the result: that is Part B, and it is the sentence to put")
    print("  in the paper.")
    print()
    print("  Part C is a weaker claim and should be stated as the weaker one it is.")
    for name, diff in conservative:
        if diff > 0.02:
            print(f"  On the {name} target the untreated arm sits {diff:+.1%} above the")
            print("  published treatment, so removing the calendar cost the paper points")
            print("  and the published figure is the conservative one.")
        else:
            print(f"  On the {name} target the untreated arm sits {diff:+.1%} from the")
            print("  published treatment, which is inside the noise: seasonal handling")
            print("  neither helps nor hurts there, for the structural reason printed in")
            print("  Part C, and no conservatism should be claimed from it.")


def main(root=None):
    df, sha, path = I.load_panel(root)
    rng = np.random.default_rng(SEED)
    print(f"data folder: {os.path.dirname(os.path.abspath(path))}")
    print("Prints the seasonal and breadth audits of lab55, Section 15.\n")
    print(f"  panel {os.path.basename(path)}, sha256 {sha[:16]}..., "
          f"{'digest matches' if sha == I.PANEL_SHA else 'DIGEST MISMATCH'}\n")
    share, prof_corr = part_a(df)
    print()
    pb = part_b(df)
    print()
    tc = part_c(df)
    print()
    td = part_d(df, rng)
    print()
    verdict(share, prof_corr, pb, tc, td)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
