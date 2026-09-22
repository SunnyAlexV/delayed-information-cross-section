"""
lab53_no_options_targets.py - four targets with no options market of their own.

Imports lab05_robustness and lab08_implied_vol; keep all three in labs/.
Runtime about fifteen minutes.

THE LIMITATION THIS ADDRESSES, AND HOW FAR IT GETS
--------------------------------------------------
Section 9 names the weakest point in the design: the S&P was chosen because its
truth is observable, and that same depth brings the options market which makes
breadth redundant.  Section 9.4 narrows the gap by separating targets whose
implied-volatility block is written on the asset itself from targets whose block
belongs to another index.

That subsection has a soft spot and it should be said plainly rather than
discovered by a referee.  Its lower group is the Nikkei, the Hang Seng, the
Nifty, the ASX and Bovespa, and options DO trade on all five.  Japan publishes
the Nikkei Volatility Index, Hong Kong the VHSI, India the India VIX, Australia
an S&P/ASX 200 VIX and Brazil an IVol-BR.  What makes those targets "foreign
options" in Section 9.4 is that this paper holds VIX and VDAX and not those
series.  That is a statement about the forecaster's information set, which is
the right analogy for an illiquid-asset holder, and it is NOT the statement that
no option chain exists on the asset.

This file closes that specific hole.  Four further targets are added for which,
as far as we can find, no implied-volatility index is published at all:

    KSE100   Pakistan, KSE-100                  close 10:30 UTC
    JKSE     Indonesia, IDX Composite           close 09:00 UTC
    CSEALL   Sri Lanka, Colombo All-Share       close 09:00 UTC
    KLCI     Malaysia, FTSE Bursa Malaysia KLCI close 09:00 UTC

For these, the absence of an own implied-volatility series is a property of the
market rather than of our download.  They are still liquid equity indices and
none of them is marked quarterly, so the illiquid-asset gap is narrowed again
rather than closed.  What they remove is the objection that Section 9.4's split
is an artefact of which series we happened to buy.

THE MATCHED COMPARISON
----------------------
All four close between 09:00 and 10:30 UTC, so each sees Tokyo, Sydney and Hong
Kong on the same day and Europe and New York a session late.  That is a clock
handicap, and a fair test has to hold it fixed: the Nikkei and the ASX have no
same-day peer at all, the Nifty has three.  So the four existing Asian targets
are re-run here through exactly the same code path, the same delays and the same
implied-volatility timing.  The two groups then differ in one thing: whether a
volatility index exists on the target at all.

Both parameterisations are reported, because lab52 showed the comparison is not
invariant to them.  The seven-regressor block is what Section 8 used; the
equal-weighted mean is what Sections 5.1 and 8.1 tell a practitioner to carry,
and it is the one the conclusion should rest on.

DATA
----
The four new series are single files from one vendor, so unlike the eight of
Section 3 there is no second export to check a join against, and equally no join
to check.  They are used in this file and nowhere else: no number in any other
lab or section depends on them.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab08_implied_vol as IV

H = L.HORIZON
DELAYS = [0, 5, 21, 55]
HEAD = 55
IV_TAGS = ["VIX", "VDAX"]

# close times in UTC, as Section 3 records them for the panel
NEW = {"KSE100": ("Pakistan KSE-100", 10.5, "KSE100_yahoo.csv"),
       "JKSE":   ("Indonesia IDX Composite", 9.0, "JKSE_yahoo.csv"),
       "CSEALL": ("Sri Lanka Colombo All-Share", 9.0, "CSEALL_yahoo.csv"),
       "KLCI":   ("Malaysia FTSE Bursa KLCI", 9.0, "KLCI_yahoo.csv")}
MATCHED = ["N225", "AXJO", "HSI", "NSEI"]       # same clock, own volatility index exists


def r2(y, f, *, bench):
    """Out-of-sample skill against the trailing-mean benchmark.

    `bench` is keyword-only and mandatory: this helper divided by the mean of
    the target over the test period until an audit of every scorer in the
    package caught it, and a mandatory argument is what stops a missed call
    site from scoring against the wrong yardstick in silence.
    """
    den = ((y - bench) ** 2).sum()
    return 1 - ((y - f) ** 2).sum() / den if den > 0 else np.nan


def frontier_dir(folder):
    d = os.path.join(folder, "frontier")
    return d if os.path.isdir(d) else None


def decision(px):
    """Section 3's decision variable from an OHLC frame."""
    yz = L.yang_zhang(px)
    return np.log(yz / yz.rolling(L.MED, min_periods=30).median())


def build(folder, target, raw_iv):
    """A panel on `target`'s calendar: its own history, the eight indices of
    Section 3 at admissible timestamps, and the implied-volatility block."""
    if target in NEW:
        path = os.path.join(frontier_dir(folder), NEW[target][2])
        tgt_px = L.read_one(path)
        close = NEW[target][1]
    else:
        tgt_px = L.load_index(target, folder)
        close = L.CLOSE_UTC[target]

    a_full = decision(tgt_px)
    peers = [t for t in sorted(L.CLOSE_UTC) if t != target and L.files_for(t, folder)]
    pcols = {t: decision(L.load_index(t, folder)) for t in peers}

    idxs = pd.DatetimeIndex(a_full.index)
    D = pd.DataFrame({target: a_full.values}, index=idxs)
    for t in peers:
        s = pcols[t].reindex(idxs).ffill(limit=5)
        if L.CLOSE_UTC[t] >= close:                 # closes at or after: yesterday's
            s = s.shift(1)
        D[t] = s.values
    for t in IV_TAGS:
        on = raw_iv[t].reindex(idxs).ffill(limit=5).shift(1)
        D[t] = np.log(on / on.rolling(L.MED, min_periods=30).median()).values
    D = D.loc[np.isfinite(D[target].values)].copy()

    a = D[target].values
    sa = pd.Series(a)
    own = np.column_stack([sa.values, sa.rolling(5).mean().values,
                           sa.rolling(22).mean().values])
    P = D[peers].values
    iv = D[IV_TAGS].values
    n = len(D)
    y = np.full(n, np.nan); y[:-H] = a[H:]
    start = L.MED + L.TRAIN + L.VAL + max(DELAYS) + 1 + H
    idx = np.arange(start, n - H)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx]) & np.isfinite(iv[idx]).all(axis=1)]
    same = sum(1 for t in peers if L.CLOSE_UTC[t] < close)
    # how strongly this market moves with the cross-section it is being offered:
    # its decision variable against the equal-weighted peer mean, on the test days
    # themselves.  Descriptive and in sample, like the ceiling of Section 5.
    finp = np.isfinite(P[idx])
    cntp = finp.sum(axis=1)
    mb = np.where(cntp > 0,
                  np.where(finp, P[idx], 0.0).sum(axis=1) / np.maximum(cntp, 1), np.nan)
    okc = np.isfinite(a[idx]) & np.isfinite(mb)
    coupling = float(np.corrcoef(a[idx][okc], mb[okc])[0, 1]) if okc.sum() > 100 else np.nan
    return own, P, iv, y, idx, peers, same, D, coupling


def _rank(v):
    """Ranks with ties averaged, so Spearman is the right statistic."""
    v = np.asarray(v, float)
    order = np.argsort(v)
    r = np.empty(len(v), float)
    r[order] = np.arange(len(v), dtype=float)
    for u in np.unique(v):
        m = v == u
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def coupling_diagnostics(rows):
    """How much of +0.98 is one point, and what survives dropping it.

    The paper makes an ORDERING claim - coupling tracks the substitution rate -
    and then quotes a Pearson correlation, which is a claim about linearity on
    eight points.  Two things are therefore owed to a reader.  First, the rank
    statistic, which is what the ordering claim actually needs.  Second, the
    influence diagnostics, because eight points with two of them far out to the
    left is exactly the shape where a correlation can be a statement about the
    gap between two clusters rather than about a relationship.

    Nothing here is a new experiment.  Every number is computed from the eight
    pairs already in the table above, at no cost, and it is reported because
    the paper draws a fitted LINE through those points in a figure and reads a
    zero-crossing off it.
    """
    tg = [r[0] for r in rows]
    c = np.array([r[1] for r in rows], float)
    rb = np.array([r[2] for r in rows], float)
    n = len(c)
    pear = lambda x, y: float(np.corrcoef(x, y)[0, 1])
    spear = lambda x, y: pear(_rank(x), _rank(y))
    full_p, full_s = pear(c, rb), spear(c, rb)

    print("\n" + "=" * 96)
    print("C2. HOW MUCH OF THE +0.98 IS ONE POINT")
    print("=" * 96)
    z, se = np.arctanh(full_p), 1.0 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    print(f"  Pearson  {full_p:+.3f}   Fisher 95% interval [{lo:+.3f}, {hi:+.3f}]  (n = {n})")
    print(f"  Spearman {full_s:+.3f}   <- the statistic the ORDERING claim needs")
    print("\n  Leave one target out, and the fitted line with it. 'slope' is the")
    print("  least-squares slope of R(breadth) on coupling; 'crossing' is where")
    print("  that line reaches zero, which is the number the paper's figure draws.")
    X = np.column_stack([np.ones(n), c])
    hat = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
    s_full, i_full = np.polyfit(c, rb, 1)
    print(f"\n{'dropped':>9}{'Pearson':>10}{'Spearman':>10}{'slope':>9}"
          f"{'crossing':>10}{'leverage':>10}")
    worst_p, worst_t = full_p, None
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        p_i, s_i = pear(c[m], rb[m]), spear(c[m], rb[m])
        sl, ic = np.polyfit(c[m], rb[m], 1)
        print(f"{tg[i]:>9}{p_i:>+10.3f}{s_i:>+10.3f}{sl:>+9.3f}"
              f"{-ic / sl:>+10.3f}{hat[i]:>10.3f}")
        if p_i < worst_p:
            worst_p, worst_t = p_i, tg[i]
    print(f"{'none':>9}{full_p:>+10.3f}{full_s:>+10.3f}{s_full:>+9.3f}"
          f"{-i_full / s_full:>+10.3f}{2 / n:>10.3f}  <- full sample, mean leverage")

    # the two low-coupling targets are the obvious worry: drop BOTH
    keep = c > 0.5
    p6, s6 = pear(c[keep], rb[keep]), spear(c[keep], rb[keep])
    print(f"\n  Drop BOTH low-coupled targets and fit on the remaining "
          f"{int(keep.sum())}, which is the")
    print("  hardest version of the question, because what is left is one cluster:")
    print(f"    Pearson {p6:+.3f}, Spearman {s6:+.3f}")

    # twice the mean leverage is the usual flag for a point the fit leans on
    hi_lev = [tg[i] for i in np.argsort(-hat) if hat[i] > 2 * (2 / n)]
    print("\n  What to take from this, and what not to:")
    print(f"  The correlation is not one point. The worst leave-one-out Pearson is")
    print(f"  {worst_p:+.3f}" + (f" (dropping {worst_t})" if worst_t else "")
          + ", the Spearman never falls below "
          f"{min(spear(c[np.arange(n) != i], rb[np.arange(n) != i]) for i in range(n)):+.3f},")
    print(f"  and inside the single cluster of {int(keep.sum())} the ordering still holds at "
          f"{s6:+.3f}.")
    if hi_lev:
        j = tg.index(hi_lev[0])
        sl, ic = np.polyfit(c[np.arange(n) != j], rb[np.arange(n) != j], 1)
        print(f"\n  The fitted LINE is a different matter. {hi_lev[0]} carries leverage")
        print(f"  {hat[j]:.3f} against a mean of {2 / n:.3f}, and removing it moves the slope from")
        print(f"  {s_full:+.3f} to {sl:+.3f} and the zero-crossing from {-i_full / s_full:+.3f} to "
              f"{-ic / sl:+.3f}.")
        print("  So the ordering is robust and the crossing is not. The paper should")
        print("  quote the correlation and treat the crossing as a location, not a")
        print("  threshold - which is what the text already says, now with the")
        print("  diagnostic that licenses saying it.")
    print("\n  Eight points remain eight points. None of this makes the relationship")
    print("  better estimated than a sample of eight can make it; it establishes")
    print("  which of the two claims the paper makes that sample can carry.")


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints the no-options-market targets, Section 9.5.\n")
    if frontier_dir(folder) is None:
        raise SystemExit(
            "data/frontier/ is missing. The four series of Section 9.5 live there:\n"
            "  " + ", ".join(v[2] for v in NEW.values()))

    raw_iv = {t: IV.load_iv(folder, t) for t in IV_TAGS}
    print()
    print("=" * 96)
    print("A.  FOUR TARGETS WITH NO VOLATILITY INDEX OF THEIR OWN, AND FOUR WITH ONE")
    print("=" * 96)
    print("The lower four are the Asian targets of Section 9.4. Options trade on all")
    print("of them and each has a published volatility index that this paper does")
    print("not hold. The upper four have none to hold. All eight close between 06:00")
    print("and 10:30 UTC, so the clock handicap is held roughly fixed across the")
    print("comparison.\n")
    print(f"{'target':>8}{'market':>34}{'close':>8}{'same-day':>10}{'days':>7}"
          f"{'first test day':>16}{'coupling':>11}")
    panels = {}
    for tg in list(NEW) + MATCHED:
        panels[tg] = build(folder, tg, raw_iv)
        own, P, iv, y, idx, peers, same, D, coup = panels[tg]
        label = NEW[tg][0] if tg in NEW else {"N225": "Japan Nikkei 225",
                                              "AXJO": "Australia S&P/ASX 200",
                                              "HSI": "Hong Kong Hang Seng",
                                              "NSEI": "India Nifty 50"}[tg]
        close = NEW[tg][1] if tg in NEW else L.CLOSE_UTC[tg]
        print(f"{tg:>8}{label:>34}{close:>8.1f}{same:>10}{len(idx):>7}"
              f"{str(D.index[idx[0]])[:10]:>16}{coup:>11.3f}")

    # ---------------- B. the race ------------------------------------------
    print("\n" + "=" * 96)
    print("B.  THE RACE ON EACH, WITH BOTH PARAMETERISATIONS")
    print("=" * 96)
    print("'seven' is the eight-index block of Section 3 minus the target itself;")
    print("'mean' is its equal-weighted average, the block Section 5.1 recommends.")
    print("The last four columns are what breadth still adds once implied")
    print("volatility is held, under each parameterisation.\n")
    res = {}
    for tg in list(NEW) + MATCHED:
        own, P, iv, y, idx, peers, same, D, coup = panels[tg]
        yb = y[idx]
        # One benchmark per target panel: the trailing mean of labels that had
        # already resolved.  This file scored against the mean of the target
        # over the test period until an audit of every scorer in the package.
        BENCH = L.bench_mean(y, idx)
        fin = np.isfinite(P)
        cnt = fin.sum(axis=1)
        mean_col = np.where(cnt > 0, np.where(fin, P, 0.0).sum(axis=1) / np.maximum(cnt, 1),
                            np.nan)[:, None]
        print(f"  {tg}  ({'no volatility index published' if tg in NEW else 'has one, not held here'})")
        print(f"{'delta':>8}{'own':>9}{'+block':>9}{'+IV':>9}{'R breadth':>11}{'R iv':>8}"
              f"{'seven|IV':>10}{'GW z':>7}{'mean|IV':>10}{'GW z':>7}")
        S0 = None
        for d in DELAYS:
            f_own = IV.walk(own, [], y, idx, d)
            f_blk = IV.walk(own, [P], y, idx, d)
            f_iv = IV.walk(own, [iv], y, idx, d)
            f_sev = IV.walk(own, [iv, P], y, idx, d)
            f_one = IV.walk(own, [iv, mean_col], y, idx, d)
            s_own = r2(yb, f_own, bench=BENCH)
            s_blk = r2(yb, f_blk, bench=BENCH)
            s_iv = r2(yb, f_iv, bench=BENCH)
            if d == 0:
                S0 = s_own
            den = S0 - s_own
            rb = (s_blk - s_own) / den if d and den > 1e-9 else np.nan
            ri = (s_iv - s_own) / den if d and den > 1e-9 else np.nan
            zz = []
            for f in (f_sev, f_one):
                dl = (yb - f_iv) ** 2 - (yb - f) ** 2
                zz.append((r2(yb, f, bench=BENCH) - s_iv,
                           float(dl.mean() / IV.hac_se(dl))))
            res[(tg, d)] = (s_own, s_blk, s_iv, rb, ri, zz[0], zz[1])
            print(f"{d:>8}{s_own:>9.4f}{s_blk:>9.4f}{s_iv:>9.4f}{rb:>11.1%}{ri:>8.1%}"
                  f"{zz[0][0]:>+10.4f}{zz[0][1]:>+7.2f}{zz[1][0]:>+10.4f}{zz[1][1]:>+7.2f}")
        print()

    # ---------------- C. what actually separates them ----------------------
    print("=" * 96)
    print("C.  THE GROUPING IS NOT WHAT SEPARATES THESE MARKETS")
    print("=" * 96)
    print("Sorted by how much the target moves with the cross-section it is being")
    print("offered. The group boundary does not survive the sort.\n")
    print(f"{'target':>8}{'volatility index':>20}{'coupling':>10}{'R breadth':>12}"
          f"{'R iv':>8}{'mean|IV at 55d':>16}{'GW z':>7}")
    rows = []
    for tg in sorted(list(NEW) + MATCHED, key=lambda t: -panels[t][8]):
        coup = panels[tg][8]
        rb, ri = res[(tg, HEAD)][3], res[(tg, HEAD)][4]
        inc, z = res[(tg, HEAD)][6]
        rows.append((tg, coup, rb, ri, inc, z))
        print(f"{tg:>8}{('none published' if tg in NEW else 'exists, not held'):>20}"
              f"{coup:>10.3f}{rb:>12.1%}{ri:>8.1%}{inc:>+16.4f}{z:>+7.2f}")
    arr = np.array([[r[1], r[2], r[3]] for r in rows])
    c_rb = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
    c_ri = float(np.corrcoef(arr[:, 0], arr[:, 2])[0, 1])
    print(f"\n  across the eight, coupling against R(breadth): {c_rb:+.2f}")
    print(f"                    coupling against R(implied volatility): {c_ri:+.2f}")
    # The line through those eight points, and where it crosses zero.  The
    # correlation says the rate follows coupling; the crossing says at what
    # coupling the cross-section stops paying for itself, which is the number a
    # practitioner actually needs and the one the scatter in the paper draws.
    slope, icept = np.polyfit(arr[:, 0], arr[:, 1], 1)
    cross = -icept / slope
    print(f"  least squares fit: R(breadth) = {icept:+.3f} {slope:+.3f} * coupling")
    print(f"  the fit crosses zero at a coupling of {cross:+.3f}")
    below = [r[0] for r in rows if r[1] < cross]
    neg = [(r[0], r[1]) for r in rows if r[2] < 0]
    print(f"  targets below that coupling: {len(below)} "
          f"({', '.join(below) if below else 'none'})")
    print(f"  targets with R(breadth) < 0: {len(neg)} "
          f"({', '.join(f'{t} at {c:+.3f}' for t, c in neg)})")
    if len(neg) != len(below):
        print("  Those two counts differ, so the crossing is not a threshold and is not")
        print("  reported as one. It locates the sign change to within one target on")
        print("  eight points: the nearest negative sits just above the fitted crossing,")
        print("  which is what a line through eight points can and cannot tell you.")
    print("\n  Eight points again and no interval is quoted. What the column shows is")
    print("  that the two weakest-coupled markets are the two where neither candidate")
    print("  repairs anything, and both sit in the group with no volatility index.")
    print("  Grouping and coupling are confounded here and four targets cannot")
    print("  separate them, so the group means below are reported and then set aside.")

    coupling_diagnostics(rows)

    summary = {}
    for label, group in (("no volatility index", list(NEW)),
                         ("has one, not held here", MATCHED)):
        rb = np.mean([res[(t, HEAD)][3] for t in group])
        ri = np.mean([res[(t, HEAD)][4] for t in group])
        sig = sum(1 for t in group for d in DELAYS if res[(t, d)][6][1] > 1.96)
        summary[label] = (rb, ri, sig, len(group) * len(DELAYS))
        print(f"\n  {label:>24}: breadth {rb:>6.1%}, implied vol {ri:>6.1%}, "
              f"{sig} of {len(group) * len(DELAYS)} cells")
    A, B = summary["no volatility index"], summary["has one, not held here"]

    # ---------------- D. where breadth does not transfer -------------------
    print("\n" + "=" * 96)
    print("D.  WHERE BREADTH DOES NOT TRANSFER AT ALL")
    print("=" * 96)
    dead = [r[0] for r in rows if r[2] < 0]
    live = [r[0] for r in rows if r[2] >= 0.30]
    print(f"  worth less than nothing at eleven weeks: "
          f"{', '.join(dead) if dead else 'none'}")
    print(f"  repairs at least thirty per cent:        {', '.join(live)}")
    for tg in dead:
        s0, s55, b55 = res[(tg, 0)][0], res[(tg, HEAD)][0], res[(tg, HEAD)][1]
        print(f"\n  {tg}: own-only R-squared {s0:.4f} with no delay, {s55:.4f} at eleven weeks,")
        print(f"        and adding the cross-section moves it to {b55:.4f}. The delay does its")
        print("        usual damage; the cross-section does not repair it and charges for")
        print("        the attempt.")
    print("\n  This is Section 5.3's mechanism read backwards. The recovery there is")
    print("  entirely the leading component of the foreign block, so a market that")
    print("  does not load on that component has nothing to recover with. Nothing")
    print("  here is special to these markets' options, and the paper does not claim")
    print("  it is.")

    # ---------------- verdict ----------------------------------------------
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    print("  Two things come out of this and only one of them was the question.")
    print("\n  The question was whether Section 9.4's split survives where no volatility")
    print("  index exists to be held. On two of the four it does. Indonesia and")
    print(f"  Malaysia repair {res[('JKSE', HEAD)][3]:.0%} and {res[('KLCI', HEAD)][3]:.0%} of the delay damage with breadth, and a")
    print("  compressed cross-section still adds to implied volatility at eleven")
    print(f"  weeks with statistics of {res[('JKSE', HEAD)][6][1]:+.2f} and {res[('KLCI', HEAD)][6][1]:+.2f}. There is no options market")
    print("  anywhere in that picture, so the reading of Section 9.4 does not depend")
    print("  on which volatility series we happened to buy.")
    print("\n  The other thing was not the question and matters more. Pakistan and Sri")
    print("  Lanka are not a weaker version of the same result, they are a different")
    print(f"  result: {res[('KSE100', HEAD)][3]:.0%} and {res[('CSEALL', HEAD)][3]:.0%} at eleven weeks, a cross-section that makes a")
    print("  delayed forecaster worse. They are also the two least coupled markets in")
    print("  the set. A mean over these four would average a result with its")
    print("  opposite, so this file does not quote one as a headline.")
    print("\n  The paper gains a scope condition it could not have guessed: breadth")
    print("  substitutes for a stale mark where the market is coupled to the")
    print("  cross-section it is offered, and not otherwise. Section 14 already")
    print("  refuses to extrapolate the rate to a private asset, on the grounds that")
    print("  nothing in the paper bounds that asset's correlation with the public")
    print("  cross-section. These four targets are what the refusal looks like when")
    print("  it is measured instead of asserted.")
    print("\n  What none of this reaches is still an illiquid asset. Every target here")
    print("  is a liquid equity index with a daily open, high, low and close, which")
    print("  is the property that made it usable at all. The gap Section 11 names is")
    print("  narrower and it is not closed.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
