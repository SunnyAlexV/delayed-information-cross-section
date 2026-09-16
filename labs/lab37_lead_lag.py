"""
lab37_lead_lag.py - prints Tables 16 and 17 and the clock test of Section 8.3.

Imports lab05_robustness and lab22_factor_benchmark; keep all three in labs/.
Runtime about a minute.

THE THING BOTH PAPERS ASSUME AND NEITHER TESTS
-----------------------------------------------
Every model in this project treats the foreign block as a set of CONTEMPORANEOUS
observations: seven closes, each the freshest one that precedes the target's,
entered at their admissible timestamp and never at any other.  That is a defensible
choice and it is also an untested one, because it quietly settles a question the
papers never ask.

Two stories fit the result equally well so far.

  PROPAGATION    volatility moves from market to market.  Tokyo is nervous today
                 and New York is nervous tomorrow.  The foreign block works
                 because it is EARLY: it contains tomorrow's domestic news.

  COMMON FACTOR  one global state drives every market at once, and the markets
                 merely report it at different clock times.  The foreign block
                 works because it is a SECOND READING of the same thing the stale
                 domestic mark used to supply.

The papers' headline - that foreign closes substitute for a stale domestic mark -
is true under both.  But the RECOMMENDATION differs sharply, and that is what
makes this worth a lab rather than a paragraph:

  under propagation, the value is in the timing, so a desk whose foreign feed is
  itself a day or two behind loses most of it;

  under a common factor, the value is in the cross-section, so the same desk
  loses almost nothing, and the paper's advice survives a much messier data
  environment than the one it was tested in.

Nobody has told a practitioner which world they are in.

WHAT IS MEASURED
----------------
A.  The cross-correlation structure, lead and lag, peer by peer.  A common factor
    observed at different clock times is roughly symmetric in the two directions;
    propagation is not.  The overlap caveat is stated in the section, not hidden.

B.  THE PRACTICAL ONE.  Age the foreign block itself.  Score the paper's model
    with peers held g days stale, g = 0, 1, 2, 3, 5, 10, at every domestic delay,
    and read the substitution rate as a surface rather than a column.  If the
    surface is flat in g, the recommendation is robust; if it collapses, the
    paper is quietly assuming a data feed most desks do not have.

C.  Where in time the global factor's information sits, and whether its CHANGE
    carries anything beyond its level.  Both comparisons hold the number of
    regressors fixed, following lab36, so lab07's estimation cost cancels
    instead of having to be subtracted.

WHAT THIS FILE DECIDED
----------------------
It was written as an exploratory check and became Section 8.3, because B
collapsed: one day of foreign staleness costs skill at nearly every delay, so
the paper's practical claim IS narrower than it read and the Limitations
section now says so.  A produced the opposite kind of result - an asymmetry
that looked like evidence of direction and turned out to be the trading clock -
and A2 is the test that separates them.  Neither outcome was the one expected
when the file was started, which is the reason to run a check rather than argue
about it.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab22_factor_benchmark as F

SEED = 20260915
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 3, 5, 13, 21, 55]
PEER_GAPS = [0, 1, 2, 3, 5, 10]        # how stale the FOREIGN feed is
FACT_LAGS = [0, 1, 2, 3, 5, 10]        # where the factor's information sits
KMAX = 5                               # lead/lag range for the correlations
N_BOOT, BLK = 2000, 10
N_DRAW = 6


# ------------------------------------------------------------------ data
def panel(folder):
    D, peers, lag = L.build(folder, TARGET)
    a = D[TARGET].values
    sa = pd.Series(a)
    own = np.column_stack([sa.values, sa.rolling(5).mean().values,
                           sa.rolling(22).mean().values])
    P = D[peers].values
    n = len(D)
    y = np.full(n, np.nan); y[:-H] = a[H:]
    start = L.MED + L.TRAIN + L.VAL + max(DELAYS) + max(PEER_GAPS) + H
    idx = np.arange(start, n - H)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])]
    return own, P, y, idx, D, peers, lag


# ------------------------------------------------------------- estimation
def walk(own, P, y, idx, delta, gap, lags=None, noise=None):
    """own held `delta` stale; the foreign block held `gap` stale.

    lags is None      -> the seven peer columns themselves
    lags is a list    -> one real-time PC1 factor at each of those lags
    noise             -> an extra uninformative column, for equal-width races
    """
    out = np.empty(len(idx)); b = mu = sd = None
    pm = psd = V = None

    def cols(rows):
        blocks = [own[rows - delta]]
        if lags is None:
            blocks.append(P[rows - gap])
        else:
            for g in lags:
                blocks.append((((P[rows - g] - pm) / psd) @ V[:, 0])[:, None])
        if noise is not None:
            blocks.append(noise[rows][:, None])
        return np.column_stack(blocks)

    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), max(cut, 1))
            if lags is not None:
                pm, psd, V, _ = F.basis(P[tr])
            X = cols(tr)
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y[tr])
            X, yy = X[ok], y[tr][ok]
            if len(yy) < 200:
                b = None; continue
            mu, sd = X.mean(0), X.std(0) + 1e-9
            b = L.cv((X - mu) / sd, yy, "cont")
        if b is None:
            out[j] = 0.0; continue
        xt = cols(np.array([t]))[0]
        out[j] = 0.0 if not np.isfinite(xt).all() \
            else L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
    return out


def boot_ci(yb, f_a, f_b, rng, n_boot=N_BOOT, block=BLK):
    d = (yb - f_b) ** 2 - (yb - f_a) ** 2
    n = len(d); nb = int(np.ceil(n / block)); offs = np.arange(block)
    st = rng.integers(0, n - block + 1, size=(n_boot, nb))
    o = np.array([d[(st[i][:, None] + offs).ravel()[:n]].mean() for i in range(n_boot)])
    lo, hi = np.percentile(o, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def corr_diff_ci(x1, y1, x2, y2, rng, n_boot=N_BOOT, block=BLK):
    """Paired block bootstrap of corr(x1,y1) - corr(x2,y2).

    The same resampled blocks drive both correlations, which is what makes the
    difference meaningful: the overlap in a rolling five-day variance inflates
    each correlation, and a paired draw lets that inflation cancel.
    """
    n = len(x1); nb = int(np.ceil(n / block)); offs = np.arange(block)
    st = rng.integers(0, n - block + 1, size=(n_boot, nb))
    o = np.empty(n_boot)
    for i in range(n_boot):
        s = (st[i][:, None] + offs).ravel()[:n]
        o[i] = np.corrcoef(x1[s], y1[s])[0, 1] - np.corrcoef(x2[s], y2[s])[0, 1]
    lo, hi = np.percentile(o, [2.5, 97.5])
    return (float(np.corrcoef(x1, y1)[0, 1] - np.corrcoef(x2, y2)[0, 1]),
            float(lo), float(hi))


def ar1_surrogates(v, rng, n_draw=N_DRAW):
    fin = np.isfinite(v)
    x = v[fin]; m = x.mean()
    v0, v1 = x[:-1] - m, x[1:] - m
    phi = float((v0 @ v1) / (v0 @ v0))
    sd = float(np.std(v1 - phi * v0))
    out = []
    for _ in range(n_draw):
        s = np.empty(len(v)); e = rng.normal(scale=sd, size=len(v))
        s[0] = m + e[0] / np.sqrt(max(1 - phi ** 2, 1e-6))
        for t in range(1, len(v)):
            s[t] = m + phi * (s[t - 1] - m) + e[t]
        s[~fin] = np.nan
        out.append(s)
    return out, phi


# ------------------------------------------------------------------ main
def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints Tables 16 and 17 and the clock test of Section 8.3.\n")

    own, P, y, idx, D, peers, lag = panel(folder)
    yb = y[idx]
    print(f"target {TARGET}, {len(idx)} test days, "
          f"{D.index[idx[0]].date()} to {D.index[idx[-1]].date()}, "
          f"{len(peers)} foreign peers")
    print("admissibility already applied: a peer closing at or after New York enters")
    print(f"at its PREVIOUS session.  Shifted by one day: "
          f"{[p for p in peers if lag[p]] or 'none'}\n")

    # ---------------- A ---------------------------------------------------
    print("=" * 102)
    print("A.  WHICH WAY DOES IT RUN?")
    print("=" * 102)
    print("corr(peer at t-k, target at t) against corr(target at t-k, peer at t).")
    print("A single global state read at different clock times is roughly symmetric;")
    print("propagation is not.  CAVEAT, and it is a real one: the decision variable is")
    print("a five-day rolling variance, so neighbouring lags share four days of data and")
    print("no correlation here is independent of the one beside it.  The asymmetry is")
    print("still readable because both directions carry the SAME overlap, and the paired")
    print("bootstrap differences them out.\n")
    a_t = D[TARGET].values
    print(f"{'peer':>7}{'peak lead k':>13}{'corr at peak':>14}{'corr at k=0':>13}"
          f"{'asym k=1':>11}{'95% CI':>24}")
    asym_pos, asym_neg, asym_val = [], [], {}
    for j, p in enumerate(peers):
        v = D[p].values
        cs = {}
        for k in range(-KMAX, KMAX + 1):
            x = v[KMAX - k: len(v) - KMAX - k]
            z = a_t[KMAX: len(v) - KMAX]
            ok = np.isfinite(x) & np.isfinite(z)
            cs[k] = float(np.corrcoef(x[ok], z[ok])[0, 1])
        peak = max(cs, key=lambda k: cs[k])
        # asymmetry at one day, paired
        x1 = v[KMAX - 1: len(v) - KMAX - 1]; y1 = a_t[KMAX: len(v) - KMAX]
        x2 = a_t[KMAX - 1: len(v) - KMAX - 1]; y2 = v[KMAX: len(v) - KMAX]
        ok = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(x2) & np.isfinite(y2)
        a, lo, hi = corr_diff_ci(x1[ok], y1[ok], x2[ok], y2[ok],
                                 np.random.default_rng(SEED + j))
        asym_val[p] = a
        if lo > 0:
            asym_pos.append(p)
        elif hi < 0:
            asym_neg.append(p)
        print(f"{p:>7}{peak:>13}{cs[peak]:>14.3f}{cs[0]:>13.3f}{a:>+11.3f}"
              + f"[{lo:>+9.3f},{hi:>+9.3f}]".rjust(24))
    print("\n'asym k=1' is corr(peer leads target by a day) minus corr(target leads peer")
    print("by a day).  Positive means the peer leads.")
    print(f"  peers that significantly LEAD the target: {len(asym_pos)} of {len(peers)} "
          f"{asym_pos if asym_pos else ''}")
    print(f"  peers that significantly LAG it:          {len(asym_neg)} of {len(peers)} "
          f"{asym_neg if asym_neg else ''}")

    # --- A2: is that asymmetry direction, or is it the clock? --------------
    print("\n" + "-" * 102)
    print("A2.  BEFORE READING ANY OF THAT AS DIRECTION")
    print("-" * 102)
    print("A row of this panel is not an instant.  The target closes at 21:00 UTC and a")
    print("peer closes hours EARLIER on the same row, so the two directions are measured")
    print("over different amounts of elapsed time:\n")
    print("    peer(t-1) -> target(t)   takes  24 + c  hours")
    print("    target(t-1) -> peer(t)   takes  24 - c  hours")
    print("\nwhere c is how many hours before the target's close that peer's close falls.")
    print("For a persistent series the shorter gap gives the higher correlation, so a")
    print("NEGATIVE asymmetry is exactly what the clock predicts even if nothing leads")
    print("anything.  The clock's prediction is testable: the size of the asymmetry")
    print("should track 2c across peers.  If it does, this table measures the clock.\n")
    print(f"{'peer':>7}{'close UTC':>11}{'shifted':>9}{'c (hours)':>11}"
          f"{'2c':>7}{'|asym|':>9}")
    cs_hours, mags = [], []
    for p in peers:
        eff = L.CLOSE_UTC[p] - (24 if lag[p] else 0)
        c = L.CLOSE_UTC[TARGET] - eff
        cs_hours.append(2 * c)
        mags.append(abs(asym_val[p]))
        print(f"{p:>7}{L.CLOSE_UTC[p]:>11.1f}{'yes' if lag[p] else 'no':>9}"
              f"{c:>11.1f}{2 * c:>7.0f}{abs(asym_val[p]):>9.3f}")
    rho = float(np.corrcoef(cs_hours, mags)[0, 1])
    print(f"\n  correlation across peers between the clock gap 2c and |asymmetry|: "
          f"{rho:+.3f}")
    clockish = rho > 0.7
    if clockish:
        print("  The asymmetry tracks the clock closely.  It is a measurement artefact of")
        print("  putting differently-timed closes on one row, not evidence that anyone")
        print("  leads anyone, and part A settles nothing about direction on its own.")
    elif not asym_pos and not asym_neg:
        print("  No peer is measurably ahead of or behind the target, and the clock does")
        print("  not need to be invoked to explain a pattern that is not there.")
    else:
        print("  The asymmetry does not track the clock, so something other than the")
        print("  timestamp is producing it and the direction above is worth taking")
        print("  seriously - subject to the overlap caveat, which remains.")

    # ---------------- B ---------------------------------------------------
    print("\n" + "=" * 102)
    print("B.  AGE THE FOREIGN FEED - THE QUESTION A DESK ACTUALLY HAS")
    print("=" * 102)
    print("Out-of-sample R-squared of the paper's model with the domestic block delta")
    print("days stale AND the foreign block g days stale.  g = 0 is the paper.\n")
    own_r2 = {}
    for d in DELAYS:
        own_r2[d] = F.r2(yb, walk_own(own, y, idx, d))
    R = {}
    print(f"{'delta':>6}{'own only':>11}" + "".join(f"{'g=' + str(g):>10}"
                                                    for g in PEER_GAPS))
    for d in DELAYS:
        row = f"{d:>6}{own_r2[d]:>11.4f}"
        for g in PEER_GAPS:
            f = walk(own, P, y, idx, d, g)
            R[(d, g)] = (f, F.r2(yb, f))
            row += f"{R[(d, g)][1]:>10.4f}"
        print(row)

    print("\nSubstitution rate: the share of the delay's damage that the foreign block")
    print("puts back.  Undefined at delta = 0, where there is no damage.\n")
    print(f"{'delta':>6}" + "".join(f"{'g=' + str(g):>10}" for g in PEER_GAPS))
    for d in DELAYS[1:]:
        den = own_r2[0] - own_r2[d]
        row = f"{d:>6}"
        for g in PEER_GAPS:
            row += (f"{(R[(d, g)][1] - own_r2[d]) / den:>10.0%}" if den > 1e-9
                    else f"{'n/a':>10}")
        print(row)

    print("\nWhat each extra day of foreign staleness costs, against g = 0:\n")
    print(f"{'delta':>6}" + "".join(f"{'g=' + str(g):>22}" for g in PEER_GAPS[1:]))
    hurt = []
    for d in DELAYS:
        row = f"{d:>6}"
        for g in PEER_GAPS[1:]:
            a, lo, hi = boot_ci(yb, R[(d, g)][0], R[(d, 0)][0],
                                np.random.default_rng(SEED + 50 * g + d))
            if hi < 0:
                hurt.append((d, g))
            row += f"{R[(d, g)][1] - R[(d, 0)][1]:>+13.4f}" + \
                   f"{'*' if hi < 0 else ' ':>2}{'':>7}"
        print(row)
    print("\n  * = staling the foreign feed by that much significantly reduces skill")
    print(f"  cells where it does: {len(hurt)} of {len(DELAYS) * (len(PEER_GAPS) - 1)}")
    g1 = [c for c in hurt if c[1] == 1]
    print(f"  of which at g = 1, a single day: {len(g1)} of {len(DELAYS)}")
    if not hurt:
        print("  Ageing the foreign feed by up to ten days costs nothing measurable at")
        print("  any delay.  The block's value is the cross-section, not the timing, and")
        print("  the paper's recommendation survives a feed far messier than its own.")
    elif len(g1) >= len(DELAYS) - 1:
        print("  One day of foreign staleness already costs skill nearly everywhere. The")
        print("  value is in the timing, and both papers are assuming a same-day foreign")
        print("  feed that they never say they assume.")
    else:
        print("  Staleness costs something at the delays and gaps marked, and nothing at")
        print("  the others.  The pattern above is the result; it is neither story whole.")

    # ---------------- B2 --------------------------------------------------
    print("\n" + "-" * 102)
    print("B2.  THE SAME SURFACE IN UNITS A DESK CAN USE")
    print("-" * 102)
    print("An R-squared of 0.39 means nothing to anyone.  So price each cell in the one")
    print("currency this project has: the age of a domestic mark.  The own-only curve is")
    print("re-run on a fine grid, and each combined model is quoted as the age of the")
    print("plain domestic mark that would forecast exactly as well.\n")
    fine = [0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 21, 26, 34, 44, 55]
    curve = {g: F.r2(yb, walk_own(own, y, idx, g)) for g in fine}
    xs = np.array(fine, dtype=float)
    vs = np.array([curve[g] for g in fine])

    def equiv_age(r):
        """The domestic-mark age with this skill, by interpolation on the curve."""
        if r >= vs[0]:
            return 0.0
        if r <= vs[-1]:
            return np.nan
        for i in range(len(xs) - 1):
            if vs[i] >= r >= vs[i + 1]:
                span = vs[i] - vs[i + 1]
                w = 0.0 if span < 1e-12 else (vs[i] - r) / span
                return float(xs[i] + w * (xs[i + 1] - xs[i]))
        return np.nan

    print("own-only skill against the age of the mark, the ruler everything is read on:")
    print("  " + "  ".join(f"{g}d:{curve[g]:.3f}" for g in fine[:8]))
    print("  " + "  ".join(f"{g}d:{curve[g]:.3f}" for g in fine[8:]))
    print("\nEffective age of a delta-day-old mark once the foreign block is added,")
    print("with that block itself g days stale.  '>55' means worse than the oldest mark")
    print("on the curve.\n")
    print(f"{'delta':>6}" + "".join(f"{'g=' + str(g):>10}" for g in PEER_GAPS))
    for d in DELAYS[1:]:
        row = f"{d:>6}"
        for g in PEER_GAPS:
            e = equiv_age(R[(d, g)][1])
            row += f"{'>55':>10}" if np.isnan(e) else f"{e:>9.1f}d"
        print(row)
    print("\nRead the top-left corner: a mark this many days old, plus today's foreign")
    print("closes, forecasts like a mark that many days old.  Read along a row to see")
    print("what each day of foreign staleness gives back to the clock.")
    print("\nThe exchange rate between the two feeds: how many days of effective")
    print("domestic freshness one day of foreign staleness costs, by least squares")
    print("along each row.\n")
    print(f"{'delta':>6}{'days lost per stale foreign day':>34}")
    slopes = []
    for d in DELAYS[1:]:
        gg, ee = [], []
        for g in PEER_GAPS:
            e = equiv_age(R[(d, g)][1])
            if not np.isnan(e):
                gg.append(float(g)); ee.append(e)
        if len(gg) >= 3:
            A = np.column_stack([np.ones(len(gg)), gg])
            s = float(np.linalg.lstsq(A, np.array(ee), rcond=None)[0][1])
            slopes.append(s)
            print(f"{d:>6}{s:>34.2f}")
    rate = float(np.mean(slopes)) if slopes else float("nan")
    worst = max(DELAYS[1:])
    e0, e1 = equiv_age(R[(worst, 0)][1]), equiv_age(R[(worst, 1)][1])
    if not np.isnan(e0) and not np.isnan(e1):
        print(f"\n  At delta = {worst}: today's foreign closes buy the desk back to a "
              f"{e0:.1f}-day-old mark;\n  one day of foreign staleness moves that to "
              f"{e1:.1f} days.")
    rising = sum(slopes[i + 1] >= slopes[i] - 1e-9 for i in range(len(slopes) - 1))
    print(f"\n  the exchange rate runs from {min(slopes):.2f} to {max(slopes):.2f} days "
          f"and rises with the domestic\n  delay at {rising} of {len(slopes) - 1} steps")
    if rising == len(slopes) - 1:
        print("  It is not one number, and the shape is the point.  When the domestic mark")
        print("  is nearly fresh the foreign block is decoration and ageing it costs")
        print("  almost nothing; when the mark is very stale the block is doing the work,")
        print("  and then a day of foreign staleness costs about a day of domestic")
        print("  freshness - one for one, which is what a second reading of ONE shared")
        print("  state would cost.  The desks that most need this block are exactly the")
        print("  ones whose feed must be current.")
    elif rate > 1.2:
        print("  More than one for one, which is what a NOISY reading of a shared state")
        print("  does: ageing a proxy costs more than ageing the thing itself.")
    else:
        print("  The rate does not move monotonically with the delay, so it is a summary")
        print("  of the surface rather than a property of it, and the table above is the")
        print("  thing to read.")

    # ---------------- C ---------------------------------------------------
    print("\n" + "=" * 102)
    print("C.  WHERE IN TIME DOES THE GLOBAL FACTOR'S INFORMATION SIT?")
    print("=" * 102)
    print("One real-time PC1 factor - eigenvectors recomputed at every refit from the")
    print("training window only, signs pinned, exactly as lab22 builds it - entered at")
    print("ONE lag at a time.  Every arm is the same width, so the estimation cost is")
    print("identical across the row and any difference within a row is information.\n")
    print(f"{'delta':>6}" + "".join(f"{'PC1(t-' + str(g) + ')':>12}" for g in FACT_LAGS))
    Fc = {}
    for d in DELAYS:
        row = f"{d:>6}"
        for g in FACT_LAGS:
            f = walk(own, P, y, idx, d, 0, lags=[g])
            Fc[(d, g)] = f
            row += f"{F.r2(yb, f):>12.4f}"
        print(row)
    print("\nAgainst the contemporaneous factor:\n")
    print(f"{'delta':>6}" + "".join(f"{'t-' + str(g):>22}" for g in FACT_LAGS[1:]))
    fhurt = []
    for d in DELAYS:
        row = f"{d:>6}"
        for g in FACT_LAGS[1:]:
            a, lo, hi = boot_ci(yb, Fc[(d, g)], Fc[(d, 0)],
                                np.random.default_rng(SEED + 70 * g + d))
            if hi < 0:
                fhurt.append((d, g))
            row += f"{F.r2(yb, Fc[(d, g)]) - F.r2(yb, Fc[(d, 0)]):>+13.4f}" + \
                   f"{'*' if hi < 0 else ' ':>2}{'':>7}"
        print(row)
    print(f"\n  * = that lag is significantly worse than the contemporaneous factor")
    print(f"  cells where it is: {len(fhurt)} of {len(DELAYS) * (len(FACT_LAGS) - 1)}")

    print("\nAnd does the factor's CHANGE carry anything beyond its level?  PC1(t) plus")
    print("PC1(t-1) against PC1(t) plus an AR(1) column matched to the factor's own")
    print("persistence - equal width, so only content can separate them.")
    pm, psd, V, _ = F.basis(P[np.isfinite(P).all(axis=1)])
    pc_full = ((P - pm) / psd) @ V[:, 0]
    draws, phi = ar1_surrogates(pc_full, np.random.default_rng(SEED + 9))
    print(f"(The surrogate's persistence, phi = {phi:.3f}, is calibrated on a full-sample")
    print("basis.  That leaks nothing: the surrogate is a regressor with no content by")
    print("construction, and only its persistence is borrowed.)\n")
    print(f"{'delta':>6}{'lag1 - noise':>15}{'95% CI':>26}")
    change = []
    for d in DELAYS:
        f_lag = walk(own, P, y, idx, d, 0, lags=[0, 1])
        fs = [walk(own, P, y, idx, d, 0, lags=[0], noise=s) for s in draws]
        f_noi = np.mean(fs, axis=0)
        a, lo, hi = boot_ci(yb, f_lag, f_noi, np.random.default_rng(SEED + 900 + d))
        if lo > 0:
            change.append(d)
        print(f"{d:>6}{F.r2(yb, f_lag) - F.r2(yb, f_noi):>+15.4f}"
              + f"[{lo:>+10.5f},{hi:>+10.5f}]".rjust(26))
    print(f"\n  delays where yesterday's factor adds to today's: {len(change)} of "
          f"{len(DELAYS)} {change if change else ''}")
    if not change:
        print("  The factor's history adds nothing to its level.  Whatever the foreign")
        print("  block carries is a state, not a motion, and a richer dynamic")
        print("  specification is not the missing piece.")
    else:
        print("  Yesterday's factor adds to today's at the delays listed, so there IS")
        print("  dynamic structure the contemporaneous design does not capture.")

    # ---------------- verdict ---------------------------------------------
    print("\n" + "=" * 102)
    print("VERDICT")
    print("=" * 102)
    print(f"  raw asymmetry: {len(asym_pos)} of {len(peers)} peers lead, "
          f"{len(asym_neg)} lag - but it tracks the trading")
    print(f"  clock at rho = {rho:+.3f}, so direction is "
          f"{'NOT established by part A' if clockish else 'not explained by the clock'}.")
    print(f"  foreign staleness hurts in {len(hurt)} of "
          f"{len(DELAYS) * (len(PEER_GAPS) - 1)} cells, "
          f"{len(g1)} of {len(DELAYS)} at a single day; one stale foreign")
    print(f"  day costs {min(slopes):.2f} to {max(slopes):.2f} days of domestic "
          f"freshness, most where the mark is oldest.")
    print(f"  lagged factor worse than contemporaneous in {len(fhurt)} of "
          f"{len(DELAYS) * (len(FACT_LAGS) - 1)} cells; "
          f"factor history adds at {len(change)} of {len(DELAYS)} delays.")

    if len(g1) >= len(DELAYS) - 1:
        print("\n  THE PRACTICAL FINDING, and it is a limitation neither paper states: the")
        print("  foreign block's value is largely its FRESHNESS.  One day of foreign")
        print("  staleness costs real skill at almost every delay.  Both papers were")
        print("  tested on a same-day foreign feed and never say they assume one; a desk")
        print("  reading yesterday's foreign closes gets materially less than the tables")
        print("  in those papers promise, and the size of the shortfall is in B2.")
    else:
        print("\n  Ageing the foreign feed costs little, so the block's value is its")
        print("  cross-section rather than its timing and the recommendation survives a")
        print("  messier feed than it was tested on.")

    if clockish and not change:
        print("  On direction, nothing here supports propagation.  The only asymmetry is")
        print("  the clock, and the factor's history adds nothing to its level, which is")
        print("  what a single global state read at staggered times looks like.")
    elif clockish and change:
        print(f"  On direction: the raw asymmetry is the clock, but yesterday's factor")
        print(f"  still adds to today's at {change} - short delays only.  That is a small")
        print("  amount of genuine dynamics sitting inside an otherwise contemporaneous")
        print("  story, and it is where a richer specification would have to start.")
    else:
        print("  On direction, part A's asymmetry is not explained by the clock and the")
        print("  tables above are the evidence; it deserves a lab of its own.")


def walk_own(own, y, idx, delta):
    """The own-only baseline, needed for the substitution rate."""
    out = np.empty(len(idx)); b = mu = sd = None
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), max(cut, 1))
            X = own[tr - delta]
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y[tr])
            X, yy = X[ok], y[tr][ok]
            if len(yy) < 200:
                b = None; continue
            mu, sd = X.mean(0), X.std(0) + 1e-9
            b = L.cv((X - mu) / sd, yy, "cont")
        xt = own[t - delta]
        out[j] = 0.0 if (b is None or not np.isfinite(xt).all()) \
            else L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
