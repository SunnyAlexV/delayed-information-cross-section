"""lab73_own_options_paired.py - the scope condition, tested within target.

    python lab73_own_options_paired.py [data folder]      (about 25 minutes)

WHY THIS FILE EXISTS
--------------------
Section 7 says that where a liquid options market exists on the asset itself,
breadth is redundant.  Until now that rested on TWO assets.  The S&P and the
DAX are the only targets in this project for which an own-index volatility
index was held, and lab51 could do no better than compare them with six targets
holding somebody else's.

That comparison has a confound lab51 states in its own Part D and Figure 3's
caption concedes outright: across those eight targets, option-market status and
COUPLING to the cross-section are perfectly separated.  Every optioned target is
more strongly coupled than every unoptioned one.  With eight points the two
cannot be told apart, so the published gradient of -0.37 is equally consistent
with "options make breadth redundant" and with "well-coupled markets are the
ones that happened to get options".

This file breaks that confound by never comparing two targets.  For each of four
new targets it runs the SAME market, over the SAME window, against the SAME
cross-section, and changes one thing: whether the implied-volatility block is
written on the asset being forecast or on somebody else's.

    A(T)  =  R2(own + peers + ivF)  -  R2(own + ivF)
    B(T)  =  R2(own + peers + ivO)  -  R2(own + ivO)

A is what the foreign cross-section still adds to a forecaster holding FOREIGN
implied volatility.  B is what it adds to one holding THEIR OWN.  Coupling,
clock, sample, estimator and peer block are identical between them because it is
one target; only the options feed moves.  If the scope condition is about
options, B collapses and A does not.  If it was about coupling all along, both
stay where they are, and Section 7 needs rewriting rather than extending.

THE FOUR SERIES, AND WHY THEIR DATES ARE SAFE
---------------------------------------------
    N225   Nikkei Stock Average Volatility Index   live from 2010-11-19
    NSEI   India VIX                               live from 2008-04-08
    HSI    HSI Volatility Index (VHSI)             live from 2011-02-21
    AXJO   S&P/ASX 200 VIX (A-VIX)                 index start 2008-01-02

Three of these four are PUBLISHED earlier than they existed: Nikkei VI is
back-calculated to 1989, VHSI to about 2001.  A back-calculated value was never
on a screen, and feeding one to a forecaster is the same error as a stale mark
running backwards - formally admissible on every timestamp and wrong.  The
exports used here begin at each index's live date and carry no back-cast rows,
which Part A checks rather than assumes.

A-VIX comes from Yahoo and not from the vendor behind the other three.
Investing.com's A-VIX export carries 75 rows misdated by one day in blocks that
track Australian daylight saving, and one fabricated print of 101.00 on a Sunday
between a Friday 33.22 and a Monday 30.88.  Yahoo's carries neither and agrees
with it on 99.92% of the 4,743 common weekday dates to within 0.5%, at a median
absolute difference of 0.0020.  The disagreement is confined to the rows one
vendor misdated, so this is a cross-vendor validation of the series and a
rejection of one vendor's copy of it, not a change of series.  lab72 is the
audit; data/README.md records the provenance.

ADMISSIBILITY, WHICH IS THE AWKWARD PART AND IS NOT NEW
-------------------------------------------------------
An own-index volatility index prints when its own cash market closes: Osaka at
06:00 UTC for N225, NSE at 10:00 for NSEI, HKEX at 08:00 for HSI, ASX at 06:00
for AXJO.  That is exactly the VIX-against-SPX problem lab08 already states and
already handles, and the handling transfers unchanged.  Both arms are reported:

    SAME-DAY   the own index enters at t.  These indices are computed
               continuously from live option quotes, so a forecaster standing
               at the cash close does observe a value of one, a few minutes'
               drift from the printed close.  This is the realistic treatment.
    STRICT     the own index enters at t-1, conceding the whole session.  This
               is a conservative bound, not the right answer.

The foreign block enters at t-1 in both arms, which is strictly admissible
everywhere and identical across targets, as in lab51.  A verdict that holds at
both ends of that interval is one the paper may claim; one that holds only
same-day is not, and this file says which it is rather than picking.

WHAT THE LENGTHS ARE, AND WHY THEY ARE ACCEPTABLE
--------------------------------------------------
Each own index starts later than the panel, so both arms for a target are
estimated on THAT target's own-IV window and are length-matched to each other.
They are shorter than the headline test and are not comparable to it: Part A
prints the usable count per target, and lab70 is what says what a set of that
length covers.  The shortest arms here sit above the 1,620-day conditional cells
for which Section S4 already publishes intervals.

WHAT THIS CANNOT SETTLE
-----------------------
Four targets is four targets.  The paired design removes the confound BETWEEN
targets; it does not turn four into a sample that can carry an interval on the
difference of differences, and none is quoted.  What it can do is say whether
the sign and rough size of the effect survive when coupling is held fixed by
construction, on four markets that were never in the original two.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab05_robustness as L
import lab08_implied_vol as IV

H = L.HORIZON
DELAYS = [0, 5, 21, 55]
FOREIGN_IV = ["VIX", "VDAX"]          # the block lab51 gave every target

# The four targets this file exists for, and the own-index series each gets.
# SPX and DAX are reported beside them as the two cases Section 7 already had,
# but they are NOT part of the paired test: for them the "foreign" block would
# have to drop the series written on themselves, so the two arms would differ in
# width as well as in content and the contrast would stop being clean.
PAIRED = {"N225": "NKVI", "NSEI": "INDIAVIX", "HSI": "VHSI", "AXJO": "AVIX"}
REFERENCE = {"SPX": "VIX", "DAX": "VDAX"}

LIVE_FROM = {"NKVI": "2010-11-19", "INDIAVIX": "2008-04-08",
             "VHSI": "2011-02-21", "AVIX": "2008-01-02"}


def r2(y, f, *, bench):
    """Out-of-sample skill against the trailing-mean benchmark.

    Keyword-only and mandatory, for the reason lab51 gives: this helper divided
    by the mean of the target over the test period in three files until an audit
    caught it, and a mandatory argument is what stops a missed call site from
    scoring against the wrong yardstick in silence.
    """
    den = ((y - bench) ** 2).sum()
    return 1 - ((y - f) ** 2).sum() / den if den > 0 else np.nan


def decision_var(series, index, shift):
    """An IV series on the panel's calendar, as a scale-free decision variable.

    Mirrors the index decision variable exactly - log of the level over its own
    trailing 252-day median - so a quiet market and a nervous one are comparable
    and no series carries its own units into the ridge.
    """
    on = series.reindex(index).ffill(limit=5)
    if shift:
        on = on.shift(shift)
    return np.log(on / on.rolling(L.MED, min_periods=30).median()).values


def build_paired(folder, target, own_series, raw_iv, own_shift):
    """One target's panel restricted to its OWN-IV window, with both IV blocks.

    The restriction is the whole point: A(T) and B(T) have to be estimated on
    the same rows or the contrast between them is partly a contrast of samples.
    """
    D, peers, lag = L.build(folder, target)
    o = raw_iv[own_series]
    # the own-IV window, intersected on the panel's own calendar
    keep = D.index >= max(D.index.min(), o.index.min())
    D = D.loc[keep]
    a = D[target].values
    sa = pd.Series(a)
    own = np.column_stack([sa.values, sa.rolling(5).mean().values,
                           sa.rolling(22).mean().values])
    ivF = np.column_stack([decision_var(raw_iv[t], D.index, 1)
                           for t in FOREIGN_IV])
    ivO = decision_var(o, D.index, own_shift)[:, None]
    n = len(D)
    y = np.full(n, np.nan); y[:-H] = a[H:]
    start = L.MED + L.TRAIN + L.VAL + max(DELAYS) + 1 + H
    idx = np.arange(start, n - H)
    ok = (np.isfinite(y[idx]) & np.isfinite(a[idx])
          & np.isfinite(ivF[idx]).all(axis=1) & np.isfinite(ivO[idx]).all(axis=1))
    idx = idx[ok]
    return own, D[peers].values, ivF, ivO, y, idx, len(peers)


def arm(own, P, ivX, y, idx, delta, bench):
    """R2 with and without the peer block, given one IV block, plus GW.

    own+ivX is nested inside own+peers+ivX, so Giacomini-White is the right
    test and Diebold-Mariano is the wrong one, as everywhere else here.
    """
    yb = y[idx]
    f_iv = IV.walk(own, [ivX], y, idx, delta)
    f_bot = IV.walk(own, [ivX, P], y, idx, delta)
    s_iv, s_bot = r2(yb, f_iv, bench=bench), r2(yb, f_bot, bench=bench)
    dl = (yb - f_iv) ** 2 - (yb - f_bot) ** 2
    z = float(dl.mean() / IV.hac_se(dl, lag=L.block_for(delta)))
    return s_iv, s_bot, s_bot - s_iv, z


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints the paired own-options test, Section S41.\n")

    tags = sorted(set(list(PAIRED.values()) + list(REFERENCE.values())
                      + FOREIGN_IV))
    raw_iv = {}
    for t in tags:
        try:
            raw_iv[t] = IV.load_iv(folder, t)
        except SystemExit as exc:
            print(f"  {t}: NOT AVAILABLE - {str(exc).splitlines()[1]}")
    print()

    print("=" * 98)
    print("A.  THE FOUR OWN-INDEX SERIES, AND WHETHER ANY OF IT IS BACK-CAST")
    print("=" * 98)
    print("  A value published before the index existed was never on a screen.")
    print("  Each export must begin at or after its index's live date.\n")
    print(f"  {'target':<8}{'series':<11}{'rows':>8}{'first':>13}{'last':>13}"
          f"{'live from':>13}{'back-cast':>11}")
    for tg, s in PAIRED.items():
        if s not in raw_iv:
            continue
        v = raw_iv[s]
        pre = int((v.index < pd.Timestamp(LIVE_FROM[s])).sum())
        print(f"  {tg:<8}{s:<11}{len(v):>8,}{str(v.index.min().date()):>13}"
              f"{str(v.index.max().date()):>13}{LIVE_FROM[s]:>13}{pre:>11}")
    print("\n  Any non-zero count in the last column invalidates that target's")
    print("  arms and the paper may not use them.")

    # ---- A2.  the weekend rows these series carry, and whether they matter --
    #
    # lab72 gates the INDEX series against the trading calendar; it does not
    # gate the implied-volatility series, and three of these four carry
    # weekend-dated rows.  That turns out not to matter, but "turns out" is not
    # a standard: the reason has to be checked, because it is a property of the
    # order of two operations and either could be reordered by a later edit.
    #
    # decision_var reindexes onto the PANEL's calendar before taking the
    # trailing median, so a weekend IV row is reachable only on a date the
    # panel itself contains.  After lab72's gate the N225 and HSI panels contain
    # no weekend dates at all, so Nikkei VI's and VHSI's weekend rows are
    # unreachable.  NSEI's panel does contain weekend dates - the genuine NSE
    # Saturday sessions - and India VIX prints on exactly those days, which it
    # should, because Nifty options trade in them.
    #
    # That last line is worth more than it looks.  The decision to exempt NSEI
    # from the trading-day gate rested on a signature test of the index alone.
    # Here a SECOND and unrelated series - a volatility index computed from
    # options rather than from the cash market - independently prints on the
    # same Saturdays.  Two different calculations agreeing that those sessions
    # happened is stronger evidence than the signature test that prompted it.
    print("\n" + "=" * 98)
    print("A2. WEEKEND ROWS IN THE IV SERIES, AND WHETHER ANY IS REACHABLE")
    print("=" * 98)
    print("  The IV series are not gated against the calendar; the panel is.")
    print("  A weekend IV row can only be used on a date the panel contains,")
    print("  because the reindex happens before the trailing median.\n")
    print(f"  {'target':<8}{'series':<11}{'weekend rows':>14}{'reachable':>11}   status")
    for tg, s in PAIRED.items():
        if s not in raw_iv:
            continue
        D, _, _ = L.build(folder, tg)
        we = raw_iv[s].index[raw_iv[s].index.dayofweek >= 5]
        reach = D.index.intersection(we)
        if len(reach) == 0:
            status = "unreachable; the panel has no weekend dates"
        elif tg in L.WEEKEND_SESSIONS:
            status = f"reachable and CORRECT: {tg} trades those sessions"
        else:
            status = "REACHABLE ON A GATED PANEL - investigate"
        print(f"  {tg:<8}{s:<11}{len(we):>14}{len(reach):>11}   {status}")
    print("\n  The NSEI row is a second, independent witness for the exemption")
    print("  lab72 makes: a volatility index computed from OPTIONS prints on the")
    print("  same Saturdays the cash index does, so those sessions are real.")

    print("\n" + "=" * 98)
    print("B.  THE PAIRED CONTRAST, TARGET BY TARGET")
    print("=" * 98)
    print("  A = what the cross-section adds given FOREIGN implied volatility")
    print("  B = what it adds given the target's OWN implied volatility")
    print("  Same target, same rows, same peers, same estimator. Only the")
    print("  options feed differs, so coupling is held fixed by construction.\n")

    out = {}
    for armname, shift in (("SAME-DAY", 0), ("STRICT", 1)):
        print(f"\n  ---- own index enters at t{'' if shift == 0 else '-1'}"
              f"   ({armname}) ----")
        for tg, s in PAIRED.items():
            if s not in raw_iv:
                continue
            own, P, ivF, ivO, y, idx, k = build_paired(folder, tg, s, raw_iv, shift)
            bench = L.bench_mean(y, idx)
            print(f"\n  {tg}  (own index {s}; {len(idx):,} test days, {k} peers)")
            print(f"  {'delta':>7}{'A: +IV_F':>10}{'A: +both':>10}{'A gain':>9}"
                  f"{'GW z':>7}   {'B: +IV_O':>10}{'B: +both':>10}{'B gain':>9}"
                  f"{'GW z':>7}")
            for d in DELAYS:
                aF = arm(own, P, ivF, y, idx, d, bench)
                aO = arm(own, P, ivO, y, idx, d, bench)
                out[(armname, tg, d)] = (aF, aO)
                print(f"  {d:>7}{aF[0]:>10.4f}{aF[1]:>10.4f}{aF[2]:>+9.4f}"
                      f"{aF[3]:>+7.2f}   {aO[0]:>10.4f}{aO[1]:>10.4f}"
                      f"{aO[2]:>+9.4f}{aO[3]:>+7.2f}", flush=True)

    print("\n" + "=" * 98)
    print("C.  WHAT HAPPENS TO BREADTH WHEN THE OPTIONS BECOME YOUR OWN")
    print("=" * 98)
    print("  The paired difference at eleven weeks, per target and per arm.")
    print("  A negative change is breadth losing value once the forecaster")
    print("  holds options on the thing they are forecasting.\n")
    print(f"  {'arm':<10}{'target':<8}{'A gain':>10}{'B gain':>10}"
          f"{'change':>10}{'A sig':>8}{'B sig':>8}")
    for armname in ("SAME-DAY", "STRICT"):
        for tg in PAIRED:
            key = (armname, tg, 55)
            if key not in out:
                continue
            aF, aO = out[key]
            print(f"  {armname:<10}{tg:<8}{aF[2]:>+10.4f}{aO[2]:>+10.4f}"
                  f"{aO[2] - aF[2]:>+10.4f}"
                  f"{('yes' if abs(aF[3]) > 2 else 'no'):>8}"
                  f"{('yes' if abs(aO[3]) > 2 else 'no'):>8}")

    print("\n" + "=" * 98)
    print("VERDICT")
    print("=" * 98)
    for armname in ("SAME-DAY", "STRICT"):
        keys = [(armname, tg, 55) for tg in PAIRED if (armname, tg, 55) in out]
        if not keys:
            continue
        drops = sum(1 for k in keys if out[k][1][2] < out[k][0][2])
        sigA = sum(1 for k in keys if abs(out[k][0][3]) > 2)
        sigB = sum(1 for k in keys if abs(out[k][1][3]) > 2)
        print(f"  {armname}: breadth is worth less given own options on "
              f"{drops} of {len(keys)} targets.")
        print(f"            it still adds significantly given FOREIGN options "
              f"in {sigA} of {len(keys)},")
        print(f"            and given OWN options in {sigB} of {len(keys)}.")
    print("\n  Read this against lab51, where the same question was asked across")
    print("  targets and could not separate options from coupling.  Here coupling")
    print("  cannot move, because the target does not change.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
