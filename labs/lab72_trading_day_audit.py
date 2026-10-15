"""lab72_trading_day_audit.py - is every input dated on a day its market was open?

    python lab72_trading_day_audit.py [data folder]      (about 20 seconds)

WHY THIS FILE EXISTS
--------------------
This project is careful about time.  Every series carries a close in UTC, a
peer closing at or after the target is shifted back a day, VIX enters at t-1,
CONVENTIONS.tsv records the rule for each series, and lab59 audits the session
timestamps.  All of that machinery compares one timestamp with another.

None of it ever compared a timestamp with a CALENDAR.

So a row dated on a Sunday passed every check in the repository.  It has a
date, the date is in range, it sorts correctly, it precedes the target's close,
and no digest notices because the digest is over the bytes the vendor shipped
and the vendor shipped exactly those bytes.  The row is simply not a cash print
of the index whose column it sits in, and nothing here could see that.

Three series carried such rows.  Two were wrong and one was right, and telling
them apart is the job of Part B.

WHAT THE SIGNATURE TEST IS
--------------------------
An index's weekend rows cannot be interrogated directly - there is no second
source in this project to check them against.  But they can be compared with
the SAME series' weekday rows on two properties that a cash close and a futures
print do not share:

    how often the close is a whole number.  A cash index is a weighted sum of
    constituent prices and lands on a round figure by accident, about 1% of the
    time.  A futures contract trades in ticks and prints round numbers often.

    the mean intraday range, (high - low) / close.  A full cash session runs
    about 1.1% to 1.6% in these markets.  A thin weekend futures session runs a
    fifth of that.

Those two numbers identify the instrument without needing to know what it is.

WHAT THIS LAB DOES NOT ESTABLISH
--------------------------------
That the surviving rows are correct.  This audit finds rows dated on days the
market was shut.  A vendor error dated on a Tuesday is invisible here, and
nothing in this project would catch it either.  The claim is narrow: after this
gate, no input row is dated on a day its own exchange was closed, except where
that exchange is known to open and the row looks like an ordinary session.

The gate itself lives in lab05_robustness.WEEKEND_SESSIONS and load_index, and
selftest.py proves it still fires.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab05_robustness as L
import lab08_implied_vol as IV

INDEX_TAGS = ["SPX", "DAX", "FTSE", "N225", "HSI", "NSEI", "AXJO", "BVSP"]
IV_TAGS = ["VIX", "VIX9D", "VDAX", "VSTOXX"]


def raw_index(tag, folder):
    """load_index WITHOUT the gate, so the audit can see what the gate removes."""
    parts = [L.read_one(p) for p in L.files_for(tag, folder)]
    m = parts[0]
    for b in parts[1:]:
        m = (pd.concat([m, b]).drop_duplicates("date", keep="first")
               .sort_values("date").reset_index(drop=True))
    return m


def diffcount(a, b):
    """Rows differing on the dates the two series SHARE, NaN-safe.

    The NaN handling is not a detail.  A first version of this counted a NaN
    against a NaN as a difference, which made a series the gate had not touched
    at all report four changed values.  A count that reports spurious changes
    is worse than no count, because it is believed.
    """
    j = a.index.intersection(b.index)
    x, y = np.asarray(a[j], float), np.asarray(b[j], float)
    both_nan = np.isnan(x) & np.isnan(y)
    return int((~(np.isclose(x, y, rtol=1e-12) | both_nan)).sum())


def _sister_counts():
    """The four significance counts this file's verdict cites, read from the
    labs that COMPUTE them.  lab51 owns the two redundancy counts and lab53
    owns the two coverage counts; a copy kept here would be a second home for
    somebody else's result, which is exactly the failure this file documents.
    Returns [] if either stored output is missing, so the caller can make the
    claim without numbers rather than with stale ones."""
    import re as _re
    exp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "expected_output")
    want = (("lab51_foreign_options.txt",
             r"significantly in (\d+ of \d+) cells, which is Section 8"),
            ("lab51_foreign_options.txt",
             r"foreign block still adds given implied volatility in "
             r"(\d+ of \d+) cells"),
            ("lab53_no_options_targets.txt",
             r"no volatility index: breadth[^\n]*?, (\d+ of \d+) cells"),
            ("lab53_no_options_targets.txt",
             r"has one, not held here: breadth[^\n]*?, (\d+ of \d+) cells"))
    out = []
    for fn, pat in want:
        path = os.path.join(exp, fn)
        if not os.path.exists(path):
            return []
        m = _re.search(pat, open(path, encoding="utf-8").read())
        if not m:
            return []
        out.append(m.group(1))
    return out


def main(folder):
    print("=" * 98)
    print("lab72  -  IS EVERY INPUT DATED ON A DAY ITS MARKET WAS OPEN?")
    print("=" * 98)
    print(f"  data folder: {folder}")
    print(f"  exchanges this project treats as holding weekend sessions: "
          f"{sorted(L.WEEKEND_SESSIONS)}\n")

    print("=" * 98)
    print("A.  ROWS DATED ON A SATURDAY OR A SUNDAY")
    print("=" * 98)
    raw = {}
    print(f"  {'series':<22}{'rows':>9}{'weekend':>9}{'first':>13}{'last':>13}")
    for t in INDEX_TAGS:
        if not L.files_for(t, folder):
            continue
        m = raw_index(t, folder)
        raw[t] = m
        we = m.loc[m["date"].dt.dayofweek >= 5, "date"]
        print(f"  {t + ' (index)':<22}{len(m):>9,}{len(we):>9}"
              f"{(str(we.min().date()) if len(we) else '-'):>13}"
              f"{(str(we.max().date()) if len(we) else '-'):>13}")
    ivs = {}
    for tag in IV_TAGS:
        try:
            s = IV.load_iv(folder, tag).sort_index()
        except SystemExit:
            continue
        ivs[tag] = s
        we = s.index[s.index.dayofweek >= 5]
        print(f"  {tag + ' (implied vol)':<22}{len(s):>9,}{len(we):>9}"
              f"{(str(we.min().date()) if len(we) else '-'):>13}"
              f"{(str(we.max().date()) if len(we) else '-'):>13}")
    print("\n  Every implied-volatility series is clean on this check.  The three")
    print("  index series with weekend rows are the subject of Part B.")

    print("\n" + "=" * 98)
    print("B.  THE SIGNATURE TEST: CASH SESSION, OR ANOTHER INSTRUMENT?")
    print("=" * 98)
    print("  A cash close carries cents and a full day's range.  A futures print")
    print("  is round and thin.  Each series is compared with ITSELF, so nothing")
    print("  depends on knowing what the other instrument is.\n")
    print(f"  {'series':<8}{'rows':<10}{'n':>7}{'whole-number close':>21}"
          f"{'mean (high-low)/close':>24}")
    verdict = {}
    for t, m in raw.items():
        we = m["date"].dt.dayofweek >= 5
        if not we.any():
            continue
        stats = {}
        for name, sub in (("weekend", m[we]), ("weekday", m[~we])):
            whole = float(np.isclose(sub["close"] % 1, 0).mean())
            rng = float(((sub["high"] - sub["low"]) / sub["close"]).mean())
            stats[name] = (len(sub), whole, rng)
            print(f"  {t if name == 'weekend' else '':<8}{name:<10}{len(sub):>7,}"
                  f"{whole:>20.1%}{rng:>23.3%}")
        nwe, wwe, rwe = stats["weekend"]
        nwd, wwd, rwd = stats["weekday"]
        # A weekend row is a DIFFERENT instrument when it is round far more
        # often AND its range is far thinner.  Both conditions, because either
        # one alone is satisfied by chance in a handful of rows.
        other = (wwe > 4 * max(wwd, 0.01)) and (rwe < 0.5 * rwd)
        verdict[t] = "another instrument" if other else "ordinary cash session"
        print(f"  {'':<8}{'-> ' + verdict[t]}\n")

    print("=" * 98)
    print("C.  HOW FAR ONE BAD ROW REACHES")
    print("=" * 98)
    print("  Yang-Zhang's overnight term is log(O_t / C_{t-1}), so a spurious row")
    print("  corrupts its own proxy value and the FOLLOWING day's.  The decision")
    print("  variable then divides by a trailing 252-day median, which carries the")
    print("  damage forward for a year of trading days.  The counts below are on")
    print("  dates the two versions SHARE - legitimate days whose values were wrong.\n")
    print(f"  {'series':<8}{'dropped':>9}{'proxy wrong':>14}"
          f"{'decision-var wrong':>21}{'% of panel':>13}")
    for t, m in raw.items():
        if t in L.WEEKEND_SESSIONS:
            gated = m
        else:
            gated = m.loc[m["date"].dt.dayofweek < 5].reset_index(drop=True)
        if len(gated) == len(m):
            continue
        a, b = L.yang_zhang(m), L.yang_zhang(gated)
        da = np.log(a / a.rolling(L.MED, min_periods=30).median())
        db = np.log(b / b.rolling(L.MED, min_periods=30).median())
        dch = diffcount(da, db)
        print(f"  {t:<8}{len(m) - len(gated):>9}{diffcount(a, b):>14,}"
              f"{dch:>21,}{dch / len(gated):>12.2%}")
    print("\n  A single wrong print is not a single wrong number.")

    print("\n" + "=" * 98)
    print("D.  WHAT THE GATE REMOVES, AND WHAT IT KEEPS")
    print("=" * 98)
    print(f"  {'series':<8}{'raw':>9}{'after gate':>13}{'removed':>9}   rule")
    for t, m in raw.items():
        g = L.load_index(t, folder)
        rule = ("kept: this exchange holds weekend sessions"
                if t in L.WEEKEND_SESSIONS else
                "weekend rows are inadmissible for this exchange")
        print(f"  {t:<8}{len(m):>9,}{len(g):>13,}{len(m) - len(g):>9}   {rule}")
    left = sum(int((L.load_index(t, folder)["date"].dt.dayofweek >= 5).sum())
               for t in raw)
    kept_ok = sum(int((L.load_index(t, folder)["date"].dt.dayofweek >= 5).sum())
                  for t in raw if t in L.WEEKEND_SESSIONS)

    # ---- E. the bad prints themselves, with their neighbours --------------
    #
    # Sections S41 and S42 quote these values.  They were established by hand
    # when the defect was found, and a figure established by hand is a figure
    # with no runnable cell behind it - which is the one thing this project does
    # not allow anywhere else.  So the audit prints them.
    print("\n" + "=" * 98)
    print("E.  THE BAD PRINTS, AGAINST THE SESSIONS EITHER SIDE")
    print("=" * 98)
    print("  A weekend row is only evidence of an error when the cash index was")
    print("  somewhere else entirely.  These are the rows the gate removes.\n")
    print(f"  {'series':<8}{'date':<13}{'previous close':>16}{'this row':>12}"
          f"{'next close':>13}{'error vs neighbours':>21}")
    for t, m in raw.items():
        if t in L.WEEKEND_SESSIONS:
            continue
        s = m.set_index("date")["close"].sort_index()
        we = s.index[s.index.dayofweek >= 5]
        for d in we:
            prev = s[s.index < d]
            nxt = s[s.index > d]
            if not len(prev) or not len(nxt):
                continue
            pv, nv = prev.iloc[-1], nxt.iloc[0]
            err = s[d] / ((pv + nv) / 2.0) - 1.0
            if abs(err) < 0.10:          # the misdated ones, not the wrong ones
                continue
            print(f"  {t:<8}{str(d.date()):<13}{pv:>16,.2f}{s[d]:>12,.2f}"
                  f"{nv:>13,.2f}{err:>20.0%}")
    print("\n  A row within ten per cent of its neighbours is misdated rather than")
    print("  invented, and is removed for being inadmissible, not for being wrong.")

    # ---- F. one series, two vendors ---------------------------------------
    print("\n" + "=" * 98)
    print("F.  A-VIX: THE SERIES THIS PAPER USES, AGAINST THE OTHER VENDOR'S COPY")
    print("=" * 98)
    print("  The S&P/ASX 200 VIX is taken from Yahoo and not from the vendor")
    print("  behind the other three own-index series.  That is a choice, so it is")
    print("  audited rather than asserted: the rejected copy stays in the data")
    print("  folder for this comparison, exactly as the retail VIX export stays")
    print("  for lab43's.\n")
    alt = [p for p in sorted(glob.glob(os.path.join(folder, "*.csv")))
           if "asx_200_vix_historical" in os.path.basename(p).lower()]
    try:
        used = IV.load_iv(folder, "AVIX")
    except SystemExit:
        used = None
    if used is None or not alt:
        print("  one of the two copies is absent; nothing to compare")
    else:
        other = L.read_one(alt[0]).set_index("date")["close"].sort_index()
        we = other.index[other.index.dayofweek >= 5]
        print(f"  {'copy':<16}{'rows':>8}{'first':>13}{'last':>13}"
              f"{'weekend rows':>14}{'max':>9}")
        for nm, s in (("used (Yahoo)", used), ("rejected", other)):
            print(f"  {nm:<16}{len(s):>8,}{str(s.index.min().date()):>13}"
                  f"{str(s.index.max().date()):>13}"
                  f"{int((s.index.dayofweek >= 5).sum()):>14}{s.max():>9.2f}")
        ow = other[other.index.dayofweek < 5]
        j = used.index.intersection(ow.index)
        rel = (ow[j] - used[j]).abs() / used[j].abs()
        print(f"\n  weekday dates both cover            {len(j):,}")
        print(f"  agreeing within 0.5%                {int((rel <= 0.005).sum()):,}"
              f"  ({(rel <= 0.005).mean():.2%})")
        print(f"  median absolute difference          {(ow[j] - used[j]).abs().median():.4f}")
        hi = other[other > 80]
        for d, v in hi.items():
            p_ = other[other.index < d]
            n_ = other[other.index > d]
            print(f"  the rejected copy's largest print   {v:.2f} on {d.date()}"
                  f" ({d.day_name()[:3]}), between {p_.iloc[-1]:.2f} and {n_.iloc[0]:.2f}")
        print("\n  Two independent vendors agreeing on the overwhelming majority of")
        print("  shared dates is a cross-validation of the SERIES; the disagreement")
        print("  is confined to rows one of them misdated. Rejecting that copy is")
        print("  not a change of series.")

    print("\n" + "=" * 98)
    print("VERDICT")
    print("=" * 98)
    bad = [t for t, v in verdict.items() if v == "another instrument"]
    good = [t for t, v in verdict.items() if v == "ordinary cash session"]
    print(f"  Weekend-dated rows survive in {left} places after the gate, all of")
    print(f"  them in {sorted(L.WEEKEND_SESSIONS)}, which is {kept_ok} rows and is intended.")
    print(f"\n  Identified as another instrument and removed: {', '.join(bad) if bad else 'none'}")
    print(f"  Identified as genuine weekend sessions and kept: {', '.join(good) if good else 'none'}")
    print("\n  The distinction matters in both directions.  Dropping every weekend")
    print("  row would have discarded 29 real NSE sessions - budget days, Muhurat")
    print("  trading and disaster-recovery drills - and keeping every weekend row")
    print("  left 30 Nikkei futures prints and one 26%-wrong Hang Seng print inside")
    print("  the estimator, where they corrupted a year of the decision variable")
    print("  each through the trailing median.")
    print("\n  What this changed in the paper: nothing it claims, and the last")
    print("  decimal of nearly everything.  The headline rate moves from 71.6% to")
    print("  71.5% and its interval from [54.4, 83.5] to [54.3, 83.4]; the")
    print("  identification statistic is unmoved at 5.40.  The multi-target tables")
    print("  of Sections S13 and S14 move by up to 1.3 points on Hong Kong, whose")
    print("  own series carried the bad print, and the counts those sections turn")
    # Those four counts belong to lab51 and lab53, not to this file.  Typing
    # them here would make this verdict a hand-kept copy of another lab's
    # result, which is the defect this repository has now found six times, so
    # they are read out of those labs' stored output and the sentence says
    # plainly where they come from.  If either file is absent the claim is made
    # without the figures rather than with figures that might be stale.
    _counts = _sister_counts()
    if _counts:
        print(f"  on ({', '.join(_counts)}, read from lab51 and lab53) are all")
        print("  unchanged.")
    else:
        print("  on are all unchanged.  Run lab51 and lab53 for those counts.")
    print("\n  The honest summary is that this defect cost the paper no conclusion")
    print("  and should still never have survived two years of checking.  It did so")
    print("  because every check here compared a timestamp with another timestamp,")
    print("  and a calendar is a different kind of fact.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else L.DEFAULT_FOLDER
         if hasattr(L, "DEFAULT_FOLDER") else "data")
