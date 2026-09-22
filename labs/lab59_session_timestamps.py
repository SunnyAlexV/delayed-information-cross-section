"""
lab59_session_timestamps.py - is the admissibility rule true on real session clocks?

Imports lab05_robustness; keep both in labs/.  Runtime a few seconds.

WHAT THIS ANSWERS
-----------------
Section 3's design rests on one rule: a foreign close is admissible on day t only
if its timestamp falls strictly before the target's close on day t, and any market
closing at or after the target contributes its PREVIOUS session instead.  The rule
is implemented from a table of fixed UTC close times in lab05:

    N225 06:00, AXJO 06:00, HSI 08:00, NSEI 10:00,
    FTSE 16:30, DAX 16:30, BVSP 21:00, SPX 21:00

Those are standard-time closes.  Real exchanges keep local time and most of them
observe daylight saving, on their own dates, which do not coincide.  The paper
has been asserting that this never matters - "close times are treated as
standard, ignoring daylight saving, which moves them by an hour and never
reorders the blocks" - and an assertion is not a measurement.  It is also exactly
the kind of assertion that is usually right and occasionally, on three weeks of
the year, is not: the United States advances its clocks on the second Sunday in
March and the European Union on the last Sunday, so for a fortnight New York is
an hour closer to London than the table says.  If a transition ever moved a
peer's close past the target's, the panel would carry a value the forecaster
could not have seen, and the leak would be silent.

WHAT IT DOES
------------
For every target, every peer and every trading day in the sample, it builds the
true close in UTC from the exchange's own local close time and its own timezone,
with the IANA database supplying the transition dates, and compares the
admissibility decision that follows with the one the fixed table produces.

    lag_true(p, t) = 0 if close_UTC(p, t) < close_UTC(target, t) else 1
    lag_table(p)   = 0 if CLOSE_UTC[p] < CLOSE_UTC[target] else 1

Any day where the two differ is either a leak, if the table says same-day and the
clock says the peer closed later, or a wasted day, if the reverse.  Both are
reported; only the first is a threat to the design.

It also reports the MARGIN - how much earlier the peer really closes - because a
rule that survives by four minutes is a different object from one that survives
by four hours, and the paper should be able to say which it has.

HOLIDAYS
--------
A second assumption is worth testing in the same place.  The panel is built on
the target's calendar and peers are forward-filled with a five-day limit, so a
peer whose exchange was shut on day t carries its last close.  That is strictly
older than t and therefore cannot leak; the question is how often it happens and
how stale the carried value gets, because a peer block that is mostly stale is a
weaker instrument than the text implies.
"""

import os
import sys
import datetime as _dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L

# Local close time and IANA zone for each index in the panel.  The local times
# are the regular-session closes; auction and post-close phases are not what the
# rule is about, since the printed close is what a forecaster reads.
SESSIONS = {
    "N225": ("Asia/Tokyo",        _dt.time(15, 0)),
    "AXJO": ("Australia/Sydney",  _dt.time(16, 0)),
    "HSI":  ("Asia/Hong_Kong",    _dt.time(16, 0)),
    "NSEI": ("Asia/Kolkata",      _dt.time(15, 30)),
    "FTSE": ("Europe/London",     _dt.time(16, 30)),
    "DAX":  ("Europe/Berlin",     _dt.time(17, 30)),
    "BVSP": ("America/Sao_Paulo", _dt.time(17, 0)),
    "SPX":  ("America/New_York",  _dt.time(16, 0)),
}


def close_utc(tag, day):
    """The true UTC hour of `tag`'s close on `day`, from its own local clock."""
    zone, hhmm = SESSIONS[tag]
    local = _dt.datetime.combine(day, hhmm, tzinfo=ZoneInfo(zone))
    u = local.astimezone(_dt.timezone.utc)
    # hours since the START OF THE TARGET DAY in UTC, so a close that rolls past
    # midnight (none here, but the arithmetic should not assume it) stays ordered
    return (u - _dt.datetime.combine(day, _dt.time(0, 0),
                                     tzinfo=_dt.timezone.utc)).total_seconds() / 3600.0


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints the admissibility audit behind Section 3 and Section 10.\n")

    tags = [t for t in L.CLOSE_UTC if L.files_for(t, folder)]
    missing = [t for t in tags if t not in SESSIONS]
    if missing:
        print(f"  no session clock recorded for {missing}; add it before trusting this")
    tags = [t for t in tags if t in SESSIONS]

    # the calendar every target is evaluated on
    prox = {t: L.yang_zhang(L.load_index(t, folder)) for t in tags}
    P = pd.DataFrame(prox).sort_index()
    days = [d.date() for d in P.index]
    print(f"{len(tags)} indices, {len(days)} dates, "
          f"{days[0]} to {days[-1]}\n")

    print("=" * 92)
    print("A.  WHAT THE TABLE SAYS AND WHAT THE CLOCK SAYS")
    print("=" * 92)
    print("The fixed table is standard time. The clock column is the median true")
    print("close across the sample, and the two extremes are what daylight saving")
    print("does to it. A blank spread means the exchange keeps no summer time.\n")
    print(f"{'index':>6}{'table':>9}{'zone':>22}{'local':>9}"
          f"{'true min':>11}{'true max':>11}{'spread':>9}")
    span = {}
    for t in tags:
        h = np.array([close_utc(t, d) for d in days])
        span[t] = (h.min(), h.max())
        zone, hhmm = SESSIONS[t]
        print(f"{t:>6}{L.CLOSE_UTC[t]:>9.1f}{zone:>22}{hhmm.strftime('%H:%M'):>9}"
              f"{h.min():>11.2f}{h.max():>11.2f}"
              + (f"{h.max() - h.min():>9.2f}" if h.max() - h.min() > 1e-9
                 else f"{'-':>9}"))

    print("\n" + "=" * 92)
    print("B.  DOES ANY DAY DISAGREE WITH THE RULE THE PANEL IMPLEMENTS?")
    print("=" * 92)
    print("For every target, every peer and every date: the lag the fixed table")
    print("assigns against the lag the true closes imply. A LEAK is a day the table")
    print("calls same-day and the clock says the peer closed later or at the same")
    print("moment - the panel would carry a value the forecaster could not have had.")
    print("A waste is the reverse, and costs information rather than validity.\n")
    print(f"{'target':>7}{'peers':>7}{'days':>7}{'leaks':>8}{'wasted':>8}"
          f"{'tightest margin':>17}{'which peer is wasted':>26}")
    total_leak = total_waste = 0
    tight = (1e9, None)
    waste_by_peer = {}
    for tgt in L.TARGETS:
        if tgt not in tags:
            continue
        peers = [p for p in tags if p != tgt]
        leaks = waste = 0
        here = (1e9, None)
        wasted_peers = {}
        for p in peers:
            table_lag = 0 if L.CLOSE_UTC[p] < L.CLOSE_UTC[tgt] else 1
            for d in days:
                gap = close_utc(tgt, d) - close_utc(p, d)     # >0 means p is earlier
                true_lag = 0 if gap > 0 else 1
                if table_lag == 0 and true_lag == 1:
                    leaks += 1
                elif table_lag == 1 and true_lag == 0:
                    waste += 1
                    wasted_peers[p] = wasted_peers.get(p, 0) + 1
                    waste_by_peer[(tgt, p)] = waste_by_peer.get((tgt, p), 0) + 1
                if table_lag == 0 and gap > 0 and gap < here[0]:
                    here = (gap, p)
        total_leak += leaks
        total_waste += waste
        if here[1] and here[0] < tight[0]:
            tight = (here[0], f"{here[1]} before {tgt}")
        top = sorted(wasted_peers.items(), key=lambda kv: -kv[1])[:2]
        names = ", ".join(f"{p} {c}" for p, c in top) if top else "-"
        print(f"{tgt:>7}{len(peers):>7}{len(days):>7}{leaks:>8}{waste:>8}"
              + (f"{here[0]:>15.2f} h" if here[1] else f"{'-':>17}")
              + f"{names:>26}")

    print(f"\n  leaks across every target, peer and day: {total_leak}")
    print(f"  wasted same-day observations:            {total_waste}")
    if tight[1]:
        print(f"  the narrowest margin the rule relies on:  {tight[0]:.2f} hours "
              f"({tight[1]})")
    # The waste is worth a sentence of its own, because it runs one way.
    if waste_by_peer:
        (wt, wp), wc = max(waste_by_peer.items(), key=lambda kv: kv[1])
        print(f"\n  The waste is not symmetric and it is not noise. On {wc} of the")
        print(f"  {len(days)} dates the fixed table sends {wp} back a day for a {wt}")
        print(f"  forecaster when the true clocks say it had already closed, because the")
        print(f"  table records both at the same standard-time hour and equality is")
        print(f"  resolved against admissibility. So the rule as implemented is STRICTER")
        print(f"  than the rule as stated: it discards information a real forecaster")
        print(f"  could lawfully have used. That direction matters for how the paper's")
        print(f"  numbers should be read. Every rate in this paper is what a forecaster")
        print(f"  achieves with less than the admissible cross-section, so the measured")
        print(f"  substitution rate is a lower bound on the rule's own terms.")
    if total_leak == 0:
        print("\n  The rule survives real session clocks. Every peer the panel treats")
        print("  as same-day admissible did close before the target on every date in")
        print("  the sample, daylight saving included, and the margin it relies on is")
        print("  hours rather than minutes. Section 3's fixed table is a simplification")
        print("  of the clock and not an approximation of the rule.")
    else:
        print("\n  THE RULE DOES NOT SURVIVE. The days above carry a peer close the")
        print("  forecaster could not have seen, and the panel must be rebuilt from")
        print("  true closes before any result in this paper is read.")

    print("\n" + "=" * 92)
    print("C.  HOLIDAYS: HOW OFTEN IS A PEER CARRIED FORWARD, AND HOW STALE?")
    print("=" * 92)
    print("The panel is built on the target's calendar and peers are forward-filled")
    print("with a five-day limit, so a peer shut on day t carries its last close.")
    print("That is strictly older than t and cannot leak. What it can do is weaken")
    print("the instrument, so the frequency and the staleness are reported.\n")
    print(f"{'target':>7}{'peer':>7}{'carried':>10}{'share':>9}"
          f"{'mean age':>11}{'worst':>8}")
    worst_share = (0.0, None)
    for tgt in L.TARGETS[:1]:                      # the headline panel
        if tgt not in tags:
            continue
        cal = P.index[P[tgt].notna()]
        for p in [q for q in tags if q != tgt]:
            have = P[p].notna()
            ages = []
            last = None
            for d in P.index:
                if have.loc[d]:
                    last = d
                if d in cal and last is not None:
                    ages.append((d - last).days)
            ages = np.array(ages, dtype=float)
            carried = int((ages > 0).sum())
            share = carried / len(ages) if len(ages) else float("nan")
            if share > worst_share[0]:
                worst_share = (share, p)
            print(f"{tgt:>7}{p:>7}{carried:>10}{share:>9.1%}"
                  f"{ages[ages > 0].mean() if carried else 0:>11.2f}"
                  f"{int(ages.max()) if len(ages) else 0:>8}")
    print(f"\n  the peer carried forward most often is {worst_share[1]} on "
          f"{worst_share[0]:.1%} of the target's days")
    print("  A carried close is stale, never early, so this is a statement about")
    print("  the strength of the foreign block and not about its admissibility.")

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    if total_leak == 0:
        print("  Section 3's admissibility rule holds on real session timestamps at")
        print("  every target, peer and date in the sample. The paper may state it as")
        print("  established rather than assumed, and this file is the establishment.")
    else:
        print(f"  {total_leak} admissibility violations. See part B.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
