"""
lab64_coupling_common_window.py - the coupling relation with every target
scored on the SAME calendar window.  Imports lab53; keep both in labs/.

WHY THIS FILE EXISTS
--------------------
Section 6's coupling result compares R(55) across eight equity targets and
reads an ordering off it.  The comparison is only meaningful if the eight rates
are measured over the same period, and in lab53 they are not.

Two of the eight end in 2021 because their vendor files do:

    KSE100   3,569 test days, first 2006-10-13   (frontier/KSE100_yahoo.csv,
                                                  last row 2021-09-30)
    CSEALL   3,429 test days, first 2006-11-30   (frontier/CSEALL_yahoo.csv,
                                                  last row 2021-09-30)

against 4,722 to 4,948 days for the other six, which run to September 2026.
So KSE-100 and Colombo are missing the whole 2021-2026 stretch, including the
2022 inflation shock, while the other six contain it.

That matters here more than it would elsewhere, because this project's own
central conditional finding is that R(delta) depends on the market state: 81%
in the stressed tercile, 53% in the middle, indeterminate in the calm.  A rate
averaged over a window that excludes a major stress episode is not comparable
to one that includes it.  And these two targets are not incidental points:
Section S17 identifies them as the two far-left observations carrying the
leverage in the fitted line, so the confound sits exactly where the relation is
most fragile.

THE FIX, AND WHY IT IS THIS ONE
-------------------------------
Intersecting the eight trading calendars day by day was the first idea and is
the wrong one: the markets keep different holidays, so the intersection is a
gappy index, and every rolling window in the scorer would silently span those
gaps.  Cutting all eight at a common END DATE instead leaves each target on its
own contiguous calendar and removes the confound just as completely.  The cost
is the 2021-2026 tail on six targets, which is the price of comparing like with
like.

This does not repair the OTHER problem with those two series - that they come
from a second vendor and carry none of the nine-join audit Section 2 applies to
the primary exports.  Only a second source for them can do that.  What this
file establishes is whether the ordering survives once the period is held
fixed, which is the question a reader of Figure 3 is entitled to ask.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab05_robustness as L
import lab08_implied_vol as IV
import lab53_no_options_targets as C

# The last date every one of the eight can be scored on.  Discovered from the
# panels rather than typed, so a longer export moves it automatically.
END = None


def spearman(x, y):
    return float(np.corrcoef(C._rank(x), C._rank(y))[0, 1])


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("The coupling relation of Section 6, with all eight targets cut at a\n"
          "common end date so that no rate is averaged over a different era.\n")
    if C.frontier_dir(folder) is None:
        raise SystemExit("data/frontier/ is missing; lab53's four series live there.")

    raw_iv = {t: IV.load_iv(folder, t) for t in C.IV_TAGS}
    targets = list(C.NEW) + C.MATCHED
    panels = {t: C.build(folder, t, raw_iv) for t in targets}

    # the common end date is the earliest LAST test day across the eight
    last = {}
    for t in targets:
        own, P, iv, y, idx, peers, same, D, coup = panels[t]
        last[t] = pd.Timestamp(D.index[idx[-1]])
    end = min(last.values())
    binding = [t for t in targets if last[t] == end]
    print("last scorable day, per target")
    for t in sorted(targets, key=lambda k: last[k]):
        print(f"  {t:>8}  {str(last[t])[:10]}"
              f"{'   <- binding' if last[t] == end else ''}")
    print(f"\n  common end date: {str(end)[:10]}  (set by {', '.join(binding)})\n")

    print("=" * 96)
    print("A.  WHAT EACH TARGET LOSES, AND WHAT ITS COUPLING BECOMES")
    print("=" * 96)
    print(f"{'target':>8}{'days full':>11}{'days common':>13}{'lost':>7}"
          f"{'coupling full':>15}{'coupling common':>17}")

    cut = {}
    for t in targets:
        own, P, iv, y, idx, peers, same, D, coup = panels[t]
        keep = idx[pd.DatetimeIndex(D.index[idx]) <= end]
        # coupling recomputed on the surviving days, exactly as lab53 defines it
        finp = np.isfinite(P[keep])
        cntp = finp.sum(axis=1)
        mb = np.where(cntp > 0,
                      np.where(finp, P[keep], 0.0).sum(axis=1) / np.maximum(cntp, 1),
                      np.nan)
        a = D[t].values
        okc = np.isfinite(a[keep]) & np.isfinite(mb)
        cc = float(np.corrcoef(a[keep][okc], mb[okc])[0, 1])
        cut[t] = (keep, cc)
        print(f"{t:>8}{len(idx):>11}{len(keep):>13}{len(idx)-len(keep):>7}"
              f"{coup:>15.3f}{cc:>17.3f}")

    print("\n" + "=" * 96)
    print("B.  R(55) ON THE COMMON WINDOW")
    print("=" * 96)
    print("Scored exactly as lab53 scores it: same benchmark, same walk-forward,\n"
          "same seven-index block; only the evaluation days differ.\n")
    print(f"{'target':>8}{'R(55) full':>13}{'R(55) common':>15}{'move':>9}"
          f"{'coupling common':>18}")

    rows = []
    for t in targets:
        own, P, iv, y, idx, peers, same, D, coup = panels[t]
        keep, cc = cut[t]
        out = {}
        for tag, ix in (("full", idx), ("common", keep)):
            yb = y[ix]
            BENCH = L.bench_mean(y, ix)
            s0 = C.r2(yb, IV.walk(own, [], y, ix, 0), bench=BENCH)
            s_own = C.r2(yb, IV.walk(own, [], y, ix, 55), bench=BENCH)
            s_blk = C.r2(yb, IV.walk(own, [P], y, ix, 55), bench=BENCH)
            den = s0 - s_own
            out[tag] = (s_blk - s_own) / den if den > 1e-9 else np.nan
        rows.append((t, out["full"], out["common"], cc))
        print(f"{t:>8}{out['full']*100:>12.1f}%{out['common']*100:>14.1f}%"
              f"{(out['common']-out['full'])*100:>+8.1f}{cc:>18.3f}")

    cp = np.array([r[3] for r in rows])
    rf = np.array([r[1] for r in rows])
    rc = np.array([r[2] for r in rows])
    ok = np.isfinite(cp) & np.isfinite(rc) & np.isfinite(rf)

    print("\n" + "=" * 96)
    print("C.  DOES THE ORDERING SURVIVE HOLDING THE PERIOD FIXED?")
    print("=" * 96)
    print(f"  full window    Pearson {np.corrcoef(cp[ok], rf[ok])[0,1]:+.3f}"
          f"   Spearman {spearman(cp[ok], rf[ok]):+.3f}")
    print(f"  common window  Pearson {np.corrcoef(cp[ok], rc[ok])[0,1]:+.3f}"
          f"   Spearman {spearman(cp[ok], rc[ok]):+.3f}")
    print("\n  leave one out, common window")
    print(f"{'dropped':>10}{'Pearson':>10}{'Spearman':>11}")
    worst_p, worst_s = 1.0, 1.0
    names = [r[0] for r in rows]
    for k in range(len(names)):
        m = ok.copy()
        m[k] = False
        p = float(np.corrcoef(cp[m], rc[m])[0, 1])
        s = spearman(cp[m], rc[m])
        worst_p, worst_s = min(worst_p, p), min(worst_s, s)
        print(f"{names[k]:>10}{p:>+10.3f}{s:>+11.3f}")
    print(f"\n  worst leave-one-out on the common window: "
          f"Pearson {worst_p:+.3f}, Spearman {worst_s:+.3f}")
    print("\n  If these are close to the full-window values the ordering claim is\n"
          "  about coupling.  If they are not, part of it was about WHEN each\n"
          "  target happened to be measured, and Section 6 has to say so.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
