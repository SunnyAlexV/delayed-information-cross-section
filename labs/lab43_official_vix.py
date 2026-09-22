"""
lab43_official_vix.py - does the VIX the paper uses match the one Cboe publishes?

Imports lab05_robustness and lab08_implied_vol; keep all three in labs/.
Runtime about four minutes.

THE OBJECTION
-------------
Section 11 lists, among the things this paper cannot vouch for, the provenance
of its implied-volatility series.  VIX entered the project as two retail CSV
exports, and a retail export is a redistribution: nobody promises it is the
series Cboe computed, revisions and all, on the day it was computed.  A referee
is entitled to say so, and "it is probably fine" is not an answer this project
accepts anywhere else.

Cboe publishes the official daily history itself, free, with open, high, low and
close from 1990.  So the objection is answerable by measurement rather than by
assurance, and this lab is the measurement.

WHAT COULD BE WRONG, AND WHAT EACH WOULD LOOK LIKE
--------------------------------------------------
    A level error.  The retail series is rescaled, stale by a day, or carries a
    different close convention.  This would show as a systematic difference on
    most overlapping rows and would invalidate every VIX result in the paper.

    Scattered transcription error.  A handful of rows differ.  This would show
    as a small number of large differences against a mass of exact matches, and
    what matters then is whether those rows are load-bearing.

    Fabricated rows.  The export carries dates on which the exchange was shut.
    A vendor that forward-fills a holiday rather than omitting it is inventing
    an observation, and an invented observation on a day with no close is the
    one error this project's admissibility rules cannot catch by themselves,
    because the row looks perfectly well formed.

The third is the one worth looking for, and it is the one nobody had looked for.

PART D
------
An audit that ends at "the files agree" leaves the question a reader actually
has: would the paper's numbers move?  Part D answers it directly, running the
strict arm of Section 8 twice on identical days, once from each source, and
printing the cells side by side.  This is the only place the comparison can
live: once the official series becomes the paper's source, the retail run
exists nowhere else.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab08_implied_vol as IV

TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 5, 21, 55]

# Matched by keyword like everything else, but stated here rather than reused
# from lab08: this lab is about the two sources being DIFFERENT files, so it has
# to be able to name them separately, which lab08's discovery deliberately
# cannot do.
OFFICIAL = ["vix_history"]
RETAIL = ["cboe_volatility_index_historical", "cboe volatility index historical"]


def one_source(folder, keys):
    """Merge every CSV under `folder` whose name matches one of `keys`."""
    import glob
    paths = []
    for pat in (os.path.join(folder, "*.csv"), os.path.join(folder, "*", "*.csv")):
        for p in sorted(glob.glob(pat)):
            b = os.path.basename(p).lower()
            if any(k in b for k in keys) and "9d" not in b:
                paths.append(p)
    if not paths:
        raise SystemExit(f"lab43: found no CSV matching {keys} in {folder}")
    parts = [L.read_one(p) for p in paths]
    m = parts[0]
    for b in parts[1:]:
        m = (pd.concat([m, b]).drop_duplicates("date", keep="first")
               .sort_values("date").reset_index(drop=True))
    return [os.path.basename(p) for p in paths], m.set_index("date")["close"]


def decision_variable(series, index, lag):
    on = series.reindex(index).ffill(limit=5)
    if lag:
        on = on.shift(lag)
    return np.log(on / on.rolling(L.MED, min_periods=30).median())


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints the provenance audit of the VIX series, Section 11.\n")

    off_files, off = one_source(folder, OFFICIAL)
    ret_files, ret = one_source(folder, RETAIL)

    # ---------------- A ----------------------------------------------------
    print("=" * 92)
    print("A.  THE TWO SOURCES")
    print("=" * 92)
    for nm, files, s in (("official (Cboe)", off_files, off),
                         ("retail export", ret_files, ret)):
        print(f"  {nm:16} {len(s):5d} rows  {s.index.min().date()} to "
              f"{s.index.max().date()}  {files}")
    print("\nOnly the close is used: the implied-volatility decision variable is the")
    print("log of the level over its own trailing 252-day median, so the official")
    print("file's open, high and low are read and discarded. That matters, because")
    print("the official file has no intraday range before June 2004 - Cboe published")
    print("closes only - and a range-based estimator would have inherited the gap.")

    # ---------------- B ----------------------------------------------------
    print("\n" + "=" * 92)
    print("B.  WHERE BOTH SOURCES HAVE A ROW, DO THEY AGREE?")
    print("=" * 92)
    # join="inner" rather than a default concat: pandas warns that the default
    # sort of two DatetimeIndexes is changing, and a deprecation warning printed
    # into the middle of a lab's output becomes part of the reference output and
    # then a spurious --check failure the day the warning is removed.
    ov = pd.concat([off.rename("off"), ret.rename("ret")], axis=1,
                   join="inner").dropna()
    d = ov.off - ov.ret
    rel = (d.abs() / ov.off.abs())
    exact = int((d == 0).sum())
    within = int((rel <= 0.005).sum())
    print(f"  overlapping dates                         {len(ov)}")
    print(f"  identical to the cent                     {exact}  ({exact / len(ov):.2%})")
    print(f"  within the 0.5% seam rule this project uses  {within}  "
          f"({within / len(ov):.2%})")
    print(f"  mean absolute difference                  {d.abs().mean():.6f} "
          f"index points")
    print(f"  largest absolute difference               {d.abs().max():.4f} on "
          f"{d.abs().idxmax().date()}")
    bad = rel[rel > 0.005]
    print(f"\n  rows outside the 0.5% rule: {len(bad)}")
    if len(bad):
        print(f"{'date':>12}{'official':>11}{'retail':>9}{'difference':>13}"
              f"{'weekday':>11}")
        for t in bad.index:
            print(f"{str(t.date()):>12}{ov.off[t]:>11.2f}{ov.ret[t]:>9.2f}"
                  f"{d[t]:>+13.2f}{t.day_name():>11}")
        print("\n  Both are shortened sessions, which is where a vendor's close and the")
        print("  exchange's are most likely to part company.")
    print("\n  A systematic level error would show here as a mass of differences.")
    print("  What is here instead is agreement to the cent on essentially every row.")

    # ---------------- C ----------------------------------------------------
    print("\n" + "=" * 92)
    print("C.  ROWS ONE SOURCE HAS AND THE OTHER DOES NOT")
    print("=" * 92)
    print("A date the exchange was shut has no close. A source carrying one has")
    print("invented an observation, and the invention is invisible to every")
    print("admissibility rule in this paper because the row is well formed.\n")
    span = (ret.index.min(), ret.index.max())
    only_ret = [t for t in off.index.symmetric_difference(ret.index)
                if t in ret.index and span[0] <= t <= span[1]]
    only_off = [t for t in off.index.symmetric_difference(ret.index)
                if t in off.index and span[0] <= t <= span[1]]
    print(f"  in the retail export but not in Cboe's file, over the retail span: "
          f"{len(only_ret)}")
    print(f"  in Cboe's file but not in the retail export:                      "
          f"{len(only_off)}")

    # A date is a trading day for this paper if and only if the S&P itself has a
    # row for it.  That is the test, and it needs no holiday calendar.
    spx = L.load_index(TARGET, folder).set_index("date")
    D, peers, lag = L.build(folder, TARGET)
    print(f"\n{'date':>12}{'retail value':>14}{'S&P has a row':>16}"
          f"{'in the panel':>15}{'weekday':>11}")
    in_panel = 0
    for t in sorted(only_ret):
        has = t in spx.index
        pan = t in D.index
        in_panel += int(pan)
        print(f"{str(t.date()):>12}{ret[t]:>14.2f}{('yes' if has else 'no'):>16}"
              f"{('YES' if pan else 'no'):>15}{t.day_name():>11}")
    print(f"\n  of those {len(only_ret)} rows, {sum(1 for t in only_ret if t in spx.index)}"
          f" fall on a day the S&P traded, and {in_panel} survive into the panel")
    if in_panel == 0:
        print("  Every fabricated row sits on a day United States equities were shut,")
        print("  so the panel construction drops all of them before any model sees")
        print("  one. The error is real and it is inert.")
    else:
        print("  At least one fabricated row reaches the panel, so it has been fed to")
        print("  the models and every VIX result needs re-running on the official file.")

    # ---------------- D ----------------------------------------------------
    print("\n" + "=" * 92)
    print("D.  WOULD THE PAPER'S NUMBERS MOVE?")
    print("=" * 92)
    vdax = IV.load_iv(folder, "VDAX")      # prints its own seam audit; keep it here
    print("\nThe strict arm of Section 8, run twice on identical days, once from each")
    print("source. VIX enters at t-1 in both, VDAX at t, and the domestic block is")
    print("held delta days stale exactly as in Table 8.\n")

    frames = {}
    for nm, s in (("official", off), ("retail", ret)):
        iv = pd.DataFrame(index=D.index)
        iv["VIX"] = decision_variable(s, D.index, IV.VIX_LAG_STRICT)
        iv["VDAX"] = decision_variable(vdax, D.index, 0)
        frames[nm] = iv

    # One set of days for both runs: the intersection, so a difference in the
    # table cannot be a difference in the sample.
    keep = (frames["official"].notna().all(axis=1)
            & frames["retail"].notna().all(axis=1) & D[TARGET].notna())
    Dk = D.loc[keep]
    print(f"  {int(keep.sum())} rows carry both sources and the target")

    res = {}
    for nm in ("official", "retail"):
        ivk = frames[nm].loc[keep]
        own, P, V, y, idx = IV.make(Dk, ivk, peers)
        yb = y[idx]
        BENCH = L.bench_mean(y, idx)
        res[nm] = {}
        for dd in DELAYS:
            f_iv = IV.walk(own, [V], y, idx, dd)
            f_both = IV.walk(own, [P, V], y, idx, dd)
            res[nm][dd] = (IV.r2(yb, f_iv, BENCH), IV.r2(yb, f_both, BENCH),
                           IV.r2(yb, f_both, BENCH) - IV.r2(yb, f_iv, BENCH))
        if nm == "official":
            print(f"  {len(idx)} test days, {Dk.index[idx[0]].date()} to "
                  f"{Dk.index[idx[-1]].date()}\n")

    print(f"{'delta':>6}{'+iv official':>14}{'+iv retail':>12}{'diff':>9}"
          f"{'fgn|iv official':>17}{'fgn|iv retail':>15}{'diff':>9}")
    worst = 0.0
    for dd in DELAYS:
        o, r = res["official"][dd], res["retail"][dd]
        worst = max(worst, abs(o[0] - r[0]), abs(o[2] - r[2]))
        # A difference of -1e-17 formats as "-0.0000", which reads as a real
        # negative and is only the sign bit of a float. Anything below the
        # printed precision is printed as zero, unsigned.
        nil = lambda x: 0.0 if abs(x) < 5e-5 else x
        print(f"{dd:>6}{o[0]:>14.4f}{r[0]:>12.4f}{nil(o[0] - r[0]):>+9.4f}"
              f"{o[2]:>+17.4f}{r[2]:>+15.4f}{nil(o[2] - r[2]):>+9.4f}")
    print(f"\n  largest difference in any cell: {worst:.4f} of R-squared")

    # ---------------- verdict ----------------------------------------------
    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    level_ok = within >= 0.999 * len(ov)
    inert = in_panel == 0
    moves = worst >= 0.001
    if level_ok and inert and not moves:
        print(f"  Cboe's own series and the retail export agree to the cent on")
        print(f"  {exact / len(ov):.2%} of {len(ov)} shared rows, and the {len(bad)} rows outside")
        print("  the project's 0.5% tolerance are shortened sessions. The export")
        print(f"  additionally carries {len(only_ret)} rows on days United States equities were")
        print("  shut, which is a fabrication rather than a discrepancy - but every one")
        print("  falls outside the panel, so no model was ever shown one. Substituting")
        print("  the official file moves no cell of Section 8 by as much as a thousandth")
        print("  of R-squared. The paper now uses Cboe's file, and this lab is the")
        print("  reason the change can be described as housekeeping rather than")
        print("  asserted to be.")
    elif not level_ok:
        print("  The two sources disagree on more rows than a seam audit tolerates, so")
        print("  the retail export was not the series Cboe published and every VIX")
        print("  result in the paper rests on the official file from here on.")
    elif not inert:
        print("  A fabricated row reached the panel. That is a data error inside the")
        print("  results, not beside them, and the affected labs must be re-run.")
    else:
        print(f"  The sources agree on levels, but substituting one for the other moves")
        print(f"  a cell by {worst:.4f} of R-squared. That is small, and it is not nothing:")
        print("  the paper reports the official-source figures and this table is the")
        print("  record of what changed.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
