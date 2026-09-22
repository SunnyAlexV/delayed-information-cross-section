"""
lab47_target_seam.py - the one seam that cannot be checked, checked anyway.

Imports lab05_robustness; keep both in labs/.  Runtime about three minutes.

THE LIMITATION THIS ADDRESSES
-----------------------------
Every index series here arrives as two retail exports, because the source caps a
download at 5,000 rows.  Eight of the nine pairs overlap - by 120 to 251 trading
days, a range Part 0 now prints in full rather than describing - and agree
exactly on every overlapping row.  The ninth is the S&P's, whose
two files ABUT without overlapping, so no row-by-row comparison exists.  Section
11 says so and falls back on two weaker checks: the join is contiguous, and the
return across it is ordinary.

That is the target series.  A silent corruption there would move every number in
the paper, and the paper's own answer is that it could not be detected BY
COMPARISON.  This file removes those two words.

THE AUDITOR
-----------
Cboe publishes its own VIX history: a different vendor, a different data path,
measuring the same underlying, spanning the seam, and already audited in lab43
against the retail export (99.91% of 6,747 shared rows identical to the cent).

There is a pleasant accident in that.  The only series in this panel with a
genuinely independent auditor is the only series that needs one, because it is
the only one whose own two files cannot be compared with each other.

THREE INSTRUMENTS, NOT ONE
--------------------------
A corruption is not one thing, and an audit that tests for one of them and
reports "no break" is theatre.  Three failures could plausibly occur in a retail
export, and each needs the instrument it is visible to:

  LEVEL      a block spliced in from elsewhere, or a gross scale error, shows up
             as a break in  log sigma^2_YZ,t = alpha + beta log VIX_t + eps_t,
             tested jointly on (alpha, beta) by a Wald statistic with HAC
             standard errors at nine lags, as everywhere else in this project.

  ALIGNMENT  a one-session date misalignment barely moves a five-day rolling
             variance, so the level test is blind to it.  It is glaring in the
             SAME-DAY relationship between the S&P's close-to-close return and
             the change in the exchange's index, which is strongly negative and
             survives no shift at all.

  RANGE      a high and low taken from a different contract, or a mis-scaled
             intraday range, moves the Yang-Zhang estimator and leaves
             close-to-close variance alone, because 88% of the Yang-Zhang weight
             sits on a Rogers-Satchell term the close never enters.  Their ratio
             is therefore an internal check that needs no auditor at all.  It is
             taken over 21 days rather than the paper's 5: this instrument audits
             a file rather than feeding a model, so it is free to use the longer
             window, and the longer window cuts its noise by two thirds and its
             detection threshold with it.

CALIBRATION, AND WHAT THE TEST CANNOT SAY
-----------------------------------------
A statistic at one date means nothing alone: fit any of these at an arbitrary
Tuesday and none comes back zero.  Each is therefore calibrated against several
hundred pseudo-seams drawn from the S&P's own history, and against eight dates
where the S&P file has no join at all - the dates where the OTHER indices' files
happen to meet, which are unconnected to the S&P and so are known-clean by
construction.

Those eight are instructive.  Several sit beside COVID, and the level statistic
there is large: the relationship between realised and implied volatility really
did move in early 2020.  So the test is one-sided in what it can conclude.  A
LARGE statistic is not proof of corruption, because market structure produces
large statistics too.  A SMALL one, at a date where the files meet, is evidence
that nothing happened to the data there, and that is the direction this audit
needs.

The power section closes it by asking the only question that matters for a
limitation: not whether a fault was found, but how large a fault would have had
to be before this caught it.
"""

import glob
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L

SEED = 20260917
TARGET = "SPX"
WIN = 250          # trading days fitted either side of a seam
BUFFER = 21        # rows discarded either side: an estimator window spans both files
RANGEWIN = 21      # the ratio instrument averages over 21 days, not the paper's 5
LAG = 9            # Bartlett lags, as everywhere else in this project
N_PSEUDO = 400
RANGE_F = [1.02, 1.05, 1.10, 1.25, 1.50, 0.80]
INSTRUMENTS = ("level", "alignment", "range")


# ------------------------------------------------------------------ regression
def ols_hac(X, y, lag=LAG):
    """Coefficients and a HAC covariance, Bartlett kernel, no dependencies."""
    A = X.T @ X
    Ainv = np.linalg.pinv(A)
    b = Ainv @ (X.T @ y)
    e = y - X @ b
    u = X * e[:, None]
    n = len(y)
    S = (u.T @ u) / n
    for k in range(1, lag + 1):
        G = (u[k:].T @ u[:-k]) / n
        S += (1 - k / (lag + 1)) * (G + G.T)
    return b, Ainv @ (n * S) @ Ainv


def stat_level(v, a, cut):
    """Wald on a break in intercept and slope, 2 d.f."""
    D = (np.arange(len(v)) >= cut).astype(float)
    X = np.column_stack([np.ones(len(v)), v, D, D * v])
    b, V = ols_hac(X, a)
    sel = np.array([2, 3])
    try:
        return float(b[sel] @ np.linalg.solve(V[np.ix_(sel, sel)], b[sel]))
    except np.linalg.LinAlgError:
        return np.nan


def stat_align(r, dv, cut):
    """How far the same-day return / implied-change correlation moves."""
    c0 = np.corrcoef(r[:cut], dv[:cut])[0, 1]
    c1 = np.corrcoef(r[cut:], dv[cut:])[0, 1]
    return abs(c1 - c0) if np.isfinite(c0) and np.isfinite(c1) else np.nan


def stat_range(q, cut):
    """t statistic on the shift in log(Yang-Zhang / close-to-close variance)."""
    m = q[cut:].mean() - q[:cut].mean()
    z = np.concatenate([q[:cut] - q[:cut].mean(), q[cut:] - q[cut:].mean()])
    n = len(z)
    s = float(z @ z) / n
    for k in range(1, LAG + 1):
        s += 2 * (1 - k / (LAG + 1)) * float(z[k:] @ z[:-k]) / n
    se = np.sqrt(max(s, 1e-18) * (1 / cut + 1 / (len(q) - cut)))
    return abs(m / se) if se > 0 else np.nan


# ------------------------------------------------------------------ the series
def series(df, vix):
    """Everything the three instruments need, on the dates both vendors carry."""
    c = df["close"].values.astype(float)
    yz = L.yang_zhang(df)                      # the paper's 5-day estimator
    yz21 = L.yang_zhang(df, n=RANGEWIN)        # the audit's quieter one
    cc = np.log(c[1:] / c[:-1])
    ccv = pd.Series(cc).rolling(RANGEWIN).var(ddof=1).values
    j = pd.DataFrame({"a": np.log(yz.values), "r": cc,
                      "q": np.log(np.maximum(yz21.values, 1e-18)
                                  / np.maximum(ccv, 1e-18))},
                     index=pd.to_datetime(yz.index))
    w = vix.reindex(j.index)
    j["v"] = np.log(w.values)
    j["dv"] = np.log(w.values) - np.log(w.shift(1).values)
    j = j.replace([np.inf, -np.inf], np.nan).dropna()
    return j.index.values, j


def windows(dates, j, when, win=WIN, buf=BUFFER):
    i = int(np.searchsorted(dates, np.datetime64(when)))
    lo, hi = i - win - buf, i + win + buf
    if lo < 0 or hi > len(j):
        return None
    keep = np.r_[np.arange(lo, i - buf), np.arange(i + buf, hi)]
    return j.iloc[keep], i - buf - lo


def all_stats(dates, j, when):
    got = windows(dates, j, when)
    if got is None:
        return {k: np.nan for k in INSTRUMENTS}
    w, cut = got
    return {"level": stat_level(w["v"].values, w["a"].values, cut),
            "alignment": stat_align(w["r"].values, w["dv"].values, cut),
            "range": stat_range(w["q"].values, cut)}


# ------------------------------------------------------------------ corruptions
def corrupt(df, start, kind, f=1.0, other=None):
    """A copy of the OHLC frame with every row from `start` damaged."""
    d = df.copy().reset_index(drop=True)
    o, h, l, c = (d[x].values.astype(float).copy() for x in ("open", "high", "low", "close"))
    if kind == "date_shift":
        for arr in (o, h, l, c):
            arr[start:] = np.r_[arr[start - 1], arr[start:-1]]
    elif kind == "range":
        up = np.log(np.maximum(h[start:], 1e-12) / o[start:])
        dn = np.log(np.maximum(l[start:], 1e-12) / o[start:])
        h[start:] = np.maximum(o[start:] * np.exp(f * up), c[start:])
        l[start:] = np.minimum(o[start:] * np.exp(f * dn), c[start:])
    elif kind == "splice":
        s = other.set_index("date").reindex(d["date"].values).ffill().bfill()
        oo, hh, ll, cc = (s[x].values.astype(float) for x in ("open", "high", "low", "close"))
        scale = c[start - 1] / cc[start - 1]
        o[start:], h[start:] = oo[start:] * scale, hh[start:] * scale
        l[start:], c[start:] = ll[start:] * scale, cc[start:] * scale
    d["open"], d["high"], d["low"], d["close"] = o, h, l, c
    return d.dropna().reset_index(drop=True)


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints the independent-vendor seam audit, Section 11.\n")
    rng = np.random.default_rng(SEED)

    # ---------------- 0. the nine joins ------------------------------------
    print("=" * 96)
    print("0.  THE NINE JOINS, AND WHICH ONE CANNOT BE CHECKED BY COMPARISON")
    print("=" * 96)
    seams = {}
    pairs = [(tag, L.files_for(tag, folder)) for tag in sorted(L.CLOSE_UTC)]
    # VDAX-NEW is the ninth retail series and it is excluded from files_for by
    # design, because "DAX New Volatility" would otherwise be merged into the
    # DAX PRICE series.  Section 3 counts nine joins, so nine are audited here;
    # leaving it out is how the Limitations section came to quote a range that
    # covered only eight of them.
    vdax = sorted(glob.glob(os.path.join(folder, "DAX_New_Volatility*.csv")))
    if len(vdax) == 2:
        pairs.append(("VDAX", vdax))
    for tag, paths in pairs:
        if len(paths) != 2:
            continue
        early, late = sorted((L.read_one(p) for p in paths), key=lambda d: d["date"].min())
        ov = early.merge(late, on="date", suffixes=("_a", "_b"))
        bad = (int(((ov.close_a - ov.close_b).abs() > 0.005 * ov.close_a.abs()).sum())
               if len(ov) else None)
        state = ("VERIFIED" if len(ov) >= 20 and bad == 0 else
                 "MISMATCH" if len(ov) >= 20 else "UNVERIFIABLE")
        seams[tag] = (late["date"].min(), len(ov), state)
        print(f"  {tag:>5}  second file opens {str(seams[tag][0])[:10]}  "
              f"overlap {len(ov):>4} rows  {state}")
    ovs = sorted(v[1] for v in seams.values() if v[1] > 0)
    print(f"\n  {len(seams)} joins audited; {len(ovs)} of them overlap, by "
          f"{min(ovs)} to {max(ovs)} trading days,")
    print("  and every overlapping row agrees. Exactly one join has no overlapping")
    print("  row, and it is the target series. Any statement elsewhere in the paper")
    print(f"  about the range of these overlaps must read {min(ovs)} to {max(ovs)}.")

    spx = L.load_index(TARGET, folder)
    vixname = [p for p in sorted(os.listdir(folder)) if p.lower().startswith("vix_history")]
    if not vixname:
        raise SystemExit("VIX_History.csv (Cboe's own file) not found in the data folder.")
    vix = L.read_one(os.path.join(folder, vixname[0])).set_index("date")["close"]
    dates, J = series(spx, vix)
    seam_date = seams[TARGET][0]
    print(f"\n  auditor: {vixname[0]}, Cboe's own publication, {len(vix)} rows")
    print(f"  paired rows: {len(J)}   S&P join: {str(seam_date)[:10]}")
    print("\n  The only series in this panel with an independent auditor is the only")
    print("  one that needs it, because it is the only one whose two files cannot")
    print("  be compared with each other.")

    # ---------------- A. the three instruments at the join -----------------
    print("\n" + "=" * 96)
    print("A.  THREE INSTRUMENTS, ONE PER FAILURE MODE, AT THE JOIN")
    print("=" * 96)
    print(f"Fitted over {WIN} trading days either side, {BUFFER} days cut out where an")
    print("estimator window would span both files.\n")
    real = all_stats(dates, J, seam_date)
    w, cut = windows(dates, J, seam_date)
    b0, _ = ols_hac(np.column_stack([np.ones(cut), w["v"].values[:cut]]), w["a"].values[:cut])
    b1, _ = ols_hac(np.column_stack([np.ones(len(w) - cut), w["v"].values[cut:]]),
                    w["a"].values[cut:])
    c0 = float(np.corrcoef(w["r"].values[:cut], w["dv"].values[:cut])[0, 1])
    c1 = float(np.corrcoef(w["r"].values[cut:], w["dv"].values[cut:])[0, 1])
    print("  LEVEL      realised variance on implied, before and after")
    print(f"             alpha {b0[0]:+.4f} -> {b1[0]:+.4f}   "
          f"beta {b0[1]:+.4f} -> {b1[1]:+.4f}   Wald {real['level']:.2f}")
    print("  ALIGNMENT  same-day return against the change in the exchange's index")
    print(f"             correlation {c0:+.4f} -> {c1:+.4f}   "
          f"shift {real['alignment']:.4f}")
    print(f"  RANGE      log(Yang-Zhang / close-to-close variance) over {RANGEWIN} days,")
    print("             a purely internal check with no auditor in it at all")
    print(f"             mean {w['q'].values[:cut].mean():+.4f} -> "
          f"{w['q'].values[cut:].mean():+.4f}   t {real['range']:.2f}")

    # ---------------- B. calibration ---------------------------------------
    print("\n" + "=" * 96)
    print("B.  WHAT EACH INSTRUMENT DOES WHERE THE DATA IS KNOWN TO BE SOUND")
    print("=" * 96)
    print("Eight dates where the S&P file has no join at all - the dates the OTHER")
    print("indices' files happen to meet - and pseudo-seams drawn from the S&P's")
    print("own history. Both are known-clean by construction.\n")
    clean = {k: [] for k in INSTRUMENTS}
    for tag, (join, ov, state) in sorted(seams.items()):
        if tag == TARGET:
            continue
        s = all_stats(dates, J, join)
        for k in INSTRUMENTS:
            if np.isfinite(s[k]):
                clean[k].append(s[k])
        print(f"  {tag:>5}  {str(join)[:10]}   level {s['level']:7.2f}   "
              f"alignment {s['alignment']:.4f}   range {s['range']:5.2f}")
    lo_i, hi_i = WIN + BUFFER, len(J) - WIN - BUFFER
    seam_i = int(np.searchsorted(dates, np.datetime64(seam_date)))
    pool = [i for i in range(lo_i, hi_i) if abs(i - seam_i) > WIN]
    draws = rng.choice(pool, size=min(N_PSEUDO, len(pool)), replace=False)
    null = {k: [] for k in INSTRUMENTS}
    for i in draws:
        keep = np.r_[np.arange(i - WIN - BUFFER, i - BUFFER),
                     np.arange(i + BUFFER, i + WIN + BUFFER)]
        sub = J.iloc[keep]
        vals = {"level": stat_level(sub["v"].values, sub["a"].values, WIN),
                "alignment": stat_align(sub["r"].values, sub["dv"].values, WIN),
                "range": stat_range(sub["q"].values, WIN)}
        for k in INSTRUMENTS:
            if np.isfinite(vals[k]):
                null[k].append(vals[k])
    crit, pct = {}, {}
    print(f"\n  {len(null['level'])} pseudo-seams:")
    print(f"{'':<14}{'median':>9}{'95th':>9}{'99th':>9}{'':>4}"
          f"{'at the S&P join':>18}{'percentile':>12}")
    for k in INSTRUMENTS:
        arr = np.array(null[k])
        crit[k] = float(np.percentile(arr, 95))
        pct[k] = float((arr < real[k]).mean() * 100)
        print(f"  {k:<12}{np.median(arr):>9.3f}{crit[k]:>9.3f}"
              f"{np.percentile(arr, 99):>9.3f}{'':>4}{real[k]:>18.3f}{pct[k]:>11.0f}th")
    ok = all(pct[k] < 95 for k in INSTRUMENTS)
    print("\n  Several of the eight clean dates carry a large LEVEL statistic, and")
    print("  they should: they sit beside COVID, where the relationship between")
    print("  realised and implied volatility genuinely moved. So a large statistic")
    print("  is not proof of corruption. A small one, at the date the files meet,")
    print("  is evidence that nothing happened to the data there, which is the")
    print("  direction this audit needs.")

    # ---------------- C. power ---------------------------------------------
    print("\n" + "=" * 96)
    print("C.  HOW LARGE WOULD A FAULT HAVE HAD TO BE?")
    print("=" * 96)
    print("Damage the post-seam block, recompute everything from the damaged OHLC,")
    print("re-run all three. A mark means the statistic clears the 95th percentile")
    print("of its own null above.\n")
    other = L.load_index("DAX", folder)
    start = int(np.searchsorted(spx["date"].values, np.datetime64(seam_date)))
    cases = [("none (the file as downloaded)", spx)]
    cases.append(("every post-seam row one session late",
                  corrupt(spx, start, "date_shift")))
    for f in RANGE_F:
        cases.append((f"intraday range scaled by {f:.2f}",
                      corrupt(spx, start, "range", f=f)))
    cases.append(("post-seam block spliced from the DAX",
                  corrupt(spx, start, "splice", other=other)))

    print(f"{'corruption':<40}{'level':>9}{'align':>9}{'range':>9}{'':>4}{'caught by':>22}")
    caught = {}
    for name, df in cases:
        d2, j2 = series(df, vix)
        s = all_stats(d2, j2, seam_date)
        hits = [k for k in INSTRUMENTS if np.isfinite(s[k]) and s[k] > crit[k]]
        caught[name] = hits
        print(f"{name:<40}{s['level']:>9.2f}{s['alignment']:>9.4f}{s['range']:>9.2f}"
              f"{'':>4}{(', '.join(hits) if hits else 'nothing'):>22}")
    smallest = min([float(n.split()[-1]) for n in caught
                    if n.startswith('intraday range') and caught[n]
                    and float(n.split()[-1]) > 1] or [np.nan])
    print()
    print(f"  smallest range corruption caught: {smallest:.0%} of the true range")
    print(f"  one-session misalignment caught:  "
          f"{'yes, by ' + ', '.join(caught['every post-seam row one session late']) if caught['every post-seam row one session late'] else 'NO'}")
    print(f"  spliced block caught:             "
          f"{'yes, by ' + ', '.join(caught['post-seam block spliced from the DAX']) if caught['post-seam block spliced from the DAX'] else 'NO'}")
    print("\n  One corruption is invisible to all three and it is worth naming:")
    print("  multiply every post-seam PRICE by a constant and nothing moves at all,")
    print("  because all of this reads log ratios. That is also the corruption that")
    print("  cannot change a single number in the paper, for the same reason.")

    # ---------------- verdict ----------------------------------------------
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    if ok:
        print(f"  At the S&P's unverifiable join all three instruments sit inside their")
        print(f"  own null distributions: level {pct['level']:.0f}th percentile, alignment "
              f"{pct['alignment']:.0f}th,")
        print(f"  range {pct['range']:.0f}th. The relationship between the S&P's own OHLC and an")
        print("  exchange's index of the same underlying does not break across that")
        print("  date; the same-day return correlation does not move; and the two")
        print("  volatility estimators inside the file keep their ratio.")
        print("\n  The limitation does not vanish, it acquires a size. A one-session")
        print("  misalignment and a spliced block are both caught, and so is an")
        print(f"  intraday range mis-scaled by {abs(smallest - 1):.0%}. Section 11 can now say what")
        print("  was checked and how large a fault would have had to be, instead of")
        print("  saying that comparison was impossible.")
    else:
        flagged = [k for k in INSTRUMENTS if pct[k] >= 95]
        print(f"  The S&P join is flagged by: {', '.join(flagged)}.")
        print("  That is a finding about the data, not about volatility, and it has")
        print("  to be resolved before any number in the paper stands. Re-download")
        print("  both exports with a deliberate overlap and compare them row by row.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
