"""
lab03_crosssection.py - does a live cross-section raise the ceiling?

Self-contained: NumPy + pandas.  Put all index CSVs in one folder and run.
Files may be split into chunks: SPX_1.csv, SPX_2.csv, ... They are merged here,
and the OVERLAP between chunks is VERIFIED rather than trusted.

THE QUESTION
------------
Our paper showed the threshold rule is the Bayes rule for a median-split target,
so no model can beat its ceiling.  The only way forward is to RAISE the ceiling,
which means raising the correlation between what you can see and what you want.

A forecaster whose own-market data is delta days stale can still see markets that
closed EARLIER TODAY.  When the S&P closes at 21:00 UTC, Tokyo (06:00), Hong Kong
(08:00), Mumbai (10:00), London and Frankfurt (16:30) have all already closed.
That is a real asymmetry, free, every trading day - no artificial delay needed.

So: does adding those live closes raise the multiple correlation enough to matter?
If not, no architecture helps, and we stop here.  That is the point of this file.

WHAT IT DOES NOT DO
-------------------
It does not fit a forecasting model.  Multiple correlation is an IN-SAMPLE upper
bound on what any linear predictor could achieve; if the bound does not move,
nothing downstream can.  If it does move, the next file fits something honest.
"""

import os, glob, re
import numpy as np
import pandas as pd
from math import asin, pi

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
DATA_DIR = next(
    (d for d in [os.path.join(_HERE, "..", "data"),
                 os.path.join(os.path.expanduser("~"), "Desktop", "indices"),
                 os.path.join(os.path.expanduser("~"), "Desktop", "data"),
                 _HERE, "."]
     if os.path.isdir(d) and glob.glob(os.path.join(d, "SPX*.csv"))),
    os.path.join(_HERE, "..", "data"))

# Close times in UTC (standard time; DST shifts by an hour but never reorders
# the blocks, and only the ORDER matters here).
# A volatility INDEX is not a price index.  "DAX New Volatility" contains the
# substring "dax", and "VDAX_a.csv" would too, so an implied-volatility download
# sitting beside the index exports can be merged into the DAX PRICE series.
# Excluded by name, so the two can share a folder safely.
NOT_AN_INDEX = ["volatility", "vix", "vstoxx", "vdax", "vhsi", "implied"]


def _is_index_file(path):
    import os as _os
    b = _os.path.basename(path).lower()
    return not any(x in b for x in NOT_AN_INDEX)


CLOSE_UTC = {"N225": 6.0, "AXJO": 6.0, "HSI": 8.0, "NSEI": 10.0,
             "FTSE": 16.5, "DAX": 16.5, "BVSP": 21.0, "SPX": 21.0}
TARGET  = "SPX"
WINDOW  = 5          # days in the volatility proxy
HORIZON = 5          # predict this many days ahead
DELAYS  = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]
MIN_OVERLAP_ROWS = 20


# ----------------------------------------------------------------------
# 1. Load and merge chunks, verifying the seam
# ----------------------------------------------------------------------
def read_one(path):
    df = pd.read_csv(path)
    df.columns = [str(c).replace('"', "").strip().lower() for c in df.columns]
    num = lambda c: pd.to_numeric(
        df[c].astype(str).str.replace('"', "").str.replace(",", "").str.strip(),
        errors="coerce")
    px = "price" if "price" in df.columns else "close"
    out = pd.DataFrame({"close": num(px), "open": num("open"),
                        "high": num("high"), "low": num("low")})
    out["date"] = pd.to_datetime(df["date"].astype(str).str.replace('"', "").str.strip(),
                                 errors="coerce", format="mixed")
    out = out.dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return out


def load_index(tag, folder):
    paths = [q for q in sorted(glob.glob(os.path.join(folder, f"{tag}*.csv")))
             if _is_index_file(q)]
    if not paths:
        raise FileNotFoundError(f"no files matching {tag}*.csv in {folder}")
    parts = [read_one(p) for p in paths]
    merged = parts[0]
    for nxt in parts[1:]:
        a, b = merged, nxt
        ov = a.merge(b, on="date", suffixes=("_a", "_b"))
        if len(ov) >= MIN_OVERLAP_ROWS:
            d = (ov["close_a"] - ov["close_b"]).abs()
            tol = 0.005 * ov["close_a"].abs().clip(lower=1e-9)   # 0.5% tolerance
            bad = int((d > tol).sum())
            print(f"    seam {tag}: {len(ov)} overlapping rows, {bad} mismatched "
                  f"(max abs diff {d.max():.4f})  -> {'VERIFIED' if bad == 0 else 'MISMATCH'}")
        else:
            # No overlap, so the chunks cannot be cross-checked against each
            # other.  That is not a reason to say nothing about the join.  Two
            # weaker checks are still available, and both are reported, because
            # a seam we could not verify has to be named as such in the paper.
            #   1. CONTIGUITY.  Exactly one business day between the last row of
            #      one chunk and the first of the next means no session is
            #      missing and none is duplicated.
            #   2. NO LEVEL SHIFT.  A botched join usually shows up as a jump.
            #      The return across the seam is compared with the series' own
            #      daily standard deviation.
            last, first = a.iloc[-1], b.iloc[0]
            gapd = int(np.busday_count(last["date"].date(), first["date"].date()))
            r = np.log(a["close"] / a["close"].shift(1)).dropna()
            jump = float(np.log(first["close"] / last["close"]))
            sd = float(r.std())
            print(f"    seam {tag}: only {len(ov)} overlapping rows -> UNVERIFIABLE "
                  f"by cross-check; falling back to:")
            print(f"      contiguity : {last['date'].date()} -> {first['date'].date()}, "
                  f"{gapd} business day{'' if gapd == 1 else 's'} "
                  f"-> {'no missing session' if gapd == 1 else 'GAP, INVESTIGATE'}")
            print(f"      level shift: return across the seam {jump:+.5f} = "
                  f"{abs(jump)/sd:.2f} sd of this series' daily returns "
                  f"-> {'no discontinuity' if abs(jump) < 3 * sd else 'DISCONTINUITY'}")
        merged = (pd.concat([a, b]).drop_duplicates("date", keep="first")
                    .sort_values("date").reset_index(drop=True))
    dups = int(merged["date"].duplicated().sum())
    if dups:
        print(f"    WARNING: {tag} merged series has {dups} duplicate dates")
    return merged


# ----------------------------------------------------------------------
# 2. Volatility proxy (Yang-Zhang) and the decision variable
# ----------------------------------------------------------------------
def yang_zhang(df, n=WINDOW):
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    oo = np.log(o[1:] / c[:-1]); uu = np.log(h[1:] / o[1:])
    dd = np.log(l[1:] / o[1:]);  cc = np.log(c[1:] / o[1:])
    rs = uu * (uu - cc) + dd * (dd - cc)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    yz = (pd.Series(oo).rolling(n).var(ddof=1).values
          + k * pd.Series(cc).rolling(n).var(ddof=1).values
          + (1 - k) * pd.Series(rs).rolling(n).mean().values)
    return pd.Series(np.maximum(yz, 1e-12), index=df["date"].values[1:])


def decision_var(v, med=252):
    """log(proxy / its own trailing median) - zero-centred, comparable across markets."""
    m = v.rolling(med, min_periods=30).median()
    return np.log(v / m)


def mult_corr(X, y):
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    if len(y) < 50: return np.nan, 0
    A = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.corrcoef(A @ b, y)[0, 1]), len(y)


def acc(rho):
    return np.nan if not np.isfinite(rho) else 0.5 + asin(max(min(rho, 1), -1)) / pi


# ----------------------------------------------------------------------
def main(folder=DATA_DIR, med=252):
    tags = [t for t in CLOSE_UTC
            if any(_is_index_file(q)
                   for q in glob.glob(os.path.join(folder, f"{t}*.csv")))]
    print(f"indices found: {tags}\n  merging and auditing seams:")
    raw = {t: load_index(t, folder) for t in tags}

    dv = {t: decision_var(yang_zhang(raw[t]), med) for t in tags}
    panel = pd.DataFrame(dv).sort_index()
    panel = panel.dropna(how="all")
    print(f"\npanel: {len(panel)} dates, {panel.index.min().date()} to {panel.index.max().date()}")
    print("missing per index (holidays etc.):")
    for t in tags:
        print(f"    {t:>5}  {panel[t].isna().mean()*100:5.1f}%")

    # peers whose close PRECEDES the target's close on the same calendar day
    peers = [t for t in tags if t != TARGET and CLOSE_UTC[t] < CLOSE_UTC[TARGET]]
    same  = [t for t in tags if t != TARGET and CLOSE_UTC[t] >= CLOSE_UTC[TARGET]]
    print(f"\ntarget {TARGET} (close {CLOSE_UTC[TARGET]:.1f} UTC)")
    print(f"  usable as LIVE (close earlier same day): {peers}")
    print(f"  excluded, closes at or after target:     {same}")

    P = panel.copy()
    # forward-fill peers only (a holiday abroad means the last known close stands);
    # never fill the target.
    P[peers] = P[peers].ffill(limit=5)
    a_t = P[TARGET].values
    idx = np.arange(len(P))
    b = np.full(len(P), np.nan)
    b[:-HORIZON] = a_t[HORIZON:]            # the outcome: target's decision var, h ahead

    print(f"\nCEILING ACCURACY = 1/2 + arcsin(rho)/pi   (h = {HORIZON})")
    print(f"{'delta':>6}{'own only':>20}{'own + live peers':>22}{'gain':>12}{'n':>8}")
    for d in DELAYS:
        own = np.full((len(P), 1), np.nan)
        own[d:, 0] = a_t[:len(P) - d]
        r1, n1 = mult_corr(own, b)
        X = np.column_stack([own[:, 0], P[peers].values])   # peers at t, own at t-d
        r2, n2 = mult_corr(X, b)
        print(f"{d:>6}{r1:>9.3f} ->{acc(r1)*100:>7.1f}%{r2:>11.3f} ->{acc(r2)*100:>7.1f}%"
              f"{(acc(r2)-acc(r1))*100:>+11.1f} pts{n2:>8}")

    print("\nMultiple correlation is an in-sample upper bound: if the gain column is")
    print("flat, no model of any kind can exploit the cross-section, and we stop.")



if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else DATA_DIR)
