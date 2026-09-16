"""
lab04_walkforward.py - the out-of-sample test of the cross-section claim.

Self-contained: NumPy + pandas.  Index CSVs in ~/Desktop/indices, named by
ticker with optional chunk numbers (SPX_1.csv, SPX_2.csv, N225_1.csv, ...).
Runtime a few minutes.

WHAT CHANGED FROM lab03
-----------------------
lab03 computed MULTIPLE CORRELATION: an in-sample upper bound on what a linear
predictor could do.  It said the cross-section raises the ceiling by up to 16.5
accuracy points at long delay.  A bound is not a result.  This file fits real
models, walk-forward, and asks how much of that bound survives.

THE FAIRNESS RULE
-----------------
The cross-model sees peers at t; the own-model sees only t-delta.  Their
information sets genuinely differ - that is the point.  But to keep the
comparison about FEATURES rather than about training-set size, both models are
trained on the SAME label set, using the more restrictive cutoff:

    a training pair (s, y_s) is admissible only if  s + HORIZON <= t - delta

so no label is used before it was resolvable by the LATER-informed model's own
data.  This handicaps the cross-model slightly.  That is deliberate: if it wins
anyway, the win is not an artefact of having more training data.

THE MODELS, at each delay
-------------------------
  persist      1{own decision variable at t-delta > 0}.  No fitting.  The
               baseline the first paper showed is the Bayes rule for own-only.
  har-own      ridge logistic on own daily/weekly/monthly log-proxy at t-delta
  cross        ridge logistic on own at t-delta PLUS six live peers at t

Ridge penalty chosen each refit on the last 250 days of the training window,
never on test data.  Features standardised with training-window statistics only.
"""

import os, glob, sys
import numpy as np
import pandas as pd

# Where to look for the CSVs, in order.  You can also pass a folder:
#     python lab04_walkforward.py "C:\\path\\to\\data"
SEARCH_DIRS = [
    os.path.join(os.path.expanduser("~"), "Desktop", "indices"),
    os.path.join(os.path.expanduser("~"), "Desktop", "data"),
    os.path.join(os.path.expanduser("~"), "Desktop"),
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".",
    ".",
]

# Files are matched by KEYWORD, so the original Investing.com download names work
# as-is ("Nikkei 225 Historical Data 3.csv") alongside tidy ones ("N225_1.csv").
ALIASES = {
    "SPX":  ["spx", "s&p 500", "s&p_500", "sp 500", "sp_500", "s&p500", "xyz"],
    "N225": ["n225", "nikkei"],
    "HSI":  ["hsi", "hang seng", "hang_seng"],
    "AXJO": ["axjo", "asx"],
    "NSEI": ["nsei", "nifty"],
    "FTSE": ["ftse"],
    "DAX":  ["dax"],
    "BVSP": ["bvsp", "bovespa"],
}


# A volatility INDEX is not a price index.  "DAX New Volatility" contains the
# substring "dax", and "VDAX_a.csv" would too, so an implied-volatility download
# sitting beside the index exports can be merged into the DAX PRICE series.
# Excluded by name, so the two can share a folder safely.
NOT_AN_INDEX = ["volatility", "vix", "vstoxx", "vdax", "vhsi", "implied"]


def _is_index_file(path):
    import os as _os
    b = _os.path.basename(path).lower()
    return not any(x in b for x in NOT_AN_INDEX)


def files_for(tag, folder):
    hits = []
    for p in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        name = os.path.basename(p).lower()
        if any(k in name for k in ALIASES[tag]) and _is_index_file(p):
            hits.append(p)
    return hits


def find_folder(folder=None):
    """First folder that actually contains recognisable index files."""
    for d in ([folder] if folder else []) + SEARCH_DIRS:
        if not d or not os.path.isdir(d):
            continue
        found = {t: files_for(t, d) for t in ALIASES}
        if sum(1 for t in found if found[t]) >= 2:
            return d
    looked = "\n  ".join(([folder] if folder else []) + SEARCH_DIRS)
    raise SystemExit(
        "Could not find the index CSVs.  Looked in:\n  " + looked +
        "\n\nPut them in one folder and either name that folder 'indices' on your\n"
        "Desktop, or pass the folder as an argument:\n"
        '    python lab04_walkforward.py "C:\\path\\to\\data"')


DATA_DIR = None
CLOSE_UTC = {"N225": 6.0, "AXJO": 6.0, "HSI": 8.0, "NSEI": 10.0,
             "FTSE": 16.5, "DAX": 16.5, "BVSP": 21.0, "SPX": 21.0}
TARGET, WINDOW, HORIZON, MED = "SPX", 5, 5, 252
TRAIN, VAL, REFIT = 1250, 250, 21
DELAYS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]
LAMBDAS = [0.3, 1.0, 3.0, 10.0, 30.0]
N_BOOT, BLOCK, SEED = 2000, 2 * HORIZON, 20260912


# ---------------------------------------------------------------- data
def read_one(path):
    df = pd.read_csv(path)
    df.columns = [str(c).replace('"', "").strip().lower() for c in df.columns]
    num = lambda c: pd.to_numeric(
        df[c].astype(str).str.replace('"', "").str.replace(",", "").str.strip(), errors="coerce")
    px = "price" if "price" in df.columns else "close"
    out = pd.DataFrame({"close": num(px), "open": num("open"),
                        "high": num("high"), "low": num("low")})
    out["date"] = pd.to_datetime(df["date"].astype(str).str.replace('"', "").str.strip(),
                                 errors="coerce", format="mixed")
    return out.dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)


def load_index(tag, folder):
    parts = [read_one(p) for p in files_for(tag, folder)]
    m = parts[0]
    for b in parts[1:]:
        m = (pd.concat([m, b]).drop_duplicates("date", keep="first")
               .sort_values("date").reset_index(drop=True))
    return m


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


# ------------------------------------------------------- ridge logistic
def fit(X, y, lam):
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1]); pen = np.full(X.shape[1], lam); pen[0] = 0.0
    for _ in range(30):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-6)
        H = X.T @ (X * W[:, None]) + np.diag(pen)
        g = X.T @ (y - p) - pen * b
        try: step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError: break
        b += step
        if np.max(np.abs(step)) < 1e-8: break
    return b


def prob(b, X):
    return 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(len(X)), X]) @ b, -30, 30)))


def fit_cv(X, y, lams=LAMBDAS, val=VAL):
    """Pick lambda on the tail of the training window, then refit on all of it."""
    if len(y) <= val + 50: return fit(X, y, lams[len(lams) // 2])
    Xtr, ytr, Xv, yv = X[:-val], y[:-val], X[-val:], y[-val:]
    best, bl = -1, lams[0]
    for lam in lams:
        acc = ((prob(fit(Xtr, ytr, lam), Xv) > 0.5).astype(float) == yv).mean()
        if acc > best: best, bl = acc, lam
    return fit(X, y, bl)


def block_boot(v, rng, n_boot=N_BOOT, block=BLOCK):
    n = len(v); nb = int(np.ceil(n / block)); offs = np.arange(block)
    st = rng.integers(0, n - block + 1, size=(n_boot, nb))
    return np.array([v[(st[i][:, None] + offs).ravel()[:n]].mean() for i in range(n_boot)])


# ---------------------------------------------------------------- main
def main(folder=None):
    rng = np.random.default_rng(SEED)
    folder = find_folder(folder)
    tags = [t for t in CLOSE_UTC if files_for(t, folder)]
    print(f"data folder: {folder}")
    for t in tags:
        print(f"    {t:>5}  " + ", ".join(os.path.basename(x) for x in files_for(t, folder)))
    if TARGET not in tags:
        raise SystemExit(f"\nNo {TARGET} file found in {folder} - cannot run.")
    print()
    proxy = {t: yang_zhang(load_index(t, folder)) for t in tags}
    panel = pd.DataFrame(proxy).sort_index().dropna(how="all")

    peers = [t for t in tags if t != TARGET and CLOSE_UTC[t] < CLOSE_UTC[TARGET]]
    print(f"panel {len(panel)} dates {panel.index.min().date()} to {panel.index.max().date()}")
    print(f"target {TARGET}; live peers {peers}\n")

    # decision variables, and own HAR features
    dv = {t: np.log(panel[t] / panel[t].rolling(MED, min_periods=30).median()) for t in tags}
    D = pd.DataFrame(dv)
    D[peers] = D[peers].ffill(limit=5)
    # Work on the TARGET's OWN TRADING CALENDAR.  The panel is a union of eight
    # market calendars, so on a US holiday the target row is missing; a 22-day
    # rolling mean then spans at least one gap almost every day and comes out NaN,
    # which silently starved the fitted models of training rows.  Peers are already
    # forward-filled onto this calendar, which is what a US forecaster actually sees.
    D = D.loc[D[TARGET].notna()].copy()
    a = D[TARGET].values
    # Own features must be CENTRED the same way the baseline is, i.e. the decision
    # variable log(proxy / trailing median), not the raw level.  Handing the fitted
    # models an uncentred level forces them to relearn a drifting threshold and they
    # collapse to chance; that was a real bug, caught by comparing to the baseline.
    sa = pd.Series(a)
    har = np.column_stack([sa.values, sa.rolling(5).mean().values, sa.rolling(22).mean().values])
    Peer = D[peers].values

    n = len(D)
    y = np.full(n, np.nan)
    y[:-HORIZON] = (a[HORIZON:] > 0).astype(float)
    y[np.isnan(a[:n - HORIZON]).nonzero()[0]] = np.nan

    start = MED + TRAIN + VAL + max(DELAYS) + HORIZON
    idx = np.arange(start, n - HORIZON)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])]
    print(f"test window {len(idx)} days, {D.index[idx[0]].date()} to {D.index[idx[-1]].date()};"
          f" base rate {y[idx].mean():.3f}\n")

    def run(delta, kind):
        preds = np.empty(len(idx)); b = None; mu = sd = None
        for j, t in enumerate(idx):
            if j % REFIT == 0:
                cut = t - delta - HORIZON                      # common fairness cutoff
                tr = np.arange(max(0, cut - TRAIN - VAL), cut)
                if kind == "har":
                    Xtr, xt = har[tr - delta], har[t - delta]
                else:
                    Xtr = np.column_stack([har[tr - delta], Peer[tr]])
                    xt = np.concatenate([har[t - delta], Peer[t]])
                ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(y[tr])
                Xtr, ytr = Xtr[ok], y[tr][ok]
                if len(ytr) < 200: b = None; continue
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
                b = fit_cv((Xtr - mu) / sd, ytr)
            xt = (har[t - delta] if kind == "har"
                  else np.concatenate([har[t - delta], Peer[t]]))
            if b is None or not np.isfinite(xt).all():
                preds[j] = 1.0
            else:
                preds[j] = float(prob(b, ((xt - mu) / sd)[None, :])[0] > 0.5)
        return preds

    yb = y[idx]
    print(f"{'delta':>6}{'persist':>20}{'har-own':>20}{'cross':>20}{'cross - persist':>22}")
    for d in DELAYS:
        p0 = (a[idx - d] > 0).astype(float)
        c0 = (p0 == yb).astype(float)
        out = [c0]
        for kind in ("har", "cross"):
            out.append((run(d, kind) == yb).astype(float))
        row = f"{d:>6}"
        for c in out:
            bs = block_boot(c, rng); lo, hi = np.percentile(bs, [2.5, 97.5])
            row += f"{c.mean()*100:>8.2f} [{lo*100:4.1f},{hi*100:4.1f}]"
        diff = out[2] - out[0]
        bs = block_boot(diff, rng); lo, hi = np.percentile(bs, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        row += f"{diff.mean()*100:>+11.2f} [{lo*100:+5.1f},{hi*100:+5.1f}]{star}"
        print(row)
    print("\n* = 95% block-bootstrap interval on the paired difference excludes zero.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
