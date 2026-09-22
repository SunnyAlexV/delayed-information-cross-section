"""
lab02_delay_curve.py  -  the honest version of the delay experiment.
Self-contained: NumPy + pandas only (both in Anaconda).  ~2 minutes.

    python lab02_delay_curve.py            # expects XYZ.csv on your Desktop

WHAT THIS REPLACES
------------------
np.py measured one thing - a median-threshold rule - and the write-up around
it claimed three more.  This file computes every column that a paper would
need, so nothing has to be asserted:

  * Yang-Zhang variance, actually implemented from Open/High/Low/Close
  * a real HAR-RV classifier, fitted walk-forward with no look-ahead
  * the two baselines a referee reaches for first (majority class, persistence)
  * confidence intervals that respect the overlap in the target
  * a measurement of the look-ahead leak in the original threshold

THE TARGET (defined once, used by everything)
---------------------------------------------
    M_t = median of the volatility proxy over the trailing 252 days, up to t
    y_t = 1 if proxy_{t+h} > M_t, else 0                      h = 5

M_t uses only data up to t, so the label is honest.  A forecaster who is
delta days late does NOT know M_t - they know M_{t-delta}.  The original
script used a threshold computed up to t while claiming to be delta days
behind, which is a small look-ahead leak.  STEP 5 measures how big it is.

THE FORECASTERS, all seeing only data up to t-delta
---------------------------------------------------
  majority     the constant that won most often in training
  persist-CC   1 if close-to-close RV at t-delta is above ITS median then
  persist-YZ   the same rule on the Yang-Zhang proxy
  har-CC       logistic regression on log RV at daily / weekly / monthly
               horizons - the HAR structure, as a classifier
  har-YZ       the same, on Yang-Zhang

YANG-ZHANG, over a window of n days
-----------------------------------
    o_i = ln(O_i / C_{i-1})     overnight jump
    u_i = ln(H_i / O_i),  d_i = ln(L_i / O_i),  c_i = ln(C_i / O_i)
    RS_i = u_i (u_i - c_i) + d_i (d_i - c_i)            Rogers-Satchell
    sig2_o  = sample variance of o over the window
    sig2_c  = sample variance of c over the window
    sig2_RS = mean of RS over the window
    k = 0.34 / (1.34 + (n+1)/(n-1))
    sig2_YZ = sig2_o + k sig2_c + (1-k) sig2_RS
"""

import os, sys
import numpy as np
import pandas as pd

FILENAME   = "XYZ.csv"
WINDOW     = 5        # days in the volatility proxy
HORIZON    = 5        # h: predict the proxy h days ahead
MED_LOOK   = 252      # trailing window for the median threshold
TRAIN      = 1250     # walk-forward training window
REFIT      = 21       # refit the logistic every REFIT days (speed)
DELAYS     = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]
N_BOOT     = 2000
BLOCK      = 8 * HORIZON   # long enough to outlast the dependence lab58 measures
SEED       = 20260912


# ----------------------------------------------------------------------
# STEP 0 - load and validate
# ----------------------------------------------------------------------
def load(filename=FILENAME):
    # Look in the repo's data folder first, then the places an ad-hoc run would
    # keep it.  Same file either way, so the numbers do not depend on which
    # branch hits.
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    candidates = [
        filename if os.path.isabs(filename) else None,
        os.path.join(here, "..", "data", "single_market", filename),
        os.path.join(here, "..", "data", filename),
        os.path.join(os.path.expanduser("~"), "Desktop", filename),
        os.path.join(here, filename),
        filename,
    ]
    path = next((c for c in candidates if c and os.path.exists(c)), None)
    if path is None:
        raise SystemExit(
            f"Could not find {filename}.  Looked in:\n  "
            + "\n  ".join(c for c in candidates if c)
            + "\nPass the full path as the first argument to run it from elsewhere.")
    df = pd.read_csv(path)
    df.columns = [str(c).replace('"', "").strip().lower() for c in df.columns]

    def num(col):
        return pd.to_numeric(
            df[col].astype(str).str.replace('"', "").str.replace(",", "").str.strip(),
            errors="coerce")

    close = num("price" if "price" in df.columns else "close")
    out = pd.DataFrame({"close": close, "open": num("open"),
                        "high": num("high"), "low": num("low")})
    out["date"] = pd.to_datetime(df["date"].astype(str).str.replace('"', "").str.strip(),
                                 errors="coerce", format="mixed")
    out = out.dropna().reset_index(drop=True)
    if out["date"].iloc[0] > out["date"].iloc[-1]:
        out = out.iloc[::-1].reset_index(drop=True)       # ascending

    # invariants - each one has caught a real bug at some point
    assert out["date"].is_monotonic_increasing, "dates not ascending after sort"
    assert (out["high"] >= out["low"]).all(), "high < low on some row"
    assert (out["high"] >= out[["open", "close"]].max(axis=1) - 1e-9).all(), "high below open/close"
    assert (out["low"] <= out[["open", "close"]].min(axis=1) + 1e-9).all(), "low above open/close"
    assert (out["close"] > 0).all(), "non-positive price"
    return out


# ----------------------------------------------------------------------
# STEP 1 - the two volatility proxies
# ----------------------------------------------------------------------
def proxy_close_to_close(close, n=WINDOW):
    r = np.log(close[1:] / close[:-1])
    return pd.Series(r ** 2).rolling(n).mean().values, 1     # offset: index 0 = day 1


def proxy_yang_zhang(o, h, l, c, n=WINDOW):
    oo = np.log(o[1:] / c[:-1])          # overnight
    uu = np.log(h[1:] / o[1:])
    dd = np.log(l[1:] / o[1:])
    cc = np.log(c[1:] / o[1:])
    rs = uu * (uu - cc) + dd * (dd - cc)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    s2o = pd.Series(oo).rolling(n).var(ddof=1).values
    s2c = pd.Series(cc).rolling(n).var(ddof=1).values
    s2rs = pd.Series(rs).rolling(n).mean().values
    yz = s2o + k * s2c + (1 - k) * s2rs
    return np.maximum(yz, 1e-12), 1


# ----------------------------------------------------------------------
# STEP 2 - target and features
# ----------------------------------------------------------------------
def trailing_median(v, look=MED_LOOK):
    """M[t] = median of v over [t-look+1, t], using only data up to t."""
    return pd.Series(v).rolling(look, min_periods=30).median().values


def make_target(v, M, h=HORIZON):
    n = len(v) - h
    y = np.full(n, np.nan)
    ok = ~np.isnan(M[:n]) & ~np.isnan(v[h:h + n])
    y[ok] = (v[h:h + n][ok] > M[:n][ok]).astype(float)
    return y


def har_features(logv):
    """Daily / weekly / monthly averages of log proxy, as of each index."""
    s = pd.Series(logv)
    return np.column_stack([s.values,
                            s.rolling(5).mean().values,
                            s.rolling(22).mean().values])


# ----------------------------------------------------------------------
# STEP 3 - logistic regression by IRLS, written out (no sklearn needed)
# ----------------------------------------------------------------------
def logistic_fit(X, y, ridge=1e-3, iters=25):
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-6)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        g = X.T @ (y - p) - ridge * b
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def logistic_predict(b, X):
    X = np.column_stack([np.ones(len(X)), X])
    return 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))


# ----------------------------------------------------------------------
# STEP 4 - walk-forward evaluation
# ----------------------------------------------------------------------
def evaluate(y, feats, Ms, proxies, delays=DELAYS):
    """
    feats[name] -> (T,3) HAR design; proxies[name] -> level series; Ms[name] -> thresholds.
    Returns dict: (model, delta) -> (predictions, labels) over the common test index.
    """
    n = len(y)
    start = TRAIN + max(delays) + HORIZON + MED_LOOK
    idx = np.arange(start, n)
    idx = idx[~np.isnan(y[idx])]
    res = {}

    ybar = y[idx]
    maj = 1.0 if np.nanmean(y[:start]) > 0.5 else 0.0
    for d in delays:
        res[("majority", d)] = np.full(len(idx), maj)

        for tag in proxies:
            v, M = proxies[tag], Ms[tag]
            # persistence: is the proxy above ITS OWN threshold, both as of t-delta
            p = (v[idx - d] > M[idx - d]).astype(float)
            res[(f"persist-{tag}", d)] = p

            # HAR logistic, refit every REFIT days on labels resolved by t-delta
            X = feats[tag]
            preds = np.empty(len(idx))
            b = None
            for j, t in enumerate(idx):
                if j % REFIT == 0 or b is None:
                    last = t - d - HORIZON          # last label known at t-delta
                    tr = np.arange(max(0, last - TRAIN), last)
                    tr = tr[~np.isnan(y[tr]) & ~np.isnan(X[tr]).any(axis=1)]
                    b = logistic_fit(X[tr], y[tr]) if len(tr) > 50 else None
                if b is None or np.isnan(X[t - d]).any():
                    preds[j] = maj
                else:
                    preds[j] = float(logistic_predict(b, X[t - d][None, :])[0] > 0.5)
            res[(f"har-{tag}", d)] = preds
    return res, ybar, idx


# ----------------------------------------------------------------------
# STEP 5 - inference that respects the overlap
# ----------------------------------------------------------------------
def block_boot(correct, rng, n_boot=N_BOOT, block=BLOCK):
    """Moving-block bootstrap on the 0/1 correctness series."""
    n = len(correct)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    out = np.empty(n_boot)
    offs = np.arange(block)
    for i in range(n_boot):
        s = (starts[i][:, None] + offs).ravel()[:n]
        out[i] = correct[s].mean()
    return out


def main(path=None):
    rng = np.random.default_rng(SEED)
    df = load(path or FILENAME)
    c = df["close"].values
    o, h, l = df["open"].values, df["high"].values, df["low"].values
    print(f"loaded {len(df)} rows, {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")

    vcc, _ = proxy_close_to_close(c)
    vyz, _ = proxy_yang_zhang(o, h, l, c)
    n = min(len(vcc), len(vyz))
    vcc, vyz = vcc[:n], vyz[:n]

    proxies = {"CC": vcc, "YZ": vyz}
    Ms = {t: trailing_median(v) for t, v in proxies.items()}
    feats = {t: har_features(np.log(np.maximum(v, 1e-14))) for t, v in proxies.items()}
    y = make_target(vcc, Ms["CC"])                     # ONE target, the CC one, for all models
    for t in proxies:
        proxies[t] = proxies[t][:len(y)]
        Ms[t] = Ms[t][:len(y)]
        feats[t] = feats[t][:len(y)]

    res, ybar, idx = evaluate(y, feats, Ms, proxies)
    print(f"test window: {len(idx)} days, {df['date'].iloc[idx[0]].date()} to "
          f"{df['date'].iloc[idx[-1]].date()};  base rate {ybar.mean():.3f}")
    print(f"overlapping {HORIZON}-day targets -> block bootstrap with block {BLOCK}\n")

    models = ["majority", "persist-CC", "persist-YZ", "har-CC", "har-YZ"]
    print(f"{'delta':>6} " + "".join(f"{m:>21}" for m in models))
    print(f"{'':>6} " + "".join(f"{'acc  [95% CI]':>21}" for _ in models))
    table = {}
    for d in DELAYS:
        row = f"{d:>6} "
        for m in models:
            corr = (res[(m, d)] == ybar).astype(float)
            bs = block_boot(corr, rng)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            table[(m, d)] = (corr.mean(), lo, hi, corr)
            row += f"{corr.mean()*100:>9.2f} [{lo*100:4.1f},{hi*100:4.1f}]"
        print(row)

    print("\nPAIRED TEST vs the persistence baseline on the same proxy")
    print("(difference in accuracy, block-bootstrap 95% CI; excludes 0 = real gain)")
    print(f"{'delta':>6}{'har-CC - persist-CC':>26}{'har-YZ - persist-YZ':>26}"
          f"{'persist-YZ - persist-CC':>28}")
    for d in DELAYS:
        row = f"{d:>6}"
        for a, b in [("har-CC", "persist-CC"), ("har-YZ", "persist-YZ"),
                     ("persist-YZ", "persist-CC")]:
            diff = table[(a, d)][3] - table[(b, d)][3]
            bs = block_boot(diff, rng)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            star = "*" if (lo > 0 or hi < 0) else " "
            row += f"{diff.mean()*100:>+16.2f} [{lo*100:+5.1f},{hi*100:+5.1f}]{star}"
        print(row)

    # --- the leak: threshold as of t (what np.py used) vs as of t-delta -----
    print("\nLOOK-AHEAD LEAK in the original threshold")
    print("np.py compared the delayed proxy against a threshold computed up to t,")
    print("which a forecaster delta days late does not have.  Size of that leak:")
    print(f"{'delta':>6}{'honest (M at t-d)':>20}{'leaky (M at t)':>17}{'leak':>9}")
    for d in DELAYS:
        v, M = proxies["CC"], Ms["CC"]
        honest = ((v[idx - d] > M[idx - d]).astype(float) == ybar).mean()
        leaky = ((v[idx - d] > M[idx]).astype(float) == ybar).mean()
        print(f"{d:>6}{honest*100:>19.2f}%{leaky*100:>16.2f}%{(leaky-honest)*100:>+9.2f}")

    print("\nDone.  Every number above came from this file.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
