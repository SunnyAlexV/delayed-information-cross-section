"""
lab05_robustness.py - the three checks the cross-market claim has not survived yet.

Self-contained: NumPy + pandas.  Same CSVs as lab04; same discovery rules.
Runtime ~10 minutes.

WHY THIS EXISTS
---------------
lab04 showed a large out-of-sample gain from live foreign closes, for ONE target
market, on ONE binary target, against a baseline that ignores information it
could have used.  Three objections follow, and each is answered here.

  A. WRONG BENCHMARK.  "+13 points over a forecaster who ignores available
     data" is close to tautological.  The sharper question is how close the
     cross-market forecast gets to one with FULLY CURRENT domestic data.

  B. ONE MARKET.  If the mechanism is real it must work with other targets.
     This needs a general timing rule, not the US-specific one lab04 used:
     a foreign close is usable if its TIMESTAMP precedes the target's close.
     Tokyo closes at 06:00 UTC, so for a Tokyo target the New York close of
     the PREVIOUS day (21:00 UTC on t-1) is still the freshest available.
     Implemented below as an explicit per-peer day offset.

  C. THE TARGET THE COMPANION PAPER ARGUED AGAINST.  A median split discards
     magnitude.  Part C repeats the test on CONTINUOUS log realised variance
     with out-of-sample R-squared and QLIKE, the loss family that is robust to
     a noisy volatility proxy.
"""

import os, glob, sys
import numpy as np
import pandas as pd

SEARCH_DIRS = [
    os.path.join(os.path.expanduser("~"), "Desktop", "indices"),
    os.path.join(os.path.expanduser("~"), "Desktop", "data"),
    os.path.join(os.path.expanduser("~"), "Desktop"),
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".",
    ".",
]
ALIASES = {
    "SPX":  ["spx", "s&p 500", "s&p_500", "sp 500", "sp_500", "s&p500", "xyz"],
    "N225": ["n225", "nikkei"], "HSI": ["hsi", "hang seng", "hang_seng"],
    "AXJO": ["axjo", "asx"],    "NSEI": ["nsei", "nifty"],
    "FTSE": ["ftse"], "DAX": ["dax"], "BVSP": ["bvsp", "bovespa"],
}
CLOSE_UTC = {"N225": 6.0, "AXJO": 6.0, "HSI": 8.0, "NSEI": 10.0,
             "FTSE": 16.5, "DAX": 16.5, "BVSP": 21.0, "SPX": 21.0}
TARGETS = ["SPX", "FTSE", "DAX", "N225", "HSI"]
WINDOW, HORIZON, MED = 5, 5, 252
TRAIN, VAL, REFIT = 1250, 250, 21
DELAYS_MULTI = [0, 5, 13, 21, 55]
DELAYS_FULL = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]
LAMBDAS = [0.3, 1.0, 3.0, 10.0, 30.0]
N_BOOT, BLOCK, SEED = 1500, 2 * HORIZON, 20260912


# Files that must NEVER be matched to an index tag, however their names read.
# "DAX New Volatility" contains the substring "dax", so a VDAX download sitting
# in the same folder as the index exports would be merged into the DAX PRICE
# series.  It happens to be harmless with the current files, because every VDAX
# date already exists in the DAX index and the merge keeps the first occurrence
# - but a single VDAX-only date would enter as a price near 20 instead of near
# 20000, and nothing would say so.  Excluded explicitly rather than left to luck.
NOT_AN_INDEX = ["volatility", "vix", "vstoxx", "vdax", "vhsi", "implied"]


def files_for(tag, folder):
    return [p for p in sorted(glob.glob(os.path.join(folder, "*.csv")))
            if any(k in os.path.basename(p).lower() for k in ALIASES[tag])
            and not any(x in os.path.basename(p).lower() for x in NOT_AN_INDEX)]


def find_folder(folder=None):
    for d in ([folder] if folder else []) + SEARCH_DIRS:
        if d and os.path.isdir(d) and sum(1 for t in ALIASES if files_for(t, d)) >= 2:
            return d
    raise SystemExit("Could not find the index CSVs. Pass the folder as an argument.")


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


# --------------------------------------------------------- estimators
def fit_logit(X, y, lam):
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1]); pen = np.full(X.shape[1], lam); pen[0] = 0.0
    for _ in range(30):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-6)
        try: step = np.linalg.solve(X.T @ (X * W[:, None]) + np.diag(pen), X.T @ (y - p) - pen * b)
        except np.linalg.LinAlgError: break
        b += step
        if np.max(np.abs(step)) < 1e-8: break
    return b


def p_logit(b, X):
    return 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(len(X)), X]) @ b, -30, 30)))


def fit_ridge(X, y, lam):
    X = np.column_stack([np.ones(len(X)), X])
    pen = np.full(X.shape[1], lam); pen[0] = 0.0
    return np.linalg.solve(X.T @ X + np.diag(pen), X.T @ y)


def p_ridge(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


def cv(X, y, kind):
    """Choose lambda on the tail of the training window."""
    f, p = (fit_logit, p_logit) if kind == "bin" else (fit_ridge, p_ridge)
    if len(y) <= VAL + 50: return f(X, y, LAMBDAS[2])
    Xtr, ytr, Xv, yv = X[:-VAL], y[:-VAL], X[-VAL:], y[-VAL:]
    best, bl = -np.inf, LAMBDAS[0]
    for lam in LAMBDAS:
        pred = p(f(Xtr, ytr, lam), Xv)
        sc = ((pred > 0.5).astype(float) == yv).mean() if kind == "bin" else -((pred - yv) ** 2).mean()
        if sc > best: best, bl = sc, lam
    return f(X, y, bl)


def block_boot(v, rng, n_boot=N_BOOT, block=BLOCK):
    n = len(v); nb = int(np.ceil(n / block)); offs = np.arange(block)
    st = rng.integers(0, n - block + 1, size=(n_boot, nb))
    return np.array([v[(st[i][:, None] + offs).ravel()[:n]].mean() for i in range(n_boot)])


def ci(v, rng):
    bs = block_boot(v, rng); return np.percentile(bs, [2.5, 97.5])


# --------------------------------------------------------- experiment
def build(folder, target):
    """Panel on the TARGET's calendar, with each peer shifted to its freshest
    observation strictly earlier than the target's close."""
    tags = [t for t in CLOSE_UTC if files_for(t, folder)]
    proxy = {t: yang_zhang(load_index(t, folder)) for t in tags}
    P = pd.DataFrame(proxy).sort_index()
    D = pd.DataFrame({t: np.log(P[t] / P[t].rolling(MED, min_periods=30).median()) for t in tags})
    peers = [t for t in tags if t != target]
    D[peers] = D[peers].ffill(limit=5)
    D = D.loc[D[target].notna()].copy()
    # peers closing at or after the target use YESTERDAY's close
    lag = {p: (0 if CLOSE_UTC[p] < CLOSE_UTC[target] else 1) for p in peers}
    for p in peers:
        if lag[p]:
            D[p] = D[p].shift(1)
    return D, peers, lag


def run_target(folder, target, delays, rng, continuous=False):
    D, peers, lag = build(folder, target)
    a = D[target].values
    sa = pd.Series(a)
    own = np.column_stack([sa.values, sa.rolling(5).mean().values, sa.rolling(22).mean().values])
    Peer = D[peers].values
    n = len(D)

    if continuous:
        y = np.full(n, np.nan); y[:-HORIZON] = a[HORIZON:]           # continuous target
    else:
        y = np.full(n, np.nan); y[:-HORIZON] = (a[HORIZON:] > 0).astype(float)
    kind = "cont" if continuous else "bin"

    start = MED + TRAIN + VAL + max(delays) + HORIZON
    idx = np.arange(start, n - HORIZON)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])]
    yb = y[idx]

    def fitrun(delta, use_peers):
        f, pr = (fit_logit, p_logit) if kind == "bin" else (fit_ridge, p_ridge)
        out = np.empty(len(idx)); b = None; mu = sd = None
        for j, t in enumerate(idx):
            if j % REFIT == 0:
                cut = t - delta - HORIZON
                tr = np.arange(max(0, cut - TRAIN - VAL), cut)
                Xtr = (np.column_stack([own[tr - delta], Peer[tr]]) if use_peers
                       else own[tr - delta])
                ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(y[tr])
                Xtr, ytr = Xtr[ok], y[tr][ok]
                if len(ytr) < 200: b = None; continue
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
                b = cv((Xtr - mu) / sd, ytr, kind)
            xt = (np.concatenate([own[t - delta], Peer[t]]) if use_peers else own[t - delta])
            if b is None or not np.isfinite(xt).all():
                out[j] = 0.5 if kind == "bin" else 0.0
            else:
                v = pr(b, ((xt - mu) / sd)[None, :])[0]
                out[j] = float(v > 0.5) if kind == "bin" else v
        return out

    return D, peers, lag, idx, yb, fitrun, a


def main(folder=None):
    rng = np.random.default_rng(SEED)
    folder = find_folder(folder)
    print(f"data folder: {folder}\n")

    # ---------------- A + B : binary target, five markets -----------------
    print("=" * 78)
    print("PARTS A and B - binary target, five target markets, general timing rule")
    print("=" * 78)
    print("A peer close counts only if its timestamp precedes the target's close;")
    print("markets closing at or after the target contribute the PREVIOUS day.\n")
    summary = {}
    for tgt in TARGETS:
        try:
            D, peers, lag, idx, yb, fitrun, a = run_target(folder, tgt, DELAYS_MULTI, rng)
        except Exception as e:
            print(f"{tgt}: skipped ({e})"); continue
        same = [p for p in peers if lag[p] == 0]; prev = [p for p in peers if lag[p] == 1]
        print(f"--- target {tgt} ({CLOSE_UTC[tgt]:.1f} UTC), {len(idx)} test days, "
              f"base rate {yb.mean():.3f}")
        print(f"    same-day peers {same}; previous-day peers {prev}")
        print(f"    {'delta':>6}{'persist':>10}{'cross':>10}{'gain':>22}{'vs current-own':>16}")
        cur = None
        for d in DELAYS_MULTI:
            p0 = ((a[idx - d] > 0).astype(float) == yb).astype(float)
            pc = (fitrun(d, True) == yb).astype(float)
            if d == 0: cur = p0.mean() * 100
            diff = pc - p0; lo, hi = ci(diff, rng)
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {d:>6}{p0.mean()*100:>10.2f}{pc.mean()*100:>10.2f}"
                  f"{diff.mean()*100:>+11.2f} [{lo*100:+5.1f},{hi*100:+5.1f}]{star}"
                  f"{pc.mean()*100-cur:>+16.2f}")
            summary[(tgt, d)] = (p0.mean() * 100, pc.mean() * 100, cur)
        print()

    # ---------------- C : continuous target, R2 and QLIKE ------------------
    print("=" * 78)
    print("PART C - continuous target: log realised variance, h=5, SPX")
    print("=" * 78)
    print("The companion paper argued a median split discards magnitude. This drops")
    print("the split entirely: out-of-sample R-squared vs the training mean, and")
    print("QLIKE, which is robust to a noisy volatility proxy.\n")
    D, peers, lag, idx, yb, fitrun, a = run_target(folder, "SPX", DELAYS_FULL, rng, continuous=True)
    ybar = None
    print(f"{'delta':>6}{'R2 own':>10}{'R2 cross':>11}{'dR2':>20}"
          f"{'QLIKE own':>12}{'QLIKE cross':>13}")
    for d in DELAYS_FULL:
        po, pc = fitrun(d, False), fitrun(d, True)
        base = yb.mean()
        sst = ((yb - base) ** 2).sum()
        r2o = 1 - ((yb - po) ** 2).sum() / sst
        r2c = 1 - ((yb - pc) ** 2).sum() / sst
        # QLIKE on the variance scale: exp() the log decision variable
        yv, fo, fc = np.exp(yb), np.exp(po), np.exp(pc)
        ql = lambda f: np.mean(yv / f - np.log(yv / f) - 1)
        de = ((yb - po) ** 2 - (yb - pc) ** 2) / sst * len(yb)
        lo, hi = ci(de, rng)
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"{d:>6}{r2o:>10.4f}{r2c:>11.4f}{r2c-r2o:>+11.4f} [{lo:+.3f},{hi:+.3f}]{star}"
              f"{ql(fo):>12.4f}{ql(fc):>13.4f}")
    print("\n* = 95% block-bootstrap interval on the paired difference excludes zero.")
    print("QLIKE is a loss: lower is better.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
