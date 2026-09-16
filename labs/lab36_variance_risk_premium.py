"""
lab36_variance_risk_premium.py - EXPLORATORY.  Cited by neither paper.

Imports lab05_robustness, lab08_implied_vol and lab22_factor_benchmark; keep all
four in labs/.  Runtime about one minute.

WHAT SECTION 6 ALREADY CONCEDED
--------------------------------
Section 6 raced the foreign cross-section against implied volatility and LOST.
Given VIX and VDAX-NEW, the seven foreign closes add nothing: the incremental
R-squared is negative at all ten delays, and at the four where lab08 decomposes
it, the gross content sits in an interval straddling zero.  The paper
reports that as redundancy rather than harm, and changes its RECOMMENDATION
accordingly.  So this file is not an attempt to rescue the foreign block from
the options market.  That argument is over and the paper lost it.

WHAT IT DID NOT TEST
--------------------
Section 6 raced the LEVEL of implied volatility.  The literature's forecasting
variable is not the level; it is the VARIANCE RISK PREMIUM,

    VRP(t) = IV(t)^2 - RV(t)

what the options market charges for variance minus what variance delivered.
Bollerslev, Tauchen and Zhou (2009) use it to predict returns, and a long line
since uses it to predict variance.  It is positive most days, because sellers of
variance are paid for bearing it.  If it carries something the level does not,
Section 6 raced a weaker horse than it thought - and if it does not, that is a
defence of Section 6 nobody has yet written down.

WHY THE PREMIUM IS THE RIGHT VARIABLE FOR *THIS* PROJECT
---------------------------------------------------------
Because it is a difference between two legs that are not equally observable.

    IV(t)   the implied leg.  A live quote.  Current, always.
    RV(t)   the realised leg.  Computed from the price history - which, for the
            domestic market, is exactly what this project holds stale.

A desk whose domestic history is delta days old cannot form the domestic
premium.  What it can form is IV_us(t)^2 - RV_us(t - delta), which IS the
premium at delta = 0 and is something else by delta = 55: part premium, part
whatever realised variance has done in the meantime.  The delay does not age
this predictor.  It changes what the predictor measures.

Germany's premium has no such problem.  Frankfurt's realised variance and
VDAX-NEW are both observable at the New York close on the same day, so the
German premium is computable in real time however stale the domestic history
is.  That is this project's hypothesis restated on a variable the papers never
touched, and it is the reason to run this.

THE DESIGN TRICK THAT MAKES IT READABLE
----------------------------------------
lab07 established that regressors cost R-squared whether or not they inform, so
a negative incremental column cannot be read as harm without subtracting that
cost - and the subtraction leaves an interval wide enough to be unhelpful.

This file avoids the problem instead of paying it.  Every comparison it turns on
is between arms with the SAME NUMBER OF REGRESSORS:

    +IV+VRPus    one extra column   the premium a delayed desk can form
    +IV+ORACLE   one extra column   the same premium with TODAY's realised leg,
                                    inadmissible at every positive delay
    +IV+VRPde    one extra column   Germany's premium, always admissible
    +IV+NOISE    one extra column   an AR(1) surrogate matched on persistence and
                                    variance, carrying nothing by construction

The estimation cost is identical across those four and cancels exactly in any
difference between them.  NOISE supplies the absolute zero; ORACLE minus VRPus
is the damage the delay does to the premium, measured without a surrogate
subtraction; VRPde minus VRPus is whether the premium you do not own stands in
for the one you cannot compute.

A built-in check: at delta = 0 the stale leg is today's leg, so VRPus and ORACLE
must agree to the last digit.  The run prints that check rather than assuming it.
"""

import io, contextlib, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab08_implied_vol as IV
import lab22_factor_benchmark as F

SEED = 20260915
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 3, 5, 13, 21, 55]
RV_WIN = 22            # matched to VIX's thirty CALENDAR days
RV_ALT = 5             # the project's own window, run as the sensitivity
N_DRAW = 6             # surrogate draws for the NOISE arm
N_BOOT, BLK = 2000, 10
VIX_LAG_STRICT = IV.VIX_LAG_STRICT

ARMS = {
    "OWN":        ["own"],
    "+FOR":       ["own", "for"],
    "+IV":        ["own", "iv"],
    "+IV+VRPus":  ["own", "iv", "vus"],
    "+IV+VRPde":  ["own", "iv", "vde"],
    "+IV+VRP":    ["own", "iv", "vus", "vde"],
    "ALL":        ["own", "for", "iv", "vus", "vde"],
    "+IV+ORACLE": ["own", "iv", "vus0"],
    "+IV+NOISE":  ["own", "iv", "noise"],
    "+IV+LEGS":   ["own", "iv", "legs"],
    "+IV+VRP+N":  ["own", "iv", "vus", "noise"],
    "+IV+WINus":  ["own", "iv", "winu"],
    "+IV+WINde":  ["own", "iv", "wind"],
}
WIN_Q, WIN_LOOK = 0.01, 1250       # causal winsorising of the premium columns


# ------------------------------------------------------------------ data
def _quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **k)


def panel(folder, rv_win=RV_WIN, quiet=False):
    """Index panel, plus the raw variance legs both premia are built from.

    Everything the panel already carries lives in log-ratio-to-median space.
    The premium cannot: a difference of variances is signed, so it has no log.
    It is divided instead by the trailing median of the IMPLIED variance, which
    leaves it dimensionless and comparable across a quiet decade and a crisis
    without pretending a negative premium is impossible.
    """
    D, peers, lag = L.build(folder, TARGET)
    if not quiet:
        print("  implied-volatility files and seams:")

    rv, iv = {}, {}
    for mkt, tag in (("SPX", "VIX"), ("DAX", "VDAX")):
        s = L.yang_zhang(L.load_index(mkt, folder), n=rv_win) * 252.0
        rv[mkt] = s.reindex(D.index).ffill(limit=5)
        q = _quiet(IV.load_iv, folder, tag) if quiet else IV.load_iv(folder, tag)
        q = q.reindex(D.index).ffill(limit=5)
        if tag == "VIX":
            q = q.shift(VIX_LAG_STRICT)      # prints 21:15 UTC, after the close
        iv[mkt] = (q / 100.0) ** 2

    keep = D[TARGET].notna()
    for m in ("SPX", "DAX"):
        keep &= iv[m].notna() & rv[m].notna()
    D = D.loc[keep]
    a = D[TARGET].values
    sa = pd.Series(a)
    n = len(D)
    y = np.full(n, np.nan); y[:-H] = a[H:]
    start = L.MED + L.TRAIN + L.VAL + max(DELAYS) + H
    idx = np.arange(start, n - H)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])]

    S = {
        "own": np.column_stack([sa.values, sa.rolling(5).mean().values,
                                sa.rolling(22).mean().values]),
        "P": D[peers].values,
    }
    for m, k in (("SPX", "u"), ("DAX", "d")):
        ivv = iv[m].loc[D.index]
        med = ivv.rolling(L.MED, min_periods=30).median()
        S["iv" + k] = ivv.values
        S["rv" + k] = rv[m].loc[D.index].values
        S["m" + k] = med.values
        S["liv" + k] = np.log(ivv.values / med.values)
    S["liv"] = np.column_stack([S["livu"], S["livd"]])
    return S, y, idx, D, len(peers)


# ------------------------------------------------------------- estimation
def design(rows, d, arms, S):
    cols = []
    for arm in arms:
        if arm == "own":
            cols.append(S["own"][rows - d])
        elif arm == "for":
            cols.append(S["P"][rows])
        elif arm == "iv":
            cols.append(S["liv"][rows])
        elif arm == "vus":                   # what a delayed desk can form
            cols.append(((S["ivu"][rows] - S["rvu"][rows - d]) / S["mu"][rows])[:, None])
        elif arm == "vus0":                  # inadmissible: today's realised leg
            cols.append(((S["ivu"][rows] - S["rvu"][rows]) / S["mu"][rows])[:, None])
        elif arm == "vde":                   # fully observable, always
            cols.append(((S["ivd"][rows] - S["rvd"][rows]) / S["md"][rows])[:, None])
        elif arm == "noise":                 # AR(1) surrogate, no information
            cols.append(S["noise"][rows][:, None])
        elif arm in ("winu", "wind"):        # the same premium, tails clipped
            cols.append(S[arm][rows][:, None])
        elif arm == "legs":                  # the difference restriction, freed
            cols.append(np.column_stack([S["ivu"][rows] / S["mu"][rows],
                                         S["rvu"][rows - d] / S["mu"][rows]]))
        else:
            raise KeyError(arm)
    return np.column_stack(cols)


def walk(S, y, idx, d, arms):
    out = np.empty(len(idx)); b = mu = sd = None
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - d - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), max(cut, 1))
            X = design(tr, d, arms, S)
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y[tr])
            X, yy = X[ok], y[tr][ok]
            if len(yy) < 200:
                b = None; continue
            mu, sd = X.mean(0), X.std(0) + 1e-9
            b = L.cv((X - mu) / sd, yy, "cont")
        xt = design(np.array([t]), d, arms, S)[0]
        out[j] = 0.0 if (b is None or not np.isfinite(xt).all()) \
            else L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
    return out


def premium(S, d, oracle=False):
    lagged = np.arange(len(S["ivu"])) - (0 if oracle else d)
    v = (S["ivu"] - S["rvu"][np.clip(lagged, 0, None)]) / S["mu"]
    v[:max(d, 0) if not oracle else 0] = np.nan
    return v


def winsor(v, q=WIN_Q, look=WIN_LOOK):
    """Clip to quantiles of the TRAILING window only, so nothing leaks forward."""
    s = pd.Series(v)
    lo = s.rolling(look, min_periods=252).quantile(q).shift(1).values
    hi = s.rolling(look, min_periods=252).quantile(1 - q).shift(1).values
    ok = np.isfinite(lo) & np.isfinite(hi)
    out = v.copy()
    out[ok] = np.clip(v[ok], lo[ok], hi[ok])
    return out


def tail_z(v, idx, d):
    """The standardised value the ridge actually sees on each test day.

    The walk standardises every column with TRAINING-window statistics and then
    applies them to a test day that may lie far outside that window.  This
    replays exactly that arithmetic for one column, which is how a column can
    score worse than noise without carrying any anti-information: one test day
    at fifteen standard deviations moves a prediction further than a hundred
    ordinary days move it back.
    """
    out = np.full(len(idx), np.nan); mu = sd = None
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            tr = np.arange(max(0, t - d - H - L.TRAIN - L.VAL), max(t - d - H, 1))
            g = v[tr][np.isfinite(v[tr])]
            mu, sd = (None, None) if len(g) < 200 else (g.mean(), g.std() + 1e-9)
        if mu is not None and np.isfinite(v[t]):
            out[j] = abs((v[t] - mu) / sd)
    return out


def ar1_surrogates(v, rng, n_draw=N_DRAW):
    """Columns with this column's persistence and variance and none of its news."""
    fin = np.isfinite(v)
    x = v[fin]
    m = x.mean()
    v0, v1 = x[:-1] - m, x[1:] - m
    phi = float((v0 @ v1) / (v0 @ v0))
    sd = float(np.std(v1 - phi * v0))
    out = []
    for _ in range(n_draw):
        s = np.empty(len(v))
        e = rng.normal(scale=sd, size=len(v))
        s[0] = m + e[0] / np.sqrt(max(1 - phi ** 2, 1e-6))
        for t in range(1, len(v)):
            s[t] = m + phi * (s[t - 1] - m) + e[t]
        s[~fin] = np.nan
        out.append(s)
    return out, phi, sd


def boot_ci(yb, f_a, f_b, rng, n_boot=N_BOOT, block=BLK):
    """Block bootstrap of mean squared-error advantage of arm a over arm b."""
    d = (yb - f_b) ** 2 - (yb - f_a) ** 2
    n = len(d); nb = int(np.ceil(n / block)); offs = np.arange(block)
    st = rng.integers(0, n - block + 1, size=(n_boot, nb))
    o = np.array([d[(st[i][:, None] + offs).ravel()[:n]].mean() for i in range(n_boot)])
    lo, hi = np.percentile(o, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def gw(yb, f_small, f_big):
    d = (yb - f_small) ** 2 - (yb - f_big) ** 2
    z = d.mean() / F.hac_se(d)
    return float(z), float(F.norm_p(z))


# ------------------------------------------------------------------ main
def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("EXPLORATORY - cited by neither paper.\n")

    rng = np.random.default_rng(SEED)
    S, y, idx, D, k = panel(folder)
    yb = y[idx]
    print(f"\ntarget {TARGET}, {len(idx)} test days, "
          f"{D.index[idx[0]].date()} to {D.index[idx[-1]].date()}, {k} foreign peers")
    print(f"realised leg: {RV_WIN}-day Yang-Zhang, annualised, matched to VIX's "
          f"thirty calendar days")
    print(f"implied leg: VIX at t-{VIX_LAG_STRICT} (it prints after the close), "
          f"VDAX-NEW at t\n")

    p_us, p_de = premium(S, 0)[idx], ((S["ivd"] - S["rvd"]) / S["md"])[idx]
    print(f"{'premium':>10}{'mean':>10}{'median':>10}{'share > 0':>12}")
    for nm, v in (("US", p_us), ("Germany", p_de)):
        g = v[np.isfinite(v)]
        print(f"{nm:>10}{g.mean():>10.3f}{np.median(g):>10.3f}{np.mean(g > 0):>12.1%}")
    print("Scaled by each market's trailing median implied variance, so 0.50 means")
    print("the options market charged half as much again as variance delivered.")

    # ---------------- A ---------------------------------------------------
    print("\n" + "=" * 100)
    print("A.  THE RACE, WITH THE DOMESTIC BLOCK HELD delta DAYS STALE")
    print("=" * 100)
    order = ["OWN", "+FOR", "+IV", "+IV+VRPus", "+IV+VRPde", "+IV+VRP", "ALL"]
    fc, r2 = {}, {}
    print(f"{'delta':>6}" + "".join(f"{nm:>13}" for nm in order))
    for d in DELAYS:
        row = f"{d:>6}"
        for nm in order:
            f = walk(S, y, idx, d, ARMS[nm])
            fc[(d, nm)] = f
            r2[(d, nm)] = F.r2(yb, f)
            row += f"{r2[(d, nm)]:>13.4f}"
        print(row)
    print("\nOWN and +FOR and +IV reproduce Section 6's table on this sample; they are")
    print("here as scale, not as news.  Every column that adds a premium sits BELOW")
    print("+IV, which on its own says nothing: lab07 showed that extra regressors cost")
    print("R-squared whether or not they inform.  That is what part B is for.")

    # ---------------- B ---------------------------------------------------
    print("\n" + "=" * 100)
    print("B.  ONE EXTRA COLUMN, FOUR WAYS - THE COMPARISON THE COST CANCELS OUT OF")
    print("=" * 100)
    print("Every arm here adds exactly ONE column to +IV, so lab07's estimation cost")
    print("is the same in all of them and vanishes from any difference between them.")
    print(f"NOISE is the mean over {N_DRAW} AR(1) surrogate draws matched to the VRPus")
    print("column's persistence and residual scale: the cost with the content removed.\n")

    ora, noi, phis = {}, {}, {}
    for d in DELAYS:
        ora[d] = walk(S, y, idx, d, ARMS["+IV+ORACLE"])
        v = premium(S, d)
        draws, phi, sd = ar1_surrogates(v, np.random.default_rng(SEED + d))
        phis[d] = phi
        fs = []
        for s in draws:
            S["noise"] = s
            fs.append(walk(S, y, idx, d, ARMS["+IV+NOISE"]))
        noi[d] = np.mean(fs, axis=0)
        S.pop("noise", None)
        # keep the draws for part D, where the column count has to match again
        S.setdefault("_draws", {})[d] = draws

    print(f"{'delta':>6}{'NOISE':>11}{'VRPus':>11}{'ORACLE':>11}{'VRPde':>11}"
          f"{'  AR(1) phi':>12}")
    for d in DELAYS:
        print(f"{d:>6}{F.r2(yb, noi[d]):>11.4f}{r2[(d, '+IV+VRPus')]:>11.4f}"
              f"{F.r2(yb, ora[d]):>11.4f}{r2[(d, '+IV+VRPde')]:>11.4f}"
              f"{phis[d]:>12.3f}")

    chk = abs(F.r2(yb, ora[0]) - r2[(0, "+IV+VRPus")])
    print(f"\n  check: at delta = 0 the stale leg IS today's leg, so VRPus and ORACLE")
    print(f"  must coincide.  |difference| = {chk:.2e}  -> "
          f"{'PASS' if chk < 1e-12 else 'FAIL'}")

    print("\nEach premium against the uninformative column of the same width.")
    print("Positive means the premium carries something an AR(1) with its persistence")
    print("does not.  NEGATIVE means the real column scores WORSE than a column with")
    print("no content at all, which is a different claim and is counted separately.\n")
    print(f"{'delta':>6}{'VRPus - NOISE':>16}{'95% CI':>26}"
          f"{'VRPde - NOISE':>16}{'95% CI':>26}")
    us_pos, de_pos, us_neg, de_neg = [], [], [], []
    for d in DELAYS:
        a1, l1, h1 = boot_ci(yb, fc[(d, "+IV+VRPus")], noi[d],
                             np.random.default_rng(SEED + 100 + d))
        a2, l2, h2 = boot_ci(yb, fc[(d, "+IV+VRPde")], noi[d],
                             np.random.default_rng(SEED + 200 + d))
        (us_pos if l1 > 0 else us_neg if h1 < 0 else []).append(d)
        (de_pos if l2 > 0 else de_neg if h2 < 0 else []).append(d)
        print(f"{d:>6}{F.r2(yb, fc[(d, '+IV+VRPus')]) - F.r2(yb, noi[d]):>+16.4f}"
              + f"[{l1:>+10.5f},{h1:>+10.5f}]".rjust(26)
              + f"{r2[(d, '+IV+VRPde')] - F.r2(yb, noi[d]):>+16.4f}"
              + f"[{l2:>+10.5f},{h2:>+10.5f}]".rjust(26))
    print(f"\n  domestic premium: beats noise at {len(us_pos)} of {len(DELAYS)} delays, "
          f"LOSES to noise at {len(us_neg)} {us_neg if us_neg else ''}")
    print(f"  Germany's premium: beats noise at {len(de_pos)} of {len(DELAYS)} delays, "
          f"LOSES to noise at {len(de_neg)} {de_neg if de_neg else ''}")
    if not us_pos and not de_pos and not (us_neg or de_neg):
        print("  Neither premium carries anything the level was not already carrying,")
        print("  and neither does damage.  Section 6 raced the level and would have")
        print("  reached the same verdict against the premium.")
    elif us_neg or de_neg:
        print("  At least one premium is significantly WORSE than a column containing")
        print("  nothing.  A variable cannot carry negative information, so either the")
        print("  surrogate is too easy a benchmark or the real column has a property the")
        print("  surrogate lacks.  Part B3 tests the obvious candidate rather than")
        print("  leaving the reader to guess.")
    elif de_pos and not us_pos:
        print("  Only the premium a delayed desk CAN compute carries anything - which is")
        print("  this project's hypothesis appearing on a variable neither paper uses.")
    else:
        print("  At least one premium beats an uninformative column of the same width,")
        print("  so the premium is a live variable here and the delays where it are")
        print("  listed above.")

    # ---------------- B3 --------------------------------------------------
    print("\n" + "=" * 100)
    print("B3.  WHY A REAL COLUMN CAN LOSE TO AN EMPTY ONE")
    print("=" * 100)
    print("The ridge standardises each column on the training window and then applies")
    print("those statistics to a test day that may lie far outside it.  A Gaussian")
    print("AR(1) never produces such a day; a variance premium in a crisis does.  The")
    print("table is the largest standardised value each column actually presents to the")
    print("model, computed by replaying the refit schedule.\n")
    print(f"{'delta':>6}{'max |z| VRPus':>16}{'max |z| NOISE':>16}"
          f"{'max |z| VRPde':>16}{'days |z|>6':>13}")
    for d in DELAYS:
        zu = tail_z(premium(S, d), idx, d)
        zn = tail_z(S["_draws"][d][0], idx, d)
        zd = tail_z((S["ivd"] - S["rvd"]) / S["md"], idx, d)
        big = int(np.nansum(zu > 6) + np.nansum(zd > 6))
        print(f"{d:>6}{np.nanmax(zu):>16.1f}{np.nanmax(zn):>16.1f}"
              f"{np.nanmax(zd):>16.1f}{big:>13}")

    print("\nSo clip the tails - at the 1st and 99th percentile of the TRAILING")
    print(f"{WIN_LOOK} days only, so nothing leaks forward - and run the same race.")
    print("If the deficit disappears, the premium's problem was its tails in a ridge,")
    print("not its content.  If it survives, the tails were never the explanation.\n")
    S["wind"] = winsor((S["ivd"] - S["rvd"]) / S["md"])
    print(f"{'delta':>6}{'WINus - NOISE':>16}{'95% CI':>26}"
          f"{'WINde - NOISE':>16}{'95% CI':>26}")
    wu_pos, wd_pos, wu_neg, wd_neg = [], [], [], []
    for d in DELAYS:
        S["winu"] = winsor(premium(S, d))
        f_wu = walk(S, y, idx, d, ARMS["+IV+WINus"])
        f_wd = walk(S, y, idx, d, ARMS["+IV+WINde"])
        a1, l1, h1 = boot_ci(yb, f_wu, noi[d], np.random.default_rng(SEED + 600 + d))
        a2, l2, h2 = boot_ci(yb, f_wd, noi[d], np.random.default_rng(SEED + 700 + d))
        (wu_pos if l1 > 0 else wu_neg if h1 < 0 else []).append(d)
        (wd_pos if l2 > 0 else wd_neg if h2 < 0 else []).append(d)
        print(f"{d:>6}{F.r2(yb, f_wu) - F.r2(yb, noi[d]):>+16.4f}"
              + f"[{l1:>+10.5f},{h1:>+10.5f}]".rjust(26)
              + f"{F.r2(yb, f_wd) - F.r2(yb, noi[d]):>+16.4f}"
              + f"[{l2:>+10.5f},{h2:>+10.5f}]".rjust(26))
    print(f"\n  clipped domestic premium: beats noise at {len(wu_pos)} of {len(DELAYS)}, "
          f"loses at {len(wu_neg)} (was {len(us_neg)} unclipped)")
    print(f"  clipped German premium:   beats noise at {len(wd_pos)} of {len(DELAYS)}, "
          f"loses at {len(wd_neg)} (was {len(de_neg)} unclipped)")
    healed = (len(wu_neg) + len(wd_neg)) < (len(us_neg) + len(de_neg))
    if healed and not (wu_pos or wd_pos):
        print("  Clipping removes the deficit and produces no surplus.  The premium's")
        print("  apparent harm was its tails meeting a standardising ridge, and its")
        print("  content is indistinguishable from nothing either way.  That is a")
        print("  caution about the estimator, not a finding about variance premia.")
    elif healed:
        print("  Clipping both removes the deficit and leaves a surplus, so the premium")
        print("  does carry something and the raw column was hiding it behind its tails.")
    elif wu_pos or wd_pos:
        print("  Clipping does not explain the deficit, yet a clipped premium still beats")
        print("  noise somewhere.  Both readings are in the table and neither is clean.")
    else:
        print("  The deficit survives clipping, so the tails were not the explanation and")
        print("  the AR(1) surrogate is flattering itself as a benchmark - most likely by")
        print("  being smoother than any real column of this persistence.")
    S.pop("winu", None); S.pop("wind", None)

    # ---------------- C ---------------------------------------------------
    print("\n" + "=" * 100)
    print("C.  WHAT THE DELAY DOES TO THE PREMIUM, AND WHETHER GERMANY STANDS IN")
    print("=" * 100)
    print("ORACLE minus VRPus is the damage: same model, same width, same everything,")
    print("except that ORACLE is handed a realised leg it has no right to see.  VRPde")
    print("minus VRPus asks whether the premium you do not own replaces the one you")
    print("cannot compute.  Neither difference needs a cost subtraction.\n")
    print(f"{'delta':>6}{'ORACLE - VRPus':>17}{'95% CI':>26}"
          f"{'VRPde - VRPus':>16}{'95% CI':>26}")
    dmg, dmg_sig, de_beats, de_loses = [], [], [], []
    for d in DELAYS:
        a1, l1, h1 = boot_ci(yb, ora[d], fc[(d, "+IV+VRPus")],
                             np.random.default_rng(SEED + 300 + d))
        a2, l2, h2 = boot_ci(yb, fc[(d, "+IV+VRPde")], fc[(d, "+IV+VRPus")],
                             np.random.default_rng(SEED + 400 + d))
        g1 = F.r2(yb, ora[d]) - r2[(d, "+IV+VRPus")]
        g2 = r2[(d, "+IV+VRPde")] - r2[(d, "+IV+VRPus")]
        dmg.append(g1)
        if l1 > 0:
            dmg_sig.append(d)
        if l2 > 0:
            de_beats.append(d)
        if h2 < 0:
            de_loses.append(d)
        print(f"{d:>6}{g1:>+17.4f}" + f"[{l1:>+10.5f},{h1:>+10.5f}]".rjust(26)
              + f"{g2:>+16.4f}" + f"[{l2:>+10.5f},{h2:>+10.5f}]".rjust(26))
    rise = sum(dmg[i + 1] >= dmg[i] - 1e-12 for i in range(len(dmg) - 1))
    print(f"\n  delays where the stale leg significantly costs something: "
          f"{len(dmg_sig)} of {len(DELAYS)} {dmg_sig if dmg_sig else ''}")
    print(f"  damage does not fall as the delay grows at {rise} of {len(dmg) - 1} steps")
    print(f"  delays where Germany's premium significantly beats the domestic one: "
          f"{len(de_beats)} of {len(DELAYS)} {de_beats if de_beats else ''}")
    print(f"  delays where it significantly LOSES to the domestic one: "
          f"{len(de_loses)} of {len(DELAYS)} {de_loses if de_loses else ''}")

    if dmg_sig and de_beats:
        share = [(r2[(d, "+IV+VRPde")] - r2[(d, "+IV+VRPus")]) /
                 (F.r2(yb, ora[d]) - r2[(d, "+IV+VRPus")])
                 for d in dmg_sig if F.r2(yb, ora[d]) - r2[(d, "+IV+VRPus")] > 1e-9]
        if share:
            print(f"  where both hold, Germany puts back "
                  f"{min(share):.0%} to {max(share):.0%} of what the delay took.")
        print("  That is the paper's substitution story, on a variable the paper never")
        print("  used, with the estimation cost cancelled rather than subtracted.")
    elif dmg_sig:
        print("  The delay measurably damages the domestic premium and Germany's does")
        print("  not step into the gap.  The channel is spoiled without being")
        print("  substitutable, which is a narrower result than the index cross-section")
        print("  gives and should not be reported as agreeing with it.")
    else:
        print("  Holding the realised leg stale costs nothing measurable, so within this")
        print("  model the premium is carried by its implied leg.  The delayed desk loses")
        print("  nothing here because there was nothing in the realised leg to lose.")
    if de_loses:
        print("  Germany's premium is significantly WORSE than the domestic one at the")
        if healed:
            print("  delays listed.  B3 showed that deficit is the column's tails meeting a")
            print("  standardising ridge, so it is not evidence against substitution - it")
            print("  is evidence that this estimator cannot be handed a raw premium.")
        else:
            print("  delays listed, and clipping its tails did not explain that away.  On")
            print("  this evidence the foreign premium is not a substitute for the")
            print("  domestic one, and saying otherwise would be reading past the table.")

    # ---------------- D ---------------------------------------------------
    print("\n" + "=" * 100)
    print("D.  IS THE DIFFERENCE THE RIGHT FORM, AND DOES THE WINDOW DECIDE IT?")
    print("=" * 100)
    print("The premium restricts the two legs to equal and opposite coefficients.  LEGS")
    print("frees them.  LEGS is two columns, so it is raced against VRPus PLUS one")
    print("uninformative column - equal width again, cost cancelled again.\n")
    print(f"{'delta':>6}{'LEGS - (VRPus+NOISE)':>23}{'95% CI':>26}")
    freed, bound = [], []
    for d in DELAYS:
        f_lg = walk(S, y, idx, d, ARMS["+IV+LEGS"])
        fs = []
        for s in S["_draws"][d]:
            S["noise"] = s
            fs.append(walk(S, y, idx, d, ARMS["+IV+VRP+N"]))
        S.pop("noise", None)
        f_vn = np.mean(fs, axis=0)
        a, lo, hi = boot_ci(yb, f_lg, f_vn, np.random.default_rng(SEED + 500 + d))
        if lo > 0:
            freed.append(d)
        if hi < 0:
            bound.append(d)
        print(f"{d:>6}{F.r2(yb, f_lg) - F.r2(yb, f_vn):>+23.4f}"
              + f"[{lo:>+10.5f},{hi:>+10.5f}]".rjust(26))
    print(f"\n  delays where freeing the legs significantly beats the premium: "
          f"{len(freed)} of {len(DELAYS)} {freed if freed else ''}")
    print(f"  delays where freeing them significantly LOSES: "
          f"{len(bound)} of {len(DELAYS)} {bound if bound else ''}")
    if freed:
        print("  The equal-and-opposite restriction is costing something: the two legs")
        print("  carry different information and the literature's difference throws part")
        print("  of it away.")
    elif bound:
        print("  Freeing the legs is significantly worse than adding an empty column, so")
        print("  the restriction is not the constraint - the raw legs are simply harder")
        print("  for a standardising ridge to hold than their difference is.")
    else:
        print("  Freeing the legs buys nothing over an uninformative column, so the")
        print("  restriction is not what is holding the premium back.")

    print(f"\nThe realised leg's window - {RV_WIN} days, matched to VIX, against the "
          f"project's {RV_ALT}:")
    S2, y2, idx2, D2, _ = panel(folder, rv_win=RV_ALT, quiet=True)
    yb2 = y2[idx2]
    print("A sign that flips between two estimates which both straddle zero is noise,")
    print("not disagreement, so a flip is only counted as MATERIAL when at least one of")
    print("the two intervals excludes zero.\n")
    print(f"{'delta':>6}{'VRPus-NOISE (' + str(RV_WIN) + 'd)':>22}"
          f"{'VRPus-NOISE (' + str(RV_ALT) + 'd)':>22}{'sign':>8}{'material?':>12}")
    flips, material = 0, 0
    for d in DELAYS:
        g22 = F.r2(yb, fc[(d, "+IV+VRPus")]) - F.r2(yb, noi[d])
        _, l22, h22 = boot_ci(yb, fc[(d, "+IV+VRPus")], noi[d],
                              np.random.default_rng(SEED + 100 + d))
        f_us = walk(S2, y2, idx2, d, ARMS["+IV+VRPus"])
        draws2, _, _ = ar1_surrogates(premium(S2, d), np.random.default_rng(SEED + d))
        fs = []
        for s in draws2:
            S2["noise"] = s
            fs.append(walk(S2, y2, idx2, d, ARMS["+IV+NOISE"]))
        S2.pop("noise", None)
        f_n2 = np.mean(fs, axis=0)
        g05 = F.r2(yb2, f_us) - F.r2(yb2, f_n2)
        _, l05, h05 = boot_ci(yb2, f_us, f_n2, np.random.default_rng(SEED + 800 + d))
        same = (g22 > 0) == (g05 > 0)
        sig = (l22 > 0 or h22 < 0) or (l05 > 0 or h05 < 0)
        flips += 0 if same else 1
        material += 1 if (not same and sig) else 0
        print(f"{d:>6}{g22:>+22.4f}{g05:>+22.4f}"
              f"{'same' if same else 'FLIP':>8}"
              f"{('-' if same else ('yes' if sig else 'no')):>12}")
    print(f"\n  sign disagreements: {flips} of {len(DELAYS)}, of which "
          f"{material} material")
    if material == 0 and flips:
        print("  The only flips are between two estimates that both straddle zero, which")
        print("  is two ways of measuring nothing rather than a contradiction.  The")
        print("  window does not decide the answer here because there is no answer for")
        print("  it to decide.")
    elif material == 0:
        print("  The premium's standing does not depend on which realised window builds")
        print("  it, so the finding is about the premium and not about a choice of")
        print("  window that happened to suit it.")
    else:
        print("  A flip between two estimates where one interval excludes zero is a real")
        print("  disagreement, so any claim made here has to name the window it was")
        print("  computed with.  Horizon matching is doing part of the work.")

    # ---------------- verdict ---------------------------------------------
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"  beats an uninformative column of the same width: domestic premium at "
          f"{len(us_pos)}/{len(DELAYS)} delays, Germany's at {len(de_pos)}/{len(DELAYS)}; "
          f"after clipping, {len(wu_pos)}/{len(DELAYS)} and {len(wd_pos)}/{len(DELAYS)}.")
    print(f"  the stale realised leg costs something at {len(dmg_sig)}/{len(DELAYS)}; "
          f"Germany's premium beats the domestic one at {len(de_beats)}/{len(DELAYS)}.")
    if not (us_pos or de_pos or wu_pos or wd_pos):
        print("  Section 6's choice of the LEVEL over the PREMIUM cost it nothing.  The")
        print("  premium is not a stronger competitor on this sample, clipped or raw, so")
        print("  the objection that Section 6 raced the wrong horse does not survive.")
        print("  That is a defence of a section this project had already conceded, and")
        print("  it is the useful result here.")
    if dmg_sig and not de_beats:
        print("  The delay damages the premium, but the foreign premium does not repair")
        print("  it.  The substitution result does not generalise from index closes to")
        print("  variance premia, and the papers should not be read as claiming it does.")
    if not dmg_sig:
        print("  The realised leg carries nothing this model can use, so the delay has")
        print("  nothing to take from it.  The premium's information, such as it is,")
        print("  lives in the implied leg - which Section 6 already had.")
    if healed:
        print("  Method note, and the part worth keeping: a raw variance premium loses")
        print("  to an empty column in a ridge that standardises on a training window.")
        print("  Clipping its tails causally repairs that.  Anyone regressing on a VRP")
        print("  this way is measuring their estimator's tail sensitivity, not the VRP.")
    if material:
        print("  Every statement above is conditional on the 22-day realised leg; the")
        print("  5-day leg materially disagrees at some delays, and that limits all of it.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
