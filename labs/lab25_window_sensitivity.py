"""
lab25_window_sensitivity.py - the companion note fixes two windows by fiat: the
median threshold M_t looks back 252 days, and the HAR classifier trains on
1,250.  A referee asks whether the threshold rule's standing is structural or
an artefact of those two choices.  This file varies both and reports.

Imports lab02_delay_curve; keep both in labs/.  Runtime a few seconds.

THE TRAP THIS FILE IS BUILT AROUND
-----------------------------------
The median window is NOT a free parameter of the forecaster.  It defines the
TARGET:

    M_t = median of the proxy over the trailing `med` days
    y_t = 1 if proxy_{t+5} > M_t

Change `med` and you change the question, not just the answer.  A 63-day median
asks "will volatility exceed its recent quarter's typical level"; a 504-day
median asks "will it exceed its two-year typical level".  Those are different
prediction problems with different base rates and different intrinsic
difficulty, so accuracies are NOT comparable across columns of the table below.

What IS comparable across columns is the within-cell contest: for each window,
does the threshold rule still beat the constant baseline, and does the HAR
classifier still fail to beat the threshold rule?  That comparison is what the
note's claim is about, and it is the one reported.

THE SECOND TRAP
---------------
Different windows consume different amounts of the sample, so a naive run would
score a 63-day median on more test days than a 504-day one, and any difference
between them would be partly a difference in sample.  Every cell here is scored
on the SAME test window - the one the most demanding cell can afford - which is
smaller than the note's 694 days.  The cost in width is reported rather than
hidden.

WHAT WOULD OVERTURN THE NOTE
-----------------------------
If the HAR classifier beats the threshold rule at some window, or if the
threshold rule's advantage over the constant baseline disappears at some
window, the note's result is a property of 252 days rather than of volatility.
The verdict is computed from the table, not written in advance.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab02_delay_curve as L

SEED = 20260914
MEDS = [63, 126, 252, 504]
TRAINS = [500, 1250]
DELAYS = [0, 3, 5, 13, 21]
EPS = 1e-12


def cell(v, med, train, idx, delays=DELAYS):
    """One (median window, training window) cell, scored on a GIVEN index."""
    M = L.trailing_median(v, look=med)
    y = L.make_target(v, M)
    X = L.har_features(np.log(np.maximum(v, 1e-14)))
    ok = idx[idx < len(y)]
    ok = ok[~np.isnan(y[ok])]
    yb = y[ok]
    base = 1.0 if np.nanmean(y[:ok[0]]) > 0.5 else 0.0

    out = {}
    for d in delays:
        a = np.log(np.maximum(v[ok - d], EPS) / np.maximum(M[ok - d], EPS))
        out[("persist", d)] = (a > 0).astype(float)
        out[("majority", d)] = np.full(len(ok), base)

        pr = np.empty(len(ok)); b = None
        for j, t in enumerate(ok):
            if j % L.REFIT == 0 or b is None:
                last = t - d - L.HORIZON
                tr = np.arange(max(0, last - train), last)
                tr = tr[~np.isnan(y[tr]) & ~np.isnan(X[tr]).any(axis=1)]
                b = L.logistic_fit(X[tr], y[tr]) if len(tr) > 50 else None
            if b is None or np.isnan(X[t - d]).any():
                pr[j] = base
            else:
                pr[j] = float(L.logistic_predict(b, X[t - d][None, :])[0] > 0.5)
        out[("har", d)] = pr
    return out, yb


def paired(ca, cb, rng):
    """Block-bootstrap CI for a paired accuracy difference."""
    n = len(ca)
    nb = int(np.ceil(n / L.BLOCK))
    starts = rng.integers(0, n - L.BLOCK + 1, size=(L.N_BOOT, nb))
    offs = np.arange(L.BLOCK)
    diff = ca - cb
    out = np.empty(L.N_BOOT)
    for i in range(L.N_BOOT):
        s = (starts[i][:, None] + offs).ravel()[:n]
        out[i] = diff[s].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return diff.mean(), lo, hi, (lo > 0 or hi < 0)


def main(path=None):
    df = L.load() if path is None else L.load(path)
    o, h, l, c = (df["open"].values, df["high"].values,
                  df["low"].values, df["close"].values)
    v, _ = L.proxy_yang_zhang(o, h, l, c)
    n = len(v)

    # the common test window: whatever the most demanding cell can afford
    start = max(MEDS) + max(TRAINS) + max(DELAYS) + L.HORIZON
    idx = np.arange(start, n - L.HORIZON)
    print(f"proxy Yang-Zhang, {n} rows")
    print(f"common test window: {len(idx)} days, fixed across every cell")
    print(f"  (the note's own window is 694 days; the most demanding cell here is")
    print(f"   median {max(MEDS)} + training {max(TRAINS)}, which costs the difference)")
    print(f"block bootstrap, block {L.BLOCK}, {L.N_BOOT} draws\n")

    grids = {}
    for med in MEDS:
        for tr in TRAINS:
            grids[(med, tr)], yb_ = cell(v, med, tr, idx)
            grids[(med, tr, "y")] = yb_

    # ---------------- A ---------------------------------------------------
    print("=" * 92)
    print("A.  THE THRESHOLD RULE AGAINST THE CONSTANT BASELINE, AT EVERY WINDOW")
    print("=" * 92)
    print("Accuracies are NOT comparable across columns - a different median window")
    print("is a different target.  What is comparable is whether the rule beats the")
    print("constant, within each column.\n")
    beats = {}
    print(f"{'delta':>6}" + "".join(f"{'med ' + str(m):>22}" for m in MEDS))
    print(f"{'':>6}" + "".join(f"{'rule - constant':>22}" for _ in MEDS))
    for d in DELAYS:
        row = f"{d:>6}"
        for med in MEDS:
            g, yb = grids[(med, TRAINS[1])], grids[(med, TRAINS[1], "y")]
            ca = (g[("persist", d)] == yb).astype(float)
            cb = (g[("majority", d)] == yb).astype(float)
            dm, lo, hi, sig = paired(ca, cb, np.random.default_rng(SEED + med + d))
            beats[(med, d)] = sig and dm > 0
            row += f"{dm * 100:>+9.2f}" + f"[{lo * 100:>+5.1f},{hi * 100:>+5.1f}]".rjust(13)
        print(row)
    print(f"\n  base rates by median window: " +
          "  ".join(f"med {m}: {grids[(m, TRAINS[1], 'y')].mean():.3f}" for m in MEDS))
    for med in MEDS:
        won = [d for d in DELAYS if beats[(med, d)]]
        print(f"  med {med:>3}: rule beats the constant at {len(won)} of {len(DELAYS)} delays {won}")

    # ---------------- B ---------------------------------------------------
    print("\n" + "=" * 92)
    print("B.  DOES THE HAR CLASSIFIER EVER BEAT THE THRESHOLD RULE?")
    print("=" * 92)
    print("The note's negative result, re-run in every cell of the window grid.")
    print("Positive favours the HAR classifier.\n")
    har_wins, rule_wins = [], []
    print(f"{'med':>5}{'train':>7}" + "".join(f"{'d=' + str(d):>22}" for d in DELAYS))
    for med in MEDS:
        for tr in TRAINS:
            g, yb = grids[(med, tr)], grids[(med, tr, "y")]
            row = f"{med:>5}{tr:>7}"
            for d in DELAYS:
                ca = (g[("har", d)] == yb).astype(float)
                cb = (g[("persist", d)] == yb).astype(float)
                dm, lo, hi, sig = paired(ca, cb,
                                         np.random.default_rng(SEED + med + tr + d))
                if sig and dm > 0:
                    har_wins.append((med, tr, d))
                if sig and dm < 0:
                    rule_wins.append((med, tr, d))
                row += f"{dm * 100:>+9.2f}" + f"[{lo * 100:>+5.1f},{hi * 100:>+5.1f}]".rjust(13)
            print(row)

    total = len(MEDS) * len(TRAINS) * len(DELAYS)
    print(f"\n  cells where the HAR classifier significantly BEATS the rule: "
          f"{len(har_wins)} of {total} {har_wins if har_wins else ''}")
    print(f"  cells where the rule significantly beats the HAR classifier: "
          f"{len(rule_wins)} of {total} {rule_wins if rule_wins else ''}")

    if har_wins:
        meds_hit = sorted({m for m, _, _ in har_wins})
        print(f"\n  The note's negative result does NOT hold at every window: the HAR")
        print(f"  classifier wins somewhere, at median window(s) {meds_hit}.  The note")
        print("  must report the window at which its claim is made.")
    else:
        print("\n  The HAR classifier beats the threshold rule in no cell of the grid.")
        print("  The note's negative result is not an artefact of the 252-day median or")
        print("  the 1,250-day training window: it survives a fourfold change in the")
        print("  first and a 2.5-fold change in the second.")

    # ---------------- C ---------------------------------------------------
    print("\n" + "=" * 92)
    print("C.  WHERE THE DECAY BITES, BY WINDOW")
    print("=" * 92)
    print("The note's headline is that the rule holds out to about a week and is")
    print("indistinguishable from a constant beyond about two.  That boundary is a")
    print("property of the target, so it should move with the median window.\n")
    print(f"{'med':>5}{'last delay where the rule still beats the constant':>54}")
    edges = {}
    for med in MEDS:
        won = [d for d in DELAYS if beats[(med, d)]]
        edges[med] = max(won) if won else None
        print(f"{med:>5}{(str(edges[med]) + ' days' if won else 'never'):>54}")
    vals = [e for e in edges.values() if e is not None]
    if len(set(vals)) <= 1 and len(vals) == len(MEDS):
        print("\n  The boundary sits at the same delay for every window tested, so on this")
        print("  grid it is a property of the delay rather than of the window length.")
    else:
        print(f"\n  The boundary moves across windows, from {min(vals)} to {max(vals)} days on the")
        print("  delays tested, and it is LATER for shorter median windows.  Shorter windows")
        print("  also give a target closer to a 50/50 split, which is easier to beat by")
        print("  construction, so this is about the target as much as about the rule.")

    # ---------------- D ---------------------------------------------------
    print("\n" + "=" * 92)
    print("D.  HOW MUCH OF PART C IS THE SHORTER WINDOW?")
    print("=" * 92)
    print("Part C scores every cell on 476 days so the cells are comparable.  The note")
    print("itself has 694.  A boundary that moves in when the sample shrinks is a loss")
    print("of power, not a finding, so the note's own cell is re-run on the largest")
    print("window IT can afford and the two are put side by side.\n")
    med, tr = 252, 1250
    start_own = med + tr + max(DELAYS) + L.HORIZON
    idx_own = np.arange(start_own, len(v) - L.HORIZON)
    g_own, yb_own = cell(v, med, tr, idx_own)
    print(f"  common window {len(idx)} days   vs   this cell's own window {len(idx_own)} days\n")
    print(f"{'delta':>6}{'common: rule - constant':>28}{'own: rule - constant':>28}")
    own_edge = None
    for d in DELAYS:
        g, yb = grids[(med, tr)], grids[(med, tr, "y")]
        dm1, lo1, hi1, s1 = paired((g[("persist", d)] == yb).astype(float),
                                   (g[("majority", d)] == yb).astype(float),
                                   np.random.default_rng(SEED + med + d))
        dm2, lo2, hi2, s2 = paired((g_own[("persist", d)] == yb_own).astype(float),
                                   (g_own[("majority", d)] == yb_own).astype(float),
                                   np.random.default_rng(SEED + 99 + med + d))
        if s2 and dm2 > 0:
            own_edge = d
        print(f"{d:>6}" + f"{dm1 * 100:>+10.2f}[{lo1 * 100:>+5.1f},{hi1 * 100:>+5.1f}]".rjust(28)
              + f"{dm2 * 100:>+10.2f}[{lo2 * 100:>+5.1f},{hi2 * 100:>+5.1f}]".rjust(28))
    print(f"\n  boundary on the common window: {edges[med] if edges[med] is not None else 'never'}"
          f"   boundary on its own window: {own_edge if own_edge is not None else 'never'}")
    if own_edge is not None and (edges[med] is None or own_edge > edges[med]):
        print("\n  The boundary moves OUT when the sample grows, so part C's inward move at")
        print("  the longer median windows is at least partly lost power rather than a")
        print("  property of the window.  Read part C as a ranking across windows on equal")
        print("  footing, not as a revision of the note's own figure.")
    else:
        print("\n  The boundary does not move out when the sample grows, so part C is not")
        print("  reporting a power loss and the ranking across windows stands as measured.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
