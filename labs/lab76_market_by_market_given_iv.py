"""
lab76_market_by_market_given_iv.py - which single foreign market, if any, adds
anything once implied volatility is held?

Imports lab05_robustness and lab08_implied_vol; keep all three in labs/.
Runtime about twelve minutes.

THE QUESTION
------------
Section 7 reports that the foreign BLOCK adds nothing detectable to a forecaster
who already holds own-index implied volatility.  That is a statement about seven
series entering together, and it leaves a reader with a reasonable follow-up: is
the block redundant because every market in it is redundant, or because seven
correlated regressors cost more to estimate than any one of them is worth?

The two have different implications.  If one market carries something the others
do not, a forecaster should hold that one rather than all seven, and the
redundancy result would be a statement about the block's construction rather
than about breadth.  Measuring each market on its own is the test, and it is the
market-by-market counterpart of what the cross-sectional literature does when it
asks which edges of a correlation structure are informative rather than treating
the structure as one object (Aste and Di Matteo).

THE DESIGN
----------
Baseline: the stale own block at delay delta, plus the implied-volatility block
the paper's admissible arm uses - VIX lagged a full day and VDAX-NEW.  Then one
foreign close is added, current, and nothing else changes: same days, same refit
schedule, same ridge, same benchmark.  The statistic is Giacomini-White on the
per-day loss differential with a HAC standard error at the delay's own
bandwidth.

WHAT THIS GRID IS AND IS NOT
----------------------------
Seven markets by six delays is forty-two cells, reported after the paper's
headline specification was fixed.  They are NOT part of the post-selection pool
Section S37 enumerates, which is closed and was declared before this file
existed; they carry their own correction here and nothing in the paper rests on
them.  Three corrections are printed rather than one, because they answer
different questions: Holm for the family-wise error, Benjamini-Hochberg for the
false discovery rate, and a joint null for the COUNT, which is the only one of
the three that respects the dependence between cells sharing evaluation days.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab08_implied_vol as IV

SEED = 20261015
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 3, 5, 13, 21, 55]
N_JOINT = 2000


def holm(ps):
    """Holm-adjusted p-values, in the order given."""
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * ps[j])
        adj[j] = min(1.0, run)
    return adj


def bh(ps):
    """Benjamini-Hochberg adjusted p-values, in the order given."""
    m = len(ps)
    order = np.argsort(ps)[::-1]
    adj = np.empty(m)
    run = 1.0
    for i, j in enumerate(order):
        run = min(run, ps[j] * m / (m - i))
        adj[j] = min(1.0, run)
    return adj


def block_starts(n, rng, block):
    nb = int(np.ceil(n / block))
    st = rng.integers(0, max(1, n - block + 1), size=nb)
    return (st[:, None] + np.arange(block)).ravel()[:n]


def main(folder=None):
    folder = L.find_folder(folder)
    rng = np.random.default_rng(SEED)
    print(f"data folder: {folder}")
    print("Prints each foreign market on its own, given implied volatility.\n")

    D, iv, peers = IV.build(folder, strict=True)
    own, P, IVM, y, idx = IV.make(D, iv, peers)
    yb = y[idx]
    bench = L.bench_mean(y, idx)
    print(f"\ntarget {TARGET}, {len(idx)} test days, {len(peers)} foreign peers")
    print(f"implied-volatility block: {', '.join(iv.columns)}, "
          f"VIX lagged a full day\n")

    print("=" * 96)
    print("A.  ONE MARKET AT A TIME, AGAINST THE IMPLIED-VOLATILITY BASELINE")
    print("=" * 96)
    print("Positive means the market adds to a forecaster already holding")
    print("implied volatility.  z is Giacomini-White on the loss differential,")
    print("HAC at the delay's own bandwidth.\n")

    rows = []
    for d in DELAYS:
        f_base = IV.walk(own, [IVM], y, idx, d)
        e_base = (yb - f_base) ** 2
        for j, p in enumerate(peers):
            f_one = IV.walk(own, [IVM, P[:, [j]]], y, idx, d)
            e_one = (yb - f_one) ** 2
            dl = e_base - e_one                     # positive: the market helps
            sst = float(((yb - bench) ** 2).sum())
            inc = dl.sum() / sst
            z = dl.mean() / IV.hac_se(dl, lag=L.block_for(d))
            # IV.norm_p IS the two-sided p-value, not a CDF: wrapping it in
            # 2 * (1 - .) turns z = 0 into p = 0, which starred a cell whose
            # increment is zero to four decimals.
            rows.append(dict(delay=d, market=p, inc=inc, z=z,
                             p=IV.norm_p(z), dl=dl))
    print(f"{'delta':>6}" + "".join(f"{p:>12}" for p in peers))
    for d in DELAYS:
        line = f"{d:>6}"
        for p in peers:
            r = [x for x in rows if x["delay"] == d and x["market"] == p][0]
            star = "*" if r["p"] < 0.05 else " "
            line += f"{r['inc']:>+11.4f}{star}"
        print(line)
    print(f"\n{'delta':>6}" + "".join(f"{p:>12}" for p in peers))
    for d in DELAYS:
        line = f"{d:>6}"
        for p in peers:
            r = [x for x in rows if x["delay"] == d and x["market"] == p][0]
            line += f"{r['z']:>12.2f}"
        print(line)

    ps = np.array([r["p"] for r in rows])
    inc = np.array([r["inc"] for r in rows])
    nom = int((ps < 0.05).sum())
    pos = int(((ps < 0.05) & (inc > 0)).sum())
    neg = int(((ps < 0.05) & (inc < 0)).sum())
    print(f"\n  cells: {len(rows)}; nominally significant at 5%: {nom} "
          f"({pos} positive, {neg} negative)")

    # ---------------- B ----------------------------------------------------
    print("\n" + "=" * 96)
    print("B.  THE SAME CELLS, CORRECTED THREE WAYS")
    print("=" * 96)
    hp, bp = holm(ps), bh(ps)
    print(f"  Holm, family-wise at 5%:            {int((hp < 0.05).sum())} of "
          f"{len(rows)} survive")
    print(f"  Benjamini-Hochberg, FDR at 5%:      {int((bp < 0.05).sum())} of "
          f"{len(rows)} survive")
    surv = [(r["market"], r["delay"], r["inc"], r["z"], h, b)
            for r, h, b in zip(rows, hp, bp) if h < 0.05 or b < 0.05]
    if surv:
        print(f"\n{'market':>12}{'delta':>7}{'increment':>12}{'z':>8}"
              f"{'Holm p':>9}{'BH p':>8}")
        for mk, d, i, z, h, b in sorted(surv, key=lambda t: t[1]):
            print(f"{mk:>12}{d:>7}{i:>+12.4f}{z:>8.2f}{h:>9.3f}{b:>8.3f}")
    else:
        print("  nothing survives either correction")

    # ---------------- C ----------------------------------------------------
    print("\n" + "=" * 96)
    print("C.  THE COUNT, AGAINST A JOINT NULL THAT KEEPS THE DEPENDENCE")
    print("=" * 96)
    print("Holm and BH treat the cells as a list.  These cells share evaluation")
    print("days, so the count's own distribution is what the question needs:")
    print("every cell demeaned under its null, then the whole grid resampled")
    print("with ONE set of block starts per replication, so the cross-cell")
    print("alignment is carried into the null rather than assumed away.\n")
    Dmat = np.column_stack([r["dl"] - r["dl"].mean() for r in rows])
    lags = [L.block_for(r["delay"]) for r in rows]
    blk = max(lags)
    n = Dmat.shape[0]
    counts = np.empty(N_JOINT, dtype=int)
    for b in range(N_JOINT):
        s = block_starts(n, rng, blk) % n
        X = Dmat[s]
        c = 0
        for j in range(X.shape[1]):
            v = X[:, j]
            se = IV.hac_se(v, lag=lags[j])
            if se > 0 and abs(v.mean() / se) > 1.96:
                c += 1
        counts[b] = c
    pj = (1 + int((counts >= nom).sum())) / (N_JOINT + 1)
    print(f"  observed count {nom}; under the joint null the count has mean "
          f"{counts.mean():.1f},")
    print(f"  standard deviation {counts.std():.1f}, 95th percentile "
          f"{np.percentile(counts, 95):.0f}")
    print(f"  p for the count: {pj:.4f} from {N_JOINT} replications")
    print(f"  a binomial at 5% of {len(rows)} cells would have mean "
          f"{0.05 * len(rows):.1f} and standard deviation "
          f"{np.sqrt(len(rows) * 0.05 * 0.95):.1f}, which is the spread the")
    print("  dependence makes wrong.")

    # ---------------- D ----------------------------------------------------
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    best = max(rows, key=lambda r: r["inc"])
    worst = min(rows, key=lambda r: r["inc"])
    print(f"  largest single-market increment: {best['market']} at delta = "
          f"{best['delay']}, {best['inc']:+.4f} (z = {best['z']:.2f})")
    print(f"  largest single-market subtraction: {worst['market']} at delta = "
          f"{worst['delay']}, {worst['inc']:+.4f} (z = {worst['z']:.2f})")
    if int((hp < 0.05).sum()) == 0 and pj > 0.05:
        print("  No market survives a family-wise correction and the count is")
        print("  inside its own joint null, so the block's redundancy given")
        print("  implied volatility is not an artefact of pooling seven series:")
        print("  no single member of it is detectable on its own either.")
    else:
        print("  At least one cell survives correction, so the redundancy result")
        print("  is about the block and not about every market in it.  Which")
        print("  market, and at which delay, is in the table above.")
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
