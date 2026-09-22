"""
lab24_proper_scores.py - the companion note scores its forecasters with 0-1
accuracy alone.  A referee objects that accuracy discards everything except
which side of a half the forecast fell on.  This file scores the note's OWN
five forecasters under proper scoring rules and under a ranking measure, and
asks whether the threshold rule's standing survives the change.

Imports lab02_delay_curve; keep both in labs/.  Runtime under a minute.

WHY THIS IS NOT THE SAME QUESTION AS THE MAIN PAPER'S
-----------------------------------------------------
The main paper already evaluates a CONTINUOUS target under QLIKE and squared
error on the variance scale, and that is where a magnitude-sensitive loss
belongs.  It cannot answer the objection raised here, because the note's four
refinements are CLASSIFIERS of a binary target.  The question here is narrower
and specific to the note: is the threshold rule's parity with the HAR
classifier an artefact of discretising at one half?

THREE MEASURES, AND WHAT EACH ONE CAN SEE
------------------------------------------
  accuracy   which side of a half.  What the note reports.
  AUC        ranking only: does the forecaster order the days correctly?
             Invariant to any monotone transform, so a rule with no calibrated
             probability can still be scored, PROVIDED it has a margin behind
             it to rank by.  The persistence rules do: the decision variable
             a = log(v_{t-delta} / M_{t-delta}), whose SIGN is the rule.
  Brier      mean squared error of a PROBABILITY, and the log score is its
  and log    likelihood counterpart.  Both are strictly proper: they are
             minimised in expectation only by the true probability, so they
             reward calibration and sharpness together.  Neither can be
             computed for a forecaster that emits a hard 0 or 1 - scoring a
             confident wrong label gives Brier 1 and log score infinity, which
             is a statement about the scoring rule rather than the forecaster.

That asymmetry is the point rather than a nuisance.  A threshold rule can be
ranked and can be graded right or wrong, but it cannot state how sure it is.
Whether that matters is an empirical question, and it is the one below.

THE NULL
--------
The note's "majority" baseline is a constant label.  Its probabilistic
counterpart is the RECURSIVE base rate - the share of ones among labels already
resolved at the time of the forecast - which is what a forecaster could
actually have quoted.  It is not the ex-post base rate, and lab20 records what
happens when those two are confused.

WHAT WOULD OVERTURN THE NOTE
-----------------------------
If the HAR classifier beats the threshold rule under Brier, log score or AUC by
more than sampling error, then the note's negative result IS an artefact of the
0-1 loss and the note has to say so.  If it does not, the note's claim survives
a strictly harder test than it was originally put to.  The verdict below is
computed from the numbers, not written in advance.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab02_delay_curve as L

SEED = 20260914
DELAYS = L.DELAYS
EPS = 1e-12


# ----------------------------------------------------------------------
# scores
# ----------------------------------------------------------------------
def brier(y, p):
    return float(np.mean((p - y) ** 2))


def logscore(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)          # the clip is reported, not hidden
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, s):
    """Rank-based AUC, ties averaged.  Equivalent to the Mann-Whitney statistic."""
    y = np.asarray(y, float)
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sorted_s = np.asarray(s)[order]
    i = 0
    while i < len(s):                        # average ranks within each tie group
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ----------------------------------------------------------------------
# walk-forward that KEEPS the probability
# ----------------------------------------------------------------------
def run(y, feats, Ms, proxies, delays=DELAYS):
    """Like lab02's evaluate(), but retains probabilities and margins.

    lab02 collapses the logistic output with `float(p > 0.5)` on the line that
    produces its predictions.  That is exactly the information this file needs,
    so the loop is repeated here rather than imported - and part 0 checks the
    repeat against lab02's own accuracies, because a rewrite is where a silent
    difference hides.
    """
    n = len(y)
    start = L.TRAIN + max(delays) + L.HORIZON + L.MED_LOOK
    idx = np.arange(start, n)
    idx = idx[~np.isnan(y[idx])]
    ybar = y[idx]

    # recursive base rate: share of ones among labels resolved by t - delta - h
    out = {}
    for d in delays:
        rec = np.empty(len(idx))
        for j, t in enumerate(idx):
            last = t - d - L.HORIZON
            past = y[:last]
            past = past[~np.isnan(past)]
            rec[j] = past.mean() if len(past) else 0.5
        out[("base rate", d)] = dict(p=rec, s=rec,
                                     lab=(rec > 0.5).astype(float))

        for tag in proxies:
            v, M = proxies[tag], Ms[tag]
            # the margin whose SIGN is the persistence rule
            a = np.log(np.maximum(v[idx - d], EPS)
                       / np.maximum(M[idx - d], EPS))
            out[(f"persist-{tag}", d)] = dict(p=None, s=a,
                                              lab=(a > 0).astype(float))

            X = feats[tag]
            pr = np.empty(len(idx)); b = None
            maj = 1.0 if np.nanmean(y[:start]) > 0.5 else 0.0
            for j, t in enumerate(idx):
                if j % L.REFIT == 0 or b is None:
                    last = t - d - L.HORIZON
                    tr = np.arange(max(0, last - L.TRAIN), last)
                    tr = tr[~np.isnan(y[tr]) & ~np.isnan(X[tr]).any(axis=1)]
                    b = L.logistic_fit(X[tr], y[tr]) if len(tr) > 50 else None
                if b is None or np.isnan(X[t - d]).any():
                    pr[j] = maj
                else:
                    pr[j] = float(L.logistic_predict(b, X[t - d][None, :])[0])
            out[(f"har-{tag}", d)] = dict(p=pr, s=pr,
                                          lab=(pr > 0.5).astype(float))
    return out, ybar, idx


def boot_diff(vals_a, vals_b, rng, lower_is_better):
    """Block bootstrap of a paired score difference, on per-day contributions."""
    n = len(vals_a)
    nb = int(np.ceil(n / L.BLOCK))
    starts = rng.integers(0, n - L.BLOCK + 1, size=(L.N_BOOT, nb))
    offs = np.arange(L.BLOCK)
    d = vals_a - vals_b
    out = np.empty(L.N_BOOT)
    for i in range(L.N_BOOT):
        s = (starts[i][:, None] + offs).ravel()[:n]
        out[i] = d[s].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    sign = -1.0 if lower_is_better else 1.0
    return d.mean(), lo, hi, (lo > 0 or hi < 0), sign


def main(path=None):
    df = L.load() if path is None else L.load(path)
    o, h, l, c = (df["open"].values, df["high"].values,
                  df["low"].values, df["close"].values)
    vcc, _ = L.proxy_close_to_close(c)
    vyz, _ = L.proxy_yang_zhang(o, h, l, c)
    n = min(len(vcc), len(vyz))
    vcc, vyz = vcc[:n], vyz[:n]
    proxies = {"CC": vcc, "YZ": vyz}
    Ms = {t: L.trailing_median(v) for t, v in proxies.items()}
    feats = {t: L.har_features(np.log(np.maximum(v, 1e-14)))
             for t, v in proxies.items()}
    y = L.make_target(vcc, Ms["CC"])

    res, ybar, idx = run(y, feats, Ms, proxies)
    rng = np.random.default_rng(SEED)
    m = len(idx)
    print(f"test window: {m} days;  realised base rate {ybar.mean():.3f}")
    print(f"block bootstrap, block {L.BLOCK}, {L.N_BOOT} draws\n")

    MODELS = ["base rate", "persist-CC", "persist-YZ", "har-CC", "har-YZ"]

    # ---------------- 0. reconciliation ----------------------------------
    print("=" * 88)
    print("0.  RECONCILIATION WITH lab02's OWN ACCURACIES")
    print("=" * 88)
    print("This file repeats lab02's walk-forward to keep the probability that")
    print("lab02 discards.  A repeat is where a silent difference hides, so")
    print("lab02's own evaluate() is called here and the accuracies compared.\n")
    ref, yb2, idx2 = L.evaluate(y, feats, Ms, proxies)
    bad = 0
    print(f"{'delta':>6}{'model':>14}{'here':>10}{'lab02':>10}")
    for d in (0, 5, 55):
        for mod in ("persist-YZ", "har-YZ"):
            a1 = float((res[(mod, d)]["lab"] == ybar).mean()) * 100
            a2 = float((ref[(mod, d)] == yb2).mean()) * 100
            ok = abs(a1 - a2) < 1e-9
            bad += (not ok)
            print(f"{d:>6}{mod:>14}{a1:>9.2f}%{a2:>9.2f}%   {'ok' if ok else 'DIFFERS'}")
    print(f"\n  {'the rewrite matches lab02 exactly' if not bad else str(bad) + ' cells differ - do not read on'}")

    # ---------------- A. the three measures ------------------------------
    print("\n" + "=" * 88)
    print("A.  ACCURACY, RANKING, AND PROBABILITY QUALITY")
    print("=" * 88)
    print("acc and AUC are available for every forecaster: the persistence rules")
    print("are ranked by the margin a = log(v/M) whose sign IS the rule.  Brier")
    print("and log score need a probability, which a hard rule does not emit;")
    print("those cells are n/a by construction, not by omission.\n")

    print("Two AUC columns, and the distance between them is the referee's point")
    print("made into a measurement.  AUC(margin) ranks days by the continuous")
    print("decision variable a forecaster HAS; AUC(label) ranks by the 0/1 it")
    print("actually EMITS, and for a binary forecast it equals balanced accuracy.")
    print("A rule that throws away its margin gives up the gap between them.\n")
    store = {}
    print(f"{'delta':>6}{'model':>13}{'acc':>9}{'AUC(margin)':>13}{'AUC(label)':>12}"
          f"{'Brier':>9}{'log':>9}")
    for d in DELAYS:
        for mod in MODELS:
            r = res[(mod, d)]
            acc = float((r["lab"] == ybar).mean())
            au = auc(ybar, r["s"])
            al = auc(ybar, r["lab"])
            if r["p"] is None:
                br = lg = None
                cells = f"{'n/a':>9}{'n/a':>9}"
            else:
                br, lg = brier(ybar, r["p"]), logscore(ybar, r["p"])
                cells = f"{br:>9.4f}{lg:>9.4f}"
            store[(mod, d)] = (acc, au, al, br, lg)
            print(f"{d:>6}{mod:>13}{acc:>8.2%}{au:>13.3f}{al:>12.3f}{cells}")
        print()

    gaps = {d: store[("persist-YZ", d)][1] - store[("persist-YZ", d)][2]
            for d in DELAYS}
    pos = [d for d in DELAYS if gaps[d] > 0]
    best, worst = max(gaps, key=gaps.get), min(gaps, key=gaps.get)
    print(f"  persist-YZ's margin out-ranks its own label at {len(pos)} of {len(DELAYS)} delays,")
    print(f"  by as much as {gaps[best]:.3f} of AUC at delta = {best}.  At delta = {worst} the")
    print(f"  label ranks {abs(gaps[worst]):.3f} HIGHER than the margin, which is possible")
    print("  because a sign is not a monotone summary of a margin once the margin's")
    print("  ordering stops tracking the outcome.  Either way the two differ, and the")
    print("  note's accuracy table cannot see the difference.\n")

    # the base rate's AUC is not a chance benchmark, and saying so is not enough
    tt = np.arange(len(ybar), dtype=float)
    a_time = auc(ybar, tt)
    a_base = store[("base rate", DELAYS[0])][1]
    print("  A caution on the base-rate row, which reads below chance.  The recursive")
    print("  base rate is a running mean of resolved labels: it is nearly constant, so")
    print("  its ranking is decided by slow drift rather than by anything it knows.")
    print(f"  Its AUC is {a_base:.3f}; ranking by calendar time alone gives {a_time:.3f}.")
    if abs(a_time - a_base) < 0.02:
        print("  Those agree, so the row is measuring drift in the target and nothing else.")
    else:
        print("  Those do NOT agree, so the drift is not a simple calendar trend - the")
        print("  running mean rises after clusters of ones, and the target mean-reverts")
        print("  against it.  We report the mechanism as unresolved rather than guess.")
    print("  In either case the row is not a chance benchmark and should not be read as")
    print("  one; chance for every other row is 0.500 by construction.\n")

    # ---------------- B. does the ranking change? ------------------------
    print("=" * 88)
    print("B.  DOES THE CHANGE OF MEASURE CHANGE THE VERDICT?")
    print("=" * 88)
    print("The note's claim is that har-YZ does not improve on persist-YZ.  It was")
    print("established on accuracy.  Here it is put to AUC as well, paired on the")
    print("same days, with a block bootstrap that respects the 5-day overlap.\n")

    flips = []
    print(f"{'delta':>6}{'acc diff':>12}{'95% CI':>20}{'AUC diff':>11}{'95% CI':>20}")
    for d in DELAYS:
        ra, rb = res[("har-YZ", d)], res[("persist-YZ", d)]
        ca = (ra["lab"] == ybar).astype(float)
        cb = (rb["lab"] == ybar).astype(float)
        dm, lo, hi, sig, _ = boot_diff(ca, cb, np.random.default_rng(SEED + d),
                                       lower_is_better=False)
        a1, a0 = store[("har-YZ", d)][1], store[("persist-YZ", d)][1]
        # AUC is not a per-day mean, so its interval comes from resampling days
        nb = int(np.ceil(len(ybar) / L.BLOCK))
        rr = np.random.default_rng(SEED + 1000 + d)
        starts = rr.integers(0, len(ybar) - L.BLOCK + 1, size=(500, nb))
        offs = np.arange(L.BLOCK)
        da = np.empty(500)
        for i in range(500):
            s = (starts[i][:, None] + offs).ravel()[:len(ybar)]
            da[i] = auc(ybar[s], ra["s"][s]) - auc(ybar[s], rb["s"][s])
        alo, ahi = np.nanpercentile(da, [2.5, 97.5])
        asig = (alo > 0 or ahi < 0)
        if sig or asig:
            flips.append((d, sig, asig))
        print(f"{d:>6}{dm * 100:>+11.2f}"
              + f"[{lo * 100:>+6.2f},{hi * 100:>+6.2f}]".rjust(20)
              + f"{a1 - a0:>+11.4f}"
              + f"[{alo:>+7.4f},{ahi:>+7.4f}]".rjust(20))

    print(f"\n  delays at which the two forecasters are separated by either measure: "
          f"{len(flips)} of {len(DELAYS)}")
    if flips:
        worse = [d for d, _, sa in flips if sa
                 and store[("har-YZ", d)][1] < store[("persist-YZ", d)][1]]
        better = [d for d, _, sa in flips if sa
                  and store[("har-YZ", d)][1] > store[("persist-YZ", d)][1]]
        for d, s_acc, s_auc in flips:
            by = " and ".join([x for x, f in (("accuracy", s_acc), ("AUC", s_auc)) if f])
            direction = ("har-YZ worse"
                         if store[("har-YZ", d)][1] < store[("persist-YZ", d)][1]
                         else "har-YZ better")
            print(f"    delta = {d}: separated by {by}  ({direction} on AUC)")
        print(f"\n  significant AUC gaps favouring the SIMPLE rule: {len(worse)}"
              f"  {worse if worse else ''}")
        print(f"  significant AUC gaps favouring the HAR classifier: {len(better)}"
              f"  {better if better else ''}")
        if worse and not better:
            print("\n  The objection is answered in the opposite direction to the one it")
            print("  anticipated.  A measure that sees the whole ranking does separate the")
            print("  two, and it separates them in the threshold rule's favour: the raw")
            print("  margin orders the days better than the fitted logistic's probability")
            print("  does.  The note's claim of parity on accuracy is, if anything, too")
            print("  generous to the refinement.")
        elif better and not worse:
            print("\n  The objection is vindicated: the HAR classifier ranks better than the")
            print("  threshold rule wherever the two are separated, and the note's negative")
            print("  result on accuracy IS an artefact of discretising at one half.")
        else:
            print("\n  The two measures disagree in direction across delays, so neither")
            print("  forecaster dominates and the note must report both columns.")
    else:
        print("\n  No delay separates them under either measure.  The note's negative result")
        print("  is not an artefact of discretising at one half: a measure that sees the")
        print("  whole ranking, and not just which side of a half the forecast fell on,")
        print("  reaches the same verdict.")

    # ---------------- C. what the threshold rule cannot do ---------------
    print("\n" + "=" * 88)
    print("C.  THE ONE THING ACCURACY HIDES")
    print("=" * 88)
    hard = [mo for mo in MODELS if res[(mo, 0)]["p"] is None]
    print("Forecasters with no probability, and so no Brier and no log score:")
    for mo in hard:
        print(f"    {mo}")
    print("\nThis is not a scoring inconvenience.  A user who needs to size a position")
    print("needs P(high volatility), not a label.  The comparison that matters for")
    print("them is between the probabilistic forecasters and the honest null.\n")
    print(f"{'delta':>6}{'har-YZ Brier':>15}{'base Brier':>13}{'diff':>10}"
          f"{'95% CI':>20}{'better?':>9}")
    wins = 0
    for d in DELAYS:
        pa = res[("har-YZ", d)]["p"]; pb = res[("base rate", d)]["p"]
        va = (pa - ybar) ** 2; vb = (pb - ybar) ** 2
        dm, lo, hi, sig, _ = boot_diff(va, vb, np.random.default_rng(SEED + 7 + d),
                                       lower_is_better=True)
        better = sig and dm < 0
        wins += better
        print(f"{d:>6}{va.mean():>15.4f}{vb.mean():>13.4f}{dm:>+10.4f}"
              + f"[{lo:>+7.4f},{hi:>+7.4f}]".rjust(20)
              + f"{'yes' if better else 'no':>9}")
    print(f"\n  delays at which a probabilistic forecaster beats the recursive base rate")
    print(f"  on Brier score: {wins} of {len(DELAYS)}")
    print("\n  Read with part B: where the two columns above disagree, the note's")
    print("  accuracy table and a risk manager's Brier score are answering different")
    print("  questions, and the note should report both rather than one.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
