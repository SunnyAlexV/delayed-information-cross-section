"""
lab60_outage_decomposition.py - is delta a stale FEATURE or a stale SYSTEM?

Imports lab05_robustness; keep both in labs/.  Runtime about eight minutes.

THE OBJECTION, IN ITS FAIR FORM
-------------------------------
The paper trains on pairs whose outcomes had resolved by t - delta, so the
training cut is c_t = t - delta - h.  Section 4 justifies that from the vintage
assumption: the label is the target's OWN decision variable, a forecaster whose
own mark is delta days old has seen that series only through t - delta, and the
newest pair they can possibly have observed is therefore the one whose outcome
lands at t - delta.

An external audit accepted the justification and then made a better point than
the one it replaced.  Whatever the reason, the consequence is that delta freezes
the WHOLE domestic information system at once: the current domestic state, the
available labels, the coefficient vintage, the ridge penalty chosen by the
validation tail, and the amount of training data.  So the design measures a
domestic-data OUTAGE of delta days, not merely a stale mark, and the paper had
been describing it as the second.  That is a real difference in what the number
means, and a reader is entitled to know which one they are being sold.

WHAT THIS FILE DOES
-------------------
It splits the outage into its two pieces by running a third arm that no
forecaster could run.

    S_own(0)        no delay anywhere            cut = t - h, features at t
    S_feat(delta)   STALE FEATURE ONLY           cut = t - h, features at t - delta
    S_out(delta)    the paper's design           cut = t - delta - h, features at t - delta

The middle arm is INFEASIBLE under the paper's own information set, and that is
stated rather than hidden: training to t - h needs labels whose outcomes land at
t, which is exactly the stretch of the target's own series the vintage assumption
withholds.  It is a decomposition device in the same sense as lab10's infeasible
median column - a quantity computed to price one effect, never a forecast this
paper reports.  Every arm shares the same feature construction, the same
estimator, the same refit schedule and the same test days, so the differences
below are the cut and nothing else.

    damage total    = S_own(0)   - S_out(delta)
    damage feature  = S_own(0)   - S_feat(delta)     what staleness of the INPUT costs
    damage vintage  = S_feat(d)  - S_out(delta)      what freezing ESTIMATION costs

WHY IT MATTERS TO THE HEADLINE
------------------------------
R(delta) divides by the total damage.  If most of that damage is the feature,
the rate is about information and the outage framing is a labelling matter.  If
much of it is the vintage, then part of what the cross-section is repairing is
the forecaster's inability to re-estimate, which is a different and smaller
claim.  The rate is therefore recomputed inside the feature-only arm as well, so
the paper can say which.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L

SEED = 20260920
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 5, 13, 21, 55]


def walk(own, P, y, idx, delta, use_peers, feature_only):
    """One walk-forward.

    `feature_only=False` is the paper's outage arm: the cut is t - delta - h.
    `feature_only=True` cuts at t - h instead, so the coefficients are as fresh
    as the estimator can make them while the origin feature stays delta days
    old.  Both train on DELAYED features, so the fitted map is the same object
    in both arms and only its vintage differs; training on current features and
    applying the fit to delayed ones would change the map as well as the cut and
    would price two things at once.
    """
    out = np.empty(len(idx))
    b = mu = sd = None
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = (t - H) if feature_only else (t - delta - H)
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), max(cut, 1))
            Xtr = (np.column_stack([own[tr - delta], P[tr]]) if use_peers
                   else own[tr - delta])
            ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(y[tr])
            Xtr, ytr = Xtr[ok], y[tr][ok]
            if len(ytr) < 200:
                b = None
                continue
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            b = L.cv((Xtr - mu) / sd, ytr, "cont")
        xt = (np.concatenate([own[t - delta], P[t]]) if use_peers
              else own[t - delta])
        out[j] = 0.0 if (b is None or not np.isfinite(xt).all()) \
            else L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
    return out


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints the outage decomposition behind Section 4.3 and Section S36.\n")

    D, peers, lag = L.build(folder, TARGET)
    a = D[TARGET].values
    sa = pd.Series(a)
    own = np.column_stack([sa.values, sa.rolling(5).mean().values,
                           sa.rolling(22).mean().values])
    P = D[peers].values
    n = len(D)
    y = np.full(n, np.nan)
    y[:-H] = a[H:]
    start = L.MED + L.TRAIN + L.VAL + max(DELAYS) + H
    idx = np.arange(start, n - H)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])]
    yb = y[idx]
    BENCH = L.bench_mean(y, idx)
    sst = ((yb - BENCH) ** 2).sum()
    print(f"target {TARGET}, {len(idx)} test days, {len(peers)} foreign peers, "
          f"{D.index[idx[0]].date()} to {D.index[idx[-1]].date()}")

    def r2(f):
        return 1 - ((yb - f) ** 2).sum() / sst

    print("\n" + "=" * 92)
    print("A.  THE TWO PIECES OF WHAT DELTA COSTS")
    print("=" * 92)
    print("Every arm shares the features, the estimator, the refit schedule and the")
    print("test days. The FEATURE-ONLY arm is infeasible under this paper's own")
    print("information set - training to t - h needs the stretch of the target's own")
    print("series the delay withholds - and exists to price one effect, never as a")
    print("forecast this paper reports.\n")

    f_own0 = walk(own, P, y, idx, 0, False, False)
    s0 = r2(f_own0)
    print(f"{'delta':>6}{'S_own(0)':>11}{'feature only':>14}{'outage':>10}"
          f"{'total':>10}{'feature':>10}{'vintage':>10}{'vintage share':>15}")
    rows = []
    for d in DELAYS[1:]:
        s_feat = r2(walk(own, P, y, idx, d, False, True))
        s_out = r2(walk(own, P, y, idx, d, False, False))
        tot, feat, vint = s0 - s_out, s0 - s_feat, s_feat - s_out
        share = vint / tot if abs(tot) > 1e-12 else float("nan")
        rows.append((d, s_feat, s_out, tot, feat, vint, share))
        print(f"{d:>6}{s0:>11.4f}{s_feat:>14.4f}{s_out:>10.4f}"
              f"{tot:>10.4f}{feat:>10.4f}{vint:>10.4f}{share:>14.1%}")

    shares = [r[6] for r in rows]
    print(f"\n  the estimation vintage accounts for {min(shares):.1%} to "
          f"{max(shares):.1%} of what the delay costs")
    if max(shares) < 0.25:
        print("  So the outage is mostly a stale INPUT. Freezing the coefficients,")
        print("  the penalty and the training window costs a minor share of the")
        print("  damage at every delay, and R(delta) divides by a denominator that")
        print("  is predominantly information rather than estimation. The naming")
        print("  matters and the measurement does not change much.")
    else:
        print("  So a material share of what the delay costs is the forecaster's")
        print("  inability to re-estimate, not the staleness of the input. The")
        print("  paper must say that R(delta) is the share of an OUTAGE repaired,")
        print("  and must not be read as the share of stale-input damage repaired.")

    print("\n" + "=" * 92)
    print("B.  DOES THE SUBSTITUTION RATE CARE WHICH ONE IT IS?")
    print("=" * 92)
    print("R(delta) recomputed inside the feature-only arm, with the cross-sectional")
    print("model given the same cut as the domestic one so the comparison stays")
    print("internal. If the two rates agree, the headline is a statement about")
    print("stale information and not about frozen estimation.\n")
    print(f"{'delta':>6}{'R outage':>11}{'R feature-only':>16}{'difference':>13}")
    diffs = []
    for d in DELAYS[1:]:
        so_o = r2(walk(own, P, y, idx, d, False, False))
        sc_o = r2(walk(own, P, y, idx, d, True, False))
        so_f = r2(walk(own, P, y, idx, d, False, True))
        sc_f = r2(walk(own, P, y, idx, d, True, True))
        r_o = (sc_o - so_o) / (s0 - so_o) if abs(s0 - so_o) > 1e-12 else np.nan
        r_f = (sc_f - so_f) / (s0 - so_f) if abs(s0 - so_f) > 1e-12 else np.nan
        diffs.append(r_o - r_f)
        print(f"{d:>6}{r_o:>11.1%}{r_f:>16.1%}{r_o - r_f:>+13.1%}")
    worst = max(abs(x) for x in diffs)
    print(f"\n  largest difference between the two rates: {worst:.1%}")
    if worst < 0.05:
        print("  The rate does not distinguish the two designs. Whatever delta is")
        print("  called, the share of the damage that breadth repairs is the same,")
        print("  so the naming is a matter of accuracy rather than of result.")
    else:
        print("  The rate DOES distinguish them, so the paper's estimand is the")
        print("  outage rate specifically, and every statement of it must say so.")

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    print("  The design freezes the domestic information system for delta days and")
    print("  the paper now names it that way. This file prices the part of that")
    print("  freeze which is not the input, so a reader can see how much of")
    print("  R(delta)'s denominator is information and how much is estimation.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
