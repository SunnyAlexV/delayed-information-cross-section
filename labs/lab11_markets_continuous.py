"""
lab11_markets_continuous.py - the five-market check, on the target the paper uses.

Imports lab05_robustness; keep both in labs/.  Runtime about four minutes.

WHY
---
Section 8 of the paper compares five target markets, and it does so on the
BINARY target, for comparability with the companion note.  Everything else in
the paper uses the continuous one, on the companion note's own argument that a
median split throws away the magnitude information a cross-sectional model
would use.  Reporting the headline on one target and the external-validity
check on another is the kind of inconsistency a referee is right to flag: if
the split discards what matters, the market comparison is running on a weaker
measurement than the rest of the paper.

So repeat it on the continuous target, with the same two losses, and put the
substitution rate on the same footing as Table 1.

Intervals bootstrap the WHOLE ratio - numerator and denominator recomputed on
the same resampled days - which is the correction lab10 applies to Table 1.
Reporting a narrower interval here than there would be worse than reporting
none.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".")
import lab05_robustness as L

SEED = 20260912
DELAYS = [0, 5, 13, 21, 55]
TARGETS = ["SPX", "FTSE", "DAX", "N225", "HSI"]
N_BOOT, BLK = 1500, 10


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}\n")
    print("=" * 86)
    print("FIVE TARGET MARKETS, CONTINUOUS TARGET (log realised variance, h = 5)")
    print("=" * 86)
    print("Same general timing rule as lab05: a peer counts only if its close precedes")
    print("the target's, otherwise it contributes the previous session.\n")

    summary = {}
    for tgt in TARGETS:
        rng = np.random.default_rng(SEED)
        try:
            D, peers, lag, idx, yb, fitrun, a = L.run_target(
                folder, tgt, DELAYS, rng, continuous=True)
        except Exception as e:
            print(f"{tgt}: skipped ({e})")
            continue
        same = [p for p in peers if lag[p] == 0]
        prev = [p for p in peers if lag[p] == 1]
        n = len(idx)
        yv = np.exp(yb)

        # cache per-day losses for every delay, both models, both losses
        cache = {}
        for d in DELAYS:
            po, pc = fitrun(d, False), fitrun(d, True)
            ho, hc = np.exp(po), np.exp(pc)
            cache[d] = {
                "sq_own": (yb - po) ** 2, "sq_cross": (yb - pc) ** 2,
                "ql_own": yv / ho - np.log(yv / ho) - 1,
                "ql_cross": yv / hc - np.log(yv / hc) - 1,
            }
        base_sq = (yb - yb.mean()) ** 2
        m = yv.mean(); base_ql = yv / m - np.log(yv / m) - 1

        def skill(d, loss, model, s=None):
            v = cache[d][f"{loss}_{model}"]
            b = base_sq if loss == "sq" else base_ql
            return 1 - (v.sum() / b.sum() if s is None else v[s].sum() / b[s].sum())

        rb = np.random.default_rng(SEED + 11)
        st = rb.integers(0, n - BLK + 1, size=(N_BOOT, int(np.ceil(n / BLK))))
        offs = np.arange(BLK)
        S = [(st[i][:, None] + offs).ravel()[:n] for i in range(N_BOOT)]

        print(f"--- target {tgt} ({L.CLOSE_UTC[tgt]:.1f} UTC), {n} test days, "
              f"{len(same)} same-day peers {same}")
        print(f"    {'delta':>6}{'R2 own':>9}{'R2 cross':>10}{'R(d) R2':>24}"
              f"{'R(d) QLIKE':>24}")
        for d in DELAYS[1:]:
            row = (f"    {d:>6}{skill(d,'sq','own'):>9.4f}"
                   f"{skill(d,'sq','cross'):>10.4f}")
            for loss in ("sq", "ql"):
                pt = ((skill(d, loss, "cross") - skill(d, loss, "own")) /
                      (skill(0, loss, "own") - skill(d, loss, "own")))
                bs = []
                for s in S:
                    den = skill(0, loss, "own", s) - skill(d, loss, "own", s)
                    if abs(den) > 1e-9:
                        bs.append((skill(d, loss, "cross", s)
                                   - skill(d, loss, "own", s)) / den)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                row += f"{pt:>8.0%} [{lo:>4.0%},{hi:>4.0%}]".rjust(24)
            print(row)
            if d == 55:
                summary[tgt] = (len(same), skill(0, 'sq', 'own'),
                                skill(55, 'sq', 'own'), skill(55, 'sq', 'cross'))
        print()

    print("=" * 86)
    print("SUMMARY AT delta = 55, ORDERED BY SAME-DAY PEER COUNT")
    print("=" * 86)
    print(f"{'target':>8}{'same-day peers':>16}{'R2 own d=0':>13}"
          f"{'R2 own d=55':>13}{'R2 cross d=55':>15}{'R(55)':>9}")
    for t in sorted(summary, key=lambda t: -summary[t][0]):
        k, s0, s55, c55 = summary[t]
        print(f"{t:>8}{k:>16}{s0:>13.4f}{s55:>13.4f}{c55:>15.4f}"
              f"{(c55-s55)/(s0-s55):>9.0%}")
    print("\nThe Nikkei is the case that carries information: it closes before every")
    print("other market in the sample, so it has NO same-day foreign observation and")
    print("should recover least if the mechanism is what we claim.  Past that one")
    print("contrast, five markets cannot resolve an ordering.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
