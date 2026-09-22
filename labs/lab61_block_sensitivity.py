"""
lab61_block_sensitivity.py - does the companion note's inference depend on the
block length it happened to pick?

Imports lab02_delay_curve and lab24_proper_scores; keep all three in labs/.
Runtime about three minutes.

THE OBJECTION
-------------
The note's intervals come from a moving-block bootstrap, and a moving-block
bootstrap has one free parameter: the block.  Too short and neighbouring blocks
are resampled as though independent when they are not, which narrows every
interval and manufactures significance.  Too long and there are too few
distinct blocks, which widens them and buries it.  Neither error announces
itself, and the note reported one block length and no sensitivity.

The number itself had drifted.  The note's methods section described a block of
2h = 10 while every table in it was produced at 8h = 40, because lab02's
constant was corrected and the prose was not.  That is fixed in the note; this
file answers the question the correction raises, which is what the choice was
ever worth.

WHAT IS VARIED, AND AGAINST WHAT
--------------------------------
Three moving-block lengths, spanning the range any reasonable reader would
defend:

    L = 10   2h.  Covers the overlap in the five-day target and nothing else.
    L = 20   4h.  A middle choice.
    L = 40   8h.  The note's, chosen to outlast the dependence the companion
             paper's bandwidth measurement finds at lag 38.

and, beside them, the STATIONARY bootstrap of Politis and Romano (1994).  That
is not a fourth block length but a different object: block lengths are drawn
from a geometric distribution with mean 1/p, so the resampled series is
stationary, and the answer does not hinge on one integer.  Each stationary run
is matched to a moving-block run by expected length, 1/p = L, so the comparison
isolates the SHAPE of the block distribution from its mean.

WHAT IS TESTED
--------------
The two claims the note actually rests on.

    parity      the paired accuracy difference, har-YZ minus persist-YZ, at
                every delay.  The note says no interval excludes zero.
    separation  the paired AUC difference, same pair, same days.  The note says
                five of ten delays separate, all in the threshold rule's
                favour.

A claim that survives every block length and both bootstrap families is a claim
about the data.  One that appears only at L = 10 is a claim about the block.
The verdict below is computed, not written in advance: if the separation count
collapses at L = 40 the note has to say so, and this file says it first.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab02_delay_curve as L
import lab24_proper_scores as S

SEED = 20260920
DRAWS = 2000
AUC_DRAWS = 800          # AUC is recomputed inside each draw, so it gets fewer
BLOCKS = [10, 20, 40]
NOTE_BLOCK = L.BLOCK     # what the note reports, read rather than repeated


# ----------------------------------------------------------------------
# the two resampling schemes, written so the only difference is the block law
# ----------------------------------------------------------------------
def moving_index(n, block, rng):
    """One moving-block resample of 0..n-1, wrapped to length n."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=nb)
    return (starts[:, None] + np.arange(block)).ravel()[:n]


def stationary_index(n, mean_block, rng):
    """One stationary-bootstrap resample (Politis and Romano 1994).

    Block lengths are geometric with mean `mean_block`, and each block starts
    at a uniform position and runs CIRCULARLY, which is what makes the
    resampled series stationary.  The moving-block scheme above is the same
    construction with the geometric law replaced by a point mass, so a
    difference between the two columns below is the block law and nothing
    else.
    """
    p = 1.0 / mean_block
    out = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        ln = int(rng.geometric(p))
        ln = min(ln, n - i)
        out[i:i + ln] = (start + np.arange(ln)) % n
        i += ln
    return out


def ci_mean(diff, block, rng, draws, stationary):
    """Percentile interval for the mean of a paired per-day difference."""
    n = len(diff)
    out = np.empty(draws)
    for i in range(draws):
        s = (stationary_index(n, block, rng) if stationary
             else moving_index(n, block, rng))
        out[i] = diff[s].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return diff.mean(), lo, hi, (lo > 0 or hi < 0)


def ci_auc(y, sa, sb, block, rng, draws, stationary):
    """Percentile interval for a paired AUC difference.

    AUC is not an average of per-day contributions, so it is recomputed inside
    every draw on the resampled days.  Both forecasters are scored on the SAME
    resampled days, which is what makes the difference paired.
    """
    n = len(y)
    out = np.empty(draws)
    for i in range(draws):
        s = (stationary_index(n, block, rng) if stationary
             else moving_index(n, block, rng))
        out[i] = S.auc(y[s], sa[s]) - S.auc(y[s], sb[s])
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return S.auc(y, sa) - S.auc(y, sb), lo, hi, (lo > 0 or hi < 0)


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

    res, ybar, idx = S.run(y, feats, Ms, proxies)
    print(f"test window: {len(ybar)} days;  realised base rate {ybar.mean():.3f}")
    print(f"the note's block: {NOTE_BLOCK};  lengths tried: "
          f"{', '.join(str(b) for b in BLOCKS)}")
    print(f"{DRAWS} draws for accuracy, {AUC_DRAWS} for AUC\n")
    if NOTE_BLOCK not in BLOCKS:
        print(f"  WARNING: the note's block {NOTE_BLOCK} is not among the lengths")
        print("  tried here, so this file no longer brackets what the note does.\n")

    # per-day paired inputs, built once so every column below scores the same
    # forecasts on the same days and only the resampling differs
    pair = {}
    for d in L.DELAYS:
        ra, rb = res[("har-YZ", d)], res[("persist-YZ", d)]
        pair[d] = ((ra["lab"] == ybar).astype(float)
                   - (rb["lab"] == ybar).astype(float),
                   ra["s"], rb["s"])

    print("=" * 100)
    print("A.  PARITY ON ACCURACY, ACROSS BLOCK LENGTHS AND BOTH BOOTSTRAPS")
    print("=" * 100)
    print("har-YZ minus persist-YZ, in percentage points.  The note's claim is")
    print("that no interval excludes zero.  'mb' is the moving block, 'sb' the")
    print("stationary bootstrap at the same EXPECTED block length.\n")
    head = f"{'delta':>6}{'diff':>9}"
    for b in BLOCKS:
        head += f"{'mb ' + str(b):>19}{'sb ' + str(b):>19}"
    print(head)
    acc_sig = {(b, k): [] for b in BLOCKS for k in ("mb", "sb")}
    for d in L.DELAYS:
        diff = pair[d][0]
        line = f"{d:>6}{diff.mean() * 100:>+8.2f}"
        for b in BLOCKS:
            for kind, st in (("mb", False), ("sb", True)):
                rng = np.random.default_rng(SEED + 97 * b + d + (7 if st else 0))
                _, lo, hi, sig = ci_mean(diff, b, rng, DRAWS, st)
                if sig:
                    acc_sig[(b, kind)].append(d)
                line += f"[{lo * 100:>+6.2f},{hi * 100:>+6.2f}]{'*' if sig else ' '}".rjust(19)
        print(line)
    print()
    for b in BLOCKS:
        for kind in ("mb", "sb"):
            ds = acc_sig[(b, kind)]
            print(f"  {kind} block {b:>2}: accuracy differences excluding zero: "
                  f"{len(ds)} of {len(L.DELAYS)}" + (f"  {ds}" if ds else ""))
    worst_acc = max(len(v) for v in acc_sig.values())
    print()
    if worst_acc == 0:
        print("  Parity holds under every block length and both bootstraps. The")
        print("  note's accuracy claim does not depend on the resampling choice.")
    else:
        print(f"  Parity FAILS somewhere: up to {worst_acc} delay(s) separate on")
        print("  accuracy under some resampling scheme. The note claims none do and")
        print("  must be corrected.")

    print("\n" + "=" * 100)
    print("B.  THE AUC SEPARATION, ACROSS BLOCK LENGTHS AND BOTH BOOTSTRAPS")
    print("=" * 100)
    print("Same pair, same days, AUC recomputed inside every draw. The note")
    print("claims five of ten delays separate, all favouring the threshold rule.")
    print("A count that shrinks as the block grows is a count that was borrowing")
    print("from a block too short to hold the dependence.\n")
    print(head)
    auc_sig = {(b, k): [] for b in BLOCKS for k in ("mb", "sb")}
    auc_dir = {(b, k): [] for b in BLOCKS for k in ("mb", "sb")}
    for d in L.DELAYS:
        _, sa, sb_ = pair[d]
        line = f"{d:>6}"
        pt = S.auc(ybar, sa) - S.auc(ybar, sb_)
        line += f"{pt:>+9.4f}"
        for b in BLOCKS:
            for kind, st in (("mb", False), ("sb", True)):
                rng = np.random.default_rng(SEED + 131 * b + d + (11 if st else 0))
                _, lo, hi, sig = ci_auc(ybar, sa, sb_, b, rng, AUC_DRAWS, st)
                if sig:
                    auc_sig[(b, kind)].append(d)
                    auc_dir[(b, kind)].append(pt < 0)
                line += f"[{lo:>+7.4f},{hi:>+7.4f}]{'*' if sig else ' '}".rjust(19)
        print(line)
    print()
    for b in BLOCKS:
        for kind in ("mb", "sb"):
            ds = auc_sig[(b, kind)]
            allthr = all(auc_dir[(b, kind)]) if ds else True
            print(f"  {kind} block {b:>2}: AUC differences excluding zero: "
                  f"{len(ds)} of {len(L.DELAYS)}" + (f"  {ds}" if ds else "")
                  + ("  all favour the threshold rule" if ds and allthr
                     else "  NOT all favour the threshold rule" if ds else ""))

    counts = {k: len(v) for k, v in auc_sig.items()}
    at40 = max(counts[(40, "mb")], counts[(40, "sb")])
    at10 = max(counts[(10, "mb")], counts[(10, "sb")])
    print()
    print(f"  separating delays at block 10: {at10};  at block 40: {at40}")
    if at40 == 0:
        print("  The separation does NOT survive the block the note uses. Whatever")
        print("  the note says about AUC must be restated as a point estimate with")
        print("  no interval evidence behind it.")
    elif at40 < at10:
        print("  The separation is weaker at the longer block, which is the expected")
        print("  direction: a short block understates the dependence and so overstates")
        print("  significance. The note must quote the count at the block it uses, and")
        print("  should say that the shorter block would have given more.")
    else:
        print("  The separation does not weaken as the block grows, so it is not an")
        print("  artefact of resampling dependent days as independent ones.")

    directions = [d for k, v in auc_dir.items() for d in v]
    print()
    print(f"  of {len(directions)} significant AUC cells across all six schemes, "
          f"{sum(directions)} favour the threshold rule")
    if directions and all(directions):
        print("  The direction is unanimous across every block length and both")
        print("  bootstrap families, which is the part of the note's claim that")
        print("  does not depend on the resampling at all.")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print("  The block length is a choice and this file prices it. The note now")
    print("  reports the block it actually used, quotes its counts at that block,")
    print("  and points here for what a shorter or a random-length block would")
    print("  have given instead.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
