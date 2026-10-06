"""joint_null.py - a joint resampling test for the count of nominally
significant cells across the secondary-analysis universe.

WHAT THIS REPLACES

Table S30 compares an observed count of 77 nominally significant cells against
"0.05n", i.e. 10.7 expected, and concludes that this "is not what noise
produces". The expectation is right - expectations are linear and survive any
dependence - but the inference needs the VARIANCE of the count, and under the
cross-cell dependence this paper insists on everywhere else that variance is far
above binomial. No variance, no simulated null, and no omnibus test is reported.

WHAT WAS WRONG WITH MY FIRST ATTEMPT

I measured a variance inflation of 9.5x on an eight-cell delay grid and carried
both it AND a "mean inflation" across to the 214-cell pool, producing a bogus
5.3 sigma. Two errors:

  1. The eight-cell delay grid and the 214-cell pool have different dependence
     GRAPHS. The pool mixes compressed-factor grids, optioned and unoptioned
     targets, nonlinear models, a third implied-volatility series, several
     horizons, targets, model classes and delay grids. Some of those cells are
     strongly coupled and some are nearly independent. An inflation factor
     measured on one graph licenses nothing about the other.

  2. The null MEAN is not a transferable quantity and should not be inflated at
     all. Under correctly sized marginal tests each null cell rejects at about
     5% whatever the correlation, so the null mean stays near 0.05n; correlation
     inflates the variance and the upper tail. Measured here: fixing the
     marginal rule moved the null mean from 1.01 to 0.70 against a theoretical
     0.40 and the inflation from 10.8x to 7.0x. The residual mean excess is
     marginal size distortion and cell heterogeneity, not correlation.

THE PROCEDURE

  1. Fix the family before looking at the revised outcome. A cell enters by
     being reported, not by surviving.
  2. Impose the null cell by cell (demean), preserving each cell's own
     dependence and the cross-cell alignment.
  3. Group cells by CALENDAR. Cells sharing an evaluation period are resampled
     with ONE set of block starts, so their joint dependence is carried into the
     replication. Groups on different calendars - daily equities, monthly or
     quarterly housing - are resampled independently within the same
     replication and then combined. Calendars are never concatenated and no cell
     is pretended onto another's time index.
  4. Recompute the SAME statistic and the SAME nominal 5% rule in every cell.
  5. Record C*(b), the count of nominally significant cells in replication b.
  6. Report the empirical null mean, sd, 90th/95th/99th percentiles and

         p_joint = (1 + #{b : C*(b) >= C_obs}) / (B + 1).

  7. BH and Bonferroni stay, for what they control - false discovery rate and
     family-wise error. They are not substitutes for this global count test and
     this is not a substitute for them.

TO RUN IT ON THE REAL 214 CELLS this harness needs, per cell: the per-day (or
per-period) statistic series, its calendar group, its bandwidth, and its
critical value. Those live across the repository's labs and are not assembled
anywhere today; assembling them is the implementation job this specifies.
"""
import math
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np


@dataclass
class Cell:
    name: str
    group: str                 # calendar group; cells in a group share an index
    x: np.ndarray              # the per-period series the statistic is built on
    M: int                     # HAC bandwidth for this cell
    cv: float                  # critical value for this cell's own rule
    stat: Callable = None      # defaults to a HAC t-statistic of the mean


def hac_se(x, M):
    n = len(x)
    xc = x - x.mean()
    s = np.dot(xc, xc) / n
    for j in range(1, min(M, n - 1) + 1):
        s += 2 * (1 - j / (M + 1)) * np.dot(xc[j:], xc[:-j]) / n
    return math.sqrt(max(s, 1e-18) / n)


def t_of(x, M):
    return x.mean() / hac_se(x, M)


def fixed_b_cv(n, M, rng, reps=3000):
    """The 95% point of |t| under iid Gaussian data, at this exact (n, M).

    1.96 is the limit as M/n -> 0.  At a bandwidth that is a non-vanishing
    fraction of the sample the Bartlett t-statistic is not standard normal, and
    the right reference point depends on b = M/n and, at short n, on n itself -
    which is the correction lab67 part A measures the cost of and lab68 applies
    as its second marginal rule.  Simulated rather than taken from a table
    because the tabulated fixed-b values are limits in n, and several of the
    samples here are not long.
    """
    ts = np.empty(reps)
    for i in range(reps):
        z = rng.standard_normal(n)
        zc = z - z.mean()
        s = np.dot(zc, zc) / n
        for j in range(1, M + 1):
            s += 2 * (1 - j / (M + 1)) * np.dot(zc[j:], zc[:-j]) / n
        ts[i] = abs(z.mean()) / math.sqrt(max(s, 1e-18) / n)
    return float(np.quantile(ts, 0.95))


def joint_null(cells: List[Cell], B=2000, block=None, seed=0):
    """The empirical null distribution of the count, by joint block resampling."""
    rng = np.random.default_rng(seed)
    groups: Dict[str, List[int]] = {}
    for i, c in enumerate(cells):
        groups.setdefault(c.group, []).append(i)
    # every cell in a group must share one index length, or they cannot share
    # block starts; that is the whole point of the grouping
    for g, ids in groups.items():
        L = {len(cells[i].x) for i in ids}
        if len(L) != 1:
            raise SystemExit(f"calendar group {g!r} holds cells of lengths {L}; "
                             f"cells sharing a group must share an index")
    null = [c.x - c.x.mean() for c in cells]       # impose the null, keep dependence
    obs = sum(1 for i, c in enumerate(cells) if abs(t_of(c.x, c.M)) > c.cv)
    counts = np.empty(B, int)
    for b in range(B):
        c_b = 0
        for g, ids in groups.items():
            n = len(cells[ids[0]].x)
            L = block or max(20, int(round(n ** (1 / 3) * 4)))
            L = min(L, max(2, n // 4))
            # ceil and an inclusive upper end, for the reason lab68 records:
            # floor makes the replication shorter than the sample and an
            # exclusive upper end never draws the last observation.
            nb = max(1, -(-n // L))
            starts = rng.integers(0, max(1, n - L + 1), nb)  # ONE per group
            for i in ids:
                y = np.concatenate([null[i][s:s + L]
                                    for s in starts])[:n]
                if abs(t_of(y, cells[i].M)) > cells[i].cv:
                    c_b += 1
        counts[b] = c_b
    p = (1 + int((counts >= obs).sum())) / (B + 1)
    return obs, counts, p


def report(cells, B=2000, block=None, seed=0, label=""):
    obs, counts, p = joint_null(cells, B=B, block=block, seed=seed)
    K = len(cells)
    gs = sorted({c.group for c in cells})
    print(f"\n{label}")
    print(f"  {K} cells in {len(gs)} calendar group(s): "
          f"{', '.join(f'{g} ({sum(1 for c in cells if c.group==g)})' for g in gs)}")
    print(f"  observed significant cells           {obs}")
    print(f"  0.05K, the quantity Table S30 uses   {0.05*K:.1f}")
    print(f"  empirical null mean                  {counts.mean():.2f}")
    print(f"  empirical null sd                    {counts.std():.2f}"
          f"   (binomial would be {math.sqrt(K*0.05*0.95):.2f})")
    print(f"  variance inflation over binomial     {counts.var()/(K*0.05*0.95):.1f}x")
    print(f"  null percentiles  90th {np.quantile(counts,0.90):.0f}"
          f"   95th {np.quantile(counts,0.95):.0f}"
          f"   99th {np.quantile(counts,0.99):.0f}")
    print(f"  p_joint = (1 + #(C* >= {obs})) / (B+1)  =  {p:.4f}")
    return obs, counts, p


if __name__ == "__main__":
    # A DEMONSTRATION that the grouping does what it claims, on synthetic data
    # only.  It is not the paper's cells and makes no claim about them; lab68
    # does the real thing on the recorded registry.  Self-contained on purpose:
    # a module whose demo reaches outside the repository is a module whose demo
    # does not run for anybody else.
    rng = np.random.default_rng(7)
    n, K = 1200, 10
    # one common AR(1) factor plus idiosyncratic noise, so the cells in a group
    # are genuinely dependent and the binomial variance is genuinely wrong
    def ar1(n, rho, rng):
        e = rng.standard_normal(n + 200)
        x = np.zeros(n + 200)
        for t in range(1, n + 200):
            x[t] = rho * x[t - 1] + e[t]
        return x[200:]
    f = ar1(n, 0.9, rng)
    cells = [Cell(f"daily cell {k}", "daily", f + 1.5 * ar1(n, 0.9, rng), 40,
                  2.05) for k in range(K)]
    # a second calendar group on its own, shorter index: resampled with its own
    # block starts inside the same replication, never concatenated with the
    # first
    m = n // 21
    g = ar1(m, 0.6, rng)
    cells += [Cell(f"monthly cell {k}", "monthly", g + 1.5 * ar1(m, 0.6, rng),
                   4, 2.05) for k in range(4)]
    obs, counts, p = report(cells, B=1500,
                            label="DEMONSTRATION - two calendar groups, "
                                  "resampled independently within each "
                                  "replication")
    print("\n  The daily cells share one set of block starts; the monthly cells")
    print("  share a different set, drawn on their own index.  Nothing is")
    print("  concatenated across calendars.")
    print(f"\n  Binomial sd would be "
          f"{math.sqrt(len(cells) * 0.05 * 0.95):.2f} against the "
          f"{counts.std():.2f} measured above, which is the entire reason")
    print("  a count cannot be judged against 0.05n on its own.")
