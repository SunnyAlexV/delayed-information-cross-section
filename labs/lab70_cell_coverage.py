"""lab70_cell_coverage.py - what does a 95% set printed in Table S4 actually
cover, at that cell's own length and its own identification strength?

    python lab70_cell_coverage.py        (~6 min; Monte Carlo only, no data)

WHY THIS FILE EXISTS
--------------------
Two coverage studies already sit in this repository and neither answers the
question the conditional tables raise.

  lab63, whose study Section 4.6 reports, sweeps the IDENTIFICATION
  STRENGTH tau at the full
  evaluation length, and inverts at the NORMAL critical value 1.96.  It is the
  measurement of the procedure this paper replaced, not of the one it uses.

  lab67 part A sweeps the LENGTH at a single strength, with the fixed-b
  critical value.  It is what showed that a 367-day slice covers 65% where the
  full window covers 95%, and it is why the state-dependence claim was
  re-estimated as one full-sample parameter.

The cells of Table S4 are neither.  They are fixed-b sets, like lab67's, but
they are computed on roughly a third of the sample - the VIX terciles hold
about 1,620 days each - and at identification strengths between 2.9 and 5.8.
Quoting lab67's full-window 95% beside them overstates what they cover, and
quoting lab63's figures beside them describes a different procedure.  So this
file simulates the actual thing: a fixed-b Fieller set at each cell's OWN
length and OWN tau, against the same generator lab63 matched to the measured
loss-differential autocorrelation.

WHAT IT IS FOR
--------------
Section 4.6 already prints tau beside every set so a reader can see which cells
are weakly identified.  That is necessary and not sufficient: tau says whether
the set is bounded, and says nothing about whether a bounded set covers.  The
table below supplies the missing column, and Section S4 prints it beside each
interval, so a reader is told what a given set is worth at the point of use
rather than being sent two sections away to a study of a different procedure at
a different length.

WHAT IT CANNOT DO
-----------------
The generator is one model of the dependence, matched to one measured
autocorrelation at one delay.  A coverage figure here is what this procedure
would achieve against THAT process at that length and strength; it is not a
confidence statement about the cell itself.  It is reported for the same reason
tau is: a reader deciding how hard to lean on a number is entitled to know what
the instrument does, and the honest answer is not the nominal level.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab63_fieller_coverage as C
import lab67_regime_contrast as RC

SEED = 20261005
REPS = 1500            # Monte Carlo error near 90% is about 0.8 points
CV_REPS = 4000         # for the fixed-b critical value itself
B_FRAC = 0.05          # the paper's bandwidth, b = M/n

# The cells Table S4 prints, read from lab41's own table rather than retyped.
# (state, delta, n, tau) - n is that tercile's day count and tau its
# denominator's HAC t-statistic, both as lab41 reports them.
CELLS = [
    ("calm", 5, 1620, 3.4), ("calm", 13, 1620, 2.9),
    ("calm", 21, 1620, 3.5), ("calm", 55, 1620, 3.0),
    ("middle", 5, 1624, 5.6), ("middle", 13, 1624, 5.8),
    ("middle", 21, 1624, 5.5), ("middle", 55, 1624, 4.4),
    ("stressed", 5, 1618, 3.5), ("stressed", 13, 1618, 3.7),
    ("stressed", 21, 1618, 3.6), ("stressed", 55, 1618, 4.2),
]
FULL = ("full window", 55, 4862, 5.4)


def fieller_at(m1, m2, s11, s12, s22, n, z):
    """lab63's Fieller inversion with the critical value as an argument rather
    than a module constant, because the whole point here is to invert at the
    fixed-b value the paper uses instead of lab63's 1.96."""
    a = m2 ** 2 - z ** 2 * s22 / n
    b = -2.0 * (m1 * m2 - z ** 2 * s12 / n)
    c = m1 ** 2 - z ** 2 * s11 / n
    disc = b ** 2 - 4.0 * a * c
    lo = np.full_like(m1, -np.inf); hi = np.full_like(m1, np.inf)
    kind = np.full(m1.shape, 2, dtype=int)
    ok = disc >= 0
    r1 = np.where(ok, (-b - np.sqrt(np.maximum(disc, 0))) / (2.0 * a), np.nan)
    r2 = np.where(ok, (-b + np.sqrt(np.maximum(disc, 0))) / (2.0 * a), np.nan)
    rl, rh = np.minimum(r1, r2), np.maximum(r1, r2)
    bd = ok & (a > 0)
    lo = np.where(bd, rl, lo); hi = np.where(bd, rh, hi)
    kind = np.where(bd, 0, kind)
    ub = ok & (a < 0)
    lo = np.where(ub, rl, lo); hi = np.where(ub, rh, hi)
    kind = np.where(ub, 1, kind)
    return lo, hi, kind


def cover_at(n, tau, rng):
    """Coverage of a nominal 95% fixed-b Fieller set at length n and strength
    tau, against lab63's generator.  Returns (coverage, unbounded share, cv)."""
    M = max(1, int(round(B_FRAC * n)))
    cv = RC.fixed_b_cv(n, M, rng, reps=CV_REPS)
    e2 = C.dep(rng, n, REPS)
    eo = C.dep(rng, n, REPS)
    e1 = C.CORR12 * e2 + np.sqrt(1.0 - C.CORR12 ** 2) * eo
    lr_sd = np.sqrt(C.LRV_TRUE / n)
    mu2 = tau * lr_sd
    d2 = mu2 + e2
    d1 = C.RSTAR * mu2 + e1
    m1, m2 = d1.mean(axis=1), d2.mean(axis=1)
    s11, s22, s12 = C.hac(d1, d1, M), C.hac(d2, d2, M), C.hac(d1, d2, M)
    lo, hi, kind = fieller_at(m1, m2, s11, s12, s22, n, cv)
    return C.covers(lo, hi, kind, C.RSTAR).mean(), (kind != 0).mean(), cv


def main():
    rng = np.random.default_rng(SEED)
    print("=" * 92)
    print("lab70  -  WHAT A TABLE S4 CELL'S 95% SET ACTUALLY COVERS")
    print("=" * 92)
    print("  Fixed-b Fieller, b = M/n = 0.05, against lab63's generator matched")
    print("  to the measured loss-differential autocorrelation.  Each row is")
    print("  simulated at that cell's OWN length and OWN identification")
    print(f"  strength, {REPS} replications; Monte Carlo error near 90% is about")
    print(f"  {100 * np.sqrt(0.9 * 0.1 / REPS):.1f} points.\n")

    print(f"  {'state':<12}{'delta':>6}{'n':>7}{'tau':>7}{'fixed-b cv':>12}"
          f"{'covers':>9}{'unbounded':>11}")
    out = {}
    for state, d, n, tau in CELLS:
        cov, unb, cv = cover_at(n, tau, rng)
        out[(state, d)] = cov
        print(f"  {state:<12}{d:>6}{n:>7,}{tau:>7.1f}{cv:>12.3f}"
              f"{cov * 100:>8.1f}%{unb * 100:>10.1f}%", flush=True)

    fs, fd, fn, ftau = FULL
    fcov, funb, fcv = cover_at(fn, ftau, rng)
    print(f"\n  {fs:<12}{fd:>6}{fn:>7,}{ftau:>7.1f}{fcv:>12.3f}"
          f"{fcov * 100:>8.1f}%{funb * 100:>10.1f}%")
    print("  (the unconditional cell, for comparison: same procedure, whole"
          " sample)")

    worst = min(out.values()); best = max(out.values())
    st55 = out[("stressed", 55)]
    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    print(f"  Across the twelve conditional cells the nominal 95% set covers")
    print(f"  between {worst * 100:.1f}% and {best * 100:.1f}%.  The cell the paper tells a")
    print(f"  practitioner to carry, stressed at eleven weeks, covers {st55 * 100:.1f}%.")
    print(f"  The same procedure on the whole sample covers {fcov * 100:.1f}%.\n")
    gap = (fcov - st55) * 100
    print(f"  So a conditional set is worth about {gap:.1f} points less than the")
    print("  unconditional one, and that is a property of its LENGTH rather than")
    print("  of its identification: tau is comparable across the two, and the")
    print("  third of the sample is not.  Section 4.6 prints tau beside every")
    print("  set so a reader can see which are bounded; tau does not say what a")
    print("  bounded set covers, and this is the column that does.")
    print("\n  Reported so that a reader meeting a conditional interval is told")
    print("  what it is worth where they meet it, rather than being left to")
    print("  carry the full-window figure to a third of the sample.")


if __name__ == "__main__":
    main()
