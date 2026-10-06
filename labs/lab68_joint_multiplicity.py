"""
lab68_joint_multiplicity.py - the omnibus test Table S30 never had.

THE DEFECT THIS REPLACES
------------------------
Table S30 sets an observed count of nominally significant secondary cells
against "0.05n" and concludes that the difference "is not what noise produces".
The expectation is right: expectations are linear and survive any dependence.
The inference is not, because it needs the VARIANCE of that count, and these
cells share evaluation days, training windows and nested delays.  No variance,
no simulated null, no test.

lab62 could not have supplied one.  It audits the grids by regex-scraping
z-statistics out of the other labs' printed text, and a scalar cannot be
resampled.  "0.05n" is not so much an oversight as the only null a scalar
admits.

The deeper reason is the one Section S37 states in its own words - "no
family-wise control was applied across them" - and it is that THE FAMILY WAS
NEVER WRITTEN DOWN.  You cannot correct across a family you have not
enumerated.

WHAT THIS LAB READS
-------------------
labs/cell_registry.py, and the five-line patch in each of the five labs that
emit reported cells, record every cell's per-day loss differential at the point
where that differential already exists, together with its calendar group, its
bandwidth and its critical value.  Recording is a side effect: no lab's printed
output changes, and REGISTRY_OFF=1 disables it so the reproduction run stays
byte for byte identical.  This lab reads that registry and does the test.

The registry is therefore a PRODUCT of the run, not an input to it.  Under
run_all.py the five source labs run before this one and leave it populated.  Run
on its own against an empty or partial registry, this lab stops and says which
labs to run first, rather than quietly testing a smaller family.

THE PROCEDURE, AS PRE-DECLARED
------------------------------
  1. The family is fixed before the revised outcome is looked at.  A cell enters
     by having been REPORTED, not by having survived.  The headline stays out of
     it, and so does the market-state ladder, which is tested on its own in
     lab67 with a statistic matched to its own claim.
  2. The null is imposed cell by cell, by demeaning, which leaves each cell's
     own serial dependence and the cross-cell alignment untouched.
  3. Cells are grouped by CALENDAR.  Cells sharing an evaluation period are
     resampled with ONE set of block starts, so their joint dependence is
     carried into the replication.  Groups are resampled independently within
     the same replication and the counts added.  Calendars are never
     concatenated and no cell is pretended onto another cell's time index.
  4. The SAME statistic and the SAME marginal rule are recomputed in every cell
     of every replication.
  5. C*(b) is the count of nominally significant cells in replication b, and

         p_joint = (1 + #{b : C*(b) >= C_obs}) / (B + 1).

  6. Benjamini-Hochberg and Bonferroni stay where they are, in lab62, for what
     they control - false discovery rate and family-wise error.  Neither is a
     substitute for a test of the global count, and this is not a substitute
     for either.

TWO MARGINAL RULES, NOT ONE
---------------------------
Rule A is the paper's own: the delay-scaled Bartlett bandwidth block_for(delta)
that the five labs actually pass, against a normal critical value of 1.96.

Rule B keeps that bandwidth and replaces the critical value with the fixed-b
95% point simulated at each cell's own (n, M), which is the right reference
distribution for a Bartlett t-statistic at a bandwidth that is a non-vanishing
fraction of the sample.  Rule B is the conservative reading and is reported
beside Rule A rather than in place of it: the pooled count is judged against a
null built with the SAME rule that produced the observed count, so each rule is
internally consistent and the two are not mixed.

WHAT I GOT WRONG FIRST, RECORDED BECAUSE THE ERROR IS INSTRUCTIVE
-----------------------------------------------------------------
I measured a variance inflation of 9.5x on an eight-cell delay grid and carried
both it and a "mean inflation" across to the pool, producing a 5.3-sigma repair.
Two errors.  The eight-cell grid and the pool have different dependence GRAPHS,
so a factor measured on one licenses nothing about the other.  And the null MEAN
is not a transferable quantity and should not be inflated at all: under a
correctly sized marginal rule each null cell rejects at about 5% whatever the
correlation, so the null mean stays near 0.05n, while correlation inflates the
variance and the upper tail.  Fixing the marginal rule on the test grid moved
its null mean from 1.01 to 0.70 against a theoretical 0.40 and its inflation
from 10.8x to 7.0x.  The residual mean excess is marginal size distortion and
cell heterogeneity, not correlation - which is exactly why Rule B is reported.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import cell_registry as REG
from joint_null import fixed_b_cv, hac_se

B = 2000
SEED = 0
CV_REPS = 40000      # per (n, M); at 4,000 the simulation noise was the result
# The five labs that record.  Named so a partial registry can say what is
# missing instead of silently testing a smaller family.
SOURCES = ("lab15_nonlinear_given_iv", "lab19_vstoxx_third_series",
           "lab51_foreign_options", "lab52_compressed_foreign_options",
           "lab53_no_options_targets")
EXPECT_CELLS = 214
EXPECT_GROUPS = 14   # 15 recorded, two of which are one calendar; see harvest()


# ----------------------------------------------------------------------
# a batched Bartlett HAC, and the check that it is the same one
# ----------------------------------------------------------------------
def hac_t_batch(Y, M):
    """HAC t-statistics of the row means of Y (K x n), Bartlett, bandwidth M.

    The same estimator as lab08's hac_se, computed for every row at once
    through the autocovariances, because the loop form is 214 cells x 2000
    replications x 80 lags away from being runnable.  Equality with the loop
    form is asserted below on the observed series, at every (n, M) the pool
    contains: a faster instrument that is not the same instrument would
    silently change what is being tested.
    """
    K, n = Y.shape
    mu = Y.mean(1)
    Yc = Y - mu[:, None]
    nfft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(Yc, n=nfft, axis=1)
    ac = np.fft.irfft(F * np.conj(F), n=nfft, axis=1)[:, :M + 1] / n
    w = 1.0 - np.arange(1, M + 1) / (M + 1)
    s = ac[:, 0] + 2.0 * (ac[:, 1:M + 1] * w).sum(1)
    return mu / np.sqrt(np.maximum(s, 1e-18) / n)


def fixed_b_cv_fast(n, M, rng, reps=CV_REPS, batches=10):
    """The fixed-b 95% point, through the batched HAC instead of a loop.

    joint_null.fixed_b_cv is the readable form and the one lab67 uses, and at
    n = 4,948 with M = 80 it is about four hours of Python per critical value,
    sixty of which are needed here.  This is the same quantity computed the same
    way, and _assert_same_fixed_b below holds the two against each other on
    identical draws.

    Returns the point AND its Monte Carlo standard error, from the spread of
    the per-batch quantiles, because a simulated critical value is an estimate
    and a cell whose statistic sits inside that estimate's own noise is not
    really being classified by the rule.  My first version returned the point
    alone at 4,000 draws, and the printed table then showed 1.897 at b = 0.0092
    and 1.974 at b = 0.0094 - a fixed-b point BELOW the normal one, which
    cannot happen, and a reader would have been right to stop there.
    """
    per = max(1, reps // batches)
    qs = []
    pool = []
    for _ in range(batches):
        t = np.abs(hac_t_batch(rng.standard_normal((per, n)), M))
        pool.append(t)
        qs.append(float(np.quantile(t, 0.95)))
    cv = float(np.quantile(np.concatenate(pool), 0.95))
    se = float(np.std(qs, ddof=1) / math.sqrt(batches))
    return cv, se


def _assert_same_fixed_b():
    """The fast fixed-b point must be the loop form's, ON THE SAME DRAWS.

    My first version of this check ran the two forms from different seeds and
    compared the resulting 95% points.  They differed by up to 0.055 and the
    check failed - on Monte Carlo noise, not on a defect, because the standard
    error of a simulated 95% quantile at 20,000 draws is about 0.015 apiece.
    An instrument check whose own noise is the size of the thing it is checking
    cannot settle anything.

    numpy fills standard_normal((k, n)) row-major, so one (k, n) call and k
    successive length-n calls from the same seed are the same stream.  Feeding
    both forms that stream turns a noisy comparison of quantiles into an exact
    comparison of every statistic, and the tolerance becomes floating point
    rather than a guess.
    """
    print("  fast fixed-b against the loop form, on identical draws:")
    worst = 0.0
    for n, M, reps in ((240, 20, 300), (400, 20, 300), (400, 40, 300),
                       (600, 30, 200), (1240, 80, 120)):
        Z = np.random.default_rng(5).standard_normal((reps, n))
        g = np.random.default_rng(5)
        loop = np.array([abs(z.mean() / hac_se(z, M))
                         for z in (g.standard_normal(n) for _ in range(reps))])
        fast = np.abs(hac_t_batch(Z, M))
        d = float(np.abs(fast - loop).max())
        worst = max(worst, d)
        # no 95% point is printed here.  It used to be, at 120 to 300 draws,
        # where its Monte Carlo error is about 0.05: one row read 1.956, below
        # the 1.96 this lab elsewhere says a fixed-b point cannot be, and
        # another gave 2.432 for a pair the results table below gives as
        # 2.129.  Both were noise in a column that existed only to pad a check
        # of the INSTRUMENT, and a diagnostic that prints numbers a reader can
        # mistake for results is worse than one that prints none.
        print(f"    n={n:>5}  M={M:>3}  b={M / n:.3f}  {reps} draws   "
              f"worst |t| difference {d:.2e}")
    if worst > 1e-9:
        raise SystemExit("the fast fixed-b point is not the loop form's")
    print(f"    worst over all of them {worst:.2e}: the same statistic, "
          f"not merely a similar one")
    return worst


def _assert_same_instrument(cells):
    """The batched HAC must equal the labs' own, cell by cell, on real data."""
    worst, where = 0.0, None
    for c in cells:
        a = float(hac_t_batch(c["x"][None, :], c["M"])[0])
        b = float(c["x"].mean() / hac_se(c["x"], c["M"]))
        d = abs(a - b) / max(abs(b), 1e-12)
        if d > worst:
            worst, where = d, c["name"]
    print(f"  batched HAC against the loop form, worst relative difference "
          f"over all {len(cells)} cells: {worst:.2e}  ({where})")
    if worst > 1e-9:
        raise SystemExit("the batched HAC is not the estimator the labs use")
    return worst


# ----------------------------------------------------------------------
# the registry, as a family
# ----------------------------------------------------------------------
def harvest():
    rows = REG.load_all()
    if not rows:
        raise SystemExit(
            "cell_registry/ is empty, so there is no family to test.\n"
            "  Run the five recording labs first, or the whole suite:\n"
            "      python run_all.py --data <your data folder>\n"
            "  and do NOT set REGISTRY_OFF=1 on that run.")
    cells = [dict(name=m["cell_id"], group=m["calendar_group"], x=s,
                  M=int(m["bandwidth"]), cv=float(m["critical_value"]),
                  family=m["family"], delta=m["delta"], n=int(s.size),
                  span=(m.get("index_start"), m.get("index_end")))
             for m, s in rows]
    # A calendar group is a CALENDAR, and two groups recorded on the same index
    # are one calendar however they were labelled.  daily-PAPER and daily-WIDE
    # were two: the suffix is a ridge PENALTY GRID, not a clock, and both arms
    # run on the identical index.  Resampling them independently threw away the
    # dependence between the most strongly coupled cells in the pool - the 24
    # matched (arm, delay) pairs correlate above 0.7 - and it did so in the
    # direction that narrows the null, which is the direction that flatters the
    # paper.  Merging is done by MEASURED span and length, not by a list of
    # names, so a future group that collides with an existing calendar is
    # merged for the same reason rather than overlooked for want of an entry.
    _key = {}
    for c in cells:
        _key.setdefault((c["n"], c["span"]), []).append(c)
    _merged = []
    for (_n, _sp), _cs in _key.items():
        _gs = sorted({c["group"] for c in _cs})
        if len(_gs) > 1:
            _name = "+".join(_gs)
            for c in _cs:
                c["group"] = _name
            _merged.append((_name, _gs, len(_cs), _n))
    for _name, _gs, _k, _n in _merged:
        print(f"  calendars merged: {', '.join(_gs)} share an index "
              f"({_sp[0]} to {_sp[1]}, n = {_n}) and are one calendar, "
              f"{_k} cells, resampled on one draw")
    seen = {c["name"].split(":")[0] for c in cells}
    missing = [s for s in SOURCES if s.split("_")[0] not in seen]
    if missing:
        raise SystemExit(
            f"the registry is missing every cell from {', '.join(missing)}, so "
            f"the family is smaller than the one the paper reports.\n"
            f"  Run those labs, then this one.")
    groups = {}
    for c in cells:
        groups.setdefault(c["group"], []).append(c)
    for g, ids in groups.items():
        lens = {c["n"] for c in ids}
        if len(lens) != 1:
            raise SystemExit(
                f"calendar group {g!r} holds cells of lengths {sorted(lens)}; "
                f"cells sharing a group must share an index, or they cannot "
                f"share block starts, which is the whole point of the grouping")
    return cells, groups


# ----------------------------------------------------------------------
# the joint null
# ----------------------------------------------------------------------
def _packs(cells, groups, block=None):
    """One resampling plan per calendar group, with the guards that plan needs.

    Split out of joint() because the calibration pass below has to resample
    exactly the same way the test does, and two copies of a resampling scheme
    is two schemes.
    """
    idx_of = {c["name"]: i for i, c in enumerate(cells)}
    pack = {}
    for g, ids in groups.items():
        Y = np.vstack([c["x"] for c in ids])
        n = Y.shape[1]
        Yc = Y - Y.mean(1, keepdims=True)          # the null, imposed
        rows = {M: [i for i, c in enumerate(ids) if c["M"] == M]
                for M in sorted({c["M"] for c in ids})}
        L = block or 60                     # >= delta + h at the longest delay
        L = min(L, max(2, n // 4))
        # ceil, not floor: n // L gives a resampled series SHORTER than the
        # original - by 28 days at n = 4,948 and 40 at n = 1,240 - so the
        # bootstrap t was computed at a different n from the observed t, and
        # under rule B scored against a critical value simulated at the
        # original (n, M).  Blocks are drawn to cover n and the index is then
        # truncated to exactly n, so every replication has the length the
        # statistic was calibrated at.
        nb = max(1, -(-n // L))
        # the resampling must preserve the length the statistic is calibrated
        # at, and must be able to draw the last observation; both were wrong
        # in the first version of this function and both biased the null count
        # DOWNWARD, which is the direction that flatters the paper
        _probe = (np.zeros(nb, int)[:, None] + np.arange(L)[None, :]).ravel()[:n]
        if len(_probe) != n:
            raise SystemExit(f"group {g}: a replication would have length "
                             f"{len(_probe)} against the observed {n}")
        if (max(1, n - L + 1) - 1) + L - 1 < n - 1:
            raise SystemExit(f"group {g}: the last observation can never be "
                             f"drawn, so the null is built on a truncated "
                             f"series")
        pack[g] = (Y, Yc, rows, n, L, nb,
                   np.array([idx_of[c["name"]] for c in ids]))
    return pack


def calibrate(cells, groups, block=None, B=B, seed=SEED + 7777):
    """Each cell's own 95% point of |t| under the resampling null.

    WHY THIS RULE EXISTS

    Rules A and B both ask whether |t| exceeds a value taken from a reference
    distribution - the normal, or the fixed-b limit - and both are wrong about
    this pool for the same reason: the cells' loss differentials move in
    time-localised episodes on the scale of the block, and a Bartlett t-test
    is not correctly sized against that whatever critical value it is given.
    Measured below, the rule's own marginal size under its own null is nearer
    9% than 5%, cell by cell, with the worst cells near a third.

    That matters twice.  It inflates the OBSERVED count, because a mis-sized
    rule rejects too often on the real data as well; and it inflates the null
    mean, which is what made the null look like evidence of dependence when it
    is mostly evidence of size.  Calibrating each cell against its own
    resampled distribution fixes both at once, and is the ordinary bootstrap
    answer to a statistic whose reference distribution is not available.

    The calibration draws are INDEPENDENT of the draws the count is then judged
    against (a different seed), because calibrating and testing on one set of
    draws would force the null mean to 0.05K by construction and the agreement
    would be arithmetic rather than evidence.
    """
    rng = np.random.default_rng(seed)
    K = len(cells)
    T = np.empty((B, K))
    pack = _packs(cells, groups, block)
    for b in range(B):
        for g, (Y, Yc, rows, n, L, nb, gi) in pack.items():
            starts = rng.integers(0, max(1, n - L + 1), nb)
            idx = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]
            Z = Yc[:, idx]
            for M, r in rows.items():
                T[b, gi[r]] = np.abs(hac_t_batch(Z[r], M))
    return {c["name"]: float(np.quantile(T[:, i], 0.95))
            for i, c in enumerate(cells)}


def joint(cells, groups, cvlist, block=None, B=B, seed=SEED):
    """The null distribution of the count, by block draws shared within a group.

    `cvlist` is a list of {cell name -> critical value} maps, one per marginal
    rule.  Every rule is scored on THE SAME resampled series, which is both
    cheaper and better: the counts below differ because the rules differ and
    for no other reason.  What is never shared is the null itself - each rule's
    observed count is judged against the null built with that same rule.

    Returns the observed counts, the null counts, the resampling p-values, and
    the per-cell rejection indicators for both.  The last of those is what lets
    the mean of the count be attributed to marginal size rather than to
    dependence: a per-cell rejection RATE is a marginal quantity and is not
    changed by sharing a draw, so it can be read off the same replications.
    """
    rng = np.random.default_rng(seed)
    nr = len(cvlist)
    K = len(cells)
    obs = [0] * nr
    obs_cell = np.zeros((nr, K), bool)
    hit_cell = np.zeros((nr, B, K), bool)
    pack = _packs(cells, groups, block)
    for g, (Y, Yc, rows, n, L, nb, gi) in pack.items():
        cv = [np.array([d[cells[i]["name"]] for i in gi]) for d in cvlist]
        for M, r in rows.items():
            t = np.abs(hac_t_batch(Y[r], M))
            for k in range(nr):
                obs[k] += int((t > cv[k][r]).sum())
                obs_cell[k, gi[r]] = t > cv[k][r]
    counts = np.zeros((nr, B), int)
    for b in range(B):
        for g, (Y, Yc, rows, n, L, nb, gi) in pack.items():
            cv = [np.array([d[cells[i]["name"]] for i in gi]) for d in cvlist]
            # integers() is exclusive on the upper end, so the old
            # `n - L` gave starts in [0, n-L-1] and the LAST observation of
            # every series was never drawn.  The moving-block range is
            # [0, n-L] inclusive.
            starts = rng.integers(0, max(1, n - L + 1), nb)   # ONE draw per group
            idx = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]
            Z = Yc[:, idx]
            for M, r in rows.items():
                t = np.abs(hac_t_batch(Z[r], M))
                for k in range(nr):
                    h = t > cv[k][r]
                    counts[k, b] += int(h.sum())
                    hit_cell[k, b, gi[r]] = h
    p = [(1 + int((counts[k] >= obs[k]).sum())) / (B + 1) for k in range(nr)]
    return obs, counts, p, obs_cell, hit_cell


def duplicates(cells):
    """Cells whose loss differential is byte-identical to an earlier cell's.

    The seven-regressor arm is recorded twice, once by the lab that runs it and
    once by the lab that compares it with a compressed one, under two names and
    with one series.  Thirty-two of the 214 are copies of that kind, so the
    pool the count is taken over is not 214 independent reports of anything -
    seven of the significant cells are the same cell twice.  The test is run on
    both pools below rather than on whichever one is larger.
    """
    seen, dup = {}, []
    for i, c in enumerate(cells):
        k = c["x"].tobytes()
        if k in seen:
            dup.append((c["name"], cells[seen[k]]["name"]))
        else:
            seen[k] = i
    keep = np.array([c["x"].tobytes() not in {cells[j]["x"].tobytes()
                                              for j in range(i)}
                     for i, c in enumerate(cells)])
    return dup, keep


def exact_upper_p(B, level=0.05):
    """The largest true p consistent with ZERO exceedances in B draws.

    (1 + 0) / (B + 1) is the smallest value the estimator can RETURN, so it is
    a floor on the estimate and not a ceiling on the truth; writing it as
    "p <= 0.0005" points the inequality the wrong way.  With no exceedance in B
    independent draws the one-sided upper confidence bound at this level is
    1 - level**(1/B), which is three times larger at B = 2,000.
    """
    return 1.0 - level ** (1.0 / B)


def report(label, rule, K, obs, counts, p, rate=None):
    binom = math.sqrt(K * 0.05 * 0.95)
    print(f"\n{label}")
    print(f"  marginal rule                        {rule}")
    print(f"  observed significant cells           {obs}")
    print(f"  0.05K, the expectation Table S30 uses{0.05 * K:7.1f}")
    print(f"  empirical null mean                  {counts.mean():.2f}")
    print(f"  empirical null sd                    {counts.std():.2f}"
          f"   (binomial would be {binom:.2f})")
    print(f"  variance inflation over binomial     "
          f"{counts.var() / binom ** 2:.1f}x")
    if rate is not None:
        # The null mean is a MARGINAL quantity: expectations are linear, so no
        # amount of cross-cell dependence can move it.  It sits above 0.05K
        # because the rule's own size under this resampling is above 5%, and the
        # sum of the per-cell sizes below reproduces the mean without any
        # dependence at all.  The dependence is in the SPREAD, and its honest
        # size is the inflation over independent cells AT THOSE SIZES, not over
        # a binomial at 5% - which is a benchmark for a rule this one is not.
        # joint_null.py has said so since it was written; this file printed the
        # mean as though dependence had caused it, and the appendix followed.
        ind = math.sqrt(float((rate * (1 - rate)).sum()))
        print(f"  the rule's own size under its own null, per cell: mean "
              f"{rate.mean():.3f}, median {np.median(rate):.3f}, "
              f"max {rate.max():.3f}")
        print(f"  sum of those per-cell sizes          {rate.sum():.2f}"
              f"   <- the null mean, with NO dependence")
        print(f"  independent-cell sd at those sizes   {ind:.2f}"
              f"   <- the right benchmark for the spread")
        print(f"  so of the {counts.std() / binom:.1f}x spread over binomial, "
              f"{ind / binom:.1f}x is size and heterogeneity and "
              f"{counts.std() / ind:.1f}x is dependence")
    print(f"  null percentiles  90th {np.quantile(counts, 0.90):.0f}"
          f"   95th {np.quantile(counts, 0.95):.0f}"
          f"   99th {np.quantile(counts, 0.99):.0f}")
    hit = int((counts >= obs).sum())
    print(f"  null maximum over {B} replications      {counts.max()}")
    print(f"  replications reaching {obs:<3}               {hit}")
    print(f"  p_joint = (1 + {hit}) / ({B}+1)  =  {p:.4f}")
    if hit == 0:
        print(f"  {p:.4f} is the smallest value {B} replications can return, "
              f"so it is a FLOOR on the")
        print(f"  estimate and not a ceiling on the truth: what was measured "
              f"is that the null never")
        print(f"  reached {obs} in {B} draws, its largest count being "
              f"{counts.max()}.  Zero exceedances in")
        print(f"  {B} draws is consistent with a true p as large as "
              f"{exact_upper_p(B):.4f}, which is the exact")
        print(f"  one-sided 95% bound, and that is the number to quote - "
              f"'p <= {p:.4f}' points the")
        print(f"  inequality the wrong way round.")


def main():
    t0 = __import__("time").time()
    print("=" * 78)
    print("lab68_joint_multiplicity  -  the omnibus test Table S30 never had")
    print("=" * 78)

    cells, groups = harvest()
    K = len(cells)
    print(f"\nTHE FAMILY, AS RECORDED")
    print(f"  cells {K}   calendar groups {len(groups)}")
    fam = {}
    for c in cells:
        fam[c["family"]] = fam.get(c["family"], 0) + 1
    for f, k in sorted(fam.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {k:>4}  {f}")
    print("\n  calendar groups, and the length every cell in each one shares:")
    for g in sorted(groups):
        print(f"    {g:<16} {len(groups[g]):>3} cells   n = {groups[g][0]['n']}")
    if K != EXPECT_CELLS or len(groups) != EXPECT_GROUPS:
        raise SystemExit(
            f"the registry holds {K} cells in {len(groups)} groups, not "
            f"{EXPECT_CELLS} in {EXPECT_GROUPS}.  Either a recording lab did "
            f"not run, or the family changed and the paper's count with it; "
            f"either way this is not the family that was pre-declared.")

    print("\nTHE INSTRUMENTS, CHECKED AGAINST THE SLOW FORMS THEY REPLACE")
    _assert_same_instrument(cells)
    _assert_same_fixed_b()

    print("\nRULE A - the paper's own marginal rule")
    print("  Bandwidth block_for(delta), which is what the five labs pass, and")
    print("  a normal critical value of 1.96.  This is the rule that produced")
    print("  the count Table S30 tabulates, so it is the rule the count is")
    print("  judged against here.")

    print("\nRULE B - the same bandwidth against its own reference distribution")
    print("  A Bartlett t-statistic at a bandwidth that is a non-vanishing")
    print("  fraction of the sample is not standard normal.  Below, the fixed-b")
    print("  95% point simulated at each distinct (n, M) in the pool, which is")
    print("  what 1.96 is standing in for.")
    rng = np.random.default_rng(12345)
    pairs = sorted({(c["n"], c["M"]) for c in cells})
    cvb, cse = {}, {}
    print(f"\n  {CV_REPS:,} draws per pair, in ten batches, the spread of whose")
    print(f"  own 95% points gives the standard error beside each value.")
    print(f"\n{'n':>8}{'M':>6}{'b = M/n':>10}{'fixed-b cv':>12}{'se':>8}"
          f"{'cells':>8}")
    for n, M in pairs:
        cvb[(n, M)], cse[(n, M)] = fixed_b_cv_fast(n, M, rng)
        k = sum(1 for c in cells if c["n"] == n and c["M"] == M)
        print(f"{n:>8}{M:>6}{M / n:>10.4f}{cvb[(n, M)]:>12.3f}"
              f"{cse[(n, M)]:>8.3f}{k:>8}")
    print(f"\n  range of the fixed-b 95% point over the pool: "
          f"{min(cvb.values()):.3f} to {max(cvb.values()):.3f}, against 1.96")
    print(f"  largest standard error among them: {max(cse.values()):.3f}")
    if min(cvb.values()) < 1.96:
        _lo = min(cvb, key=cvb.get)
        print(f"  The smallest, {cvb[_lo]:.3f} at b = {_lo[1] / _lo[0]:.4f}, "
              f"is BELOW 1.96, which the fixed-b")
        print(f"  point cannot be at b > 0.  It sits "
              f"{(1.96 - cvb[_lo]) / cse[_lo]:.1f} standard errors under, so "
              f"this is")
        _bhi = max(pairs, key=lambda p: p[1] / p[0])
        print(f"  simulation noise and not a result - and it is also the "
              f"finding: at b of a")
        print(f"  hundredth the correction is inside its own Monte Carlo "
              f"error.  It is only at the")
        print(f"  short end that it bites, where b reaches "
              f"{_bhi[1] / _bhi[0]:.3f} and the point reaches "
              f"{max(cvb.values()):.3f}.  A long sample")
        print(f"  does not need this correction; a short one does, which is "
              f"the same lesson")
        print(f"  lab67 part A draws about Table S7's subsamples.")

    cvs_a = {c["name"]: c["cv"] for c in cells}
    cvs_b = {c["name"]: cvb[(c["n"], c["M"])] for c in cells}
    # how much of rule B's count is at the mercy of the simulation: a cell whose
    # statistic sits within two standard errors of its own threshold is not
    # being classified by the rule, it is being classified by the draw
    fragile = sum(1 for c in cells
                  if abs(abs(hac_t_batch(c["x"][None, :], c["M"])[0])
                         - cvb[(c["n"], c["M"])]) < 2 * cse[(c["n"], c["M"])])
    print(f"  cells sitting closer to their own threshold than twice THEIR "
          f"OWN standard error")
    print(f"  from the column above, and so liable to fall either side of "
          f"rule B on the draws alone: {fragile}")
    # WHICH cells the two rules disagree about, computed rather than asserted.
    # An earlier version of this lab printed the sentence "what rule B actually
    # changes is the six cells at n = 1,240" from the shape of the critical
    # values alone, having never looked at a single statistic.  It was also
    # wrong: the cells that fall are not the short ones.
    flip = []
    for c in cells:
        t = abs(float(hac_t_batch(c["x"][None, :], c["M"])[0]))
        if t > c["cv"] and t <= cvs_b[c["name"]]:
            flip.append((c, t))
    print(f"\n  cells significant under rule A and not under rule B: "
          f"{len(flip)}")
    for c, t in sorted(flip, key=lambda ct: -ct[1]):
        print(f"    {c['name']:<34} n={c['n']:>5}  M={c['M']:>3}  "
              f"|t|={t:.3f}  against 1.96 and "
              f"{cvs_b[c['name']]:.3f}")
    if flip:
        _ns = sorted({c["n"] for c, _ in flip})
        print(f"    their lengths: {', '.join(f'{n:,}' for n in _ns)} - "
              f"so the rule does not fall only on the")
        print(f"    short series.  Which cells it falls on is not a property "
              f"of length: it is whichever")
        print(f"    statistics happen to sit between 1.96 and their own "
              f"fixed-b point, and the three above")
        print(f"    are listed rather than characterised for that reason.")

    print("\nRULE C - each cell against its own resampled distribution")
    print("  Rules A and B take a critical value from a reference distribution")
    print("  the statistic does not have on this pool.  Rule C takes each")
    print("  cell's own 95% point of |t| under the same resampling null, on")
    print("  INDEPENDENT draws, which is the ordinary bootstrap answer and the")
    print("  only one of the three that is correctly sized by construction.")
    cvs_c = calibrate(cells, groups)
    _cvc = np.array([cvs_c[c["name"]] for c in cells])
    print(f"\n  calibrated critical values: {_cvc.min():.2f} to "
          f"{_cvc.max():.2f}, median {np.median(_cvc):.2f}, against 1.96")
    print(f"  cells needing a critical value above 2.5: "
          f"{int((_cvc > 2.5).sum())} of {K}")

    obs, cnt, p, obs_cell, hit_cell = joint(cells, groups,
                                            [cvs_a, cvs_b, cvs_c])
    obs_a, obs_b, obs_c = obs
    cnt_a, cnt_b, cnt_c = cnt[0], cnt[1], cnt[2]
    p_a, p_b, p_c = p
    rate = hit_cell.mean(1)
    report("RESULT, RULE A", "Bartlett block_for(delta), cv = 1.96",
           K, obs_a, cnt_a, p_a, rate[0])
    report("RESULT, RULE B",
           "Bartlett block_for(delta), fixed-b cv at each (n, M)",
           K, obs_b, cnt_b, p_b, rate[1])
    report("RESULT, RULE C",
           "Bartlett block_for(delta), each cell's own resampled 95% point",
           K, obs_c, cnt_c, p_c, rate[2])

    print("\n" + "=" * 78)
    print("THE POOL ITSELF: DUPLICATES, AND WHICH WAY THE SIGNIFICANT CELLS POINT")
    print("=" * 78)
    dup, keep = duplicates(cells)
    print(f"  cells whose series is byte-identical to another cell's: "
          f"{len(dup)} of {K}")
    for a, b in dup[:4]:
        print(f"    {a:<40} is {b}")
    if len(dup) > 4:
        print(f"    ... and {len(dup) - 4} more of the same shape")
    print(f"  distinct cells                       {int(keep.sum())}")
    for _k, (_nm, _o, _c, _pp) in enumerate(
            (("rule A", obs_a, cnt_a, p_a), ("rule B", obs_b, cnt_b, p_b),
             ("rule C", obs_c, cnt_c, p_c))):
        _od = int(obs_cell[_k][keep].sum())
        _cd = hit_cell[_k][:, keep].sum(1)
        _hd = int((_cd >= _od).sum())
        _pd = (1 + _hd) / (B + 1)
        print(f"  {_nm}, distinct cells only: {_od} observed against a null "
              f"of mean {_cd.mean():.1f}, sd {_cd.std():.1f}, 95th "
              f"{np.quantile(_cd, 0.95):.0f}, p "
              f"{'<= ' if _hd == 0 else '= '}{_pd:.4f}")
    _tt = np.array([float(hac_t_batch(c["x"][None, :], c["M"])[0])
                    for c in cells])
    _sig = obs_cell[0]
    print(f"\n  of the {int(_sig.sum())} cells significant under rule A, "
          f"{int((_tt[_sig] < 0).sum())} have a NEGATIVE statistic and "
          f"{int((_tt[_sig] > 0).sum())} a positive one.")
    print("  The count is two-sided and the test is direction-agnostic, which")
    print("  is deliberate: the question is whether the pool holds more")
    print("  structure than chance, not whether the structure helps.  But a")
    print("  reader told that a count of significant cells exceeds chance will")
    print("  assume the cells point the paper's way, and most of them do not:")
    print("  on these grids the added block more often significantly HURTS.")
    _sigc = obs_cell[2]
    print(f"  under rule C: {int(_sigc.sum())} significant, "
          f"{int((_tt[_sigc] < 0).sum())} negative, "
          f"{int((_tt[_sigc] > 0).sum())} positive.")

    print("\n" + "=" * 78)
    print("WHAT THIS REPLACES, IN NUMBERS")
    print("=" * 78)
    # The count below is THIS file's, measured from the cell registry under
    # rule A, and it is not the count Table S30 prints: that table's comes
    # from lab62, which scrapes z-statistics out of the other labs' printed
    # text, and the two instruments differ by a cell or two on the same pool.
    # Attributing this number to Table S30 was the project's own recurring
    # defect in miniature - a figure credited to a source that does not print
    # it - and a reader who compared the two would have found them disagreeing
    # with no explanation.  The SHAPE of the argument is what belongs to Table
    # S30; the number belongs here.
    print(f"  the shape of Table S30's argument:  a count far above "
          f"{0.05 * K:.1f} expected, with no variance attached")
    print(f"  this file's own count, rule A:  {obs_a} observed")
    print(f"  what the null actually is, rule A:  mean {cnt_a.mean():.1f}, "
          f"sd {cnt_a.std():.1f}, 95th percentile "
          f"{np.quantile(cnt_a, 0.95):.0f}")
    print(f"  so a count of {np.quantile(cnt_a, 0.95):.0f} is ordinary noise, "
          f"and the null mean is "
          f"{cnt_a.mean() / (0.05 * K):.1f}x the tabulated expectation")
    # "p <= x" when nothing reached the observed count, "p = x" when something
    # did - and when nothing did, the number that bounds the truth is the exact
    # one, not the estimator's floor.
    _pa = (f"p_joint = {p_a:.4f}, bounded above by {exact_upper_p(B):.4f}"
           if int((cnt_a >= obs_a).sum()) == 0 else f"p_joint = {p_a:.4f}")
    _pb = (f"p_joint = {p_b:.4f}, bounded above by {exact_upper_p(B):.4f}"
           if int((cnt_b >= obs_b).sum()) == 0 else f"p_joint = {p_b:.4f}")
    print(f"  the observed {obs_a} is still outside it:  {_pa}, the null's "
          f"largest count in {B} draws being {cnt_a.max()}")
    print(f"  on the conservative rule B:  {obs_b} observed, null 95th "
          f"percentile {np.quantile(cnt_b, 0.95):.0f}, {_pb}")
    # "on both marginal rules" was printed unconditionally beside a verdict
    # taken from max(p_a, p_b); had the two rules disagreed the sentence would
    # have asserted agreement the code never established.  Both are stated now.
    _ok = [p_a < 0.05, p_b < 0.05, p_c < 0.05]
    _names = ["A", "B", "C"]
    if all(_ok):
        verdict = "carry signal on all three rules"
    elif any(_ok):
        verdict = ("carry signal on rule" + ("s " if sum(_ok) > 1 else " ")
                   + ", ".join(n for n, o in zip(_names, _ok) if o)
                   + " and NOT on rule"
                   + ("s " if sum(_ok) < 2 else " ")
                   + ", ".join(n for n, o in zip(_names, _ok) if not o)
                   + ", which is a disagreement between the rules and not a "
                     "result")
    else:
        verdict = "do NOT carry signal, on any of the three rules"
    print(f"  on rule C, the only correctly sized one:  {obs_c} observed, "
          f"null mean {cnt_c.mean():.1f}, 95th percentile "
          f"{np.quantile(cnt_c, 0.95):.0f}, p_joint = {p_c:.4f}")
    print(f"\n  VERDICT: the secondary grids collectively {verdict}.")
    print(f"  What Table S30 cannot claim any more is the SIZE of the margin "
          f"it implied, and")
    print(f"  neither can rule A.  The null rule A is judged against sits "
          f"{cnt_a.mean() / (0.05 * K):.1f} times above")
    print(f"  0.05K because the rule itself rejects at "
          f"{rate[0].mean():.3f} rather than 0.05 under its own null, not "
          f"because")
    print(f"  the cells are correlated; correcting that, rule C gives "
          f"{obs_c} observed against a null")
    print(f"  mean of {cnt_c.mean():.1f} and p_joint = {p_c:.4f}, against rule "
          f"A's {p_a:.4f}.  The direction of the")
    print(f"  conclusion survives on all rules that reject; the MARGIN is a "
          f"property of the rule,")
    print(f"  and the margin to quote is rule C's.")
    print(f"\n  Not tested here, deliberately: the headline, which stays "
          f"outside both pools,")
    print(f"  and the market-state ladder, which lab67 tests with a statistic "
          f"matched to")
    print(f"  its own claim rather than by counting significant cells.")
    print(f"\nruntime {__import__('time').time() - t0:.1f}s")


if __name__ == "__main__":
    main()
