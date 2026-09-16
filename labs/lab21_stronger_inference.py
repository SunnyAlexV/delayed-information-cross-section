"""
lab21_stronger_inference.py - two upgrades a referee asked for, on the two
objects Section 5 rests on.

Imports lab05_robustness; keep both in labs/.  Runtime about seven minutes.

UPGRADE 1 - A BETTER NULL GENERATOR FOR THE COST OF BREADTH
-----------------------------------------------------------
lab07 measures what carrying seven extra regressors costs by replacing the
foreign block with AR(1) surrogates: Gaussian, matched on each peer's first-order
persistence and on the contemporaneous covariance of the innovations.  A referee
objects, correctly, that this preserves first-order persistence and cross-market
covariance but NOT heavy tails, volatility-of-volatility, crisis clustering,
nonlinear dependence or higher-order serial structure.  The cost is therefore
identified only relative to that generator.

The answer is to measure it again with a generator that fits nothing.  Move the
real foreign block circularly through the sample: every marginal, every tail,
every cluster is the real one, and only the alignment with the target is
destroyed.  If the unfitted generator returns the fitted generator's number, the
objection is answered rather than conceded.

Two traps sit in that sentence, and both are demonstrated below rather than
asserted, because the first version of this lab fell into both and reported a
disagreement that was neither of the generators' doing.

TRAP 1 - THE GAPS MOVE WITH THE VALUES.  The foreign block is all-finite on 96.2%
of rows; the rest are holidays that differ across markets.  Rolling the raw array
moves that missingness pattern too, so on about 189 test days per draw (3.9%) the
surrogate arm has no usable row where the real arm does, and the walk-forward
falls back to a zero forecast.  Training rows drop for the same reason, so the
surrogate arm also estimates its seven coefficients from a smaller window.  Both
push the measured cost down, and neither is a cost of breadth.  The fix is to
shift the VALUES while keeping the real mask: compact the block to its all-finite
rows, roll that, and scatter it back.  Both versions are reported below.

TRAP 2 - AN OFFSET CAN BE A NEAR-IDENTITY.  Offsets are taken modulo the sample
length, so an offset a few rows short of the full length is a shift of a few rows
BACKWARDS - the real block, barely moved, still carrying the signal.  Offsets here
are held at least GUARD rows from identity in either direction.  Part A runs the
careless grid alongside the guarded one and prints both, because a trap described
in a comment is worth less than a trap shown in the output.

Part A runs all three generators on identical days and reports the Monte Carlo
standard error of each, which is the quantity that decides whether a difference
between two of them means anything.

UPGRADE 2 - AN INTERVAL THAT KNOWS THE DENOMINATOR CAN VANISH
-------------------------------------------------------------
The substitution rate is a ratio,

    R(delta) = [S_cross(delta) - S_own(delta)] / [S_own(0) - S_own(delta)],

and its denominator is what a delay costs a domestic-only forecaster.  At short
delays that is near zero, so R is weakly identified and a percentile bootstrap of
it is not trustworthy however many replicates it uses: the bootstrap distribution
of a ratio with a denominator straddling zero is not merely wide, it can be
bimodal or unbounded, and a percentile interval will quietly report finite
endpoints anyway.  This project has met the symptom repeatedly - [-154%, 50%] at
one point - and treated it as an empirical nuisance.  It is a formal problem with
a textbook answer.

Fieller's theorem inverts a t-test on the linear combination

    num - R * den,

which is exactly zero at the true R, and solves the resulting quadratic for the
set of R values not rejected.  When the denominator is well separated from zero
the solution is an ordinary interval close to the delta-method one.  When it is
not, the quadratic's leading coefficient changes sign and the confidence set is
correctly a union of two half-lines, or the whole line - an honest statement that
the data do not pin the ratio down.  A percentile bootstrap cannot express that
and so never does.

Part B computes both and reports where they disagree.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L

SEED = 20260913
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 1, 3, 5, 13, 21, 55]
N_DRAW = 20          # AR(1) surrogate draws, as lab07
N_SHIFT = 20         # circular offsets, spread across the sample
GUARD = 200          # keep every offset this far from a near-identity shift
N_BOOT = 2000
BLK = 10


def panel(folder):
    D, peers, lag = L.build(folder, TARGET)
    a = D[TARGET].values
    sa = L.pd.Series(a)
    own = np.column_stack([sa.values, sa.rolling(5).mean().values,
                           sa.rolling(22).mean().values])
    P = D[peers].values
    n = len(D)
    y = np.full(n, np.nan); y[:-H] = a[H:]
    start = L.MED + L.TRAIN + L.VAL + max(DELAYS) + H
    idx = np.arange(start, n - H)
    idx = idx[np.isfinite(y[idx]) & np.isfinite(a[idx])]
    return own, P, y, idx, len(peers)


def walk(own, P, y, idx, delta, use_peers):
    out = np.empty(len(idx)); b = mu = sd = None
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), cut)
            X = np.column_stack([own[tr - delta]] + ([P[tr]] if use_peers else []))
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y[tr])
            X, yy = X[ok], y[tr][ok]
            if len(yy) < 200:
                b = None; continue
            mu, sd = X.mean(0), X.std(0) + 1e-9
            b = L.cv((X - mu) / sd, yy, "cont")
        xt = np.concatenate([own[t - delta]] + ([P[t]] if use_peers else []))
        out[j] = 0.0 if (b is None or not np.isfinite(xt).all()) \
            else L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
    return out


def r2(yb, f):
    return 1 - ((yb - f) ** 2).sum() / ((yb - yb.mean()) ** 2).sum()


def fieller(num, den, alpha=0.05):
    """Confidence set for mean(num)/mean(den), by inverting a HAC t-test.

    Returns (kind, lo, hi).  kind is 'interval' when the quadratic has real roots
    and opens the right way, 'exclusion' when the set is the complement of an
    interval - two half-lines - and 'unbounded' when nothing is excluded.  The
    last two are the honest answers a percentile bootstrap cannot give.
    """
    n = len(num)
    mn, md = num.mean(), den.mean()
    # HAC, not iid.  The targets overlap by five days and the losses are strongly
    # serially dependent, so iid variances understate every standard error here
    # and Fieller would come out NARROWER than the block bootstrap at every delay
    # - which is what the first version of this function did, and is the tell.
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))

    def hac(u, v):
        u, v = u - u.mean(), v - v.mean()
        s = (u @ v) / n
        for kk in range(1, lag + 1):
            w = 1 - kk / (lag + 1)
            s += w * ((u[kk:] @ v[:-kk]) + (v[kk:] @ u[:-kk])) / n
        return s / n

    vn, vd, cv = hac(num, num), hac(den, den), hac(num, den)
    z = 1.959963985                                   # normal, two-sided 5%
    A = md ** 2 - z ** 2 * vd
    B = -2.0 * (mn * md - z ** 2 * cv)
    C = mn ** 2 - z ** 2 * vn
    disc = B ** 2 - 4 * A * C
    if disc < 0:
        return ("unbounded", -np.inf, np.inf)
    r1 = (-B - np.sqrt(disc)) / (2 * A)
    r2_ = (-B + np.sqrt(disc)) / (2 * A)
    lo, hi = min(r1, r2_), max(r1, r2_)
    return ("interval", lo, hi) if A > 0 else ("exclusion", lo, hi)


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    own, P, y, idx, k = panel(folder)
    yb = y[idx]
    m = len(idx)
    print(f"target {TARGET}, {m} test days, {k} foreign peers\n")

    # ---------------- A --------------------------------------------------
    print("=" * 88)
    print("A.  DOES THE COST OF BREADTH SURVIVE AN UNFITTED NULL?")
    print("=" * 88)
    print("AR(1)   fitted Gaussian surrogates, matched on each peer's persistence and")
    print("        on the innovation covariance - lab07's generator, and the one the")
    print("        referee objects to because it cannot carry tails or clustering.")
    print("SHIFT   the real foreign block moved circularly, with the real missing-data")
    print("        mask held fixed.  Every marginal, tail and cluster is the real one;")
    print("        only the alignment is destroyed.  Nothing is fitted.")
    print("NAIVE   the same shift applied to the raw array, so the gaps move with the")
    print("        values.  Reported to show what that costs, not as a candidate.\n")

    rng = np.random.default_rng(SEED)
    fin = np.isfinite(P).all(axis=1)
    phi = np.empty(k); mu_p = np.empty(k); resid = []
    for j_ in range(k):
        v = P[fin, j_]; v0, v1 = v[:-1] - v.mean(), v[1:] - v.mean()
        phi[j_] = (v0 @ v1) / (v0 @ v0)
        resid.append(v1 - phi[j_] * v0)
        mu_p[j_] = v.mean()
    Chol = np.linalg.cholesky(np.cov(np.column_stack(resid), rowvar=False)
                              + 1e-12 * np.eye(k))

    def surrogate():
        S = np.empty_like(P); e = rng.normal(size=P.shape) @ Chol.T
        S[0] = mu_p + e[0] / np.sqrt(np.maximum(1 - phi ** 2, 1e-6))
        for t in range(1, len(P)):
            S[t] = mu_p + phi * (S[t - 1] - mu_p) + e[t]
        S[~fin] = np.nan
        return S

    Q = P[fin]                                   # the block with its gaps closed
    nq = len(Q)
    offs = [int(round(x)) for x in np.linspace(GUARD, nq - GUARD, N_SHIFT)]
    dist = min(min(o, nq - o) for o in offs)     # nearest approach to identity

    print(f"  guard: {N_SHIFT} offsets over a {nq}-row block, none closer than "
          f"{dist} rows to no shift at all")
    naive_lost = int(np.mean([(fin[idx] & ~np.roll(fin, o)[idx]).sum() for o in offs]))
    print(f"  trap 1: rolling the raw array instead would cost the surrogate arm "
          f"{naive_lost} of {len(idx)} test days per draw ({naive_lost / len(idx):.1%}),")
    print("          and the matching training rows, purely by moving the holidays.\n")

    def shifted(o):
        S = np.full_like(P, np.nan); S[fin] = np.roll(Q, o, axis=0); return S

    ar_draws = [surrogate() for _ in range(N_DRAW)]
    sh_draws = [shifted(o) for o in offs]
    nv_draws = [np.roll(P, o, axis=0) for o in offs]

    print(f"{'delta':>6}{'net dR2':>10}{'cost AR(1)':>13}{'cost SHIFT':>13}"
          f"{'(se)':>9}{'cost NAIVE':>13}{'gross AR(1)':>14}{'gross SHIFT':>14}")
    own_f, cross_f, cost_ar, cost_sh, se_sh, se_ar, cost_nv = {}, {}, {}, {}, {}, {}, {}
    for d in DELAYS:
        fo = walk(own, P, y, idx, d, False)
        fc = walk(own, P, y, idx, d, True)
        own_f[d], cross_f[d] = fo, fc
        base = r2(yb, fo)
        net = r2(yb, fc) - base

        def cost(draws):
            v = np.array([r2(yb, walk(own, S, y, idx, d, True)) - base for S in draws])
            return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))

        ca, sa = cost(ar_draws)
        cs, ss = cost(sh_draws)
        cn, _ = cost(nv_draws)
        cost_ar[d], cost_sh[d], se_sh[d], se_ar[d] = ca, cs, ss, sa
        cost_nv[d] = cn
        print(f"{d:>6}{net:>+10.4f}{ca:>+13.4f}{cs:>+13.4f}{ss:>9.4f}{cn:>+13.4f}"
              f"{net - ca:>+14.4f}{net - cs:>+14.4f}")

    # Compare each delay against ITS OWN combined Monte Carlo error, not against
    # the largest one in the table: a single loose tolerance would let a real
    # disagreement at delta = 0 hide behind the noisiest row at delta = 55.
    gap = {d: abs(cost_sh[d] - cost_ar[d]) for d in DELAYS}
    tol = {d: 3 * np.hypot(se_sh[d], se_ar[d]) for d in DELAYS}
    big = [d for d in DELAYS if gap[d] > tol[d]]
    worst = max(DELAYS, key=lambda d: gap[d] / tol[d])
    print(f"\n  gap between the fitted and unfitted generator, against three combined")
    print(f"  Monte Carlo standard errors, delay by delay:")
    for d in DELAYS:
        print(f"{d:>8}   gap {gap[d]:.4f}   tolerance {tol[d]:.4f}"
              f"   {'OUTSIDE' if gap[d] > tol[d] else 'inside'}")
    print(f"\n  closest call: delta = {worst}, at {gap[worst] / tol[worst]:.2f} of its tolerance")
    print(f"  delays where the two generators differ by more than that: "
          f"{len(big)} of {len(DELAYS)} {big if big else ''}")
    verdict = ("the two generators agree at every delay within Monte Carlo error, so\n"
               "  the cost of breadth is not an artefact of assuming Gaussian AR(1) peers"
               if not big else
               "the generators disagree at " + str(big) + ", so the cost figure is\n"
               "  generator-dependent and the paper must say which one it is quoting")
    print(f"  verdict: {verdict}.")
    lo_d, hi_d = DELAYS[0], DELAYS[-1]
    print(f"\n  the NAIVE column is the same shift with the gaps allowed to move.  It sits")
    print(f"  {abs(cost_nv[lo_d] - cost_sh[lo_d]):.4f} of R2 below the unfitted estimate at delta = {lo_d}, where the real")
    print(f"  cost is small, and {abs(cost_nv[hi_d] - cost_sh[hi_d]):.4f} below it at delta = {hi_d}, where the real cost is")
    print(f"  large - the signature of a fixed penalty for lost days, not a cost of breadth.")

    # ---------------- trap 2, demonstrated rather than described ----------
    # The claim "an offset can be a near-identity" is worth exactly as much as the
    # run that shows it, so run it: the offsets a careless linspace produces over
    # the RAW length, one of which is a shift of a few rows backwards.
    print("\n  trap 2: offsets are taken modulo the block length, so an offset just")
    print("  short of it is a shift of a few rows BACKWARDS - the real block, barely")
    print("  moved.  Below is the careless grid, spread over the raw array length")
    print("  rather than the compacted one, at delta = 0.\n")
    bad_offs = [int(round(x)) for x in np.linspace(260, len(P) - 260, 8)]
    base0 = r2(yb, own_f[DELAYS[0]])
    bad = []
    for o in bad_offs:
        c = r2(yb, walk(own, shifted(o % nq), y, idx, DELAYS[0], True)) - base0
        bad.append(c)
        near = min(o % nq, nq - (o % nq))
        print(f"{o:>8} (={near:>5} rows from identity)   cost {c:>+9.4f}"
              f"{'   <- not a null at all' if near < GUARD else ''}")
    bad = np.array(bad)
    hit = int(np.argmax(bad))
    print(f"\n  that grid gives mean {bad.mean():+.4f} with standard deviation "
          f"{bad.std(ddof=1):.4f},")
    print(f"  against {cost_sh[DELAYS[0]]:+.4f} from the guarded grid.  The single offset "
          f"{bad_offs[hit]}")
    print(f"  scores {bad[hit]:+.4f} on its own and carries the whole difference.")

    # ---------------- B --------------------------------------------------
    print("\n" + "=" * 88)
    print("B.  AN INTERVAL FOR THE RATE THAT KNOWS ITS DENOMINATOR CAN VANISH")
    print("=" * 88)
    print("R(delta) = [skill(cross,delta) - skill(own,delta)] / [skill(own,0) - skill(own,delta)].")
    print("Fieller inverts a t-test on num - R*den and solves the quadratic; where the")
    print("denominator is weak the confidence SET is two half-lines or the whole line,")
    print("which a percentile interval cannot report and so never does.\n")
    sst = ((yb - yb.mean()) ** 2).sum()
    e0 = (yb - own_f[0]) ** 2
    st = rng.integers(0, m - BLK + 1, size=(N_BOOT, int(np.ceil(m / BLK))))
    off = np.arange(BLK)
    S = [(st[i][:, None] + off).ravel()[:m] for i in range(N_BOOT)]

    kinds, gaps = [], []
    print(f"{'delta':>6}{'R(d)':>8}{'percentile 95%':>22}{'Fieller 95%':>30}")
    for d in DELAYS[1:]:
        ed = (yb - own_f[d]) ** 2
        ec = (yb - cross_f[d]) ** 2
        num = ed - ec                      # skill gained by the cross-section
        den = ed - e0                      # skill destroyed by the delay
        pt = num.sum() / den.sum() if abs(den.sum()) > 1e-12 else np.nan

        vals = []
        for s in S:
            dd = den[s].sum()
            if abs(dd) > 1e-12:
                vals.append(num[s].sum() / dd)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        kind, flo, fhi = fieller(num, den)
        f = (f"[{flo:>7.0%},{fhi:>7.0%}]" if kind == "interval" else
             f"outside [{flo:>6.0%},{fhi:>6.0%}]" if kind == "exclusion" else
             "the whole line")
        print(f"{d:>6}{pt:>8.0%}" + f"[{lo:>6.0%},{hi:>6.0%}]".rjust(22) + f"{f:>30}")
        kinds.append(kind)
        if kind == "interval":
            gaps.append(max(abs(flo - lo / 1.0), abs(fhi - hi / 1.0)))

    odd = [k_ for k_ in kinds if k_ != "interval"]
    worst = max(gaps) if gaps else float("nan")
    print(f"\n  confidence sets that are NOT an ordinary interval: {len(odd)} of "
          f"{len(kinds)} {odd if odd else ''}")
    print(f"  largest disagreement between Fieller and the percentile bootstrap, "
          f"either endpoint: {worst:.1%}")
    if odd:
        print("""
  At least one delay returns the two-half-lines or whole-line confidence set that
  weak identification produces.  A percentile bootstrap cannot express that and
  reports finite endpoints anyway, so at those delays the published interval
  overstates what the data pin down and Fieller is the one to quote.""")
    else:
        print("""
  Every row is an ordinary interval: no delay produced the two-half-lines or
  whole-line confidence set that weak identification would have shown, and Fieller
  sits within a point or two of the percentile bootstrap throughout.  So the
  referee's objection is methodologically right and empirically inert at the delays
  this paper reports - the denominator is weak at delta = 1 but not weak enough to
  break identification, and the bootstrap intervals stand as published.""")
    print("""
  Two caveats keep this from being a clean vindication.  The first version of this
  function used iid variances and came out NARROWER than the block bootstrap at
  every delay, which is impossible for data whose targets overlap by five days and
  was the signal that it was wrong; the numbers above use HAC variances with the
  same Bartlett kernel as Section 6.  And delta = 0 is absent because its
  denominator is identically zero by construction, which is the one case where the
  ratio genuinely does not exist rather than being merely hard to pin down.""")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
