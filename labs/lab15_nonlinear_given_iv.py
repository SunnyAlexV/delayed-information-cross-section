"""
lab15_nonlinear_given_iv.py - is the foreign block redundant given implied
volatility only because the estimator is linear?

Imports lab05_robustness and lab08_implied_vol; keep all three in labs/.
Runtime about eight and a half minutes.

THE OBJECTION, STATED PRECISELY
-------------------------------
Section 7 concludes that given VIX and VDAX the foreign cross-section adds
nothing: lab08's 'foreign | iv' column is negative at all ten delays.  A
referee objects that this conclusion is tied to linear specifications, and that
a non-linear model might find cross-market spillover patterns that bypass the
single dominant global volatility factor.

lab09 already tested non-linearity - and found nothing that clears its own
estimation cost - but it tested it on own + foreign.  It never saw the implied
volatility block.  So the redundancy conclusion specifically, which is the one
the objection names, has in fact only ever been established under ridge.  That
is a real gap and this file closes it.

It is worth being clear about what lab09 does and does not already settle,
because two different inferences are easy to confuse:

  lab09 answers   "can a non-linear map of the foreign block beat a linear map
                   of the foreign block, when neither has implied volatility?"
  lab15 answers   "can a non-linear map of the foreign block add anything to a
                   forecaster who ALREADY HAS implied volatility?"

The second does not follow from the first in either direction.  A non-linear
foreign map could be redundant with the linear one and still add something on
top of VIX, or vice versa.

THE ARMS - all hold own domestic data delta days stale, all see VIX and VDAX
----------------------------------------------------------------------------
  IV          own(3) + iv(2).                            5 features
  IV+F        IV plus the seven foreign closes.  Linear. 12
  IV+F+FAC    IV+F plus g, g^2, g x own_daily, g x VIX,  16
              where g is the equal-weighted foreign mean.  This is lab09's
              FACTOR arm carried onto the IV feature set, plus the one term
              lab09 could not have: an interaction between the global factor
              and implied volatility.  If foreign closes matter only when the
              options market is already nervous - a plausible reading of the
              objection - this term is where that would show.
  IV+F+RFF20  IV+F plus 20 random Fourier features of the foreign block.  32
  IV+F+RFF60  the same with 60.                           72

Every arm keeps the columns of the one before it, so the ladder is strictly
nested: IV inside IV+F inside each non-linear arm.  That is deliberate and it
matters twice.  It makes Giacomini-White the right test at every comparison and
Diebold-Mariano the wrong one, as in lab06 and lab09.  And it means a non-linear
arm can always reproduce the linear fit exactly by zeroing its extra
coefficients, so when one scores worse it is not because the added features
displaced the linear ones - it is because estimating them cost more than they
returned.

WHY RANDOM FOURIER FEATURES
---------------------------
The objection names a shallow neural network.  Fitting one would break the
"NumPy and pandas, nothing else" promise, and writing backpropagation out by
hand would put an unaudited optimiser between the data and the answer.  Random
Fourier features (Rahimi and Recht, 2007) give the same object without either
cost: z(x) = sqrt(2/m) cos(Wx + b) with W and b drawn once from fixed
distributions IS a one-hidden-layer network with a cosine activation, a random
first layer, and an output layer fitted by ridge.  As m grows it approximates a
Gaussian RBF kernel, so the map it can represent is not a polynomial and not
restricted to any particular interaction order.

What it is not: a network whose first layer is LEARNED.  That is a genuine
limitation and it is stated rather than hidden.  Two things bound how much it
matters.  First, a learned first layer has strictly more free parameters than a
fixed one, so whatever it gains it must pay for on the same estimation-cost
axis that lab07 measured and that the RFF60 arm here exercises directly - the
argument extends a fortiori, not by hope.  Second, the two capacities below
bracket the question empirically: if the answer were "more flexibility finds
more signal", RFF60 would beat RFF20, and the direction of that comparison is
reported whichever way it goes.

W and b are drawn ONCE, outside the walk-forward, from a fixed seed.  They are
therefore a fixed function of the foreign block, not something estimated on the
test set; only the output layer is refitted per fold, exactly like the ridge
coefficients in every other lab.  The foreign block is standardised first, using
the mean and standard deviation of the BURN-IN period alone - the days before
the first forecast origin - so no test day contributes to the scaling.  The
bandwidth is the median heuristic computed on that same burn-in.

WHY THE STRICT TIMING ONLY
--------------------------
lab08 runs two VIX timings.  The GENEROUS one hands the options market a full
extra day of information, which makes implied volatility stronger and therefore
makes the foreign block MORE redundant, not less.  STRICT is the arm least
favourable to the redundancy conclusion and so the right one to attack it on.

WHAT WOULD COUNT AS A PROBLEM
-----------------------------
If any non-linear arm beats IV by a Giacomini-White statistic above 1.96 at any
delay, Section 7's redundancy conclusion is a statement about ridge rather than
about information, and has to be rewritten.  If none does, the conclusion holds
against the flexibility actually tested, and the paper says exactly that -
which is still weaker than "no non-linear model could ever find anything", and
is worded that way on purpose.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L

BENCH = None          # set in main(); read by the scorers below
import lab08_implied_vol as IV

SEED = 20260913
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 3, 5, 13, 21, 55]
N_DRAW = 25            # surrogate draws for the estimation-cost column
M_RFF = (20, 60)       # hidden width of the two random-feature arms
ARMS = ["IV", "IV+F", "IV+F+FAC", "IV+F+RFF20", "IV+F+RFF60"]

# THE PENALTY GRID, AND WHY THERE ARE TWO
# ---------------------------------------
# Every other lab selects the ridge penalty from lab05's LAMBDAS, which stops
# at 30.  lab12 part B found validation already landing on that upper edge 52%
# to 84% of the time with TEN features, and part D showed that widening the grid
# does not move the answer - for ten features.
#
# The widest arm here carries seventy-two.  A grid that cannot go past 30 would
# force it to be under-regularised, and it would then lose for a reason that has
# nothing to do with whether the information is there.  Concluding "non-linearity
# does not help" from a test that would not let it regularise is precisely the
# unfairness the objection is entitled to complain about.
#
# So the comparison is run twice.  PAPER is lab05's grid, which reproduces lab08
# exactly and is checked against it below.  WIDE extends it by a further factor
# of thirty.  The verdict is read off WIDE, because that is the fair test; PAPER
# is kept because reproducing lab08 digit for digit is what proves this file is
# measuring the same thing Section 7 measured.
GRIDS = {"PAPER": list(L.LAMBDAS),
         "WIDE": list(L.LAMBDAS) + [100.0, 300.0, 1000.0]}


class penalty:
    """Temporarily swap the ridge grid lab05.cv selects from."""

    def __init__(self, grid):
        self.grid = grid

    def __enter__(self):
        self.old = L.LAMBDAS
        L.LAMBDAS = self.grid

    def __exit__(self, *exc):
        L.LAMBDAS = self.old
        return False


def hac_se(d, lag=None):
    """Newey-West standard error of a mean, at lab05's measured bandwidth.

    The default is L.HAC_LAG, not the automatic 4 (n/100)^(2/9) rule.  That
    rule gives 9 lags here and truncates the kernel while the loss
    differential still carries 0.52 autocorrelation, which understates every
    standard error in this project by up to 54% and every statistic formed
    from one in the paper's favour.  lab58 part 6 measures it.
    """
    n = len(d); d = d - d.mean()
    if lag is None:
        lag = L.HAC_LAG
    s = (d @ d) / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * ((d[k:] @ d[:-k]) / n)
    return np.sqrt(max(s, 1e-18) / n)


def norm_p(z):
    from math import erf, sqrt
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


class RFF:
    """One fixed random layer: x -> sqrt(2/m) cos(W x + b).

    Everything here is drawn once and never refitted, so this object is a fixed
    feature map.  The scaling statistics come from the burn-in only.
    """

    def __init__(self, P, burn_end, m, rng):
        B = P[:burn_end]
        B = B[np.isfinite(B).all(axis=1)]
        self.mu = B.mean(axis=0)
        self.sd = B.std(axis=0) + 1e-9
        Z = (B - self.mu) / self.sd
        # median heuristic: bandwidth = median pairwise distance on the burn-in
        take = Z[rng.choice(len(Z), size=min(500, len(Z)), replace=False)]
        d2 = ((take[:, None, :] - take[None, :, :]) ** 2).sum(-1)
        med = np.sqrt(np.median(d2[np.triu_indices(len(take), 1)]))
        self.W = rng.normal(size=(P.shape[1], m)) / max(med, 1e-9)
        self.b = rng.uniform(0, 2 * np.pi, size=m)
        self.m = m

    def __call__(self, rows):
        Z = (rows - self.mu) / self.sd
        return np.sqrt(2.0 / self.m) * np.cos(Z @ self.W + self.b)


def design(own_rows, P_rows, iv_rows, kind, rff):
    """Feature matrix for one arm.  own_rows is already delayed; the rest current."""
    base = np.column_stack([own_rows, iv_rows])
    if kind == "IV":
        return base
    X = np.column_stack([base, P_rows])
    if kind == "IV+F":
        return X
    if kind == "IV+F+FAC":
        g = P_rows.mean(axis=1)
        return np.column_stack([X, g, g ** 2, g * own_rows[:, 0], g * iv_rows[:, 0]])
    if kind.startswith("IV+F+RFF"):
        return np.column_stack([X, rff[int(kind[8:])](P_rows)])
    raise ValueError(kind)


def walk(own, P, iv, y, idx, delta, kind, rff):
    out = np.empty(len(idx)); b = mu = sd = None
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), cut)
            Xtr = design(own[tr - delta], P[tr], iv[tr], kind, rff)
            ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(y[tr])
            Xtr, ytr = Xtr[ok], y[tr][ok]
            if len(ytr) < 200:
                b = None; continue
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            b = L.cv((Xtr - mu) / sd, ytr, "cont")
        xt = design(own[t - delta][None, :], P[t][None, :], iv[t][None, :],
                    kind, rff)[0]
        out[j] = 0.0 if (b is None or not np.isfinite(xt).all()) \
            else L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
    return out


def r2(yb, f, bench):
    return 1 - ((yb - f) ** 2).sum() / ((yb - bench) ** 2).sum()


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    D, ivdf, peers = IV.build(folder, strict=True)
    own, P, iv, y, idx = IV.make(D, ivdf, peers)
    yb = y[idx]
    global BENCH
    BENCH = L.bench_mean(y, idx)
    k = P.shape[1]
    burn_end = int(idx[0])

    rng = np.random.default_rng(SEED)
    rff = {m: RFF(P, burn_end, m, rng) for m in M_RFF}

    print(f"\ntarget {TARGET}, {len(idx)} test days, {k} foreign peers, "
          f"VIX at t-1 (STRICT)")
    print(f"burn-in for the random layer: rows 0 to {burn_end}, "
          f"all strictly before the first forecast origin")
    widths = {"IV": 5, "IV+F": 5 + k, "IV+F+FAC": 9 + k,
              "IV+F+RFF20": 5 + k + 20, "IV+F+RFF60": 5 + k + 60}
    print("features: " + ",  ".join(f"{a} {widths[a]}" for a in ARMS))
    for g, lams in GRIDS.items():
        print(f"ridge grid {g:>5}: {lams}")

    # --- forecasts, under both penalty grids ------------------------------
    F = {}
    for g, lams in GRIDS.items():
        with penalty(lams):
            for d in DELAYS:
                for a in ARMS:
                    F[(g, a, d)] = walk(own, P, iv, y, idx, d, a, rff)

    # --- does this file measure the same thing Section 7 measured? --------
    print("\n" + "=" * 86)
    print("0.  RECONCILIATION WITH lab08")
    print("=" * 86)
    print("Under the paper's own grid the two linear arms here are lab08's '+iv' and")
    print("'+both' columns.  The reference is not pasted in: lab08's own walk() is")
    print("called on the same panel and its output compared.  This file builds its")
    print("design matrix a different way - one design() call per arm rather than a")
    print("list of blocks - so agreement is a real check on both, and if it fails")
    print("nothing below is about Section 7.\n")
    print(f"{'delta':>6}{'IV here':>10}{'lab08 +iv':>12}"
          f"{'IV+F here':>12}{'lab08 +both':>13}")
    agree = True
    with penalty(GRIDS["PAPER"]):
        for d in DELAYS:
            a1 = r2(yb, F[("PAPER", "IV", d)], BENCH)
            a2 = r2(yb, F[("PAPER", "IV+F", d)], BENCH)
            b1 = r2(yb, IV.walk(own, [iv], y, idx, d), BENCH)
            b2 = r2(yb, IV.walk(own, [P, iv], y, idx, d), BENCH)
            ok = abs(a1 - b1) < 5e-9 and abs(a2 - b2) < 5e-9
            agree &= ok
            print(f"{d:>6}{a1:>10.4f}{b1:>12.4f}{a2:>12.4f}{b2:>13.4f}"
                  f"{'   ok' if ok else '   MISMATCH'}")
    print(f"\n  {'both arms reconcile with lab08 to within 5e-9' if agree else 'RECONCILIATION FAILED'}")

    # --- the objection, tested -------------------------------------------
    print("\n" + "=" * 86)
    print("A.  DOES ANY NON-LINEAR FOREIGN MAP ADD TO A FORECASTER WHO HAS VIX?")
    print("=" * 86)
    print("Every arm is compared against IV - own stale domestic data plus implied")
    print("volatility.  dR2 and the GW statistic ARE the redundancy claim; lab08 made")
    print("them for IV+F alone, under ridge and under the paper's grid.")
    print("A positive GW statistic above 1.96 would overturn Section 7.\n")
    tab = {}
    for g in GRIDS:
        print(f"  --- ridge grid: {g} " + "-" * (63 - len(g)))
        print(f"{'delta':>6}{'R2 IV':>9}" + "".join(
            f"{a.replace('IV+F', 'F'):>22}" for a in ARMS[1:]))
        print(f"{'':>6}{'':>9}" + "".join(
            f"{'dR2   GW z      p':>22}" for _ in ARMS[1:]))
        for d in DELAYS:
            row = f"{d:>6}{r2(yb, F[(g, 'IV', d)], BENCH):>9.4f}"
            for a in ARMS[1:]:
                dr = r2(yb, F[(g, a, d)], BENCH) - r2(yb, F[(g, "IV", d)], BENCH)
                dl = (yb - F[(g, "IV", d)]) ** 2 - (yb - F[(g, a, d)]) ** 2
                z = dl.mean() / hac_se(dl)
                tab[(g, a, d)] = (dr, z)
                row += f"{dr:>+9.4f}{z:>7.2f}{norm_p(z):>6.3f}"
            print(row)
        print()

    # --- estimation cost of the added flexibility ------------------------
    print("=" * 86)
    print("B.  WHAT THE ADDED FLEXIBILITY COSTS")
    print("=" * 86)
    print("As in lab07 and lab09: the foreign block is replaced by AR(1) surrogates")
    print("with the same persistence and the same cross-market innovation covariance")
    print("but no predictive content, so any R2 difference between arms is what the")
    print("extra regressors take whether or not they help.  Measured against IV+F,")
    print("because the question is the cost of BENDING the foreign block rather than")
    print("the cost of carrying it, and under the WIDE grid, because that is the grid")
    print("the verdict is read from.\n")
    fin = np.isfinite(P).all(axis=1)
    phi = np.empty(k); mu_p = np.empty(k); resid = []
    for j in range(k):
        v = P[fin, j]; v0, v1 = v[:-1] - v.mean(), v[1:] - v.mean()
        phi[j] = (v0 @ v1) / (v0 @ v0)
        resid.append(v1 - phi[j] * v0)
        mu_p[j] = v.mean()
    Chol = np.linalg.cholesky(np.cov(np.column_stack(resid), rowvar=False)
                              + 1e-12 * np.eye(k))
    srng = np.random.default_rng(SEED + 5)

    def surrogate():
        S = np.empty_like(P); e = srng.normal(size=P.shape) @ Chol.T
        S[0] = mu_p + e[0] / np.sqrt(np.maximum(1 - phi ** 2, 1e-6))
        for t in range(1, len(P)):
            S[t] = mu_p + phi * (S[t - 1] - mu_p) + e[t]
        S[~fin] = np.nan
        return S

    draws = [surrogate() for _ in range(N_DRAW)]
    print(f"{'delta':>6}" + "".join(
        f"{a.replace('IV+F', 'F'):>18}" for a in ARMS[2:]))
    print(f"{'':>6}" + "".join(f"{'cost     net':>18}" for _ in ARMS[2:]))
    with penalty(GRIDS["WIDE"]):
        for d in DELAYS:
            row = f"{d:>6}"
            base = [r2(yb, walk(own, S, iv, y, idx, d, "IV+F", rff), BENCH) for S in draws]
            for a in ARMS[2:]:
                c = float(np.mean([r2(yb, walk(own, S, iv, y, idx, d, a, rff), BENCH) - b0
                                   for S, b0 in zip(draws, base)]))
                net = (r2(yb, F[("WIDE", a, d)], BENCH)
                       - r2(yb, F[("WIDE", "IV+F", d)], BENCH)) - c
                row += f"{c:>+10.4f}{net:>+8.4f}"
            print(row)

    # --- capacity ---------------------------------------------------------
    print("\n" + "=" * 86)
    print("C.  DOES MORE FLEXIBILITY FIND MORE?")
    print("=" * 86)
    print("If the objection were right, the wider random layer would beat the")
    print("narrower one.  Positive means RFF60 is better.  WIDE grid.\n")
    print(f"{'delta':>6}{'R2 RFF20':>11}{'R2 RFF60':>11}{'difference':>13}")
    for d in DELAYS:
        r20 = r2(yb, F[("WIDE", "IV+F+RFF20", d)], BENCH)
        r60 = r2(yb, F[("WIDE", "IV+F+RFF60", d)], BENCH)
        print(f"{d:>6}{r20:>11.4f}{r60:>11.4f}{r60 - r20:>+13.4f}")

    print("\n" + "=" * 86)
    print("VERDICT   (read from the WIDE grid)")
    print("=" * 86)
    for a in ARMS[1:]:
        win = [d for d in DELAYS if tab[("WIDE", a, d)][1] > 1.96]
        lose = [d for d in DELAYS if tab[("WIDE", a, d)][1] < -1.96]
        print(f"  {a:>11}: adds to IV at {len(win)} of {len(DELAYS)} delays"
              f"{' (delta ' + ', '.join(map(str, win)) + ')' if win else ''};"
              f"  IV beats it at {len(lose)}"
              f"{' (delta ' + ', '.join(map(str, lose)) + ')' if lose else ''}")
    print()
    print("Read the GW column, not dR2 and not 'net'.  'net' is a difference of two")
    print("estimated quantities carrying no interval, which is the trap lab07")
    print("documented; it is printed because the cost of flexibility is the mechanism,")
    print("but no claim rests on its sign.")
    print()
    print("What this supports, if the GW column holds: the redundancy of the foreign")
    print("block given implied volatility is not an artefact of the linear estimator.")
    print("It survives a quadratic global factor, an interaction between that factor")
    print("and implied volatility, and a random-feature approximation to an RBF kernel")
    print("at two capacities and under a penalty grid wide enough to regularise it.")
    print()
    print("What it cannot support: that no non-linear model could ever find anything.")
    print("A learned first layer was not tested, and this sample could not pay for")
    print("one.  That is the same weaker claim lab09 draws, now covering the case")
    print("Section 7 actually rests on.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
