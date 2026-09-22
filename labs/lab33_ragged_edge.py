"""
lab33_ragged_edge.py - the problem in its native form, and a better estimator.

Cited by Section 11.  Imports lab05_robustness; keep both in labs/.

This was an exploratory file for most of the project's life, on the reasoning
that it answers a different question from the papers'.  That was the wrong call.
It contains the only model in this repository that beats the paper's own
specification at the delays the paper is about, and a limitation section that
says "nothing here bounds how much a better estimator could recover" cannot sit
beside an uncited file that recovers more.  Section 11 now reports it.

WHY THIS IS DIFFERENT FROM EVERYTHING ELSE HERE
------------------------------------------------
Every other file in this project imposes the delay: hold the domestic block
delta days stale, leave the foreign block current, measure.  That is a clean
experiment and it is not how data arrives.  Real panels have a RAGGED EDGE -
each series is current up to its own last observation, the edge is jagged, and
it moves every day.

A state-space model handles that natively.  Write one latent global volatility
factor following an AR(1), have every observed series load on it, and let the
Kalman filter do the only thing it needs to do differently when a series is
missing: drop that row from the measurement equation for that day.  No
imputation, no artificial delay, no separate model per delay.  The filter
produces a posterior for the latent state given whatever happened to be
observable, which is exactly the object the papers have been approximating.

THE MODEL
---------
    state       f(t)   = phi * f(t-1) + eta(t),        Var(eta) = q
    measurement x_i(t) = lambda_i * f(t) + eps_i(t),   Var(eps_i) = r_i

estimated in two steps, after Doz, Giannone and Reichlin: principal components
on the complete training rows give the loadings and the idiosyncratic variances,
an AR(1) on the resulting factor gives phi and q, and the filter is then run
forward.  Two-step rather than full maximum likelihood because it is transparent,
because it needs no EM loop, and because the point here is the ragged edge rather
than the last decimal of the estimates.

HOW THE RAGGED EDGE IS COMPUTED, AND WHY IT IS FAST
----------------------------------------------------
A forecaster at time t knows the target up to t - delta and the peers up to t.
Getting that by re-running a filter from scratch at every t is what a first
version of this file did, and it was unusably slow.  The same object comes from
two passes:

  1. one full causal pass over the whole sample gives x(s) = E[f(s) | everything
     observed up to s], for every s;
  2. for each delta, start from x(s - delta) and propagate delta steps forward
     using the PEERS ONLY, never touching the target.

Step 2 is exactly what the filter would do if the target column were missing for
the last delta days, and the two passes together cost O(n * delta) instead of
O(n * window * refits).

The state-space parameters are estimated ONCE, on the initial training block that
precedes every test day, and then held fixed; only the forecasting ridge refits
on the usual schedule.  Frozen parameters are weaker than refitting them and they
are also unambiguously free of look-ahead, which is the trade made here and is
stated rather than buried.

WHAT IT IS COMPARED AGAINST
----------------------------
The filter's state estimate is fed into the ordinary forecasting regression in
place of the stale domestic level, so the comparison against the paper's
approach is like for like: same target, same horizon, same ridge, same days.

FOUR ARMS, AND THE ONE THAT MATTERS
------------------------------------
  STALE   the domestic model on delta-day-old data
  PAPER   the domestic model plus seven foreign closes, fitted per delay
  FILTER  the Kalman state estimate under a ragged edge, in place of the level
  FILTER+ the state estimate alongside the domestic history

The interesting comparison is FILTER against PAPER.  Both see the same data.
PAPER learns a mapping from foreign closes to the target; FILTER assumes a
structure - one common factor, observed with noise - and infers where the latent
state is.  If the structure is right it should win, especially at long delay
where PAPER has the most parameters to estimate and the least signal.

A HONEST CAVEAT, IN ADVANCE
----------------------------
A single AR(1) factor is a strong restriction.  lab28 found the foreign block IS
essentially one factor, which is what makes the restriction defensible here, but
the target's own dynamics are richer than AR(1) - the HAR structure in every
other file exists precisely because volatility has more than one time scale.  So
a loss for FILTER would be evidence against THIS state space, not against
filtering.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab22_factor_benchmark as F

SEED = 20260914
H = F.H
DELAYS = F.DELAYS
N_BOOT = 2000
BLK = L.BLOCK


def fit_state_space(M):
    """Two-step estimates from a training block M (rows = days, cols = series).

    Returns loadings, idiosyncratic variances, and the AR(1) parameters of the
    factor.  Columns are standardised inside, and the standardisation is returned
    so the filter can apply the same transform to new rows.
    """
    ok = np.isfinite(M).all(axis=1)
    Z = M[ok]
    if len(Z) < 100:
        return None
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    Zs = (Z - mu) / sd
    C = np.cov(Zs, rowvar=False)
    w, V = np.linalg.eigh(C)
    j = int(np.argmax(w))
    lam = V[:, j]
    if lam[0] < 0:                      # pin the sign, as lab22 does
        lam = -lam
    f = Zs @ lam                        # the factor, up to scale
    s = f.std() + 1e-12
    f, lam = f / s, lam * s             # scale so Var(f) = 1 in training
    resid = Zs - np.outer(f, lam)
    r = np.maximum(resid.var(axis=0), 1e-6)
    f0, f1 = f[:-1], f[1:]
    phi = float(np.dot(f0, f1) / (np.dot(f0, f0) + 1e-12))
    phi = max(min(phi, 0.999), -0.999)
    q = float(max(np.var(f1 - phi * f0), 1e-6))
    return dict(mu=mu, sd=sd, lam=lam, r=r, phi=phi, q=q)


def filter_pass(par, M):
    """Full causal pass over the panel; returns filtered mean and variance."""
    mu, sd, lam, r = par["mu"], par["sd"], par["lam"], par["r"]
    phi, q = par["phi"], par["q"]
    Z = (M - mu) / sd
    n = len(Z)
    xs = np.empty(n); ps = np.empty(n)
    x = 0.0
    p = q / max(1 - phi * phi, 1e-6)
    for t in range(n):
        x = phi * x
        p = phi * phi * p + q
        obs = np.isfinite(Z[t])
        if obs.any():
            l, rr, z = lam[obs], r[obs], Z[t][obs]
            s_ = p * float(np.dot(l, l / rr)) + 1.0
            k = (p * (l / rr)) / s_
            x = x + float(np.dot(k, z - l * x))
            p = p * (1.0 - float(np.dot(k, l)))
        xs[t] = x; ps[t] = p
    return xs, ps


def ragged_state(par, M, delta):
    """State at every s given the target to s-delta and the peers to s.

    Starts from the full-information state at s-delta and propagates forward
    delta steps updating on the PEER columns only.
    """
    mu, sd, lam, r = par["mu"], par["sd"], par["lam"], par["r"]
    phi, q = par["phi"], par["q"]
    Z = (M - mu) / sd
    n = len(Z)
    xs, ps = filter_pass(par, M)
    if delta == 0:
        return xs
    out = np.full(n, np.nan)
    lam_p, r_p = lam[1:], r[1:]                 # peers only; column 0 is the target
    for t in range(delta, n):
        x, p = xs[t - delta], ps[t - delta]
        for u in range(t - delta + 1, t + 1):
            x = phi * x
            p = phi * phi * p + q
            obs = np.isfinite(Z[u, 1:])
            if obs.any():
                l, rr, z = lam_p[obs], r_p[obs], Z[u, 1:][obs]
                s_ = p * float(np.dot(l, l / rr)) + 1.0
                k = (p * (l / rr)) / s_
                x = x + float(np.dot(k, z - l * x))
                p = p * (1.0 - float(np.dot(k, l)))
        out[t] = x
    return out


def walk(own, P, y, idx, delta, arm, st=None):
    """One arm.  `st` is the precomputed ragged-edge state, one value per row."""
    out = np.empty(len(idx))
    b = fmu = fsd = None

    def design(rows_own, rows_P, rows_st):
        if arm == "STALE":
            return rows_own
        if arm == "PAPER":
            return np.column_stack([rows_own, rows_P])
        if arm == "FILTER":
            return np.column_stack([rows_st, rows_own[:, 1], rows_own[:, 2]])
        return np.column_stack([rows_st, rows_own])          # FILTER+

    for j_, t in enumerate(idx):
        if j_ % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), cut)
            Xtr = design(own[tr - delta], P[tr],
                         st[tr] if st is not None else np.zeros(len(tr)))
            ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(y[tr])
            Xtr, ytr = Xtr[ok], y[tr][ok]
            if len(ytr) < 200:
                b = None; continue
            fmu, fsd = Xtr.mean(0), Xtr.std(0) + 1e-9
            b = L.cv((Xtr - fmu) / fsd, ytr, "cont")
        xt = design(own[t - delta][None, :], P[t][None, :],
                    np.array([st[t]]) if st is not None else np.zeros(1))[0]
        out[j_] = 0.0 if (b is None or not np.isfinite(xt).all()) \
            else L.p_ridge(b, ((xt - fmu) / fsd)[None, :])[0]
    return out


def boot_ci(yb, fa, fb, rng, n_boot=N_BOOT, block=BLK):
    d = (yb - fb) ** 2 - (yb - fa) ** 2
    n = len(d); nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    offs = np.arange(block)
    o = np.empty(n_boot)
    for i in range(n_boot):
        s = (starts[i][:, None] + offs).ravel()[:n]
        o[i] = d[s].mean()
    lo, hi = np.percentile(o, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints the state-space comparison of Section 11.\n")
    own, P, y, idx, k = F.panel(folder)
    yb = y[idx]
    BENCH = L.bench_mean(y, idx)
    a = own[:, 0]
    M = np.column_stack([a, P])
    print(f"target {F.TARGET}, {len(idx)} test days, {k} foreign peers\n")

    # ---------------- 0. does the filter recover a known state? -----------
    print("=" * 96)
    print("0.  DOES THE FILTER RECOVER A STATE IT IS GIVEN?")
    print("=" * 96)
    print("Simulate the model the filter assumes, hide the first series for the last")
    print("`delta` days, and see how well the filtered state tracks the truth.  With")
    print("nothing hidden it should be near-perfect; the fall-off as days are hidden")
    print("is the quantity the whole exercise turns on.\n")
    rng = np.random.default_rng(SEED)
    n_sim, k_sim = 3000, 8
    phi_t, q_t = 0.97, 0.06
    f_t = np.empty(n_sim); f_t[0] = 0.0
    for t in range(1, n_sim):
        f_t[t] = phi_t * f_t[t - 1] + rng.normal(0, np.sqrt(q_t))
    lam_t = rng.uniform(0.6, 1.0, k_sim)
    Xs = np.outer(f_t, lam_t) + rng.normal(0, 0.5, (n_sim, k_sim))
    par_t = fit_state_space(Xs[:2000])
    print(f"  true phi {phi_t:.3f} -> estimated {par_t['phi']:.3f}")
    print(f"{'days hidden':>13}{'corr(filtered, true)':>24}")
    print("  (measured on one FIXED 400-day window for every row.  A first version")
    print("   averaged over the whole block, where hiding one of eight series at the")
    print("   end moved nothing; a second used a window that grew with the number of")
    print("   days hidden, so the rows were not comparable with each other.)\n")
    Msim = Xs[2000:]
    ftrue = f_t[2000:]
    WIN = 400          # the SAME window for every row, so the rows compare
    for hide in (0, 1, 5, 21, 55):
        st = ragged_state(par_t, Msim, hide)
        ok = np.isfinite(st)
        tail = np.zeros(len(st), bool); tail[-WIN:] = True
        ok = ok & tail
        c = float(np.corrcoef(st[ok], ftrue[ok])[0, 1])
        print(f"{hide:>13}{c:>24.4f}")
    c0 = float(np.corrcoef(*(lambda st: (st[-WIN:], ftrue[-WIN:]))(
        ragged_state(par_t, Msim, 0)))[0, 1])
    c55 = float(np.corrcoef(*(lambda st: (st[-WIN:], ftrue[-WIN:]))(
        ragged_state(par_t, Msim, 55)))[0, 1])
    print(f"\n  hiding the target for 55 days costs {c0 - c55:.4f} of correlation.")
    if abs(c0 - c55) < 0.02:
        print("  That is not a null result, it is the mechanism.  Seven other series")
        print("  load on the same factor, so removing one of eight measurements barely")
        print("  moves the state estimate however long it is hidden.  Whether the real")
        print("  cross-section behaves like that is what part A tests - here it is only")
        print("  established that the filter behaves as its own assumptions imply.")
    else:
        print("  The estimate degrades materially as the target is hidden, so the")
        print("  cross-section is not pinning the factor on its own and part A's")
        print("  comparison is between two weak reconstructions rather than one strong")
        print("  one and one stale value.")

    # ---------------- A ---------------------------------------------------
    print("\n" + "=" * 96)
    print("A.  THE RAGGED-EDGE FILTER AGAINST THE PAPER'S REGRESSION")
    print("=" * 96)
    print("STALE   domestic model on delta-day-old data")
    print("PAPER   domestic model plus seven foreign closes, fitted per delay")
    print("FILTER  Kalman state under a ragged edge, in place of the stale level")
    print("FILTER+ the state estimate alongside the domestic history\n")
    par = fit_state_space(M[:L.TRAIN + L.VAL])
    print(f"  state space estimated once on the first {L.TRAIN + L.VAL} rows: "
          f"phi = {par['phi']:.3f}, q = {par['q']:.4f}")
    print(f"  loading on the target series: {par['lam'][0]:.3f}; "
          f"peer loadings {par['lam'][1:].min():.3f} to {par['lam'][1:].max():.3f}\n")
    STATE = {d: ragged_state(par, M, d) for d in DELAYS}

    ARMS = ["STALE", "PAPER", "FILTER", "FILTER+"]
    print(f"{'delta':>6}" + "".join(f"{arm:>10}" for arm in ARMS))
    R = {}
    for d in DELAYS:
        row = f"{d:>6}"
        for arm in ARMS:
            f = walk(own, P, y, idx, d, arm, STATE[d])
            R[(d, arm)] = f
            row += f"{F.r2(yb, f, BENCH):>10.4f}"
        print(row)

    # ---------------- B ---------------------------------------------------
    print("\n" + "=" * 96)
    print("B.  IS THE STRUCTURE WORTH IMPOSING?")
    print("=" * 96)
    print("Positive favours the filter. Both arms see the same data; the filter")
    print("assumes one common factor observed with noise, the regression assumes")
    print("nothing and estimates ten coefficients.\n")
    print(f"{'delta':>6}{'FILTER - PAPER':>17}{'95% CI':>24}{'winner':>11}")
    fil, pap = [], []
    for d in DELAYS:
        fa, fb = R[(d, "FILTER")], R[(d, "PAPER")]
        dm, lo, hi = boot_ci(yb, fa, fb, np.random.default_rng(SEED + d))
        if lo > 0:
            fil.append(d)
        if hi < 0:
            pap.append(d)
        w = "FILTER" if lo > 0 else ("PAPER" if hi < 0 else "neither")
        print(f"{d:>6}{F.r2(yb, fa, BENCH) - F.r2(yb, fb, BENCH):>+17.4f}"
              + f"[{lo:>+9.5f},{hi:>+9.5f}]".rjust(24) + f"{w:>11}")
    print(f"\n  delays where the filter significantly wins: {len(fil)} of {len(DELAYS)} "
          f"{fil if fil else ''}")
    print(f"  delays where the regression significantly wins: {len(pap)} of {len(DELAYS)} "
          f"{pap if pap else ''}")

    if fil and not pap:
        print("\n  The structure pays. Assuming one common factor observed with noise beats")
        print("  estimating a free mapping, which is what Section 5's estimation-cost")
        print("  result predicts once the restriction is the RIGHT one.")
    elif pap and not fil:
        print("\n  The restriction costs more than it saves. A single AR(1) factor is too")
        print("  poor a description of volatility's dynamics - which have more than one")
        print("  time scale, as the HAR structure everywhere else in this project assumes")
        print("  - so this is evidence against THIS state space, not against filtering.")
    elif fil and pap:
        print("\n  Each wins somewhere, and the crossover delay is the finding.")
    else:
        print("\n  Neither separates at any delay: a strong structural restriction and a")
        print("  free regression reach the same place from the same data.")

    # ---------------- C ---------------------------------------------------
    print("\n" + "=" * 96)
    print("C.  WHAT THE FILTER IS ACTUALLY DOING")
    print("=" * 96)
    print("Correlation between the filtered state and the TRUE current domestic state")
    print("that the forecaster is not allowed to see. This is the reconstruction")
    print("quality itself, separate from whether it helps the forecast.\n")
    print(f"{'delta':>6}{'corr(state, truth)':>22}{'corr(stale, truth)':>22}{'gain':>9}")
    gains = []
    for d in DELAYS:
        st = STATE[d][idx]
        truth, stale = a[idx], a[idx - d]
        ok = np.isfinite(st) & np.isfinite(truth) & np.isfinite(stale)
        c1 = float(np.corrcoef(st[ok], truth[ok])[0, 1])
        c2 = float(np.corrcoef(stale[ok], truth[ok])[0, 1])
        gains.append((d, c1 - c2))
        print(f"{d:>6}{c1:>22.4f}{c2:>22.4f}{c1 - c2:>+9.4f}")
    pos = [d for d, g in gains if g > 0]
    print(f"\n  delays where the filtered state tracks the unobservable truth better")
    print(f"  than the stale value does: {len(pos)} of {len(DELAYS)} {pos if pos else ''}")
    print("\n  This is reconstruction quality, separate from whether it helps the")
    print("  forecast.  A method can reconstruct well and still lose the forecasting")
    print("  comparison, and separating the two is the reason this part exists.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
