"""
lab01_ceiling.py  -  Stage 2 of the delayed-information project.
Self-contained. NumPy only. No imports from any other file of ours.
Run it from anywhere:  python lab01_ceiling.py      (~20 seconds)

WHAT THIS FILE IS FOR
---------------------
Before we can ask "how much skill do you lose by being late", we need to know
how much skill was available in the first place.  That number is the CEILING,
and this lab derives it in closed form and then checks the derivation against
a simulation.  If the two disagree, everything downstream is worthless, so
this file ends in a PASS/FAIL gate rather than a conclusion.

THE WORLD
---------
Volatility is not observed.  What we see is a noisy proxy of it.  So:

    latent log-variance   x_t = mu + phi (x_{t-1} - mu) + eta * e_t     e ~ N(0,1)
    what we observe       z_t = x_t + s * q_t                           q ~ N(0,1)
    what we predict       y_t = (1/k) sum_{j=1..k} z_{t+j}

x is the thing that carries the signal; s is the measurement noise that makes
the problem hard; y is "average realised variance over the next k days".

THE FOUR PIECES OF MATHEMATICS
------------------------------
(1) The stationary variance of the latent state.
        Var(x) = eta^2 / (1 - phi^2)
    Because Var(x) = phi^2 Var(x) + eta^2 in steady state.

(2) The steady-state Kalman filter.  Write A for the PREDICTED variance
    A = phi^2 P + eta^2, where P is the posterior (filtered) variance.  The
    update gives P = A s^2 / (A + s^2).  Substituting one into the other:

        A^2 + A ( s^2 (1 - phi^2) - eta^2 ) - eta^2 s^2 = 0

    a quadratic we can solve exactly.  Then P = A s^2/(A+s^2) and the Kalman
    gain is K = A/(A+s^2).  No iteration needed - this is the Riccati
    equation solved in closed form for the scalar case.

(3) The ceiling.  By the law of total variance,
        Var(y) = Var(m) + E[Var(y | information)],   m = E[y | information]
    so the best R^2 any forecaster with that information could reach is
    Var(m)/Var(y).  With information stopping delta days ago:

        m_delta = mu + phi^delta * w_k * (xhat_{t-delta} - mu)
        w_k     = (1/k) sum_{j=1..k} phi^j = phi (1 - phi^k) / (k (1 - phi))

    and since Var(x) = Var(xhat) + P  (the filtered estimate and its error are
    orthogonal), Var(xhat) = Var(x) - P.  Therefore

        Var(m_delta) = phi^(2 delta) * w_k^2 * (Var(x) - P)

    The whole delay effect is the single factor phi^(2 delta).  EXACT - there
    is no eigenvalue sum here, which is exactly why we use a continuous state
    rather than a discrete regime chain.

(4) The variance of the target.  The k future observations are correlated:
        Var(sum_{j=1..k} x_{t+j}) = Var(x) * [ k + 2 sum_{d=1}^{k-1} (k-d) phi^d ]
    and the k independent measurement errors add s^2 each, so
        Var(y) = Var(x)*[k + 2 sum_{d=1}^{k-1}(k-d) phi^d]/k^2  +  s^2/k

    Ceiling(delta, k) = phi^(2 delta) * w_k^2 * (Var(x) - P) / Var(y)
"""

import numpy as np

# ----------------------------------------------------------------------
# PARAMETERS.  Chosen for legibility, not realism.  phi = 0.98 gives a
# latent half-life of about 34 trading days, which is in the right region
# for daily log realised variance once measurement error is corrected for.
# ----------------------------------------------------------------------
PHI   = 0.98      # persistence of the latent log-variance
ETA   = 0.30      # innovation standard deviation of the latent state
S     = 0.45      # measurement-noise standard deviation of the proxy
MU    = -9.0      # long-run mean (log variance; exp(-9) ~ 1.2e-4 daily var)
K     = 5         # target horizon: average over the next 5 days
SEED  = 20260912
T_SIM = 400_000   # simulated days for the Monte Carlo check
BURN  = 5_000     # discard while the filter settles
TOL   = 0.02      # relative tolerance for the PASS/FAIL gate (2%)

DELTAS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]


# ======================================================================
# PART 1 - closed forms
# ======================================================================
def var_x(phi, eta):
    """Stationary variance of the latent AR(1) state."""
    return eta**2 / (1.0 - phi**2)


def riccati(phi, eta, s):
    """
    Solve the scalar steady-state Riccati equation exactly.
        A^2 + A(s^2(1-phi^2) - eta^2) - eta^2 s^2 = 0
    Returns (A, P, K): predicted variance, posterior variance, Kalman gain.
    """
    b = s**2 * (1.0 - phi**2) - eta**2
    c = -(eta**2) * s**2
    A = (-b + np.sqrt(b*b - 4.0*c)) / 2.0      # positive root
    P = A * s**2 / (A + s**2)
    K = A / (A + s**2)
    return A, P, K


def w_k(phi, k):
    """Averaging weight: (1/k) sum_{j=1..k} phi^j, in closed form."""
    return phi * (1.0 - phi**k) / (k * (1.0 - phi))


def var_y(phi, eta, s, k):
    """Variance of the k-day average of future observations."""
    vx = var_x(phi, eta)
    acc = k + 2.0 * sum((k - d) * phi**d for d in range(1, k))
    return vx * acc / k**2 + s**2 / k


def ceiling(delta, phi, eta, s, k):
    """Maximum R^2 attainable by any forecaster whose data stops delta days ago."""
    _, P, _ = riccati(phi, eta, s)
    return phi**(2*delta) * w_k(phi, k)**2 * (var_x(phi, eta) - P) / var_y(phi, eta, s, k)


# ======================================================================
# PART 2 - simulation and the exact filter
# ======================================================================
def simulate(n, rng):
    """Generate the latent state and its noisy proxy."""
    e = rng.normal(0.0, ETA, n)
    x = np.empty(n)
    x[0] = MU + rng.normal(0.0, np.sqrt(var_x(PHI, ETA)))   # start stationary
    for t in range(1, n):
        x[t] = MU + PHI * (x[t-1] - MU) + e[t]
    z = x + rng.normal(0.0, S, n)
    return x, z


def kalman_filter(z, phi, mu, gain):
    """
    Steady-state Kalman filter, written out so every step is visible.
        predict:  xbar = mu + phi (xhat_{t-1} - mu)
        update:   xhat = xbar + K (z_t - xbar)
    Using the steady-state gain from the start costs only the first few points.
    """
    n = len(z)
    xhat = np.empty(n)
    xhat[0] = z[0]
    for t in range(1, n):
        xbar = mu + phi * (xhat[t-1] - mu)
        xhat[t] = xbar + gain * (z[t] - xbar)
    return xhat


def build_target(z, k):
    """y[t] = mean of z[t+1 .. t+k].  Cumulative sums so it is O(n), not O(nk)."""
    c = np.concatenate(([0.0], np.cumsum(z)))
    return (c[k+1:] - c[1:-k]) / k          # length len(z) - k


# ======================================================================
# MAIN
# ======================================================================
def main():
    rng = np.random.default_rng(SEED)

    vx = var_x(PHI, ETA)
    A, P, KG = riccati(PHI, ETA, S)
    vy = var_y(PHI, ETA, S, K)
    wk = w_k(PHI, K)

    print("=" * 70)
    print("PART 1  -  closed forms")
    print("=" * 70)
    print(f"  phi                      {PHI}")
    print(f"  latent half-life         {np.log(2)/-np.log(PHI):8.2f} days")
    print(f"  R^2 half-life            {np.log(2)/(-2*np.log(PHI)):8.2f} days   (half, because R^2 is a variance ratio)")
    print(f"  Var(x)  stationary       {vx:10.5f}")
    print(f"  A       predicted var    {A:10.5f}")
    print(f"  P       posterior var    {P:10.5f}")
    print(f"  K       Kalman gain      {KG:10.5f}")
    print(f"  Var(xhat) = Var(x) - P   {vx - P:10.5f}   ({100*(1-P/vx):.1f}% of the state is recoverable)")
    print(f"  w_k     averaging weight {wk:10.5f}")
    print(f"  Var(y)                   {vy:10.5f}")

    # --- simulate once, reuse for every check ---------------------------
    x, z = simulate(T_SIM, rng)
    xhat = kalman_filter(z, PHI, MU, KG)
    y = build_target(z, K)
    n = len(y)

    print()
    print("=" * 70)
    print("PART 2  -  does the filter behave as the algebra says?")
    print("=" * 70)
    checks = []
    mc_vx  = x[BURN:].var()
    mc_P   = (x[BURN:n] - xhat[BURN:n]).var()
    mc_vxh = xhat[BURN:n].var()
    mc_vy  = y[BURN:].var()
    for name, closed, mc in [("Var(x)", vx, mc_vx),
                             ("P  (filter error var)", P, mc_P),
                             ("Var(xhat)", vx - P, mc_vxh),
                             ("Var(y)", vy, mc_vy)]:
        rel = abs(closed - mc) / abs(closed)
        ok = rel < TOL
        checks.append(ok)
        print(f"  {name:<24} closed {closed:10.5f}   simulated {mc:10.5f}   "
              f"rel.err {rel:7.4f}  {'ok' if ok else 'FAIL'}")

    print()
    print("=" * 70)
    print("PART 3  -  the ceiling, and the phi^(2 delta) delay law")
    print("=" * 70)
    print(f"  {'delta':>6}{'closed form':>14}{'simulated':>13}{'rel.err':>10}"
          f"{'ratio to d=0':>14}{'phi^(2d)':>11}")
    base_closed = ceiling(0, PHI, ETA, S, K)
    for d in DELTAS:
        cf = ceiling(d, PHI, ETA, S, K)
        # m_delta uses the belief held delta days before the forecast date
        m = MU + (PHI**d) * wk * (xhat[BURN-d:n-d] - MU)
        mc = m.var() / mc_vy
        rel = abs(cf - mc) / abs(cf)
        ok = rel < TOL
        checks.append(ok)
        print(f"  {d:>6}{cf:>14.6f}{mc:>13.6f}{rel:>10.4f}"
              f"{cf/base_closed:>14.5f}{PHI**(2*d):>11.5f}  {'ok' if ok else 'FAIL'}")
    print("\n  The last two columns are the point: the ceiling falls EXACTLY as")
    print("  phi^(2 delta).  One parameter governs the entire cost of being late.")

    print()
    print("=" * 70)
    print("PART 4  -  the k axis:  delay decay is the SAME for every k")
    print("=" * 70)
    print(f"  {'k':>4}{'ceiling at d=0':>17}{'at d=21':>12}{'ratio':>10}{'phi^42':>10}")
    for k in (1, 5, 21, 63):
        c0 = ceiling(0, PHI, ETA, S, k)
        c21 = ceiling(21, PHI, ETA, S, k)
        print(f"  {k:>4}{c0:>17.6f}{c21:>12.6f}{c21/c0:>10.5f}{PHI**42:>10.5f}")
    print("\n  Changing k changes HOW MUCH signal there is, never how fast delay")
    print("  destroys it.  If real data shows a k-dependent decay RATE, then a")
    print("  single-persistence model is wrong - which is a testable prediction,")
    print("  not an assumption.")

    print()
    print("=" * 70)
    passed = all(checks)
    print(f"GATE: {sum(checks)}/{len(checks)} checks within {TOL:.0%}  ->  "
          f"{'PASS - the apparatus can be trusted' if passed else 'FAIL - do not proceed'}")
    print("=" * 70)
    return passed


if __name__ == "__main__":
    main()
