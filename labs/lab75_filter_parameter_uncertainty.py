"""
lab75_filter_parameter_uncertainty.py - does the filter still win when its own
parameters are not treated as known?

Imports lab05_robustness, lab22_factor_benchmark, lab33_ragged_edge and
lab50_filtered_headline; keep all five in labs/.  Runtime about ten minutes.

THE OBJECTION
-------------
Section 11 reports the headline inside a filtered specification: the state-space
model beats the paper's ridge as the carrier of the cross-section at the three
longest delays, and the rate at eleven weeks moves up by about four points.  The
state space is fitted once, on the first training block, and then frozen.

The filtering literature has a standing warning about exactly this.  A filter
run with parameters estimated from the same data, and then treated as known,
reports a state it is more certain about than it has any right to be, and the
apparent gain can be an artefact of that certainty rather than information
(Monoyios, on filtering with parameter uncertainty).  The test is whether the
gain survives when the parameters are re-estimated as the window grows, and
whether the filter's own innovations behave the way the model says they must.

THREE QUESTIONS, IN ORDER
-------------------------
  1. REAL-TIME PARAMETERS.  Re-estimate the state space on an expanding window
     and give each test day the vintage a forecaster could have had on that day,
     exactly as the forecasting ridge is refitted.  If the filter's advantage is
     a frozen-parameter artefact, this is where it goes.
  2. HOW MUCH THE PARAMETERS ARE WORTH KNOWING.  Resample the estimation block
     in moving blocks, refit, re-filter, and report the spread of the
     reconstructed state against the spread the filter reports for itself.  A
     filter whose parameter band is wide next to its own posterior band is one
     whose stated precision is mostly assumption.
  3. WHETHER THE INNOVATIONS ARE INNOVATIONS.  Standardised one-step
     innovations should have unit variance and no serial correlation.  If their
     variance is above one the model is understating its own uncertainty, and
     the band in question 2 is the floor rather than the answer.

Nothing in the paper changes on the strength of this file; it is a test of a
result the paper already reports, run after submission.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab22_factor_benchmark as F
import lab33_ragged_edge as RE
import lab50_filtered_headline as FH

SEED = 20261015
H = L.HORIZON
DELAYS = [0, 3, 5, 13, 21, 55]
HEAD = 55
TRAIN0 = L.TRAIN + L.VAL          # the block lab50 freezes on
VINTAGE = 252                     # how often the state space is re-estimated
N_ENS = 200                       # parameter draws for the uncertainty band
N_BOOT = 1500


def innovations(par, M):
    """Standardised one-step innovations, one column per series.

    The same recursion filter_pass runs, with nu(t) = z(t) - lam * x(t|t-1) and
    its variance kept, so the diagnostics below read the model's own numbers
    rather than a residual computed after the fact.
    """
    mu, sd, lam, r = par["mu"], par["sd"], par["lam"], par["r"]
    phi, q = par["phi"], par["q"]
    Z = (M - mu) / sd
    n, k = Z.shape
    out = np.full((n, k), np.nan)
    x = 0.0
    p = q / max(1 - phi * phi, 1e-6)
    for t in range(n):
        x = phi * x
        p = phi * phi * p + q
        obs = np.isfinite(Z[t])
        if obs.any():
            l, rr, z = lam[obs], r[obs], Z[t][obs]
            nu = z - l * x                      # innovation
            fv = p * l * l + rr                 # its variance, per series
            out[t, obs] = nu / np.sqrt(fv)
            s_ = p * float(np.dot(l, l / rr)) + 1.0
            kg = (p * (l / rr)) / s_
            x = x + float(np.dot(kg, nu))
            p = p * (1.0 - float(np.dot(kg, l)))
    return out


def ljung_box(v, lags):
    """Q statistic and its degrees of freedom for a standardised series."""
    v = v[np.isfinite(v)]
    v = v - v.mean()
    n = len(v)
    q = 0.0
    for k in range(1, lags + 1):
        rk = float(v[:n - k] @ v[k:]) / float(v @ v)
        q += rk * rk / (n - k)
    return n * (n + 2) * q, lags


def chi2_upper(q, df):
    """P(chi2_df > q), by the Wilson-Hilferty normal approximation."""
    from math import erf, sqrt
    z = ((q / df) ** (1.0 / 3.0) - (1 - 2.0 / (9 * df))) / sqrt(2.0 / (9 * df))
    return 0.5 * (1 - erf(z / sqrt(2.0)))


def simulate(par, n, rng):
    """One panel of length n from the fitted model, in the original units."""
    phi, q, lam, r = par["phi"], par["q"], par["lam"], par["r"]
    f = np.empty(n)
    f[0] = rng.normal(0.0, np.sqrt(q / max(1 - phi * phi, 1e-6)))
    e = rng.normal(0.0, np.sqrt(q), size=n)
    for t in range(1, n):
        f[t] = phi * f[t - 1] + e[t]
    Z = np.outer(f, lam) + rng.normal(0.0, np.sqrt(r), size=(n, len(lam)))
    return Z * par["sd"] + par["mu"]


def block_idx(n, rng, block):
    nb = int(np.ceil(n / block))
    st = rng.integers(0, max(1, n - block + 1), size=nb)
    return (st[:, None] + np.arange(block)).ravel()[:n]


def main(folder=None):
    folder = L.find_folder(folder)
    rng = np.random.default_rng(SEED)
    print(f"data folder: {folder}")
    print("Prints the filtered headline under parameter uncertainty.\n")

    own, P, y, idx, k = F.panel(folder)
    yb = y[idx]
    BENCH = L.bench_mean(y, idx)
    M = np.column_stack([own[:, 0], P])
    n = len(idx)
    print(f"target {F.TARGET}, {n} test days, {k} foreign peers")

    par0 = RE.fit_state_space(M[:TRAIN0])
    if par0 is None:
        raise SystemExit("the state space could not be fitted on the training block")
    print(f"frozen fit, first {TRAIN0} rows: phi = {par0['phi']:.4f}, "
          f"q = {par0['q']:.4f}, loading on the target = {par0['lam'][0]:.4f}\n")

    # ---------------- A ----------------------------------------------------
    print("=" * 96)
    print("A.  THE SAME MODELS WITH PARAMETERS A FORECASTER COULD HAVE HAD")
    print("=" * 96)
    print("The state space is re-estimated every", VINTAGE, "rows on everything")
    print("strictly before the forecast's own training cut, and each test day")
    print("uses the vintage in force on that day.  The forecasting ridge on top")
    print("is unchanged, so the only difference from Section 11 is that the")
    print("filter's parameters are no longer known in advance.\n")

    cuts = list(range(TRAIN0, len(M), VINTAGE))
    vintages = {}
    for c in cuts:
        pv = RE.fit_state_space(M[:c])
        if pv is not None:
            vintages[c] = pv
    print(f"  {len(vintages)} vintages, from row {min(vintages)} to "
          f"{max(vintages)}")
    phis = np.array([v["phi"] for v in vintages.values()])
    lam0 = np.array([v["lam"][0] for v in vintages.values()])
    print(f"  phi across vintages {phis.min():.4f} to {phis.max():.4f}; "
          f"target loading {lam0.min():.3f} to {lam0.max():.3f}\n")

    st_frozen = {d: RE.ragged_state(par0, M, d) for d in DELAYS}
    st_live = {}
    for d in DELAYS:
        per_vintage = {c: RE.ragged_state(pv, M, d) for c, pv in vintages.items()}
        s_live = np.full(len(M), np.nan)
        for t in range(len(M)):
            avail = [c for c in vintages if c <= t - d - H]
            if not avail:
                continue
            s_live[t] = per_vintage[max(avail)][t]
        st_live[d] = s_live

    f_own = {d: RE.walk(own, P, y, idx, d, "STALE") for d in DELAYS}
    f_pap = {d: RE.walk(own, P, y, idx, d, "PAPER") for d in DELAYS}
    f_frz = {d: RE.walk(own, P, y, idx, d, "FILTER", st_frozen[d]) for d in DELAYS}
    f_liv = {d: RE.walk(own, P, y, idx, d, "FILTER", st_live[d]) for d in DELAYS}
    S = {}
    for nm, f in (("own", f_own), ("ridge", f_pap), ("frozen", f_frz),
                  ("live", f_liv)):
        S[nm] = {d: FH.r2(yb, f[d], bench=BENCH) for d in DELAYS}

    print(f"{'delta':>6}{'own only':>11}{'cross, ridge':>15}"
          f"{'filter, frozen':>17}{'filter, live':>15}{'live - ridge':>15}")
    for d in DELAYS:
        print(f"{d:>6}{S['own'][d]:>11.4f}{S['ridge'][d]:>15.4f}"
              f"{S['frozen'][d]:>17.4f}{S['live'][d]:>15.4f}"
              f"{S['live'][d] - S['ridge'][d]:>+15.4f}")
    win_f = [d for d in DELAYS if S["frozen"][d] > S["ridge"][d]]
    win_l = [d for d in DELAYS if S["live"][d] > S["ridge"][d]]
    print(f"\n  frozen parameters: the filter is ahead at {len(win_f)} of "
          f"{len(DELAYS)} delays, {win_f}")
    print(f"  live parameters:   the filter is ahead at {len(win_l)} of "
          f"{len(DELAYS)} delays, {win_l}")

    def rate_of(sown, scross, d):
        return (scross - sown[d]) / (sown[0] - sown[d]) \
            if abs(sown[0] - sown[d]) > 1e-12 else float("nan")

    print(f"\n{'delta':>6}{'R, ridge':>11}{'R, frozen':>12}{'R, live':>10}"
          f"{'live - frozen':>16}")
    moves = []
    for d in DELAYS[1:]:
        rp = rate_of(S["own"], S["ridge"][d], d)
        rf = rate_of(S["own"], S["frozen"][d], d)
        rl = rate_of(S["own"], S["live"][d], d)
        moves.append((d, rp, rf, rl))
        print(f"{d:>6}{rp:>11.1%}{rf:>12.1%}{rl:>10.1%}{rl - rf:>+16.1%}")
    d0, rp0, rf0, rl0 = [m for m in moves if m[0] == HEAD][0]
    print(f"\n  at eleven weeks: {rp0:.1%} as published, {rf0:.1%} with the "
          f"frozen filter, {rl0:.1%} with live parameters")

    # the paired difference between the published model and the live filter
    e_pap = (yb - f_pap[HEAD]) ** 2
    e_liv = (yb - f_liv[HEAD]) ** 2
    sst = float(((yb - BENCH) ** 2).sum())
    dd = (e_pap - e_liv) / sst * len(yb)
    bs = np.empty(N_BOOT)
    blk = L.block_for(HEAD)
    for i in range(N_BOOT):
        bs[i] = dd[block_idx(len(dd), rng, blk) % len(dd)].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  live filter minus ridge, in R-squared at eleven weeks: "
          f"{dd.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"  blocks of {blk} days, {N_BOOT} replications")

    # ---------------- B ----------------------------------------------------
    print("\n" + "=" * 96)
    print("B.  WHAT THE PARAMETERS ARE WORTH KNOWING")
    print("=" * 96)
    print("Parameter sets are drawn by simulating the fitted model over the")
    print("estimation block and refitting, then the whole panel is re-filtered")
    print("with each draw.  The spread of the reconstruction across those draws")
    print("is the part of the uncertainty the frozen filter does not report.")
    print()
    print("A moving-block resample of the block itself will not do here, and")
    print("the reason is worth stating: the factor is persistent, the joins")
    print("between blocks break that persistence, and the refit comes back with")
    print("phi biased down and q biased up - far enough that the band excludes")
    print("its own point estimate, which is a sign the instrument is wrong")
    print("rather than a finding about the parameter.\n")
    xs0, ps0 = RE.filter_pass(par0, M)
    ens = np.full((N_ENS, len(M)), np.nan)
    kept = 0
    pars = []
    for i in range(N_ENS):
        pv = RE.fit_state_space(simulate(par0, TRAIN0, rng))
        if pv is None:
            continue
        pars.append(pv)
        ens[kept] = RE.filter_pass(pv, M)[0]
        kept += 1
    ens = ens[:kept]
    print(f"  {kept} of {N_ENS} draws produced a fit\n")
    pe = np.array([p["phi"] for p in pars])
    qe = np.array([p["q"] for p in pars])
    le = np.array([p["lam"][0] for p in pars])
    print("  The draws are refitted the way the data were, so they carry the")
    print("  estimator's own attenuation: a factor extracted from noisy series")
    print("  is a noisy factor, and an AR(1) coefficient fitted to it comes back")
    print("  low.  The raw quantiles therefore sit below the point estimate and")
    print("  are not a band for it.  What the draws measure honestly is the")
    print("  SPREAD, so the band below is recentred on the point estimate and")
    print("  the raw quantiles are printed beside it rather than hidden.\n")
    print(f"{'':>17}{'point':>9}{'recentred 95%':>22}{'raw draw quantiles':>26}")
    for nm, arr, pt in (("phi", pe, par0["phi"]), ("q", qe, par0["q"]),
                        ("target loading", le, par0["lam"][0])):
        lo_r, med, hi_r = np.percentile(arr, [2.5, 50, 97.5])
        print(f"  {nm:>15}{pt:>9.4f}"
              + f"[{pt + lo_r - med:.4f}, {pt + hi_r - med:.4f}]".rjust(22)
              + f"[{lo_r:.4f}, {hi_r:.4f}]".rjust(26))

    rows = idx
    own_sd = np.sqrt(ps0[rows])
    par_sd = np.nanstd(ens[:, rows], axis=0)
    ratio = np.nanmean(par_sd / np.maximum(own_sd, 1e-12))
    print(f"\n  the filter's own posterior standard deviation, averaged over the")
    print(f"  test days: {own_sd.mean():.4f}")
    print(f"  the spread of the reconstruction across parameter draws: "
          f"{par_sd.mean():.4f}")
    print(f"  parameter spread as a share of the filter's own: {ratio:.0%}")
    tot = np.sqrt(own_sd ** 2 + par_sd ** 2).mean()
    print(f"  the two together: {tot:.4f}, which is {tot / own_sd.mean():.2f} "
          f"times the band the frozen filter reports")

    # the reconstruction the section leans on, with its band
    tgt = M[rows, 0]
    fin = np.isfinite(tgt) & np.isfinite(xs0[rows])
    c0 = float(np.corrcoef(xs0[rows][fin], tgt[fin])[0, 1])
    ce = np.array([float(np.corrcoef(e[rows][fin], tgt[fin])[0, 1])
                   for e in ens if np.isfinite(e[rows][fin]).all()])
    clo, cmed, chi = np.percentile(ce, [2.5, 50, 97.5])
    print(f"\n  correlation between the filtered state and the target's own")
    print(f"  decision variable: {c0:.4f}, recentred band "
          f"[{c0 + clo - cmed:.4f}, {c0 + chi - cmed:.4f}]")

    # ---------------- C ----------------------------------------------------
    print("\n" + "=" * 96)
    print("C.  ARE THE INNOVATIONS INNOVATIONS?")
    print("=" * 96)
    print("Standardised one-step innovations, on the rows the test uses.  Under")
    print("the model they have unit variance and no serial correlation; a")
    print("variance above one means the filter is reporting more precision than")
    print("it has, which is what makes the band in B a floor.\n")
    nu = innovations(par0, M)[rows]
    print(f"{'series':>8}{'mean':>9}{'variance':>11}{'LB(10) p':>11}"
          f"{'LB(20) p':>11}{'|z|>1.96':>11}")
    names = ["target"] + [f"peer {i}" for i in range(1, nu.shape[1])]
    bad = 0
    for j, nm in enumerate(names):
        v = nu[:, j]
        v = v[np.isfinite(v)]
        if len(v) < 100:
            continue
        q10, df10 = ljung_box(v, 10)
        q20, df20 = ljung_box(v, 20)
        p10, p20 = chi2_upper(q10, df10), chi2_upper(q20, df20)
        out = float(np.mean(np.abs(v) > 1.96))
        if v.var() > 1.5 or p10 < 0.05:
            bad += 1
        print(f"{nm:>8}{v.mean():>9.3f}{v.var():>11.3f}{p10:>11.3f}"
              f"{p20:>11.3f}{out:>11.1%}")
    print(f"\n  series whose innovations fail unit variance or whiteness: "
          f"{bad} of {len(names)}")

    # ---------------- D ----------------------------------------------------
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    survives = S["live"][HEAD] > S["ridge"][HEAD]
    print(f"  With parameters a forecaster could have had, the filter is "
          f"{'still' if survives else 'no longer'} the better")
    print(f"  cross-sectional model at eleven weeks: {S['live'][HEAD]:.4f} "
          f"against the ridge's {S['ridge'][HEAD]:.4f}.")
    print(f"  The rate at eleven weeks reads {rl0:.1%} under live parameters "
          f"against {rf0:.1%} frozen")
    print(f"  and {rp0:.1%} in the published specification.")
    if lo <= 0 <= hi:
        print("  The paired difference against the ridge does not exclude zero,")
        print("  so the filtered specification is not separated from the")
        print("  published one on these days.")
    else:
        print("  The paired difference against the ridge excludes zero.")
    print(f"  Parameter uncertainty adds {ratio:.0%} to the filter's own band on")
    print("  the reconstruction, and the innovation diagnostics above say")
    print("  whether even that band is an honest floor.")
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
