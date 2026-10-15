"""
lab32_masked_training.py - EXPLORATORY.  Cited by neither paper.

One model for every delay, instead of one model per delay.

THE IDEA
--------
Both papers treat the delay as an experimental condition: fix delta, fit a model,
measure, repeat.  Ten delays means ten models, and a forecaster who turns up
seven days stale - a delay nobody fitted - has nothing to use.

Masked training inverts that.  Draw a random delay for every training row, hold
the domestic block that stale, tell the model how stale it is, and fit ONCE.
Staleness stops being a condition the experiment imposes and becomes a feature
the model conditions on.  This is what the masking objective in sequence models
does, reduced to something a ridge regression can carry, so that the idea can be
tested without also testing a neural network.

THE DESIGN MATRIX
-----------------
    own(t - delta)          three columns, held stale
    foreign(t)              seven columns, always current
    s = log(1 + delta)      how stale the domestic block is
    s * own_level           the one interaction that matters: how much to
                            discount the stale level as it ages

Twelve columns against the paper's ten, fitted once per refit instead of once
per refit per delay.

TWO DELIBERATE HANDICAPS
------------------------
Each training row is used ONCE, with a single randomly drawn delay, so the pooled
model sees exactly as many rows as each per-delay model does.  Masked training
would normally revisit a row under several masks; not doing so keeps the sample
sizes comparable and makes this the conservative comparison.

And the training window is cut at t - max(delta) - h for every row regardless of
the delay drawn, so no row can borrow information a long-delay forecaster would
not have had.  That costs the pooled model recent data the per-delay models at
short delay do get.

WHAT WOULD MAKE THIS WORTH HAVING
----------------------------------
Matching the per-delay models would be enough on its own, because one model is
cheaper than ten and covers delays nobody fitted.  Part B tests exactly that:
delays held out of training entirely.  Part C is the harder version - train only
on short delays and test on long ones, where the model must have learned how
staleness behaves rather than memorised each case.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab22_factor_benchmark as F

SEED = 20260914
H = F.H
GRID = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]     # delays the masked model trains on
TEST = [0, 3, 5, 13, 21, 55]                  # the paper's reporting grid
UNSEEN = [7, 17, 30, 44]                      # never drawn in training
SHORT = [0, 1, 2, 3, 5, 8, 13, 21]            # part C trains on these only
FAR = [34, 55]                                # ... and is tested on these
N_BOOT = 2000
BLK = L.BLOCK


def feats(own_stale, P_now, delta):
    """Twelve columns: stale domestic, current foreign, staleness, interaction."""
    n = own_stale.shape[0]
    s = np.full((n, 1), np.log1p(delta))
    return np.column_stack([own_stale, P_now, s, s[:, 0] * own_stale[:, 0]])


def per_delay(own, P, y, idx, delta):
    """The paper's arm: own + foreign, fitted at this delay only."""
    out = np.empty(len(idx)); b = mu = sd = None
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            cut = t - delta - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), cut)
            X = np.column_stack([own[tr - delta], P[tr]])
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y[tr])
            X, yy = X[ok], y[tr][ok]
            if len(yy) < 200:
                b = None; continue
            mu, sd = X.mean(0), X.std(0) + 1e-9
            b = L.cv((X - mu) / sd, yy, "cont")
        xt = np.concatenate([own[t - delta], P[t]])
        out[j] = 0.0 if (b is None or not np.isfinite(xt).all()) \
            else L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
    return out


def masked(own, P, y, idx, train_grid, test_delays, seed, per_row_cut=False):
    """Fit ONE model per refit on randomly-delayed rows; score at every delay."""
    rng = np.random.default_rng(seed)
    out = {d: np.empty(len(idx)) for d in test_delays}
    b = mu = sd = None
    cut_back = max(train_grid)
    for j, t in enumerate(idx):
        if j % L.REFIT == 0:
            # one cut for every row, at the LONGEST delay, so no row can borrow
            # information a long-delay forecaster would not have had
            cut = t - cut_back - H
            tr = np.arange(max(0, cut - L.TRAIN - L.VAL), cut)
            draws = rng.choice(train_grid, size=len(tr))
            if per_row_cut:
                # OPTIMISTIC: each row cut at its OWN drawn delay, so short-delay
                # rows use fresher data than a long-delay forecaster ever had.
                # Not a legitimate protocol for one shared model - run only to
                # price the conservative cut used above.
                cut = t - min(train_grid) - H
                tr = np.arange(max(0, cut - L.TRAIN - L.VAL), cut)
                draws = rng.choice(train_grid, size=len(tr))
                keep = (tr + draws) <= (t - H)
                tr, draws = tr[keep], draws[keep]
            rows, ys = [], []
            for dd in np.unique(draws):
                sel = tr[draws == dd]
                Xd = feats(own[sel - dd], P[sel], dd)
                okd = np.isfinite(Xd).all(axis=1) & np.isfinite(y[sel])
                if okd.any():
                    rows.append(Xd[okd]); ys.append(y[sel][okd])
            if not rows:
                b = None; continue
            X = np.vstack(rows); yy = np.concatenate(ys)
            if len(yy) < 200:
                b = None; continue
            mu, sd = X.mean(0), X.std(0) + 1e-9
            b = L.cv((X - mu) / sd, yy, "cont")
        for d in test_delays:
            xt = feats(own[t - d][None, :], P[t][None, :], d)[0]
            out[d][j] = 0.0 if (b is None or not np.isfinite(xt).all()) \
                else L.p_ridge(b, ((xt - mu) / sd)[None, :])[0]
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
    print("EXPLORATORY - cited by neither paper.\n")
    own, P, y, idx, k = F.panel(folder)
    yb = y[idx]
    BENCH = L.bench_mean(y, idx)
    print(f"target {F.TARGET}, {len(idx)} test days, {k} foreign peers")
    print(f"masked model trains on delays {GRID}")
    print(f"block bootstrap, block {BLK}, {N_BOOT} draws\n")

    # ---------------- A ---------------------------------------------------
    print("=" * 96)
    print("A.  ONE MASKED MODEL AGAINST SIX SEPARATELY-FITTED ONES")
    print("=" * 96)
    print("Positive favours the masked model.  It is handicapped twice: one delay")
    print("drawn per row, and every training window cut at the longest delay.\n")
    mk = masked(own, P, y, idx, GRID, TEST + UNSEEN, SEED)
    print(f"{'delta':>6}{'per-delay':>12}{'masked':>10}{'difference':>13}"
          f"{'95% CI':>24}{'':>8}")
    beats, loses = [], []
    for d in TEST:
        fp = per_delay(own, P, y, idx, d)
        fm = mk[d]
        dm, lo, hi = boot_ci(yb, fm, fp, np.random.default_rng(SEED + d))
        if lo > 0:
            beats.append(d)
        if hi < 0:
            loses.append(d)
        tag = "masked" if lo > 0 else ("per-delay" if hi < 0 else "neither")
        print(f"{d:>6}{F.r2(yb, fp, BENCH):>12.4f}{F.r2(yb, fm, BENCH):>10.4f}"
              f"{F.r2(yb, fm, BENCH) - F.r2(yb, fp, BENCH):>+13.4f}"
              + f"[{lo:>+9.5f},{hi:>+9.5f}]".rjust(24) + f"{tag:>11}")
    print(f"\n  delays where ONE masked model significantly beats its own per-delay")
    print(f"  counterpart: {len(beats)} of {len(TEST)} {beats if beats else ''}")
    print(f"  delays where the per-delay model significantly wins: "
          f"{len(loses)} of {len(TEST)} {loses if loses else ''}")
    if not loses:
        print("\n  One model, fitted once, is not worse than six fitted separately at any")
        print("  delay - while being cheaper and defined at delays nobody fitted.")
    elif loses and not beats:
        print("\n  Pooling costs accuracy at every delay where the two separate.  The")
        print("  convenience of one model is real but it is not free, and the price is")
        print("  in the table.")
    else:
        print("\n  The two trade places across the grid; neither dominates, and where")
        print("  they differ is reported above rather than summarised away.")

    # ---------------- B ---------------------------------------------------
    print("\n" + "=" * 96)
    print("B.  DELAYS NOBODY FITTED")
    print("=" * 96)
    print(f"{UNSEEN} were never drawn in training.  The per-delay approach has no")
    print("model for them at all without refitting; the masked model simply answers.")
    print("The comparison is therefore against a model fitted specially for each,")
    print("which is the strongest possible opponent.\n")
    print(f"{'delta':>6}{'fitted for it':>15}{'masked':>10}{'difference':>13}"
          f"{'95% CI':>24}")
    worse = []
    for d in UNSEEN:
        fp = per_delay(own, P, y, idx, d)
        fm = mk[d]
        dm, lo, hi = boot_ci(yb, fm, fp, np.random.default_rng(SEED + 3 + d))
        if hi < 0:
            worse.append(d)
        print(f"{d:>6}{F.r2(yb, fp, BENCH):>15.4f}{F.r2(yb, fm, BENCH):>10.4f}"
              f"{F.r2(yb, fm, BENCH) - F.r2(yb, fp, BENCH):>+13.4f}"
              + f"[{lo:>+9.5f},{hi:>+9.5f}]".rjust(24))
    print(f"\n  unseen delays where the masked model is significantly worse than one")
    print(f"  fitted specially for that delay: {len(worse)} of {len(UNSEEN)} "
          f"{worse if worse else ''}")
    if not worse:
        print("  It interpolates. A forecaster arriving at an arbitrary staleness gets an")
        print("  answer as good as one fitted for that staleness, from a model that never")
        print("  saw it.")
    else:
        print("  Interpolation costs something at those delays, and a practitioner who")
        print("  cares about them should refit rather than rely on the pooled model.")

    # ---------------- C ---------------------------------------------------
    print("\n" + "=" * 96)
    print("C.  THE HARD VERSION: TRAIN SHORT, TEST LONG")
    print("=" * 96)
    print(f"Train on {SHORT} only, then test at {FAR} - delays longer than anything")
    print("the model has ever been fitted on. Interpolation is easy; this is not.\n")
    mk2 = masked(own, P, y, idx, SHORT, FAR, SEED + 1)
    print(f"{'delta':>6}{'fitted for it':>15}{'extrapolated':>14}{'difference':>13}"
          f"{'95% CI':>24}")
    ext_ok = []
    for d in FAR:
        fp = per_delay(own, P, y, idx, d)
        fm = mk2[d]
        dm, lo, hi = boot_ci(yb, fm, fp, np.random.default_rng(SEED + 11 + d))
        if hi >= 0:
            ext_ok.append(d)
        print(f"{d:>6}{F.r2(yb, fp, BENCH):>15.4f}{F.r2(yb, fm, BENCH):>14.4f}"
              f"{F.r2(yb, fm, BENCH) - F.r2(yb, fp, BENCH):>+13.4f}"
              + f"[{lo:>+9.5f},{hi:>+9.5f}]".rjust(24))
    print(f"\n  extrapolated delays not significantly worse than a purpose-fitted")
    print(f"  model: {len(ext_ok)} of {len(FAR)} {ext_ok if ext_ok else ''}")
    if len(ext_ok) == len(FAR):
        print("\n  The model learned how staleness behaves rather than memorising each")
        print("  case: trained on nothing beyond three weeks, it answers at eleven weeks")
        print("  as well as a model fitted there. That is the claim worth making for")
        print("  masked training, and it is the one this part was built to test.")
    elif ext_ok:
        print("\n  It extrapolates some of the way and not all of it. The delay at which")
        print("  it breaks down is the useful number, not the fact that it does.")
    else:
        print("\n  Extrapolation fails. What the masked model learned is specific to the")
        print("  delays it saw, so the single-model convenience does not extend past the")
        print("  range it was trained on - and a paper reporting only interpolation would")
        print("  have overstated what the method does.")


    # ---------------- D ---------------------------------------------------
    print("\n" + "=" * 96)
    print("D.  HOW MUCH OF THAT LOSS IS THE PROTOCOL RATHER THAN THE METHOD?")
    print("=" * 96)
    print("Part A cuts every training window at the LONGEST delay, so the pooled model")
    print("gives up 55 days of recent data that the per-delay model at delta = 0 keeps.")
    print("That is the right choice for one shared model and it is not free.  Rerunning")
    print("with each row cut at its own delay is NOT a legitimate protocol - short-delay")
    print("rows then see data a long-delay forecaster never had - but it brackets how")
    print("much of the gap is the cut and how much is pooling itself.\n")
    mk3 = masked(own, P, y, idx, GRID, TEST, SEED + 2, per_row_cut=True)
    print(f"{'delta':>6}{'per-delay':>12}{'conservative':>14}{'optimistic':>12}"
          f"{'cut costs':>12}{'pooling costs':>15}")
    cut_cost, pool_cost = [], []
    for d in TEST:
        fp = per_delay(own, P, y, idx, d)
        rp, rc, ro = F.r2(yb, fp, BENCH), F.r2(yb, mk[d], BENCH), F.r2(yb, mk3[d], BENCH)
        cut_cost.append(ro - rc); pool_cost.append(rp - ro)
        print(f"{d:>6}{rp:>12.4f}{rc:>14.4f}{ro:>12.4f}{ro - rc:>+12.4f}"
              f"{rp - ro:>+15.4f}")
    print(f"\n  cost of the conservative cut: {min(cut_cost):+.4f} to {max(cut_cost):+.4f} of R2")
    print(f"  cost of pooling itself:       {min(pool_cost):+.4f} to {max(pool_cost):+.4f} of R2")
    if max(cut_cost) > max(pool_cost):
        print("\n  Most of part A's gap is the cut, not the pooling.  The method is closer")
        print("  to competitive than part A alone suggests - but the cut is required for")
        print("  one shared model to be honest, so the gap in part A is the one a")
        print("  practitioner would actually face.  Both numbers belong in any write-up.")
    else:
        print("\n  Pooling itself, not the cut, is what costs.  Sharing one set of")
        print("  coefficients across delays is the binding constraint, and no change of")
        print("  protocol recovers it.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
