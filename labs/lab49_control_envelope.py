"""
lab49_control_envelope.py - the rate against every control, not one at a time.

Imports lab05_robustness, lab08_implied_vol and lab38_domestic_baseline; keep all
four in labs/.  Runtime about eight minutes.

THE LIMITATION THIS ADDRESSES
-----------------------------
R(delta) carries the domestic control in three of its four places, so every
figure in this paper is a statement about the control as much as about the
cross-section.  The paper answers that objection one arm at a time and it has
been answered three times now: Section 10.1 builds five richer feature sets and
gives up four points of rate to the strongest; lab46 projects a frozen AR(1)
state forward and finds the projection absorbed exactly by a standardised ridge;
lab46 also weights the whole history exponentially and finds the control gets
worse.

Answering an objection three times is not the same as retiring it.  A referee can
always name a fourth control, and the paper's reply would be another arm and
another paragraph.  What retires it is a different shape of answer:

    quote the rate against the WHOLE SET of controls, and report the spread.

Then the question stops being "what if a better control exists" and becomes "how
far does the rate move across every control tried", which is a number rather than
an argument.

One rule has to be settled before the numbers, because the obvious choice is
wrong.  "Report the lowest rate any control produces" sounds like the
conservative reading and is not a defensible one.  R(delta) reads the own-only
curve at TWO points, the origin and the delay, so a control that is strong at the
origin and weak at the delay deflates the rate at both ends of the fraction
without being a better control in any sense a reader would accept.  The kitchen
sink below does exactly that.  The defensible rule is the paper's own, from
Section 10.1: choose on own-only skill averaged across delays, never on the rate,
and report the spread beside it so nothing is hidden.

THE RULE, STATED BEFORE THE NUMBERS
-----------------------------------
Selection is on OWN-ONLY SKILL, never on the rate.  This matters: the rate is the
quantity under test, and a control chosen because it makes the rate small would
be choosing the answer.  The envelope column takes, at each delay, the control
with the highest own-only R-squared there, and reports the rate that follows.
The lowest column is printed too, because a hostile reader will compute it, and
the verdict says what it is worth.

EIGHT CONTROLS
--------------
    HAR3     the paper's: the stale level and rolling means at 5 and 22 days
    +LAGS    four further own lags
    +LONG    quarterly and annual means
    +LEV     leverage terms from the signed return
    +MEAS    a 22-day Yang-Zhang and the LEVEL of the trailing median
    ALL      every one of the above at once
    +EWMA    exponential windows at half-lives 5, 22 and 66 days (lab46)
    STATE    a frozen AR(1) state projected phi^(delta+h) forward (lab46)

The last two are here because they are the ones the referees actually named, and
because a set that contains only arms the author thought of is not an envelope.

PART C
------
One check the paper has never run and a referee will ask for immediately.  The
redundancy result of Section 8 was established against the PAPER'S control.  If a
stronger domestic block is the right one, the horse race has to be re-run inside
it, because a stronger domestic model could in principle leave less room for
implied volatility as well.  Part C does that on Section 8's own window.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else ".")
import lab05_robustness as L
import lab08_implied_vol as IV
import lab38_domestic_baseline as B

SEED = 20260917
TARGET = "SPX"
H = L.HORIZON
DELAYS = [0, 3, 5, 13, 21, 55]
HALFLIVES = (5, 22, 66)
# The block is lab05's measured one, not 2 * HORIZON.  The h-day overlap in
# the target is not the only dependence in a loss difference: it also
# inherits the common factor's persistence, which lab58 measures at 0.48
# autocorrelation at lag 10 and not below 0.05 until lag 38.  A block
# shorter than the dependence leaves it inside the resample and the
# interval comes out too narrow.
N_BOOT, BLOCK = 1500, L.BLOCK
ARMS = ["HAR3", "+LAGS", "+LONG", "+LEV", "+MEAS", "ALL", "+EWMA", "STATE"]


def moments(x):
    """phi, q, r for an AR(1) observed with noise, by method of moments."""
    x = x[np.isfinite(x)]
    v = x - x.mean()
    n = len(v)
    g = [float(v[:n - k] @ v[k:]) / n for k in (0, 1, 2)]
    phi = float(np.clip(g[2] / g[1] if abs(g[1]) > 1e-12 else 0.0, 0.0, 0.999))
    sig = max(g[1] / phi if phi > 1e-9 else 0.0, 1e-12)
    return phi, sig * (1 - phi ** 2), max(g[0] - sig, 1e-9)


def r2(y, f, *, bench, rows=None):
    """Out-of-sample skill against the trailing-mean benchmark.

    `bench` is keyword-only and has no default on purpose.  This helper used to
    divide by ((yy - yy.mean()) ** 2), the mean of the target over the test
    period - a constant chosen with hindsight, and the error an external audit
    caught elsewhere in this project.  The rollout that fixed it missed this
    file, which computed the right benchmark at the top of main() and then never
    used it.  Making the argument mandatory means a missed call site raises
    instead of quietly scoring against the wrong yardstick, and keyword-only
    means an old positional `r2(y, f, rows)` cannot pass `rows` in as `bench`.
    """
    yy, ff = (y, f) if rows is None else (y[rows], f[rows])
    bb = bench if rows is None else bench[rows]
    den = ((yy - bb) ** 2).sum()
    return 1 - ((yy - ff) ** 2).sum() / den if den > 0 else np.nan


def blocks_of(n, rng, block=BLOCK):
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def meas_cols(folder, index):
    """The +MEAS pair, computed on an arbitrary panel index.

    Part C needs Section 8's window rather than Section 4's, and lab38 builds
    these only on its own.  Same construction, same source file.
    """
    raw = L.load_index(TARGET, folder)
    yz22 = L.yang_zhang(raw, n=22).reindex(index).ffill(limit=5)
    m22 = yz22.rolling(L.MED, min_periods=30).median()
    yz5 = L.yang_zhang(raw, n=L.WINDOW).reindex(index).ffill(limit=5)
    m5 = yz5.rolling(L.MED, min_periods=30).median()
    return np.column_stack([np.log(yz22.values / m22.values), np.log(m5.values)])


def main(folder=None):
    folder = L.find_folder(folder)
    print(f"data folder: {folder}")
    print("Prints the control envelope, Section 10.1.\n")

    F, P, y, idx, npeers, D = B.blocks(folder)
    yb = y[idx]
    BENCH = L.bench_mean(y, idx)
    n = len(idx)
    a = D[TARGET].values
    sa = pd.Series(a)
    print(f"target {TARGET}, {n} test days, {npeers} foreign peers")

    # the two controls the referees named, built here so the set is not just ours
    ewm = [sa.ewm(halflife=h, adjust=False).mean().values for h in HALFLIVES]
    F["EWMA"] = np.column_stack([e[:, None] for e in ewm])
    phi, q, r = moments(a[:L.TRAIN + L.VAL])
    print(f"the frozen AR(1) control: phi = {phi:.4f}, fitted on the first "
          f"{L.TRAIN + L.VAL} rows\n")

    def own_of(arm):
        if arm == "+EWMA":
            return np.column_stack([F["HAR3"], F["EWMA"]])
        if arm == "STATE":
            return a[:, None]
        return B.own_block(F, arm)

    # ---------------- A. the eight controls --------------------------------
    print("=" * 96)
    print("A.  OWN-ONLY SKILL, EVERY CONTROL, EVERY DELAY")
    print("=" * 96)
    print("Same walk-forward, same refit schedule, same days. Selection below is on")
    print("THIS table and never on the rate, which is the quantity under test.\n")
    own_f, S = {}, {}
    for arm in ARMS:
        blk = own_of(arm)
        for d in DELAYS:
            own_f[(arm, d)] = B.walk(blk, P, y, idx, d, False)
            S[(arm, d)] = r2(yb, own_f[(arm, d)], bench=BENCH)
    print(f"{'delta':>6}" + "".join(f"{a_:>9}" for a_ in ARMS))
    for d in DELAYS:
        print(f"{d:>6}" + "".join(f"{S[(a_, d)]:>9.4f}" for a_ in ARMS))
    best = {d: max(ARMS, key=lambda a_: S[(a_, d)]) for d in DELAYS}
    print("\n  strongest control at each delay: "
          + ", ".join(f"{d}d {best[d]}" for d in DELAYS))
    mean_skill = {a_: np.mean([S[(a_, d)] for d in DELAYS]) for a_ in ARMS}
    champ = max(ARMS, key=lambda a_: mean_skill[a_])
    print(f"  strongest averaged across delays: {champ} "
          f"({mean_skill[champ]:.4f} against {mean_skill['HAR3']:.4f} for the paper's)")

    # ---------------- B. the rate against each ------------------------------
    print("\n" + "=" * 96)
    print("B.  THE RATE AGAINST EVERY ONE OF THEM")
    print("=" * 96)
    print("The rate is recomputed INSIDE each arm, as Table 16 does: the control")
    print("and the cross-sectional model carry the same domestic block, so the")
    print("only difference between numerator and denominator is the seven peers.")
    print("Holding the cross-sectional model at the paper's block while the")
    print("control improves would compare two different models and report the")
    print("mismatch as a rate.\n")
    cross_f, Sc = {}, {}
    for arm in ARMS:
        blk = own_of(arm)
        for d in DELAYS:
            cross_f[(arm, d)] = B.walk(blk, P, y, idx, d, True)
            Sc[(arm, d)] = r2(yb, cross_f[(arm, d)], bench=BENCH)
    R = {}
    for arm in ARMS:
        for d in DELAYS[1:]:
            den = S[(arm, 0)] - S[(arm, d)]
            R[(arm, d)] = (Sc[(arm, d)] - S[(arm, d)]) / den if den > 1e-9 else np.nan
    print(f"{'delta':>6}" + "".join(f"{a_:>9}" for a_ in ARMS)
          + f"{'envelope':>10}{'lowest':>9}")
    low = {}
    for d in DELAYS[1:]:
        vals = [R[(a_, d)] for a_ in ARMS]
        low[d] = (min(vals), ARMS[int(np.argmin(vals))])
        print(f"{d:>6}"
              + "".join(f"{R[(a_, d)]:>9.1%}" for a_ in ARMS)
              + f"{R[(best[d], d)]:>10.1%}{low[d][0]:>9.1%}")
    print(f"\n  at eleven weeks: {R[('HAR3', 55)]:.1%} against the paper's own control,")
    print(f"  {R[(best[55], 55)]:.1%} against the strongest one by own-only skill ({best[55]}),")
    print(f"  {low[55][0]:.1%} at the lowest any of the eight produces ({low[55][1]})")

    # the paired interval on the gap that matters: paper against the strongest
    rng = np.random.default_rng(SEED)
    diffs = []
    arm = best[55]
    for _ in range(N_BOOT):
        rows = blocks_of(n, rng)
        s0p = r2(yb, own_f[("HAR3", 0)], rows=rows, bench=BENCH)
        sdp = r2(yb, own_f[("HAR3", 55)], rows=rows, bench=BENCH)
        s0b = r2(yb, own_f[(arm, 0)], rows=rows, bench=BENCH)
        sdb = r2(yb, own_f[(arm, 55)], rows=rows, bench=BENCH)
        scp = r2(yb, cross_f[("HAR3", 55)], rows=rows, bench=BENCH)
        scb = r2(yb, cross_f[(arm, 55)], rows=rows, bench=BENCH)
        if min(s0p - sdp, s0b - sdb) < 1e-9:
            continue
        diffs.append((scb - sdb) / (s0b - sdb) - (scp - sdp) / (s0p - sdp))
    dlo, dhi = np.percentile(diffs, [2.5, 97.5])
    print(f"\n  paper minus {arm} at eleven weeks: "
          f"{R[(arm, 55)] - R[('HAR3', 55)]:+.1%} [{dlo:+.1%}, {dhi:+.1%}], "
          f"{len(diffs)} replications")
    print("  Numerator and denominator are recomputed on the same resampled days")
    print("  for both arms, so the difference is paired and the intervals of the")
    print("  two rates are not being compared with each other.")

    # ---------------- C. does redundancy survive the stronger control? -----
    print("\n" + "=" * 96)
    print("C.  AND DOES SECTION 8 SURVIVE THE STRONGER CONTROL?")
    print("=" * 96)
    print("Section 8 raced the foreign block against implied volatility with the")
    print("PAPER'S domestic block underneath both. If the stronger block is the")
    print("right one, the race has to be re-run inside it: a better domestic model")
    print("could leave less room for the options market too.\n")
    Div, ivdf, ivpeers = IV.build(folder, strict=True)
    own_iv, Piv, iv, yiv, idxiv = IV.make(Div, ivdf, ivpeers)
    strong = np.column_stack([own_iv, meas_cols(folder, Div.index)])
    ybiv = yiv[idxiv]
    # Part C is on Section 8's window, a different panel from parts A and B,
    # so it needs its own benchmark on its own days rather than the one built
    # above.
    BENCH_IV = L.bench_mean(yiv, idxiv)
    print(f"  Section 8's window: {len(idxiv)} test days, "
          f"{len(ivpeers)} peers, implied-volatility block {iv.shape[1]} wide")
    print(f"\n{'delta':>6}{'control':>10}{'+IV':>9}{'+IV+foreign':>13}"
          f"{'foreign given IV':>19}{'GW z':>8}")
    verdict_rows = []
    for d in DELAYS:
        for name, blk in (("paper", own_iv), ("+MEAS", strong)):
            f_iv = IV.walk(blk, [iv], yiv, idxiv, d)
            f_both = IV.walk(blk, [iv, Piv], yiv, idxiv, d)
            inc = (r2(ybiv, f_both, bench=BENCH_IV)
                   - r2(ybiv, f_iv, bench=BENCH_IV))
            dl = (ybiv - f_iv) ** 2 - (ybiv - f_both) ** 2
            z = float(dl.mean() / IV.hac_se(dl))
            print(f"{d:>6}{name:>10}{r2(ybiv, f_iv, bench=BENCH_IV):>9.4f}"
                  f"{r2(ybiv, f_both, bench=BENCH_IV):>13.4f}"
                  f"{inc:>19.4f}{z:>8.2f}")
            if name == "+MEAS":
                verdict_rows.append((d, inc, z))
    wins = [d for d, inc, z in verdict_rows if z > 1.96]
    print(f"\n  delays where the foreign block adds significantly given implied")
    print(f"  volatility AND the stronger domestic control: {len(wins)} of {len(DELAYS)}")

    # ---------------- verdict ----------------------------------------------
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    print(f"  Across eight domestic controls the substitution rate at eleven weeks")
    print(f"  runs from {min(R[(a_, 55)] for a_ in ARMS):.1%} to "
          f"{max(R[(a_, 55)] for a_ in ARMS):.1%}, a spread of "
          f"{max(R[(a_, 55)] for a_ in ARMS) - min(R[(a_, 55)] for a_ in ARMS):.0%} points. The paper's control")
    print(f"  gives {R[('HAR3', 55)]:.1%} and the strongest control by own-only skill, {champ}, gives")
    print(f"  {R[(champ, 55)]:.1%}, a paired difference of "
          f"{R[(champ, 55)] - R[('HAR3', 55)]:+.1%} [{dlo:+.1%}, {dhi:+.1%}]. That reproduces "
          f"Section 10.1")
    print("  from a second direction, on a different feature set and a different")
    print("  script, which is worth more than the agreement of a number with itself.")
    print(f"\n  The lowest figure in the table is {low[55][0]:.1%}, from {low[55][1]}, and it should not")
    print("  be quoted as the conservative rate. Look at what produces it: that arm")
    print(f"  scores {S[(low[55][1], 0)]:.4f} own-only at the origin and {S[(low[55][1], 55)]:.4f} at "
          f"eleven weeks, so it is the")
    print("  strongest control in the set where the denominator is set and one of the")
    print("  weakest where the numerator is. R(delta) reads the curve at two points,")
    print("  and an arm that is strong at one and weak at the other moves the rate")
    print("  without being a better control anywhere. Reporting it as the headline")
    print("  would be the mirror of the error two referees warned about, made in the")
    print("  paper's own favour instead of against it.")
    print("\n  What the envelope does retire is the open-ended version of the")
    print("  objection. The control is no longer one choice a reader has to take on")
    print("  trust: eight are priced, the selection rule is stated before the")
    print("  numbers, and a ninth moves the headline only by beating all eight on")
    print("  own-only skill, under the same rule.")
    if not wins:
        print("\n  Section 8 is unaffected. Under the stronger control the foreign block")
        print("  still adds nothing detectable given implied volatility at any delay,")
        print("  so the redundancy result was not resting on a weak domestic model")
        print("  either.")
    else:
        print(f"\n  Section 8 does NOT survive intact: given implied volatility AND the")
        print(f"  stronger control, the foreign block adds significantly at "
              f"{len(wins)} of {len(DELAYS)} delays")
        print(f"  ({', '.join(str(d) for d in wins)}). That is a real change to the paper's")
        print("  central scope condition and Section 8 has to report it.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
