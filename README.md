# What substitutes for a stale mark?

Two working papers and the code that produced every number in them.


below.

The question behind both: when the data you care about has gone stale, what can
you buy instead of timeliness?

That question is easy to ask about a private equity portfolio marked once a
quarter, or property appraised on a cycle, and impossible to answer there,
because the truth is never observed. So the experiment runs where the truth
*is* observed. A forecaster of S&P 500 realised variance is handicapped by
holding domestic data δ days old while foreign index closes stay current — an
asymmetry that trading calendars supply for free, since Tokyo, Hong Kong,
Mumbai, London and Frankfurt have all closed by the time New York does.

## What the papers claim

**`papers/what-substitutes-for-a-stale-mark.pdf`** — the main result. Two
candidate substitutes are priced against each other. Over 4,862 out-of-sample
days spanning 2007 to 2026, domestic-only R² falls from 0.499 to below zero as δ
runs from nothing to eleven weeks; adding seven foreign closes holds it at 0.347,
recovering 72% of the loss by eleven weeks and 81% under QLIKE. Implied
volatility recovers more than all of it — with VIX and VDAX in hand a forecaster
three weeks stale beats one holding fully current domestic data — and given
implied volatility the foreign block adds nothing at any delay.

Four things the paper spends as much space on as the headline:

- **Why it stops at roughly three-quarters.** Foreign markets span only about
  70% of the S&P's volatility state (contemporaneous R² = 0.695), so the
  unrecovered third is local by construction. The foreign block is also nearly
  one-dimensional — first principal component 66.5%, first two 76.9% — so adding
  more correlated indices cannot raise the ceiling.
- **What breadth costs.** Carrying seven extra regressors is not free. The cost
  is about one point of R² at zero delay and grows with delay. Netting it out is
  what turns "the model got slightly worse at δ = 0" into something
  interpretable. `lab22` then stops inferring the cheaper representation and
  fits it: one real-time global factor beats all seven series at every delay,
  by 0.012 to 0.019 of R², with the margin growing as the inferred cost does.
  An equal-weighted mean captures nearly all of that, so what is bought is the
  decision to compress rather than the machinery of compressing.
- **Which substitute wins, and where.** Section 7 runs the horse race. Where a
  liquid options market exists on the stale asset, breadth is redundant; the
  cross-sectional result belongs to assets with no such market, which is the case
  that motivates the question and the one this experiment cannot demonstrate.
- **Whether the linear estimator is the binding constraint.** `lab09` and `lab15`
  test it rather than asserting it. Squaring every feature makes the forecast
  worse; letting the global factor enter quadratically and interact with the
  domestic state changes nothing measurable; and a random-Fourier layer — a
  one-hidden-layer network with a random first layer — is significantly worse at
  every delay, with the wider layer worse than the narrower one. The claim drawn
  is deliberately weaker than "linearity is sufficient" — this sample cannot pay
  for the extra parameters.
- **Which loss is actually proxy-robust.** `lab10` corrects an error in an
  earlier draft: QLIKE is variance-scale and robust, R² on log variance is not.
  Both robust losses put the substitution rate *above* the headline, so the
  number the paper leads with is the conservative one.
- **What does not survive testing.** Several claims were withdrawn rather than
  defended. They are listed below.

**`papers/threshold-rule-at-its-ceiling.pdf`** — the companion note, which
argues that splitting a volatility target at its median discards exactly the
magnitude information a cross-sectional model would use, and shows the resulting
threshold rule sits close to what a restricted-information benchmark says is
attainable. It is why the main paper's headline uses a continuous target.

An earlier version of that note called the benchmark a *ceiling* and said no
regression could beat the sign rule. A referee objected and `lab20` settled it:
the measured rule exceeds the benchmark at eight of ten delays. The referee's
proposed mechanism — that the HAR classifier's extra lags break a sufficiency
condition — is not what fails; those lags add nothing significant. What fails is
the centring assumption. P(b>0 | a) estimated non-parametrically is not monotone
and crosses one half at a = +0.27, not at zero, so the rule's threshold is in the
wrong place and the formula derived under centring is an approximation rather
than a bound. The note says so now.

That ceiling table was, until recently, the one thing in either paper this
repository could not regenerate. It came from a scratch script that read a CSV
from the author's Desktop and compared it against accuracies pasted in as
literals from an earlier run — so nobody else could run it, and if `lab02` had
ever changed, the comparison would have gone on agreeing with a number that no
longer existed. It is `lab02b_threshold_ceiling.py` now: same data as everything
else, accuracies recomputed by calling `lab02`'s own `evaluate()`, nothing
pasted. Every figure is unchanged, which is the good outcome and was not the
guaranteed one.

## Reproducing it

```
pip install -r requirements.txt
python run_all.py --check
```

**Extract the archive to a real folder before running anything.** Launching
`run_all.py` straight out of the zip preview copies only that one file into a
temporary directory, leaving `labs/` and `data/` behind; the script checks for
this and says so rather than failing on an unreadable `[WinError 267]`.

`--check` runs all twenty-four labs and diffs each against the stored output in
`expected_output/`. Two lines are exempt from the comparison — the data folder
path and the runtime — and nothing else is. A clean run prints
`all 24 labs reproduced the reference output exactly`.

The reference output in `expected_output/` was generated on Linux. Four of the
twenty-four labs — 04 through 07 — were separately reproduced on Windows during
development and matched digit for digit. The other twenty, including every lab
added after that check, have only been run on Linux. The cross-platform claim is
therefore narrower than a reader might assume from a clean `--check`, and it is
stated at that width on purpose: a suite that has only ever run on one operating
system has not been shown to be portable, whatever its own diff says. Running
`python run_all.py --check` on a second platform is what would widen it.

Every file is read and written as UTF-8 explicitly. Python's `open()` otherwise
follows the *locale*, which is UTF-8 here and cp1252 on a default Windows
install, and cp1252 does not reject the bytes it cannot interpret — it decodes
them into different characters and carries on. `verify_paper.py` failed on
Windows for exactly this reason while passing on Linux, on a single interval
containing a U+2212 minus sign, which reads as a stale number and is not one.
`PYTHONWARNDEFAULTENCODING=1 python -W always run_all.py --check` reports any
call that reintroduces the gap; it is currently silent.

There is a second check, and it covers something `--check` does not:

```
python verify_paper.py
```

`run_all.py --check` proves the *scripts* still produce the same numbers. It says
nothing about whether the *paper* still quotes them. `verify_paper.py` reads both
paper sources in `papers/source/` and asserts that every cost, gross, interval and
test statistic they print matches `expected_output/`, that no superseded value has
survived, and that no withdrawn claim has crept back. Two hundred and sixty-one checks; exit 1 on
any disagreement. Two stale numbers reached a draft before this existed.

Both commands must be run **from the extracted repository folder**. They resolve
`labs/`, `data/`, `expected_output/` and `papers/source/` relative to themselves,
so a copy of a single script saved loose on the Desktop will not work and, worse,
may be an older version than the one in the repo.

Drop `--check` to just run and read. Pass a lab name to run one:
`python run_all.py lab07`. The whole suite takes about forty minutes. `lab07` is seven minutes of it, because its
estimation-cost figure averages over a hundred surrogate draws, and `lab15` is eight,
because it runs five feature sets across two penalty grids and then prices three of them
against surrogates.

Requirements are NumPy and pandas. Nothing else — the ridge regressions, the
IRLS logistic, the Kalman filter and the bootstraps are all written out, partly
so that the walk-forward's fairness rule is visible rather than buried in a
library call.

## What is in each lab

| lab | establishes |
|---|---|
| `lab01_ceiling.py` | The closed-form ceiling, checked against an explicit Kalman filter at 14 parameter settings. Passes to within 2%. |
| `lab02_delay_curve.py` | The decay curve on a single market, with a matched-persistence baseline rather than a strawman. |
| `lab02b_threshold_ceiling.py` | The ceiling the threshold rule is aiming at, from the orthant probability, against the accuracy it reaches. Produces the companion note's main table. |
| `lab03_crosssection.py` | Seam audit of the data joins, time-zone admissibility, and the in-sample upper bound. If the bound does not move, nothing downstream can. |
| `lab04_walkforward.py` | Out-of-sample walk-forward on the binary target. |
| `lab05_robustness.py` | The reframed benchmark, five target markets under a general timing rule, and the continuous target with R² and QLIKE. |
| `lab06_inference.py` | Giacomini-White and Clark-West (the domestic model is *nested*, which rules out Diebold-Mariano), simultaneous bands across the delay grid, a placebo, and a test of the convergence claim. |
| `lab07_estimation_cost.py` | What breadth costs, measured two ways, and the gross-versus-net decomposition with intervals. |
| `lab08_implied_vol.py` | The implied-volatility horse race. Four arms — stale domestic only, plus foreign closes, plus VIX and VDAX, plus both — run twice on VIX timing. |
| `lab09_nonlinearity.py` | Whether a non-linear map finds more than the linear one. Squared features and a global-factor interaction, both priced against their own estimation cost. |
| `lab10_loss_scale.py` | Which scale each loss lives on. QLIKE is variance-scale and proxy-robust; R² on log variance is not. Adds Patton's MSE with Duan smearing, and bootstraps the *whole* substitution ratio rather than its numerator. |
| `lab11_markets_continuous.py` | The five-market external-validity check, repeated on the continuous target the rest of the paper uses. |
| `lab12_appendix.py` | Every setting stated — HAC kernel and bandwidth, bootstrap replicates, block length, ridge grid — plus sensitivity tests for the two that could be doing hidden work. |
| `lab13_origin_median.py` | Whether the trailing median leaks the future into the target. Reruns everything with the outcome normalised at the *forecast origin*, which a practitioner could actually do. |
| `lab14_appraisal_smoothing.py` | Whether a clean lag is the right model of a *stale mark*. Compares it against periodic revision held flat between marks, which is what an appraisal-based holder actually sees. |
| `lab15_nonlinear_given_iv.py` | Whether the redundancy result is an artefact of the linear estimator. Repeats `lab09`'s question on the feature set that actually carries it — the one including implied volatility — and adds a random-Fourier layer, which is a one-hidden-layer network with a random first layer. |
| `lab16_clark_west_shrinkage.py` | Whether this project's own Clark-West column survives per-arm penalty tuning, a known hazard of the test. Reports an adjustment diagnostic and applies the remedy, and checks the primary test is untouched. |
| `lab17_horizons.py` | Whether five days is the result or the setting. The whole experiment at h = 1, 5, 10 and 21, with the target/feature overlap at h = 1 measured rather than assumed. |
| `lab18_economic_reading.py` | What the R² column means to someone sizing a position: how far wrong the variance forecast is as a multiple of the truth, split by which side it errs on. No strategy, no P&L. |
| `lab19_vstoxx_third_series.py` | Whether redundancy is an artefact of a thin options block. Adds spot VSTOXX on the 1,240 days where all three series exist, holding the days fixed so the third series is separated from the shorter, calmer window. |
| `lab20_ceiling_is_not_a_ceiling.py` | Two referee corrections to the companion note, tested rather than conceded. The "ceiling" is exceeded at eight of ten delays; the sufficiency condition is not what fails, the centring assumption is. |
| `lab21_stronger_inference.py` | Re-measures the cost of breadth with a generator that fits nothing — the real foreign block shifted circularly, real missing-data mask held fixed — and gets the AR(1) figure at every delay. Prints both ways of getting that null wrong: gaps that move with the values, and an offset that is a near-identity. Part B is a HAC Fieller confidence set for the substitution rate. |
| `lab22_factor_benchmark.py` | Fits the cheaper representation Section 5 used to infer: real-time walk-forward PCA of the foreign block, one factor against seven regressors. |
| `lab23_compressed_everywhere.py` | Carries the compressed block through all three losses, and retests redundancy given implied volatility with ONE regressor instead of seven, which separates "no information" from "estimation cost". |

Run them in that order. Each one exists because the previous one raised an
objection that could not be answered without it.

## Data

`data/` holds daily OHLC for eight equity indices — S&P 500 (SPX), Nikkei 225
(N225), S&P/ASX 200 (AXJO), Hang Seng (HSI), Nifty 50 (NSEI), FTSE 100 (FTSE),
DAX (DAX), Bovespa (BVSP) — plus two implied-volatility indices, CBOE VIX and
VDAX-NEW. January 2000 to September 2026, two exports each because the provider
caps any single export at 5,000 rows.

All series are retail end-of-day exports from **Investing.com**, downloaded in
September 2026. Columns are `Date, Price, Open, High, Low, Vol., Change %`;
`Price` is the close and is the only price column the labs read besides OHLC.
There is no point-in-time guarantee — these are current-vintage files, not a
historical archive, and the paper says so. They are included so the results
can be reproduced exactly; they remain the provider's data, not the author's,
and `LICENSE` says so. The code, the papers and this README are MIT. The environment is NumPy and pandas
at the versions in `requirements.txt`; nothing else is imported anywhere.

`lab03` audits every join before using it. Seven of the eight overlap by 120 to
129 trading days and agree exactly across every overlapping row. The eighth, the
S&P itself, has no overlap: the exports abut. That seam is reported as
unverifiable and falls back to two weaker checks — contiguity across the join
and the absence of a level shift. It prints this on every run, including the
failure mode, and the paper says so.

`data/single_market/XYZ.csv` is a shorter S&P export used only by `lab02`, kept
separate because the file discovery in later labs would otherwise treat it as a
third S&P chunk.

The price data are a retail export with no point-in-time guarantee. That is a
limitation, and the paper states it.

## A known hazard of the Clark-West column, tested here

Clark-West adds back, in full, the estimation penalty the larger model pays for
its extra coefficients — because its adjustment is derived on the null that those
coefficients are zero in population. Shrinkage removes part of that penalty. So
if the two arms are shrunk by different amounts, the adjustment restores more
than was ever charged and the surplus inflates the statistic.

This project's `cv()` picks the ridge penalty on a validation slice, separately,
for each arm, so the precondition is present in our own Section 6, and the two
arms really do disagree: different penalties at 18%–50% of refits, with the
cross-sectional arm usually the more heavily penalised — the direction that
would inflate the statistic.

`lab16` tests it rather than reasoning about it. The decisive check needs no
calibration from anywhere: impose one penalty on both arms, chosen on the
restricted arm, so the two shrinkages are equal by construction. Doing that moves
no Clark-West statistic by more than 0.05 and no Giacomini-White statistic by
more than 0.06, and changes no verdict at any delay. `lab16` also reports the
ratio of the mean adjustment to the mean loss differential as a description of
how hard the correction is working — not as a threshold, since this repository
has no external calibration for what value should worry you.

## What breadth does not buy

`lab18` translates the R² column into the units a risk manager works in: how far
wrong the variance forecast is as a *multiple* of the truth. At eleven weeks the
stale domestic model is outside a factor of two on 59.2% of days and the
cross-sectional model on 37.1% — a 22-point improvement, and the same fact as the
headline in a form someone can act on.

Splitting that tail changes the recommendation. The stale model does not fail by
*under*-forecasting; it fails by over-forecasting, its median forecast/realised
ratio drifting from 1.26 at δ = 0 to 1.90 at eleven weeks while the
cross-sectional model's stays at 1.27. Almost the entire improvement is breadth
pulling that overstatement back. The understating tail — forecasts below half the
truth, the side that leaves a position too large — sits between 8.6% and 11.4% of
days and is unmoved at every delay, changing by under a point and in the wrong
direction at three of four. Adopting the cross-section as protection against being
caught short by a volatility spike would be adopting it for something it does not
do. It stops a stale forecaster systematically over-estimating risk, which is a
real cost and a different one.

## Claims that were tested and withdrawn

Kept here because the reasoning is more useful than the conclusion.

- **λ₂^(2δ) decay as a general result.** True only for a two-state chain. For
  three states the gaps run −0.037 to −0.103. The non-asymptotic object is the
  Dobrushin coefficient.
- **An optimal delay δ\*.** Derived twice, wrong twice. The first attempt
  predicted a level floor that simulation refused to produce; the second
  linearised φ̂^δ, which is exponential in δ and cannot be linearised. Finding
  out *why* led to the real mechanism, which is bias rather than variance.
- **That the cross-section harms a current forecast.** The small negative at
  δ = 0 does not survive simultaneous bands across the delay grid. The paper
  claims only failure to help.
- **That the substitution rate plateaus.** It is still rising at seven weeks.
  Only R(55) vs R(34) fails to reject equality.
- **A band where the signal is real but cannot pay for its own estimation.**
  Tempting, and not supported: at δ = 1 the gross interval includes zero.
- **Anything about the δ = 0 row.** Two losses say the cross-section is slightly
  worse with current domestic data; the variance-scale robust loss says slightly
  better, and its interval covers zero. A sign that moves with a monotone
  transform of the target is not a finding, so the claim — and the disagreement
  with Korkusuz and Jayawardena it was used to press — is withdrawn.

Three literature gates were run against the project, and each killed something.
The ceiling's novelty went to Andersen & Bollerslev (1998); the bias mechanism
to Buccheri & Corsi (2021); the cross-market channel to Engle, Ito & Lin (1990)
and, most closely, to a preprint posted three weeks before this work. What
survived is the delay sweep itself and the framing of the cross-section as a
*substitute* for timeliness rather than an *augmentation* of a current
information set.

## The horse race

Korkusuz (2025) and Buncic & Gisler (2016) both find implied volatility drives
much of the cross-market gain, so a referee will reasonably ask whether the
foreign block here is a slow proxy for options data available with no delay at
all. `lab08` runs that race, and Section 7 of the paper is built on the answer.

Implied volatility dominates. At δ = 21 the domestic-only model scores 0.108,
adding foreign closes gives 0.375, and adding VIX and VDAX instead gives 0.507 —
more than the entire delay-induced loss, so a forecaster three weeks stale with
current implied volatility beats one with fully current domestic data. On top of
implied volatility the foreign block adds nothing: the incremental figure is
negative at all ten delays, and once the estimation cost from `lab07` is netted
out the gross content is indistinguishable from zero at every delay tested. This
holds whether VIX is lagged a day to respect our own timing rule or used same-day
in the objection's favour.

A referee pressed a sharper version: redundancy measured with ridge is redundancy
under *linear* specifications, and a non-linear model might find cross-market
structure that bypasses the single global factor. `lab09` had already tested
non-linearity - but on the domestic-plus-foreign feature set, never on the one
carrying the redundancy claim. `lab15` closes that. Two things make it a fair test
rather than a formality. The penalty grid is widened thirtyfold, because the widest
arm carries 72 regressors and the appendix shows validation already pinned to the
top of the paper's grid with ten; and on the wider grid the implied-volatility
baseline is itself slightly weaker, so the bar the foreign arms must clear has been
lowered. Nothing clears it: no arm adds at any delay, both random-feature arms are
significantly worse at all six, and the 60-wide layer is worse than the 20-wide one
everywhere. More flexibility finds less, which is what over-parameterisation looks
like when the signal is not there.

One thing did change, in the objection's favour. On the wider grid the *linear*
increment stops being significantly negative from the second delay on. That is the
result `lab07`'s estimation-cost argument predicts: given implied volatility the
foreign block is not harmful, it is empty, and the apparent harm was the bill for
estimating regressors that carry nothing. The conclusion is unchanged; its sign is
better explained.

The measurement is unaffected; the recommendation is not. Where a liquid options
market exists on the stale asset, use it. What that leaves is the case the paper
opens with — private marks, appraisal cycles, fund NAVs — where no options market
on the asset exists, and which this experiment, run on the most heavily optioned
index in the world, cannot itself demonstrate. That is the stated scope of the
result.

`data/` holds two VSTOXX series and neither is in the headline block.

The *Mini Futures* contract starts in 2013 and is not used at all. The **spot**
index — filed under "STOXX 50 Volatility VSTOXX EUR", symbol **V2TX**, which is why
a search for "VSTOXX" returns only the five futures contracts FVSc1–FVSc5 — was
obtained afterwards and is used by `lab19` alone. Its seam is clean (253 overlapping
rows, 0 mismatched), but the export runs **2012-12-28 to 2025-03-28**, so it covers
neither the start of the sample nor the present.

That matters more than it sounds. The walk-forward needs 252 + 1,250 + 250 days of
burn-in plus the longest delay before it yields a single test day, so *requiring*
all three implied-volatility series moves the test window from 4,525 days beginning
September 2008 to 1,240 days beginning April 2020 — a 73% cut that discards the 2008
crisis and starts after the March 2020 spike (peak VIX 82.7 over the full sample,
41.4 inside what remains). Adding it to the headline would make the result less
established, not more, so it is a sensitivity run instead.

To fix it properly you would re-export from 1999 (the index's real start) to the
present, in three or four overlapping chunks under the 5,000-row cap. The current
two chunks were pulled from 2013 onward.

Adding it needs one code change beyond the download: `VSTOXX` is not in `IV_TAGS`,
so nothing currently loads it. The keyword discovery is already prepared —
`IV_EXCLUDE` keeps "mini", "futures" and "fvs" out of the VSTOXX match, because the
spot export and the futures export *both* contain the string `vstoxx` and without
that guard `load_iv` would concatenate an index and a futures contract into one
column. That is the same failure mode as the `DAX_New_Volatility` collision this
repository already had to fix once.

## Author

Sunny Alex Vellanikaran — [github.com/SunnyAlexV](https://github.com/SunnyAlexV)
