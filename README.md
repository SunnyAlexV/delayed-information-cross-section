# What substitutes for a stale mark?

Two working papers and the code that produced every number in them.

A third paper by the same author — *Validation-Tuned Shrinkage Invalidates the
Clark-West Test for Nested Forecast Comparison* — is referenced here as [19] and
bears directly on this one's Section 6. Its own replication scripts live with
that paper, not in this repository; what lives here is `lab16`, which applies
its diagnostic and its remedy to the statistics in this paper. See *A known
hazard of the Clark-West column, tested here* below.

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

That benchmark table was, until recently, the one thing in either paper this
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

Both commands work whether the repository keeps its folders or arrives flat.
The tidy layout has `labs/`, `data/`, `expected_output/` and `papers/source/`
as siblings of the two scripts, and that is what the paths below describe. A
web upload that drags files rather than folders flattens all of that into one
directory; the scripts detect it and run anyway, because a verifier that only
works under one directory arrangement is a verifier that quietly stops running.

**Extract the archive to a real folder before running anything.** Launching
`run_all.py` straight out of the zip preview copies only that one file into a
temporary directory, leaving `labs/` and `data/` behind; the script checks for
this and says so rather than failing on an unreadable `[WinError 267]`.

`--check` runs all sixty-three labs and diffs each against the stored output in
`expected_output/`. Two lines are exempt from the comparison — the data folder
path and the runtime — and nothing else is. A clean run prints
`all 63 labs reproduced the reference output exactly`.

### The paper and its Internet Appendix

`papers/what-substitutes-for-a-stale-mark.pdf` is the paper. `papers/stale-mark-internet-appendix.pdf` carries the robustness and scope work behind its
Section 10 and the two subsections it points into: the eight alternative domestic
controls, the Clark-West caveat, the temporal split, the multiple-testing adjustment, the
leakage check, the five-market comparison, the four scope conditions, the implementable
target at every delay, the coupling diagnostics, the house price overlap audit, the
seam audit, the state-space alternative, the unfitted-generator estimation bill, the
non-linear redundancy test, the horizon sensitivity and the matched-horizon race. The
paper states every one of those results; the Internet Appendix shows the working. `verify_paper.py` reads the two together,
so a figure cannot be lost by moving it from one into the other.

### Reproducing the papers' numbers

Every number in the three documents is printed by one of forty-nine scripts, all of them in the
table below, and all run under a fixed seed. The papers themselves carry only a short
availability statement; the mapping from result to script lives here, because it is a
property of the repository rather than of the argument.

    python run_all.py --check     rerun every lab and diff against expected_output/
    python verify_data.py         check your copy of the inputs against the manifest
    python verify_paper.py        check every figure in all three documents
    python build_registry.py      build results_registry.tsv and run five build gates

`build_registry.py` exists because every error an external audit found in these
documents was one error wearing different clothes: a number copied rather than
referenced, the original moved, the copy left agreeing with a run that no longer
existed. A check that a figure appears *somewhere* is satisfied by whichever copy
happens to be current, so three separate quantities each sat at two different
intervals for months while their checks passed.

The registry is built **from** `verify_paper.py`'s own checks rather than beside
them, because a registry maintained separately would be one more copy to drift.
Running the verifier with `VERIFY_DUMP` set records every figure it demands and
which lab that demand came from; `results_registry.tsv` is that record, one row
per checked result with its label, script, section, kind and figures. Over it run
five gates:

1. **Conflicting intervals.** One point estimate quoted in prose with two
   different intervals. Table rows are excluded, because a column legitimately
   repeats a value across cells; a sentence names its quantity, so a sentence
   cannot.
2. **Prose p-values against stored statistics.** A p-value written beside a z
   must be that z's p-value, to the precision the text itself quotes.
3. **Reversed contrast signs.** A claim that a contrast favours one side, sitting
   beside a negative statistic, requires the direction of the contrast to be
   stated in the same passage.
4. **Unlinked numerical literals.** Any distinctive figure in the documents that
   no check ties to a lab. These are not errors, they are unguarded surface, and
   the gate is a ratchet held in `registry_ceiling.txt`: the count may fall and
   may not rise.
5. **Negative signs through the PDF.** A minus that does not survive rendering
   and text extraction turns a figure into a different figure. Two reviewers in
   a row read the companion note's δ = 55 correlation as positive and called the
   benchmark beside it impossible; it is negative. The gate extracts every built
   PDF — the three documents and the six journal-variant files — and checks that
   each negative figure still carries its sign.

The equity results run on the data described in the papers' data section. The two illiquid
labs, `lab55_illiquid_measured.py` and `lab56_seasonal_and_breadth.py`, run on the
twenty-metro Case-Shiller panel in `data/illiquid/`, which ships with `PROVENANCE.md`
recording a SHA-256 digest of every series as computed at the source before transfer.

Four of the scripts were additionally reproduced on a second machine running a different
operating system and matched digit for digit. The rest have been run only on the platform
that generated the reference output, and the papers state that rather than the stronger
claim. The seam audit prints on every run, including the seam it could not verify.

`check_on_windows.bat` runs the same thing by double-click, for anyone who
would rather not open a terminal. It finds Python, runs the check, writes
`check_windows.log`, and prints the one line that matters at the end.

The reference output in `expected_output/` was generated on Linux. Four of the
sixty-three labs — 04 through 07 — were separately reproduced on Windows during
development and matched digit for digit. The other fifty-nine, including every lab
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
paper sources (in `papers/source/`, in `papers/`, or beside itself) and asserts that every cost, gross, interval and
test statistic they print matches `expected_output/`, that no superseded value has
survived, and that no withdrawn claim has crept back. One thousand one hundred and forty-seven checks across all three documents; exit 1 on
any disagreement. Two stale numbers reached a draft before this existed.

Both commands must be run **from the extracted repository folder**. They resolve
`labs/`, `data/`, `expected_output/` and the paper sources relative to themselves,
so a copy of a single script saved loose on the Desktop will not work and, worse,
may be an older version than the one in the repo.

Drop `--check` to just run and read. Pass a lab name to run one:
`python run_all.py lab07`. The whole suite takes about eighty minutes. `lab07` is seven minutes of it, because its
estimation-cost figure averages over a hundred surrogate draws, and `lab15` is eight,
because it runs five feature sets across two penalty grids and then prices three of them
against surrogates.

Requirements are NumPy and pandas. Nothing else — the ridge regressions, the
IRLS logistic, the Kalman filter and the bootstraps are all written out, partly
so that the walk-forward's fairness rule is visible rather than buried in a
library call.

### The submission versions, and the scripts that make them

The paper is submitted to different journals in different shapes: a forecasting
journal wants the ragged-edge experiment in the lead, an empirical finance
journal wants the question in the title, and a financial econometrics journal
has a forty-page limit the base manuscript does not meet. Three hand-edited
copies would drift in exactly the way this repository spends its life
preventing, and the drift would be invisible because each copy would look
internally consistent.

So there is one source, in `papers/source/`, and a generator:

    python build_variants.py      regenerate papers/variants/, verify each
    python measure_pages.py       page count in the geometry a page limit means
    python verify_letters.py      the cover letters against what they describe
    python build_manifest.py      write data/MANIFEST.tsv from a local data copy

`build_variants.py` holds only the *differences* between versions — a title, an
abstract, some framing, and for one journal a list of sections to relocate —
and then rebuilds each version from the base, renumbers every heading and every
in-text reference, renders the PDFs through `build_pdf.py` so all five documents
share one set of render settings, and runs `verify_paper.py` over the result. A
correction to the base therefore reaches all three submissions or fails loudly.
Nothing is deleted to meet the page limit: sections are moved into the Internet
Appendix, keeping every check that was pinned to them.

`measure_pages.py` exists because a page limit is not a count of pages in the
PDF an author renders. It is a count in a typewritten geometry — US Letter,
one-inch margins, 12-point serif, double spaced — and this repository's house
format is A4, 10.5pt and single spaced. The same manuscript is 27 pages in one
and 43 in the other, and neither number is wrong. The script reports two
columns, because the guidelines do not say whether table bodies are double
spaced too, and a single number there would be a preference dressed as a
measurement.

Two things are deliberately **not** in this repository. `contact.txt` holds the
corresponding author's telephone number, which JFEc's instructions require on a
title page and which has no business in a public source file; the published
front matter says the number is supplied with the submission, and
`build_variants.py` inserts the real one into the journal copies if that file is
present. `submission/` holds the cover letters, which are correspondence. A
clone missing either builds and verifies cleanly — the checks that read them say
so and pass rather than failing for everyone but the author. `papers/variants/`
is generated and not committed, for the same reason: built on the author's
machine it carries the number the source keeps out.

`verify_repo.py` answers the question a reader asks first: is everything here
actually part of reproducing the result? It checks that every lab ships with the
stored output it is diffed against and every stored output with its lab, that no
second copy of a manuscript source exists, and that no hidden directory ships —
one did, a `.pre-jfec/` snapshot of all three manuscripts taken before a risky
edit, invisible in a file listing and 817 diff lines behind the live source. It
also holds the README to the repository: lab counts, gate counts, one table row
per lab, every script named, every filename resolving. It exempts nothing,
including itself.

`build_ssrn.py` binds the paper and its Internet Appendix into the single PDF
SSRN accepts, paper first. A reader who finds the paper there and cannot see
the robustness work has been shown the claims without the working, so the two
travel together. A concatenation is where documents go missing quietly — a
dropped last page, or a re-encoded font that stops a minus sign extracting —
so the merge is checked rather than trusted: the page count must be the sum of
its inputs, a sentence from near the start and near the end of each input must
still extract from the result, and every negative figure must still carry its
sign. The companion note posts as its own SSRN entry. The script also writes the
SSRN form's own fields — title, abstract, keywords, JEL codes — lifted out of
the documents rather than retyped, because an abstract typed into a web form
is a copy of figures the papers keep current, and it is the copy every reader
sees before they open the PDF. It refuses any field that still contains
markup: the first version produced `S&amp;P 500` and `&delta; days`, which
would have gone into the abstract field exactly as written.

`repo-about.txt` holds the one sentence that is not in this repository at all: the description GitHub shows beside the URL the papers print. It went stale twice — once naming a paper title that no longer existed, once claiming a script count that matched nothing — because it lives on the website where no check could reach it. It lives here now, its count is checked against `labs/`, and publishing it is a paste rather than a retype.

What ships is 157 files, and 126 of them are the sixty-three labs and the
sixty-three outputs they are checked against. That pairing *is* the
reproduction; there is nothing to thin out without thinning the claim.

Two of the three target journals review double-anonymised and all three ask for
the title page as a separate file, so `build_variants.py` also emits a blinded
pair per journal: `manuscript-anonymous`, `internet-appendix-anonymous` and
`title-page`. Blinding is not a matter of deleting a byline. Four things carry
identity here and only the first is obvious: the byline and contact block; the
two in-text citations of the author's own companion note, which a referee reads
as a name whatever the title page says; the repository URL, which is
`github.com/SunnyAlexV` and so is the author's name spelled out; and the
document title element. After stripping them the builder searches the result for
the name again and **refuses to write the variant** if it finds it — which it
did, twice: once because the Internet Appendix ships alongside the manuscript
and was not being blinded at all, and once because the appendix's front matter
uses centred `class="t"` paragraphs rather than the manuscript's byline classes,
so the manuscript's patterns matched nothing and removed nothing. The title page
lists what was removed, so an editor does not have to diff two files to satisfy
themselves that none of it was content.

`verify_letters.py` checks the one document nobody re-derives. A cover letter is
written once and then the paper gains checks, loses pages and changes length,
and the letter goes on telling an editor the old numbers. Every count in the
letters — scripts, checks, words, pages, input files — is recomputed from the
thing it describes, and every interval the letter quotes is looked up in the
manuscript it is sent with.

## What is in each lab

| lab | establishes |
|---|---|
| `lab01_ceiling.py` | The closed-form ceiling, checked against an explicit Kalman filter at 14 parameter settings. Passes to within 2%. |
| `lab02_delay_curve.py` | The decay curve on a single market, with a matched-persistence baseline rather than a strawman. |
| `lab02b_threshold_ceiling.py` | The Gaussian benchmark the threshold rule is aiming at, from the orthant probability, against the accuracy it reaches. Produces the companion note's main table. |
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
| `lab24_proper_scores.py` | Scores the companion note's own five forecasters under AUC and under the Brier and log scores, not 0-1 accuracy. Reports what a hard rule cannot be scored on at all, and what discretising a margin costs in ranking. |
| `lab25_window_sensitivity.py` | Varies the 252-day median and the 1,250-day training window over a grid, every cell on one common test window. Includes a power control separating a real window effect from a smaller sample. |
| `lab26_quantile_target.py` | Generalises the note's ceiling off the median: derives the Bayes cut z_q/rho, integrates the bivariate normal benchmark numerically, and checks that integrator against the closed form at q = 0.5. |
| `lab27_regime_conditioning.py` | Re-runs both headline results inside market states, on a ladder of stress definitions from the top third of VIX days down to the top 5%, plus named crisis episodes. Every figure is scaled by the benchmark model's own error inside the same regime, because raw squared errors are larger in stress whatever the model does. |
| `lab28_orthogonal_breadth.py` | Splits the foreign block into its real-time global component and the remainder orthogonal to it, and asks which half carries the recovery. Checks first that the rotation reproduces the raw block. |
| `lab29_single_regime.py` | **Exploratory — cited by neither paper.** Breaks lab27's pooled episode figure into one row per crisis, drops each episode in turn to see whether one carries it, and walks the whole 2007–2010 arc phase by phase. Also measures how much crisis history each model had actually been fitted on. |
| `lab30_information_bound.py` | **Exploratory.** A ceiling from the entropy power inequality: R2 <= 1 - [N(Y)/Var(Y)] exp(-2I). Needs no distributional assumption, so it survives where the Gaussian ceiling did not. Estimator validated against the one closed form, and debiased by permutation. |
| `lab31_mark_to_model.py` | **Exploratory.** The practitioner's method — estimate factor exposures, roll the stale mark forward on current factor returns — against the paper's regression on the same data. |
| `lab32_masked_training.py` | **Exploratory.** One model for every delay, trained by drawing a random staleness per row, against ten separately fitted ones. Includes delays never trained on, and a train-short/test-long extrapolation. |
| `lab33_ragged_edge.py` | **Exploratory.** A one-factor Kalman filter on the ragged edge — the target missing for the last delta days, the peers current — against the paper's per-delay regression. |
| `lab34_data_snooping.py` | White's Reality Check and Hansen's SPA across the nine forecasters this project has built, so the headline is tested against the number of models tried rather than one at a time. Size validated by simulation first. |
| `lab35_purged_cv.py` | Varies the gap between training and forecast date, from removing the purge entirely to adding a 21-day embargo on top, to test whether the walk-forward design's existing gap is sufficient. |
| `lab36_variance_risk_premium.py` | **Exploratory.** Races the variance risk premium, not the implied-volatility level Section 6 tested, and separates the premium a delayed desk can actually form from the one it cannot. Every comparison holds the regressor count fixed so the estimation cost cancels. |
| `lab38_domestic_baseline.py` | Varies the domestic control that R(delta) is measured against: five stronger blocks, the rate recomputed inside each. Finds the level of volatility, which the median normalisation hides, worth more at long delays than the whole foreign cross-section. |
| `lab41_conditional_anatomy.py` | Takes R(delta) apart into the error the cross-section removes and the error the delay adds, by market state, and applies Appendix C's Fieller construction to all twelve cells. Stress multiplies the numerator by 10.7 and the denominator by 6.2. |
| `lab42_factor_model.py` | The factor model the results assemble, and its three predictions: geometric decay at a rate matching the factor's own persistence (0.9646 against 0.9678), near-flat cross-sectional skill, and a ceiling of 69.6% the curve reaches. |
| `lab40_level_or_normaliser.py` | Tests whether Section 9.1's level result is information or the model undoing the paper's own normalisation, by re-dating the normalising median three ways. The implementable target keeps 87-102% of the gain, so it is information. |
| `lab39_temporal_stability.py` | Splits the test window, walks a rolling R(delta), and compares the halves within VIX terciles. The stressed-market rate is stable; the calm-market rate is not identified. |
| `lab37_lead_lag.py` | Propagation or a common factor: the lead-lag asymmetry against what the trading clock alone predicts, and what it costs when the foreign feed is itself stale, priced in days of domestic mark age. |
| `lab43_official_vix.py` | Audits the paper's VIX against Cboe's own published history: 99.91% of 6,747 shared rows identical to the cent, two shortened sessions outside the 0.5% seam rule, and fifteen rows in the retail export on days United States equities were shut. None reaches the panel; substituting the official file moves no cell of Section 8. |
| `lab44_horizon_matched_iv.py` | The horse race with the horizons matched, using Cboe's nine-day VIX9D against a five-day target. Nine-day beats thirty-day at every delay (GW 4.02 to 4.19), and the foreign cross-section still adds nothing on top of it. |
| `lab45_effective_age_interval.py` | An interval for the paper's most-quoted number. Resamples test days in moving blocks and re-interpolates the effective age inside each replication: 4.6 days becomes 4.6 [2.8, 7.7], and a day of foreign staleness costs 2.1 [1.4, 3.3] days of domestic freshness as a paired difference. |
| `lab46_own_only_filter.py` | Answers two referees who called the own-only control a straw man because a ridge cannot project a stale observation forward. Fits a frozen AR(1)-plus-noise state space on the domestic series alone, shows the projection is absorbed exactly by a standardised ridge (1.8e-11 across six delays), and finds exponential weighting of the whole history worse at every delay. The rate against that weaker control would read 74.3% rather than 71.4%. |
| `lab47_target_seam.py` | Closes the paper's oldest data limitation. The S&P's two exports abut without overlapping, so no row-by-row check exists; this audits the join against Cboe's own VIX history using three instruments, one per failure mode, each calibrated on 400 pseudo-seams. The join sits at the 70th, 4th and 25th percentiles. A one-session misalignment, a spliced block and a range mis-scaled by 25% are all detected. |
| `lab48_age_by_regime.py` | Asks the calm-market cell again in a coordinate with no denominator that can vanish: the effective age, matched on mean squared error. It still does not answer, 4.5 days [0.3, 37.0] against the stressed 3.8 [2.2, 6.4], so the non-result is about markets rather than about ratios. Along the stress ladder the age falls to 2.3 days [1.0, 4.5] in the top 5% of VIX days. |
| `lab49_control_envelope.py` | Prices the domestic control as a set rather than one arm at a time: eight blocks, the rate recomputed inside each, selection on own-only skill and never on the rate. The rate at eleven weeks spans 60.8% to 74.4%, and 67.5% against the strongest control. Also re-runs Section 8's horse race inside that control, where redundancy holds at 0 of 6 delays. |
| `lab50_filtered_headline.py` | Recomputes the headline inside the state-space specification Section 11 prices but declines to adopt. The rate rises to 75.4% from 71.6%, paired +3.8% [+1.5%, +6.6%], and the effective age falls to 3.7 days from 4.6. The better estimator makes the paper's claim larger, so the published figure is the conservative one. |
| `lab51_foreign_options.py` | Asks whose options market redundancy needs. With an option chain on the asset itself, implied volatility beats breadth by 31 points and breadth adds in 0 of 8 cells. With only another index's options market, the gap falls to 3 points and breadth adds significantly in 5 of 24 cells. The scope condition behind Section 8, measured for the first time. |
| `lab52_compressed_foreign_options.py` | Re-runs the eight-target race with the block Section 5.1 tells a practitioner to carry, one column instead of seven. The split sharpens: 0 of 8 cells significant where the option chain is the asset's own against 16 of 24 where it is another index's, and the gradient behind it goes from -0.38 to -0.78. Also explains the one target that did not fit: Bovespa was paying for six coefficients it did not need. |
| `lab53_no_options_targets.py` | Four targets with no published volatility index at all - Pakistan, Indonesia, Sri Lanka, Malaysia - against the four Asian targets that have one this paper does not hold, on the same clock and the same code path. Two behave like the matched group; two return negative rates. What sorts them is coupling to the cross-section, which tracks the substitution rate at +0.98 across the eight. |
| `lab54_monthly_feasibility.py` | The rehearsal for `lab55`, and the file that established the design could be re-specified at all. The daily design cannot be pointed at a quarterly-marked asset: it needs 1,752 observations with an intraday range, which at one mark per quarter is 438 years. So the experiment is re-run on month-end marks alone, and it survives: 48.7% [34, 76] at a one-quarter delay, 43.8% [28, 75] once the mark is smoothed the way an appraisal is. What was missing was two to four decades of marks, not a method, and `lab55` supplies them. |
| `lab55_illiquid_measured.py` | The paper's largest limitation, closed with a measurement rather than an argument. Twenty metropolitan Case-Shiller house price indices, monthly 1987 to 2026: smoothed by construction, no intraday range, no option chain on any of them, and a real cross-section of twenty marks published on the same day. At one quarterly appraisal cycle of extra staleness the rate is 67.8% [45, 92] on the next month's return, positive on 14 of 14 metros, and 19.0% [6, 31] on this paper's own target type, positive on 13 of 14. Two findings beyond the headline: the measured persistence of real indices (0.601 to 0.939, mean 0.804) exceeds the rho = 0.6 Section 13 imposed, so the synthetic case was conservative; and with all thirteen peers as separate regressors the variance rate is NEGATIVE, turning positive only under the compressed block, which makes compression part of the method rather than a robustness check. The peer block is chosen by a rule that cannot see the rate and breaks ties toward fewer regressors, so the headline is the low end of its own menu. |
| `lab56_seasonal_and_breadth.py` | The two ways `lab55` could be wrong. These series are not seasonally adjusted and month-of-year explains 19.9% of monthly return variance with profiles correlating +0.894 across metros, so a fresh peer could have substituted by revealing the calendar. A placebo peer built to carry the seasonal and nothing else earns 4.2% where the real cross-section earns 73.0%, and -2.8% against 18.6% on the variance target: a test that could have convicted the design and did not. Then the other direction - widening from 13 peers to 19, six of them genuinely ragged - moves the rate by less than a point, so thirteen peers already span the one national cycle this panel contains. |
| `lab57_housing_overlap.py` | Whether `lab55`'s rate is the index's own three-month moving average, rather than information. Each published figure averages three months of closings, so consecutive monthly changes overlap and a manufactured rate would look like a rate. The design is re-run on non-overlapping quarterly changes - the index read every third month, on all three phases of the grid, so no two quantities share a transaction month - where the artefact explanation predicts zero. Pooled across metros the rate is 57% at one quarter of staleness, spanning 49% to 63% across phases, positive in 35 of 42 metro-phases and with each phase's interval excluding zero. Lower than 67.8%, which is what a modest smoother contribution would look like, and nowhere near zero. De-smoothing by Geltner's reverse filter agrees and adds a finding: with the moving average inverted out, one month of staleness costs 0.0005 of R-squared, so R(delta) has no denominator and no rate is quotable - smoothing is most of what makes a stale mark expensive. Two estimators are reported throughout, because a mean of per-metro ratios is fragile on a sample this size. |
| `lab58_ratio_inference.py` | Whether the paper's interval actually covers R(delta), which nobody had measured for a ratio of skill differences computed from rolling, re-estimated, regularised forecasts of an overlapping target. Applying Fieller and Dufour correctly is an implementation; the coverage is the result. Eighty simulated histories per regime, the truth taken as the mean over a hundred noise redraws on that same history. An iid bootstrap covers 64% at a nominal 95%, which is the failure this design invites. The paper's moving-block bootstrap at 2h covers 92%, Fieller/HAC at 2h covers 95%, and lengthening either to 6h or 12h moves coverage to 96-98% - conservative rather than optimistic. The same file is why HAC_LAG equals the block length: the automatic bandwidth understates the standard error by up to 34%, worst at the long delays the paper leads with, while HAC(40), HAC(80) and a kernel-free bootstrap all agree. |

| `lab59_session_timestamps.py` | Whether the admissibility rule is true on real exchange clocks rather than on the table of standard-time closes that implements it. Every close is rebuilt in UTC from its exchange's own local closing time and timezone, daylight-saving transitions included, on all 7,021 dates, and the lag the table assigns is compared with the lag the true clocks imply for every target, peer and date. There are zero leaks: not one day on which a real session clock would have reordered a block, with the narrowest margin the rule relies on anywhere at 2.00 hours. The audit also finds the implementation is stricter than the rule it implements, on 6,099 target-peer-days across the panel, because the table records two markets at the same standard-time hour and equality is resolved against admissibility. So the design discards information a forecaster could lawfully have used, which makes every rate here a lower bound on the rule's own terms. Holidays are reported in the same place: a peer is carried forward on at most 5.8% of the target's days, with a mean age under three days. |

| `lab60_outage_decomposition.py` | Whether delta is a stale FEATURE or a stale SYSTEM. The training cut of equation (4) is forced by the vintage assumption, but its consequence is that delta freezes the current domestic state, the available labels, the coefficient vintage, the ridge penalty and the quantity of training data all at once, so the design measures a domestic-data outage rather than a stale mark. This file splits the two with a third arm that no forecaster could run: stale input, coefficients refitted to t - h. Freezing the estimation accounts for 0.4% of the delay's damage at one week and 2.7% at eleven, so the denominator R(delta) divides by is information and not estimation, and the rate recomputed inside the infeasible arm is at most 2.8 points higher, which makes the paper's design the conservative one. The arm is labelled infeasible everywhere it appears, in the same sense as lab10's infeasible median column. |
| `lab61_block_sensitivity.py` | What the companion note's block length was worth. Every interval in the note comes from a moving-block bootstrap, whose block is a free parameter no data choose: too short resamples dependent blocks as independent ones and narrows every interval, too long leaves too few distinct blocks and widens them. The note had also described a block of 2h = 10 while producing its tables at 8h = 40. This varies the block over 10, 20 and 40 and runs the stationary bootstrap of Politis and Romano beside each at the same expected block length, so the shape of the block law is separated from its mean. Parity on accuracy holds under all six schemes; the AUC separation falls from seven delays at block 10 to five at 20 and 40, and its direction is unanimous across every scheme. Section 6.4 and Table 6 of the note. |
| `lab62_post_selection.py` | How much of the secondary tables is chance. lab34 prices the search behind the headline and nothing else; everything reported after the headline specification was fixed carried per-cell significance at the ordinary 5% level with no family-wise control. This enumerates those grids by name, counts their cells against the 0.05n a grid of pure noise would show, pools all 214 secondary cells and applies Bonferroni and Benjamini-Hochberg across the pool. 79 cells are nominally significant against 10.7 expected, so the grids collectively carry signal; 5 survive a false-discovery correction and 2 a family-wise one, so the other 74 are exploratory and the papers label them so. The headline is deliberately excluded from the pool. Section S37 and Table S30. |

Run them in that order. Each one exists because the previous one raised an
objection that could not be answered without it.

## Data

**The input data is not in this repository.** Every vendor behind it prohibits
redistribution, and an earlier version of this repository should not have carried
their files. `data/README.md` sets out the terms source by source; the short
version:

| source | series | may we redistribute? |
|---|---|---|
| Investing.com | the eight equity indices, VDAX-NEW, the retail VIX export, spot VSTOXX, VSTOXX mini futures, `single_market/XYZ.csv` | no — distribution forbidden without written permission |
| Cboe | `VIX_History.csv`, `VIX9D_History.csv` | no — one copy for personal non-commercial use only |
| Yahoo Finance | the four `frontier/` series | no — reproduction and distribution forbidden |
| FRED / S&P Cotality | the twenty-metro Case-Shiller panel, the two national Case-Shiller series | no — FRED labels these *Copyrighted: Pre-approval Required* |
| FRED / FHFA | `illiquid/USSTHPI_fred.csv` | **yes** — *Public Domain: Citation Requested*, and it is here |

Using those series to do research and publishing the results is a different act
from republishing the files, and only the second is what the terms forbid. One
of the thirty-five input files is redistributable and it is the one still here.

What ships instead is `data/MANIFEST.tsv`: per file, its source, row count, date
span, the SHA-256 of the exact bytes the published numbers were computed from, a
digest of the derived series on those dates, and that digest taken one calendar
year at a time. The derived digests are the point — a digest over a vendor's CSV
is a statement about that vendor's formatting, whereas a digest over the closing
prices actually fed to the estimator survives the vendor adding a month of rows.

```
python make_all.py --data <folder>              # every step below, in order
```

or the steps separately:

```
python verify_data.py  <folder>                 # per file: IDENTICAL, SAME DATA, DIFFERENT, MISSING
python run_all.py --check --data <folder>       # all 63 scripts against the stored output
python build_pdf.py                             # render the three documents
python verify_paper.py                          # all three documents against that output
```

`make_all.py` exists because the order matters and was only written down here:
running `verify_paper.py` before `run_all.py` compares the paper against the
previous output and passes when it should fail. `--fast` skips the two-hour lab
run and says so in its summary.

`verify_data.py` distinguishes a fresher export from a different series, and
names the calendar years in which a different series disagrees. `run_all.py`
without `--data` prints the same instructions rather than failing obscurely.

`data/README.md` gives the download recipe for each file: which page, which
frequency, which columns, and the two traps worth knowing — the 5,000-row export
cap that makes each retail series arrive as two files with a deliberate overlap,
and the spot VSTOXX being filed as "STOXX 50 Volatility VSTOXX EUR" (symbol
**V2TX**) where a search for "VSTOXX" returns futures contracts instead.

The panel itself is daily OHLC for eight equity indices — S&P 500 (SPX), Nikkei
225 (N225), S&P/ASX 200 (AXJO), Hang Seng (HSI), Nifty 50 (NSEI), FTSE 100
(FTSE), DAX (DAX), Bovespa (BVSP) — plus three implied-volatility indices: CBOE
VIX, its nine-day counterpart VIX9D, and VDAX-NEW, January 2000 to September
2026. `lab47_target_seam.py` audits all nine two-file joins: eight overlap by 120
to 251 trading days and agree exactly on every overlapping row, and the ninth,
the S&P itself, has no overlap because its exports abut. That seam is reported as
unverifiable and falls back on contiguity, the absence of a level shift, and an
independent audit against Cboe's own VIX history.

`data/illiquid/PROVENANCE.md` records, for every housing series, a SHA-256 digest
computed at the source before the data moved and re-computed on the stored file
afterwards, together with the single transformation applied to the metro panel: a
rounding to four decimals whose largest relative error anywhere is 1.0e-6.

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

- **λ₂^(2δ) decay as a general result.** True only for a two-state chain;
  simulation on three states broke it. The non-asymptotic object is the Dobrushin
  coefficient. The three-state figures that once stood here came from an
  exploration that is not in this repository, so they are no longer quoted: every
  number in this README is one the scripts print.
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
