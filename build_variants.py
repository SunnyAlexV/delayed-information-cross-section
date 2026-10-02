"""
build_variants.py - one source, three journal-targeted versions.

    python build_variants.py            build all three, verify each
    python build_variants.py ijf        build one

WHY A GENERATOR AND NOT THREE FILES
-----------------------------------
Three hand-edited copies of a paper is the same mistake this project spent its
whole life correcting: a number copied rather than referenced, the original
moved, the copy left agreeing with a run that no longer existed.  Three
manuscripts would drift in exactly that way, and the drift would be invisible
because each would look internally consistent.

So the content has ONE source, papers/source/, and this file holds only the
DIFFERENCES: a title, an abstract, some framing, and for one journal a set of
sections to relocate.  Rebuild after any correction to the base and all three
variants carry it.  Each variant is then run through verify_paper.py, which
reads a manuscript together with the appendix beside it, so a section moved out
of the paper and into the appendix keeps every check that was pinned to it.

WHAT EACH JOURNAL WANTS, AND WHY THE VARIANT DIFFERS
----------------------------------------------------
IJF   Forecasting, not finance.  The lead is the controlled ragged-edge
      experiment, the substitution rate as a general performance measure, the
      outage-versus-feature-only distinction, and inference for a weakly
      identified ratio under overlapping horizons.  The risk is reading as
      finance-specific, so a paragraph shows the design transferring to
      macroeconomic release delays, supply-chain lags and appraised values.

JEF   Empirical finance.  One question, asked in the title, answered in the
      abstract.  The private-equity and housing motivation moves out of the
      opening pages; housing stays as the external-validity test.  The lead is
      the identifying design and the evidence, not the practitioner story.

JFEC  Financial econometrics, and a 40-double-spaced-page limit.  Two jobs:
      present R(delta), its weak-identification inference and the real-time
      outage design as a reusable framework rather than one application, and
      cut enough to fit.  The cut RELOCATES sections into the Internet
      Appendix rather than deleting them, so nothing checkable is lost.
"""

import html as _html
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "papers", "source")
OUT = os.path.join(HERE, "papers", "variants")
PAPER = "what-substitutes-for-a-stale-mark.html"
SUPP = "stale-mark-internet-appendix.html"


# ----------------------------------------------------------------------
# the pieces that differ, per journal
# ----------------------------------------------------------------------
IJF_TITLE = ("Forecasting Under Information Delay: A Ragged-Edge Experiment<br>"
             "and a Substitution Rate for Timeliness")

# ----------------------------------------------------------------------
# The figures the abstracts quote, READ from the labs rather than typed here.
#
# Each of the three abstracts below used to carry its own copy of the
# conditional rate, the middle tercile and the headline.  That is four copies
# of each figure across this repository, and the copies drifted: when the
# inference moved to fixed-b critical values the source paper's headline became
# 71% [54, 83] and all three abstracts here kept [59, 81], which no check
# could see because the abstract check required only the "71%".  The same
# happened to the tercile figures.  So the abstracts now carry placeholders and
# the values arrive from the same outputs verify_paper.py reads.
# ----------------------------------------------------------------------
def _fig(labname, pattern, groups=(1, 2, 3)):
    src = open(os.path.join(HERE, "expected_output", labname + ".txt"),
               encoding="utf-8").read()
    m = re.search(pattern, " ".join(src.split()))
    if not m:
        raise SystemExit(
            f"build_variants: {labname} no longer prints the figure the "
            f"abstracts quote (pattern: {pattern}).  The abstracts are built "
            f"from the labs on purpose; fix the pattern rather than typing the "
            f"number back in.")
    pt, lo, hi = (m.group(g) for g in groups)
    return f"{int(pt)}% [{int(lo)}, {int(hi)}]"


FIGS = {
    # lab41's fixed-b Fieller sets by market state, at the headline delay
    "STRESSED": _fig("lab41_conditional_anatomy",
                     r"stressed 55 (\d+)% [\d.]+ \[([+-]?\d+)%, ?([+-]?\d+)%\]"),
    "MIDDLE": _fig("lab41_conditional_anatomy",
                   r"middle 55 (\d+)% [\d.]+ \[([+-]?\d+)%, ?([+-]?\d+)%\]"),
    # lab13's origin-dated headline, from part C2, which is the FIXED-b table.
    # Part C above it prints a row of the same shape from the percentile
    # bootstrap, so an unscoped search takes that one -- it did, and produced
    # the target-dated bootstrap figure instead of the headline.  Same scoping
    # as verify_paper.py uses for the same number.
    "HEADLINE": _fig("lab13_origin_median",
                     r"C2\..{0,900}?\s55\s+\d+% \[\s*\d+%,\s*\d+%\]\s+"
                     r"(\d+)% \[\s*(\d+)%,\s*(\d+)%\]"),
}


def _fill(s):
    """Substitute the lab-read figures into an abstract template."""
    for k, v in FIGS.items():
        s = s.replace("{" + k + "}", v)
    left = re.findall(r"\{([A-Z_]+)\}", s)
    if left:
        raise SystemExit(f"build_variants: unfilled placeholder(s) {left}")
    return s


IJF_ABSTRACT = """How much forecasting performance is lost when one input arrives late, and how
much of that loss can a timely substitute repair? We answer both in a controlled
ragged-edge experiment: a forecaster's own history is withheld for &delta; periods
while every other source stays current, so the delay is imposed rather than
observed and its cost is measured against the same forecaster undelayed. The loss
is summarised by a substitution rate R(&delta;), the share of delay-induced
forecasting loss a substitute returns, which is unit-free and
comparable across targets, losses and horizons, and which goes negative exactly when a
substitute costs more to estimate than it returns. Three things make it usable.
First, the design freezes a whole information system rather than one feature, and
we separate the two: an infeasible feature-only arm prices staleness of the input
against staleness of the estimated model. Second, R(&delta;) is a ratio whose
denominator vanishes as the delay cost goes to zero, so it is weakly identified at
short delays; we invert a HAC t-test after Fieller and report where the data do not
pin it down. Third, the targets overlap, and we set the bootstrap block and the HAC
bandwidth from measured dependence rather than convention. Applied to S&amp;P 500
realised variance with seven foreign equity closes as the timely substitute, a
current cross-section returns {STRESSED} of what eleven weeks of staleness takes in
this sample's stressed markets, cut in real time, {MIDDLE} in the middle tercile and
no stable value in calm ones where delay costs little; the unconditional rate pools the
three and is {HEADLINE}.
In both own-index option cases examined, the compressed cross-section adds no
detectable information once own-index implied volatility is included;
read as a clock, a fifty-five-day-old mark plus today's foreign closes forecasts as
well as a mark 4.6 days old [2.5, 8.2]. The same design run on twenty metropolitan
house price indices, where marks are smoothed and no options chain is listed,
returns 57% at one quarterly cycle of extra staleness."""

IJF_GENERAL = """<p><b>Where else this design applies.</b> Nothing in the construction is specific
to volatility or to equities. It needs three things: a target whose outcome is
eventually observed, one input that can be withheld on a schedule, and at least one
other input that stays current. Macroeconomic nowcasting has the same shape, where
a quarterly national accounts release is late and monthly indicators are not, and
the substitution rate would price a survey against the release it stands in for.
Supply-chain forecasting has it, where point-of-sale data arrive weekly and shipment
data monthly. Appraised asset values have it most sharply, since the mark is stale
by construction and the question of what may stand in for it is the one a valuation
committee actually asks. In each case R(&delta;) answers the same question in the
same unit, the outage-versus-feature-only decomposition separates a late system from
a late number, and the weak-identification caution applies wherever the delay costs
little, because that is where the denominator vanishes. Section 8 is the first of
these carried out: a panel whose marks are smoothed by construction rather than
withheld by design.</p>

"""

JEF_TITLE = ("How Much of the Forecasting Loss from Stale Domestic<br>"
             "Information Can a Timely Cross-Section Repair?")

JEF_ABSTRACT = """We measure how much of the forecasting loss caused by stale own-asset information
a timely cross-section of other markets can repair, and we identify the measurement
rather than assume it. Withholding an S&amp;P 500 realised-variance forecaster's own
history for &delta; days while leaving every other source current makes the delay an
imposed treatment rather than an observed condition, and the resulting loss is
summarised by a substitution rate R(&delta;), the share of that loss a substitute
returns. In this sample's stressed markets, cut in real time, a current foreign cross-section
returns {STRESSED} of what eleven weeks of staleness takes, stable across both halves
of the sample; it is {MIDDLE} in the middle tercile and has no stable value in calm
markets. The unconditional rate pools the three and is {HEADLINE} on the implementable
target and the original model,
with four specifications spanning 68% to 75% and every interval excluding zero. Read
as a clock rather than a share, a fifty-five-day-old mark plus today's foreign closes
forecasts as well as a plain domestic mark 4.6 days old [2.5, 8.2]. The rate is a
ratio whose denominator vanishes at short delays, so its inference is treated as the
weak-identification problem it is, by inverting a HAC t-test after Fieller. Two scope
conditions bound the result and both are identified in the examined markets rather
than assumed. In both own-index option cases examined, the compressed cross-section
adds no detectable information once own-index implied volatility is included, while a
chain written on another asset does not substitute in the same way; and coupling, the target's own
correlation with the cross-section offered to it, orders the rate across eight equity
markets. Twenty metropolitan house price indices, smoothed by construction and with
no listed options chain, carry the measurement to an illiquid market as an
external-validity test."""

JEF_OPENING = """<p>A forecaster whose own data arrives late is in a different position from one whose
data is merely noisy, and the difference is measurable. This paper measures it. We
withhold a forecaster's own history for &delta; days, leave every other source current,
and ask how much of the resulting forecasting loss a timely cross-section of other
markets returns. The answer is summarised by one number, the substitution rate
R(&delta;), and the paper is about identifying that number rather than about the
practitioners who would use it.</p>

<p>Two things make the exercise an econometric one rather than a descriptive one. The
delay is imposed on a schedule rather than observed, so the comparison is against the
same forecaster undelayed on the same days, and nothing is inferred from a
cross-section of forecasters who happen to differ in timeliness. And R(&delta;) is a
ratio whose denominator is what the delay costs, which goes to zero as the delay does;
at short delays it is weakly identified, and Section 4.6 treats that rather than
reporting a percentile interval and moving on.</p>

"""

# The JFEc variant relocates Section 2, Related work, to the Internet Appendix
# to meet the page limit.  That is defensible for a literature survey and not
# defensible for the sentence that says what is new: it left a manuscript whose
# roadmap sent a referee to a supplementary file to find out how the paper
# differs from what is already known, and whose body cited Engle, Rapach and
# Buncic in the reference list and engaged none of them.  The survey stays in
# the appendix; the distinction comes back into the introduction, where a
# referee decides in two minutes whether the paper is new.
JFEC_PRIOR = """<p><b>What is new here.</b> Volatility spillover research asks whether foreign
markets help forecast a domestic one that is otherwise current: Engle et al. (1990) separate
meteor showers from heat waves, Rapach et al. (2013) find the United States leads, and Buncic
and Gisler (2016) put a wider cross-section to work on realised variance. In each the domestic
series is fresh and the question is whether breadth adds to it. Here the domestic series is
stale by construction, so breadth is not an addition but a substitute, and what is measured is
not whether it helps but what share of a known loss it returns. That is why the object is a
ratio rather than a coefficient or a skill score: a buyer choosing between a wider data feed and
an options subscription needs the two priced on one ruler. The second difference follows from
the first. Because both candidates are priced in the same unit on the same days, breadth and
implied volatility can be raced against each other under delay rather than studied apart, which
Section 7 does. Section 2 places the paper against the wider literature in full.</p>

"""


JFEC_FRAMEWORK = """<p><b>What this paper offers beyond one cross-section.</b> Three components are
separable from the application and reusable without it. The first is the estimand:
R(&delta;) is a ratio of forecasting losses, unit-free,
comparable across targets, losses and horizons, negative exactly when a substitute costs
more to estimate than it returns, and defined so that the benchmark
divides out of it exactly, which is what lets rates measured against different
benchmarks be compared at all. The second is the identifying design: withholding one
input on a schedule while others stay current makes the delay a treatment rather than
a condition, and the infeasible feature-only arm of Section S36 separates a late
information system from a late number, which is a distinction any real-time design
must make and most do not. The third is the inference: a ratio whose denominator
vanishes is weakly identified, and the procedure here inverts a HAC t-test after
Fieller with the block and the bandwidth set from measured dependence rather than
convention, so the construction reports unbounded sets where the data warrant them.
A researcher with a different target, a different substitute and a different loss can
take all three unchanged; what they would replace is the cross-section, not the
apparatus. Section 4 states the framework in that order, and the empirical sections
are one instance of it.</p>

"""

# What stands in the manuscript where a relocated section used to be.  Each
# keeps the float and says enough for it to be read on its own.
# ---------------------------------------------------------------------------
# What stays in the body when construction detail moves to the appendix.
#
# JFEc asks for forty double-spaced pages and this manuscript, assembled in the
# order the journal asks for - every table and every figure on a page of its own
# at the back - measures fifty-one.  Eleven pages have to leave the body, and
# the question is which eleven.  Nothing is deleted: each block below replaces a
# section that moves to the Internet Appendix in full, and what the stub keeps
# is the part a referee cannot read the results without.
#
# The rule applied to every one of these: a stub keeps the DEFINITIONS and the
# DISCLOSURES, and sends the audits, the vendor detail and the discussion to the
# appendix.  An equation a later section cites stays; a paragraph defending a
# choice goes.  [[APP...]] is replaced by the appendix section the block became,
# which is computed, because a pointer typed by hand into a stub is the stale
# cross-reference this project has spent six rounds removing.
DATA_STUB = """<p>The data are daily open, high, low and close for eight equity
indices: S&amp;P 500 (SPX), Nikkei 225 (N225), S&amp;P/ASX 200 (AXJO), Hang Seng
(HSI), Nifty 50 (NSEI), FTSE 100 (FTSE), DAX (DAX) and Bovespa (BVSP), from
January 2000 to September 2026, of which the out-of-sample evaluation uses 4,862
days once the burn-in, training and validation blocks are spent, together with
three implied-volatility series: CBOE VIX, its nine-day counterpart VIX9D, and
VDAX-NEW. The two CBOE series are Cboe's own published histories; the eight
indices and VDAX-NEW are retail end-of-day exports. Section [[APPDATA]] of the
Internet Appendix carries the rest: every file and vendor, the nine export joins
and the audit that checks them, the four further index series behind the coupling
result and the weaker verification they carry, and the two of those four the
vendor has since withdrawn.</p>

<p>Two construction rules belong here, because the results turn on them. A
foreign close is admissible only when its timestamp falls before the target's, so
a market closing at or after the target contributes its previous session. The
audit rebuilding all 7,021 dates in UTC from each exchange's own closing time,
daylight saving included, finds no date on which the true clocks would have
reordered a block, and the narrowest margin the rule relies on anywhere is two
hours. The same rule is awkward for the VIX, and the awkwardness is not
cosmetic: its printed daily close is stamped 21:15 UTC, fifteen minutes after the
21:00 S&amp;P cash close, so that print is inadmissible. A 21:00 value of it
existed and a retail end-of-day export does not carry it. Section 7 therefore
leads with the previous day's close, which obeys our own rule and discards a full
day of information, and reports the same-day close as a sensitivity check. The
21:00 observation lies between the two in time, from which nothing follows about
where its skill would lie.</p>
"""

EST_STUB = """<p>Volatility is the Yang-Zhang estimator (2000) over rolling
five-day windows, which uses the whole daily range and so sees more of each
session than the close alone. Over the <i>n</i> sessions ending at <i>t</i>,</p>
[[EQ:1]]
<p>where the three terms are the overnight gap, the drift within the session and
the Rogers-Satchell range net of that drift (Rogers and Satchell 1991), k is the
weight for which the sum is minimum-variance and drift-independent under a
Brownian price, and n = 5 throughout. Section [[APPEST]] of the Internet Appendix
defines the three components term by term.</p>
"""

TGT_STUB = """<p>Let <i>i</i> index the eight series and <i>t</i> the target's
trading day. The decision variable each model sees is the log of that series'
variance against its own recent level, and the quantity to be forecast, with
horizon h = 5 trading days and target market denoted SPX, is the target's own
decision variable five days later:</p>
[[EQ:2]]
[[EQ:3]]
<p>Dividing by a series' own trailing median puts Tokyo, S&atilde;o Paulo and the
VIX on one scale and keeps each series' units out of the regression; the
implied-volatility series enter the same way. One definition per place, stated
once: unless a caption says otherwise, the equity-result tables in this paper
report the target-dated rate and every headline sentence reports the
implementable origin-dated one. The two differ by at most a point at every delay,
and Table S12 of the Internet Appendix gives both columns. The target is left
continuous, because splitting it at its median discards exactly the magnitude
information a cross-sectional model would use (Vellanikaran 2026). Section
[[APPTGT]] of the Internet Appendix carries the dating argument in full.</p>
"""

FEAT_STUB = """<p>Figure 1 is the design. Every model is a ridge regression
fitted by walk-forward: a burn-in for the trailing median, a training block for
the coefficients, a validation block for the penalty, and then the
&delta;&nbsp;+&nbsp;h days at the end that are withheld. Writing &delta; for the
delay imposed on the target's own block, the last date a model may use and the
window it is fitted on are</p>
[[EQ:4]]
<p>and the coefficients solve</p>
[[EQ:5]]
<p>over that window, with &lambda; chosen on the validation block. The domestic
block is the target's own three features dated t &minus; &delta;; the foreign
block adds the seven peer closes dated t, which is the whole of what delay costs
and what breadth might repair; the implied-volatility block adds the admissible
option-implied series. Section [[APPFEAT]] of the Internet Appendix gives the
feature definitions, the penalty grid, the refit cadence and the ragged-edge
handling.</p>
"""

LOSS_STUB = """<p>Losses need care here. With an imperfect volatility proxy only
a restricted family of losses preserves the ranking a researcher would obtain
against true latent volatility (Patton 2011), and that family is defined on the
variance scale, while this paper's decision variable is a normalised log
variance. We therefore report three losses, and the benchmark they divide by has
to be stated first, because it is not a single number. At each test day t it is
b<sub>t</sub>, the mean of the labels a forecaster had already seen, taken over
the 1,500 days ending at t &minus; h, where labels stop resolving for someone
whose own data is current. Two of its properties are load-bearing. It is not
delay-matched: one benchmark serves every model at every delay, because
R(&delta;) divides by S<sub>own</sub>(0) &minus; S<sub>own</sub>(&delta;), and a
yardstick that moved with &delta; would mix the damage the delay does with the
change in the yardstick. And it is a forecast rather than a fitted constant: a
mean taken over the test period would be chosen with hindsight, which is not what
out-of-sample skill means.</p>

<p>Write n for the number of test days. Realised variance is recovered from the
target by undoing the normalisation at the date the target carries it,
v<sub>t</sub> = exp(y<sub>t</sub>) M<sub>t+h</sub> =
&sigma;&sup2;<sub>YZ,t+h</sub>. A forecaster at t does not know M<sub>t+h</sub>,
so the matching forecast can only use the median they do have, w<sub>t</sub> =
&gamma;<sub>t</sub> exp(f<sub>t</sub>) M<sub>t</sub>, which carries the median's
h-day drift as forecast error rather than hiding it. The three losses are</p>
[[EQ:6]]
[[EQ:7]]
[[EQ:8]]
<p>where &gamma;<sub>t</sub> is Duan's smearing factor (1983), estimated from the
same delayed training window at each refit. Skill under QLIKE and MSE<sub>var</sub>
is defined as in (6), one minus the model's loss over the benchmark's, so all
three columns are shares of a benchmark loss removed and are read the same way.
That is only true if the benchmark is carried onto each scale the way a model
forecast is, and it is. Of the three, R&sup2; in (6) is the paper's headline and
is not proxy-robust; QLIKE in (7) is. Section [[APPLOSS]] of the Internet
Appendix carries the reconstruction in full: what the smeared benchmark is on the
natural scale and why it is not the trailing mean of variance, the two ways of
getting it wrong and which direction each one errs in, the demonstration that
QLIKE is robust where R&sup2; is not, and the alternative benchmark built
directly from resolved variances, which leaves the rate identical to the decimal
place because the benchmark cancels from a ratio of skill differences.</p>
"""

DC_STATEMENT = """<p>Every number in this paper and in its Internet Appendix is
printed by a script, and the scripts, their reference output and the source of
every document are in a public repository, named in the data availability
statement supplied to the editor. One command reproduces every lab's reference
output byte for byte; a second reads the source of both documents and fails on
any figure that disagrees with it; a third builds a machine-readable registry of
every quoted figure and runs five gates over it. The input files are not
redistributed, because every vendor behind them forbids it, so what ships instead
is a manifest recording for each file its source, its date span, the SHA-256 of
the exact bytes used and a digest of the derived series year by year: a reader
with the same exports reproduces the paper bitwise, and a reader with a fresher
export is told which file and which year differ rather than left to guess. The
publicly available inputs are cited in the reference list with their repository
and series identifiers, as this journal's data-citation policy asks, and every
number is computed with NumPy (Harris et al. 2020) and pandas (McKinney 2010)
and no other third-party library.
Section [[APPDC]] of the Internet Appendix carries the full account, including the
five gates, the three scripts that cannot run without a licensed input, and what
was and was not reproduced on a second machine.</p>
"""

RETAIN = {
    "Table 5": (
        "<p>The limits of every claim in this paper are set out in Table 5, which "
        "grades each one by the strength of the evidence behind it and names where "
        "that evidence is. The grades are used strictly: <i>established</i> means "
        "the estimate excludes zero and survives the robustness set; "
        "<i>supported</i> means it is consistent across the tests run but rests on "
        "a sample too small to resolve alternatives; <i>suggestive</i> means the "
        "sign is stable and the magnitude is not; and <i>not claimed</i> marks a "
        "question this paper raises and leaves open. Nothing in the paper should be "
        "quoted at a grade stronger than its row. The discussion behind each grade "
        "is in the Internet Appendix.</p>\n\n"),
    # The stronger-control finding is the strongest objection to the design, and
    # in the source it lives in Section 9 - which this variant relocates to the
    # appendix, so adding it there put it exactly where a referee would say it
    # had been buried.  It is repeated in the retained stub so the JFEc body
    # carries it too, alongside Figure 3, which is the float that keeps this
    # section in the manuscript at all.
    "Figure 3": (
        "<p><b>The benchmark was improvable, and improving it costs the rate "
        "three to four points.</b> Median normalisation removes the level of a "
        "series' own volatility, and a domestic block given that level back is "
        "stronger than this paper's at every delay: own-only R&sup2; rises from "
        "0.0070 to 0.0572 at eleven weeks, and the substitution rate measured "
        "inside that arm falls from 72% to 68%, a paired difference of "
        "&minus;4.1 points with interval [&minus;9.7, &minus;0.6]. The direction "
        "is the same at all eight delays and the size barely varies, so this is "
        "one result and not two. It is stated here rather than left to the "
        "appendix because it is the strongest objection to the design: part of "
        "what the cross-section appears to restore is a level the normalisation "
        "threw away, and a fair reading carries the 68% alongside the 72%. "
        "Table 1 reports it as its own row.</p>\n\n"
        "<p>The robustness set is reported in full in the Internet Appendix: eight "
        "alternative domestic controls, a multiple-testing adjustment, a temporal "
        "split, a leakage check, a second estimator for the inference, and the two "
        "scope conditions that bound where the result holds, the option chain and "
        "coupling. One of those two is worth the space here, because it is what a "
        "reader needs in order to know whether the result applies to their own "
        "asset. Coupling is the target's own correlation with the cross-section it "
        "is offered, it is computable before anything is bought, and it orders the "
        "substitution rate across eight equity markets.</p>\n\n"),
}

# The journal copies carry a LONGER disclosure than the SSRN copy.  Both say the
# same thing about who did the research; the journal one adds the declaration
# Elsevier prescribes - its own template sentence, naming the tool and the
# reason and stating that the author reviewed the content and takes full
# responsibility - which IJF and JEF require and SSRN has no use for.  Keeping
# the difference here, beside the title and abstract that already differ per
# journal, is what stops it becoming a hand-edited second copy of the section.
# One paragraph, not two.  The second restated the first: "solely responsible for
# all contents" and "retained full responsibility for the resulting work" are the
# same undertaking written twice, and a disclosure that repeats itself reads as
# boilerplate rather than as a statement someone stands behind.
JOURNAL_DISCLOSURE = """<h2>Disclosure of AI use</h2>
<p>The research questions, mathematics, empirical design, interpretation of results, and
decisions about the paper's claims are the author's own. During preparation of this
manuscript and its Internet Appendix, the author used Claude (Anthropic) as an assistive
tool for drafting and revising text and code; the author independently specified the
analyses, executed them, and reviewed, tested, and validated the code and all reported
outputs. The author is solely responsible for all contents of the paper, including every
claim, design decision, and reported number. No AI system is an author of this work or
held authorial responsibility for any part of it.</p>

"""


# JFEc states a 100-word limit on the abstract.  The base abstract is 119 and
# is already posted on SSRN, so it is not touched: the generator exists exactly
# so that one journal's constraint does not reach the other copies.  The cut is
# not only arithmetic.  The base opens with a scene-setting sentence that earns
# its place in a working paper and not in a hundred words, and it says
# "Handicapping", which implies the delay is imposed without saying so.  This
# version spends the recovered words on making that explicit, since a referee's
# first objection to the design is that the S&P 500 does not in fact report
# late - and the answer is that nobody said it did.
JFEC_ABSTRACT = """We define the substitution rate R(&delta;), the share of a delay's forecast damage that a
current substitute repairs: a ratio of dependent loss differentials whose denominator vanishes
where delay costs little, so inference inverts a HAC t-test after Fieller. In S&amp;P 500
variance forecasts handicapped by &delta; days, a current foreign cross-section returns
{STRESSED} of eleven weeks' staleness in stressed markets, {HEADLINE} unconditionally. In both
own-index option cases examined, the compressed cross-section adds no detectable information
once own-index implied volatility is included. Twenty house price indices carry the measurement
to an appraisal-like market."""


# The companion note's reference entry, and what it becomes in an anonymous
# copy.  Kept here as a pair of constants rather than inline so the blinding
# and the refusal below cannot disagree about what they are removing, and so a
# change to the entry has one place to be made.
SSRN_NOTE = ("SSRN working paper 7504558. "
             "<code>ssrn.com/abstract=7504558</code>")
# The same entry as a pattern, because the JFEc conversion rewrites the words
# around the identifier ("SSRN Working Paper 7504558") and a literal
# replacement then removes nothing.  Blinding has to survive a reference style
# change, so it matches the identifier rather than the sentence carrying it.
SSRN_NOTE_RE = re.compile(r"SSRN [Ww]orking [Pp]aper 7504558\.\s*"
                          r"<code>ssrn\.com/abstract=7504558</code>")
ANON_NOTE = ("Working paper. [Identifier withheld for anonymous review; "
             "supplied to the editor.]")


VARIANTS = {
    "ijf": {
        "name": "International Journal of Forecasting",
        "review": "double-anonymised",   # stated on the journal's own author guidelines
        "title": IJF_TITLE,
        "abstract": _fill(IJF_ABSTRACT),
        "keywords": ("information delay; ragged edge; forecast evaluation; "
                     "substitution rate; weak identification; overlapping horizons"),
        # No word limit is stated in the instructions we hold for this journal,
        # so none is enforced.  A guessed limit is worse than no limit: it would
        # fail a build for a rule nobody set.
        "abstract_max": None,
        "insert_after_intro": IJF_GENERAL,
        "move": [],
    },
    "jef": {
        "name": "Journal of Empirical Finance",
        "review": "double-anonymised",   # stated on the journal's own author guidelines
        "title": JEF_TITLE,
        "abstract": _fill(JEF_ABSTRACT),
        "keywords": ("stale prices; information delay; volatility forecasting; "
                     "cross-sectional information; weak identification"),
        "abstract_max": None,
        "replace_opening": JEF_OPENING,
        "move": [],
    },
    "jfec": {
        "name": "Journal of Financial Econometrics",
        # "the Price of Being Late" reads as a monetary valuation, which is the
        # one thing Table 5 records the paper as NOT claiming.  The title names
        # the two substitutes and the condition instead.
        "title": "Breadth, Implied Volatility, and Forecasting with Stale Information",
        # "The first numbered section should follow the introduction of the
        # article" - the journal's own sentence, and the instruction this
        # package missed until a referee read it.
        "unnumbered_intro": True,
        "abstract": _fill(JFEC_ABSTRACT),          # 100 words, the journal's limit
        "keywords": None,
        "abstract_max": 100,
        "insert_after_intro": JFEC_PRIOR + JFEC_FRAMEWORK,
        # Relocated to the Internet Appendix to meet the 40-page limit.  Nothing
        # is deleted, so every check pinned to this text still finds it, and the
        # choice of what goes is editorial rather than mechanical: the estimand
        # (4.5) and its weak-identification inference (4.6) are what a financial
        # econometrics referee is being asked to judge, so they stay in the body
        # whatever the length costs.  What leaves is the literature survey, the
        # proxy-robustness discussion, the robustness summary that already
        # points at the appendix, and the limitations prose whose full version
        # is Section S33 - but Table 5, the claim-by-claim grade table, stays,
        # because it is the paper's scope discipline and a referee wants it in
        # front of them.
        # Base numbering.  2 is Related work, 3 is Data, 9 is Robustness and
        # scope, 10 is Limitations.  Data moves with a stub rather than
        # wholesale: a manuscript with no data section is a desk return.
        "move": [2, 3, 9, 10],
        "move_stub": {3: (DATA_STUB, "APPDATA")},
        # 4.4 used to be the subsection that left.  It holds the three loss
        # functions AND equations (6), (7) and (8), and relocating it left the
        # body saying "skill under any one of (6), (7) or (8)" about equations
        # that were no longer in the document.  For a financial econometrics
        # journal the loss functions are not background a referee can be sent
        # to an appendix for: they define what every number in the paper is
        # measured in.  6.1 goes instead - an implementation check on whether
        # fitting the compressed representation directly changes anything,
        # almost the same length, and carrying no equations to strand.
        # 4.1 to 4.3 are the construction: the estimator, the target and the
        # feature blocks.  Each leaves the equations a later section cites and
        # sends its own derivation and defence to the appendix.  Figure 1 is
        # retained from 4.3, because one page showing the walk-forward design
        # replaces eight hundred words describing it - which is the whole trade
        # this length cut is making.
        "move_sub": ["4.1", "4.2", "4.3", "4.4", "6.1"],
        "sub_stub": {"4.1": (EST_STUB, "APPEST"),
                     "4.2": (TGT_STUB, "APPTGT"),
                     "4.3": (FEAT_STUB, "APPFEAT"),
                     "4.4": (LOSS_STUB, "APPLOSS")},
        "keep_float": {10: "Table 5", 9: "Figure 3", "6.1": "Table 4",
                       "4.3": "Figure 1"},
        # The journal asks for a data availability STATEMENT; the SSRN copy
        # carries two pages of account, which is the right thing in a preprint
        # and the wrong thing here.
        "statement": ("Data and code availability", DC_STATEMENT, "APPDC"),
        # Paragraphs, not sections.  The introduction's own statement of the
        # contribution duplicates the two blocks this variant inserts after it
        # for a financial-econometrics reader, and the further-work paragraphs
        # are proposals rather than findings: of everything in the manuscript
        # they are what a referee can read in the appendix with the least loss.
        # They are relocated there, not cut.
        # Related work leaves no stub, so the roadmap needs a name for where it
        # went.
        "token_for": {2: "APPPRIOR"},
        # The roadmap, rewritten because this variant moves three of the things
        # it points at.  Written in base numbering: the renumbering treats it
        # like any other paragraph.
        "replace_text": [
            # 2.6, the bandwidth argument: the current position, stated as the
            # current position
            ("An earlier version of this paper tried to settle it: the bandwidth "
             "grew with the delay, the growth was justified by how much long-run "
             "variance a shorter kernel retained, and the paper then had to defend "
             "both the rule and the retained-variance measurement behind it. "
             "Neither defence is needed, because neither fixes the problem.",
             "Growing the bandwidth with the delay, and justifying the growth by "
             "how much long-run variance a shorter kernel retains, settles "
             "nothing: neither the rule nor the measurement behind it fixes the "
             "problem."),
            ("which is why the earlier version of this paper found a shortfall it "
             "could not close by lengthening: it was reading a property of the "
             "kernel as a property of the data",
             "so a shortfall measured against it cannot be closed by lengthening: "
             "that reads a property of the kernel as a property of the data"),
            # Figure 2's discussion: name the denominator, drop the confession
            (", and this sentence used to quote the second number while describing "
             "the first", ""),
            # the variance-scale losses: why no interval is quoted
            ("That is why QLIKE, not MSE, is the robust loss we lean on, and it is "
             "a stronger reason than the one this paper used to give. A previous "
             "version quoted a resampled interval of [11, 114] at five days here; "
             "the bootstrap returns finite endpoints at every one of these cells "
             "and so concealed the identification problem, and those endpoints "
             "were in any case not the ones the script produced.",
             "That is why QLIKE, not MSE, is the robust loss we lean on. A "
             "percentile bootstrap returns finite endpoints at every one of these "
             "cells and so conceals the identification problem rather than "
             "reporting it, which is the strongest reason not to quote one here."),
            # the spanning bound: state what the section is entitled to
            ("A sentence stood here comparing a five-day-ahead foreign-only "
             "R&sup2; of 0.399 against 0.498 from domestic history and read the "
             "ratio as bracketing the fitted 72%. Neither figure is produced by "
             "any script in the replication, a single ratio cannot bracket "
             "anything, and the comparison is withdrawn rather than reconstructed: "
             "the contemporaneous spanning figure above is the bound this section "
             "is entitled to",
             "The contemporaneous spanning figure above is the bound this section "
             "is entitled to"),

            # The robustness set, enumerated in the manuscript and indexed in
            # the appendix's own contents.  A list of what is elsewhere is the
            # "secondary robustness detail" a length cut is for.
            ("The robustness set is reported in full in the Internet Appendix: "
             "eight alternative domestic controls, a multiple-testing "
             "adjustment, a temporal split, a leakage check, a second estimator "
             "for the inference, and the two scope conditions that bound where "
             "the result holds, the option chain and coupling. One of those two "
             "is worth the space here, because it is what a reader needs in "
             "order to know whether the result applies to their own asset. "
             "Coupling is",
             "The robustness set is reported in full in the Internet Appendix, "
             "including the two scope conditions that bound where the result "
             "holds: the option chain and coupling. Coupling is worth the space "
             "here, because it is what tells a reader whether the result "
             "applies to their own asset. It is"),
            # The timing audit, whose full account is the appendix section the
            # sentence already names.
            ("The audit rebuilding all 7,021 dates in UTC from each exchange's "
             "own closing time, daylight saving included, finds no date on which "
             "the true clocks would have reordered a block, and the narrowest "
             "margin the rule relies on anywhere is two hours.",
             "An audit rebuilding every date in UTC from each exchange's own "
             "closing time, daylight saving included, finds no date on which the "
             "true clocks would have reordered a block."),
        ],
        "relocate_text": [
            # The coverage study's implementation.  The manuscript keeps the
            # finding and the consequence; the five coverage figures and the
            # taper argument move to the appendix, which did not carry them
            # before - so this is a relocation and not a deletion.
            ("A coverage study on a generator matched to the dependence these "
             "differentials carry finds that no bandwidth reaches the nominal "
             "level while the inversion uses normal critical values: coverage "
             "runs from 89.4% to 93.7% for a nominal 95% across the lengths "
             "tried. What the remaining gap does with length depends on how "
             "strongly the ratio is identified. Where the denominator sits ten "
             "standard errors from zero the gap stops closing once the kernel is "
             "longer than the dependence, 92.6% at both eighty and a hundred and "
             "sixty. At the two weaker strengths it is still closing at a "
             "hundred and sixty, 93.4% and 93.7%, and has not arrived. Neither "
             "reading rescues the length, and the reason is visible in what the "
             "kernel does. The generator's dependence stops at lag forty-five, "
             "so a kernel of eighty truncates nothing; what it does is taper, "
             "weighting the covariance at lag h by one minus h over the "
             "bandwidth, and that weighting discards a share of the long-run "
             "variance at every finite bandwidth. A fraction computed against "
             "the true long-run variance therefore cannot reach one at any "
             "length, so a shortfall measured against it cannot be closed by "
             "lengthening: that reads a property of the kernel as a property of "
             "the data. What is left once the taper is priced is not a length.",
             "A coverage study on a generator matched to the dependence these "
             "differentials carry finds that no bandwidth reaches the nominal "
             "level while the inversion uses normal critical values, and that "
             "the shortfall does not close as the kernel lengthens: the "
             "Bartlett taper discards a share of the long-run variance at every "
             "finite bandwidth, so a fraction measured against the true "
             "long-run variance cannot reach one at any length. The Internet "
             "Appendix reports the coverage at each length and the arithmetic "
             "behind that statement. What is left once the taper is priced is "
             "not a length.",
             "Coverage of the normal-critical-value inversion, measured"),
        ],
        "swap_paras": [
            # The contribution, stated where a referee looks for it rather than
            # assembled from four findings and two later blocks.
            ("We cannot observe the truth for a private portfolio",
             "<p>We cannot observe the truth for a private portfolio, so we run "
             "the experiment where the truth is observable. We take a forecaster "
             "of S&amp;P 500 realised variance, withhold their own data for "
             "&delta; days, leave the other sources untouched, and measure what "
             "each gives back. For foreign equities the asymmetry costs nothing "
             "to construct: Tokyo closes at 06:00 UTC, Hong Kong at 08:00, "
             "Mumbai at 10:00, London and Frankfurt at 16:30, New York at 21:00. "
             "On any given day the rest of the world has already finished trading "
             "when New York does.</p>\n\n"
             "<p>Three things are separable from that application and are what "
             "this paper contributes: an estimand, a design that imposes the "
             "staleness rather than observing it, and inference for a ratio of "
             "dependent loss differentials whose denominator vanishes exactly "
             "where the question stops being interesting. Each is set out "
             "below.</p>"),
            # The findings, at headline level.  The state-by-state overlap, the
            # leave-one-out correlations, the post-selection counts and the
            # effective-age clock are results and are reported in the sections
            # that establish them; an introduction that carries them too makes
            # the contribution visible and hard to absorb.
            ("Four findings carry it. First, breadth substitutes for timeliness",
             "<p>Four findings carry it. First, breadth substitutes for "
             "timeliness and the amount can be priced, and the amount is worth "
             "stating where it is determined rather than where it is largest. A "
             "current foreign cross-section returns 81% [67, 90] in this "
             "sample's stressed markets, cut in real time on a rolling VIX "
             "tercile, and the estimate is stable across both halves of the "
             "sample taken separately. In calm markets the same quantity spans "
             "more than a hundred points and cannot be pinned down at all, for "
             "the mechanical reason that where delay costs almost nothing the "
             "share of that cost repaired has no stable value. Between the two "
             "sits a middle tercile at 53% [28, 76], lower at the point estimate "
             "but not separated from the stressed one: under the construction of "
             "Section 4.6 the three states&rsquo; confidence sets overlap at "
             "every delay. Pooling the three, a current foreign cross-section "
             "returns 71% [54, 83] of what the delay took on the implementable "
             "target, with the original model, the stronger control, the "
             "compressed specifications and the state-space alternative all "
             "landing between 68% and 75%. Table 1 collects them beside the "
             "question each one answers and says which to carry. Read as a clock "
             "rather than a share, a fifty-five-day-old mark plus today's "
             "foreign close forecasts as well as a mark 4.6 days old [2.5, 8.2]. "
             "Second, an options market on the asset itself makes the "
             "cross-section redundant, but only on the asset itself: where the "
             "only chain available is written on another index, a compressed "
             "cross-section still adds in thirteen of twenty-four cells, of "
             "which few survive a post-selection correction, so what the "
             "evidence supports is the pattern and not any single cell. Third, "
             "which of those worlds a holder is in can be measured before "
             "anything is bought, because the target's own correlation with the "
             "cross-section orders the substitution rate across the eight "
             "markets examined, at a rank correlation of +0.976; what eight "
             "points do not support is the fitted line itself, which Table 5 "
             "grades accordingly. Where that correlation is small the "
             "cross-section repairs nothing and charges for the attempt. Fourth, "
             "on the house price panel one quarterly appraisal cycle of extra "
             "staleness gives 19.0% [6, 31] on the variance coordinate, positive "
             "on thirteen metros of fourteen. That is the coordinate this "
             "paper's target is defined on, and the harder of the two: a "
             "trailing twelve-month variance overlaps its own current window, so "
             "the denominator it divides by is small. On next month's return, "
             "which is the quantity an owner cares about but is not this paper's "
             "target, the same design gives 67.8% [45, 92], positive on fourteen "
             "of fourteen.</p>"),
            # Two future studies described at length become one paragraph that
            # names them.
            ("Two studies would matter more than any of those",
             "<p>Two studies would matter more than any of those, and both are "
             "named rather than attempted because each is a paper. The first "
             "closes the gap Section 10 leaves open, by measuring a delay that "
             "was imposed here on an asset where it is real. The second turns "
             "the rate into money: R(&delta;) is a share of forecast loss, and "
             "what a holder wants to know is what that share is worth in a "
             "decision, which needs a loss function this paper does not "
             "have.</p>"),
            ("The second turns the rate into money", ""),
            (
            "The paper proceeds as follows",
            "<p>The paper proceeds as follows. Sections 3 and 4 give the data "
            "and the design, including the estimator, the loss functions and the "
            "definition of R(&delta;); the construction detail behind them is in "
            "the Internet Appendix, each section of it named where the text "
            "hands over. Section S26 states the specification the measurements "
            "support. Sections 5 to 7 measure what breadth recovers, where it "
            "stops, and how it fares against implied volatility. Section 8 "
            "carries the measurement to house prices. Section 9 tests the result "
            "and bounds where it holds, with the working in the Internet "
            "Appendix. Section S30 says what a practitioner should do with it, "
            "Section 10 what the paper cannot say, and Section 11 what would "
            "test it next. The literature this paper sits in is reviewed in "
            "Section [[APPPRIOR]] of the Internet Appendix rather than here, "
            "which is where this version's length limit put it.</p>")],
        "trim_paras": ("Positioning, contribution and further work", [
            "Nothing turns on which row is called the headline",
            "The contribution is four-fold",
            "The other three follow from applying it",
            "The universe here is eight equity indices",
            "asset classes is, and it is the sharper test",
            "Two smaller threads are open and named where they arise",
        ]),
        # Author-date in the journal's own shape: year out of its parentheses,
        # co-authors initials-first with a serial comma, headline capitals,
        # "volume: pages" with the issue dropped.  See _jfec_reference.
        "refs": "jfec",
    },
}


# ----------------------------------------------------------------------
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


# ---------------------------------------------------------------------------
# JFEc's reference style, applied to the base list rather than kept as a second
# copy of it.
#
# The base documents carry one house style - "Brown, G. W., Ghysels, E. and
# Gredil, O. (2023). Title in sentence case. Journal 36(3), 945-986." - and the
# journal asks for another: "Adams, G., R. Jones, and J. Doe. 1999. Article
# Title. Journal Title 9: 226-232."  Four things differ: the year leaves its
# parentheses and follows the names, co-authors put initials before the
# surname, the serial comma arrives, the title takes headline capitals, and the
# issue number goes with the volume replaced by "volume: pages".
#
# Maintaining a second reference list by hand is the copied-figure failure in
# another costume, so the conversion is computed.  Capitalisation is the one
# risky step: a word is capitalised only when it is ENTIRELY lower case, so
# MIDAS, GDP, S&P, Kalman, German and Diebold-Mariano survive untouched, and the
# Chicago stop words stay down unless they open the title or follow a colon.
_STOP = {"a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet", "of",
         "in", "on", "at", "to", "from", "by", "with", "as", "into", "onto",
         "over", "under", "than", "that", "upon", "across", "about", "between",
         "through", "during", "without", "within", "after", "before", "per",
         "via", "vs"}
# An initials group: "G. W.", "E.", "K. I. M.", "J.-M.".  A surname never
# matches it, which is what lets the author list be paired without a parser.
# The final period is optional because the author block is stripped of its
# trailing full stop before splitting, which leaves the last author's initials
# as "O" rather than "O." - and an initials group that fails to match is read as
# a surname, which is how "Gredil, O." became "Gredil, and O."
_INITIALS = re.compile(r"^(?:[A-Z]\.?(?:-[A-Z]\.?)?\s*)+$")


def _dots(ini):
    """'O' -> 'O.', 'H. R' -> 'H. R.', leaving 'J.-M.' alone."""
    return re.sub(r"([A-Z])(?![.\w])", r"\1.", ini).strip()


def _head_case(title):
    """Headline capitals, changing only words that are entirely lower case.

    MIDAS, GDP, S&P, Kalman, German and Diebold-Mariano are already capitalised
    in the base list and must survive, so a word is touched only when its
    alphabetic core is all lower case.  The core is taken WITHOUT surrounding
    punctuation: an earlier version tested the raw token, so every word ending
    in a colon or a comma was left alone and the output read "Net Asset values:
    The Case of Private Equity".
    """
    out, after_colon = [], True       # True while the next word opens the title
    toks = title.split(" ")
    words = [k for k, t in enumerate(toks) if t.strip()]
    last = words[-1] if words else -1
    for k, tok in enumerate(toks):
        if not tok.strip():
            out.append(tok)
            continue
        lead = re.match(r"^[\"\u2018\u2019\'(\[]*", tok).group(0)
        trail = re.search(r"[\"\u2018\u2019\'),.;:?!\]]*$", tok).group(0)
        core = tok[len(lead):len(tok) - len(trail)] if trail else tok[len(lead):]
        down = core.lower() in _STOP and not after_colon and k != last
        parts = []
        for n, piece in enumerate(core.split("-")):
            if piece.islower() and piece.isalpha() and not (down and n == 0):
                piece = piece[:1].upper() + piece[1:]
            parts.append(piece)
        out.append(lead + "-".join(parts) + trail)
        after_colon = tok.rstrip().endswith(":")
    return " ".join(out)


def _jfec_authors(block_):
    """'Brown, G. W., Ghysels, E. and Gredil, O.' -> 'Brown, G. W., E. Ghysels, and O. Gredil'.

    The base list writes every author surname-first; the journal writes the
    first that way and the rest initials-first, with a serial comma.  Splitting
    on commas and "and" gives an alternating sequence of surnames and initials
    groups, so the pairing needs no grammar: anything matching _INITIALS belongs
    to the surname before it.
    """
    # "et al." is not a name and must not be paired with one: left to the
    # loop below it becomes a surname and the list reads "and et al".
    _etal = ""
    block_ = block_.strip()
    _m_etal = re.search(r"\s+et al\.?$", block_)
    if _m_etal:
        _etal, block_ = " et al.", block_[:_m_etal.start()]
    toks = [t.strip() for t in re.split(r",|\s+and\s+", block_.strip().rstrip("."))
            if t.strip()]
    pairs = []
    for tok in toks:
        if _INITIALS.match(tok) and pairs:
            pairs[-1][1] = _dots((pairs[-1][1] + " " + tok).strip())
        else:
            pairs.append([tok, ""])
    if not pairs:
        return block_.strip().rstrip(".")
    first = (f"{pairs[0][0]}, {pairs[0][1]}" if pairs[0][1] else pairs[0][0])
    rest = [f"{ini} {sur}".strip() for sur, ini in pairs[1:]]
    if _etal:
        # With "et al." there is no last author to put "and" in front of.
        return ", ".join([first] + rest) + "," + _etal
    if not rest:
        return first
    if len(rest) == 1:
        return f"{first}, and {rest[0]}"
    return first + ", " + ", ".join(rest[:-1]) + ", and " + rest[-1]


def _jfec_reference(item):
    """One <li> of the base reference list, in the journal's own style."""
    # A dataset citation is already in the shape the journal's data-citation
    # policy prescribes - "[dataset] Authors, Year, Title, Publisher
    # (repository), Identifier" - which is not author-date and must not be
    # pushed into it.
    if item.lstrip().startswith("[dataset]"):
        return item
    m = re.match(r"\s*(.*?)\s*\((\d{4}[a-z]?)\)\.\s*(.*)$", item, re.S)
    if not m:
        return item
    who, year, tail = _jfec_authors(m.group(1)), m.group(2), m.group(3).strip()
    # the title runs to the first full stop that is followed by a space and
    # either an italic journal name or a capital letter - not to a full stop
    # inside "S&P 500." or an initial
    # Where the title ends and the source begins.  This used to be found by
    # looking for a full stop before the italic journal name, which failed on
    # every title ending in a question mark - "What Is the Role of the United
    # States?" kept its volume and issue in the base style while the rest of the
    # list had been converted, and a half-converted reference list is worse than
    # an unconverted one because it reads as deliberate.
    cuts = []
    _i = tail.find("<i>")
    if _i >= 0:
        # A book chapter's source begins at "In", which may be followed by the
        # editors before the italic book title: "In Elliott, G., Granger,
        # C. W. J. and Timmermann, A. (eds.), <i>Handbook ...</i>".  Cutting at
        # the italic tag put the editors in the TITLE, where headline casing
        # turned "(eds.)" into "(Eds.)" and a full stop was appended after the
        # comma.  The "In" is required to open a sentence, so a lower-case "in"
        # inside a title cannot be mistaken for it.
        _in = re.search(r"(?:^|\.\s)(In\s)", tail[:_i])
        cuts.append(_in.start(1) if _in else _i)
    for mark in ("SSRN", "arXiv"):
        _j = tail.find(mark)
        if _j >= 0:
            cuts.append(_j)
    if cuts:
        at = min(c for c in cuts if c >= 0)
        title = _head_case(tail[:at].strip().rstrip(" ."))
        src = tail[at:].strip()
    else:
        title, src = _head_case(tail.rstrip(".")), ""
    # volume(issue), pages -> volume: pages
    src = re.sub(r"(</i>)\s*(\d+)\(([\d\u2013&;a-z-]+)\),\s*", r"\1 \2: ", src)
    src = re.sub(r"(</i>)\s*(\d+),\s*", r"\1 \2: ", src)
    # an unpublished title is quoted rather than plain, as the journal's own
    # working-paper example has it
    if src.startswith("In "):
        # "Shaw, J. 1986a. 'Chapter Title.' In R. Jones and M. Brown (eds.),
        # Book Title, vol. 3." - the journal quotes a chapter title where it
        # leaves a journal article's plain.
        out = f'{who}. {year}. &ldquo;{title}.&rdquo; {src}'
        out = re.sub(r"\.\.(\s|$)", r".\1", out)
        return re.sub(r"\s+", " ", out).strip()
    if src.startswith(("SSRN", "arXiv")):
        title = '"' + title + '."'
        out = f"{who}. {year}. {title} {src}"
    elif src:
        # A title that ends in its own punctuation does not take another stop:
        # "What Is the Role of the United States?." is not a reference style.
        stop = "" if title.endswith(("?", "!")) else "."
        out = f"{who}. {year}. {title}{stop} {src}"
    else:
        out = f"{who}. {year}. {title}."
    # "Corsi, F." + ". 2009." gives "Corsi, F.. 2009."; the journal's own
    # example writes one stop, not two.
    out = re.sub(r"\.\.(\s|$)", r".\1", out)
    if "SSRN" in out:
        out = out.replace("SSRN working paper", "SSRN Working Paper")
    return re.sub(r"\s+", " ", out).replace(" .", ".").strip()


def _jfec_refs(html):
    """Rewrite every reference list in a document into the journal's style."""
    def one(mo):
        body = re.sub(r"<li>(.*?)</li>",
                      lambda li: "<li>" + _jfec_reference(li.group(1)) + "</li>",
                      mo.group(2), flags=re.S)
        return mo.group(1) + body + mo.group(3)
    return re.sub(r'(<ul class="refs">)(.*?)(</ul>)', one, html, flags=re.S)


def _trim_paras(raw, wants):
    """Cut the paragraphs containing each substring, and return them.

    A length cut that DELETES prose is a length cut that loses an argument, so
    what comes out here goes into the appendix with everything else that moved.
    Matching is on the tag-stripped text, because the distinctive phrase in a
    paragraph is usually wrapped in markup somewhere.
    """
    out = []
    for want in wants:
        hit = None
        for m in re.finditer(r"<p[^>]*>.*?</p>", raw, re.S):
            if want in " ".join(re.sub(r"<[^>]+>", " ", m.group(0)).split()):
                hit = m
                break
        if hit is None:
            raise SystemExit(f"build_variants: no paragraph contains "
                             f"{want!r}, so the trim list is out of date with "
                             f"the manuscript")
        out.append(hit.group(0))
        raw = raw[:hit.start()] + raw[hit.end():]
    return raw, out


_SECREF = (r"(Sections?\s+)((?:S?\d+(?:\.\d+)?)"
           r"(?:\s*(?:,|and|&amp;|to|through|&ndash;|-)\s*"
           r"S?\d+(?:\.\d+)?)*)")


def _unnumber_intro(docs):
    """Make the introduction unnumbered and shift every other number down one.

    "The first numbered section should follow the introduction of the article"
    is the journal's own sentence, and it is the one instruction on that page a
    reader cannot see has been missed: the manuscript looked conventional with a
    numbered Section 1.  The shift is applied AFTER the relocation renumbering,
    on the finished documents, because by then every heading and every reference
    is already consistent and all that is left is to subtract one from each and
    give the introduction a name instead of a number.

    References to the introduction become "the introduction", including inside a
    list - "Sections 1, 7" has to become "the introduction and Section 6" rather
    than "Sections the introduction, 6", which is what a naive substitution
    produces and what a referee would read.
    """
    out = []
    for doc in docs:
        doc = re.sub(r'(<h2[^>]*>)1\.\s*(Introduction)</h2>', r"\1\2</h2>", doc)
        # headings, deepest first so a parent renumber cannot eat a child
        doc = re.sub(r'(<h3>)(\d+)(\.\d+)',
                     lambda m: f"{m.group(1)}{int(m.group(2)) - 1}{m.group(3)}"
                     if int(m.group(2)) > 1 else m.group(0), doc)
        doc = re.sub(r'(<h2(?: class="pb")?>)(\d+)\.',
                     lambda m: f"{m.group(1)}{int(m.group(2)) - 1}."
                     if int(m.group(2)) > 1 else m.group(0), doc)

        def _one(lead, body):
            parts = re.split(r"(\s*(?:,|and|&amp;|to|through|&ndash;|-)\s*)", body)
            intro, nums, seps = False, [], []
            for i, tok in enumerate(parts):
                if i % 2:
                    seps.append(tok)
                    continue
                if tok.startswith("S"):
                    nums.append(tok)
                    continue
                n, _, sub = tok.partition(".")
                if int(n) == 1:
                    intro = True          # a subsection of it would be odd,
                    nums.append(None)         # and there are none to lose
                else:
                    nums.append(f"{int(n) - 1}" + (f".{sub}" if sub else ""))
            kept = [x for x in nums if x is not None]
            if not intro:
                return lead + body
            if not kept:
                return "the introduction"
            word = "Section " if len(kept) == 1 else "Sections "
            joined = kept[0] if len(kept) == 1 else \
                ", ".join(kept[:-1]) + " and " + kept[-1]
            return "the introduction and " + word + joined

        def _sub(m):
            lead, body = m.group(1), m.group(2)
            if "1" not in re.findall(r"\d+", body) and "S" not in body:
                # plain shift
                return lead + re.sub(r"(?<!S)(\d+)(\.\d+)?",
                                     lambda g: f"{int(g.group(1)) - 1}"
                                     + (g.group(2) or ""), body)
            return _one(lead, body)

        doc = re.sub(_SECREF, _sub, doc)
        out.append(doc)
    return out


def _put_eqs(stub, eqs):
    """Replace [[EQ:n]] with the source's own paragraph for equation n."""
    def one(m):
        n = int(m.group(1))
        if n not in eqs:
            raise SystemExit(f"build_variants: a stub asks for equation ({n}), "
                             f"which this source does not define")
        return eqs[n]
    return re.sub(r"\[\[EQ:(\d+)\]\]", one, stub)


def _spell(n):
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + ("-" + _ONES[o] if o else "")


# Every reference in a manuscript section is written from the manuscript's own
# standpoint.  "Section S33 of the Internet Appendix", "a caveat developed in
# the Internet Appendix", "argued at length there rather than here" - each is
# correct where it was written and wrong the instant the paragraph containing it
# is relocated INTO the Internet Appendix, where it becomes a document naming
# itself as somewhere else and a "here" pointing at the wrong document.
#
# This was shipping.  The JFEc appendix carried fifteen such sentences, one of
# them opening a relocated section with "The Internet Appendix examines the
# principal identified threats" - inside the Internet Appendix - and another
# with "argued at length in Section S33 of the Internet Appendix rather than
# here", where S33 and "here" are the same document.  Nothing caught it,
# because every figure in the moved text was still correct: relocation breaks
# deixis, not arithmetic, and the checks only read arithmetic.
#
# The rewrite is mechanical and deliberately narrow.  Whatever it misses is
# caught rather than shipped: verify_paper.py now fails on any surviving
# self-reference in an appendix, so a form not handled here becomes a build
# failure and not a sentence in front of a referee.
_OF_IA = re.compile(
    r"((?:Sections?|Tables?|Figures?)\s+S\d+(?:\.\d+)?"
    r"(?:\s*(?:,|and|to|through|&ndash;|-)\s*S?\d+(?:\.\d+)?)*"
    r"(?:&rsquo;s|'s)?)\s+of\s+the\s+Internet\s+Appendix")


def _repoint(chunk):
    """Rewrite a relocated manuscript section to speak from the appendix."""
    # 1. a cross-reference that names the document it is now inside
    chunk = _OF_IA.sub(lambda m: m.group(1), chunk)
    # 2. "there rather than here" inverts on relocation: "here" meant the
    #    manuscript, and the section is no longer in it
    chunk = chunk.replace("rather than here", "rather than in the manuscript")
    # 3. any remaining mention becomes an internal one.  Ordered longest-first
    #    so "The Internet Appendix to this paper" does not leave "to this
    #    paper" dangling off "This appendix".
    for a, b in (("The Internet Appendix to this paper", "This appendix"),
                 ("the Internet Appendix to this paper", "this appendix"),
                 ("The Internet Appendix", "This appendix"),
                 ("the Internet Appendix", "this appendix")):
        chunk = chunk.replace(a, b)
    return chunk


def section_span(raw, n):
    """(start, end) of numbered top-level section n in the manuscript."""
    m = re.search(r'<h2(?: class="pb")?>' + str(n) + r'\.', raw)
    if not m:
        return None
    nxt = re.search(r'<h2(?: class="pb")?>(?:\d+\.|Data and code|Disclosure)',
                    raw[m.end():])
    return (m.start(), m.end() + nxt.start() if nxt else len(raw))


def _contact(path):
    """The unpublished contact details, as key: value lines.

    It held a bare telephone number until the address had to join it.  Keeping
    both here rather than in the manuscript is the same rule as before: the
    source is published, a journal's instructions are not the public's business,
    and a detail written into a public file is one a scraper finds without
    reading the paper.
    """
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and ":" in line:
            k, v = line.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out


def _date_of(raw):
    """The manuscript's own date, read out of it.

    This was a literal here, which made the title page a FOURTH copy of a date
    the three documents already state.  A title page that disagrees with the
    manuscript it introduces is the kind of thing an editor notices and an
    author never does, and the copy would have gone stale the first time the
    date moved - which it did, the day after it was written.
    """
    m = re.search(r'<p class="date">([^<]+)</p>', raw)
    if not m:
        raise SystemExit("build_variants: the manuscript states no date, so the "
                         "title page cannot take one from it")
    return m.group(1).strip()


def _title_page(spec, title_html, raw):
    """The separate title page all three journals ask for.

    JFEc names what it must carry - "the title, name(s) of the author(s), and
    institutional affiliation for each author", plus the corresponding author's
    address, telephone and email - so the fields are not a guess.  It is built
    from the manuscript rather than typed, so a change of title or contact
    detail reaches it.  It also states what was removed from the anonymous
    manuscript, because an editor should not have to diff two files to find out.
    """
    front = re.search(r'<p class="front">(.*?)</p>', raw, re.S)
    front = re.sub(r"<[^>]+>", "", front.group(1)) if front else ""
    # The manuscript's paragraph already begins "Corresponding author." inside a
    # <b>, and stripping the tags keeps the words.  Wrapping that in a fresh
    # label below printed "Corresponding author. Corresponding author." on the
    # title page - the first line of the first file an editor opens.  The label
    # is removed here and added once, by the line that formats it.
    front = re.sub(r"^\s*Corresponding author\.\s*", "", front)
    # The postal address.  JFEc asks the title page for the "name, address,
    # telephone number, and e-mail address of the author responsible for
    # correspondence", and address is the one of the four the manuscript has
    # never carried - rightly, since the source is public and a home address in
    # a public file is worse than a phone number in one.  So it lives in
    # contact.txt like the telephone, is inserted here into the journal copies
    # only, and if it is missing the title page SAYS it is missing rather than
    # quietly omitting a required field.  A silent omission is how a submission
    # gets desk-returned for an administrative reason.
    _cf = os.path.join(HERE, "contact.txt")
    _addr = _contact(_cf).get("address", "") if os.path.isfile(_cf) else ""
    if _addr and not _addr.startswith("<"):
        _addr_line = f'<span class="nb">{_addr}</span> '
    else:
        _addr_line = ('<b style="background:#ffe08a">[POSTAL ADDRESS REQUIRED '
                      '&mdash; set "address:" in contact.txt and rebuild]</b> ')
    # Placed AFTER the name and affiliation and BEFORE the telephone, which is
    # the order the journal lists the four fields in and the order a reader
    # expects: who, where, how to telephone, how to write.  Prepending it put
    # a street address ahead of the author's own name on the first line of the
    # first file an editor opens.
    _anchor2 = next((a for a in ("Telephone ", "Email ") if a in front), None)
    if _anchor2:
        front = front.replace(_anchor2, _addr_line + _anchor2, 1)
        _addr_line = ""  # consumed
    kw = re.search(r'<p class="kw">(.*?)</p>', raw, re.S)
    css = re.search(r"<style>.*?</style>", raw, re.S)
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>Title page</title>\n{css.group(0) if css else ''}\n</head>\n"
        "<body>\n"
        f"<h1>{title_html}</h1>\n"
        '<p class="byline">Sunny Alex Vellanikaran</p>\n'
        '<p class="aff">Independent</p>\n'
        f'<p class="date">{_date_of(raw)}</p>\n'
        '<div class="frontrule"></div>\n'
        f'<p class="front"><b>Corresponding author.</b> {_addr_line}{front}</p>\n'
        + (f'<p class="kw">{kw.group(1)}</p>\n' if kw else "")
        + '<p class="front"><b>Submitted to</b> ' + spec["name"] + ".</p>\n"
        '<p class="front"><b>Anonymity.</b> The accompanying manuscript carries '
        "no author details. Four things were removed to achieve that and none "
        "of them is content: this title block; the two in-text citations of the "
        "author's own companion note, which appear there as "
        "&ldquo;(Author 2026)&rdquo; and as &ldquo;Author (2026)&rdquo; in the "
        "reference list, with the note's public identifier withheld because it "
        "resolves to a page naming the author; the public repository URL in the "
        "data availability statement, which contains the author's name and is "
        "supplied to the editor separately; and the document title element. "
        "The companion note is posted and is supplied to the editor with this "
        "submission. Every figure, "
        "table, section and reference is otherwise identical to this "
        "submission's manuscript.</p>\n"
        "</body>\n</html>\n")


def build(key, spec):
    dst = os.path.join(OUT, key)
    os.makedirs(dst, exist_ok=True)
    raw = open(os.path.join(SRC, PAPER), encoding="utf-8").read()
    app = open(os.path.join(SRC, SUPP), encoding="utf-8").read()

    # swap the SSRN-length disclosure for the journal-length one
    _i = raw.find("<h2>Disclosure of AI use</h2>")
    _j = raw.find("<h2", _i + 10)
    if _i < 0 or _j < 0:
        raise SystemExit("build_variants: no disclosure section to replace")
    raw = raw[:_i] + JOURNAL_DISCLOSURE + raw[_j:]

    # The telephone number.  JFEc requires one on the title page - "name,
    # address, telephone number, and e-mail address of the author responsible
    # for correspondence" - so it cannot simply be dropped.  But the source is
    # published on GitHub, and a phone number written into a public source file
    # is one a scraper finds without reading the paper.  So the source carries
    # the sentence and not the number, and contact.txt - which is not published -
    # supplies it here, into the copies that are uploaded to a journal and
    # nowhere else.  Without that file the variants build fine and say the
    # number is supplied with the submission, which is true of any clone.
    _cf = os.path.join(HERE, "contact.txt")
    if os.path.isfile(_cf):
        _tel = _contact(_cf).get("telephone", "")
        if _tel:
            # INSERTED, not substituted.  The source used to carry the sentence
            # "Telephone supplied with the submission." as a placeholder for
            # this line to replace.  That sentence is true of a journal
            # submission and meaningless on a public preprint, where it reads
            # as leftover machinery in the first block a reader sees.  So the
            # published source now says nothing about a telephone at all, and
            # the journal copies - the only place one is required - have the
            # real line inserted before the email.
            _anchor = '<span class="nb">Email '
            if _anchor not in raw:
                raise SystemExit("build_variants: no email span to insert the "
                                 "telephone before")
            raw = raw.replace(
                _anchor,
                f'<span class="nb">Telephone {_tel}.</span> ' + _anchor, 1)

    if spec.get("title"):
        raw = re.sub(r"<h1>.*?</h1>", "<h1>" + spec["title"] + "</h1>", raw,
                     count=1, flags=re.S)
        raw = re.sub(r"<title>.*?</title>",
                     "<title>" + re.sub(r"<[^>]+>", " ", spec["title"]) + "</title>",
                     raw, count=1, flags=re.S)
    if spec.get("abstract"):
        raw = re.sub(r'(<div class="abs">\s*<p><b>Abstract</b></p>\s*<p>).*?(</p>)',
                     lambda m: m.group(1) + spec["abstract"] + m.group(2),
                     raw, count=1, flags=re.S)
    if spec.get("refs") == "jfec":
        # The journal's own reference style, computed from the base list rather
        # than maintained as a second copy of it.  Both documents are converted,
        # because a manuscript and its appendix in two different styles is the
        # first thing a copy editor sees.
        raw, app = _jfec_refs(raw), _jfec_refs(app)
    if spec.get("keywords"):
        raw = re.sub(r"(<b>Keywords:</b>)[^<]*", r"\1 " + spec["keywords"] + ".", raw,
                     count=1)
    if spec.get("replace_opening"):
        i = raw.index("<h2>1. Introduction</h2>")
        j = raw.index("<p>", i)
        # replace the first two paragraphs of the introduction
        k = j
        for _ in range(2):
            k = raw.index("</p>", k) + 4
        raw = raw[:j] + spec["replace_opening"] + raw[k:]
    if spec.get("insert_after_intro"):
        i = raw.index("<h2>1. Introduction</h2>")
        nxt = re.search(r'<h2(?: class="pb")?>\d+\.', raw[i + 10:])
        at = i + 10 + nxt.start()
        # Before the roadmap, not after it.  Appended at the end of the
        # introduction these blocks left the paper saying "the paper proceeds as
        # follows", listing the sections, and then starting two more arguments -
        # which is what a referee noticed first.  A roadmap is the last thing in
        # an introduction or it is not a roadmap.
        _road = re.search(r"<p[^>]*>\s*The paper proceeds as follows", raw[i:at])
        if _road:
            at = i + _road.start()
        raw = raw[:at] + spec["insert_after_intro"] + raw[at:]

    moved = []
    keep = spec.get("keep_float", {})
    _swaps = list(spec.get("swap_paras", []))
    # [[TOKEN]] -> the moved block it points at, resolved to an appendix
    # section number once those are assigned.  A stub that names its appendix
    # section by hand is a cross-reference nobody rebuilds.
    tokens = {}
    # A stub that carries an equation takes the SOURCE's own paragraph for it,
    # through [[EQ:n]], rather than a copy typed here.  The first version of
    # these stubs transcribed three equations into a different wrapper, and the
    # equation-numbering check reported that the manuscript cited (1), (2) and
    # (3) while defining only (6) to (11): a copied equation is a copied figure
    # with more ways to be wrong.
    _eqs = {int(m.group(1)): m.group(0) for m in re.finditer(
        r'<p class="eq">(?:(?!</p>).)*?<span class="n">\((\d+)\)</span></p>',
        raw, re.S)}
    if spec.get("statement"):
        # An end section that a journal wants as a STATEMENT rather than as two
        # pages: the short form stays under the same heading and the full
        # account moves to the appendix like any relocated section.
        _nm, _txt, _tok = spec["statement"]
        _i = raw.find(f"<h2>{_nm}</h2>")
        _j = raw.find("<h2", _i + 10)
        if _i < 0 or _j < 0:
            raise SystemExit(f"build_variants: no '{_nm}' section to shorten")
        moved.append((_nm, raw[_i:_j]))
        tokens[_tok] = _nm
        raw = raw[:_i] + f"<h2>{_nm}</h2>\n" + _put_eqs(_txt, _eqs) + "\n" \
            + raw[_j:]
    for _n, _tok in spec.get("token_for", {}).items():
        # A relocated section that leaves no stub still needs to be nameable:
        # the roadmap has to say where the reader will find it, and the only
        # honest pointer is the appendix number the build assigns.
        tokens[_tok] = _n
    if spec.get("trim_paras"):
        _name, _wants = spec["trim_paras"]
        raw, _cut = _trim_paras(raw, _wants)
        moved.append((_name, f'<h2 class="pb">{_name}</h2>\n' + "\n".join(_cut)))
    for n in spec.get("move", []):
        span = section_span(raw, n)
        if not span:
            continue
        chunk = raw[span[0]:span[1]]
        head = re.search(r'<h2(?: class="pb")?>\d+\.\s*([^<]*)</h2>', chunk)
        title = head.group(1).strip() if head else f"Section {n}"
        stay = ""
        float_html = ""
        if n in keep:
            # Lift the named float out of the moved chunk and leave it in the
            # manuscript.  A length cut may relocate discussion; it may not
            # silently drop a table or a figure the paper's argument rests on,
            # and verify_paper.py counts both.
            want = keep[n]
            if want.startswith("Table"):
                tm = re.search(r"<table>(?:(?!</table>).)*?<b>" + re.escape(want)
                               + r"\.</b>(?:(?!</table>).)*?</table>", chunk, re.S)
            else:
                tm = re.search(r"<figure>(?:(?!</figure>).)*?<b>" + re.escape(want)
                               + r"\.</b>(?:(?!</figure>).)*?</figure>", chunk, re.S)
            if tm:
                chunk = chunk[:tm.start()] + chunk[tm.end():]
                float_html = RETAIN.get(want, "") + tm.group(0) + "\n\n"
        # A section may also leave a STUB behind: the definitions and the
        # disclosures a later section cannot be read without, with the audits
        # and the discussion in the appendix.  Keeping the heading means the
        # manuscript still has a Data section, which is what a referee looks
        # for, rather than a hole where one was.
        stub = ""
        if n in spec.get("move_stub", {}):
            stub, _tok = spec["move_stub"][n]
            stub = _put_eqs(stub, _eqs)
            tokens[_tok] = n
        if stub or float_html:
            stay = (f'<h2 class="pb">{n}. {title}</h2>\n' + stub + float_html)
        raw = raw[:span[0]] + stay + raw[span[1]:]
        moved.append((n, chunk))

    for sub in spec.get("move_sub", []):
        m = re.search(r"<h3>" + re.escape(sub) + r"\s*([^<]*)</h3>", raw)
        if not m:
            continue
        nxt = re.search(r'<h3>|<h2(?: class="pb")?>', raw[m.end():])
        end = m.end() + (nxt.start() if nxt else 0)
        chunk = raw[m.start():end]
        # A relocated SUBSECTION can carry a float the body still cites, which
        # the whole-section path has always handled and this one did not: the
        # first build that moved 6.1 took Table 4 with it and left a reference
        # to a caption that was no longer in either document.  Same rule, same
        # retention, keyed on the subsection.
        stay = ""
        float_html = ""
        sub_title = m.group(1).strip()
        want = keep.get(sub)
        if want:
            if want.startswith("Table"):
                tm = re.search(r"<table>(?:(?!</table>).)*?<b>" + re.escape(want)
                               + r"\.</b>(?:(?!</table>).)*?</table>", chunk, re.S)
            else:
                tm = re.search(r"<figure>(?:(?!</figure>).)*?<b>" + re.escape(want)
                               + r"\.</b>(?:(?!</figure>).)*?</figure>", chunk, re.S)
            if tm:
                chunk = chunk[:tm.start()] + chunk[tm.end():]
                float_html = RETAIN.get(want, "") + tm.group(0) + "\n\n"
        stub = ""
        if sub in spec.get("sub_stub", {}):
            stub, _tok = spec["sub_stub"][sub]
            stub = _put_eqs(stub, _eqs)
            tokens[_tok] = sub
        # The heading comes back only with a stub.  A float retained from a
        # relocated subsection floats inside its parent section, which is how
        # Table 4 has shipped since 6.1 was first moved; re-emitting a heading
        # for it would put back the subsection the move exists to remove.
        stay = (f'<h3>{sub} {sub_title}</h3>\n' + stub + float_html) if stub \
            else float_html
        raw = raw[:m.start()] + stay + raw[end:]
        moved.append((sub, chunk))

    # Paragraphs that have to be REWRITTEN rather than moved, because what they
    # say about the manuscript stops being true when sections leave it.  The
    # roadmap is the case this exists for: "Section 2 places it against what is
    # already known" renumbered correctly to "Section S41" and read as a map of
    # this paper that sends the reader to an appendix at its first step.  A
    # replacement is written in BASE numbering like everything else here, so the
    # renumbering below treats it exactly as it treats the text around it.
    # Surgical substitutions, for sentences rather than paragraphs.  Used here
    # to put the manuscript in the present tense: several passages explained
    # what an earlier version had said and why it was withdrawn, which belongs
    # in a response to referees and reads as throat-clearing inside a paper.
    # Each replacement keeps the substance and drops the history; the record of
    # what changed lives in the repository and in the preprint.
    for _old, _new in spec.get("replace_text", []):
        # Whitespace-tolerant, because the source wraps its paragraphs and a
        # sentence written here on one line will not match one written there
        # across three.  Still required to match EXACTLY ONCE: a replacement
        # that silently matches nothing is an edit that did not happen.
        _pat = re.compile(r"\s+".join(re.escape(w) for w in _old.split()))
        _n = len(_pat.findall(raw))
        if _n != 1:
            raise SystemExit(f"build_variants: the text to replace appears {_n} "
                             f"times, not once: {_old[:70]!r}")
        raw = _pat.sub(lambda _m: _new.replace("\\", "\\\\"), raw, count=1)

    # Text that leaves the manuscript and must land somewhere: unlike the
    # substitutions above, which drop revision history the repository keeps,
    # this moves an argument the paper still needs - simulation detail whose
    # figures exist in no other document - into the appendix, where the
    # relocated sections already go.  Removing it without relocating it would
    # delete results, which is what the checks reported when the first attempt
    # did exactly that.
    for _old, _new, _title in spec.get("relocate_text", []):
        _pat = re.compile(r"\s+".join(re.escape(w) for w in _old.split()))
        _n = len(_pat.findall(raw))
        if _n != 1:
            raise SystemExit(f"build_variants: the text to relocate appears {_n} "
                             f"times, not once: {_old[:70]!r}")
        raw = _pat.sub(lambda _m: _new, raw, count=1)
        moved.append((_title, f'<h2 class="pb">{_title}</h2>\n<p>{_old}</p>'))

    for _want, _new in _swaps:
        _hit = None
        for _m in re.finditer(r"<p[^>]*>.*?</p>", raw, re.S):
            if _want in " ".join(re.sub(r"<[^>]+>", " ", _m.group(0)).split()):
                _hit = _m
                break
        if _hit is None:
            raise SystemExit(f"build_variants: no paragraph contains {_want!r} to "
                             f"swap, so the replacement is out of date with the "
                             f"manuscript")
        raw = raw[:_hit.start()] + _new + raw[_hit.end():]

    if moved:
        last_before = max(int(x) for x in re.findall(r'<h2 class="pb">S(\d+)\.', app))
        # renumber what is left so the manuscript reads 1, 2, 3, ...
        kept = [int(m.group(1)) for m in
                re.finditer(r'<h2(?: class="pb")?>(\d+)\.', raw)]
        remap = {old: i + 1 for i, old in enumerate(sorted(kept))}
        def _renum(m):
            return m.group(0).replace(m.group(1) + ".",
                                      str(remap[int(m.group(1))]) + ".", 1)
        raw = re.sub(r'<h2(?: class="pb")?>(\d+)\.', _renum, raw)
        # subsection HEADINGS carry the old parent number until they are
        # remapped too; leaving them produced a manuscript whose Section 3 held
        # subsections labelled 4.1 to 4.6 while the text cited 3.2
        #
        # The CHILD number needs the same treatment and did not get it.  Moving
        # 4.4 out renumbered the parents and left the survivors as 3.1, 3.2,
        # 3.3, 3.5, 3.6 - a methodology section that skips a subsection, in the
        # file a referee reads first, with nothing to explain the hole.  So the
        # survivors of each parent are renumbered consecutively, in document
        # order, and the map is kept for the in-text references below.
        sub_new = {}
        _seen = {}
        for _m in re.finditer(r'<h3>(\d+)\.(\d+)', raw):
            _par, _chi = int(_m.group(1)), int(_m.group(2))
            _newpar = remap.get(_par, _par)
            _seen[_newpar] = _seen.get(_newpar, 0) + 1
            sub_new[f"{_par}.{_chi}"] = f"{_newpar}.{_seen[_newpar]}"
        raw = re.sub(r'<h3>(\d+\.\d+)',
                     lambda m: f"<h3>{sub_new.get(m.group(1), m.group(1))}", raw)
        # In-text cross-references must follow the headings, or the manuscript
        # points at sections it no longer has.  verify_paper.py checks this and
        # caught it when the first version renumbered only the headings.  A
        # reference to a section that MOVED now names the Internet Appendix
        # section it became.
        # A reference to something that MOVED names the appendix section it
        # became - for whole sections and for relocated subsections alike,
        # since "Section 4.4" must not survive as "Section 3.4" pointing at a
        # subsection the manuscript no longer has.
        app_of, sub_of = {}, {}
        for i, (n, _c) in enumerate(moved, start=1):
            if isinstance(n, int):
                app_of[n] = f"S{last_before + i}"
            else:
                sub_of[n] = f"S{last_before + i}"

        # The stubs' pointers, now that the appendix numbers exist.
        for _tok, _key in tokens.items():
            _dest = app_of.get(_key) or sub_of.get(_key)
            if not _dest:
                raise SystemExit(f"build_variants: [[{_tok}]] points at {_key!r}, "
                                 f"which was not relocated")
            raw = raw.replace(f"[[{_tok}]]", _dest)
        if "[[" in raw:
            raise SystemExit("build_variants: an unresolved [[TOKEN]] is left in "
                             "the manuscript: "
                             + ", ".join(re.findall(r"\[\[[^\]]+\]\]", raw)))

        # Every reference form the manuscript uses: "Section 7", "Sections 4.5
        # and 11", "Sections 1, 9.2", "Section 6.1".  A first version handled
        # only the bare form and left "and 11" pointing at a section the
        # manuscript no longer had; verify_paper.py caught it.
        def _sub_one(m):
            n = int(m.group(1))
            tail = m.group(2) or ""
            if tail:                                   # 6.1 -> renumbered 6
                if f"{n}{tail}" in sub_of:             # unless it moved out
                    return sub_of[f"{n}{tail}"]
                # a surviving subsection whose child number shifted because an
                # earlier sibling left: the heading moved, so the citation must
                if f"{n}{tail}" in sub_new:
                    return sub_new[f"{n}{tail}"]
                return f"{remap.get(n, n)}{tail}"
            if n in remap:
                return str(remap[n])
            return app_of.get(n, str(n))

        def _refs(m):
            lead, rest = m.group(1), m.group(2)
            # Only BARE numbers are remapped.  An "S" prefix means the appendix's
            # own numbering, which this map knows nothing about: feeding "S13" to
            # _sub_one hands it the integer 13 and invites it to renumber an
            # appendix section as though it were a manuscript one.  It happens
            # not to bite today only because the manuscript has fewer sections
            # than the appendix does, which is luck and not a design.
            rest = re.sub(r"(S?)(\d+)(\.\d+)?",
                          lambda g: g.group(0) if g.group(1)
                          else _sub_one(re.match(r"(\d+)(\.\d+)?", g.group(0))),
                          rest)
            return lead + rest
        # Ranges are written "Sections 5 to 7" as well as as lists, and the
        # first version of this pattern knew only about lists.  So the range
        # matched its FIRST number and stopped: "Sections 5 to 7" became
        # "Sections 4 to 7" while the paper's sections ran 4, 5, 6 - an
        # introduction that contradicted itself on its second page, in the
        # file an editor opens first.  Nothing caught it, because both 4 and 7
        # existed and every dangling-reference check passed.
        _RE = (r"(Sections?\s+)((?:S?\d+(?:\.\d+)?)"
               r"(?:\s*(?:,|and|&amp;|to|through|&ndash;|-)\s*"
               r"S?\d+(?:\.\d+)?)*)")
        raw = re.sub(_RE, _refs, raw)
        # The appendix cites manuscript sections too, so it is renumbered by the
        # same map.  Leaving it alone pointed the appendix at sections the
        # manuscript no longer had, which verify_paper.py reports because it
        # reads the two documents together.  References that already name an
        # appendix section (S12, S33) are untouched: _sub_one only sees the
        # bare digits, and the "S" prefix is preserved by the outer pattern.
        def _refs_app(m):
            lead, rest = m.group(1), m.group(2)
            if "S" in rest:
                return m.group(0)
            return lead + re.sub(r"(\d+)(\.\d+)?", _sub_one, rest)
        app = re.sub(_RE, _refs_app, app)
        # the relocated sections become new appendix sections at the end
        last = last_before
        add = []
        for i, (n, chunk) in enumerate(moved, start=1):
            # A relocated section's OWN cross-references are in base numbering
            # and have to follow the renumbering like everything else: the Data
            # section pointed at "Section 9" for the coupling result, which the
            # move made Section 8, and the cross-reference guard found it in the
            # appendix rather than in the manuscript - which is the same error
            # one document further on.
            chunk = re.sub(_RE, _refs_app, _repoint(chunk))
            # a relocated block may be a whole section (<h2>) or one subsection
            # (<h3>); both arrive here and both keep their own heading text
            head = re.search(r'<h[23](?: class="pb")?>(?:[\d.]+\s*)?([^<]*)'
                             r'</h[23]>', chunk)
            title = head.group(1).strip() if head else f"Relocated section {n}"
            cut = re.search(r"</h[23]>", chunk)
            body = chunk[cut.end():] if cut else chunk
            add.append(f'<h2 class="pb">S{last + i}. {title}'
                       '<span style="font-weight:400;font-style:italic">'
                       '&nbsp;&middot;&nbsp;Moved from the manuscript for length'
                       "</span></h2>\n" + body)
        app = app.replace("</body>", "\n".join(add) + "\n</body>")
        # The appendix keeps its own books: a sentence counting its sections and
        # a claim map that every section carrying a verdict must appear in.
        # Relocating sections into it changes both, and verify_paper.py checks
        # both, so the builder updates them rather than leaving a variant that
        # contradicts its own front matter.
        # The count has to be taken the way the check takes it.  The first
        # version of this block counted '<h2 class="pb">S<n>.' headings, which
        # is 28 of the 41 sections: the page-break class is a typesetting
        # choice, not a marker of what a section is.  verify_paper.py counts
        # every '>S<n>.' heading, so the builder does too - a builder that
        # derives a number by a different rule than the checker will agree by
        # luck and disagree silently the moment the two rules diverge.
        _new = sorted({int(x) for x in re.findall(r">S(\d+)\.", app)})
        _hi, _n_sub = max(_new), len([x for x in _new if x > 0])
        app = re.sub(r"[A-Za-z-]+ substantive sections, S1 through S\d+",
                     f"{_spell(_n_sub).capitalize()} substantive sections, "
                     f"S1 through S{_hi}", app, count=1)
        # Enumerated, not written as a range.  The coverage check reads the map
        # by collecting every S<n> that appears in it, so "S38 to S41" maps S38
        # and S41 and leaves S39 and S40 uncovered - which is what it reported.
        # Every other row in the map enumerates; this one now does too.
        _reloc = ", ".join(f"S{i}" for i in range(last_before + 1, _hi + 1))
        row = ('<tr class="botrule"><td>Sections the manuscript cites but does not '
               'carry</td><td>' + _reloc +
               '</td><td>relocated from the manuscript for length; the claims they '
               'support are graded in the manuscript\'s own claim table</td></tr>')
        # Both edits are scoped to the Table S0 span, not the whole document.
        # Unscoped, '</tbody>\n</table>' hit the first table in the file, so the
        # new claim-map row was appended to a different table entirely and the
        # map went on missing the relocated sections.  The check itself reads
        # only the span from the "Table S0." caption to its first closing tag,
        # so that is the span the builder edits.
        _s0 = re.search(r"Table S0\..*?</table>", app, re.S)
        if not _s0:
            raise SystemExit("build_variants: no Table S0 span in the appendix")
        _blk = _s0.group(0)
        _blk = re.sub(r'<tr class="botrule">(?!.*<tr class="botrule">)',
                      "<tr>", _blk, count=1, flags=re.S)
        _blk = _blk.replace("</tbody>", row + "\n</tbody>", 1)
        app = app[:_s0.start()] + _blk + app[_s0.end():]

    if spec.get("unnumbered_intro"):
        # Applied last, to both documents at once, because the appendix cites
        # manuscript sections and a shift applied to one and not the other
        # points half the cross-references at the wrong place.
        raw, app = _unnumber_intro([raw, app])

    # --- the blinded pair, for journals that review double-anonymised -------
    # IJF and JEF both run double-anonymised review and both ask for the title
    # page as a SEPARATE file from a manuscript carrying no author details.
    # JFEc asks for a separate title page too.  Producing these by hand is how
    # a name survives into a "blinded" file: the obvious identifiers go and the
    # unobvious ones stay.  There are four kinds here and only the first is
    # obvious.
    #
    #   1. the byline, affiliation and contact block
    #   2. the two in-text citations of the author's own companion note, and
    #      its entry in the reference list - a reviewer who reads "(Vellanikaran
    #      2026)" has the author's name whatever the title page says
    #   3. the repository URL in the availability statement, which is
    #      github.com/SunnyAlexV and so is the author's name spelled out
    #   4. the running title and any <title> element
    #
    # The blinded manuscript keeps every number and every section, so it is the
    # same paper: only identity is removed, and what was removed is listed in
    # the title page file so the editor can see it was not content.
    _title = re.search(r"<h1>(.*?)</h1>", raw, re.S)
    _blind = raw
    _blind = re.sub(r'<p class="byline">.*?</p>\s*', "", _blind, flags=re.S)
    _blind = re.sub(r'<p class="aff">.*?</p>\s*', "", _blind, flags=re.S)
    _blind = re.sub(r'<div class="frontrule"></div>\s*<p class="front">.*?</p>\s*',
                    "", _blind, flags=re.S)
    _blind = _blind.replace("(Vellanikaran 2026)", "(Author 2026)")
    # The self-citation, in either reference style.  The base list writes
    # "Vellanikaran, S. A. (2026)." and the JFEc conversion writes
    # "Vellanikaran, S. A. 2026."; pinning the first spelling meant the
    # conversion silently carried the author's name into a file named
    # "anonymous", and the guard below - which exists for exactly this - caught
    # it on the first build.  One pattern now covers both shapes.
    _blind = re.sub(r"Vellanikaran, S\. A\.\s*\(?(\d{4})\)?\.",
                    r"Author (\1).", _blind)
    _blind = _blind.replace(
        "github.com/SunnyAlexV/delayed-information-cross-section",
        "[repository URL withheld for anonymous review; supplied to the editor]")
    # The companion note's reference entry used to read "Working note." and now
    # carries its SSRN identifier, which is right for the version of record and
    # wrong for an anonymous one: the abstract page names the author, so a
    # reviewer who follows the link is one click from knowing who wrote this.
    # Replacing the byline and leaving a resolvable pointer to it blinds
    # nothing, which is the same failure the appendix byline was.
    _blind = SSRN_NOTE_RE.sub(ANON_NOTE, _blind)
    _blind = re.sub(r"<title>.*?</title>",
                    "<title>Manuscript for anonymous review</title>", _blind,
                    flags=re.S)
    # The Internet Appendix ships WITH the blinded manuscript as supplementary
    # material, so blinding one and not the other blinds nothing: the appendix
    # carried the byline and one "(Vellanikaran 2026)" of its own.  It goes
    # through the same treatment and the same refusal.
    _bapp = app
    _bapp = re.sub(r'<p class="byline">.*?</p>\s*', "", _bapp, flags=re.S)
    _bapp = re.sub(r'<p class="aff">.*?</p>\s*', "", _bapp, flags=re.S)
    # The appendix does not use the manuscript's byline classes: its three
    # front-matter lines are centred class="t" paragraphs.  Reusing the
    # manuscript's patterns removed nothing at all and the author's name went
    # straight through into a file labelled anonymous, which is precisely what
    # the refusal below is for - the blinding rule has to match the markup in
    # front of it, not the markup it expects.
    _bapp = re.sub(r'<p class="t" style="text-align:center">(?:<i>)?'
                   r'(?:Sunny Alex Vellanikaran|Independent)(?:</i>)?</p>\s*',
                   "", _bapp)
    _bapp = re.sub(r'<div class="frontrule"></div>\s*<p class="front">.*?</p>\s*',
                   "", _bapp, flags=re.S)
    _bapp = _bapp.replace("(Vellanikaran 2026)", "(Author 2026)")
    _bapp = re.sub(r"Vellanikaran, S\. A\. \(2026\)\.", "Author (2026).", _bapp)
    _bapp = _bapp.replace(
        "github.com/SunnyAlexV/delayed-information-cross-section",
        "[repository URL withheld for anonymous review; supplied to the editor]")
    _bapp = SSRN_NOTE_RE.sub(ANON_NOTE, _bapp)
    _bapp = re.sub(r"<title>.*?</title>",
                   "<title>Internet Appendix for anonymous review</title>",
                   _bapp, flags=re.S)
    _cfb = os.path.join(HERE, "contact.txt")
    for _doc, _what in ((_blind, "manuscript"), (_bapp, "Internet Appendix")):
        # The fixed list, plus whatever contact.txt currently holds.  Reading
        # the file rather than restating its contents is the point: a postal
        # address was added to the title page and this list did not learn
        # about it, so for one build a new identifying field had no refusal
        # behind it.  Anything contact.txt supplies to a journal copy is by
        # definition something the anonymous copy must not contain.
        _secret = []
        if os.path.isfile(_cfb):
            for _v in _contact(_cfb).values():
                _v = _v.strip().rstrip(".")
                if len(_v) > 6 and not _v.startswith("<"):
                    _secret.append(_v)
                    # and the distinctive pieces of it, since a line break or a
                    # reformat would defeat a whole-string match
                    _secret += [_p.strip() for _p in _v.split(",")
                                if len(_p.strip()) > 5]
        # NOTE: no private string is written out here.  An earlier version
        # listed the telephone number as a literal so the guard would still
        # fire when contact.txt was absent, which published the number to
        # every reader of this file - the guard against a leak was the leak.
        # The private values come from contact.txt at run time (_secret above)
        # and from nowhere else; when that file is missing the guard says so
        # and checks only the identifiers the papers already carry in public.
        _public = ["Sunny Alex Vellanikaran", "Vellanikaran", "SunnyAlexV",
                   "sunnyalex1234@gmail.com",
                   # the SSRN identifier of the author's own companion note:
                   # not a name, but one click from one
                   "7504558", "ssrn.com/abstract"]
        if not _secret:
            print("  build_variants: contact.txt absent, so the blinding check "
                  "covers the public identifiers only; the contact-derived "
                  "checks are skipped")
        for _who in _public + _secret:
            if _who in re.sub(r"<title>.*?</title>", "", _doc, flags=re.S):
                raise SystemExit(
                    f"build_variants: '{_who}' survives into the blinded {key} "
                    f"{_what}; blinding is not safe to ship until that is gone")

    open(os.path.join(dst, "manuscript-anonymous.html"), "w",
         encoding="utf-8").write(_blind)
    open(os.path.join(dst, "internet-appendix-anonymous.html"), "w",
         encoding="utf-8").write(_bapp)
    open(os.path.join(dst, "title-page.html"), "w", encoding="utf-8").write(
        _title_page(spec, _title.group(1) if _title else "", raw))

    # The abstract's word limit, where the journal states one.  This check
    # exists because the limit was known, was designed around - two guards in
    # verify_paper.py were moved OUT of the abstract to make room under it -
    # and was then exceeded anyway, by nineteen words, with nothing to notice.
    # A constraint that is honoured by memory is a constraint that drifts, and
    # this is the one dimension of the paper that had no check on it at all.
    _cap = spec.get("abstract_max")
    if _cap:
        _ab = re.search(r'<div class="abs">\s*<p><b>Abstract</b></p>\s*<p>(.*?)</p>',
                        raw, re.S)
        if not _ab:
            raise SystemExit(f"build_variants: {key} has no abstract to measure")
        _words = len(" ".join(_html.unescape(
            re.sub(r"<[^>]+>", " ", _ab.group(1))).split()).split())
        if _words > _cap:
            raise SystemExit(
                f"build_variants: the {key} abstract is {_words} words and the "
                f"journal allows {_cap}; it is {_words - _cap} over and would be "
                f"checked before an editor reads a line of it")

    open(os.path.join(dst, PAPER), "w", encoding="utf-8").write(raw)
    open(os.path.join(dst, SUPP), "w", encoding="utf-8").write(app)

    # Five documents land in this folder and two of them NAME THE AUTHOR.
    # "stale-mark-internet-appendix.pdf" sits next to
    # "internet-appendix-anonymous.pdf", one identified and one not, and only
    # attention stands between them and a broken anonymity - a single wrong
    # click is unrecoverable, because the editor then has the name.  Moving the
    # identified pair into a subfolder was tried and reverted: four other
    # scripts path into papers/variants/<key>/<name>.html, so the move fixed a
    # naming hazard by creating four silent ones.  What ships instead is an
    # upload manifest, written from this spec rather than typed, naming exactly
    # what goes up and what must not.  verify_repo.py checks it exists and
    # lists the blinded three.
    # All three target journals review double-anonymised, which is why every
    # variant folder holds a blinded pair; if a fourth is ever added that does
    # not, this manifest is the thing to branch.
    #
    # The letter's filename is looked up rather than derived: key.upper() gives
    # "JFEC" and the file is "JFEc", so a manifest that built the name from the
    # key pointed the author at a file that does not exist.  If the lookup
    # misses, the manifest says so instead of inventing a path.
    _letter = {"ijf": "IJF-cover-letter.md", "jef": "JEF-cover-letter.md",
               "jfec": "JFEc-cover-letter.md"}.get(key, "")
    if not _letter or not os.path.isfile(os.path.join(HERE, "submission", _letter)):
        raise SystemExit(f"build_variants: no cover letter on disk for {key}; the "
                         f"upload manifest would name a file the author cannot find")
    _lines = [
        "UPLOAD THESE, IN THIS ORDER", "",
        "  0.  the cover letter        submission/" + _letter
        + "  (pasted, not uploaded)",
        "  1.  title-page.pdf          submitted, NOT for review",
        "  2.  manuscript-anonymous.pdf          the manuscript for review",
        "  3.  internet-appendix-anonymous.pdf   the appendix for review",
        "",
        "DO NOT UPLOAD - THESE TWO NAME THE AUTHOR", "",
        "  " + PAPER[:-5] + ".pdf",
        "  " + SUPP[:-5] + ".pdf",
        "",
        "They are the identified copies, kept for reading. Sending either one",
    ] + ([
        "ends the anonymity this journal's double-anonymised review depends on,",
        "and that cannot be undone.",
    ] if spec.get("review") == "double-anonymised" else [
        "ends the anonymity if the review is in fact anonymous, which cannot be",
        "undone. Nothing here asserts that it is: neither this journal's General",
        "Instructions nor its submission page states a review model, and its",
        "instructions require a title page naming the author. The package",
        "satisfies either model - an anonymous manuscript plus a title page -",
        "which is why the uncertainty costs nothing.",
    ])
    open(os.path.join(dst, "UPLOAD-THESE.txt"), "w", encoding="utf-8").write(
        "\n".join(_lines) + "\n")
    return dst, len(" ".join(re.sub(r"<[^>]+>", " ", raw).split()).split())


def readme(rows):
    """A note describing the variants, derived from the spec that built them.

    Written rather than typed for the same reason the variants themselves are
    generated: a hand-written note saying "Section 2 moved to the appendix"
    goes on saying it after the spec stops moving Section 2.  Everything below
    is read out of VARIANTS and out of the build, so the note is wrong only if
    the build is.
    """
    out = ["# Journal variants", "",
           "One source, three submissions.  Each directory holds a manuscript and",
           "its Internet Appendix, in HTML and PDF.  They are GENERATED by",
           "`build_variants.py` from `papers/source/`: edit the source, not these.",
           "Every variant is re-verified against the 63 lab outputs after it is",
           "built, and the count below is that verifier's own count.", "",
           "Page counts in the geometry a page limit means (US Letter, 12pt,",
           "one-inch margins, double spaced) come from `measure_pages.py`, which",
           "reports two columns because the guidelines do not say whether tables",
           "are double spaced.  The rendered PDFs here are the house format:",
           "A4, 10.5pt, single spaced, and much shorter in pages.", ""]
    for k, name, words, ok, _ in rows:
        v = VARIANTS[k]
        out += [f"## `{k}/` - {name}", "",
                f"- {words:,} words in the manuscript; checks: "
                f"{'pass' if ok else 'FAIL'}"]
        out.append("- title: " + ("retitled for this journal" if v.get("title")
                                  else "unchanged from the base paper"))
        out.append("- abstract: " + ("rewritten for this readership"
                                     if v.get("abstract") else "unchanged"))
        if v.get("keywords"):
            out.append("- keywords: " + v["keywords"])
        if v.get("insert_after_intro"):
            out.append("- an added framing passage after the introduction")
        if v.get("replace_opening"):
            out.append("- a replaced opening passage")
        if v.get("move") or v.get("move_sub"):
            moved = ", ".join(["Section " + str(n) for n in v.get("move", [])] +
                              ["Section " + s for s in v.get("move_sub", [])])
            out.append(f"- moved to the Internet Appendix for length: {moved}."
                       " Nothing is deleted; the relocated sections keep their"
                       " text and are renumbered into the appendix, and every"
                       " in-text reference is rewritten to match.")
        for n, f in (v.get("keep_float") or {}).items():
            out.append(f"- kept in the body although Section {n} moved: {f}")
        out.append("")
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(
        "\n".join(out).rstrip() + "\n")


def main(argv):
    keys = [a for a in argv[1:] if a in VARIANTS] or list(VARIANTS)
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for k in keys:
        dst, words = build(k, VARIANTS[k])
        r = subprocess.run([sys.executable, os.path.join(HERE, "verify_paper.py"),
                            os.path.join(dst, PAPER),
                            os.path.join(SRC, "why-a-simple-threshold-rule-is-hard-to-beat.html")],
                           capture_output=True, text=True)
        ok = "every checked figure" in r.stdout
        # Render here, not in a separate command.  A variant whose HTML is
        # regenerated and whose PDF is not is the drift this whole file exists
        # to prevent, and it is the copy a journal receives.  build_pdf.py is
        # invoked rather than reimplemented so all five documents come out of
        # one set of render settings.
        pdf = subprocess.run([sys.executable, os.path.join(HERE, "build_pdf.py"),
                              "--src=" + dst, "--out=" + dst,
                              PAPER[:-5], SUPP[:-5], "manuscript-anonymous",
                              "internet-appendix-anonymous", "title-page"],
                             capture_output=True, text=True)
        if pdf.returncode != 0:
            ok = False
            r = subprocess.CompletedProcess(
                [], 0, stdout=r.stdout + "\nPDF render failed:\n" + pdf.stderr, stderr="")
        rows.append((k, VARIANTS[k]["name"], words, ok, r.stdout.strip()))
    if set(keys) == set(VARIANTS):
        readme(rows)
    print(f"{'variant':>6}  {'words':>6}  {'checks':>7}  journal")
    for k, name, words, ok, out in rows:
        print(f"{k:>6}  {words:>6}  {'pass' if ok else 'FAIL':>7}  {name}")
        if not ok:
            for ln in out.splitlines()[1:6]:
                print("          " + ln)
    return 0 if all(r[3] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
