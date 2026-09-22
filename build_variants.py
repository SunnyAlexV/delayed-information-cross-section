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

IJF_ABSTRACT = """How much forecasting performance is lost when one input arrives late, and how
much of that loss can a timely substitute repair? We answer both in a controlled
ragged-edge experiment: a forecaster's own history is withheld for &delta; periods
while every other source stays current, so the delay is imposed rather than
observed and its cost is measured against the same forecaster undelayed. The loss
is summarised by a substitution rate R(&delta;), the share of delay-induced
forecasting loss a substitute returns, which is unit-free, bounded below by zero
and comparable across targets, losses and horizons. Three things make it usable.
First, the design freezes a whole information system rather than one feature, and
we separate the two: an infeasible feature-only arm prices staleness of the input
against staleness of the estimated model. Second, R(&delta;) is a ratio whose
denominator vanishes as the delay cost goes to zero, so it is weakly identified at
short delays; we invert a HAC t-test after Fieller and report where the data do not
pin it down. Third, the targets overlap, and we set the bootstrap block and the HAC
bandwidth from measured dependence rather than convention. Applied to S&amp;P 500
realised variance with seven foreign equity closes as the timely substitute, a
current cross-section returns 71% [59, 81] of what eleven weeks of staleness takes.
In both own-index option cases examined, the compressed cross-section adds no
detectable information once own-index implied volatility is included;
read as a clock, a fifty-five-day-old mark plus today's foreign closes forecasts as
well as a mark 4.6 days old [2.5, 8.4]. The same design run on twenty metropolitan
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
returns. A current foreign cross-section returns 71% [59, 81] of what eleven weeks of
staleness takes, on the implementable target and the original model,
with four specifications spanning 68% to 75% and every interval excluding zero. Read
as a clock rather than a share, a fifty-five-day-old mark plus today's foreign closes
forecasts as well as a plain domestic mark 4.6 days old [2.5, 8.4]. The rate is a
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

JFEC_FRAMEWORK = """<p><b>What this paper offers beyond one cross-section.</b> Three components are
separable from the application and reusable without it. The first is the estimand:
R(&delta;) is a ratio of forecasting losses, unit-free, bounded below by zero,
comparable across targets, losses and horizons, and defined so that the benchmark
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
    "Figure 3": (
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
JOURNAL_DISCLOSURE = """<h2>Disclosure of AI use</h2>
<p>The research questions, mathematics, empirical design, interpretation of results, and
decisions about the paper's claims are the author's own. The author is solely
responsible for all contents of the paper, including every claim, design decision, and
reported number. No AI system is an author of this work or held authorial responsibility
for any part of it.</p>

<p>During preparation of this manuscript and its Internet Appendix, the author used
Claude (Anthropic) as an assistive tool for drafting and revising text and code. The
author independently specified the analyses, executed them, reviewed, tested, and
validated the code and all reported outputs, and retained full responsibility for the
resulting work.</p>

<p>In the wording requested by certain target journals: during preparation of this work,
the author used Claude (Anthropic) to assist with drafting and revising manuscript text
and research code. After using this tool, the author reviewed and edited the content as
needed and takes full responsibility for the publication.</p>

"""


VARIANTS = {
    "ijf": {
        "name": "International Journal of Forecasting",
        "title": IJF_TITLE,
        "abstract": IJF_ABSTRACT,
        "keywords": ("information delay; ragged edge; forecast evaluation; "
                     "substitution rate; weak identification; overlapping horizons"),
        "insert_after_intro": IJF_GENERAL,
        "move": [],
    },
    "jef": {
        "name": "Journal of Empirical Finance",
        "title": JEF_TITLE,
        "abstract": JEF_ABSTRACT,
        "keywords": ("stale prices; information delay; volatility forecasting; "
                     "cross-sectional information; weak identification"),
        "replace_opening": JEF_OPENING,
        "move": [],
    },
    "jfec": {
        "name": "Journal of Financial Econometrics",
        "title": None,                      # keeps the base title
        "abstract": None,                   # keeps the base abstract
        "keywords": None,
        "insert_after_intro": JFEC_FRAMEWORK,
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
        "move": [2, 9, 10],
        "move_sub": ["4.4"],
        "keep_float": {10: "Table 5", 9: "Figure 3"},
    },
}


# ----------------------------------------------------------------------
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def _spell(n):
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + ("-" + _ONES[o] if o else "")


def section_span(raw, n):
    """(start, end) of numbered top-level section n in the manuscript."""
    m = re.search(r'<h2(?: class="pb")?>' + str(n) + r'\.', raw)
    if not m:
        return None
    nxt = re.search(r'<h2(?: class="pb")?>(?:\d+\.|Data and code|Disclosure)',
                    raw[m.end():])
    return (m.start(), m.end() + nxt.start() if nxt else len(raw))


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
        f'<p class="front"><b>Corresponding author.</b> {front}</p>\n'
        + (f'<p class="kw">{kw.group(1)}</p>\n' if kw else "")
        + '<p class="front"><b>Submitted to</b> ' + spec["name"] + ".</p>\n"
        '<p class="front"><b>Anonymity.</b> The accompanying manuscript carries '
        "no author details. Four things were removed to achieve that and none "
        "of them is content: this title block; the two in-text citations of the "
        "author's own companion note, which appear there as "
        "&ldquo;(Author 2026)&rdquo; and as &ldquo;Author (2026)&rdquo; in the "
        "reference list; the public repository URL in the data availability "
        "statement, which contains the author's name and is supplied to the "
        "editor separately; and the document title element. Every figure, "
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
        _tel = open(_cf, encoding="utf-8").read().strip()
        if _tel:
            raw = raw.replace("Telephone supplied with the submission.",
                              f"Telephone {_tel}.", 1)

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
        raw = raw[:at] + spec["insert_after_intro"] + raw[at:]

    moved = []
    keep = spec.get("keep_float", {})
    for n in spec.get("move", []):
        span = section_span(raw, n)
        if not span:
            continue
        chunk = raw[span[0]:span[1]]
        head = re.search(r'<h2(?: class="pb")?>\d+\.\s*([^<]*)</h2>', chunk)
        title = head.group(1).strip() if head else f"Section {n}"
        stay = ""
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
                lead = RETAIN.get(want, "")
                stay = (f'<h2 class="pb">{n}. {title}</h2>\n' + lead
                        + tm.group(0) + "\n\n")
        raw = raw[:span[0]] + stay + raw[span[1]:]
        moved.append((n, chunk))

    for sub in spec.get("move_sub", []):
        m = re.search(r"<h3>" + re.escape(sub) + r"\s*([^<]*)</h3>", raw)
        if not m:
            continue
        nxt = re.search(r'<h3>|<h2(?: class="pb")?>', raw[m.end():])
        end = m.end() + (nxt.start() if nxt else 0)
        chunk = raw[m.start():end]
        raw = raw[:m.start()] + raw[end:]
        moved.append((sub, chunk))

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
        raw = re.sub(r'<h3>(\d+)(\.\d+)',
                     lambda m: (f"<h3>{remap.get(int(m.group(1)), m.group(1))}"
                                f"{m.group(2)}"), raw)
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
                return f"{remap.get(n, n)}{tail}"
            if n in remap:
                return str(remap[n])
            return app_of.get(n, str(n))

        def _refs(m):
            lead, rest = m.group(1), m.group(2)
            rest = re.sub(r"(\d+)(\.\d+)?", _sub_one, rest)
            return lead + rest
        _RE = (r"(Sections?\s+)((?:S?\d+(?:\.\d+)?)"
               r"(?:\s*(?:,|and|&amp;)\s*S?\d+(?:\.\d+)?)*)")
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
            # a relocated block may be a whole section (<h2>) or one subsection
            # (<h3>); both arrive here and both keep their own heading text
            head = re.search(r'<h[23](?: class="pb")?>[\d.]+\s*([^<]*)</h[23]>',
                             chunk)
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
    _blind = re.sub(r"Vellanikaran, S\. A\. \(2026\)\.",
                    "Author (2026).", _blind)
    _blind = _blind.replace(
        "github.com/SunnyAlexV/delayed-information-cross-section",
        "[repository URL withheld for anonymous review; supplied to the editor]")
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
    _bapp = re.sub(r"<title>.*?</title>",
                   "<title>Internet Appendix for anonymous review</title>",
                   _bapp, flags=re.S)
    for _doc, _what in ((_blind, "manuscript"), (_bapp, "Internet Appendix")):
        for _who in ("Sunny Alex Vellanikaran", "Vellanikaran", "SunnyAlexV",
                     "sunnyalex1234@gmail.com", "7920 640772"):
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

    open(os.path.join(dst, PAPER), "w", encoding="utf-8").write(raw)
    open(os.path.join(dst, SUPP), "w", encoding="utf-8").write(app)
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
                            os.path.join(SRC, "threshold-rule-at-its-ceiling.html")],
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
