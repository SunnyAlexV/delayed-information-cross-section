"""selftest.py - do the gates actually fire?  One command, one verdict.

    python selftest.py              run every case
    python selftest.py --list       show the register without running it
    python selftest.py bootstrap    run the cases whose name matches

WHY THIS FILE EXISTS
--------------------
Every verifier in this repository passes.  That is not evidence of anything on
its own: a check that reads the wrong surface, or compares a thing to itself,
or was written against a sentence that has since been reworded, passes exactly
like a check that works.  Four such gates have been found here, each after
weeks of passing.

So the question "is the paper checked?" is the wrong question.  The right one
is "for each way this paper has been wrong before, does a gate exist, and does
it FAIL when the defect is reintroduced?"  That is what this file answers.

Each case below names a defect class that actually occurred, the gate that now
covers it, and a one-line perturbation that reintroduces it.  The case passes
only if the gate fails with the expected signature.  A gate that stops firing -
because a sentence was reworded, a file moved, or a check was quietly
loosened - shows up here as a FAILED CASE, which is the opposite of what a
green verifier would tell you.

It runs against a COPY of the shipping set in a temporary directory, so a
crash mid-run cannot leave an edited file behind.  The run is read-only with
respect to the repository.

HOW TO READ THE RESULT
----------------------
    15 of 15 gates fire      every known defect class is covered and live
    14 of 15 gates fire      one gate has gone blind; the name tells you which

The number is a floor, not a ceiling.  It says nothing about defect classes
nobody has thought of yet - the README figures went unchecked for a year
because nobody had thought to ask whether that file was a surface.  When a new
class is found, the fix is not finished until a case for it appears here.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, what the class is, file, find, replace, gate argv, expected text in output)
#
# `find`/`replace` reintroduce the defect.  The gate must then FAIL, and its
# output must contain `expect` - so a case cannot pass because something
# unrelated broke.
PAPER = ["verify_paper.py",
         "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
         "papers/source/why-a-simple-threshold-rule-is-hard-to-beat.html"]
REPO = ["verify_repo.py"]
LETTERS = ["verify_letters.py"]
VARIANTS = ["build_variants.py"]
REGISTRY = ["build_registry.py"]

CASES = [
    # ---- the paper against the labs ---------------------------------------
    ("figure-vs-lab",
     "a figure in the manuscript no longer matches the lab that produces it",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "that share is 42.0 points higher", "that share is 41.0 points higher",
     PAPER, "42.0 points higher"),

    ("band-excludes-its-own-row",
     "the specification band is quoted at a precision that excludes one of the "
     "rows it summarises - round(75.3) = 75 certified a band Table 1 "
     "contradicts.  The state-space row reads 75.3%, so a band quoted to the "
     "nearest point drops it",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "all landing between 67.4% and 75.3%,",
     "all landing between 67% and 75%,",
     PAPER, "all landing between 67.4% and 75.3%"),

    ("ai-disclosure-weakened",
     "the AI disclosure is trimmed back to the generic form, dropping what the "
     "tool did NOT produce and how a reader can check it",
     # the disclosure is injected by build_variants into the JOURNAL copies, so
     # it is not in papers/source/ at all - the first version of this case
     # perturbed the source and reported STALE, which is the register working:
     # a case that no longer reaches its target says so instead of passing.
     "papers/variants/jfec/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     # A short fragment, because the paragraph is rewrapped whenever a word
     # changes and a one-line needle then matches nothing and reports STALE.
     "produced by that tool",
     "produced somehow",
     ["verify_paper.py", "papers/variants/jfec/breadth-implied-volatility-and-forecasting-with-stale-information.html",
      "papers/source/why-a-simple-threshold-rule-is-hard-to-beat.html"],
     "disclosure"),

    # ---- the README, which had no figure check at all until 4 Oct ---------
    ("readme-figure-vs-lab",
     "a README lab row quotes a figure that lab's own output does not produce - "
     "sixteen were wrong when the gate was first written",
     "README.md",
     "is 42.0 points higher in the top 5%", "is 43.0 points higher in the top 5%",
     REPO, "README's lab67_regime_contrast row quotes 43.0"),

    ("readme-multiplicity-sentence",
     "the README's multiplicity sentence goes stale in a way the row-scoped "
     "figure check cannot see, because the wrong number still occurs in that "
     "lab's output in a different role",
     "README.md",
     "the largest being 62", "the largest being 64",
     REPO, "multiplicity sentence"),

    # ---- the repository as a shipped artefact -----------------------------
    ("absolute-path-in-shipped-script",
     "a shipped script names a path that exists only on the machine it was "
     "written on, so it runs for the author and nobody else",
     "verify_repo.py",
     "HERE = os.path.dirname(os.path.abspath(__file__))",
     "HERE = os.path.dirname(os.path.abspath(__file__))\n_SCRATCH = '/tmp/claude-0/x.txt'",
     REPO, "absolute path"),

    ("provenance-vs-manifest",
     "data/illiquid/PROVENANCE.md duplicates eight digests and five counts that "
     "live in the manifest, and nothing read it until 4 Oct",
     "data/illiquid/PROVENANCE.md",
     "| 206 | 3713 |", "| 207 | 3713 |",
     REPO, "PROVENANCE.md gives"),

    ("provenance-vs-lab",
     "PROVENANCE.md quotes two figures from lab54 and can drift from them",
     "data/illiquid/PROVENANCE.md",
     "needs about 115", "needs about 116",
     REPO, "month-marks"),

    ("lab-output-bijection",
     "a lab ships without the stored output it is diffed against, or the "
     "reverse - lab67 sat in labs/ for days while --check never ran it",
     "run_all.py",
     '("lab68_joint_multiplicity.py"', '("lab68_joint_multiplicity_MISSING.py"',
     REPO, "lab"),

    # ---- the letters against the package ----------------------------------
    ("letter-vs-package",
     "a cover letter states a check count, word count or page count that the "
     "artefact it describes does not have",
     # the needle is the LAB COUNT, not the check count.  The first version
     # perturbed "runs 2,726 checks", which moves whenever a check is added -
     # so the case went stale within the day and reported itself stale, which
     # is the register behaving correctly and still a case that was not
     # testing anything.  69 labs is stable across everything but a new lab.
     #
     # The lab count went too, when counts of the apparatus came out of the
     # letters, and the case reported itself skipped rather than passing on a
     # pattern that matched nothing.  Repointed, not relaxed: the page count
     # is what this class is actually about and it is the figure the editors
     # read, measured by measure_pages.py from the built PDF.
     "submission/JFEc-cover-letter.md",
     r"re:The file is \d+ pages", "The file is 41 pages",
     LETTERS, "states pages"),

    ("prose-figure-from-a-different-lab",
     "the prose explains a table with a figure measured by a DIFFERENT lab.  "
     "Section 6 read '+0.3019 gross' eleven lines under a Table 3 row printing "
     "+0.3009: both numbers were right about their own lab, so every figure "
     "check passed, and the check behind the sentence was simply reading the "
     "wrong surface.  A referee found it by reading the page.  The gate now "
     "requires a four-decimal figure explained beside a table to be a figure "
     "that table prints",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "+0.3009 gross against +0.2495 net", "+0.3019 gross against +0.2495 net",
     PAPER, "which no table in this manuscript prints"),

    ("repeated-word-in-prose",
     "a word is typed twice in a sentence.  'it is ahead by by 0.0120' sat in "
     "the Internet Appendix through four review rounds and 2,955 passing "
     "checks, because every check in the file reads a number and none reads a "
     "sentence as a sentence",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information-internet-appendix.html",
     "it is ahead by 0.0120", "it is ahead by by 0.0120",
     PAPER, "is repeated"),

    ("two-spellings-in-one-document",
     "the same word is spelled two ways in one document - toward and towards, "
     "or -ise beside -ize.  No dictionary is involved and there is nothing to "
     "allow: the document is read against itself, so the check cannot be "
     "widened until it passes",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information-internet-appendix.html",
     "towards a ceiling below one", "toward a ceiling below one",
     PAPER, "both spellings"),

    ("letter-cross-reference-off-by-one",
     "a cover letter sends the editor to an appendix section that exists and "
     "is about something else.  The letter cited a section for the two "
     "withdrawn series which is about the volatility proxy instead; the "
     "perturbation below moves the citation off S17, which is where the "
     "withdrawal and the leave-one-out table that replaces it now live, and "
     "onto S38, which exists and is about something else.  Every check in "
     "place passed, because each one asked whether a number was right and "
     "none asked whether a POINTER was - an existence check would have passed "
     "too.  The gate now reads the subject out of the citing sentence and "
     "requires the cited section to carry it",
     "submission/JFEc-cover-letter.md",
     "Section S17 of the", "Section S38 of the",
     LETTERS, "which that section does not mention"),

    # ---- the page-limit overage, which is allowed but not by default -----
    ("overage-defended-by-default",
     "the manuscript runs over the journal's page guidance and the cover "
     "letter does not actually ask for the length - an overage that passes "
     "because nobody looked is the thing the gate exists to stop.  The "
     "manuscript IS over on a total-page reading, so the perturbation is the "
     "letter rather than the limit: take out the request and the build has to "
     "fail",
     "submission/JFEc-cover-letter.md",
     "I would be grateful for the editorial office's direction on how to handle the",
     "I hope the length will not be a difficulty for",
     ["build_jfec_submission.py"], "it does not ask for the length in terms"),

    ("length-claimed-on-the-flattering-count",
     "the letter states the count that looks compliant and not the one an "
     "editor will get by opening the file.  The guidance does not say floats "
     "are excluded from it, so quoting only the text-and-references figure "
     "claims compliance on a reading the journal has not granted.  The letter "
     "must state BOTH counts, and the gate now requires the larger one",
     "submission/JFEc-cover-letter.md",
     "The file is 45 pages, of which 40 pages are text and references",
     "The text and references are 40 pages",
     ["build_jfec_submission.py"], "it does not state both counts"),

    # ---- build_variants, which hid the same defect three times ------------
    # ---- the result tables, which were hand-maintained until 5 Oct -------
    ("table-cell-vs-lab",
     "a result-table cell drifts from the lab token it is a copy of.  These "
     "tables were maintained by hand, so every lab change staled about a "
     "hundred cells at once and they were patched one at a time - work that "
     "had to be redone on the next change.  papers/TABLE_MAP.tsv now says "
     "which lab token each mapped cell is, and build_tables.py asserts it",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information-internet-appendix.html",
     # a cell mapped one-to-one to a lab21 token, and unique in the file:
     # a perturbation that matches twice tests nothing, which this register
     # reports as STALE rather than passing
     "<td>0.0008</td>", "<td>0.0009</td>",
     ["build_tables.py", "--check"], "should be"),

    # ---- the calendar, which nothing in this project ever checked ---------
    ("trading-day-gate-removed",
     "an input row dated on a day its own exchange was shut reaches the "
     "estimator.  Every timing rule here compares one timestamp with another, "
     "so a Sunday row satisfied all of them: 30 Nikkei futures prints and one "
     "26%-wrong Hang Seng print sat in the panel for two years, each "
     "corrupting about a year of that target's decision variable through the "
     "trailing median",
     "labs/lab05_robustness.py",
     '    if tag not in WEEKEND_SESSIONS:\n'
     '        m = m.loc[m["date"].dt.dayofweek < 5].reset_index(drop=True)\n',
     "",
     ["verify_repo.py", "--data", "@DATA"],
     "trading-day gate is not being applied"),

    ("weekend-exemption-abused",
     "a market is added to WEEKEND_SESSIONS to make the trading-day gate stop "
     "complaining, rather than the bad rows being identified.  This is the "
     "likelier mistake of the two and the one a passing gate would hide, so "
     "the exemption list is itself checked against the signature test",
     "labs/lab05_robustness.py",
     'WEEKEND_SESSIONS = {"NSEI"}', 'WEEKEND_SESSIONS = {"NSEI", "N225"}',
     ["verify_repo.py", "--data", "@DATA"],
     "hides the defect instead of fixing it"),

    ("typed-figure-in-generated-prose",
     "a result figure is TYPED into the prose build_variants writes rather than "
     "read from a lab - the condensation carried 19.4 / 10.3 / 64 this way",
     "build_variants.py",
     "{HEADLINE} at eleven weeks and removes",
     "71% [54, 83] at eleven weeks and removes",
     ["build_variants.py", "jfec"], "TYPED into prose"),

    # ---- the figures the sweep could not see --------------------------------
    ("bare-decimal-with-nothing-behind-it",
     "a figure is written into prose as a bare decimal rather than a "
     "percentage.  The exhaustive sweep covered only figures carrying a "
     "per-cent sign, and this paper writes every skill level, every loss "
     "difference and every correlation without one, so the sweep that called "
     "itself exhaustive was watching a minority of its own figures: twenty-one "
     "came through the hole and most of them were stale",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "a loss of 99% of the skill it began with",
     "a loss of 0.98765 of the skill it began with",
     PAPER, "is neither demanded by a check nor listed as a non-result"),

    ("two-decimal-statistic-with-nothing-behind-it",
     "a Giacomini-White statistic is perturbed.  Statistics are written to TWO "
     "decimal places, and the sweep covers bare decimals only from three "
     "places up, because two-place decimals are mostly section numbers and "
     "equation constants.  So they are reached through a lead-in that says a "
     "statistic is coming - 'z = ', or 'statistics of/from/between' - which is "
     "what found three eleven-week statistics still carrying their "
     "pre-correction values",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information-internet-appendix.html",
     "the three largest statistics are the Nifty's +1.83",
     "the three largest statistics are the Nifty's +1.84",
     PAPER, "is neither demanded by a check nor listed as a non-result"),

    ("dead-allowlist-entry",
     "the sweep's allowlist keeps an entry for a figure that is no longer in "
     "any document.  An allowlist with unreachable entries is a place to hide "
     "a figure: the entry sits there as permission, and a genuine result that "
     "happens to print the same digits later inherits it silently.  Four such "
     "entries were found when this gate was added",
     "verify_paper.py",
     r're:        "2\.01": \(1, "the second of those three withdrawn statistics"\),',
     '        "2.01": (1, "the second of those three withdrawn statistics"),\n'
     '        "0.123456": (1, "a figure that is in no document"),',
     PAPER, "permission with nothing behind it"),

    # ---- an order that was load-bearing and undeclared ---------------------
    ("cross-lab-read-out-of-order",
     "a lab that reads a SIBLING's stored output is run before that sibling. "
     "Five labs do this, correctly, so that a figure belonging to lab06 does "
     "not get a second home in lab07 - but it makes run_all's ORDER part of "
     "the result, and nothing said so. A regeneration that ran lab07 first "
     "left it quoting lab06's pre-correction placebo figures, and the only "
     "report of it was verify_paper saying the two labs disagreed",
     "labs/lab07_estimation_cost.py",
     '"..", "expected_output", "lab06_inference.txt")',
     '"..", "expected_output", "lab73_own_options_paired.txt")',
     ["verify_repo.py"], "so a full rerun leaves"),

    # ---- a condensation that matched itself --------------------------------
    ("swap-matches-its-own-replacement",
     "a journal variant's paragraph swap whose needle matches a paragraph the "
     "SAME build inserted.  A condensation names the findings it condenses, so "
     "it carries the phrases the emptying swaps look for; when it lands earlier "
     "in the document than the source paragraph, the emptying swap deletes the "
     "condensation and leaves the long source paragraph in the submitted copy. "
     "That is what the jfec future-work entry did.  Every check stayed green, "
     "because the words were all still there, and the only symptom was the "
     "manuscript sitting one page over the journal's limit",
     "build_variants.py",
     '            ("The second turns the rate into money", ""),',
     '            ("what that share is worth in a decision", ""),',
     ["build_variants.py"], "matching a condensation"),

    # ---- the four classes the length-reduction passes created --------------
    #
    # All four shipped in a package that 2,980 checks had passed, and an outside
    # reader found them by reading the document.  They are one failure in four
    # coats: relocation changed what a sentence pointed at, and nothing was
    # reading the sentence.
    ("appendix-section-number-reused",
     "two appendix sections carry the same number.  The relocated sections "
     "counted from the last heading the build could SEE, and two sections "
     "added later were written as a bare <h2> while the pattern required "
     "class='pb' - so the appendix shipped with two S41s and two S42s, and "
     "the manuscript's three citations of 'Section S40' named neither of them "
     "in particular",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information-internet-appendix.html",
     '<h2 class="pb">S41. Whether any input is dated',
     '<h2 class="pb">S40. Whether any input is dated',
     ["build_variants.py"], "same number"),

    ("announced-count-delivers-nothing",
     "a paragraph announces a count and the paragraph that delivered it was "
     "relocated to the appendix.  'Two things bound what that costs here.' "
     "stood alone for three builds, immediately above a sentence beginning "
     "'MSEvar is therefore', with nothing between them",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "<p>Two things are disclosed rather than repaired. The first is where",
     "<p>Two things are disclosed rather than repaired.</p>\n<p>The first is where",
     PAPER, "announces a count and delivers nothing"),

    ("blanket-inference-claim-beside-its-exceptions",
     "a universal claim about which construction produced every interval, in "
     "a paper whose own Section 4.6 exempts three families from it.  The "
     "introduction made that claim two paragraphs under three bootstrap "
     "figures, and Section 4.5 made it again",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "Every interval this paper reports for a rate on the daily equity panel",
     "Every interval this paper reports for a rate",
     PAPER, "states no scope"),

    ("title-word-quoted-after-a-retitling",
     "the conclusion explains what a word 'in this title' means, and the "
     "title does not contain that word.  It survived three retitlings because "
     "a check REQUIRED the sentence, so the one gate that touched it was "
     "holding the defect in place",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "that, R(&delta;) remains a ratio on one ruler",
     "that, &ldquo;price&rdquo; in this title means a ratio on one ruler",
     PAPER, "and the title does not contain it"),

    # ---- a licensed input, published ---------------------------------------
    ("licensed-input-ships",
     "a file the manifest marks as not redistributable is in the tree and "
     "ships.  .gitignore was the only thing keeping the vendor CSVs out, and "
     "verify_repo could not even read an entry with a glob in the middle of a "
     "path - 'labs/*.csv' matched nothing, so an input dropped beside the labs "
     "to satisfy a lab's default search path was counted into the package.  The "
     "gate now tests the SHA-256 recorded in the manifest, so a renamed or "
     "moved copy fails too.  Here the one redistributable input is relabelled "
     "to prove the gate reads that column rather than a list of names",
     "data/MANIFEST.tsv",
     "illiquid/USSTHPI_fred.csv\tFRED / FHFA\tyes",
     "illiquid/USSTHPI_fred.csv\tFRED / FHFA\tno",
     REPO, "not redistributable"),

    # ---- an interval nobody computed ---------------------------------------
    ("interval-no-lab-produces",
     "an interval is printed that no lab output contains.  build_tables.py "
     "rewrites table CELLS from the labs and a table NOTE is prose, so six "
     "intervals reached the submitted PDF with endpoints nobody computed - the "
     "stronger control at [-9.7, -0.6] where the lab says [-10.6, -0.3], the "
     "stress contrast at [25.4, 55.8] where lab67 says [25.2, 55.8], the same "
     "contrast on the rate at [-7.3, 37.6] where lab67 says [-7.4, +37.5], and "
     "three more in the appendix.  Each was right to a tenth of a point, which "
     "is why reading did not find them.  The gate asks one thing of every "
     "interval in either document: some lab prints it",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "42.0 points [25.2, 55.8], estimated",
     "42.0 points [25.4, 55.8], estimated",
     PAPER, "no lab output contains it"),

    # ---- a table nobody was rewriting -------------------------------------
    ("hand-maintained-table-drifts",
     "an appendix table is not in papers/TABLE_MAP.tsv, so build_tables.py "
     "never rewrites it and it is maintained by hand.  Table S1 was one of "
     "eight such tables, and all seven of its Clark-West entries sat exactly "
     "0.01 above lab06's, with four more cells wrong across the other two "
     "rows - a whole statistics table that no lab output reproduced, in a "
     "paper whose claim is that every figure comes from a numbered script",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information-internet-appendix.html",
     '<tr class="botrule"><td>Clark-West</td><td>6.13</td>',
     '<tr class="botrule"><td>Clark-West</td><td>6.14</td>',
     PAPER, "and lab06 computes"),

    # ---- a table from a run that no longer exists --------------------------
    ("stale-table-contradicts-its-own-paragraph",
     "an unmapped table holds a run the lab no longer produces.  Table S15, the "
     "de-smoothed housing panel, carried twenty cells from an older lab57 - and "
     "one of them reversed a sign the paragraph above it depends on: the text "
     "says one month of staleness costs -0.0012 'the sign having gone the wrong "
     "way' while the table printed +0.0005.  Eight of the appendix's thirty-five "
     "tables are maintained this way; two of them had drifted",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information-internet-appendix.html",
     "<td>0.2230</td>", "<td>0.1951</td>",
     PAPER, "the appendix does not print it"),

    ("unmapped-table-cell-no-lab-prints",
     "a single result-table cell holds a figure no script produces.  "
     "derive_table_map.py keys only the rows it can key unambiguously, so "
     "roughly three cells in five are hand-maintained and covered only if "
     "somebody wrote a check naming that cell: Table S24's zero-delay "
     "cross-model R-squared read 0.7277 where lab54 prints 0.7359, in a table "
     "whose other nineteen cells matched to the fourth decimal.  The gate asks "
     "of every distinctive cell in either document what the interval gate asks "
     "of every interval: some lab prints it",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information-internet-appendix.html",
     "<td>0.7694</td><td>0.7359</td>", "<td>0.7694</td><td>0.7277</td>",
     PAPER, "Table S24 prints 0.7277"),

    # ---- the renumbering, which nothing was reading ------------------------
    ("appendix-points-at-a-renumbered-subsection",
     "a cross-reference survives a renumbering pointing two numbers away from "
     "the section it means.  _unnumber_intro shifted every section number down "
     "one, but any reference containing the digit 1 took a branch written for "
     "'Section 1' that returned the reference untouched - so a subsection "
     "ending .1 came back unshifted while .2 to .6 came back right, and the "
     "appendix sentence telling a reader where the estimator's constants are "
     "named Section 3.1 of a manuscript whose Methodology is Section 2.  "
     "verify_paper runs on the sources, where the numbering is the base "
     "paper's and every reference is correct by construction, so only a gate "
     "on the BUILT pair can see this",
     "build_variants.py",
     "                rebuilt = nums[0]",
     "                return lead + body\n                rebuilt = nums[0]",
     ["build_variants.py", "jfec"], "which the manuscript does not have"),

    ("mixed-reference-list-skipped-by-the-renumberer",
     "a reference that names a manuscript section and an appendix section in "
     "one list is skipped whole.  The appendix's renumberer bailed out "
     "whenever an S appeared anywhere in the reference, to avoid feeding an "
     "appendix number to a manuscript map - which protects the S-number by "
     "abandoning its neighbour: 'Sections 6.1 and S28' kept 6.1 in base "
     "numbering, and 6.1 is a subsection this variant RELOCATES",
     "build_variants.py",
     "        _refs_app = _refs",
     "        def _refs_app(m):\n"
     "            return m.group(0) if 'S' in m.group(2) else _refs(m)",
     ["build_variants.py", "jfec"], "which the manuscript does not have"),

    # ---- a result computed from data nobody can obtain ---------------------
    ("withdrawn-series-result-reported",
     "a figure computed from a series that can no longer be obtained is "
     "reported in the documents.  The two withdrawn targets' rates sat in "
     "Section S14 and their coupling in three other sentences, and the only "
     "check against it was a forbid on ONE phrasing, which three other "
     "phrasings walked past - including a figure that still labelled both of "
     "them as plotted points after their markers had been removed.  The gate "
     "is structural instead of lexical: a sentence that names a withdrawn "
     "target may not also carry a figure that reads as a measurement of it, "
     "so the documents can still say what was dropped and why and cannot "
     "quote what it returned",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information-internet-appendix.html",
     "The other two cannot be answered here, and the reason is not a result.",
     "The other two cannot be answered here. Pakistan returned &minus;31.1%.",
     PAPER, "also carries a figure"),

    ("withdrawn-series-coupling-quoted-without-a-name",
     "the same leak by a route the sentence gate cannot see: a coupling "
     "figure from the withdrawn series quoted without naming whose it is.  "
     "'the eight targets there differ in coupling from 0.81 to -0.11' names "
     "no target, so neither the forbid on one phrasing nor the "
     "names-and-figures gate saw it.  Coupling is a correlation and every one "
     "this study can re-run is positive, so the gate is the sign: a negative "
     "figure beside the word, outside a figure's own axis, is a figure from a "
     "series nobody can obtain",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "the six targets there differ in coupling from 0.612 to 0.808",
     "the eight targets there differ in coupling from 0.81 to &minus;0.11",
     PAPER, "a negative coupling figure is quoted"),

    # ---- a gate that preserved the error it was written to catch -----------
    ("cover-letter-misdescribes-the-papers-inference",
     "the cover letter tells the editor the paper's intervals are block "
     "bootstraps with Fieller as a check.  They were, when the gate for this "
     "was written - and the gate REQUIRED that phrase, so when Section 4.6 "
     "made the fixed-b Fieller inversion the reported instrument and kept the "
     "bootstrap in three named families, the check went on holding the letter "
     "to the superseded method.  The first thing an editor reads misdescribed "
     "the paper, and the check was what kept it there.  The gate now reads "
     "which instrument the manuscript calls the reported one and requires the "
     "letter to agree, so it fails in either direction",
     "submission/JFEc-cover-letter.md",
     "The reported sets therefore invert a HAC t-test after Fieller",
     "The intervals come from a moving-block bootstrap of the whole ratio, "
     "with Fieller",
     PAPER, "moving-block bootstrap"),

    # ---- an abstract nothing was reading --------------------------------
    ("abstract-coupling-count-goes-stale",
     "a journal variant's abstract orders the coupling result across a number "
     "of markets the study no longer scores.  The three variant abstracts are "
     "CONSTANTS in build_variants.py rather than text in the source, so every "
     "check that read the source abstract passed while one variant's own "
     "abstract still said 'eight equity markets' after the family had dropped "
     "to six.  The count is now read from lab53's scored targets, and because "
     "build_variants runs verify_paper against each variant, a variant's "
     "abstract is checked and not only the source's",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "That is an appraisal-like published index, not an appraised book.",
     "Coupling orders the rate across eight equity markets. That is an "
     "appraisal-like published index, not an appraised book.",
     PAPER, "equity markets; lab53 scores"),

    ("abstract-claims-a-non-detection-that-is-a-detection",
     "an abstract claims breadth adds nothing detectable on ANY of the "
     "own-options markets.  Appendix S40 reports the opposite on one of them: "
     "the N225 cell clears a statistic of two and breadth does not merely fail "
     "to help but costs 0.0427.  A significant negative is stronger evidence "
     "for redundancy than a null and is still a detection, so the blanket "
     "non-detection is an overclaim in the one paragraph an editor is certain "
     "to read.  Checked against lab73's own count of significant cells in that "
     "arm rather than against a phrase",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "adds nothing detectable on five of the six markets examined and\nsignificantly subtracts on the sixth",
     "adds nothing detectable on any of the six markets examined",
     PAPER, "adds nothing detectable on ANY"),

    # ---- the manuscript disagreeing with its own appendix ------------------
    ("body-contradicts-its-own-appendix-on-a-count",
     "Section 7 tells a reader breadth is insignificant given the target's own "
     "options 'just as it is given their own', and lab73 reports that arm "
     "significant on one of four: the Nikkei, where breadth does not merely "
     "fail to help but costs 0.0427.  Appendix S40 states that correctly, so "
     "the manuscript and its own appendix disagreed about a count and every "
     "figure-level check passed, because no figure was wrong - a count stated "
     "in words was.  Both arms are now read from lab73's verdict",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "given foreign implied volatility it is insignificant on four of four, and given the target's own it is insignificant on three and significantly <i>negative</i> on the fourth",
     "it is insignificant given foreign options in nought of four, just as it is given their own",
     PAPER, "own-options arm"),

    # ---- a design parameter in the code and in no document ----------------
    ("undisclosed-forward-fill-cap",
     "the panel carries a peer forward when its exchange was shut, and the "
     "appendix reported how OFTEN that happens and how old a carried close is "
     "on average but not the cap, which is the one number a replicator cannot "
     "infer from the other two.  The gate reads the limit out of "
     "lab05_robustness.py and demands the appendix state it, so it fires "
     "whether the prose drifts or the code changes",
     "labs/lab05_robustness.py",
     "ffill(limit=5)", "ffill(limit=7)",
     PAPER, "forward-fill cap"),

    # ---- a check that REQUIRED the discrepancy it should have caught ------
    ("check-enshrines-a-lab-paper-discrepancy",
     "the worst shape a check can take.  Section 6's two variance shares were "
     "verified by a PAIR OF LITERALS - 'first component 66.6%' in the lab "
     "against 'accounts for 66.5%' in the paper - so this package asserted "
     "that the lab printed one value and the document another and passed on a "
     "discrepancy of exactly one tenth, twice.  It did not fail to catch the "
     "error, it required it.  Both needles are now derived from the lab's own "
     "figure, so the paper has to follow the lab wherever the lab moves",
     "expected_output/lab22_factor_benchmark.txt",
     "first component 66.6%", "first component 65.1%",
     PAPER, "six-peer first component"),

    # ---- a second lab quoting the first by hand, and wrong ----------------
    ("second-lab-quotes-the-first-by-hand",
     "lab52 told the reader the seven-regressor gradient 'read -0.38' and "
     "lab51, the only script that computes it, prints -0.37.  The appendix "
     "followed lab52's copy, and the check paired the two literals, so a "
     "hand-copied figure wrong in the second decimal passed.  lab52 now reads "
     "lab51's stored output, the appendix needle is derived from lab51, and "
     "the echo is compared with it.  A figure copied between labs by hand is "
     "the most common defect in this repository",
     "expected_output/lab51_foreign_options.txt",
     "-0.37 on the levels", "-0.39 on the levels",
     PAPER, "gradient on the levels"),

    # ---- a re-run moved a headline and the sentences comparing with it did not
    ("rerun-moves-a-headline-quoted-in-a-comparison",
     "Section 8's monthly rate was re-run on the Case-Shiller vintage that can "
     "still be obtained and moved from 67.8% to 67.9%.  Four appendix "
     "sentences compare something AGAINST that headline, and three of them "
     "kept the old value, so the same figure stood at two values one paragraph "
     "apart - and the one check on the worst of them carried the old literal, "
     "so a gate written to prove the paper reports a gap was the thing pinning "
     "the stale figure.  All four now read lab55's return-target row",
     "expected_output/lab55_illiquid_measured.txt",
     "      3     0.3506       0.5953     67.9%",
     "      3     0.3506       0.5953     68.4%",
     PAPER, "S18 the magnitude gap"),

    # ---- a lab documenting a SIBLING's figure by hand ----------------------
    ("lab-docstring-states-a-siblings-figure",
     "three shipped labs summarised a sibling's result with the figures typed "
     "into their docstrings, and all three had gone stale - six numbers for "
     "lab33's filter comparison in lab50, two for lab38's LEVEL column in "
     "lab40, two standard errors and a percentage for lab58's bandwidth table "
     "in lab65.  No document quoted them, so no document check could see "
     "them, and what shipped was a repository whose own documentation "
     "disagreed with the labs it imports.  A figure of four or more decimals "
     "in a module docstring must now appear in some lab's output; three "
     "decimals and fewer are left alone so the gate cannot argue about "
     "rounding",
     "labs/lab50_filtered_headline.py",
     "A docstring cannot read a file, so it names no figure at",
     "It beat the paper by +0.0124 [+0.00114, +0.02192] at 21.  It names none at",
     ["verify_repo.py"], "which no lab output prints"),

    # ---- a bound measured on the specification the paper declines ----------
    ("bound-measured-on-the-declined-arm",
     "Section 7's redundancy claim is a BOUND - it rules out a gross gain "
     "larger than some figure and leaves anything smaller open - and that "
     "figure was measured on lab08's same-day arm, the specification the paper "
     "calls inadmissible, while being presented beside the admissible table.  "
     "The arithmetic did not close: the strict net is -0.0103 at zero delay "
     "against a printed cost of -0.0107, which gives +0.0004 and not the "
     "+0.0016 the appendix quoted.  lab08 now decomposes BOTH arms, the "
     "admissible one first, and the bound is read from it: it widens from "
     "0.019 to 0.022, so the correction weakens the paper's own claim.  The "
     "gate also counts the parsed rows, because the two arms carry the same "
     "four delays and a dict keyed on delay had silently kept the last arm's",
     "expected_output/lab08_implied_vol.txt",
     "    55      -0.0095   -0.0175   +0.0080     [-0.0067,+0.0218]",
     "    55      -0.0095   -0.0175   +0.0080     [-0.0067,+0.0244]",
     PAPER, "equivalence bound"),

    # ---- a result table made of ANOTHER table's numbers --------------------
    ("one-lab-row-feeds-two-tables-in-one-document",
     "Table S32's four calm rows carried Table S31's numbers - lab69's CC, RS "
     "and PK rows pasted into the coverage table, and lab70's calm rows into "
     "Table S33 beneath it - so two result tables in the submitted appendix "
     "were made of another table's measurements.  Every cell was a figure some "
     "lab prints, so the cell-provenance gate passed on all of them; every row "
     "was rectangular; and the prose around both tables was correct, which is "
     "why reading it did not reveal the tables.  TABLE_MAP held the evidence "
     "all along: rows mapped to lab69 inside a table whose other rows mapped "
     "to lab70.  A lab source row feeding two tables of ONE document is now a "
     "failure, since that is what a paste produces; across documents it stays "
     "allowed, because the appendix reproduces the headline table",
     "papers/TABLE_MAP.tsv",
     "breadth-implied-volatility-and-forecasting-with-stale-information-internet-appendix.html\t31\t2\t5\tlab69_proxy_robustness.txt\t8\tCC#0\t5\t{0}",
     "breadth-implied-volatility-and-forecasting-with-stale-information-internet-appendix.html\t32\t2\t5\tlab69_proxy_robustness.txt\t8\tCC#0\t5\t{0}",
     ["build_tables.py", "--check"], "DUPLICATE SOURCE"),

    # ---- two consecutive paragraphs saying the same thing -----------------
    ("adjacent-paragraphs-restate-each-other",
     "Section 4.4 said 'the benchmark carries the same correction on the same "
     "schedule, so a skill score differences most of it away' and then opened "
     "the NEXT paragraph with 'The correction is applied to the benchmark on "
     "the same schedule, so a skill score differences most of it away'; "
     "Section S16 priced the median's drift twice in two consecutive "
     "paragraphs.  Neither is a repeated word, so the doubled-word check could "
     "not see them, and both cost page-limit lines in a manuscript that pays "
     "for every addition with a cut.  ADJACENT is the whole rule and is what "
     "keeps it from firing on a correct document: 22 pairs of NON-adjacent "
     "paragraphs share a "
     "ten-word run, nearly all a limitations or roadmap section restating a "
     "result on purpose",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "<p>The correction is applied to the benchmark on the same schedule,",
     "<p>The benchmark carries the same correction on the same schedule, so a "
     "skill score differences most of it away.</p>\n\n<p>The correction is "
     "applied to the benchmark on the same schedule,",
     PAPER, "two consecutive paragraphs share a ten-word run"),

    # ---- the introduction asserting what the design treats as an exception -
    ("intro-overstates-the-clock",
     "Section 1 said 'on any given day the rest of the world has already "
     "finished trading when New York does'.  CLOSE_UTC puts BVSP at 21:00, "
     "the same hour as SPX, which is why Section 3's rule sends it back a "
     "session and Section 4.3 calls it 'Bovespa a session late'.  So the "
     "introduction asserted as a fact about the design the one case the design "
     "handles as an exception, in the argument the clock rule rests on.  The "
     "count of markets finishing before the target is read from the code",
     "labs/lab05_robustness.py",
     '"BVSP": 21.0', '"BVSP": 20.0',
     PAPER, "counts the markets that finish before the target"),

    # ---- a caption claiming one convention for rows computed on two --------
    ("table-caption-claims-one-convention-for-two",
     "Table 1's caption said 'all are the implementable origin-dated figures "
     "of Section 4.2'.  lab23's Part C states in terms that it computes the "
     "TARGET-dated convention and it produces two of those rows, as lab49 and "
     "lab50 do for two more; only the headline row is origin-dated.  The band "
     "and Section 1's sentence are both read off these rows, so the caption "
     "claimed six figures sit on one ruler when four sit on the other",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "the four comparison rows are on the target-dated convention",
     "all are the implementable origin-dated figures",
     PAPER, "claims one convention for two"),

    # ---- a citation standing where a predicate should be -------------------
    ("citation-replaces-a-missing-predicate",
     "Section 4.1 read 'because the Rogers-Satchell term is (Rogers and "
     "Satchell 1991) and the other two are demeaned': the words the citation "
     "was meant to follow had been lost.  Every check in verify_paper reads a "
     "number and the doubled-word check reads pairs of identical words, so "
     "neither could see a sentence with its predicate missing.  A copula "
     "immediately followed by a parenthesised author and year occurs nowhere "
     "in these documents when the prose is right, so the rule cannot cry wolf",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "the Rogers-Satchell term is itself drift-independent (Rogers and Satchell 1991)",
     "the Rogers-Satchell term is (Rogers and Satchell 1991)",
     PAPER, "a citation stands where a predicate should be"),

    # ---- a conditional cut described as real-time that is not --------------
    ("conditional-cut-called-real-time",
     "the paper said the VIX tercile was cut 'in real time on a rolling VIX "
     "tercile' in five places.  lab41 takes the LEVEL at t-1 and the "
     "BOUNDARIES from percentiles of the whole test window, so the level is "
     "real-time and the boundary is not.  Part E now runs the trailing-year "
     "boundary on identical forecasts and losses: the stressed rate at eleven "
     "weeks reads 71% against 81%, the middle 1% against 53% and the calm 84% "
     "against 47%, so the ordering the stressed column rests on inverts under "
     "the cut the paper claimed.  No document may call the cut rolling or "
     "real-time while lab41 takes its boundaries from the whole window",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "Conditioning on a VIX tercile, the level taken at t",
     "Conditioning in real time on a rolling VIX tercile, the level taken at t",
     PAPER, "called real-time"),

    # ---- a selection whose stated criterion selects nothing ----------------
    ("exclusion-criterion-that-excludes-nothing",
     "lab42 fitted the decay line over eight of ten delays under the comment "
     "'delays where S_own is positive, so log exists', and S_own is positive "
     "at all ten: the criterion excludes nothing, while the exclusion of "
     "delta = 55 is what carries the result.  On all ten the line fits worse, "
     "R2 0.9692 against 0.9980, and lands CLOSER to the factor's persistence.  "
     "The real reason is leverage - 1/S at S_own(55) = 0.0072 is seventy times "
     "the first point's - and both fits are reported now, with the appendix "
     "demanding each",
     "expected_output/lab42_factor_model.txt",
     "    slope = -0.07196 per day, fit R-squared = 0.9692",
     "    slope = -0.07196 per day, fit R-squared = 0.9881",
     PAPER, "all-ten fit"),

    # ---- a lab asserting a conclusion the paper has withdrawn -------------
    ("lab-asserts-a-retracted-prediction",
     "lab42 counted '4 of 4 predictions hold' and its verdict credited the "
     "model with predicting a ceiling 'from two independently measured "
     "quantities'.  Section S26 retracts exactly that: once S_own(55) has "
     "gone, the ceiling and the measured rate are the same two numbers and "
     "cannot disagree.  The shipped replication package was asserting a "
     "confirmation the manuscript withdraws, which is what a referee running "
     "the code would read first",
     "expected_output/lab42_factor_model.txt",
     "3 of 3 testable predictions hold", "4 of 4 predictions hold",
     PAPER, "counting the ceiling as a prediction"),

    # ---- a bootstrap resampling shorter than the measured dependence ------
    ("bootstrap-block-shorter-than-the-dependence",
     "lab34's Reality Check and SPA tests resampled at a mean block of ten "
     "days while computing their standard errors at L.block_for(delay), forty "
     "to eighty.  Ten days is 2*HORIZON, the length lab05's own note records "
     "as measured wrong: the loss differences carry 0.48 autocorrelation at "
     "lag 10 and the coverage study finds 86% for a nominal 95% there.  A "
     "block shorter than the dependence tightens the null and LOWERS a "
     "p-value, on the test that licenses the headline.  Both lengths are run "
     "now, the measured one is quoted, and the conclusion survives either - "
     "which is the point of measuring rather than assuming",
     "expected_output/lab34_data_snooping.txt",
     "    21           3.99         3.29e-05              0.0010",
     "    21           3.99         3.29e-05            < 0.0005",
     PAPER, "what the short block changed"),

    # ---- a point estimate and its interval on different scales ------------
    ("interval-on-a-different-scale-from-its-point",
     "lab35 printed the purge inflation as a difference of two R-squareds and "
     "its interval as percentiles of the raw mean squared-error differential.  "
     "The two scales differ by n/sum((y-bench)^2), 1.056 on this panel, so "
     "Section S7 quoted a point and an interval measured differently and the "
     "interval understated the inflation by six per cent at both endpoints.  "
     "boot_ci now scales to R-squared units the way lab05 does, and takes the "
     "benchmark as a required argument so a missed call site raises",
     "expected_output/lab35_purged_cv.txt",
     "     0              +0.0023   [ +0.00072, +0.00400]",
     "     0              +0.0023   [ +0.00068, +0.00379]",
     PAPER, "purge load-bearing"),

    ("grid-claims-an-audit-that-does-not-cover-it",
     "a table caption says its cells are part of the Section S37 "
     "post-selection pool when lab68, which computes that pool, does not list "
     "the lab filling the table among its SOURCES.  Table S23's twelve cells "
     "carried the claim for six review rounds: the grid is the compressed "
     "mean-and-factor comparison, priced in Section S6 with the other eight "
     "specifications, and no cell of it was ever in the 198.  Every figure in "
     "the table was right, which is why 4,446 checks saw nothing - the defect "
     "is a claim about which audit covers the grid, not about a number",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information-internet-appendix.html",
     "Both arms are among the nine specifications",
     "Cells in this grid are part of the post-selection audit of Section S37; "
     "both arms are among the nine specifications",
     PAPER, "lab68 does not"),

    ("audit-enumeration-credits-the-wrong-table",
     "Section S37 enumerates the grids in its pool and credits a term to a "
     "table that does not hold it.  The largest term, 64 cells, was credited "
     "to Table S23, which holds twelve, while those cells are lab52's "
     "eight-market grid in Section S13.  The enumeration's arithmetic had "
     "just been corrected from 102 to 198 and its sum is gated; the pointers "
     "were not, so a term could sum correctly and name the wrong grid",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information-internet-appendix.html",
     "for the compressed foreign block of Section S13",
     "for the compressed foreign block in Table S23",
     PAPER, "enumeration credits a term"),

    ("agreement-measured-over-the-rows-the-threshold-kept",
     "a lab reports how closely a set of rows agrees, having chosen that set "
     "with the same tolerance the agreement is then measured against.  lab14 "
     "collected the rows whose gap was inside eight points and reported that "
     "they agreed to within seven, so a row that disagreed was removed from "
     "the set whose agreement was being reported and the sentence could not "
     "have come out any other way.  The stated reason for dropping the one "
     "excluded row is a property of its COMPARATOR, which is measurable "
     "without looking at the gap, and that is what the lab selects on now",
     "expected_output/lab14_appraisal_smoothing.txt",
     "rows whose comparator has a usable denominator agree to within",
     "rows with real staleness agree to within",
     PAPER, "circular"),

    ("joint-band-resampled-and-studentised-at-different-lengths",
     "a simultaneous band resamples the delay vector in blocks of one length "
     "and studentises each delay by a HAC standard error of another.  A block "
     "of length b destroys dependence past lag b exactly as a Bartlett kernel "
     "at bandwidth b does, so two lengths answer two questions and the "
     "critical value belongs to neither; lab06 resampled at block_for(0) "
     "while studentising at block_for(delta), and the shorter length was the "
     "one the published critical value came from",
     "expected_output/lab06_inference.txt",
     "HAC standard error at the same 80 lags",
     "HAC standard error at the same 40 lags",
     PAPER, "answer different ones"),

    ("comment-states-a-figure-no-lab-prints",
     "a lab comment names a four-decimal figure as the current value of "
     "something a lab computes.  Three such figures went stale at once, every "
     "one of them inside a comment recording an earlier fix of this same "
     "defect: a comment is not re-recorded by a rerun, so the note "
     "written to prevent the error becomes the error.  The gate rounds the "
     "PRINTED figure to the comment's precision rather than rounding the "
     "comment's figure down until something matches, and the figures a "
     "comment is entitled to name are listed one by one with their reasons",
     "labs/lab14_appraisal_smoothing.py",
     "The figures are deliberately not repeated in",
     "The blur arm runs 0.5056 to 0.4782.  The figures are not repeated in",
     REPO, "no lab output prints it"),

    ("appendix-pointer-sends-the-reader-to-the-wrong-subject",
     "a pointer into the Internet Appendix resolves to a section that exists "
     "and is about something else.  The cover letters have been checked this "
     "way since one of them cited the volatility-proxy section for the two "
     "withdrawn series; the documents went unchecked for longer.  The gate "
     "reads the subject out of the "
     "citing CLAUSE - not the sentence, which in a paper this long names "
     "eight indices at once - and requires the distinctive proper nouns in it "
     "to appear in the section cited.  It found two on its first run: a "
     "clause left unparseable by a length-reduction edit, and a geographic "
     "label the cited section does not use",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "with the same Bartlett kernel used in Section S2, because",
     "with the same Bartlett kernel used in Section S20, because",
     PAPER, "sends the reader to the wrong subject"),

    ("a-margin-quoted-as-absent-when-the-lab-finds-it-present",
     "a section says an interval does not exclude zero and the lab it comes "
     "from says it does.  Section S20 rests the weaker of its two readings on "
     "the live-parameter interval spanning zero, which is the sentence a "
     "referee would check first, so the sign of that set is read out of the "
     "lab rather than taken from the prose around it",
     "expected_output/lab75_filter_parameter_uncertainty.txt",
     "at eleven weeks: +0.0059 [-0.0124, +0.0271]",
     "at eleven weeks: +0.0059 [+0.0024, +0.0271]",
     PAPER, "now excludes zero"),

    ("frozen-and-live-arms-compared-across-different-panels",
     "two columns of the same comparison are quoted side by side while one of "
     "them was computed on another panel.  Section S20 prints 75.3% frozen "
     "beside 72.6% live and the whole paragraph is the DIFFERENCE between "
     "them, so the frozen column of the live-parameter file has to reproduce "
     "the frozen file's own figure; if it does not, the two numbers are not "
     "comparable and the paragraph means nothing",
     "expected_output/lab75_filter_parameter_uncertainty.txt",
     "71.5% as published, 75.3% with the frozen filter",
     "71.5% as published, 74.1% with the frozen filter",
     PAPER, "not comparable with the figure"),

    ("a-post-closure-arm-joins-the-closed-multiplicity-pool",
     "a specification run after the post-selection pool was enumerated is "
     "added to that pool.  The pool is a family fixed before the outcomes are "
     "read, so a count that grows once the results are known is the defect "
     "the audit exists to prevent - and it grows in the direction that helps, "
     "because a new arm raises the null mean against which the observed count "
     "is judged.  The gate reads lab68's own declared family",
     "labs/lab68_joint_multiplicity.py",
     '"lab53_no_options_targets")',
     '"lab53_no_options_targets",\n           '
     '"lab75_filter_parameter_uncertainty")',
     PAPER, "cannot join it"),

    ("punctuation-a-condensation-left-behind",
     "a built document carries a doubled full stop, a doubled comma or a "
     "space before one.  The condensed introduction shipped 'couples between "
     "0.612 and 0.808..' to the journal: a figure pattern ending in a "
     "character class that contains the full stop had swallowed the source "
     "sentence's own period, and the replacement supplied another.  No figure "
     "check could see it, because both documents state the same number and "
     "the defect is the punctuation around it.  The gate reads the text as a "
     "reader sees it, so tags are stripped without a space in their place",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information-internet-appendix.html",
     "is the number to read.</p>", "is the number to read..</p>",
     PAPER, "a doubled full stop"),

    ("abstract-names-a-different-cut-from-the-paper",
     "the condensed abstract names a different market-state cut from the one "
     "the manuscript names beside the same figure.  Reading the boundary from "
     "a lab makes it current and does not make it right: the lab prints a "
     "LADDER of rungs, and the first pattern written over it matched the top "
     "rung rather than the one the contrast was measured on, so a figure "
     "measured on the top 5% of days was about to be published as the top "
     "33%.  The abstract's cut is pinned to the manuscript's own sentence",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-information.html",
     "higher in the top 5% of VIX days than in the bottom 67%",
     "higher in the top 10% of VIX days than in the bottom 67%",
     VARIANTS, "where the manuscript names"),

    ("availability-statement-promises-coverage-the-repository-lacks",
     "the data-and-code availability statement describes an index of the "
     "figures as more complete than it is.  It read 'a registry naming, for "
     "EVERY figure quoted in either document, the script that produces it', "
     "and thirty-six table cells were in neither that registry nor the cell "
     "map - a compliance claim nobody had measured, in the one paragraph a "
     "replicator reads first.  The gate holds the sentence and the measured "
     "coverage together in BOTH directions, so closing the gap later forces "
     "the sentence to be rewritten the other way rather than left as an "
     "understatement",
     "papers/source/breadth-implied-volatility-and-forecasting-with-stale-"
     "information.html",
     "Between them they do not reach every cell of every table",
     "Between them they reach every cell of every table",
     REGISTRY, "does not say the indexes are partial"),

    ("editor-note-counts-inputs-differently-from-the-manifest",
     "a document supplied to the editor states a number of input files, or a "
     "per-source count, that the manifest does not have.  The data-access "
     "note opened 'Forty input files' and gave Yahoo five of them where the "
     "manifest has thirty-eight and three: the counts were written while the "
     "two withdrawn targets were still in, and nothing read this file, "
     "because it is a note rather than a letter.  It is the one document "
     "whose whole purpose is to be exact about the inputs, and it sat two "
     "files wrong beside a cover letter with the right number",
     "submission/JFEc-data-access-note.md",
     "| Yahoo Finance | 3 |", "| Yahoo Finance | 5 |",
     LETTERS, "the manifest has"),
]

def run(argv, cwd):
    r = subprocess.run([sys.executable] + [argv[0]] + argv[1:],
                       capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout + r.stderr


def main(argv):
    # Two cases need the vendor data, which does not ship.  They take the folder
    # from --data and SKIP without it, which is the same path the cover-letter
    # cases already use: a case whose input is deliberately absent is untestable
    # here, not broken.  The alternative - baking a path in - is the defect class
    # this register already covers.
    data = None
    if "--data" in argv:
        i = argv.index("--data")
        if i + 1 < len(argv):
            data = argv[i + 1]
    picks = [a for a in argv[1:]
             if not a.startswith("-") and a != data]
    cases = [c for c in CASES if not picks or any(p in c[0] for p in picks)]
    if data is None:
        cases = [c for c in cases if "@DATA" not in c[5]]
        _nodata = [c[0] for c in CASES if "@DATA" in c[5]]
    else:
        cases = [(n, w, r, f, rp, [data if g == "@DATA" else g for g in gate], e)
                 for (n, w, r, f, rp, gate, e) in cases]
        _nodata = []

    if "--list" in argv:
        print(f"{len(CASES)} defect classes in the register:\n")
        for name, what, *_ in CASES:
            print(f"  {name}\n      {what}\n")
        return 0

    tmp = tempfile.mkdtemp(prefix="selftest-")
    root = os.path.join(tmp, "repo")
    try:
        # the shipping set, plus the two files it needs that do not ship
        man = subprocess.run([sys.executable, os.path.join(HERE, "verify_repo.py"),
                              "--manifest"], capture_output=True, text=True, cwd=HERE)
        files = [l.strip() for l in man.stdout.splitlines() if l.strip()]
        for extra in ("contact.txt",):
            if os.path.isfile(os.path.join(HERE, extra)):
                files.append(extra)
        for rel in files:
            dst = os.path.join(root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(HERE, rel), dst)
        # submission/ and papers/variants/ are deliberately absent from a
        # clone - correspondence and generated output.  The first version of
        # this file copied them unconditionally and died with FileNotFoundError
        # in a clone, which is precisely the defect class the register's own
        # clone test exists for.  A case whose inputs are not present SKIPS and
        # says so; it does not pass, and it does not take the run down.
        for sub in (("submission",), ("papers", "variants")):
            src = os.path.join(HERE, *sub)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(root, *sub),
                                dirs_exist_ok=True)

        print(f"testing {len(cases)} defect classes against a copy of the "
              f"shipping set\n")
        # A gate that does not fire is BLIND and is a failure.  A case whose
        # INPUT is deliberately absent - the cover letters and the generated
        # variants are not in a clone - is SKIPPED, which is not a failure and
        # must not be reported as one, or a reader running this from a clone is
        # told two gates are broken when they are merely untestable there.
        fired, blind, skipped = 0, [], []
        for name, what, rel, find, repl, gate, expect in cases:
            path = os.path.join(root, rel)
            if not os.path.isfile(path):
                skipped.append((name, f"{rel} is not present here"))
                print(f"  skip   {name}  ({rel} not in this copy)")
                continue
            orig = open(path, encoding="utf-8").read()
            # A case may name its target as a REGEX, written "re:<pattern>".
            # Two cases have now gone stale because they pinned a number that
            # legitimately moves - a check count, then a lab count - and a
            # case that cannot find its target tests nothing.  With a regex the
            # case says "whatever this count currently is, break it", which is
            # what it meant all along.
            if find.startswith("re:"):
                _m = list(re.finditer(find[3:], orig))
                if len(_m) != 1:
                    skipped.append((name, f"the pattern matches {len(_m)} "
                                          f"places in {rel}, not one"))
                    print(f"  skip   {name}  (pattern matches {len(_m)})")
                    continue
                find = _m[0].group(0)
            if orig.count(find) != 1:
                blind.append((name, f"the perturbation no longer matches {rel} "
                                    f"({orig.count(find)} hits) - the case is "
                                    f"stale, not the gate"))
                print(f"  STALE {name}: perturbation does not match any more")
                continue
            try:
                open(path, "w", encoding="utf-8").write(orig.replace(find, repl, 1))
                code, out = run(gate, root)
                ok = code != 0 and expect in out
                if ok:
                    fired += 1
                    print(f"  fires  {name}")
                else:
                    blind.append((name, f"gate exited {code} and did not say "
                                        f"{expect!r}"))
                    print(f"  BLIND  {name}  <-- the defect was reintroduced and "
                          f"nothing caught it")
            finally:
                open(path, "w", encoding="utf-8").write(orig)

        testable = len(cases) - len(skipped)
        print(f"\n  {fired} of {testable} testable gates fire"
              + (f"   ({len(skipped)} skipped: inputs not present in this copy)"
                 if skipped else ""))
        for n in _nodata:
            print(f"    not run  {n}: needs the vendor data; pass --data <folder>")
        for n, why in blind:
            print(f"    BLIND  {n}: {why}")
        for n, why in skipped:
            print(f"    skipped  {n}: {why}")
        if not blind:
            print("  every testable defect class in the register is covered by "
                  "a gate that still works")
        return 1 if blind else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
