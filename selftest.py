"""selftest.py - do the gates actually fire?  One command, one verdict.

    python selftest.py              run every case
    python selftest.py --list       show the register without running it
    python selftest.py bootstrap    run the cases whose name matches

WHY THIS FILE EXISTS
--------------------
Every verifier in this repository passes.  That is not evidence of anything on
its own: a check that reads the wrong surface, or compares a thing to itself,
or was written against a sentence that has since been reworded, passes exactly
like a check that works.  This project has found four of those, and each one
had been passing for weeks.

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
         "papers/source/what-substitutes-for-a-stale-mark.html",
         "papers/source/why-a-simple-threshold-rule-is-hard-to-beat.html"]
REPO = ["verify_repo.py"]
LETTERS = ["verify_letters.py"]

CASES = [
    # ---- the paper against the labs ---------------------------------------
    ("figure-vs-lab",
     "a figure in the manuscript no longer matches the lab that produces it",
     "papers/source/what-substitutes-for-a-stale-mark.html",
     "that share is 42.0 points higher", "that share is 41.0 points higher",
     PAPER, "42.0 points higher"),

    ("band-excludes-its-own-row",
     "the specification band is quoted at a precision that excludes one of the "
     "rows it summarises - round(75.4) = 75 certified a band Table 1 contradicts",
     "papers/source/what-substitutes-for-a-stale-mark.html",
     "all landing between 67.4% and 75.4%,",
     "all landing between 67% and 75%,",
     PAPER, "all landing between 67.4% and 75.4%"),

    ("ai-disclosure-weakened",
     "the AI disclosure is trimmed back to the generic form, dropping what the "
     "tool did NOT produce and how a reader can check it",
     # the disclosure is injected by build_variants into the JOURNAL copies, so
     # it is not in papers/source/ at all - the first version of this case
     # perturbed the source and reported STALE, which is the register working:
     # a case that no longer reaches its target says so instead of passing.
     "papers/variants/jfec/what-substitutes-for-a-stale-mark.html",
     # A short fragment, because the paragraph is rewrapped whenever a word
     # changes and a one-line needle then matches nothing and reports STALE.
     "produced by that tool",
     "produced somehow",
     ["verify_paper.py", "papers/variants/jfec/what-substitutes-for-a-stale-mark.html",
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
     "the largest being 68", "the largest being 64",
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
     "submission/JFEc-cover-letter.md",
     r"re:all \d+ analysis scripts", "all 3 analysis scripts",
     LETTERS, "analysis scripts"),

    ("letter-cross-reference-off-by-one",
     "a cover letter sends the editor to an appendix section that exists and "
     "is about something else.  The letter cited S39 for the two withdrawn "
     "series, which are in S38; S39 is the volatility proxy.  Every check in "
     "place passed, because each one asked whether a number was right and "
     "none asked whether a POINTER was - an existence check would have passed "
     "too.  The gate now reads the subject out of the citing sentence and "
     "requires the cited section to carry it",
     "submission/JFEc-cover-letter.md",
     "Section S38 of the", "Section S39 of the",
     LETTERS, "which that section does not mention"),

    # ---- the page-limit overage, which is allowed but not by default -----
    ("overage-defended-by-default",
     "the manuscript runs over the journal's page guidance and the cover "
     "letter does not actually ask for the length - an overage that passes "
     "because nobody looked is the thing the gate exists to stop.  The "
     "manuscript is now inside the guidance and the letter asks for nothing, "
     "which is the right state and an untestable one, so the perturbation "
     "tightens the guidance instead: given ANY overage, a letter that does not "
     "ask for the length in terms has to fail",
     "build_jfec_submission.py",
     "re:\\nLIMIT = \\d+",
     "\nLIMIT = 30",
     ["build_jfec_submission.py"], "defend the length deliberately"),

    # ---- build_variants, which hid the same defect three times ------------
    # ---- the result tables, which were hand-maintained until 5 Oct -------
    ("table-cell-vs-lab",
     "a result-table cell drifts from the lab token it is a copy of.  These "
     "tables were maintained by hand, so every lab change staled about a "
     "hundred cells at once and they were patched one at a time - work that "
     "had to be redone on the next change.  papers/TABLE_MAP.tsv now says "
     "which lab token each mapped cell is, and build_tables.py asserts it",
     "papers/source/stale-mark-internet-appendix.html",
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
     "{HEADLINE} of eleven weeks' staleness and removes",
     "71% [54, 83] of eleven weeks' staleness and removes",
     ["build_variants.py", "jfec"], "TYPED into prose"),

    # ---- the figures the sweep could not see --------------------------------
    ("bare-decimal-with-nothing-behind-it",
     "a figure is written into prose as a bare decimal rather than a "
     "percentage.  The exhaustive sweep covered only figures carrying a "
     "per-cent sign, and this paper writes every skill level, every loss "
     "difference and every correlation without one, so the sweep that called "
     "itself exhaustive was watching a minority of its own figures: twenty-one "
     "came through the hole and most of them were stale",
     "papers/source/what-substitutes-for-a-stale-mark.html",
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
     "papers/source/stale-mark-internet-appendix.html",
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
     're:        "2\.01": \(1, "the second of those three withdrawn statistics"\),',
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
