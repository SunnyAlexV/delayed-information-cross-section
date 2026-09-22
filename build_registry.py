"""
build_registry.py - the results registry, and the four build gates over it.

Run it with no arguments:

    python build_registry.py            builds results_registry.tsv and checks it
    python build_registry.py --list     also prints every unlinked literal

WHY THIS EXISTS
---------------
Every error an external audit found in this paper was the same error: a number
was copied rather than referenced, the original moved, and the copy went on
agreeing with a run that no longer existed.  The headline interval sat at
[63, 79] in the abstract and the introduction while Table 2 said [59, 81].  The
effective age was quoted three times at two different intervals.  A whole
interval column of one table was a bandwidth behind the paragraph underneath it.
In each case a check existed and passed, because a check that a figure appears
SOMEWHERE is satisfied by whichever copy happens to be current.

The registry closes that by making every quoted number an entry with a name.
It is not a second copy of the results, which would be one more thing to drift:
it is BUILT from the verifier's own checks, so a number is in the registry
exactly when something ties it to a lab, and the coverage gate below is a
measurement of how much of the papers that is.

THE FIVE GATES
--------------
1. CONFLICTING INTERVALS.  A point estimate quoted with two different intervals
   is an error unless the two are different quantities that share a digit
   string.  Those exceptions are declared in EXEMPT below, with a reason, so
   the list of them is short, visible and reviewable rather than implicit.

2. PROSE P-VALUES AGAINST STORED STATISTICS.  A p-value written beside a z in
   the text must be the p-value of that z.  The paragraph under Table S1 said
   z = 2.57 gives p = 0.0090; it gives 0.0102, and the 2.57 belonged to a
   different delay.

3. REVERSED CONTRAST SIGNS.  A sentence that names a winner must agree in sign
   with the number beside it.  "In the threshold rule's favour" beside a
   negative gap is correct only if the contrast is stated in the direction that
   makes it so, which is why the direction must be stated at all.

4. UNLINKED LITERALS.  Any distinctive number in either document that no check
   ties to a lab.  These are not errors; they are unguarded surface, and the
   gate is a ratchet: the count may fall and may not rise.

5. NEGATIVE SIGNS THROUGH THE PDF.  A minus that does not survive rendering and
   extraction turns a figure into a different figure.  Two reviewers in a row
   read the threshold note's delta = 55 correlation as positive and called the
   benchmark beside it impossible; it is negative.
"""

import os
import re
import subprocess
import sys
from math import erfc, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, "results_registry.tsv")
CEIL = os.path.join(HERE, "registry_ceiling.txt")

# Point estimates that legitimately carry more than one interval, because the
# same digits name different quantities.  Each needs a reason: an entry here is
# a claim that two identical numbers are unrelated, and that claim is checkable
# by reading the two sites.
# Point estimates that legitimately carry more than one interval in PROSE,
# because the same digits name different quantities in two sentences.  The list
# is empty, and that is the finding rather than an oversight: once table rows
# are excluded, no figure in these documents is quoted twice in prose with two
# different intervals.  An earlier version of this file exempted 71%, 72% and
# 67% because they collided across TABLE rows, and those exemptions then blinded
# the gate to the real headline conflict when it was reintroduced as a test.
# An exemption written against the wrong filter is worse than none: it reads as
# diligence while switching the check off.  Anything added here needs a reason
# that survives that test.
EXEMPT = {}


def two_sided_p(z):
    return erfc(abs(z) / sqrt(2.0))


def plainer(path):
    """The document as one line of tag-free text, dashes normalised."""
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    txt = " ".join(re.sub(r"<[^>]+>", " ", raw).split())
    for a, b in (("−", "-"), ("&minus;", "-"), ("–", "-"),
                 ("—", "-"), ("&mdash;", "-"), ("&ndash;", "-"),
                 ("&nbsp;", " "), ("&amp;", "&"), (" ", " ")):
        txt = txt.replace(a, b)
    return " ".join(txt.split())


def dump_needles():
    """Run the verifier with VERIFY_DUMP and read back what it demanded.

    The registry is built from the checks rather than beside them on purpose.
    A registry assembled independently would be a second copy of the results,
    and this project's whole failure mode is second copies.
    """
    out = os.path.join(HERE, ".registry_dump.tsv")
    env = dict(os.environ, VERIFY_DUMP=out)
    r = subprocess.run([sys.executable, os.path.join(HERE, "verify_paper.py")],
                       env=env, capture_output=True, text=True)
    if not os.path.isfile(out):
        raise SystemExit("verify_paper.py produced no dump; the registry cannot "
                         "be built without it.\n" + r.stdout + r.stderr)
    rows = []
    with open(out, encoding="utf-8") as fh:
        for ln in fh:
            if "\t" in ln:
                lab, needle = ln.rstrip("\n").split("\t", 1)
                rows.append((lab, needle))
    os.remove(out)
    return rows, r.returncode, r.stdout


NUM = re.compile(r"[-+]?\d+(?:\.\d+)?%?")
# A unit word may sit between the estimate and its interval: the effective age
# is written "4.6 days old [2.5, 8.4]" in one place and "4.6 days [2.5, 8.4]"
# in another.  A pattern that demanded the bracket immediately after the number
# matched neither, so the gate passed a tamper test that restored the real
# [2.8, 7.7] conflict.  The unit is consumed and the key is the bare figure.
IV = re.compile(r"([-+]?\d+(?:\.\d+)?%?)"
                r"(?:\s+(?:days?|day|points?|months?|weeks?|of R&sup2;|old)){0,2}"
                r"\s*\[\s*([-+]?\d+(?:\.\d+)?%?)\s*,"
                r"\s*([-+]?\d+(?:\.\d+)?%?)\s*\]")


def classify(needle):
    """What kind of result a needle carries."""
    if IV.search(needle):
        return "point+interval"
    if re.search(r"\[\s*[-+]?\d", needle):
        return "interval"
    if re.search(r"\bp\s*(?:=|&le;|<=|<)\s*0?\.\d", needle):
        return "p-value"
    if NUM.fullmatch(needle.strip()):
        return "statistic"
    if NUM.search(needle):
        return "sentence with figures"
    return "text"


def script_of(label):
    m = re.search(r"lab(\d+[a-z]?)", label)
    return f"lab{m.group(1)}" if m else ""


def section_of(label):
    m = re.match(r"(S?\d+(?:\.\d+)?)\s", label)
    return m.group(1) if m else ""


def build():
    needles, rc, stdout = dump_needles()
    rows = []
    for lab, needle in needles:
        rows.append({
            "label": lab,
            "script": script_of(lab),
            "section": section_of(lab),
            "kind": classify(needle),
            "value": needle,
            "figures": " ".join(NUM.findall(needle)),
        })
    with open(REG, "w", encoding="utf-8") as fh:
        cols = ("label", "script", "section", "kind", "value", "figures")
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(r[c].replace("\t", " ") for c in cols) + "\n")
    return rows, rc, stdout


# Trailing punctuation counts as part of the word: "resampled days, gives" is
# prose, and a pattern that stopped at the comma saw only one word and filtered
# the sentence out, which let a restored effective-age conflict pass the gate.
WORDS = re.compile(r"([A-Za-z][A-Za-z'&-]*[,;:)]?\s+){3}$")


def gate_conflicting_intervals(docs):
    """Gate 1: one point estimate, two intervals, in PROSE.

    A table legitimately repeats a value down a column: 51% is the rate at one
    delay for one target and at another delay for another, and those rows carry
    different intervals because they are different cells.  Flagging those buries
    the signal under two dozen non-errors.  What is never legitimate is the same
    figure quoted twice in a SENTENCE with two different intervals, because a
    sentence names its quantity.  So an occurrence counts only when at least
    three words run straight into it, which is what distinguishes prose from a
    row of a table once the tags are stripped.
    """
    seen = {}
    for name, txt in docs.items():
        for m in IV.finditer(txt):
            if not WORDS.search(txt[max(0, m.start() - 60):m.start()]):
                continue
            pt, lo, hi = m.groups()
            lead = " ".join(txt[max(0, m.start() - 60):m.start()].split()[-3:])
            seen.setdefault(pt, {}).setdefault(f"[{lo}, {hi}]", []).append(
                f"{name}: ...{lead} {pt} [{lo}, {hi}]")
    bad = []
    for pt, ivs in seen.items():
        if len(ivs) > 1 and pt not in EXEMPT:
            bad.append((pt, sorted(ivs), [v[0] for v in ivs.values()]))
    return bad


def gate_prose_pvalues(docs):
    """Gate 2: a p-value written beside a z must be that z's p-value."""
    bad = []
    pat = re.compile(r"z\s*=\s*([-+]?\d+\.\d+)[^.]{0,60}?\bp\s*=\s*(0?\.\d+)")
    for name, txt in docs.items():
        for m in pat.finditer(txt):
            z, p = float(m.group(1)), float(m.group(2))
            want = two_sided_p(z)
            # The tolerance is half a unit in the LAST DECIMAL THE TEXT QUOTES,
            # not a flat number.  A flat 0.0015 was tried first and it swallowed
            # the real error this gate exists for: z = 2.57 written beside
            # p = 0.0090 when the answer is 0.0102, a 12% relative error that
            # happens to be 0.0012 in absolute terms.  Quoting four decimals is
            # a claim to four decimals.
            dec = len(m.group(2).split(".")[1])
            tol = 0.5 * 10 ** (-dec)
            if abs(want - p) > tol:
                bad.append((name, m.group(0)[:90], f"{want:.{max(dec, 4)}f}"))
    return bad


def gate_contrast_signs(docs):
    """Gate 3: a named winner must state the direction of its contrast.

    A sentence reading "in the threshold rule's favour" beside a gap of
    -0.0299 is correct only if the contrast runs HAR minus threshold, and a
    reader cannot know that unless the text says so.  The gate therefore looks
    for a possessive claim of victory sitting near a negative statistic, and
    requires an orientation phrase in the same window.

    An earlier version searched forward from the number for the word "favour",
    which found nothing when the claim preceded the number, as it does in the
    one place this matters.  It also matched "least favourable" and "favours
    the objection", neither of which is a claim about a contrast; both are
    excluded by requiring the possessive form.
    """
    CLAIM = re.compile(r"in (?:the )?[a-z][\w' -]{2,30}'s favour")
    ORIENT = re.compile(r"minus the |minus persist|negative number is|"
                        r"ositive favours|minus \+MEAS|paper minus|"
                        r"minus the threshold rule")
    NEG = re.compile(r"(?<![\w.])-\d+\.\d+")
    bad = []
    for name, txt in docs.items():
        for m in CLAIM.finditer(txt):
            win = txt[max(0, m.start() - 400):m.end() + 400]
            if NEG.search(win) and not ORIENT.search(win):
                bad.append((name, m.group(0),
                            "a victory is claimed beside a negative statistic "
                            "with no orientation stated in the sentence"))
    return bad


def gate_unlinked(docs, rows):
    """Gate 4: distinctive numbers in the documents that no check ties to a lab.

    'Distinctive' means a decimal with at least two places, or a bracketed
    endpoint: integers, years, section numbers, percentages of the form 5% and
    counts are excluded, because guarding those would drown the signal.
    """
    linked = set()
    for r in rows:
        linked.update(r["figures"].split())
    # also treat a literal inside a longer checked sentence as linked
    checked_text = " ".join(r["value"] for r in rows)
    out = {}
    tok = re.compile(r"(?<![\w.])[-+]?\d+\.\d{2,}%?(?![\w])")
    for name, txt in docs.items():
        miss = []
        for m in tok.finditer(txt):
            t = m.group(0)
            if t in linked or t.lstrip("+") in linked or t in checked_text:
                continue
            miss.append(t)
        out[name] = miss
    return out


def gate_minus_signs():
    """Gate 5: a minus sign that does not survive extraction is a wrong number.

    Two reviewers read the threshold note's delta = 55 correlation as POSITIVE
    0.1202 and concluded the benchmark beside it was arithmetically impossible.
    It is -0.1202.  Whether their extractor dropped the sign or their eye did,
    the lesson is the same: a negative figure that reaches a reader unsigned is
    a different number, and the one place it can silently go missing is between
    the HTML and the rendered PDF.

    So every figure the source marks negative is looked for in the extracted
    text of the built PDF, and must be found carrying a minus of some kind.
    Needs pdftotext; if the PDFs are not built the gate reports that rather
    than passing quietly.
    """
    out = []
    pairs = (("papers/what-substitutes-for-a-stale-mark.pdf",
              "papers/source/what-substitutes-for-a-stale-mark.html"),
             ("papers/stale-mark-internet-appendix.pdf",
              "papers/source/stale-mark-internet-appendix.html"),
             ("papers/threshold-rule-at-its-ceiling.pdf",
              "papers/source/threshold-rule-at-its-ceiling.html"))
    # The journal variants are shipped documents too, and they are the ones a
    # reader of THIS gate would least expect to be covered: they are generated,
    # so nobody proofreads them.  A minus sign that survives in papers/ and is
    # dropped in a variant is a per-render fault, not a per-source one, so the
    # gate has to see the renders it did not write.  Discovered rather than
    # listed, so a fourth variant is covered the day it exists.
    _vd = os.path.join(HERE, "papers", "variants")
    if os.path.isdir(_vd):
        pairs = list(pairs) + [
            (os.path.join(_vd, v, n + ".pdf"), os.path.join(_vd, v, n + ".html"))
            for v in sorted(os.listdir(_vd))
            if os.path.isdir(os.path.join(_vd, v))
            for n in ("what-substitutes-for-a-stale-mark",
                      "stale-mark-internet-appendix")
            if os.path.isfile(os.path.join(_vd, v, n + ".html"))]
    for pdf, src in pairs:
        pdf, src = os.path.join(HERE, pdf), os.path.join(HERE, src)
        if not os.path.isfile(pdf):
            out.append(f"{os.path.relpath(pdf, HERE)} is not built, so its signs cannot "
                       "be checked; run build_pdf.py first")
            continue
        try:
            txt = subprocess.run(["pdftotext", "-layout", pdf, "-"],
                                 capture_output=True, text=True, timeout=120).stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            out.append("pdftotext is unavailable, so gate 5 cannot run")
            return out
        html = open(src, encoding="utf-8").read()
        # every figure the source writes with an explicit minus entity
        negs = sorted(set(re.findall(r"&minus;(\d+\.\d+)", html)))
        missing = []
        for n in negs:
            hits = [m for m in re.finditer(re.escape(n), txt)]
            if not hits:
                continue          # the figure may live only in a caption variant
            signed = any(txt[max(0, m.start() - 2):m.start()].strip()
                         and txt[max(0, m.start() - 2):m.start()].strip()[-1]
                         in "-\u2212\u2013" for m in hits)
            if not signed:
                missing.append(n)
        if missing:
            out.append(f"{os.path.relpath(pdf, HERE)}: {len(missing)} negative figure(s) "
                       f"extract without a sign: {', '.join(missing[:8])}")
    return out


def main(argv):
    rows, rc, stdout = build()
    docs = {}
    src = os.path.join(HERE, "papers", "source")
    for f, nm in (("what-substitutes-for-a-stale-mark.html", "paper"),
                  ("stale-mark-internet-appendix.html", "appendix"),
                  ("threshold-rule-at-its-ceiling.html", "companion")):
        p = os.path.join(src, f)
        if os.path.isfile(p):
            docs[nm] = plainer(p)

    print(f"results registry: {len(rows)} entries written to "
          f"{os.path.basename(REG)}")
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    for k in sorted(kinds, key=lambda x: -kinds[x]):
        print(f"  {kinds[k]:>4}  {k}")
    scripts = sorted({r["script"] for r in rows if r["script"]})
    print(f"  {len(scripts)} scripts named by at least one entry")

    fails = []
    print("\ngate 1: conflicting intervals")
    bad = gate_conflicting_intervals(docs)
    for pt, ivs, sites in bad:
        fails.append(f"'{pt}' is quoted in prose with {len(ivs)} different "
                     f"intervals: {', '.join(ivs)}")
        print(f"  FAIL  {fails[-1]}")
        for st in sites:
            print(f"          {st}")
    if not bad:
        print(f"  ok    no point estimate carries two intervals "
              f"({len(EXEMPT)} declared exemptions)")

    print("gate 2: prose p-values against stored statistics")
    bad = gate_prose_pvalues(docs)
    for nm, frag, want in bad:
        fails.append(f"{nm}: '{frag}' but that z gives p = {want}")
        print(f"  FAIL  {fails[-1]}")
    if not bad:
        print("  ok    every p-value written beside a z matches it")

    print("gate 3: reversed contrast signs")
    bad = gate_contrast_signs(docs)
    for nm, frag, why in bad:
        fails.append(f"{nm}: '{frag}' - {why}")
        print(f"  FAIL  {fails[-1]}")
    if not bad:
        print("  ok    every named winner states its orientation")

    print("gate 4: unlinked numerical literals")
    miss = gate_unlinked(docs, rows)
    total = sum(len(v) for v in miss.values())
    for nm in sorted(miss):
        print(f"  {len(miss[nm]):>5}  {nm}")
    ceiling = None
    if os.path.isfile(CEIL):
        ceiling = int(open(CEIL).read().split()[0])
    if ceiling is None:
        with open(CEIL, "w", encoding="utf-8") as fh:
            fh.write(f"{total}\n")
        print(f"  ---   {total} unlinked; ceiling recorded, this run sets it")
    elif total > ceiling:
        fails.append(f"unlinked literals rose from {ceiling} to {total}; the "
                     "registry is a ratchet and the count may not rise")
        print(f"  FAIL  {fails[-1]}")
    else:
        if total < ceiling:
            with open(CEIL, "w", encoding="utf-8") as fh:
                fh.write(f"{total}\n")
            print(f"  ok    {total} unlinked, down from {ceiling}; ceiling lowered")
        else:
            print(f"  ok    {total} unlinked, at the ceiling")
    print("gate 5: negative signs survive PDF text extraction")
    for line in gate_minus_signs():
        fails.append(line)
        print(f"  FAIL  {line}")
    else:
        pass
    if not any("extract without a sign" in f or "not built" in f or
               "pdftotext" in f for f in fails):
        print("  ok    every negative figure in the built PDFs extracts with its sign")

    if "--list" in argv:
        for nm in sorted(miss):
            for t in sorted(set(miss[nm])):
                print(f"      {nm}: {t}")

    print()
    if rc != 0:
        fails.append("verify_paper.py itself failed; the registry is built from "
                     "its checks and cannot be trusted while it does")
        print("  the verifier reported:\n" + "\n".join(
            "    " + ln for ln in stdout.strip().splitlines()[-6:]))
    if fails:
        print(f"{len(fails)} build gate failure(s).")
        return 1
    print("all five build gates pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
