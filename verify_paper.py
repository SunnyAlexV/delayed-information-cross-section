"""
verify_paper.py - does every number in the paper still match the labs?

    python verify_paper.py                 checks papers/source/ automatically
    python verify_paper.py other.html      checks a specific file

Run it from the extracted repository folder.  It needs expected_output/ and
papers/source/ beside it, so a loose copy on the Desktop will not work.

WHY THIS EXISTS
---------------
Twice during drafting, a number in the paper drifted from the lab output it came
from, and both times a human reader caught it rather than the author.  The cause
was mundane and will recur: a batch of edits aborted partway, silently discarding
the replacements that had already succeeded, and nothing downstream compared the
prose against its source.

Reproducibility checks like run_all.py --check confirm the SCRIPTS still produce
the same numbers.  They say nothing about whether the PAPER still quotes them.
This closes that gap: it extracts every figure the paper takes from a lab, finds
the same figure in expected_output/, and reports any that disagree.

It is deliberately literal.  It does not parse prose or guess intent; it looks
for specific strings that must be present and specific strings that must not.
A check that cannot be trusted is worse than none, so where a number cannot be
located unambiguously the script says so rather than passing quietly.
"""

import ast
import glob
import math
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# expected_output/ beside this script is the tidy layout; a flat upload puts the
# .txt files here instead.  Same reasoning as the paper-source search below.
EXP = (os.path.join(HERE, "expected_output")
       if os.path.isdir(os.path.join(HERE, "expected_output"))
       else (HERE if os.path.exists(os.path.join(HERE, "lab05_robustness.txt"))
             else os.path.join(HERE, "expected_output")))


def read_text(path):
    """Read UTF-8, everywhere, on every platform.

    Python's open() defaults to the LOCALE encoding, which is UTF-8 on Linux and
    macOS but cp1252 on a default Windows install.  The paper source contains
    U+2212 MINUS SIGN.  cp1252 does not raise on those bytes - it silently
    decodes them as three Latin-1 characters - so the file reads "fine" and only
    the string comparisons fail, one check at a time, with no hint of the cause.
    That is exactly what happened: this script passed here and failed on the
    author's machine on a single interval, which looks like a stale number and
    is not.  A verifier that is itself unportable is worse than no verifier.
    """
    d, f = os.path.split(os.path.abspath(path))
    if d == os.path.abspath(EXP) and f.endswith(".txt") and f[:-4] not in USED:
        USED.append(f[:-4])
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# Every lab output this script has opened, in the order it opened them.  The
# record is what makes the script list in Section 12 self-maintaining: any lab
# whose numbers are checked against the main paper must be a lab the main paper
# names, and the only reliable list of the former is the set of outputs this
# verifier actually reads while checking it.  Keeping that list by hand failed
# twice - once when lab27 and lab28 were cited by new sections while the text
# still said "twenty scripts" and listed neither, and once when lab41 and lab42
# supplied the whole of Section 6 and part of Section 10.2 without ever being
# named, so a reader could not find where those figures came from.
USED = []


def lab(name):
    return " ".join(read_text(os.path.join(EXP, name + ".txt")).split())


# Prose spells small counts; a check that compared a computed integer against
# the document would have to hard-code the word, which is the copied figure
# this project keeps catching.  Table S7's conversion introduced three of
# these in one paragraph ("ten of the twelve endpoints", "twelve of sixteen
# to eleven of sixteen"), so the word is derived from the count instead.
_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve "
          "thirteen fourteen fifteen sixteen seventeen eighteen nineteen "
          "twenty").split()


_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty",
         70: "seventy", 80: "eighty", 90: "ninety"}


def _num_word(n):
    """The English word for a small count, for comparison against prose."""
    if 0 <= n < len(_WORDS):
        return _WORDS[n]
    if 20 < n < 100:
        _t, _u = divmod(n, 10)
        return _TENS[_t * 10] + ("-" + _WORDS[_u] if _u else "")
    if n in _TENS:
        return _TENS[n]
    raise AssertionError(f"_num_word has no word for {n}; extend it rather than "
                         f"writing the word into the check")


# lab39's stressed-tercile interval table, parsed once and reachable from
# everywhere.  Three separate checks used to pin these figures as literals -
# the abstract's, Table 7's cross-reference and Table 20's - and between them
# they froze a conditional rate that no script produced.  They now all read
# this, so a move in the lab reaches every copy or fails loudly.
_CELL = r"(\d+)%\[ *([+-]?\d+)%, *([+-]?\d+)%\]"


def tercile_boot():
    """{delta: {state: (point, lo, hi)}} from lab39's pooled bootstrap table.

    All THREE columns, not one.  An earlier version of this file parsed the
    stressed column alone -- the helper was even named for it -- and the
    consequence was that Table S4's other eight cells kept the values they had
    before the block was widened: eight cells that matched no script, every one
    of them narrower than the lab, and 2,009 passing checks that could not see
    them because nothing was pinned to them.  A check that covers one column of
    a table does not cover the table.
    """
    txt = lab("lab39_temporal_stability")       # flattened to one line
    pat = re.compile(r"(\d+) " + _CELL + " " + _CELL + " " + _CELL)
    out = {}
    for m in pat.finditer(txt):
        g = m.groups()
        out[int(g[0])] = {"calm": g[1:4], "middle": g[4:7], "stressed": g[7:10]}
    return out


def tercile_fieller():
    """{delta: {state: (point, lo, hi)}} from lab41's fixed-b Fieller sets.

    This is what the paper reports for the conditional rate, on the same rule
    Section 4.5 states for the unconditional one: the rate the paper quotes
    inverts the HAC t-test, and lab39's bootstrap is the disclosed comparison.
    """
    txt = read_text(os.path.join(EXP, "lab41_conditional_anatomy.txt"))
    body = txt.split("Fieller 95% set", 1)[-1].split("sets are not ordinary", 1)[0]
    # the table now carries a tau column between R and the set, so the pattern
    # allows it and captures it: tau is the axis the coverage tables are
    # indexed by, and the documents quote it.
    pat = re.compile(r"(calm|middle|stressed)\s+(\d+)\s+(\d+)%\s+([\d.]+)\s+"
                     r"\[([+-]?\d+)%, *([+-]?\d+)%\]")
    out = {}
    for m in pat.finditer(body):
        st, d, pt, tau, lo, hi = m.groups()
        out.setdefault(int(d), {})[st] = (pt, lo, hi, tau)
    return out


def spell(n):
    """Small English cardinals, enough to spell a script or check count."""
    ones = ["zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
            "eighty", "ninety"]
    if n < 20:
        return ones[n]
    if n < 100:
        return tens[n // 10] + ("-" + ones[n % 10] if n % 10 else "")
    head = ones[n // 100] + " hundred"
    return head if n % 100 == 0 else head + " and " + spell(n % 100)


def rows_lab07():
    """delta -> (net, cost, gross, lo, hi) from lab07's final table."""
    out = {}
    for ln in read_text(os.path.join(EXP,
                                     "lab07_estimation_cost.txt")).splitlines():
        m = re.match(r"\s*(\d+)\s+([-+][\d.]+)\s+([-+][\d.]+)\s+([-+][\d.]+)\s+"
                     r"\[([-+][\d.]+),([-+][\d.]+)\]", ln)
        if m:
            out[int(m.group(1))] = m.groups()[1:]
    return out


# Escaped rather than typed, so this file stays pure ASCII on disk: a verifier
# that can itself be corrupted by a careless re-save is not much of a verifier.
DELTA = "&delta; "
MINUS = "\u2212"


def mis_decoded(text):
    """Name the encoding, if this text is UTF-8 that was read as something else.

    The first version of this check listed the mojibake sequences to look for,
    and listed the Latin-1 ones.  cp1252 maps the same bytes to DIFFERENT
    characters - 0x88 is U+02C6 there, not U+0088 - so the check sat in the file
    looking reassuring and caught nothing.  Enumeration was the wrong tool.

    Round-tripping is the right one and does not depend on remembering a table.
    If the whole text can be encoded back to cp1252 or Latin-1, and those bytes
    are then valid UTF-8 that decodes to something DIFFERENT and non-ASCII, then
    the bytes on disk were UTF-8 all along and were read with the wrong codec.
    Correctly-read text fails the test at the last step: it decodes back to
    itself, or the bytes are not valid UTF-8 at all.
    """
    for enc in ("cp1252", "latin-1"):
        try:
            back = text.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if back != text and any(ord(c) > 127 for c in back):
            return enc
    return None


IV_PCT = re.compile(r"\[\s*\+?(-?[\d.]+)%,\s*\+?(-?[\d.]+)%\]")


def plain_text_of(html):
    """plain() for a string rather than a path, with the same substitutions.

    Needed by the converse check, which has to mask table bodies BEFORE the
    tags are stripped and so cannot use plain() on the file.
    """
    out = " ".join(re.sub(r"<[^>]+>", " ", html).split())
    for a, b in ((MINUS, "-"), ("&minus;", "-"), ("\u2013", "-"),
                 ("\u2014", "-"), ("&mdash;", "-"), ("&ndash;", "-"),
                 ("\u2264", "<="), ("\u2265", ">="),
                 ("&nbsp;", " "), ("&amp;", "&"), ("\u00a0", " ")):
        out = out.replace(a, b)
    return " ".join(out.split())


def norm_iv(t):
    """Drop the percent signs INSIDE a bracketed interval, on both sides of a check.

    The labs print an interval as [67%, 72%] and the paper renders it [67, 72]
    beside a point estimate that already carries the unit.  Both are defensible
    and the paper should pick one, but a checker that compares typography
    instead of numbers turns a house-style decision into a false failure, and
    the alternative - rewriting the print format of fourteen labs - risks the
    numbers to fix the punctuation.  So the comparison is made on the numbers.
    """
    return IV_PCT.sub(r"[\1, \2]", t)


def plain(path):
    """Paper source as one line of tag-free, dash-normalised text."""
    paper = read_text(path)
    enc = mis_decoded(paper)
    if enc:
        raise SystemExit(
            f"{os.path.basename(path)} is UTF-8 that was saved after being read\n"
            f"as {enc}.  Recover it from the original rather than re-saving this\n"
            "copy: every check below would fail, and would look like stale numbers.")
    txt = " ".join(re.sub(r"<[^>]+>", " ", paper).split())
    for a, b in ((MINUS, "-"), ("&minus;", "-"), ("\u2013", "-"),
                 ("\u2014", "-"), ("&mdash;", "-"), ("&ndash;", "-"),
                 ("\u2264", "<="), ("\u2265", ">="),
                 ("&nbsp;", " "), ("&amp;", "&"), ("\u00a0", " ")):
        txt = txt.replace(a, b)
    return " ".join(txt.split())


def main(path, companion=None):
    # The paper ships with an Internet Appendix carrying the robustness work.  Checks
    # read the pair, so a figure that leaves the paper without arriving in the
    # supplement still fails: splitting a document must not be a way to lose a
    # number quietly.
    _supp = os.path.join(os.path.dirname(os.path.abspath(path)),
                         "stale-mark-internet-appendix.html")
    # The README is read once, here, because two groups of checks need it:
    # the companion note's script credits and the main paper's script count.
    # It used to be read halfway down, which put it out of reach of the first
    # group and raised UnboundLocalError the moment they were retargeted.
    _rd_path = os.path.join(HERE, "README.md")
    RDME = read_text(_rd_path) if os.path.isfile(_rd_path) else ""
    txt = norm_iv(plain(path))
    if os.path.isfile(_supp):
        txt = txt + " " + norm_iv(plain(_supp))
    else:
        txt = txt
    fails, checks = [], 0
    # Every needle any want() demanded, kept so that the last check in
    # this file can ask the converse question: is there a quoted result
    # interval in either document that NO check asked for?  That is the
    # shape the stale conditional rate hid in for months -- 81% [74, 87]
    # was not wrong against a check, it was invisible to every check.
    _demanded = []
    _cdemanded = []
    # cwant()'s needles, kept separately: the note is a separate
    # document and want() cannot see it, so coverage over it has to be
    # measured against the needles that were aimed at it.
    _cdem_for_cover = []
    _dump_path = os.environ.get("VERIFY_DUMP")
    _DUMP = open(_dump_path, "w") if _dump_path else None

    _TAG = re.compile(r"<[a-zA-Z/][^>]*>")

    def _no_tags(label, needle):
        """A needle containing an HTML tag can never match.

        plain() strips tags before any comparison, so a needle written against
        the raw source is a check that silently does nothing - and a forbid()
        written that way is worse than absent, because it reads as a ban while
        permitting exactly what it names.  One such ban on the phrase
        '<i>Given implied volatility, the foreign block adds nothing.</i>'
        survived a tamper test by passing it.  Tags are now a loud failure.
        """
        nonlocal checks
        if _TAG.search(needle):
            checks += 1
            fails.append(f"{label}: needle contains an HTML tag and can never match "
                         f"tag-stripped text: '{needle}'")
            return False
        return True

    def want(label, needle):
        nonlocal checks
        if not _no_tags(label, needle):
            return
        checks += 1
        # VERIFY_DUMP exists for one job: syncing the documents after a lab
        # rerun moves numbers.  Running this file against the OLD lab outputs
        # records the needle each check demanded then; running it against the
        # new ones records what it demands now; pairing the two by label gives
        # an exact old-to-new substitution list.  That is the safe way to do a
        # large sync - a blind numeric sweep over the documents was tried once
        # and produced 208 collisions, because the same four digits mean
        # different things in different sections.
        if _DUMP is not None:
            _DUMP.write(f"{label}\t{needle}\n")
        _demanded.append(norm_iv(needle.replace(MINUS, "-")))
        if norm_iv(needle.replace(MINUS, "-")) not in txt:
            fails.append(f"{label}: MISSING  '{needle}'")

    def forbid(label, needle):
        nonlocal checks
        if not _no_tags(label, needle):
            return
        checks += 1
        if norm_iv(needle.replace(MINUS, "-")) in txt:
            fails.append(f"{label}: STALE    '{needle}' should not appear")

    # The companion note is a separate document, so want() and forbid() cannot
    # see it: they search the paper and the Internet Appendix.  Its helpers used
    # to be defined inside one of the two companion blocks, which meant a check
    # written in the other block raised UnboundLocalError, and a check written
    # outside both searched the wrong documents and failed for the wrong reason.
    # Both happened.  They are defined once here instead, so a companion check
    # works wherever it is written.
    ctxt = plain(companion) if companion and os.path.isfile(companion) else ""

    def cwant(label, needle):
        nonlocal checks
        if not _no_tags(label, needle):
            return
        _cdem_for_cover.append(norm_iv(needle.replace(MINUS, "-")))
        checks += 1
        if not ctxt:
            return
        # dumped like want(), so build_registry.py sees the companion's checked
        # figures.  Hoisting these helpers dropped this line once, and the
        # registry's unlinked-literal ratchet caught it immediately: the note's
        # count jumped from 77 to 191 because every figure it checks had
        # stopped being recorded as checked.
        if _DUMP is not None:
            _DUMP.write(f"{label}\t{needle}\n")
        if needle.replace(MINUS, "-") not in ctxt:
            fails.append(f"{label}: MISSING  '{needle}'")

    def cforbid(label, needle):
        nonlocal checks
        if not _no_tags(label, needle):
            return
        checks += 1
        if not ctxt:
            return
        if needle.replace(MINUS, "-") in ctxt:
            fails.append(f"{label}: STALE    '{needle}' should not appear")

    def only_iv_re(label, stem_re, lo, hi):
        """As only_iv, but the stem is a regex whose match is followed by the
        interval.  Used where the same figure is written three ways: '4.6 days
        old [..]' in the abstract, '4.6 days [..]' in one appendix section and
        a bare '6.8 [..]' in a list where the unit was stated once for the row.
        """
        nonlocal checks
        checks += 1
        want_iv = norm_iv(f"[{lo}, {hi}]")
        found = re.findall(stem_re + r"\s*(\[[^\]]{1,24}\])", txt)
        bad = [x for x in found if x != want_iv]
        if not found:
            fails.append(f"{label}: nothing matched {stem_re!r}")
        elif bad:
            fails.append(f"{label}: {len(bad)} of {len(found)} interval(s) matching "
                         f"{stem_re!r} disagree with the labs' {want_iv}: "
                         f"{', '.join(sorted(set(bad)))}")

    def only_iv(label, stem, lo, hi, reach=44):
        """Every interval attached to `stem` must be the one the labs print.

        want() proves at-least-one: a figure quoted in four places passes on
        the strength of whichever copy happens to be current, and the other
        three can carry a pre-correction interval indefinitely.  That is not
        hypothetical.  The effective age was quoted three times; the abstract
        was tied to lab45 and satisfied the check on its own, while both
        Internet Appendix copies sat at a superseded [2.8, 7.7] for as long as
        nothing looked at them.  This finds EVERY occurrence of the stem and
        requires the first interval within `reach` characters of each to agree,
        so a site that drifts is a site that fails.
        """
        nonlocal checks
        checks += 1
        want_iv = norm_iv(f"[{lo}, {hi}]")
        seen, bad, i = 0, [], txt.find(stem)
        while i != -1:
            m = re.search(r"\[[^\]]{1,24}\]", txt[i + len(stem):i + len(stem) + reach])
            if m:
                seen += 1
                if m.group(0) != want_iv:
                    bad.append(m.group(0))
            i = txt.find(stem, i + 1)
        if not seen:
            fails.append(f"{label}: no interval found after any '{stem}'")
        elif bad:
            fails.append(f"{label}: {len(bad)} of {seen} interval(s) after '{stem}' "
                         f"disagree with the labs' {want_iv}: {', '.join(sorted(set(bad)))}")

    r7 = rows_lab07()

    # --- Table 2 and the prose around it, against lab07 -------------------
    for d, (net, cost, gross, lo, hi) in r7.items():
        want(f"lab07 d={d} cost", cost)
        want(f"lab07 d={d} gross", gross)
    want("lab07 d=0 interval", f"[{r7[0][3]}, {r7[0][4]}]")
    want("lab07 d=1 interval", f"[{r7[1][3]}, {r7[1][4]}]")
    want("lab07 d=2 interval", f"[{r7[2][3]}, {r7[2][4]}]")

    # Every superseded cost/gross value must be gone from the two rows of Table 2
    # that carry them.  These used to be forbidden as bare tokens anywhere in the
    # paper, and that is a needle too loose to keep: "-0.0141" is a superseded
    # cost AND, since Section 8.5 was added, the foreign increment given the
    # nine-day block, so the bare form failed on a correct paper.  Scoped to the
    # rows it is about, the check still catches what it was written for.
    _t2 = txt.split("Cost of breadth")[-1].split("Gross content clears zero")[0] \
        if "Cost of breadth" in txt else ""
    checks += 1
    if not _t2:
        fails.append("Table 2's cost and gross rows could not be located")
    # The list is historical, and history is not static: the benchmark
    # correction moved the cost and gross columns, and two values that had been
    # superseded came back as current ones (-0.0294 is now the cost at delta = 8
    # and +0.0353 the gross at delta = 2).  Forbidding a figure the lab now
    # prints would fail a correct paper, so anything lab07 currently produces is
    # removed from the list rather than deleted from it by hand.
    _live7 = {v for row in r7.values() for v in row}
    for old in ("-0.0119", "-0.0141", "-0.0166", "-0.0232", "-0.0294", "-0.0547",
                "+0.0016", "+0.0135", "+0.0353", "+0.1063", "+0.1583", "+0.3152"):
        if old in _live7:
            continue
        checks += 1
        if old.replace(MINUS, "-") in _t2:
            fails.append(f"superseded lab07 value: STALE '{old}' is still in "
                         f"Table 2's cost or gross row")
    forbid("superseded interval", "[-0.0077, +0.0104]")
    forbid("surrogate draw count", "ten draws")

    # --- sup-t and the GW row, against lab06 ------------------------------
    l6 = lab("lab06_inference")
    # Read out of lab06, not transcribed: the value moves with the block length
    # and with the studentisation, and both have been corrected since this line
    # was first written as the literal 6.51.
    _m6 = re.search(r"sup-t critical value ([\d.]+)", l6)
    checks += 1
    if not _m6:
        fails.append("lab06 no longer prints a sup-t critical value")
    else:
        want("lab06 sup-t critical value", _m6.group(1))
    # The claim Table S1's caption makes about its own p-values, checked against
    # them.  The caption said "all entries have p <= 0.011" and that was true
    # under the nine-lag kernel; the measured bandwidth moved the QLIKE row out
    # past 1% and a caption is exactly the place a stale claim survives, because
    # nobody rereads one.
    _p6 = re.findall(r"\s(\d+) -?[\d.]+ \((0?\.\d+)\) -?[\d.]+ \((0?\.\d+)\) "
                     r"-?[\d.]+ \((0?\.\d+)\)", l6)
    _seven = {"3", "5", "8", "13", "21", "34", "55"}
    _pq = [float(q) for d, g, q, c in _p6 if d in _seven]
    _pgc = [float(x) for d, g, q, c in _p6 if d in _seven for x in (g, c)]
    if _pq and _pgc:
        checks += 1
        if max(_pgc) >= 0.01:
            fails.append(f"Table S1: a squared-loss or Clark-West p-value is now "
                         f"{max(_pgc):.3f}, but the caption says every one is below 1%")
        checks += 1
        if not (max(_pq) < 0.05):
            fails.append(f"Table S1: the largest QLIKE p-value is now {max(_pq):.3f}, "
                         f"but the caption says the row clears 5% throughout")
        want("S1 QLIKE p range",
             f"from p = {min(_pq):.3f} at &delta; = "
             f"{[d for d, g, q, c in _p6 if float(q) == min(_pq)][0]}")
        # The MAIN PAPER quotes the same range in a sentence of its own, and
        # that sentence was unguarded: it still said "between 0.010 and 0.034"
        # after the bandwidth moved and the appendix's copy had been corrected,
        # so the two documents disagreed about the same statistic and nothing
        # said so.  Both copies are pinned from here now.
        want("4.6 QLIKE p range in the body",
             f"between {min(_pq):.3f} and {max(_pq):.3f} on QLIKE")

    # The GW column is read out of lab06's own part-1 table rather than listed
    # here.  It was a literal list, and it moved twice in one week: once when the
    # benchmark changed and again when the HAC bandwidth did.  Only the
    # significant ones are required of the paper, since those are the ones it
    # quotes; the row below checks that the set has not silently grown or shrunk.
    _gw6 = re.findall(r"(?:^|\s)\d+ (-?[\d.]+) \(0?\.\d+\) "
                      r"-?[\d.]+ \(0?\.\d+\)", l6)
    checks += 1
    if len(_gw6) < 8:
        fails.append(f"lab06: parsed {len(_gw6)} GW statistics from part 1, expected 10")
    for z in [g for g in _gw6 if abs(float(g)) > 1.96 and float(g) > 0]:
        want(f"lab06 GW {z}", z)
    # Both documents make claims about the p-value COLUMNS, and those claims had
    # survived the bandwidth correction unchanged: the paragraph under Table S1
    # said every statistic from delta = 3 on has p <= 0.011 and all but one are
    # below 0.01, and attributed z = 2.57 to delta = 3, where lab06 prints 2.12
    # and p = 0.034.  The caption directly above it said the QLIKE row clears 1%
    # nowhere.  The two were read off different runs.  The bounds are derived
    # here from lab06's own bracketed p-values, over the delays the section
    # quotes (delta >= 3), so a rerun moves both documents or fails them.
    # lab() collapses the file to one line, so part 1 is isolated by its header
    # and the rows are matched inline rather than anchored to line starts.
    _t6 = l6.split("GW (sq loss)")[-1].split("GW and Clark-West are HAC")[0]
    _P6 = re.compile(r"(\d+) (-?[\d.]+) \((0?\.\d+)\) (-?[\d.]+) \((0?\.\d+)\) "
                     r"(-?[\d.]+) \((0?\.\d+)\)")
    _rows6 = [(int(m.group(1)), m.group(3), m.group(4), m.group(5), m.group(7))
              for m in _P6.finditer(_t6)]
    _rows6 = [r for r in _rows6 if r[0] >= 3]
    checks += 1
    if len(_rows6) != 7:
        fails.append(f"lab06: parsed {len(_rows6)} delta>=3 rows for the Table S1 "
                     "claims, expected 7")
    else:
        _ql = {d: (z, p) for d, _sp, z, p, _cw in _rows6}
        _sq = [float(sp) for _d, sp, _z, _p, _cw in _rows6]
        _cw6 = [float(cw) for *_r, cw in _rows6]
        _lo_d = min(_ql, key=lambda d: float(_ql[d][1]))
        _hi_d = max(_ql, key=lambda d: float(_ql[d][1]))
        want("S1 QLIKE range",
             f"the QLIKE row runs from p = {_ql[_lo_d][1]} at &delta; = {_lo_d} "
             f"to p = {_ql[_hi_d][1]} at &delta; = {_hi_d}")
        want("S1 worst two-sided p",
             f"every statistic from &delta; = 3 onward has p &le; {_ql[_hi_d][1]}")
        checks += 1
        if max(_sq + _cw6) >= 0.01:
            fails.append("lab06: a squared-loss or Clark-West p-value at delta >= 3 is "
                         "no longer below 0.01, which Section S2 states of both rows")
        # The caption's verdict on the QLIKE row is stated in words, so it is
        # derived from the p-values and demanded in words.  Asserting only that
        # the LAB still has the property leaves the caption free to say the
        # opposite, which is how a tamper test found this check passing while
        # the caption claimed the row clears 1% throughout.
        checks += 1
        _qmin, _qmax = float(_ql[_lo_d][1]), float(_ql[_hi_d][1])
        _verdict = ("clears 5% throughout and 1% nowhere"
                    if _qmax < 0.05 <= 1 and _qmin >= 0.01
                    else "clears 1% throughout" if _qmax < 0.01
                    else None)
        if _verdict is None:
            fails.append(f"lab06: the QLIKE row now spans p = {_qmin} to {_qmax}, "
                         "which is neither of the two verdicts Table S1's caption "
                         "knows how to state; the caption needs rewriting by hand")
        else:
            want("S1 caption QLIKE verdict", f"so it {_verdict}")
        # the one-sided restatement must halve the two-sided bounds, not assert
        # a bound the halving does not reach: at z = 2.12 one-sided p is 0.017,
        # so "every entry under 0.01" was false as well as stale.
        # The bound the paper states is parsed and compared numerically rather
        # than reconstructed, because any bound at or above the largest
        # one-sided p-value is honest and the paper may round it.  What is
        # checked is that it HOLDS: 0.01 did not, since z = 2.12 gives 0.017.
        _os = re.search(r"the whole QLIKE row under (0?\.\d+)", txt)
        checks += 1
        if not _os:
            fails.append("Section S2 no longer states a one-sided bound for the "
                         "QLIKE row")
        elif float(_os.group(1)) < float(_ql[_hi_d][1]) / 2:
            fails.append(f"Section S2 claims the one-sided QLIKE row is under "
                         f"{_os.group(1)}, but lab06's weakest two-sided p is "
                         f"{_ql[_hi_d][1]}, so one-sided it is "
                         f"{float(_ql[_hi_d][1]) / 2:.4f}")
        forbid("S1 stale one-sided claim",
               "would put every entry under 0.01")
        forbid("S1 stale p-value ceiling", "has p &le; 0.011")
        forbid("S1 misattributed z", "at &delta; = 3, where z = 2.57")

    # --- the delta = 0 row on four loss scales, against lab10 part 3 ------
    # Every figure in this paragraph was pre-correction, and two of the three
    # intervals it printed were malformed: [0.0120, -0.0021] runs backwards and
    # [0.0065, +0.0212] does not contain the 0.0042 it was attached to.  Nothing
    # checked the paragraph, and nothing checked interval WELL-FORMEDNESS
    # anywhere, so a transcription error that a reader spots at a glance had
    # been sitting in the supplement.  Both gaps are closed here.
    _p3 = read_text(os.path.join(EXP, "lab10_loss_scale.txt"))
    # The header is fenced by rule lines, so the body is the block after the
    # first rule that follows it.  The rules are split on as whole LINES: a
    # bare "=" * 40 needle splits an 84-character rule into empty pieces, which
    # is how the first version of this check parsed nothing and said so.
    _p3 = re.split(r"\n=+\n", _p3.split("3.  THE delta = 0 ROW")[-1])[1]
    _L3 = re.compile(r"\s+(sq\(log\)|QLIKE|MSE\(nat\)|MSE\(nrm\))\s+[\d.]+\s+[\d.]+"
                     r"\s+([-+][\d.]+)\s+\[([-+][\d.]+),([-+][\d.]+)\]")
    _r3 = {m.group(1): m.groups()[1:] for m in _L3.finditer(_p3)}
    checks += 1
    if len(_r3) != 4:
        fails.append(f"lab10 part 3: parsed {len(_r3)} of 4 loss rows for the "
                     "delta = 0 paragraph")
    else:
        def _n(x):                      # 0.0099, not +0.0099, in the prose
            return x.lstrip("+-")
        # the first of the three carries the word "interval"; the other two
        # inherit it, which is house style and not a difference in quantity
        for _key, _lead in (("sq(log)", "worse by {}, interval ["),
                            ("QLIKE", "worse by {}, ["),
                            ("MSE(nat)", "worse by {}, [")):
            _d, _lo, _hi = _r3[_key]
            checks += 1
            if not _d.startswith("-"):
                fails.append(f"lab10 part 3: the {_key} difference at delta = 0 is "
                             f"now {_d}; Section S8 says the cross-section is worse "
                             "on all three")
            want(f"S8 delta=0 {_key}",
                 _lead.format(_n(_d)) + f"{_lo}, {_hi}]")
        _d, _lo, _hi = _r3["MSE(nrm)"]
        want("S8 delta=0 normalised", f"comes out at {_d}, [{_lo}, {_hi}]")
        checks += 1
        _ex = sum(1 for k in ("sq(log)", "QLIKE", "MSE(nat)")
                  if _r3[k][1][0] == _r3[k][2][0])
        if _ex != 2:
            fails.append(f"lab10 part 3: {_ex} of the three losses now exclude zero "
                         "at delta = 0; Section S8 says two of the three do")

    # --- every interval in both documents must be well formed -------------
    # A ratio interval written [0.0120, -0.0021] is wrong on its face, and the
    # supplement carried one for as long as no check looked.  This sweeps every
    # bracketed numeric pair in both documents and requires the low endpoint to
    # be no greater than the high one.  Percentages, decimals and signed values
    # all parse; anything that is not a pair of numbers is skipped, so tables of
    # citations and ranges written with words are unaffected.
    checks += 1
    _mal = []
    for _m in re.finditer(r"\[\s*([-+−]?[\d.]+)%?\s*,\s*([-+−]?[\d.]+)%?\s*\]",
                          txt):
        try:
            _a = float(_m.group(1).replace("−", "-"))
            _b = float(_m.group(2).replace("−", "-"))
        except ValueError:
            continue
        if _a > _b:
            _mal.append(_m.group(0))
    if _mal:
        fails.append(f"{len(_mal)} interval(s) run backwards: "
                     f"{', '.join(sorted(set(_mal))[:6])}")

    # --- the block the simultaneous band is actually resampled in ---------
    # The band's prose said "moving blocks of ten days" while lab06 resampled
    # in forty and the critical value quoted beside it, 2.42, is the forty-day
    # one.  The block-length sweep elsewhere in this file reads the LABS; this
    # reads the sentence, which is where the stale number lived.  Spelled,
    # because the paper writes block lengths in words.
    _bl6 = re.search(r"bootstrap block (\d+) days", l6)
    checks += 1
    if not _bl6:
        fails.append("lab06 no longer reports the block its simultaneous band uses")
    else:
        want("band block length in words",
             f"in moving blocks of {spell(int(_bl6.group(1)))} days")
        for _n in range(5, 81, 5):
            if _n != int(_bl6.group(1)):
                forbid(f"stale band block {_n}",
                       f"in moving blocks of {spell(_n)} days")

    # --- pointwise, simultaneous and equivalence are three claims ---------
    # The audit's objection is that non-rejection gets written as equality.
    # The paper distinguishes the three levels explicitly now, and the
    # distinction has to survive editing, because it is the thing that keeps
    # "adds nothing detectable" from collapsing back into "adds nothing".
    for _lvl, _needle in (
            ("pointwise", "Pointwise : an individual interval excludes zero"),
            ("simultaneous",
             "Simultaneous : the effect survives a band covering the whole delay "
             "grid at once"),
            ("equivalence",
             "Equivalence : the effect lies inside a margin declared negligible in "
             "advance, which this paper tests nowhere and therefore never claims")):
        want(f"inference vocabulary: {_lvl}", _needle)
    want("non-detection is not zero",
         "no incremental contribution was detected; it is not a cell where the "
         "contribution is zero")

    # --- the results registry, and the claims the paper makes about it -----
    # The registry is built from THIS file's own dump, so it cannot be built
    # unless the checks exist; what can go wrong is the other direction, with
    # the paper describing gates that were quietly dropped.  The builder, the
    # ratchet and the four gate descriptions are therefore required to exist
    # and to match what the paper says they do.
    for _f, _why in ((os.path.join(HERE, "build_registry.py"),
                      "the paper's availability statement names it"),
                     (os.path.join(HERE, "registry_ceiling.txt"),
                      "the unlinked-literal ratchet has nowhere to record its "
                      "ceiling without it")):
        checks += 1
        if not os.path.isfile(_f):
            fails.append(f"{os.path.basename(_f)} is missing, and {_why}")
    _br = os.path.join(HERE, "build_registry.py")
    if os.path.isfile(_br):
        _brt = read_text(_br)
        for _gate in ("gate_conflicting_intervals", "gate_prose_pvalues",
                      "gate_contrast_signs", "gate_unlinked"):
            checks += 1
            if f"def {_gate}(" not in _brt:
                fails.append(f"build_registry.py no longer defines {_gate}, but the "
                             "paper's availability statement describes four gates")
        # the ratchet must be a ratchet: the comparison has to be an increase
        checks += 1
        if "total > ceiling" not in _brt:
            fails.append("build_registry.py no longer fails when the unlinked count "
                         "rises, so the ratchet the paper describes is not one")
    want("registry described in the availability statement",
         "builds a machine-readable registry of every figure the verifier ties to a lab")
    want("registry gates described",
         "no point estimate may be quoted in prose with two different intervals")
    want("registry is derived, not parallel",
         "built from the verifier's own checks rather than maintained beside them")

    # --- scope, terminology and redistribution, the reviewer's last items --
    # Three claims that were written carefully and guarded by nothing, which is
    # how the carefully written ones go stale first.
    #
    # "Price" is a loaded word in a finance paper.  This one uses it to mean a
    # ratio on a common ruler, never an amount of money, and says so twice.
    # Both statements are required, because the title carries the word.
    want("price is defined, not assumed",
         "it means that the two things being compared can be put on one ruler")
    want("price in the title is defined",
         "in this title means a ratio on one ruler and not an amount")
    # The own-asset options result rests on two targets.  The claim table says
    # so; that sentence is what keeps a scope limit from reading as precision.
    want("options scope is two targets",
         "two targets only, SPX and DAX, each on the 4,525-day window")
    # Table 5's grades are the paper's own vocabulary and were checked by
    # nothing.  "Established" is defined as excluding zero AND surviving the
    # robustness set; "supported" as resting on a sample too small to resolve
    # alternatives.  Four rows carried "established" on two targets, on eight
    # markets, or on dependent metropolitan indices, and one of them cited the
    # very cells Section S37's audit calls exploratory.  A row whose evidence
    # names a small sample, or points at the audit, may not be graded
    # "established" by the paper's own definition.
    _T5 = re.compile(r"<tr><td>(.{10,200}?)</td><td>(established|supported|"
                     r"suggestive|not claimed)</td><td>(.{10,600}?)</td></tr>",
                     re.S)
    _rawM = read_text(path)
    _rows5 = [(re.sub(r"<[^>]+>", "", c).strip(),
               g,
               " ".join(re.sub(r"<[^>]+>", " ", e).split()))
              for c, g, e in _T5.findall(_rawM)]
    checks += 1
    if len(_rows5) < 10:
        fails.append(f"Table 5: parsed {len(_rows5)} claim rows, expected at least 10")
    _THIN = ("two targets only", "eight targets", "eight points",
             "Section S37", "metro-phases", "14 of 14 metros")
    for _claim, _grade, _ev in _rows5:
        checks += 1
        _hit = [t for t in _THIN if t in _ev]
        if _grade == "established" and _hit:
            fails.append(f"Table 5: '{_claim[:60]}' is graded established but its "
                         f"evidence names {_hit[0]!r}, which is the paper's own "
                         "definition of supported")
    # and the grade definitions themselves must stay put, since the regrade
    # above is only meaningful against them
    want("Table 5 grade definitions",
         "established means the estimate excludes zero and survives the robustness "
         "set")
    # The definition was about a LEVEL excluding zero, while one row of the table
    # grades a COMPARISON - "the effect is larger in stressed markets" - whose
    # evidence is two sets that do not overlap.
    want("Table 5's grade definition covers the comparative row",
         "for the one row that grades a comparison rather than a level it means the "
         "two confidence sets being compared do not overlap and go on not "
         "overlapping across that set")
    want("Table 5 grade definitions, the other three",
         "supported means it is consistent across the tests run but rests on a "
         "sample too small to resolve alternatives")
    # The paper stated the number of scope conditions in three places and said
    # "two" in one and "four" in the other two.  The count is now derived from
    # the Internet Appendix's own section markers, so the three cannot diverge
    # again and a new scope section forces the sentences to be updated.
    # The marker was relabelled from "established" to "supported in the examined
    # markets", because two own-index option cases cannot establish a universal
    # scope condition.  The count is read off the new marker.
    _scope_n = len(re.findall(r"&middot;&nbsp;Scope condition supported in the "
                              r"examined markets",
                              read_text(_supp) if os.path.isfile(_supp) else ""))
    checks += 1
    if _scope_n == 0:
        fails.append("the Internet Appendix no longer marks any section as "
                     "establishing a scope condition, but the paper counts them")
    else:
        want("scope condition count in Section 10",
             f"the {spell(_scope_n)} scope conditions that bound where the result holds")
        want("scope condition count in the availability statement",
             f"the {spell(_scope_n)} scope conditions.")
        want("scope conditions are named",
             "the option chain and coupling")
        forbid("stale scope-condition count", "the four scope conditions")
        # A scope condition checked on two own-index markets is identified
        # there, not established in general; the later sections already say so
        # and the early statement of it has to match them.
        want("scope condition is identified, not established",
             "identified in the examined markets rather than assumed")
        forbid("scope condition overstated at first mention",
               "scope condition on that rate is established rather than assumed")
    # Three phrases the audit named as overclaiming, removed and banned so they
    # cannot return: a result is not "unambiguous", a diagnostic is not "the
    # whole story", and an objection is not "retired".
    for _over in ("The answer is unambiguous",
                  "the correlation is the whole story",
                  "retires the objection"):
        forbid("overclaiming scope language", _over)
    want("coupling is bounded, not the whole story",
         "coupling is the part of the story this design can see")

    # The data cannot be redistributed and the paper names each vendor's terms
    # rather than asserting a blanket prohibition.  The count of input files is
    # read out of the manifest, which is the thing that would actually change.
    _mf = os.path.join(HERE, "data", "MANIFEST.tsv")
    checks += 1
    if not os.path.isfile(_mf):
        fails.append("data/MANIFEST.tsv is missing, so the data-availability statement "
                     "points at a file listing that does not exist")
    else:
        _rows = [r for r in read_text(_mf).splitlines()[1:] if r.strip()]
        _free = [r for r in _rows if r.split("\t")[2].strip().lower() == "yes"]
        checks += 1
        if not _rows:
            fails.append("data/MANIFEST.tsv lists no input files")
        else:
            _cl = os.path.join("/mnt/user-data/outputs", "JFEc-cover-letter.md")
            if os.path.isfile(_cl):
                _clt = " ".join(read_text(_cl).split())
                checks += 1
                _need = (f"Of {spell(len(_rows))} input files, "
                         f"{spell(len(_free))} is public domain"
                         if len(_free) == 1 else
                         f"Of {spell(len(_rows))} input files, "
                         f"{spell(len(_free))} are public domain")
                if _need not in _clt:
                    fails.append(f"the cover letter does not say '{_need}', but the "
                                 f"manifest lists {len(_rows)} files of which "
                                 f"{len(_free)} are redistributable")
    # No vendor export may ship.  The manifest marks exactly which files are
    # redistributable, and the repository must contain those and no others:
    # a licence statement is only as good as what sits beside it on disk.
    if os.path.isfile(_mf):
        _ok_files = {r.split("\t")[0] for r in read_text(_mf).splitlines()[1:]
                     if r.strip() and r.split("\t")[2].strip().lower() == "yes"}
        _shipped = []
        for _root, _dirs, _files in os.walk(os.path.join(HERE, "data")):
            for _f in _files:
                if _f.endswith(".csv"):
                    _rel = os.path.relpath(os.path.join(_root, _f),
                                           os.path.join(HERE, "data"))
                    _shipped.append(_rel.replace(os.sep, "/"))
        checks += 1
        _extra = sorted(set(_shipped) - _ok_files)
        if _extra:
            fails.append(f"{len(_extra)} input file(s) ship in data/ that the manifest "
                         f"does not mark redistributable: {', '.join(_extra[:5])}")

    for _vendor in ("Investing.com's terms forbid distributing data obtained from the "
                    "site without written permission",
                    "Cboe's permit one downloaded copy for personal non-commercial use",
                    "Yahoo's forbid reproducing or distributing any portion",
                    "Copyrighted: Pre-approval Required"):
        want("redistribution terms stated per vendor", _vendor)

    # --- the review's last wording repairs, each with its ban -------------
    # Four statements stronger than the evidence: an "impossible" that is only
    # a warning sign, an "unmeasurable" that the Fieller analysis shows is
    # merely imprecise, a universal redundancy claim resting on two own-index
    # cases, and a section opening that claims every objection was tested.
    for _lbl, _need, _ban in (
            ("overlap is evidence, not proof",
             "whether iid standard errors come out smaller depends on the signs and "
             "magnitudes of the serial covariances",
             "which is impossible for overlapping targets"),
            ("calm market is imprecise, not unidentified",
             "the Fieller analysis does not classify it as unidentified",
             "averages a well-measured result with an unmeasurable one"),
            ("own-index redundancy is scoped to the cases examined",
             "In both own-index option cases examined, the compressed cross-section "
             "adds no detectable information",
             "An option chain on the asset itself makes that cross-section redundant"),
            # The needle stops short of the subject, because build_variants.py
            # repoints it: the JFEc variant relocates this section into the
            # Internet Appendix, where "The Internet Appendix examines" becomes
            # "This appendix examines".  Pinning the subject pinned the one word
            # the relocation has to change.
            ("Section 9 opens on identified threats",
             "examines the principal identified threats to each central claim",
             "Every claim above has been tested against the objection that would "
             "retire it")):
        want(_lbl, _need)
        forbid(f"stale: {_lbl}", _ban)
    forbid("scope condition overclaimed",
           "&middot; Scope condition established")

    # --- the VIX horizon ratio, in one unit --------------------------------
    # "Six times longer" divides thirty CALENDAR days by five TRADING days.
    # Thirty calendar days is about twenty-one trading days, so the ratio is
    # about four.  The same mix produced "four times closer than thirty" for
    # VIX9D, whose nine calendar days are about six trading days against the
    # target's five: near-matched, not matched.
    want("horizon conversion stated",
         "thirty calendar days is about twenty-one trading days")
    want("VIX9D is near-matched, not matched",
         "ine calendar days is about six trading days")
    for _mix in ("six times longer than the thing it forecasts",
                 "six times longer than the thing it is forecasting",
                 "four times closer than thirty"):
        forbid("calendar and trading days mixed", _mix)

    # --- the title page against JFEc's own requirements --------------------
    # "It should also include name, address, telephone number, and e-mail
    # address of the author responsible for correspondence."  All four are
    # required, so none may be tidied away: a phone number looks like clutter
    # right up until the desk rejects the submission for missing it.  Keywords
    # and JEL codes belong with the abstract, not above it.
    for _lbl, _needle in (
            ("title page: corresponding author named",
             "Corresponding author. Sunny Alex Vellanikaran"),
            # The manuscript names the affiliation and no country.  A country
            # on a title page is a CORRESPONDENCE address, never a nationality,
            # and this author's is supplied to a journal rather than printed on
            # a public preprint - so what is checked here is that the
            # affiliation survives, not that a country follows it.
            ("title page: affiliation", "Sunny Alex Vellanikaran, Independent."),

            ("title page: JEL codes", "JEL codes: C53, C58, G17, G12, C22.")):
        want(_lbl, _needle)
    # The telephone and email are required but are NOT written out here: this
    # file is published, and a literal contact detail in source code is one a
    # scraper finds without reading the paper.  Their PRESENCE is what the
    # journal requires, so presence is what is checked.
    #
    # The PUBLISHED source now carries the sentence without the number, because
    # the repository is public and the paper's own front matter was the one
    # place the number still appeared.  contact.txt, which is not published,
    # supplies it when build_variants.py makes the copies that go to a journal.
    # So both forms are accepted here and the STRICTER rule - a real number -
    # is applied below to anything built into papers/variants/, which is what
    # actually gets uploaded.  Checking only the loose form everywhere would
    # let a submission go out with the placeholder still in it.
    #
    # The strict rule is conditional on contact.txt EXISTING.  Without that
    # condition a plain clone - which has no contact.txt, so no number to
    # insert - failed its own variant build, and a repository whose standard
    # command fails for everyone but the author is a repository nobody runs.
    # The rule is therefore: if a number was available to insert, it must have
    # been inserted; if none was, the placeholder stands and says so.
    _is_variant = os.sep + "variants" + os.sep in os.path.abspath(path)
    _have_tel = os.path.isfile(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "contact.txt"))
    # The published source names no telephone at all now: the placeholder
    # sentence it used to carry read as leftover machinery on a public preprint,
    # which is what SSRN posts.  So the requirement applies to the JOURNAL
    # copies only, and only when contact.txt gave the builder a number to insert.
    _tel_required = _is_variant and _have_tel
    _tel_pat = r"Telephone \+\d[\d\s]{6,}\."
    # No label may appear twice in the contact block.  It did: the title-page
    # builder re-added "Corresponding author." to text that already began with
    # it, and a doubled label is invisible to every figure check in this file
    # while being the first thing a desk editor reads.
    checks += 1
    if re.search(r"Corresponding author\.\s*Corresponding author\.", txt):
        fails.append("the contact block says 'Corresponding author.' twice")

    _contact_checks = [("title page: email",
                        r"Email [^\s@]+@[^\s@]+\.[a-z]{2,}\.")]
    if _tel_required:
        _contact_checks.insert(0, ("title page: telephone", _tel_pat))
    for _lbl, _pat in _contact_checks:
        checks += 1
        if not re.search(_pat, txt):
            fails.append(f"{_lbl}: the title page carries no line matching "
                         f"{_pat!r}, which JFEc requires of the corresponding author")

    # Keywords are required; WHICH keywords is a per-journal choice, so the
    # check is that a keyword line exists with several entries, not that it
    # names any particular term.
    checks += 1
    _kw = re.search(r"Keywords:(?:</b>)?\s*([^<]{10,400})", _rawM)
    if not _kw or _kw.group(1).count(";") < 1:
        fails.append("the title page carries no keyword list with at least two "
                     "entries, which JFEc requires")

    # the contact block must not borrow the centred italic affiliation class,
    # which is what wrapped it mid-phrase and orphaned the address
    checks += 1
    if 'class="aff"><b>Corresponding author' in _rawM:
        fails.append("the corresponding-author block is styled with the centred "
                     "one-line affiliation class again, which wraps it mid-phrase")
    # keywords sit inside the abstract block, where the journal asks for them
    checks += 1
    _absblk = _rawM[_rawM.find('<div class="abs">'):]
    _absblk = _absblk[:_absblk.find("</div>")]
    if "Keywords:" not in _absblk:
        fails.append("the keywords are outside the abstract block; JFEc asks for "
                     "them with the abstract")

    # --- the housing derivative claim --------------------------------------
    # "No option chain is written on any of the twenty" is a claim about the
    # world, stated in the historical absolute.  CME lists futures on the
    # Case-Shiller composite and on ten metro indices, so a blanket denial that
    # any derivative exists is false; what is true, and what the comparison
    # actually needs, is that no options chain and no daily own-market implied
    # volatility index is currently listed for these targets.
    want("housing derivative claim is current, not absolute",
         "no options chain on any of the twenty is currently listed and no daily "
         "own-market implied-volatility index exists for them")
    want("housing futures acknowledged",
         "CME does list futures on the composite and on ten of the metro indices")
    for _abs in ("no option chain is written on any of the twenty",
                 "smoothed, unoptioned market"):
        forbid("absolute housing derivative claim", _abs)

    # --- the review's five wording repairs, each with its ban ------------
    # Every one of these was a statement the text could not support: a false
    # identity, a ceiling on a quantity that exceeds it, an absolute claim
    # about weak identification, a sentence contradicted by the subsection
    # under it, a cross-reference to a section the paper does not have, and a
    # universal quantifier with an unlabelled exception.  Each repair is
    # required and each original is banned, because a reverted edit is how
    # these came back the first time.
    for _lbl, _need, _ban in (
            ("4.4 benchmark is not the trailing mean of variance",
             "exponentiated using its own smearing factor and reconstructed using "
             "the median available at the forecast origin",
             "which makes it the trailing mean of the variance itself"),
            ("4.5 restoration point, not a ceiling",
             "a natural restoration reference point at 100%",
             "the ruler has a zero and a ceiling"),
            ("4.6 Fieller is one construction, not the only one",
             "the asymptotic Fieller-type inversion used here is one standard "
             "construction with that property rather than the only one",
             "the only coverage-correct answer available"),
            ("6 points at the subsection that does the fitting",
             "fits that compressed representation directly",
             "We have not built that, and we make no claim about how much it "
             "would recover"),
            ("claim table cross-reference",
             "converting it needs a decision problem the holder supplies",
             "(Sections 4.5, 13)"),
            ("4.2 names its exception",
             "Table 1 is the labelled exception",
             "This table, like every table here, is computed on the target-dated")):
        want(_lbl, _need)
        forbid(f"stale: {_lbl}", _ban)
    # the cross-reference ban is only meaningful if the paper really stops at 11
    _secs = [int(m) for m in re.findall(r'<h2 class="pb">(\d+)\.', _rawM)]
    checks += 1
    if not _secs:
        fails.append("the main paper has no numbered top-level sections to check "
                     "its own cross-references against")
    else:
        for _m in re.finditer(r"Sections? ([0-9]+(?:\.[0-9]+)?)"
                              r"(?: and ([0-9]+(?:\.[0-9]+)?))?", txt):
            for _g in _m.groups():
                if _g and "." not in _g and int(_g) > max(_secs):
                    checks += 1
                    fails.append(f"a cross-reference points at Section {_g}, but the "
                                 f"main paper ends at Section {max(_secs)}")

    # --- the own-options versus other-asset counts, against lab52 ---------
    # A reviewer read Table S10's DAX cell, z = -2.06, as contradicting the
    # claim that the foreign block is significant in none of eight own-options
    # cells.  Both statements are true and they are about different things: the
    # table's columns are the SEVEN-REGRESSOR specification, where three of the
    # eight own-options cells are significantly NEGATIVE, while the counts in
    # the text are the COMPRESSED one, where none of the eight reaches
    # significance at all.  Nothing said so.  All four counts are derived here
    # from lab52, which prints both specifications side by side, so the two can
    # never again be quoted as though they were one.
    _l52 = read_text(os.path.join(EXP, "lab52_compressed_foreign_options.txt"))
    _R52 = re.compile(r"^\s*([A-Z0-9]+)\s+(\d+)\s+[\d.]+\s+([-+][\d.]+)\s+"
                      r"([-+][\d.]+)\s+([-+][\d.]+)\s+([-+][\d.]+)\s*$", re.M)
    _rows52 = _R52.findall(_l52)
    _OWN = {"SPX", "DAX"}
    checks += 1
    if len(_rows52) != 32:
        fails.append(f"lab52: parsed {len(_rows52)} of 32 target-delay rows for the "
                     "own-options comparison")
    else:
        def _cnt(group, col, want_positive=None):
            zs = [float(r[col]) for r in _rows52 if (r[0] in _OWN) == group]
            if want_positive is None:
                return sum(abs(z) > 1.959963985 for z in zs)
            return sum((z > 1.959963985) if want_positive else (z < -1.959963985)
                       for z in zs)
        _own_sev = _cnt(True, 3)
        _own_cmp = _cnt(False if False else True, 5)
        _oth_cmp_pos = _cnt(False, 5, want_positive=True)
        _own_sev_neg = _cnt(True, 3, want_positive=False)
        checks += 1
        if _own_cmp != 0:
            fails.append(f"lab52: {_own_cmp} of the eight own-options cells are now "
                         "significant under the compressed block, but the papers say "
                         "none are")
        checks += 1
        if _own_sev != _own_sev_neg:
            fails.append(f"lab52: {_own_sev - _own_sev_neg} own-options cell(s) are "
                         "significantly POSITIVE in the seven-regressor specification; "
                         "Table S10's caption says none is")
        want("S10 caption names the specification",
             "Those two columns are the seven-regressor specification, not the "
             "compressed one the section's counts use")
        want("S10 negative-cell count",
             f"{spell(_own_sev_neg).capitalize()} of the eight own-options cells are "
             "significantly negative in this specification")
        want("the two counts share a specification",
             "using the compressed one-regressor peer block and the same pointwise "
             "criterion in both groups")
        want("other-asset count",
             f"adds significantly in {spell(_oth_cmp_pos)} of twenty-four cells")

    # --- the post-selection audit, against lab62 --------------------------
    # lab34 prices the search behind the HEADLINE and nothing else.  Everything
    # reported after the headline specification was fixed - four wide grids and
    # a third implied series - carried per-cell significance at the ordinary 5%
    # level with no family-wise control.  lab62 pools them and corrects; Table
    # S30 is rebuilt from its output, and the verdict sentence has to match the
    # counts rather than describe them.
    l62 = read_text(os.path.join(EXP, "lab62_post_selection.txt"))
    checks += 1
    if "PARSED NOTHING" in l62:
        fails.append("lab62: a secondary family parsed nothing, so the post-selection "
                     "audit is incomplete and Section S37 must not quote it")
    _F62 = re.compile(r"^\s*(\S.*?)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+[\d.]+\s*$", re.M)
    _fam = [(m.group(1), m.group(2), m.group(3), m.group(4))
            # rules are split on as whole LINES: a bare "=" * 40 needle cuts an
            # 96-character rule into empty pieces, which is how this parsed
            # nothing on its first run
            for m in _F62.finditer(re.split(
                r"\n=+\n", l62.split("A.  THE SECONDARY FAMILIES")[-1])[1])]
    _sv = re.compile(r"^\s*(\S.*?)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", re.M)
    _surv = {m.group(1): (m.group(2), m.group(3), m.group(4))
             for m in _sv.finditer(l62.split("C.  WHICH CELLS SURVIVE")[-1])}
    checks += 1
    if len(_fam) != 5:
        fails.append(f"lab62: parsed {len(_fam)} of 5 secondary families for Table S30")
    else:
        for _nm, _n, _k, _e in _fam:
            checks += 1
            if _nm not in _surv:
                fails.append(f"lab62: family '{_nm}' is in part A but not part C")
                continue
            _s5, _bh, _bo = _surv[_nm]
            if _s5 != _k:
                fails.append(f"lab62: family '{_nm}' reports {_k} significant cells in "
                             f"part A and {_s5} in part C")
            want(f"Table S30 {_nm[:34]}", f"{_n} {_k} {_e} {_bh} {_bo}")
    _P62 = {k: re.search(p, l62) for k, p in (
        ("pooled", r"secondary cells pooled\s+(\d+)"),
        ("unc", r"significant at 5% uncorrected\s+(\d+)"),
        ("exp", r"expected by chance if all were noise\s+([\d.]+)"),
        ("bh", r"surviving Benjamini-Hochberg\s+(\d+)"),
        ("bonf", r"surviving Bonferroni\s+(\d+)"))}
    checks += 1
    if not all(_P62.values()):
        fails.append("lab62 no longer prints the pooled correction Section S37 quotes")
    else:
        _pv = {k: m.group(1) for k, m in _P62.items()}
        want("Table S30 pooled row",
             f"pooled {_pv['pooled']} {_pv['unc']} {_pv['exp']} {_pv['bh']} "
             f"{_pv['bonf']}")
        want("S37 both readings",
             f"{spell(int(_pv['unc'])).capitalize()} nominally significant cells against "
             f"{_pv['exp']} expected")
        _lo = int(_pv["unc"]) - int(_pv["bh"])
        want("S37 exploratory count",
             f"only {spell(int(_pv['bh']))} of those {spell(int(_pv['unc']))} survive")
        want("S37 exploratory label",
             f"The {spell(_lo)} that do not are exploratory")
        # the audit only means something if the grids beat chance AND most
        # cells fail correction; if either stops being true the section is wrong
        checks += 1
        if not (int(_pv["unc"]) > float(_pv["exp"])):
            fails.append("lab62: the secondary grids no longer show more significant "
                         "cells than chance, so Section S37's first reading is false")
        checks += 1
        if int(_pv["bh"]) >= int(_pv["unc"]):
            fails.append("lab62: every nominally significant secondary cell now survives "
                         "correction, so Section S37's exploratory label is wrong")
    # the headline must stay OUT of the pool, and the section must say why
    want("S37 headline excluded",
         "the headline comparison is excluded from the pool")
    # Every audited table must carry the label, not just one of them.  A
    # single want() here passed a tamper test that stripped the label from one
    # of the two captions, because the other still satisfied it: the same
    # at-least-one weakness only_iv() exists to close.  Checked per table, in
    # the RAW source, so each caption is examined on its own.
    _AUDITED = ("S17", "S23")
    _rawA = read_text(_supp) if os.path.isfile(_supp) else ""
    for _tb in _AUDITED:
        checks += 1
        _i = _rawA.find(f"<b>Table {_tb}.</b>")
        if _i < 0:
            fails.append(f"Table {_tb} is named in Section S37's audit but is not in "
                         "the Internet Appendix")
            continue
        _cap = _rawA[_i:_rawA.find("</caption>", _i)]
        if "exploratory" not in _cap or "Section S37" not in _cap:
            fails.append(f"Table {_tb}'s caption does not carry Section S37's "
                         "exploratory label, though its cells are in the audit")

    # --- the marking experiment, against lab14 ----------------------------
    l14 = lab("lab14_appraisal_smoothing")
    for pct in ("64%", "71%", "70%"):
        want(f"lab14 rate {pct}", pct)
    # The blur arm's two endpoints and the clean-lag arm's, READ from lab14's
    # own table.  A literal "0.456" stood here, which froze a figure lab14's
    # narration carried from before Section 4.4's benchmark correction while
    # lab14's own table said 0.4782 - a check that enforced the stale value and
    # passed every round.  The appendix had copied the whole sentence.
    _R14 = re.compile(r"^\s*(\d+)\s+[\d.]+\s+([\d.]+)\s+-?\d+% \[[^\]]*\]\s+"
                      r"([\d.]+)\s+-?\d+% \[")
    _b14 = {}
    for _ln in read_text(os.path.join(EXP,
                                      "lab14_appraisal_smoothing.txt")).splitlines():
        _m = _R14.match(_ln)
        if _m:
            _b14[int(_m.group(1))] = (_m.group(2), _m.group(3))   # clean, blur
    checks += 1
    if len(_b14) < 8:
        fails.append(f"lab14: read {len(_b14)} of 8 mean-lag rows, so Section S15's "
                     f"two curves rest on a short read")
    else:
        _f14, _l14e = min(_b14), max(_b14)
        want("S15 gives the blur arm's two endpoints",
             f"R&sup2; falls from {_b14[_f14][1]} to {_b14[_l14e][1]} across a mean lag")
        want("S15 gives the clean-lag arm's two endpoints",
             f"against {_b14[_f14][0]} to +{_b14[_l14e][0]} under a clean lag")
        checks += 1
        if float(_b14[_l14e][1]) <= float(_b14[_l14e][0]):
            fails.append("Section S15 says blur is a much milder handicap than lag; "
                         "lab14 no longer agrees at the longest mean lag")
        forbid("S15's pre-correction blur curve",
               "R&sup2; falls from 0.499 to 0.456")

    # --- Table 1, against lab10 -------------------------------------------
    # This block exists because of a hole found late: Table 1 is the paper's
    # headline table and nothing here read the lab that prints it.  The two
    # figures below were checked against the PAPER only - the verifier asserted
    # that the text says 80%, never that any script produces 80% - which is the
    # same class of mistake as the sentence that claimed 348 checks while the
    # script made 414.  Every cell of Table 1 is now rebuilt from lab10's own
    # output and required to appear as a whole row, so a transposed digit
    # cannot pass by matching some other number elsewhere in the paper.
    l10 = lab("lab10_loss_scale")
    RATE = re.compile(r"^\s*(\d+)\s+(-?\d+)% \[\s*(-?\d+)%,\s*(-?\d+)%\]"
                      r"\s+(-?\d+)% \[\s*(-?\d+)%,\s*(-?\d+)%\]")
    SKILL = re.compile(r"^\s*(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+"
                       r"(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$")
    # The RATES come from lab10 part 1a, the fixed-b Fieller sets, not from
    # part 1's bootstrap table above it.  Table 1 and Table 2 both carry the
    # inverted sets now, and part 1's rows have the same SHAPE, so an unscoped
    # scan takes the bootstrap's and reports a correct table as unsupported.
    # The skill columns are unaffected and come from the whole output.
    rates, skill = {}, {}
    _l10whole = read_text(os.path.join(EXP, "lab10_loss_scale.txt"))
    _l10inv = _l10whole.split("1a. THE SAME RATES, INVERTED")[-1]
    RATEI = re.compile(r"^\s*(\d+)\s+[\d.]+\s+(-?\d+)% \[\s*(-?\d+)%,"
                       r"\s*(-?\d+)%\]\s+(-?\d+)% \[\s*(-?\d+)%,"
                       r"\s*(-?\d+)%\]")
    for ln in _l10inv.splitlines():
        m = RATEI.match(ln)
        if m:
            rates[int(m.group(1))] = m.groups()[1:]
    for ln in _l10whole.splitlines():
        m = SKILL.match(ln)
        if m:
            skill[int(m.group(1))] = m.groups()[1:]
    if len(skill) < 9 or len(rates) < 8:
        fails.append("lab10: could not parse the skill and rate tables Table 1 "
                     f"is built from ({len(skill)} skill rows, {len(rates)} rates)")
    for d in sorted(skill):
        checks += 1
        r2o, r2c, qo, qc = skill[d][0], skill[d][1], skill[d][2], skill[d][3]
        if d in rates:
            a, alo, ahi, q, qlo, qhi = rates[d]
            row = (f"{d} {r2o} {r2c} {a}% [{alo}, {ahi}] "
                   f"{qo} {qc} {q}% [{qlo}, {qhi}]")
        else:                # delta = 0 has no rate to report; the table says n/a
            row = f"{d} {r2o} {r2c} n/a {qo} {qc} n/a"
        if row.replace(MINUS, "-") not in txt:
            fails.append(f"Table 1 delta={d}: lab10 no longer supports the row "
                         f"'{row}'")
    # The simultaneous band has to be internally coherent, and the lab prints
    # the evidence for that rather than asserting it.  Bonferroni ignores the
    # dependence across delays; these ten columns are overlapping windows of one
    # series, so the bootstrap's critical value must come out BELOW it.  When it
    # does not, the studentisation and the resampling disagree about how much
    # dependence there is, which is exactly the bug that produced a critical
    # value of 9.54 against a Bonferroni 2.81.
    _l06 = lab("lab06_inference")
    _sup = re.search(r"sup-t critical value ([\d.]+)", _l06)
    _bon = re.search(r"Bonferroni would use ([\d.]+) for (\d+) delays", _l06)
    checks += 1
    if not (_sup and _bon):
        fails.append("lab06 no longer prints the sup-t and Bonferroni critical values "
                     "side by side")
    else:
        checks += 1
        if float(_sup.group(1)) >= float(_bon.group(1)):
            fails.append(f"lab06: the sup-t critical value {_sup.group(1)} is not below "
                         f"Bonferroni's {_bon.group(1)} for {_bon.group(2)} dependent "
                         f"delays, so the band's studentisation and its resampling "
                         f"disagree about the dependence")
        checks += 1
        if "is NOT expected" in _l06:
            fails.append("lab06 flags its own critical-value comparison as unexpected")

    # THE TRUNCATION LENGTH, and the three ways this project has got it wrong.
    #
    #   (i)   It was 2h, on the reasoning that the only dependence to respect
    #         was the h-day overlap.  lab58 measured 0.48 autocorrelation at
    #         lag 10 and the coverage study found 86% for a nominal 95%.
    #   (ii)  It became 8h, and the change reached the labs that imported it
    #         and missed the nine that spelled it out for themselves, which is
    #         how an appendix table came to carry a narrower interval than the
    #         headline for the same quantity.
    #   (iii) 8h is still too short at the LONG delays, because a delay-delta
    #         differential carries the delay's own overlap on top of the
    #         horizon's.  lab65 measures a Bartlett(40) kernel keeping only
    #         76-80% of the long-run variance at delta = 55.
    #
    # So the length is now a FUNCTION of the delay, and these checks pin all
    # three failures shut: one declaration, no private copies, and every
    # delay-indexed bootstrap actually asking for the delay's own length.
    _l02p = os.path.join(HERE, "labs", "lab02_delay_curve.py")
    _l02 = read_text(_l02p)
    _base = re.search(r"^BLOCK\s*=\s*(\d+)\s*\*\s*HORIZON", _l02, re.M)
    _cap = re.search(r"^BLOCK_CAP\s*=\s*(\d+)\s*\*\s*HORIZON", _l02, re.M)
    checks += 1
    if not _base or not _cap:
        fails.append("lab02 no longer declares BLOCK and BLOCK_CAP as multiples of "
                     "the horizon, so the truncation rule cannot be checked")
    else:
        _b, _c = int(_base.group(1)), int(_cap.group(1))
        checks += 1
        if _c <= _b:
            fails.append(f"lab02's BLOCK_CAP ({_c}h) does not exceed BLOCK ({_b}h), "
                         f"so block_for() can never widen at the long delays and the "
                         f"coverage shortfall lab65 measures is left uncorrected")
        # the rule itself, exercised rather than read
        checks += 1
        try:
            _ns = {}
            exec(compile("HORIZON = %d\n" % 1 +
                         "BLOCK = %d\nBLOCK_CAP = %d\n" % (_b, _c) +
                         re.search(r"^def block_for\b.*?^    return [^\n]*\n",
                                   _l02, re.M | re.S).group(0), "<rule>", "exec"), _ns)
            _bf = _ns["block_for"]
            if _bf(0) != _b:
                fails.append(f"lab02.block_for(0) is {_bf(0)}, not BLOCK ({_b}); the "
                             f"delay-0 figures would move and every published "
                             f"short-delay number with them")
            if _bf(10 ** 6) != _c:
                fails.append("lab02.block_for does not cap at BLOCK_CAP, so a long "
                             "delay can ask for more blocks than the sample has")
            if any(_bf(i) > _bf(i + 1) for i in range(0, 200)):
                fails.append("lab02.block_for is not monotone in the delay")
        except Exception as _e:                                    # noqa: BLE001
            fails.append(f"lab02.block_for could not be exercised: "
                         f"{type(_e).__name__}: {_e}")

    # No lab may keep a private copy of the length.  lab02 owns it; lab04 and
    # lab05 used to declare their own and are the reason this check exists.
    for _lp in sorted(glob.glob(os.path.join(HERE, "labs", "lab*.py"))):
        _nm = os.path.basename(_lp)
        if _nm == "lab02_delay_curve.py":
            continue
        _ls = read_text(_lp)
        checks += 1
        _own = re.findall(r"^(?:N_BOOT,\s*)?(?:BLOCK|BLK)\b[^=\n]*=\s*(?:\d+,\s*)?"
                          r"(\d+)\s*\*\s*HORIZON", _ls, re.M)
        if _own:
            fails.append(f"{_nm} declares its own bootstrap block as {_own[0]}h; the "
                         f"length is lab02's and must be imported, or it drifts from "
                         f"the argument that sets it")

    # Every statistic formed INSIDE a delay loop must be truncated at that
    # delay's length.  This is the check that would have caught failure (ii)
    # above, and it is the third version of it - the first two were not strong
    # enough, each in a way worth recording.
    #
    # A regex over the call text found only the direct calls.  It looked
    # straight past lab05's ci(), which wrapped the bootstrap one level down,
    # and so certified a file whose every interval truncated at the delay-0
    # length however long the delay was - the headline continuous-target table
    # among them.  So the reachability is computed transitively here: a
    # function that calls a bootstrap, or calls something that does, is itself
    # a place a delay has to arrive.
    #
    # Then accepting the loop variable ANYWHERE in the arguments was too
    # loose.  stats(yb5, fitrun(d, False), fitrun(d, True)) mentions d twice
    # and passes it as no bandwidth at all; it had in fact lost the argument
    # entirely and would have raised TypeError on the next full run.  So the
    # delay must arrive as a bare positional argument or as a delay/lag/block
    # keyword, and a mention nested inside another call does not count.
    # Reachability is computed ACROSS the labs, not within one file, and a
    # function counts as truncation-bearing if its body mentions a truncation
    # length at all - BLOCK, BLK, block, lag, HAC_LAG, band.
    #
    # Both halves of that were learned the hard way.  Keyed on a list of
    # primitive names, the check missed block_mean_ci() and boot_ci(), which
    # build their resample indices inline with rng.integers() and so call no
    # primitive at all.  Computed per file, it missed rel_improvement(), which
    # lab29 calls inside a delay loop but lab27 defines - the bearing set for
    # lab29 never contained it.  Between them those two blind spots hid the
    # VIX-tercile Fieller sets, which are quoted in the abstract.
    _TRUNC = {"BLOCK", "BLK", "block", "lag", "HAC_LAG", "BAND", "band"}
    _DGRID = re.compile(r"^[A-Z_]*DELAY[A-Z_]*$|^FINE$")
    _labfiles = sorted(glob.glob(os.path.join(HERE, "labs", "lab*.py")))

    def _idents(node):
        out = set()
        for _n in ast.walk(node):
            if isinstance(_n, ast.Name):
                out.add(_n.id)
            elif isinstance(_n, ast.Attribute):
                out.add(_n.attr)
        return out

    def _callees(node):
        out = set()
        for _n in ast.walk(node):
            if isinstance(_n, ast.Call):
                _f = _n.func
                if isinstance(_f, ast.Name):
                    out.add(_f.id)
                elif isinstance(_f, ast.Attribute):
                    out.add(_f.attr)
        return out

    # Bearing is computed PER LAB and then joined across labs through the
    # actual imports.  A single global map keyed on the bare function name was
    # tried first and is wrong in both directions: it reported lab18's
    # report(), which computes percentiles and touches no block at all, purely
    # because another lab defines a bearing function of the same name.  A
    # checker that cries wolf gets switched off, so the alias is resolved.
    _trees, _own, _bearlab = {}, {}, {}
    for _lp in _labfiles:
        try:
            _trees[_lp] = ast.parse(read_text(_lp))
        except SyntaxError as _e:                                  # noqa: BLE001
            fails.append(f"{os.path.basename(_lp)} does not parse: {_e}")
            continue
        _d = {}
        for _n in ast.walk(_trees[_lp]):
            if isinstance(_n, ast.FunctionDef):
                _i, _c = _d.setdefault(_n.name, (set(), set()))
                _i.update(_idents(_n))
                _c.update(_callees(_n))
        _own[_lp] = _d
        _b = {k for k, (_i, _c) in _d.items() if _i & _TRUNC}
        _g = True
        while _g:
            _g = False
            for _k, (_i, _c) in _d.items():
                if _k not in _b and (_c & _b):
                    _b.add(_k)
                    _g = True
        _bearlab[_lp] = _b

    # DELAY-AWARE is not the same as delay-RECEIVING, and conflating them was
    # a false negative that hid the paper's tercile intervals.  lab39's
    # boot(d, s, rng) takes the delay as its first argument and used it to
    # select the forecast while truncating at a fixed block - so a rule that
    # asked only "is the delay passed?" saw a correct call.  What has to be
    # true is that the delay reaches the BLOCK.  A function qualifies only if
    # its own body computes block_for(), or it takes a block/lag parameter
    # that the caller can set.
    _aware = set()
    _selfband = set()
    for _lp2 in _labfiles:
        if _lp2 not in _trees:
            continue
        _s2 = read_text(_lp2)
        for _n in ast.walk(_trees[_lp2]):
            if not isinstance(_n, ast.FunctionDef):
                continue
            _body = ast.get_source_segment(_s2, _n) or ""
            _params = {_a.arg for _a in _n.args.args} | \
                      {_a.arg for _a in _n.args.kwonlyargs}
            if "block_for(" in _body or (_params & {"block", "lag", "band"}):
                _aware.add(_n.name)
            # A function that derives its OWN bandwidth from the sample size -
            # fixed-b inference does exactly that - needs no delay passed to
            # it, because it is not using a delay-dependent truncation at all.
            # Without this the guard would demand the delay-dependent rule be
            # threaded into the very calls that were converted away from it,
            # which is the guard enforcing a design the paper has replaced.
            if "har_band(" in _body:
                _selfband.add(_n.name)

    # module stem -> path, so an import can be followed
    _stem = {os.path.basename(x)[:-3]: x for x in _labfiles}

    def _visible(lp):
        """Bearing names reachable from lab `lp`: its own, plus imported ones."""
        src = read_text(lp)
        out = set(_bearlab.get(lp, ()))
        for _m in re.finditer(r"^\s*import\s+(lab\w+)\s+as\s+(\w+)\s*$",
                              src, re.M):
            _t = _stem.get(_m.group(1))
            if _t:
                out |= _bearlab.get(_t, set())
        for _m in re.finditer(r"^\s*from\s+(lab\w+)\s+import\s+([^\n]+)$",
                              src, re.M):
            _t = _stem.get(_m.group(1))
            if not _t:
                continue
            _want = {x.strip().split(" as ")[0].strip()
                     for x in _m.group(2).split(",")}
            out |= (_want & _bearlab.get(_t, set()))
        return out

    for _lp in _labfiles:
        _nm = os.path.basename(_lp)
        if _lp not in _trees:
            continue
        _src = read_text(_lp)
        if re.search(r"^BLOCK_(?:M|Q)\s*=|^N_BOOT, BLOCK_(?:M|Q)\s*=", _src, re.M):
            continue                      # a different sampling frequency
        _vis = _visible(_lp)
        _miss = set()
        for _node in ast.walk(_trees[_lp]):
            if not isinstance(_node, ast.For) or \
                    not isinstance(_node.target, ast.Name):
                continue
            _nms = {_x.id for _x in ast.walk(_node.iter)
                    if isinstance(_x, ast.Name)}
            if not any(_DGRID.match(_x) for _x in _nms):
                continue
            _dv = _node.target.id
            # A BANDWIDTH SWEEP inside a delay loop is exempt, and on
            # principle rather than by file name.  lab63 and lab65 exist to
            # price the truncation length: their inner loop runs over BANDS or
            # SWEEP and evaluates the same differential at every length, so
            # demanding they pass the delay's own length would forbid the one
            # measurement that justifies the rule.  The exemption is keyed on
            # the sweep variable, so a lab that simply forgot to pass a length
            # is still caught.
            # Variables ASSIGNED from block_for() inside the loop count as
            # passing it: `rule = L.block_for(d)` then `f(g, rule)` is the
            # rule reaching the estimator, and requiring the call itself to
            # spell block_for() reported a line that was already correct.
            _swept = {"REF_LAG"}
            for _sub in ast.walk(_node):
                if isinstance(_sub, ast.Assign):
                    _seg2 = ast.get_source_segment(_src, _sub) or ""
                    if "block_for(" in _seg2:
                        for _tg in _sub.targets:
                            if isinstance(_tg, ast.Name):
                                _swept.add(_tg.id)
            for _sub in ast.walk(_node):
                if isinstance(_sub, ast.For) and isinstance(_sub.target, ast.Name):
                    _sn = {_x.id for _x in ast.walk(_sub.iter)
                           if isinstance(_x, ast.Name)}
                    if _sn & {"BANDS", "SWEEP", "BLOCKS"}:
                        _swept.add(_sub.target.id)
                if isinstance(_sub, ast.comprehension) and \
                        isinstance(_sub.target, ast.Name):
                    _sn = {_x.id for _x in ast.walk(_sub.iter)
                           if isinstance(_x, ast.Name)}
                    if _sn & {"BANDS", "SWEEP", "BLOCKS"}:
                        _swept.add(_sub.target.id)
            for _c in ast.walk(_node):
                if not isinstance(_c, ast.Call):
                    continue
                _f = _c.func
                _cn = (_f.id if isinstance(_f, ast.Name) else
                       _f.attr if isinstance(_f, ast.Attribute) else None)
                if _cn not in _vis or _cn in _selfband:
                    continue
                _seg = ast.get_source_segment(_src, _c) or ""
                if "block_for(" in _seg or "BLOCK_Q" in _seg or "BLOCK_M" in _seg:
                    continue
                # the delay must reach the truncation, so the callee has to
                # be delay-aware as well as delay-receiving
                _ok = (_cn in _aware or _cn in _swept) and any(
                    isinstance(_a, ast.Name) and _a.id in ({_dv} | _swept)
                    for _a in _c.args)
                for _k in _c.keywords:
                    if _k.arg in ("delay", "lag", "block"):
                        _ks = ast.get_source_segment(_src, _k.value) or ""
                        if "block_for(" in _ks or _ks.strip() == _dv:
                            _ok = True
                if not _ok:
                    _miss.add((_c.lineno, _cn))
        checks += 1
        if _miss:
            fails.append(f"{_nm}: {sorted(_miss)} form a statistic inside a delay "
                         f"loop without passing that delay, so they truncate at "
                         f"the delay-0 length while the headline truncates longer")

    # THE FIXED-DELAY CASE.  The check above scans delay LOOPS, which is the
    # shape nearly every lab has - and it is blind to a lab that works at ONE
    # delay held in a module constant, because such a lab has no loop to find.
    # That blind spot hid lab45 and lab48, which are where the paper's
    # headline rate, its effective-age reading and its VIX-tercile intervals
    # come from: the abstract's own numbers were still truncated at the
    # delay-0 length after everything else had moved.
    #
    # Scope is deliberately one level deep rather than the transitive closure
    # used above.  Closing over every caller reaches main() and panel(), which
    # are bearing only because something inside them eventually bootstraps,
    # and demanding a truncation argument of main() is meaningless.
    _DFIX = re.compile(r"^(DELTA|DELAY|HEADLINE_DELAY)\s*=\s*\d+", re.M)
    for _lp in _labfiles:
        _nm = os.path.basename(_lp)
        if _lp not in _trees:
            continue
        _src = read_text(_lp)
        if re.search(r"^BLOCK_(?:M|Q)\s*=|^N_BOOT, BLOCK_(?:M|Q)\s*=", _src, re.M):
            continue
        _m = _DFIX.search(_src)
        if not _m:
            continue
        _dname = _m.group(1)
        # The resampling primitives themselves.  Named here rather than
        # reusing the transitive set above, which by design reaches every
        # caller up to main().
        _PRIM = {"block_boot", "blocks_of", "hac_se", "_hac", "hac", "hac_cov"}
        _miss = set()
        for _c in ast.walk(_trees[_lp]):
            if not isinstance(_c, ast.Call):
                continue
            _f = _c.func
            _cn = (_f.id if isinstance(_f, ast.Name) else
                   _f.attr if isinstance(_f, ast.Attribute) else None)
            if _cn not in _PRIM:
                continue                    # only the primitive calls themselves
            _seg = ast.get_source_segment(_src, _c) or ""
            if "block_for(" in _seg or "BLOCK_Q" in _seg or "BLOCK_M" in _seg \
                    or "REF_LAG" in _seg:
                continue
            _miss.add(_c.lineno)
        checks += 1
        if _miss:
            fails.append(f"{_nm} holds its delay in the constant {_dname} and "
                         f"bootstraps at line(s) {sorted(_miss)} without "
                         f"block_for({_dname}), so it truncates at the delay-0 "
                         f"length while the rest of the paper truncates longer")

    # The converse of the exemption above: a lab whose bootstrap block is a
    # monthly or quarterly constant must not call block_for() anywhere, because
    # the two are lengths in different units and mixing them silently produces
    # a block three to seven years long.
    for _lp in sorted(glob.glob(os.path.join(HERE, "labs", "lab*.py"))):
        _nm = os.path.basename(_lp)
        _src = read_text(_lp)
        _code = "\n".join(_l.split("#", 1)[0] for _l in _src.splitlines())
        checks += 1
        if re.search(r"\bBLOCK_(?:M|Q)\s*=", _code) and "block_for(" in _code:
            fails.append(f"{_nm} defines a monthly or quarterly bootstrap block and "
                         f"also calls block_for(), which returns trading days; the "
                         f"two are different units and one of them is wrong")

    # Section 4.6's figures, read from lab66.  These replace four checks that
    # pinned lab65's retained-variance numbers, and those are gone from the
    # paper on purpose: under fixed-b the bandwidth needs no justification, so
    # the retained fraction no longer does any work - which is fortunate,
    # because that fraction turned out to depend on an arbitrary reference
    # truncation and is documented in lab65 as unreliable.
    _l66p = os.path.join(EXP, "lab66_har_alternatives.txt")
    checks += 1
    if not os.path.isfile(_l66p):
        fails.append("expected_output/lab66_har_alternatives.txt is missing, so "
                     "Section 4.6's coverage figures rest on nothing")
    else:
        _l66 = read_text(_l66p)
        _u = re.search(r"THE VALUE USED: b = ([\d.]+), M = (\d+) at n = (\d+), "
                       r"cv = ([\d.]+)", _l66)
        checks += 1
        if not _u:
            fails.append("lab66 no longer reports the critical value the labs use")
        else:
            _b66, _m66, _n66, _cv66 = _u.groups()
            want("4.6 fixed-b setting",
                 f"fixed b = M/n = {_b66}")
            want("4.6 fixed-b bandwidth and critical value",
                 f"bandwidth of {int(_m66)} days and a two-sided critical value of "
                 f"{_cv66} rather than 1.96")
        # the coverage each procedure reaches, at the three strengths lab66 runs
        def _cov66(tag, band):
            out = []
            for _blk in re.split(r"\n\s*tau = ", _l66)[1:]:
                _m = re.search(r"^\s*" + re.escape(tag) + r"\s+" +
                               (str(band) if band else "none") +
                               r"\s+\S+\s+([\d.]+)%", _blk, re.M)
                if _m:
                    out.append(_m.group(1))
            return out
        _fb = _cov66("fixed-b", int(_m66) if _u else 243)
        # raw text, not a pattern: _cov66 escapes what it is given, so a
        # pre-escaped string would have its backslashes escaped again and
        # match nothing.  The symptom was zero rows read and a check that
        # reported the lab as broken when the lab was fine.
        _nm = _cov66("normal (the paper's)", 80)
        checks += 1
        if len(_fb) != 3 or len(_nm) != 3:
            fails.append(f"lab66: could not read both coverage rows "
                         f"({len(_fb)} fixed-b, {len(_nm)} normal)")
        else:
            want("4.6 fixed-b coverage",
                 f"reaches {_fb[0]}%, {_fb[1]}% and {_fb[2]}% at")
            want("4.6 coverage it replaces",
                 f"against {_nm[0]}%, {_nm[1]}% and {_nm[2]}% for")
            # Section 4.6 now says WHY b = 0.05 rather than presenting it as
            # given, and the reason is that lab66 cannot separate it from
            # b = 0.033.  That comparison is quoted, so it is read: both
            # coverage triples, the largest gap between them, the Monte Carlo
            # standard error at lab66's own replication count, and the width
            # penalty.  A referee's objection was that the paper adopted the
            # value its own summary argues against; the answer is only honest
            # while these numbers are the lab's.
            _alt = _cov66("fixed-b", 160)
            checks += 1
            if len(_alt) != 3:
                fails.append(f"lab66: could not read the b = 0.033 coverage row "
                             f"({len(_alt)} of 3)")
            else:
                want("4.6 alternative b coverage",
                     f"At b = 0.033 coverage is {_alt[0]}%, {_alt[1]}% and "
                     f"{_alt[2]}%")
                _gap = max(abs(float(x) - float(y))
                           for x, y in zip(_fb, _alt))
                checks += 1
                if round(_gap, 1) != 0.4:
                    fails.append(f"Section 4.6 says the largest gap between the two "
                                 f"b values is four tenths of a point; lab66 now "
                                 f"makes it {_gap:.1f}")
            _reps = re.search(r"(\d+) replications per coverage cell", _l66)
            checks += 1
            if not _reps:
                fails.append("lab66 no longer states its replication count, so the "
                             "Monte Carlo error Section 4.6 quotes rests on nothing")
            else:
                _se = 100 * math.sqrt(2 * 0.95 * 0.05 / int(_reps.group(1)))
                checks += 1
                if round(_se, 1) != 0.9:
                    fails.append(f"Section 4.6 quotes a Monte Carlo standard error of "
                                 f"nine tenths of a point on a difference; at "
                                 f"{_reps.group(1)} replications it is {_se:.2f}")
                want("4.6 names the replication count behind that error",
                     f"at {spell(int(_reps.group(1)) // 100 * 100).replace(' and ', ' ')} "
                     f"replications".replace("one thousand two hundred",
                                             "twelve hundred"))
            # the widths the choice is actually made on
            def _wid66(band):
                out = []
                for _blk in re.split(r"\n\s*tau = ", _l66)[1:]:
                    # coverage, THEN bounded-only coverage, THEN the width.
                    # The pattern used to take the first number after the first
                    # percentage and silently started reading the new
                    # bounded-only column when it arrived, which turned a width
                    # penalty of 0.7 points into 40.
                    _m = re.search(r"^\s*fixed-b\s+" + str(band) +
                                   r"\s+\S+\s+[\d.]+%\s+[\d.]+%\s+([\d.]+)",
                                   _blk, re.M)
                    if _m:
                        out.append(float(_m.group(1)))
                return out
            _w243, _w160 = _wid66(243), _wid66(160)
            checks += 1
            if len(_w243) != 3 or len(_w160) != 3:
                fails.append("lab66: could not read both width columns")
            else:
                _pen = [round(100 * (a - b), 1) for a, b in zip(_w243, _w160)]
                want("4.6 width penalty of the wider b",
                     f"by {_pen[0]}, {_pen[1]} and {_pen[2]} points of width")
                checks += 1
                if any(x <= 0 for x in _pen):
                    fails.append(f"Section 4.6 says b = 0.05 gives the WIDER set at "
                                 f"all three strengths; lab66 now makes the "
                                 f"differences {_pen}")
            want("4.6 says the evidence does not make the choice",
                 "the choice is not one this evidence can make")
            # Section 4.6 now says WHICH of the three coverage figures applies,
            # by quoting the measured tau ranges.  Both are computed, and the
            # claim that no cell sits at either extreme of the study must hold.
            _t21b = sorted(float(_m) for _m in re.findall(
                r"\n\s+\d+\s+-?\d+%\s+([\d.]+)\s+\[",
                read_text(os.path.join(EXP, "lab21_stronger_inference.txt"))))
            _tfb = tercile_fieller()
            _ttb = sorted(float(_tfb[_d][_s][3]) for _d in _tfb for _s in _tfb[_d])
            checks += 1
            if len(_t21b) != 6:
                fails.append(f"lab21: read {len(_t21b)} of 6 tau values for the "
                             f"range Section 4.6 quotes")
            else:
                want("4.6 tau range across the delay grid",
                     f"from {_t21b[0]} to {_t21b[-1]} across the delay grid")
            want("4.6 tau range across the market states",
                 f"from {_ttb[0]} to {_ttb[-1]} across the market states")
            _lo_t, _hi_t = min(_t21b + _ttb), max(_t21b + _ttb)
            _taus66 = [float(_m) for _m in re.findall(r"\n\s*tau = ([\d.]+)", _l66)]
            checks += 1
            if not _taus66:
                fails.append("lab66 no longer states the identification strengths it "
                             "runs, so Section 4.6's 'middle of the three' is unmoored")
            elif _lo_t <= min(_taus66) or _hi_t >= max(_taus66):
                fails.append(f"Section 4.6 says no cell sits at the coverage study's "
                             f"extremes of {min(_taus66)} and {max(_taus66)}; the "
                             f"measured range is {_lo_t} to {_hi_t}")
            want("4.6 says which figure applies to the middle group",
                 "the applicable figure for all of those is the middle of the three")
            want("4.6 says which figure applies to the calm rungs",
                 "for those the applicable figure is the strongest of the three")
            want("4.6 quotes the strongest coverage figure",
                 f"the strongest of the three, {_fb[0]}%")
            want("4.6 claims nothing weaker than the middle",
                 "Nothing this paper claims rests on a set weaker than the middle "
                 "strength")
            # the bounded-set answer to "coverage bought by declining to
            # identify".  lab66 measures it; before this column existed the
            # objection could only be argued about, and the arithmetic bound a
            # reader can derive from the unbounded fraction alone (assuming an
            # unbounded set always covers) is far below what is actually
            # measured, because most of them are two half-lines and can miss.
            def _col66(tag, band, idx):
                out = []
                for _blk in re.split(r"\n\s*tau = ", _l66)[1:]:
                    _m = re.search(r"^\s*" + re.escape(tag) + r"\s+" +
                                   (str(band) if band else "none") +
                                   r"\s+\S+" + r"\s+([\d.]+)%" * 2 +
                                   r"\s+[\d.]+\s+([\d.]+)%", _blk, re.M)
                    if _m:
                        out.append(_m.group(idx))
                return out
            _bnd = _col66("fixed-b", int(_m66) if _u else 243, 2)
            _unb = _col66("fixed-b", int(_m66) if _u else 243, 3)
            _unb_n = _col66("normal (the paper's)", 80, 3)
            checks += 1
            if len(_bnd) != 3 or len(_unb) != 3 or len(_unb_n) != 3:
                fails.append(f"lab66: could not read the bounded-only and unbounded "
                             f"columns ({len(_bnd)}, {len(_unb)}, {len(_unb_n)})")
            else:
                want("4.6 bounded-set coverage at the two weaker strengths",
                     f"{_fb[1]}% against {_bnd[1]}% at the middle strength and "
                     f"{_fb[2]}% against {_bnd[2]}% at the weakest")
                # and the claim that the two agree to within a tenth must hold
                _worst_b = max(abs(float(a) - float(b))
                               for a, b in zip(_fb, _bnd))
                checks += 1
                if _worst_b > 0.1:
                    fails.append(f"Section 4.6 says total and bounded-set coverage "
                                 f"agree to within a tenth of a point; lab66 now "
                                 f"differs by {_worst_b:.1f}")
                # the unbounded fractions the paragraph describes in words
                checks += 1
                if not (0.60 <= float(_unb[2]) / 100 <= 0.70
                        and 0.55 <= float(_unb_n[2]) / 100 <= 0.65):
                    fails.append(f"Section 4.6 says two sets in three come back "
                                 f"unbounded against three in five before; lab66 now "
                                 f"gives {_unb[2]}% and {_unb_n[2]}%")
                want("4.6 answers the buy-coverage objection",
                     "Declining to identify is not, on this evidence, what buys the "
                     "coverage")
            # Part D: the generator's correlation, measured on the real
            # differentials by lab21 and then priced.  Both the measured values
            # and the coverage move are read, and the paper's claim that the
            # assumption is not load-bearing must be the lab's verdict, not the
            # author's hope.
            _cset = re.search(r"generator sets that correlation to ([\d.]+)", _l66)
            _cmove = re.search(r"largest move in coverage between the assumed "
                               r"[\d.]+ and the measured values: ([\d.]+) points",
                               " ".join(_l66.split()))
            _l21c = " ".join(read_text(os.path.join(
                EXP, "lab21_stronger_inference.txt")).split())
            _cmeas = re.search(r"mean over the grid: ([\d.]+), at delta = 55: "
                               r"([\d.]+)", _l21c)
            checks += 1
            if not (_cset and _cmove and _cmeas):
                fails.append("lab66 Part D or lab21's correlation block is missing, "
                             "so Section 4.6's correlation paragraph rests on nothing")
            else:
                want("4.6 the correlation the generator assumes",
                     f"sets that correlation to {_cset.group(1)}")
                want("4.6 the correlation actually measured",
                     f"it is {_cmeas.group(2)} at eleven weeks and "
                     f"{_cmeas.group(1)} averaged over the delay grid")
                # plain() collapses the source's line breaks, so the needle
                # must not carry one.
                want("4.6 what the assumption is worth",
                     f"moves coverage by at most {_cmove.group(1)} points")
                checks += 1
                if "the assumption is not load-bearing" not in _l66:
                    fails.append("lab66 Part D no longer finds the correlation "
                                 "assumption harmless, but Section 4.6 says it is")
            # and Part D discloses that its own baseline row is a second
            # estimate of Part B's, on the same discipline lab63 follows.
            _pd = re.findall(r"largest gap is ([\d.]+) standard errors", _l66)
            checks += 1
            if not _pd:
                fails.append("lab66 Part D no longer reconciles its baseline row "
                             "with Part B's, so the file reports one cell twice "
                             "without saying so")
            elif any(float(_x) >= 3.0 for _x in _pd):
                fails.append(f"lab66's two estimates of one cell disagree by "
                             f"{max(_pd)} standard errors")
            forbid("4.6 presents b as chosen by the evidence",
                   "the coverage study selects b = 0.05")
        # Section 4.6's claim about WHAT the remaining gap does with length,
        # read off lab63's own sweep.  The sentence used to say the shortfall
        # "stops shrinking once the kernel is long enough that truncation bias
        # is gone", which is true of the strongest identification row and false
        # of the two weaker ones, where coverage is still rising at 160 by 1.7
        # and 1.9 points.  A claim that is true of one row and false of two is
        # the kind a reader checks, so it is now read from all three.
        _l63c = read_text(os.path.join(EXP, "lab63_fieller_coverage.txt"))
        _sweep = {}
        for _m in re.finditer(r"^\s*([\d.]+)\s+(\d+)\s+[\d.]+%\s+([\d.]+)%",
                              _l63c, re.M):
            _sweep.setdefault(_m.group(1), {})[int(_m.group(2))] = float(_m.group(3))
        checks += 1
        if len(_sweep) != 3 or any(len(v) != 3 for v in _sweep.values()):
            fails.append(f"lab63: could not read the coverage-against-length sweep "
                         f"({ {k: len(v) for k, v in _sweep.items()} })")
        else:
            # lab-to-lab agreement, which nothing here used to check.  lab63's
            # two parts and lab66 all estimate coverage of the SAME cell -- the
            # paper's own procedure at bandwidth 80 -- on independent draws, and
            # Section 4.6 quotes lab63 in one sentence and lab66 in the next.
            # Three numbers for one quantity is what a referee tests, so the
            # three are compared here against lab63's own stated Monte Carlo
            # tolerance.  A disagreement beyond it means the labs are not
            # measuring the same thing and the section is quoting two of them.
            _rec = re.search(r"largest gap is ([\d.]+) standard errors", _l63c)
            checks += 1
            if not _rec:
                fails.append("lab63 no longer reconciles its two parts, so the two "
                             "coverage figures Section 4.6 quotes are unexplained")
            elif float(_rec.group(1)) >= 3.0:
                fails.append(f"lab63's two parts disagree by "
                             f"{_rec.group(1)} standard errors about the same cell")
            _se63 = re.search(r"on a DIFFERENCE\s+between two of them, (\d+\.\d+)",
                              " ".join(_l63c.split()))
            checks += 1
            if not _se63:
                fails.append("lab63 no longer prints the Monte Carlo error on a "
                             "difference, so no tolerance is stated for any of this")
            elif _nm and len(_nm) == 3:
                _tol = 3.0 * float(_se63.group(1))
                _pairs = [(_sweep[_k][80], float(_v))
                          for _k, _v in zip(sorted(_sweep, key=float, reverse=True),
                                            _nm)]
                for _i, (_a, _b) in enumerate(_pairs):
                    checks += 1
                    if abs(_a - _b) > _tol:
                        fails.append(f"lab63 and lab66 disagree about coverage of "
                                     f"the same cell by {abs(_a - _b):.1f} points, "
                                     f"beyond the {_tol:.1f} lab63 allows")
            want("4.6 the two studies are independent replications",
                 "on independent draws")
            # "the bootstrap reaches about 92%" appears in three places across
            # the two documents and was a literal in all three.  It is the
            # figure that licenses reporting the secondary grids on the
            # bootstrap at all, so it is the last figure that should be typed.
            # Read from lab63's boot column, over the identification strengths
            # that bracket the tau this paper's own cells actually have.
            _bootcov = {}
            for _m in re.finditer(r"^\s*([\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s+"
                                  r"[\d.]+%", _l63c, re.M):
                _bootcov[float(_m.group(1))] = float(_m.group(3))
            checks += 1
            if len(_bootcov) < 4:
                fails.append(f"lab63: read {len(_bootcov)} bootstrap coverage rows, "
                             f"so the 'about 92%' both documents quote is unmoored")
            else:
                _tlo, _thi = min(_t21b + _ttb), max(_t21b + _ttb)
                # strictly inside the measured range, with no padding.  A
                # +/-2 window let in tau = 1.0 and 1.5, where the bootstrap
                # covers 96-97% only because its intervals there are three to
                # fourteen units wide; averaging those in moved the figure to
                # 94% and would have had the paper claim coverage the grids it
                # describes do not have.
                _brack = [v for k, v in _bootcov.items() if _tlo <= k <= _thi]
                checks += 1
                if not _brack:
                    fails.append(f"lab63 runs no identification strength near the "
                                 f"{_tlo}-{_thi} this paper's cells have, so its "
                                 f"bootstrap figure does not apply to them")
                else:
                    _b92 = round(sum(_brack) / len(_brack))
                    for _lbl, _need in (
                            ("4.5 the bootstrap's coverage, for the grids it is "
                             "still used on",
                             f"cover about {_b92}% rather than 95%"),
                            ("1 the bootstrap's coverage",
                             f"the bootstrap reaches about {_b92}%"),
                            ("S2 the bootstrap's coverage",
                             f"both of the earlier procedures reach about {_b92}%")):
                        want(_lbl, _need)
            _all = [v for row in _sweep.values() for v in row.values()]
            want("4.6 states the coverage range over the lengths tried",
                 f"from {min(_all)}% to {max(_all)}% for a nominal 95%")
            _flat = [k for k, row in _sweep.items() if row[160] <= row[80]]
            _rising = [k for k, row in _sweep.items() if row[160] > row[80]]
            checks += 1
            if len(_flat) != 1 or len(_rising) != 2:
                fails.append(f"Section 4.6 says the gap stops closing at one "
                             f"identification strength and is still closing at two; "
                             f"lab63 now makes it {len(_flat)} and {len(_rising)}")
            else:
                _s = _sweep[_flat[0]]
                want("4.6 the strength where the gap stops closing",
                     f"{_s[80]}% at both eighty and a hundred and sixty")
                _r = sorted(_sweep[k][160] for k in _rising)
                want("4.6 the two strengths still closing at 160",
                     f"closing at a hundred and sixty, {_r[0]}% and {_r[1]}%")
            forbid("4.6 stale 'truncation bias is gone' claim",
                   "the shortfall\nstops shrinking once the kernel is long enough")
            forbid("4.6 stale truncation-bias framing",
                   "stops shrinking once the kernel is long enough that truncation "
                   "bias is gone")
        # and the taper reading, which is what makes the sweep make sense: the
        # generator's dependence STOPS at a stated lag, so a longer kernel
        # truncates nothing and the loss that remains is the Bartlett weighting.
        _k63 = re.search(r"truncated-MA\(phi=[\d.]+, K=(\d+)\)", _l63c)
        checks += 1
        if not _k63:
            fails.append("lab63 no longer states its generator's dependence length, "
                         "so Section 4.6's taper reading rests on nothing")
        else:
            want("4.6 names the generator's dependence length",
                 f"dependence stops at lag {spell(int(_k63.group(1)))}")
            want("4.6 taper, not truncation", "what it does is taper")

    # The companion note's own checks.  The guard is not decoration: without it
    # these run against an empty document and every cwant() below passes
    # vacuously, which is the quietest way a checker can stop checking.  It was
    # lost when the block above it was replaced, and the symptom was that the
    # note's checks became conditional on an unrelated file existing.
    if companion:
        _l02b = lab("lab02b_threshold_ceiling")
        _sh = re.search(r"by ([\d.]+) points on average across all ten delays", _l02b)
        # All three gaps, not two.  The note used to quote the third ("3.05
        # above the non-centred one") without printing the column it was
        # measured against, so a reader checking it against Table 3's two
        # printed columns got 2.69 and concluded the table was wrong.  Table 3
        # now prints all three, and all three are pinned here.
        _ex = re.search(r"([-+][\d.]+) points above the centred benchmark,\s+"
                        r"([-+][\d.]+) above the\s+non-centred sign-rule accuracy, "
                        r"and ([-+][\d.]+) above the non-centred\s+Bayes accuracy",
                        _l02b, re.I)
        _rg = re.search(r"measured accuracy runs from (\d+)% to (\d+)% of the Bayes "
                        r"benchmark, exceeding it at (\d+) of 10 delays", _l02b)
        _bg = re.search(r"worth at most ([\d.]+) accuracy points over the\s+zero cut, "
                        r"at delta = (\d+), and at most ([\d.]+) points", _l02b)
        checks += 1
        if not (_sh and _ex and _bg and _rg):
            fails.append("lab02b no longer prints the non-centred benchmark the "
                         "companion note quotes")
        else:
            _c = norm_iv(plain(companion))
            for _lbl, _n in (
                    ("note centring shift",
                     f"moves the sign rule's own accuracy by {_sh.group(1)} "
                     f"points on average"),
                    ("note excess centred",
                     f"sits {_ex.group(1).lstrip('+')} points above the centred column"),
                    ("note excess non-centred sign rule",
                     f"{_ex.group(2).lstrip('+')} above the non-centred same-sign column"),
                    ("note excess best cut",
                     f"{_ex.group(3).lstrip('+')} above the best-cut benchmark"),
                    # The range the note quotes for measured-over-benchmark had
                    # no check on it and went stale: Section 5 said "between
                    # 90% and 106%" while its own Table 3 ran from 84.6% and
                    # its own abstract said 85-106%.  Two of the ten rows sat
                    # below the figure the prose claimed was the floor.  Both
                    # ends now come out of the lab.
                    ("note benchmark range",
                     f"sit between {_rg.group(1)}% and {_rg.group(2)}% of the "
                     f"benchmark"),
                    ("note benchmark exceedance",
                     f"above it at {spell(int(_rg.group(3)))} of the ten delays"),
                    ("note bayes gain near",
                     f"worth at most {_bg.group(3)} accuracy points at any delay out "
                     f"to three weeks"),
                    ("note bayes gain far",
                     f"rising to {_bg.group(1)} points at {DELTA}= {_bg.group(2)}")):
                # cwant() rather than a hand-rolled comparison, because cwant
                # records what it demanded for build_registry.py.  Rolled by
                # hand, these seven checks were invisible to the registry, so
                # every figure they pin counted as an unlinked literal and the
                # ratchet rose whenever one was added - penalising the addition
                # of a check.
                cwant(_lbl, _n)
            checks += 1
            if "heavier tails than the normal approximation allows, which makes" in _c:
                fails.append("the companion note has gone back to explaining the "
                             "excess over its Gaussian benchmark by heavy tails, "
                             "which lab02b measures and does not support")
            checks += 1
            if "an approximation whose error we have not attributed" not in _c:
                fails.append("the companion note no longer says the excess over its "
                             "Gaussian benchmark is unattributed, which is what "
                             "lab02b found")

    # The compressed-block cell count is quoted in four places across the two
    # documents - the abstract, Section 9, the claim map and Section S13 - and
    # it moved when the HAC bandwidth was corrected.  Three of the four went on
    # saying sixteen while lab52 said thirteen, because each was its own
    # sentence and nothing tied them to the lab or to each other.  All four are
    # now required to agree with the lab, in words.
    _fc52 = re.search(r"another index's mean \+?-?[\d.]+ [\d.]+ (\d+) of (\d+)",
                      " ".join(lab("lab52_compressed_foreign_options").split()))
    _W = {"0": "none", "8": "eight", "12": "twelve", "13": "thirteen",
          "14": "fourteen", "15": "fifteen", "16": "sixteen", "24": "twenty-four"}
    # the phrase Table 7, the abstract, Section 9 and Section S13 must all use
    CELLS = (f"{_W.get(_fc52.group(1))} of {_W.get(_fc52.group(2))}"
             if _fc52 and _W.get(_fc52.group(1)) and _W.get(_fc52.group(2))
             else None)
    checks += 1
    if not _fc52:
        fails.append("lab52 no longer prints the compressed foreign-options cell count")
    else:
        _a, _b = _W.get(_fc52.group(1)), _W.get(_fc52.group(2))
        if not (_a and _b):
            fails.append(f"lab52's cell count is now {_fc52.group(1)} of "
                         f"{_fc52.group(2)}, which this check cannot spell")
        else:
            # The word "cells" is NOT part of the needle.  One of the places
            # quoting this count writes it without that word, so a needle
            # carrying it left that copy unguarded - and a tamper on exactly
            # that copy passed.
            #
            # The requirement is structural rather than a count, because the
            # count is not stable: Section 9 used to quote this figure and now
            # defers to Sections 7 and 8 for the condition it belongs to, so
            # demanding a fixed number of occurrences would fail a correct
            # paper the next time a section is reorganised.  What must hold is
            # that both documents carry it - the paper in its abstract and its
            # claim map, the Internet Appendix in the section that establishes
            # it - so neither can drift from lab52 while the other stays right.
            _n = f"{_a} of {_b}"
            for _lbl, _doc in (("the paper", norm_iv(plain(path))),
                               ("the Internet Appendix",
                                norm_iv(plain(_supp)) if os.path.isfile(_supp) else "")):
                checks += 1
                if _n not in _doc:
                    fails.append(f"'{_n}', lab52's compressed cell count, is not in "
                                 f"{_lbl}; both documents quote it and both must "
                                 f"agree with the lab")
            checks += 1
            if norm_iv(plain(path)).count(_n) < 2:
                fails.append(f"'{_n}' appears once in the paper; the abstract and the "
                             f"claim map both quote it, so one of them has drifted")
            for _bad in ("sixteen", "fifteen", "fourteen", "twelve", "eleven"):
                if _bad == _a:
                    continue
                checks += 1
                if f"{_bad} of {_b}" in norm_iv(txt):
                    fails.append(f"a superseded compressed cell count, '{_bad} of "
                                 f"{_b}', is still in the text; lab52 says {_a}")

    # Section S35's admissibility audit, against lab59.  Section 3 rests on the
    # rule this measures, so the paper is not allowed to state the result unless
    # the lab still produces it - and, more importantly, is not allowed to keep
    # stating it if the lab ever finds a leak.
    _l59 = lab("lab59_session_timestamps")
    _lk = re.search(r"leaks across every target, peer and day: (\d+)", _l59)
    _ws = re.search(r"wasted same-day observations:\s+(\d+)", _l59)
    _tg = re.search(r"the narrowest margin the rule relies on:\s+([\d.]+) hours "
                    r"\(([^)]+)\)", _l59)
    _wc = re.search(r"On (\d+) of the\s+(\d+) dates the fixed table sends (\w+) "
                    r"back a day for a (\w+)", _l59)
    checks += 1
    if not (_lk and _ws and _tg and _wc):
        fails.append("lab59 no longer prints the admissibility audit Sections 3 and "
                     "S35 quote")
    else:
        checks += 1
        if _lk.group(1) != "0":
            fails.append(f"lab59 finds {_lk.group(1)} admissibility leaks on real "
                         f"session clocks, but Section 3 states there are none; the "
                         f"panel must be rebuilt from true closes before any result "
                         f"in this paper is read")
        # the needles carry no tags: `plain()` has already stripped them from txt
        want("3 admissibility leaks", f"There are {_lk.group(1)} leaks")
        want("3 admissibility margin",
             f"relies on is {_tg.group(1)} hours, between "
             f"{_tg.group(2).replace(' before ', ' and ')}")
        want("S35 waste total", f"{_ws.group(1)} such days occur")
        want("S35 waste case",
             f"On {_wc.group(1)} of the {int(_wc.group(2)):,} dates the table sends "
             f"{_wc.group(3)} back a day for a {_wc.group(4)} forecaster")
        # Anchored on S35's own continuation, not on the bare claim: Section 3
        # carries the same sentence, and a needle on the shared half passed
        # while S35 itself said the opposite.  Exactly the spurious-match bug
        # this file was bitten by twice before.
        want("S35 direction", "the implementation is stricter than the rule "
                              "it implements: it discards information a real "
                              "forecaster could lawfully have used")
        want("3 direction summary",
             "the implementation is stricter than the rule it implements, which is "
             "the direction that costs information rather than validity")
        want("3 dates audited", f"on all {int(_wc.group(2)):,} dates")

    # Section 8.1's housing variance target is BACKWARD-looking, and the paper
    # described it for several drafts as "this paper's target type at a coarser
    # clock", which it is not: the paper's own target is the variance of the
    # five days after the origin, and this one overlaps its own current window
    # in WIN_M - 1 of WIN_M months.  The near-0.95 own-only R2 is that overlap,
    # not skill.  The window is read out of lab55 and the superseded
    # equivalence claim is forbidden, so the caveat cannot quietly go missing
    # while the figure it explains stays in the table.
    _l55src = read_text(os.path.join(HERE, "labs", "lab55_illiquid_measured.py"))
    _mwin = re.search(r"^WIN_M, MED_M = (\d+), (\d+)", _l55src, re.M)
    checks += 1
    if not _mwin:
        fails.append("lab55 no longer declares the variance window Section 8.1 describes")
    else:
        _W = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
              7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
              12: "twelve", 23: "twenty-three", 24: "twenty-four"}
        _n = int(_mwin.group(1))
        _months, _minus1 = _W.get(_n), _W.get(_n - 1)
        if not (_months and _minus1):
            fails.append(f"lab55's variance window is now {_n} months, which this "
                         f"check cannot spell; add it to the table above")
        else:
            want("8.1 variance window", f"trailing {_months}-month realised variance")
            want("8.1 variance overlap",
                 f"overlaps the current one in {_minus1} of {_months} months")
    want("8.1 variance is backward-looking", "The housing one is backward-looking")
    want("8.1 closer analogue",
         "The return target is the closer analogue of the paper's")
    forbid("8.1 superseded equivalence claim",
           "which is this paper's target type at a coarser clock and changes only the "
           "asset class")
    forbid("8.1 superseded own-target-type phrasing",
           "On this paper's own target type it is 19.0%")

    # Section 4.4's definition of the benchmark, checked against the code that
    # implements it rather than against itself.  The audit that found the
    # benchmark error found it through this paragraph: equation (6) carried a
    # scalar m described as "the training-window mean", while the code used
    # something else.  A prose definition of a quantity the whole paper divides
    # by is worth tying to its implementation, so the window, the cut and the
    # two load-bearing properties are all required here, and the superseded
    # wording is forbidden.
    _src = read_text(os.path.join(HERE, "labs", "lab05_robustness.py"))
    _mw = re.search(r"^TRAIN, VAL, REFIT = (\d+), (\d+),", _src, re.M)
    _mh = re.search(r"^WINDOW, HORIZON, MED = \d+, (\d+),", _src, re.M)
    checks += 1
    if not (_mw and _mh):
        fails.append("lab05 no longer declares the window and horizon that Section 4.4 "
                     "states")
    else:
        _win = int(_mw.group(1)) + int(_mw.group(2))
        want("4.4 benchmark window", f"over the {_win:,} days ending at t " + MINUS + " h")
        checks += 1
        if f"cut = t - horizon" not in _src:
            fails.append("lab05's benchmark no longer cuts at t - horizon, which "
                         "Section 4.4 states")
    want("4.4 benchmark is a forecast", "it is a forecast, not a fitted constant")
    want("4.4 benchmark not delay-matched", "It is not delay-matched")
    want("4.4 benchmark smeared", "Leaving the benchmark unsmeared while the models are "
                                  "smeared")
    forbid("4.4 superseded benchmark wording", "Let m be the training-window mean of y")
    # written tag-free: plain() turns each tag into a space, so the raw-source
    # spelling of this formula never appeared in the text being searched
    forbid("4.4 superseded benchmark symbol",
           "&sum; t ( y t " + MINUS + " m )&sup2;")

    # The headline interval is quoted by name in three places outside Table 2,
    # and it moved when the bootstrap block was corrected.  Two of those three
    # went on saying [64, 79] for months, because a bare pair of numbers in
    # prose is guarded by nothing.  Any OTHER two-number interval presented as
    # the headline's is forbidden here, so the next time the block or the
    # benchmark moves, the prose fails with the table instead of behind it.
    if 55 in rates:
        _a, _lo, _hi, *_ = rates[55]
        checks += 1
        want("headline interval in Table 2", f"{_a}% [{_lo}, {_hi}]")
        for _phrase in ("interval Table 2 already carries",
                        "a quantity whose interval is"):
            for _m in re.finditer(re.escape(_phrase), txt):
                _win = txt[max(0, _m.start() - 60):_m.end() + 60]
                _iv = re.findall(r"\[\s*(-?\d+),\s*(-?\d+)\]", _win)
                checks += 1
                for _pair in _iv:
                    if list(_pair) != [_lo, _hi]:
                        fails.append(
                            f"headline interval quoted as [{_pair[0]}, {_pair[1]}] "
                            f"near '{_phrase}', but lab10 gives [{_lo}, {_hi}]")

    # Section 4's new paragraph on where the two losses disagree quotes four
    # cells of Table 2 in prose.  A prose sentence can drift while the table it
    # describes stays correct, so both endpoints are tied back to lab10 here.
    # The claim itself is directional: R2 above the benchmark at the two longest
    # delays, QLIKE below it.  If a rerun ever flips that, the paragraph is
    # wrong in substance and not merely in its digits, so the sign is checked
    # too rather than only the figures.
    for d in (34, 55):
        if d in skill:
            r2o, qo = skill[d][0], skill[d][2]
            checks += 1
            want(f"4 loss disagreement R2 at delta={d}", r2o.replace("-", MINUS))
            checks += 1
            want(f"4 loss disagreement QLIKE at delta={d}", qo.replace("-", MINUS))
            checks += 1
            if not (float(r2o) > 0 > float(qo)):
                fails.append(f"lab10: at delta={d} the two losses no longer straddle "
                             f"the benchmark (R2 {r2o}, QLIKE {qo}), but Section 4 "
                             f"says they do")
    want("4 loss disagreement reading",
         "Where the two conflict we give weight to QLIKE")

    # The third loss is quoted in the text rather than tabulated, and which of
    # the two variance-scale losses it is matters: the paper reported the
    # median-NORMALISED one as Patton's MSE for several drafts, which overstated
    # the rate by up to 29 points.  Both columns are now parsed out of lab10's
    # Part 4 and the paper is required to quote the natural-scale figure and to
    # carry the gap, so the mislabelling cannot quietly return.
    # Part 4 now prints THREE columns, not two, because the gap between the
    # normalised and the natural loss is two effects rather than one: taking the
    # 1/M^2 weight off, and moving the forecast's median from M_{t+h} to M_t.
    # The middle column reconstructs both sides at t+h - infeasible, and there
    # only to hold the median fixed - so the first difference is the weight
    # alone and the second is the drift alone.  The verifier reads all six
    # numbers per row, because a decomposition whose parts were not checked
    # against the code is exactly the kind of tidy arithmetic that drifts.
    GAP = re.compile(r"^\s*(\d+)\s+(-?[\d.]+)%\s+(-?[\d.]+)%\s+(-?[\d.]+)%"
                     r"\s+([-+][\d.]+)%\s+([-+][\d.]+)%\s+([-+][\d.]+)%\s*$")
    gaps = {}
    # lab() collapses whitespace, so Part 4's rows are parsed from the file
    # itself, which is where the line structure survives.
    for ln in read_text(os.path.join(EXP, "lab10_loss_scale.txt")).splitlines():
        m = GAP.match(ln)
        if m:
            gaps[int(m.group(1))] = m.groups()[1:]
    checks += 1
    if len(gaps) < 8:
        fails.append(f"lab10 Part 4 no longer prints all three variance-scale "
                     f"rate columns ({len(gaps)} rows parsed, 8 expected)")
    else:
        for d, (nrm, inf, nat, w, dr, gp) in sorted(gaps.items()):
            want(f"S25 delta={d} row",
                 f"{d} {nrm}% {inf}% {nat}% {w} {dr} {gp}")
            # the two pieces must still add to the total in the code itself,
            # which is a statement about lab10 and not about the paper
            checks += 1
            # three figures each rounded to a tenth, so the sum can miss by up
            # to 0.15 without anything being wrong; wider than that is not
            # rounding
            if abs(float(w) + float(dr) - float(gp)) > 0.16:
                fails.append(f"lab10 Part 4 delta={d}: weight {w} + drift {dr} "
                             f"does not make the total {gp}")
        # the claim the paper leans on: the drift piece is small everywhere
        _drift = max(abs(float(v[4])) for v in gaps.values())
        want("4.4 bounds the drift piece",
             f"the drift worth at most {_drift:.1f} points at any delay")
        want("S25 bounds the drift piece",
             f"the drift never worth more than {_drift:.1f} points at any delay")
        _w1, _g1 = gaps[1][3], gaps[1][5]
        _w55, _g55x = gaps[55][3], gaps[55][5]
        want("S25 splits the one-day gap",
             f"{_w1.lstrip('+')} of the {_g1.lstrip('+')} points at one day")
        want("S25 splits the eleven-week gap",
             f"{_w55.lstrip('+')} of the {_g55x.lstrip('+')} at eleven weeks")
        # the identity the middle column rests on has to be checked by the
        # script, not merely described by the appendix
        checks += 1
        if "the middle column IS the normalised loss reweighted: True" not in l10:
            fails.append("lab10 no longer proves the reweighting identity the "
                         "S25 decomposition rests on")
        _n55, _t55, _g55 = gaps[55][0], gaps[55][2], gaps[55][5]
        want("8 quotes the natural-scale rate at 55",
             f"{round(float(_t55))}% at eleven weeks")
        want("4.4 names the size of the correction",
             f"by {round(float(_g55))} points at eleven weeks")
        forbid("8 still quotes the normalised rate",
               f"the substitution rate reads 85% at five days and {_n55[:2]}%")
    # and the zero-delay row must no longer be described as changing sign
    forbid("S9 stale sign-flip claim",
           "the one place in this paper where the choice of loss decides the sign")
    want("S9 records the artefact", "The flip was an artefact")

    # lab05 is the module the other labs import, and its own Part C prints the
    # same two skill columns lab10 does.  Checking both against Table 1 is not
    # duplication: it is the only thing that would catch the two labs drifting
    # apart, which run_all.py --check cannot see because each would still match
    # its own stored output.
    l05 = lab("lab05_robustness")
    PARTC = re.compile(r"^\s*(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([-+][\d.]+)\s+"
                       r"\[([-+][\d.]+),([-+][\d.]+)\]")
    partc = {}
    for ln in read_text(os.path.join(EXP,
                                     "lab05_robustness.txt")).splitlines():
        m = PARTC.match(ln)
        if m:
            partc[int(m.group(1))] = m.groups()[1:]
    if len(partc) < 10:
        fails.append(f"lab05: could not parse Part C ({len(partc)} rows, 10 expected)")
    for d, (own, cross, _dr, _lo, _hi) in sorted(partc.items()):
        checks += 1
        if d in skill and (own, cross) != (skill[d][0], skill[d][1]):
            fails.append(f"lab05 and lab10 disagree at delta={d}: "
                         f"{own}/{cross} against {skill[d][0]}/{skill[d][1]}")
    checks += 1
    if "4862 test days" not in l05:
        fails.append("lab05: the test window is no longer 4,862 days, which "
                     "Section 4 states")

    # --- Table 8, the horse race, against lab08 ---------------------------
    # The section this table carries is the one the whole paper turns on, and
    # it was checked against nothing.  Every cell of every row, from the STRICT
    # arm - the t-1 specification the paper reports - is required here.
    l08 = read_text(os.path.join(EXP, "lab08_implied_vol.txt"))
    # The file holds two arms, t-1 and same-day, printed in the same shape; take
    # the block between the t-1 banner and the next banner so the same-day
    # numbers can never be mistaken for the ones the paper reports.
    strict = l08.split("STRICT - VIX at t-1")[-1]
    strict = strict.split("=" * 40, 2)[-1].split("=" * 40)[0]
    HORSE = re.compile(r"^\s*(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+"
                       r"(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s*$")
    horse = {}
    for ln in strict.splitlines():
        m = HORSE.match(ln)
        if m:
            horse[int(m.group(1))] = m.groups()[1:]
    if len(horse) < 10:
        fails.append(f"lab08: could not parse the strict horse race "
                     f"({len(horse)} rows found, 10 expected)")
    for d in (0, 3, 5, 13, 21, 55):
        checks += 1
        if d not in horse:
            fails.append(f"lab08: no delta={d} row in the strict arm")
            continue
        own, fo, iv, both, gain, z, _p = horse[d]
        row = f"{d} {own} {fo} {iv} {both} {gain} {z}"
        if row.replace(MINUS, "-") not in txt:
            fails.append(f"Table 8 delta={d}: lab08 no longer supports the row "
                         f"'{row}'")
    checks += 1
    if f"{len(horse)} test days" not in l08 and "4525 test days" not in l08:
        fails.append("lab08: the 4,525-day window Table 8's caption states is gone")

    # --- the GROSS content given implied volatility, against lab08 ---------
    # Section 7 and Section S28 both quote this block, and neither was checked.
    # All four intervals were stale at the third decimal, and the paper named
    # the wrong one as its binding bound: it cited delta = 55 where delta = 21
    # has the larger upper endpoint.  The bound is what makes "redundant" a
    # bounded statement instead of a claim of equality, so it is derived here
    # and the paper has to quote the least favourable endpoint, not a
    # convenient one.
    _g08 = re.compile(r"\n\s*(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([-+][\d.]+)\s+"
                      r"\[(-?[\d.]+),([-+][\d.]+)\]\s+redundant")
    _gr = {int(m.group(1)): m.groups()[1:] for m in _g08.finditer(l08)}
    checks += 1
    if len(_gr) != 4:
        fails.append(f"lab08: parsed {len(_gr)} of 4 gross-content rows given "
                     "implied volatility")
    else:
        def _r3(x):                       # the papers quote three decimals
            return f"{float(x):+.3f}".replace("+", "").replace("-", MINUS) \
                if float(x) >= 0 else f"{float(x):.3f}".replace("-", MINUS)
        _ds = sorted(_gr)

        def _join(items):                 # "a, b, c and d", the paper's style
            return ", ".join(items[:-1]) + " and " + items[-1]

        # the points are quoted with their sign; the delta = 0 gross was
        # TRUNCATED to +0.001 where lab08 prints +0.0016, which rounds to
        # +0.002, so rounding is done here rather than trusted
        want("S28 gross content points",
             _join([f"+{float(_gr[d][2]):.3f} at &delta; = {d}" for d in _ds]))
        want("S28 gross content intervals",
             _join([f"[{_r3(_gr[d][3])}, +{float(_gr[d][4]):.3f}]" for d in _ds]))
        # the binding equivalence bound is the LARGEST upper endpoint, and the
        # delay it comes from has to be named
        _bd = max(_ds, key=lambda d: float(_gr[d][4]))
        _bv = f"{float(_gr[_bd][4]):.3f}"
        want("S28 equivalence bound",
             f"rules out a gross gain larger than {_bv} of R&sup2;, at &delta; = {_bd}")
        want("7 equivalence bound",
             f"[{_r3(_gr[_bd][3])}, +{_bv}] at &delta; = {_bd}, rules out a gross "
             f"gain above about {_bv} of R&sup2;")
        checks += 1
        if any(float(_gr[d][3]) > 0 for d in _ds):
            fails.append("lab08: a gross-content interval given implied volatility no "
                         "longer covers zero, so 'redundant' is the wrong word and "
                         "Sections 7 and S28 need rewriting rather than renumbering")
        # non-rejection must not be written as equality anywhere
        want("7 redundant is not equal",
             "is used in this paper to mean exactly that and never to mean equal")
        for _eq in ("the foreign block adds nothing once implied volatility is held,"
                    " at every delay",
                    "the foreign block adds nothing. The increment from adding",
                    "they add nothing to it, and paying to estimate nothing"):
            forbid("non-rejection written as equality", _eq)

    # --- Table 12, the horizon-matched race, against lab44 ----------------
    # Every cell of the table is rebuilt by joining two of lab44's own tables:
    # the columns through "GW z" come from part A, the foreign increment given
    # the nine-day block from part B.  Building the row across two sections is
    # deliberate - it is the join the paper makes, so it is the join to check.
    l44 = read_text(os.path.join(EXP, "lab44_horizon_matched_iv.txt"))
    # The header line sits BEFORE its ruler, so the block of interest is the
    # second field of a split on the ruler, not the first. Getting this wrong
    # silently yields zero rows, and a parse that finds nothing must fail loudly
    # rather than pass vacuously - hence the row-count check below.
    _secA = l44.split("A.  DOES MATCHING THE HORIZON")[-1].split("=" * 60)[1]
    _secB = l44.split("B.  THE QUESTION SECTION 8 ASKS")[-1].split("=" * 60)[1]
    A9 = re.compile(r"^\s*(\d+)" + r"\s+(-?[\d.]+)" * 4 + r"\s+([-+][\d.]+)"
                    r"\s+(-?[\d.]+)\s+([-+][\d.]+)\s+(-?[\d.]+)\s*$")
    B9 = re.compile(r"^\s*(\d+)" + r"\s+([-+][\d.]+)\s+(-?[\d.]+)" * 3 + r"\s*$")
    pa = {int(m.group(1)): m.groups()[1:]
          for m in (A9.match(ln) for ln in _secA.splitlines()) if m}
    pb = {int(m.group(1)): m.groups()[1:]
          for m in (B9.match(ln) for ln in _secB.splitlines()) if m}
    if len(pa) != 5 or len(pb) != 5:
        fails.append(f"lab44: could not parse its own tables "
                     f"({len(pa)} rows in part A, {len(pb)} in part B, 5 each)")
    for d in sorted(pa):
        checks += 1
        if d not in pb:
            fails.append(f"lab44: delta={d} is in part A but not part B")
            continue
        own, iv30, iv9, term, gap, z9, _t, _zt = pa[d]
        _g30, _z30, g9, z9f, _gt, _zt2 = pb[d]
        row = f"{d} {own} {iv30} {iv9} {term} {gap} {z9} {g9} {z9f}"
        if row.replace(MINUS, "-") not in txt:
            fails.append(f"Table 12 delta={d}: lab44 no longer supports the row "
                         f"'{row}'")
    # The two claims the section rests on, as lab44's own verdict counters.
    checks += 1
    if "nine-day beats thirty-day significantly at 5 of 5 delays" not in \
            " ".join(l44.split()):
        fails.append("lab44: the nine-day index no longer beats the thirty-day one "
                     "at every delay, which Section 8.5 states")
    checks += 1
    if "given   IV9: 0 of 5" not in l44:
        fails.append("lab44: the foreign block now helps somewhere given the "
                     "horizon-matched block, which would overturn Section 8")
    checks += 1
    if "2098 test days" not in l44:
        fails.append("lab44: the window is no longer 2,098 days, which Section 8.5 "
                     "and Table 12's caption both state")
    want("8.5 window", "2,098 test days from May 2018")
    want("S24 window share", "46% of Table S22's window")
    # Table S19's cells are checked above; the RANGES the paragraph beneath it
    # states were checked by nothing, and both were stale.  It said the horizon
    # effect carries Giacomini-White statistics of 4.02 to 4.19 where lab44's
    # own column runs 3.13 to 3.30, and put the foreign increment between -2.41
    # and -3.38 where the table printed in the same document runs -2.24 to
    # -3.29.  A narrative that summarises a table has to be built from it.
    if len(pa) == 5 and len(pb) == 5:
        _gaps = [float(pa[d][4]) for d in pa]
        _z9s = [float(pa[d][5]) for d in pa]
        _fgn = [float(pb[d][3]) for d in pb]
        want("S24 horizon effect range",
             f"worth between {min(_gaps):.4f} and {max(_gaps):.4f} of R&sup2;")
        want("S24 horizon effect statistics",
             f"Giacomini-White statistics of {min(_z9s):.2f} to {max(_z9s):.2f}")
        # stated weakest-first, which for a negative column is the maximum
        want("S24 foreign increment statistics",
             f"with statistics between {max(_fgn):.2f} and {min(_fgn):.2f}"
             .replace("-", MINUS))
        checks += 1
        if max(_fgn) >= 0:
            fails.append("lab44: the foreign increment given IV9 is no longer "
                         "negative at every delay, which Section S24 states")
    # The window effect, stated as a range in the text and printed per delay in
    # part A0; check the extremes rather than the prose alone.
    _sec0 = l44.split("A0.  HOW MUCH OF ANY DIFFERENCE")[-1].split("=" * 60)[1]
    _wd = [float(m.group(1)) for m in
           re.finditer(r"^\s*\d+\s+-?[\d.]+\s+-?[\d.]+\s+(-[\d.]+)\s*$",
                       _sec0, re.M)]
    checks += 1
    if len(_wd) != 5:
        fails.append(f"lab44: part A0 no longer prints five window rows ({len(_wd)})")
    else:
        want("8.5 window cost low", f"{min(abs(x) for x in _wd):.4f}")
        want("8.5 window cost high", f"{max(abs(x) for x in _wd):.4f}")
    # The slope result and the two-tenor result, both quoted in the text.
    _secC = l44.split("C.  THE SLOPE ON ITS OWN")[-1]
    _sz = [float(m.group(1)) for m in
           re.finditer(r"^\s*\d+\s+[\d.]+\s+[\d.]+\s+[-+][\d.]+\s+(-[\d.]+)\s*$",
                       _secC, re.M)]
    checks += 1
    if len(_sz) != 5 or max(_sz) > -1.96:
        fails.append("lab44: the term-structure slope is no longer worse than the "
                     "nine-day level at every delay, which Section 8.5 states")
    else:
        want("8.5 slope range low", f"{MINUS}{abs(max(_sz)):.2f}")
        want("8.5 slope range high", f"{MINUS}{abs(min(_sz)):.2f}")

    # --- Section 5.3's two full-sample descriptives, against lab22 --------
    # Found in the last pre-publication pass: the span and the dimension of the
    # live cross-section were quoted in Section 5.3 and produced by nothing.
    # They had been computed once by hand and typed in, in a repository whose
    # whole claim is that no number reaches the paper that way.
    l22b = " ".join(lab("lab22_factor_benchmark").split())
    for label, inlab, inpaper in (
            ("span", "contemporaneous R-squared of the target on them: 0.6952",
             "same day gives R"),
            ("first component", "first component 66.5%", "accounts for 66.5%"),
            ("first two", "first two 76.9%", "the first two for 76.9%")):
        checks += 1
        if inlab not in l22b:
            fails.append(f"lab22 no longer prints the {label} ('{inlab}')")
        want(f"5.3 {label}", inpaper)
    want("5.3 span value", "= 0.695.")
    checks += 1
    if "6 same-day peers ['N225', 'AXJO', 'HSI', 'NSEI', 'FTSE', 'DAX']" not in l22b:
        fails.append("lab22: the descriptives are no longer computed on the six "
                     "same-day peers Section 5.3 names")
    want("5.3 six series", "six foreign volatility series")

    # --- Section 8.5's crisis check, against lab44 part B2 ----------------
    _secB2 = l44.split("B2.  IS THE HORIZON RESULT JUST MARCH 2020")[-1] \
        .split("=" * 60)[1]
    B2 = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+\+([\d.]+)\s+([\d.]+)"
                    r"\s+(-[\d.]+)\s+(-[\d.]+)\s*$")
    pb2 = {int(m.group(1)): m.groups()[1:]
           for m in (B2.match(ln) for ln in _secB2.splitlines()) if m}
    checks += 1
    if len(pb2) != 5:
        fails.append(f"lab44: part B2 no longer prints five rows ({len(pb2)})")
    else:
        gains = [float(v[2]) for v in pb2.values()]
        zs = [float(v[3]) for v in pb2.values()]
        fgn = [float(v[4]) for v in pb2.values()]
        fz = [float(v[5]) for v in pb2.values()]
        want("8.5 crisis gain range", f"{min(gains):.4f} to {max(gains):.4f} of R")
        want("8.5 crisis z range", f"{min(zs):.2f} to {max(zs):.2f}")
        want("8.5 crisis foreign range",
             f"{MINUS}{abs(max(fgn)):.4f} and {MINUS}{abs(min(fgn)):.4f}")
        want("8.5 crisis foreign z",
             f"{MINUS}{abs(max(fz)):.2f} to {MINUS}{abs(min(fz)):.2f}")
    checks += 1
    if "nine-day still beats thirty-day significantly at 5 of 5" not in l44:
        fails.append("lab44: the horizon result no longer survives removing 2020, "
                     "which Section 8.5 states that it does")
    for _n in ("2,004 days", "15 February to 30 June 2020"):
        want(f"8.5 crisis window {_n}", _n)
    checks += 1
    if "94 test days removed, 2004 remain" not in l44:
        fails.append("lab44: the 2020 exclusion no longer leaves 2,004 days")

    # --- Section 9.3's interval on the headline, against lab45 ------------
    # The effective age went unqualified for most of this project's life while
    # every figure around it carried an interval.  It is checked here cell by
    # cell, including the paired differences, which are the part that answers
    # the question the marginal intervals cannot.
    l45 = read_text(os.path.join(EXP, "lab45_effective_age_interval.txt"))
    AGE = re.compile(r"^\s*(\d+)\s+([\d.]+)d\s+\[([\d.]+), ([\d.]+)\] days"
                     r"\s+(\d+)\s+(\d+)\s*$")
    ages = {int(m.group(1)): m.groups()[1:]
            for m in (AGE.match(ln) for ln in l45.splitlines()) if m}
    checks += 1
    if len(ages) != 3:
        fails.append(f"lab45: could not parse the age table ({len(ages)} rows, 3)")
    for g, label in ((0, "same-day"), (1, "one-day"), (5, "week-behind")):
        checks += 1
        if g not in ages:
            fails.append(f"lab45: no g={g} row")
            continue
        pt, lo, hi, zero, off = ages[g]
        want(f"9.3 age g={g}", f"{pt} [{lo}, {hi}]" if g else f"{pt} days old [{lo}, {hi}]")
        # and every OTHER copy of the same figure, not merely the one that is
        # current.  The g=0 age is the paper's most-quoted number and lived in
        # three places at two different intervals.
        only_iv_re(f"9.3 age g={g} everywhere",
                   re.escape(pt) + r"(?: days(?: old)?)?", lo, hi)
        checks += 1
        if (zero, off) != ("0", "0"):
            fails.append(f"lab45 g={g}: replications now fall at zero or off the "
                         f"grid ({zero}, {off}), which Section 9.3 says none do")
    PAIR = re.compile(r"^\s*g = (\d+) minus g = 0\s+([\d.]+)d\s+"
                      r"\[([\d.]+), ([\d.]+)\] days\s+(yes|no)\s*$")
    pairs = {int(m.group(1)): m.groups()[1:]
             for m in (PAIR.match(ln) for ln in l45.splitlines()) if m}
    checks += 1
    if len(pairs) != 2:
        fails.append(f"lab45: could not parse the paired differences ({len(pairs)})")
    for g in sorted(pairs):
        pt, lo, hi, excl = pairs[g]
        checks += 1
        if excl != "yes":
            fails.append(f"lab45: the g={g} paired difference no longer excludes "
                         "zero, which Section 9.3 states")
        want(f"9.3 exchange rate g={g}", f"{pt} days" if g == 1 else f"{pt} [{lo}, {hi}]")
    # The literal that stood here, "[1.4, 3.3]", froze an endpoint lab45 does
    # not print: it gives [1.4, 3.2].  Two sentences in the appendix carried
    # the wrong figure for as long as this check demanded it.
    _m45i = re.search(r"g = 1 minus g = 0\s+[\d.]+d\s+\[([\d.]+), ([\d.]+)\]",
                      read_text(os.path.join(EXP,
                                             "lab45_effective_age_interval.txt")))
    checks += 1
    if not _m45i:
        fails.append("lab45 no longer prices one day of foreign staleness")
    else:
        want("9.3 exchange rate interval",
             f"[{_m45i.group(1)}, {_m45i.group(2)}]")

    # --- the Fieller construction, against lab21 --------------------------
    # The size of the Fieller-versus-bootstrap disagreement is stated in both
    # documents and was checked in neither.  Section 4.6 said 2.4 points while
    # Section S3 said 1.6 and lab21 prints 1.6, so the paper contradicted its
    # own appendix on the one number the section exists to report.  Both
    # phrasings are found and both must carry the lab's figure.
    l21 = read_text(os.path.join(EXP, "lab21_stronger_inference.txt"))
    _gap21 = re.search(r"largest disagreement between Fieller and the percentile "
                       r"bootstrap, either endpoint: ([\d.]+)%", l21)
    _set21 = re.search(r"confidence sets that are NOT an ordinary interval: "
                       r"(\d+) of (\d+)", l21)
    checks += 1
    if not (_gap21 and _set21):
        fails.append("lab21 no longer prints the Fieller summary Section 4.6 quotes")
    else:
        checks += 1
        if _set21.group(1) != "0":
            fails.append(f"lab21: {_set21.group(1)} of {_set21.group(2)} Fieller sets "
                         "are no longer ordinary intervals, which both documents "
                         "state none are")
        # The phrasing changed with the procedure, and so does this pattern.
        # While the inversion used normal critical values both documents said
        # the Fieller set sat "within N points of the bootstrap's" and offered
        # that as reassurance.  Under fixed-b the two disagree by 8.3 points
        # and the agreement is gone, so what has to be stated in both places is
        # the DISAGREEMENT, with the lab's own figure.  Matching the old
        # wording would now pass only on a document that still made the old
        # claim, which is the wrong way round for a checker.
        # Scoped to the sentences that are actually about the bootstrap.  The
        # bare phrase matched a third passage comparing two CONTROLS "by up to
        # 4.1 points", which has nothing to do with the Fieller gap, and the
        # check duly reported the paper as disagreeing with a lab it was not
        # quoting.  The window is what keeps the two apart.
        _sites = [_m.group(1) for _m in
                  re.finditer(r"by up to ([\d.]+) (?:percentage )?points", txt)
                  if "bootstrap" in txt[max(0, _m.start() - 180):_m.end() + 180]]
        checks += 1
        if len(_sites) < 2:
            fails.append(f"the Fieller gap is stated in {len(_sites)} place(s); "
                         "Section 4.6 and Section S3 both state it")
        _bad21 = [x for x in _sites if x != _gap21.group(1)]
        checks += 1
        if _bad21:
            fails.append(f"Fieller gap: {len(_bad21)} of {len(_sites)} site(s) say "
                         f"{', '.join(sorted(set(_bad21)))} where lab21 prints "
                         f"{_gap21.group(1)}")
    # and the delta = 55 pair the appendix quotes as the illustration
    # a tau column now sits between the rate and the two sets, so the pattern
    # allows it and captures it: the appendix quotes tau beside the pair, so
    # that a reader can find the coverage row that applies to it.
    _r55 = re.search(r"\n\s+55\s+\d+%\s+([\d.]+)\s+\[\s*(\d+)%,\s*(\d+)%\]\s+"
                     r"\[\s*(\d+)%,\s*(\d+)%\]", l21)
    checks += 1
    if not _r55:
        fails.append("lab21 no longer prints the delta=55 Fieller and bootstrap pair")
    else:
        _t55r, _bl, _bh, _fl, _fh = _r55.groups()
        want("S3 Fieller illustration",
             f"at &delta; = 55, [{_fl}, {_fh}] against [{_bl}, {_bh}]")
        # and tau beside it, so a reader can find the coverage row that
        # applies.  Without this the appendix reported a set and left the
        # coverage evidence unattachable to it.
        want("S3 names tau for the illustration",
             f"where the denominator sits {_t55r} standard errors from zero")
        # the two ranges the appendix quotes beside it, both computed
        _t21 = sorted(float(m) for m in re.findall(
            r"\n\s+\d+\s+-?\d+%\s+([\d.]+)\s+\[", l21))
        checks += 1
        if len(_t21) != 6:
            fails.append(f"lab21: read {len(_t21)} of 6 tau values from the delay "
                         f"table, so the range Section S3 quotes rests on nothing")
        else:
            want("S3 tau range across the delay grid",
                 f"it runs from {_t21[0]} to {_t21[-1]}")
        # tercile_fieller() is called again here rather than reusing the _tf
        # bound several hundred lines below: a name defined later in the same
        # function is not available here, and reordering the blocks to share it
        # would move checks that are deliberately grouped by section.
        _tfa = tercile_fieller()
        _tt = sorted(float(_tfa[_d][_s][3]) for _d in _tfa for _s in _tfa[_d])
        want("S3 tau range across the market states",
             f"from {_tt[0]} to {_tt[-1]}")
        # and the claim that no cell sits at either extreme of the coverage
        # study must BE true, not merely written.
        checks += 1
        if min(_t21 + _tt) <= 1.5 or max(_t21 + _tt) >= 10.0:
            fails.append(f"Section S3 says no cell sits at the coverage study's "
                         f"extremes of 1.5 and 10; the measured range is "
                         f"{min(_t21 + _tt)} to {max(_t21 + _tt)}")

    # --- Section 11's state-space comparison, against lab33 ---------------
    # lab33 was exploratory and uncited while Section 11 claimed nothing here
    # bounded what a better estimator could do.  It is cited now, so it is
    # checked like anything else the paper quotes.
    l33 = read_text(os.path.join(EXP, "lab33_ragged_edge.txt"))
    FILT = re.compile(r"^\s*(\d+)\s+([-+][\d.]+)\s+\[\s*([-+][\d.]+), "
                      r"([-+][\d.]+)\]\s+(\w+)\s*$")
    filt = {int(m.group(1)): m.groups()[1:]
            for m in (FILT.match(ln) for ln in l33.splitlines()) if m}
    checks += 1
    if len(filt) != 6:
        fails.append(f"lab33: could not parse the filter comparison ({len(filt)})")
    for d in (21, 55, 0):
        checks += 1
        if d not in filt:
            fails.append(f"lab33: no delta={d} row")
            continue
        diff, lo, hi, who = filt[d]
        # The paper quotes the lab's own precision rather than a rounded form:
        # a five-decimal interval beside a four-decimal point estimate is less
        # elegant than rounding it and is exactly what the lab prints, which is
        # the property that makes the check worth having.
        want(f"11 filter delta={d}", f"by {diff.lstrip('+-')}")
        want(f"11 filter interval delta={d}", f"[{lo}, {hi}]")
    # The winning set is READ from lab33 and then required to be what the
    # appendix says, rather than pinned to the pair that happened to win.  It
    # was pinned to "2 of 6 [21, 55]", and when the delay-21 interval widened
    # to include zero the check reported the lab as broken - which was exactly
    # backwards: the lab was right and the appendix had become wrong.  Reading
    # it makes the failure name the document instead.
    _w33 = re.search(r"delays where the filter significantly wins: "
                     r"(\d+) of 6 \[([\d, ]*)\]", l33)
    checks += 1
    if not _w33:
        fails.append("lab33 no longer reports which delays the filter wins at")
    else:
        _wins = [x.strip() for x in _w33.group(2).split(",") if x.strip()]
        _named = ["&delta; = " + x for x in _wins]
        checks += 1
        if len(_wins) == 1:
            want("11 filter winning delay",
                 f"beats the specification used throughout this paper at "
                 f"{_named[0]}")
        # every delay the filter does NOT significantly win at must not be
        # described as beaten
        for _d in ("21", "55"):
            if _d in _wins:
                continue
            checks += 1
            if f"paper at &delta; = {_d} by" in txt:
                fails.append(f"the appendix says the filter beats the paper at "
                             f"delta = {_d}, but lab33 finds that interval no "
                             f"longer excludes zero")
    for _c in ("0.8424", "0.1802"):
        checks += 1
        if _c not in l33:
            fails.append(f"lab33: the reconstruction correlation {_c} is gone")
    want("11 filter reconstruction", "correlates 0.84 with the truth at eleven weeks")
    want("11 filter reconstruction stale", "against 0.18 for the stale value")

    # --- Section 10.1's estimator-class arm, against lab46 ----------------
    # Two referees argued the own-only control is a straw man because a ridge
    # cannot project a stale observation forward.  Section 10.1 answers with a
    # state-space model and the answer has two halves, so both are checked.
    # The first is the identity: a deterministic projection is one constant per
    # column and a standardised ridge absorbs it, so the two arms must agree to
    # within the guard added to the standard deviation.  That is a claim about
    # the ESTIMATOR, not about volatility, and it would fail loudly if the walk
    # ever stopped standardising - which is the reason to check the exponent
    # rather than the sentence around it.
    l46 = read_text(os.path.join(EXP, "lab46_own_only_filter.txt"))
    l46f = " ".join(l46.split())
    checks += 1
    _m = re.search(r"largest difference the projection makes: ([\d.]+)e-(\d+)", l46f)
    if not _m:
        fails.append("lab46: the projection identity line is gone")
    elif int(_m.group(2)) < 9:
        fails.append(f"lab46: the projection now moves the forecast by "
                     f"{_m.group(0).split(': ')[1]}, which is larger than the "
                     "1e-9 standardisation guard Section 10.1 attributes it to")
    else:
        # The paper writes this as 1.8 x 10^-11 in tags; plain() strips the tags
        # and leaves the exponent as its own word, so the needle is that form.
        want("10.1 projection residue", "1.8 &times; 10 -11")
    # The second half is empirical: exponential weighting has to LOSE, and the
    # paper names three of the margins.  Anchor them to the lab's own table
    # rather than to the prose, so a re-run that moves them fails here first.
    E46 = re.compile(r"^\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-+][\d.]+)\s*$")
    ew = {int(m.group(1)): m.groups()[1:]
          for m in (E46.match(ln) for ln in l46.splitlines()) if m}
    checks += 1
    if len(ew) != 6:
        fails.append(f"lab46: could not parse the exponential table ({len(ew)} rows, 6)")
    for d in (0, 21, 55):
        checks += 1
        if d not in ew:
            fails.append(f"lab46: no delta={d} row in the exponential table")
            continue
        if not ew[d][2].startswith("-"):
            fails.append(f"lab46: exponential weighting now WINS at delta={d} "
                         f"({ew[d][2]}), which Section 10.1 says it never does")
            continue
        want(f"10.1 exponential margin delta={d}", ew[d][2].lstrip("-"))
    checks += 1
    if "the exponential control is better at 0 of 6 delays" not in l46f:
        fails.append("lab46: exponential weighting now wins somewhere, so "
                     "Section 10.1's 'worse at every delay tested' is stale")
    for label, inlab, inpaper in (
            ("training block", "fitted on the first 1500 rows", "first 1,500 rows"),
            ("half-lives", "half-lives (5, 22, 66) days",
             "half-lives of 5, 22 and 66 days"),
            ("feature counts", "Six features against", "Six features against three"),
            ("persistence", "phi = 0.9265", "0.9265"),
            ("half-life", "half-life of the state: 9.1", "9.1 trading days"),
            ("state equals observation", "the raw observation: 4.42e-08",
             "4.4 &times; 10 -8"),
            ("inflation", "higher at every delay, by up to 4.1%", "4.1 points"),
            ("rate against the weak arm", "74.3% at eleven weeks", "74.3%"),
            ("rate against the paper's arm", "against the published 71.4%", "71.4%"),
            ("ruler point", "EWMA ruler: 4.2 days", "4.2 days rather than 4.6"),
            ("ruler width", "-2% as a share of the point", "2% narrower")):
        checks += 1
        if inlab not in l46f:
            fails.append(f"lab46 no longer prints the {label} ('{inlab}')")
        want(f"10.1 {label}", inpaper)
    # The rate quoted against the paper's own control has to be the one Table 14
    # already carries, or Section 10.1 and Section 9.3 are quoting two different
    # pipelines at each other.
    checks += 1
    if "71%" not in txt:
        fails.append("10.1: Table 14's rounded 71% is gone, so the comparison "
                     "in 10.1 no longer lands on a figure the paper reports")

    # --- Section 11's seam audit, against lab47 ---------------------------
    # The paper's oldest limitation was that one join could not be checked by
    # comparison.  It is checked now, so the three percentiles and the power
    # statements are anchored to the lab rather than transcribed.  The power
    # line is the one that matters: a limitation is worth what it rules out.
    l47 = read_text(os.path.join(EXP, "lab47_target_seam.txt"))
    l47f = " ".join(l47.split())
    PCT = re.compile(r"^\s*(level|alignment|range)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+"
                     r"[\d.]+\s+(\d+)th\s*$")
    pcts = {m.group(1): int(m.group(2))
            for m in (PCT.match(ln) for ln in l47.splitlines()) if m}
    checks += 1
    if len(pcts) != 3:
        fails.append(f"lab47: could not parse the three percentiles ({len(pcts)} of 3)")
    for k, v in sorted(pcts.items()):
        checks += 1
        if v >= 95:
            fails.append(f"lab47: the {k} instrument now flags the S&P join at the "
                         f"{v}th percentile, which Section 11 says it does not")
    if len(pcts) == 3:
        want("11 seam percentiles",
             f"{pcts['level']}th, {pcts['alignment']}th and {pcts['range']}th percentiles")
    for label, inlab in (
            ("misalignment caught", "one-session misalignment caught: yes, by alignment"),
            ("splice caught", "spliced block caught: yes, by level, alignment"),
            ("smallest range caught", "smallest range corruption caught: 125%")):
        checks += 1
        if inlab not in l47f:
            fails.append(f"lab47 no longer reports the {label} ('{inlab}')")
    want("11 seam power", "an intraday range mis-scaled by 25% is caught")
    want("11 seam power limit", "while 10% is not")
    want("11 seam calibration", "400 pseudo-seams")
    checks += 1
    if "overlap    0 rows  UNVERIFIABLE" not in l47f.replace("  ", " ").replace(
            " 0 rows UNVERIFIABLE", "    0 rows  UNVERIFIABLE"):
        if "UNVERIFIABLE" not in l47:
            fails.append("lab47 no longer finds an unverifiable join at all")

    # --- Section 10.2's calm cell in a second coordinate, against lab48 ----
    # The claim is a NEGATIVE one - that the calm cell fails in both
    # coordinates - so the check has to confirm the failure rather than a
    # number.  If a future run made the calm age determinate, this fires.
    l48 = read_text(os.path.join(EXP, "lab48_age_by_regime.txt"))
    # The row now carries TWO intervals for the rate - the percentile bootstrap
    # and the fixed-b Fieller set - because lab63 and lab66 establish that the
    # bootstrap under-covers and the fixed-b set does not.  Groups, in order:
    # point, boot lo, boot hi, fixed-b lo, fixed-b hi, age, age lo, age hi,
    # at-zero count, off-grid count.  The fixed-b field can also read
    # "outside[...]" or "the whole line" when the set is not an interval, and
    # those forms are matched so a lab that stops identifying the rate is
    # reported rather than silently unparsed.
    AGE48 = re.compile(
        r"^\s*(.+?)\s+(-?\d+)%\s+\[\s*(-?\d+)%,\s*(-?\d+)%\]\s+"
        r"\[\s*(-?\d+)%,\s*(-?\d+)%\]\s+"
        r"([\d.]+)\s+\[\s*([\d.]+),\s*([\d.]+)\]\s+(\d+)\s+(\d+)\s*$")
    cells = {}
    for ln in l48.splitlines():
        m = AGE48.match(ln)
        if m:
            cells[m.group(1).strip()] = m.groups()[1:]
    checks += 1
    if len(cells) < 5:
        fails.append(f"lab48: could not parse the state table ({len(cells)} rows)")
    for state, label in (("calm (bottom tercile)", "10.2 calm age"),
                         ("stressed (top tercile)", "10.2 stressed age"),
                         ("VIX top 10%", "10.2 decile age"),
                         ("VIX top 5%", "10.2 top-5 age")):
        checks += 1
        if state not in cells:
            fails.append(f"lab48: no '{state}' row")
            continue
        # groups: point, boot lo, boot hi, fixed-b lo, fixed-b hi, age,
        # age lo, age hi, at-zero, off-grid.  The two fixed-b fields were
        # inserted when lab48 began reporting the covering interval beside
        # the bootstrap one, and this unpack is what has to move with them.
        _, _, _, _flo, _fhi, pt, lo, hi, at0, off = cells[state]
        want(label, f"{pt} days" if state.startswith("calm")
             else f"{pt} days old [{lo}, {hi}]" if "tercile" in state
             else f"{pt} [{lo}, {hi}]")
        if state.startswith("calm"):
            want("10.2 calm interval", f"[{lo}, {hi}]")
            checks += 1
            if int(at0) + int(off) == 0:
                fails.append("lab48: every calm replication now places the age on "
                             "the grid, so Section 10.2's count of failures is stale")
            want("10.2 calm failures", f"{int(at0) + int(off)} of 1,500 replications")
    checks += 1
    _w = re.search(r"([\d.]+)x on the age", " ".join(l48.split()))
    if not _w:
        fails.append("lab48: the relative-width comparison is gone")
    else:
        want("10.2 calm relative width", f"{_w.group(1)} times as wide")
    checks += 1
    if "Changing" not in l48 or "coordinate does not rescue the calm cell" not in \
            " ".join(l48.split()):
        fails.append("lab48 no longer concludes that the calm cell survives the "
                     "change of coordinate, which Section 10.2 states")

    # --- Section 10.1's control envelope, against lab49 -------------------
    l49 = " ".join(read_text(os.path.join(EXP, "lab49_control_envelope.txt")).split())
    for label, inlab, inpaper in (
            ("spread", "runs from 60.8% to 74.4%", "runs from 60.8% to 74.4%"),
            ("strongest rate", "strongest control by own-only skill, +MEAS, gives", "67.5%"),
            ):
        checks += 1
        if inlab not in l49:
            fails.append(f"lab49 no longer prints the {label} ('{inlab}')")
        want(f"10.1 envelope {label}", inpaper)
    # The three skill LEVELS Section 10.1 quotes are read out of lab49 rather
    # than transcribed: the benchmark correction moved every level in this lab
    # while leaving every rate untouched, and literals kept here would have gone
    # on agreeing with themselves.
    # the paired difference, read out of lab49 rather than transcribed
    _pg49 = re.search(r"paper minus \+MEAS at eleven weeks: (-?[\d.]+%) "
                      r"\[(-?[\d.]+%), (-?[\d.]+%)\]", l49)
    checks += 1
    if not _pg49:
        fails.append("lab49 no longer prints the paired gap Section 10.1 quotes")
    else:
        want("10.1 envelope paired gap",
             f"{_pg49.group(1)} [{_pg49.group(2)}, {_pg49.group(3)}]")
        # The gap is stated twice, in Section 10.1 and again in the Internet
        # Appendix, and the check above passed on whichever copy was current:
        # the main paper carried [-7.2, -1.5] against the appendix's correct
        # [-9.7, -0.7] for months.  Table S3's delta = 55 row also reads -4.1%
        # but is lab38's quantity with its own endpoint, so the stem includes
        # the phrase that names THIS one rather than the bare number.
        _g = _pg49.group(1).replace("%", "").replace(MINUS, "-")
        only_iv_re("envelope gap quoted consistently",
                   r"paired difference of " + re.escape(_g) + "%",
                   _pg49.group(2).rstrip("%").replace(MINUS, "-"),
                   _pg49.group(3).rstrip("%").replace(MINUS, "-"))
    _ch49 = re.search(r"strongest averaged across delays: \+MEAS "
                      r"\(([\d.]+) against ([\d.]+) for the paper's\)", l49)
    checks += 1
    if not _ch49:
        fails.append("lab49 no longer prints the champion's averaged own-only skill")
    else:
        want("10.1 envelope champion skill",
             f"{_ch49.group(1)} against the paper's {_ch49.group(2)}")
    _ar49 = re.search(r"scores ([\d.]+) own-only at the origin and (-?[\d.]+) "
                      r"at eleven weeks", l49)
    checks += 1
    if not _ar49:
        fails.append("lab49 no longer prints the lowest arm's two endpoints")
    else:
        want("10.1 envelope origin score", f"{_ar49.group(1)} own-only at the origin")
        checks += 1
        want("10.1 envelope delay score", f"{_ar49.group(2)} at eleven weeks")
    checks += 1
    if "significantly given implied" not in l49 or "0 of 6" not in l49:
        fails.append("lab49: the foreign block now adds significantly given implied "
                     "volatility under the stronger control, which Section 10.1 denies")
    want("10.1 envelope redundancy", "statistics from -1.94 to -2.86")

    # --- Section 11's filtered headline, against lab50 --------------------
    l50 = " ".join(read_text(os.path.join(EXP, "lab50_filtered_headline.txt")).split())
    for label, inlab, inpaper in (
            ("filtered rate", "at eleven weeks: 71.6% as published, 75.4% filtered",
             "75.4% against the published 71.6%"),
            ("where it wins", "better cross-sectional model at 3 of 6 delays",
             "better model at three of six delays")):
        checks += 1
        if inlab not in l50:
            fails.append(f"lab50 no longer prints the {label} ('{inlab}')")
        want(f"11 filtered {label}", inpaper)
    # The rate move and the three ages are read out of lab50 rather than
    # transcribed.  All four widened when this file's bootstrap block was
    # corrected: it had been resampling at 2h, like eight others.
    _pm50 = re.search(r"paired difference (\+[\d.]+%) \[(\+?-?[\d.]+%), "
                      r"(\+?-?[\d.]+%)\]", l50)
    _pub = re.search(r"paper's cross-sectional model: ([\d.]+) days "
                     r"\[([\d.]+), ([\d.]+)\]", l50)
    _fil = re.search(r"filtered cross-sectional model: ([\d.]+) days "
                     r"\[([\d.]+), ([\d.]+)\]", l50)
    _pa = re.search(r"paired difference: (-?[\d.]+) days "
                    r"\[(-?[\d.]+), (-?[\d.]+)\]", l50)
    checks += 1
    if not (_pm50 and _pub and _fil and _pa):
        fails.append("lab50 no longer prints the paired comparisons Section 11 quotes")
    else:
        want("11 filtered paired move",
             f"{_pm50.group(1)} [{_pm50.group(2)}, {_pm50.group(3)}]")
        want("11 filtered age",
             f"{_fil.group(1)} days old [{_fil.group(2)}, {_fil.group(3)}]")
        want("11 filtered published age",
             f"rather than the published {_pub.group(1)}")
        want("11 filtered paired age",
             f"{_pa.group(1).replace('-', MINUS)} days "
             f"[{_pa.group(2).replace('-', MINUS)}, "
             f"{_pa.group(3).replace('-', MINUS)}]")
    checks += 1
    if "the headline is LARGER" not in l50:
        fails.append("lab50: the filtered headline is no longer larger than the "
                     "published one, which Section 11 states")

    # --- Section 9.4's whose-options-market table, against lab51 ----------
    # Table 16 is rebuilt from the lab's own summary rather than compared
    # cell by cell to prose, so a re-run that moves any target fails here
    # before anyone reads the section.
    l51 = read_text(os.path.join(EXP, "lab51_foreign_options.txt"))
    ROW51 = re.compile(r"^\s*([A-Z0-9]+)\s+([\d.]+)\s+([\d.]+%)\s+([\d.]+%)\s+"
                       r"([+-][\d.]+)\s+([+-][\d.]+)\s*$")   # both columns signed
    t51 = {m.group(1): m.groups()[1:]
           for m in (ROW51.match(ln) for ln in l51.splitlines()) if m}
    checks += 1
    if len(t51) != 8:
        fails.append(f"lab51: could not parse the eight-target summary ({len(t51)})")
    for tag in sorted(t51):
        rho, riv, rb, inc, z = t51[tag]
        want(f"9.4 row {tag}", f"{rho} {riv} {rb} {inc} {z}")
    l51f = " ".join(l51.split())
    for label, inlab, inpaper in (
            ("own-options group", "own options 72.8% 103.3%", "103% of the delay"),
            ("foreign group", "foreign options 64.8% 68.1%", "68% against 65%"),
            ("own redundancy", "significantly in 0 of 8 cells", "none of the eight cells"),

            ("gradient", "-0.38 on the levels", "seven-regressor version read -0.38")):
        checks += 1
        if inlab not in l51f:
            fails.append(f"lab51 no longer prints the {label} ('{inlab}')")
        want(f"9.4 {label}", inpaper)
    # The count of significant cells and whether any sits at the headline delay,
    # both read out of lab51.  Section 9.4 named three targets as significant at
    # eleven weeks and none of the three survived the HAC bandwidth correction,
    # so the section now reports the count, the one cell that does clear, and
    # the withdrawal.  Both halves are checked: a count that drifts, and a
    # section that goes back to naming targets it can no longer support.
    _c51 = re.search(r"foreign block still adds given implied volatility in "
                     r"(\d+) of (\d+) cells(, none of them at eleven weeks\.|"
                     r", significantly at eleven weeks for ([^.]+)\.)", l51f)
    checks += 1
    if not _c51:
        fails.append("lab51 no longer prints its foreign-options significance count")
    else:
        want("9.4 foreign significance",
             f"significantly in {_c51.group(1)} of {_c51.group(2)} cells")
        checks += 1
        if _c51.group(3).startswith(", none"):
            forbid("9.4 withdrawn named targets", "for the FTSE, the Nifty and the ASX")
            want("9.4 withdrawal stated",
                 "none of which clears the threshold")
        else:
            want("9.4 named targets", _c51.group(4))
    want("9.4 scope sentence",
         "about the forecaster's information set and not about which option chains exist")
    checks += 1
    if "31 points" not in l51f:
        fails.append("lab51: the own-options gap is no longer 31 points, which the "
                     "abstract and Sections 9.4 and 14 all quote")
    want("abstract whose-options",
         "three points apart rather than thirty-one")

    # --- Section 9.4's estimation-cost correction, against lab52 ----------
    # Section 9.4's counts are quoted twice over: once with the seven-regressor
    # block, which matches Section 8's design, and once with the block the paper
    # tells a practitioner to carry.  The second is the one the conclusion rests
    # on, so both rows of the group table are anchored here, and so is the
    # Bovespa reversal, which is the reason the correction had to be made.
    l52 = " ".join(read_text(os.path.join(EXP,
                   "lab52_compressed_foreign_options.txt")).split())
    # Every figure below is read out of lab52's own four-row summary and its two
    # closing diagnostics rather than transcribed.  These were literals until a
    # benchmark correction moved all of them at once, and a literal kept here
    # cannot fail a paper that has gone stale with it.
    def _l52(pat, what):
        m = re.search(pat, l52)
        if not m:
            fails.append(f"lab52 no longer prints the {what}")
        return m
    _os = _l52(r"own options seven (-?\+?[\d.]+)", "own-options seven-regressor increment")
    _oc = _l52(r"own options mean (\+?-?[\d.]+) ([\d.]+) (\d+ of \d+)",
               "own-options compressed row")
    _fs = _l52(r"another index's seven (\+?-?[\d.]+) [\d.-]+ (\d+ of \d+)",
               "foreign-options seven-regressor row")
    _fc = _l52(r"another index's mean (\+?-?[\d.]+) ([\d.]+) (\d+ of \d+)",
               "foreign-options compressed row")
    _bv = _l52(r"Bovespa, seven regressors: (-?[\d.]+) compressed: (\+?-?[\d.]+)",
               "Bovespa reversal")
    _sp = _l52(r"correlates (-?[\d.]+) with the increment", "spanning null correlation")
    _sg = _l52(r"target correlates (-?[\d.]+)", "sharpened gradient")
    for label, inlab, inpaper in (
            ("own seven", _os and _os.group(0),
             _os and _fc and f"from {_os.group(1)} to {_oc.group(1)}"),
            ("own compressed", _oc and _oc.group(0),
             _oc and f"stays at {_oc.group(3)}"),
            ("foreign seven", _fs and _fs.group(0),
             _fs and _fc and f"from {_fs.group(1)} to {_fc.group(1)}"),
            ("foreign compressed", _fc and _fc.group(0),
             _fs and _fc and f"from {_fs.group(2)} to <b>{_fc.group(3)}</b>"),
            ("mean statistic", _fc and _fc.group(0),
             _fc and f"mean Giacomini-White statistic of {_fc.group(2)}"),
            ("sharpened gradient", _sg and _sg.group(0),
             _sg and f"correlates {_sg.group(1)} across the eight"),
            ("bovespa seven", _bv and _bv.group(0),
             _bv and f"{_bv.group(1)} with seven regressors"),
            ("bovespa compressed", _bv and _bv.group(0),
             _bv and f"{_bv.group(2)} with one"),
            ("spanning null", _sp and _sp.group(0),
             _sp and f"correlates {_sp.group(1)} with the increment")):
        if inlab is None or inpaper is None:
            continue
        checks += 1
        if inlab not in l52:
            fails.append(f"lab52 no longer prints the {label} ('{inlab}')")
        want(f"9.4 compressed {label}", inpaper.replace("<b>", "").replace("</b>", ""))
    checks += 1
    if "the clock is not the cause" not in l52:
        fails.append("lab52: breaking admissibility now rescues Bovespa, so Section "
                     "9.4's ruling-out of the clock is stale")
    want("9.4 clock diagnostic", "leaves the increment negative at all three delays")

    # --- Table S31's common-window coupling, against lab64 ------------------
    # Section S38 exists because two of the eight targets end in 2021 and are
    # the two the fitted line leans on.  Its table is rebuilt here from the
    # lab's own rows, so a re-run that moves any of them fails rather than
    # printing a stale comparison, and the ordering statistics that carry the
    # section's claim are pinned as numbers rather than as a sentence.
    _l64p = os.path.join(EXP, "lab64_coupling_common_window.txt")
    if os.path.isfile(_l64p):
        _l64 = read_text(_l64p)
        _R64 = re.compile(r"^\s*([A-Z0-9]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
                          r"(-?[\d.]+)\s+(-?[\d.]+)\s*$")
        _rows = [m.groups() for m in (_R64.match(ln) for ln in _l64.splitlines()) if m]
        checks += 1
        if len(_rows) != 8:
            fails.append(f"lab64: parsed {len(_rows)} target rows, Table S31 needs 8")
        else:
            _nm = {"KSE100": "KSE-100", "CSEALL": "Colombo All-Share",
                   "JKSE": "IDX Composite", "KLCI": "FTSE Bursa KLCI",
                   "NSEI": "Nifty 50", "N225": "Nikkei 225",
                   "HSI": "Hang Seng", "AXJO": "S&P/ASX 200"}
            _r64 = {m.group(1): (m.group(2), m.group(3))
                    for m in re.finditer(
                        r"^\s*([A-Z0-9]+)\s+(-?[\d.]+)%\s+(-?[\d.]+)%\s+"
                        r"[+-][\d.]+\s+(-?[\d.]+)\s*$", _l64, re.M)}
            for _tg, _full, _com, _lost, _cf, _cc in _rows:
                _rf, _rc = _r64.get(_tg, (None, None))
                if _rf is None:
                    checks += 1
                    fails.append(f"lab64 prints no R(55) row for {_tg}")
                    continue
                want(f"Table S31 {_tg}",
                     f"{_nm[_tg]} {int(_full):,} {int(_com):,} {_cf} {_cc} "
                     f"{_rf}% {_rc}%".replace("-", MINUS))
        for _lbl, _pat, _txt in (
                ("S38 common-window Spearman",
                 r"common window\s+Pearson ([+-][\d.]+)\s+Spearman ([+-][\d.]+)",
                 "is +0.976 on the full window and {1} on the common one"),
                ("S38 worst leave-one-out",
                 r"worst leave-one-out on the common window: "
                 r"Pearson ([+-][\d.]+), Spearman ([+-][\d.]+)",
                 "the worst leave-one-out value is {1} in both")):
            _m = re.search(_pat, _l64)
            checks += 1
            if not _m:
                fails.append(f"lab64 no longer prints the {_lbl}")
            else:
                want(_lbl, _txt.format(_m.group(1), _m.group(2)))

    # --- Section 9.5's no-options targets, against lab53 -------------------
    # Table 17 is rebuilt from the lab's own sorted summary, so a re-run that
    # moves any target - or reorders them - fails here rather than in print.
    # The coupling correlation is the subsection's claim and is checked as a
    # number, not as a sentence.
    l53 = read_text(os.path.join(EXP, "lab53_no_options_targets.txt"))
    R53 = re.compile(r"^\s*([A-Z0-9]+)\s+(none published|exists, not held)\s+"
                     r"(-?[\d.]+)\s+(-?[\d.]+)%\s+(-?[\d.]+)%\s+([+-][\d.]+)\s+"
                     r"([+-][\d.]+)\s*$")
    t53 = [m.groups() for m in (R53.match(ln) for ln in l53.splitlines()) if m]
    checks += 1
    if len(t53) != 8:
        fails.append(f"lab53: could not parse the eight-target table ({len(t53)})")
    for tag, kind, coup, rb, ri, inc, z in t53:
        want(f"9.5 row {tag}", f"{coup} {rb}% {ri}% {inc} {z}")
    checks += 1
    if [r[1] for r in t53].count("none published") != 4:
        fails.append("lab53: the four no-volatility-index targets are no longer four")
    l53f = " ".join(l53.split())
    for label, inlab, inpaper in (
            ("coupling correlation", "coupling against R(breadth): +0.98",
             "coupling and R(breadth) correlate +0.98"),
            ("indonesia rate", "Malaysia repair 42% and 44%", "41.8% and 44.0%"),
            ("dead pair", "worth less than nothing at eleven weeks: CSEALL, KSE100",
             "&minus;29.7% and &minus;11.4%"),
            ("pakistan coupling", "KSE100 none published -0.115",
             "negative at &minus;0.115")):
        checks += 1
        if inlab not in l53f:
            fails.append(f"lab53 no longer prints the {label} ('{inlab}')")
        want(f"9.5 {label}", inpaper.replace("&minus;", MINUS).replace(MINUS, "-"))
    # the two conclusions the subsection rests on, as behaviour rather than text
    # The two statistics Section 9.5 quotes, read out of lab53's own table rather
    # than listed here.  They were +3.52 and +2.00 under the nine-lag kernel and
    # a literal kept in this file would have gone on agreeing with a stale
    # paper.  The second one falls below 1.96 at the measured bandwidth, so the
    # section has to say so rather than quote it as significant, and the check
    # below requires the lab and the paper to agree about which of the two
    # clears.
    _jk = [r for r in t53 if r[0] == "JKSE"]
    _kl = [r for r in t53 if r[0] == "KLCI"]
    checks += 1
    if not (_jk and _kl):
        fails.append("lab53 no longer prints the Indonesian and Malaysian rows that "
                     "Section 9.5 quotes")
    else:
        _zj, _zk = _jk[0][6], _kl[0][6]
        want("9.5 statistics", f"{_zj} and {_zk}")
        checks += 1
        _both = abs(float(_zj)) > 1.96 and abs(float(_zk)) > 1.96
        if _both:
            forbid("9.5 stale hedge", "falls short of the threshold")
        else:
            want("9.5 significance hedge", "falls short of the threshold")
    want("9.5 provenance", "no published implied-volatility index")
    want("9.5 containment", "no other number in this paper depends on them")
    want("14 coupling rule", "tracks the substitution rate at +0.98")

    # --- Figure 3's geometry, against lab53 --------------------------------
    # The scatter draws the same eight points as the table beside it, plus a
    # fitted line.  The line is a claim, so it is checked: its crossing comes
    # from the lab rather than from the drawing, and the caption's refusal to
    # call that crossing a threshold is checked too, because it would be the
    # easy overclaim to make from a picture.
    l53 = " ".join(lab("lab53_no_options_targets").split())
    for _label, _needle in (
            ("fit", "least squares fit: R(breadth) = -0.249 +1.154 * coupling"),
            ("crossing", "the fit crosses zero at a coupling of +0.216"),
            ("negatives", "targets with R(breadth) < 0: 2"),
            ("not a threshold", "the crossing is not a threshold and is not")):
        checks += 1
        if _needle not in l53:
            fails.append(f"lab53 no longer prints the {_label} that Figure 3 draws "
                         f"('{_needle}')")
    want("fig3 crossing", "crosses zero at a coupling of 0.22")
    want("fig3 not a threshold", "is not a threshold")
    want("fig3 both groups", "the relation runs through both groups")
    checks += 1
    if 'aria-label' not in txt and '<svg' in read_text(path):
        pass
    _svgs = len(re.findall(r"<svg", read_text(path)))
    checks += 1
    if _svgs < 3:
        fails.append(f"the paper has {_svgs} figures; Figure 3 (coupling against the "
                     "rate) has gone")
    # every figure must carry an aria-label, since the SVGs are hand written
    checks += 1
    _unlabelled = len(re.findall(r"<svg(?![^>]*aria-label)", read_text(path)))
    if _unlabelled:
        fails.append(f"{_unlabelled} figure(s) have no aria-label")

    # --- Section 15's monthly feasibility study, against lab54 -------------
    # Table 22 is rebuilt from the lab's two runs.  The clean and the appraised
    # rows come from separate passes, so they are matched by delay rather than
    # by position, and the subsection's claim - that the shape survives and the
    # appraised column barely differs - is checked as behaviour.
    l54 = read_text(os.path.join(EXP, "lab54_monthly_feasibility.txt"))
    R54 = re.compile(r"^\s*(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)%\s+"
                     r"\[(-?\d+)%, (-?\d+)%\]\s*$")
    runs, cur = [], []
    for ln in l54.splitlines():
        if "test months," in ln:
            if cur:
                runs.append(cur)
            cur = []
        m = R54.match(ln)
        if m:
            cur.append(m.groups())
    if cur:
        runs.append(cur)
    checks += 1
    if len(runs) != 2 or not all(len(r) == 4 for r in runs):
        fails.append(f"lab54: expected two runs of four delays, got "
                     f"{[len(r) for r in runs]}")
    else:
        clean, appr = runs
        for (d, so, sc, rate, lo, hi) in clean:
            want(f"15 monthly delay {d} own", so)
            want(f"15 monthly delay {d} cross", sc)
            want(f"15 monthly delay {d} rate", f"{rate}% [{lo}, {hi}]")
        for (d, _, _, rate, lo, hi) in appr:
            want(f"15 appraised delay {d}", f"{rate}% [{lo}, {hi}]")
        checks += 1
        one = [r for r in clean if r[0] == "1"][0]
        if not (int(one[4]) < 0 < int(one[5])):
            fails.append("lab54: the one-month rate no longer contains zero, so "
                         "Section 15's 'the shape is the daily result' is stale")
    l54f = " ".join(l54.split())
    for label, inlab, inpaper in (
            ("burn-in arithmetic", "quarterly 4 438", "four hundred and thirty-eight years"),
            ("test months", "109 test months", "109 test months"),
            ("width requirement", "40% 121 27", "121 test months"),
            ("wider requirement", "30% 215 35", "215")):
        checks += 1
        if inlab not in l54f:
            fails.append(f"lab54 no longer prints the {label} ('{inlab}')")
        want(f"15 {label}", inpaper)
    checks += 1
    if "frequency was not the obstruction" not in l54f:
        fails.append("lab54 no longer concludes that the measurement survives at "
                     "monthly frequency, which Section 15 states")
    want("9.1 restated limitation", "what was missing was a series and not a method")
    want("11 feasibility pointer", "survives being re-specified at monthly frequency")

    # --- Section 16's measured illiquid study, against lab55 and lab56 -----
    # Every figure in Section 16 is rebuilt from the two labs rather than
    # spot-checked.  Table 23 is matched metro by metro across three separate
    # printed blocks in lab55 (persistence, coupling, and the two per-metro rate
    # tables), because a row that silently paired the wrong metro's coupling with
    # the wrong metro's rate is exactly the error a reader could not catch.
    l55 = read_text(os.path.join(EXP, "lab55_illiquid_measured.txt"))
    l56 = read_text(os.path.join(EXP, "lab56_seasonal_and_breadth.txt"))
    l55f, l56f = " ".join(l55.split()), " ".join(l56.split())

    # persistence and coupling, per metro
    pers = {m.group(1).strip(): m.group(5) for m in re.finditer(
        r"^\s{2,}([A-Za-z ]+?)\s+(\d{4}-\d{2})\s+(\d+)\s+([\d.]+)%\s+([\d.]+)\s*$",
        l55, re.M)}
    coup = {m.group(1).strip(): m.group(2) for m in re.finditer(
        r"^\s{2,}([A-Za-z ]+?)\s+(0\.\d{3})\s*$", l55, re.M)}
    permetro = []
    for chunk in l55.split("Per metro at one quarterly appraisal cycle")[1:]:
        permetro.append({m.group(1).strip(): m.group(4) for m in re.finditer(
            r"^\s{2,}([A-Za-z ]+?)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)%\s*$",
            chunk, re.M)})
    checks += 1
    if len(permetro) != 2:
        fails.append(f"lab55: expected two per-metro rate tables, got {len(permetro)}")
    balanced = ["Boston", "Charlotte", "Chicago", "Cleveland", "Denver", "Las Vegas",
                "Los Angeles", "Miami", "New York", "Portland", "San Diego",
                "San Francisco", "Tampa", "Washington"]
    checks += 1
    if sorted(coup) != sorted(balanced):
        fails.append("lab55: the coupling block no longer lists exactly the fourteen "
                     f"balanced metros; got {sorted(coup)}")
    for city in balanced:
        for what, val in (("persistence", pers.get(city)), ("coupling", coup.get(city))):
            checks += 1
            if val is None:
                fails.append(f"lab55 no longer prints {what} for {city}")
        if len(permetro) == 2:
            row = [city, "1987-01", pers.get(city), coup.get(city),
                   permetro[0].get(city), permetro[1].get(city)]
            checks += 1
            if any(v is None for v in row):
                fails.append(f"lab55: Table 23's row for {city} cannot be rebuilt")
            else:
                # txt is tag-free and minus-normalised, so the row is checked as the
                # sequence of its cell values rather than as markup.
                cells = " ".join([row[0], row[1], row[2], row[3],
                                  row[4] + "%", row[5] + "%"])
                if cells not in txt:
                    fails.append(f"Table 23's row for {city} does not match lab55: "
                                 f"expected '{cells}'")

    # the panel's headline descriptors
    for label, needle in (
            ("persistence span", "autocorrelation spans 0.601 to 0.939, mean 0.804"),
            ("coupling span", "range 0.662 to 0.866, width 0.203, mean 0.763"),
            ("return test months", "310 test months per metro"),
            ("variance test months", "250 test months per metro")):
        checks += 1
        if needle not in l55f:
            fails.append(f"lab55 no longer prints the {label} ('{needle}')")
    want("16 persistence span", "from 0.601 at Cleveland to 0.939 at Phoenix with a "
                                "mean of 0.804")
    want("16 coupling span", "spans 0.662 to 0.866 with a mean of 0.763")
    want("16 observations", "9,143")
    # The band ratio, READ from lab55, and lab55's own idea of lab53's range
    # checked against lab53.  A literal "0.35 times the width" stood here and
    # froze a figure derived from a stale pair: lab55 hardcoded lab53's
    # coupling span as "0.228 to 0.808", which is lab53's SECOND-lowest
    # coupling, so the equity band looked 37% narrower than it is and this
    # panel's band looked a larger fraction of it - the direction that
    # flatters the slope test the section declines to run.
    _m55b = re.search(r"A gradient estimated over a band ([\d.]+) times", l55f)
    checks += 1
    if not _m55b:
        fails.append("lab55 no longer prints the band ratio Section S31 quotes")
    else:
        want("16 slope refused", f"{_m55b.group(1)} times the width")
    # lab53's range is lab53's to state.  lab55 compares against it and so
    # must agree with it; nothing checked that until the comparison went
    # stale, and a lab quoting another lab from memory is the failure this
    # project has now found five times.
    _l53c = read_text(os.path.join(EXP, "lab53_no_options_targets.txt"))
    _c53 = [float(_m) for _m in re.findall(
        r"^\s{2,}\w+\s+[A-Za-z0-9 &/.-]+?\s+[\d.]+\s+\d+\s+\d+\s+"
        r"\d{4}-\d{2}-\d{2}\s+(-?[\d.]+)\s*$", _l53c, re.M)]
    _m55r = re.search(r"coupling range in lab53:\s+(-?[\d.]+) to (-?[\d.]+), "
                      r"width ([\d.]+)", l55f)
    checks += 1
    if len(_c53) < 8 or not _m55r:
        fails.append(f"could not read lab53's couplings ({len(_c53)} found) or "
                     f"lab55's quotation of their range")
    else:
        _lo53, _hi53 = min(_c53), max(_c53)
        checks += 1
        if (abs(float(_m55r.group(1)) - _lo53) > 5e-4
                or abs(float(_m55r.group(2)) - _hi53) > 5e-4):
            fails.append(f"lab55 says lab53's coupling range is "
                         f"{_m55r.group(1)} to {_m55r.group(2)}; lab53's own "
                         f"table gives {_lo53:.3f} to {_hi53:.3f}")
        want("S31 gives lab53's coupling range as lab53 prints it",
             f"against {'&minus;' if _lo53 < 0 else ''}{abs(_lo53):.3f} to "
             f"{_hi53:.3f} in Section S14".replace("&minus;", "-"))
        want("S31 records the superseded range and why it was wrong",
             "which is its second-lowest coupling rather than its lowest")

    # Table 24, the peer block priced as a set
    price = re.findall(r"^\s{2,}(every peer separately|peer mean \(lab52's block\)|"
                       r"peer mean \+ its 3-month mean)\s+(\d+)\s+([\d.]+)\s+"
                       r"([\d.]+)\s+([+-][\d.]+)\s*$", l55, re.M)
    menu = re.findall(r"^\s{2,}(every peer separately|peer mean \(lab52's block\)|"
                      r"peer mean \+ its 3-month mean)((?:\s+-?[\d.]+%){4})"
                      r"(   <- selected)?\s*$", l55, re.M)
    checks += 1
    if len(price) != 6 or len(menu) != 6:
        fails.append(f"lab55: expected six priced blocks and six rate rows, got "
                     f"{len(price)} and {len(menu)}")
    else:
        for (_, cols, so, sc, bill) in price:
            for v in (cols, so, sc):
                want(f"16 block price {v}", v)
            want("16 block bill " + bill,
                 bill if not bill.startswith("-") else MINUS + bill[1:])
        for (_, rates, _) in menu:
            for r in rates.split():
                want("16 block rate " + r,
                     r if not r.startswith("-") else MINUS + r[1:])
        # the finding that makes compression part of the method
        volrow = [m for m in menu if m[0] == "every peer separately"][1]
        neg = [r for r in volrow[1].split() if r.startswith("-")]
        checks += 1
        if len(neg) != 3:
            fails.append("lab55: the widest peer block no longer makes the variance "
                         f"rate negative at exactly three delays ({neg}), which "
                         "Section 16 states")
        cmprow = [m for m in menu if m[0] == "peer mean (lab52's block)"][1]
        checks += 1
        if any(r.startswith("-") for r in cmprow[1].split()):
            fails.append("lab55: the compressed block no longer makes the variance "
                         "rate positive at every delay, which Section 16 states")
        want("16 compression is the method", "compression is part of the method and "
                                             "not a robustness check")
        # Both figures come out of the variance target's price table above,
        # which `price` has already parsed: the widest block's bill and the
        # compressed one's.  They were literals until a benchmark correction
        # moved them.
        _wide = [p_ for p_ in price if p_[0] == "every peer separately"][1][4]
        _cmp = [p_ for p_ in price if p_[0] == "peer mean (lab52's block)"][1][4]
        want("16 bill sizes", f"cost {_wide.lstrip('+-')} of")
        want("16 bill removed", f"Compression leaves {_cmp.lstrip('+-')} of it")

    # Table 25, the two coordinates, rebuilt row by row
    rows = re.findall(r"^\s*(\d)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)%\s+"
                      r"\[(-?\d+)%, (-?\d+)%\]\s+(\d+) of (\d+)\s*$", l55, re.M)
    checks += 1
    if len(rows) != 8:
        fails.append(f"lab55: expected eight delay rows across two targets, got {len(rows)}")
    else:
        for (d, so, sc, rate, lo, hi, pos, tot) in rows:
            unit = "month" if d == "1" else "months"
            iv = f"{rate}% [{lo.replace('-', MINUS)}, {hi.replace('-', MINUS)}]"
            want(f"16 delay {d} own", so)
            want(f"16 delay {d} cross", sc)
            want(f"16 delay {d} rate", iv)
            want(f"16 delay {d} count", f"{pos} of {tot}")
        # the shape claim: one month cannot be told from zero on either coordinate
        for (d, _, _, _, lo, hi, _, _) in rows:
            if d == "1":
                checks += 1
                if not (int(lo) < 0 < int(hi)):
                    fails.append("lab55: the one-month rate no longer contains zero, "
                                 "so Section 16's 'the shape of Table 1 survives' is stale")
        # both headline coordinates must exclude zero at a quarterly staleness
        for (d, _, _, rate, lo, hi, _, _) in rows:
            if d == "3":
                checks += 1
                if int(lo) <= 0:
                    fails.append(f"lab55: the quarterly rate {rate}% no longer excludes "
                                 "zero, which is Section 16's central claim")

    # Table 26 and the placebo, against lab56
    for label, needle, inpaper in (
            ("seasonal share", "month-of-year pattern explains 19.9% of monthly return",
             "19.9% of monthly return variance"),
            ("worst metro", "up to 41.7% at Cleveland", "41.7% at Cleveland"),
            ("profile correlation", "mean +0.894, min +0.542, max +0.993",
             "correlate +0.894 across metros"),
            ("return placebo", "at delta = 3: real 73.0%, placebo 4.2%",
             "placebo earns 4.2% where the real peer mean earns 73.0%"),
            ("variance placebo", "at delta = 3: real 18.6%, placebo -2.8%",
             "earns 18.6%"),
            ("return treatments",
             "at delta = 3: dummies 67.8%, deseasonalised 71.0%, raw 73.0%",
             "73.0% against the published 67.8%, with causal deseasonalisation in "
             "between at 71.0%"),
            ("variance treatments",
             "at delta = 3: dummies 19.0%, deseasonalised 16.0%, raw 18.6%",
             "19.0%, 16.0% and 18.6%"),
            ("return breadth", "at delta = 3: 13 peers 67.8%, 19 peers 67.5%",
             "67.8% to 67.5%"),
            ("variance breadth", "at delta = 3: 13 peers 19.0%, 19 peers 19.9%",
             "19.0% to 19.9%")):
        checks += 1
        if needle not in l56f:
            fails.append(f"lab56 no longer prints the {label} ('{needle}')")
        want(f"16 {label}", inpaper)
    # the placebo verdict must actually be the one lab56 reaches
    checks += 1
    if "The rate is information and not" not in l56f:
        fails.append("lab56 no longer concludes the rate is information rather than a "
                     "calendar effect, which Section 16 states")
    want("16 placebo purpose", "a test that could have convicted the design and did not")

    # --- Section 9's placebo, which now reports SETS rather than points ----
    # The sentence used to quote "-1% [-33, 29]", which is lab10's rate at one
    # day's delay under the REAL block: a figure from a different experiment,
    # and the one error of this round that was in the posted version.  The
    # placebo returns a negative rate at every delay tested, which is a
    # stronger null than the sentence claimed, and both of the properties the
    # paper draws from it are recomputed here rather than read.
    _l06p = read_text(os.path.join(EXP, "lab06_inference.txt"))
    _plc = {}
    for _m in re.finditer(r"^\s*(\d+)\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+"
                          r"(-?\d+\.\d)%\s+[\d.]+\s+\[\s*(-?\d+)%,"
                          r"\s*(-?\d+)%\]", _l06p.split("3. PLACEBO")[-1], re.M):
        _plc[int(_m.group(1))] = (_m.group(2), int(_m.group(3)), int(_m.group(4)))
    checks += 1
    if len(_plc) != 3:
        fails.append(f"lab06's placebo now prints {len(_plc)} delays, not 3, so "
                     f"Section 9's falsification test rests on a short read")
    else:
        _pk = sorted(_plc)
        for _d in _pk:
            _pt, _lo, _hi = _plc[_d]
            want(f"9 the placebo set at delta={_d}",
                 f"-{abs(float(_pt)):.1f}% [{_lo}, {_hi}]".replace("[-", "[-"))
        # property one: every set contains zero
        checks += 1
        if not all(_plc[_d][1] <= 0 <= _plc[_d][2] for _d in _pk):
            fails.append(f"Section 9 says each placebo set contains zero; lab06 "
                         f"now gives {[(d, _plc[d]) for d in _pk]}")
        want("9 the placebo sets contain zero",
             "Each contains zero, so the placebo cannot be distinguished from no "
             "effect at all")
        # property two: every set lies entirely below the headline's.  The
        # headline is re-read here rather than taken from `rates`, because by
        # this point in the file that name has been rebound to a different
        # table - the fifth name collision this file has been caught by.
        _hm = re.search(r"^\s*55\s+[\d.]+\s+(\d+)% \[\s*(-?\d+)%,\s*(-?\d+)%\]",
                        read_text(os.path.join(EXP, "lab10_loss_scale.txt")).split(
                            "1a. THE SAME RATES, INVERTED")[-1], re.M)
        checks += 1
        if not _hm:
            fails.append("lab10 part 1a no longer prints the headline set, so the "
                         "placebo's disjointness claim rests on nothing")
        else:
            _hlo = int(_hm.group(2))
            checks += 1
            if not all(_plc[_d][2] < _hlo for _d in _pk):
                fails.append(f"Section 9 says each placebo set lies entirely below "
                             f"the headline's, whose lower endpoint is {_hlo}; the "
                             f"placebo upper endpoints are "
                             f"{[_plc[d][2] for d in _pk]}")
            want("9 the placebo sets are disjoint from the headline",
                 "each lies entirely below the headline")
        forbid("9's placebo quotes the real block's one-day rate",
               "returns a rate of -1% [-33, 29]")
    # Table 26's own rows
    pb = re.findall(r"^\s*(\d)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)%\s+"
                    r"(\d+) of (\d+)\s*$", l56, re.M)
    checks += 1
    if len(pb) < 8:
        fails.append(f"lab56: expected at least eight delay rows to rebuild Table 26, "
                     f"got {len(pb)}")

    # Section 16 must not re-assert the limitation it removes, and Section 11
    # must not still call the illiquid case undemonstrated.
    forbid("11 stale limitation",
           "The illiquid-asset case that motivates the whole exercise is argued rather "
           "than demonstrated")
    forbid("15 stale closing", "this paper did not have such a series")
    want("16 remaining limits", "nobody holds a Case-Shiller index")
    want("12 measured pointer", "the illiquid case itself, argued rather than demonstrated")

    # --- every in-text SECTION reference must resolve --------------------
    # The table-reference check below has caught a shifted table three times.
    # Renumbering a section is the same hazard with no guard on it, and this
    # paper's sections were renumbered once to make room for Section 16.
    #
    # Three things were wrong with the first version of this, and all three let
    # a dangling pointer through 1,782 checks.
    #
    # It read only "Section N", singular, and took only the first number.
    # Every reference that had outlived its target was therefore invisible to
    # it, because they are all in plural lists: "(Sections 8.1, S18)" and
    # "(Sections 1, 9.2)" in the claim table, and "Sections 9.4 and 9.5" in the
    # appendix - five dead pointers, two of them in the evidence column of the
    # table that exists to stop the paper over-claiming.  The reference grammar
    # is now read the way the paper writes it: singular or plural, lists,
    # ranges, and both sides of a range.
    #
    # It collected only bare-numbered headings, so the S-prefixed subsections
    # were not in the set of things a reference could resolve to - which is how
    # two subsections of S28 came to be numbered "8.2" and "8.3", sitting among
    # S1 to S41 with nothing objecting.
    #
    # And it ran over the manuscript alone.  main() is called once, with the
    # manuscript, and the appendix reaches most checks only through the
    # concatenated `txt`; a check that reads `read_text(path)` therefore never
    # saw the appendix at all.  Both documents are walked here, each held to
    # the numbers it can own: the manuscript owns bare numbers, the appendix
    # owns S-numbers, and each cites the other's freely.
    for _doc in [path, _supp]:
        if not os.path.isfile(_doc):
            continue
        _src = read_text(_doc)
        _is_app = "Internet Appendix for" in _src[:2000]
        _who = "the Internet Appendix" if _is_app else "the manuscript"
        # Both levels, and both shapes at each level.  The appendix numbers its
        # sections on one flat counter across two levels - S1 is an umbrella
        # and S2 to S8 are h3 headings beneath it - so an h3 pattern that
        # required a dot ("S28.1") left S2 to S8 and S10 to S15 out of the set
        # a reference could resolve to, and reported every pointer at them as
        # dangling.
        _heads = set()
        for _lv in ("h2", "h3"):
            _heads |= {m.group(1) for m in
                       re.finditer(r"<" + _lv + r"[^>]*>(S?\d+(?:\.\d+)?)[.\s]", _src)}
        _REF = (r"Sections?\s+((?:S?\d+(?:\.\d+)?)"
                r"(?:\s*(?:,|and|&amp;|to|through|&ndash;|-)\s*S?\d+(?:\.\d+)?)*)")
        _seen = set()
        for _m in re.finditer(_REF, norm_iv(plain(_doc))):
            for _n in re.findall(r"S?\d+(?:\.\d+)?", _m.group(1)):
                _seen.add(_n)
        for _n in sorted(_seen):
            if _n.startswith("S") != _is_app:
                continue
            checks += 1
            if _n not in _heads:
                fails.append(f"{_who} refers to Section {_n}, which it does not have")

        # --- and a section may not cite itself for its own working --------
        # "the price of estimating parameters instead of knowing them, and
        # Section 6 measures a cost of the right order" sat inside Section 6.
        # Both the reference and the section exist, so nothing above could see
        # it; it is only wrong once you notice which section you are reading.
        # Captions are exempt: "Figure 1. The design of Section 4, on two
        # panels." sits inside Section 4 and names what the figure draws rather
        # than sending a reader anywhere.  Only running prose can misdirect.
        _nocap = re.sub(r"<(figcaption|caption)[^>]*>.*?</\1>", " ", _src, flags=re.S)
        _bnd = [(m.start(), m.group(1)) for m in
                re.finditer(r"<h2[^>]*>(S?\d+)\.\s", _nocap)] + [(len(_nocap), None)]
        for _i, (_st, _nm) in enumerate(_bnd[:-1]):
            _body = re.sub(r"<[^>]+>", " ", _nocap[_st:_bnd[_i + 1][0]])
            checks += 1
            if re.search(r"Sections?\s+" + re.escape(_nm) + r"(?![\d.])", _body):
                fails.append(f"Section {_nm} of {_who} refers the reader to Section "
                             f"{_nm}, which is the section they are reading")

        # --- an appendix may not name itself as somewhere else ------------
        # Relocating a manuscript section into the Internet Appendix carries
        # its deixis with it: "Section S33 of the Internet Appendix rather than
        # here" became a sentence inside the Internet Appendix in which S33 and
        # "here" are the same document.  Forty-one such cross-references were
        # shipping in the JFEc appendix and thirty-two in the SSRN one, none of
        # them visible to a check that reads arithmetic.  build_variants.py
        # rewrites what it relocates; this catches a form it does not handle,
        # and stops the phrasing being reintroduced by hand.
        if not _is_app:
            continue

        # --- every appendix heading must carry the S prefix ---------------
        # Two subsections of S28 were numbered "8.2" and "8.3", in the
        # manuscript's scheme, sitting among S1 to S41.  Nothing referenced
        # them, so the resolution check above could not see it: an orphaned
        # heading dangles in the other direction, and only a reader looking at
        # the contents page would notice the appendix counting 8.2 after S28.1.
        _bare = [m.group(1) for m in
                 re.finditer(r"<h[23][^>]*>(\d+(?:\.\d+)?)[.\s]", _src)]
        checks += 1
        if _bare:
            fails.append(
                f"these Internet Appendix headings are numbered in the "
                f"manuscript's scheme instead of the appendix's: "
                f"{', '.join(_bare)}; every appendix heading takes an S prefix")

        _flat = norm_iv(plain(_doc))
        _self_ia = re.findall(
            r"(?:Sections?|Tables?|Figures?)\s+S\d+(?:\.\d+)?"
            r"(?:\s*(?:,|and|to|through|&ndash;|-)\s*S?\d+(?:\.\d+)?)*"
            r"(?:'s)?\s+of\s+the\s+Internet\s+Appendix", _flat)
        checks += 1
        if _self_ia:
            fails.append(
                f"the Internet Appendix cites itself as another document "
                f"{len(_self_ia)} times, e.g. '{_self_ia[0]}'; inside the "
                f"appendix that qualifier is wrong and should be dropped")
        checks += 1
        if "rather than here" in _flat:
            fails.append("the Internet Appendix says 'rather than here' about a "
                         "section it contains; the phrase was written in the "
                         "manuscript and inverts when the paragraph is relocated")

    # The loop above rebound _src per document; everything below is about the
    # manuscript, so put it back rather than inheriting the appendix.
    _src = read_text(path)

    # --- every equation the text cites must be defined in this document ---
    # Section 3.5 read "skill at delay d under any one of (6), (7) or (8)"
    # while the manuscript defined 1, 2, 3, 4, 5 and 9: relocating a
    # subsection to the Internet Appendix took three numbered equations with
    # it and nothing followed the references.  Every figure check passed,
    # because a figure check reads numbers and not cross-references.
    _eqdef = {int(n) for n in re.findall(r'<span class="n">\((\d+)\)</span>', _src)}
    _eqref = set()
    for _m in re.finditer(r"\((\d+)\)(?:,\s*\((\d+)\))*(?:\s*(?:or|and)\s*\((\d+)\))?",
                          re.sub(r"<[^>]+>", " ", _src)):
        for _g in _m.groups():
            if _g:
                _eqref.add(int(_g))
    # only numbers that are plainly equation citations, not years or footnotes
    _eqref = {n for n in _eqref if n <= max(_eqdef | {0}) + 6 and n > 0}
    checks += 1
    _lost = sorted(n for n in _eqref if n not in _eqdef)
    if _lost:
        fails.append(
            "the text cites equation(s) " + ", ".join(f"({n})" for n in _lost) +
            " but this document defines only " +
            ", ".join(f"({n})" for n in sorted(_eqdef)) +
            "; a relocated subsection probably took them with it")
    # and the numbering itself may not skip
    checks += 1
    if _eqdef and sorted(_eqdef) != list(range(1, max(_eqdef) + 1)):
        _gap = [n for n in range(1, max(_eqdef) + 1) if n not in _eqdef]
        fails.append("the equation numbering skips " +
                     ", ".join(f"({n})" for n in _gap) +
                     "; a reader counts from one and finds a hole")

    # --- the roadmap may not give one section two different jobs ----------
    # "Sections 4 to 7 measure what breadth recovers ... Section 7 carries the
    # measurement to house prices" passed every check in this file: both 4 and
    # 7 exist, so nothing dangled, and the sentence is only wrong when you read
    # it.  It came from build_variants renumbering the first number of a range
    # and not the second.  The general form of that mistake is a roadmap that
    # claims the same section twice, which is mechanical and so is checkable:
    # expand the ranges, count the numbers, and fail on a repeat.
    _road = re.search(r"The paper proceeds as follows\.(.{0,1400}?)(?:</p>|$)",
                      _src, re.S)
    if _road:
        _rtxt = re.sub(r"<[^>]+>", " ", _road.group(1))
        _claimed = []
        for _m in re.finditer(r"Sections?\s+((?:S?\d+)"
                              r"(?:\s*(?:,|and|&amp;|to|through)\s*S?\d+)*)",
                              _rtxt):
            _g = _m.group(1)
            _rng = re.match(r"(\d+)\s+(?:to|through)\s+(\d+)$", _g.strip())
            if _rng:
                _a, _b = int(_rng.group(1)), int(_rng.group(2))
                if 0 < _b - _a < 20:
                    _claimed += [str(i) for i in range(_a, _b + 1)]
                    continue
            _claimed += [n for n in re.findall(r"(?<!S)\b(\d+)\b", _g)]
        checks += 1
        _dupe = sorted({n for n in _claimed if _claimed.count(n) > 1}, key=int)
        if _dupe:
            fails.append(
                "the roadmap gives " +
                ", ".join(f"Section {n}" for n in _dupe) +
                " more than one job; a range was probably renumbered at one end "
                "only, so the introduction contradicts itself")
    # These four sections must still EXIST and still carry their subject.  The
    # check used to pin each to a number, which is right for one manuscript and
    # wrong for a journal variant that renumbers after relocating a section to
    # the appendix; the loop above already guarantees that every in-text
    # reference resolves to whatever numbering the document actually uses, so
    # what is left to protect is that a section has not been deleted or
    # retitled out of existence.
    _titles = " | ".join(m.group(1) for m in
                         re.finditer(r"<h2[^>]*>\d+\.\s*([^<]*)", _src))
    # "The illiquid asset" was renamed "External validity: appraisal-like
    # markets" on a referee's point that the old title named the object rather
    # than the job.  The check is on the SUBJECT, not on the wording, which is
    # what it was always meant to protect: a section that still carries the
    # house-price evidence under either title passes, and a section deleted
    # under either title fails.
    for _kw in ("Data", "Methodology", "appraisal-like markets", "Limitations"):
        checks += 1
        if _kw not in _titles and _kw not in txt:
            fails.append(f"no section is titled '{_kw}' any more, in the manuscript "
                         "or the appendix it was moved to")
    # The cover letter is not part of the manuscript, so nothing else here reads
    # it - and it spent a week telling an editor that "every interval in the
    # paper is obtained by inverting a t-test after Fieller", which the paper
    # had already corrected to a block bootstrap with Fieller as a check.  A
    # covering letter that misstates the paper's own method is worse than a
    # stale figure, because it is the first thing read and the last thing
    # anyone re-reads.  Checked when it is present.
    _cl = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "mnt", "user-data", "outputs",
                       "JFEc-cover-letter.md")
    _cl = _cl if os.path.isfile(_cl) else "/mnt/user-data/outputs/JFEc-cover-letter.md"
    if os.path.isfile(_cl):
        _ct = " ".join(read_text(_cl).split())
        checks += 1
        if "every interval in the paper is obtained by inverting a t-test" in _ct:
            fails.append("the cover letter says every interval comes from inverting a "
                         "t-test after Fieller; the paper's intervals are block "
                         "bootstraps and Fieller is a check on them")
        checks += 1
        if "moving-block bootstrap of the whole ratio" not in _ct:
            fails.append("the cover letter no longer describes the paper's intervals "
                         "as block bootstraps of the whole ratio")
        # the script count it quotes must be the number of labs that exist
        _n_labs = len(glob.glob(os.path.join(HERE, "expected_output", "lab*.txt")))
        checks += 1
        if f"all {_n_labs} analysis scripts" not in _ct:
            fails.append(f"the cover letter does not say 'all {_n_labs} analysis "
                         f"scripts', but {_n_labs} lab outputs ship")

    # Every TABLE and FIGURE reference in either document must resolve to a
    # caption that exists.  The table half of this existed; the figure half did
    # not, and renumbering the paper's figures after Sections 7 and 8 moved out
    # left "Figure 4" pointing at nothing from inside the Internet Appendix.
    # A dangling figure reference survives every figure check in this file,
    # because those compare numbers and this is a pointer.
    _pa, _ap = read_text(path), (read_text(_supp) if os.path.isfile(_supp) else "")
    _cap = lambda _s, _p: {m.group(1) for m in re.finditer(_p, _s)}
    for _lbl, _pat, _have in (
            ("Table", r"\bTable (\d+)\b", _cap(_pa, r"<b>Table (\d+)\.</b>")),
            ("Figure", r"\bFigure (\d+)\b", _cap(_pa, r"<b>Figure (\d+)\.</b>")),
            ("Table S", r"\bTable (S\d+)\b", _cap(_ap, r"<b>Table (S\d+)\.</b>")),
            ("Figure S", r"\bFigure (S\d+)\b", _cap(_ap, r"<b>Figure (S\d+)\.</b>"))):
        checks += 1
        _bad = sorted(set(re.findall(_pat, _pa + _ap)) - _have,
                      key=lambda x: int(x.lstrip("S")))
        if _bad:
            fails.append(f"these {_lbl} references resolve to no caption in either "
                         f"document: {', '.join(_lbl.rstrip(' S') + ' ' + b for b in _bad)}")

    # Section 4.3's outage framing and Section S36, against lab60.  The audit
    # that produced this section accepted the training cut and objected to the
    # NAME: delta freezes the whole domestic information system, not one input,
    # so the paper had been selling an outage as a stale mark.  Both halves are
    # checked - the figures that price the difference, and the fact that the
    # paper still calls the experiment what it is.
    _l60 = lab("lab60_outage_decomposition")
    _vs = re.search(r"the estimation vintage accounts for ([\d.]+)% to ([\d.]+)% "
                    r"of what the delay costs", _l60)
    _rd = re.search(r"largest difference between the two rates: ([\d.]+)%", _l60)
    checks += 1
    if not (_vs and _rd):
        fails.append("lab60 no longer prices the outage split that Section 4.3 and "
                     "Section S36 quote")
    else:
        want("4.3 vintage share",
             f"accounts for {_vs.group(1)}% of the damage at a week and "
             f"{_vs.group(2)}% at eleven weeks")
        want("4.3 rate difference",
             f"is {_rd.group(1)} points higher at most")
        want("S36 vintage share",
             f"{_vs.group(1)}% of the delay's damage at one week and "
             f"{_vs.group(2)}% at eleven")
        want("S36 rate difference", f"differ by at most {_rd.group(1)} points")
        # the direction matters: the infeasible arm must be the HIGHER one, or
        # the paper's claim to be conservative is backwards
        checks += 1
        # lab() collapses whitespace, so an anchored per-line pattern matches
        # nothing here and the check silently passes.  It did, once.
        _dirs = re.findall(r"\s\d+ ([\d.]+)% ([\d.]+)% ([-+][\d.]+)%", _l60)
        checks += 1
        if len(_dirs) < 4:
            fails.append(f"lab60: parsed {len(_dirs)} rate comparisons, expected 4")
        if _dirs and not all(float(x[2]) <= 0 for x in _dirs):
            fails.append("lab60: the feature-only rate is no longer at least as high "
                         "as the outage rate at every delay, but Section S36 says the "
                         "paper's design is the conservative one")
    want("4.3 outage named", "domestic-data outage of &delta; days")
    want("S36 infeasible labelled",
         "It is infeasible under this paper's own information set")
    forbid("4.3 stale-mark-only framing",
           "delta does not only make one number old. It freezes the whole")

    # Section 4.4's variance-scale benchmark.  This paper asserted, for one
    # revision, that the smeared constant log benchmark "collapses" to the
    # trailing mean of the variance.  It does not: the target is normalised, so
    # it collapses to the trailing mean of variance over its own median, and an
    # audit caught it.  The two figures that show how different are read out of
    # lab10 part 5, the false claim is forbidden, and the direction of the
    # median error the same audit found is required to be disclosed.
    _l10 = lab("lab10_loss_scale")
    _bd = re.search(r"mean natural variance: ([\d.]+%); correlation between them: "
                    r"([\d.]+)", _l10)
    checks += 1
    if not _bd:
        fails.append("lab10 part 5 no longer measures the gap between the "
                     "reconstructed benchmark and the trailing mean of natural "
                     "variance, which Section 4.4 quotes")
    else:
        want("4.4 benchmark gap",
             f"differ by {_bd.group(1).rstrip('%')}% of the mean level and correlate "
             f"{_bd.group(2)}")
    forbid("4.4 superseded collapse claim",
           "the smeared forecast is the window mean of exp(y u ) itself: the trailing "
           "mean of the variance")
    forbid("4.4 superseded practitioner claim",
           "the trailing mean of the variance, which is also what a practitioner would "
           "quote")
    want("4.4 benchmark is normalised",
         "the trailing mean of variance over its own median")
    want("4.4 median direction disclosed",
         "giving it the median at t + h while the model only holds the median at t")
    # the rate must be invariant to the choice, which is what makes the
    # correction a correction to reporting rather than to conclusions
    _inv = re.search(r"largest difference the choice makes to R\(delta\): ([\d.]+)%",
                     _l10)
    checks += 1
    if not _inv:
        fails.append("lab10 part 5 no longer reports what the benchmark choice does "
                     "to R(delta)")
    elif float(_inv.group(1)) > 0.05:
        fails.append(f"the variance-scale benchmark choice now moves R(delta) by "
                     f"{_inv.group(1)}%, but Section 4.4 says the rate is identical "
                     f"under either")

    # Moving a section between documents creates two faults that no figure check
    # can see, and the move of Sections 7 and 8 created four of the first and
    # none of the second only because they were looked for.
    #
    #   A SELF-REFERENCE.  Text that said "Section S28 shows it is a bill" was
    #   true in the paper and absurd once it WAS Section S28.  It reads as a
    #   pointer to working that is in fact the paragraph above it.
    #
    #   A DANGLING REFERENCE.  A pointer to a section that no longer exists
    #   under that number, which renumbering produces silently.
    _both = (read_text(_supp) if os.path.isfile(_supp) else "") + read_text(path)
    _sheads = {m.group(1) for m in
               re.finditer(r"<h[23][^>]*>(S\d+)[.\s]",
                           read_text(_supp) if os.path.isfile(_supp) else "")}
    # The reference forms this has to see.  "Section (S\d+)" alone - which is
    # all this check used to match - reads SINGULAR references only: the
    # pattern needs a space after "Section", so "Sections S13, S37" does not
    # match at any member, not even the first, and both documents contain that
    # form.  Every member of a list, and every number spanned by a "S13 to S15"
    # range, is a pointer that can dangle, so all of them are collected.  The
    # gap was tolerable while both documents were hand-written and stopped
    # being tolerable when the journal variants began relocating manuscript
    # sections into the appendix and rewriting references to match: a rewrite
    # that misses a list member is the exact fault this is here to catch.
    def _srefs(text):
        out, pat = set(), (r"Sections?\s+((?:S\d+)"
                           r"(?:\s*(?:,|and|to|&amp;|&)\s*S?\d+)*)")
        for g in re.findall(pat, text):
            nums = re.findall(r"S?(\d+)", g)
            out |= {"S" + n for n in nums}
            for m in re.finditer(r"S(\d+)\s+to\s+S?(\d+)", g):
                a, b = int(m.group(1)), int(m.group(2))
                if 0 < b - a < 40:
                    out |= {f"S{i}" for i in range(a, b + 1)}
        return out
    checks += 1
    _dangle = sorted(_srefs(_both) - _sheads,
                     key=lambda s: int(s[1:]))
    if _dangle:
        fails.append(f"these Internet Appendix sections are referenced but do not "
                     f"exist: {', '.join(_dangle)}")
    if os.path.isfile(_supp):
        _as = read_text(_supp)
        _spans = [(m.start(), m.group(1)) for m in
                  re.finditer(r"<h[23][^>]*>(S\d+)[.\s]", _as)] + [(len(_as), None)]
        _self = []
        for _i, (_st, _nm) in enumerate(_spans[:-1]):
            _body = _as[_st:_spans[_i + 1][0]]
            _self += [_nm for _m in re.finditer(r"Section (S\d+)\b", _body)
                      if _m.group(1) == _nm]
        checks += 1
        if _self:
            fails.append(f"these Internet Appendix sections point at themselves for "
                         f"their own working: {', '.join(sorted(set(_self)))}")

    # Sections 7 and 8 were cut to pointers for JFEc's page limit, and their
    # bodies moved into the Internet Appendix rather than being deleted.  The
    # check is that the working is still SOMEWHERE: a pointer section whose
    # target has gone is worse than either half on its own.
    _app = read_text(_supp) if os.path.isfile(_supp) else ""
    for _sec, _kw in (("S28", "The implied-volatility race in full"),
                      ("S31", "The house price measurement in full")):
        checks += 1
        if _kw not in _app:
            fails.append(f"Section {_sec} ('{_kw}') is gone from the Internet "
                         f"Appendix, but the paper points at it for the working")
    for _lbl, _n in (("7 points at its working",
                      "Section S28 of the Internet Appendix gives the race in full"),
                     ("8 points at its working",
                      "Section S31 of the Internet Appendix runs the whole design")):
        want(_lbl, _n)
    # and the moved material must still be there, not merely the heading
    for _lbl, _n in (("S28 carries the race table", "0.5547"),
                     ("S31 carries the metro table", "67.8% [45, 92]")):
        checks += 1
        if norm_iv(_n) not in norm_iv(_app):
            fails.append(f"{_lbl}: '{_n}' is not in the Internet Appendix, so the "
                         f"working the paper points at did not survive the move")

    # the paper must keep the conventional shape it was restructured into
    # Cut to JFEc's 40-double-spaced-page limit, the paper lost two whole
    # sections to the Internet Appendix and everything after them moved up by
    # one.  This list is the shape it now has; it is here so that a later edit
    # cannot quietly drop a section without the verifier saying which one.
    # The list is by TITLE, not by number.  Numbering is a per-manuscript
    # choice once a journal variant relocates a section to the appendix, but
    # the subject matter is not: a section may move, it may not disappear.  A
    # title found in the appendix counts, because that is where a relocated
    # section legitimately lives.
    for _want in ("Introduction", "Related work", "Data", "Methodology",
                  "Robustness and scope", "Limitations",
                  "Conclusion and further work"):
        checks += 1
        if _want not in _src and _want not in txt:
            fails.append(f"the section '{_want}' is gone from the manuscript and "
                         "from the appendix; a length cut may relocate a section "
                         "but may not drop one")

    # --- a style guard the author set and every new section can break ------
    # Both papers are written without em dashes.  That is a deliberate choice,
    # not an accident, and new prose reintroduced four of them once already.
    # The Internet Appendix is held to the same rule as the paper, because a
    # house style that stops at the document boundary is not a house style.  An
    # em dash standing alone in a table cell is a not-applicable marker rather
    # than prose punctuation, so those are excluded rather than counted.
    _emdash_docs = (("main paper", path),)
    _ia_p = os.path.join(os.path.dirname(os.path.abspath(path)),
                         "stale-mark-internet-appendix.html")
    if os.path.isfile(_ia_p):
        _emdash_docs += (("Internet Appendix", _ia_p),)
    if companion and os.path.isfile(companion):
        _emdash_docs += (("companion", companion),)
    for _label, _p in _emdash_docs:
        _raw = read_text(_p)
        checks += 1
        _prose = re.sub(r"<td>\s*(&mdash;|\u2014)\s*</td>", "<td></td>", _raw)
        _n = _prose.count("&mdash;") + _prose.count("\u2014")
        if _n:
            fails.append(f"{_label}: {_n} em dash(es) reintroduced; this paper is "
                         "written without them")

    # Drafting language, in the same spirit as the em-dash rule.  Two kinds are
    # forbidden outright.  "Moved here ... for length" is a note to ourselves
    # about where a paragraph used to live, which tells a reader nothing and
    # tells a referee that the document was assembled in a hurry.  And framing
    # an objection as a referee's discloses a review history the submitted
    # version has no business carrying: the objections are good, so they are
    # stated as objections.  The one permitted use is the generic reader in the
    # opening paragraph, where "a referee who does not" describes who the
    # Internet Appendix is for rather than reporting what one said.
    for _label, _p in _emdash_docs:
        _raw = read_text(_p)
        checks += 1
        # "for length" can sit behind a section number, so the gap may contain
        # periods; it may not contain a tag.
        _moved = len(re.findall(r"[Mm]oved here[^<>]{0,80}for length", _raw))
        if _moved:
            fails.append(f"{_label}: {_moved} 'moved here ... for length' note(s); "
                         "these are drafting notes, not content")
        checks += 1
        _ref = [m.group(0) for m in
                re.finditer(r"[^.<>]{0,40}referee[^.<>]{0,40}", _raw)
                if "a referee who does not can check" not in m.group(0)]
        if _ref:
            fails.append(f"{_label}: the submitted text still attributes "
                         f"{len(_ref)} objection(s) to a referee: "
                         + "; ".join(x.strip()[:60] for x in _ref[:3]))

    # --- the VIX provenance audit, against lab43 --------------------------
    l43 = " ".join(lab("lab43_official_vix").split())
    for label, inlab, inpaper in (
            ("shared rows", "overlapping dates 6747", "6,747 dates"),
            ("exact share", "identical to the cent 6741 (99.91%)", "99.91% of rows"),
            ("mean difference", "mean absolute difference 0.000222",
             "0.000222 index points"),
            ("rows outside tolerance", "rows outside the 0.5% rule: 2",
             "two rows fall outside"),
            ("fabricated rows",
             "in the retail export but not in Cboe's file, over the retail span: 15",
             "fifteen rows on dates United States equities were shut"),
            ("none reach the panel", "and 0 survive into the panel",
             "All fifteen fall outside the panel"),
            ("substitution is inert", "largest difference in any cell: 0.0000",
             "by as much as a thousandth")):
        checks += 1
        if inlab not in l43:
            fails.append(f"lab43 no longer prints the {label} ('{inlab}')")
        want(f"lab43 {label}", inpaper)
    # The two shortened sessions are named in the paper; they must be the two the
    # lab actually flags, not two dates that once were.
    for _dt in ("2008-11-28", "2012-12-24"):
        checks += 1
        if _dt not in l43:
            fails.append(f"lab43 no longer flags {_dt} as outside the seam rule")
    want("11 shortened session 1", "28 November 2008")
    want("11 shortened session 2", "24 December 2012")
    want("11 fabricated date", "11 September 2001")
    checks += 1
    if "2001-09-11" not in l43:
        fails.append("lab43 no longer lists 2001-09-11 among the fabricated rows")

    # --- the estimation-cost reading of non-linearity, against lab09 ------
    l09 = lab("lab09_nonlinearity")
    # "FACTOR net of cost at delta=0", "+0.0115" used to be checked here.  It
    # was removed once it became clear that the paper never states that figure:
    # the needle was satisfied by the upper endpoint of an unrelated interval in
    # Section 6, and changing that interval revealed it.  A check that passes on
    # a coincidence is worse than no check, and the honest repair is to drop it
    # rather than to add a sentence to the paper so the needle has something to
    # find.
    # Two per-cell needles on lab09's SQ column used to sit here as well, and
    # they went the same way and for the same reason.  Neither document quotes
    # lab09's cells any more - the non-linearity argument is carried by lab15
    # and Section S22 - so the paper side of those pairs had nothing to find.
    # One of them nevertheless passed for months, because "-0.0131" is also the
    # AR(1) surrogate cost in Section 5; changing the SQ column revealed it.
    # What lab09 still supports is its verdict, so that is what is checked.
    checks += 1
    if "SQ: beats LIN at 0 of 6 delays" not in l09:
        fails.append("lab09: squaring every input now beats the linear map "
                     "somewhere, so the non-linearity question is no longer "
                     "settled the way the package says it is")
    checks += 1
    if "FACTOR: beats LIN at 0 of 6 delays" not in l09:
        fails.append("lab09: the global-factor interaction now beats the linear "
                     "map somewhere, so the non-linearity question is no longer "
                     "settled the way the package says it is")

    # --- Section 3's stated settings, against lab12 -----------------------
    # The settings paragraph reads like boilerplate, which is exactly why it
    # drifts: nothing in the paper changes when a window length in prose stops
    # matching the window the scripts use.  lab12 prints them; they are checked.
    # The bandwidth the settings paragraph states, read out of the source that
    # defines it so the prose cannot drift from the constant.
    _hm = re.search(r"^HAC_LAG = BLOCK", read_text(os.path.join(
        HERE, "labs", "lab05_robustness.py")), re.M)
    checks += 1
    if not _hm:
        fails.append("lab05 no longer ties HAC_LAG to BLOCK, which Section 5 "
                     "and Section S13 both state")
    _hb = re.search(r"HAC bandwidth\s+(\d+) lags, measured", lab("lab12_appendix"))
    _HACLAG = _hb.group(1) if _hb else "?"
    l12 = " ".join(lab("lab12_appendix").split())
    for label, inlab, inpaper in (
            ("training window", "rolling training window 1250 days",
             "fitted on the first 1,250 days"),
            ("validation tail", "validation tail 250 days",
             "scored on the last 250"),
            ("refit frequency", "refit frequency every 21 days",
             "refitted every 21 days"),
            ("median window", "trailing median window 252 days",
             "trailing 252-day median"),
            ("HAC bandwidth", f"{_HACLAG} lags, measured",
             f"run to L = {_HACLAG} lags")):
        checks += 1
        if inlab not in l12:
            fails.append(f"lab12 no longer prints the {label} ('{inlab}')")
        want(f"lab12 {label}", inpaper)

    # --- the origin-dated robustness check, against lab13 -----------------
    l13 = " ".join(lab("lab13_origin_median").split())
    checks += 1
    if "target-dated (paper): R(55) = 72%" not in l13 \
            or "origin-dated: R(55) = 71%" not in l13:
        fails.append("lab13: the origin-dated comparison is no longer 72% "
                     "against 71%, which Section 3 quotes")
    # Both intervals here were literals, so widening the bootstrap block moved
    # them in lab13 and left the check asserting the old pair.  Derived now.
    # Scoped to lab13's part C2, the FIXED-b table, and not to part C above
    # it.  Both parts print a delta=55 row of the same shape, so an unscoped
    # search takes the first one - the percentile bootstrap - which is the
    # procedure lab63 measures at about 92% coverage.  The paper quotes the
    # covering one, so the check has to read the covering one.
    _l13txt = read_text(os.path.join(EXP, "lab13_origin_median.txt"))
    _c2 = _l13txt.find("C2.")
    checks += 1
    if _c2 < 0:
        fails.append("lab13 no longer prints its fixed-b table, so the headline "
                     "rate the paper quotes rests on nothing")
    _m13 = re.search(r"\s*55\s+(\d+)% \[\s*(\d+)%,\s*(\d+)%\]\s+(\d+)% \["
                     r"\s*(\d+)%,\s*(\d+)%\]",
                     _l13txt[_c2:] if _c2 >= 0 else _l13txt)
    checks += 1
    if not _m13:
        fails.append("lab13 no longer prints the delta=55 rate under both targets")
    else:
        _tp, _tl, _th, _op, _ol, _oh = _m13.groups()
        want("lab13 origin-dated rate",
             f"the substitution rate is {_op}% [{_ol}, {_oh}] against "
             f"{_tp}% [{_tl}, {_th}] at eleven weeks")
    want("4.2 quotes the implementable rate in headlines",
         "quotes the origin-dated rate wherever it states a headline")
    # This check sat directly under the comment above explaining that literals
    # here go stale when the block moves, and was itself a literal carrying the
    # pre-correction [63, 79].  The abstract and the introduction agreed with
    # the check and disagreed with Table 2 and with Section 4.2's own summary
    # row.  Derived now, and every site that says "returns <rate>%" is required
    # to carry the same interval, because two of the three sites were wrong and
    # a single-site check would have passed on the third.
    if _m13:
        _op, _ol, _oh = _m13.group(4), _m13.group(5), _m13.group(6)
        # This used to require the abstract to LEAD with the unconditional
        # implementable rate.  Section 4 of the paper says that rate is "the
        # average of a result and a non-result", so leading with it was the
        # paper contradicting itself in its first hundred words, and a referee
        # said so.  The abstract now leads with the conditional rate.  The
        # check is not dropped, it is made stricter: both rates must appear in
        # the abstract, each with its own interval, and the unconditional one
        # must be labelled as such - which is harder to satisfy by accident
        # than the single phrase it replaces.
        _abs_m = re.search(r'<div class="abs">.*?</div>', read_text(path), re.S)
        _abs_t = " ".join(re.sub(r"<[^>]+>", " ",
                                 _abs_m.group(0) if _abs_m else "").split())
        for _lbl, _need in (
                # derived from lab39, not transcribed.  This was the third
                # hardcoded copy of the same interval; together they pinned an
                # abstract figure to a constant no script produced.
                # the Fieller set, matching the rule Section 4.5 states: the
                # rate this paper quotes inverts the HAC t-test.  Reading the
                # bootstrap here was how the abstract and Section 9 came to
                # disagree with the appendix about the same cell.
                ("abstract carries the conditional rate with its interval",
                 (lambda _s: (f"{_s[55]['stressed'][0]}% "
                              f"[{int(_s[55]['stressed'][1])}, "
                              f"{int(_s[55]['stressed'][2])}]") if 55 in _s
                  else "81%")(tercile_fieller())),
                ("abstract carries the unconditional rate with its interval",
                 f"{_op}%"),
                ("abstract labels the unconditional rate as unconditional",
                 "unconditional")):
            checks += 1
            if _need not in _abs_t:
                fails.append(f"{_lbl}: the abstract does not contain {_need!r}")
        want("introduction states the implementable rate",
             f"returns {_op}% [{_ol}, {_oh}] of what the delay took")
        want("4.2 summary row states the implementable rate",
             f"{_op}% [{_ol}, {_oh}] at eleven weeks")
        only_iv_re("implementable rate quoted consistently",
                   r"returns " + re.escape(_op) + "%", _ol, _oh)

    # --- Table 5, against lab15 -------------------------------------------
    # Parsed from the WIDE block of section A, which is the grid the paper's
    # table reports.  Both the dR2 and the GW statistic of every cell are
    # checked, because a table this wide is exactly where a hand-copied digit
    # goes unnoticed.
    l15 = read_text(os.path.join(EXP, "lab15_nonlinear_given_iv.txt"))
    wide = l15.split("ridge grid: WIDE")[-1].split("=" * 40)[0]
    n15 = 0
    for ln in wide.splitlines():
        m = re.match(r"\s*(\d+)\s+([\d.]+)((?:\s*[-+][\d.]+\s*[-+]?[\d.]+\s+[\d.]+){4})\s*$", ln)
        if not m:
            continue
        n15 += 1
        d = int(m.group(1))
        want(f"lab15 d={d} R2 IV", m.group(2))
        cells = re.findall(r"([-+][\d.]+)\s*([-+]?[\d.]+)\s+[\d.]+", m.group(3))
        for arm, (dr, z) in zip(("F", "FAC", "RFF20", "RFF60"), cells):
            want(f"lab15 d={d} {arm} dR2", dr)
            want(f"lab15 d={d} {arm} GW", z)
    checks += 1
    if n15 != 6:
        fails.append(f"lab15: parsed {n15} WIDE rows from the lab output, expected 6")

    # The prose quotes the RANGE of the linear GW statistics past the first
    # delay.  That range is derivable, so derive it rather than trusting the
    # draft: an earlier one quoted -0.70, which is the FACTOR column's weakest
    # statistic, not the linear column's.
    lin_z = []
    for ln in wide.splitlines():
        m = re.match(r"\s*(\d+)\s+[\d.]+\s+[-+][\d.]+\s+([-+][\d.]+)\s+[\d.]+", ln)
        if m and int(m.group(1)) > 0:
            lin_z.append(float(m.group(2)))
    if len(lin_z) == 5:
        want("lab15 linear GW range",
             f"{max(lin_z):.2f} and {min(lin_z):.2f}".replace("-", "-"))
    else:
        checks += 1
        fails.append(f"lab15: found {len(lin_z)} linear GW statistics past delay 0, expected 5")

    # --- Table 4, against lab16 -------------------------------------------
    # Section 6.1 discloses that our own companion paper's result bears on our
    # own Clark-West column.  Every cell of that disclosure is checked, because
    # a caveat with a wrong number in it is worse than no caveat.
    l16 = read_text(os.path.join(EXP, "lab16_clark_west_shrinkage.txt"))

    def part(letter):
        """Body of one lettered section: past its own closing rule, up to the next."""
        after = l16.split(f"\n{letter}.  ", 1)[1]
        body = after.split("=" * 40, 2)          # [header text, '', body...]
        return body[2].split("=" * 40)[0] if len(body) > 2 else ""

    pre, diag, rem, prim = (part(c) for c in "ABCD")
    n16 = 0
    for ln in pre.splitlines():
        m = re.match(r"\s*(\d+)\s+[\d.]+\s+[\d.]+\s+(\d+)%\s+\d+%\s*$", ln)
        if m:
            n16 += 1
            want(f"lab16 d={m.group(1)} differing refits", m.group(2) + "%")
    for ln in diag.splitlines():
        m = re.match(r"\s*(\d+)\s+[-+][\d.]+\s+[\d.]+\s+([\d.]+)\s+[-+]?[\d.]+\s+[-+]?[\d.]+\s*$", ln)
        if m:
            want(f"lab16 d={m.group(1)} adjustment ratio", m.group(2))
    for ln in rem.splitlines():
        m = re.match(r"\s*(\d+)\s+([-+]?[\d.]+)\s+([-+]?[\d.]+)\s+[-+][\d.]+\s+[\d.]+\s+[\d.]+\s*$", ln)
        if m:
            want(f"lab16 d={m.group(1)} CW separate", m.group(2))
            want(f"lab16 d={m.group(1)} CW common", m.group(3))
    for ln in prim.splitlines():
        m = re.match(r"\s*(\d+)\s+([-+]?[\d.]+)\s+([-+]?[\d.]+)\s+[-+][\d.]+\s+[\d.]+\s+[\d.]+\s*$", ln)
        if m:
            want(f"lab16 d={m.group(1)} GW common", m.group(3))
    checks += 1
    if n16 != 6:
        fails.append(f"lab16: parsed {n16} delay rows from part A, expected 6")
    # The caveat must stand on this repository's own evidence.  It used to lean on
    # an unpublished companion paper; that paper has no public replication, so the
    # section was rewritten to rest on lab16 alone and the citation must not return.
    want("lab16 disclosure", "S3. A caveat on the Clark-West column")
    want("lab16 remedy is decisive",
         "imposing a common penalty moves no Clark-West statistic by more than 0.05")
    want("lab16 ratio not sold as a threshold",
         "We offer that ratio as a description, not a threshold")
    # The remedy equalises the shrinkage and so closes the differential-shrinkage
    # channel.  It does NOT make the nesting exact: a ridge fit on the wider design
    # is not the narrower fit with zeros appended, at any common penalty.  An
    # earlier draft said the remedy "restores the nesting", which claims the whole
    # of what it buys only part of, so both the accurate sentence and a ban on the
    # overclaim are pinned here.
    want("lab16 remedy is partial",
         "A common penalty reduces the departure from exact nesting rather than "
         "completely removing it")
    want("lab16 residual named",
         "the residual non-nesting of two ridge fits on designs of different width")
    for phrase in ("the Clark-West column is invalid",
                   "we withdraw the Clark-West",
                   "equalising the shrinkage restores the nesting",
                   "restores the nesting at the level the correction assumes",
                   "the mechanism is present in principle but undetectable"):
        forbid("lab16 overclaim", phrase)
    # Bootstrap intervals CAN be inverted into tests; the reason this paper runs a
    # separate one is the nesting, not the form of an interval.  The old blanket
    # claim must not come back, in either document.
    forbid("S2 interval-vs-test overclaim",
           "they do not test a hypothesis")
    want("S2 interval inversion conceded",
         "any interval can be inverted into a test of the values it excludes")
    for phrase in ("companion paper", "by the present author",
                   "Validation-tuned shrinkage",
                   "validation-tuned shrinkage invalidates",
                   "rejection reaches 0.51"):
        forbid("withdrawn companion citation", phrase)

    # --- Tables 7 and 8, against lab17 and lab18 --------------------------
    l17 = read_text(os.path.join(EXP, "lab17_horizons.txt"))
    tail = l17.split("C.  IS THE HEADLINE")[-1]
    # Anchored as whole rows, not as bare percentages.  A first version checked
    # "70%" on its own and passed while the lab said something else, because the
    # paper happens to contain "71%" elsewhere (the origin-dated median run).  A
    # check that any nearby number can satisfy is not a check.
    n17 = 0
    # The own-only R2 at delta = 0 for each horizon, read out of lab17's own
    # per-horizon blocks rather than transcribed here.  It was a literal until
    # a benchmark correction moved all four and this file went on agreeing
    # with its own copy.
    OWN = {h: f"{float(v):.3f}" for h, v in
           re.findall(r"--- h = (\d+) \([^)]*\) -+\s*\n[^\n]*\n\s*0\s+(-?[\d.]+)", l17)}
    # lab17's part C gained a fixed-b column, and Table S18 carries THAT set
    # rather than the bootstrap one beside it.  The row is therefore
    # delta, point, bootstrap interval, QLIKE point, fixed-b interval, and the
    # table quotes the last of those.  The pattern that stood here expected the
    # pre-conversion four-column row and matched nothing, which reported the
    # lab as broken rather than the pattern.
    for ln in tail.splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)%\s*\[\s*(-?\d+)%,\s*(-?\d+)%\]"
                     r"\s+(\d+)%\s*\[\s*(-?\d+)%,\s*(-?\d+)%\]\s*$", ln)
        if m:
            n17 += 1
            h, pt, _blo, _bhi, ql, lo, hi = m.groups()
            unit = "day" if h == "1" else "days"
            want(f"lab17 h={h} row",
                 f"{h} {unit} {OWN[h]} {pt}% [{lo}%, {hi}%] {ql}%")
    checks += 1
    if n17 != 4:
        fails.append(f"lab17: parsed {n17} horizon rows, expected 4")
    # the overlap at h = 1 must be disclosed, not buried
    # Read out of lab17 rather than fixed here, for the same reason OWN is.
    want("lab17 h=1 own R2", OWN["1"])
    for c in ("0.953", "0.692", "0.576", "0.417"):
        want("lab17 target autocorrelation", c)

    l18 = read_text(os.path.join(EXP, "lab18_economic_reading.txt"))
    rows18 = {}
    for ln in l18.splitlines():
        m = re.match(r"\s*(\d+)\s+(domestic|cross)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)%", ln)
        if m:
            rows18[(m.group(1), m.group(2))] = m.groups()[2:]
    checks += 1
    if len(rows18) != 8:
        fails.append(f"lab18: parsed {len(rows18)} model rows, expected 8")
    else:
        for key in (("0", "domestic"), ("55", "domestic"), ("55", "cross")):
            med, out2, _ = rows18[key]
            want(f"lab18 d={key[0]} {key[1]} median", med)
            want(f"lab18 d={key[0]} {key[1]} outside 2x", out2 + "%")
        # the understating-tail range is the finding the section turns on
        und = sorted(float(v[2]) for v in rows18.values())
        want("lab18 understating tail range", f"{und[0]:.1f}% and {und[-1]:.1f}%")
    forbid("lab18 overclaim", "trading edge")
    forbid("lab18 overclaim", "Sharpe")

    # --- lab19, which now survives only as the summary in Section 11 -------
    # Section 8.5 and its table were cut; the VSTOXX result is quoted in the
    # limitations instead, so only those figures are checked here.
    l19 = read_text(os.path.join(EXP, "lab19_vstoxx_third_series.txt"))
    for tok, label in (("1,240", "lab19 reduced window"), ("82.7", "lab19 peak VIX full"),
                       ("41.4", "lab19 peak VIX reduced")):
        checks += 1
        if tok.replace(",", "") not in l19.replace(",", ""):
            fails.append(f"{label}: lab19 no longer prints {tok}")
        want(label, tok)
    want("12 VSTOXX verdict",
         "it adds nothing at any delay and is mildly negative at all six")

    # --- Table 3, against lab22 -------------------------------------------
    l22 = read_text(os.path.join(EXP, "lab22_factor_benchmark.txt"))
    n22 = 0
    ROW22 = re.compile(r"\s*(\d+)" + r"\s+([-+]?[\d.]+)" * 5 +
                       r"\s+\w+\s+([-+][\d.]+)\s+([-+]?[\d.]+)\s+[\d.]+\s*$")
    for ln in l22.splitlines():
        m = ROW22.match(ln)
        if m:
            n22 += 1
            d, ownr, mean, pc1, pc2, full, gap, z = m.groups()
            want(f"lab22 d={d} row", f"{ownr} {mean} {pc1} {pc2} {full} {gap} {z}")
    checks += 1
    if n22 != 6:
        fails.append(f"lab22: parsed {n22} rows from part A, expected 6")
    # Narrow on purpose.  Section 9 now WITHDRAWS this claim and quotes it while
    # doing so, so a forbid on the bare phrase fires on the withdrawal itself.
    # What must not return is the assertion form, which cites Section 5 as
    # showing it rather than as having fitted it.
    forbid("superseded factor claim",
           "Section 5 shows that a cheaper representation of the foreign block")

    # --- Section 6.1's three claims about Table 4, all recomputed ---------
    # Every one was wrong against the table printed six lines above it, and the
    # significance claim was wrong in the direction that flatters the paper: it
    # said the gain clears 5% at delta = 55, which is the LEAST significant
    # cell in the table.
    _R22 = re.compile(r"^\s*(\d+)" + r"\s+([-+]?[\d.]+)" * 5 +
                      r"\s+\w+\s+([-+][\d.]+)\s+([-+]?[\d.]+)\s+[\d.]+\s*$")
    _g22, _z22, _m22 = {}, {}, {}
    for _ln in l22.splitlines():
        _m = _R22.match(_ln)
        if _m:
            _d = int(_m.group(1))
            _g22[_d] = float(_m.group(7))
            _z22[_d] = float(_m.group(8))
            _m22[_d] = round(float(_m.group(4)) - float(_m.group(3)), 4)
    checks += 1
    if len(_g22) != 6:
        fails.append(f"lab22: read {len(_g22)} of 6 rows for Section 6.1's claims")
    else:
        _ds22 = sorted(_g22)
        want("6.1's range for the PC1 gain",
             f"at every delay, by {min(_g22.values()):.3f} to "
             f"{max(_g22.values()):.3f} of R&sup2;")
        _steps = sum(1 for a, b in zip(_ds22, _ds22[1:]) if _g22[b] > _g22[a])
        want("6.1 counts the rising steps instead of claiming monotonicity",
             f"larger at {_num_word(_steps)} of the {_num_word(len(_ds22) - 1)} steps")
        checks += 1
        if _steps == len(_ds22) - 1:
            fails.append("the PC1 margin is now monotone in delay, so Section 6.1's "
                         "step count understates it")
        _sig22 = re.search(r"does so at z > 1\.96 at (\d+)(?:: \[([^\]]*)\])?", l22)
        checks += 1
        if not _sig22:
            fails.append("lab22 no longer tallies which delays clear 1.96")
        else:
            _which22 = ([int(x) for x in _sig22.group(2).split(",")]
                        if _sig22.group(2) else [])
            checks += 1
            if _which22 != [0]:
                fails.append(f"Section 6.1 says the gain clears 5% at zero delay "
                             f"alone; lab22 now gives {_which22}")
            else:
                want("6.1 names the only significant delay and its statistic",
                     f"clears conventional significance at &delta; = {_which22[0]} "
                     f"alone, where the Giacomini-White statistic is "
                     f"{_z22[_which22[0]]:.2f}, and at no other delay")
            want("6.1 names the weakest cell",
                 f"to {min(_z22.values()):.2f} at eleven weeks")
            checks += 1
            if min(_z22.values()) != _z22[max(_z22)]:
                fails.append("Section 6.1 calls the eleven-week cell the weakest in "
                             "Table 4; it no longer is")
            checks += 1
            if any(_z22[b] > _z22[a] for a, b in zip(_ds22, _ds22[1:])):
                fails.append(f"Section 6.1 says the Giacomini-White statistic falls "
                             f"monotonically; lab22's column is "
                             f"{[_z22[d] for d in _ds22]}")
            want("6.1 counts the consistent sign",
                 f"a consistent sign at "
                 f"{_num_word(sum(1 for g in _g22.values() if g > 0))} "
                 f"of {_num_word(len(_g22))} delays")
        want("6.1's margin over the equal-weighted mean",
             f"by {min(_m22.values()):.4f} of R&sup2; at zero delay and by at most "
             f"{max(_m22.values()):.3f} at any delay")
        checks += 1
        if _m22[min(_ds22)] != min(_m22.values()):
            fails.append("Section 6.1 names the zero-delay cell as the smallest "
                         "margin over the equal-weighted mean; it no longer is")
    checks += 1
    if "lab07 inferred cost" in l22:
        fails.append("lab22 is printing lab07's cost figures again; they were "
                     "hard-coded into the script and both were wrong")
    # The withdrawn ceiling comparison.  This used to REQUIRE the sentence that
    # announced the withdrawal, which pinned a piece of revision history into
    # the manuscript: a journal variant that states its current position
    # instead - the contemporaneous spanning bound, and what it does and does
    # not bound - then failed a check for being cleaner than the preprint.
    # What has to be guarded is the CLAIM, so the ban is on the claim and on
    # the two figures nothing produces, in whichever document they appear.
    forbid("the withdrawn ceiling comparison is asserted again",
           "or 80% of it, which brackets the 72% the fitted model achieves")
    # Two shapes are correct and a third is not.  The preprint NAMES what it
    # withdrew, so the two unproduced figures appear there inside the sentence
    # that retracts them; a journal variant drops the history and must then not
    # contain them at all.  What is wrong in either is the figures without the
    # retraction, which is the comparison coming back.
    checks += 1
    if "A sentence stood here comparing" in txt:
        want("the withdrawal names what it withdraws",
             "Neither figure is produced by any script in the replication")
    else:
        forbid("the unproduced foreign-only ceiling figure is back",
               "foreign-only R&sup2; of 0.399")
        forbid("the unproduced domestic-history figure is back",
               "0.498 from domestic history")
    want("6 states the bound it is entitled to",
         "an in-sample bound on a different quantity from the rate")

    # --- six figures a tamper audit found unpinned, each read from its lab -
    _l07c = read_text(os.path.join(EXP, "lab07_estimation_cost.txt"))
    _m07 = re.search(r"^\s*0\s+(-[\d.]+)\s+([\d.]+)\s+\[", _l07c, re.M)
    checks += 1
    if not _m07:
        fails.append("lab07 no longer prints the surrogate cost at zero delay")
    else:
        want("6 quotes lab07's cost POINT estimate, not its interval endpoint",
             f"draws this gives -{abs(float(_m07.group(1))):.4f} at &delta; = 0, "
             f"with a Monte Carlo standard error of {_m07.group(2)}")
    _m07d = re.search(r"\b([A-Za-z]+|\d+) draws\.", _l07c)
    checks += 1
    if not _m07d:
        fails.append("lab07 no longer states how many surrogate draws it takes")
    else:
        _d7 = _m07d.group(1).lower()
        _d7 = spell(int(_d7)) if _d7.isdigit() else _d7
        want("6 names lab07's surrogate draw count", f"Averaged over {_d7} draws")
        want("Table 3's caption names the same draw count",
             f"Cost from AR(1) surrogates, {_d7} draws")
        for _wrong in ("one hundred", "a hundred", "ten", "twenty", "fifty"):
            if _wrong == _d7:
                continue
            forbid(f"surrogate draw count given as {_wrong}",
                   f"Averaged over {_wrong} draws")
        # Section S21 compares lab21's twenty-draw column against Table 3's
        _m21d = re.search(r"guard: (\d+) offsets", read_text(
            os.path.join(EXP, "lab21_stronger_inference.txt")))
        checks += 1
        if not _m21d:
            fails.append("lab21 no longer states its offset count")
        else:
            _n21 = int(_m21d.group(1))
            want("S21 names its own draw count",
                 f"The AR(1) column here is {spell(_n21)} draws")
            want("S21 names Table 3's draw count", f"not the {_d7} behind Table 3")
            checks += 1
            if _n21 >= int(_m07d.group(1) if _m07d.group(1).isdigit() else 0):
                fails.append(f"Section S21 explains the spread by Table 3 resting "
                             f"on more draws, which needs lab07's count to exceed "
                             f"{_n21}")
            _c21, _se7 = {}, {}
            for _ln in read_text(os.path.join(
                    EXP, "lab21_stronger_inference.txt")).splitlines():
                _m = re.match(r"^\s*(\d+)\s+[-+][\d.]+\s+(-[\d.]+)\s+(-[\d.]+)"
                              r"\s+([\d.]+)\s+", _ln)
                if _m:
                    _c21[int(_m.group(1))] = (float(_m.group(2)), float(_m.group(4)))
            for _ln in _l07c.splitlines():
                _m = re.match(r"^\s*(\d+)\s+(-[\d.]+)\s+([\d.]+)\s+\[", _ln)
                if _m:
                    _se7[int(_m.group(1))] = (float(_m.group(2)), float(_m.group(3)))
            checks += 1
            if not ({0, 21} <= set(_c21) and {0, 21} <= set(_se7)):
                fails.append("could not read both cost columns at delta = 0 and 21 "
                             "for Section S21's bookkeeping paragraph")
            else:
                want("S21 compares the two tables at zero delay",
                     f"-{abs(_c21[0][0]):.4f} against -{abs(_se7[0][0]):.4f} "
                     f"at &delta; = 0")
                want("S21 compares the two tables at three weeks",
                     f"-{abs(_c21[21][0]):.4f} against -{abs(_se7[21][0]):.4f} "
                     f"at &delta; = 21")
                _sp21 = abs(abs(_c21[21][0]) - abs(_se7[21][0]))
                _sp0 = abs(abs(_c21[0][0]) - abs(_se7[0][0]))
                want("S21 prices the three-week spread against the standard error",
                     f"it is {_sp21:.4f} against Table 3's own Monte Carlo "
                     f"standard error of {_se7[21][1]:.4f}")
                want("S21 prices the zero-delay spread, which is larger",
                     f"it is {_sp0:.4f} against {_se7[0][1]:.4f}")
                checks += 1
                if not (_sp21 < _se7[21][1] and _sp0 > _se7[0][1]):
                    fails.append(f"Section S21 says the spread is inside Table 3's "
                                 f"Monte Carlo error at three weeks and about twice "
                                 f"it at zero delay; the spreads are {_sp0:.4f} and "
                                 f"{_sp21:.4f}")
    _m07g = re.search(r"Gross content first clears zero at delta = (\d+)", _l07c)
    checks += 1
    if not _m07g:
        fails.append("lab07 no longer states where gross content first clears zero")
    else:
        _gd = int(_m07g.group(1))
        _W1 = {1: "one day", 2: "two days", 3: "three days"}
        want("6 names the delay where gross content first clears zero",
             f"Gross content clears zero at {_W1.get(_gd, str(_gd) + ' days')}")
        for _other in (1, 2, 3):
            if _other == _gd:
                continue
            forbid(f"6 says gross clears zero at {_W1[_other]}",
                   f"Gross content clears zero at {_W1[_other]}")
    _m48 = re.search(r"relative width, calm against stressed: ([\d.]+)x on the "
                     r"rate, ([\d.]+)x on the age",
                     read_text(os.path.join(EXP, "lab48_age_by_regime.txt")))
    checks += 1
    if not _m48:
        fails.append("lab48 no longer prints the calm-against-stressed relative "
                     "widths, which both documents quote")
    else:
        want("the relative width quoted in days is lab48's age figure",
             f"the calm cell is {_m48.group(2)} times as wide relative to its own")
        # The sentence appears TWICE in the paper, and a want() is satisfied by
        # any one of them, so the occurrences are counted: a single stale copy
        # is how Tables S4, S12 and 1 went wrong, and the 7.3x this replaced
        # was wrong in both paper copies.
        _flat48 = plain_text_of(read_text(path))
        _n48 = _flat48.count("times as wide relative to its own")
        _n48ok = _flat48.count(f"{_m48.group(2)} times as wide relative to its own")
        checks += 1
        if _n48ok != _n48:
            fails.append(f"{_n48} sentences in the paper give the calm-against-"
                         f"stressed relative width and only {_n48ok} of them say "
                         f"{_m48.group(2)}x, which is lab48's figure for the age")
        forbid("the relative width in days quotes the rate figure instead",
               f"the calm cell is {_m48.group(1)} times as wide")
    _m13inc = re.search(r"^\s*55\s+([\d.]+)\s+([\d.]+)\s+\+([\d.]+)\s+"
                        r"([\d.]+)\s+([\d.]+)\s+\+([\d.]+)", read_text(
                            os.path.join(EXP, "lab13_origin_median.txt")), re.M)
    checks += 1
    if not _m13inc:
        fails.append("lab13 no longer prints the eleven-week row Section S16 quotes")
    else:
        want("S16's two increments and their own-only skill",
             f"worth +{_m13inc.group(3)} of R&sup2; at eleven weeks under the "
             f"target-dated outcome and +{_m13inc.group(6)} under the origin-dated "
             f"one, on own-only skill of +{_m13inc.group(1)} and "
             f"+{_m13inc.group(4)}")
    _l08c = read_text(os.path.join(EXP, "lab08_implied_vol.txt"))
    _i08 = {}
    for _ln in _l08c.split("same-day")[0].splitlines():
        _m = re.match(r"^\s*(\d+)\s+([-\d.]+)\s+[-\d.]+\s+([\d.]+)\s+", _ln)
        if _m:
            _i08[int(_m.group(1))] = (_m.group(2), _m.group(3))
    checks += 1
    if not ({0, 21, 55} <= set(_i08)):
        fails.append("lab08's first table no longer prints the rows Section S28 "
                     "quotes")
    else:
        want("S28's implied-volatility curve and the domestic one",
             f"from {_i08[0][1]} to {_i08[55][1]} across eleven weeks, where "
             f"the domestic-only forecaster falls from {_i08[0][0]} to "
             f"{_i08[55][0]}")
        want("S28's three-week cell",
             f"today's VIX scores {_i08[21][1]}, above the {_i08[0][0]} of a "
             f"forecaster with fully current domestic data")
    for _lname, _scope, _col, _label, _phrase in (
            ("lab23_compressed_everywhere", "B.  ", 8, "S28.1",
             "no Giacomini-White statistic reaches 1.96: the largest is {hi}"),
            ("lab28_orthogonal_breadth", "A.  ", 7, "S27",
             "separates the two at every delay, from z = {lo} to {hi}")):
        _body = read_text(os.path.join(EXP, _lname + ".txt"))
        if _scope in _body:
            _body = _body.split(_scope, 1)[1]
        _zs = []
        for _ln in _body.splitlines():
            _f = _ln.split()
            if len(_f) >= _col and re.fullmatch(r"\d+", _f[0]):
                for _tok in _f[1:]:
                    if re.fullmatch(r"-?\d\.\d\d", _tok):
                        _zs.append(abs(float(_tok)))
        checks += 1
        if len(_zs) < 6:
            fails.append(f"{_lname}: read {len(_zs)} statistics for {_label}'s "
                         f"quoted range")
        else:
            want(f"{_label} quotes {_lname}'s statistic range",
                 _phrase.format(lo=f"{min(_zs):.2f}", hi=f"{max(_zs):.2f}"))

    # --- Table S17, every cell, against lab15's WIDE grid -----------------
    # Three of its cells were a different grid's: the RFF(20) statistic at
    # thirteen and twenty-one days read -2.12, which is lab15's STRICT grid at
    # five days, and the foreign statistic at twenty-one read -0.69 against
    # -0.64.  The prose beside it quoted the lab correctly, so the table was
    # the stale artefact - and nothing here read the table.
    _l15w = read_text(os.path.join(EXP, "lab15_nonlinear_given_iv.txt"))
    _w15 = _l15w.split("WIDE grid")[0]
    _w15 = _w15[_w15.rfind("delta    R2 IV"):] if "delta    R2 IV" in _w15 else ""
    _g15 = {}
    for _ln in _w15.splitlines():
        _f = _ln.split()
        if len(_f) == 14 and re.fullmatch(r"\d+", _f[0]):
            _g15[int(_f[0])] = (_f[1], _f[2], _f[3], _f[5], _f[6],
                                _f[8], _f[9], _f[11], _f[12])
    checks += 1
    if len(_g15) != 6:
        fails.append(f"lab15: read {len(_g15)} of 6 WIDE-grid rows, so Table S17 "
                     f"rests on a short read")
    else:
        for _d in sorted(_g15):
            _r15 = _g15[_d]
            want(f"Table S17 row delta={_d}",
                 f"{_d} {_r15[0]} {_r15[1]} {_r15[2]} {_r15[3]} {_r15[4]} "
                 f"{_r15[5]} {_r15[6]} {_r15[7]} {_r15[8]}")

    # --- the horizon-matched gain range, against lab44 --------------------
    _l44 = read_text(os.path.join(EXP, "lab44_horizon_matched_iv.txt"))
    _g44 = [float(_m) for _m in re.findall(
        r"^\s*\d+\s+[-\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\+([\d.]+)\s+[\d.]+",
        _l44, re.M)]
    checks += 1
    if len(_g44) < 5:
        fails.append(f"lab44: read {len(_g44)} horizon-matched gains, so Section "
                     f"S30's range rests on a short read")
    else:
        want("S30's horizon-matched gain range",
             f"beats thirty-day VIX at every delay we test, by {min(_g44):.4f} to "
             f"{max(_g44):.4f} of R&sup2;")

    # --- Table S26 counts its own rows, and the span it quotes ------------
    _l55p = read_text(os.path.join(EXP, "lab55_illiquid_measured.txt"))
    _pm = re.findall(r"^\s{2,}([A-Za-z ]+?)\s+(\d{4}-\d{2})\s+(\d+)\s+"
                     r"[\d.]+%\s+([\d.]+)\s*$", _l55p, re.M)
    checks += 1
    if len(_pm) < 20:
        fails.append(f"lab55: read {len(_pm)} metro persistence rows, expected 20")
    else:
        _full55 = [r for r in _pm if r[1] == min(x[1] for x in _pm)]
        _top55 = max(_pm, key=lambda r: float(r[3]))
        want("Table S26's caption counts the panel and the rows separately",
             f"all {spell(len(_pm))} metropolitan indices and the span quoted in "
             f"the text is over all {spell(len(_pm))}; the rows below are the "
             f"{spell(len(_full55))} with a full history")
        want("Table S26's caption names the most persistent metro",
             f"The most persistent of the {spell(len(_pm))} is "
             f"{_top55[0].strip()} at {_top55[3]}")
        checks += 1
        if _top55 in _full55:
            fails.append(f"Table S26's caption says the most persistent metro is "
                         f"not a row of the table; {_top55[0].strip()} now is one")

    # --- Section S31.1's one-month cell, and the estimator behind it ------
    # S31.1 explained a mean-of-ratios reading of 160.3% as a fresh
    # cross-section beating a fresh mark.  It isn't that: the pooled ratio on
    # the same two columns is 57%, and the 160.3% is a mean of fourteen ratios
    # with one near-zero denominator among them - the failure Section S18
    # introduces pooling to avoid.
    _R55p = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+"
                       r"(n/a|[\d.]+%)\s+(n/a|\[\s*-?\d+%, ?-?\d+%\])", re.M)
    _ret55 = _l55p.split("C.  THE HEADLINE: NEXT MONTH'S RETURN")[1].split(
        "Per metro at")[0] if "C.  THE HEADLINE: NEXT MONTH'S RETURN" in _l55p else ""
    _rt = {}
    for _m in _R55p.finditer(_ret55):
        _rt[int(_m.group(1))] = (_m.group(2), _m.group(3), _m.group(4), _m.group(5))
    checks += 1
    if not ({0, 1} <= set(_rt)):
        fails.append("lab55's return-target table no longer prints the zero- and "
                     "one-month rows that Section S31.1 reads")
    else:
        _o0, _c0r = float(_rt[0][0]), float(_rt[0][1])
        _o1, _c1r = float(_rt[1][0]), float(_rt[1][1])
        _pool = 100 * (_c1r - _o1) / (_o0 - _o1)
        want("S31.1 quotes the mean-of-ratios reading and its interval",
             f"reads {_rt[1][2]}, with an interval of "
             f"{_rt[1][3].replace('%', '').replace('[ ', '[')}")
        want("S31.1 gives the pooled arithmetic",
             f"{_rt[1][1]} against {_rt[1][0]} over {_rt[0][0]} against "
             f"{_rt[1][0]} - gives {_pool:.0f}%")
        want("S31.1 gives the zero-delay pair that DOES show the Section 7 effect",
             f"the cross arm scores {_rt[0][1]} against {_rt[0][0]}")
        checks += 1
        if not _c0r > _o0:
            fails.append("Section S31.1 says the cross arm beats the fresh mark at "
                         "zero delay on this panel; lab55 no longer agrees")
        checks += 1
        if _pool > 100:
            fails.append(f"Section S31.1 says the pooled one-month rate carries no "
                         f"suggestion of exceeding one; it is {_pool:.0f}%")
        for _shape in ("exceeds 100%, which is Section 7's point",
                       "exceeds 100%, which is Section 7&rsquo;s point",
                       "above 100%, which is Section 7's point",
                       "it exceeds 100%, which is Section 7"):
            forbid("S31.1 explains the one-month cell by the Section 7 effect",
                   _shape)



    # --- the companion note's two repairs, against lab20 -------------------
    if companion:
        for phrase in ("restricted-information Gaussian benchmark",
                       "a = +0.27", "among functions of a alone",
                       "chosen with hindsight"):
            checks += 1
            if phrase not in ctxt:
                fails.append(f"companion repair: MISSING  '{phrase}'")
        # the two claims the referee falsified must not survive
        for phrase in ("No use of the magnitude of a, by a regression or anything else",
                       "There is no headroom for a better model to occupy",
                       "crosses one half exactly at a = 0"):
            checks += 1
            if phrase in ctxt:
                fails.append(f"companion overclaim: STALE  '{phrase}'")
        # The note used to list its six scripts by filename in its own
        # Reproducibility section.  A paper is not a README, so the credit now
        # lives in the repository's README and this check follows it there: the
        # requirement was never that the note print a filename, it was that the
        # script its numbers come from be findable.
        checks += 1
        if "lab20_ceiling_is_not_a_ceiling.py" not in RDME:
            fails.append("README.md does not name lab20_ceiling_is_not_a_ceiling.py, "
                         "whose figures the companion note quotes")
        # The note states how many scripts stand behind it, and that number is a
        # claim like any other: adding lab61 made it seven and the sentence said
        # six.  The list is enumerated here rather than counted from the
        # verifier's own reads, because those happen in two places and one of
        # them runs after the main paper's count is taken.
        _NOTE_SCRIPTS = ("lab02_delay_curve", "lab02b_threshold_ceiling",
                         "lab20_ceiling_is_not_a_ceiling", "lab24_proper_scores",
                         "lab25_window_sensitivity", "lab26_quantile_target",
                         "lab61_block_sensitivity")
        for _ns in _NOTE_SCRIPTS:
            checks += 1
            if _ns + ".py" not in RDME:
                fails.append(f"README.md does not name {_ns}.py, which the companion "
                             "note's Reproducibility section counts")
            checks += 1
            if not os.path.isfile(os.path.join(EXP, _ns + ".txt")):
                fails.append(f"{_ns}.txt is missing from expected_output, so the "
                             "companion note's script count is not reproducible")
        # cwant() belongs to the second companion block further down; this one
        # predates it and tests ctxt directly.
        # --- the threshold note's four presentation repairs -----------------
        # A duplicate table number, a computability claim that is really a
        # usefulness claim, a proper score filed under continuous forecasting,
        # and four forecasters whose common evaluation target was never stated.
        # Every lab builds the label from the close-to-close proxy alone, so
        # the last of these is a statement the code supports and the note did
        # not make.
        for _lbl, _need in (
                ("no duplicate Table 6",
                 "Table 7. Block-length and bootstrap-family sensitivity"),
                ("hard labels: usefulness, not computability",
                 "makes the Brier score equivalent to the error rate and makes the "
                 "log score infinite after any error"),
                ("proper scores filed correctly",
                 "the Brier and log scores evaluate probability forecasts of the "
                 "binary event, which is what they are proper for"),
                ("one evaluation target for all forecasters",
                 "CC and YZ denote alternative information variables a forecaster "
                 "may hold, not alternative evaluation targets")):
            cwant(_lbl, _need)
        cforbid("stale hard-label claim",
                "cannot be computed for a forecaster that emits a hard label at all")
        cforbid("stale continuous-forecasting terminology",
                "using AUC and the Brier and log scores; for continuous volatility "
                "forecasting")
        # two tables may not share a number
        _nums = re.findall(r"<b>Table (\d+)\.</b>", read_text(companion))
        checks += 1
        if len(_nums) != len(set(_nums)):
            _dup = sorted({n for n in _nums if _nums.count(n) > 1})
            fails.append(f"the companion note numbers more than one table "
                         f"{', '.join('Table ' + n for n in _dup)}")

        # The companion's abstract states a non-rejection.  It must state the
        # bound rather than say the HAR classifier does not "improve", which
        # reads as a demonstrated equality.  This check lives inside the
        # companion block because want() reads the paper and the appendix: a
        # companion needle written outside here searches the wrong documents
        # and fails for the wrong reason, which is how it was first written.
        for _lbl, _needle in (
                ("companion abstract bounds its non-rejection",
                 "a failure to separate and not an equality"),
                ("note script count",
                 f"produced by {spell(len(_NOTE_SCRIPTS))} scripts"),
                ("note script self-import",
                 f"One of the {spell(len(_NOTE_SCRIPTS))} imports another")):
            checks += 1
            if _needle not in ctxt:
                fails.append(f"{_lbl}: MISSING  '{_needle}'")
        # Section 5 spends two paragraphs saying A* is not a ceiling; a table
        # column headed "ceiling" undid that, and is now forbidden rather than
        # merely fixed.
        for phrase in ("ceiling A*", "% of ceiling",
                       "sit between 90% and 106% of the ceiling"):
            checks += 1
            if phrase in ctxt:
                fails.append(f"companion: '{phrase}' calls the benchmark a ceiling, "
                             "which Section 5 spends two paragraphs denying")
        # Table 3 used to carry one column headed "Gaussian benchmark A*".  It
        # now carries two, because the same-sign figure and the attainable one
        # part company once rho goes negative, and only the second is a
        # benchmark.  The requirement is unchanged in substance: the attainable
        # column must be labelled as the benchmark and the other must not be.
        checks += 1
        if "Gaussian benchmark, best cut" not in ctxt:
            fails.append("companion: Table 3 no longer heads its attainable column "
                         "'Gaussian benchmark, best cut'")
        checks += 1
        if "Gaussian benchmark A*" in ctxt:
            fails.append("companion: Table 3 again calls the same-sign accuracy the "
                         "Gaussian benchmark; at negative rho that column is below "
                         "one half and bounds nothing")
        checks += 1
        if "the episode count grows in only two honest ways" not in ctxt.lower():
            fails.append("companion: Section 7 no longer says what would repair the "
                         "tail-target event shortage")

    # --- Tables 6 and 7, against lab23 ------------------------------------
    l23 = read_text(os.path.join(EXP, "lab23_compressed_everywhere.txt"))
    b23 = l23.split("B.  REDUNDANCY GIVEN")[-1].split("=" * 40)[2] \
        if l23.count("=" * 40) > 2 else ""
    n23 = 0
    ROW23 = re.compile(r"\s*(\d+)" + r"\s+([\d.]+)" * 4 +
                       r"\s+([-+][\d.]+)\s+([-+]?[\d.]+)"
                       r"\s+([-+][\d.]+)\s+([-+]?[\d.]+)\s*$")
    for ln in b23.splitlines():
        m = ROW23.match(ln)
        if m:
            n23 += 1
            d, iv, mn, pc, full, im, iz, pm_, pz = m.groups()
            want(f"lab23 d={d} redundancy row",
                 f"{iv} {mn} {pc} {full} {im} {iz} {pm_} {pz}")
    checks += 1
    if n23 != 6:
        fails.append(f"lab23: parsed {n23} redundancy rows, expected 6")
    # the compressed substitution rates and their intervals
    # Table 1 carries the FIXED-B set for these rows now, not the bootstrap.
    # It used to carry the bootstrap for two rows and the Fieller set for two
    # others while its caption named the bootstrap for all four, which is the
    # same kind of caption-against-content mismatch as lab25's header.  Both
    # columns are parsed: the Fieller set is what must appear, and the
    # bootstrap's endpoints are banned from the document so the old pair cannot
    # quietly return.
    _n23r = 0
    for ln in l23.split("C.  THE SUBSTITUTION RATE")[-1].splitlines():
        m = re.match(r"\s*(FULL|MEAN|PC1|PC2)\s+([\d.]+)%\s+"
                     r"\[\s*([\d.]+)%,\s*([\d.]+)%\]\s+[\d.]+\s+"
                     r"([\d.]+)\s+\[\s*([\d.]+)%,\s*([\d.]+)%\]", ln)
        if m:
            _n23r += 1
            a, pt, blo, bhi, tau, flo, fhi = m.groups()
            if a in ("FULL", "MEAN", "PC1"):
                want(f"lab23 {a} rate is the fixed-b set",
                     f"{pt}% [{flo}, {fhi}]")
                forbid(f"lab23 {a} bootstrap pair returned",
                       f"{pt}% [{blo}, {bhi}]")
            # and the fixed-b set must be the wider of the two, which is the
            # direction Section 4.6 measures; if it stopped being so, the
            # caption's claim about the bootstrap being narrower would be wrong
            checks += 1
            if float(fhi) - float(flo) <= float(bhi) - float(blo):
                fails.append(f"lab23 {a}: the fixed-b set [{flo}, {fhi}] is no "
                             f"wider than the bootstrap's [{blo}, {bhi}], but "
                             f"Table 1's caption says the bootstrap is narrower")
    checks += 1
    if _n23r != 4:
        fails.append(f"lab23: parsed {_n23r} of 4 rate rows with both constructions")
    # renumber-proof: the journal variants shift section numbers (4.6 becomes
    # 3.6 in the JFEc layout), so a needle carrying the number passes on the
    # source and fails on a variant that is in fact correct.
    want("Table 1 caption names one construction",
         "Every interval here is the fixed-b Fieller set")
    forbid("Table 1 caption names the bootstrap for every row",
           "Intervals are the full-ratio moving-block bootstrap of Section 4.5")
    # the sign flip is the finding; it must not be reported as a reversal
    want("lab23 sign flip", "The sign flips and the conclusion does not")
    forbid("lab23 overclaim", "the foreign block adds nothing given implied volatility.")

    # --- Table 4 and Section 5.2, against lab21 ---------------------------
    # Row-anchored: every one of these figures is four digits after a sign and
    # several appear elsewhere in the paper on their own, so the whole row is the
    # needle.  This is the check that would have caught the earlier draft, whose
    # SHIFT column was the NAIVE column by mistake.
    l21 = read_text(os.path.join(EXP, "lab21_stronger_inference.txt"))
    a21 = l21.split("A.  DOES THE COST")[-1].split("B.  AN INTERVAL")[0]
    ROW21 = re.compile(r"\s*(\d+)\s+([-+][\d.]+)\s+([-+][\d.]+)\s+([-+][\d.]+)"
                       r"\s+([\d.]+)\s+([-+][\d.]+)\s+([-+][\d.]+)\s+([-+][\d.]+)\s*$")
    n21 = 0
    for ln in a21.splitlines():
        m = ROW21.match(ln)
        if m:
            n21 += 1
            d, net, car, csh, se, cnv, gar, gsh = m.groups()
            want(f"lab21 d={d} generator row",
                 f"{net} {car} {csh} {se} {cnv} {gar} {gsh}")
    checks += 1
    if n21 != 7:
        fails.append(f"lab21: parsed {n21} generator rows, expected 7")

    # the verdict, and the two traps, must match what the lab actually printed
    # Each entry is (what the lab prints, the phrase the paper must carry).  The
    # paper side is anchored rather than bare: "0.0060" on its own also occurs in
    # a Section 5 interval, so a bare needle passes whatever Table 4 says - the
    # spurious-match bug this file was bitten by once already.
    # Each figure is read out of lab21's own text and then required of the
    # paper, so a rerun that moves one fails here instead of agreeing with a
    # literal kept in this file.  The paper side stays anchored in a phrase:
    # "0.0060" on its own also occurs in a Section 5 interval, so a bare needle
    # would pass whatever Table 4 said - the spurious-match bug this file was
    # bitten by once already.
    _gap = max(re.findall(r"gap ([\d.]+) +tolerance", l21), key=float, default=None)
    _naive = re.search(r"([\d.]+) of R2 below the unfitted estimate at delta = 0, "
                       r"where the real\s+cost is small, and ([\d.]+) below it", l21)
    _ident = re.search(r"scores \+([\d.]+) on its own", l21)
    _sd = re.search(r"with standard deviation ([\d.]+),", l21)
    _lost = re.search(r"would cost the surrogate arm (\d+) of \d+ test days per draw "
                      r"\(([\d.]+%)\)", l21)
    _close = re.search(r"closest call: delta = \d+, at ([\d.]+) of its tolerance", l21)
    for got, phrase, label in (
            (_gap, "largest gap is {} of R&sup2; at &delta; = 13",
             "lab21 largest gap"),
            (_close and _close.group(1), "which is {} of the tolerance there",
             "lab21 closest call"),
            (_lost and _lost.group(1), "about {} test days per draw",
             "lab21 test days lost per naive draw"),
            (_naive and _naive.group(1),
             "sits {} of R&sup2; below the corrected estimate",
             "lab21 naive penalty at d=0"),
            (_naive and _naive.group(2), "and {} below it at &delta; = 55",
             "lab21 naive penalty at d=55"),
            (_ident and _ident.group(1), "scored +{} on its own",
             "lab21 near-identity draw"),
            (_sd and _sd.group(1), "standard deviation of {}",
             "lab21 near-identity spread")):
        checks += 1
        if got is None:
            fails.append(f"{label}: lab21 no longer prints the figure the paper quotes")
        else:
            want(label, phrase.format(got))
    # the paper may claim agreement only while the lab prints agreement
    checks += 1
    if "the two generators agree at every delay" not in l21:
        fails.append("lab21: the lab no longer prints the agreement verdict")
    want("lab21 agreement claim", "The two generators agree at every delay")
    forbid("lab21 stale disagreement", "the two generators disagree by more than")

    # --- the Fieller paragraph, against lab21 part B ----------------------
    b21 = l21.split("B.  AN INTERVAL")[-1]
    checks += 1
    if "confidence sets that are NOT an ordinary interval: 0 of 6" not in b21:
        fails.append("lab21: a confidence set is no longer an ordinary interval, "
                     "but Section 6 still says every one is")
    want("lab21 Fieller ordinary intervals",
         "one day to eleven weeks the confidence set is an ordinary interval")
    # Both of these were literals and both moved when the block widened.  The
    # gap and the delta = 55 pair now come out of lab21's own output.
    _l21raw = read_text(os.path.join(EXP, "lab21_stronger_inference.txt"))
    _gap = re.search(r"either endpoint: ([\d.]+)%", _l21raw)
    checks += 1
    if not _gap:
        fails.append("lab21 no longer prints the largest Fieller/bootstrap gap")
    else:
        want("lab21 Fieller endpoint gap", f"{_gap.group(1)} percentage points")
    checks += 1
    row55 = [ln for ln in b21.splitlines() if ln.strip().startswith("55 ")]
    m55 = re.search(r"\[\s*(\d+)%,\s*(\d+)%\]\s*\[\s*(\d+)%,\s*(\d+)%\]",
                    row55[0]) if row55 else None
    if not m55:
        fails.append("lab21 part B: could not parse the delta = 55 row")
    else:
        plo, phi, flo, fhi = m55.groups()
        want("lab21 Fieller d=55 pair",
             f"[{flo}%, {fhi}%] against [{plo}%, {phi}%]")
    checks += 1
    if _gap and f"{_gap.group(1)}%" not in b21:
        fails.append("lab21 part B: the endpoint gap is not in the lab output")

    # --- Table 5 and Section 5.3, against lab28 ---------------------------
    l28 = read_text(os.path.join(EXP, "lab28_orthogonal_breadth.txt"))
    R28 = re.compile(r"\s*(\d+)" + r"\s+([-+]?[\d.]+)" * 4 +
                     r"\s+([-+][\d.]+)\s+([-+][\d.]+)\s+([\d.]+)\s*$")
    n28 = 0
    for ln in l28.split("A.  WHICH HALF")[-1].splitlines():
        m = R28.match(ln)
        if m:
            n28 += 1
            d, ownr, pc1, rst, ful, dp, dr, z = m.groups()
            want(f"lab28 d={d} split row", f"{ownr} {pc1} {rst} {ful} {dp} {dr} {z}")
    checks += 1
    if n28 != 6:
        fails.append(f"lab28: parsed {n28} rows from part A, expected 6")
    checks += 1
    if "ORTHOGONAL remainder beats domestic-only: 0 of 6" not in l28:
        fails.append("lab28: the orthogonal remainder now adds somewhere, but Section "
                     "5.3 still says it adds nowhere")
    want("lab28 verdict", "is <em>negative</em> at every delay".replace("<em>", "")
         .replace("</em>", ""))
    want("lab28 rotation check", "to within 0.0010 of R&sup2; at every delay")
    checks += 1
    if "largest gap: 0.0010 of R2" not in l28:
        fails.append("lab28: the rotation check no longer gives a 0.0010 gap")
    forbid("lab28 stale mechanism claim", "bilateral diffusion between particular markets, ")

    # --- Tables 14 and 15 and Section 8, against lab27 --------------------
    # Table S7 moved from lab27's part A (moving-block bootstrap) to its part
    # A2 (fixed-b Fieller), because the share of own MSE removed is a ratio of
    # two means and this table carried the one comparative claim still resting
    # on a resampled interval.  Both parts are parsed: the table reads A2, and
    # the prose comparing the two constructions reads both and is recomputed
    # rather than trusted, which is the Table S4 lesson applied before the fact.
    l27 = read_text(os.path.join(EXP, "lab27_regime_conditioning.txt"))
    _p27a = l27.split("A.  DOES THE CROSS-SECTION")[1].split("A2. THE SAME LADDER")[0]
    _p27f = l27.split("A2. THE SAME LADDER")[1].split("B.  IS BREADTH")[0]
    _A27, _F27 = {}, {}
    for _m in re.finditer(r"^\s*(\d+)\s{2,}(\S.*?)\s{2,}(\d+)\s+[+-]\d\.\d{4}\s+"
                          r"(-?\d+\.\d)%\s+\[\s*([+-]\s*\d+\.\d)%,\s*"
                          r"([+-]\s*\d+\.\d)%\]\s+(yes|no)\s*$", _p27a, re.M):
        _A27[(int(_m[1]), _m[2].strip())] = (_m[4],
                                             float(_m[5].replace(" ", "")),
                                             float(_m[6].replace(" ", "")), _m[7])
    for _m in re.finditer(r"^\s*(\d+)\s{2,}(\S.*?)\s{2,}(\d+)\s+(\d+)\s+\d\.\d{3}\s+"
                          r"(-?\d+\.\d)%\s+\[\s*([+-]\s*\d+\.\d)%,\s*"
                          r"([+-]\s*\d+\.\d)%\]\s+(-?\d+\.\d)\s+(\w+)\s*$",
                          _p27f, re.M):
        _F27[(int(_m[1]), _m[2].strip())] = (_m[5],
                                             float(_m[6].replace(" ", "")),
                                             float(_m[7].replace(" ", "")),
                                             _m[8], _m[9], int(_m[4]), int(_m[3]))
    checks += 1
    if len(_A27) != 28 or len(_F27) != 28:
        fails.append(f"lab27: parsed {len(_A27)} bootstrap and {len(_F27)} fixed-b "
                     f"ladder cells, expected 28 of each")
    _LAD27 = ["VIX top 33%", "VIX top 10%", "VIX top 5%", "episodes, peak +/-20d"]
    _CALM27 = ["VIX bottom 67%", "outside every episode"]
    # Table S7's own six rows, in the order the table prints them, each demanded
    # with the test-day count, the bandwidth and tau beside it so that no cell
    # can be matched by a figure borrowed from a neighbouring row.
    _S7ROWS = ["VIX top 5%", "VIX top 10%", "VIX top 33%",
               "episodes, peak +/-20d", "VIX bottom 67%", "outside every episode"]
    _S7LBL = {"episodes, peak +/-20d": "episodes, peak &plusmn;20 days"}
    n27 = 0
    for _k in _S7ROWS:
        _f = _F27.get((55, _k))
        checks += 1
        if _f is None:
            fails.append(f"lab27 part A2 no longer prints '{_k}' at eleven weeks, "
                         f"so that row of Table S7 rests on nothing")
            continue
        n27 += 1
        want(f"Table S7 row {n27} ({_k})",
             f"{_S7LBL.get(_k, _k)} {_f[6]:,} {_f[5]} {_f[0]}% "
             f"[{_f[1]:.1f}, {_f[2]:.1f}] {_f[3]}")
        checks += 1
        if _f[4] != "interval":
            fails.append(f"Table S7's '{_k}' row is now a {_f[4]} set, which the "
                         f"table prints as an ordinary interval")
    checks += 1
    if n27 != 6:
        fails.append(f"lab27: matched {n27} Table S7 rows, expected 6")
    # the comparative claim, recomputed under BOTH constructions
    _ft, _fc = _F27[(55, "VIX top 5%")], _F27[(55, "VIX bottom 67%")]
    _at, _ac = _A27[(55, "VIX top 5%")], _A27[(55, "VIX bottom 67%")]
    checks += 1
    if _ft[1] <= _fc[2]:
        fails.append(f"lab27: the severest and calmest rungs no longer separate "
                     f"under fixed-b ([{_ft[1]}, {_ft[2]}] against "
                     f"[{_fc[1]}, {_fc[2]}]), but Section S10 still says they do")
    checks += 1
    if "sets disjoint: yes" not in l27:
        fails.append("lab27 part A2 no longer reports the two rungs as disjoint")
    checks += 1
    if "intervals disjoint: yes" not in l27:
        fails.append("lab27: the severest and calmest rungs no longer separate, but "
                     "Section 8.1 still says they do")
    want("Table S7 gap under fixed-b",
         f"disjoint sets, {_ft[1]:.1f} against {_fc[2]:.1f}")
    want("Table S7 gap against the bootstrap's",
         f"the gap between the two sets is {_ft[1] - _fc[2]:.1f} points where the "
         f"bootstrap's was {_at[1] - _ac[2]:.1f}")
    # the three construction comparisons Section S10 makes, all computed
    _wider = [round((_F27[(55, k)][2] - _F27[(55, k)][1])
                    - (_A27[(55, k)][2] - _A27[(55, k)][1]), 1) for k in _S7ROWS]
    _outw = sum((_F27[(55, k)][1] < _A27[(55, k)][1])
                + (_F27[(55, k)][2] > _A27[(55, k)][2]) for k in _S7ROWS)
    checks += 1
    if not all(w > 0 for w in _wider):
        fails.append(f"Section S10 says all six Table S7 sets are wider than the "
                     f"bootstrap's; the width differences are {_wider}")
    want("S10 width range", f"all six are wider, by {min(_wider):.1f} to "
                            f"{max(_wider):.1f} points")
    want("S10 endpoints moved outward",
         f"{_num_word(_outw)} of the twelve endpoints moved outward")
    _exc = [k for k in _S7ROWS
            if not (_F27[(55, k)][1] < _A27[(55, k)][1]
                    and _F27[(55, k)][2] > _A27[(55, k)][2])]
    checks += 1
    if sorted(_exc) != sorted(["VIX top 33%", "episodes, peak +/-20d"]):
        fails.append(f"Section S10 names the top-33% upper end and the narrow "
                     f"window's lower end as the only endpoints that moved "
                     f"inward; lab27 now makes the exceptions {_exc}")
    want("S10 names the top-33% exception",
         f"the upper end of the top-33% rung, {_F27[(55, 'VIX top 33%')][2]:.1f} "
         f"against {_A27[(55, 'VIX top 33%')][2]:.1f}")
    want("S10 names the narrow-window exception",
         f"the lower end of the narrow calendar window, "
         f"{_F27[(55, 'episodes, peak +/-20d')][1]:.1f} against "
         f"{_A27[(55, 'episodes, peak +/-20d')][1]:.1f}")
    _fs = sum(1 for d in (0, 5, 21, 55) for k in _LAD27
              if _F27[(d, k)][4] == "interval" and _F27[(d, k)][1] > 0)
    _as = sum(1 for d in (0, 5, 21, 55) for k in _LAD27 if _A27[(d, k)][3] == "yes")
    want("S10 count of separated stress cells",
         f"falls from {_num_word(_as)} of sixteen to {_num_word(_fs)} of sixteen")
    checks += 1
    if f"STRESS rungs: {_fs} of 16" not in _p27f or f"STRESS rungs: {_as} of 16" not in _p27a:
        fails.append(f"lab27 no longer reports {_as} of 16 bootstrap and {_fs} of 16 "
                     f"fixed-b stress cells above zero")
    _chg = [(d, k) for d in (0, 5, 21, 55) for k in _LAD27
            if (_A27[(d, k)][3] == "yes")
            != (_F27[(d, k)][4] == "interval" and _F27[(d, k)][1] > 0)]
    checks += 1
    if _chg != [(5, "episodes, peak +/-20d")]:
        fails.append(f"Section S10 names delta = 5 on the narrow calendar window as "
                     f"the single cell whose status changes; lab27 now gives {_chg}")
    _cA, _cF = _A27[(5, "episodes, peak +/-20d")], _F27[(5, "episodes, peak +/-20d")]
    want("S10 names the cell that changes status",
         f"{_cF[0]}%, which reads [{_cA[1]:.1f}, {_cA[2]:.1f}] under the bootstrap "
         f"and [-{abs(_cF[1]):.1f}, {_cF[2]:.1f}] under the inversion")
    want("S10 gives that cell's days and tau",
         f"{_cF[6]} days with a denominator t -statistic of {_cF[3]}")
    # the ladder's SHAPE, counted: both documents said "four delays by six
    # states", which is the six rows Table S7 prints, and then counted the
    # twenty-eight cells lab27 actually runs in the same sentence.
    _st27 = sorted({k for _, k in _F27})
    _dl27 = sorted({d for d, _ in _F27})
    want("S10 gives the ladder's dimensions",
         f"{_num_word(len(_dl27))} delays by {_num_word(len(_st27))} states, of "
         f"which this table shows {_num_word(len(_S7ROWS))}")
    want("4.6 gives the ladder's dimensions",
         f"{_num_word(len(_dl27))} delays by {_num_word(len(_st27))} market states")
    checks += 1
    if len(_dl27) * len(_st27) != len(_F27):
        fails.append(f"the ladder's dimensions do not multiply out: "
                     f"{len(_dl27)} delays by {len(_st27)} states against "
                     f"{len(_F27)} cells")
    want("S10 counts the ladder's cells",
         f"all {_num_word(len(_F27))} cells returned ordinary bounded intervals")
    def _taus27(keys, delays=(0, 5, 21, 55)):
        v = [float(_F27[(d, k)][3]) for d in delays for k in keys]
        return min(v), max(v)
    _s55, _c55 = _taus27(_LAD27, (55,)), _taus27(_CALM27, (55,))
    _all27 = _taus27(_LAD27 + _CALM27)
    want("S10 tau range in Table S7",
         f"{_s55[0]:.1f} to {_s55[1]:.1f} on the stress rungs and "
         f"{_c55[0]:.1f} to {_c55[1]:.1f} on the calm ones")
    want("S10 tau range over the full ladder",
         f"it runs {_all27[0]:.1f} to {_all27[1]:.1f}")
    want("4.6 tau range on Table S7's stress rungs",
         f"from {_s55[0]:.1f} to {_s55[1]:.1f} on the stressed rungs of Table S7")
    want("4.6 tau on Table S7's calm rungs",
         f"reach {_c55[0]:.1f} and {_c55[1]:.1f}")
    want("4.6 full-ladder tau range",
         f"&tau; runs from {_all27[0]:.1f} to {_all27[1]:.1f}")
    checks += 1
    if not (_c55[0] >= 10.0 and _s55[1] < 10.0):
        fails.append(f"Section 4.6 says Table S7's calm rungs sit at ten standard "
                     f"errors and its stress rungs at the middle strength; lab27 "
                     f"now gives calm {_c55} and stress {_s55}")
    checks += 1
    if any(_F27[k][4] != "interval" for k in _F27):
        fails.append("Section S10 says all twenty-eight cells returned bounded "
                     "intervals; at least one is now unbounded or an exclusion set")
    want("S10 says every cell came back bounded",
         "all twenty-eight cells returned ordinary bounded intervals")
    forbid("claim table still cites the bootstrap rungs",
           f"removes 63.1% [{_at[1]:.1f}, {_at[2]:.1f}] of domestic error")
    want("lab27 direction", "it roughly doubles, and it rises monotonically")
    checks += 1
    if "on STRESS rungs: 0 of 16" not in l27:
        fails.append("lab27: the foreign block now adds given implied volatility on some "
                     "stressed rung, but Section 8.2 still says none")
    want("lab27 redundancy in stress", "the foreign block adds significantly in none")
    # The three row tokens that used to live here - "VIX top 5% 367 63.1%" and
    # its two neighbours - are gone, not dropped: the Table S7 block above
    # demands every one of the six rows with its test-day count, its bandwidth,
    # its set AND its tau, which is strictly more than those three pinned and
    # covers the three rows they missed.
    for tok, phrase, label in (
            ("324", "severest rung carries 324 days", "lab27 severest rung size"),):
        checks += 1
        if tok not in l27:
            fails.append(f"{label}: '{tok}' not in lab27 output")
        want(label, phrase)

    # --- Table S20: why the variance-scale rates carry no interval --------
    # The caption claimed a bootstrap construction for intervals the paper no
    # longer prints.  Inverting those columns is the honest answer and lab10
    # part 1a already did it: neither variance-scale rate is identified at most
    # delays.  Every figure in the paragraph that now reports this is read out
    # of that table, counts included, because a count of exclusion sets is
    # exactly the kind of figure that goes stale silently on a rerun.
    _s20 = read_text(os.path.join(EXP, "lab10_loss_scale.txt"))
    _s20b = _s20.split("1a. THE SAME RATES, INVERTED")[1].split("\n1b.")[0]
    _SET = (r"(?P<p{i}>-?\d+)% (?:outside \[(?P<xlo{i}>\d+)%,(?P<xhi{i}>\d+)%\]"
            r"|(?P<whole{i}>the whole line)"
            r"|\[\s*(?P<lo{i}>-?\d+)%,\s*(?P<hi{i}>-?\d+)%\])")
    _S20ROW = re.compile(
        r"^\s*(?P<d>\d+)\s+(?P<tau>[\d.]+)\s+"
        + _SET.format(i=1) + r"\s+" + _SET.format(i=2) + r"\s+"
        + _SET.format(i=3) + r"\s+" + _SET.format(i=4) + r"\s*$")
    _nrm, _nat = {}, {}
    for _ln in _s20b.splitlines():
        _m = _S20ROW.match(_ln)
        if _m:
            _d = int(_m.group("d"))
            _nrm[_d] = (_m.group("p3"),
                        "outside" if _m.group("xlo3") is not None else "other",
                        _m.group("xlo3"), _m.group("xhi3"))
            _nat[_d] = (_m.group("p4"),
                        "the whole line" if _m.group("whole4") else "bounded",
                        _m.group("lo4"), _m.group("hi4"))
    checks += 1
    if len(_nrm) != 8:
        fails.append(f"lab10 part 1a: read {len(_nrm)} of 8 variance-scale rows, so "
                     f"Section S25's identification paragraph rests on a short read")
    else:
        checks += 1
        if not all(v[1] == "outside" for v in _nrm.values()):
            fails.append("Section S25 says the normalised rate returns an exclusion "
                         "set at all eight delays; lab10 no longer agrees")
        want("S25 says the normalised rate is an exclusion set everywhere",
             f"returns an exclusion set at all {_num_word(len(_nrm))} delays")
        _lowband = min(int(v[2]) for v in _nrm.values())
        want("S25 gives the band the normalised set excludes",
             f"whose excluded middle begins no lower than {_lowband} at any delay")
        _d55 = _nrm[55]
        want("S25 quotes the normalised set at eleven weeks",
             f"the set is everything outside [{_d55[2]}, {_d55[3]}]")
        _wl = sorted(d for d in _nat if _nat[d][1] == "the whole line")
        _bd = sorted(d for d in _nat if _nat[d][2] is not None)
        checks += 1
        if len(_wl) != 6 or len(_bd) != 2:
            fails.append(f"Section S25 says the natural-scale rate is the whole line "
                         f"at six of eight delays and bounded at two; lab10 now "
                         f"gives {len(_wl)} and {len(_bd)}")
        else:
            want("S25 counts the natural-scale whole-line sets",
                 f"returns the whole real line at {_num_word(len(_wl))} of the "
                 f"{_num_word(len(_wl) + len(_bd))} delays")
            _w = [int(_nat[d][3]) - int(_nat[d][2]) for d in _bd]
            want("S25 gives the two bounded widths",
                 f"sets {_w[0]} and {_w[1]} points wide at the other two")
            want("S25 quotes the two bounded sets",
                 f"{_nat[_bd[0]][0]}% [{_nat[_bd[0]][2]}, {_nat[_bd[0]][3]}] at "
                 f"&delta; = {_bd[0]} and {_nat[_bd[1]][0]}% "
                 f"[{_nat[_bd[1]][2]}, {_nat[_bd[1]][3]}] at &delta; = {_bd[1]}")
    _mni = re.search(r"sets that are NOT an ordinary interval: (\d+) of (\d+) \[(.*?)\]\n",
                     _s20b, re.S)
    checks += 1
    if not _mni:
        fails.append("lab10 part 1a no longer reports how many of its sets are not "
                     "ordinary intervals, so Section S25's count rests on nothing")
    else:
        _tot = int(_mni.group(1))
        _inmse = _mni.group(3).count("'mse'") + _mni.group(3).count("'msen'")
        want("S25 counts the sets that are not ordinary intervals",
             f"{_num_word(_tot)} of {spell(int(_mni.group(2)))} sets are not "
             f"ordinary intervals")
        checks += 1
        if _inmse != _tot:
            fails.append(f"Section S25 says every one of the {_tot} non-interval sets "
                         f"is in the two variance-scale columns; lab10 now puts "
                         f"{_inmse} of them there")
        want("S25 says all of them are in the two variance-scale columns",
             f"every one of the {_num_word(_tot)} is in these two columns")
    _tau10 = [float(x) for x in re.findall(r"^\s*\d+\s+([\d.]+)\s+-?\d+% \[",
                                           _s20b, re.M)]
    checks += 1
    if not _tau10:
        fails.append("lab10 part 1a no longer prints tau, so Section S25's closing "
                     "sentence has no support")
    else:
        want("S25 gives the tau range of the columns the paper leans on",
             f"denominator t -statistics of {min(_tau10)} to {max(_tau10)}")
    forbid("Table S20 caption still names a bootstrap interval",
           "Intervals in the paper bootstrap the whole ratio on the same resampled "
           "days")
    want("Table S20 caption says why it carries no interval",
         "This table carries no interval column, and the paragraph after it says "
         "why")

    # --- Tables 16 and 17 and the clock test, against lab37 ---------------
    # Every figure Section 8.3 quotes is anchored to the ROW it came from, not
    # to a bare token.  A bare "29%" appears in half a dozen unrelated places in
    # this output; a row that reads "5 51% 29% ..." appears once.
    l37 = read_text(os.path.join(EXP, "lab37_lead_lag.txt"))
    l37f = " ".join(l37.split())

    RATE = {3:  "3 36% 9% -8% -15% -14% -10%",
            5:  "5 51% 29% 13% -0% -11% -8%",
            13: "13 64% 52% 42% 34% 22% -1%",
            21: "21 68% 57% 50% 44% 33% 16%",
            55: "55 71% 64% 58% 53% 46% 30%"}
    # Stripping tags and collapsing whitespace turns an HTML table row into
    # exactly the shape the lab prints it in, so the WHOLE row is the needle.
    # Cell by cell would be six loose tokens per row, and "9%" alone occurs in
    # four unrelated places in this paper.
    for d, row in RATE.items():
        checks += 1
        if row not in l37f:
            fails.append(f"Table 15 row delta={d} is no longer what lab37 prints")
        want(f"Table 15 row delta={d}", row)

    AGE = {3:  "3 1.7d 2.7d 3.3d 3.7d 3.6d 3.4d",
           5:  "5 1.9d 3.0d 4.0d 5.0d 5.9d 5.6d",
           13: "13 2.8d 4.2d 5.5d 6.8d 8.9d 13.2d",
           21: "21 3.5d 5.2d 6.7d 8.1d 10.5d 15.0d",
           55: "55 4.6d 6.8d 8.4d 10.0d 12.6d 19.4d"}
    for d, row in AGE.items():
        checks += 1
        if row not in l37f:
            fails.append(f"Table 18 row delta={d} is no longer what lab37 prints")
        want(f"Table 18 row delta={d}", row)

    # --- Table S8's panel is NOT Table 2's, and the caption said it was ---
    # The caption claimed "same test days as Table 2; g = 0 reproduces it".
    # lab37 builds its own (delta, g) grid and ends with a different window,
    # so the g = 0 column is Table 2's comparison recomputed rather than
    # Table 2's column.  Section S4 had already conceded the panels differ.
    _n37 = re.search(r"target SPX, ([\d,]+) test days", l37)
    _n2 = re.search(r"The test window runs from [^.]*?, ([\d,]+) days", txt)
    checks += 1
    if not (_n37 and _n2):
        fails.append("could not read both panels' test-day counts for Table S8's "
                     "caption")
    else:
        _d37n = int(_n37.group(1).replace(",", ""))
        _d2n = int(_n2.group(1).replace(",", ""))
        checks += 1
        if _d37n == _d2n:
            fails.append(f"Table S8's caption says its panel differs from Table 2's; "
                         f"both are now {_d2n} days")
        want("Table S8's caption gives both test-day counts",
             f"ends with {_d37n:,} test days against Table 2's {_d2n:,}")
        _o37 = re.search(r"^\s*55\s+([\d.]+)\s", l37, re.M)
        _o2 = re.search(r"\b55 ([\d.]+) 0\.3736\b", txt)
        checks += 1
        if not (_o37 and _o2):
            fails.append("could not read both panels' own-only R2 at eleven weeks")
        else:
            want("Table S8's caption gives both own-only figures",
                 f"at eleven weeks is {_o37.group(1)} where Table 2 has "
                 f"{_o2.group(1)}")
            checks += 1
            if _o37.group(1) == _o2.group(1):
                fails.append("Table S8's caption contrasts two own-only figures that "
                             "are now identical")
        _r37_21 = re.search(r"\n\s*21\s+(\d+)%", l37)
        _r37_55 = re.search(r"\n\s*55\s+(\d+)%", l37)
        _t2a = re.search(r"\b21 [\d.]+ [\d.]+ (\d+)%", txt)
        _t2b = re.search(r"\b55 [\d.]+ [\d.]+ (\d+)%", txt)
        checks += 1
        if not all((_r37_21, _r37_55, _t2a, _t2b)):
            fails.append("could not read the g = 0 and Table 2 rates at three and "
                         "eleven weeks for Table S8's caption")
        else:
            want("Table S8's caption contrasts the four rates",
                 f"it reads {_r37_21.group(1)}% at three weeks and "
                 f"{_r37_55.group(1)}% at eleven where Table 2 reads "
                 f"{_t2a.group(1)}% and {_t2b.group(1)}%")
            checks += 1
            if max(abs(int(_r37_21.group(1)) - int(_t2a.group(1))),
                   abs(int(_r37_55.group(1)) - int(_t2b.group(1)))) > 1:
                fails.append("Section S12 says the g = 0 column is within a point of "
                             "Table 2's at every delay; it no longer is")
    forbid("Table S8's caption claims it reproduces Table 2",
           "same test days as Table 2")

    # the exchange rate, quoted as a list in the prose
    checks += 1
    if "3 0.13" not in l37f or "55 1.44" not in l37f:
        fails.append("lab37: the exchange-rate table no longer runs 0.13 to 1.44")
    want("8.3 exchange rate", "0.13, 0.35, 1.03, 1.13 and 1.44 days")
    checks += 1
    if "rises with the domestic delay at 4 of 4 steps" not in l37f:
        fails.append("lab37: the exchange rate no longer rises at every step, but "
                     "Section 8.3 still says it rises at every step")
    want("8.3 monotone exchange rate", "rising at every step")

    # the count of cells where staling the foreign feed hurts
    checks += 1
    if "cells where it does: 25 of 30" not in l37f:
        fails.append("lab37: the foreign-staleness cell count is no longer 25 of 30")
    want("8.3 staleness cell count", "significantly in 25 of the 30 cells")
    checks += 1
    if "of which at g = 1, a single day: 5 of 6" not in l37f:
        fails.append("lab37: a single day of foreign staleness no longer hurts at 5 of 6 "
                     "delays, but Section 8.3 still says it does")
    want("8.3 one-day count", "in 5 of the 6 delays a single day is already enough")

    # the clock test
    checks += 1
    if "the clock gap 2c and |asymmetry|: +0.933" not in l37f:
        fails.append("lab37: the clock correlation is no longer +0.933")
    want("8.3 clock correlation", "with a correlation of +0.93")
    # Row-anchored on both sides.  A bare "0.051" occurs elsewhere in this paper,
    # so a check on the token alone stayed silent when the figure was perturbed -
    # which is how this comment came to be written.
    for row, phrase in (
            ("FTSE 16.5 no 4.5 9 0.013",
             "0.013 and 0.019 for the FTSE and the DAX"),
            ("DAX 16.5 no 4.5 9 0.019",
             "four and a half hours ahead of New York"),
            ("BVSP 21.0 yes 24.0 48 0.051",
             "against 0.051 for Bovespa")):
        checks += 1
        if row not in l37f:
            fails.append(f"lab37: the clock row '{row}' is no longer printed")
        want("8.3 clock row " + row.split()[0], phrase)
    checks += 1
    if "peers that significantly LEAD the target: 0 of 7" not in l37f:
        fails.append("lab37: some peer now significantly LEADS the target, but Section "
                     "8.3 still says all seven appear to lag it")
    want("8.3 direction of the raw asymmetry", "all seven peers appear to LAG the target")

    # part C, quoted in the closing paragraphs
    checks += 1
    if "cells where it is: 26 of 30" not in l37f:
        fails.append("lab37: the lagged-factor count is no longer 26 of 30")
    want("8.3 factor lag count", "beats every lagged version in 26 of 30 cells")
    checks += 1
    if "yesterday's factor adds to today's: 2 of 6 [3, 5]" not in l37f:
        fails.append("lab37: yesterday's factor no longer adds at exactly delta 3 and 5, "
                     "but Section 8.3 still names those two delays")
    want("8.3 factor history delays",
         "does add to today's at &delta; = 3 and &delta; = 5, and at no longer delay")

    # --- Section 9 and the abstract, against labs 38, 39, 34, 35 and 11 ------
    l38 = lab("lab38_domestic_baseline")
    l39 = lab("lab39_temporal_stability")
    l34 = lab("lab34_data_snooping")
    l35 = lab("lab35_purged_cv")
    l11 = lab("lab11_markets_continuous")

    def pair(label, lab_needle, lab_text, paper_needle):
        """Both sides of a claim: the lab still prints it, the paper still says it."""
        nonlocal checks
        checks += 1
        if lab_needle not in lab_text:
            fails.append(f"{label}: lab no longer prints '{lab_needle}'")
        want(label, paper_needle)

    # 9.1 the control and the normalisation
    # lab38 prints all six arms on one line, so the LAB side is anchored on the
    # whole row and the PAPER side on the two columns Table 19 lifts from it.
    # Table S3's first two columns are lab38 part A's HAR3 and +MEAS arms.  The
    # row is located by delay and the two cells are read out of it, so a rerun
    # that moves either arm fails the paper rather than a copy kept here.  The
    # part A block is isolated first: the same delays appear again in parts C
    # and D, and a bare row needle would match whichever came first.
    _a38 = l38.split("Against HAR3")[0]
    # part C carries the rates and the paired interval; it is isolated the same
    # way, because parts A and C share every delay label.
    _c38 = read_text(os.path.join(EXP, "lab38_domestic_baseline.txt"))
    _c38 = _c38.split("R(d) +MEAS   difference")[-1].split("=" * 40)[0]
    for d in ("5", "13", "21", "55"):
        checks += 1
        m = re.search(r"\s" + d + r" (-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+) "
                      r"(-?[\d.]+) (-?[\d.]+)", _a38)
        if not m:
            fails.append(f"Table S3 delta={d}: lab38 part A no longer prints a "
                         f"six-arm row")
        else:
            har3, meas = m.group(1), m.group(5)
            want(f"Table S3 row delta={d}",
                 f"{d} {har3.replace('-', '&minus;')} {meas.replace('-', '&minus;')}")
        # Columns 4 to 7 were checked by nothing at all, and the whole interval
        # column of Table S3 sat at the nine-lag values while the paragraph
        # underneath it quoted the measured-bandwidth ones.  The table said
        # delta = 13 and 21 exclude zero; lab38 and the prose said they do not.
        # The rest of the row is read out of lab38 part C here, so the table
        # cannot disagree with the sentence that interprets it.
        checks += 1
        mc = re.search(r"\n\s+" + d + r"\s+(-?\d+)%\s+(-?\d+)%\s+(-?[\d.]+)%\s+"
                       r"\[\s*(-?[\d.]+)%,\s*([-+][\d.]+)%\]", _c38)
        if not mc:
            fails.append(f"Table S3 delta={d}: lab38 part C no longer prints a "
                         f"paired-difference row")
        else:
            _p, _c, _df, _lo, _hi = mc.groups()
            want(f"Table S3 rates and interval delta={d}",
                 f"{_p}% {_c}% {_df.replace('-', MINUS)}% "
                 f"[{_lo.replace('-', MINUS)}, {_hi.replace('-', MINUS)}]")
    # Both counts are read out of lab38 rather than transcribed, and so is the
    # LIST of delays at which the rate moves.  A section that names four delays
    # while the lab now finds two is the sort of claim a correction leaves
    # behind, because the count and the list live in different sentences.
    _cells38 = re.search(r"significantly beats the paper's: (\d+) of (\d+)", l38)
    _rate38 = re.search(r"rate moves significantly with the control: (\d+) of (\d+) "
                        r"\[([^\]]*)\]", l38)
    _lvl38 = re.search(r"contributes more than the second measure: (\d+) of (\d+)", l38)
    checks += 1
    if not (_cells38 and _rate38 and _lvl38):
        fails.append("lab38 no longer prints the counts Section 9.1 quotes")
    else:
        want("9.1 cells improved",
             f"beats the paper's in {_cells38.group(1)} of {_cells38.group(2)} cells")
        _W = {"9": "nine", "8": "eight", "7": "seven", "6": "six", "5": "five",
              "4": "four", "3": "three", "2": "two", "1": "one"}
        want("9.1 level does the work",
             f"at {_W.get(_lvl38.group(1), _lvl38.group(1))} delays out of "
             f"{_W.get(_lvl38.group(2), _lvl38.group(2))}")
        _ds = [d.strip() for d in _rate38.group(3).split(",") if d.strip()]
        _phr = (" and ".join(_ds) if len(_ds) < 3
                else ", ".join(_ds[:-1]) + " and " + _ds[-1])
        want("9.1 rate moves", f"significantly at &delta; = {_phr}")
    # lab38's part D, read rather than frozen: these two checks pinned four
    # figures from before Section 4.4's benchmark correction (-0.0351, -0.0428,
    # +0.0308, +0.0736) against a lab printing +0.0070, -0.0004, +0.0702 and
    # +0.0706.  They passed every round because they compared the paper to
    # themselves rather than to lab38.
    _D38 = re.compile(r"^\s*(\d+)\s+([-+]?[\d.]+)\s+([-+]?[\d.]+)\s+"
                      r"([-+]?[\d.]+)\s+([-+]?[\d.]+)\s+([+-][\d.]+)\s*$")
    _d38 = {}
    for _ln in read_text(os.path.join(EXP,
                                      "lab38_domestic_baseline.txt")).splitlines():
        _m = _D38.match(_ln)
        if _m:
            _d38[int(_m.group(1))] = tuple(_m.groups()[1:])
    checks += 1
    if not ({0, 55} <= set(_d38)):
        fails.append("lab38's part D no longer prints the HAR3, +VAR22 and +LEVEL "
                     "columns at zero and eleven weeks, which Section 9.1 quotes")
    else:
        def _sgn38(x):
            return x if x[0] in "+-" else "+" + x
        _h38, _v38, _lv38 = _d38[55][0], _d38[55][1], _d38[55][2]
        # part D's columns are HAR3, +VAR22, +LEVEL, +MEAS, level - var22, so
        # the margin Section 9.1 quotes is the LAST one, not the fourth.
        _marg0, _marg55 = _d38[0][4], _d38[55][4]
        want("9.1 level split",
             f"the paper's block scores {_sgn38(_h38)}, adding the second realised "
             f"measure scores {_sgn38(_v38)}, which is worse, and adding the level "
             f"alone scores {_sgn38(_lv38)}")
        want("9.1 level grows with delay",
             f"{_sgn38(_marg0)} of R&sup2; at zero delay against "
             f"{_sgn38(_marg55)} at eleven weeks")
        checks += 1
        if float(_marg55) <= float(_marg0):
            fails.append("Section 9.1 says the level's margin grows with staleness; "
                         "lab38 no longer agrees between zero and eleven weeks")
        # The pre-correction figures must not come back INTO THIS PARAGRAPH.
        # Banning them document-wide fails a correct appendix: Section S26
        # records -0.0351 by name as the value Section 4.4's correction
        # superseded, which is the paper being honest about its own history.
        _p91 = txt.split("the paper's block scores")[-1][:600] \
            if "the paper's block scores" in txt else ""
        checks += 1
        if not _p91:
            fails.append("Section 9.1's level-split paragraph could not be located")
        for _stale in ("-0.0351", "-0.0428", "+0.0308", "+0.0736"):
            if _stale in {v for row in _d38.values() for v in row}:
                continue
            checks += 1
            if _stale in _p91:
                fails.append(f"pre-correction lab38 figure {_stale} is back in "
                             f"Section 9.1's level-split paragraph")

    # 9.2 stability and the conditional rate
    pair("9.2 halves overlap", "intervals do not overlap: 0 of 4", l39,
         "77% against 67% at eleven weeks, with overlapping intervals")
    pair("9.2 within-tercile null", "differ SIGNIFICANTLY within a tercile: 0 of 12",
         l39, "not one of twelve cells differs significantly")
    want("9.2 path range", "22% in the window ending April 2019 to 91% in the window "
                           "ending April 2015")
    want("9.2 trend", "22 points a decade at five days and 12 at eleven weeks")
    for row in ("5 42%", "13 43%", "21 41%", "55 47%"):
        d = row.split()[0]
        checks += 1
        if row not in l39:
            fails.append(f"Table 20 delta={d}: lab39 no longer prints the calm rate")
    # READ from the labs, not listed here.  These were hardcoded literals, and
    # that is how the abstract came to quote a conditional rate of
    # "81% [74, 87]" that no script produced: the check pinned the document to
    # a constant instead of to a lab, so the document and the lab could drift
    # apart and the drift was invisible.  The first repair fixed that for the
    # STRESSED column and left the other two unpinned, which is how Table S4
    # kept eight pre-repair cells.  Both labs are parsed here now, all twelve
    # cells of each, and the two documents are held to the Fieller sets with
    # the bootstrap disclosed beside them.
    _tf, _tb = tercile_fieller(), tercile_boot()
    for _nm, _tt in (("lab41 Fieller", _tf), ("lab39 bootstrap", _tb)):
        checks += 1
        if sorted(_tt) != [5, 13, 21, 55] or any(
                sorted(_tt[_d]) != ["calm", "middle", "stressed"] for _d in _tt):
            fails.append(f"{_nm}: could not parse all twelve tercile cells "
                         f"({sum(len(v) for v in _tt.values())} of 12)")
    # the conditional rate the documents quote is the Fieller set, at every
    # delay and in every state.  want() searches the paper and the appendix
    # together, so this pins the value; the table itself is parsed below, so
    # that a cell cannot pass by matching some other section's copy of it.
    for _d in sorted(_tf):
        for _st in ("calm", "middle", "stressed"):
            _pt, _lo, _hi, _tau = _tf[_d][_st]
            want(f"tercile {_st} d={_d} is lab41's Fieller set",
                 f"{_pt}% [{int(_lo)}, {int(_hi)}]")
    # Table S4 parsed out of the appendix and matched cell by cell.  This is
    # the check whose absence let eight cells go stale: pinning a value
    # somewhere in the document is not the same as pinning the table.
    _appx_raw = read_text(_supp) if os.path.isfile(_supp) else ""
    _s4 = re.search(r"Table S4\.(.{0,2400}?)</table>", _appx_raw, re.S)
    checks += 1
    if not _s4:
        fails.append("Internet Appendix: Table S4 not found")
    else:
        # each state cell is "NN% [lo, hi] <span class=tau>&tau; T</span>" now,
        # so a cell is no longer tag-free: the pattern takes everything up to
        # the closing </td> and the comparison below strips the tags.  The
        # earlier tag-free pattern silently matched nothing when tau arrived,
        # which would have left the table unguarded again.
        _rows = re.findall(r"<td>(\d+)</td>\s*<td>(.+?)</td>\s*"
                           r"<td>(.+?)</td>\s*<td>(.+?)</td>",
                           _s4.group(1), re.S)
        checks += 1
        if len(_rows) != 4:
            fails.append(f"Table S4: parsed {len(_rows)} of 4 rows")
        for _r in _rows:
            _d = int(_r[0])
            for _st, _cell in zip(("calm", "middle", "stressed"), _r[1:]):
                _pt, _lo, _hi, _tau = _tf[_d][_st]
                _got = norm_iv(re.sub(r"<[^>]+>", " ", _cell)
                               .replace("&minus;", "-")
                               .replace(MINUS, "-").replace("&nbsp;", " "))
                _got = re.sub(r"\s+", " ", _got).strip()
                _need = f"{_pt}% [{int(_lo)}, {int(_hi)}] &tau; {_tau}"
                checks += 1
                if _got != _need:
                    fails.append(f"Table S4 {_st} d={_d}: says '{_got}' where "
                                 f"lab41's Fieller set is '{_need}'")
    # Section S2's Fieller-versus-bootstrap comparison: both sides read from
    # their own lab, and both gaps computed.  The sentence it replaced printed
    # ONE interval on both sides and called the identity agreement, so the
    # numbers here are pinned in pairs rather than individually: a check that
    # each value appears somewhere would have passed that sentence too.
    _fs, _bs = _tf[55]["stressed"], _tb[55]["stressed"]
    _fc, _bc = _tf[55]["calm"], _tb[55]["calm"]
    _gap_c = max(abs(int(_fc[1]) - int(_bc[1])), abs(int(_fc[2]) - int(_bc[2])))
    _gap_s = max(abs(int(_fs[1]) - int(_bs[1])), abs(int(_fs[2]) - int(_bs[2])))
    # plain() leaves &rsquo; alone, so the needle carries the entity as the
    # source does -- the two other needles in this file with a possessive do
    # the same.
    want("S2 calm pair, Fieller against bootstrap",
         f"by {_gap_c} points, [{int(_fc[1])}, {int(_fc[2])}] against the "
         f"bootstrap&rsquo;s [{int(_bc[1])}, {int(_bc[2])}]")
    want("S2 stressed pair, Fieller against bootstrap",
         f"they differ by {_gap_s}, [{int(_fs[1])}, {int(_fs[2])}] against "
         f"[{int(_bs[1])}, {int(_bs[2])}]")
    # the two sides must actually differ; that is the whole point of the
    # sentence and the thing its predecessor got wrong.
    checks += 1
    if _fs[1:3] == _bs[1:3] or _fc[1:3] == _bc[1:3]:
        fails.append("Section S2 presents the Fieller set against the bootstrap's "
                     "as different intervals, but the labs now give the same one")
    # and the caption's count of how often the Fieller set is the wider
    _wider = sum(1 for _d in _tf for _s in _tf[_d]
                 if int(_tf[_d][_s][2]) - int(_tf[_d][_s][1])
                 >= int(_tb[_d][_s][2]) - int(_tb[_d][_s][1]))
    want("Table S4 caption counts where the Fieller set is wider",
         f"at least as wide in {spell(_wider)} of the {spell(12)}")
    _worst = max(max(abs(int(_tf[_d][_s][1]) - int(_tb[_d][_s][1])),
                     abs(int(_tf[_d][_s][2]) - int(_tb[_d][_s][2])))
                 for _d in _tf for _s in _tf[_d])
    want("Table S4 caption states the largest disagreement",
         f"at one endpoint by as much as {_worst} points")
    # How often each state's set excludes zero, COUNTED.  The claim table and
    # Section S2 both said the calm column "does not exclude zero at three of
    # the four delays", which implied that at one delay it did.  It never has:
    # that sentence was read off the stale Table S4, whose delta=55 calm cell
    # said [6, 84].  A count that is written rather than counted is how a
    # corrected table leaves a stale sentence behind it.
    def _excl(st, tab):
        return sum(1 for _d in tab
                   if int(tab[_d][st][1]) > 0 or int(tab[_d][st][2]) < 0)
    _nz_calm = _excl("calm", _tf)
    checks += 1
    if _nz_calm:
        fails.append(f"the calm set now excludes zero at {_nz_calm} of four delays, "
                     f"but both documents say it does so at none")
    want("claim table counts the calm exclusions",
         "excludes zero at none of the four delays")
    want("S2 counts the calm exclusions",
         "at none of the four delays does it exclude zero")
    forbid("stale calm exclusion count", "three of four delays; where delay costs")
    forbid("S2 stale calm exclusion count",
           "at three of the four delays it does not exclude zero")
    # "larger in stressed markets" is a COMPARATIVE claim, and the tercile sets
    # overlap at every delay under this construction, so the tercile interval
    # cannot carry it.  The claim table must say what does: the nested rungs,
    # whose top and bottom intervals are disjoint.  Both read from their labs.
    _nsep = sum(1 for _d in _tf
                if int(_tf[_d]["stressed"][1]) > int(_tf[_d]["middle"][2]))
    checks += 1
    if _nsep:
        fails.append(f"the stressed set now sits entirely above the middle one at "
                     f"{_nsep} of four delays, so the claim table's appeal to the "
                     f"nested rungs is no longer necessary and should be revisited")
    want("claim table names the overlap rather than implying separation",
         "the three states&rsquo; sets overlap so the ordering rests on the nested "
         "rungs")
    forbid("claim table cites the tercile interval alone for a comparison",
           "81% [67, 90], stable across both halves of the sample (Section S10)")
    # the two nested-rung figures the claim table now leans on, READ from
    # lab27, and required to be disjoint: they are the interval evidence for
    # "larger in stressed markets" now that the tercile sets overlap, so if
    # they stopped being disjoint the grade would have nothing behind it.
    # These read lab27's part A2 - the same cells Table S7 prints - because the
    # claim table and the table it cites must not drift apart: that is exactly
    # how Table S4 and the "three of four delays" count came to disagree.
    _top = _F27[(55, "VIX top 5%")]
    _out = _F27[(55, "outside every episode")]
    checks += 1
    if not (_top and _out):
        fails.append("lab27 no longer prints the top-5% and outside-episode rungs "
                     "at eleven weeks, so the claim table's ordering evidence "
                     "rests on nothing")
    else:
        want("claim table quotes the top rung",
             f"removes {_top[0]}% [{_top[1]:.1f}, {_top[2]:.1f}] of domestic error")
        want("claim table quotes the bottom rung",
             f"against {_out[0]}% [{_out[1]:.1f}, {_out[2]:.1f}] "
             f"outside every episode")
        checks += 1
        if _top[1] <= _out[2]:
            fails.append(f"the claim table says the top and bottom rungs do not "
                         f"overlap, but lab27 now gives [{_top[1]}, {_top[2]}] and "
                         f"[{_out[1]}, {_out[2]}]")
        want("claim table says the rungs do not overlap",
             "two fixed-b sets that do not overlap")
        # and it must name the construction, so a reader cannot take these for
        # the resampled sets they replaced
        forbid("claim table calls the fixed-b sets bootstrap intervals",
               "two intervals that do not overlap")

    # and the literals that survived the first repair may not come back.
    # and the two prose literals that survived the first repair may not come
    # back.  Only these two are banned by value: the table's own cells are
    # parsed above, and the rest of the stale set ([36, 66], [8, 51] and the
    # like) are four-digit shapes that occur innocently elsewhere -- [36, 66]
    # is a window-grid cell in this same appendix -- so banning them by value
    # would be a check that cries wolf.  A ban has to be as specific as the
    # thing it bans.
    for _bad in ("[74, 87]", "53% [37, 67]"):
        forbid(f"pre-repair tercile literal {_bad}", _bad)

    # 9.3 data snooping
    pair("9.3 RC size", "Reality Check rejects at 5%: 5.5%", l34,
         "rejects 5.5% and 7.0% of the time")
    pair("9.3 survives", "Reality Check: 5 of 6 [3, 5, 13, 21, 55]", l34,
         "survives both tests at &delta; = 3, 5, 13, 21 and 55 with p below 0.0005")
    # Three claims that tamper-testing found unpinned: a ratio the paper states
    # in words, the exponential-control gap, and the note's decay figure.  Each
    # is derived from figures that ARE pinned, which is exactly the kind of
    # claim that goes stale quietly: the inputs move, the check on the inputs
    # passes, and the sentence about them does not.
    #
    # 1. "the cross-sectional model still holds 0.3736, about 73% of what was
    #    available with fully current data" -- a ratio of two Table 2 cells.
    _l10r = read_text(os.path.join(EXP, "lab10_loss_scale.txt"))
    # the R2 block, whose rows are "delta own cross ..." with no n/a column
    _c0 = re.search(r"^\s*0\s+(0\.\d+)\s+(0\.\d+)\s+0\.\d+", _l10r, re.M)
    _c55 = re.search(r"^\s*55\s+([-\d.]+)\s+(0\.\d+)\s+[-\d.]+", _l10r, re.M)
    checks += 1
    if not (_c0 and _c55):
        fails.append("lab10 no longer prints the cross-model R2 at both ends, so "
                     "Section 5's 'about 73% of what was available' rests on nothing")
    else:
        # The denominator is the DOMESTIC model at zero delay - "what was
        # available with fully current data" - not the cross model's own
        # zero-delay figure.  This check divided by the latter and the sentence
        # described the former, so both were self-consistent and the pair said
        # something false: 0.3736/0.5092 is 73% and 0.3736/0.5191 is 72%.
        _share = round(100 * float(_c55.group(2)) / float(_c0.group(1)))
        _other = round(100 * float(_c55.group(2)) / float(_c0.group(2)))
        want("5 the share of current-data skill the cross-section keeps",
             f"still holds {_c55.group(2)}, which is {_share}% of the "
             f"{_c0.group(1)} a fully current domestic mark delivered")
        want("5 names the other denominator it is not using",
             f"against the cross-sectional model's own zero-delay figure of "
             f"{_c0.group(2)} the same ratio reads {_other}%")
        checks += 1
        if _share == _other:
            fails.append("Section 5 contrasts two denominators that now give the "
                         "same share, so the contrast should be dropped")
    # 2. the exponential-control gap, from lab46's own summary line
    _l46 = read_text(os.path.join(EXP, "lab46_own_only_filter.txt"))
    _m46 = re.search(r"higher at every delay, by up to ([\d.]+)% - ([\d.]+)% at "
                     r"eleven weeks\s*against the published ([\d.]+)%",
                     " ".join(_l46.split()))
    checks += 1
    if not _m46:
        fails.append("lab46 no longer prints the exponential-control gap, so the "
                     "appendix's 'by up to 4.1 points' rests on nothing")
    else:
        want("S the exponential-control gap",
             f"higher at all five delays, by up to {_m46.group(1)} points, reading "
             f"{_m46.group(2)}% at eleven weeks where the paper's control "
             f"reads {_m46.group(3)}%")
    # A batch of prose figures a tamper sweep found unpinned.  Each was a
    # literal that a reader could check against a lab but no check did, which
    # is how the conditional rate drifted in the first place.  Every one is
    # read from its lab's own summary line.
    #
    # Table S28, the seasonal placebo, against lab56.  A tamper audit found
    # its first result cell unread: the table's whole point is that the real
    # series earns the rate and the placebo earns nothing, so both columns
    # have to be pinned or the contrast is unguarded.
    _l56 = read_text(os.path.join(EXP, "lab56_seasonal_and_breadth.txt"))
    _s28 = re.findall(r"^\s*(\d+)\s+[\d.]+\s+[\d.]+\s+(-?[\d.]+)%\s+"
                      r"\d+ of \d+", _l56, re.M)
    checks += 1
    if len(_s28) < 8:
        fails.append(f"lab56: read {len(_s28)} placebo-table rows, so Table S28 "
                     f"rests on nothing")
    else:
        for _d, _r in _s28[:8]:
            want(f"Table S28 rate d={_d} ({_r}%)", f"{_r}%")
    # Table S29, the outage decomposition, against lab60
    _l60 = read_text(os.path.join(EXP, "lab60_outage_decomposition.txt"))
    _s29 = re.findall(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
                      r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%", _l60, re.M)
    checks += 1
    if len(_s29) < 3:
        fails.append(f"lab60: read {len(_s29)} decomposition rows, so Table S29 "
                     f"rests on nothing")
    else:
        for _row in _s29:
            # the row as one needle: cell-by-cell needles passed on another
            # row's identical copy, because several of these values repeat.
            want(f"Table S29 d={_row[0]} row",
                 f"{_row[1]} {_row[2]} {_row[3]} {_row[4]} {_row[5]} "
                 f"{_row[6]} {_row[7]}%")

    # lab54's monthly rates, clean and appraised, at the two delays the
    # appendix quotes.  One of these four was wrong by a digit -- "95%" where
    # the lab says 54.6% -- and it had survived every round because nothing
    # read it.  It was also impossible on its face: the point estimate sat
    # outside its own interval, [47, 75].  A check on the pair catches that
    # shape whatever causes it, so the pair is what is demanded, and the
    # containment is asserted separately.
    _l54 = read_text(os.path.join(EXP, "lab54_monthly_feasibility.txt"))
    for _lbl, _head in (("clean", "clean monthly marks"),
                        ("appraised", "appraised monthly marks")):
        _blk = _l54.split(_head)[-1].split("=====")[0] if _head in _l54 else ""
        _r54 = {int(_m.group(1)): (_m.group(2), _m.group(3), _m.group(4))
                for _m in re.finditer(
                    r"^\s*(\d+)\s+[-\d.]+\s+[\d.]+\s+([\d.]+)%\s+"
                    r"\[\s*(-?\d+)%,\s*(-?\d+)%\]", _blk, re.M)}
        checks += 1
        if not all(_d in _r54 for _d in (3, 6)):
            fails.append(f"lab54 no longer prints the {_lbl} monthly rate at "
                         f"three and six months, so the appendix's figures for it "
                         f"rest on nothing")
            continue
        for _d in (3, 6):
            _pt, _lo, _hi = _r54[_d]
            want(f"lab54 {_lbl} rate at {_d} months",
                 f"{round(float(_pt))}% [{_lo}, {_hi}]")
            # a point estimate outside its own interval is the shape the "95%"
            # typo took; assert it cannot be written again
            checks += 1
            if not (float(_lo) <= float(_pt) <= float(_hi)):
                fails.append(f"lab54 {_lbl} at {_d} months: the point estimate "
                             f"{_pt}% is outside its own interval "
                             f"[{_lo}, {_hi}]")
    forbid("the impossible six-month pair", "at six months 95% [47, 75]")

    # Table S12, cell by cell against lab13's part C2.  The converse check
    # found it stale in the same shape as Table S4: only its delta = 55 row
    # matched a lab, because that is the row the abstract quotes and the only
    # one anybody had updated when the inference moved to fixed-b.  The other
    # four rows matched neither of lab13's tables.  All ten cells are pinned
    # now, and the caption is pinned with them.
    _l13c2 = read_text(os.path.join(EXP, "lab13_origin_median.txt"))
    _c2i = _l13c2.find("C2.")
    _s12 = list(re.finditer(
        r"^\s*(\d+)\s+(\d+)% \[\s*(\d+)%,\s*(\d+)%\]\s+"
        r"(\d+)% \[\s*(\d+)%,\s*(\d+)%\]",
        _l13c2[_c2i:] if _c2i >= 0 else "", re.M))
    checks += 1
    if len(_s12) != 5:
        fails.append(f"lab13 part C2: parsed {len(_s12)} of 5 rows, so Table S12 "
                     f"rests on nothing")
    else:
        for _m in _s12:
            _d = _m.group(1)
            want(f"Table S12 d={_d} target-dated",
                 f"{_m.group(2)}% [{_m.group(3)}, {_m.group(4)}]")
            want(f"Table S12 d={_d} origin-dated",
                 f"{_m.group(5)}% [{_m.group(6)}, {_m.group(7)}]")
            # the gap column must be the difference of the two point estimates
            _gap = int(_m.group(5)) - int(_m.group(2))
            checks += 1
            if _gap not in (-1, 0):
                fails.append(f"Table S12 d={_d}: lab13 now makes the origin-minus-"
                             f"target gap {_gap}, which the table's gap column and "
                             f"the paper's 'at most a point' claim do not allow")
    # renumber-proof, like Table 1's: the JFEc layout shifts 4.6 to 3.6
    want("Table S12 caption names the construction its cells use",
         "Intervals are the fixed-b Fieller sets of Section")
    forbid("Table S12 caption names the bootstrap for cells that are not",
           "Intervals bootstrap the whole ratio. The last column is the gap")

    # the quarterly phase table, cell by cell against lab57.  The converse
    # check found four of its intervals orphaned; the rest of the table
    # happened to be demanded elsewhere, which is exactly the pattern that
    # makes a partly-checked table look checked.
    _l57p = read_text(os.path.join(EXP, "lab57_housing_overlap.txt"))
    _p57 = list(re.finditer(
        r"^\s*(\w{3}/\w{3}/\w{3}/\w{3})\s+(\d+)\s+[\d.]+\s+[\d.]+\s+"
        r"([\d.]+)%\s+\[\s*(-?\d+)%,\s*(-?\d+)%\]\s+"
        r"([\d.]+)%\s+\[\s*(-?\d+)%,\s*(-?\d+)%\]", _l57p, re.M))
    checks += 1
    if len(_p57) != 6:
        fails.append(f"lab57: parsed {len(_p57)} of 6 phase rows, so the "
                     f"appendix's quarterly phase table rests on nothing")
    else:
        for _m in _p57:
            _ph, _dq = _m.group(1), _m.group(2)
            want(f"phase table {_ph} d={_dq} mean rate",
                 f"{_m.group(3)}% [{_m.group(4)}, {_m.group(5)}]")
            want(f"phase table {_ph} d={_dq} pooled rate",
                 f"{_m.group(6)}% [{_m.group(7)}, {_m.group(8)}]")

    # the quarterly housing rate and its count, from lab57's own summary line.
    # The paper states 57% in two places and 35 of 42 beside it; the count was
    # pinned and the rate was not.
    _l57q = " ".join(read_text(os.path.join(EXP,
                                            "lab57_housing_overlap.txt")).split())
    _m57q = re.search(r"delta = 1 quarter : phase mean of 'mean R'\s+[\d.]+%; "
                      r"of 'pooled R'\s+([\d.]+)%, phases spanning "
                      r"\[[\d.]+%, [\d.]+%\]; positive in (\d+) of (\d+)", _l57q)
    checks += 1
    if not _m57q:
        fails.append("lab57 no longer prints the pooled quarterly rate, so the "
                     "paper's 57% at one quarter rests on nothing")
    else:
        want("the quarterly housing rate",
             f"the rate is {round(float(_m57q.group(1)))}% at one quarter of "
             f"staleness and positive in {_m57q.group(2)} of {_m57q.group(3)} "
             f"metro-phases")
    # the horizon series the appendix quotes, read across lab17's horizon
    # BLOCKS at the headline delay -- not down the delays inside one block,
    # which is what a first attempt at this check did and which produced a
    # plausible-looking series belonging to a different comparison.
    _l17h = read_text(os.path.join(EXP, "lab17_horizons.txt"))
    _hz = []
    for _blk in re.split(r"--- h = ", _l17h)[1:]:
        _hn = re.match(r"(\d+)", _blk)
        _r55 = re.search(r"^\s*55\s+[-\d.]+\s+[\d.]+\s+[+-][\d.]+\s+(\d+)%\s+"
                         r"\[[^\]]*\]\s+(\d+)%", _blk, re.M)
        if _hn and _r55:
            _hz.append((int(_hn.group(1)), _r55.group(1), _r55.group(2)))
    checks += 1
    if len(_hz) < 4:
        fails.append(f"lab17: read {len(_hz)} horizon blocks at the headline delay, "
                     f"so the appendix's horizon series rests on nothing")
    else:
        _hz.sort()
        want("the QLIKE horizon series",
             ", ".join(f"{q}%" for _h, _r, q in _hz[:4]))
        # and the R2 band it calls flat over the first three horizons
        want("the horizon band called flat",
             "flat at " + ", ".join(f"{r}%" for _h, r, _q in _hz[:2])
             + f" and {_hz[2][1]}%")

    # ---- the last nine prose figures no needle mentioned.  Found by dumping
    # every needle the verifier demands and differencing it against every
    # percentage in the documents, which is a stronger instrument than a
    # sampled tamper sweep: it is exhaustive.
    #
    # the peer-count line fit (Section 5)
    _l07f = read_text(os.path.join(EXP, "lab07_estimation_cost.txt"))
    _m07f = re.search(r"line through the origin: dR2 = (-[\d.]+) per peer, "
                      r"fraction of variation explained ([\d.]+)", _l07f)
    checks += 1
    if not _m07f:
        fails.append("lab07 no longer prints the peer-count line fit")
    else:
        want("5 the peer-count line's explained fraction",
             f"accounts for {round(float(_m07f.group(2)) * 100)}% of the variation")
    # what estimation consumes at three weeks (Section 5)
    _m07e = re.search(r"At delta = (\d+) estimation eats (\d+)% of what the", _l07f)
    checks += 1
    if not _m07e:
        fails.append("lab07 no longer prints what estimation consumes")
    else:
        want("5 what estimation consumes",
             f"estimation consumes {_m07e.group(2)}% of what the cross-section")
    # the real-time first component's share (Section 6)
    _l22r = read_text(os.path.join(EXP, "lab22_factor_benchmark.txt"))
    _m22r = re.search(r"d=0: ([\d.]+)%", _l22r)
    checks += 1
    if not _m22r:
        fails.append("lab22 no longer prints the real-time component share")
    else:
        want("6 the real-time first component's share",
             f"component explains {_m22r.group(1)}% of the block")
    # the same-day-VIX substitution rate the appendix rejects (Internet Appendix)
    _l08r = read_text(os.path.join(EXP, "lab08_implied_vol.txt"))
    # the SECOND rate table: lab08 prints one for the admissible lagged VIX and
    # one for the inadmissible same-day close, with identical row shapes.  An
    # unscoped search takes the first and the appendix is quoting the second.
    _blocks08 = re.split(r"^ delta  R\(d\) foreign", _l08r, flags=re.M)
    _m08r = re.search(r"^\s*21\s+\d+%\s+(\d+)%", _blocks08[2], re.M) \
        if len(_blocks08) > 2 else None
    checks += 1
    if not _m08r:
        fails.append("lab08 no longer prints the delta=21 implied-volatility rate")
    else:
        want("S the inadmissible same-day rate",
             f"a substitution rate of {_m08r.group(1)}%")
    # the carry-forward frequency and which peer carries it (Internet Appendix).
    # lab03 prints a DIFFERENT quantity -- missing days over the whole panel,
    # where N225 is 6.4% and FTSE the largest -- so this reads lab59, which
    # measures carry-forward over the target's own test days.  The two look
    # alike and are not, which is a good reason to pin the sentence to the one
    # it actually came from.
    _l59 = read_text(os.path.join(EXP, "lab59_session_timestamps.txt"))
    _m59 = re.search(r"the peer carried forward most often is (\w+) on "
                     r"([\d.]+)% of the target's days", _l59)
    checks += 1
    if not _m59:
        fails.append("lab59 no longer prints the carry-forward frequency")
    else:
        want("S the carry-forward frequency",
             f"carried forward most often is {_m59.group(1)}, on "
             f"{_m59.group(2)}% of the target's days")
    # the identity between the prediction and the measurement, both computed
    # from lab10's three R2 values rather than transcribed to two decimals
    _l10id = read_text(os.path.join(EXP, "lab10_loss_scale.txt"))
    _r0 = re.search(r"^\s*0\s+(0\.\d+)\s+(0\.\d+)\s+0\.\d+", _l10id, re.M)
    _r55 = re.search(r"^\s*55\s+([-\d.]+)\s+(0\.\d+)\s+[-\d.]+", _l10id, re.M)
    checks += 1
    if not (_r0 and _r55):
        fails.append("lab10 no longer prints the R2 pair the identity uses")
    else:
        # S own (0) is the FIRST column of lab10's row 0; the second is the
        # cross model, and using it put the identity 1.4 points out.
        _s0, _sc55, _so55 = (float(_r0.group(1)), float(_r55.group(2)),
                             float(_r55.group(1)))
        _pred = 100 * _sc55 / _s0
        _meas = 100 * (_sc55 - _so55) / (_s0 - _so55)
        want("S the identity's two figures",
             f"gap between {_pred:.2f}% and {_meas:.2f}%")
        checks += 1
        if round(_pred - _meas, 1) != 0.4:
            fails.append(f"the appendix calls the gap 0.4 points; lab10 now makes "
                         f"it {_pred - _meas:.2f}")

    # ---- the claim table's own figures, which a tamper sweep found unpinned.
    # It is the table a referee reads hardest and it was the least checked.
    #
    # the specification band, checked against Table 1's own rate column: the
    # band must be the rounded min and max of the specifications it summarises,
    # so it cannot survive a row moving.
    _t1 = re.search(r"Table 1\.(.{0,6000}?)</table>", read_text(path), re.S)
    checks += 1
    if not _t1:
        fails.append("Table 1 not found, so the specification band Sections 1 and "
                     "12 quote cannot be checked against it")
    else:
        # Rows are taken by their LABEL, not by their position.  Reading rows
        # two to five worked until the table gained its middle-tercile and calm
        # rows, at which point a positional check reported that the band "rests
        # on nothing" - of a table that had just been made more complete.  The
        # band is about four named specifications; naming them is what the
        # check means and is what cannot drift.
        _SPECS = ("strongest domestic control", "equal-weighted cross-section",
                  "one real-time factor", "state-space alternative")
        _t1rows = {}
        for _m in re.finditer(r"<tr[^>]*><td>([^<]*)</td><td>([^<]*)</td>",
                              _t1.group(1)):
            _v = re.match(r"\s*([\d.]+)%", _m.group(2))
            if _v:
                _t1rows[" ".join(_m.group(1).split())] = float(_v.group(1))
        checks += 1
        if not all(_k in _t1rows for _k in _SPECS):
            fails.append(f"Table 1: the four specification rows the band "
                         f"summarises are not all there; found "
                         f"{sorted(_t1rows)}")
        else:
            _mid = [_t1rows[_k] for _k in _SPECS]
            _blo, _bhi = round(min(_mid)), round(max(_mid))
            want("claim table's specification band",
                 f"{_blo}% to {_bhi}% across four specifications")
            want("1 states the same band",
                 f"all landing between {_blo}% and {_bhi}%")
    # the stressed-against-middle pair the claim table now quotes
    _cf = tercile_fieller()
    want("claim table's stressed-against-middle pair",
         f"{_cf[55]['stressed'][0]}% [{int(_cf[55]['stressed'][1])}, "
         f"{int(_cf[55]['stressed'][2])}] against {_cf[55]['middle'][0]}% "
         f"[{int(_cf[55]['middle'][1])}, {int(_cf[55]['middle'][2])}]")
    # the coupling ordering's worst leave-one-out RANK correlation.  lab53
    # prints a Pearson column beside the Spearman one and the paper's claim is
    # about the ordering, so it is the Spearman floor that has to be read; the
    # Pearson worst is a different number in the same table.
    _l53c = read_text(os.path.join(EXP, "lab53_no_options_targets.txt"))
    _sp = [float(_m.group(2)) for _m in re.finditer(
        r"^\s+\w+\s+([+-][\d.]+)\s+([+-][\d.]+)\s+[+-][\d.]+\s+[+-][\d.]+"
        r"\s+[\d.]+\s*$", _l53c, re.M)]
    checks += 1
    if len(_sp) < 8:
        fails.append(f"lab53: read {len(_sp)} leave-one-out rows, so the claim "
                     f"table's worst leave-one-out rests on nothing")
    else:
        want("claim table's worst leave-one-out rank correlation",
             f"worst leave-one-out +{min(_sp):.3f}")
    # the de-smoothed housing count, at the quarterly cycle
    _l57 = read_text(os.path.join(EXP, "lab57_housing_overlap.txt"))
    _m57 = re.search(r"^\s*3\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[+-][\d.]+\s+"
                     r"(\d+) of (\d+)\s*$", _l57, re.M)
    checks += 1
    if not _m57:
        fails.append("lab57 no longer prints the de-smoothed count at three "
                     "quarters, so the claim table's '14 of 14' rests on nothing")
    else:
        want("claim table's de-smoothed count",
             f"still adds at three quarters on {_m57.group(1)} of "
             f"{_m57.group(2)} metros")
    # the implied-volatility gross bound: the largest upper endpoint lab08's
    # gross interval reaches across the delays it runs
    _l08 = read_text(os.path.join(EXP, "lab08_implied_vol.txt"))
    _g08 = [float(_m.group(1)) for _m in re.finditer(
        r"\[[-+][\d.]+,\+([\d.]+)\]\s+redundant", _l08)]
    checks += 1
    if len(_g08) < 3:
        fails.append(f"lab08: read {len(_g08)} gross intervals, so the claim "
                     f"table's ruled-out gross gain rests on nothing")
    else:
        want("claim table's ruled-out gross gain",
             f"gross gain above {max(_g08):.3f} of R&sup2; ruled out")

    # Table 2's own cells, row by row, against lab10.  The converse check found
    # all twenty-seven of them orphaned: the paper's MAIN results table had no
    # check reading its interval columns, only its surrounding prose.  That is
    # the Table S4 situation in the table a reader looks at first, and it
    # survived because the prose figures it quotes ARE checked, so nothing
    # noticed that the table they summarise was not.
    # Scoped to part 1a, the inverted sets, for the reason the `rates` block
    # above gives: part 1's bootstrap rows have the same shape.
    _l10t2 = read_text(os.path.join(EXP, "lab10_loss_scale.txt"))
    _l10fb = _l10t2.split("1a. THE SAME RATES, INVERTED")[-1]
    _t2 = [(_m.group(1), _m.groups()[1:]) for _m in re.finditer(
        r"^\s*(\d+)\s+[\d.]+\s+(-?\d+)% \[\s*(-?\d+)%,\s*(-?\d+)%\]\s+"
        r"(-?\d+)% \[\s*(-?\d+)%,\s*(-?\d+)%\]", _l10fb, re.M)]
    checks += 1
    if len(_t2) != 8:
        fails.append(f"lab10: parsed {len(_t2)} of 8 rate rows, so Table 2's "
                     f"interval columns rest on nothing")
    else:
        for _d, _g in _t2:
            want(f"Table 2 d={_d} R2 rate is the inverted set",
                 f"{_g[0]}% [{_g[1]}, {_g[2]}]")
            want(f"Table 2 d={_d} QLIKE rate is the inverted set",
                 f"{_g[3]}% [{_g[4]}, {_g[5]}]")

    # the QLIKE range Section 5 quotes across the delay grid, from lab10's
    # own QLIKE column rather than from the two ends written by hand
    _l10q = read_text(os.path.join(EXP, "lab10_loss_scale.txt"))
    _qr = [int(_m.group(1)) for _m in
           re.finditer(r"^\s*\d+\s+\d+% \[[^\]]*\]\s+(\d+)% \[", _l10q, re.M)]
    checks += 1
    if len(_qr) < 5:
        fails.append(f"lab10: read {len(_qr)} QLIKE rates, so Section 5's "
                     f"'40% to 80%' range rests on nothing")
    else:
        _qlo = min(x for x in _qr if x > 0)
        want("5 the QLIKE range across the grid",
             f"larger numbers, {_qlo}% to {max(_qr)}%")
    # the monthly peer-block rates Section S quotes as the negative case
    _l55 = read_text(os.path.join(EXP, "lab55_illiquid_measured.txt"))
    # scoped to the right table: lab55 prints TWO rows labelled "every peer
    # separately", for two different quantities, and the first one matched.
    # The paper quotes the second, under the "what the rates would have been"
    # heading, so the search starts there.
    _l55b = _l55.split("What the rates would have been under each block")[-1]
    _m55 = re.search(r"^\s*every peer separately\s+(-?[\d.]+)%\s+(-?[\d.]+)%"
                     r"\s+(-?[\d.]+)%", _l55b, re.M)
    checks += 1
    if not _m55:
        fails.append("lab55 no longer prints the widest-block monthly rates")
    else:
        # plain() has already turned &minus; into "-", so the needle carries
        # the plain sign; converting it to the entity matched nothing.
        want("S the widest block's monthly rates",
             f"at {_m55.group(1)}%, {_m55.group(2)}% and {_m55.group(3)}%")
    # the horizon band Section S quotes as flat
    _l17 = read_text(os.path.join(EXP, "lab17_horizons.txt"))
    _h17 = [int(_m.group(1)) for _m in
            re.finditer(r"^\s*\d+\s+[-\d.]+\s+[\d.]+\s+[+-][\d.]+\s+(\d+)%",
                        _l17, re.M)]
    checks += 1
    if len(_h17) < 4:
        fails.append(f"lab17: read {len(_h17)} horizon rates, so the appendix's "
                     f"flat-band claim rests on nothing")

    # the principal-component variance shares (Section 6)
    _l22 = " ".join(read_text(os.path.join(EXP,
                                           "lab22_factor_benchmark.txt")).split())
    _mpc = re.search(r"first component ([\d.]+)%, first two ([\d.]+)%", _l22)
    checks += 1
    if not _mpc:
        fails.append("lab22 no longer prints the variance shares Section 6 quotes")
    else:
        want("6 the first component's share",
             f"first principal component accounting for {_mpc.group(1)}% of the "
             f"variation and its first two for {_mpc.group(2)}%")
    # the filtered-headline comparison (Internet Appendix)
    _l50 = " ".join(read_text(os.path.join(EXP,
                                           "lab50_filtered_headline.txt")).split())
    _m50 = re.search(r"([\d.]+)% against ([\d.]+)% at eleven weeks, a paired "
                     r"([+-][\d.]+)% \[([+-][\d.]+)%, ?([+-][\d.]+)%\]", _l50)
    checks += 1
    if not _m50:
        fails.append("lab50 no longer prints the filtered-headline pair")
    else:
        want("S the filtered headline and its paired difference",
             f"becomes {_m50.group(1)}% against the published {_m50.group(2)}%, a "
             f"paired difference of {_m50.group(3)}% [{_m50.group(4)}%, "
             f"{_m50.group(5)}%]")
    # the control-envelope comparison (Internet Appendix)
    _l49 = " ".join(read_text(os.path.join(EXP,
                                           "lab49_control_envelope.txt")).split())
    _m49 = re.search(r"([\d.]+)%, a paired difference of ([+-][\d.]+)% "
                     r"\[([+-][\d.]+)%, ?([+-][\d.]+)%\]", _l49)
    checks += 1
    if not _m49:
        fails.append("lab49 no longer prints the control-envelope pair")
    else:
        want("S the control-envelope rate and its paired difference",
             f"the rate inside it is {_m49.group(1)}%, a paired difference of "
             f"{_m49.group(2)}%")
    want("9.3 delta-0 absence", "the best margin across all nine models is +0.0014")
    # The availability statement says how many scripts a reader without the
    # pre-approved Case-Shiller panel can run.  Both numbers are counted from
    # the labs folder and from which labs actually read that file, so the
    # sentence cannot drift as labs are added: it said "all sixty-seven" with
    # no qualification while three of them needed an input the reader may not
    # have, which is the kind of statement a referee tests by trying it.
    _labs_all = sorted(f for f in os.listdir(os.path.join(HERE, "labs"))
                       if re.fullmatch(r"lab\d+[a-z]?_.*\.py", f))
    _needs_cs = set()
    for _f in _labs_all:
        _s = read_text(os.path.join(HERE, "labs", _f))
        if "metro20" in _s:
            _needs_cs.add(_f[:-3])
    # plus anything importing one of those
    for _f in _labs_all:
        _s = read_text(os.path.join(HERE, "labs", _f))
        if any(re.search(r"\bimport\s+" + re.escape(_m), _s) for _m in list(_needs_cs)):
            _needs_cs.add(_f[:-3])
    checks += 1
    if not _needs_cs:
        fails.append("no lab reads the Case-Shiller metro panel any more, but the "
                     "availability statement says three do")
    else:
        want("availability names the restricted-input count",
             f"{spell(len(_needs_cs))} of the {spell(len(_labs_all))} read the "
             f"twenty-metro")
        want("availability names what a reader without it can run",
             f"runs {spell(len(_labs_all) - len(_needs_cs))} and is told which "
             f"{spell(len(_needs_cs))} were")
        forbid("availability claims an unqualified full rerun",
               f"reruns all {spell(len(_labs_all))} scripts under a fixed seed and "
               f"compares each line against the stored output; and")

    # 9.4 purging
    pair("9.4 purge load-bearing", "flatters the result: 2 of 6 [0, 3]", l35,
         "+0.0024 [+0.00062, +0.00387] at zero delay and +0.0021 "
         "[+0.00017, +0.00382] at three days")
    pair("9.4 embargo", "embargo significantly reduces skill: 0 of 18", l35,
         "costs nothing detectable in any of eighteen cells")

    # 9.5 five markets, every row anchored to lab11
    # These four intervals per market were hard-coded here for most of this
    # project's life, which meant the check could only ever confirm that the
    # paper still said what it said.  Widening the bootstrap block moved every
    # one of them and the literals did not follow, so they are now parsed out of
    # lab11's own output: the check fails when the paper and the lab disagree,
    # which is the only thing it was ever supposed to test.
    _l11raw = read_text(os.path.join(EXP, "lab11_markets_continuous.txt"))
    _mkt, _cur = {}, None
    for _ln in _l11raw.splitlines():
        _h = re.match(r"--- target (\w+) ", _ln)
        if _h:
            _cur = _h.group(1); _mkt[_cur] = {}
        elif _cur:
            # The FIXED-B column, not the bootstrap one.  lab11 prints both
            # side by side - "R(d) R2" is the resampled set and "fixed-b R2"
            # the inverted one - and Table 21 carries the inverted one, like
            # every other rate table in this paper.  Reading the first pair on
            # the line takes the bootstrap's and fails a correct table.
            _r = re.match(r"\s*(\d+)\s+-?[\d.]+\s+-?[\d.]+\s+"
                          r"\d+% \[\s*-?\d+%,\s*-?\d+%\]\s+"
                          r"\d+% \[\s*-?\d+%,\s*-?\d+%\]\s+[\d.]+\s+"
                          r"(\d+)% \[\s*(-?\d+)%,\s*(-?\d+)%\]", _ln)
            if _r:
                _mkt[_cur][int(_r.group(1))] = (f"{_r.group(2)}% [{_r.group(3)}, "
                                                f"{_r.group(4)}]")
    checks += 1
    if any(len(_mkt.get(m, {})) < 4 for m in ("SPX", "FTSE", "DAX", "HSI", "N225")):
        fails.append("lab11 no longer prints four delays for all five markets, "
                     "which Table 21 is built from")
        _mkt = {m: {d: "" for d in (5, 13, 21, 55)} for m in
                ("SPX", "FTSE", "DAX", "HSI", "N225")}
    for mk, r5, r13, r21, r55 in (
            (m, _mkt[m][5], _mkt[m][13], _mkt[m][21], _mkt[m][55])
            for m in ("SPX", "FTSE", "DAX", "HSI", "N225")):
        # Anchored on the WHOLE row: "60% [45, 73]" alone also appears in Table 20,
        # so a per-cell needle passed a deliberately corrupted Table 20 row.
        checks += 1
        pt, rng = r55.split("% [")
        lo, hi = rng.rstrip("]").split(", ")
        if (f"{pt}% [ {lo}%, {hi}%]" not in l11
                and f"{pt}% [{lo}%, {hi}%]" not in l11
                and f"{pt}% [ {lo}%,{hi}%]" not in l11):   # lab11's own spacing
            fails.append(f"Table 21 {mk}: lab11 no longer prints {r55}")
        want(f"Table 21 row {mk}", f"{r5} {r13} {r21} {r55}")

    # --- which tables may still claim a bootstrap interval ----------------
    # Section 4.6 names the exceptions instead of calling them "the secondary
    # grids", so the list is checkable and is checked: the captions are read
    # out of the two documents and compared against it.  Table S4, S6, S12 and
    # S18 each went stale by being converted in the row the abstract quotes
    # and left alone everywhere else, and in every case a caption went on
    # naming a construction its cells no longer used.
    _BOOT_OK = {"S3", "S14", "S24", "S25"}
    _capboot = set()
    _rawP = read_text(path)
    _rawS = read_text(_supp) if _supp and os.path.isfile(_supp) else ""
    for _src in (path, _supp):
        if not _src or not os.path.isfile(_src):
            continue
        _doc = read_text(_src)
        for _cm in re.finditer(r"<b>Table (S?\d+)\.</b>(.*?)</caption>", _doc, re.S):
            _body = plain_text_of(_cm.group(2))
            # Only tables that actually PRINT an interval are in scope.  Table
            # 3 names a bootstrap in its caption because a significance
            # statement in the prose comes from one, and has no interval
            # column at all; flagging it would be the guard crying wolf.
            _full = re.search(re.escape(_cm.group(0)) + r"(.*?)</table>", _doc, re.S)
            _cells = plain_text_of(_full.group(1)) if _full else ""
            if not re.search(r"\[\s*[-+\u2212]?[\d.]+\s*,", _cells):
                continue
            if re.search(r"bootstrap|resampl", _body, re.I) and not re.search(
                    r"fixed-b|Fieller", _body, re.I):
                _capboot.add(_cm.group(1))
    checks += 1
    if _capboot - _BOOT_OK:
        fails.append(f"Section 4.6 names Tables {', '.join(sorted(_BOOT_OK))} as the "
                     f"only tables still on the bootstrap, but these captions "
                     f"claim a resampled interval and no fixed-b one: "
                     f"{', '.join(sorted(_capboot - _BOOT_OK))}")
    checks += 1
    if _BOOT_OK - _capboot:
        fails.append(f"Section 4.6 says Tables {', '.join(sorted(_BOOT_OK - _capboot))} "
                     f"keep the bootstrap, but their captions no longer say so")
    want("4.6 names the three families that keep the bootstrap",
         "Three families keep it, each for a stated reason")
    want("4.6 gives the reason Table S3 keeps it",
         "which is a difference of two ratios and not itself a ratio of two means")
    want("4.6 gives the reason the monthly tables keep it",
         f"Tables {', '.join(sorted(_BOOT_OK - {'S3'}, key=lambda x: int(x[1:])))} "
         f"are run at a different frequency on fewer than five hundred observations"
         .replace("S14, S24, S25", "S14, S24 and S25"))
    want("4.6 says everything else is inverted",
         "including the market-state ladder of Table S7, is a fixed-b Fieller set")
    forbid("4.6 still calls the exceptions the secondary grids",
           "the same holds for the secondary grids of the Internet Appendix")
    # the two captions the conversion rewrote, which must name the construction
    for _tb in ("S6", "S18"):
        want(f"Table {_tb} caption names the construction its cells use",
             "Intervals are the fixed-b Fieller sets of Section")
    want("Table S18 states why the five-day row reproduces the headline",
         "reproduces the headline&rsquo;s endpoints and not merely its point "
         "estimate")

    # --- cross-references, checked against what the cited section holds ---
    # A cold read found ten citations pointing at the wrong section, several to
    # a section on an unrelated subject.  None of them is a figure, so nothing
    # in this file looked at them.  The guard is a table of claims whose
    # subject is distinctive enough to locate by text, and it LOCATES the
    # subject rather than comparing against a number written here, so it
    # survives the JFEc variant's renumbering.
    _secs, _psecs = {}, {}
    _heads = [(m.group(1), m.start()) for m in
              re.finditer(r"<h[23][^>]*>(S\d+(?:\.\d+)?)\.", _rawS)]
    for _i, (_nm, _at) in enumerate(_heads):
        _end = _heads[_i + 1][1] if _i + 1 < len(_heads) else len(_rawS)
        _secs[_nm] = plain_text_of(_rawS[_at:_end]).lower()
    _ph = [(m.group(1), m.start()) for m in
           re.finditer(r"<h2[^>]*>(\d+)\.\s", _rawP)]
    for _i, (_nm, _at) in enumerate(_ph):
        _end = _ph[_i + 1][1] if _i + 1 < len(_ph) else len(_rawP)
        _psecs[_nm] = plain_text_of(_rawP[_at:_end]).lower()
    checks += 1
    if len(_secs) < 30 or len(_psecs) < 8:
        fails.append(f"could not split the documents into sections for the "
                     f"cross-reference guard ({len(_secs)} appendix, "
                     f"{len(_psecs)} paper)")
    _XREF = [
        (r"coupling result of Section (\d+)", "paper",
         "the substitution rate against coupling"),
        (r"nested rungs of Section (S\d+)", "supp", "rungs are nested"),
        (r"bandwidth Section (S\d+) measures", "supp",
         "the automatic bandwidth returns 0.0495"),
        (r"Section (\d+) reports that a target's coupling", "paper",
         "the substitution rate against coupling"),
        (r"the factor model in Section (S\d+)", "supp",
         "six real-time components orthogonal"),
        (r"Section (S\d+) prices the search behind the headline", "supp",
         "size of both first validated by simulation"),
        (r"multiplicity is priced in Section (S\d+)", "supp",
         "size of both first validated by simulation"),
        (r"carries its own correction in Section (S\d+)", "supp",
         "size of both first validated by simulation"),
        (r"non-linearity result of Section (S\d+)", "supp",
         "rff(60) is below rff(20)"),
    ]
    for _pat, _which, _subject in _XREF:
        _found = re.findall(_pat, _rawP + " " + _rawS)
        checks += 1
        if not _found:
            fails.append(f"the cross-reference guard's pattern {_pat!r} matches "
                         f"nothing, so it is a check that cannot fail")
            continue
        _map = _psecs if _which == "paper" else _secs
        _holders = [_k for _k, _v in _map.items() if _subject.lower() in _v]
        checks += 1
        if len(_holders) != 1:
            fails.append(f"the cross-reference guard cannot locate '{_subject}': "
                         f"{len(_holders)} sections contain it ({_holders})")
            continue
        _should = _holders[0]
        for _cited in set(_found):
            checks += 1
            _norm = _cited.lstrip("S") if _which == "paper" else _cited
            if _norm != _should:
                fails.append(f"a citation sends the reader to Section {_cited} for "
                             f"'{_subject}', which is in Section {_should}")
    # A TABLE citation is checked against that table's own caption, not against
    # a section: the appendix's table numbers and section numbers do not run
    # together - Table S22 sits inside Section S28.
    _TREF = [(r"seven-regressor column is Table (S\d+)", "four information sets"),
             (r"the negative in Table (S\d+) was a bill", "four information sets")]
    _caps = {_m.group(1): plain_text_of(_m.group(2)).lower() for _m in
             re.finditer(r"<b>Table (S?\d+)\.</b>(.*?)</caption>", _rawP + _rawS,
                         re.S)}
    for _pat, _subject in _TREF:
        _found = re.findall(_pat, _rawP + " " + _rawS)
        checks += 1
        if not _found:
            fails.append(f"the table-reference guard's pattern {_pat!r} matches "
                         f"nothing, so it is a check that cannot fail")
            continue
        for _cited in set(_found):
            checks += 1
            if _cited not in _caps:
                fails.append(f"a citation points at Table {_cited}, which no "
                             f"caption in either document defines")
            elif _subject not in _caps[_cited]:
                fails.append(f"a citation sends the reader to Table {_cited} for "
                             f"'{_subject}', which is not what that caption says")

    # 9.1's normalisation test, against lab40
    l40 = lab("lab40_level_or_normaliser")
    # Derived from lab40's own output rather than transcribed, so a rerun that
    # moves the share or the rate correction fails the paper instead of
    # agreeing with a stale copy kept here.
    _m40 = re.search(r"the implementable target keeps between (\d+)% and (\d+)% "
                     r"of the gain", l40)
    checks += 1
    if not _m40:
        fails.append("lab40 no longer prints the share the implementable target keeps")
    else:
        want("9.1 normaliser test",
             f"retains between {_m40.group(1)}% and {_m40.group(2)}% of the gain")
    # the eleven-week gains on the paper's target and on the implementable one
    _g40 = re.search(r"55 \+?(-?[\d.]+) \+?(-?[\d.]+) (\d+)%\s", l40)
    checks += 1
    if not _g40:
        fails.append("lab40 no longer prints the delta=55 gain comparison")
    else:
        want("9.1 normaliser numbers",
             f"+{_g40.group(2)} against +{_g40.group(1)} at eleven weeks")
    _r40 = re.search(r"C t-delta 55 (\d+)% (\d+)%", l40)
    checks += 1
    if not _r40:
        fails.append("lab40 no longer prints the rate correction on target C")
    else:
        want("9.1 correction survives",
             f"moves R(55) from {_r40.group(1)}% to {_r40.group(2)}%")

    # the conclusions must quote what Section 9 corrected, not what it replaced
    # 9.1 reports the control as a second specification, not a retraction; the
    # paper must not go back to telling the reader its own tables are wrong.
    forbid("10 asks the reader to correct the tables",
           "should subtract roughly four points from ours")
    # Three states, not two.  The analysis cuts on a rolling VIX tercile and the
    # middle one is resolved at 53% [37, 67], so "averages two states" collapsed a
    # regime the paper measures.  The needle now pins the three-state form, which
    # is also what stops the two-state phrasing coming back.
    want("12 conditional limitation",
         "The unconditional estimate pools three market states in which the "
         "mechanism behaves differently")
    forbid("stale two-state limitation",
           "The unconditional rate averages two states")
    # Anchored on Section 13's own wording: the bare interval also appears in
    # 9.2, so a needle on it alone would pass a Section 13 that had gone stale.
    # "tight" was fair of the bootstrap's 18-point set and is not fair of the
    # Fieller set that replaced it, which is 23 points wide.  What is true
    # of it, and checkable, is that the stressed state is the narrowest of
    # the three, so that is what the sentence claims and this pins.
    want("13 conditional recommendation",
         "about 80% of the delay-induced loss recovered in stressed markets, where "
         "the estimate is stable and the narrowest of the three states")
    forbid("13 stale tightness claim", "stable and tight")
    # and it must BE the narrowest, on lab41's own sets.
    _w55 = {_s: int(_tf[55][_s][2]) - int(_tf[55][_s][1])
            for _s in ("calm", "middle", "stressed")}
    checks += 1
    if min(_w55, key=_w55.get) != "stressed":
        fails.append(f"Section 13 calls the stressed set the narrowest of the "
                     f"three; lab41 now makes it {_w55}")
    # the calm set is what the prose calls "more than a hundred points"
    checks += 1
    if _w55["calm"] <= 100:
        fails.append(f"Sections 1 and 9.2 say the calm set spans more than a "
                     f"hundred points; lab41 now spans {_w55['calm']}")
    want("9.2 calm span", "spans more than a hundred points")
    forbid("9.2 stale calm span", "spans ninety points")
    # the three sets overlap at every delay under the Fieller construction,
    # where the bootstrap separated stressed from middle at two delays.  The
    # documents must not claim the separation the new sets do not show.
    _sep = sum(1 for _d in _tf
               if int(_tf[_d]["stressed"][1]) > int(_tf[_d]["middle"][2]))
    checks += 1
    if _sep:
        fails.append(f"lab41 now separates stressed from middle at {_sep} of 4 "
                     f"delays, but both documents say the sets overlap at every one")
    want("tercile sets overlap", "overlap at every delay")
    forbid("stale tercile separation claim",
           "resolved and materially lower than the stressed figure")
    want("13 names both unconditional figures",
         "72% against the paper's own control and 68% against the stronger control")
    forbid("13 stale unconditional advice",
           "around 70% by eleven weeks, against a cost of one to five points")
    # 4.5 states the identification problem and names the construction that
    # answers it; both halves are checked, because the first without the second
    # is a caveat with no remedy attached.
    want("4.5 states the identification problem",
         "approaches zero and the ratio is weakly identified, so a uniformly "
         "valid confidence procedure must be free to return an unbounded set "
         "(Dufour 1997)")
    # Two constructions now coexist and the paper must not blur them.  The
    # abstract's rate quotes the fixed-b Fieller set (lab13 part C2); the delay
    # and specification tables still carry the block bootstrap's interval
    # columns.  An earlier draft said the bootstrap was the source of every
    # published interval, which the abstract contradicts, and the draft before
    # that said Fieller produced them all, which the tables contradict.  So the
    # claim is pinned three ways: the split must be stated, and neither
    # single-source sentence may come back.
    # Section 4.5 now states ONE construction for every reported rate, because
    # the tables were converted; the sentence that named the split is gone and
    # must not come back, and the remaining bootstrap families are named in 4.6.
    want("4.5 names one construction for every reported rate",
         "Every interval this paper reports for the rate inverts a HAC t-test "
         "after Fieller (1954)")
    want("4.5 says where the bootstrap still appears and why",
         "it is reported in the tables only where the quantity is not a ratio of "
         "two means")
    forbid("4.5 still describes a two-construction split",
           "the interval columns of the delay and specification tables are a "
           "bootstrap of the whole ratio")
    forbid("4.5 reverts to bootstrap-as-only-source",
           "The published intervals are a bootstrap of the whole ratio, and each is "
           "checked by inverting a HAC t-test after Fieller (1954)")
    # Table 2's caption carries the bootstrap column and must say so, and must
    # say that the abstract's interval is not that column -- the caption used to
    # explain the 72/71 point-estimate gap and leave the 6-point interval gap
    # unexplained, which is how a reader concludes the two are the same interval.
    # Table 2's caption used to explain why its interval was NOT the
    # abstract's.  They are the same construction now, and the caption says so.
    want("Table 2 caption names the construction its cells use",
         "Intervals are the fixed-b Fieller sets of Section")
    want("Table 2 caption names the construction, singular",
         "the same construction the abstract and every other table here use")
    forbid("Table 2 caption still separates its interval from the abstract's",
           "which is not the construction the abstract&rsquo;s interval uses")
    # the two stale sentences the fixed-b switch left behind in 4.6: the
    # appendix no longer "leaves every interval standing" (it widens them by
    # 8.3 points) and the inversion no longer uses normal critical values.
    forbid("4.6 stale 'leaves every interval standing'",
           "leaves every published interval standing")
    forbid("4.6 stale 'normal critical values' in the construction paragraph",
           "it uses an estimated HAC covariance and normal critical values")
    forbid("the paper claims Fieller produced the intervals",
           "Every interval here is therefore obtained by inverting a t-test after Fieller")
    forbid("4.5 claims Fieller produced the interval",
           "The interval is therefore built by inverting a t-test after Fieller")
    # the Fieller result went against the objection; the paper must not claim
    # a pathology the computation did not find.
    forbid("9.2 stale Fieller caveat", "the quantity is not identified")
    l41 = lab("lab41_conditional_anatomy")
    checks += 1
    if "calm: 0 of 4 sets are not ordinary intervals" not in l41:
        fails.append("lab41: the calm Fieller sets are no longer ordinary intervals, "
                     "but Section 9.2 says they are")
    want("9.2 Fieller applied", "Every one comes back an ordinary interval")
    want("9.2 VIX tension", "can see it whether or not any option trades on the thing "
                            "they own")

    # --- every in-text table reference must point at a table that exists,
    # and at the right one.  Deleting a table has silently shifted these three
    # times; a caption keyword check is what finally caught it.
    import re as _re
    _caps = {int(m.group(1)): m.group(2)
             for m in _re.finditer(r"<b>Table (\d+)\.</b>\s*(.{0,90})",
                                   read_text(path), _re.S)}
    for _n in sorted({int(x) for x in _re.findall(r"Table (\d+)(?!\.</b>)",
                                                  read_text(path))}):
        checks += 1
        if _n not in _caps:
            fails.append(f"the text refers to Table {_n}, which does not exist")
    # Table 5 was the four-information-set race until Sections 7 and 8 were cut
    # to pointers for JFEc's page limit and their tables moved to the Internet
    # Appendix.  The paper's tables closed up behind them, so 5 is now the
    # claim-boundary table and the race is Table S22.  Both identities are
    # checked, each in its own document.
    for _n, _kw in ((1, "under each specification"),
                    (2, "Out-of-sample skill"), (5, "claim"),
                    (23, "when the foreign block is itself"),
                    (24, "Effective age"), (25, "raced against each other"),
                    (26, "sorted by coupling"), (19, "by market state, pooled"),
                    (20, "taken apart"), (21, "by target market")):
        checks += 1
        if _n in _caps and _kw not in _re.sub(r"<[^>]+>", "", _caps[_n]):
            fails.append(f"Table {_n} is no longer the '{_kw}' table; a reference "
                         "to it now points somewhere else")

    # --- the operational recommendation, and the horizon note -------------
    l23 = lab("lab23_compressed_everywhere")
    checks += 1
    _m23 = re.search(r"PC1\s+([\d.]+)%\s+\[\s*([\d.]+)%,\s*([\d.]+)%\]", l23)
    if not _m23:
        fails.append("lab23 no longer prints a PC1 rate for the compressed block")
    want("14 compression is the model to run",
         "a single real-time factor beating all seven series at every delay and raising "
         "R(55) to 75.2%")
    want("14 seven regressors not recommended",
         "compress the cross-section first and treat the seven-regressor figures as the "
         "conservative benchmark")
    want("8 horizon mismatch",
         "measured over a horizon about four times the length of the thing it is "
         "forecasting, not six")
    want("8 matched instrument named", "a nine-day index, VIX9D, formerly VXST")
    # the 81% sentence was duplicated once already
    checks += 1
    if txt.count("81% in each half of the sample taken separately") != 2:
        fails.append("the 'each half' sentence appears "
                     f"{txt.count('81% in each half of the sample taken separately')} "
                     "times; it belongs once in Section 10.2 and once in Section 14")

    # --- Table 20, the ratio taken apart, against lab41 -------------------
    for row in ("5 0.0284 0.1567 5.5x 0.0673 0.2434 3.6x",
                "13 0.0345 0.3787 11.0x 0.0798 0.5110 6.4x",
                "21 0.0412 0.5224 12.7x 0.1003 0.6755 6.7x",
                "55 0.0583 0.8043 13.8x 0.1234 0.9927 8.0x"):
        d = row.split()[0]
        checks += 1
        if row not in l41:
            fails.append(f"Table 20 delta={d}: lab41 no longer prints '{row}'")
        want(f"Table 20 row delta={d}", row.replace("x", "&times;"))
    checks += 1
    if "stress multiplies N by 10.7 and D by 6.2" not in l41:
        fails.append("lab41: the N and D multipliers are no longer 10.7 and 6.2")
    want("9.2 numerator moves", "stress multiplies N by 10.7 and D by 6.2")
    want("9.2 absolute reading",
         "removes 0.8043 of squared error in stressed markets against 0.0583 in calm")

    # --- Section 6, the model, against lab42 ------------------------------
    l42 = lab("lab42_factor_model")
    # Derived, not transcribed: every figure Section 6 quotes is read out of
    # lab42's own text here, so a lab rerun that moves a number fails the
    # paper rather than silently agreeing with a copy kept in this file.
    def pull(pat, label):
        m = re.search(pat, l42)
        if not m:
            fails.append(f"{label}: lab42 no longer prints a value matching {pat!r}")
        return m
    for pat, papern, label in (
            (r"fit R-squared = ([0-9.]+)", "R&sup2; = {}", "6 decay fit"),
            (r"slope = (-[0-9.]+) per day", "slope {} per day", "6 decay slope"),
            (r"implied persistence phi = exp\(slope/2\) = ([0-9.]+)",
             "implying &phi; = {}", "6 implied persistence"),
            (r"measured persistence of the factor += ([0-9.]+)", "is {}",
             "6 measured persistence"),
            (r"predicted asymptote R\(infinity\) = ([0-9.]+%)",
             "giving a ceiling of {}", "6 predicted ceiling")):
        checks += 1
        m = pull(pat, label)
        if m:
            want(label, papern.format(m.group(1)))
    checks += 1
    if "4 of 4 predictions hold" not in l42:
        fails.append("lab42: the model's predictions no longer all hold, but Section 6 "
                     "presents them as holding")
    # the agreement the section claims is the gap between the two persistences
    mi = re.search(r"implied persistence phi = exp\(slope/2\) = ([0-9.]+)", l42)
    mm = re.search(r"measured persistence of the factor += ([0-9.]+)", l42)
    if mi and mm:
        checks += 1
        want("6 agreement",
             f"The two agree to {abs(float(mi.group(1)) - float(mm.group(1))):.4f}")
    # the flatness contrast: both losses are printed by lab42's part 2
    mc = re.search(r"S_cross falls from [0-9.]+ to [0-9.]+ across 55 days, "
                   r"a loss of (\d+)%", l42)
    mo = re.search(r"S_own +falls from [0-9.]+ to [0-9.]+, a loss of (\d+)%", l42)
    if mc and mo:
        checks += 1
        want("6 flatness",
             f"it falls {mc.group(1)}%, against {mo.group(1)}% for the domestic model")
    want("6 ceiling met",
         "against a measured 69.8% at &delta; = 34 and 71.6% at &delta; = 55")
    # (16) is algebraically tied to the rate it "predicts" once S_own(delta) is
    # near zero, and the benchmark correction made the tie tighter rather than
    # looser.  The section is required to say so and forbidden from claiming an
    # unfitted prediction.  The number moved from (12) to (16) when the
    # appendix was renumbered to continue the manuscript instead of colliding
    # with it; this check pinned the literal including the number, which is
    # why it fired, and firing was correct.
    want("6 ceiling is interpretation",
         "a factor-model interpretation of the observed plateau rather than a "
         "prediction of it")
    want("6 ceiling identity shown",
         "this collapses to S cross (55) &divide; S own (0), which is equation (16)")
    forbid("6 stale unfitted-prediction claim",
           "Nothing in that prediction was fitted to the curve it predicts")
    want("6 what it cannot do", "What the model does not explain is why the mechanism "
                                "is stronger in stress")

    # the abstract's three claims, each tied to the section that establishes it
    # Read out of lab45's own age table, the same row Section 9.3 quotes, rather
    # than transcribed.  The abstract carried the pre-correction interval for a
    # week after Section 9.3 had been updated, because nothing tied the two
    # copies of the same figure to each other.
    if 0 in ages:
        _pt, _lo, _hi = ages[0][0], ages[0][1], ages[0][2]
        checks += 1
        want("abstract effective age",
             f"forecasts as well as a mark {_pt} days old [{_lo}, {_hi}]")
    _m45a = re.search(r"g = 1 minus g = 0\s+([\d.]+)d\s+\[([\d.]+), ([\d.]+)\]",
                      read_text(os.path.join(EXP,
                                             "lab45_effective_age_interval.txt")))
    checks += 1
    if not _m45a:
        fails.append("lab45 no longer prices one day of foreign staleness, which "
                     "the abstract quotes")
    else:
        want("abstract exchange rate",
             f"costs {_m45a.group(1)} days [{_m45a.group(2)}, {_m45a.group(3)}] "
             f"of domestic freshness")
    # scoped to the figure: the abstract now says "in this sample's stressed
    # markets", and a check that pinned the following word would fail on a
    # correctly hedged sentence.
    # READ from lab41, like the tercile block above.  This was the second
    # hardcoded copy of the same figure, and between them they let the abstract
    # quote a conditional rate that no script produced.  It reads the Fieller
    # set, because that is the construction Section 4.5 says the quoted rate
    # uses; reading the bootstrap here left the abstract and Table S4 stating
    # different widths for one cell.
    checks += 1
    if 55 in _tf:
        _p55, _l55, _h55, _t55 = _tf[55]["stressed"]
        want("abstract conditional rate",
             f"{_p55}% [{int(_l55)}, {int(_h55)}] in")
    else:
        fails.append("lab41 no longer prints the stressed Fieller set at eleven "
                     "weeks, so the abstract's conditional rate rests on nothing")
    # The abstract's stronger-control figure is read out of lab38 rather than
    # matched against the prose.  This check used to assert the prose verbatim,
    # which is worthless: it passed while the abstract carried an interval that
    # belonged to a different table in a different lab, because the needle had
    # been written from the paper instead of from the output.  A check that
    # cannot fail when the paper is wrong is not a check.
    _l38 = read_text(os.path.join(EXP, "lab38_domestic_baseline.txt"))
    _m38 = re.search(r"\s*55\s+(\d+)%\s+(\d+)%\s+([-+\d.]+)%\s+\[\s*([-+\d.]+)%,\s*([-+\d.]+)%\]",
                     _l38)
    checks += 1
    if not _m38:
        fails.append("lab38 no longer prints the delta=55 control comparison that "
                     "the abstract's stronger-control figure comes from")
    else:
        _own, _alt, _diff, _lo, _hi = _m38.groups()
        # the abstract now names which control it means without repeating the
        # superlative, so the check pins the figure and the reference to it
        # The 100-word abstract JFEc allows cannot carry this, so the guard moved
        # to Section 10, which is where the envelope is argued.  The constraint is
        # unchanged: the stronger control's figure must be quoted BESIDE the
        # paper's own, so the headline is never shown without its harder test.
        want("10 quotes the stronger control beside its own",
             f"{_alt}% against the strongest control alongside its own")
        # The abstract points at the section of the appendix that runs the eight
        # controls, and the FIRST version of this check hard-coded the section
        # number by copying the abstract's own sentence - which is how the
        # abstract came to cite S9, where the envelope is not, for a whole
        # revision without the verifier noticing.  The number is now read out of
        # the appendix: whichever section heading owns the paragraph that reports
        # the envelope is the one the abstract has to name.
        _env = re.search(r'<h[23][^>]*>S(\d+)\.(?:(?!<h[23]).)*?'
                         r'the substitution rate across the eight runs',
                         read_text(_supp), re.S)
        checks += 1
        if not _env:
            fails.append("the Internet Appendix no longer has a section reporting "
                         "the eight-control envelope, which the abstract cites")
        else:
            # The sentence is in the appendix, where "of the Internet Appendix"
            # would be the appendix naming itself, so the needle is the bare
            # cross-reference.
            want("the paper names the envelope's section",
                 f"against the stronger control of Section S{_env.group(1)}")
        want("Table 1 stronger-control difference",
             f"is {_diff.lstrip('+-')} points below the paper's own")
        want("Table 1 stronger-control interval", f"[{_lo}, {_hi}]")
        # and the figure must NOT be paired with any interval of its own
        forbid("abstract invents an interval for the stronger control",
               f"{_alt}% [50, 86] against the strongest domestic control")
    want("abstract normalisation finding",
         "Restoring that level is worth more at long delays than the entire cross-section")
    want("compression advice", "one real-time factor")

    # The recommendation in Section 12 quotes Table 15 back at the reader; the
    # table rows are checked above, this checks that the sentence still agrees
    # with them rather than drifting into a rounder-sounding claim.
    want("12 foreign-feed condition",
         "36% into 9% at three days and its 51% into 29% at five")

    # --- Section 13's prediction rests on lab28; Section 9's seam on lab03 ---
    # A further-research section that mispredicts what the paper already measured
    # is worse than no further-research section, so the prediction is tied to the
    # count lab28 actually prints.
    l28o = lab("lab28_orthogonal_breadth")
    checks += 1
    if "ORTHOGONAL remainder beats domestic-only: 0 of 6" not in l28o:
        fails.append("lab28: the orthogonal remainder now beats the domestic model "
                     "somewhere, but Section 13 still predicts from its beating it "
                     "nowhere")
    want("13 prediction from 5.3",
         "the remainder is negative at every delay and beats the stale domestic "
         "model at none")
    want("13 cross-asset test", "Breadth ACROSS asset classes is")
    want("13 FX first", "current by construction, not by luck of the clock")

    # --- Table 7, the claim-boundary table -------------------------------
    # A table that grades the paper's own claims is the easiest table in the
    # paper to let drift, because nothing downstream reads it: a figure could be
    # softened here while the section it summarises still said something
    # stronger.  So every figure in it is required to be a figure the paper
    # already carries and the labs already support, and the grades are required
    # to be drawn from the fixed vocabulary the paragraph above the table
    # defines.  A grade invented later fails here rather than in review.
    _grades = ("established", "supported", "suggestive", "not claimed")
    _t13 = re.search(r"<b>Table 5\.</b>.*?</table>", read_text(path), re.S)
    checks += 1
    if not _t13:
        fails.append("Table 5, the claim-boundary table, is missing from the paper")
    else:
        _body = _t13.group(0)
        _cells = re.findall(r"<td>([^<]*)</td>", _body)
        _used = [c.strip() for c in _cells if c.strip() in _grades]
        checks += 1
        if len(_used) < 10:
            fails.append(f"Table 7 grades only {len(_used)} claims; it is meant to "
                         "carry every claim the paper makes")
        checks += 1
        _bad = [c.strip() for c in _cells
                if re.fullmatch(r"[a-z ]{4,12}", c.strip()) and c.strip() not in _grades
                and len(c.strip().split()) <= 2]
        if _bad:
            fails.append("Table 7 uses grades outside the defined four: "
                         + ", ".join(sorted(set(_bad))))
        # each of these figures must ALSO appear in the section the row cites,
        # so the table cannot quietly disagree with the body of the paper
        # the figure has to be IN the table and ALSO somewhere outside it.
        # Counting occurrences document-wide was the first version of this and
        # it was too weak: a figure altered inside the table still passed on the
        # strength of the two copies elsewhere.
        _outside = read_text(path).replace(_body, "")
        for _fig, _where in ((_m13 and f"{_m13.group(4)}% [{_m13.group(5)}, "
                              f"{_m13.group(6)}]" or "71%", "Section 6"),
                             # derived from lab39, not transcribed, for the
                             # same reason as the abstract copy above
                             ((lambda _s: (f"{_s[55]['stressed'][0]}% "
                                           f"[{int(_s[55]['stressed'][1])}, "
                                           f"{int(_s[55]['stressed'][2])}]")
                               if 55 in _s
                               else "81%")(tercile_fieller()), "Section S10"),
                             # derived from lab52, not transcribed: this entry
                             # was a literal and went stale with the bandwidth
                             (CELLS or "thirteen of twenty-four", "Section S13"),
                             ("35 of 42", "Section 9.2"),
                             ("14 of 14", "Section S18")):
            checks += 1
            if _fig not in _body:
                fails.append(f"Table 7 no longer quotes '{_fig}' for {_where}")
            checks += 1
            if _fig not in _outside and _fig not in read_text(_supp):
                fails.append(f"Table 7 quotes '{_fig}' but {_where} no longer does, "
                             "so the summary and the body disagree")

    # --- the econometric core is cited, and sits in the methodology ---------
    # Fieller's theorem carried an entire section of this paper for several
    # drafts without Fieller appearing in the reference list, which is the kind
    # of omission a referee at an econometrics journal notices first.  The
    # construction and the identification result behind it are now both cited,
    # and the section lives in the methodology rather than in back matter.
    _paper_src = read_text(path)
    # Author and year, in EITHER reference style.  These two were pinned as
    # literals in the base style, "Fieller, E. C. (1954)", and the JFEc variant
    # converts the list to the journal's own shape, "Fieller, E. C. 1954." - so
    # the literal failed on a variant whose reference list is in fact correct.
    # A check that only passes under one journal's house style is a check that
    # will be edited away the first time it fires, so it now asks the question
    # it means: is this author cited, with this year, in this list?
    for _lbl, _who, _yr in (
            ("Fieller is in the reference list", "Fieller, E. C.", "1954"),
            ("Dufour is in the reference list", "Dufour, J.-M.", "1997")):
        checks += 1
        if not re.search(re.escape(_who) + r"\s*\(?" + _yr + r"\)?\.", _paper_src):
            fails.append(f"{_lbl}: MISSING '{_who} ({_yr})' in either "
                         f"reference style")
    checks += 1
    if "Inference for a ratio whose denominator can vanish</h3>" not in _paper_src:
        fails.append("4.6 is a methodology subsection: MISSING 'Inference for a "
                     "ratio whose denominator can vanish</h3>'")
    checks += 1
    if re.search(r'<h2[^>]*>Appendix\.', _paper_src):
        fails.append("the Fieller material is back in an unnumbered appendix; it "
                     "belongs in the methodology where its cross-references point")
    # every numbered citation in the paper must have a reference to land on
    _nrefs = re.search(r'<ul class="refs">(.*?)</ul>', _paper_src, re.S)
    checks += 1
    if not _nrefs:
        fails.append("the paper's reference list is missing")
    else:
        _n = _nrefs.group(1).count("<li>")
        checks += 1
        if _n < 30:
            fails.append(f"the reference list has shrunk to {_n} entries")
        # Chicago author-date: no numbered citation may survive the conversion,
        # and every author-date citation must name a surname the list carries.
        _sur = {re.sub("<[^>]*>", "", x).split(",")[0].strip()
                for x in re.findall(r"<li>(.*?)</li>", _nrefs.group(1), re.S)}
        checks += 1
        _left = [m.group(0) for m in
                 re.finditer(r"(?<![\d%])\s\[\d{1,2}(?:\s*,\s*\d{1,2})*\](?![\d])",
                             _paper_src)
                 if max(int(z) for z in re.findall(r"\d+", m.group(0))) <= 32]
        if _left:
            fails.append("numbered citations survived the author-date conversion: "
                         + ", ".join(sorted(set(_left))[:6]))
        checks += 1
        _bad = sorted({m.group(1) for m in
                       re.finditer(r"\((?:see )?([A-Z][A-Za-z\u00C0-\u017F'-]+)(?: et al\.| and [A-Z][A-Za-z\u00C0-\u017F'-]+)? (?:19|20)\d{2}\)",
                                   _paper_src)} - _sur)
        if _bad:
            fails.append("these author-date citations name nobody in the reference "
                         "list: " + ", ".join(_bad[:6]))

    l03 = lab("lab03_crosssection")
    checks += 1
    if l03.count("-> VERIFIED") != 7 or "seam SPX: only 0 overlapping rows" not in l03:
        fails.append("lab03: the seam audit no longer shows seven verified index "
                     "seams and an unverifiable SPX one, which Section 9 now states")
    want("12 unverified seam",
         "eight overlap by 120 to 251 days and agree exactly on every overlapping "
         "row, while the S&P's two files abut without overlapping")

    # --- Section 3 must say why the NON-robust loss carries the headline ----
    want("3 headline-loss defence",
         "the headline number in this paper is the one loss of the three that is "
         "NOT proxy-robust")
    want("3 headline-loss reason", "a ratio needs a denominator a reader can hold")

    # ================================================================
    # Section S18 and the Section 9.2 summary, against lab57
    # ----------------------------------------------------------------
    # The overlap audit is the objection that could have retired Section 9.2,
    # so every figure the paper quotes from it is rebuilt from the lab rather
    # than spot-checked, including the six rows of Table S14 cell by cell.
    l57 = read_text(os.path.join(EXP, "lab57_housing_overlap.txt"))

    # the three per-phase rows at delta = 1 quarter, and the three at delta = 2
    R57 = re.compile(r"\s*(\S+/\S+)\s+(\d)\s+([\d.]+)\s+([\d.]+)\s+"
                     r"([-\d.]+)%\s+\[(-?\d+)%, (-?\d+)%\]\s+"
                     r"([-\d.]+)%\s+\[(-?\d+)%, (-?\d+)%\]\s+(\d+)/(\d+)\s*$")
    n57 = 0
    for ln in l57.splitlines():
        m = R57.match(ln)
        if not m:
            continue
        phase, d, sown, scross, mor, alo, ahi, rom, blo, bhi, pos, nus = m.groups()
        n57 += 1
        want(f"lab57 {phase} d={d} S_own", sown)
        want(f"lab57 {phase} d={d} S_cross", scross)
        want(f"lab57 {phase} d={d} pooled", f"{rom}%")
        want(f"lab57 {phase} d={d} pooled interval", f"[{blo}, {bhi}]")
    checks += 1
    if n57 != 6:
        fails.append(f"lab57: matched {n57} phase rows, expected 6")

    # every per-phase pooled interval must exclude zero, which is the claim
    # Section 9.2 makes and the one a wider interval would falsify
    checks += 1
    if "3 exclude zero on the pooled one" not in l57:
        fails.append("lab57: the three phases no longer all exclude zero on the "
                     "pooled interval, which Section 9.2 and S18 both state")
    # The quarterly summary figures are read out of lab57 rather than kept here:
    # this file borrowed lab55's scorer and inherited its benchmark, so every
    # level in it moved when that was corrected.
    _q57 = re.search(r"delta = 1 quarter : phase mean of 'mean R' +[\d.]+%; "
                     r"of 'pooled R' +([\d.]+)%, phases spanning "
                     r"\[([\d.]+)%, ([\d.]+)%\]", l57)
    _d57 = re.search(r"one month of staleness now costs (-?[\d.]+) of R-squared", l57)
    _i57 = re.search(r"the peer block still adds (\+[\d.]+) of R-squared", l57)
    if not (_q57 and _d57 and _i57):
        fails.append("lab57 no longer prints its quarterly summary in the shape "
                     "Section 9.2 quotes")
        _q57 = _d57 = _i57 = None
    for tok, label in ([] if _q57 is None else
                       [(_q57.group(1) + "%", "lab57 pooled mean"),
                        (_q57.group(2) + "%", "lab57 phase low"),
                        (_q57.group(3) + "%", "lab57 phase high"),
                        ("35 of 42", "lab57 metro-phases positive"),
                        (_d57.group(1), "lab57 de-smoothed delay cost"),
                        (_i57.group(1), "lab57 de-smoothed increment"),
                        ("0.567", "lab57 mean rho")]):
        checks += 1
        if tok.replace(",", "") not in l57.replace(",", ""):
            fails.append(f"{label}: lab57 no longer prints {tok}")
    # the paper rounds the pooled mean to 57%; both forms are checked so that a
    # change in the lab cannot pass by matching only the rounded prose
    want("9.2 quarterly rate", "the rate is 57% at one quarter of staleness")
    want("9.2 quarterly span", "spanning 49% to 63% across phases")
    want("9.2 quarterly breadth", "positive in 35 of 42 metro-phases")
    if _d57 and _i57:
        want("9.2 de-smoothed cost",
             f"one month of staleness costs {_d57.group(1).replace('-', MINUS)} of R")
        want("9.2 de-smoothed increment", f"still adds {_i57.group(1)} of R")
    want("S18 mean rho", "averages 0.567 across metros")
    # and the paper must not claim the quarterly figure as a second headline,
    # nor claim it agrees with the monthly one, because it is lower
    forbid("9.2 quarterly overclaim",
           "the quarterly figure confirms 67.8%")
    want("9.2 reports the gap rather than the agreement",
         "57% is visibly below 67.8%")

    # ================================================================
    # Section S17, against lab53's diagnostics
    # ----------------------------------------------------------------
    # The paper makes an ORDERING claim and draws a LINE, and S17 exists to say
    # which of the two eight points can carry.  Both halves are checked, so a
    # future edit cannot keep the reassuring half and drop the caution.
    l53 = read_text(os.path.join(EXP, "lab53_no_options_targets.txt"))
    for tok, label in (("Pearson  +0.979", "lab53 Pearson"),
                       ("[+0.884, +0.996]", "lab53 Fisher interval"),
                       ("Spearman +0.976", "lab53 Spearman"),
                       ("Pearson +0.903, Spearman +0.943", "lab53 six-cluster"),
                       ("0.734", "lab53 KSE100 leverage"),
                       ("+1.455", "lab53 slope without KSE100"),
                       ("+0.309", "lab53 crossing without KSE100")):
        checks += 1
        if tok not in l53:
            fails.append(f"{label}: lab53 no longer prints '{tok}'")
    want("S17 Fisher interval", "Fisher interval of [+0.884, +0.996]")
    want("S17 Spearman", "the Spearman rank correlation, which is what the "
                         "ordering claim actually needs, is +0.976")
    want("S17 leave-one-out floor", "never falls below +0.976")
    want("S17 single cluster", "the correlation is +0.903 and the Spearman +0.943")
    want("S17 leverage", "leverage of 0.734 against a mean of 0.250")
    want("S17 slope shift", "from +1.154 to +1.455")
    want("S17 crossing shift", "from +0.216 to +0.309")
    want("11 coupling caution", "the ordering is robust and the crossing is not")
    forbid("11 crossing as threshold",
           "below a coupling of 0.22 the cross-section can be expected to fail")

    # ================================================================
    # The nine export joins, against lab47
    # ----------------------------------------------------------------
    # This check exists because the paper carried two different ranges for the
    # same quantity, 120 to 251 in Section 3 and 120 to 129 in the limitations,
    # and nothing compared them.  The lab now prints the range and both places
    # in the paper are checked against it.
    l47 = read_text(os.path.join(EXP, "lab47_target_seam.txt"))
    m47 = re.search(r"(\d+) joins audited; (\d+) of them overlap, by (\d+) to (\d+)", l47)
    checks += 1
    if not m47:
        fails.append("lab47 no longer prints the audited-join summary")
    else:
        njoin, nov, olo, ohi = m47.groups()
        for label in ("3 export overlap", "12 export overlap"):
            want(label, f"overlap by {olo} to {ohi}")
        checks += 1
        if txt.count(f"{olo} to {ohi} days") + txt.count(f"{olo} to {ohi} trading days") < 2:
            fails.append("the export-overlap range is stated fewer than twice; "
                         "Section 3 and Section 12 must both carry it")
        # the paper spells small counts as words, so the digit the lab prints
        # has to be converted before it can be looked for.
        _w = {"8": "eight", "9": "nine", "10": "ten"}.get(njoin, njoin)
        want("3 joins counted", f"which leaves {_w} joins to be checked")
        _wov = {"7": "Seven", "8": "Eight", "9": "Nine"}.get(nov, nov)
        want("3 joins overlapping", f"{_wov} of the {_w} overlap")

    # ================================================================
    # Figure 1, against the constants the code actually uses
    # ----------------------------------------------------------------
    # The timeline is drawn from lab05's constants by labs/fig_timeline.py, so
    # the figure cannot drift from the design.  What CAN drift is the caption,
    # which restates the numbers in prose, so the caption is checked here.
    _fig = txt.split("Figure 1.")[-1].split("</figcaption>")[0] if "Figure 1." in txt else ""
    checks += 1
    if not _fig:
        fails.append("Figure 1's caption could not be located")
    for tok in ("1,250", "250-day", "+h"):
        checks += 1
        if tok not in _fig:
            fails.append(f"Figure 1's caption no longer states {tok}")
    want("4.3 points at the timeline", "Figure 1 draws the whole arrangement")

    # ================================================================
    # Sections and subsections are set in one font
    # ----------------------------------------------------------------
    # The main paper and the Internet Appendix italicised subsections while the
    # companion note did not, so the three documents disagreed with each other
    # about their own heading style.  All three now set h2 and h3 in the body
    # face, bold, roman, differing only in size, and the rule is checked because
    # a stylesheet is exactly the kind of thing that gets edited once and never
    # looked at again.  Size may differ: that is the hierarchy.  Family, weight
    # and slope may not.
    for _label, _p in _emdash_docs:
        _css = read_text(_p)
        _rules = {}
        for _h in ("h2", "h3"):
            _m = re.search(rf"^{_h}\{{([^}}]*)\}}", _css, re.M)
            _rules[_h] = _m.group(1) if _m else None
        checks += 1
        if not all(_rules.values()):
            fails.append(f"{_label}: could not find both h2 and h3 rules in its "
                         "stylesheet")
            continue
        for _h, _r in _rules.items():
            checks += 1
            if "font-style" in _r:
                fails.append(f"{_label}: {_h} sets font-style, so subsections and "
                             "sections are not in the same font")
            checks += 1
            if "font-family" in _r:
                fails.append(f"{_label}: {_h} overrides font-family, so headings "
                             "leave the body face")
        checks += 1
        _w = [re.search(r"font-weight:(\d+)", _r) for _r in _rules.values()]
        if not all(_w) or len({m.group(1) for m in _w}) != 1:
            fails.append(f"{_label}: h2 and h3 do not share a font-weight")
        checks += 1
        _sz = [float(re.search(r"font-size:([\d.]+)pt", _r).group(1))
               for _r in _rules.values()]
        if not _sz[0] > _sz[1]:
            fails.append(f"{_label}: h3 is not smaller than h2 ({_sz[1]} against "
                         f"{_sz[0]}), so the heading levels do not read as a "
                         "hierarchy")

    # ================================================================
    # A paper is not a README: where a filename may and may not appear
    # ----------------------------------------------------------------
    # Script names accumulated in these documents until a figure caption
    # credited a plotting script and the companion note listed six labs inline,
    # which reads like a laboratory notebook rather than a paper.  The rule
    # settled on is narrow enough to enforce: a lab script is named in the
    # repository's README and nowhere in the three documents, while the four
    # commands a reader actually types may appear in the statements whose job
    # is to tell them what to run.  Captions name nothing, because a caption
    # describes a result rather than the machinery behind it.
    TOOLS = ("run_all.py", "verify_paper.py", "verify_data.py", "build_pdf.py",
             "make_all.py", "build_manifest.py")
    for _label, _p in _emdash_docs:
        _raw = read_text(_p)
        # no lab script anywhere in a document
        checks += 1
        _labs = sorted(set(re.findall(r"\blab\d+[a-z]?_[a-z_0-9]+\.py", _raw)))
        if _labs:
            fails.append(f"{_label} names lab scripts in its prose, which belong in "
                         "the README: " + ", ".join(_labs))
        # and no filename of any kind inside a caption
        checks += 1
        _caps = re.findall(r"<(?:figcaption|caption)[^>]*>(.*?)</(?:figcaption|caption)>",
                           _raw, re.S)
        _incap = sorted({m for c in _caps
                         for m in re.findall(r"[\w/]+\.py", c)})
        if _incap:
            fails.append(f"{_label} names a script inside a caption: "
                         + ", ".join(_incap))
        # the tools that ARE allowed must still be ones that exist
        checks += 1
        _named = sorted(set(re.findall(r"\b([\w]+\.py)\b", _raw)))
        _ghost = [n for n in _named
                  if n not in TOOLS and not os.path.isfile(os.path.join(HERE, n))]
        if _ghost:
            fails.append(f"{_label} names scripts that are neither the reader's "
                         "commands nor files in the repository: " + ", ".join(_ghost))

    # ================================================================
    # Section 4.4 and S9 must tell the same story about the zero-delay row
    # ----------------------------------------------------------------
    # These two passages disagreed for one revision: S9 was corrected to say
    # the three losses point the same way and 4.4 still said the two robust
    # ones disagree with each other.  Neither sentence carries a number, so no
    # figure check could have caught it, and a reader comparing the two would
    # have found the paper contradicting itself about its own loss functions.
    want("4.4 zero-delay direction",
         "both point in the same negative direction but the natural-scale MSE "
         "is too imprecise to establish an effect")
    forbid("4.4 stale zero-delay disagreement",
           "where they disagree with each other as well")
    forbid("4.4 stale withdrawal", "withdraws the claim instead of picking a winner")
    want("4.4 rests no claim on that row",
         "no economic claim in this paper rests on that row")
    # and the illiquid claim must stay inside what a published index supports
    forbid("abstract overclaims the illiquid case",
           "the illiquid case that motivates the exercise is measured rather than")
    forbid("1 overclaims the illiquid case", "The last step is therefore to stop extrapolating")
    # The abstract used to carry the housing scope condition too; at 100 words it
    # cannot, so the introduction is now the only place that states it and the
    # guard rests there alone.  The claim it protects is the one this paper is
    # most likely to be over-read on: a published index is not a marked book.
    want("1 scopes the housing test", "narrow that extrapolation as far as public data allows")
    want("1 states the index-is-not-a-book limit", "a published index is not a book")

    # --- the AI disclosure -------------------------------------------------
    # There are TWO legitimate forms of this section and the checks have to know
    # both.  The SSRN copy states the author's contribution, the tool's role and
    # the denial of AI authorship, and stops there.  The journal copies add the
    # declaration Elsevier prescribes, because IJF and JEF require it and SSRN
    # does not.  build_variants.py swaps one for the other, so this file reads
    # whichever it was handed and validates THAT form completely - checking only
    # the phrases common to both would leave each form's own content unguarded,
    # which is how a section gets quietly shortened.
    _flat = re.sub(r"\s+", " ", read_text(path))
    # Which disclosure this document carries used to be detected by the sentence
    # "In the wording requested by certain target journals", which was also the
    # sentence that made the journal form read as though it had been written for
    # a different journal and left in.  Removing it therefore made the journal
    # variant look like the SSRN source and fired the wrong three checks.  The
    # form is now detected by a sentence that belongs to the journal disclosure
    # on its merits.
    _journal_form = ("The author is solely responsible for all contents of the paper"
                     in _flat)
    _shared = (
        ("disclosure names the tool",
         "the author used Claude (Anthropic) as an assistive tool for drafting "
         "and revising text and"),
        ("disclosure denies AI authorship",
         "No AI system is an author of this work"),
        ("disclosure records the author's own checking",
         "reviewed, tested, and validated the code"),
    )
    _ssrn_only = (
        ("disclosure attributes the research to the author",
         "The research questions, mathematical development, empirical design, data "
         "construction, analysis, interpretation of results, and decisions regarding "
         "the paper's claims were the author's own"),
        ("disclosure states sole responsibility",
         "takes sole responsibility for the manuscript's contents, including all "
         "methods, results, interpretations, and reported numbers"),
        ("disclosure denies AI authorial responsibility",
         "no AI system held authorial responsibility for any part of it"),
    )
    _journal_only = (
        ("disclosure attributes the research to the author",
         "The research questions, mathematics, empirical design, interpretation of "
         "results, and decisions about the paper's claims are the author's own"),
        ("disclosure states sole responsibility",
         "The author is solely responsible for all contents of the paper, including "
         "every claim, design decision, and reported number"),
        ("disclosure denies AI authorial responsibility",
         "or held authorial responsibility for any part of it"),
        # The publisher's template sentence was a third paragraph restating the
        # first two and announcing that it did so.  In a submission to one named
        # journal it reads as a disclosure assembled for a different one, so it
        # is gone and its absence is now the thing checked.
    )
    for _lbl, _needle in _shared + (_journal_only if _journal_form else _ssrn_only):
        checks += 1
        if _needle not in _flat:
            fails.append(f"{_lbl}: MISSING '{_needle[:60]}...'")


    # --- one target definition per place, said where it is met -------------
    # The paper computes its tables on the target-dated definition and quotes
    # the implementable origin-dated one in headline sentences.  That is a
    # defensible arrangement and an indefensible one to leave implicit: a
    # reader meeting 71% in the abstract and 72% in the first table has to
    # reconcile them unaided, and one reader already did.  Both statements of
    # the convention are required, so neither can be dropped as redundant.
    want("4.2 states the convention once",
         "the equity-result tables in this paper report the target-dated rate and "
         "every headline sentence reports the implementable origin-dated one")
    want("Table 2 repeats it where the gap is met",
         "reads 72% at eleven weeks where the abstract&rsquo;s implementable "
         "figure is 71%")


    # ================================================================
    # The Internet Appendix's own numbering, and the envelope it cites
    # ----------------------------------------------------------------
    # Two collisions got through by hand: a second Table S16 when a moved
    # subsection brought its own table with it, and a section count left at
    # twenty-four after a twenty-fifth was appended.  Both are the kind of
    # error that a reader notices and an author never does, so both are counted
    # here rather than read.
    _ia = os.path.join(os.path.dirname(os.path.abspath(path)),
                       "stale-mark-internet-appendix.html")
    checks += 1
    if not os.path.isfile(_ia):
        fails.append("the Internet Appendix source is missing")
    else:
        _iatext = read_text(_ia)
        _caps = re.findall(r"<b>Table (S\d+)\.</b>", _iatext)
        checks += 1
        _dup = sorted({c for c in _caps if _caps.count(c) > 1})
        if _dup:
            fails.append("the Internet Appendix uses these table numbers twice: "
                         + ", ".join(_dup))
        checks += 1
        _nums = [int(c[1:]) for c in _caps]
        if _nums != list(range(0, len(_nums))):
            fails.append(f"the Internet Appendix's tables are not S0..S{len(_nums)-1} "
                          f"in order: {_caps}")
        _secs = sorted({int(m) for m in re.findall(r">S(\d+)\.", _iatext)})
        _n_sub = len([x for x in _secs if x > 0])
        checks += 1
        if _secs != list(range(0, max(_secs) + 1)):
            fails.append(f"the Internet Appendix's sections are not contiguous: {_secs}")
        # The count is of SUBSTANTIVE sections, S1 upward; S0 is the claim map
        # and counting it would be the opposite error to the one that left the
        # count at twenty-four.  The sentence has to name the range as well as
        # the number, so neither can drift without the other failing.
        want("S0 states the section count",
             f"{spell(_n_sub).capitalize()} substantive sections, S1 through "
             f"S{max(_secs)}, plus this claim map, is enough")
        # and every section must appear somewhere in the claim map or be the map
        _mapped = set()
        _m0 = re.search(r"Table S0\..*?</table>", _iatext, re.S)
        if _m0:
            _mapped = {int(x) for x in re.findall(r"S(\d+)", _m0.group(0))}
        # S1 and S9 are umbrella headings that the sections beneath them sit
        # under, not tests, and are identified as such by carrying no verdict
        # label.  Everything that reports a verdict has to be in the map.
        _verdicted = {int(m) for m in re.findall(
            r'>S(\d+)\.[^<]*<span style="font-weight:400', _iatext)}
        checks += 1
        _missing = [f"S{x}" for x in _secs
                    if x > 0 and x in _verdicted and x not in _mapped]
        if _missing:
            fails.append("these Internet Appendix sections are in neither the claim "
                         "map nor its ranges: " + ", ".join(_missing))

    # the eight-control envelope the abstract now distinguishes itself from
    _l49 = lab("lab49_control_envelope")
    _m49 = re.search(r"runs from ([\d.]+)% to ([\d.]+)%", _l49)
    checks += 1
    if not _m49:
        fails.append("lab49 no longer prints the control envelope's range, which "
                     "the abstract cites")
    else:
        want("abstract envelope range", f"{_m49.group(1)}% to {_m49.group(2)}%")

    # ================================================================
    # S3's remedy must be the one lab16 actually runs
    # ----------------------------------------------------------------
    # S3's prose said the remedy re-runs the statistic on an UNPENALISED fit
    # while its own table said the columns impose a COMMON ridge penalty.  Those
    # are different experiments and lab16 runs the second one, so the prose
    # described a procedure nobody executed.  The lab's own column headings are
    # now the authority.
    _l16 = lab("lab16_clark_west_shrinkage")
    checks += 1
    if "CW separate" not in _l16 or "CW common" not in _l16:
        fails.append("lab16 no longer reports separate and common penalty "
                     "columns, which S3 describes")
    want("S3 names the common-penalty remedy",
         "imposes a single ridge penalty on both arms, selected on the "
         "restricted one")
    forbid("S3 stale unpenalised remedy",
           "the remedy re-runs the statistic on an unpenalised fit")
    forbid("S3 stale unpenalised lean", "lean on the unpenalised re-run")

    # ================================================================
    # Table 1's sample sizes, against the labs each row comes from
    # ----------------------------------------------------------------
    # The first version of this caption said every equity row was on 4,525 test
    # days.  That is the options window of Section 8 and NO row of Table 1 uses
    # it: five rows are on lab05's 4,862 days and the stressed row is the upper
    # tercile of those.  A caption naming a sample none of its rows uses is the
    # kind of error that survives indefinitely because it looks like detail, so
    # each count is now read out of the lab that produces it.
    _n_main = re.search(r"(\d+) test days", lab("lab05_robustness"))
    _n_str = re.search(r"stressed\s+55\s+[\d.]+\s+[\d.]+\s+\d+%\s+(\d+)",
                       read_text(os.path.join(EXP,
                                              "lab41_conditional_anatomy.txt")))
    _n_opt = re.search(r"(\d+) test days", lab("lab08_implied_vol"))
    checks += 1
    if not (_n_main and _n_str and _n_opt):
        fails.append("could not read Table 1's sample sizes out of lab05, lab41 "
                     "and lab08")
    else:
        _m, _st, _op = (int(_n_main.group(1)), int(_n_str.group(1)),
                        int(_n_opt.group(1)))
        # the sample size is the claim; which section number carries it is a
        # per-manuscript matter once a variant renumbers
        want("Table 1 main sample", f"same {_m:,} test days as Section")
        want("Table 1 stressed sample", f"the {_st:,} of those days")
        want("Table 1 disclaims the options window",
             f"No row here uses the {_op:,}-day window of Section")
        forbid("Table 1 stale sample claim",
               f"all are on the same {_op:,} test days")
        checks += 1
        if _st >= _m:
            fails.append(f"lab41's stressed subset ({_st}) is not smaller than "
                         f"lab05's sample ({_m}), so the caption is wrong")

    # ================================================================
    # Section 4.3's training window, against lab05's own constants
    # ----------------------------------------------------------------
    # This check exists because the paper got this wrong twice over. Equation
    # (4) used to define T(t) as 1,250 days ending at t, while the code fits
    # 1,500 days ending at t - delta - h: the window is longer than the paper
    # said AND stops earlier than the paper said. Nothing downstream was
    # affected - every number came from the code - but the methodology section
    # described a different experiment from the one that was run, which is the
    # worst place in a paper to be loose. Both numbers are now derived from
    # lab05 rather than typed, so neither can drift again.
    # labs/ is a sibling in the tidy layout and this folder in a flat upload,
    # which is the same search the rest of this script does for its inputs.
    for _d in (os.path.join(HERE, 'labs'), HERE):
        if os.path.isfile(os.path.join(_d, 'lab05_robustness.py')) and _d not in sys.path:
            sys.path.insert(0, _d)
    import lab05_robustness as _L5                                # noqa: E402
    _tot = _L5.TRAIN + _L5.VAL
    want("4.3 window length", f"the window holds {_tot:,} days")
    # plain() strips the <sub> tags to spaces, so the needles are written the
    # way the stripped text reads rather than the way the source does.
    want("4.3 window end", "c t = t - &delta; - h")
    want("4.3 window set",
         f"T &delta; (t) = {{c t - {_tot - 1}, &hellip;, c t }}")
    want("4.3 two-step penalty",
         f"candidate penalties are fitted on the first {_L5.TRAIN:,} days")
    want("4.3 final refit", f"refitted on all {_tot:,}")
    # This guarded a sentence that was itself wrong: the validation tail is held
    # out of the CANDIDATE FITS, and is precisely the data that performs the
    # selection.  An audit caught the contradiction, and the check now pins the
    # corrected statement rather than the comfortable one.
    want("4.3 says what the tail is held out of",
         "held out of the candidate fits, which is what lets it choose between "
         "them, and is not held out of the final model")
    forbid("4.3 stale window definition", "T(t) = {t &minus; 1249")
    forbid("4.3 stale window prose", "on a rolling 1,250-day window")
    want("Figure 1 caption states the refit",
         f"refitted on all {_tot:,}, so the tail is held out of the selection")
    checks += 1
    if f"{_L5.MED:,} + {_L5.TRAIN:,} + {_L5.VAL:,} = {_L5.MED + _tot:,}" not in txt:
        fails.append("Section 9.1's burn-in arithmetic no longer adds the three "
                     f"blocks to {_L5.MED + _tot:,}, so it disagrees with 4.3")

    # ================================================================
    # The data are not redistributed, and the paper must say so
    # ----------------------------------------------------------------
    # Every vendor behind the inputs forbids redistribution, so the repository
    # ships digests instead of files.  A paper that still claims to ship the
    # data would be making a promise the repository does not keep, and that is
    # exactly the kind of drift this script exists to catch.
    want("availability names the restriction",
         "Every vendor behind the inputs prohibits redistributing their files")
    want("availability names the one exception",
         "the FHFA index that FRED places in the public domain")
    want("availability names the three commands", "python verify_data.py")
    forbid("availability still promises the data",
           "the scripts, the data,")
    checks += 1
    _man = os.path.join(HERE, "data", "MANIFEST.tsv")
    if not os.path.exists(_man):
        fails.append("data/MANIFEST.tsv is missing, so the availability "
                     "statement's promise cannot be kept")
    else:
        _rows = read_text(_man).strip().splitlines()
        checks += 1
        if len(_rows) - 1 != 35:
            fails.append(f"the manifest lists {len(_rows) - 1} files; the paper "
                         "says thirty-five")
        want("availability file count", "thirty-five input files exactly one")

    # --- Section 12 must name every script the paper's numbers come from ---
    # This check caught two real misses.  Sections 5.3, 8.1 and 8.2 were added
    # citing lab27 and lab28 while the text still said "twenty scripts" and
    # listed neither; later lab41 and lab42 supplied the whole of Section 6 and
    # part of Section 10.2 without ever being named, so a reader could not find
    # where those figures came from.  The list is no longer typed here: it is
    # whatever this verifier actually read while checking the main paper, so a
    # lab added tomorrow is covered the moment its output is checked.
    CITED = sorted(USED)
    # The script list used to sit in an appendix of the paper.  It is archiving,
    # not argument, so it now lives in the repository's README and the check
    # follows it there: the claim is unchanged, only its address is.
    checks += 1
    if not RDME:
        fails.append("README.md is missing, so the mapping from result to script "
                     "that the paper's availability statement points at does not exist")
    for name in CITED:
        checks += 1
        if name + ".py" not in RDME:
            fails.append(f"README.md does not name {name}.py, whose figures the "
                         "paper quotes")
    # Section 10.3 states the same count in words, in a sentence about how many
    # models were tried - the worst possible place for a stale number, and it was
    # stale ("thirty-eight") because the guard below only ever read Section 12's
    # sentence.  Both sentences are now tied to the same computed figure.
    checks += 1
    if f"{spell(len(CITED)).capitalize()} scripts stand behind this paper" not in txt:
        fails.append(f"Section 10.3 does not say "
                     f"'{spell(len(CITED)).capitalize()} scripts stand behind this "
                     f"paper', but {len(CITED)} are checked")
    # --- the two script counts the appendix gives, reconciled -------------
    _nlabs = len([f for f in os.listdir(os.path.join(HERE, "labs"))
                  if re.match(r"lab\w+\.py$", f)])
    checks += 1
    if _nlabs <= len(CITED):
        fails.append(f"the appendix explains why {len(CITED)} scripts are quoted "
                     f"against {_nlabs} reproduced, which needs the second to be "
                     f"the larger")
    else:
        want("the appendix reconciles its two script counts",
             f"smaller than the {spell(_nlabs)} the replication reproduces: "
             f"{spell(_nlabs - len(CITED))} scripts earn their place by being run "
             f"and agreeing without a number of theirs being quoted")

    checks += 1
    if f"one of {spell(len(CITED))} scripts" not in RDME:
        fails.append(f"README.md's script count does not say "
                     f"'{spell(len(CITED))}', but {len(CITED)} are checked")
    # The gate count in the availability statement, against build_registry.py's
    # own numbered list.  It said "four gates" and listed four after a fifth was
    # added, which no figure check could see: it is a claim about the repository
    # made inside the paper, and the only place that number is defined is the
    # file that implements the gates.  Same rule as the README's copy of it.
    checks += 1
    _bp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "build_registry.py")
    if os.path.isfile(_bp):
        _ng = len(re.findall(r"^\d+\. [A-Z][A-Z ]", read_text(_bp), re.M))
        if f"runs {spell(_ng)} gates over it" not in txt:
            fails.append(f"the availability statement does not say 'runs "
                         f"{spell(_ng)} gates over it', but build_registry.py "
                         f"defines {_ng}")

    # The count in the availability statement is of the scripts run_all.py
    # reruns, which is every lab - not the smaller set the paper CITES.  It
    # said "fifty-nine" while sixty-three existed, in the one sentence telling
    # a referee what the reproduction command does.
    checks += 1
    _nlabs = len([f for f in os.listdir(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "labs"))
        if re.fullmatch(r"lab\d+[a-z]?_.*\.py", f)])
    if f"reruns all {spell(_nlabs)} scripts" not in txt:
        fails.append(f"the availability statement does not say 'reruns all "
                     f"{spell(_nlabs)} scripts', but labs/ holds {_nlabs}")

    # and the paper must still say where its numbers come from, even though the
    # list has gone: a reader of the PDF alone has to be told.
    for _label, _needle in (("availability repo", "github.com/SunnyAlexV"),
                            ("availability check command", "run_all.py --check"),
                            ("availability verifier", "verify_paper.py"),
                            ("availability seed", "under a fixed seed"),
                            ("availability platform caveat",
                             "reproduced on a second machine running a different "
                             "operating system")):
        checks += 1
        if _needle not in txt:
            fails.append(f"the paper's data and code availability statement no longer "
                         f"carries the {_label} ('{_needle}')")
    # The list must not run the other way either.  Naming a script the paper
    # quotes nothing from is how lab04 sat in this sentence for months after the
    # binary target left the paper: harmless-looking, and it makes the count
    # wrong, which is the one thing the sentence is for.
    checks += 1
    named = sorted(set(re.findall(r"lab\d+[a-z]?_[a-z_0-9]+", txt)))
    extra = [n for n in named if n not in CITED]
    if extra:
        fails.append("Section 12 names " + ", ".join(extra)
                     + ", whose figures this script checks nowhere: either check "
                       "them or stop claiming the paper's numbers come from them")
    # An exploratory lab must NOT be cited by either paper.  The list is derived
    # from the outputs themselves rather than typed here, so a lab added later
    # is covered the moment it declares itself, and nobody has to remember to
    # come back and extend a hardcoded list.
    # lab*.txt, not *.txt: in a flat layout EXP is the repository root, where
    # requirements.txt also ends in .txt and is not a lab output.
    for fn in sorted(os.listdir(EXP)):
        if not (fn.startswith("lab") and fn.endswith(".txt")):
            continue
        if "EXPLORATORY - cited by neither paper" in read_text(os.path.join(EXP, fn)):
            forbid("exploratory lab cited", fn[:-4])

    # --- claims that must stay withdrawn ----------------------------------
    # The cross-platform claim went stale once labs were added after the Windows
    # run; it must not creep back to the stronger wording.
    forbid("stale portability claim",
           "verified on two machines running different operating systems")

    for phrase in ("no amount of foreign data", "domestic series carries nothing",
                   "lies between them", "sits between them",
                   "brackets the specification",
                   "belongs to assets that have no such market",
                   "Every entry shown has p"):
        forbid("withdrawn claim", phrase)

    # --- the companion note, against lab02b -------------------------------
    # The note's ceiling table used to come from a scratch script that read the
    # author's Desktop and pasted lab02's accuracies in as literals, so nothing
    # in the repository could regenerate it and nothing could check it.  It is
    # lab02b now, and every cell of every row it prints is checked here.
    if companion:
        rows = {}
        for ln in read_text(
                os.path.join(EXP, "lab02b_threshold_ceiling.txt")).splitlines():
            m = re.match(r"\s*(\d+)\s+([-+]?[\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s+"
                         r"([\d.]+)%\s+([\d.]+)%\s*$", ln)
            if m:
                rows[int(m.group(1))] = m.groups()[1:]
        if not rows:
            fails.append("lab02b: could not parse the ceiling table at all")
        # Table 3's shape is checked in full further down, against both the
        # same-sign and the best-cut columns.  This older pass only confirms
        # that lab02b still parses and that the note has not lost a delay; it
        # no longer asserts a row layout, which changed when the benchmark
        # column was split in two.
        checks += 1
        if len(rows) != 10:
            fails.append(f"lab02b: parsed {len(rows)} of 10 rows from the ceiling "
                         "table")
        # The note must name the script that actually produces its table.
        checks += 1
        if "ceiling_check.py" in ctxt:
            fails.append("companion: still credits ceiling_check.py, which is gone")
        checks += 1
        if "lab02b_threshold_ceiling.py" not in RDME:
            fails.append("README.md does not name lab02b_threshold_ceiling.py, "
                         "whose figures the companion note quotes")

        # --- Section 6.1, against lab24 -------------------------------------
        # --- Table 3 and the exceed count, against lab02b ------------------
        # Two reviewers in a row stopped at the delta = 55 row.  The first read
        # rho as positive and called 46.17% impossible; it is -0.1202 and the
        # arithmetic was right.  The second saw the real problem underneath: at
        # rho < 0 the same-sign accuracy 1/2 + arcsin(rho)/pi falls BELOW one
        # half, so the rule is beaten by its own negation and the column headed
        # "Gaussian benchmark" named something no forecaster would accept.  The
        # benchmark is now the best cut on the delayed state, which lab02b
        # already computed, and every figure here is read out of it.
        l02b = read_text(os.path.join(EXP, "lab02b_threshold_ceiling.txt"))
        # Four columns, not three: centred, the non-centred SIGN RULE, the
        # non-centred Bayes cut, and measured.  The sign-rule column used to be
        # skipped here because Table 3 did not print it.
        _NC = re.compile(r"\n\s*(\d+)\s+[-\d.]+\s+[-\d.]+\s+([\d.]+)%\s+([\d.]+)%"
                         r"\s+([\d.]+)%\s+[\d.]+\s+([\d.]+)%")
        _nc = {m.group(1): (m.group(2), m.group(3), m.group(4), m.group(5))
               for m in _NC.finditer(l02b)}
        _RT = re.compile(r"\n\s*(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s*$", re.M)
        _rt = {m.group(1): m.group(4) for m in _RT.finditer(l02b)}
        _OLD = re.compile(r"\n\s*(\d+)\s+(-?[\d.]+)\s+([\d.]+)%\s+([\d.]+)%"
                          r"\s+([\d.]+)%\s+([\d.]+)%")
        _old = {m.group(1): m.groups()[1:] for m in _OLD.finditer(l02b)}
        checks += 1
        if not (len(_nc) == 10 and len(_rt) == 10 and len(_old) == 10):
            fails.append(f"lab02b: parsed {len(_old)} rho rows, {len(_nc)} non-centred "
                         f"rows and {len(_rt)} ratio rows; Table 3 needs 10 of each")
        else:
            for _d in sorted(_old, key=int):
                _rho, _same, _p3, _h3, _oldpc = _old[_d]
                _sameC, _sameNC, _bayes, _measured = _nc[_d]
                # The non-centred same-sign column is printed too.  It used to
                # be computed and not shown, while the prose quoted a gap
                # measured against it - so the one figure a reader was most
                # likely to check was the one they could not.
                cwant(f"Table 3 delta={_d}",
                      f"{_d} {_rho.replace('-', MINUS)} {_same} {_sameNC} {_bayes} "
                      f"{_p3} {_h3} {_rt[_d]}")
            # the benchmark column must be attainable at every delay: the best
            # cut is at least one half by construction, the same-sign figure is
            # not, and the table must not quote the second as a benchmark
            _below = [d for d in _nc if float(_nc[d][0]) < 50.0]
            checks += 1
            if not _below:
                fails.append("lab02b: no delay now has a same-sign accuracy below one "
                             "half, so Table 3's two-column caption describes a "
                             "distinction the data no longer contains")
            else:
                cwant("Table 3 caption explains the same-sign columns",
                      "Both fall below one half where &rho; &lt; 0, which means the "
                      "rule is beaten by its own negation there and neither column "
                      "can bound anything")
                # "Attainable" without a qualifier reads as attainable in the
                # data-generating process.  It is attainable under the FITTED
                # approximation only, and the note's own result proves the
                # distinction: measured accuracy exceeds the benchmark at six of
                # ten delays, which cannot happen against a true bound.
                cwant("benchmark is defined by its model",
                      "the Bayes accuracy implied by the fitted non-centred Gaussian "
                      "model, maximised over thresholds on the delayed state")
                # the emphasis tags around "under that fitted approximation"
                # leave a space before the comma once tags are stripped, so the
                # sentence is checked in two pieces rather than one
                cwant("attainable is qualified",
                      "attainable by the Bayes classifier under that fitted "
                      "approximation")
                cwant("attainable is not a bound",
                      "not a bound on what any forecaster could achieve in the "
                      "data-generating process")
                # the heading names the model compactly; the caption carries the
                # full definition, because a heading long enough to hold it
                # crushed the five numeric columns beside it
                cwant("the column heading names the model",
                      "Gaussian benchmark, best cut (fitted model)")
                cforbid("heading implies an unconditional benchmark",
                        "Gaussian benchmark, best cut persist-YZ")
                cforbid("unqualified attainability",
                        "maximises the non-centred Gaussian accuracy over the cut on "
                        "the delayed state and is attainable at every delay")
                # headroom is an estimate in one sample, not a property of the world
                cwant("headroom is an estimate",
                      "The estimated headroom is small relative to sampling "
                      "uncertainty in this sample")
                cforbid("headroom stated as fact",
                        "The headroom a better model could occupy is small relative "
                        "to sampling uncertainty,")
            checks += 1
            _bad = [d for d in _nc if float(_nc[d][2]) < 50.0]
            if _bad:
                fails.append(f"lab02b: the best-cut benchmark is below one half at "
                             f"{_bad}, which cannot happen and means the maximisation "
                             "is broken")
            # the count and the range, both read out of the lab's own summary
            _sum = re.search(r"measured accuracy runs from (\d+)% to (\d+)% of the "
                             r"Bayes benchmark, exceeding it at (\d+) of (\d+) delays",
                             l02b)
            checks += 1
            if not _sum:
                fails.append("lab02b no longer prints the Bayes-benchmark range and "
                             "count the abstract quotes")
            else:
                _lo, _hi, _n, _tot = _sum.groups()
                cwant("abstract benchmark range",
                      f"sit at {_lo}-{_hi}% of that restricted-information "
                      "Gaussian benchmark")
                cwant("abstract exceed count",
                      f"exceeding it at {spell(int(_n))} of {spell(int(_tot))} delays")
                cwant("Section 5 exceed count",
                      f"exceeds the benchmark at {spell(int(_n))} of the "
                      f"{spell(int(_tot))} delays below")
                for _w in ("seven", "eight", "nine", "five"):
                    if _w != spell(int(_n)):
                        cforbid(f"stale exceed count {_w}",
                                f"exceeding it at {_w} of ten delays")
                cforbid("stale same-sign benchmark claim",
                        "exceeds A* at seven of the ten delays below")
            # the negative-correlation delays must be named, not buried
            _neg = re.search(r"delays where the correlation is negative: (\d+)", l02b)
            checks += 1
            if not _neg:
                fails.append("lab02b no longer reports which delays have a negative "
                             "correlation")
            elif int(_neg.group(1)) > 0:
                cwant("abstract names the negative-correlation problem",
                      "the correlation turns negative at the two longest delays, where "
                      "a same-sign rule is beaten by its own negation")

        l24 = read_text(os.path.join(EXP, "lab24_proper_scores.txt"))
        # Table 4 rows, whole-row anchored: several of these figures occur
        # elsewhere in the note on their own.
        R24 = re.compile(r"\s*(\d+)\s+(persist-YZ|har-YZ|base rate)\s+([\d.]+)%"
                         r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+|n/a)\s+([\d.]+|n/a)\s*$")
        n24 = 0
        for ln in l24.splitlines():
            m = R24.match(ln)
            if m and (int(m.group(1)) in (0, 5, 21)
                      and not (m.group(2) == "base rate" and int(m.group(1)) != 21)):
                d, mod, acc, am, al, br, lg = m.groups()
                n24 += 1
                cwant(f"lab24 d={d} {mod} row",
                      f"{acc}% {am} {al} {br} {lg}")
        checks += 1
        if n24 != 7:
            fails.append(f"lab24: matched {n24} rows for the note's Table 4, expected 7")
        # The delta = 0 AUC gap had its POINT estimate checked against lab24 and
        # its INTERVAL written out as a literal, so the note quoted
        # [-0.0542, -0.0066] against the lab's [-0.0590, -0.0026] and passed.
        # The whole row is derived now.
        _a24 = re.search(r"\n\s+0\s+[-+][\d.]+\s+\[[^\]]*\]\s+(-?[\d.]+)\s+"
                         r"\[(-?[\d.]+),\s*(-?[\d.]+)\]", l24)
        checks += 1
        if not _a24:
            fails.append("lab24 no longer prints the delta = 0 AUC row the note quotes")
        else:
            cwant("lab24 AUC gap at d=0",
                  f"the gap is {_a24.group(1)} with a block-bootstrap interval of "
                  f"[{_a24.group(2)}, {_a24.group(3)}]")
        for tok, phrase, label in (
                ("0.055", "by as much as 0.055 of AUC at &delta; = 0",
                 "lab24 discretising cost"),):
            checks += 1
            if tok not in l24:
                fails.append(f"{label}: '{tok}' not in lab24 output")
            cwant(label, phrase)
        # How MANY delays separate, and which.  The note said seven of ten and
        # that the pattern "holds out to delta = 13"; lab24 separates five, at
        # delta = 0, 1, 2, 3 and 5, and covers zero from delta = 8 on.  Both the
        # count and the list are read out of the lab, and the orientation of the
        # contrast has to be stated, because a negative number favouring the
        # threshold rule is unreadable without it.
        _sep24 = re.search(r"separated by either measure: (\d+) of (\d+)", l24)
        _fav24 = re.search(r"favouring the SIMPLE rule: (\d+)\s+\[([\d, ]*)\]", l24)
        checks += 1
        if not (_sep24 and _fav24):
            fails.append("lab24 no longer prints the separation count the note quotes")
        else:
            cwant("lab24 separation count",
                  f"they separate at {spell(int(_sep24.group(1)))} of "
                  f"{spell(int(_sep24.group(2)))} delays")
            _ds = [d.strip() for d in _fav24.group(2).split(",") if d.strip()]
            checks += 1
            if len(_ds) != int(_sep24.group(1)):
                fails.append(f"lab24: {len(_ds)} delays favour the simple rule but "
                             f"{_sep24.group(1)} are separated; the note names one list")
            cwant("lab24 separation list",
                  "&delta; = " + ", ".join(_ds[:-1]) + " and " + _ds[-1])
        # --- the note's Tables 1 and 2, against lab02 -----------------------
        # Neither table was checked against anything.  Both carried the
        # intervals of the shorter block the note's methods section used to
        # describe, while their point estimates had been re-synced, so the
        # note tabulated accuracies from one run beside intervals from
        # another and its abstract quoted the older pair.  Every cell of both
        # tables is rebuilt from lab02 here.
        _l02 = read_text(os.path.join(EXP, "lab02_delay_curve.txt"))
        _T1 = re.compile(r"\n\s*(\d+)\s+((?:[\d.]+ \[\s*[\d.]+,\s*[\d.]+\]\s+){4}"
                         r"[\d.]+ \[\s*[\d.]+,\s*[\d.]+\])")
        _t1 = {int(m.group(1)): re.findall(r"([\d.]+) \[\s*([\d.]+),\s*([\d.]+)\]",
                                           m.group(2))
               for m in _T1.finditer(_l02)}
        checks += 1
        if len(_t1) != 10:
            fails.append(f"lab02: parsed {len(_t1)} of 10 accuracy rows for Table 1")
        for _d in sorted(_t1):
            _cells = _t1[_d]
            cwant(f"Table 1 delta={_d}",
                  f"{_d} {_cells[0][0]} " + " ".join(
                      f"{a} [{lo}, {hi}]" for a, lo, hi in _cells[1:]))
        _blk2 = _l02.split("PAIRED TEST vs the persistence baseline")[-1]
        _T2 = re.compile(r"\n\s*(\d+)\s+((?:[-+][\d.]+ \[\s*[-+][\d.]+,\s*[-+][\d.]+\]"
                         r"\s+){2}[-+][\d.]+ \[\s*[-+][\d.]+,\s*[-+][\d.]+\])")
        _t2 = {int(m.group(1)): re.findall(
            r"([-+][\d.]+) \[\s*([-+][\d.]+),\s*([-+][\d.]+)\]", m.group(2))
            for m in _T2.finditer(_blk2)}
        checks += 1
        if len(_t2) != 10:
            fails.append(f"lab02: parsed {len(_t2)} of 10 paired rows for Table 2")

        # ctxt is plain()-normalised, where &minus; has already become "-", so
        # these needles use the ASCII sign rather than the typographic one.
        def _sg(x):
            return x

        for _d in sorted(_t2):
            if f" {_d} " not in " ".join(re.findall(r"\n\s*(\d+)\s", _blk2)) and False:
                continue
            _row = f"{_d} " + " ".join(f"{_sg(a)} [{_sg(lo)}, {_sg(hi)}]"
                                       for a, lo, hi in _t2[_d])
            # the note tabulates a subset of delays; a delay it shows must match,
            # one it omits is not required to appear
            checks += 1
            if f"{_d} {_sg(_t2[_d][0][0])} [" in ctxt and _row not in ctxt:
                fails.append(f"Table 2 delta={_d}: lab02 no longer supports the row "
                             f"'{_row}'")
        # Table 2's own caption makes a claim about every cell
        checks += 1
        _exc2 = [d for d in _t2 if any(float(lo) > 0 or float(hi) < 0
                                       for _a, lo, hi in _t2[d])]
        if _exc2:
            fails.append(f"lab02: paired differences now exclude zero at {_exc2}, but "
                         "Table 2's caption says no interval does")
        # the abstract quotes two of these cells by name
        if 0 in _t2:
            # column order is har-CC - persist-CC, har-YZ - persist-YZ,
            # persist-YZ - persist-CC; the abstract's "Yang-Zhang proxy" figure
            # is the THIRD of those, the proxy contrast, not the second
            _yz, _har = _t2[0][2], _t2[0][0]
            cwant("abstract YZ cell",
                  f"(+2.0 points at zero delay, 95% CI [{_sg(_yz[1])}, {_yz[2]}])")
            cwant("abstract HAR cell",
                  f"(-4.0 points, [{_sg(_har[1])}, {_har[2]}])")
            # and Section 4.2's equivalence bound is the largest upper endpoint
            _mx = max((float(hi), d, i) for d in _t2
                      for i, (_a, _lo, hi) in enumerate(_t2[d]))
            cwant("4.2 equivalence bound",
                  f"at &delta; = {_mx[1]}, leaves a HAR advantage of up to {_mx[0]:.1f} "
                  "accuracy points unexcluded")

        # --- Section 4.1's two constant-class hurdles, against lab20 --------
        # The note used to call 53.2% "the constant-forecast level" and justify
        # it by saying two of three constructions agree, which counts the
        # hindsight rule as a vote it is not entitled to.  Only two constants
        # are available to a forecaster, they differ by 6.3 points, and lab20
        # part C2 shows the gap changes the answer for all four forecasters.
        #
        # C2 determines each crossing over MANY bootstrap seeds, because the
        # first version of it used one and reported delta = 5 as the threshold
        # rule's crossing against the higher hurdle; that delay clears in 1 seed
        # in 20.  Every crossing the note quotes is therefore the FIRM one, and
        # the unresolved delays are quoted with their seed counts.
        l20 = read_text(os.path.join(EXP, "lab20_ceiling_is_not_a_ceiling.txt"))
        _c2 = l20.split("C2.  DOES THE CHOICE OF CONSTANT")[-1].split("\nD.  ")[0]
        _h20 = re.search(r"above ([\d.]+)%\s+above ([\d.]+)%", _c2)
        _CELL = r"(?:through delta = \d+|never|(?:\d+|never) firm, \d+ knife-edge)"
        _X20 = re.compile(r"\s*(persist-CC|persist-YZ|har-CC|har-YZ)\s+"
                          r"(" + _CELL + r")\s{2,}(" + _CELL + r")\s*$", re.M)
        _x20 = {m.group(1): (m.group(2).strip(), m.group(3).strip())
                for m in _X20.finditer(_c2)}
        checks += 1
        if not (_h20 and len(_x20) == 4):
            fails.append(f"lab20 part C2: parsed {len(_x20)} of 4 forecaster rows "
                          "for Section 4.1's hurdle comparison")
        else:
            _hi, _lo20 = _h20.group(1), _h20.group(2)
            checks += 1
            if float(_hi) <= float(_lo20):
                fails.append(f"lab20: the rolling-window constant {_hi}% is no longer "
                             f"above the expanding-window one {_lo20}%, so calling "
                             "the first conservative is backwards")
            _hiR, _loR = f"{float(_hi):.1f}", f"{float(_lo20):.1f}"
            cwant("4.1 conservative hurdle named",
                  "a conservative constant-class hurdle")
            cwant("4.1 higher hurdle", f"the conservative {_hiR}% hurdle")

            def _firm(cell):
                """The FIRM crossing delay in a C2 cell, or None."""
                m = re.match(r"through delta = (\d+)$", cell)
                if m:
                    return m.group(1)
                m = re.match(r"(\d+) firm, \d+ knife-edge$", cell)
                return m.group(1) if m else None

            _ty, _te = _firm(_x20["persist-YZ"][0]), _firm(_x20["persist-YZ"][1])
            checks += 1
            if not (_ty and _te):
                fails.append("lab20 part C2: the threshold rule no longer has a firm "
                             "crossing against both hurdles, which Section 4.1 quotes")
            else:
                cwant("4.1 threshold rule both hurdles",
                      f"entirely above it through &delta; = {_ty}; against the {_loR}% "
                      f"expanding-window rule it does so through &delta; = {_te}")
                cwant("4.1 crossing is the all-seed one",
                      "stated at the last delay that clears in every bootstrap seed")
            # the knife-edge cells, quoted with their seed counts
            _NS = re.search(r"bootstrap seeds per cell: (\d+)", _c2)
            _kn = dict()
            for m in re.finditer(r"(persist-CC|persist-YZ|har-CC|har-YZ) at ([\d.]+)%: "
                                 r"delta = (\d+) clears in (\d+) of (\d+) seeds", _c2):
                _kn[(m.group(1), m.group(2))] = (m.group(3), m.group(4), m.group(5))
            checks += 1
            if not _NS:
                fails.append("lab20 part C2 no longer reports its seed count")
            elif ("persist-YZ", _hi) in _kn and ("persist-YZ", _lo20) in _kn:
                _a, _ka, _n = _kn[("persist-YZ", _hi)]
                _b, _kb, _ = _kn[("persist-YZ", _lo20)]
                cwant("4.1 knife-edge delays",
                      f"At &delta; = {_a} the lower endpoint sits within a quarter of a "
                      f"point of the higher hurdle and clears it in {_ka} of {_n} seeds; "
                      f"at &delta; = {_b} it clears the lower hurdle in {_kb} of {_n}")
            else:
                fails.append("lab20 part C2: the threshold rule's crossings are no "
                             "longer knife-edge at either hurdle, but Section 4.1 "
                             "names two delays as unresolved")
            # the other three, which the paragraph names to show the shift is general
            checks += 1
            if _firm(_x20["persist-CC"][0]) != _firm(_x20["har-YZ"][0]):
                fails.append("lab20: persist-CC and har-YZ no longer share a firm "
                             "crossing against the higher hurdle, which Section 4.1 pairs")
            cwant("4.1 other forecasters",
                  f"persist-CC and har-YZ run through &delta; = "
                  f"{_firm(_x20['persist-CC'][0])} against the higher hurdle and "
                  f"&delta; = {_firm(_x20['persist-CC'][1])} against the lower")
            checks += 1
            if _firm(_x20["har-CC"][0]) is not None:
                fails.append("lab20: har-CC now clears the higher hurdle firmly "
                             "somewhere, but Section 4.1 says it never does")
            elif ("har-CC", _hi) in _kn:
                _d0, _k0, _n0 = _kn[("har-CC", _hi)]
                cwant("4.1 har-CC never clears",
                      f"har-CC clears the higher one at no delay firmly, clearing even "
                      f"&delta; = {_d0} in only {_k0} of {_n0} seeds, while clearing "
                      f"the lower through &delta; = {_firm(_x20['har-CC'][1])}")
            _n20 = re.search(r"FIRM answer depends on which constant is used: "
                             r"(\d+) of (\d+)", _c2)
            checks += 1
            if not _n20:
                fails.append("lab20 part C2 no longer reports how many forecasters the "
                             "choice of constant moves")
            elif _n20.group(1) != _n20.group(2):
                fails.append(f"lab20: the choice of constant now moves {_n20.group(1)} "
                             f"of {_n20.group(2)} forecasters, but Section 4.1 says all")
            else:
                cwant("4.1 all four move", "changes the answer for all four")
        # the superseded justification must not come back
        for _st in ("two of the three constructions agree on it, and report",
                    "Taking 53.2% as the reference",
                    "the 53.2% constant-forecast hurdle"):
            cforbid("4.1 stale hurdle justification", _st)

        # --- what the note may say about its own block, against lab61 ------
        # The note's methods section used to defend the block length with two
        # arguments the companion paper has since withdrawn: a retained-variance
        # measurement that moves from 89% to 68% with an unstated reference
        # truncation, and a "coverage experiment puts the optimum at 16h" that
        # no coverage experiment found (coverage was still improving at 160).
        # Worse, Table 7's caption said the note's own tables used a flat block
        # of 40 while every interval in them is resampled at block_for(delta),
        # 40 rising to 80 -- the caption contradicted the code beneath it, and
        # nothing here could see it because no check read lab61's own statement
        # of what the note uses.  Now one does.
        _l61hdr = read_text(os.path.join(EXP, "lab61_block_sensitivity.txt"))
        _mnb = re.search(r"the note's block: block_for\(delta\), (\d+) at delta "
                         r"(\d+) to (\d+) at delta (\d+)", _l61hdr)
        checks += 1
        if not _mnb:
            fails.append("lab61 no longer states which block the note uses, so the "
                         "note's description of it rests on nothing")
        else:
            _lo61, _hi61 = _mnb.group(1), _mnb.group(3)
            # plain() strips tags but leaves &delta; alone, so the needles
            # carry the entity exactly as the source does.
            cwant("3 note states its own block",
                  f"running 8h = {_lo61} at &delta; = 0 to 16h = {_hi61} at "
                  f"eleven weeks")
            cwant("6.4 caption states the block the tables use",
                  f"the delay-dependent block, {_lo61} at &delta; = 0 rising to "
                  f"{_hi61} at eleven weeks")
        cforbid("6.4 caption claims a flat block the tables do not use",
                "own tables use mb 40")
        for _lbl, _gone in (
                ("3 retired retained-variance defence",
                 "retaining only about three quarters of the long-run variance"),
                ("3 retired coverage-optimum claim",
                 "where a coverage experiment puts the optimum"),
                ("3 retired bias-variance cap argument",
                 "32h would leave only thirty distinct blocks"),
                ("3 retired analytic coverage figure",
                 "turns a nominal 95% interval into roughly a 91% one")):
            cforbid(_lbl, _gone)
        cwant("3 says the length is shown to be worth what it is worth",
              "the length here is defended the only way it can be: by showing what "
              "it is worth")
        # The note's Table 2, the paired accuracy differences, against lab02.
        # A tamper audit found its first result cell unread, which for a table
        # whose caption asserts "No interval excludes zero" leaves the whole
        # claim unguarded.  The note prints a SUBSET of lab02's delays, so the
        # rows it prints are read off the note itself and their values demanded
        # from the lab: that pins every printed cell without requiring the note
        # to print delays it deliberately omits.
        _l02p = read_text(os.path.join(EXP, "lab02_delay_curve.txt"))
        _p02 = {int(_m.group(1)): _m.groups()[1:] for _m in re.finditer(
            r"^\s*(\d+)" +
            r"\s+([+-][\d.]+) \[\s*([+-][\d.]+),\s*([+-][\d.]+)\]" * 3,
            _l02p, re.M)}
        _nt2 = re.search(r"<b>Table 2\.</b>(.{0,3000}?)</table>",
                         read_text(companion), re.S) if companion else None
        checks += 1
        if not (_p02 and _nt2):
            fails.append("lab02's paired-difference table or the note's Table 2 is "
                         "missing, so the note's Table 2 rests on nothing")
        else:
            _shown = [int(_m.group(1)) for _m in re.finditer(
                r"<tr[^>]*><td>(\d+)</td><td>[^<]*</td>", _nt2.group(1))]
            checks += 1
            if not _shown:
                fails.append("the note's Table 2 has no parsable rows")
            _excl = False
            for _d in _shown:
                checks += 1
                if _d not in _p02:
                    fails.append(f"the note's Table 2 prints delay {_d}, which "
                                 f"lab02 does not compute")
                    continue
                _g = _p02[_d]
                for _k in range(3):
                    _pt, _lo, _hi = _g[3 * _k], _g[3 * _k + 1], _g[3 * _k + 2]
                    cwant(f"note Table 2 d={_d} col{_k + 1}",
                          f"{_pt} [{_lo}, {_hi}]")
                    if float(_lo) > 0 or float(_hi) < 0:
                        _excl = True
            checks += 1
            if _excl:
                fails.append("a paired accuracy difference in the note's Table 2 "
                             "now has an interval excluding zero, but its caption "
                             "says none does")
            cwant("note Table 2 caption's assertion", "No interval excludes zero")

        # The note's Table 6, the Gaussian benchmark by quantile, against
        # lab26.  A tamper audit found its first result cell unread.  Rows are
        # demanded whole: the majority column is q * 100 and repeats across
        # blocks, so a cell needle would pass on another row.
        _l26q = read_text(os.path.join(EXP, "lab26_quantile_target.txt"))
        _q26 = re.findall(r"^\s*(0\.\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+"
                          r"([\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s*$",
                          _l26q, re.M)
        checks += 1
        if len(_q26) < 4:
            fails.append(f"lab26: read {len(_q26)} quantile rows, so the note's "
                         f"Table 6 rests on nothing")
        else:
            for _r in [_x for _x in _q26 if _x[2] == "0"]:
                cwant(f"note Table 6 q={_r[0]} row",
                      f"{_r[0]} {_r[1]} {_r[4]} {float(_r[0]) * 100:.1f}% "
                      f"{_r[5]}% {_r[6]}%")

        # three note figures no needle mentioned, found by differencing the
        # verifier's own demanded needles against every percentage in it.
        #
        # the majority rule's accuracy, which the note says the others converge on
        _l02m = read_text(os.path.join(EXP, "lab02_delay_curve.txt"))
        _m02m = re.search(r"^\s*0\s+([\d.]+) \[", _l02m, re.M)
        checks += 1
        if not _m02m:
            fails.append("lab02 no longer prints the majority rule's accuracy")
        else:
            # The paragraph that quoted this said "Every column falls with
            # delay" and "all start near 66-69%" and "converge on the majority
            # rule's 46.83% by three weeks", and the note's own Table 1 refutes
            # all three: the majority column is flat by construction, two
            # columns rise once, har-CC starts at 61.82, and at three weeks
            # har-CC is seven points above the majority rule.  The rewritten
            # paragraph is checked against the table instead.
            _T1 = {}
            for _ln in _l02m.splitlines():
                _m = re.match(r"^\s*(\d+)\s+([\d.]+) \[[^\]]*\]\s+([\d.]+) \[[^\]]*\]"
                              r"\s+([\d.]+) \[[^\]]*\]\s+([\d.]+) \[[^\]]*\]"
                              r"\s+([\d.]+) \[", _ln)
                if _m:
                    _T1[int(_m.group(1))] = tuple(_m.groups()[1:])
            checks += 1
            if len(_T1) < 10:
                fails.append(f"lab02: read {len(_T1)} of 10 delay rows, so the "
                             f"note's Section 4.1 paragraph rests on a short read")
            else:
                _d0 = min(_T1)
                _maj = _T1[_d0][0]
                _four = [float(_T1[_d0][i]) for i in (1, 2, 3, 4)]
                # the rises, counted
                _ds1 = sorted(_T1)
                _rises = [(i, a, b) for i in (1, 2, 3, 4)
                          for a, b in zip(_ds1, _ds1[1:])
                          if float(_T1[b][i]) > float(_T1[a][i])]
                checks += 1
                if len(_rises) != 2:
                    fails.append(f"the note says two of the four forecasters rise "
                                 f"once across the grid; lab02 now gives "
                                 f"{len(_rises)} rises")
                cwant("4.1 the majority rule's level",
                      f"the majority column is a constant {_maj}% at every delay "
                      f"by construction")
                cwant("4.1 the two columns that rise",
                      f"har-CC from {_T1[13][3]} to {_T1[21][3]} between thirteen "
                      f"and twenty-one days and persist-CC from {_T1[21][1]} to "
                      f"{_T1[34][1]} between twenty-one and thirty-four")
                cwant("4.1 the zero-delay range, and the column outside it",
                      f"start between {min(x for i, x in enumerate(_four) if i != 2):.2f}% "
                      f"and {max(_four):.2f}% at zero delay, har-CC lower at "
                      f"{_T1[_d0][3]}%")
                _d55 = max(_T1)
                _gap55 = max(abs(float(_T1[_d55][i]) - float(_maj))
                             for i in (1, 2, 3, 4))
                cwant("4.1 where the four actually converge",
                      f"by eleven weeks all four sit within "
                      f"{_num_word(int(_gap55) + 1)} points of the majority rule")
                cwant("4.1 and where they do not",
                      f"har-CC is {_num_word(round(float(_T1[21][3]) - float(_maj)))} "
                      f"above it")
                forbid_note_41 = "Every column falls with delay"
                checks += 1
                if forbid_note_41 in ctxt:
                    fails.append("the note says every column falls with delay; the "
                                 "majority column is flat and two others rise")
        # the Bayes rule and the majority class at the 90th percentile, which is
        # the note's tail result
        _l26t = read_text(os.path.join(EXP, "lab26_quantile_target.txt"))
        _m26t = re.search(r"^\s*0\.90\s+[\d.]+\s+0\s+[\d.]+\s+[\d.]+\s+"
                          r"([\d.]+)%", _l26t, re.M)
        checks += 1
        if not _m26t:
            fails.append("lab26 no longer prints the Bayes rule at the 90th "
                         "percentile, so the note's tail result rests on nothing")
        else:
            cwant("the tail result's two levels",
                  f"attains only {float(_m26t.group(1)):.1f}% against a "
                  f"majority-class 90.0%")
        # and the retained-variance range the note now cites when it withdraws
        # the argument it used to make from that measurement
        _l65n = read_text(os.path.join(EXP, "lab65_hac_coverage_real.txt"))
        _r65 = [float(_m.group(1)) for _m in re.finditer(
            r"^\s*\d+\s+[\d.]+%\s+([\d.]+)%", _l65n, re.M)]
        checks += 1
        if len(_r65) < 4:
            fails.append(f"lab65: read {len(_r65)} denominator retained fractions, "
                         f"so the note's withdrawn-argument sentence rests on "
                         f"nothing")
        else:
            cwant("3 the retained-variance range the note withdraws",
                  f"runs from {round(max(_r65))}% to {round(min(_r65))}% depending "
                  f"on a reference truncation")

        # The note's decay sentence, "Accuracy falls from about 68% at zero
        # delay", was a literal: tamper-testing it changed the number and every
        # check still passed.  It is the threshold rule's own accuracy at delay
        # zero, which is lab02's persist-YZ column, so it is read from there.
        _l02d = read_text(os.path.join(EXP, "lab02_delay_curve.txt"))
        _m02 = re.search(r"^\s*0\s+[\d.]+ \[[^\]]*\]\s+[\d.]+ \[[^\]]*\]\s+"
                         r"([\d.]+) \[", _l02d, re.M)
        checks += 1
        if not _m02:
            fails.append("lab02 no longer prints the threshold rule's accuracy at "
                         "delay zero, so the note's decay sentence rests on nothing")
        else:
            cwant("4.1 the decay starts where lab02 says it does",
                  f"Accuracy falls from about {round(float(_m02.group(1)))}% at "
                  f"zero delay")

        # --- Section 6.4 and Table 6, against lab61 -------------------------
        # The note's inference has one free parameter and reported no
        # sensitivity to it, while its methods section named a block of 10 and
        # its tables were produced at 40.  lab61 varies the block over 10, 20
        # and 40 under two bootstrap families; every cell of Table 6 is read out
        # of its summary lines, so the table cannot drift from the run and the
        # note cannot quote a count from a block it does not use.
        l61 = read_text(os.path.join(EXP, "lab61_block_sensitivity.txt"))
        _B61 = re.compile(r"(mb|sb) block\s+(\d+): (accuracy|AUC) differences "
                          r"excluding zero: (\d+) of (\d+)(?:\s+\[([\d, ]*)\])?"
                          r"(\s+all favour the threshold rule)?")
        _r61 = {}
        for _m in _B61.finditer(l61):
            _r61[(_m.group(1), int(_m.group(2)), _m.group(3))] = (
                _m.group(4), _m.group(5), _m.group(6), bool(_m.group(7)))
        # The expected count is READ from lab61's own block list, not fixed at
        # 12.  It was 12 because the sweep had three lengths and two schemes;
        # adding a fourth length made the file emit 16 and the check fail for
        # the one reason that is not a mistake - the lab did more work.
        _bl61 = re.search(r"^BLOCKS\s*=\s*\[([0-9,\s]+)\]",
                          read_text(os.path.join(HERE, "labs",
                                                 "lab61_block_sensitivity.py")), re.M)
        # Four lines per block length: two schemes (moving and stationary)
        # times two statistics (accuracy and AUC).  Three lengths gave the 12
        # this was pinned at.
        _nb = 4 * len([x for x in _bl61.group(1).split(",") if x.strip()]) \
            if _bl61 else 12
        checks += 1
        if len(_r61) != _nb:
            fails.append(f"lab61: parsed {len(_r61)} of {_nb} summary lines for "
                         f"Table 6 ({_nb // 4} block lengths x 2 schemes x 2 "
                         f"statistics)")
        else:
            for _k, _b in (("mb", 10), ("sb", 10), ("mb", 20),
                           ("sb", 20), ("mb", 40), ("sb", 40)):
                _ac, _tot, _, _ = _r61[(_k, _b, "accuracy")]
                _au, _tot2, _lst, _all = _r61[(_k, _b, "AUC")]
                checks += 1
                if _ac != "0":
                    fails.append(f"lab61: {_k} block {_b} now separates {_ac} delays on "
                                 "accuracy; Table 6 says none do under any scheme")
                checks += 1
                if not _all:
                    fails.append(f"lab61: {_k} block {_b} has a significant AUC cell "
                                 "favouring the HAR classifier; Table 6 says none do")
                _cells = ", ".join(x.strip() for x in (_lst or "").split(",")
                                   if x.strip())
                cwant(f"Table 6 {_k}{_b}",
                      f"{_k} {_b} {_ac} of {_tot} {_au} of {_tot2} {_cells} yes")
            # the two facts the paragraph draws out of the table
            _at10 = max(int(_r61[(k, 10, "AUC")][0]) for k in ("mb", "sb"))
            _at40 = max(int(_r61[(k, 40, "AUC")][0]) for k in ("mb", "sb"))
            checks += 1
            if not _at40 <= _at10:
                fails.append(f"lab61: the AUC count RISES with the block ({_at10} at "
                             f"10, {_at40} at 40); Section 6.4 says it falls")
            cwant("6.4 count falls with the block",
                  f"It stands at {spell(_at10)} of ten delays with a block of 10")
            cwant("6.4 reported count",
                  f"We therefore report {spell(_at40)}, the count at the block this "
                  "note uses")
            # The scheme count is read, not fixed at six: lab61's sweep has
            # eight and its own narration said six, which is the defect the
            # companion-note audit found in this very line.  The note gets it
            # right, so a pattern that insisted on "six" would have failed the
            # corrected lab.
            _tot61 = re.search(r"of (\d+) significant AUC cells across all "
                               r"(\S+) schemes, (\d+) favour the threshold rule",
                               l61)
            checks += 1
            if not _tot61:
                fails.append("lab61 no longer prints the direction tally Section 6.4 "
                             "quotes")
            elif _tot61.group(1) != _tot61.group(3):
                fails.append(f"lab61: {_tot61.group(3)} of {_tot61.group(1)} "
                             "significant AUC cells favour the threshold rule, but "
                             "Section 6.4 says all of them do")
            else:
                cwant("6.4 names the number of schemes",
                      f"Across the {_tot61.group(2)} schemes")
                cwant("6.4 direction tally",
                      f"there are {spell(int(_tot61.group(1)))} significant AUC cells "
                      f"and all {spell(int(_tot61.group(3)))} favour the threshold rule")
            # The note must report the block it actually ran, and that is no
            # longer a single number: its intervals are resampled at
            # block_for(delta), so the note states a RANGE.  This check used to
            # build the needle as "8h = {number}", hardcoding the multiple
            # while reading the number - which would have demanded the
            # self-contradictory "8h = 80" the moment the length moved.  Both
            # ends are read now.
            _nb61 = re.search(r"the note's block: block_for\(delta\), (\d+) at "
                              r"delta (\d+) to (\d+) at delta (\d+)", l61)
            checks += 1
            if not _nb61:
                fails.append("lab61 no longer reports the note's own block length")
            else:
                _lo61, _hi61 = _nb61.group(1), _nb61.group(3)
                _h61 = re.search(r"^HORIZON\s*=\s*(\d+)", read_text(
                    os.path.join(HERE, "labs", "lab02_delay_curve.py")), re.M)
                _hh = int(_h61.group(1)) if _h61 else 5
                cwant("3.4 block length",
                      f"running {int(_lo61) // _hh}h = {_lo61} at &delta; = 0 to "
                      f"{int(_hi61) // _hh}h = {_hi61} at eleven weeks")
                cforbid("3.4 stale block length", "block length 2h = 10")

        cwant("lab24 contrast orientation",
              "the HAR classifier minus the threshold rule, so a negative number is "
              "the threshold rule ahead")
        for _stale in ("separate at seven of ten delays",
                       "[-0.0542, -0.0066]",
                       "the pattern holds out to &delta; = 13"):
            cforbid("lab24 stale AUC claim", _stale)
        checks += 1
        if "significant AUC gaps favouring the HAR classifier: 0" not in l24:
            fails.append("lab24: the HAR classifier now wins somewhere, but the note "
                         "still says it never does")
        cwant("lab24 direction", "in the threshold rule's favour every time")
        # the Brier deterioration claim: the note names delays 21, 34 and 55
        l24c = l24.split("C.  THE ONE THING ACCURACY HIDES")[-1]
        worse = []
        for ln in l24c.splitlines():
            m = re.match(r"\s*(\d+)\s+[\d.]+\s+[\d.]+\s+\+([\d.]+)\s+"
                         r"\[\s*\+([\d.]+),", ln)
            if m:
                worse.append(int(m.group(1)))     # diff > 0 and interval lower bound > 0
        # Which delays deteriorate significantly is a consequence of the
        # interval width, so widening the bootstrap block changed the list.  It
        # is read out of lab24 now, and the note has to name exactly that list.
        checks += 1
        if not worse:
            fails.append("lab24 no longer shows a significant Brier deterioration "
                         "at any delay, but the note claims one")
        else:
            _w = (" and ".join([", ".join(str(x) for x in worse[:-1]), str(worse[-1])])
                  if len(worse) > 1 else str(worse[0]))
            cwant("lab24 Brier deterioration",
                  f"significantly so at &delta; = {_w}")

        # --- Section 6.2, against lab25 -------------------------------------
        l25 = read_text(os.path.join(EXP, "lab25_window_sensitivity.txt"))
        checks += 1
        if "significantly BEATS the rule: 0 of 40" not in l25:
            fails.append("lab25: the HAR classifier now beats the rule in some cell, "
                         "but Section 6.2 still claims none")
        # Both counts READ from lab25.  The note said "fifteen" where lab25
        # says 16 of 40, and this check pinned the fifteen.
        _c25b = re.search(r"rule significantly beats the HAR classifier: (\d+) of",
                          l25)
        checks += 1
        if not _c25b:
            fails.append("lab25 no longer prints how many cells the rule wins")
        else:
            cwant("lab25 headline",
                  f"the HAR classifier significantly beats the threshold rule in "
                  f"none, and the threshold rule significantly beats the HAR "
                  f"classifier in {spell(int(_c25b.group(1)))}")
        # The power caveat.  A cwant pinned "moves that boundary back out from
        # zero to three days" - the NEGATION of lab25's own verdict line, "The
        # boundary does not move out when the sample grows, so part C is not
        # reporting a power loss".  The check therefore enforced the error.
        checks += 1
        _m25b = re.search(r"boundary on the common window: (\d+)\s+boundary on its "
                          r"own window: (\d+)", l25)
        if not _m25b:
            fails.append("lab25 no longer prints both boundaries, so the note's "
                         "power caveat rests on nothing")
        else:
            _moved25 = int(_m25b.group(2)) > int(_m25b.group(1))
            checks += 1
            if _moved25:
                fails.append(f"lab25's boundary now moves out from {_m25b.group(1)} "
                             f"to {_m25b.group(2)}, so the note's paragraph saying "
                             f"it does not has to be rewritten")
            else:
                cwant("lab25 power caveat",
                      "It is not a loss of power, and an earlier version of this "
                      "note said it was")
                cwant("lab25 power caveat keeps the boundary where lab25 puts it",
                      f"leaves the boundary at {spell(int(_m25b.group(2)))}")
                checks += 1
                if "moves that boundary back out" in ctxt:
                    fails.append("the note says the boundary moves out, which lab25 "
                                 "explicitly denies")
        checks += 1
        _m25 = re.search(r"boundary on its own window: (\d+)", l25)
        if not _m25:
            fails.append("lab25 no longer prints the power control's boundary")

        # --- the note figures a cold audit found unpinned -------------------
        # Each was wrong and is corrected; none was read by any check, which is
        # why a reversion of each went unnoticed in the first tamper pass.
        # The HAR bound the abstract quotes, from lab02's own Table 2 row.
        _l02b = read_text(os.path.join(EXP, "lab02_delay_curve.txt"))
        _m02b = re.search(r"^\s*21\s+[+-][\d.]+ \[\s*-?[\d.]+,\s*\+([\d.]+)\]",
                          _l02b, re.M)
        checks += 1
        if not _m02b:
            fails.append("lab02 no longer prints the three-week paired difference, "
                         "so the note's abstract quotes an unsupported bound")
        else:
            cwant("the note's abstract HAR bound",
                  f"a HAR advantage of up to {_m02b.group(1)} accuracy points "
                  f"unexcluded")
            # The bound is quoted TWICE, in the abstract and in Section 4.2,
            # and a cwant is satisfied by either, so one copy could go stale
            # unseen - which it had: the abstract said 12.5 against the
            # section's 12.7.  The occurrences are counted.
            _n02b = ctxt.count("accuracy points unexcluded")
            _n02ok = ctxt.count(f"up to {_m02b.group(1)} accuracy points unexcluded")
            checks += 1
            if _n02b != _n02ok:
                fails.append(f"the note quotes the HAR bound {_n02b} times and only "
                             f"{_n02ok} of them say {_m02b.group(1)}, which is "
                             f"lab02's three-week upper endpoint")
        # the Yang-Zhang column's margin, which the note said falls after zero
        # delay and which lab02 gives as equal at zero and thirteen days
        _yz = {int(_m.group(1)): _m.group(2) for _m in re.finditer(
            r"^\s*(\d+)\s+[+-][\d.]+ \[[^\]]*\]\s+[+-][\d.]+ \[[^\]]*\]"
            r"\s+([+-][\d.]+) \[", _l02b, re.M)}
        checks += 1
        if not ({0, 13} <= set(_yz)):
            fails.append("lab02 no longer prints the Yang-Zhang margin at zero and "
                         "thirteen days")
        elif _yz[0] == _yz[13]:
            cwant("4.2 does not claim the proxy margin falls after zero delay",
                  f"worth about two accuracy points at zero delay and the same "
                  f"{_yz[13].lstrip('+')} at thirteen")
        # Section 6.3's balanced-accuracy interval and episode count, from lab26
        _l26n = read_text(os.path.join(EXP, "lab26_quantile_target.txt"))
        _m26b = re.search(r"q = 0\.90.*?\n\s*0\s+note a>0\s+[\d.]+%\s+[\d.]+%"
                          r"\s+([\d.]+)\s+\[\s*([\d.]+),\s*([\d.]+)\]",
                          _l26n, re.S)
        checks += 1
        if not _m26b:
            fails.append("lab26 no longer prints the q = 0.90 zero-delay row, so "
                         "Section 6.3's interval rests on nothing")
        else:
            cwant("6.3 quotes lab26's balanced-accuracy interval",
                  f"balanced accuracy is {_m26b.group(1)} with an interval of "
                  f"[{_m26b.group(2)}, {_m26b.group(3)}]")
        _m26e = re.search(r"^\s*0\.90\s+(\d+)\s+(\d+)\s+", _l26n, re.M)
        checks += 1
        if not _m26e:
            fails.append("lab26 no longer prints the positive-day and episode "
                         "counts at the 90th percentile")
        else:
            cwant("6.3 quotes lab26's episode count",
                  f"{_m26e.group(1)} positive days which, with five-day overlap, "
                  f"resolve into {_num_word(int(_m26e.group(2)))} independent "
                  f"episodes")
        # the last delay at which the rule beats a constant, per median window,
        # read from lab25: the note said "zero days at 252 and 504" where the
        # lab says 0 days at 252 and NEVER at 504
        _w25 = dict(re.findall(r"^\s*(\d+)\s{2,}(\d+ days|never)\s*$", l25, re.M))
        checks += 1
        if len(_w25) < 4:
            fails.append(f"lab25: read {len(_w25)} of 4 median-window boundary "
                         f"rows, so the note's window sentence rests on a short read")
        else:
            _by = {}
            for _k, _v in _w25.items():
                _by.setdefault(_v, []).append(int(_k))
            cwant("6.2 names the window at which the rule never beats a constant",
                  "and no delay at all at 504" if _w25.get("504") == "never"
                  else f"at 504 the rule still beats the constant at {_w25['504']}")
        # Table 5's caption may not promise intervals its body does not print
        _cap5 = re.search(r"<b>Table 5\.</b>(.*?)</caption>(.*?)</table>",
                          read_text(companion), re.S) if companion else None
        checks += 1
        if not _cap5:
            fails.append("the note's Table 5 could not be located")
        else:
            _has_iv = bool(re.search(r"\[\s*[-+\u2212]?[\d.]+\s*,",
                                     plain_text_of(_cap5.group(2))))
            # The test is on the CLAIM shape, not on the word: a caption may
            # legitimately mention intervals while saying they are elsewhere,
            # and a crude substring test then fails a caption that is telling
            # the truth.  What is forbidden is describing an interval column of
            # this table - "with 95% ... intervals", "Intervals are ...".
            _c5t = plain_text_of(_cap5.group(1)).lower()
            # "intervals are in lab25's output" is the caption saying where
            # they are NOT, so the pattern requires a predicate that describes
            # them as the table's own: "intervals are <construction>", not
            # "intervals are in <somewhere else>".
            _says_iv = bool(re.search(r"with 95%[^.]{0,40}intervals?\b", _c5t)
                            or re.search(r"\bintervals? (are|is) (?!in\b)", _c5t))
            checks += 1
            if _says_iv and not _has_iv:
                fails.append("the note's Table 5 caption promises 95% intervals and "
                             "its body prints none; lab25 has them, the table does "
                             "not, and a caption may not describe a column that is "
                             "not there")

        # --- Section 6.3, against lab26 -------------------------------------
        l26 = read_text(os.path.join(EXP, "lab26_quantile_target.txt"))
        R26 = re.compile(r"\s*(0\.\d\d)\s+([-\d.]+)\s+0\s+([\d.]+)\s+"
                         r"([-\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s*$")
        n26 = 0
        for ln in l26.splitlines():
            m = R26.match(ln)
            if m:
                q, z, rho, cstar, ceil, note_ = m.groups()
                n26 += 1
                cwant(f"lab26 q={q} row", f"{ceil}% {note_}%")
        checks += 1
        if n26 != 4:
            fails.append(f"lab26: matched {n26} delta=0 quantile rows, expected 4")
        cwant("lab26 tail ceiling",
              "the Bayes rule itself attains 90.11% against a majority-class 90.00%")
        cwant("lab26 quadrature check",
              "largest absolute difference of 1.41e-07")
        checks += 1
        if "largest absolute difference: 1.41e-07" not in l26:
            fails.append("lab26: the quadrature check no longer agrees to 1.4e-07")
        cwant("lab26 objective clash",
              "At fifteen of twenty quantile-delay cells")
        checks += 1
        if "different rules: 15 of 20" not in l26:
            fails.append("lab26: the objectives no longer disagree at 15 of 20 cells")
        # the note must not claim it settled the tail question
        for phrase in ("the threshold rule dominates at the 90th percentile",
                       "resolves the tail target"):
            checks += 1
            if phrase in ctxt:
                fails.append(f"companion overclaim: STALE  '{phrase}'")

    # ------------------------------------------------------------------
    # THE CONVERSE CHECK: a quoted interval no check asked for
    #
    # Everything above asks "does the document contain what the lab says?".
    # That leaves the other direction open, and it is the direction the stale
    # conditional rate hid in: "81% [74, 87]" did not fail a check, it was
    # invisible to every check, because no check mentioned the cell it came
    # from.  So this asks the converse: every "NN% [a, b]" a document quotes
    # must be a value some want() above demanded.  A figure nobody asked for
    # is a figure nobody is checking.
    #
    # Scope: the whole document, table bodies included.  Restricting it to
    # prose and captions left a real hole: a table cell can be altered while a
    # want() still passes on another copy of the same figure elsewhere in the
    # document, which is what a tamper audit found in Table 1 -- the paper's
    # own summary table.  Including the tables was tried on the assumption it
    # would report hundreds of cells the row parsers cover implicitly; it
    # reports none, because they do cover them, so there is no reason to
    # exclude them and one good reason not to.
    _dem = set()
    for _d in _demanded:
        for _mm in re.finditer(r"(-?\d+(?:\.\d+)?)% \[\s*(-?\d+(?:\.\d+)?),"
                               r"\s*(-?\d+(?:\.\d+)?)\]", _d):
            _dem.add(_mm.group(0))
        for _mm in re.finditer(r"\[\s*(-?\d+(?:\.\d+)?),\s*"
                               r"(-?\d+(?:\.\d+)?)\]", _d):
            _dem.add(_mm.group(0))
    _orphans = []
    for _label, _path in (("paper", path),) + (
            (("appendix", _supp),) if os.path.isfile(_supp) else ()):
        _raw = read_text(_path)
        _prose = norm_iv(plain_text_of(_raw))
        for _mm in re.finditer(r"(-?\d+(?:\.\d+)?)% \[\s*(-?\d+(?:\.\d+)?),"
                               r"\s*(-?\d+(?:\.\d+)?)\]", _prose):
            _tok = _mm.group(0)
            if _tok in _dem:
                continue
            # NO fallback on the bracket alone.  There used to be one, for
            # needles that are a bare interval, and it made the check blind to
            # the point estimate: changing Table 1's headline from 71% to 112%
            # passed, because "[54, 83]" had been demanded somewhere.  A
            # needle that is a bare interval is now covered by requiring the
            # bracket to appear inside a demanded needle that also carries a
            # point estimate, which the loop above collects.
            _orphans.append((_label, _tok,
                             " ".join(_prose[max(0, _mm.start() - 70):
                                             _mm.end() + 20].split())))
    # ------------------------------------------------------------------
    # PASS-3 GUARDS: a lab's own PROSE read against that lab's own TABLE.
    #
    # The first two cold passes closed the gap between the documents and the
    # lab outputs.  What the third pass found was a different surface: free
    # prose inside a lab, asserting a count or a ratio that the lab's own
    # printed table denies - "six schemes" over eight, "ten draws" beside
    # n_draw = 100, "three to twelve times as many days" beside 85 days
    # against a 61-day quarter.  Seventeen instances, all in sentences that
    # were typed rather than interpolated.
    #
    # Every one of them is now computed by the lab from the same arrays that
    # print the table, which is the real fix.  These checks are the ratchet
    # on that fix: they re-derive each quantity from the PRINTED table and
    # require the lab's sentence to agree.  A future edit that re-hardcodes
    # one of these - including an edit that reruns the lab and installs the
    # new reference, which is what hides this class of error from
    # run_all --check - fails here instead.
    def _p3(label, ok, msg):
        nonlocal checks
        checks += 1
        if not ok:
            fails.append(f"{label}: {msg}")

    # lab39: the verdict's magnitude claim against part C's difference column
    _p3_39 = read_text(os.path.join(EXP, "lab39_temporal_stability.txt"))
    _p3_39g = [abs(int(_m)) for _m in re.findall(
        r"^\s{2,}(?:calm|middle|stressed)\s+\d+\s+\d+%\s+\d+%\s+([+-]\d+)%\s+\[",
        _p3_39, re.M)]
    _p3_39m = re.search(r"halves differ by up to (\d+) points,\s*\n\s*and by more "
                        r"than fifteen in (\d+) of the (\d+) cells", _p3_39)
    _p3("lab39 verdict shape", bool(_p3_39m) and len(_p3_39g) == 12,
         f"part C gave {len(_p3_39g)} within-tercile rows and the verdict's "
         f"magnitude sentence was {'found' if _p3_39m else 'NOT found'}")
    if _p3_39m and len(_p3_39g) == 12:
        _p3("lab39 largest gap", int(_p3_39m.group(1)) == max(_p3_39g),
             f"verdict says up to {_p3_39m.group(1)} points; part C's largest "
             f"difference is {max(_p3_39g)}")
        _p3("lab39 gap count", int(_p3_39m.group(2)) == sum(1 for g in _p3_39g if g > 15)
             and int(_p3_39m.group(3)) == len(_p3_39g),
             f"verdict says more than fifteen in {_p3_39m.group(2)} of "
             f"{_p3_39m.group(3)}; part C gives "
             f"{sum(1 for g in _p3_39g if g > 15)} of {len(_p3_39g)}")

    # lab12: three typed claims, all three now derived - and all three checked
    # against the tables that sit above them
    _p3_12 = read_text(os.path.join(EXP, "lab12_appendix.txt"))
    _p3_12w = {}
    for _m in re.finditer(r"^\s{2,}(\d+)\s+[+-][\d.]+((?:\s+\[[+-][\d.]+,[+-][\d.]+\]){4})"
                          r"\s*$", _p3_12, re.M):
        _iv = re.findall(r"\[([+-][\d.]+),([+-][\d.]+)\]", _m.group(2))
        _p3_12w[int(_m.group(1))] = [float(_h) - float(_l) for _l, _h in _iv]
    _p3("lab12 block table", len(_p3_12w) == 4,
         f"part C's block table gave {len(_p3_12w)} delays, expected 4")
    _p3_12m = re.search(r"width rises with the block at (\d+) of the (\d+) delays, and "
                        r"the\s*\n\s*longest block \((\d+)\) gives the widest interval "
                        r"at (\d+) of them", _p3_12)
    if _p3_12m and len(_p3_12w) == 4:
        _rise = sum(1 for v in _p3_12w.values()
                    if all(b > a for a, b in zip(v, v[1:])))
        _long = sum(1 for v in _p3_12w.values() if v[-1] == max(v))
        _p3("lab12 width rises", int(_p3_12m.group(1)) == _rise
             and int(_p3_12m.group(2)) == len(_p3_12w),
             f"says width rises at {_p3_12m.group(1)} of {_p3_12m.group(2)} "
             f"delays; the printed intervals rise at {_rise} of {len(_p3_12w)}")
        _p3("lab12 longest block widest", int(_p3_12m.group(4)) == _long,
             f"says the longest block is widest at {_p3_12m.group(4)} delays; "
             f"the printed intervals make it widest at {_long}")
    else:
        _p3("lab12 width sentence", False,
             "part C's width sentence is no longer in the form this check reads")
    _p3_12n = re.search(r"narrow grid \[([\d., ]+)\]", _p3_12)
    _p3_12v = re.search(r"wide grid\s+\[([\d., ]+)\]", _p3_12)
    _p3_12f = re.search(r"(\d+)-fold lower and (\d+)-fold higher", _p3_12)
    if _p3_12n and _p3_12v and _p3_12f:
        _nar = [float(x) for x in _p3_12n.group(1).split(",")]
        _wid = [float(x) for x in _p3_12v.group(1).split(",")]
        _p3("lab12 grid widening", int(_p3_12f.group(1)) == round(min(_nar) / min(_wid))
             and int(_p3_12f.group(2)) == round(max(_wid) / max(_nar)),
             f"part D says {_p3_12f.group(1)}-fold lower and "
             f"{_p3_12f.group(2)}-fold higher; the printed grids give "
             f"{min(_nar) / min(_wid):.0f}-fold and {max(_wid) / max(_nar):.0f}-fold")
    else:
        _p3("lab12 grid sentence", False,
             "part D's grid widening sentence or one of its grids is missing")
    _p3_12d = [float(_m) for _m in re.findall(
        r"^\s{2,}\d+\s+[+-][\d.]+\s+[+-][\d.]+\s+([+-][\d.]+)\s*$", _p3_12, re.M)]
    _p3_12s = re.search(r"favours the\s*\n?\s*cross-sectional model at (\d+) of the "
                        r"(\d+) delays and runs against it at\s*\n?\s*(\d+)", _p3_12)
    if _p3_12s and _p3_12d:
        _p3("lab12 grid direction", int(_p3_12s.group(1)) == sum(1 for d in _p3_12d if d > 0)
             and int(_p3_12s.group(2)) == len(_p3_12d)
             and int(_p3_12s.group(3)) == sum(1 for d in _p3_12d if d < 0),
             f"says {_p3_12s.group(1)} of {_p3_12s.group(2)} favour and "
             f"{_p3_12s.group(3)} runs against; part D's difference column gives "
             f"{sum(1 for d in _p3_12d if d > 0)} of {len(_p3_12d)} and "
             f"{sum(1 for d in _p3_12d if d < 0)}")
    else:
        _p3("lab12 direction sentence", False,
             f"part D's direction sentence was {'found' if _p3_12s else 'NOT found'} "
             f"and its difference column gave {len(_p3_12d)} rows")

    # lab09: the row the verdict describes must be a row that exists, and must
    # be the kind of row the sentence says it is - net of cost positive beside
    # a negative dR2.  The sentence used to name neither the model nor the
    # delay, so nothing could check it.
    _p3_09 = read_text(os.path.join(EXP, "lab09_nonlinearity.txt"))
    _p3_09t = {}
    for _nm, _hd in (("SQ", "R2 SQ"), ("FACTOR", "R2 FAC")):
        _seg = _p3_09.split(_hd, 1)
        if len(_seg) == 2:
            _p3_09t[_nm] = {int(_m[0]): (float(_m[1]), float(_m[2]), float(_m[3]))
                            for _m in re.findall(
                                r"^\s{2,}(\d+)\s+[\d.]+\s+[\d.]+\s+([+-][\d.]+)\s+"
                                r"(-?[\d.]+)\s+[+-][\d.]+\s+([+-][\d.]+)\s*$",
                                _seg[1].split("delta")[0], re.M)}
    _p3_09m = re.search(r"a dR2 of (-?[\d.]+) and a GW statistic of (-?[\d.]+) "
                        r"\((SQ|FACTOR) at delta = (\d+)\)", _p3_09)
    _p3("lab09 names its row", bool(_p3_09m),
         "the verdict no longer names the model and delay whose row it describes")
    if _p3_09m:
        _row = _p3_09t.get(_p3_09m.group(3), {}).get(int(_p3_09m.group(4)))
        _p3("lab09 row exists", _row is not None,
             f"the verdict cites {_p3_09m.group(3)} at delta = {_p3_09m.group(4)}, "
             f"which is not a row of that table")
        if _row:
            _p3("lab09 row figures",
                 abs(_row[0] - float(_p3_09m.group(1))) < 5e-5
                 and abs(_row[1] - float(_p3_09m.group(2))) < 5e-3,
                 f"the verdict quotes dR2 {_p3_09m.group(1)} and GW "
                 f"{_p3_09m.group(2)}; the table's row gives {_row[0]:+.4f} "
                 f"and {_row[1]:.2f}")
            _p3("lab09 row is the kind described", _row[2] > 0 and _row[0] < 0,
                 f"the verdict describes a positive net-of-cost beside a negative "
                 f"dR2; the cited row has net {_row[2]:+.4f} and dR2 {_row[0]:+.4f}")

    # lab16: two ranges and a 6-of-6, all from part A's two percentage columns
    _p3_16 = read_text(os.path.join(EXP, "lab16_clark_west_shrinkage.txt"))
    _p3_16r = [(float(a), float(b), int(c), int(d)) for a, b, c, d in re.findall(
        r"^\s{2,}\d+\s+([\d.]+)\s+([\d.]+)\s+(\d+)%\s+(\d+)%\s*$", _p3_16, re.M)]
    _p3_16m = re.search(r"arms agree at (\d+)% to (\d+)% of refits, and the cross arm is"
                        r"\s*\n\s*penalised harder at (\d+)% to (\d+)% of them", _p3_16)
    _p3_16h = re.search(r"MEAN penalty is the\s*\n\s*higher of the two \((\d+) of (\d+)\)",
                        _p3_16)
    _p3("lab16 part A table", len(_p3_16r) >= 6 and bool(_p3_16m) and bool(_p3_16h),
         f"part A gave {len(_p3_16r)} rows; the agreement sentence was "
         f"{'found' if _p3_16m else 'NOT found'} and the mean-penalty tally "
         f"{'found' if _p3_16h else 'NOT found'}")
    if len(_p3_16r) >= 6 and _p3_16m and _p3_16h:
        _ag = [100 - r[2] for r in _p3_16r]
        _hd = [r[3] for r in _p3_16r]
        _p3("lab16 agreement range",
             int(_p3_16m.group(1)) == min(_ag) and int(_p3_16m.group(2)) == max(_ag),
             f"says the arms agree at {_p3_16m.group(1)}% to {_p3_16m.group(2)}%; "
             f"the differing column gives {min(_ag)}% to {max(_ag)}%")
        _p3("lab16 harder range",
             int(_p3_16m.group(3)) == min(_hd) and int(_p3_16m.group(4)) == max(_hd),
             f"says the cross arm is penalised harder at {_p3_16m.group(3)}% to "
             f"{_p3_16m.group(4)}%; the column gives {min(_hd)}% to {max(_hd)}%")
        _p3("lab16 mean penalty tally",
             int(_p3_16m is not None and _p3_16h.group(1))
             == sum(1 for r in _p3_16r if r[1] > r[0])
             and int(_p3_16h.group(2)) == len(_p3_16r),
             f"says the cross arm's mean penalty is higher at "
             f"{_p3_16h.group(1)} of {_p3_16h.group(2)}; the two mean columns "
             f"make it higher at {sum(1 for r in _p3_16r if r[1] > r[0])} of "
             f"{len(_p3_16r)}")

    # lab25: the two fold-changes are the grid's own extent
    _p3_25 = read_text(os.path.join(EXP, "lab25_window_sensitivity.txt"))
    _p3_25g = [(int(a), int(b)) for a, b in re.findall(
        r"^\s{2,}(\d+)\s+(\d+)\s+[+-][\d.]+\[", _p3_25, re.M)]
    _p3_25m = re.search(r"it survives an? ([\d.]+)-fold change in the\s*\n\s*first and "
                        r"an? ([\d.]+)-fold change in the second", _p3_25)
    _p3("lab25 grid sentence", bool(_p3_25m) and len(_p3_25g) >= 8,
         f"part B gave {len(_p3_25g)} grid rows and the fold sentence was "
         f"{'found' if _p3_25m else 'NOT found'}")
    if _p3_25m and _p3_25g:
        _med = [g[0] for g in _p3_25g]
        _trn = [g[1] for g in _p3_25g]
        _p3("lab25 median fold",
             abs(float(_p3_25m.group(1)) - max(_med) / min(_med)) < 0.05,
             f"says a {_p3_25m.group(1)}-fold change in the median; the grid runs "
             f"{min(_med)} to {max(_med)}, a factor of {max(_med) / min(_med):.1f}")
        _p3("lab25 training fold",
             abs(float(_p3_25m.group(2)) - max(_trn) / min(_trn)) < 0.05,
             f"says a {_p3_25m.group(2)}-fold change in the training window; the "
             f"grid runs {min(_trn)} to {max(_trn)}, a factor of "
             f"{max(_trn) / min(_trn):.1f}")

    # lab40: the verdict's three lists are the significance table's three lists.
    # A verdict that spells its own conclusion ("at NO delay") and prints the
    # lists it rests on can be checked; the sentence it replaced said neither.
    _p3_40 = read_text(os.path.join(EXP, "lab40_level_or_normaliser.txt"))
    _p3_40t = dict(re.findall(r"^\s+(A t\+5|B origin|C t-delta): \d+ of \d+ (\[[^\]]*\])\s*$",
                              _p3_40, re.M))
    _p3_40m = re.search(r"at (NO delay|\w+ delays?) on any of the three targets "
                        r"\(A t\+5: (\[[^\]]*\]); B origin: (\[[^\]]*\]); "
                        r"C t-delta: (\[[^\]]*\])\)", _p3_40)
    _p3("lab40 verdict shape", bool(_p3_40m) and len(_p3_40t) == 3,
         f"the significance table gave {len(_p3_40t)} targets and the verdict "
         f"sentence was {'found' if _p3_40m else 'NOT found'}")
    if _p3_40m and len(_p3_40t) == 3:
        for _i, _k in enumerate(("A t+5", "B origin", "C t-delta"), start=2):
            _p3(f"lab40 {_k} list", _p3_40m.group(_i) == _p3_40t[_k],
                 f"the verdict gives {_p3_40m.group(_i)} for {_k}; the table "
                 f"gives {_p3_40t[_k]}")
        _p3("lab40 verdict wording",
             (_p3_40m.group(1) == "NO delay")
             == all(v == "[]" for v in _p3_40t.values()),
             f"the verdict says '{_p3_40m.group(1)}' while the table's lists are "
             f"{list(_p3_40t.values())}")

    # lab57: a widest single phase is not an envelope of three.  The verdict
    # used to call the union of the three a single phase's interval, which
    # widens the honest claim by six points at the top.
    _p3_57 = read_text(os.path.join(EXP, "lab57_housing_overlap.txt"))
    _p3_57r = [(int(a), int(b)) for a, b in re.findall(
        r"^\s{2,}\w{3}/\w{3}/\w{3}/\w{3}\s+1\s+[\d.]+\s+[\d.]+\s+[\d.-]+%\s+"
        r"\[\s*-?\d+%,\s*-?\d+%\]\s+[\d.]+%\s+\[\s*(-?\d+)%,\s*(-?\d+)%\]",
        _p3_57, re.M)]
    _p3_57m = re.search(r"widest single-phase bootstrap interval on the pooled figure "
                        r"is \[(-?\d+)%, (-?\d+)%\],\s*\n\s*and the envelope of all "
                        r"three is \[(-?\d+)%, (-?\d+)%\]", _p3_57)
    _p3("lab57 phase intervals", bool(_p3_57m) and len(_p3_57r) == 3,
         f"found {len(_p3_57r)} one-quarter phase rows and the interval sentence "
         f"was {'found' if _p3_57m else 'NOT found'}")
    if _p3_57m and len(_p3_57r) == 3:
        _wide = max(_p3_57r, key=lambda t: t[1] - t[0])
        _p3("lab57 widest single phase",
             (int(_p3_57m.group(1)), int(_p3_57m.group(2))) == _wide,
             f"says the widest single phase is [{_p3_57m.group(1)}%, "
             f"{_p3_57m.group(2)}%]; the widest printed row is "
             f"[{_wide[0]}%, {_wide[1]}%]")
        _p3("lab57 envelope",
             int(_p3_57m.group(3)) == min(r[0] for r in _p3_57r)
             and int(_p3_57m.group(4)) == max(r[1] for r in _p3_57r),
             f"says the envelope is [{_p3_57m.group(3)}%, {_p3_57m.group(4)}%]; "
             f"the three rows span [{min(r[0] for r in _p3_57r)}%, "
             f"{max(r[1] for r in _p3_57r)}%]")

    # lab29: the days-per-cell ratio, which was asserted as "three to twelve"
    # against quarters of 61 days and phases of 85 to 380 - a sentence that
    # overstated the phase view's advantage by a factor of two at the bottom
    # of the range, where the argument needs it most.
    _p3_29 = read_text(os.path.join(EXP, "lab29_single_regime.txt"))
    _p3_29q = [int(_m) for _m in re.findall(r"^\s{2,}\d{4}Q\d\s+(\d+)\s+-?[\d.]+\s",
                                            _p3_29, re.M)]
    _p3_29p = [int(_m) for _m in re.findall(
        r"^\s{2,}[a-z][a-z -]+\s+\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}\s+(\d+)\s+",
        _p3_29, re.M)]
    _p3_29m = re.search(r"mean quarter in the arc \((\d+) days\):\s*\n\s*the phases "
                        r"carry ([\d.]+) to ([\d.]+) times as many", _p3_29)
    _p3("lab29 ratio sentence", bool(_p3_29m) and len(_p3_29q) >= 8 and len(_p3_29p) == 4,
         f"the arc gave {len(_p3_29q)} quarters and {len(_p3_29p)} phases; the "
         f"ratio sentence was {'found' if _p3_29m else 'NOT found'}")
    if _p3_29m and _p3_29q and len(_p3_29p) == 4:
        _mq = sum(_p3_29q) / len(_p3_29q)
        _p3("lab29 mean quarter", int(_p3_29m.group(1)) == round(_mq),
             f"says the mean quarter is {_p3_29m.group(1)} days; the printed "
             f"quarters average {_mq:.1f}")
        _rr = sorted(p / _mq for p in _p3_29p)
        _p3("lab29 ratio range",
             abs(float(_p3_29m.group(2)) - _rr[0]) < 0.06
             and abs(float(_p3_29m.group(3)) - _rr[-1]) < 0.06,
             f"says the phases carry {_p3_29m.group(2)} to {_p3_29m.group(3)} "
             f"times as many days; the printed day counts give "
             f"{_rr[0]:.1f} to {_rr[-1]:.1f}")
        _p3("lab29 smallest phase is not large", _rr[0] < 2.0,
             f"the sentence's warning that the smallest phase is barely larger "
             f"than a quarter no longer matches its ratio of {_rr[0]:.1f}")

    # lab54 / Section S29: the width the appendix says its intervals have.
    # It said "Thirty to forty points" two sentences before saying "from the
    # forty-two observed here", and Table S24's quarterly row gives 42 and 47.
    # An understatement of a width is a claim of precision the data does not
    # give, so it is read from the lab rather than left to prose.
    _p3_54 = read_text(os.path.join(EXP, "lab54_monthly_feasibility.txt"))
    _m54 = re.search(r"price it\. (\d+) to (\d+) points of", _p3_54)
    _p3("lab54 width sentence", bool(_m54),
         "lab54's verdict no longer prints the width range Section S29 quotes")
    if _m54:
        want("S29 quotes lab54's interval widths",
             f"{_num_word(int(_m54.group(1))).capitalize()} to "
             f"{_num_word(int(_m54.group(2)))} points of width")
    # The same figure again, this time as the arithmetic's starting width:
    # "from the forty-two observed here" must be the narrower of the two.
    if _m54:
        want("S29's narrowing arithmetic starts from the width it measured",
             f"from the {_num_word(int(_m54.group(1)))} observed here")

    # Three tolerances that used to be read back as measurements: a threshold
    # inside an "if" is not a result, and each of these sentences quoted its
    # own branch condition.  All three are now the measured value, and each is
    # checked against the column the measurement comes from.
    _p3_29b = re.split(r"delta = \d+:\s+all six episodes", _p3_29)[1:]
    _m29b = re.findall(r"moves the pooled figure by more than ([\d.]+) points", _p3_29)
    _p3("lab29 episode-move blocks", len(_p3_29b) == len(_m29b) == 3,
         f"found {len(_p3_29b)} episode blocks and {len(_m29b)} move bounds, "
         f"expected three of each")
    if len(_p3_29b) == len(_m29b) == 3:
        for _i, (_blk, _said) in enumerate(zip(_p3_29b, _m29b)):
            _mv = [abs(float(_x)) for _x in re.findall(
                r"^\s{2,}[A-Za-z0-9 ]+\s+\d+\s+[+-]?[\d.]+%\s+\[[^\]]+\]\s+"
                r"([+-][\d.]+)%\s*$", _blk, re.M)]
            _p3(f"lab29 move bound {_i}",
                 len(_mv) == 6 and abs(float(_said) - max(_mv)) < 0.051,
                 f"the block says no episode moves it by more than {_said} points; "
                 f"its {len(_mv)} printed moves reach "
                 f"{max(_mv) if _mv else float('nan'):.1f}")
    _p3_14 = read_text(os.path.join(EXP, "lab14_appraisal_smoothing.txt"))
    _g14 = [abs(int(_x)) for _x in re.findall(
        r"gap\s+([+-]\d+)%\s+match\s*$", _p3_14, re.M)]
    _m14 = re.search(r"The (\w+) rows with real staleness agree to within (\d+) points",
                     _p3_14)
    _p3("lab14 staleness rows", bool(_m14) and len(_g14) >= 2,
         f"found {len(_g14)} matching rows and the agreement sentence was "
         f"{'found' if _m14 else 'NOT found'}")
    if _m14 and _g14:
        _p3("lab14 row count", _m14.group(1) == _num_word(len(_g14)),
             f"says {_m14.group(1)} rows agree; {len(_g14)} rows are marked match")
        _p3("lab14 agreement bound", int(_m14.group(2)) == max(_g14),
             f"says they agree to within {_m14.group(2)} points; the printed gaps "
             f"reach {max(_g14)}")

    # lab58's citation of lab65, checked against lab65's own header and row.
    # lab58 attributed its two figures to a bandwidth-20 column; lab65's sweep
    # is 9/40/80/160/320 and has no such column, so the figures it was quoting
    # are the bandwidth-40 ones.  A lab citing a column that does not exist is
    # the strongest form of this failure, because the citation reads as
    # precision.
    _p3_65 = read_text(os.path.join(EXP, "lab65_hac_coverage_real.txt"))
    _h65 = re.search(r"^ delta  series((?:\s+ret@\d+)+)\s+rule\s+ret@rule\s+"
                     r"95% at (\d+) is", _p3_65, re.M)
    _p3("lab65 header", bool(_h65), "lab65's bandwidth sweep header has changed shape")
    if _h65:
        _bw65 = [int(_x) for _x in re.findall(r"ret@(\d+)", _h65.group(1))]
        _p3("lab65 has no bandwidth-20 column", 20 not in _bw65,
             f"lab65's sweep is {_bw65}; a check written on the assumption that "
             f"it has no 20 column must be revisited now that it does")
        _p3("lab65 prices the length this project uses", 40 in _bw65
             and int(_h65.group(2)) == 40,
             f"lab65's sweep is {_bw65} and its coverage column is at "
             f"{_h65.group(2)}, not 40")
        _r65 = re.search(r"^\s+55\s+d2((?:\s+[\d.]+%){5})\s+\d+\s+[\d.]+%\s+"
                         r"([\d.]+)%\s+([\d.]+)%\s*$", _p3_65, re.M)
        _l58 = read_text(os.path.join(EXP, "lab58_ratio_inference.txt"))
        _m58 = re.search(r"Bartlett\((\d+)\), it\s*\n\s*finds about three quarters of "
                         r"it retained at the headline delay - a\s*\n\s*nominal 95% "
                         r"interval worth roughly (\d+)%", _l58)
        _p3("lab58 cites lab65 in the form this check reads",
             bool(_r65) and bool(_m58),
             f"lab65's delta = 55 row was {'found' if _r65 else 'NOT found'} and "
             f"lab58's citation {'found' if _m58 else 'NOT found'}")
        if _r65 and _m58:
            _ret = [float(_x) for _x in re.findall(r"([\d.]+)%", _r65.group(1))]
            _i40 = _bw65.index(40) if 40 in _bw65 else None
            _p3("lab58 cites the bandwidth it names",
                 int(_m58.group(1)) in _bw65,
                 f"lab58 attributes its figures to Bartlett({_m58.group(1)}), which "
                 f"is not one of lab65's bandwidths {_bw65}")
            _p3("lab58's 'three quarters' is lab65's bandwidth-40 retention",
                 _i40 is not None and 0.70 <= _ret[_i40] / 100 <= 0.80,
                 f"lab58 calls it three quarters; lab65's ret@40 at delta = 55 is "
                 f"{_ret[_i40] if _i40 is not None else float('nan')}%")
            _p3("lab58's coverage figure is lab65's at the same bandwidth",
                 abs(float(_m58.group(2)) - round(float(_r65.group(2)))) < 0.51,
                 f"lab58 says roughly {_m58.group(2)}%; lab65's 95%-at-40 column "
                 f"gives {_r65.group(2)}% at delta = 55")

    # lab56's comparison against lab55's interval, which lab56 used to quote as
    # "some twenty points wide" from memory.  The lab now makes the comparison
    # without the number; the arithmetic is checked here, where both labs are
    # readable at once.
    _p3_56 = read_text(os.path.join(EXP, "lab56_seasonal_and_breadth.txt"))
    _m56 = re.search(r"by only ([\d.]+)%, which is an order of magnitude inside the",
                     _p3_56)
    _l55v = read_text(os.path.join(EXP, "lab55_illiquid_measured.txt"))
    _m55v = re.search(r"On the paper's OWN target type at this clock, [\d.]+% "
                      r"\[(\d+)%, (\d+)%\]", _l55v)
    _p3("lab56/lab55 comparison readable", bool(_m56) and bool(_m55v),
         f"lab56's gap sentence was {'found' if _m56 else 'NOT found'} and lab55's "
         f"interval on that coordinate {'found' if _m55v else 'NOT found'}")
    if _m56 and _m55v:
        _w55 = int(_m55v.group(2)) - int(_m55v.group(1))
        _p3("lab56's gap really is an order of magnitude inside lab55's interval",
             float(_m56.group(1)) * 10 <= _w55,
             f"lab56 calls a {_m56.group(1)}-point gap an order of magnitude inside "
             f"lab55's interval, which is {_w55} points wide")

    # lab07's three borrowed figures, each now read and each now checked.
    # The placebo pair came from lab06 at two decimals (-0.019/-0.044 against
    # -0.0180/-0.0419); the "warning signs" sentence asserted +0.0063 against a
    # P2 table printing +0.0060, compared it with a "+0.0016 point estimate"
    # that appears nowhere in the lab, and called the ratio four when it is two;
    # and P1's own figure was typed as -0.0100 against an implied -0.0096.
    _p3_07 = read_text(os.path.join(EXP, "lab07_estimation_cost.txt"))
    _p3_06 = read_text(os.path.join(EXP, "lab06_inference.txt"))
    _m07p = re.search(r"shifted peers cost (-[\d.]+)\s*\n\s*at delta = 5 but "
                      r"(-[\d.]+) at delta = 21", _p3_07)
    _p3("lab07 quotes lab06's placebo", bool(_m07p),
         "lab07 no longer prints lab06's placebo pair in the form this check reads")
    if _m07p:
        for _d, _said in ((5, _m07p.group(1)), (21, _m07p.group(2))):
            _ok = re.search(rf"^\s*{_d}\s+({re.escape(_said)})\b", _p3_06, re.M) \
                  or _said in _p3_06
            _p3(f"lab07's placebo figure at delta = {_d} is lab06's",
                 bool(_ok),
                 f"lab07 says lab06's placebo at delta = {_d} is {_said}, which "
                 f"does not appear in lab06's output")
    _m07h = re.search(r"the delta = 0 gap read ([+-][\d.]+), ([\d.]+) times\s*\n\s*"
                      r"the ([+-][\d.]+) the point estimate below gives", _p3_07)
    _m07t = re.search(r"^\s+1000\s+([+-][\d.]+)\s+[+-][\d.]+\s", _p3_07, re.M)
    _m07g = re.search(r"^\s+0\s+[+-][\d.]+\s+[+-][\d.]+\s+([+-][\d.]+)\s+\[",
                      _p3_07, re.M)
    _p3("lab07 warning-signs sentence", bool(_m07h) and bool(_m07t) and bool(_m07g),
         f"the sentence was {'found' if _m07h else 'NOT found'}, P2's heaviest row "
         f"{'found' if _m07t else 'NOT found'}, the gross table's delta = 0 row "
         f"{'found' if _m07g else 'NOT found'}")
    if _m07h and _m07t and _m07g:
        _p3("lab07's heavy-shrinkage figure is P2's own",
             _m07h.group(1) == _m07t.group(1),
             f"the sentence says {_m07h.group(1)}; P2's lambda = 1000 row gives "
             f"{_m07t.group(1)}")
        _p3("lab07's point estimate is the gross table's own",
             _m07h.group(3) == _m07g.group(1),
             f"the sentence says {_m07h.group(3)}; the gross table's delta = 0 "
             f"point estimate is {_m07g.group(1)}")
        _p3("lab07's multiple is the quotient of the two figures it names",
             abs(float(_m07h.group(2))
                 - float(_m07t.group(1)) / float(_m07g.group(1))) < 0.051,
             f"the sentence says {_m07h.group(2)} times; "
             f"{_m07t.group(1)} / {_m07g.group(1)} is "
             f"{float(_m07t.group(1)) / float(_m07g.group(1)):.1f}")
    _m07s = re.search(r"tight ([+-][\d.]+) from P1", _p3_07)
    _m07l = re.search(r"implied cost of the full (\d+)-peer cross-section: "
                      r"([+-][\d.]+)", _p3_07)
    _p3("lab07's P1 reference", bool(_m07s) and bool(_m07l),
         f"the P1 reference was {'found' if _m07s else 'NOT found'} and P1's own "
         f"implied cost {'found' if _m07l else 'NOT found'}")
    if _m07s and _m07l:
        _p3("lab07 quotes P1 as P1 prints it",
             _m07s.group(1) == _m07l.group(2),
             f"the sentence says {_m07s.group(1)} from P1; P1's line through the "
             f"origin implies {_m07l.group(2)}")

    # ------------------------------------------------------------------
    # THE LABS' OWN PROSE: a bounded inventory of typed quantifiers.
    #
    # Everything above checks a sentence this project already knows about.  This
    # closes the class instead: sweep_lab_prose.py finds every count, ratio or
    # multiple that a lab PRINTS without COMPUTING it, and the list below is the
    # complete set that may remain, each with the reason it is not a
    # measurement.  A new typed quantifier anywhere in the sixty-seven labs
    # fails here, which is the only construction that stops the nineteen
    # sentences of this round from becoming twenty next month.
    #
    # An entry may be added only with a reason that survives the test applied to
    # every one of these: would the lab's own table still agree with the
    # sentence after a rerun?  "It is right today" is not that reason.
    _PROSE_OK = {
        ("lab10_loss_scale.py", "one to"):
            "'the one to believe' - a pronoun, not a count",
        ("lab29_single_regime.py", "one to"):
            "'the one to read' - a pronoun, not a count",
        ("lab53_no_options_targets.py", "two of"):
            "'two of the four' - checked against its own table, which lists "
            "CSEALL and KSE100 as the two that fail",
        ("lab56_seasonal_and_breadth.py", "six of"):
            "'six of them ragged' - 19 peers minus the 13 balanced ones, and "
            "both numbers are in the same sentence",
        ("lab57_housing_overlap.py", "14 of 14"):
            "Section 9.2's monthly figure, quoted from the paper and checked "
            "against it below",
        ("lab61_block_sensitivity.py", "five of"):
            "the companion note's own claim, which this lab exists to test; "
            "checked against the note",
        ("lab63_fieller_coverage.py", "two of"):
            "'between two of them' - names a pair, and the figure beside it is "
            "interpolated",
    }
    try:
        import sweep_lab_prose as _sweep
        _found = {(r[0], r[2]) for r in _sweep.sweep("quantifiers")}
    except Exception as _e:                       # noqa: BLE001 - reported, not raised
        _found = None
        checks += 1
        fails.append(f"sweep_lab_prose.py could not be run, so the labs' prose "
                     f"is unchecked this run: {_e}")
    if _found is not None:
        checks += 1
        _new = sorted(_found - set(_PROSE_OK))
        if _new:
            fails.append(
                "a lab PRINTS a count or ratio it does not COMPUTE, and it is "
                "not on the allowlist in verify_paper.py: "
                + "; ".join(f"{f} [{t}]" for f, t in _new)
                + " - compute it from the same arrays that print the table, or "
                  "add it to _PROSE_OK with the reason it is not a measurement")
        checks += 1
        _gone = sorted(set(_PROSE_OK) - _found)
        if _gone:
            # An allowlist entry whose sentence no longer exists is a permission
            # with nothing behind it, and the next sentence to use those words
            # inherits it silently.  That is how "90%" nearly acquired blanket
            # permission in the coverage check below.
            fails.append(
                "an allowlist entry in _PROSE_OK no longer matches anything, so "
                "it is permission with nothing behind it: "
                + "; ".join(f"{f} [{t}]" for f, t in _gone)
                + " - remove it")

    # ------------------------------------------------------------------
    # EXHAUSTIVE COVERAGE: every percentage in prose or a caption must either
    # be a value some check demanded, or be named below as a figure that is
    # deliberately not a lab output.  There is no third category.
    #
    # This replaces a sampled tamper sweep, which is how the gap was first
    # measured and which can only ever estimate it.  Differencing the
    # verifier's own demanded needles against the documents is exhaustive and
    # runs in a second.  When it was first written it reported fourteen
    # figures; nine were real results nobody was checking and are now pinned,
    # and five are the list below.
    #
    # Everything on this list is here for a stated reason, and a figure may not
    # be added to it to make a failure go away.
    _NOT_A_RESULT = {
        # Each entry is (maximum occurrences with nothing behind them, why).
        # other papers' numbers, cited from the literature
        "4.6%": (1, "cited from the literature, not measured here"),
        "14.5%": (1, "cited from the literature, not measured here"),
        # a superseded value the text names in order to say it was wrong
        "90%": (2, "a mislabelled column quoted to retract it, and a "
                   "hypothetical accuracy against a spike target"),
        # thresholds and conceptual reference points, not measurements
        "100%": (8, "the restoration reference point, not a ceiling"),
        "5%": (12, "a nominal significance level"),
        "95%": (9, "a nominal confidence level"),
        "10%": (2, "a nominal level or a VIX decile boundary"),
        "33%": (2, "a tercile boundary"),
        "67%": (2, "a tercile boundary"),
        "90.0%": (1, "the majority class at the 90th percentile, which is q itself"),
        "50.0%": (1, "the majority class at the median, which is q itself"),
        "65.0%": (1, "the majority class at q = 0.65, which is q itself"),
        "75.0%": (1, "the majority class at q = 0.75, which is q itself"),
    }
    # An allowlist with no bound is a place to hide a figure: "90%" is allowed
    # because the appendix quotes a mislabelled column in order to retract it
    # and the note names a hypothetical, and without a bound a genuine 90%
    # result appearing anywhere would inherit that permission silently.  The
    # bound is a MAXIMUM and a TOTAL rather than per document, because the
    # journal variants move motivation between the paper and the appendix and
    # the shortest variant drops some of it, so an exact per-document count
    # fails a variant that is in fact correct.
    _allow_seen = {}
    _uncovered = []
    for _label, _p, _pool in (("paper", path, _demanded),
                              ("appendix", _supp, _demanded),
                              ("note", companion, _cdem_for_cover)):
        if not _p or not os.path.isfile(_p):
            continue
        _raw3 = read_text(_p)
        _m3 = re.sub(r"<table.*?</table>", lambda m: " " * len(m.group(0)),
                     _raw3, flags=re.S)
        _m3 = re.sub(r"<style.*?</style>", lambda m: " " * len(m.group(0)),
                     _m3, flags=re.S)
        _m3 = re.sub(r"<svg.*?</svg>", lambda m: " " * len(m.group(0)),
                     _m3, flags=re.S)
        for _cm in re.finditer(r"<caption>.*?</caption>", _raw3, re.S):
            _m3 = _m3[:_cm.start()] + _cm.group(0) + _m3[_cm.end():]
        _txt3 = norm_iv(plain_text_of(_m3))
        for _mm in re.finditer(r"(?<![\w.\-])(\d+\.\d+|\d{1,3})%", _txt3):
            _tok = _mm.group(0)
            # Coverage is tested BEFORE the allowlist, so a token that some
            # check demands is never charged to the allowlist's budget.  The
            # other order made the bound count every occurrence rather than
            # every UNVERIFIED one: adding a correct, checked "67%" to a
            # caption then failed the run, and the only ways out were to raise
            # the bound, which weakens it, or to reword the caption, which is
            # the check dictating prose.
            if any(_tok in _d for _d in _pool):
                continue
            if _tok in _NOT_A_RESULT:
                _allow_seen[_tok] = _allow_seen.get(_tok, 0) + 1
                continue
            _uncovered.append((_label, _tok,
                               " ".join(_txt3[max(0, _mm.start() - 70):
                                              _mm.end() + 25].split())))
    # the allowlist may not be used more often than it says
    for _k, (_n, _why) in sorted(_NOT_A_RESULT.items()):
        checks += 1
        _got = _allow_seen.get(_k, 0)
        if _got > _n:
            fails.append(f"allowlist entry {_k} occurs {_got} times across the "
                         f"documents with nothing behind it, above the {_n} it is "
                         f"allowed ({_why}); a new occurrence has to be accounted "
                         f"for, not absorbed")
    checks += 1
    if _uncovered:
        for _label, _tok, _ctx in _uncovered[:10]:
            fails.append(f"uncovered figure in the {_label}: '{_tok}' is neither "
                         f"demanded by a check nor listed as a non-result  "
                         f"...{_ctx}")
        if len(_uncovered) > 10:
            fails.append(f"... and {len(_uncovered) - 10} more uncovered figures")

    # CONTAINMENT: a point estimate must lie inside its own interval.  The
    # appendix carried "95% [47, 75]" for months, a figure impossible on its
    # face, and no reader or referee caught it because no check looked at the
    # shape.  This does, everywhere, for both documents and the note.
    _bad_pair = []
    for _label, _p in (("paper", path), ("appendix", _supp),
                       (("note", companion) if companion else ("note", None))):
        if not _p or not os.path.isfile(_p):
            continue
        _txt2 = norm_iv(plain_text_of(read_text(_p)))
        for _mm in re.finditer(r"(-?\d+(?:\.\d+)?)% \[\s*(-?\d+(?:\.\d+)?),"
                               r"\s*(-?\d+(?:\.\d+)?)\]", _txt2):
            _pt, _lo, _hi = (float(_mm.group(i)) for i in (1, 2, 3))
            if _lo > _hi:
                continue          # an exclusion set, printed low-then-high
            if not (_lo <= _pt <= _hi):
                _bad_pair.append((_label, _mm.group(0),
                                  " ".join(_txt2[max(0, _mm.start() - 60):
                                                 _mm.end() + 20].split())))
    checks += 1
    if _bad_pair:
        for _label, _tok, _ctx in _bad_pair[:8]:
            fails.append(f"impossible pair in the {_label}: '{_tok}' puts the point "
                         f"estimate outside its own interval  ...{_ctx}")

    checks += 1
    if _orphans:
        for _label, _tok, _ctx in _orphans[:12]:
            fails.append(f"unchecked result interval in the {_label}: '{_tok}' is "
                         f"quoted but no check asked for it  ...{_ctx}")
        if len(_orphans) > 12:
            fails.append(f"... and {len(_orphans) - 12} more unchecked intervals")

    open('/tmp/claude-0/demanded.txt','w').write('\n'.join(_demanded))
    open('/tmp/claude-0/cdemanded.txt','w').write('\n'.join(_cdemanded))
    _n_out = len([f for f in os.listdir(EXP)
                  if f.startswith("lab") and f.endswith(".txt")])
    print(f"verify_paper: {checks} checks against {_n_out} lab outputs")
    if fails:
        for f in fails:
            print("  " + f)
        raise SystemExit(f"\n{len(fails)} problem(s): the paper and the labs disagree.")
    print("  every checked figure in the paper matches its lab output")


# The paper sources live in papers/source/, and that is the layout the README
# describes.  They are looked for in three places rather than one because the
# single-path version broke in practice: GitHub's web uploader flattens a nested
# folder unless the folder itself is dragged, so a repository uploaded that way
# ends up with the HTML at the top level and this script exits before it checks
# anything.  A verifier that only runs under one directory layout is a verifier
# that quietly stops running.  Same discipline as the labs, which search several
# folders for the data rather than insisting on one.
SEARCH = [os.path.join(HERE, "papers", "source"),
          os.path.join(HERE, "papers"),
          HERE]


def find_paper(name):
    for d in SEARCH:
        q = os.path.join(d, name)
        if os.path.exists(q):
            return q
    return os.path.join(SEARCH[0], name)      # report the canonical path on failure


DEFAULT = find_paper("what-substitutes-for-a-stale-mark.html")
COMPANION = find_paper("why-a-simple-threshold-rule-is-hard-to-beat.html")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.exists(path):
        raise SystemExit(
            f"Could not find the paper source at\n  {path}\n"
            "Run this from the extracted repository folder. The sources are\n"
            "looked for in papers/source/, in papers/, and beside this script;\n"
            "otherwise pass the .html path as an argument.\n"
            "It needs the HTML source, not the PDF: the checks read the tables.")
    # Check the companion note too, but only on a default run: given an explicit
    # path the caller asked about one file and should get checks on that file.
    # A second argument names the companion note explicitly.  Without it an
    # explicit manuscript path checks that manuscript alone, which is right for
    # a one-file question but wrong for a journal variant: the script count and
    # the note's own checks need the companion in view.
    _comp = (sys.argv[2] if len(sys.argv) > 2 and os.path.exists(sys.argv[2])
             else COMPANION if len(sys.argv) <= 1 and os.path.exists(COMPANION)
             else None)
    main(path, _comp)
