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

import glob
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
         "set; supported means it is consistent across the tests run but rests on a "
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
            ("Section 9 opens on identified threats",
             "The Internet Appendix examines the principal identified threats to "
             "each central claim",
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
            ("title page: address", "Independent, United Kingdom."),

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
    _tel_pat = (r"Telephone \+\d[\d\s]{6,}\." if (_is_variant and _have_tel) else
                r"Telephone (?:\+\d[\d\s]{6,}|supplied with the submission)\.")
    for _lbl, _pat in (("title page: telephone", _tel_pat),
                       ("title page: email",
                        r"Email [^\s@]+@[^\s@]+\.[a-z]{2,}\.")):
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
    want("lab14 blur R2 floor", "0.456")

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
    rates, skill = {}, {}
    for ln in read_text(os.path.join(EXP, "lab10_loss_scale.txt")).splitlines():
        m = RATE.match(ln)
        if m:
            rates[int(m.group(1))] = m.groups()[1:]
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

    # No lab may quietly keep a short bootstrap block.  The block was widened
    # from 2h to 8h on measured dependence, and the change reached the labs that
    # imported it and missed the nine that spelled it out for themselves, which
    # is how an appendix table came to carry a narrower interval than the
    # headline for the same quantity.  Every lab is scanned for a block
    # expressed as a multiple of the horizon, and anything below lab05's is a
    # failure rather than a matter of taste.
    _blk_src = re.search(r"^N_BOOT, BLOCK, SEED = \d+, (\d+) \* HORIZON",
                         read_text(os.path.join(HERE, "labs",
                                                "lab05_robustness.py")), re.M)
    checks += 1
    if not _blk_src:
        fails.append("lab05 no longer declares its bootstrap block as a multiple of "
                     "the horizon, so the sweep below cannot compare against it")
    else:
        _mult = int(_blk_src.group(1))
        for _lp in sorted(glob.glob(os.path.join(HERE, "labs", "lab*.py"))):
            _ls = read_text(_lp)
            _short = [m for m in
                      re.findall(r"(?:BLOCK|BLK|blocks)\s*=\s*(?:\d+,\s*)?"
                                 r"(\d+)\s*\*\s*(?:L\.)?HORIZON", _ls)
                      if int(m) < _mult]
            checks += 1
            if _short:
                fails.append(f"{os.path.basename(_lp)}: bootstrap block still "
                             f"{_short[0]}h, against lab05's {_mult}h; a block shorter "
                             f"than the measured dependence returns intervals that are "
                             f"too narrow")

    # The companion note claimed for several drafts that "the close-to-close
    # proxy discards the overnight gap entirely", and read the two-proxy
    # comparison as a test of whether overnight moves matter.  It is wrong on
    # the arithmetic: log(C_s / C_{s-1}) = log(O_s / C_{s-1}) + log(C_s / O_s),
    # so a close-to-close return CONTAINS the overnight move and merely fails to
    # separate it.  What the comparison tests is the value of the richer
    # measurement.  Both the false claim and the reading it licensed are
    # forbidden, and the identity itself is required to be shown.
    if companion and os.path.isfile(companion):
        _cn = norm_iv(plain(companion))
        for _lbl, _n in (
                ("note overnight identity",
                 "log(C s /C s-1 ) = o s + c s"),
                ("note overnight correction",
                 "it contains it, unseparated"),
                ("note overnight reading",
                 "tests the value of the richer measurement, not whether overnight "
                 "moves matter")):
            checks += 1
            if norm_iv(_n.replace(MINUS, "-")) not in _cn:
                fails.append(f"{_lbl}: MISSING from the companion  '{_n}'")
        for _lbl, _n in (
                ("note stale overnight claim",
                 "discards the overnight gap entirely"),
                ("note stale overnight reading",
                 "a direct test of whether that gap carries regime information")):
            checks += 1
            if norm_iv(_n) in _cn:
                fails.append(f"{_lbl}: the companion still says '{_n}', which is "
                             f"arithmetically false")

    # The companion note's Gaussian benchmark, against lab02b.  Equation (3) is
    # an orthant probability and the note used to explain the excess over it by
    # heavy tails, which was never measured.  The non-centred version is now
    # computed, and the honest finding is that it does not explain the excess
    # either.  Both halves are checked: the figures, and the fact that the note
    # does not go back to claiming an explanation it does not have.
    if companion and os.path.isfile(companion):
        _l02b = lab("lab02b_threshold_ceiling")
        _sh = re.search(r"by ([\d.]+) points on average across all ten delays", _l02b)
        _ex = re.search(r"([-+][\d.]+) points above the centred benchmark, "
                        r"([-+][\d.]+) above the", _l02b)
        _bg = re.search(r"worth at most ([\d.]+) accuracy points over the\s+zero cut, "
                        r"at delta = (\d+), and at most ([\d.]+) points", _l02b)
        checks += 1
        if not (_sh and _ex and _bg):
            fails.append("lab02b no longer prints the non-centred benchmark the "
                         "companion note quotes")
        else:
            _c = norm_iv(plain(companion))
            for _lbl, _n in (
                    ("note centring shift", f"moves the benchmark by {_sh.group(1)} "
                                            f"points on average"),
                    ("note excess centred",
                     f"sits {_ex.group(1).lstrip('+')} points above the centred figure"),
                    ("note excess non-centred",
                     f"and {_ex.group(2).lstrip('+')} above the non-centred one"),
                    ("note bayes gain near",
                     f"worth at most {_bg.group(3)} accuracy points at any delay out "
                     f"to three weeks"),
                    ("note bayes gain far",
                     f"rising to {_bg.group(1)} points at {DELTA}= {_bg.group(2)}")):
                checks += 1
                if norm_iv(_n.replace(MINUS, "-")) not in _c:
                    fails.append(f"{_lbl}: MISSING from the companion  '{_n}'")
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
    want("9.3 exchange rate interval", "[1.4, 3.3]")

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
        _sites = re.findall(r"within ([\d.]+) (?:percentage )?points of the "
                            r"bootstrap's", txt)
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
    _r55 = re.search(r"\n\s+55\s+\d+%\s+\[\s*(\d+)%,\s*(\d+)%\]\s+"
                     r"\[\s*(\d+)%,\s*(\d+)%\]", l21)
    checks += 1
    if not _r55:
        fails.append("lab21 no longer prints the delta=55 Fieller and bootstrap pair")
    else:
        _bl, _bh, _fl, _fh = _r55.groups()
        want("S3 Fieller illustration",
             f"at &delta; = 55, [{_fl}, {_fh}] against [{_bl}, {_bh}]")

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
    checks += 1
    if "delays where the filter significantly wins: 2 of 6 [21, 55]" not in l33:
        fails.append("lab33: the filter no longer wins at exactly delta 21 and 55, "
                     "which Section 11 states")
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
            ("ruler width", "-7% as a share of the point", "7% narrower")):
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
    AGE48 = re.compile(r"^\s*(.+?)\s+(-?\d+)%\s+\[\s*(-?\d+)%,\s*(-?\d+)%\]\s+"
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
        _, _, _, pt, lo, hi, at0, off = cells[state]
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
    want("16 slope refused", "0.35 times the width")

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
    _src = read_text(path)
    _heads = {m.group(1) for m in re.finditer(r"<h2[^>]*>(\d+)\.\s", _src)}
    _heads |= {m.group(1) for m in re.finditer(r"<h3[^>]*>(\d+\.\d+)\s", _src)}
    for _n in sorted(set(re.findall(r"Section (\d+(?:\.\d+)?)", txt))):
        checks += 1
        if _n not in _heads:
            fails.append(f"the text refers to Section {_n}, which does not exist")
    # These four sections must still EXIST and still carry their subject.  The
    # check used to pin each to a number, which is right for one manuscript and
    # wrong for a journal variant that renumbers after relocating a section to
    # the appendix; the loop above already guarantees that every in-text
    # reference resolves to whatever numbering the document actually uses, so
    # what is left to protect is that a section has not been deleted or
    # retitled out of existence.
    _titles = " | ".join(m.group(1) for m in
                         re.finditer(r"<h2[^>]*>\d+\.\s*([^<]*)", _src))
    for _kw in ("Data", "Methodology", "The illiquid asset", "Limitations"):
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
    _m13 = re.search(r"\s*55\s+(\d+)% \[\s*(\d+)%,\s*(\d+)%\]\s+(\d+)% \["
                     r"\s*(\d+)%,\s*(\d+)%\]",
                     read_text(os.path.join(EXP, "lab13_origin_median.txt")))
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
        want("abstract leads with the implementable rate",
             f"returns {_op}% [{_ol}, {_oh}] of what eleven weeks of staleness")
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
    for ln in tail.splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)%\s*\[\s*(\d+)%,\s*(\d+)%\]\s+(\d+)%\s*$", ln)
        if m:
            n17 += 1
            h, pt, lo, hi, ql = m.groups()
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
    for ln in l23.split("C.  THE SUBSTITUTION RATE")[-1].splitlines():
        m = re.match(r"\s*(FULL|MEAN|PC1|PC2)\s+([\d.]+)%\s+\[\s*([\d.]+)%,\s*([\d.]+)%\]", ln)
        if m and m.group(1) in ("FULL", "MEAN", "PC1"):
            a, pt, lo, hi = m.groups()
            want(f"lab23 {a} rate", f"{pt}% [{lo}%, {hi}%]")
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
    l27 = read_text(os.path.join(EXP, "lab27_regime_conditioning.txt"))
    lad = l27.split("Reading the ladder")[-1].split("cells where")[0]
    R27 = re.compile(r"\s*([A-Za-z0-9 ,%/+-]+?)\s+([\d.]+)%\s+"
                     r"\[\s*([+-][\d.]+)%,\s*([+-][\d.]+)%\]\s*$")
    n27 = 0
    for ln in lad.splitlines():
        m = R27.match(ln)
        if m:
            n27 += 1
            _, pt, lo, hi = m.groups()
            want(f"lab27 ladder row {n27}", f"{pt}% [{lo}%, {hi}%]")
    checks += 1
    if n27 != 6:
        fails.append(f"lab27: parsed {n27} ladder rows, expected 6")
    checks += 1
    if "intervals disjoint: yes" not in l27:
        fails.append("lab27: the severest and calmest rungs no longer separate, but "
                     "Section 8.1 still says they do")
    want("lab27 direction", "it roughly doubles, and it rises monotonically")
    checks += 1
    if "on STRESS rungs: 0 of 16" not in l27:
        fails.append("lab27: the foreign block now adds given implied volatility on some "
                     "stressed rung, but Section 8.2 still says none")
    want("lab27 redundancy in stress", "the foreign block adds significantly in none")
    for tok, phrase, label in (
            ("324", "severest rung carries 324 days", "lab27 severest rung size"),
            ("367", "VIX top 5% 367 63.1%", "lab27 top-5% row"),
            ("1729", "VIX top 33% 1,729 43.3%", "lab27 top-33% row"),
            ("3074", "VIX bottom 67% 3,074 29.9%", "lab27 calm row")):
        checks += 1
        if tok not in l27:
            fails.append(f"{label}: '{tok}' not in lab27 output")
        want(label, phrase)

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
    want("9.1 level split", "scores -0.0428, which is worse, and adding the level "
                            "alone scores +0.0308")
    want("9.1 level grows with delay",
         "+0.0011 of R&sup2; at zero delay against +0.0736 at eleven weeks")

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
    for row in ("64% [49, 81]", "74% [64, 84]", "77% [68, 85]", "81% [74, 87]"):
        want("Table 20 stressed " + row.split("%")[0], row)

    # 9.3 data snooping
    pair("9.3 RC size", "Reality Check rejects at 5%: 5.5%", l34,
         "rejects 5.5% and 7.0% of the time")
    pair("9.3 survives", "Reality Check: 5 of 6 [3, 5, 13, 21, 55]", l34,
         "survives both tests at &delta; = 3, 5, 13, 21 and 55 with p below 0.0005")
    want("9.3 delta-0 absence", "the best margin across all nine models is +0.0014")

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
            _r = re.match(r"\s*(\d+)\s+-?[\d.]+\s+-?[\d.]+\s+(\d+)% \[\s*(-?\d+)%,"
                          r"\s*(-?\d+)%\]", _ln)
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
        if f"{pt}% [ {lo}%, {hi}%]" not in l11:      # lab11's own spacing
            fails.append(f"Table 21 {mk}: lab11 no longer prints {r55}")
        want(f"Table 21 row {mk}", f"{r5} {r13} {r21} {r55}")

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
    want("12 conditional limitation",
         "The unconditional rate averages two states in which the mechanism behaves "
         "differently")
    # Anchored on Section 13's own wording: the bare interval also appears in
    # 9.2, so a needle on it alone would pass a Section 13 that had gone stale.
    want("13 conditional recommendation",
         "about 80% of the delay-induced loss recovered in stressed markets, where "
         "the estimate is stable and tight")
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
    # The paper's intervals ARE a block bootstrap of the whole ratio; Fieller is
    # the CHECK on them, and lab21 prints the two side by side.  A draft of this
    # section claimed Fieller produced the published intervals, which the tables
    # contradict, so the claim is now pinned in both directions: the bootstrap
    # must be named as the source and Fieller as the check, and the paper must
    # not say the intervals are obtained from Fieller.
    want("4.5 names the bootstrap as the source",
         "The published intervals are a bootstrap of the whole ratio, and each is "
         "checked by inverting a HAC t-test after Fieller (1954)")
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
    # (12) is algebraically tied to the rate it "predicts" once S_own(delta) is
    # near zero, and the benchmark correction made the tie tighter rather than
    # looser.  The section is required to say so and forbidden from claiming an
    # unfitted prediction.
    want("6 ceiling is interpretation",
         "a factor-model interpretation of the observed plateau rather than a "
         "prediction of it")
    want("6 ceiling identity shown",
         "this collapses to S cross (55) &divide; S own (0), which is equation (12)")
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
    want("abstract exchange rate",
         "costs 2.1 days [1.4, 3.3] of domestic freshness")
    # scoped to the figure: the abstract now says "in this sample's stressed
    # markets", and a check that pinned the following word would fail on a
    # correctly hedged sentence.
    want("abstract conditional rate", "81% [74, 87] in")
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
            want("the paper names the envelope's section",
                 f"against the stronger control of Section S{_env.group(1)} of the "
                 f"Internet Appendix")
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
                             ("81% [74, 87]", "Section S10"),
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
    for _lbl, _needle in (
            ("Fieller is in the reference list", "Fieller, E. C. (1954)"),
            ("Dufour is in the reference list", "Dufour, J.-M. (1997)"),
            ("4.6 is a methodology subsection",
             "Inference for a ratio whose denominator can vanish</h3>")):
        checks += 1
        if _needle not in _paper_src:
            fails.append(f"{_lbl}: MISSING '{_needle}'")
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
    _journal_form = "In the wording requested by certain target journals" in _flat
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
        # the publisher's own template sentence, which is the whole reason the
        # journal form is longer than the SSRN one
        ("disclosure carries the publisher's template",
         "After using this tool, the author reviewed and edited the content as "
         "needed and takes full responsibility for the publication"),
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
        _NC = re.compile(r"\n\s*(\d+)\s+[-\d.]+\s+[-\d.]+\s+([\d.]+)%\s+[\d.]+%"
                         r"\s+([\d.]+)%\s+[\d.]+\s+([\d.]+)%")
        _nc = {m.group(1): (m.group(2), m.group(3), m.group(4))
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
                _sameNC, _bayes, _measured = _nc[_d]
                cwant(f"Table 3 delta={_d}",
                      f"{_d} {_rho.replace('-', MINUS)} {_same} {_bayes} "
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
                cwant("Table 3 caption explains the two columns",
                      "where &rho; &lt; 0 it falls below one half, which means the rule "
                      "is beaten by its own negation and cannot bound anything")
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
            _bad = [d for d in _nc if float(_nc[d][1]) < 50.0]
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
        checks += 1
        if len(_r61) != 12:
            fails.append(f"lab61: parsed {len(_r61)} of 12 summary lines for Table 6")
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
            _tot61 = re.search(r"of (\d+) significant AUC cells across all six "
                               r"schemes, (\d+) favour the threshold rule", l61)
            checks += 1
            if not _tot61:
                fails.append("lab61 no longer prints the direction tally Section 6.4 "
                             "quotes")
            elif _tot61.group(1) != _tot61.group(2):
                fails.append(f"lab61: {_tot61.group(2)} of {_tot61.group(1)} "
                             "significant AUC cells favour the threshold rule, but "
                             "Section 6.4 says all of them do")
            else:
                cwant("6.4 direction tally",
                      f"there are {spell(int(_tot61.group(1)))} significant AUC cells "
                      f"and all {spell(int(_tot61.group(2)))} favour the threshold rule")
            # the note must report the block it actually ran
            _nb61 = re.search(r"the note's block: (\d+)", l61)
            checks += 1
            if not _nb61:
                fails.append("lab61 no longer reports the note's own block length")
            else:
                cwant("3.4 block length",
                      f"with block length 8h = {_nb61.group(1)} and 2,000 resamples")
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
        cwant("lab25 headline", "the HAR classifier significantly beats the threshold "
                                "rule in none, and the threshold rule significantly beats "
                                "the HAR classifier in fifteen")
        cwant("lab25 power caveat", "moves that boundary back out from zero to three days")
        checks += 1
        _m25 = re.search(r"boundary on its own window: (\d+)", l25)
        if not _m25:
            fails.append("lab25 no longer prints the power control's boundary")

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
COMPANION = find_paper("threshold-rule-at-its-ceiling.html")


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
