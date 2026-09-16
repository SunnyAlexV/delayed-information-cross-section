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
    txt = plain(path)
    fails, checks = [], 0

    def want(label, needle):
        nonlocal checks
        checks += 1
        if needle.replace(MINUS, "-") not in txt:
            fails.append(f"{label}: MISSING  '{needle}'")

    def forbid(label, needle):
        nonlocal checks
        checks += 1
        if needle.replace(MINUS, "-") in txt:
            fails.append(f"{label}: STALE    '{needle}' should not appear")

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
    for old in ("-0.0119", "-0.0141", "-0.0166", "-0.0232", "-0.0294", "-0.0547",
                "+0.0016", "+0.0135", "+0.0353", "+0.1063", "+0.1583", "+0.3152"):
        checks += 1
        if old.replace(MINUS, "-") in _t2:
            fails.append(f"superseded lab07 value: STALE '{old}' is still in "
                         f"Table 2's cost or gross row")
    forbid("superseded interval", "[-0.0077, +0.0104]")
    forbid("surrogate draw count", "ten draws")

    # --- sup-t and the GW row, against lab06 ------------------------------
    l6 = lab("lab06_inference")
    want("lab06 sup-t critical value", "6.51")
    for z in ("3.18", "4.48", "5.08", "5.34", "5.66", "6.21", "7.02"):
        if z not in l6:
            fails.append(f"lab06: {z} not in lab output but quoted as a GW statistic")
        want(f"lab06 GW {z}", z)

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
    # the third loss is quoted in the text rather than tabulated
    checks += 1
    if "90% [ 49%,105%]" not in l10:
        fails.append("lab10: the MSE(var) rate at delta=55 is no longer 90%, "
                     "which the text quotes")
    want("lab10 MSE(var) rate at 55", "90%")

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
    want("8.5 window share", "46% of Table 8's window")
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
             "moved no cell of Section 8 by as much as a thousandth")):
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
    for label, needle in (("SQ dR2 at delta=0", "-0.0131"),
                          ("SQ dR2 at delta=3", "-0.0436"),
                          ("FACTOR net of cost at delta=0", "+0.0115")):
        checks += 1
        if needle not in l09:
            fails.append(f"lab09 {label}: {needle} is no longer printed")
        want(f"lab09 {label}", needle)
    checks += 1
    if "SQ: beats LIN at 0 of 6 delays" not in l09:
        fails.append("lab09: squaring every input now beats the linear map "
                     "somewhere, which Section 5 says it never does")

    # --- Section 3's stated settings, against lab12 -----------------------
    # The settings paragraph reads like boilerplate, which is exactly why it
    # drifts: nothing in the paper changes when a window length in prose stops
    # matching the window the scripts use.  lab12 prints them; they are checked.
    l12 = " ".join(lab("lab12_appendix").split())
    for label, inlab, inpaper in (
            ("training window", "rolling training window 1250 days",
             "rolling 1,250-day window"),
            ("validation tail", "validation tail 250 days",
             "250-day validation tail"),
            ("refit frequency", "refit frequency every 21 days",
             "refitted every 21 days"),
            ("median window", "trailing median window 252 days",
             "trailing 252-day median"),
            ("HAC bandwidth", "9 lags at n = 4862",
             "Bartlett kernel at nine lags")):
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
    want("lab13 origin-dated rate",
         "the substitution rate is 71% against 72% at eleven weeks")

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
    want("lab16 disclosure", "7.1 A caveat on the Clark-West column")
    want("lab16 remedy is decisive",
         "imposing a common penalty moves no Clark-West statistic by more than 0.05")
    want("lab16 ratio not sold as a threshold",
         "We offer that ratio as a description, not a threshold")
    for phrase in ("the Clark-West column is invalid",
                   "we withdraw the Clark-West"):
        forbid("lab16 overclaim", phrase)
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
    OWN = {"1": "0.921", "5": "0.499", "10": "0.338", "21": "0.161"}
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
    want("lab17 h=1 own R2", "0.921")
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
    want("11 VSTOXX verdict",
         "the third series adds nothing at any delay and is mildly negative at all six")

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
        ctxt = plain(companion)
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
        checks += 1
        if "lab20_ceiling_is_not_a_ceiling.py" not in ctxt:
            fails.append("companion: does not name lab20_ceiling_is_not_a_ceiling.py")
        # Section 5 spends two paragraphs saying A* is not a ceiling; a table
        # column headed "ceiling" undid that, and is now forbidden rather than
        # merely fixed.
        for phrase in ("ceiling A*", "% of ceiling",
                       "sit between 90% and 106% of the ceiling"):
            checks += 1
            if phrase in ctxt:
                fails.append(f"companion: '{phrase}' calls the benchmark a ceiling, "
                             "which Section 5 spends two paragraphs denying")
        checks += 1
        if "Gaussian benchmark A*" not in ctxt:
            fails.append("companion: Table 3 no longer heads its column "
                         "'Gaussian benchmark A*'")
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
    for tok, phrase, label in (
            ("0.0060", "largest gap is 0.0060 of R&sup2; at &delta; = 13",
             "lab21 largest gap"),
            ("0.60", "which is 0.60 of the tolerance there",
             "lab21 closest call"),
            ("187", "about 187 test days per draw - 3.8% -",
             "lab21 test days lost per naive draw"),
            ("0.0207", "sits 0.0207 of R&sup2; below the corrected estimate",
             "lab21 naive penalty at d=0"),
            ("0.0011", "and 0.0011 below it at &delta; = 55",
             "lab21 naive penalty at d=55"),
            ("+0.1893", "scored +0.1893 on its own",
             "lab21 near-identity draw"),
            ("0.0709", "standard deviation of 0.0709",
             "lab21 near-identity spread")):
        checks += 1
        if tok not in l21:
            fails.append(f"{label}: NOT IN LAB OUTPUT  '{tok}'")
        want(label, phrase)
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
    want("lab21 Fieller endpoint gap", "2.4 percentage points")
    want("lab21 Fieller d=55", "[63%, 79%] against [64%, 78%]")
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
    if "2.4%" not in b21:
        fails.append("lab21 part B: the 2.4% endpoint gap is not in the lab output")

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
            fails.append(f"Table 16 row delta={d} is no longer what lab37 prints")
        want(f"Table 16 row delta={d}", row)

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
    # whole row and the PAPER side on the two columns Table 17 lifts from it.
    for labrow, papercols in (
            ("5 0.3371 0.3400 0.3262 0.3525 0.3505 0.3673", "5 0.3371 0.3505"),
            ("13 0.2011 0.2009 0.1791 0.1988 0.2226 0.1969", "13 0.2011 0.2226"),
            ("21 0.1121 0.1117 0.0806 0.1027 0.1448 0.0904", "21 0.1121 0.1448"),
            ("55 -0.0351 -0.0353 -0.0968 -0.0499 0.0172 -0.1100", "55 -0.0351 0.0172")):
        d = papercols.split()[0]
        checks += 1
        if labrow not in l38:
            fails.append(f"Table 17 delta={d}: lab38 no longer prints '{labrow}'")
        want(f"Table 18 row delta={d}", papercols)
    pair("9.1 cells improved", "significantly beats the paper's: 14 of 45", l38,
         "beats the paper's in 14 of 45 cells")
    pair("9.1 level does the work", "contributes more than the second measure: 9 of 9",
         l38, "at nine delays out of nine")
    pair("9.1 rate moves", "rate moves significantly with the control: 4 of 8", l38,
         "significantly at &delta; = 13, 21, 34 and 55")
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
            fails.append(f"Table 18 delta={d}: lab39 no longer prints the calm rate")
    for row in ("64% [49, 81]", "74% [64, 84]", "77% [68, 85]", "81% [74, 87]"):
        want("Table 18 stressed " + row.split("%")[0], row)

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
    for mk, r5, r13, r21, r55 in (
            ("SPX", "51% [37, 66]", "64% [53, 74]", "67% [58, 76]", "72% [64, 79]"),
            ("FTSE", "74% [56, 90]", "82% [70, 92]", "84% [74, 93]", "88% [80, 94]"),
            ("DAX", "55% [41, 68]", "69% [58, 78]", "71% [62, 78]", "74% [67, 80]"),
            ("HSI", "61% [34, 86]", "68% [50, 86]", "69% [54, 84]", "69% [58, 80]"),
            ("N225", "44% [17, 71]", "54% [33, 73]", "55% [36, 73]", "60% [45, 73]")):
        # Anchored on the WHOLE row: "60% [45, 73]" alone also appears in Table 18,
        # so a per-cell needle passed a deliberately corrupted Table 18 row.
        checks += 1
        pt, rng = r55.split("% [")
        lo, hi = rng.rstrip("]").split(", ")
        if f"{pt}% [ {lo}%, {hi}%]" not in l11:      # lab11's own spacing
            fails.append(f"Table 19 {mk}: lab11 no longer prints {r55}")
        want(f"Table 19 row {mk}", f"{r5} {r13} {r21} {r55}")

    # 9.1's normalisation test, against lab40
    l40 = lab("lab40_level_or_normaliser")
    checks += 1
    if "the implementable target keeps between 87% and 102% of the gain" not in l40:
        fails.append("lab40: the share retained is no longer 87% to 102%")
    want("9.1 normaliser test", "retains between 87% and 102% of the gain")
    checks += 1
    if "C t-delta 55 0.0168 0.0847 +0.0680" not in l40:
        fails.append("lab40: the delta=55 row on the implementable target has changed")
    want("9.1 normaliser numbers", "+0.0680 against +0.0667 at eleven weeks")
    checks += 1
    if "C t-delta 55 74% 70%" not in l40:
        fails.append("lab40: the rate correction on target C is no longer 74% to 70%")
    want("9.1 correction survives", "moves R(55) from 74% to 70%")

    # the conclusions must quote what Section 9 corrected, not what it replaced
    # 9.1 reports the control as a second specification, not a retraction; the
    # paper must not go back to telling the reader its own tables are wrong.
    forbid("10 asks the reader to correct the tables",
           "should subtract roughly four points from ours")
    want("10 conditional limitation",
         "the unconditional rate averages two states in which the mechanism behaves "
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
    want("3 vanishing denominator",
         "approaches zero and the ratio need not have a bounded confidence set")
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
    for _n, _kw in ((1, "Out-of-sample skill"), (8, "four information sets"),
                    (12, "with the horizons matched"),
                    (14, "when the foreign block is itself"),
                    (15, "Effective age"), (17, "by market state, pooled"),
                    (18, "taken apart"), (19, "by target market")):
        checks += 1
        if _n in _caps and _kw not in _re.sub(r"<[^>]+>", "", _caps[_n]):
            fails.append(f"Table {_n} is no longer the '{_kw}' table; a reference "
                         "to it now points somewhere else")

    # --- the operational recommendation, and the horizon note -------------
    l23 = lab("lab23_compressed_everywhere")
    checks += 1
    if "PC1 75.2% [ 68.2%, 81.8%]" not in l23:
        fails.append("lab23: R(55) for the compressed block is no longer 75.2%")
    want("14 compression is the model to run",
         "a single real-time factor beating all seven series at every delay and raising "
         "R(55) to 75.2%")
    want("14 seven regressors not recommended",
         "compress the cross-section first and treat the seven-regressor figures as the "
         "conservative benchmark")
    want("8 horizon mismatch",
         "measured over a horizon six times longer than the thing it is forecasting")
    want("8 matched instrument named", "a nine-day index, VIX9D, formerly VXST")
    # the 81% sentence was duplicated once already
    checks += 1
    if txt.count("81% in each half of the sample taken separately") != 2:
        fails.append("the 'each half' sentence appears "
                     f"{txt.count('81% in each half of the sample taken separately')} "
                     "times; it belongs once in Section 10.2 and once in Section 14")

    # --- Table 18, the ratio taken apart, against lab41 -------------------
    for row in ("5 0.0284 0.1567 5.5x 0.0673 0.2434 3.6x",
                "13 0.0345 0.3787 11.0x 0.0798 0.5110 6.4x",
                "21 0.0412 0.5224 12.7x 0.1003 0.6755 6.7x",
                "55 0.0583 0.8043 13.8x 0.1234 0.9927 8.0x"):
        d = row.split()[0]
        checks += 1
        if row not in l41:
            fails.append(f"Table 18 delta={d}: lab41 no longer prints '{row}'")
        want(f"Table 18 row delta={d}", row.replace("x", "&times;"))
    checks += 1
    if "stress multiplies N by 10.7 and D by 6.2" not in l41:
        fails.append("lab41: the N and D multipliers are no longer 10.7 and 6.2")
    want("9.2 numerator moves", "stress multiplies N by 10.7 and D by 6.2")
    want("9.2 absolute reading",
         "removes 0.8043 of squared error in stressed markets against 0.0583 in calm")

    # --- Section 6, the model, against lab42 ------------------------------
    l42 = lab("lab42_factor_model")
    for labn, papern, label in (
            ("fit R-squared = 0.9984", "R&sup2; = 0.9984", "6 decay fit"),
            ("slope = -0.07217 per day", "slope -0.07217 per day", "6 decay slope"),
            ("implied persistence phi = exp(slope/2) = 0.9646", "implying &phi; = 0.9646",
             "6 implied persistence"),
            ("measured persistence of the factor = 0.9678", "is 0.9678",
             "6 measured persistence"),
            ("predicted asymptote R(infinity) = 69.6%", "predicted ceiling of 69.6%",
             "6 predicted ceiling")):
        checks += 1
        if labn not in l42:
            fails.append(f"{label}: lab42 no longer prints '{labn}'")
        want(label, papern)
    checks += 1
    if "4 of 4 predictions hold" not in l42:
        fails.append("lab42: the model's predictions no longer all hold, but Section 6 "
                     "presents them as holding")
    want("6 agreement", "The two agree to 0.0033")
    want("6 ceiling met", "reaches 69.8% at &delta; = 34 and 71.6% at &delta; = 55")
    want("6 flatness", "it falls 29%, against 107% for the domestic model")
    want("6 what it cannot do", "What the model does not explain is why the mechanism "
                                "is stronger in stress")

    # the abstract's three claims, each tied to the section that establishes it
    want("abstract effective age",
         "forecasts as well as a mark 4.6 days old [2.8, 7.7]")
    want("abstract exchange rate",
         "costs 2.1 days [1.4, 3.3] of domestic freshness")
    want("abstract conditional rate",
         "81% [74, 87] at eleven weeks and identical in both halves")
    want("abstract stronger control", "it is 68% rather than 72%")
    want("abstract normalisation finding",
         "restoring it is worth more at long delays than the entire cross-section")
    want("abstract compression advice",
         "one real-time factor, or an equal-weighted mean, does as well for a "
         "fraction of the estimation cost")

    # The recommendation in Section 12 quotes Table 15 back at the reader; the
    # table rows are checked above, this checks that the sentence still agrees
    # with them rather than drifting into a rounder-sounding claim.
    want("12 foreign-feed condition",
         "turns Table 14's 36% into 9% at three days and its 51% into 29% at five")

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

    l03 = lab("lab03_crosssection")
    checks += 1
    if l03.count("-> VERIFIED") != 7 or "seam SPX: only 0 overlapping rows" not in l03:
        fails.append("lab03: the seam audit no longer shows seven verified index "
                     "seams and an unverifiable SPX one, which Section 9 now states")
    want("9 unverified seam",
         "eight overlap and agree exactly on every overlapping row, while the "
         "S&P's two files abut without overlapping")

    # --- Section 3 must say why the NON-robust loss carries the headline ----
    want("3 headline-loss defence",
         "the headline number in this paper is the one loss of the three that is "
         "NOT proxy-robust")
    want("3 headline-loss reason", "a ratio needs a denominator a reader can hold")

    # --- Section 12 must name every script the paper's numbers come from ---
    # This check caught two real misses.  Sections 5.3, 8.1 and 8.2 were added
    # citing lab27 and lab28 while the text still said "twenty scripts" and
    # listed neither; later lab41 and lab42 supplied the whole of Section 6 and
    # part of Section 10.2 without ever being named, so a reader could not find
    # where those figures came from.  The list is no longer typed here: it is
    # whatever this verifier actually read while checking the main paper, so a
    # lab added tomorrow is covered the moment its output is checked.
    CITED = sorted(USED)
    for name in CITED:
        checks += 1
        if name + ".py" not in txt:
            fails.append(f"Section 12 does not name {name}.py, whose figures the "
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
    if f"one of {spell(len(CITED))} scripts" not in txt:
        fails.append(f"Section 12's script count does not say "
                     f"'{spell(len(CITED))}', but {len(CITED)} are checked")
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
        ctxt = plain(companion)
        rows = {}
        for ln in read_text(
                os.path.join(EXP, "lab02b_threshold_ceiling.txt")).splitlines():
            m = re.match(r"\s*(\d+)\s+([-+]?[\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s+"
                         r"([\d.]+)%\s+([\d.]+)%\s*$", ln)
            if m:
                rows[int(m.group(1))] = m.groups()[1:]
        if not rows:
            fails.append("lab02b: could not parse the ceiling table at all")
        for d in (0, 1, 3, 5, 8, 13, 21, 55):
            checks += 1
            if d not in rows:
                fails.append(f"lab02b: no delta={d} row in the lab output")
                continue
            rho, ceil, obs, har, pct = rows[d]
            row = f"{d} {rho} {ceil} {obs} {har} {pct}"
            if row not in ctxt:
                fails.append(f"companion delta={d} row: MISSING  '{row}'")
        # The note must name the script that actually produces its table.
        checks += 1
        if "ceiling_check.py" in ctxt:
            fails.append("companion: still credits ceiling_check.py, which is gone")
        checks += 1
        if "lab02b_threshold_ceiling.py" not in ctxt:
            fails.append("companion: does not name lab02b_threshold_ceiling.py")

        # --- Section 6.1, against lab24 -------------------------------------
        def cwant(label, needle):
            nonlocal checks
            checks += 1
            if needle.replace(MINUS, "-") not in ctxt:
                fails.append(f"{label}: MISSING  '{needle}'")

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
        for tok, phrase, label in (
                ("-0.0299", "-0.0299 with a block-bootstrap interval of "
                            "[-0.0542, -0.0066]", "lab24 AUC gap at d=0"),
                ("0.055", "by as much as 0.055 of AUC at &delta; = 0",
                 "lab24 discretising cost")):
            checks += 1
            if tok not in l24:
                fails.append(f"{label}: '{tok}' not in lab24 output")
            cwant(label, phrase)
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
        checks += 1
        if worse != [21, 34, 55]:
            fails.append(f"lab24: the Brier deterioration is now at {worse}, but the "
                         "note says delta = 21, 34 and 55")
        cwant("lab24 Brier deterioration",
              "significantly so at &delta; = 21, 34 and 55")

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
        if "boundary on its own window: 3" not in l25:
            fails.append("lab25: the power control no longer gives a boundary of 3")

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
    main(path, COMPANION if len(sys.argv) <= 1 and os.path.exists(COMPANION) else None)
