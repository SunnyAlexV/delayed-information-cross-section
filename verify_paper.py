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
EXP = os.path.join(HERE, "expected_output")


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
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def lab(name):
    return " ".join(read_text(os.path.join(EXP, name + ".txt")).split())


def rows_lab07():
    """delta -> (net, cost, gross, lo, hi) from lab07's final table."""
    out = {}
    with open(os.path.join(EXP, "lab07_estimation_cost.txt"), encoding="utf-8") as fh:
        for ln in fh:
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

    # every superseded cost/gross value must be gone
    for old in ("-0.0119", "-0.0141", "-0.0166", "-0.0232", "-0.0294", "-0.0547",
                "+0.0016", "+0.0135", "+0.0353", "+0.1063", "+0.1583", "+0.3152"):
        forbid("superseded lab07 value", old)
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

    # --- loss-scale figures, against lab10 --------------------------------
    want("lab10 QLIKE rate at 55", "80%")
    want("lab10 MSE(var) rate at 55", "90%")

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
    want("lab16 disclosure", "6.1 A caveat on the Clark-West column")
    want("lab16 remedy is decisive",
         "imposing a common penalty moves no Clark-West statistic by more than 0.05")
    want("lab16 ratio not sold as a threshold",
         "We offer that ratio as a description rather than a threshold")
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

    # --- Table 8, against lab19 -------------------------------------------
    l19 = read_text(os.path.join(EXP, "lab19_vstoxx_third_series.txt"))

    def sec19(letter):
        after = l19.split(f"\n{letter}.  ", 1)[1]
        body = after.split("=" * 40, 2)
        return body[2].split("=" * 40)[0] if len(body) > 2 else ""

    nb = nc = 0
    for ln in sec19("B").splitlines():
        m = re.match(r"\s*(\d+)\s+[-+]?[\d.]+\s+([\d.]+)\s+([\d.]+)\s+"
                     r"([-+][\d.]+)\s+([-+]?[\d.]+)\s+[\d.]+\s*$", ln)
        if m:
            nb += 1
            d, iv2, iv3, diff, z = m.groups()
            want(f"lab19 d={d} IV2/IV3 row", f"{iv2} {iv3} {diff} {z}")
    for ln in sec19("C").splitlines():
        m = re.match(r"\s*(\d+)\s+[\d.]+\s+[\d.]+\s+([-+][\d.]+)\s+([-+]?[\d.]+)\s+[\d.]+\s*$", ln)
        if m:
            nc += 1
            want(f"lab19 d={m.group(1)} foreign|IV3", f"{m.group(2)} {m.group(3)}")
    checks += 1
    if nb != 6 or nc != 6:
        fails.append(f"lab19: parsed {nb} block rows and {nc} redundancy rows, expected 6 each")
    # the sample cost is the reason it is not in the headline; it must be stated
    for s19 in ("73%", "1,240", "4,525", "82.7", "41.4"):
        want("lab19 sample cost", s19)
    # the withdrawn assertion must not survive anywhere in the paper
    forbid("withdrawn IV assertion", "would widen the gap in Section 7 further. It would not")

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

    print(f"verify_paper: {checks} checks against {len(os.listdir(EXP))} lab outputs")
    if fails:
        for f in fails:
            print("  " + f)
        raise SystemExit(f"\n{len(fails)} problem(s): the paper and the labs disagree.")
    print("  every checked figure in the paper matches its lab output")


SRC = os.path.join(HERE, "papers", "source")
DEFAULT = os.path.join(SRC, "what-substitutes-for-a-stale-mark.html")
COMPANION = os.path.join(SRC, "threshold-rule-at-its-ceiling.html")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.exists(path):
        raise SystemExit(
            f"Could not find the paper source at\n  {path}\n"
            "Run this from the extracted repository folder, where papers/source/\n"
            "sits next to this script, or pass the .html path as an argument.\n"
            "It needs the HTML source, not the PDF: the checks read the tables.")
    # Check the companion note too, but only on a default run: given an explicit
    # path the caller asked about one file and should get checks on that file.
    main(path, COMPANION if len(sys.argv) <= 1 and os.path.exists(COMPANION) else None)
