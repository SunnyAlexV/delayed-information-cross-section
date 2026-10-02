"""
lab62_post_selection.py - how many of the secondary tables' significant cells
should be there by chance alone?

Reads the stored output of the labs it audits; needs no data and no imports
from the other labs.  Runtime under a second.

THE OBJECTION
-------------
lab34 already prices the search behind the HEADLINE: nine specifications, a
Reality Check and an SPA test, and the headline survives both.  That covers the
one comparison the paper was designed around.  It covers nothing else.

After the headline specification was fixed, the paper and its appendix went on
to report several wide secondary grids - seven markets by four delays here, four
unoptioned targets by four delays there, four nonlinear arms by six delays - and
marked individual cells significant inside them at the ordinary 5% level.  No
family-wise or false-discovery control was applied across those grids, and none
of them was pre-specified.  A reader is entitled to know how many of those cells
would be significant if every one of them were noise.

WHAT THIS FILE DOES
-------------------
It enumerates the secondary families by name, counts their cells, counts the
cells significant at 5% two-sided, and compares that count with 0.05n.  Then it
pools every secondary cell in the paper and applies two corrections to the pool:

    Bonferroni      controls the probability of ANY false rejection.  Severe,
                    and the right instrument if a single cell is to be leaned on.
    Benjamini-      controls the expected SHARE of rejections that are false.
    Hochberg        The right instrument for a grid read as a pattern, which is
                    how these grids are read.

Cells that survive neither are exploratory: real enough to report, not firm
enough to carry a claim, and the papers are required to say so in those words.

WHAT THIS FILE IS NOT
---------------------
It is not a retraction of the headline.  The headline is a single pre-specified
comparison whose multiplicity lab34 already prices, and it is deliberately NOT
in the pool below: mixing it in would let the secondary grids dilute a test that
was never part of them.  Nor is a cell that fails a correction thereby false.
The verdict here is about what may be CLAIMED, not about what is true.
"""

import os
import re
import sys
from math import erfc, sqrt

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
EXP = os.path.join(os.path.dirname(HERE), "expected_output")
if not os.path.isdir(EXP):
    EXP = os.path.join(HERE, "expected_output")

ALPHA = 0.05

# Each family: a display name, the lab whose output holds it, the section the
# rows live in, a row pattern, and which capture groups are z-statistics.  The
# families are named rather than discovered, because "every number in the repo
# that looks like a z" would sweep in the headline and the diagnostics and make
# the correction meaningless.
FAMILIES = [
    ("compressed foreign block given implied volatility, seven markets",
     "lab52_compressed_foreign_options",
     None,
     r"^\s*([A-Z0-9]+)\s+(\d+)\s+[-+]?[\d.]+\s+[-+][\d.]+\s+([-+][\d.]+)\s+"
     r"[-+][\d.]+\s+([-+][\d.]+)\s*$",
     (3, 4)),
    ("seven foreign closes given implied volatility, five optioned markets",
     "lab51_foreign_options",
     None,
     r"^\s*(\d+)\s+[-+]?[\d.]+\s+[-+]?[\d.]+\s+[-+]?[\d.]+\s+[-+]?[\d.]+\s+"
     r"(?:nan%|[-+]?[\d.]+%)\s+(?:nan%|[-+]?[\d.]+%)\s+([-+][\d.]+)\s+"
     r"([-+][\d.]+)\s*$",
     (3,)),
    ("breadth given implied volatility, four unoptioned targets",
     "lab53_no_options_targets",
     None,
     r"^\s*(\d+)\s+[-+]?[\d.]+\s+[-+]?[\d.]+\s+[-+]?[\d.]+\s+"
     r"(?:nan%|[-+]?[\d.]+%)\s+(?:nan%|[-+]?[\d.]+%)\s+[-+][\d.]+\s+"
     r"([-+][\d.]+)\s+[-+][\d.]+\s+([-+][\d.]+)\s*$",
     (2, 3)),
    ("nonlinearity given implied volatility, four arms",
     "lab15_nonlinear_given_iv",
     "ridge grid: PAPER",
     r"^\s*(\d+)\s+[\d.]+((?:\s+[-+][\d.]+\s+([-+][\d.]+)\s+[\d.]+){4})\s*$",
     "inline"),
    ("VSTOXX as a third implied-volatility series",
     "lab19_vstoxx_third_series",
     "B.  DOES THE THIRD SERIES ADD",
     r"^\s*(\d+)\s+[-+]?[\d.]+\s+[-+]?[\d.]+\s+[-+]?[\d.]+\s+[-+][\d.]+\s+"
     r"([-+][\d.]+)\s+[\d.]+\s*$",
     (2,)),
]


def two_sided_p(z):
    """Two-sided normal p-value, via the complementary error function."""
    return erfc(abs(z) / sqrt(2.0))


def read(name):
    with open(os.path.join(EXP, name + ".txt"), encoding="utf-8") as fh:
        return fh.read()


def collect(name, lab, anchor, pat, groups):
    """Every z-statistic in one family, with a label for each."""
    txt = read(lab)
    if anchor:
        if anchor not in txt:
            return None
        # A section header sits BETWEEN two rule lines, so the body is the
        # block after the first rule that follows the anchor, not the text
        # before it.  Taking the text before it is how the VSTOXX family
        # silently parsed nothing on the first run; the missing-family report
        # in part A exists because that failure must be loud.
        rest = txt.split(anchor, 1)[1]
        # Two anchor shapes appear in these files.  A section TITLE sits between
        # two rule lines, so its rows are in the block after the next rule; a
        # dashed sub-heading is followed directly by its rows.  Which one this
        # is can be read off the distance to the next rule: a title's trailing
        # rule arrives within a few characters, a sub-heading's next rule only
        # after the table.  Getting this wrong yields zero rows, which is why
        # part A reports an empty family as INCOMPLETE rather than as zero.
        nxt = re.search(r"\n=+\n", rest)
        if nxt and nxt.start() < 80:
            rest = rest[nxt.end():]
        txt = re.split(r"\n=+\n", rest)[0]
    out = []
    rx = re.compile(pat, re.M)
    for m in rx.finditer(txt):
        tag = m.group(1)
        if groups == "inline":
            # the four-arm grid repeats "dR2 z p" across one line
            for z in re.findall(r"[-+][\d.]+\s+([-+][\d.]+)\s+[\d.]+", m.group(2)):
                out.append((f"{tag}", float(z)))
        else:
            for g in groups:
                out.append((f"{tag}", float(m.group(g))))
    return out


def bh(ps, alpha=ALPHA):
    """Benjamini-Hochberg: the largest k with p_(k) <= k*alpha/m, then all below."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    kmax = 0
    for rank, i in enumerate(order, start=1):
        if ps[i] <= rank * alpha / m:
            kmax = rank
    keep = set(order[:kmax])
    return keep, (kmax * alpha / m if kmax else 0.0)


def main(_path=None):
    print("Prints the post-selection audit behind Section S33 and the")
    print("'exploratory' label the secondary tables now carry.\n")

    print("=" * 96)
    print("A.  THE SECONDARY FAMILIES, ONE AT A TIME")
    print("=" * 96)
    print("Each family is a grid the paper reports AFTER the headline")
    print("specification was fixed, and inside which individual cells are")
    print("marked significant at the ordinary 5% level. 'expected' is 0.05n,")
    print("the number of significant cells a grid of pure noise would show.\n")
    print(f"{'family':>62}{'cells':>7}{'sig 5%':>8}{'expected':>10}{'ratio':>8}")
    pool, missing = [], []
    for nm, lab, anchor, pat, groups in FAMILIES:
        got = collect(nm, lab, anchor, pat, groups)
        if not got:
            missing.append(nm)
            print(f"{nm[:62]:>62}{'PARSED NOTHING':>33}")
            continue
        zs = [z for _t, z in got]
        k = sum(abs(z) > 1.959963985 for z in zs)
        exp = ALPHA * len(zs)
        print(f"{nm[:62]:>62}{len(zs):>7}{k:>8}{exp:>10.1f}"
              f"{(k / exp if exp else float('nan')):>8.1f}")
        pool.extend((nm, t, z) for t, z in got)
    if missing:
        print(f"\n  {len(missing)} family(ies) parsed nothing; the audit is "
              f"INCOMPLETE and must not be quoted:")
        for nm in missing:
            print(f"    {nm}")

    print("\n" + "=" * 96)
    print("B.  THE POOLED CORRECTION")
    print("=" * 96)
    print("Every secondary cell above, pooled, because the papers read these")
    print("grids together. The headline comparison is deliberately excluded:")
    print("lab34 prices its own search, and pooling it here would let these")
    print("grids dilute a test that was never part of them.\n")
    ps = [two_sided_p(z) for _n, _t, z in pool]
    m = len(ps)
    k_unc = sum(p <= ALPHA for p in ps)
    bonf = ALPHA / m if m else float("nan")
    k_bonf = sum(p <= bonf for p in ps)
    keep, cut = bh(ps)
    print(f"  secondary cells pooled                    {m}")
    print(f"  significant at 5% uncorrected             {k_unc}")
    print(f"  expected by chance if all were noise      {ALPHA * m:.1f}")
    print(f"  Bonferroni threshold (5% family-wise)     {bonf:.6f}")
    print(f"  surviving Bonferroni                      {k_bonf}")
    print(f"  Benjamini-Hochberg cutoff (5% FDR)        {cut:.6f}")
    print(f"  surviving Benjamini-Hochberg              {len(keep)}")
    print(f"  significant uncorrected but not under BH  {k_unc - len(keep)}")

    print("\n" + "=" * 96)
    print("C.  WHICH CELLS SURVIVE, AND BY WHICH FAMILY")
    print("=" * 96)
    print("A cell that survives Benjamini-Hochberg may carry a claim. One")
    print("significant only uncorrected is exploratory: worth reporting,")
    print("not worth leaning on.\n")
    print(f"{'family':>62}{'sig 5%':>8}{'BH':>5}{'Bonf':>6}")
    for nm, _lab, _a, _p, _g in FAMILIES:
        idx = [i for i, (n, _t, _z) in enumerate(pool) if n == nm]
        if not idx:
            continue
        s = sum(ps[i] <= ALPHA for i in idx)
        b = sum(i in keep for i in idx)
        bo = sum(ps[i] <= bonf for i in idx)
        print(f"{nm[:62]:>62}{s:>8}{b:>5}{bo:>6}")

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    if k_unc <= ALPHA * m:
        print("  The secondary grids show no more significant cells than noise")
        print("  would produce. Every one of them is exploratory and the papers")
        print("  must say so; none may carry a claim on its own.")
    elif len(keep) == 0:
        print(f"  {k_unc} cells are significant uncorrected against {ALPHA * m:.1f}")
        print("  expected by chance, so the grids collectively carry signal, but")
        print("  NOT ONE cell survives a false-discovery correction across the")
        print("  secondary set. Every secondary cell is therefore exploratory and")
        print("  the papers must label them so.")
    else:
        print(f"  {len(keep)} of {k_unc} nominally significant secondary cells survive")
        print("  a 5% false-discovery correction across the whole secondary set.")
        print("  Those may carry a claim; the remaining")
        print(f"  {k_unc - len(keep)} are exploratory and the papers must label them so.")
    print()
    print("  In no case does this touch the headline, which is one pre-specified")
    print("  comparison with its own Reality Check and SPA test in lab34.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
