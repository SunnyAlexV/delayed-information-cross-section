"""check_citation_only.py - prove that a lab's output changed ONLY in its
citations, before its stored reference output is replaced.

WHY THIS EXISTS

Twenty-odd labs print a sentence naming the paper section their result belongs
to, and a cross-reference audit found those names pointing at sections from two
earlier numberings of the manuscript.  Correcting them changes PRINTED output,
so every one of those labs now differs from the reference copy it is diffed
against, and the reference copies have to be refreshed.

Refreshing a reference output is the one operation in this repository that can
hide a real change: copy the new run over the old and whatever else moved moves
with it, silently, into the thing that defines correctness.  So the refresh is
gated on a proof.  Both sides are normalised by replacing every section and
table reference with a placeholder; if the normalised texts are then IDENTICAL,
the only difference was the citation, and nothing numerical came along with it.
Any lab that fails that test is printed in full and not refreshed.

    python check_citation_only.py            report, refresh nothing
    python check_citation_only.py --apply    refresh only the labs that pass
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
EXP = os.path.join(HERE, "expected_output")

# A reference, in the two shapes these labs print.  Matched in two passes.
#
# The PAIR pass handles the plural forms ("Tables S8 and S9", "Sections 9.1 and
# 10.2") and uses a BACKREFERENCE so both members are the same family - both
# plain or both S-prefixed.  Without that anchor the pattern swallowed whatever
# followed "and":
#
#     "see Table S30 and 77 cells are significant" -> "see <REF> cells ..."
#     "Table 5 and 3.2x inflation"                 -> "<REF>.2x inflation"
#
# which would let a count move from 77 to 83, or a ratio from 3.2 to 9.9, and
# still be classified as a citation change - and then --apply would write that
# over the reference output.  That substitution is exactly what this file exists
# to refuse, so it is refused by construction rather than by hoping no lab ever
# prints such a line.  The singular "Section 9 and ..." never extends, because a
# single reference followed by "and" is a sentence, not a pair.
#
# The ONE pass then handles single references.  Both end in (?!\d) rather than
# (?![\w.]) so that a reference closing a sentence - "Section S12." - still
# matches while "9.1" is not clipped to "9".
_PAIR = re.compile(r"\b(?:Sections|Tables) (S?)\d+(?:\.\d+)?\s+and\s+"
                   r"\1\d+(?:\.\d+)?(?!\d)")
_ONE = re.compile(r"\b(?:Sections?|Tables?) S?\d+(?:\.\d+)?(?!\d)")
# Lines run_all.py itself excludes from the comparison, for the same reasons
_IGNORE = (re.compile(r"^data folder:"), re.compile(r"^runtime "),
           re.compile(r"^panel: (?:/|[A-Za-z]:|<your data folder>)"))


def _lines(text):
    """The comparable lines, exactly as run_all.py --check compares them."""
    return [ln.rstrip() for ln in text.splitlines()
            if not any(p.match(ln) for p in _IGNORE)]


def _norm(text):
    """The same lines with every section and table reference blanked out."""
    return [_ONE.sub("<REF>", _PAIR.sub("<REF>", ln))
            for ln in _lines(text)]


def main(argv):
    apply = "--apply" in argv
    if not os.path.isdir(OUT):
        raise SystemExit("output/ does not exist; run run_all.py first")
    clean, dirty, same = [], [], []
    for f in sorted(os.listdir(EXP)):
        if not f.endswith(".txt"):
            continue
        p_new = os.path.join(OUT, f)
        if not os.path.isfile(p_new):
            continue
        new = open(p_new, encoding="utf-8").read()
        old = open(os.path.join(EXP, f), encoding="utf-8").read()
        a, b = _norm(old), _norm(new)
        # "unchanged" must be judged on the lines run_all.py actually compares,
        # not on the raw file: the data-folder and runtime lines differ on every
        # run by design, and comparing them made every lab look like it had a
        # citation change.  That is the same class of error as a check reading
        # the wrong surface, committed inside the tool built to prevent one.
        if a == b and _lines(old) == _lines(new):
            same.append(f)
        elif a == b:
            clean.append(f)
        else:
            dirty.append((f, a, b))

    print(f"{len(same)} unchanged, {len(clean)} changed in citations only, "
          f"{len(dirty)} changed in substance")
    for f in clean:
        print(f"  citation-only  {f[:-4]}")
    for f, a, b in dirty:
        print(f"\n  SUBSTANTIVE    {f[:-4]}")
        n = 0
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                n += 1
                if n <= 6:
                    print(f"      - {x[:150]}")
                    print(f"      + {y[:150]}")
        if len(a) != len(b):
            print(f"      (line count {len(a)} -> {len(b)})")
        if n > 6:
            print(f"      ... and {n - 6} more differing line(s)")

    if dirty:
        print("\nNothing was refreshed: a lab changed in substance, which is a "
              "result moving and not a citation being corrected.  Read the "
              "diff above before deciding.")
        return 1
    if apply:
        for f in clean:
            with open(os.path.join(OUT, f), encoding="utf-8") as fh:
                data = fh.read()
            with open(os.path.join(EXP, f), "w", encoding="utf-8") as fh:
                fh.write(data)
        print(f"\nrefreshed {len(clean)} reference output(s), all proved "
              f"citation-only")
    elif clean:
        print("\nRe-run with --apply to refresh those reference outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
