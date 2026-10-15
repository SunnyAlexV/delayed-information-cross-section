"""
sweep_lab_prose.py - the labs' own sentences, searched for figures nobody reads.

    python sweep_lab_prose.py              all three sweeps
    python sweep_lab_prose.py --quantifiers    counts, ratios and multiples only
    python sweep_lab_prose.py --numbers        decimal and percentage literals
    python sweep_lab_prose.py --crossrefs      figures beside another source's name

WHY THIS FILE EXISTS
--------------------
A lab's table is reproduced byte for byte on every run, so a figure in a table
cannot go stale without run_all.py --check failing.  A lab's PROSE is
reproduced just as faithfully, and that is the trap: a sentence that was typed
rather than computed is reproduced wrong forever.  The check confirms that the
wrong sentence has not changed.  It cannot tell that it was wrong.

Reading the labs turned up nineteen such sentences, among them

    lab29  "three to twelve times as many days per cell"   measured 1.4 to 6.2
    lab54  "Thirty to forty points of width"               measured 42 to 47
    lab61  "across all six schemes"                        it ran eight
    lab55  lab53's coupling range as 0.228 to 0.808        lab53 says -0.115
    lab58  two figures attributed to lab65's bandwidth 20  lab65 has no such column

Reading sixty-seven files is not a method that survives contact with a deadline,
so the surface is searched here instead.  The three sweeps look for the three
shapes the failure takes:

  --quantifiers  a count, a ratio or a multiple written as a word or a digit
                 and NOT adjacent to an interpolated value: "fifteen of the
                 twelve", "3 of the 4 delays", "8-fold", "four times".  This is
                 the sweep that found most of the nineteen.
  --numbers      a decimal or percentage literal in a printed string, which is
                 either a threshold the lab applies (fine, and usually named as
                 one) or a measurement it should be computing (not fine).
  --crossrefs    a figure in a literal that also names another lab, the note or
                 a table: one lab quoting another from memory, which is the
                 form that reads as precision and has the longest reach,
                 because the documents then quote the lab.

An interpolated field is treated as evidence that the figure is computed, so
f"{n} of {m} delays" never appears and "three of the four delays" always does.
The sweeps are deliberately noisy in one direction only: they over-report
structural constants ("seven foreign closes", "eight targets"), which are
cheap to dismiss, rather than under-reporting a claim.

A narrowed form of the first sweep runs inside verify_paper.py as a ratchet
with a bounded allowlist, so a NEW typed quantifier fails the verifier.  This
file is the wider search a person reads when adding one.
"""

import io
import glob
import os
import re
import sys
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))

WORDS = (r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
         r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
         r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|dozen)")

# A count, a ratio or a multiple.  The trailing context matters: "three of the
# four" is a count and "one of them" is a pronoun, and only the first can go
# stale against a table.
QUANTIFIER = re.compile(
    rf"(?:\b{WORDS}\b(?:[ -]{WORDS}\b)?\s+(?:of|to|times|in)\b"
    rf"|[\d.]+-fold"
    rf"|\b\d+ of (?:the )?\d+\b"
    rf"|\b\d+(?:\.\d+)? times\b)", re.I)

NUMBER = re.compile(r"[+-]?\d+\.\d+%?|\b\d{1,3}%")

CROSSREF = re.compile(r"lab\d\d|the note|Table S?\d+|Section S?\d+(?:\.\d+)?")

# Pronouns and set-membership phrases, which carry no count.
PRONOUN = re.compile(r"^(?:one|two|the one|one of)$", re.I)

# Section and table numbers are references, not measurements, and a format spec
# is not prose.
NOT_A_FIGURE = re.compile(r"^(?:5%|95%|10%|90%|2\.5%|97\.5%|100%|0\.05|1\.96)$")

SENTINEL = "\x00"          # where an interpolated field was


def literals(path):
    """Every printed string literal in a file, with interpolations blanked.

    Docstrings are excluded: they describe the code to a reader of the code,
    and a stale sentence there is a comment, not a printed claim.

    F-STRINGS AND THE TOKENISER.  Until Python 3.12 an f-string arrived here as
    one STRING token and was scanned like any other literal.  From 3.12 it
    arrives as FSTRING_START, then FSTRING_MIDDLE for each run of literal text,
    nested tokens for each replacement field, and FSTRING_END - so a sweep that
    reads STRING tokens alone sees every plain literal and NOT ONE f-string.
    Almost every sentence a lab prints is an f-string, which made this function
    nearly inert on a modern interpreter while still returning rows, and a sweep
    that returns fewer rows than it should is worse than one that crashes.  It
    was caught only because verify_paper.py fails when an allowlist entry stops
    matching, which is the lower bound that check exists to provide.

    So FSTRING_MIDDLE runs are reassembled here into one literal per f-string,
    with a sentinel where each replacement field was - the same surface the old
    tokenisation produced, so the allowlist and the sweeps are unchanged.
    """
    src = open(path, encoding="utf-8").read()
    out = []
    # Set while an f-string is open: [line, pieces].  FSTRING_START may nest
    # (an f-string inside a replacement field), so this is a stack.
    stack = []
    START = getattr(tokenize, "FSTRING_START", None)
    MIDDLE = getattr(tokenize, "FSTRING_MIDDLE", None)
    END = getattr(tokenize, "FSTRING_END", None)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if START is not None and tok.type == START:
            stack.append([tok.start[0], []])
            continue
        if stack and MIDDLE is not None and tok.type == MIDDLE:
            stack[-1][1].append(tok.string)
            continue
        if stack and END is not None and tok.type == END:
            line, pieces = stack.pop()
            # A replacement field sits between two MIDDLE runs, and produced no
            # MIDDLE token of its own, so join on the sentinel rather than "".
            s = SENTINEL.join(pieces)
            if stack:                       # an f-string inside a field: the
                stack[-1][1].append(s)      # outer literal absorbs it
            else:
                out.append((line, s, s))
            continue
        if stack:
            # Tokens inside a replacement field are code, not prose.  The
            # sentinel for the field is supplied by the join above.
            continue
        if tok.type != tokenize.STRING:
            continue
        s = tok.string
        if s.lstrip("rbuf").startswith(('"""', "'''")):
            continue
        out.append((tok.start[0], s, re.sub(r"\{[^{}]*\}", SENTINEL, s)))
    return out


def near_interpolation(blanked, start, end, pad=3):
    return SENTINEL in blanked[max(0, start - pad):end + pad]


def sweep(which):
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "labs", "*.py"))):
        name = os.path.basename(path)
        for line, raw, blanked in literals(path):
            if which == "crossrefs" and not CROSSREF.search(blanked):
                continue
            pat = {"quantifiers": QUANTIFIER, "numbers": NUMBER,
                   "crossrefs": NUMBER}[which]
            for m in pat.finditer(blanked):
                if near_interpolation(blanked, m.start(), m.end()):
                    continue
                tok = m.group(0)
                if which == "quantifiers" and PRONOUN.match(tok.strip()):
                    continue
                if which in ("numbers", "crossrefs") and NOT_A_FIGURE.match(tok):
                    continue
                rows.append((name, line, tok, " ".join(raw.split())[:96]))
    return rows


def main(argv):
    picks = [a[2:] for a in argv if a.startswith("--")]
    picks = picks or ["quantifiers", "numbers", "crossrefs"]
    total = 0
    for which in picks:
        rows = sweep(which)
        total += len(rows)
        print(f"\n{'=' * 78}\n{which}: {len(rows)} candidates\n{'=' * 78}")
        for name, line, tok, text in rows:
            print(f"{name}:{line}  [{tok}]  {text}")
    print(f"\n{total} candidates in all.  Each is a figure a lab PRINTS and does "
          f"not COMPUTE.\nMost are structural constants and are fine.  The test "
          f"for the rest is whether\nthe lab's own table would still agree with "
          f"the sentence after a rerun.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
