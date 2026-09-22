"""
verify_letters.py - the cover letters, checked against the things they describe.

    python verify_letters.py [directory]      default: ./submission

WHY THIS FILE EXISTS
--------------------
A cover letter is the one document in a submission that nobody re-derives.  It
is written once, and then the paper is revised - a check is added, a section is
relocated, a page count changes - and the letter goes on stating the old
numbers to the one reader who decides whether the paper is sent out at all.
That is this project's standard failure: a number copied rather than referenced.

The letters here claim seven kinds of fact, and each is recomputed rather than
trusted:

    * the number of analysis scripts            labs/lab*.py
    * the number of verifier checks             verify_paper.py's own report
    * the manuscript's word count               the built variant
    * its double-spaced page counts             measure_pages.py's two renders
    * the number of input files, and how        data/MANIFEST.tsv
      many may be redistributed
    * the headline figures it quotes            the variant's own abstract
    * the journal it names                      build_variants.py's spec

The page-count check is the slow one, because it renders.  --fast skips it and
says so, which is right while editing the prose and wrong before sending.
"""

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VAR = os.path.join(HERE, "papers", "variants")
PAPER = "what-substitutes-for-a-stale-mark"

# letter file -> variant directory
LETTERS = {"IJF-cover-letter.md": "ijf",
           "JEF-cover-letter.md": "jef",
           "JFEc-cover-letter.md": "jfec"}

def words(path):
    return len(" ".join(re.sub(r"<[^>]+>", " ", open(path, encoding="utf-8").read())
                        .split()).split())


def checks_for(variant):
    """The verifier's own count, taken from the verifier rather than recalled."""
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "verify_paper.py"),
         os.path.join(VAR, variant, PAPER + ".html"),
         os.path.join(HERE, "papers", "source", "threshold-rule-at-its-ceiling.html")],
        capture_output=True, text=True)
    m = re.search(r"verify_paper: ([\d,]+) checks", r.stdout)
    ok = "every checked figure" in r.stdout
    return (int(m.group(1).replace(",", "")) if m else None), ok


def page_counts():
    """Both columns from measure_pages.py, parsed out of its own output."""
    r = subprocess.run([sys.executable, os.path.join(HERE, "measure_pages.py")],
                       capture_output=True, text=True)
    out = {}
    for ln in r.stdout.splitlines():
        m = re.match(r"\s+(\S+)/(\S+)\s+(\d+) pp\s+(\d+) pp", ln)
        if m:
            out[(m.group(1), m.group(2))] = (int(m.group(3)), int(m.group(4)))
    return out


def main(argv):
    fast = "--fast" in argv
    where = next((a for a in argv[1:] if not a.startswith("-")),
                 os.path.join(HERE, "submission"))
    n_labs = len([f for f in os.listdir(os.path.join(HERE, "labs"))
                  if re.fullmatch(r"lab\d+[a-z]?_.*\.py", f)])
    man = [l.split("\t") for l in
           open(os.path.join(HERE, "data", "MANIFEST.tsv"),
                encoding="utf-8").read().splitlines()[1:] if l.strip()]
    n_inputs = len(man)
    n_free = sum(1 for r in man if r[2].strip() == "yes")
    pages = {} if fast else page_counts()

    # The cover letters are correspondence, not code, and whether they belong in
    # a public repository is the author's call.  If the directory is absent the
    # check reports that and passes, rather than failing every clone that does
    # not carry them - a step in make_all.py that fails by default for everyone
    # but the author is a step people learn to ignore.
    if not os.path.isdir(where):
        print("verify_letters: no submission/ directory, so the cover letters "
              "are not\n  part of this copy; nothing to check")
        return 0

    fails, seen = [], 0
    for fn, key in sorted(LETTERS.items()):
        path = os.path.join(where, fn)
        if not os.path.isfile(path):
            fails.append(f"{fn}: not found in {os.path.relpath(where, HERE)}")
            continue
        seen += 1
        txt = open(path, encoding="utf-8").read()
        src = os.path.join(VAR, key, PAPER + ".html")
        body = open(src, encoding="utf-8").read()

        # the counts the letter asserts about the package
        n_checks, ok = checks_for(key)
        if not ok:
            fails.append(f"{fn}: the variant it describes does not pass its checks")
        # Each fact gets the exact phrasing the letters use, not a proximity
        # rule.  The first version of this looked for a number within sixty
        # characters of "script" and reported "says 12 where analysis scripts
        # is 63" - it had matched "12-point serif ... the manuscript", because
        # MANUSCRIPT contains SCRIPT.  A check that cries wolf is worse than no
        # check: it is read once and then skipped.  A letter need not state any
        # of these; it must not state one wrongly, so an absent phrase passes
        # and a present one is compared exactly.
        for what, pat, val in (
                ("analysis scripts", r"\b([\d,]+) (?:analysis )?scripts\b", n_labs),
                ("verifier checks", r"\b([\d,]+) checks\b", n_checks),
                ("manuscript words", r"manuscript is ([\d,]+) words\b", words(src)),
                ("input files", r"\b(?:of )?([\d,]+|thirty-five) input files\b",
                 n_inputs)):
            for m in re.finditer(pat, txt, re.I):
                g = m.group(1)
                got_n = (35 if g.lower() == "thirty-five"
                         else int(g.replace(",", "")))
                if got_n != val:
                    fails.append(f"{fn}: says {g} where {what} is {val}")
        if "thirty-five" in txt.lower() and n_inputs != 35:
            fails.append(f"{fn}: 'thirty-five' input files is now {n_inputs}")
        if re.search(r"\bone is public domain\b", txt) and n_free != 1:
            fails.append(f"{fn}: 'one is public domain', but the manifest marks "
                         f"{n_free} redistributable")

        # the page counts, which are the numbers most likely to go stale
        if not fast:
            floors, strict = pages.get((key, PAPER), (None, None))
            if floors is None:
                fails.append(f"{fn}: no page measurement for {key}")
            else:
                stated = set(re.findall(r"(\d+) pages", txt)) | \
                         set(re.findall(r"is (\d+) pages", txt))
                stated = {int(x) for x in stated}
                if stated and not stated <= {floors, strict}:
                    fails.append(f"{fn}: states pages {sorted(stated)}; measured "
                                 f"{floors} (floats single) and {strict} (all double)")

        # Figures quoted from the paper must be IN the paper.  Read out of the
        # LETTER, not from a list of expected quotes: a list only fires on the
        # quotes still present, so editing 71% [59, 81] to 71% [63, 79] simply
        # removed the trigger and the check passed on a letter that now
        # contradicted the paper.  That is the exact drift this file is for, so
        # the letter is scanned for anything shaped like a result and each one
        # is looked up.  &nbsp; and the minus entity are normalised first,
        # because the paper writes intervals with both.
        _plain = re.sub(r"<[^>]+>", " ", body)
        for a, b in (("&minus;", "-"), ("&nbsp;", " "), ("−", "-"),
                     (" ", " "), (" ", " ")):
            _plain = _plain.replace(a, b)
        _plain = " ".join(_plain.split())
        for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*(%|days? old|days?)?\s*"
                             r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
                             txt):
            pt, lo, hi = m.group(1), m.group(3), m.group(4)
            if not re.search(re.escape(pt) + r"[^\[]{0,24}\[\s*" + re.escape(lo) +
                             r"\s*,\s*" + re.escape(hi) + r"\s*\]", _plain):
                fails.append(f"{fn}: quotes {pt} [{lo}, {hi}], which the {key} "
                             f"manuscript does not state")
        # and bare percentages that the letter attributes to a result
        for m in re.finditer(r"returns (\d+)%", txt):
            if not re.search(r"\b" + m.group(1) + r"%", _plain):
                fails.append(f"{fn}: says it returns {m.group(1)}%, which the "
                             f"{key} manuscript does not state")

        # and the journal it addresses must be the one the variant was built for
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "bv", os.path.join(HERE, "build_variants.py"))
        bv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bv)
        name = bv.VARIANTS[key]["name"]
        if name not in txt:
            fails.append(f"{fn}: does not name {name}, which is the journal the "
                         f"{key} variant is built for")

    print(f"verify_letters: {seen} of {len(LETTERS)} letters checked"
          + ("  (--fast: page counts NOT measured)" if fast else ""))
    for f in fails:
        print("  FAIL  " + f)
    if not fails:
        print("  every number in the cover letters matches what it describes")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
